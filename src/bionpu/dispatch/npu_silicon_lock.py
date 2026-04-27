# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""NPU silicon dispatch mutex + pre-flight wedge check.

THIS IS THE CANONICAL way to serialise NPU silicon dispatch in this
repo. Every code path that submits work to /dev/accel/accel0 — both
the in-process pyxrt path (:mod:`bionpu.dispatch.npu`) and the
subprocess-based microtest harnesses (under
``tests/aie2p_microtests/*/harness.py``) — MUST wrap silicon
submissions in :func:`npu_silicon_lock`.

Why this module exists
----------------------

Background: during the 2026-04-25 9-agent swarm, multiple agents
observed firmware wedges they attributed to sibling-agent silicon
submissions. The original mutex discipline was a shell-level
``flock`` wrapper, but two paths (the swarm plan text vs. the
per-agent prompts) referenced different lock paths
(``state/swarm/npu-silicon.lock`` vs. ``/tmp/bionpu-npu-silicon.lock``)
which broke the serialisation. T5 documented 13 distinct context
IDs across hwctx 14 AND 15 active concurrently. The wedges may or
the candidate findings) but the contention confound made it
impossible to distinguish.

This module makes the discipline mechanical:

* Single canonical lock path (``/tmp/bionpu-npu-silicon.lock``) —
  no plan/prompt drift possible.
* Python context manager — easier to audit than ``flock`` shell
  wrappers, can't be bypassed by typos.
* Pre-flight dmesg wedge check — refuses to submit on top of a
  device that's already showing fresh ``aie2_dump_ctx`` /
  ``Firmware timeout`` activity (operator must wait for TDR drain
  or explicitly override).
* Stale-lock recovery — if the holder PID is dead (segfault,
  killed, parent died), the lock is silently taken over after
  verifying.
* PID + label sidecar — diagnostic visibility into who currently
  holds the device.

Usage
-----

In-process pyxrt::

    from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

    with npu_silicon_lock(label=f"crispr_pam_filter:{my_chunk_id}"):
        run = kernel(opcode, bo_instr, instr_size, *bos)
        state = run.wait()

Subprocess-based harness::

    from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

    with npu_silicon_lock(label="dma_compression_loopback_microtest"):
        proc = subprocess.run(
            [str(runner), "-x", str(xclbin), ...],
            capture_output=True, text=True, timeout=timeout_sec,
        )

Wedge override (recovery scenarios only)::

    # When you've manually drained the device and need to test the
    # wedge directly:
    with npu_silicon_lock(label="recovery", check_wedge=False):
        ...

Diagnostic::

    from bionpu.dispatch.npu_silicon_lock import lock_status
    print(lock_status())  # who holds the lock, is the holder alive
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterator

# Single canonical paths. Any other reference in the codebase is wrong.
LOCK_PATH: Path = Path("/tmp/bionpu-npu-silicon.lock")
PID_PATH: Path = Path("/tmp/bionpu-npu-silicon.pid")

# dmesg patterns indicating a fresh AIE2P firmware wedge.
# These are the strings the amdxdna driver writes to ring buffer
# when the AIE2P firmware times out a kernel dispatch.
WEDGE_PATTERNS: tuple[str, ...] = (
    "aie2_dump_ctx",
    "Firmware timeout",
    "aie2_tdr_work",
)

# How many dmesg lines to scan for fresh wedge evidence. 200 is large
# enough to catch a multi-context wedge cascade but small enough to
# stay below dmesg ring-buffer noise from unrelated subsystems.
DMESG_TAIL_LINES: int = 200

class WedgeDetected(RuntimeError):
    """Fresh ``aie2_dump_ctx`` / ``Firmware timeout`` in dmesg tail.

    The pre-flight check refuses to submit on top of a wedged device.
    Wait for TDR drain (5-10 min by default) or explicitly override
    via ``check_wedge=False`` (recovery scenarios only).
    """

class LockTimeout(RuntimeError):
    """Couldn't acquire NPU silicon mutex within ``timeout_sec``."""

@dataclasses.dataclass(frozen=True)
class LockHolderInfo:
    """Snapshot of the current lock holder, read from the PID sidecar."""

    pid: int
    label: str
    acquired_at: float

    @property
    def age_sec(self) -> float:
        return max(0.0, time.time() - self.acquired_at)

def _read_pid_file() -> LockHolderInfo | None:
    try:
        text = PID_PATH.read_text()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(text)
        return LockHolderInfo(
            pid=int(data["pid"]),
            label=str(data.get("label", "")),
            acquired_at=float(data.get("acquired_at", 0.0)),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

def _is_pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live process. Robust to permission errors."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it — still counts as alive
        return True

def _snapshot_dmesg(tail_lines: int = DMESG_TAIL_LINES) -> list[str]:
    """Best-effort dmesg snapshot. Returns [] if dmesg is unavailable."""
    try:
        proc = subprocess.run(
            ["dmesg", "-T", "--ctime"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return proc.stdout.splitlines()[-tail_lines:]
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return []

def _scan_for_wedge(dmesg_tail: list[str]) -> tuple[bool, str | None]:
    """Conservative scan: any wedge pattern in the tail counts as fresh.

    Returns (found, sample_line) — the sample line lets the caller
    surface the matched dmesg entry in the WedgeDetected message.
    """
    for line in dmesg_tail:
        for pattern in WEDGE_PATTERNS:
            if pattern in line:
                return True, line.strip()
    return False, None

def lock_status() -> dict:
    """Read-only diagnostic snapshot. Safe to call without holding the lock."""
    holder = _read_pid_file()
    return {
        "lock_path": str(LOCK_PATH),
        "pid_path": str(PID_PATH),
        "lock_exists": LOCK_PATH.exists(),
        "holder": dataclasses.asdict(holder) if holder else None,
        "holder_alive": _is_pid_alive(holder.pid) if holder else None,
    }

@contextlib.contextmanager
def npu_silicon_lock(
    *,
    timeout_sec: float = 60.0,
    check_wedge: bool = True,
    label: str = "",
    poll_interval_sec: float = 0.5,
) -> Iterator[LockHolderInfo]:
    """Serialise NPU silicon dispatch + pre-flight wedge check.

    On entry: acquires an exclusive ``flock`` on :data:`LOCK_PATH` and
    writes ``{pid, label, acquired_at}`` to :data:`PID_PATH`. If
    ``check_wedge`` (default), takes a dmesg snapshot and raises
    :class:`WedgeDetected` if any fresh ``aie2_dump_ctx`` /
    ``Firmware timeout`` / ``aie2_tdr_work`` pattern is visible.

    On exit (normal or exception): releases the lock + clears the PID
    sidecar.

    Stale-lock recovery: if the lock file is held by a PID that no
    longer exists (segfault, killed, parent crash), the sidecar is
    silently cleared and the next acquire attempt proceeds. The
    underlying ``flock`` is automatically released by the kernel when
    the holding process exits, so ``LOCK_EX | LOCK_NB`` will succeed.

    Args:
        timeout_sec: max time to wait for the lock. Raises
            :class:`LockTimeout` if exceeded.
        check_wedge: take a dmesg snapshot before yielding; raise
            :class:`WedgeDetected` on fresh wedge. Set ``False`` ONLY
            for recovery scenarios where the operator has manually
            verified the device.
        label: human-readable tag written to the PID sidecar (test
            name, agent ID, kernel name). Surfaced via
            :func:`lock_status` for diagnostic.
        poll_interval_sec: wait between non-blocking acquire retries.

    Yields:
        :class:`LockHolderInfo` describing the freshly acquired lock
        (own PID, label, acquired_at).

    Raises:
        LockTimeout: lock not acquired within ``timeout_sec``.
        WedgeDetected: dmesg snapshot shows fresh wedge entries.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_sec
    acquired = False
    holder_info: LockHolderInfo | None = None

    try:
        while not acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                holder = _read_pid_file()
                if holder is not None and not _is_pid_alive(holder.pid):
                    # Stale sidecar from a dead process; clear and retry.
                    # The kernel-level flock has already been released.
                    try:
                        PID_PATH.unlink()
                    except FileNotFoundError:
                        pass
                    time.sleep(0.05)
                    continue
                if time.monotonic() >= deadline:
                    raise LockTimeout(
                        f"NPU silicon lock not acquired within {timeout_sec}s. "
                        f"Current holder: {holder!r}"
                    )
                time.sleep(poll_interval_sec)

        holder_info = LockHolderInfo(
            pid=os.getpid(),
            label=label,
            acquired_at=time.time(),
        )
        PID_PATH.write_text(
            json.dumps(
                {
                    "pid": holder_info.pid,
                    "label": holder_info.label,
                    "acquired_at": holder_info.acquired_at,
                }
            )
        )

        if check_wedge:
            dmesg_tail = _snapshot_dmesg()
            found, sample = _scan_for_wedge(dmesg_tail)
            if found:
                raise WedgeDetected(
                    f"Fresh wedge pattern in dmesg tail; refusing to submit. "
                    f"Sample line: {sample!r}. Wait for TDR drain (~5-10 min) "
                    f"or pass check_wedge=False to override (recovery only)."
                )

        yield holder_info

    finally:
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            os.close(fd)
        except Exception:
            pass
