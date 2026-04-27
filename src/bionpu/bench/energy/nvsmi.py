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

"""GPU energy reader via ``nvidia-smi``.

Implements the GPU whole-board reader contract from
``bionpu/bench/POWER_DOMAINS.md`` §2 and the joule-integration window
from ``bionpu/bench/UNITS.md`` §3.1. Subsumes the design of 's
``NvidiaSmiReal`` (in ``tracks/crispr/baseline/run_baseline.py``) into
the canonical home for future tasks to import from.

Method auto-selection
---------------------

At construction, the reader probes ``nvidia-smi --help-query-gpu`` for
the ``total_energy_consumption`` field:

- Supported (driver ≥ 535 on most GeForce; ≥ 470 on data-center cards)
  → :attr:`method` = ``"total_energy_consumption"``. The reader takes
  one driver-integrated counter sample at :meth:`start` and one at
  :meth:`stop`, returning ``(end - start) / 1000`` joules (counter is
  in millijoules).
- Not supported → :attr:`method` = ``"power.draw"``. The reader spawns
  a daemon thread polling ``power.draw`` at :data:`_DEFAULT_POLL_HZ`
  (10 Hz; used 4 Hz, this implementation tightens to 10 Hz for
  better resolution at the cost of ~3x more nvidia-smi invocations
  per second). :meth:`stop` joins the thread, then trapezoidal-
  integrates the ``(t_monotonic, watts)`` samples.

Multi-GPU
---------

``gpu_index=None`` (default) sums across every GPU reported by
``nvidia-smi``. Pass ``gpu_index=0`` (or any int) to pin to a single
GPU. The driver-integrated counter path queries with ``--id=<i>`` and
the trapezoidal path filters polled lines by GPU index.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar

from ..units import EnergySource

# 10 Hz poll for power.draw fallback ( used 4 Hz; we tighten for
# better resolution on bursty workloads — POWER_DOMAINS.md §2.4 flags
# the under-resolution risk).
_DEFAULT_POLL_HZ: float = 10.0

class NvidiaSmiUnavailableError(RuntimeError):
    """Raised when ``nvidia-smi`` is not on PATH or returns no parseable data.

    Catch sites either fall back to
    :class:`bionpu.bench.harness.NvidiaSmiStub` deliberately or skip the
    GPU energy comparison.
    """

def _probe_nvidia_smi() -> str | None:
    """Return the path to nvidia-smi if available + working, else None.

    Working = on PATH AND a power.draw query returns a numeric value.
    """
    bin_path = shutil.which("nvidia-smi")
    if not bin_path:
        return None
    try:
        proc = subprocess.run(
            [bin_path, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    try:
        float(out[0].strip())
    except ValueError:
        return None
    return bin_path

def _has_total_energy_consumption(bin_path: str) -> bool:
    """Probe whether the installed driver supports ``total_energy_consumption``.

    Two-step verification: (a) the help text lists the field; (b) an
    actual query returns a numeric value. Both required because some
    older drivers list the field but error on query.
    """
    try:
        proc = subprocess.run(
            [bin_path, "--help-query-gpu"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if "total_energy_consumption" not in proc.stdout:
        return False
    # Confirm by an actual query
    try:
        q = subprocess.run(
            [
                bin_path,
                "--query-gpu=total_energy_consumption",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if q.returncode != 0:
        return False
    out = q.stdout.strip().splitlines()
    if not out:
        return False
    try:
        float(out[0].strip())
    except ValueError:
        return False
    return True

def _read_total_energy_mj(bin_path: str, gpu_index: int | None) -> float:
    """One-shot read of total_energy_consumption (mJ); sums across GPUs if None."""
    args = [
        bin_path,
        "--query-gpu=total_energy_consumption",
        "--format=csv,noheader,nounits",
    ]
    if gpu_index is not None:
        args.insert(1, "--id=" + str(gpu_index))
    proc = subprocess.run(args, capture_output=True, text=True, timeout=5.0)
    if proc.returncode != 0:
        raise NvidiaSmiUnavailableError(
            f"nvidia-smi total_energy_consumption query failed: "
            f"exit={proc.returncode} stderr={proc.stderr.strip()!r}"
        )
    total_mj = 0.0
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        total_mj += float(line)
    return total_mj

def _read_power_draw_watts(bin_path: str, gpu_index: int | None) -> float:
    """One-shot read of power.draw (W); sums across GPUs if None."""
    args = [bin_path, "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
    if gpu_index is not None:
        args.insert(1, "--id=" + str(gpu_index))
    proc = subprocess.run(args, capture_output=True, text=True, timeout=5.0)
    if proc.returncode != 0:
        raise NvidiaSmiUnavailableError(
            f"nvidia-smi power.draw query failed: "
            f"exit={proc.returncode} stderr={proc.stderr.strip()!r}"
        )
    total_w = 0.0
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        total_w += float(line)
    return total_w

@dataclass
class NvidiaSmiReader:
    """Real GPU energy reader with auto-selected integration method.

    At construction, probes the host for ``nvidia-smi`` and the
    ``total_energy_consumption`` capability. Records the chosen method
    on :attr:`method` so the harness's measurement passport can cite
    it (POWER_DOMAINS.md §2.3).

    Args:
        gpu_index: GPU to read. ``None`` (default) sums across all GPUs;
            an int pins to a single GPU index.
        poll_hz: Polling rate for the trapezoidal-integration fallback.
            Ignored when the driver supports ``total_energy_consumption``.

    Raises:
        NvidiaSmiUnavailableError: nvidia-smi not on PATH or returns no
            parseable data. POWER_DOMAINS.md §2 says fail loud rather
            than report zero.
    """

    is_real: bool = True
    source: EnergySource = EnergySource.REAL_NVIDIA_SMI
    gpu_index: int | None = None
    poll_hz: float = _DEFAULT_POLL_HZ
    method: str = "power.draw"  # "total_energy_consumption" or "power.draw"

    _bin_path: str = ""
    _start_mj: float = 0.0  # used when method == total_energy_consumption
    _samples: list[tuple[float, float]] = field(default_factory=list)
    _stop_evt: threading.Event | None = None
    _thread: threading.Thread | None = None
    _n_samples: int = 0

    # Class-level flag; tests monkeypatch _probe_nvidia_smi directly.
    METHOD_DRIVER_INTEGRATED: ClassVar[str] = "total_energy_consumption"
    METHOD_TRAPEZOIDAL: ClassVar[str] = "power.draw"

    def __init__(
        self,
        gpu_index: int | None = None,
        poll_hz: float = _DEFAULT_POLL_HZ,
    ) -> None:
        self.is_real = True
        self.source = EnergySource.REAL_NVIDIA_SMI
        self.gpu_index = gpu_index
        self.poll_hz = poll_hz
        bin_path = _probe_nvidia_smi()
        if bin_path is None:
            raise NvidiaSmiUnavailableError(
                "nvidia-smi not on PATH or returned no parseable power.draw line; "
                "install NVIDIA driver + nvidia-utils-* or skip GPU energy comparison "
                "(POWER_DOMAINS.md §2)"
            )
        self._bin_path = bin_path
        if _has_total_energy_consumption(bin_path):
            self.method = self.METHOD_DRIVER_INTEGRATED
        else:
            self.method = self.METHOD_TRAPEZOIDAL
        self._samples = []
        self._stop_evt = None
        self._thread = None
        self._n_samples = 0
        self._start_mj = 0.0

    # ----- driver-integrated path -----

    def _start_driver(self) -> None:
        self._start_mj = _read_total_energy_mj(self._bin_path, self.gpu_index)

    def _stop_driver(self) -> float:
        end_mj = _read_total_energy_mj(self._bin_path, self.gpu_index)
        # Counter is monotonic per the driver; if we somehow get
        # end < start treat as zero rather than negative.
        delta_mj = max(0.0, end_mj - self._start_mj)
        return delta_mj / 1000.0  # mJ → J

    # ----- trapezoidal fallback path -----

    def _poll_loop(self) -> None:
        assert self._stop_evt is not None
        interval = 1.0 / max(0.1, self.poll_hz)
        while not self._stop_evt.is_set():
            t = time.monotonic()
            try:
                w = _read_power_draw_watts(self._bin_path, self.gpu_index)
            except NvidiaSmiUnavailableError:
                # Transient; let stop() decide based on samples count.
                self._stop_evt.wait(interval)
                continue
            self._samples.append((t, w))
            self._stop_evt.wait(interval)

    def _start_trapezoidal(self) -> None:
        self._samples = []
        self._stop_evt = threading.Event()
        # Take a sample at start so even very short windows have ≥1
        # endpoint to anchor the trapezoid.
        try:
            w0 = _read_power_draw_watts(self._bin_path, self.gpu_index)
            self._samples.append((time.monotonic(), w0))
        except NvidiaSmiUnavailableError:
            pass
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _stop_trapezoidal(self) -> float:
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Take a final sample post-stop so the integration window covers
        # right up to the user's stop() call.
        try:
            w_end = _read_power_draw_watts(self._bin_path, self.gpu_index)
            self._samples.append((time.monotonic(), w_end))
        except NvidiaSmiUnavailableError:
            pass
        samples = self._samples
        self._n_samples = len(samples)
        if len(samples) < 2:
            # Window too short (or polling never returned a parseable
            # line). Per POWER_DOMAINS.md §2 / never-fabricate, signal
            # via 0.0 joules and let the caller decide whether to fall
            # back to a stub.
            return 0.0
        joules = 0.0
        for (t0, w0), (t1, w1) in zip(samples, samples[1:], strict=False):
            dt = max(0.0, t1 - t0)
            joules += 0.5 * (w0 + w1) * dt
        return joules

    # ----- EnergyReader Protocol -----

    def start(self) -> None:
        if self.method == self.METHOD_DRIVER_INTEGRATED:
            self._start_driver()
        else:
            self._start_trapezoidal()

    def stop(self) -> float:
        if self.method == self.METHOD_DRIVER_INTEGRATED:
            return self._stop_driver()
        return self._stop_trapezoidal()

__all__ = [
    "NvidiaSmiReader",
    "NvidiaSmiUnavailableError",
]
