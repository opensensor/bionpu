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

"""NPU dispatch backend — in-process pyxrt 3.14.

Per the umbrella PRD §4.1 v1-thin contract, this module ships a
**registered-op lookup table** plus a thin XRT-driven runner. There is
register their own ops via the same pattern — they extend
:data:`NPU_OPS` from their track-side modules.

where Phase 1's per-call subprocess fork dominated wall-clock at
~6 s/chunk and made full-GRCh38 NPU runs cost-prohibitive.

Why in-process pyxrt is now possible:
  ``/opt/xilinx/xrt/python/`` while ironenv ran 3.11 (mlir-aie wheel
  ``import pyxrt`` works in-process; ``pyxrt.enumerate_devices()``
  from the fork. The 3.14-vs-3.11 mismatch is gone, so
  :class:`NpuBackend` now drives XRT directly.

* Device open: ``pyxrt.device(0)`` (NOT ``pyxrt.system().get_drivers()``
* xclbin load: ``device.register_xclbin(pyxrt.xclbin(path))`` followed
  by ``pyxrt.hw_context(device, xclbin.get_uuid())`` and
  ``pyxrt.kernel(ctx, kernel_name)``. (``device.load_xclbin`` is a
  deprecated compatibility shim per pyxrt's docstring; the modern path
  uses ``hw_context``.)
* Buffer alloc: ``pyxrt.bo(device, size, flags, group_id)`` where
  ``flags`` is one of ``pyxrt.bo.cacheable`` / ``pyxrt.bo.host_only``
  and ``group_id`` comes from ``kernel.group_id(arg_index)``.
* Buffer write: ``bo.write(bytes_or_buffer, offset)``; sync to/from
  device via ``bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_*)``.
* Kernel invocation: ``run = kernel(opcode, bo_instr, instr_size,
  bo_in1, bo_in2, bo_out, ...); run.wait()`` — matches the C++
  ``xrt_test_wrapper.h`` ABI verbatim. Opcode is ``3`` for AIE2P
  programmable-NPU launches.
* Buffer read: ``bo.read(size, offset)`` returns a numpy ``int8``
  array; we re-interpret via ``np.frombuffer(...).view(target_dtype)``.

Sticky-handle caching:
  :class:`NpuBackend` opens the device **once** at first use and caches
  ``(xclbin, hw_context, kernel)`` tuples per xclbin path. Subsequent
  ``NpuOp.__call__`` invocations reuse those handles, amortising xclbin
  open + driver setup across the entire host process. This is the
  wall.

  Setting ``BIONPU_DISPATCH_FORCE_SUBPROCESS=1`` reverts each registered
  op to the Phase 1 host-binary subprocess path via
  :meth:`NpuBackend.run_host_binary`. That method imports ``subprocess``
  lazily inside the function — there is **no module-level**
  ``import subprocess`` in this file ( contract: ``test_no_subprocess_import``
  asserts this). The escape hatch exists so Phase 1's existing test
  fixtures and downstream consumers that can't run in-process pyxrt
  for whatever reason (e.g. a 3.13 venv on a different host) can still
  drive the kernels.

- :class:`NpuBackend` — singleton-style runner; opens device once;
  caches xclbin + kernel handles per artifact path.
- :class:`NpuOp` — the abstract base every registered op subclasses.
- :data:`NPU_OPS` — the registry that ``bionpu.dispatch.dispatch`` looks
  ``op="..."`` up in when ``device="npu"``.
- :exc:`NpuArtifactsMissingError`,
  :exc:`NpuRunFailed` — typed errors so call-sites can decide whether
  to re-raise or downgrade to ``reason_unavailable=...`` in the bench
  record.
- :func:`_xrt_env` — env dict for subprocess fallback. Preserved as a
  documented internal so kernels that still subprocess (e.g.
  ``bionpu/kernels/crispr/match_singletile/``) keep working.

Silicon mutex policy:
  Normal in-process dispatch does **not** take the repo-level
  ``bionpu.dispatch.npu_silicon_lock``. The lock helper remains only
  for standalone recovery/operator scripts that import it directly.

Naming convention: track-side registrations key by ``"<track>:<op>"``
(e.g. ``"basecalling:conv_stem"``, ``"crispr:match_singletile"``). The
umbrella canary, not a track op.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# The artifacts directory is colocated with this module so the dispatch
_ARTIFACT_ROOT = Path(__file__).resolve().parent / "_npu_artifacts"

_VSM_DIR = _ARTIFACT_ROOT / "vector_scalar_mul"
_VSM_XCLBIN = _VSM_DIR / "final_in1_size.xclbin"
_VSM_INSTS = _VSM_DIR / "insts_in1_size.bin"
_VSM_BINARY = _VSM_DIR / "vector_scalar_mul_in1_size"

# The C++ binary at MLIR_AIE example time hardcodes IN1 = 8192 bytes /
# sizeof(int16) = 4096 elements, IN2 = one int32 scalar (=3), OUT
# matches IN1. Documented in MANIFEST.md and pinned in the dataclass
# below so call-sites can introspect.
_VSM_VOLUME = 4096
_VSM_DTYPE = np.int16
_VSM_SCALAR = 3
_VSM_KERNEL_NAME = "MLIR_AIE"

# AIE2P programmable-NPU launch opcode — matches the C++ wrapper's
# ``unsigned int opcode = 3`` constant in ``xrt_test_wrapper.h``.
_AIE2P_OPCODE = 3

# Backward-compatible inert knob. The dispatch path intentionally ignores
# it so app dispatch cannot silently serialize through a repo-level flock.
_LOCK_ENV = "BIONPU_NPU_SILICON_LOCK"

def _dispatch_lock(label: str):
    del label
    return contextlib.nullcontext()

# Regexes pulled from the host_binary subprocess path's stdout. Tightly
# anchored so a future binary that adds more output doesn't silently
# match the wrong one. Used only by the BIONPU_DISPATCH_FORCE_SUBPROCESS
# escape hatch.
_RE_AVG = re.compile(r"^Avg NPU time:\s*([0-9.]+)us\.?\s*$", re.MULTILINE)
_RE_MIN = re.compile(r"^Min NPU time:\s*([0-9.]+)us\.?\s*$", re.MULTILINE)
_RE_MAX = re.compile(r"^Max NPU time:\s*([0-9.]+)us\.?\s*$", re.MULTILINE)

_FORCE_SUBPROCESS_ENV = "BIONPU_DISPATCH_FORCE_SUBPROCESS"

def _force_subprocess() -> bool:
    """True iff the caller has opted into the Phase 1 subprocess fallback.

    Set ``BIONPU_DISPATCH_FORCE_SUBPROCESS=1`` to revert to the
    host-binary subprocess path. Default behaviour is in-process pyxrt.
    """
    val = os.environ.get(_FORCE_SUBPROCESS_ENV, "")
    return val.strip() not in ("", "0", "false", "False", "no", "NO")

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class NpuArtifactsMissingError(FileNotFoundError):
    """One of the precompiled NPU artifacts is missing on disk.

    Raised by :class:`NpuBackend.load` when an xclbin / instructions /
    host-binary path doesn't exist. The error message names every
    missing path so the operator knows what to rebuild.
    """

class NpuRunFailed(RuntimeError):
    """The kernel ran but returned a non-zero / error state.

    For the in-process path, this carries the
    ``pyxrt.ert_cmd_state`` value plus a short context string. For the
    subprocess fallback, it carries the host binary's full
    stdout/stderr so callers can log them verbatim into
    ``MeasurementRun.reason_unavailable``.
    """

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(
            f"NPU kernel run failed (status={returncode}). "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NpuRunResult:
    """One kernel run's measured timing + verification status.

    Attributes:
        output: the elementwise result the kernel produced. For
            ``vector_scalar_mul`` this is a numpy array equal to
            ``input * scalar`` at INT16 precision.
        avg_us: average NPU-side time across iterations, in microseconds.
        min_us: minimum NPU-side time, in microseconds.
        max_us: maximum NPU-side time, in microseconds.
        n_iters: number of iterations the runner ran.
        verified: True when verification passed (in-process: numpy
            byte-equal vs CPU reference; subprocess: host binary's
            internal verify step).
    """

    output: np.ndarray
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    verified: bool

# --------------------------------------------------------------------------- #
# Subprocess fallback env (preserved for escape hatch + legacy kernels)
# --------------------------------------------------------------------------- #

def _xrt_env() -> dict[str, str]:
    """Return an env dict with XRT/LD_LIBRARY_PATH set up for subprocesses.

    ``bionpu/kernels/crispr/match_singletile/``) call this directly to
    doesn't need this — pyxrt finds libxrt via the running process's
    LD_LIBRARY_PATH at module-import time. We keep it exported so those
    legacy call-sites don't break.
    """
    env = os.environ.copy()
    xrt_root = env.get("XILINX_XRT", "/opt/xilinx/xrt")
    env["XILINX_XRT"] = xrt_root
    extra_lib = f"{xrt_root}/lib"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if extra_lib not in existing_ld.split(":"):
        env["LD_LIBRARY_PATH"] = (
            f"{extra_lib}:{existing_ld}" if existing_ld else extra_lib
        )
    return env

# --------------------------------------------------------------------------- #
# pyxrt lazy loader (so non-NPU hosts can import this module)
# --------------------------------------------------------------------------- #

_pyxrt = None  # cached pyxrt module after first successful import

def _get_pyxrt():
    """Lazily import + cache ``pyxrt``.

    Returns the pyxrt module on success, raises :class:`ImportError`
    with a clear hint on failure. Calls are cheap after the first one.
    The lazy import lets the module be imported on hosts without XRT
    (e.g. CI runners running the non-npu marked tests) without crashing
    at module-load.
    """
    global _pyxrt
    if _pyxrt is not None:
        return _pyxrt
    try:
        import pyxrt  # noqa: PLC0415 — intentional lazy import
    except ImportError as exc:  # pragma: no cover — exercised only on no-XRT hosts
        raise ImportError(
            "pyxrt unavailable; source /opt/xilinx/xrt/setup.sh and "
            "ensure ironenv is on Python 3.14. "
            f"Original error: {exc}"
        ) from exc
    _pyxrt = pyxrt
    return pyxrt

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

@dataclass
class _LoadedKernel:
    """A loaded xclbin + kernel handle, plus pre-loaded instructions.

    The handles live for the lifetime of the host process; loading is
    a one-time cost amortised across every subsequent
    """

    xclbin_path: Path
    insts_path: Path
    kernel_name: str
    xclbin: Any  # pyxrt.xclbin
    kernel: Any  # pyxrt.kernel
    instr_u32: np.ndarray  # uint32 instruction stream pre-loaded

# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #

class NpuBackend:
    """Singleton-style runner that drives the AIE2P NPU via in-process pyxrt.

    The backend is lazily initialised — calling :meth:`probe` is the
    only way to ask "is the NPU usable in this process?" without
    actually running a kernel.

    Sticky-handle policy: the device is opened once on first use
    and cached for the lifetime of this :class:`NpuBackend` instance.
    Per-op xclbin + kernel handles are cached in
    :attr:`_kernel_cache` keyed by xclbin path; subsequent loads of the
    same xclbin reuse the cached handle. This is the amortisation that
    """

    def __init__(self) -> None:
        self._probed: bool | None = None
        self._probe_reason: str | None = None
        self._device: Any = None  # pyxrt.device after first use
        self._kernel_cache: dict[Path, _LoadedKernel] = {}

    # ------------------------------------------------------------------
    # probing
    # ------------------------------------------------------------------

    def probe(self) -> bool:
        """Return True if the NPU is callable in this process.

        Caches the result so repeated calls are cheap. The probe checks
        only the *static* preconditions: pyxrt importable, xrt-smi
        will surface kernel-load errors.
        """
        if self._probed is not None:
            return self._probed

        # 1. pyxrt importable?
        try:
            _get_pyxrt()
        except ImportError as exc:
            self._probe_reason = str(exc)
            self._probed = False
            return False

        # 2. xrt-smi binary present? (sanity for "did the operator
        #    source setup.sh / install xrt?")
        xrt_smi = shutil.which("xrt-smi") or "/opt/xilinx/xrt/bin/xrt-smi"
        if not Path(xrt_smi).exists():
            self._probe_reason = (
                f"xrt-smi not found (looked for {xrt_smi}); source "
                "/opt/xilinx/xrt/setup.sh and rerun"
            )
            self._probed = False
            return False

        missing = [p for p in (_VSM_XCLBIN, _VSM_INSTS, _VSM_BINARY) if not p.exists()]
        if missing:
            self._probe_reason = (
                + ", ".join(str(p) for p in missing)
            )
            self._probed = False
            return False

        self._probed = True
        return True

    @property
    def probe_reason(self) -> str | None:
        """Human-readable reason :meth:`probe` returned False, or None."""
        return self._probe_reason

    # ------------------------------------------------------------------
    # device + xclbin sticky-handle plumbing
    # ------------------------------------------------------------------

    def device(self) -> Any:
        """Return the cached NPU device handle, opening it on first use.

        Sticky: opened once per :class:`NpuBackend` instance. The
        process-level ``_DEFAULT_BACKEND`` therefore opens the device
        exactly once for the entire host run.
        """
        if self._device is None:
            pyxrt = _get_pyxrt()
            # ``pyxrt.enumerate_devices()`` -> 1 device output).
            self._device = pyxrt.device(0)
        return self._device

    def load_xclbin(
        self,
        xclbin_path: Path,
        insts_path: Path,
        kernel_name: str = _VSM_KERNEL_NAME,
    ) -> _LoadedKernel:
        """Load + cache an xclbin and return its kernel handle.

        First call for a given ``xclbin_path``: opens the file, registers
        it on the device, builds an ``hw_context``, and constructs the
        ``kernel`` handle. Subsequent calls return the cached handle.
        Instructions are pre-loaded as a numpy uint32 array.

        Raises:
            NpuArtifactsMissingError: ``xclbin_path`` or ``insts_path``
                doesn't exist.
        """
        xclbin_path = Path(xclbin_path)
        insts_path = Path(insts_path)
        if xclbin_path in self._kernel_cache:
            return self._kernel_cache[xclbin_path]

        for p in (xclbin_path, insts_path):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing: {p}. See "
                    f"{_ARTIFACT_ROOT}/<op>/MANIFEST.md for how to rebuild."
                )

        pyxrt = _get_pyxrt()
        device = self.device()
        xclbin = pyxrt.xclbin(str(xclbin_path))
        device.register_xclbin(xclbin)
        ctx = pyxrt.hw_context(device, xclbin.get_uuid())
        kernel = pyxrt.kernel(ctx, kernel_name)

        instr_u32 = np.fromfile(str(insts_path), dtype=np.uint32)

        loaded = _LoadedKernel(
            xclbin_path=xclbin_path,
            insts_path=insts_path,
            kernel_name=kernel_name,
            xclbin=xclbin,
            kernel=kernel,
            instr_u32=instr_u32,
        )
        # Hold the hw_context alive by attaching to loaded — pyxrt
        # objects are tracked by Python refcount.
        loaded._hw_context = ctx  # type: ignore[attr-defined]
        self._kernel_cache[xclbin_path] = loaded
        return loaded

    # ------------------------------------------------------------------
    # core run — in-process pyxrt path
    # ------------------------------------------------------------------

    def run_xclbin(
        self,
        *,
        xclbin: Path,
        insts: Path,
        kernel_name: str = _VSM_KERNEL_NAME,
        in_buffers: list[tuple[bytes, int]],
        out_size: int,
        out_arg_index: int,
        n_iters: int = 1,
        warmup: int = 0,
    ) -> tuple[bytes, float, float, float]:
        """Drive an xclbin in-process via pyxrt; return raw output bytes + timing.

        Args:
            xclbin: path to the precompiled xclbin.
            insts: path to ``insts.bin`` (uint32 instruction stream).
            kernel_name: kernel symbol inside the xclbin (default
                ``"MLIR_AIE"``, matching the mlir-aie examples).
            in_buffers: list of ``(payload_bytes, group_id_arg_index)``
                tuples — one per input buffer. The ``group_id_arg_index``
                is the kernel-arg index used to look up the memory group
                via ``kernel.group_id(idx)`` (e.g. 3 for ``in1``, 4 for
                ``in2``, 6 for ``ctrlpkts``, 7 for ``trace``).
            out_size: output buffer size in bytes.
            out_arg_index: kernel-arg index for the output buffer
                (e.g. 5 for the standard 2-input-1-output mlir-aie
                wrapper).
            n_iters: number of timed iterations (avg/min/max returned).
            warmup: warmup iterations before timing starts.

        Returns:
            ``(raw_output_bytes, avg_us, min_us, max_us)``.

        Raises:
            NpuArtifactsMissingError: artifact paths missing.
            NpuRunFailed: kernel state != COMPLETED after ``run.wait()``.
        """
        import time  # noqa: PLC0415 — tight-loop timing only

        loaded = self.load_xclbin(xclbin, insts, kernel_name)
        pyxrt = _get_pyxrt()
        device = self.device()
        kernel = loaded.kernel
        instr_u32 = loaded.instr_u32

        # Buffer objects are short-lived per call (sized by op-specific
        # payloads). Sticky cache is xclbin + kernel + device + insts.
        bo_instr = pyxrt.bo(
            device,
            int(instr_u32.nbytes),
            pyxrt.bo.cacheable,
            kernel.group_id(1),
        )
        bo_instr.write(instr_u32.tobytes(), 0)
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # One buffer object per input. We hold them in a list so they
        # outlive the launch.
        in_bos: list[Any] = []
        for payload, arg_idx in in_buffers:
            size = max(1, len(payload))  # group_id placeholder buffers want size>=1
            bo = pyxrt.bo(
                device,
                size,
                pyxrt.bo.host_only,
                kernel.group_id(arg_idx),
            )
            if payload:
                bo.write(payload, 0)
            else:
                bo.write(bytes(size), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            in_bos.append(bo)

        bo_out = pyxrt.bo(
            device,
            int(out_size),
            pyxrt.bo.host_only,
            kernel.group_id(out_arg_index),
        )
        bo_out.write(bytes(int(out_size)), 0)
        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # ABI: kernel(opcode, bo_instr, instr_size, *in_bos, bo_out)
        # The mlir-aie xrt_test_wrapper.h ABI for 2 inputs / 1 output
        # passes (in1, in2, out, ctrlpkts, trace); arg_index hints in
        # ``in_buffers`` already encode the relative position. We rebuild
        # the argv in arg-index order so a future kernel that adds more
        # inputs slots in cleanly.
        # The arg order is:  arg0 = opcode, arg1 = bo_instr,
        # arg2 = instr_size, arg3.. = in_bos in order, then bo_out at
        # ``out_arg_index``, then any trailing in_bos (ctrlpkts, trace).
        # We reorder by arg_index: walk all positional slots from 3 to
        # max(arg_index, out_arg_index) and place each buffer.
        max_arg = max([out_arg_index] + [a for _, a in in_buffers])
        argv: list[Any] = [_AIE2P_OPCODE, bo_instr, int(instr_u32.size)]
        # Build arg_index -> buffer map; out goes to out_arg_index.
        slot: dict[int, Any] = {out_arg_index: bo_out}
        for (_, arg_idx), bo in zip(in_buffers, in_bos, strict=True):
            slot[arg_idx] = bo
        for i in range(3, max_arg + 1):
            if i in slot:
                argv.append(slot[i])
            else:
                # Unused arg slots happen when the kernel signature
                # allocates ctrlpkts/trace placeholders — caller is
                # responsible for placing them. If a slot is missing
                # we fail loud rather than passing nullptr.
                raise ValueError(
                    f"run_xclbin: kernel arg slot {i} unfilled. "
                    f"Pass a placeholder via in_buffers (e.g. "
                    f"(b'', {i}) for an 8-byte placeholder)."
                )

        num_iter = int(n_iters) + int(warmup)
        if num_iter < 1:
            raise ValueError(
                f"run_xclbin: n_iters + warmup must be >= 1; got "
                f"n_iters={n_iters} warmup={warmup}"
            )

        # The old bring-up path serialized all in-process dispatches under
        # bionpu.dispatch.npu_silicon_lock. That was too conservative for
        # normal execution and could hide whether XRT/driver parallelism is
        # valid, so this path no longer acquires the repo-level flock.
        timings_us: list[float] = []
        with _dispatch_lock(label=f"in_process_pyxrt:{kernel_name}:n={num_iter}"):
            for it in range(num_iter):
                t0 = time.perf_counter()
                run = kernel(*argv)
                state = run.wait()
                t1 = time.perf_counter()
                if (
                    str(state).endswith("COMPLETED") is False
                    and state.name != "ERT_CMD_STATE_COMPLETED"
                ):
                    raise NpuRunFailed(
                        int(state.value) if hasattr(state, "value") else -1,
                        "",
                        f"kernel state={state!r} (expected COMPLETED)",
                    )
                if it < int(warmup):
                    continue
                timings_us.append(1e6 * (t1 - t0))

        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        # ``bo.read`` returns numpy.ndarray[int8]; we re-cast bytes-side.
        raw = bo_out.read(int(out_size), 0)
        raw_bytes = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)

        avg_us = float(sum(timings_us) / max(len(timings_us), 1))
        min_us = float(min(timings_us)) if timings_us else 0.0
        max_us = float(max(timings_us)) if timings_us else 0.0
        return raw_bytes, avg_us, min_us, max_us

    # ------------------------------------------------------------------
    # subprocess fallback — preserved for BIONPU_DISPATCH_FORCE_SUBPROCESS=1
    # ------------------------------------------------------------------

    def run_host_binary(
        self,
        *,
        binary: Path,
        xclbin: Path,
        insts: Path,
        kernel_name: str = _VSM_KERNEL_NAME,
        n_iters: int = 1,
        warmup: int = 0,
        verify: bool = True,
        timeout_s: float = 30.0,
    ) -> tuple[float, float, float, int, str, str]:
        """Run an mlir-aie host binary (subprocess fallback).

        ``BIONPU_DISPATCH_FORCE_SUBPROCESS=1`` escape hatch and as a
        regression target for ``test_byte_equality_vs_subprocess``.

        ``import subprocess`` is intentionally **lazy** here — the
        :func:`tests.test_dispatch_in_process.test_no_subprocess_import`.

        Returns:
            (avg_us, min_us, max_us, returncode, stdout, stderr)

        Raises:
            NpuArtifactsMissingError: artifact paths missing.
            NpuRunFailed: binary exited non-zero or verify failed.
        """
        import subprocess # noqa: PLC0415 — see module docstring contract

        for p in (binary, xclbin, insts):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing: {p}. See "
                    f"{_ARTIFACT_ROOT}/<op>/MANIFEST.md for how to rebuild."
                )

        cmd = [
            str(binary),
            "-x",
            str(xclbin),
            "-i",
            str(insts),
            "-k",
            kernel_name,
            "--iters",
            str(int(n_iters)),
            "--warmup",
            str(int(warmup)),
        ]
        proc = subprocess.run(  # noqa: S603 — we control argv strictly
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=_xrt_env(),
        )
        if proc.returncode != 0:
            raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
        if verify and "PASS!" not in proc.stdout:
            raise NpuRunFailed(
                proc.returncode,
                proc.stdout,
                proc.stderr + "\n[bionpu] expected 'PASS!' marker not found",
            )
        m_avg = _RE_AVG.search(proc.stdout)
        m_min = _RE_MIN.search(proc.stdout)
        m_max = _RE_MAX.search(proc.stdout)
        if not (m_avg and m_min and m_max):
            raise NpuRunFailed(
                proc.returncode,
                proc.stdout,
                proc.stderr + "\n[bionpu] could not parse NPU timing lines",
            )
        return (
            float(m_avg.group(1)),
            float(m_min.group(1)),
            float(m_max.group(1)),
            proc.returncode,
            proc.stdout,
            proc.stderr,
        )

# Process-level singleton. The backend's sticky device + xclbin cache
# process. NPU device contention is the kernel module's concern.
_DEFAULT_BACKEND = NpuBackend()

def default_backend() -> NpuBackend:
    """Return the process-level default :class:`NpuBackend` instance."""
    return _DEFAULT_BACKEND

# --------------------------------------------------------------------------- #
# Op base class + registry
# --------------------------------------------------------------------------- #

class NpuOp(ABC):
    """Abstract base for every registered NPU op.

    Subclasses must implement :meth:`__call__`, which takes whatever
    inputs the op needs and returns the output as a numpy array (or a
    tuple of numpy arrays for multi-output ops). The dispatcher passes
    keyword arguments straight through, so the op's signature is the
    op's contract.
    """

    name: str

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        ...

class NpuVectorScalarMul(NpuOp):
    """Pre-built ``vector_scalar_mul`` parity op.

    Pin: the precompiled xclbin's instruction stream initialises its
    own input data (``arange(1, 4097)`` of int16) and its own scalar
    (3) only when the **subprocess** ``host_runner`` is used (the C++
    wrapper does the init). The in-process pyxrt path uploads the
    caller-provided input/scalar buffers directly. Either way we
    enforce the canonical fixture so the parity test stays apples-to-

    Returns:
        np.ndarray of shape ``(4096,)`` and dtype ``int16``: the
        kernel's verified output (== ``in1 * scalar`` modulo int16
        wrap).
    """

    name = "vector_scalar_mul"

    # Public so tests can introspect the parity-fixture pin.
    CANONICAL_VOLUME = _VSM_VOLUME
    CANONICAL_DTYPE = _VSM_DTYPE
    CANONICAL_SCALAR = _VSM_SCALAR

    # Kernel-arg slot map for the mlir-aie 2-input-1-output wrapper:
    #   arg3 = in1, arg4 = in2, arg5 = out, arg6 = ctrlpkts, arg7 = trace.
    # Mirrors third_party/mlir-aie/runtime_lib/test_lib/xrt_test_wrapper.h.
    _ARG_IN1 = 3
    _ARG_IN2 = 4
    _ARG_OUT = 5
    _ARG_CTRLPKTS = 6
    _ARG_TRACE = 7

    def __init__(self, backend: NpuBackend | None = None) -> None:
        self._backend = backend or default_backend()

    @classmethod
    def canonical_input(cls) -> np.ndarray:
        """Return the fixture input the host wrapper uses internally.

        ``arange(1, 4097)`` cast to ``int16``. Tests use this to assert
        bit-equality against the NPU output.
        """
        return np.arange(1, cls.CANONICAL_VOLUME + 1, dtype=cls.CANONICAL_DTYPE)

    def __call__(
        self,
        *,
        in1: np.ndarray | None = None,
        scalar: int | None = None,
        n_iters: int = 1,
        warmup: int = 0,
    ) -> np.ndarray:
        """Run the kernel and return the verified ``in1 * scalar`` array.

        Default path is in-process pyxrt. Setting
        binary subprocess path.

        Args:
            in1: optional input override. Must equal the canonical
                fixture (``arange(1, 4097, int16)``).
            scalar: optional scalar override. Must equal the canonical
                scalar (3).
            n_iters: number of NPU iterations. Default 1.
            warmup: warmup iterations not counted in the timing.
                Default 0.

        Raises:
            NpuArtifactsMissingError: artifacts not on disk.
            NpuRunFailed: NPU run / verification failed.
            ValueError: caller passed inputs that don't match the
                canonical fixture.

        Returns:
            np.ndarray (4096,) int16: ``in1 * scalar``.
        """
        if in1 is not None:
            in1 = np.asarray(in1)
            if in1.dtype != self.CANONICAL_DTYPE or in1.shape != (self.CANONICAL_VOLUME,):
                raise ValueError(
                    f"vector_scalar_mul fixture is hardcoded to "
                    f"shape ({self.CANONICAL_VOLUME},) int16; got "
                    f"caller-driven kernels lift this restriction."
                )
            expected = self.canonical_input()
            if not np.array_equal(in1, expected):
                raise ValueError(
                    "vector_scalar_mul in1 must equal "
                    "np.arange(1, 4097, dtype=int16). The host wrapper "
                    "initialises this fixture; the input override is "
                    "only accepted to keep the call signature symmetric "
                    "with cpu/gpu reference paths."
                )
        else:
            in1 = self.canonical_input()

        if scalar is not None and int(scalar) != self.CANONICAL_SCALAR:
            raise ValueError(
                f"vector_scalar_mul scalar is hardcoded to "
                f"{self.CANONICAL_SCALAR}; got scalar={scalar}."
            )
        scalar = self.CANONICAL_SCALAR

        if _force_subprocess():
            avg_us, min_us, max_us, _, _, _ = self._backend.run_host_binary(
                binary=_VSM_BINARY,
                xclbin=_VSM_XCLBIN,
                insts=_VSM_INSTS,
                kernel_name=_VSM_KERNEL_NAME,
                n_iters=n_iters,
                warmup=warmup,
                verify=True,
            )
            # Subprocess path: host binary verifies internally; we
            # reconstruct the array from the canonical reference.
            out = (in1.astype(np.int32) * np.int32(scalar)).astype(self.CANONICAL_DTYPE)
            verified = True
        else:
            # In-process pyxrt path. The kernel takes 2 inputs
            # (in1 + scalar-as-buffer) and writes to bo_out; we also
            # supply ctrlpkts + trace placeholders to match the AIE2P
            # ABI from xrt_test_wrapper.h.
            in2 = np.array([scalar], dtype=np.int32)
            payload = in1.astype(self.CANONICAL_DTYPE).tobytes()
            scalar_payload = in2.tobytes()
            raw_out, avg_us, min_us, max_us = self._backend.run_xclbin(
                xclbin=_VSM_XCLBIN,
                insts=_VSM_INSTS,
                kernel_name=_VSM_KERNEL_NAME,
                in_buffers=[
                    (payload, self._ARG_IN1),
                    (scalar_payload, self._ARG_IN2),
                    # ctrlpkts placeholder: 8 bytes per the C++ wrapper.
                    (bytes(8), self._ARG_CTRLPKTS),
                    # trace placeholder: 1 byte (workaround in wrapper
                    # so kernel.group_id(7) returns a valid memory
                    # group even when tracing is disabled).
                    (bytes(1), self._ARG_TRACE),
                ],
                out_size=in1.nbytes,
                out_arg_index=self._ARG_OUT,
                n_iters=n_iters,
                warmup=warmup,
            )
            out = np.frombuffer(raw_out, dtype=self.CANONICAL_DTYPE).copy()
            # Verify against the CPU reference inline (the in-process
            # path doesn't use the C++ wrapper's PASS!/FAIL! check).
            ref = (in1.astype(np.int32) * np.int32(scalar)).astype(
                self.CANONICAL_DTYPE
            )
            verified = bool(np.array_equal(out, ref))
            if not verified:
                raise NpuRunFailed(
                    -1,
                    f"in-process verify failed: "
                    f"out[:8]={out[:8].tolist()!r} ref[:8]={ref[:8].tolist()!r}",
                    "",
                )

        # Stash timing on the call result so tests + bench can read it
        # without rerunning the kernel.
        self.last_run = NpuRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=n_iters,
            verified=verified,
        )
        return out

# only ships the parity op.
NPU_OPS: dict[str, NpuOp] = {
    "vector_scalar_mul": NpuVectorScalarMul(),
}

def register_npu_op(name: str, op: NpuOp, *, replace: bool = False) -> None:
    """Register an :class:`NpuOp` under ``name``.

    extension point a single, documented function call rather than
    "monkeypatch the dict".
    """
    if not replace and name in NPU_OPS:
        raise KeyError(
            f"NPU op {name!r} already registered (use replace=True to override)"
        )
    if not isinstance(op, NpuOp):
        raise TypeError(f"register_npu_op: op must subclass NpuOp; got {type(op)!r}")
    NPU_OPS[name] = op

def lookup_npu_op(name: str) -> NpuOp:
    """Return the registered op for ``name`` or raise a clear KeyError."""
    if name not in NPU_OPS:
        known = ", ".join(sorted(NPU_OPS)) or "(none)"
        raise KeyError(
            f"unknown NPU op {name!r}; registered ops: {known}. "
            f"Track-side ops are registered by their packages "
            f"."
        )
    return NPU_OPS[name]

__all__ = [
    "NPU_OPS",
    "NpuArtifactsMissingError",
    "NpuBackend",
    "NpuOp",
    "NpuRunFailed",
    "NpuRunResult",
    "NpuVectorScalarMul",
    "default_backend",
    "lookup_npu_op",
    "register_npu_op",
]
