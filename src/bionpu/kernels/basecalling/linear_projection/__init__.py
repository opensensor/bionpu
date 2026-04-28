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

"""Dorado fast Linear projection (CRF head) — NPU op.

Registers ``dorado_fast_linear_projection`` with the NPU dispatch
registry. Implements ``Linear(in=96, out=256, bias=False) + Clamp(-5,5)``
on a fixed time-major input ``(L=334, N=1, 96)`` FP32, returning
``(L=334, N=1, 256)`` FP32.

The CRF head per is bias-free; the post-clamp range [-5, 5] is the
final FP32 logit feed for the host-side beam decoder. The op runs on a
single AIE2P tile with weights resident on tile (W_proj = 256*96 =
24,576 FP32 = 96 KiB — too big for one tile, so we shard along the
output dim into 4 OC groups of 64 outputs each, 24 KiB per group).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    default_backend,
    register_npu_op,
)

T_LSTM = 334
# Production-shape (long) timestep count, paired with the L=1667 LSTM
# artifact. The encoder's production_long path produces 1667 timesteps
# from a 10000-sample raw signal.
L1667_T_LSTM = 1667
HIDDEN = 96
OUT_DIM = 256
OC_GROUP_SIZE = 64
N_OC_GROUPS = OUT_DIM // OC_GROUP_SIZE  # 4
WB_LEN = OUT_DIM * HIDDEN  # 24576 (no bias)
CLAMP_LO = -5.0
CLAMP_HI = 5.0

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_NAME = "dorado_fast_linear_projection"
_ARTIFACT_DIR = _PACKAGE_ROOT / _ARTIFACT_NAME
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

# DESIGN-fusion.md stage 3 (per-timestep fused) artifact paths.
# Wire format: bf16 input + bf16 weight (single 48-KiB slab) +
# bf16 output. Built via `make experiment=fused-perts NPU2=1 all`.
_FUSED_ARTIFACT_NAME = "dorado_fast_linear_projection_fused_perts"
_FUSED_ARTIFACT_DIR = _PACKAGE_ROOT / _FUSED_ARTIFACT_NAME
_FUSED_XCLBIN = _FUSED_ARTIFACT_DIR / "final.xclbin"
_FUSED_INSTS = _FUSED_ARTIFACT_DIR / "insts.bin"
_FUSED_HOST_RUNNER = _FUSED_ARTIFACT_DIR / "host_runner"

# DESIGN-fusion.md stage 4 (single-call-per-dispatch entry symbol)
# artifact paths. Wire format identical to stage 3 (bf16 input +
# bf16 weight slab + bf16 output). Built via
# `make experiment=fused-dispatch NPU2=1 seq=334 all`.
# See linear_projection_fused_dispatch.py for the fallback rationale
# (collapses to stage-3 MLIR shape due to IRON Kernel ABI limits).
_FUSED_DISPATCH_ARTIFACT_DIR = (
    _PACKAGE_ROOT / "dorado_fast_linear_projection_fused_dispatch"
)
_FUSED_DISPATCH_XCLBIN = _FUSED_DISPATCH_ARTIFACT_DIR / "final.xclbin"
_FUSED_DISPATCH_INSTS = _FUSED_DISPATCH_ARTIFACT_DIR / "insts.bin"
_FUSED_DISPATCH_HOST_RUNNER = _FUSED_DISPATCH_ARTIFACT_DIR / "host_runner"

# Match plain decimal AND scientific notation (e.g. "1.76837e+06us").
_RE_AVG = re.compile(
    r"^Avg NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MIN = re.compile(
    r"^Min NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MAX = re.compile(
    r"^Max NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)

@dataclass(frozen=True)
class LinearProjRunResult:
    output: np.ndarray  # (L, N=1, 256) FP32
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int


# Kernel-arg slots for both linear_projection runners (per_group + fused).
# Mirrors runner.cpp / runner_fused.cpp: kernel(opcode, bo_instr,
# instr_size, bo_input, bo_weight, bo_output, bo_trace).
_ARG_INPUT = 3
_ARG_WEIGHT = 4
_ARG_OUTPUT = 5
_ARG_TRACE = 6


def _resolve_dispatch_impl(impl: str | None) -> str:
    """Return the dispatch implementation: 'subprocess' (default) or 'pyxrt'.

    Priority: explicit ``impl`` arg, else env
    ``BIONPU_LINEAR_PROJECTION_DISPATCH``, else ``"subprocess"``.
    Raises :class:`ValueError` for unknown values.
    """
    chosen = (impl or os.environ.get(
        "BIONPU_LINEAR_PROJECTION_DISPATCH", "subprocess"
    )).strip().lower()
    if chosen not in ("subprocess", "pyxrt"):
        raise ValueError(
            f"BIONPU_LINEAR_PROJECTION_DISPATCH / _impl must be one of "
            f"'subprocess' or 'pyxrt'; got {chosen!r}."
        )
    return chosen


def _run_pyxrt_linear_projection(
    *,
    xclbin_path: Path,
    insts_path: Path,
    input_bytes: bytes,
    weight_bytes: bytes,
    out_size: int,
    n_iters: int,
    warmup: int,
) -> tuple[bytes, float, float, float]:
    """Drive a linear-projection xclbin in-process via pyxrt.

    Used by both the per-group (FP32) and fused-perts (bf16) ops.
    The kernel arg layout is identical for both runners (see
    runner.cpp / runner_fused.cpp):
        arg3 = input, arg4 = weight, arg5 = output, arg6 = trace.
    Wire-format byte-packing is the caller's responsibility.

    Returns ``(raw_output_bytes, avg_us, min_us, max_us)``.
    """
    backend = default_backend()
    return backend.run_xclbin(
        xclbin=xclbin_path,
        insts=insts_path,
        kernel_name="MLIR_AIE",
        in_buffers=[
            (input_bytes, _ARG_INPUT),
            (weight_bytes, _ARG_WEIGHT),
            # Trace placeholder: 1 byte (matches the C++ runner, which
            # constructs bo_trace with size 1 to keep group_id(6) valid
            # when tracing is disabled).
            (bytes(1), _ARG_TRACE),
        ],
        out_size=int(out_size),
        out_arg_index=_ARG_OUTPUT,
        n_iters=int(n_iters),
        warmup=int(warmup),
    )

def _xrt_env() -> dict[str, str]:
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

def _pack_input(x: np.ndarray, seq_len: int = T_LSTM) -> np.ndarray:
    """``(L, N=1, 96)`` -> ``(L * 96,)`` FP32."""
    if x.shape != (seq_len, 1, HIDDEN):
        raise ValueError(
            f"linear_projection input must have shape ({seq_len}, 1, {HIDDEN}); "
            f"got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(f"linear_projection input must be FP32; got {x.dtype}")
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)

def _pack_wb(weight: np.ndarray, seq_len: int = T_LSTM) -> np.ndarray:
    """Pack ``(256, 96)`` weight into the host buffer.

    The IRON kernel runs ``seq_len * N_OC_GROUPS`` slabs per call; each
    slab is one OC group (64 rows x 96 cols = 6144 FP32). The host
    materialises ``seq_len`` repetitions of the 4-group cycle:
    ``seq_len * N_OC_GROUPS * OC_GROUP_SIZE * HIDDEN`` FP32 (~31 MB at
    L=334; ~155 MB at L=1667). A future revision can use BD-level
    repeat patterns to drop the bloat.
    """
    if weight.shape != (OUT_DIM, HIDDEN):
        raise ValueError(
            f"linear_projection weight must have shape "
            f"({OUT_DIM}, {HIDDEN}); got {weight.shape}"
        )
    W = weight.astype(np.float32, copy=False)
    # One cycle: (N_OC_GROUPS, OC_GROUP_SIZE, HIDDEN)
    cycle = W.reshape(N_OC_GROUPS, OC_GROUP_SIZE, HIDDEN)
    repeated = np.broadcast_to(
        cycle, (seq_len, N_OC_GROUPS, OC_GROUP_SIZE, HIDDEN)
    ).reshape(-1)
    return np.ascontiguousarray(repeated, dtype=np.float32)

def _unpack_output(flat: np.ndarray, seq_len: int = T_LSTM) -> np.ndarray:
    """``(L * 256,)`` -> ``(L, 1, 256)`` FP32."""
    expected = seq_len * OUT_DIM
    if flat.size != expected:
        raise ValueError(
            f"linear_projection output length mismatch: got {flat.size}, "
            f"expected {expected}"
        )
    return flat.reshape(seq_len, 1, OUT_DIM)

class DoradoFastLinearProjection(NpuOp):
    """CRF linear head + clamp on AIE2P.

    Two on-disk artifacts:

    - ``seq_len=T_LSTM`` (=334, default): legacy short-shape artifact.
      Loaded from ``_npu_artifacts/dorado_fast_linear_projection/``.
    - ``seq_len=L1667_T_LSTM`` (=1667): production-shape (long)
      artifact paired with the LSTM L=1667 path. Loaded from
      ``_npu_artifacts/dorado_fast_linear_projection_L1667/``.
    """

    name = "dorado_fast_linear_projection"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, OUT_DIM)
    HIDDEN_SIZE = HIDDEN
    OUTPUT_SIZE = OUT_DIM
    SEQ_LEN = T_LSTM

    def __init__(self, seq_len: int = T_LSTM) -> None:
        self.seq_len = int(seq_len)
        self.last_run: LinearProjRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_XCLBIN, _INSTS, _HOST_RUNNER))

    @property
    def artifact_dir(self) -> Path:
        if self.seq_len == T_LSTM:
            return _ARTIFACT_DIR
        return _PACKAGE_ROOT / f"{_ARTIFACT_NAME}_L{self.seq_len}"

    @property
    def xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def host_runner(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_available(self) -> bool:
        return all(
            p.exists() for p in (self.xclbin, self.insts, self.host_runner)
        )

    def __call__(
        self,
        *,
        x: np.ndarray,
        weight: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 60.0,
        _impl: str | None = None,
    ) -> np.ndarray:
        impl = _resolve_dispatch_impl(_impl)
        xclbin = self.xclbin
        insts = self.insts
        host_runner = self.host_runner
        artifact_dir = self.artifact_dir
        seq_len = self.seq_len
        # Subprocess-only artifacts: host_runner is required for the
        # subprocess path; pyxrt path needs only xclbin + insts.
        required = (xclbin, insts) if impl == "pyxrt" else (
            xclbin, insts, host_runner,
        )
        for p in required:
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name} "
                    f"(seq_len={seq_len}): {p}. See "
                    f"{artifact_dir}/MANIFEST.md to rebuild."
                )

        x_packed = _pack_input(x, seq_len=seq_len)
        wb_packed = _pack_wb(weight, seq_len=seq_len)

        if impl == "pyxrt":
            out_size = seq_len * OUT_DIM * 4  # FP32
            raw_out, avg_us, min_us, max_us = _run_pyxrt_linear_projection(
                xclbin_path=xclbin,
                insts_path=insts,
                input_bytes=x_packed.tobytes(),
                weight_bytes=wb_packed.tobytes(),
                out_size=out_size,
                n_iters=n_iters,
                warmup=warmup,
            )
            flat = np.frombuffer(raw_out, dtype=np.float32).copy()
            out = _unpack_output(flat, seq_len=seq_len)
            self.last_run = LinearProjRunResult(
                output=out,
                avg_us=avg_us,
                min_us=min_us,
                max_us=max_us,
                n_iters=int(n_iters),
            )
            return out

        with tempfile.TemporaryDirectory(prefix="t51_linear_") as td:
            tdp = Path(td)
            x_path = tdp / "input.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(host_runner),
                "-x", str(xclbin),
                "-i", str(insts),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(seq_len),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv fully controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
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
            avg_us = float(m_avg.group(1))
            min_us = float(m_min.group(1))
            max_us = float(m_max.group(1))

            flat = np.fromfile(out_path, dtype=np.float32)
        out = _unpack_output(flat, seq_len=seq_len)
        self.last_run = LinearProjRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

LINEAR_PROJECTION_TILE_MEMORY_USED_BYTES: int | None = None

register_npu_op(
    "dorado_fast_linear_projection",
    DoradoFastLinearProjection(),
)

# --------------------------------------------------------------------------- #
# DESIGN-fusion.md stage 3: per-timestep fused (bf16 weights)
# --------------------------------------------------------------------------- #
#
# Same math (Linear(96 -> 256, bias=False) + clamp[-5, 5]) but bf16
# weights/inputs/outputs and ONE kernel call per timestep (334) instead
# of one call per (timestep, OC group) tuple (1336). Reuses the
# lstm_cell_bf16 bf16 conversion helpers — same precision contract.

# Local import to avoid circular-import surprises if either of the two
# packages is loaded before the other; cycle is benign because neither
# module pulls anything from the other at module load.
from bionpu.kernels.basecalling.lstm_cell_bf16 import (  # noqa: E402
    bf16_bytes_to_fp32,
    fp32_to_bf16_bytes,
)


def _pack_input_bf16(x: np.ndarray, seq_len: int = T_LSTM) -> np.ndarray:
    """``(L, N=1, 96)`` fp32 -> ``(L * 96,)`` bf16-as-uint16."""
    if x.shape != (seq_len, 1, HIDDEN):
        raise ValueError(
            f"linear_projection input must have shape ({seq_len}, 1, {HIDDEN}); "
            f"got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(
            f"linear_projection input must be FP32 (host-side fp32 -> bf16 "
            f"narrowing happens here); got {x.dtype}"
        )
    flat_fp32 = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    return fp32_to_bf16_bytes(flat_fp32)


def _pack_wb_bf16(weight: np.ndarray) -> np.ndarray:
    """Pack ``(256, 96)`` weight into the kernel's flat bf16 slab.

    Stage-3 wire format: ONE 48-KiB bf16 slab per dispatch (no
    per-timestep repetition; the kernel acquires the slab once at
    start and holds it for the whole 334-timestep loop).
    Layout matches ``torch.nn.Linear``: ``W[oc, j]`` at
    ``slab[oc * HIDDEN + j]`` (row-major, no transpose).
    """
    if weight.shape != (OUT_DIM, HIDDEN):
        raise ValueError(
            f"linear_projection weight must have shape "
            f"({OUT_DIM}, {HIDDEN}); got {weight.shape}"
        )
    W = weight.astype(np.float32, copy=False)
    flat_fp32 = np.ascontiguousarray(W.reshape(-1), dtype=np.float32)
    return fp32_to_bf16_bytes(flat_fp32)


def _unpack_output_bf16(buf_u16: np.ndarray, seq_len: int = T_LSTM) -> np.ndarray:
    """``(L * 256,)`` bf16-as-uint16 -> ``(L, 1, 256)`` fp32."""
    expected = seq_len * OUT_DIM
    if buf_u16.size != expected:
        raise ValueError(
            f"linear_projection fused output length mismatch: got "
            f"{buf_u16.size}, expected {expected}"
        )
    return bf16_bytes_to_fp32(buf_u16, (seq_len, 1, OUT_DIM))


class DoradoFastLinearProjectionFusedPerts(NpuOp):
    """Stage-3 fused CRF linear head + clamp on AIE2P.

    DESIGN-fusion.md stage 3: bf16 weights, one kernel call per
    timestep, full 256-output GEMM computed by looping over the 4 OC
    groups internally with weight-stationary access. Target wall time:
    ~141 ms (4x faster than the per-group artifact's ~564 ms in
    results/basecalling/b-m6/measurements.json).

    Two on-disk artifacts:

    - ``seq_len=T_LSTM`` (=334, default): legacy short-shape artifact.
    - ``seq_len=L1667_T_LSTM`` (=1667): production-shape (long)
      artifact paired with the LSTM L=1667 path.
    """

    name = "dorado_fast_linear_projection_fused_perts"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, OUT_DIM)
    HIDDEN_SIZE = HIDDEN
    OUTPUT_SIZE = OUT_DIM
    SEQ_LEN = T_LSTM

    def __init__(self, seq_len: int = T_LSTM) -> None:
        self.seq_len = int(seq_len)
        self.last_run: LinearProjRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(
            p.exists()
            for p in (_FUSED_XCLBIN, _FUSED_INSTS, _FUSED_HOST_RUNNER)
        )

    @property
    def artifact_dir(self) -> Path:
        if self.seq_len == T_LSTM:
            return _FUSED_ARTIFACT_DIR
        return _PACKAGE_ROOT / f"{_FUSED_ARTIFACT_NAME}_L{self.seq_len}"

    @property
    def xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def host_runner(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_available(self) -> bool:
        return all(
            p.exists() for p in (self.xclbin, self.insts, self.host_runner)
        )

    def __call__(
        self,
        *,
        x: np.ndarray,
        weight: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 60.0,
        _impl: str | None = None,
    ) -> np.ndarray:
        impl = _resolve_dispatch_impl(_impl)
        xclbin = self.xclbin
        insts = self.insts
        host_runner = self.host_runner
        artifact_dir = self.artifact_dir
        seq_len = self.seq_len
        required = (
            (xclbin, insts)
            if impl == "pyxrt"
            else (xclbin, insts, host_runner)
        )
        for p in required:
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name} "
                    f"(seq_len={seq_len}): {p}. Build via "
                    f"`make experiment=fused-perts NPU2=1 seq={seq_len} all` "
                    f"in this kernel directory, then copy the build outputs "
                    f"to {artifact_dir}."
                )

        x_packed = _pack_input_bf16(x, seq_len=seq_len)
        wb_packed = _pack_wb_bf16(weight)

        if impl == "pyxrt":
            out_size = seq_len * OUT_DIM * 2  # bf16
            raw_out, avg_us, min_us, max_us = _run_pyxrt_linear_projection(
                xclbin_path=xclbin,
                insts_path=insts,
                input_bytes=x_packed.tobytes(),
                weight_bytes=wb_packed.tobytes(),
                out_size=out_size,
                n_iters=n_iters,
                warmup=warmup,
            )
            buf_u16 = np.frombuffer(raw_out, dtype=np.uint16).copy()
            out = _unpack_output_bf16(buf_u16, seq_len=seq_len)
            self.last_run = LinearProjRunResult(
                output=out,
                avg_us=avg_us,
                min_us=min_us,
                max_us=max_us,
                n_iters=int(n_iters),
            )
            return out

        with tempfile.TemporaryDirectory(prefix="t51_linear_fused_") as td:
            tdp = Path(td)
            x_path = tdp / "input.bf16.bin"
            wb_path = tdp / "wb.bf16.bin"
            out_path = tdp / "output.bf16.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(host_runner),
                "-x", str(xclbin),
                "-i", str(insts),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(seq_len),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv fully controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
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
            avg_us = float(m_avg.group(1))
            min_us = float(m_min.group(1))
            max_us = float(m_max.group(1))

            buf_u16 = np.fromfile(out_path, dtype=np.uint16)
        out = _unpack_output_bf16(buf_u16, seq_len=seq_len)
        self.last_run = LinearProjRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out


register_npu_op(
    "dorado_fast_linear_projection_fused_perts",
    DoradoFastLinearProjectionFusedPerts(),
)


# --------------------------------------------------------------------------- #
# DESIGN-fusion.md stage 4: single-call-per-dispatch entry symbol
# --------------------------------------------------------------------------- #
#
# Same math + same wire format as stage 3 (DoradoFastLinearProjectionFusedPerts);
# distinct artifact directory and distinct kernel symbol. The IRON
# Kernel ABI does not support objectfifo-handle args, so stage 4
# collapses to stage 3's MLIR shape (one func.call per scf.for
# iteration; one acquire/release pair per timestep on input/output;
# weight acquired once). See linear_projection_fused_dispatch.py for
# the full fallback rationale and gaps.yaml for the entry.


class DoradoFastLinearProjectionFusedDispatch(NpuOp):
    """Stage-4 fused CRF linear head + clamp on AIE2P.

    DESIGN-fusion.md stage 4: distinct kernel symbol with an inner-
    loop micro-optimisation (aie::reduce_add for the per-OC scalar
    reduction tail). MLIR shape is identical to stage 3 because the
    IRON Kernel ABI doesn't accept objectfifo-handle args (see
    linear_projection_fused_dispatch.py top-of-file rationale).

    Marginal speedup over stage 3 is uncertain — stage 3 already
    silicon-validated at 9.68 ms/iter, far exceeding the 141 ms
    DESIGN-fusion.md prediction. The realistic stage-4 ceiling is
    roughly 9.68 ms divided by a small constant (collapsed scalar
    reduction tail) — a few hundred us at best, possibly indistinguishable
    from stage 3 within run-to-run variance.
    """

    name = "dorado_fast_linear_projection_fused_dispatch"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, OUT_DIM)
    HIDDEN_SIZE = HIDDEN
    OUTPUT_SIZE = OUT_DIM
    SEQ_LEN = T_LSTM

    def __init__(self) -> None:
        self.last_run: LinearProjRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(
            p.exists()
            for p in (
                _FUSED_DISPATCH_XCLBIN,
                _FUSED_DISPATCH_INSTS,
                _FUSED_DISPATCH_HOST_RUNNER,
            )
        )

    def __call__(
        self,
        *,
        x: np.ndarray,
        weight: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 60.0,
    ) -> np.ndarray:
        for p in (
            _FUSED_DISPATCH_XCLBIN,
            _FUSED_DISPATCH_INSTS,
            _FUSED_DISPATCH_HOST_RUNNER,
        ):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. Build via "
                    f"`make experiment=fused-dispatch NPU2=1 seq={T_LSTM} all`"
                    f" in this kernel directory, then copy the build outputs "
                    f"to {_FUSED_DISPATCH_ARTIFACT_DIR}."
                )

        # Wire format identical to stage 3 — reuse the same packers.
        x_packed = _pack_input_bf16(x)
        wb_packed = _pack_wb_bf16(weight)

        with tempfile.TemporaryDirectory(prefix="t51_linear_dispatch_") as td:
            tdp = Path(td)
            x_path = tdp / "input.bf16.bin"
            wb_path = tdp / "wb.bf16.bin"
            out_path = tdp / "output.bf16.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(_FUSED_DISPATCH_HOST_RUNNER),
                "-x", str(_FUSED_DISPATCH_XCLBIN),
                "-i", str(_FUSED_DISPATCH_INSTS),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(T_LSTM),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv fully controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
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
            avg_us = float(m_avg.group(1))
            min_us = float(m_min.group(1))
            max_us = float(m_max.group(1))

            buf_u16 = np.fromfile(out_path, dtype=np.uint16)
        out = _unpack_output_bf16(buf_u16)
        self.last_run = LinearProjRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out


register_npu_op(
    "dorado_fast_linear_projection_fused_dispatch",
    DoradoFastLinearProjectionFusedDispatch(),
)


# --------------------------------------------------------------------------- #
# Batched stage 3: process N_CHUNKS chunks per silicon dispatch
# --------------------------------------------------------------------------- #
#
# Purely a host-amortisation trick. The C++ kernel is unchanged; the
# IRON-side outer loop walks N_CHUNKS chunks of L timesteps in a single
# silicon dispatch. Single xclbin load + single weight slab + single
# XRT submit collapses (some of) the ~100 ms/chunk host-side dispatch
# overhead measured in B3b.
#
# Wire format:
#   - input:  N_CHUNKS * L * HIDDEN bf16 (chunks concatenated)
#   - weight: WB_LEN bf16 (single 48-KiB slab; SHARED across batch)
#   - output: N_CHUNKS * L * OUT_DIM bf16 (chunks concatenated)
#
# Op signature is `op(x_batch=..., weight=..., n_iters=..., warmup=...)`
# with `x_batch` shape `(N_CHUNKS, L, 1, HIDDEN)`. Output:
# `(N_CHUNKS, L, 1, OUT_DIM)`. The artifact is selected by the
# baked-in batch size on disk; the op infers N_CHUNKS from the input.

# Artifact directory naming mirrors the Makefile suffix so n=1, 2, 4,
# 8 builds coexist:
#   _npu_artifacts/dorado_fast_linear_projection_fused_batched_n{N}/
_FUSED_BATCHED_ARTIFACT_DIR_TEMPLATE = (
    _PACKAGE_ROOT / "dorado_fast_linear_projection_fused_batched_n{n}"
)


def _fused_batched_artifact_paths(n_chunks: int):
    base = Path(
        str(_FUSED_BATCHED_ARTIFACT_DIR_TEMPLATE).format(n=int(n_chunks))
    )
    return base, base / "final.xclbin", base / "insts.bin", base / "host_runner"


def _pack_input_bf16_batched(x_batch: np.ndarray, n_chunks: int) -> np.ndarray:
    """``(N_CHUNKS, L, N=1, 96)`` fp32 -> ``(N_CHUNKS * L * 96,)`` bf16-as-uint16."""
    expected_shape = (n_chunks, T_LSTM, 1, HIDDEN)
    if x_batch.shape != expected_shape:
        raise ValueError(
            f"linear_projection batched input must have shape "
            f"{expected_shape}; got {x_batch.shape}"
        )
    if x_batch.dtype != np.float32:
        raise ValueError(
            f"linear_projection batched input must be FP32; got "
            f"{x_batch.dtype}"
        )
    flat_fp32 = np.ascontiguousarray(x_batch.reshape(-1), dtype=np.float32)
    return fp32_to_bf16_bytes(flat_fp32)


def _unpack_output_bf16_batched(
    buf_u16: np.ndarray, n_chunks: int,
) -> np.ndarray:
    """``(N_CHUNKS * L * 256,)`` bf16-as-uint16 -> ``(N_CHUNKS, L, 1, 256)`` fp32."""
    expected = n_chunks * T_LSTM * OUT_DIM
    if buf_u16.size != expected:
        raise ValueError(
            f"linear_projection batched output length mismatch: got "
            f"{buf_u16.size}, expected {expected}"
        )
    return bf16_bytes_to_fp32(buf_u16, (n_chunks, T_LSTM, 1, OUT_DIM))


class DoradoFastLinearProjectionFusedBatched(NpuOp):
    """Batched stage-3 fused CRF linear head + clamp on AIE2P.

    Same math + precision contract as DoradoFastLinearProjectionFusedPerts;
    chunk-outer batching at the IRON layer amortises host-side dispatch
    overhead (xclbin load, BO alloc, XRT submit, output sync) across
    N_CHUNKS chunks. Single silicon dispatch processes N_CHUNKS chunks
    of L=334 timesteps with a SHARED 48-KiB bf16 weight slab.

    The artifact suffix bakes in the chunk count so n=1, 2, 4, 8 builds
    coexist on disk. The op infers N_CHUNKS from the input shape and
    selects the matching on-disk artifact.
    """

    name = "dorado_fast_linear_projection_fused_perts_batched"

    HIDDEN_SIZE = HIDDEN
    OUTPUT_SIZE = OUT_DIM
    SEQ_LEN = T_LSTM
    SUPPORTED_N_CHUNKS = (1, 2, 4, 8)

    def __init__(self) -> None:
        self.last_run: LinearProjRunResult | None = None

    @classmethod
    def artifacts_present(cls, n_chunks: int = 1) -> bool:
        if n_chunks not in cls.SUPPORTED_N_CHUNKS:
            return False
        _, x, i, h = _fused_batched_artifact_paths(n_chunks)
        return all(p.exists() for p in (x, i, h))

    def __call__(
        self,
        *,
        x_batch: np.ndarray,
        weight: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 120.0,
    ) -> np.ndarray:
        if x_batch.ndim != 4:
            raise ValueError(
                "linear_projection batched x_batch must have rank 4 "
                f"(N_CHUNKS, L, 1, HIDDEN); got shape {x_batch.shape}"
            )
        n_chunks = int(x_batch.shape[0])
        if n_chunks not in self.SUPPORTED_N_CHUNKS:
            raise ValueError(
                f"N_CHUNKS={n_chunks} not in supported set "
                f"{self.SUPPORTED_N_CHUNKS}; rebuild with "
                f"BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS={n_chunks} make "
                f"experiment=fused-batched seq={T_LSTM} all if needed."
            )

        artifact_dir, xclbin, insts, host_runner = _fused_batched_artifact_paths(
            n_chunks
        )
        for p in (xclbin, insts, host_runner):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name} (n_chunks={n_chunks}): "
                    f"{p}. Build via "
                    f"`BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS={n_chunks} "
                    f"make experiment=fused-batched NPU2=1 seq={T_LSTM} all` "
                    f"in this kernel directory, then copy the build outputs "
                    f"to {artifact_dir}."
                )

        x_packed = _pack_input_bf16_batched(x_batch, n_chunks)
        # Weight is the SAME single 48-KiB slab as stage 3 (shared).
        wb_packed = _pack_wb_bf16(weight)

        with tempfile.TemporaryDirectory(prefix="t51_linear_batched_") as td:
            tdp = Path(td)
            x_path = tdp / "input.bf16.bin"
            wb_path = tdp / "wb.bf16.bin"
            out_path = tdp / "output.bf16.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(host_runner),
                "-x", str(xclbin),
                "-i", str(insts),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(T_LSTM),
                "--n-chunks", str(int(n_chunks)),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv fully controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
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
            avg_us = float(m_avg.group(1))
            min_us = float(m_min.group(1))
            max_us = float(m_max.group(1))

            buf_u16 = np.fromfile(out_path, dtype=np.uint16)
        out = _unpack_output_bf16_batched(buf_u16, n_chunks)
        self.last_run = LinearProjRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out


register_npu_op(
    "dorado_fast_linear_projection_fused_perts_batched",
    DoradoFastLinearProjectionFusedBatched(),
)


__all__ = [
    "CLAMP_HI",
    "CLAMP_LO",
    "DoradoFastLinearProjection",
    "DoradoFastLinearProjectionFusedBatched",
    "DoradoFastLinearProjectionFusedDispatch",
    "DoradoFastLinearProjectionFusedPerts",
    "HIDDEN",
    "L1667_T_LSTM",
    "LINEAR_PROJECTION_TILE_MEMORY_USED_BYTES",
    "LinearProjRunResult",
    "N_OC_GROUPS",
    "OC_GROUP_SIZE",
    "OUT_DIM",
    "T_LSTM",
    "WB_LEN",
    "_pack_input",
    "_pack_input_bf16",
    "_pack_input_bf16_batched",
    "_pack_wb",
    "_pack_wb_bf16",
    "_unpack_output",
    "_unpack_output_bf16",
    "_unpack_output_bf16_batched",
]
