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

"""Dorado fast LSTM cell — bf16 NPU op.

Registers ``dorado_fast_lstm_cell_bf16`` with :data:`bionpu.dispatch.NPU_OPS`.
Mirrors the public surface of 's ``dorado_fast_lstm_cell`` (FP32
scalar) but lowers the per-timestep math to AIE2P bf16 vector intrinsics
(``aie::tanh<bfloat16>`` etc.). Targets : 's 3rd-order
Padé tanh produces ~3e-3 max error per call which cumulatively drifts
to end-to-end max-abs 2.303 across 5 LSTMs × 200 timesteps.

Op contract (FP32 in, FP32 out — keeps the encoder's FP32 interface):
    inputs:  x (L, 1, 96) fp32, weights and biases all fp32
    outputs: y (L, 1, 96) fp32
    internal: bf16 throughout the kernel; FP32→bf16 cast happens host-
              side once, bf16→FP32 cast on the output once.

Same DMA topology as (input + folded-bias-prefix weight stream
+ output) — bf16 doesn't free a DMA channel; the bias-folding
workaround for still applies.

Tile-memory used (parsed from
``build/aie_L334.mlir.prj/main_core_0_2.ld.script``):
    stack 0x400 + 2 * 0x2A00 (weight depth-2 buffer) + 4 * 0xC0
    (input/output depth-2 buffers) + .bss (h, c, gate_acc, bias_cache;
    all bf16)
    ≈ 23.5 KiB on a 64 KiB AIE2P tile (vs 's 51 KiB FP32 footprint).
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
    register_npu_op,
)

# Pinned LSTM cell geometry — same as (interface-compatible).
T_LSTM = 334
HIDDEN = 96
INPUT_DIM = 96
N_GATES = 4

# Per-gate weight slab sizes (counts in element-of-type, regardless of
# whether the type is fp32 or bf16).
WIH_GATE = HIDDEN * INPUT_DIM  # 9216 per gate
WHH_GATE = HIDDEN * HIDDEN     # 9216 per gate
BIH_GATE = HIDDEN              # 96 per gate
BHH_GATE = HIDDEN              # 96 per gate

WB_LEN = N_GATES * (WIH_GATE + WHH_GATE) + N_GATES * (BIH_GATE + BHH_GATE)
# 4*(9216 + 9216) + 4*(96 + 96) = 74,496 elements (148 KiB at bf16).

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_DIR = _PACKAGE_ROOT / "dorado_fast_lstm_cell_bf16"
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

_RE_AVG = re.compile(
    r"^Avg NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MIN = re.compile(
    r"^Min NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MAX = re.compile(
    r"^Max NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)

# Tile-local memory used by the AIE compute tile, parsed from the
# linker script `main_core_*.ld.script` produced by aiecc. Buffers
# only (stack + weight depth-2 + input depth-2 + output depth-2);
# .bss section adds a few KB of static state (h_state/c_state/
# gate_acc/bias_cache).
LSTM_CELL_BF16_TILE_MEMORY_USED_BYTES = 0x400 + 2 * 0x2A00 + 4 * 0xC0  # 22784

@dataclass(frozen=True)
class LstmCellBf16RunResult:
    output: np.ndarray  # (L, N=1, 96) FP32 (re-cast from bf16)
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int

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

# ---------------------------------------------------------------------------
# bf16 conversion helpers
# ---------------------------------------------------------------------------

def _try_bf16_dtype():
    """Return ml_dtypes.bfloat16 or None if not installed."""
    try:
        from ml_dtypes import bfloat16  # noqa: WPS433
        return bfloat16
    except Exception:  # pragma: no cover - ml_dtypes always present in env
        return None

def fp32_to_bf16_bytes(x: np.ndarray) -> np.ndarray:
    """Convert a contiguous fp32 numpy array to bf16, returned as a
    uint16 view (the wire format the kernel + host runner expect).

    Uses :mod:`ml_dtypes` when available (round-to-nearest-even per
    IEEE 754); falls back to a manual round-to-nearest-even cast.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    bf = _try_bf16_dtype()
    if bf is not None:
        return x.astype(bf).view(np.uint16).reshape(x.shape)
    # Fallback: manual round-to-nearest-even.
    u = x.view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).reshape(x.shape)

def bf16_bytes_to_fp32(buf_u16: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert a uint16-on-disk bf16 buffer back to fp32 numpy."""
    bf = _try_bf16_dtype()
    if bf is not None:
        return buf_u16.view(bf).astype(np.float32).reshape(shape)
    # Fallback: bf16 -> fp32 is just zero-padding the mantissa.
    expanded = (buf_u16.astype(np.uint32) << 16).view(np.float32)
    return expanded.reshape(shape)

def _pack_input_bf16(x: np.ndarray) -> np.ndarray:
    """``(L, N=1, 96)`` fp32 -> ``(L * 96,)`` bf16-as-uint16 contiguous."""
    if x.shape != (T_LSTM, 1, INPUT_DIM):
        raise ValueError(
            f"lstm_cell_bf16 input must have shape ({T_LSTM}, 1, {INPUT_DIM}); "
            f"got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(
            f"lstm_cell_bf16 input must be FP32 (FP32->bf16 cast happens "
            f"in this op); got dtype={x.dtype}"
        )
    flat_fp32 = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    return fp32_to_bf16_bytes(flat_fp32)

def _pack_wb_bf16(
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
) -> np.ndarray:
    """Pack PyTorch LSTM weights into the kernel's flat WB buffer at bf16.

    Same gate-major layout as 's :func:`bionpu.kernels.basecalling
    .lstm_cell._pack_wb`; only the element type changes (fp32 → bf16
    via :func:`fp32_to_bf16_bytes`).
    """
    if weight_ih.shape != (4 * HIDDEN, INPUT_DIM):
        raise ValueError(
            f"weight_ih must have shape ({4*HIDDEN}, {INPUT_DIM}); "
            f"got {weight_ih.shape}"
        )
    if weight_hh.shape != (4 * HIDDEN, HIDDEN):
        raise ValueError(
            f"weight_hh must have shape ({4*HIDDEN}, {HIDDEN}); "
            f"got {weight_hh.shape}"
        )
    if bias_ih.shape != (4 * HIDDEN,):
        raise ValueError(
            f"bias_ih must have shape ({4*HIDDEN},); got {bias_ih.shape}"
        )
    if bias_hh.shape != (4 * HIDDEN,):
        raise ValueError(
            f"bias_hh must have shape ({4*HIDDEN},); got {bias_hh.shape}"
        )
    Wih = weight_ih.astype(np.float32, copy=False)
    Whh = weight_hh.astype(np.float32, copy=False)
    bih = bias_ih.astype(np.float32, copy=False)
    bhh = bias_hh.astype(np.float32, copy=False)

    parts: list[np.ndarray] = []
    for g in range(N_GATES):
        parts.append(Wih[g * HIDDEN : (g + 1) * HIDDEN].reshape(-1))
        parts.append(Whh[g * HIDDEN : (g + 1) * HIDDEN].reshape(-1))
    for g in range(N_GATES):
        parts.append(bih[g * HIDDEN : (g + 1) * HIDDEN])
        parts.append(bhh[g * HIDDEN : (g + 1) * HIDDEN])
    flat_fp32 = np.concatenate(parts).astype(np.float32)
    return fp32_to_bf16_bytes(flat_fp32)

def _unpack_output_bf16(buf_u16: np.ndarray) -> np.ndarray:
    """``(L * 96,)`` bf16-as-uint16 -> ``(L, 1, 96)`` fp32."""
    expected = T_LSTM * HIDDEN
    if buf_u16.size != expected:
        raise ValueError(
            f"lstm output length mismatch: got {buf_u16.size}, expected {expected}"
        )
    return bf16_bytes_to_fp32(buf_u16, (T_LSTM, 1, HIDDEN))

class DoradoFastLstmCellBf16(NpuOp):
    """Single bf16 LSTM cell forward over a fixed time sequence."""

    name = "dorado_fast_lstm_cell_bf16"

    INPUT_SHAPE = (T_LSTM, 1, INPUT_DIM)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    HIDDEN_SIZE = HIDDEN
    SEQ_LEN = T_LSTM
    PRECISION = "bf16"

    def __init__(self) -> None:
        self.last_run: LstmCellBf16RunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_XCLBIN, _INSTS, _HOST_RUNNER))

    def __call__(
        self,
        *,
        x: np.ndarray,
        weight_ih: np.ndarray,
        weight_hh: np.ndarray,
        bias_ih: np.ndarray,
        bias_hh: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 120.0,
    ) -> np.ndarray:
        for p in (_XCLBIN, _INSTS, _HOST_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{_ARTIFACT_DIR}/MANIFEST.md to rebuild."
                )

        x_packed = _pack_input_bf16(x)
        wb_packed = _pack_wb_bf16(weight_ih, weight_hh, bias_ih, bias_hh)

        with tempfile.TemporaryDirectory(prefix="t64_lstm_bf16_") as td:
            tdp = Path(td)
            x_path = tdp / "input.bf16.bin"
            wb_path = tdp / "wb.bf16.bin"
            out_path = tdp / "output.bf16.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(_HOST_RUNNER),
                "-x", str(_XCLBIN),
                "-i", str(_INSTS),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(T_LSTM),
                "--hidden", str(HIDDEN),
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
        self.last_run = LstmCellBf16RunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

register_npu_op(
    "dorado_fast_lstm_cell_bf16",
    DoradoFastLstmCellBf16(),
)

__all__ = [
    "BIH_GATE",
    "BHH_GATE",
    "DoradoFastLstmCellBf16",
    "HIDDEN",
    "INPUT_DIM",
    "LSTM_CELL_BF16_TILE_MEMORY_USED_BYTES",
    "LstmCellBf16RunResult",
    "N_GATES",
    "T_LSTM",
    "WB_LEN",
    "WHH_GATE",
    "WIH_GATE",
    "_pack_input_bf16",
    "_pack_wb_bf16",
    "_unpack_output_bf16",
    "bf16_bytes_to_fp32",
    "fp32_to_bf16_bytes",
]
