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

"""Dorado fast LSTM cell — NPU op.

Registers ``dorado_fast_lstm_cell`` with :data:`bionpu.dispatch.NPU_OPS`.
Implements one ``nn.LSTM(input=96, hidden=96, num_layers=1, batch_first=False,
bidirectional=False)`` forward pass on a fixed time-major input
``(L=334, N=1, 96)`` FP32, returning ``(L, N, 96)`` FP32. The reverse-
direction LSTM in Bonito's alternating-direction stack is implemented as
``flip(input, dim=0)`` -> forward LSTM -> ``flip(output, dim=0)`` per
``architecture.md §2`` (and the ONNX export in ).

The kernel is FP32 scalar on a single AIE2P tile, with weights streamed
gate-at-a-time per timestep (see DESIGN.md for the tile-memory budget
walkthrough). LSTM weights total 4*(96*96 + 96*96) + 4*(96+96) = 74,496
FP32 = 291 KiB — far above the 64 KiB tile data memory budget. Sharding
strategy: per timestep, per gate, the host streams one ``W_ih_gate``
(9216 FP32 = 36 KiB) and one ``W_hh_gate`` (36 KiB) sequentially through
a single double-buffer-1 ObjectFifo. The kernel computes partial gate
sums incrementally without ever holding more than one gate's weights at
a time. will revisit with bf16 / weight stationary patterns.

Tile-memory used (parsed from the build's ``main_core_*.ld.script``):
~ 41 KiB on a 64 KiB budget (single-buffered weight block + state).
Recorded in ``LSTM_CELL_TILE_MEMORY_USED_BYTES`` and the corresponding
``MANIFEST.md`` entry.
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

# Pinned LSTM cell geometry. T_LSTM = 334 = 2000 // 6 + 1 from the conv
# stem stride-6 layer ( layer 1 + layers 2/3 emit a sequence
# of length 334 timesteps for a 2000-sample raw input).
T_LSTM = 334
HIDDEN = 96
INPUT_DIM = 96
N_GATES = 4  # i, f, g, o (PyTorch LSTM order: i, f, g, o)

# Per-gate weight slab sizes
WIH_GATE = HIDDEN * INPUT_DIM  # 9216 FP32 per gate
WHH_GATE = HIDDEN * HIDDEN     # 9216 FP32 per gate
BIH_GATE = HIDDEN              # 96 FP32 per gate
BHH_GATE = HIDDEN              # 96 FP32 per gate

# Total flat weight buffer per LSTM layer (FP32):
#   W_ih (4 * 9216) + W_hh (4 * 9216) + b_ih (4 * 96) + b_hh (4 * 96)
#   = 36864 + 36864 + 384 + 384 = 74496 FP32 = 291 KiB
WB_LEN = N_GATES * (WIH_GATE + WHH_GATE) + N_GATES * (BIH_GATE + BHH_GATE)

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_DIR = _PACKAGE_ROOT / "dorado_fast_lstm_cell"
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

# Match plain decimal AND scientific notation (e.g. "1.76837e+06us"); the
# host runner's iostream emits scientific when the float is >= ~1e6 us.
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
class LstmCellRunResult:
    output: np.ndarray  # (L, N=1, 96) FP32
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

def _pack_input(x: np.ndarray) -> np.ndarray:
    """``(L, N=1, 96)`` -> ``(L * 96,)`` FP32 contiguous (time-major)."""
    if x.shape != (T_LSTM, 1, INPUT_DIM):
        raise ValueError(
            f"lstm_cell input must have shape ({T_LSTM}, 1, {INPUT_DIM}); "
            f"got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(f"lstm_cell input must be FP32; got dtype={x.dtype}")
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)

def _pack_wb(
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
) -> np.ndarray:
    """Pack PyTorch LSTM weights into the kernel's flat WB buffer.

    PyTorch ``nn.LSTM`` packs gates as rows ``[i, f, g, o]`` in
    ``weight_ih_l0`` (and same in ``weight_hh_l0``); the kernel mirrors
    that order. Returns a length-:data:`WB_LEN` FP32 contiguous buffer:

      [W_ih_i (96*96), W_hh_i (96*96), W_ih_f, W_hh_f,
       W_ih_g, W_hh_g, W_ih_o, W_hh_o,
       b_ih_i (96), b_hh_i (96), b_ih_f, b_hh_f,
       b_ih_g, b_hh_g, b_ih_o, b_hh_o]

    This packing puts each gate's complete weight set together so the
    streaming pattern (one gate at a time) reads contiguous bytes.
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
    return np.concatenate(parts).astype(np.float32)

def _unpack_output(flat: np.ndarray) -> np.ndarray:
    """``(L * 96,)`` -> ``(L, 1, 96)`` FP32."""
    expected = T_LSTM * HIDDEN
    if flat.size != expected:
        raise ValueError(
            f"lstm output length mismatch: got {flat.size}, expected {expected}"
        )
    return flat.reshape(T_LSTM, 1, HIDDEN)

class DoradoFastLstmCell(NpuOp):
    """Single LSTM cell forward over a fixed time sequence."""

    name = "dorado_fast_lstm_cell"

    INPUT_SHAPE = (T_LSTM, 1, INPUT_DIM)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    HIDDEN_SIZE = HIDDEN
    SEQ_LEN = T_LSTM

    def __init__(self) -> None:
        self.last_run: LstmCellRunResult | None = None

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

        x_packed = _pack_input(x)
        wb_packed = _pack_wb(weight_ih, weight_hh, bias_ih, bias_hh)

        with tempfile.TemporaryDirectory(prefix="t51_lstm_cell_") as td:
            tdp = Path(td)
            x_path = tdp / "input.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

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

            flat = np.fromfile(out_path, dtype=np.float32)
        out = _unpack_output(flat)
        self.last_run = LstmCellRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

# Tile-local memory used by the AIE compute tile, parsed from the
# linker script `main_core_*.ld.script` produced by aiecc. Recorded
# here so the equivalence runner can record it without re-running the
# build. See MANIFEST.md for the per-buffer breakdown.
LSTM_CELL_TILE_MEMORY_USED_BYTES: int | None = None

register_npu_op(
    "dorado_fast_lstm_cell",
    DoradoFastLstmCell(),
)

__all__ = [
    "BIH_GATE",
    "BHH_GATE",
    "DoradoFastLstmCell",
    "HIDDEN",
    "INPUT_DIM",
    "LSTM_CELL_TILE_MEMORY_USED_BYTES",
    "LstmCellRunResult",
    "N_GATES",
    "T_LSTM",
    "WB_LEN",
    "WHH_GATE",
    "WIH_GATE",
    "_pack_input",
    "_pack_wb",
    "_unpack_output",
]
