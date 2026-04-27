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

"""Dorado fast LSTM cell — bf16 NPU op with N:M-sparse weight DMA.

Registers ``dorado_fast_lstm_cell_bf16_compressed`` with
:data:`bionpu.dispatch.NPU_OPS`. Sibling of 's bf16 cell with
the **single difference** that the IRON weight FIFO is a
:class:`aie.iron.SparseFifo` instead of a vanilla
:class:`aie.iron.ObjectFifo`. Math is identical to because
SparseFifo presents the *dense* view to the consumer.

Op contract (FP32 in, FP32 out — keeps the encoder's FP32 interface):
    inputs:  x (L, 1, 96) fp32, weights and biases all fp32 (N:M-pruned
             upstream by `tracks.basecalling.quant.sparsity_pass`).
    outputs: y (L, 1, 96) fp32
    internal: bf16 throughout the kernel; FP32 → bf16 cast happens
              host-side, bf16 → FP32 on the output.

The pruning is **structural** (real zeros where the N:M pattern says
they go); the fork's BD-emit pass at AIEDmaToNpu.cpp:655 currently
writes ``Enable_Compression = 0`` unconditionally so the wire DMA
volume equals the dense total in the active build. Documented as
 in the kernel's ; weight-DMA-volume reduction
becomes Phase 3 future-work pending the BD-emit pass closure.

The accuracy contract is independent of the wire-DMA-volume goal:
bf16-compressed (N:M-pruned) vs bf16-dense, per-cell **max-abs ≤ 0.1**
(matches 's INT8-ish drift envelope; falsifiable).
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

# Re-export the bf16 conversion helpers + geometry constants from
# the dense sibling so callers don't need to know whether they're
# packaging weights for the dense or the compressed kernel — the
# packing format is the same in both cases (the only difference is
# whether the input weight tensor was N:M-pruned offline).
from bionpu.kernels.basecalling.lstm_cell_bf16 import (  # noqa: E402
    BIH_GATE,
    BHH_GATE,
    HIDDEN,
    INPUT_DIM,
    N_GATES,
    T_LSTM,
    WHH_GATE,
    WIH_GATE,
    WB_LEN,
    _pack_input_bf16,
    _pack_wb_bf16,
    _unpack_output_bf16,
    bf16_bytes_to_fp32,
    fp32_to_bf16_bytes,
)

# Default N:M pattern matches the sparsity_pass module default.
DEFAULT_N: int = 2
DEFAULT_M: int = 4

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_DIR = _PACKAGE_ROOT / "dorado_fast_lstm_cell_bf16_compressed"
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
_RE_DENSE = re.compile(
    r"^DMA dense bytes:\s*([0-9]+)\s*$", re.MULTILINE
)
_RE_COMP = re.compile(
    r"^DMA theoretical compressed bytes \([0-9]+:[0-9]+\):\s*([0-9]+)\s*$",
    re.MULTILINE,
)
_RE_MEAS = re.compile(
    r"^DMA measured wire bytes:\s*([0-9]+).*$", re.MULTILINE
)

# Tile-local memory used by the AIE compute tile — same as 
# (the SparseFifo lowers to the same ObjectFifo storage; the
# discardable compression attributes don't change the storage type).
LSTM_CELL_BF16_COMPRESSED_TILE_MEMORY_USED_BYTES = 0x400 + 2 * 0x2A00 + 4 * 0xC0  # 22784

@dataclass(frozen=True)
class LstmCellBf16CompressedRunResult:
    output: np.ndarray  # (L, N=1, 96) FP32 (re-cast from bf16)
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    sparsity_n: int
    sparsity_m: int
    dma_dense_bytes: int | None
    dma_theoretical_compressed_bytes: int | None
    dma_measured_wire_bytes: int | None

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

class DoradoFastLstmCellBf16Compressed(NpuOp):
    """Single bf16 LSTM cell forward with N:M-sparse weight DMA."""

    name = "dorado_fast_lstm_cell_bf16_compressed"

    INPUT_SHAPE = (T_LSTM, 1, INPUT_DIM)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    HIDDEN_SIZE = HIDDEN
    SEQ_LEN = T_LSTM
    PRECISION = "bf16"
    SPARSITY_N = DEFAULT_N
    SPARSITY_M = DEFAULT_M

    def __init__(self) -> None:
        self.last_run: LstmCellBf16CompressedRunResult | None = None

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
        sparsity_n: int = DEFAULT_N,
        sparsity_m: int = DEFAULT_M,
        measure_dma: bool = True,
    ) -> np.ndarray:
        for p in (_XCLBIN, _INSTS, _HOST_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{_ARTIFACT_DIR}/MANIFEST.md to rebuild."
                )

        x_packed = _pack_input_bf16(x)
        wb_packed = _pack_wb_bf16(weight_ih, weight_hh, bias_ih, bias_hh)

        with tempfile.TemporaryDirectory(prefix="t32_lstm_bf16_compressed_") as td:
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
                "-N", str(int(sparsity_n)),
                "-M", str(int(sparsity_m)),
            ]
            if measure_dma:
                cmd.append("--measure-dma")

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

            dma_dense = int(_RE_DENSE.search(proc.stdout).group(1)) if _RE_DENSE.search(proc.stdout) else None
            dma_compressed = int(_RE_COMP.search(proc.stdout).group(1)) if _RE_COMP.search(proc.stdout) else None
            dma_measured = int(_RE_MEAS.search(proc.stdout).group(1)) if _RE_MEAS.search(proc.stdout) else None

            buf_u16 = np.fromfile(out_path, dtype=np.uint16)
        out = _unpack_output_bf16(buf_u16)
        self.last_run = LstmCellBf16CompressedRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
            sparsity_n=int(sparsity_n),
            sparsity_m=int(sparsity_m),
            dma_dense_bytes=dma_dense,
            dma_theoretical_compressed_bytes=dma_compressed,
            dma_measured_wire_bytes=dma_measured,
        )
        return out

register_npu_op(
    "dorado_fast_lstm_cell_bf16_compressed",
    DoradoFastLstmCellBf16Compressed(),
)

__all__ = [
    "DEFAULT_M",
    "DEFAULT_N",
    "DoradoFastLstmCellBf16Compressed",
    "HIDDEN",
    "INPUT_DIM",
    "LSTM_CELL_BF16_COMPRESSED_TILE_MEMORY_USED_BYTES",
    "LstmCellBf16CompressedRunResult",
    "N_GATES",
    "T_LSTM",
    "WB_LEN",
]
