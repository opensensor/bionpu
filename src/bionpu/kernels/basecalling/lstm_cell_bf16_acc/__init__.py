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

"""Dorado fast LSTM cell — bf16 input/output + FP32 recurrent state
NPU op.

Registers ``dorado_fast_lstm_cell_bf16_acc`` with
:data:`bionpu.dispatch.NPU_OPS`. Mixed-precision sibling of 's
``dorado_fast_lstm_cell_bf16``: the wire format (bf16 input, bf16
weights+biases, bf16 output) is **identical**, but the on-tile
recurrent state (h, c, gate_acc) is held at FP32 between timesteps.

Per the AM020 cross-walk (`docs/.md` §):
's end-to-end max-abs of 2.458 was diagnosed in 's
gaps.yaml as "bf16 cast quant noise on Dorado fast's
trained-large weights"; the AM020 read corrects this — the actual
wall is the **recurrent-state writeback** at bf16 width. Each
timestep narrows h_t / c_t from FP32 accumulator (23 mantissa bits;
AM020 Ch. 4 p. 65) to bf16 storage (8 mantissa bits) — losing 15
mantissa bits per step. Across 5 LSTMs × ~200 timesteps = ~1000
narrowings, the compound drift hits 2.3-2.5 max-abs.

's design: keep h, c, gate_acc as FP32 in tile DM. The
multiplier inputs (x_t, h_state-narrowed, weights) stay bf16 — that's
where the bf16 hardware path lives (AM020 Ch. 4 p. 65) and where
the 40× perf win demonstrated comes from. The C++ kernel
documents the storage discipline in its header.

**Toolchain caveat (RQ4)**: AM020 Ch. 4 p. 67 documents an explicit
accumulator-to-accumulator register move primitive (512 bits/cycle
on AIE-ML; carry-forward expected on AIE2P) and a cascade stream
primitive (512 bits inter-tile, no precision loss). The IRON
layer does NOT expose either. The C++ kernel falls back to FP32
tile-DM static storage to preserve the same precision invariant —
the accumulator-to-FP32-store conversion is hardware-free per AM020
Ch. 4 p. 65 ("Floating-Point Vector Unit"). This is documented in
``gaps.yaml`` as a real RQ4 finding: hardware supports it; toolchain
doesn't expose it.

Op contract (FP32 in, FP32 out — keeps the encoder's FP32 interface):
    inputs:  x (L, 1, 96) fp32, weights and biases all fp32
    outputs: y (L, 1, 96) fp32
    internal: bf16 multiplier inputs, FP32 accumulators (gate_acc,
              h_state, c_state); FP32→bf16 cast for matmul/activation
              inputs only.
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
from bionpu.kernels.basecalling.lstm_cell_bf16 import (
    HIDDEN,
    INPUT_DIM,
    N_GATES,
    T_LSTM,
    WB_LEN,
    _pack_input_bf16,
    _pack_wb_bf16,
    _unpack_output_bf16,
    bf16_bytes_to_fp32,
    fp32_to_bf16_bytes,
)

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_DIR = _PACKAGE_ROOT / "dorado_fast_lstm_cell_bf16_acc"
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
# linker script `main_core_*.ld.script` produced by aiecc. Same
# DMA-buffer footprint as (the wire format is identical); the
# .bss section is ~1.5 KiB larger because gate_acc + h_state + c_state
# are FP32 (vs 's bf16 storage).
LSTM_CELL_BF16_ACC_TILE_MEMORY_USED_BYTES = 0x400 + 2 * 0x2A00 + 4 * 0xC0  # 22784

@dataclass(frozen=True)
class LstmCellBf16AccRunResult:
    output: np.ndarray  # (L, N=1, 96) FP32 (re-cast from bf16 on the wire)
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

class DoradoFastLstmCellBf16Acc(NpuOp):
    """Mixed-precision LSTM cell: bf16 multiplier inputs, FP32
    recurrent state.
    """

    name = "dorado_fast_lstm_cell_bf16_acc"

    INPUT_SHAPE = (T_LSTM, 1, INPUT_DIM)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    HIDDEN_SIZE = HIDDEN
    SEQ_LEN = T_LSTM
    PRECISION = "mixed-fp32-state"

    def __init__(self) -> None:
        self.last_run: LstmCellBf16AccRunResult | None = None

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

        with tempfile.TemporaryDirectory(prefix="t64d_lstm_bf16_acc_") as td:
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
        self.last_run = LstmCellBf16AccRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

register_npu_op(
    "dorado_fast_lstm_cell_bf16_acc",
    DoradoFastLstmCellBf16Acc(),
)

__all__ = [
    "DoradoFastLstmCellBf16Acc",
    "HIDDEN",
    "INPUT_DIM",
    "LSTM_CELL_BF16_ACC_TILE_MEMORY_USED_BYTES",
    "LstmCellBf16AccRunResult",
    "N_GATES",
    "T_LSTM",
    "WB_LEN",
    "bf16_bytes_to_fp32",
    "fp32_to_bf16_bytes",
]
