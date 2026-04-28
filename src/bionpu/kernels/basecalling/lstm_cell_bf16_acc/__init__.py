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
 as "bf16 cast quant noise on Dorado fast's
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
```` as a real RQ4 finding: hardware supports it; toolchain
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
from bionpu.dispatch._pyxrt_helpers import (
    resolve_dispatch_impl,
    run_pyxrt_with_buffers,
)
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

# Kernel-arg slot layout for the lstm_cell_bf16_acc runner (see
# runner.cpp: kernel(opcode, bo_instr, instr_size, bo_input, bo_weight,
# bo_output, bo_trace)). Slot 6 is the 1-byte trace placeholder.
_ARG_INPUT = 3
_ARG_WEIGHT = 4
_ARG_OUTPUT = 5
_ARG_TRACE = 6

# Constants mirroring runner.cpp expand_wb (bf16 variant). Same layout
# as the FP32 sibling but stored as uint16 (bf16-as-uint16 byte
# pattern); chunk size in *elements* (uint16) is identical.
_HALF_IN = INPUT_DIM // 2  # 48
_WEIGHT_HALF_LEN = HIDDEN * _HALF_IN  # 4608
_BIAS_LEN = N_GATES * 2 * HIDDEN  # 768
_CHUNK_LEN = _BIAS_LEN + _WEIGHT_HALF_LEN  # 5376


def _expand_wb_bf16(compact: np.ndarray, seq_len: int) -> np.ndarray:
    """Expand the compact bf16-as-uint16 host WB into the IRON
    per-timestep stream.

    Mirrors ``expand_wb`` in lstm_cell_bf16_acc/runner.cpp byte-for-byte
    (same layout as the FP32 lstm_cell sibling, just bf16-wide
    elements). Returns a ``uint16`` ndarray of length
    ``seq_len * N_GATES * 4 * _CHUNK_LEN``.
    """
    if compact.size != WB_LEN:
        raise ValueError(
            f"compact bf16 wb length mismatch: got {compact.size}, "
            f"expected {WB_LEN}"
        )
    cp = np.ascontiguousarray(compact, dtype=np.uint16).reshape(-1)

    off = 0
    Wih_gates: list[np.ndarray] = []
    Whh_gates: list[np.ndarray] = []
    WIH_GATE = HIDDEN * INPUT_DIM
    WHH_GATE = HIDDEN * HIDDEN
    BIH_GATE = HIDDEN
    BHH_GATE = HIDDEN
    for _ in range(N_GATES):
        Wih_gates.append(cp[off : off + WIH_GATE].reshape(HIDDEN, INPUT_DIM))
        off += WIH_GATE
        Whh_gates.append(cp[off : off + WHH_GATE].reshape(HIDDEN, HIDDEN))
        off += WHH_GATE
    bias_slab = np.empty(_BIAS_LEN, dtype=np.uint16)
    for g in range(N_GATES):
        bih = cp[off : off + BIH_GATE]; off += BIH_GATE
        bhh = cp[off : off + BHH_GATE]; off += BHH_GATE
        bias_slab[(g * 2) * HIDDEN : (g * 2 + 1) * HIDDEN] = bih
        bias_slab[(g * 2 + 1) * HIDDEN : (g * 2 + 2) * HIDDEN] = bhh

    halves = np.empty((N_GATES, 4, HIDDEN, _HALF_IN), dtype=np.uint16)
    for g in range(N_GATES):
        halves[g, 0] = Wih_gates[g][:, 0:_HALF_IN]
        halves[g, 1] = Wih_gates[g][:, _HALF_IN:INPUT_DIM]
        halves[g, 2] = Whh_gates[g][:, 0:_HALF_IN]
        halves[g, 3] = Whh_gates[g][:, _HALF_IN:HIDDEN]

    cycle = np.empty((N_GATES, 4, _CHUNK_LEN), dtype=np.uint16)
    for g in range(N_GATES):
        for h in range(4):
            cycle[g, h, :_BIAS_LEN] = bias_slab
            cycle[g, h, _BIAS_LEN:] = halves[g, h].reshape(-1)
    repeated = np.broadcast_to(
        cycle, (seq_len, N_GATES, 4, _CHUNK_LEN)
    ).reshape(-1)
    return np.ascontiguousarray(repeated, dtype=np.uint16)

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_NAME = "dorado_fast_lstm_cell_bf16_acc"
_ARTIFACT_DIR = _PACKAGE_ROOT / _ARTIFACT_NAME

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

    def __init__(self, seq_len: int = T_LSTM) -> None:
        self.seq_len = int(seq_len)
        self.last_run: LstmCellBf16AccRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return cls(seq_len=T_LSTM).artifacts_available()

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
        return all(p.exists() for p in (self.xclbin, self.insts, self.host_runner))

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
        _impl: str | None = None,
    ) -> np.ndarray:
        impl = resolve_dispatch_impl(
            _impl, env_var="BIONPU_LSTM_CELL_BF16_ACC_DISPATCH"
        )
        required = (
            (self.xclbin, self.insts) if impl == "pyxrt"
            else (self.xclbin, self.insts, self.host_runner)
        )
        for p in required:
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{self.artifact_dir}/MANIFEST.md to rebuild."
                )

        x_packed = _pack_input_bf16(x, seq_len=self.seq_len)
        wb_packed = _pack_wb_bf16(weight_ih, weight_hh, bias_ih, bias_hh)

        if impl == "pyxrt":
            wb_expanded = _expand_wb_bf16(wb_packed, self.seq_len)
            # Output is bf16 on the wire (the kernel narrows
            # h_acc -> bfloat16 before storing y_t; see
            # lstm_cell_bf16_acc.cc). The FP32-output flip was reverted
            # because the L334 silicon was never rebuilt and a runner
            # that read FP32 from a bf16-emitting xclbin produced
            # garbage at production_long=0 (each pair of bf16s reread
            # as one FP32; second half of every layer = zero).
            out_size = self.seq_len * HIDDEN * 2  # bf16-as-uint16
            raw_out, avg_us, min_us, max_us = run_pyxrt_with_buffers(
                xclbin_path=self.xclbin,
                insts_path=self.insts,
                in_buffers=[
                    (x_packed.tobytes(), _ARG_INPUT),
                    (wb_expanded.tobytes(), _ARG_WEIGHT),
                    (bytes(1), _ARG_TRACE),
                ],
                out_size=out_size,
                out_arg_index=_ARG_OUTPUT,
                n_iters=n_iters,
                warmup=warmup,
            )
            buf_u16 = np.frombuffer(raw_out, dtype=np.uint16).copy()
            out = _unpack_output_bf16(buf_u16, seq_len=self.seq_len)
            self.last_run = LstmCellBf16AccRunResult(
                output=out,
                avg_us=avg_us,
                min_us=min_us,
                max_us=max_us,
                n_iters=int(n_iters),
            )
            return out

        with tempfile.TemporaryDirectory(prefix="t64d_lstm_bf16_acc_") as td:
            tdp = Path(td)
            x_path = tdp / "input.bf16.bin"
            wb_path = tdp / "wb.bf16.bin"
            out_path = tdp / "output.bf16.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(self.seq_len),
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
        out = _unpack_output_bf16(buf_u16, seq_len=self.seq_len)
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
