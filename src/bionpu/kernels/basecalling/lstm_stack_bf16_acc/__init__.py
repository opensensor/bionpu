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

"""Dorado fast 5-layer alternating mixed-precision LSTM stack.

Mixed-precision sibling of 's
:class:`bionpu.kernels.basecalling.lstm_stack_bf16.DoradoFastLstmStackBf16`.
Same alternating-direction pattern (Bonito's flip → forward LSTM →
flip per architecture.md §2 — leading layer at index 4 is
reverse=True).

The preferred path uses the cell xclbin's ``stack_host_runner``: one
process, one XRT context, five sequential layer runs. If that runner is
not installed, this falls back to the original Python composite that
chains five :class:`DoradoFastLstmCellBf16Acc` calls.

Cross-walk note (AM020 §):
The "ideal" design per the cross-walk would chain layer-N's
output BM accumulator state into layer-N+1's input via the cascade
stream primitive (AM020 Ch. 4 p. 67, 512-bit cascade per cycle, no
precision loss). The IRON lowering does not expose this for
AIE2P; the composite falls back to **per-cell host-side state**
(each cell call drains its bf16 output and reseeds h, c on its own
internal reset path on (t==0, g==0, chunk==0)).

Within a single LSTM (one `DoradoFastLstmCellBf16Acc` call), the
state IS preserved at FP32 across all L=334 timesteps (this is the
load-bearing change vs — see lstm_cell_bf16_acc.cc). The
inter-layer boundary loses a single bf16 round-trip per layer (5
narrowings end-to-end) — small compared to 's ~1000
in-recurrence narrowings.

Why composite (rationale carried over from 's lstm_stack):
  - Per-layer drift validation is the load-bearing research surface.
  - Per-read weight reloads are cleaner with the per-layer call.
  - Host-side flips are free during the contiguous L*N*96 buffer build.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bionpu.dispatch.npu import NpuOp, NpuRunFailed, register_npu_op
from bionpu.kernels.basecalling.lstm_cell_bf16_acc import (
    _ARTIFACT_NAME,
    _PACKAGE_ROOT,
    HIDDEN,
    T_LSTM,
    DoradoFastLstmCellBf16Acc,
    _pack_input_bf16,
    _pack_wb_bf16,
)

LSTM_DIRECTIONS = [True, False, True, False, True]
N_LAYERS = 5
_RE_AVG = re.compile(
    r"^Avg NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_TOTAL = re.compile(
    r"^Total NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
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

@dataclass(frozen=True)
class LstmStackBf16AccRunResult:
    """Per-call timing rollup for the 5-layer mixed-precision LSTM stack."""

    output: np.ndarray
    per_layer_avg_us: list[float]
    total_us: float
    n_iters: int

class DoradoFastLstmStackBf16Acc(NpuOp):
    """5-layer alternating-direction mixed-precision LSTM stack
."""

    name = "dorado_fast_lstm_stack_bf16_acc"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    N_LAYERS = N_LAYERS
    DIRECTIONS = tuple(LSTM_DIRECTIONS)
    PRECISION = "mixed-fp32-state"

    def __init__(self, seq_len: int = T_LSTM) -> None:
        self.seq_len = int(seq_len)
        self.last_run: LstmStackBf16AccRunResult | None = None
        self._cell = DoradoFastLstmCellBf16Acc(seq_len=self.seq_len)

    @property
    def artifact_dir(self) -> Path:
        if self.seq_len == T_LSTM:
            return _PACKAGE_ROOT / _ARTIFACT_NAME
        return _PACKAGE_ROOT / f"{_ARTIFACT_NAME}_L{self.seq_len}"

    @property
    def xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def stack_host_runner(self) -> Path:
        return self.artifact_dir / "stack_host_runner"

    def batched_artifacts_available(self) -> bool:
        return all(p.exists() for p in (self.xclbin, self.insts, self.stack_host_runner))

    @classmethod
    def artifacts_present(cls) -> bool:
        return DoradoFastLstmCellBf16Acc.artifacts_present()

    def __call__(
        self,
        *,
        x: np.ndarray,
        weights_per_layer: list[dict[str, np.ndarray]],
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 600.0,
    ) -> np.ndarray:
        if len(weights_per_layer) != N_LAYERS:
            raise ValueError(
                f"weights_per_layer must have {N_LAYERS} entries; got "
                f"{len(weights_per_layer)}"
            )
        if self.batched_artifacts_available():
            return self._call_batched_host_runner(
                x=x,
                weights_per_layer=weights_per_layer,
                timeout_s=timeout_s,
            )
        cur = np.ascontiguousarray(x, dtype=np.float32)
        per_layer_us: list[float] = []
        total_us = 0.0
        for i, reverse in enumerate(LSTM_DIRECTIONS):
            w = weights_per_layer[i]
            if reverse:
                cur = np.ascontiguousarray(cur[::-1, :, :], dtype=np.float32)
            cur = self._cell(
                x=cur,
                weight_ih=w["weight_ih"],
                weight_hh=w["weight_hh"],
                bias_ih=w["bias_ih"],
                bias_hh=w["bias_hh"],
                n_iters=n_iters,
                warmup=warmup,
                timeout_s=timeout_s,
            )
            if reverse:
                cur = np.ascontiguousarray(cur[::-1, :, :], dtype=np.float32)
            avg_us = self._cell.last_run.avg_us if self._cell.last_run else 0.0
            per_layer_us.append(avg_us)
            total_us += avg_us
        self.last_run = LstmStackBf16AccRunResult(
            output=cur,
            per_layer_avg_us=per_layer_us,
            total_us=total_us,
            n_iters=int(n_iters),
        )
        return cur

    def _call_batched_host_runner(
        self,
        *,
        x: np.ndarray,
        weights_per_layer: list[dict[str, np.ndarray]],
        timeout_s: float,
    ) -> np.ndarray:
        x_packed = _pack_input_bf16(x, seq_len=self.seq_len)
        wb_packed = [
            _pack_wb_bf16(
                w["weight_ih"],
                w["weight_hh"],
                w["bias_ih"],
                w["bias_hh"],
            )
            for w in weights_per_layer
        ]

        with tempfile.TemporaryDirectory(prefix="t64d_lstm_stack_bf16_acc_") as td:
            tdp = Path(td)
            x_path = tdp / "input.bf16.bin"
            out_path = tdp / "output.f32.bin"
            wb_paths = [tdp / f"wb_layer_{i}.bf16.bin" for i in range(N_LAYERS)]
            x_packed.tofile(x_path)
            for path, wb in zip(wb_paths, wb_packed, strict=True):
                wb.tofile(path)

            cmd = [
                str(self.stack_host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
            ]
            for path in wb_paths:
                cmd.extend(["--wb", str(path)])
            cmd.extend(
                [
                    "--output", str(out_path),
                    "--seq", str(self.seq_len),
                    "--hidden", str(HIDDEN),
                ]
            )
            proc = subprocess.run(  # noqa: S603
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
            m_total = _RE_TOTAL.search(proc.stdout)
            if not (m_avg and m_total):
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse stack timing lines",
                )
            flat = np.fromfile(out_path, dtype=np.float32)

        expected = self.seq_len * HIDDEN
        if flat.size != expected:
            raise ValueError(
                f"lstm stack output length mismatch: got {flat.size}, "
                f"expected {expected}"
            )
        out = flat.reshape(self.seq_len, 1, HIDDEN)
        total_us = float(m_total.group(1))
        avg_us = float(m_avg.group(1))
        self.last_run = LstmStackBf16AccRunResult(
            output=out,
            per_layer_avg_us=[avg_us] * N_LAYERS,
            total_us=total_us,
            n_iters=1,
        )
        return out

register_npu_op(
    "dorado_fast_lstm_stack_bf16_acc",
    DoradoFastLstmStackBf16Acc(),
)

__all__ = [
    "DoradoFastLstmStackBf16Acc",
    "LSTM_DIRECTIONS",
    "LstmStackBf16AccRunResult",
    "N_LAYERS",
]
