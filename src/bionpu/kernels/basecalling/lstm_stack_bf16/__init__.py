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

"""Dorado fast 5-layer alternating bf16 LSTM stack — composite NPU op
.

bf16 sibling of 's :class:`bionpu.kernels.basecalling.lstm_stack
.DoradoFastLstmStack`. Same alternating-direction pattern (Bonito's
``flip → forward LSTM → flip`` per architecture.md §2 — leading layer
at index 4 is reverse=True). Composite op with no own xclbin: chains
five :class:`DoradoFastLstmCellBf16` calls.

Why composite (rationale carried over from 's lstm_stack):
  - Per-layer drift validation is the load-bearing research surface for
    's bf16 fix.
  - Per-read weight reloads are cleaner with the per-layer call.
  - Host-side flips are free during the contiguous L*N*96 buffer build.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bionpu.dispatch.npu import NpuOp, register_npu_op
from bionpu.kernels.basecalling.lstm_cell_bf16 import (
    DoradoFastLstmCellBf16,
    HIDDEN,
    T_LSTM,
)

LSTM_DIRECTIONS = [True, False, True, False, True]
N_LAYERS = 5

@dataclass(frozen=True)
class LstmStackBf16RunResult:
    """Per-call timing rollup for the 5-layer bf16 LSTM stack."""

    output: np.ndarray
    per_layer_avg_us: list[float]
    total_us: float
    n_iters: int

class DoradoFastLstmStackBf16(NpuOp):
    """5-layer alternating-direction bf16 LSTM stack."""

    name = "dorado_fast_lstm_stack_bf16"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    N_LAYERS = N_LAYERS
    DIRECTIONS = tuple(LSTM_DIRECTIONS)
    PRECISION = "bf16"

    def __init__(self) -> None:
        self.last_run: LstmStackBf16RunResult | None = None
        self._cell = DoradoFastLstmCellBf16()

    @classmethod
    def artifacts_present(cls) -> bool:
        return DoradoFastLstmCellBf16.artifacts_present()

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
        self.last_run = LstmStackBf16RunResult(
            output=cur,
            per_layer_avg_us=per_layer_us,
            total_us=total_us,
            n_iters=int(n_iters),
        )
        return cur

register_npu_op(
    "dorado_fast_lstm_stack_bf16",
    DoradoFastLstmStackBf16(),
)

__all__ = [
    "DoradoFastLstmStackBf16",
    "LSTM_DIRECTIONS",
    "LstmStackBf16RunResult",
    "N_LAYERS",
]
