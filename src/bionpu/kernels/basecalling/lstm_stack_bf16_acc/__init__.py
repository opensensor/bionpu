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

"""Dorado fast 5-layer alternating mixed-precision LSTM stack —
composite NPU op.

Mixed-precision sibling of 's
:class:`bionpu.kernels.basecalling.lstm_stack_bf16.DoradoFastLstmStackBf16`.
Same alternating-direction pattern (Bonito's flip → forward LSTM →
flip per architecture.md §2 — leading layer at index 4 is
reverse=True). Composite op with no own xclbin: chains five
:class:`DoradoFastLstmCellBf16Acc` calls.

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

from dataclasses import dataclass

import numpy as np

from bionpu.dispatch.npu import NpuOp, register_npu_op
from bionpu.kernels.basecalling.lstm_cell_bf16_acc import (
    DoradoFastLstmCellBf16Acc,
    HIDDEN,
    T_LSTM,
)

LSTM_DIRECTIONS = [True, False, True, False, True]
N_LAYERS = 5

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

    def __init__(self) -> None:
        self.last_run: LstmStackBf16AccRunResult | None = None
        self._cell = DoradoFastLstmCellBf16Acc()

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
