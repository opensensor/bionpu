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

"""Dorado fast 5-layer alternating LSTM stack — composite NPU op.

Registers ``dorado_fast_lstm_stack`` with the NPU dispatch registry as a
**composite op**: it doesn't ship its own xclbin. Instead it chains five
calls to ``dorado_fast_lstm_cell`` (the per-layer NPU kernel) with
host-side flips between layers to implement Bonito's alternating-direction
pattern (architecture.md §2: forward, reverse, forward, reverse, forward
with the leading layer at index 4 being reverse=True per the on-disk
ordering).

Why composite (and not one fused xclbin):
  - Per-layer validation of the FP32 NPU vs CPU drift is the load-bearing
    research surface for (the LSTM cell is 's 77%-of-GPU-time
    bottleneck and 's main optimization target).
  - Fusing five LSTM-cell xclbins into one would tie us to a single set
    of weights; 's full FASTQ pipeline wants per-read weight reloads.
  - The host-side ``flip`` between layers is a memory permutation,
    NPU-native per but free during the fifo construction since
    we already build a contiguous L*N*96 buffer.

 may revisit and fuse if profiling justifies; for v1 we ship the
composite chain and document the per-layer per-call drift in
``results/basecalling/b-m4/measurements.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bionpu.dispatch.npu import NpuOp, register_npu_op
from bionpu.kernels.basecalling.lstm_cell import (
    DoradoFastLstmCell,
    HIDDEN,
    T_LSTM,
)

LSTM_DIRECTIONS = [True, False, True, False, True]
N_LAYERS = 5

@dataclass(frozen=True)
class LstmStackRunResult:
    """Per-call timing rollup for the 5-layer LSTM stack."""

    output: np.ndarray  # (L, N=1, 96) FP32
    per_layer_avg_us: list[float]
    total_us: float
    n_iters: int

class DoradoFastLstmStack(NpuOp):
    """5-layer alternating-direction LSTM stack."""

    name = "dorado_fast_lstm_stack"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    N_LAYERS = N_LAYERS
    DIRECTIONS = tuple(LSTM_DIRECTIONS)

    def __init__(self) -> None:
        self.last_run: LstmStackRunResult | None = None
        self._cell = DoradoFastLstmCell()

    @classmethod
    def artifacts_present(cls) -> bool:
        return DoradoFastLstmCell.artifacts_present()

    def __call__(
        self,
        *,
        x: np.ndarray,
        weights_per_layer: list[dict[str, np.ndarray]],
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 600.0,
    ) -> np.ndarray:
        """Run the 5-layer stack on x.

        Args:
            x: ``(L=334, 1, 96)`` FP32 input.
            weights_per_layer: list of length 5; each entry has keys
                ``weight_ih`` ``(384, 96)``, ``weight_hh`` ``(384, 96)``,
                ``bias_ih`` ``(384,)``, ``bias_hh`` ``(384,)``.
            n_iters / warmup / timeout_s: forwarded to the LSTM cell op.

        Returns:
            ``(L, 1, 96)`` FP32 — the post-stack hidden state ready for
            the linear projection.
        """
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
        self.last_run = LstmStackRunResult(
            output=cur,
            per_layer_avg_us=per_layer_us,
            total_us=total_us,
            n_iters=int(n_iters),
        )
        return cur

register_npu_op(
    "dorado_fast_lstm_stack",
    DoradoFastLstmStack(),
)

__all__ = [
    "DoradoFastLstmStack",
    "LSTM_DIRECTIONS",
    "LstmStackRunResult",
    "N_LAYERS",
]
