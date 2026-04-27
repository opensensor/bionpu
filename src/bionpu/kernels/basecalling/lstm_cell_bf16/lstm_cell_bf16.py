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

# lstm_cell_bf16.py — IRON lowering for Dorado fast LSTM cell, bf16 path
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bf16 sibling of `bionpu/kernels/basecalling/lstm_cell/lstm_cell.py`
#. Same single-tile, half-gate sharded
# DMA pattern ( forced bias slab to be folded into every
# weight chunk; that constraint is unchanged here — bf16 doesn't
# magically gain DMA channels). The only changes vs :
#   - dtype on every ObjectFifo + Kernel arg switches to bfloat16.
#   - chunk size is half-precision -> chunk byte size halves.
#   - nothing else moves: same depth=2 fifos, same gate/chunk
#     iteration counts, same Worker/Runtime structure.
#
# # Tile-memory walkthrough (post-shrink)
#
# State (h, c, gate_acc[4], biases) at bf16:
#   h, c, gate_acc, x_t : at bf16 -> half the fp32 footprint.
#   bias_cache (768)    : 768 * 2 = 1536 B.
#   gate_acc[4][96]     : 4 * 96 * 4 = 1536 B (held fp32 internally).
#   h_state, c_state    : 96 * 2 + 96 * 2 = 384 B.
# State total: ~4 KiB (vs 's ~6 KiB, savings from halved buffers).
#
# Fifos (depth=2, bf16):
#   weight_in_buff_{0,1}: 2 * (768 + 4608) bf16 = 2 * 10752 B = 21504 B.
#   input_in_buff_{0,1}:  2 * 96 bf16 = 384 B.
#   output_out_buff_{0,1}:2 * 96 bf16 = 384 B.
#   stack: 1 KiB.
# Total approx: 28 KiB on a 64 KiB tile (vs 's ~51 KiB) — bf16
# unlocks substantial slack for future fusions / multi-cell pipelining
#.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

HIDDEN = 96
INPUT_DIM = 96
HALF_IN = INPUT_DIM // 2  # 48
N_GATES = 4
N_HALVES_PER_GATE = 4  # 2 ih + 2 hh

WEIGHT_HALF_LEN = HIDDEN * HALF_IN  # 4608

# Bias buffer (folded into every chunk's prefix, see ):
# 4 gates * 2 (ih, hh) * 96 = 768 bf16 = 1536 B.
BIAS_LEN = N_GATES * 2 * HIDDEN  # 768

# bfloat16 dtype for ObjectFifo / Kernel buffer types. The aie.iron
# Python bindings carry bfloat16 via numpy's bfloat16-compatible type.
# We use np.dtype("bfloat16") via ml_dtypes when available; fall back
# to a uint16 surrogate (same byte size, transparent to IRON).
try:
    from ml_dtypes import bfloat16  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover - ml_dtypes always present in env
    _BF16 = np.dtype(np.uint16)

def my_dorado_fast_lstm_cell_bf16(dev, L: int):
    """Return the MLIR for one bf16 LSTM-cell forward over L timesteps.

    Same DMA topology as 's FP32 cell (input/weight in, output out,
    biases folded into every weight-chunk's prefix); per-element type
    drops from FP32 to bf16 throughout.
    """
    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_with_bias_ty = np.ndarray[
        (BIAS_LEN + WEIGHT_HALF_LEN,), np.dtype[_BF16]
    ]

    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE  # 334 * 4 * 4 = 5344

    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * (BIAS_LEN + WEIGHT_HALF_LEN),),
        np.dtype[_BF16],
    ]

    lstm_kernel = Kernel(
        "dorado_fast_lstm_cell_bf16",
        "lstm_cell_bf16.o",
        [
            in_step_ty,
            chunk_with_bias_ty,
            out_step_ty,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)
    of_weight = ObjectFifo(chunk_with_bias_ty, name="weight_in", depth=2)
    of_output = ObjectFifo(out_step_ty, name="output_out", depth=2)

    def core_body(of_input, of_weight, of_output, lstm_fn):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)

        for t in range_(L):
            elem_in = of_input.acquire(1)
            elem_out = of_output.acquire(1)
            for g in range_(N_GATES):
                for chunk in range_(4):
                    w_chunk = of_weight.acquire(1)
                    g_i32 = arith.index_cast(i32, g)
                    t_i32 = arith.index_cast(i32, t)
                    chunk_i32 = arith.index_cast(i32, chunk)
                    lstm_fn(elem_in, w_chunk, elem_out, g_i32, t_i32, chunk_i32)
                    of_weight.release(1)
            of_input.release(1)
            of_output.release(1)

    worker = Worker(
        core_body,
        fn_args=[
            of_input.cons(),
            of_weight.cons(),
            of_output.prod(),
            lstm_kernel,
        ],
    )

    rt = Runtime()
    with rt.sequence(in_total_ty, weight_total_ty, out_total_ty) as (X, W, Y):
        rt.start(worker)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weight.prod(), W)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device",
                   help="AIE Device (npu/npu2)")
    p.add_argument("-L", "--seq", type=int, default=334,
                   help="LSTM sequence length (default 334)")
    return p.parse_args(argv)

def _select_device(name: str):
    if name == "npu":
        return NPU1Col1()
    if name == "npu2":
        return NPU2()
    raise ValueError(f"[ERROR] Device name {name!r} is unknown")

if __name__ == "__main__":
    opts = _parse_args(sys.argv[1:])
    dev = _select_device(opts.device)
    module = my_dorado_fast_lstm_cell_bf16(dev, L=opts.seq)
    print(module)
