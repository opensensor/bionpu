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

# lstm_cell_int8.py — IRON lowering for Dorado fast LSTM cell, INT8
#                      path with FP32 recurrent state
#                      -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# INT8 sibling of `bionpu/kernels/basecalling/lstm_cell_bf16/lstm_cell_bf16.py`
#. Closes (the "no AIE2P INT8 LSTM
# kernel" surface 's INT8 sweep needed). Per brief, this
# **does not** depend on the fork's CascadeFifo / AccumFifo — uses
# existing IRON primitives only.
#
# What changed vs (bf16) and (bf16_acc):
#   - Wire dtype: int8 (1 byte/elem) for input, output, weight slab.
#   - Bias prefix dtype: float32 (4 bytes/elem). Carries the
#     pre-multiplied bias (b_ih + b_hh) per gate plus the per-gate
#     scale chain that the kernel uses to dequantize gate_acc.
# - Recurrent state: FP32 in tile DM (per 's lesson —
#     INT8 narrowing on h_t / c_t is the silicon wall).
#   - Per-channel calibration: applied **host-side** when packing
#     weight slabs (each output channel's row in W is independently
#     scaled to its own int8 grid; the per-channel scale is then
#     *folded into the FP32 bias prefix* so the on-tile dequant is a
#     single per-gate scalar multiply). See runner.cpp expand_wb.
#
# # Tile-memory walkthrough (post-shrink vs 's bf16 22.3 KiB):
#
# Wire-format chunk size:
#   prefix:  784 floats (aligned) = 3136 B
#   weight:  HIDDEN * HALF_IN INT8 = 4608 B
# total: 7744 B per chunk (vs 's 10752 B bf16 chunk)
# depth=2 cons buffer: 2 * 7744 = 15488 B (vs 's 21504 B)
#
# State (h, c, gate_acc, biases, scales) on tile:
#   h_state, c_state    : 96 * 4 + 96 * 4 = 768 B (FP32, persistent)
#   gate_acc[4][96]     : 4 * 96 * 4 = 1536 B (INT32)
#   bias_cache (768)    : 768 * 4 = 3072 B (FP32)
#   per_gate_scale_x[4] : 16 B
#   per_gate_scale_h[4] : 16 B
#   h_scale, y_scale    : 8 B
# State total: ~5.5 KiB (vs 's bf16 ~4 KiB; the INT32 gate_acc
# + FP32 bias slab pay for the precision discipline).
#
# Fifos (depth=2):
#   weight_in_buff_{0,1}: 2 * 7744 = 15488 B
#   input_in_buff_{0,1}:  2 * 96 INT8 = 192 B
#   output_out_buff_{0,1}:2 * 96 INT8 = 192 B
#   stack: 1 KiB
# Total approx: ~24 KiB on a 64 KiB tile (vs 's 28 KiB) —
# INT8 wire format saves on the weight-buffer footprint enough to
# offset the FP32 bias cache.

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

# Half-gate weight slab: HIDDEN * HALF_IN INT8.
WEIGHT_HALF_LEN = HIDDEN * HALF_IN  # 4608

# Bias buffer (folded into every chunk's prefix, see ):
# 4 gates * 2 (ih, hh) * 96 = 768 FP32 = 3072 B.
BIAS_LEN = N_GATES * 2 * HIDDEN  # 768

# Scale prefix: per_gate_scale_x[4] + per_gate_scale_h[4] + h_scale +
# y_scale = 10 floats. Plus padding to 32-byte alignment for the int8
# slab that follows.
SCALE_PREFIX = 4 + 4 + 1 + 1  # 10 floats
FLOAT_PREFIX = SCALE_PREFIX + BIAS_LEN  # 778 floats
# Pad to 32-byte alignment: 778 → 784 floats (3136 bytes = 32 * 98).
FLOAT_PREFIX_ALIGNED = 784  # floats
BIAS_PREFIX_BYTES = FLOAT_PREFIX_ALIGNED * 4  # 3136

# Wire-format chunk byte size: prefix (FP32, 4B) + weight (INT8, 1B).
# Expressed as INT8 elements (1B) for the IRON ObjectFifo type.
CHUNK_BYTES = BIAS_PREFIX_BYTES + WEIGHT_HALF_LEN  # 3136 + 4608 = 7744

def my_dorado_fast_lstm_cell_int8(dev, L: int):
    """Return the MLIR for one INT8 LSTM-cell forward over L timesteps.

    Same DMA topology as 's bf16 cell (input/weight in, output
    out, prefix folded into every weight chunk). Only the per-element
    type changes:
      - input/output: bf16 -> int8 (1 byte/elem)
      - weight chunk: bf16 -> int8 (1 byte/elem) AFTER a 3136-byte
        FP32 prefix carrying the per-gate scales + biases
    """
    # IRON ObjectFifo types are expressed as numpy ndarrays. We carry
    # the prefix + weight slab as a flat int8 buffer (3136 + 4608 =
    # 7744 bytes); the kernel reinterprets the first BIAS_PREFIX_BYTES
    # as FP32 via a byte-pointer cast in C++.
    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[np.int8]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[np.int8]]
    chunk_with_bias_ty = np.ndarray[
        (CHUNK_BYTES,), np.dtype[np.int8]
    ]

    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE  # 334 * 4 * 4 = 5344

    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[np.int8]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[np.int8]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * CHUNK_BYTES,),
        np.dtype[np.int8],
    ]

    lstm_kernel = Kernel(
        "dorado_fast_lstm_cell_int8",
        "lstm_cell_int8.o",
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
    module = my_dorado_fast_lstm_cell_int8(dev, L=opts.seq)
    print(module)
