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

# lstm_cell.py — IRON lowering for Dorado fast LSTM cell -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Lowers a single Bonito LSTM cell (`nn.LSTM(input=96, hidden=96, num_layers=1,
# bidirectional=False, batch_first=False)`) over a fixed time sequence
# of length L=334 onto a single AIE2P tile.
#
# # Why this is a real toolchain gap
#
# `mlir-aie/programming_examples/` ships no LSTM / RNN example for AIE2 or
# AIE2P (verified 2026-04-25 against `mlir-aie@979629649`). The closest
# patterns are the matrix-multiplication examples (no recurrent state) and
# the SiLU/RMSNorm/LayerNorm (no per-timestep weight reuse). Per the
# umbrella PRD §4.4 ("LSTM is the bottleneck and the research target for
# "), this lowering is the load-bearing first version. It is FP32
# scalar — bf16 vector intrinsics arrive in . See `gaps.yaml` for the
# `expressibility` entry filed against this absence.
#
# # Tile-memory walkthrough
#
# Tile data memory is 64 KiB (4 * 16 KiB banks). Per-timestep working
# set:
#   - h, c, gate_acc[4]: 96+96+4*96 = 576 FP32 = 2.25 KiB
#   - x_t (input frame buffer): 96 FP32 = 384 B
#   - bias_ih, bias_hh (4 gates * 96): 768 FP32 = 3 KiB
#   - tmp/dot product registers: a few KiB
# State total ≈ 6 KiB.
#
# Full LSTM weight set: 4*(96*96 + 96*96) FP32 + 4*(96+96) FP32 =
# 74,496 FP32 = 291 KiB. NOWHERE near a 64 KiB tile.
#
# Sharding strategy: stream weights as **half-gate** chunks (one gate's
# W_ih or W_hh sliced along the input dim into halves of (96, 48) =
# 4608 FP32 = 18 KiB each). Per timestep, per gate, the kernel reads:
#   1. W_ih_gate first half  (18 KiB) — covers x_t[0:48]
#   2. W_ih_gate second half (18 KiB) — covers x_t[48:96]
#   3. W_hh_gate first half  (18 KiB) — covers h[0:48]
#   4. W_hh_gate second half (18 KiB) — covers h[48:96]
# (4 acquires per gate × 4 gates = 16 acquires per timestep; 334 ts ≈
# 5344 weight acquires per LSTM-layer call.)
#
# Tile-resident buffers (depth=2 on weights, depth=2 on input/output):
#   - weight_in_buff (depth=2, 4608 FP32): 36 KiB
#   - input_in_buff  (depth=2, 96 FP32):   768 B
#   - output_out_buff (depth=2, 96 FP32):  768 B
#   - state on data section (h, c, biases, x_t, gate_accs): ~6 KiB
#   - stack: 1 KiB
#   - total ≈ 44 KiB — within 64 KiB budget.
#
# # Buffer layout pinned with the host runner
#
# Input fifo (`input_in`):
#   float32[L * 96]; per-timestep frame is contiguous (timestep-major,
#   channel-minor). The fifo cycles `L = 334` 96-element elements.
#
# Weights fifo (`weight_in`):
#   float32[L * N_GATES * 4 * (HIDDEN * HALF_IN)] = float32[L * 4 * 4 *
#   4608] = float32[5344 * 4608] ≈ 24.6M FP32 (~98 MB). Per-timestep
#   layout cycles through:
#     gate=0: W_ih_h0, W_ih_h1, W_hh_h0, W_hh_h1
#     gate=1: ...
#   where each half is FP32[(HIDDEN, HALF_IN)] in row-major (out_dim,
#   in_dim_half).
#
# Output fifo (`output_out`):
#   float32[L * 96]; same layout as input.
#
# Biases are folded into the wb buffer at the start (one bias slab per
# gate per W type, prepended). The kernel reads them once on its first
# acquire of each gate's first weight half (TODO: simpler — just have a
# separate small bias fifo).

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

HIDDEN = 96
INPUT_DIM = 96
HALF_IN = INPUT_DIM // 2  # 48; sharding granularity for W_ih and W_hh
N_GATES = 4

# Acquire counts: per timestep, per gate, the kernel consumes
#   2 (W_ih halves) + 2 (W_hh halves) = 4 weight half-slabs.
# Per timestep that's 16 weight acquires.
N_HALVES_PER_GATE = 4  # 2 ih + 2 hh

WEIGHT_HALF_LEN = HIDDEN * HALF_IN  # 4608 FP32

# Bias buffer (small, single-shot at start): 4 gates * 2 (ih, hh) * 96.
BIAS_LEN = N_GATES * 2 * HIDDEN  # 768 FP32

def my_dorado_fast_lstm_cell(dev, L: int):
    """Return the MLIR for one LSTM-cell forward pass over L timesteps.

    AIE2P tile DMA constraint: a single CoreTile has 2 input + 2 output
    DMA channels. We have **3 logical inputs** (input frames, weights,
    biases) and **1 output** (frames). To fit, we **fold biases into
    the weight stream** as a 768-FP32 prefix on the first weight chunk
    only — the kernel reads them once and caches them in static memory.
    All subsequent kernel calls receive a 0-bias placeholder buffer
    that they ignore. (See `gaps.yaml` for the IRON DMA-
    channel-count blocker that drove this fold.)
    """
    # Tile-resident types
    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[np.float32]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[np.float32]]
    # Weight chunk: a half-gate slab. Biases are folded into the *first*
    # chunk of the cycle (offset 0..768 holds biases; the rest is the
    # half-gate W_ih). For non-first chunks, the first 768 floats are
    # ignored by the kernel. This costs one extra 768-float space per
    # chunk = 5344 * 768 = 4.1M FP32 of host-side bloat (~16 MB).
    chunk_with_bias_ty = np.ndarray[
        (BIAS_LEN + WEIGHT_HALF_LEN,), np.dtype[np.float32]
    ]

    # Whole-runtime buffers
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE  # 334 * 4 * 4 = 5344

    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[np.float32]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[np.float32]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * (BIAS_LEN + WEIGHT_HALF_LEN),),
        np.dtype[np.float32],
    ]

    # External AIE-tile kernel; entry symbol matches `extern "C"` in
    # lstm_cell.cc.
    lstm_kernel = Kernel(
        "dorado_fast_lstm_cell_fp32",
        "lstm_cell.o",
        [
            in_step_ty,           # x_t (96 floats)
            chunk_with_bias_ty,   # current weight chunk + bias prefix
            out_step_ty,          # y_t (= h_t, 96 floats)
            np.int32,             # gate index (0..3) — kernel uses for accum
            np.int32,             # timestep index (t==0 triggers reset)
            np.int32,             # chunk index (0..3 = ih_h0, ih_h1, hh_h0, hh_h1)
        ],
    )

    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)
    of_weight = ObjectFifo(chunk_with_bias_ty, name="weight_in", depth=2)
    of_output = ObjectFifo(out_step_ty, name="output_out", depth=2)

    def core_body(of_input, of_weight, of_output, lstm_fn):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)

        # Per-chunk acquire/release to keep within ObjectFifo depth=2.
        # The kernel C++ takes one chunk at a time and identifies which
        # half (h0/h1) and W type (ih/hh) via a chunk index in [0, 4).
        # Per timestep we make 4 gates × 4 chunks = 16 kernel calls.
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
    module = my_dorado_fast_lstm_cell(dev, L=opts.seq)
    print(module)
