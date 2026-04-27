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

# conv1d_layer3.py — IRON lowering for Dorado fast Conv stem layer 3 -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Lowers Dorado `fast@v5.0.0` third Conv1d stem layer
# (in=16, out=96, kernel=19, stride=6, padding=9, bias=true) onto a
# single AIE2P tile. Stride 6 + kernel 19 is the asymmetric layer that
# reduces the time dimension by 6× — input length 2018 (after padding)
# produces output length 334.
#
# # Tile-memory budget
#
# The full weight buffer is 96*16*19 + 96 = 29280 floats = 117 KiB; the
# full input buffer is 16*2018 = 32288 floats = 126 KiB; the full
# output buffer is 96*334 = 32064 floats = 125 KiB. None of these fit
# in 64 KiB. We solve this by chunking along the **output time axis**:
# emit T_chunk_out output samples per kernel invocation, where each
# invocation needs k + (T_chunk_out - 1) * stride = 19 + (T_chunk_out -
# 1)*6 input samples per channel.
#
# For T_chunk_out = 32:
#   chunk_in_len_per_ch = 19 + 31*6 = 205 samples
#   sig: 1 * 16 * 205 * 4 = 13120 B
#   wb:  2 * 29280 * 4    = 234240 B  -- still doesn't fit!
#
# Strategy: keep weights resident on tile (single big sub-buffer) by
# *tiling* across output channels. Per kernel invocation, we process
# OC_GROUP = 16 output channels (1/6 of the 96), which needs
# 16*16*19 + 16 = 4880 floats = 19 KiB of weights. We do 6 passes per
# input chunk to cover all 96 output channels, with a different weight
# slice each pass.
#
# For T_chunk_out = 32, OC_GROUP = 16 (single-buffered to fit budget):
#   sig:    1 * 16 * 205 * 4 = 13120 B
#   wb:     2 * 4880 * 4    = 39040 B   (per-OC-group weights)
#   output: 1 * 16 * 32 * 4 = 2048 B    (per-OC-group output for chunk)
#   stack:  1024 B
#   total: ~ 55 KiB — within 64 KiB.
#
# Number of OC groups = 96 / 16 = 6.
# Number of time chunks: T_OUT / T_chunk_out = 334 / 32 → not divisible.
# We choose T_chunk_out that divides 334 cleanly: 334 = 2 * 167 (prime).
# 334 / 2 = 167 chunks of 2; 334 / 167 = 2 chunks of 167. Neither is
# convenient. Solution: pad output to 336 (= 32 * (334 / 32 + 1)) =
# divisible? 32 * 11 = 352, 32 * 10 = 320. Pick T_chunk_out=2:
#   chunk_in_len_per_ch = 19 + 6 = 25 samples
#   sig: 1 * 16 * 25 * 4 = 1600 B (per chunk)
#   wb:  2 * 4880 * 4 = 39040 B
#   output: 1 * 16 * 2 * 4 = 128 B
#   total ≈ 41 KiB.  Fits, and we get 167 time chunks per OC group.
#   Total kernel invocations = 6 OC groups × 167 time chunks = 1002.
#
# Pick T_chunk_out = 167 (single time chunk per OC group):
#   chunk_in_len_per_ch = 19 + 166*6 = 1015 samples
#   sig: 1 * 16 * 1015 * 4 = 64960 B  -- TOO BIG.
#
# Pick T_chunk_out = 14 (divides T_OUT not exactly: 334/14 = 23.857).
# Pick T_chunk_out = 1 (every output sample is its own chunk):
#   chunk_in_len_per_ch = 19 samples
#   sig: 1 * 16 * 19 * 4 = 1216 B
#   wb:  2 * 4880 * 4 = 39040 B
#   output: 1 * 16 * 1 * 4 = 64 B
#   total ≈ 41 KiB — within budget. Total kernel invocations =
#   6 OC groups × 334 time samples = 2004. Per-call overhead ~ ~1 us
#   on the host runner pattern → ~2 ms total. Acceptable for v1.
#
# Going with T_chunk_out = 1 to maximize fit; can revisit.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

CONV_K = 19
CONV_IN_CH = 16
CONV_OUT_CH = 96
PAD = 9
STRIDE = 6
OC_GROUP_SIZE = 16  # 96 / 6 groups; sized so weight slice fits the tile.
N_OC_GROUPS = CONV_OUT_CH // OC_GROUP_SIZE  # 6

def my_dorado_fast_conv_stem_layer3(dev, T_in_padded: int, T_out: int):
    """
    T_in_padded : input padded length per channel (= 2000 + 2*9 = 2018)
    T_out       : output length (= 334)
    """
    # We process one output time sample at a time per kernel call; per
    # call the input slice is K = 19 samples per channel and the output
    # is OC_GROUP_SIZE = 16 floats (one slice of the 96 output channels).
    # The host streams through OC_GROUPS x T_out = 6 x 334 = 2004 slices.

    # Per-slice tile-resident shapes:
    chunk_in_len = CONV_K  # 19 samples per channel
    chunk_out_len = 1  # one output time sample per kernel call

    # Whole-runtime tensors on the host side:
    #   - signal: T_out time slices, each (CONV_IN_CH * CONV_K) floats.
    #     Slice t covers input samples [t*STRIDE : t*STRIDE + CONV_K].
    #   - wb: N_OC_GROUPS slices, each (OC_GROUP_SIZE * CONV_IN_CH * CONV_K +
    #     OC_GROUP_SIZE) = 4880 floats.
    #   - output: N_OC_GROUPS * T_out * OC_GROUP_SIZE = 32064 floats
    #     reshapeable to (96, 334) once the host concatenates by
    #     (oc_group, time, oc_in_group) and transposes.
    #
    # Layout assumption for the runtime: the host produces one
    # signal-fifo element per *time slice*, but since the kernel is
    # called once per (oc_group × time) tuple, the host re-feeds the
    # same time slice 6 times in sequence (oc_group cycles inside the
    # outer time loop). Equivalently: we can have the fifo cycle
    # internally — for v1 we use the explicit replication, which is the
    # simplest IRON pattern.
    #
    # To halve the number of kernel calls, the IRON program below takes
    # one signal slice and runs all 6 OC groups against it before
    # advancing time. This means: per-time-step, signal arrives once,
    # weights cycle 6 times.

    sig_slice_len = CONV_IN_CH * chunk_in_len  # 16*19 = 304 floats
    wb_slice_len = OC_GROUP_SIZE * CONV_IN_CH * CONV_K + OC_GROUP_SIZE  # 4880
    out_slice_len = OC_GROUP_SIZE * chunk_out_len  # 16

    # Whole-runtime totals.
    #
    # Note on wb_total_len: IRON's ObjectFifo expects the host runtime
    # to feed exactly as many fifo-element copies as the consumer
    # acquires. The consumer here acquires N_OC_GROUPS*T_out = 2004
    # times, so the host must materialise the wb buffer with the same
    # total volume on its side (= T_out repetitions of the 6-group
    # cycle). For v1 we accept the host-side memory bloat
    # (T_out * N_OC_GROUPS * wb_slice_len floats); a future revision
    # can use BD-level repeat patterns to avoid this. ~37 MB at FP32 is
    # well within host RAM.
    sig_total_len = T_out * sig_slice_len
    wb_total_len = T_out * N_OC_GROUPS * wb_slice_len
    out_total_len = T_out * N_OC_GROUPS * out_slice_len  # = T_out * 96

    sig_chunk_ty = np.ndarray[(sig_slice_len,), np.dtype[np.float32]]
    wb_chunk_ty = np.ndarray[(wb_slice_len,), np.dtype[np.float32]]
    out_chunk_ty = np.ndarray[(out_slice_len,), np.dtype[np.float32]]

    sig_total_ty = np.ndarray[(sig_total_len,), np.dtype[np.float32]]
    wb_total_ty = np.ndarray[(wb_total_len,), np.dtype[np.float32]]
    out_total_ty = np.ndarray[(out_total_len,), np.dtype[np.float32]]

    conv_kernel = Kernel(
        "dorado_fast_conv_stem_layer3_fp32",
        "conv1d_layer3.o",
        [sig_chunk_ty, wb_chunk_ty, out_chunk_ty],
    )

    # ObjectFifos: one signal slice per time step; weights cycle through
    # N_OC_GROUPS slices, host feeds them T_out * N_OC_GROUPS times so
    # each time step iterates through all 6 weight slices. The signal
    # fifo is single-element-per-time-step, so the kernel acquires it
    # once per time step then releases after the OC group loop.
    of_signal = ObjectFifo(sig_chunk_ty, name="signal_in", depth=1)
    of_wb = ObjectFifo(wb_chunk_ty, name="wb_in", depth=2)
    of_output = ObjectFifo(out_chunk_ty, name="output_out", depth=1)

    def core_body(of_signal, of_wb, of_output, conv_fn):
        for _ in range_(T_out):
            elem_signal = of_signal.acquire(1)
            for _g in range_(N_OC_GROUPS):
                elem_wb = of_wb.acquire(1)
                elem_out = of_output.acquire(1)
                conv_fn(elem_signal, elem_wb, elem_out)
                of_wb.release(1)
                of_output.release(1)
            of_signal.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_signal.cons(), of_wb.cons(), of_output.prod(), conv_kernel],
    )

    rt = Runtime()
    with rt.sequence(sig_total_ty, wb_total_ty, out_total_ty) as (S, W, O):
        rt.start(worker)
        rt.fill(of_signal.prod(), S)
        rt.fill(of_wb.prod(), W)
        rt.drain(of_output.cons(), O, wait=True)

    return Program(dev, rt).resolve_program()

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--dev", required=True, dest="device", help="AIE Device (npu/npu2)"
    )
    p.add_argument("--in-time", type=int, default=2018,
                   help="Padded input length per channel (default 2018)")
    p.add_argument("--out-time", type=int, default=334,
                   help="Output time length (default 334)")
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
    module = my_dorado_fast_conv_stem_layer3(
        dev, T_in_padded=opts.in_time, T_out=opts.out_time
    )
    print(module)
