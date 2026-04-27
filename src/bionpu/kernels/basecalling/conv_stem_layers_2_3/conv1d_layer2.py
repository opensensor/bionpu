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

# conv1d_layer2.py — IRON lowering for Dorado fast Conv stem layer 2 -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Lowers Dorado `fast@v5.0.0` second Conv1d stem layer
# (in=16, out=16, kernel=5, stride=1, padding=2, bias=true) onto a single
# AIE2P tile. Multi-channel sibling of 's `conv1d_layer1.py`. The
# kernel proper is in `conv1d_layer2.cc`; this script emits the MLIR-AIE
# module that wires its input/weight/output through ObjectFifos.
#
# # Tile-memory budget (per-chunk, depth=2)
#   - signal_in:  2 * (16 * (T_chunk + 4)) * 4B  (per-chunk multi-channel input)
#   - wb_in:      2 * 1296 * 4B                  (16*16*5 weights + 16 bias)
#   - output_out: 2 * (16 * T_chunk) * 4B        (per-chunk output)
# For T_chunk=200:
#   signal: 2 * 16 * 204 * 4 = 26112 B
#   wb:     2 * 1296 * 4    = 10368 B
#   output: 2 * 16 * 200 * 4 = 25600 B
#   total ≈ 62 KiB — uncomfortable on a 64 KiB tile.
# So we drop signal_in / output_out depth to 1 (single-buffered), keeping
# wb double-buffered for steady-state weight reuse:
#   signal: 1 * 16 * 204 * 4 = 13056 B
#   wb:     2 * 1296 * 4    = 10368 B
#   output: 1 * 16 * 200 * 4 = 12800 B
#   stack:  1024 B
#   total ≈ 37 KiB — within the 64 KiB budget.
#
# Layout pinned with the host runner:
#   - in1 (signal): float32[N * 16 * (T_chunk+4)]; chunk-major then
#     channel-major then time-major.
#   - in2 (wb):     float32[16*16*5 + 16] = float32[1296]; weights
#     in (oc, ic, k) row-major then bias (oc,).
#   - out:          float32[N * 16 * T_chunk]; chunk-major then
#     out-channel-major then time-major.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

CONV_K = 5
CONV_IN_CH = 16
CONV_OUT_CH = 16
PAD = 2

def my_dorado_fast_conv_stem_layer2(dev, T: int, T_chunk: int):
    if T % T_chunk != 0:
        raise ValueError(
            f"T ({T}) must be a multiple of T_chunk ({T_chunk})."
        )
    N = T // T_chunk

    chunk_in_len = T_chunk + 2 * PAD
    chunk_out_len = T_chunk

    # Whole-runtime tensors (host-side flat buffers)
    sig_total_len = N * CONV_IN_CH * chunk_in_len
    out_total_len = N * CONV_OUT_CH * chunk_out_len

    wb_len = CONV_OUT_CH * CONV_IN_CH * CONV_K + CONV_OUT_CH

    sig_chunk_ty = np.ndarray[
        (CONV_IN_CH * chunk_in_len,), np.dtype[np.float32]
    ]
    wb_ty = np.ndarray[(wb_len,), np.dtype[np.float32]]
    out_chunk_ty = np.ndarray[
        (CONV_OUT_CH * chunk_out_len,), np.dtype[np.float32]
    ]

    sig_total_ty = np.ndarray[(sig_total_len,), np.dtype[np.float32]]
    out_total_ty = np.ndarray[(out_total_len,), np.dtype[np.float32]]

    conv_kernel = Kernel(
        "dorado_fast_conv_stem_layer2_fp32",
        "conv1d_layer2.o",
        [sig_chunk_ty, wb_ty, out_chunk_ty, np.int32],
    )

    # Single-buffered signal (depth=1) to fit 64 KiB tile budget; output
    # depth=2 to allow ping-pong (the v1=depth=1 path produced a 9-position
    # NaN window at positions 43..51 of every chunk; see gaps.yaml
    # ).
    of_signal = ObjectFifo(sig_chunk_ty, name="signal_in", depth=1)
    of_wb = ObjectFifo(wb_ty, name="wb_in")
    of_output = ObjectFifo(out_chunk_ty, name="output_out", depth=2)

    def core_body(of_signal, of_wb, of_output, conv_fn):
        elem_wb = of_wb.acquire(1)
        for _ in range_(N):
            elem_signal = of_signal.acquire(1)
            elem_out = of_output.acquire(1)
            conv_fn(elem_signal, elem_wb, elem_out, chunk_out_len)
            of_signal.release(1)
            of_output.release(1)
        of_wb.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_signal.cons(), of_wb.cons(), of_output.prod(), conv_kernel],
    )

    rt = Runtime()
    with rt.sequence(sig_total_ty, wb_ty, out_total_ty) as (S, W, O):
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
    p.add_argument(
        "-t", "--time", type=int, default=2000,
        help="Total output time T (default 2000)",
    )
    p.add_argument(
        "-c", "--chunk", type=int, default=200,
        help="Per-iteration output chunk size T_chunk (default 200)",
    )
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
    module = my_dorado_fast_conv_stem_layer2(
        dev, T=opts.time, T_chunk=opts.chunk
    )
    print(module)
