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

# conv1d_layer1.py — IRON lowering for Dorado fast stem Conv1d -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Lowers the first Dorado `fast@v5.0.0` Conv1d stem layer
# (in=1, out=16, kernel=5, stride=1, padding=2, bias=true) onto a single
# AIE2P tile. The kernel proper is in `conv1d_layer1.cc`; this script
# emits the MLIR-AIE module that wires its input/weight/output through
# ObjectFifos.
#
# Modeled on `mlir-aie/programming_examples/basic/vector_scalar_mul/
# vector_scalar_mul.py` (the AIE2P-tested IRON pattern T0.1 verifies);
# the conv2d example targets AIE2 1x1 i8 and is not a clean match for
# Dorado's FP32 1D path.
#
# # Tile-memory budget
#
# AIE2P tile data memory is 64 KiB total (4 × 16 KiB banks). The full
# (1, 16, T=2000) FP32 output (128 KB) doesn't fit, so we stream the
# work in N chunks of `T_chunk` output samples each. Per-chunk
# tile-local footprint with depth=2 ObjectFifos:
#
#   - signal_in:     2 × (T_chunk + 4) × 4 B
#   - wb_in:         2 × 96 × 4 B  (weights + bias)
#   - output_out:    2 × (16 × T_chunk) × 4 B
#
# For T_chunk = 200 this is ≈ 28 KiB, well inside 64 KiB and matching
# the slack we want for the conv stem's BN-folded successor in .
#
# # Buffer layout pinned with the host runner
#
#   - in1 (signal): float32[N × (T_chunk+4)]. The host pre-pads the
#     full signal with 2 leading + 2 trailing zeros (Conv1d padding=2)
#     and splits into N overlapping chunks where chunk i starts at
#     offset i*T_chunk in the padded buffer; consecutive chunks share
#     the 4-sample boundary context.
#   - in2 (wb):     float32[80 + 16] = float32[96]. Weights (16, 5)
#     row-major then bias (16,). Loaded once per run.
#   - out:          float32[N × 16 × T_chunk]. NCL contiguous within
#     each chunk; the host concatenates chunks along the time axis to
#     recover the (1, 16, T) result.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

CONV_K = 5
CONV_IN_CH = 1
CONV_OUT_CH = 16
PAD = 2  # padding=2 on each side per architecture.md §1

def my_dorado_fast_conv_stem_layer1(dev, T: int, T_chunk: int):
    """Return the MLIR module for total time T split into N=T/T_chunk chunks."""
    if T % T_chunk != 0:
        raise ValueError(
            f"T ({T}) must be a multiple of T_chunk ({T_chunk}); pick "
            f"T_chunk that divides {T} evenly."
        )
    N = T // T_chunk

    # Per-chunk tile-resident shapes
    chunk_in_len = T_chunk + 2 * PAD  # 4-sample overlap per chunk
    chunk_out_len = T_chunk

    # Whole-runtime tensors (host-side flat buffers)
    sig_total_len = N * chunk_in_len
    out_total_len = N * CONV_OUT_CH * chunk_out_len

    wb_len = CONV_OUT_CH * CONV_K + CONV_OUT_CH

    # Tile-resident type per-chunk
    sig_chunk_ty = np.ndarray[(chunk_in_len,), np.dtype[np.float32]]
    wb_ty = np.ndarray[(wb_len,), np.dtype[np.float32]]
    out_chunk_ty = np.ndarray[(CONV_OUT_CH * chunk_out_len,), np.dtype[np.float32]]

    # Whole-runtime types
    sig_total_ty = np.ndarray[(sig_total_len,), np.dtype[np.float32]]
    out_total_ty = np.ndarray[(out_total_len,), np.dtype[np.float32]]

    # External AIE-tile kernel definition. Entry symbol must match
    # `extern "C"` in conv1d_layer1.cc.
    conv_kernel = Kernel(
        "dorado_fast_conv_stem_layer1_fp32",
        "conv1d_layer1.o",
        [sig_chunk_ty, wb_ty, out_chunk_ty, np.int32],
    )

    # Three ObjectFifos: signal stream (per-chunk), weights+bias (once),
    # output stream (per-chunk). depth=2 enables double-buffering on
    # the per-chunk fifos.
    of_signal = ObjectFifo(sig_chunk_ty, name="signal_in")
    of_wb = ObjectFifo(wb_ty, name="wb_in")
    of_output = ObjectFifo(out_chunk_ty, name="output_out")

    # Core body: load wb once, then loop over N chunks.
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
        help="Per-iteration output chunk size T_chunk (default 200; must divide T)",
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
    module = my_dorado_fast_conv_stem_layer1(
        dev, T=opts.time, T_chunk=opts.chunk
    )
    print(module)
