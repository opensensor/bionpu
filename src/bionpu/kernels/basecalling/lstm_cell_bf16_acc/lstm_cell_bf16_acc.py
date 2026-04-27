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

# lstm_cell_bf16_acc.py — IRON lowering for Dorado fast LSTM cell, bf16
#                          input/output + FP32 recurrent state path
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Mixed-precision sibling of `bionpu/kernels/basecalling/lstm_cell_bf16
# /lstm_cell_bf16.py`.
#
# **The IRON-side topology is byte-identical to .** Same
# half-gate sharded DMA pattern ( forced bias-into-weight
# folding; that constraint is unchanged here — bf16 doesn't free a DMA
# channel and FP32-internal-state doesn't change the wire format
# either). Only the kernel binary differs (the C++ `dorado_fast_lstm_
# cell_bf16_acc` symbol).
#
# Why a separate IRON file even though the topology is identical?
# Because Kernel("...", "...o") binds to the C++ object file by name
# at IRON-resolution time. 's IRON file binds to
# `lstm_cell_bf16.o` and the C++ entry symbol `dorado_fast_lstm_cell_
# bf16`; needs to bind to `lstm_cell_bf16_acc.o` and
# `dorado_fast_lstm_cell_bf16_acc`. Re-using 's IRON would
# either fight the symbol resolution or force editing (which
# the task brief forbids).
#
# Cross-walk note (AM020 §): The hardware-natively-supported
# AM-to-AM register move primitive (Ch. 4 p. 67, 512 bits/cycle) is
# **not exposed at the IRON layer** — there is no
# `Worker.persist_accumulator(...)` or analogous API. Likewise the
# cascade stream primitive (Ch. 4 p. 67) is not exposed for AIE2P at
# the IRON level. The C++ kernel therefore falls back to FP32
# tile-DM static storage to preserve the same precision invariant
# (since the accumulator-to-FP32-store path is lossless on AIE-ML/AIE2P
# per AM020 Ch. 4 p. 65). This **is the toolchain gap** the task
# brief calls out as a likely real RQ4 finding; documented in
# `gaps.yaml`.

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

try:
    from ml_dtypes import bfloat16  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover - ml_dtypes always present in env
    _BF16 = np.dtype(np.uint16)

def my_dorado_fast_lstm_cell_bf16_acc(dev, L: int):
    """Return the MLIR for one mixed-precision LSTM-cell forward
    over L timesteps.

    Wire-format identical to 's bf16 cell: bf16 inputs, bf16
    weights+biases, bf16 outputs, same chunked DMA. The change is
    internal: the compute-tile kernel keeps gate_acc, h_state, c_state
    at FP32 between timesteps (preserves 23-mantissa-bit recurrent
    precision per AM020 cross-walk). See lstm_cell_bf16_acc.cc for the
    storage-discipline rationale.
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
        "dorado_fast_lstm_cell_bf16_acc",
        "lstm_cell_bf16_acc.o",
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
    module = my_dorado_fast_lstm_cell_bf16_acc(dev, L=opts.seq)
    print(module)
