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

# linear_projection.py — IRON lowering for Dorado fast CRF linear head -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Lowers `Linear(96 -> 256, bias=False) + Clamp(-5, 5)` (the Dorado
# fast CRF head, per architecture.md §3 and dorado_fast.py CRFHead) onto
# a single AIE2P tile with output-dim sharding.
#
# Per kernel call: process one (timestep, OC group) tuple. There are
# T_LSTM=334 timesteps × N_OC_GROUPS=4 groups = 1336 calls. Per call:
#   - x_t: HIDDEN=96 floats
#   - W_group: OC_GROUP_SIZE * HIDDEN = 64 * 96 = 6144 floats (24 KiB)
#   - y_group: OC_GROUP_SIZE = 64 floats
#
# Tile-memory budget per call (depth=2 weight buffer + tiny input/output):
#   - weight (depth=2): 2 * 6144 * 4 = 49152 B (48 KiB)
#   - input (depth=2): 2 * 96 * 4 = 768 B
#   - output (depth=2): 2 * 64 * 4 = 512 B
#   - stack: 1024 B
#   Total ≈ 51 KiB. Within 64 KiB.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

HIDDEN = 96
OUT_DIM = 256
OC_GROUP_SIZE = 64
N_OC_GROUPS = OUT_DIM // OC_GROUP_SIZE  # 4
WB_PER_GROUP = OC_GROUP_SIZE * HIDDEN  # 6144

def my_dorado_fast_linear_projection(dev, L: int):
    in_step_ty = np.ndarray[(HIDDEN,), np.dtype[np.float32]]
    weight_chunk_ty = np.ndarray[(WB_PER_GROUP,), np.dtype[np.float32]]
    out_chunk_ty = np.ndarray[(OC_GROUP_SIZE,), np.dtype[np.float32]]

    in_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[np.float32]]
    weight_total_ty = np.ndarray[
        (L * N_OC_GROUPS * WB_PER_GROUP,), np.dtype[np.float32]
    ]
    out_total_ty = np.ndarray[
        (L * N_OC_GROUPS * OC_GROUP_SIZE,), np.dtype[np.float32]
    ]

    linear_kernel = Kernel(
        "dorado_fast_linear_projection_fp32",
        "linear_projection.o",
        [in_step_ty, weight_chunk_ty, out_chunk_ty],
    )

    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)
    of_weight = ObjectFifo(weight_chunk_ty, name="weight_in", depth=2)
    of_output = ObjectFifo(out_chunk_ty, name="output_out", depth=2)

    def core_body(of_input, of_weight, of_output, linear_fn):
        for t in range_(L):
            elem_in = of_input.acquire(1)
            for g in range_(N_OC_GROUPS):
                elem_w = of_weight.acquire(1)
                elem_out = of_output.acquire(1)
                linear_fn(elem_in, elem_w, elem_out)
                of_weight.release(1)
                of_output.release(1)
            of_input.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_input.cons(), of_weight.cons(), of_output.prod(),
                 linear_kernel],
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
    p.add_argument("-L", "--seq", type=int, default=334)
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
    module = my_dorado_fast_linear_projection(dev, L=opts.seq)
    print(module)
