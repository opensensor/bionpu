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

# linear_projection_fused.py — IRON lowering for stage-3 fused
#                                Dorado-fast CRF linear head.
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# DESIGN-fusion.md stage 3 ("per-timestep fused call"). Same math as
# linear_projection.py but:
#
#   - weights are bf16 (precision-neutral on AIE2P; multiplier already
#     narrows fp32 inputs to bf16 per AM020 Appendix A)
#   - weights are acquired ONCE per dispatch as a single (256, 96) bf16
#     slab — 48 KiB; fits cleanly in tile L1
#   - per-timestep loop in the kernel core_body: for each t in 0..L,
#     acquire 96-bf16 input, compute the full 256-output GEMM (looping
#     over the 4 OC groups internally with weight-stationary access),
#     release 256-bf16 output
#
# Total kernel function calls per dispatch: L (one per timestep) instead
# of L * 4 (one per (timestep, OC group) tuple).
#
# Tile-memory budget per dispatch (depth=2 on input/output, depth=1 on
# the load-once weight slab):
#   - weight (depth=1): 1 * 24576 * 2 = 49152 B (48 KiB)
#   - input  (depth=2): 2 *    96 * 2 =   384 B
#   - output (depth=2): 2 *   256 * 2 =  1024 B
#   - stack  (~ same as production)  =  1024 B
#   Total ≈ 51.5 KiB. Within 64 KiB.
#
# Wire format note: the host packs the weight as ONE 48-KiB bf16 slab
# (no per-timestep repetition); the per-timestep input is bf16; output
# is bf16. The host runner converts fp32 inputs to bf16 host-side and
# bf16 outputs back to fp32 (matching the lstm_cell_bf16 convention).

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
WB_LEN = OUT_DIM * HIDDEN  # 24576

try:
    from ml_dtypes import bfloat16  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover - ml_dtypes always present in env
    _BF16 = np.dtype(np.uint16)


def my_dorado_fast_linear_projection_fused_perts(dev, L: int):
    """Return the MLIR for the stage-3 fused linear projection.

    See DESIGN-fusion.md for the precision contract and the
    weight-acquire-once mechanism. Wire format: bf16 input
    (L * 96 elements), bf16 weight (24576 elements; full slab,
    no per-timestep repetition), bf16 output (L * 256 elements).
    """
    in_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    weight_full_ty = np.ndarray[(WB_LEN,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(OUT_DIM,), np.dtype[_BF16]]

    in_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[(WB_LEN,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * OUT_DIM,), np.dtype[_BF16]]

    # Symbol: dorado_fast_linear_projection_fused_perts in
    # linear_projection_fused.cc. Distinct from the per-group fp32
    # symbol so the existing artifact's behaviour is unchanged.
    linear_kernel = Kernel(
        "dorado_fast_linear_projection_fused_perts",
        "linear_projection_fused.o",
        [in_step_ty, weight_full_ty, out_step_ty],
    )

    # depth=1 on the weight fifo: the kernel acquires once and holds
    # the slab for the entire dispatch. depth=2 on input/output gives
    # the standard ping-pong streaming.
    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)
    of_weight = ObjectFifo(weight_full_ty, name="weight_in", depth=1)
    of_output = ObjectFifo(out_step_ty, name="output_out", depth=2)

    def core_body(of_input, of_weight, of_output, linear_fn):
        # Acquire the full bf16 weight slab once at kernel start
        # (DESIGN-fusion.md stage 3, point 2).
        elem_w_full = of_weight.acquire(1)
        for t in range_(L):
            elem_in = of_input.acquire(1)
            elem_out = of_output.acquire(1)
            # One kernel call per timestep; the C++ kernel walks the
            # 4 OC groups internally with weight-stationary access
            # over elem_w_full.
            linear_fn(elem_in, elem_w_full, elem_out)
            of_input.release(1)
            of_output.release(1)
        of_weight.release(1)

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
    module = my_dorado_fast_linear_projection_fused_perts(dev, L=opts.seq)
    print(module)
