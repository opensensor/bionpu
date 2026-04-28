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

# linear_projection_fused_dispatch.py — IRON lowering for stage-4
#                                        Dorado-fast CRF linear head.
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# DESIGN-fusion.md stage 4 ("single kernel call per dispatch").
#
# IMPORTANT — fallback rationale
# ===============================
# The literal stage-4 spec wants the kernel function to receive
# ObjectFifo handles + a count L and walk the fifos for all L
# timesteps inside C++ (one func.call per dispatch). This is NOT
# expressible at the current IRON-Python layer:
#
#   * aie.iron.Kernel only accepts arg types of (numpy ndarray /
#     numpy dtype / scalar). There is no objectfifo-handle ABI to
#     C++ kernels.
#   * No equivalent of aie.objectfifo.acquire / release is exposed
#     to peano-compiled C++ kernels — the stateful-transform pass
#     emits acquire/release at the MLIR layer, lowered from
#     scf.for-driven IRON Python.
#
# As authorized by the prompt's done-criteria escape hatch
# ("If you hit a fundamental ObjectFifo-passing-into-kernel
# limitation in IRON Python, document it in gaps.yaml + fall back
# to the documented Python-loop-with-runtime-SCF approach"), this
# file emits the same SCF-driven shape as stage 3, but registers a
# distinct entry symbol so the dispatch layer can A/B between them.
# See gaps.yaml#B1-fused-dispatch-kernel-fifo-walk for the analysis.
#
# What's still meaningfully "stage 4" about this artifact:
#
#   * Distinct kernel symbol (dorado_fast_linear_projection_fused_dispatch)
#     and distinct artifact directory.
#   * Inner-loop micro-optimisation in the C++ kernel
#     (aie::reduce_add for the per-OC reduction tail instead of a
#     manual scalar sum).
#   * Identical wire format to stage 3 (bf16 input + bf16 weight slab
#     + bf16 output) so test fixtures and silicon-validation script
#     reuse cleanly.

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


def my_dorado_fast_linear_projection_fused_dispatch(dev, L: int):
    """Return the MLIR for the stage-4 fused linear projection.

    See top-of-file rationale for why this collapses to stage-3's
    SCF-loop shape; the structural difference is the kernel symbol
    and the inner-loop micro-opt in linear_projection_fused_dispatch.cc.
    Wire format: bf16 input (L * 96 elements), bf16 weight (24576;
    full slab), bf16 output (L * 256 elements).
    """
    in_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    weight_full_ty = np.ndarray[(WB_LEN,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(OUT_DIM,), np.dtype[_BF16]]

    in_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[(WB_LEN,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * OUT_DIM,), np.dtype[_BF16]]

    # Stage-4 entry symbol — distinct from stage 3
    # (dorado_fast_linear_projection_fused_perts) so the dispatch
    # registry can pick whichever artifact is built / faster.
    linear_kernel = Kernel(
        "dorado_fast_linear_projection_fused_dispatch",
        "linear_projection_fused_dispatch.o",
        [in_step_ty, weight_full_ty, out_step_ty],
    )

    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)
    of_weight = ObjectFifo(weight_full_ty, name="weight_in", depth=1)
    of_output = ObjectFifo(out_step_ty, name="output_out", depth=2)

    def core_body(of_input, of_weight, of_output, linear_fn):
        # Acquire the full bf16 weight slab once at kernel start
        # (DESIGN-fusion.md stage 4, point 1).
        elem_w_full = of_weight.acquire(1)
        # FUSED_DISPATCH_SINGLE_CALL_PER_DISPATCH_SENTINEL_PY
        # The "for t in range_(L)" + per-iteration func.call is the
        # MLIR-level shape of stage 3. Stage 4's "one call per
        # dispatch" interpretation is implemented at the kernel-
        # symbol layer (distinct entry point) because the IRON
        # Kernel ABI doesn't support objectfifo-handle args. See
        # the top-of-file fallback rationale.
        for t in range_(L):
            elem_in = of_input.acquire(1)
            elem_out = of_output.acquire(1)
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
    module = my_dorado_fast_linear_projection_fused_dispatch(dev, L=opts.seq)
    print(module)
