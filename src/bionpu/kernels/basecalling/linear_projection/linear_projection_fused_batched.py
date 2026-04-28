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

# linear_projection_fused_batched.py — IRON lowering for the batched
#                                       Dorado-fast CRF linear head.
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Batched variant of stage-3 fused-perts: process N_CHUNKS chunks of L
# timesteps in a single silicon dispatch. Single xclbin load per
# N_CHUNKS chunks, single weight slab (48 KiB bf16) shared across the
# batch, the IRON-side kernel walks input/output sequentially through
# the full batch.
#
# This mirrors the CRISPR pam_filter LAUNCH_CHUNKS pattern (see
# pam_filter.py:235): an env var multiplies the per-dispatch input/
# output size; the C++ kernel itself is unchanged (still one call per
# timestep — chunk-batching is purely a host-amortisation trick).
#
# The amortisation target: B3b's bench measured fused-perts kernel
# avg us/chunk = 10.65 ms vs total wall per chunk = 112 ms. The 100
# ms delta is host-side dispatch ceremony (xclbin reload, BO alloc,
# XRT submit, output sync). Batching N chunks into one dispatch
# collapses (some of) that overhead to once-per-N-chunks. As N grows,
# per-chunk wall should approach the kernel time.
#
# Wire format for an N_CHUNKS batch:
#   - input:  N_CHUNKS * L * HIDDEN bf16 (concatenated chunks)
#   - weight: WB_LEN bf16 (single 48-KiB slab; SHARED across batch)
#   - output: N_CHUNKS * L * OUT_DIM bf16 (concatenated chunks)
#
# Build via:
#   BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS=8 \
#     make NPU2=1 experiment=fused-batched seq=334 all
# Supported batch sizes: {1, 2, 4, 8}.

import argparse
import os
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

# Batched-dispatch knob. Mirrors the CRISPR pam_filter LAUNCH_CHUNKS
# pattern (pam_filter.py:235): the env var multiplies the per-dispatch
# input/output size. Default=1 builds an artifact byte-for-byte
# equivalent in shape to stage-3 fused-perts (the batched=1 sanity
# build).
BATCH_CHUNKS = int(os.environ.get("BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS", "1"))
if BATCH_CHUNKS <= 0:
    raise ValueError(
        "BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS must be > 0; "
        f"got {BATCH_CHUNKS}"
    )
# Supported set per the task contract. Unsupported values (e.g. 3, 5)
# would still emit MLIR but haven't been silicon-validated.
if BATCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        "BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS must be one of "
        f"{{1, 2, 4, 8}}; got {BATCH_CHUNKS}"
    )

try:
    from ml_dtypes import bfloat16  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover - ml_dtypes always present in env
    _BF16 = np.dtype(np.uint16)


def my_dorado_fast_linear_projection_fused_batched(dev, L: int,
                                                   n_chunks: int):
    """Return the MLIR for the batched fused linear projection.

    The C++ kernel symbol is unchanged from stage-3 fused-perts
    (``dorado_fast_linear_projection_fused_perts`` in
    linear_projection_fused.cc) — the chunk-outer batching lives in
    IRON, not in C++. Each kernel call still processes ONE timestep.

    Wire format for a batched dispatch:
      - input:  ``n_chunks * L * HIDDEN`` bf16 (chunks concatenated)
      - weight: ``WB_LEN`` bf16 (single 48-KiB slab; shared)
      - output: ``n_chunks * L * OUT_DIM`` bf16 (chunks concatenated)

    Tile-memory budget (depth=2 on input/output, depth=1 on weight):
      - weight (depth=1): 1 * 24576 * 2 = 49152 B (48 KiB)
      - input  (depth=2): 2 *    96 * 2 =   384 B
      - output (depth=2): 2 *   256 * 2 =  1024 B
      - stack  (~ same as stage 3)      =  1024 B
      Total ≈ 51.5 KiB. Within 64 KiB. The wider host-visible buffers
      live in shim-DMA address space, not on the compute tile.
    """
    in_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    weight_full_ty = np.ndarray[(WB_LEN,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(OUT_DIM,), np.dtype[_BF16]]

    # WIDENED runtime tensor types: input/output scale by n_chunks; the
    # weight slab is unchanged (single shared 48-KiB load). This is the
    # entire host-amortisation mechanism — N times more useful work
    # per xclbin load + BO alloc + XRT submit.
    total_in_len = n_chunks * L * HIDDEN
    total_out_len = n_chunks * L * OUT_DIM
    in_total_ty = np.ndarray[(total_in_len,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[(WB_LEN,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(total_out_len,), np.dtype[_BF16]]

    # SAME kernel symbol as stage-3 fused-perts. The C++ kernel
    # processes one timestep per call regardless of the batch size,
    # so we reuse linear_projection_fused.o byte-for-byte.
    linear_kernel = Kernel(
        "dorado_fast_linear_projection_fused_perts",
        "linear_projection_fused.o",
        [in_step_ty, weight_full_ty, out_step_ty],
    )

    # depth=1 on weight: acquired ONCE for the entire batched dispatch.
    # depth=2 on input/output: standard ping-pong streaming, drains
    # n_chunks * L per-timestep slots over the batch.
    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)
    of_weight = ObjectFifo(weight_full_ty, name="weight_in", depth=1)
    of_output = ObjectFifo(out_step_ty, name="output_out", depth=2)

    def core_body(of_input, of_weight, of_output, linear_fn):
        # FUSED_BATCHED_CHUNK_OUTER_LOOP_SENTINEL
        # Acquire the bf16 weight slab once for the whole batched
        # dispatch (n_chunks * L timesteps share the same weight).
        elem_w_full = of_weight.acquire(1)
        for chunk in range_(n_chunks):
            for t in range_(L):
                elem_in = of_input.acquire(1)
                elem_out = of_output.acquire(1)
                # One kernel call per timestep — same as stage 3.
                # The chunk-outer loop is purely host-amortisation:
                # we feed n_chunks * L timesteps through the same
                # tile-resident weight without re-loading.
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
    p.add_argument(
        "-n", "--n-chunks", dest="n_chunks", type=int, default=None,
        help=(
            "Number of chunks per dispatch. Defaults to "
            "BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS env var (or 1)."
        ),
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
    n_chunks = opts.n_chunks if opts.n_chunks is not None else BATCH_CHUNKS
    if n_chunks not in (1, 2, 4, 8):
        raise ValueError(
            f"--n-chunks must be one of {{1, 2, 4, 8}}; got {n_chunks}"
        )
    module = my_dorado_fast_linear_projection_fused_batched(
        dev, L=opts.seq, n_chunks=n_chunks,
    )
    print(module)
