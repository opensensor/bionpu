# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_int8_matmul — IRON-Python lowering for the BERT body INT8 matmul.
#
# The workhorse kernel of the DNABERT-Epi AIE2P scorer port. v0 is a
# single-compute-tile correctness path; the weight matrix is
# tile-resident (works for the classifier head: M=47, K=768, N=2).
# Memtile streaming for larger N (qkvo: 768; ffn1/2: 3072) is the
# v0.5 follow-up — see MANIFEST.md.
#
# Shapes (per CLI flags, default = head specialization):
#   x:           M × K  int8           (token-major)
#   w:           N × K  int8           (output-channel-major)
#   scales:      K + 1  fp32           (combined per-output FP32 scales)
#   y:           M × N  int8           (token-major)
#
# Topology:
#   shim ──x ──→ compute_tile
#   shim ──w ──→ compute_tile
#   shim ──s ──→ compute_tile
#   compute_tile ──y ──→ shim
#
# The four ObjectFifos all live on the same column. v0.5 will split
# w (and possibly y) across N-axis to multiple tiles when shape grows.

from __future__ import annotations

import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.dialects.aie import device
from aie.helpers.dialects.ext.scf import _for as range_

# Element types — IRON's MemRef-style declarations.
from aie.dialects.aiex import npu_dma_memcpy_nd  # noqa: F401  (helps lowering import)
from aie.helpers.taplib import TensorAccessPattern  # noqa: F401  (memtile path)


def emit_mlir(M: int, K: int, N: int, target: str = "npu2") -> str:
    """Emit IRON Python -> MLIR-AIE for the given matmul shape.

    Returns the MLIR-AIE module as a string. Caller pipes to
    aiecc to produce the xclbin.
    """
    if target == "npu2":
        dev = NPU2Col1()
    elif target == "npu":
        dev = NPU1Col1()
    else:
        raise ValueError(f"unknown target: {target!r}")

    # ObjectFifo element types: int8 buffers + one float32 scales buffer.
    # IRON's array typing uses (shape, elemTy) tuples on the Iron API.
    # NB: the head shape (M=47, K=768, N=2) gives:
    #   x  =  47 × 768 = 36 096  bytes (one shim DMA fill)
    #   w  =   2 × 768 =  1 536  bytes
    #   s  =   2 + 1   = 12     bytes (combined per-output + bias)
    #   y  =  47 × 2   = 94     bytes (round up for DMA alignment)
    # Tile-resident: every input fits inside the compute tile DM at
    # once. v0 keeps the kernel single-pass (no chunking inside the
    # tile); v0.5 introduces an outer K-chunking loop for the larger
    # qkvo / ffn shapes.

    of_x = ObjectFifo("x_in", (M, K), depth=2, elemTy="i8")
    of_w = ObjectFifo("w_in", (N, K), depth=2, elemTy="i8")
    of_s = ObjectFifo("s_in", (N + 1,), depth=2, elemTy="f32")
    of_y = ObjectFifo("y_out", (M, N), depth=2, elemTy="i8")

    # Compute kernel — scalar inner loop, fused-scale requantize, int8
    # saturate. The C++ symbol is `bert_int8_matmul_head` for the head
    # specialization; rebuild with `make M=... K=... N=...` to produce
    # the qkvo / ffn variants. The kernel signature mirrors the
    # ObjectFifo element layouts above.
    matmul = Kernel(
        f"bert_int8_matmul_{N}",          # symbol resolved per-shape
        "bert_int8_matmul.o",
        [(M, K), (N, K), (N + 1,), (M, N)],
    )

    def core_body(of_x, of_w, of_s, of_y):
        # One launch per (input chunk × weight chunk). v0 = one chunk
        # of each; v0.5 will iterate K-chunks here.
        x_buf = of_x.acquire(1)
        w_buf = of_w.acquire(1)
        s_buf = of_s.acquire(1)
        y_buf = of_y.acquire(1)
        matmul(x_buf, w_buf, s_buf, y_buf)
        of_x.release(1)
        of_w.release(1)
        of_s.release(1)
        of_y.release(1)

    worker = Worker(core_body, [of_x.cons(), of_w.cons(), of_s.cons(), of_y.prod()])

    rt = Runtime()
    with rt.sequence(of_x.prod(), of_w.prod(), of_s.prod(), of_y.cons()) as (x, w, s, y):
        # The host runner (runner.cpp) writes x/w/s into shim BD slots;
        # the runtime sequence below dispatches a single launch per
        # input chunk.
        rt.start(worker)
        rt.fill(of_x.prod(), x)
        rt.fill(of_w.prod(), w)
        rt.fill(of_s.prod(), s)
        rt.drain(of_y.cons(), y, wait=True)

    program = Program(dev, rt)
    placer = SequentialPlacer()
    return program.resolve_program(placer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--target", choices=["npu", "npu2"], default="npu2")
    p.add_argument("--M", type=int, default=47, help="batch × seq length")
    p.add_argument("--K", type=int, default=768, help="reduction axis (BERT hidden)")
    p.add_argument("--N", type=int, default=2, help="output channels")
    args = p.parse_args()
    sys.stdout.write(emit_mlir(M=args.M, K=args.K, N=args.N, target=args.target))


if __name__ == "__main__":
    main()
