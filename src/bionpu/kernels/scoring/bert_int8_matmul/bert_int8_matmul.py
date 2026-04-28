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
#   scales:      N + 1  fp32           (combined per-output FP32 scales)
#   y:           M × N  int8           (token-major)
#
# Topology (single column, single compute tile):
#   shim ──x── compute_tile ──y── shim
#   shim ──w── compute_tile
#   shim ──s── compute_tile

from __future__ import annotations

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1


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
        raise ValueError(f"unknown device target: {target!r}")

    # ObjectFifo element types — flat byte arrays per IRON convention.
    #
    # AIE2P CoreTiles only have 2 input + 2 output DMA channels, so
    # we cannot afford 3 separate input ObjectFifos (x, w, s). The
    # smaller two (w + scales) are concatenated into a single byte
    # buffer; the kernel C++ slices it back into typed pointers via
    # known byte offsets. Layout (head shape):
    #   ws[0 .. N*K - 1]                  = int8 weights (N rows × K cols)
    #   ws[N*K .. N*K + (N+1)*4 - 1]      = float32 fused scales (N+1 entries)
    # AIE2P shim DMA requires transfer length to be a multiple of 4
    # bytes (`aie.dma_bd` op constraint). Pad each fifo size up to the
    # next multiple of 4 so the lowering succeeds. The kernel C++
    # writes only the useful prefix; the host runner reads the same
    # useful prefix and ignores any trailing pad bytes.
    def _round4(n: int) -> int:
        return ((n + 3) // 4) * 4

    x_size = _round4(M * K)
    ws_bytes = _round4(N * K + (N + 1) * 4)   # int8 weights + float32 scales
    y_size = _round4(M * N)
    x_ty  = np.ndarray[(x_size,), np.dtype[np.int8]]
    ws_ty = np.ndarray[(ws_bytes,), np.dtype[np.int8]]
    y_ty  = np.ndarray[(y_size,), np.dtype[np.int8]]

    # depth=1 in v0 — there's only one chunk per launch and the
    # 36 KB x buffer is large enough that depth=2 (72 KB) blows the
    # 64 KB tile DM cap. v0.5 will introduce K-chunking so each
    # ObjectFifo holds a small slice and depth=2 fits comfortably.
    of_x  = ObjectFifo(x_ty,  name="x_in",  depth=1)
    of_ws = ObjectFifo(ws_ty, name="ws_in", depth=1)
    of_y  = ObjectFifo(y_ty,  name="y_out", depth=1)

    # Compute kernel — scalar inner loop, fused-scale requantize, INT8
    # saturate. Symbol is `bert_int8_matmul_<N>` so the same .py / .cc
    # compile to multiple specializations via the Makefile.
    matmul_sym = f"bert_int8_matmul_{N}"
    matmul = Kernel(
        matmul_sym,
        "bert_int8_matmul.o",
        [x_ty, ws_ty, y_ty],
    )

    # v0 = one chunk in / one chunk out per launch. v0.5 will
    # iterate K-chunks here for the larger qkvo / ffn shapes. The
    # Kernel must be passed *into* the body as an arg (not closed
    # over) so the IRON resolver can bind the symbol at lower-time.
    def core_body(of_x, of_ws, of_y, matmul_kernel):
        x_buf  = of_x.acquire(1)
        ws_buf = of_ws.acquire(1)
        y_buf  = of_y.acquire(1)
        matmul_kernel(x_buf, ws_buf, y_buf)
        of_x.release(1)
        of_ws.release(1)
        of_y.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_x.cons(), of_ws.cons(), of_y.prod(), matmul],
    )

    rt = Runtime()
    with rt.sequence(x_ty, ws_ty, y_ty) as (x, ws, y):
        rt.start(worker)
        rt.fill(of_x.prod(), x)
        rt.fill(of_ws.prod(), ws)
        rt.drain(of_y.cons(), y, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--target", choices=["npu", "npu2"], default="npu2")
    p.add_argument("--M", type=int, default=47)
    p.add_argument("--K", type=int, default=768)
    p.add_argument("--N", type=int, default=2)
    args = p.parse_args()
    sys.stdout.write(str(emit_mlir(M=args.M, K=args.K, N=args.N, target=args.target)))


if __name__ == "__main__":
    main()
