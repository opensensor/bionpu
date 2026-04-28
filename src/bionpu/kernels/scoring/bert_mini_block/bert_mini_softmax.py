# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_mini_softmax — IRON-Python lowering for the BERT-mini attention
# softmax. Per-row reduction over a (PAD,) bf16 vector. Host loops over
# M=47 rows × NUM_HEADS=4 heads externally and submits one launch per
# row; per-row dispatch keeps the tile-DM budget tiny (PAD * 2 bytes
# in + PAD * 2 bytes out = 256 B for PAD=64).
#
# The host wire format batches all (M*HEADS) rows into the single
# launch: in_total = M*HEADS*PAD bf16, out_total = M*HEADS*PAD bf16.
# This matches the dispatch-overhead-amortised pattern used by the
# qkvo lowering — one shim DMA, multiple per-row reductions inside
# the IRON `range_` loop.

from __future__ import annotations

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1


# Default shape for BERT-mini. Per-head softmax is (M, M); padded to
# PAD elements per row so the reduction vector is a multiple of the
# bf16 lane width on AIE2P (32).
_M_DEFAULT = 47
_PAD_DEFAULT = 64
_NUM_HEADS_DEFAULT = 4

try:
    from ml_dtypes import bfloat16  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover
    _BF16 = np.dtype(np.uint16)


def _round4(n: int) -> int:
    return ((n + 3) // 4) * 4


def emit_mlir(M: int = _M_DEFAULT,
              PAD: int = _PAD_DEFAULT,
              NUM_HEADS: int = _NUM_HEADS_DEFAULT,
              target: str = "npu2") -> str:
    if target == "npu2":
        dev = NPU2Col1()
    elif target == "npu":
        dev = NPU1Col1()
    else:
        raise ValueError(f"unknown device target: {target!r}")

    if PAD % 32 != 0:
        raise ValueError(f"PAD ({PAD}) must be a multiple of 32 (bf16 lane)")

    # Per-row element shapes (one row per kernel invocation; one tile
    # processes all NUM_ROWS=M*NUM_HEADS rows in sequence inside
    # range_).
    NUM_ROWS = M * NUM_HEADS
    in_row_size = _round4(PAD * 2)        # bf16 = 2 bytes
    out_row_size = _round4(PAD * 2)
    in_total_bytes = _round4(NUM_ROWS * in_row_size)
    out_total_bytes = _round4(NUM_ROWS * out_row_size)

    # Element types. IRON ObjectFifo carries bf16 as uint16 byte storage.
    row_ty = np.ndarray[(PAD,), np.dtype[np.uint16]]
    in_total_ty = np.ndarray[(NUM_ROWS * PAD,), np.dtype[np.uint16]]
    out_total_ty = np.ndarray[(NUM_ROWS * PAD,), np.dtype[np.uint16]]

    # ObjectFifos — one row in, one row out per iteration; depth=2 for
    # ping-pong overlap of DMA + compute on AIE2P.
    of_in = ObjectFifo(row_ty, name="sm_in", depth=2)
    of_out = ObjectFifo(row_ty, name="sm_out", depth=2)

    softmax = Kernel(
        "bert_mini_attention_softmax",
        "bert_mini_block.o",
        [row_ty, row_ty],
    )

    def core_body(of_in, of_out, softmax):
        for _ in range_(NUM_ROWS):
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            softmax(x, y)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), softmax],
    )

    rt = Runtime()
    with rt.sequence(in_total_ty, out_total_ty) as (x_total, y_total):
        rt.start(worker)
        rt.fill(of_in.prod(), x_total)
        rt.drain(of_out.cons(), y_total, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--target", choices=["npu", "npu2"], default="npu2")
    p.add_argument("--M", type=int, default=_M_DEFAULT)
    p.add_argument("--PAD", type=int, default=_PAD_DEFAULT)
    p.add_argument("--NUM_HEADS", type=int, default=_NUM_HEADS_DEFAULT)
    args = p.parse_args()
    sys.stdout.write(str(emit_mlir(
        M=args.M, PAD=args.PAD, NUM_HEADS=args.NUM_HEADS,
        target=args.target,
    )))


if __name__ == "__main__":
    main()
