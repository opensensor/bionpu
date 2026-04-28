# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_mini_gelu — IRON-Python lowering for the BERT-mini FFN GELU.
# Per-row tanh-approximation GELU over a (HIDDEN_PAD,) bf16 vector.
#
# Mirrors bert_mini_softmax.py's per-row dispatch discipline:
# host loops over (M * NUM_GELU_ROWS_PER_CALL) rows externally and the
# IRON range_ loop runs one kernel invocation per row. NUM_GELU_ROWS
# defaults to M=47 (one row per FFN expansion-layer activation token).
#
# Tile DM budget at HIDDEN=1024 / depth=2 (BERT-mini FFN expansion):
#   in row  (1024 * 2 * 2) = 4096 B
#   out row (1024 * 2 * 2) = 4096 B
#   stack                 ≈ 1024 B
#   total                 ≈ 9 KB / 64 KB

from __future__ import annotations

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1


_M_DEFAULT = 47
_HIDDEN_DEFAULT = 1024  # BERT-mini FFN expansion dim (4 × hidden=256)


def _round4(n: int) -> int:
    return ((n + 3) // 4) * 4


def emit_mlir(M: int = _M_DEFAULT,
              HIDDEN: int = _HIDDEN_DEFAULT,
              target: str = "npu2") -> str:
    if target == "npu2":
        dev = NPU2Col1()
    elif target == "npu":
        dev = NPU1Col1()
    else:
        raise ValueError(f"unknown device target: {target!r}")

    if HIDDEN % 16 != 0:
        raise ValueError(
            f"GELU HIDDEN ({HIDDEN}) must be a multiple of 16 (bf16 lane "
            f"for the upstream tanh-approximation reference)"
        )

    NUM_ROWS = M
    in_row_size = _round4(HIDDEN * 2)        # bf16 = 2 bytes
    out_row_size = _round4(HIDDEN * 2)
    in_total_bytes = _round4(NUM_ROWS * in_row_size)
    out_total_bytes = _round4(NUM_ROWS * out_row_size)

    row_ty = np.ndarray[(HIDDEN,), np.dtype[np.uint16]]
    in_total_ty = np.ndarray[(NUM_ROWS * HIDDEN,), np.dtype[np.uint16]]
    out_total_ty = np.ndarray[(NUM_ROWS * HIDDEN,), np.dtype[np.uint16]]

    of_in = ObjectFifo(row_ty, name="gelu_in", depth=2)
    of_out = ObjectFifo(row_ty, name="gelu_out", depth=2)

    gelu = Kernel(
        "bert_mini_gelu",
        "bert_mini_block.o",
        [row_ty, row_ty],
    )

    def core_body(of_in, of_out, gelu):
        for _ in range_(NUM_ROWS):
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            gelu(x, y)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), gelu],
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
    p.add_argument("--HIDDEN", type=int, default=_HIDDEN_DEFAULT)
    args = p.parse_args()
    sys.stdout.write(str(emit_mlir(M=args.M, HIDDEN=args.HIDDEN, target=args.target)))


if __name__ == "__main__":
    main()
