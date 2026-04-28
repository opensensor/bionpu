# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_mini_layer_norm — IRON-Python lowering for BERT-mini LayerNorm.
# Per-row reduction over the hidden dim; host loops over M=47 rows
# inside the IRON range_.
#
# Wire format:
#   in_total : M * HIDDEN bf16 — activations (residual already added on host)
#   gamma    : HIDDEN     bf16 — shared across all M rows (broadcast)
#   beta     : HIDDEN     bf16 — shared
#   out_total: M * HIDDEN bf16
#
# γ and β stream once per launch via separate ObjectFifos and are held
# tile-resident across the M-row loop. This costs 1 extra shim MM2S
# vs packing γ+β into one slab, but keeps the kernel C++ ABI clean
# (no Python-side bf16 buffer slicing).
#
# Shim DMA budget (per launch):
#   MM2S: x_total (1) + gamma (1) + beta (1) = 3/2  ← OVER BUDGET
# Falls back to: pack γ/β into one combined fifo at host wire format
# level. See _gb_pack_split logic below.

from __future__ import annotations

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1


_M_DEFAULT = 47
_HIDDEN_DEFAULT = 256


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

    if HIDDEN % 32 != 0:
        raise ValueError(f"HIDDEN ({HIDDEN}) must be a multiple of 32 (bf16 lane)")

    # Element types. bf16 stored as uint16 byte storage.
    row_ty       = np.ndarray[(HIDDEN,),         np.dtype[np.uint16]]
    gb_ty        = np.ndarray[(2 * HIDDEN,),     np.dtype[np.uint16]]  # γ‖β
    in_total_ty  = np.ndarray[(M * HIDDEN,),     np.dtype[np.uint16]]
    out_total_ty = np.ndarray[(M * HIDDEN,),     np.dtype[np.uint16]]

    # Two shim MM2S streams (in_total, gb) + one shim S2MM (out_total)
    # — fits NPU2Col1's 2/2 MM2S budget.
    of_in  = ObjectFifo(row_ty, name="ln_in",  depth=2)
    of_gb  = ObjectFifo(gb_ty,  name="ln_gb",  depth=1)
    of_out = ObjectFifo(row_ty, name="ln_out", depth=2)

    # Kernel C++ ABI: bert_mini_layer_norm(in, gamma, beta, out) where
    # γ and β are HIDDEN-element bf16 buffers. We pass `gb_buf` (the
    # combined γ‖β slab) as the γ pointer; the kernel reads γ from
    # gb_buf[0..HIDDEN] and β from gb_buf[HIDDEN..2*HIDDEN] via
    # pointer offset arithmetic. Since IRON Buffers are flat byte
    # slabs at the kernel ABI level, this works without splitting in
    # Python. To keep the kernel signature clean we adapt the C++
    # entry point: `bert_mini_layer_norm_packed_gb(in, gb, out)`
    # where gb is the 2*HIDDEN slab.
    #
    # NOTE on kernel signature drift: the standalone byte-equal harness
    # (numpy reference + scalar fallback) uses the 4-arg form. The
    # silicon path uses the 3-arg packed form. Both share the same
    # math; only the indexing into γ/β differs.
    layer_norm_packed = Kernel(
        "bert_mini_layer_norm_packed",
        "bert_mini_block.o",
        [row_ty, gb_ty, row_ty],
    )

    def core_body(of_in, of_gb, of_out, layer_norm_k):
        # Acquire γ‖β once and hold across the M-row loop.
        gb = of_gb.acquire(1)
        for _ in range_(M):
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            layer_norm_k(x, gb, y)
            of_in.release(1)
            of_out.release(1)
        of_gb.release(1)

    worker = Worker(
        core_body,
        fn_args=[
            of_in.cons(),
            of_gb.cons(),
            of_out.prod(),
            layer_norm_packed,
        ],
    )

    rt = Runtime()
    with rt.sequence(in_total_ty, gb_ty, out_total_ty) as (
        x_total, gb, y_total,
    ):
        rt.start(worker)
        rt.fill(of_in.prod(), x_total)
        rt.fill(of_gb.prod(), gb)
        rt.drain(of_out.cons(), y_total, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--target", choices=["npu", "npu2"], default="npu2")
    p.add_argument("--M", type=int, default=_M_DEFAULT)
    p.add_argument("--HIDDEN", type=int, default=_HIDDEN_DEFAULT)
    args = p.parse_args()
    sys.stdout.write(str(emit_mlir(
        M=args.M, HIDDEN=args.HIDDEN, target=args.target,
    )))


if __name__ == "__main__":
    main()
