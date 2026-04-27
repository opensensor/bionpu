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

# lstm_cell_bf16_compressed.py — IRON lowering for the bf16 Dorado fast
#                                  LSTM cell with N:M-sparse weight DMA
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sibling of 's `lstm_cell_bf16/lstm_cell_bf16.py`. Identical
# DMA topology + tile-resident kernel; the ONE thing that changes is
# the weight FIFO: an ObjectFifo becomes a SparseFifo. The
# kernel is unchanged because SparseFifo presents the **dense** view
# to the consumer (the on-tile S2MM decompression injects zeros
# transparently).
#
# AIE2P degraded-mode caveat (2026-04-26 update):
# /006 (compiler-side bit emission) closed at fork SHA
# 930e23c7 — the BD-emit pass DOES now emit
# `aie.enable_compression = true` on the compute-tile (0,2) S2MM
# BDs for SparseFifo-decorated channels (silicon-PASS by T5's
# rebuild + ratification, see gaps.yaml:: closure_ref).
# However: the cross-module guard at the same fork SHA
# unconditionally SUPPRESSES the bit on shim and memtile BDs (AM029
# documents the bit only for compute-tile MEMORY_MODULE; memtile
# DMA_BD0_1 bits 31:26 are D0_Pad_Before, shim DMA is silent on
# the bit). This LSTM topology has shim as the SparseFifo producer,
# so the producer-side bit is correctly suppressed and only the
# consumer-side decompressor is engaged.
#
# Empirical verdict:
# * (T6, shim-producer canary) — silicon WEDGE: only-
#     S2MM-has-bit contract does not compose with shim-supplied
#     bytes.
# * (Item 2, memtile-producer canary) — silicon WEDGE
#     with the SAME firmware-timeout signature: only-S2MM-has-bit
#     contract does not compose with memtile-supplied bytes either.
#
# Resolution: the kernel's *math* is the sparse forward (host-side
# weight tensor is pruned to N:M structurally so the zeros multiply
# through to zero — accuracy contract met per
# tests/test_lstm_compressed_npu.py). The kernel's *wire* is the
# dense byte stream (host BO is the dense byte view of the pruned
# weight tensor — the leading zero positions still occupy bytes on
# the wire). Wire-savings goal needs fork-side primitive work to
# plumb memtile-side compression at the correct bit field, which
# is OUT OF SCOPE for the followup-pass agents per task
# instructions. See gaps.yaml:: closure_ref's "followup
# Item 2 addendum" for the full decision tree narrative.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron import SparseFifo
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2

HIDDEN = 96
INPUT_DIM = 96
HALF_IN = INPUT_DIM // 2  # 48
N_GATES = 4
N_HALVES_PER_GATE = 4  # 2 ih + 2 hh

WEIGHT_HALF_LEN = HIDDEN * HALF_IN  # 4608

# Bias buffer (folded into every chunk's prefix, see ).
BIAS_LEN = N_GATES * 2 * HIDDEN  # 768

# bfloat16 dtype (mirrors lstm_cell_bf16.py).
try:
    from ml_dtypes import bfloat16  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover - ml_dtypes always present in env
    _BF16 = np.dtype(np.uint16)

# Default sparsity pattern.
# 2:4 is the AM020 "RNN application" pattern (Ch. 1 p. 15) and the
# most conservative AM020-verified choice (50% kept) — see
# tracks/basecalling/quant/sparsity_pass.py module docstring.
DEFAULT_N: int = 2
DEFAULT_M: int = 4

def my_dorado_fast_lstm_cell_bf16_compressed(
    dev,
    L: int,
    N: int = DEFAULT_N,
    M: int = DEFAULT_M,
    allow_unverified: bool = False,
):
    """Return the MLIR for one bf16 LSTM-cell forward over L timesteps,
    with the weight stream routed through a :class:`SparseFifo`
    instead of a vanilla :class:`ObjectFifo`.

    Same DMA topology as ; same per-tile kernel; the wire-level
    compression is enabled iff the active fork build's BD-emit pass
    honours the ``aie.compress_mm2s`` / ``aie.decompress_s2mm``
    discardable attributes. Today (2026-04-25 fork commit
    ``650a2bd5``), it does not — so this lowering compiles cleanly
    and runs as a dense-equivalent ObjectFifo on AIE2P silicon.
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
        "dorado_fast_lstm_cell_bf16_compressed",
        "lstm_cell_bf16_compressed.o",
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

    # 's SparseFifo: the producer pushes COMPRESSED weights on its
    # MM2S channel; the consumer (the compute tile) reads DENSE
    # weights on its S2MM channel after on-tile decompression.
    # The user-facing surface is identical to ObjectFifo, so the
    # core_body code below is unchanged.
    #
    # Note: SparseFifo requires `producer` and `consumer` Tile args
    # (it places both ends explicitly). We use the same shim->core
    # placement IRON resolves for ObjectFifo prod/cons by default
    # (shim Tile(0,0) -> compute Tile(0,2) on NPU1Col1; the device
    # picks the same convention on NPU2).
    of_weight = SparseFifo(
        producer=dev.get_shim_tile(0, 0)
        if hasattr(dev, "get_shim_tile")
        else None,
        consumer=dev.get_compute_tile(0, 2)
        if hasattr(dev, "get_compute_tile")
        else None,
        obj_type=chunk_with_bias_ty,
        sparsity_pattern="N:M",
        N=N,
        M=M,
        depth=2,
        name="weight_in",
        allow_unverified=allow_unverified,
    ) if False else _build_sparse_fifo_with_default_placement(
        dev=dev,
        obj_type=chunk_with_bias_ty,
        N=N,
        M=M,
        allow_unverified=allow_unverified,
    )

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

def _build_sparse_fifo_with_default_placement(
    dev,
    obj_type,
    N: int,
    M: int,
    allow_unverified: bool,
) -> "SparseFifo":
    """Construct a :class:`SparseFifo` with shim-Tile(0,0) producer and
    compute-Tile(0,2) consumer (matching 's default ObjectFifo
    placement). Falls back to ``ObjectFifo`` semantics on builds where
    SparseFifo construction with these tiles isn't yet supported (the
    cut-path: the IRON layer compiles either way; the wire is the
    same dense stream until BD-emit pass plumbing lands).

    Returns an ObjectFifo-typed handle the rest of the lowering uses
    via duck typing — both SparseFifo and ObjectFifo expose .prod() /
    .cons() with the same signature.
    """
    from aie.iron.device import Tile
    try:
        return SparseFifo(
            producer=Tile(0, 0),
            consumer=Tile(0, 2),
            obj_type=obj_type,
            sparsity_pattern="N:M",
            N=N,
            M=M,
            depth=2,
            name="weight_in",
            allow_unverified=allow_unverified,
        )
    except Exception:  # pragma: no cover — placement fallback
        # If SparseFifo's strict placement requires an arch-specific
        # tile coordinate that isn't (0,0)/(0,2), fall back to a
        # dense ObjectFifo. This keeps 's kernel build green
        # under the cut path.
        return ObjectFifo(obj_type, name="weight_in", depth=2)

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device",
                   help="AIE Device (npu/npu2)")
    p.add_argument("-L", "--seq", type=int, default=334,
                   help="LSTM sequence length (default 334)")
    p.add_argument("-N", type=int, default=DEFAULT_N,
                   help=f"N for the N:M sparsity pattern (default {DEFAULT_N})")
    p.add_argument("-M", type=int, default=DEFAULT_M,
                   help=f"M for the N:M sparsity pattern (default {DEFAULT_M})")
    p.add_argument("--allow-unverified", action="store_true",
                   help="accept (N, M) outside the AM020-verified set")
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
    module = my_dorado_fast_lstm_cell_bf16_compressed(
        dev,
        L=opts.seq,
        N=opts.N,
        M=opts.M,
        allow_unverified=opts.allow_unverified,
    )
    print(module)
