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
# minimizer.py — IRON lowering for the (w, k) minimizer kernel.
#
# v1 adds multi-pass hash-slice partitioning (mirrors kmer_count v0.5):
# the kernel emits only minimizers whose canonical falls into the active
# slice (low n_passes_log2 bits == pass_idx). Each pass produces an
# xclbin with (pass_idx, n_passes_log2) baked in as IRON arith.constants.
# The host runner dispatches N_PASSES separate xclbins per chunk (one
# per slice) and unions the results.
#
# Topology mirrors v0 (broadcast → N_TILES tiles → memtile join → shim);
# only the kernel-arg list grew (5 → 7 args) to carry the multi-pass
# selectors.
#
# Build-time parameters (all baked into the xclbin via env vars):
#   K, W                  — minimizer params; per-build constants.
#   LAUNCH_CHUNKS         — n_tiles fan-out (1, 2, 4, 8). Default 4.
#   N_PASSES              — hash-slice partition count (1, 4, 8, 16).
#                           Default 1 (back-compat with v0).
#   PASS_IDX              — which slice this xclbin handles
#                           (0..N_PASSES-1).
#   N_CHUNKS_PER_LAUNCH   — chunks per dispatch (default 1).
#
# CLI:
#   python3 minimizer.py -d npu2 --k 15 --w 10 --launch-chunks 4 \
#       --n-passes 4 --pass-idx 0
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/genomics/kmer_count/kmer_count.py.

import argparse
import os as _os
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1


# ---------------------------------------------------------------------------
# Pinned constants (matches minimizer_constants.h byte-equal).
# ---------------------------------------------------------------------------

LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_MINIMIZER_LAUNCH_CHUNKS", "4"))
if LAUNCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_MINIMIZER_LAUNCH_CHUNKS={LAUNCH_CHUNKS} not in {{1, 2, 4, 8}}"
    )

N_TILES = LAUNCH_CHUNKS

# Default (k, w). Override via CLI / env.
K_DEFAULT = int(_os.environ.get("BIONPU_MINIMIZER_K", "15"))
W_DEFAULT = int(_os.environ.get("BIONPU_MINIMIZER_W", "10"))

# v1: multi-pass hash-slice partitioning. log2 ∈ {0, 2, 3, 4} → values
# {1, 4, 8, 16}. Build-time constant; one xclbin per (k, w, n_tiles,
# n_passes, pass_idx).
N_PASSES = int(_os.environ.get("BIONPU_MINIMIZER_N_PASSES", "1"))
if N_PASSES not in (1, 4, 8, 16):
    raise ValueError(
        f"BIONPU_MINIMIZER_N_PASSES={N_PASSES} not in {{1, 4, 8, 16}}"
    )

# PASS_IDX — which slice this xclbin handles. 0..N_PASSES-1.
PASS_IDX = int(_os.environ.get("BIONPU_MINIMIZER_PASS_IDX", "0"))


def _n_passes_log2(n: int) -> int:
    return {1: 0, 4: 2, 8: 3, 16: 4}[n]


# Streaming chunk + per-(k,w) overlap (4-byte aligned for aiecc dma_bd).
# v1: kept at 4096 bytes (16384 bases) — same as v0 — for throughput.
# A chunk=1024 v1 experiment fit the cap fully at chr22 but blew up
# e2e wall (15+ min). Cap-fire mitigation deferred to v2 (widen
# partial_out 32K→64K).
SEQ_CHUNK_BYTES = 4096

# Overlap covers the prior chunk's tail (w + k - 1) bases. Round to
# nearest 4-byte boundary; both pinned configs land on 8 bytes.
_OVERLAP_BY_KW: dict[tuple[int, int], int] = {
    (15, 10): 8,  # 24 bases → 6 bytes → round to 8 (aligned)
    (21, 11): 8,  # 31 bases → 8 bytes (aligned; 32 bases ok)
}

# Per-tile partial-out element size: 32 KiB (4-byte prefix + 16-byte
# records + zero pad). 16 bytes/record × 2046 max = 32_736 + 4 = 32_740
# fits in 32 768.
PARTIAL_OUT_BYTES_PADDED = 32768
RECORD_BYTES = 16
MAX_EMIT_IDX = 2046

# How many input chunks one xclbin launch processes. v0: 1 (per-chunk
# dispatch). Mirrors kmer_count.py:189-202 — short fixtures need
# self-consistent dispatches.
#
# v1 batched-dispatch: setting N_CHUNKS_PER_LAUNCH > 1 amortises silicon
# dispatch cost over N_BATCH chunks. The runner MUST pad short tail
# batches with zero-actual-bytes chunks or the kernel will block
# acquiring undelivered chunks.
N_CHUNKS_PER_LAUNCH = int(_os.environ.get(
    "BIONPU_MINIMIZER_N_CHUNKS_PER_LAUNCH", "1"))
if N_CHUNKS_PER_LAUNCH not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_MINIMIZER_N_CHUNKS_PER_LAUNCH={N_CHUNKS_PER_LAUNCH} "
        "not in {1, 2, 4, 8}"
    )


# ---------------------------------------------------------------------------
# IRON Program builder.
# ---------------------------------------------------------------------------


def minimizer(dev, *, k: int, w: int, n_passes: int = 1, pass_idx: int = 0):
    """Build the IRON program for the (w, k) minimizer kernel.

    Args:
        dev: AIE device.
        k: k-mer length.
        w: window length.
        n_passes: hash-slice partition count (1, 4, 8, 16). BUILD-TIME.
        pass_idx: which slice this xclbin handles (0..n_passes-1).
            BUILD-TIME — the host runner dispatches one xclbin per
            pass_idx and unions the results.

    Topology:
        shim ─seq_in─ broadcast ──▶ tile_0 .. tile_{N_TILES-1}
                                          │
                                    (compute canonical;
                                     run sliding-window min;
                                     hash-slice filter at emit time)
                                          │
                                    partial_minimizer_<i>
                                          │
                              memtile .join(N_TILES, ...)
                                          │
                                          ▼
                                      shim drain

    Note: every tile receives the SAME input chunk (broadcast), so
    every tile produces the SAME minimizer output. The host runner
    reads tile_0's slot only and ignores the others. v1 retains the v0
    broadcast topology; per-tile partition is a future enhancement.
    """
    if (k, w) not in _OVERLAP_BY_KW:
        raise ValueError(
            f"unsupported (k, w)=({k}, {w}); pinned: "
            f"{tuple(_OVERLAP_BY_KW.keys())}"
        )
    if n_passes not in (1, 4, 8, 16):
        raise ValueError(
            f"n_passes must be in {{1, 4, 8, 16}}, got {n_passes!r}"
        )
    if not (0 <= pass_idx < n_passes):
        raise ValueError(
            f"pass_idx must be in [0, {n_passes}), got {pass_idx!r}"
        )

    n_passes_log2 = _n_passes_log2(n_passes)

    overlap_bytes = _OVERLAP_BY_KW[(k, w)]
    seq_in_chunk_bytes = SEQ_CHUNK_BYTES + overlap_bytes
    if seq_in_chunk_bytes % 4 != 0:
        raise AssertionError(
            f"seq_in chunk size {seq_in_chunk_bytes} not 4-byte aligned"
        )

    # ----- whole-runtime tensor types (host-visible flat buffers) -----
    seq_in_size = N_CHUNKS_PER_LAUNCH * seq_in_chunk_bytes
    joined_partial_bytes = N_TILES * PARTIAL_OUT_BYTES_PADDED
    sparse_out_size = N_CHUNKS_PER_LAUNCH * joined_partial_bytes

    seq_in_ty = np.ndarray[(seq_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    seq_chunk_ty = np.ndarray[(seq_in_chunk_bytes,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[
        (PARTIAL_OUT_BYTES_PADDED,), np.dtype[np.uint8]
    ]
    joined_chunk_ty = np.ndarray[
        (joined_partial_bytes,), np.dtype[np.uint8]
    ]

    # ----- external kernel function -----
    # v1 7-arg signature: (packed_in, partial_out, n_input_bytes,
    #                      pass_idx, n_passes_log2, tile_idx, n_tiles_log2)
    tile_symbol = f"minimizer_tile_k{k}_w{w}"
    tile_fn = Kernel(
        tile_symbol,
        "minimizer_tile.o",
        [
            seq_chunk_ty,                  # packed_in
            partial_chunk_ty,              # partial_out
            np.int32,                      # n_input_bytes
            np.int32,                      # pass_idx (v1)
            np.int32,                      # n_passes_log2 (v1)
            np.int32,                      # tile_idx (unused in v1)
            np.int32,                      # n_tiles_log2 (unused in v1)
        ],
    )

    # ----- ObjectFifos -----
    of_seq_in = ObjectFifo(seq_chunk_ty, name="seq_in", depth=2)
    of_sparse = ObjectFifo(joined_chunk_ty, name="sparse_out", depth=2)

    join_offsets = [i * PARTIAL_OUT_BYTES_PADDED for i in range(N_TILES)]

    # depth=1 on the per-tile sub-FIFOs to stay within the 64 KiB
    # CoreTile DM cap (mirrors kmer_count's argument).
    partial_fifos = of_sparse.prod().join(
        join_offsets,
        obj_types=[partial_chunk_ty] * N_TILES,
        names=[f"partial_minimizer_{i}" for i in range(N_TILES)],
        depths=[1] * N_TILES,
    )

    if N_TILES not in (1, 2, 4, 8):
        raise ValueError(f"N_TILES must be in {{1,2,4,8}}, got {N_TILES}")
    n_tiles_log2 = {1: 0, 2: 1, 4: 2, 8: 3}[N_TILES]

    def make_tile_body(tile_idx_const: int):
        def tile_body(of_seq, of_partial, kernel_fn):
            for _ in range_(N_CHUNKS_PER_LAUNCH):
                elem_seq = of_seq.acquire(1)
                elem_partial = of_partial.acquire(1)
                kernel_fn(
                    elem_seq,
                    elem_partial,
                    seq_in_chunk_bytes,         # n_input_bytes
                    pass_idx,                   # build-time int (v1)
                    n_passes_log2,              # build-time int (v1)
                    tile_idx_const,             # build-time int
                    n_tiles_log2,               # build-time int
                )
                of_seq.release(1)
                of_partial.release(1)
        return tile_body

    tile_workers = [
        Worker(
            make_tile_body(i),
            fn_args=[
                of_seq_in.cons(),
                partial_fifos[i].prod(),
                tile_fn,
            ],
        )
        for i in range(N_TILES)
    ]

    rt = Runtime()
    with rt.sequence(seq_in_ty, sparse_out_ty) as (S, Out):  # noqa: N806
        rt.start(*tile_workers)
        rt.fill(of_seq_in.prod(), S)
        rt.drain(of_sparse.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    global LAUNCH_CHUNKS, N_TILES, N_PASSES, PASS_IDX, N_CHUNKS_PER_LAUNCH

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--dev",
        required=True,
        dest="device",
        choices=("npu", "npu2"),
    )
    p.add_argument(
        "--k",
        required=True,
        type=int,
        choices=(15, 21),
    )
    p.add_argument(
        "--w",
        required=True,
        type=int,
        choices=(10, 11),
    )
    p.add_argument(
        "--launch-chunks",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
    )
    p.add_argument(
        "--n-passes",
        type=int,
        choices=(1, 4, 8, 16),
        default=None,
        help=f"Hash-slice partition count (default {N_PASSES}). BUILD-TIME.",
    )
    p.add_argument(
        "--pass-idx",
        type=int,
        default=None,
        help=f"Which slice this xclbin handles (0..n_passes-1; default {PASS_IDX}).",
    )
    p.add_argument(
        "--n-chunks-per-launch",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
    )
    opts = p.parse_args(sys.argv[1:])

    if opts.launch_chunks is not None:
        LAUNCH_CHUNKS = opts.launch_chunks
        N_TILES = LAUNCH_CHUNKS
    if opts.n_passes is not None:
        N_PASSES = opts.n_passes
    if opts.pass_idx is not None:
        PASS_IDX = opts.pass_idx
    if opts.n_chunks_per_launch is not None:
        N_CHUNKS_PER_LAUNCH = opts.n_chunks_per_launch
    if not (0 <= PASS_IDX < N_PASSES):
        raise ValueError(
            f"--pass-idx={PASS_IDX} must be in [0, --n-passes={N_PASSES})"
        )

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    print(minimizer(
        dev,
        k=opts.k,
        w=opts.w,
        n_passes=N_PASSES,
        pass_idx=PASS_IDX,
    ))


if __name__ == "__main__":
    main()
