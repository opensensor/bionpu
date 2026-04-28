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
# primer_scan.py — IRON lowering for the v0 primer/adapter scan kernel.
#
# Topology mirrors minimizer v0 (broadcast → N_TILES tiles → memtile join
# → shim). The kernel scans each chunk of the query for exact matches
# of a primer (forward + reverse-complement); the primer canonical
# arrives in the chunk header so each dispatch can use a different
# primer at runtime without rebuilding the xclbin (Path B).
#
# Build-time parameters:
#   P                     — primer length (13, 20, or 25). Per-build
#                           constant (folds the per-P mask into the kernel).
#   LAUNCH_CHUNKS         — n_tiles fan-out (1, 2, 4, 8). Default 4.
#   N_CHUNKS_PER_LAUNCH   — chunks per dispatch (default 1).
#
# CLI:
#   python3 primer_scan.py -d npu2 --p 13 --launch-chunks 4
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/genomics/minimizer/minimizer.py.

import argparse
import os as _os
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1


# ---------------------------------------------------------------------------
# Pinned constants (matches primer_scan_constants.h byte-equal).
# ---------------------------------------------------------------------------

LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_PRIMER_SCAN_LAUNCH_CHUNKS", "4"))
if LAUNCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_PRIMER_SCAN_LAUNCH_CHUNKS={LAUNCH_CHUNKS} not in {{1, 2, 4, 8}}"
    )

N_TILES = LAUNCH_CHUNKS

# Default P. Override via CLI / env.
P_DEFAULT = int(_os.environ.get("BIONPU_PRIMER_SCAN_P", "13"))

# Streaming chunk + 8-byte overlap (covers up to P=33 bases).
SEQ_CHUNK_BYTES = 4096
SEQ_OVERLAP_BYTES = 8

# Per-tile partial-out element size: 32 KiB (4-byte prefix + 16-byte
# records + zero pad). 16 bytes/record × 2046 max = 32 740 fits in 32 768.
PARTIAL_OUT_BYTES_PADDED = 32768
RECORD_BYTES = 16
MAX_EMIT_IDX = 2046

# How many input chunks one xclbin launch processes.
N_CHUNKS_PER_LAUNCH = int(_os.environ.get(
    "BIONPU_PRIMER_SCAN_N_CHUNKS_PER_LAUNCH", "1"))
if N_CHUNKS_PER_LAUNCH not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_PRIMER_SCAN_N_CHUNKS_PER_LAUNCH={N_CHUNKS_PER_LAUNCH} "
        "not in {1, 2, 4, 8}"
    )

_VALID_P: tuple[int, ...] = (13, 20, 25)


# ---------------------------------------------------------------------------
# IRON Program builder.
# ---------------------------------------------------------------------------


def primer_scan(dev, *, p: int):
    """Build the IRON program for the v0 primer-scan kernel.

    Args:
        dev: AIE device.
        p: primer length (13, 20, or 25). Build-time constant.

    Topology:
        shim ─seq_in─ broadcast ──▶ tile_0 .. tile_{N_TILES-1}
                                          │
                                    (compute fwd + rc;
                                     compare to primer_fwd in header;
                                     emit on match.)
                                          │
                                    partial_primer_<i>
                                          │
                              memtile .join(N_TILES, ...)
                                          │
                                          ▼
                                      shim drain

    Note: every tile receives the SAME input chunk (broadcast), so
    every tile produces the SAME match output. The host runner reads
    tile_0's slot only and ignores duplicates from siblings (matches
    minimizer v0's broadcast topology and keeps the artifact dir naming
    consistent).
    """
    if p not in _VALID_P:
        raise ValueError(f"unsupported P={p}; pinned: {_VALID_P}")

    seq_in_chunk_bytes = SEQ_CHUNK_BYTES + SEQ_OVERLAP_BYTES  # 4104
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
    # 5-arg signature: (packed_in, partial_out, n_input_bytes,
    #                   tile_idx, n_tiles_log2)
    # Primer canonical pair lives in the chunk header (Path B).
    tile_symbol = f"primer_scan_tile_p{p}"
    tile_fn = Kernel(
        tile_symbol,
        "primer_scan_tile.o",
        [
            seq_chunk_ty,                  # packed_in (header + payload)
            partial_chunk_ty,              # partial_out (32 KiB slot)
            np.int32,                      # n_input_bytes
            np.int32,                      # tile_idx (unused for v0)
            np.int32,                      # n_tiles_log2 (unused for v0)
        ],
    )

    # ----- ObjectFifos -----
    of_seq_in = ObjectFifo(seq_chunk_ty, name="seq_in", depth=2)
    of_sparse = ObjectFifo(joined_chunk_ty, name="sparse_out", depth=2)

    join_offsets = [i * PARTIAL_OUT_BYTES_PADDED for i in range(N_TILES)]

    # depth=1 on the per-tile sub-FIFOs to stay within the 64 KiB
    # CoreTile DM cap (mirrors minimizer/kmer_count).
    partial_fifos = of_sparse.prod().join(
        join_offsets,
        obj_types=[partial_chunk_ty] * N_TILES,
        names=[f"partial_primer_{i}" for i in range(N_TILES)],
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
    global LAUNCH_CHUNKS, N_TILES, N_CHUNKS_PER_LAUNCH

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--dev",
        required=True,
        dest="device",
        choices=("npu", "npu2"),
    )
    p.add_argument(
        "--p",
        required=True,
        type=int,
        choices=_VALID_P,
    )
    p.add_argument(
        "--launch-chunks",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
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
    if opts.n_chunks_per_launch is not None:
        N_CHUNKS_PER_LAUNCH = opts.n_chunks_per_launch

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    print(primer_scan(dev, p=opts.p))


if __name__ == "__main__":
    main()
