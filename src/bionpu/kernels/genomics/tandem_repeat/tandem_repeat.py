# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# tandem_repeat.py — IRON lowering for the v0 short tandem repeat kernel.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import os as _os

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2


LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_TANDEM_REPEAT_LAUNCH_CHUNKS", "4"))
if LAUNCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_TANDEM_REPEAT_LAUNCH_CHUNKS={LAUNCH_CHUNKS} "
        "not in {1, 2, 4, 8}"
    )
N_TILES = LAUNCH_CHUNKS

SEQ_CHUNK_BYTES = 4096
SEQ_OVERLAP_BYTES = 12
HEADER_BYTES = 8
PARTIAL_OUT_BYTES_PADDED = 32768
RECORD_BYTES = 16
MAX_EMIT_IDX = 2046

N_CHUNKS_PER_LAUNCH = int(_os.environ.get(
    "BIONPU_TANDEM_REPEAT_N_CHUNKS_PER_LAUNCH", "1"))
if N_CHUNKS_PER_LAUNCH not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_TANDEM_REPEAT_N_CHUNKS_PER_LAUNCH={N_CHUNKS_PER_LAUNCH} "
        "not in {1, 2, 4, 8}"
    )


def tandem_repeat(dev):
    """Build the IRON program for short tandem repeat (STR) detection."""
    seq_in_chunk_bytes = SEQ_CHUNK_BYTES + SEQ_OVERLAP_BYTES  # 4108
    if seq_in_chunk_bytes % 4 != 0:
        raise AssertionError(
            f"seq_in chunk size {seq_in_chunk_bytes} not 4-byte aligned"
        )

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

    tile_fn = Kernel(
        "tandem_repeat_tile",
        "tandem_repeat_tile.o",
        [
            seq_chunk_ty,
            partial_chunk_ty,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    of_seq_in = ObjectFifo(seq_chunk_ty, name="seq_in", depth=2)
    of_sparse = ObjectFifo(joined_chunk_ty, name="sparse_out", depth=2)
    join_offsets = [i * PARTIAL_OUT_BYTES_PADDED for i in range(N_TILES)]
    partial_fifos = of_sparse.prod().join(
        join_offsets,
        obj_types=[partial_chunk_ty] * N_TILES,
        names=[f"partial_tr_{i}" for i in range(N_TILES)],
        depths=[1] * N_TILES,
    )
    n_tiles_log2 = {1: 0, 2: 1, 4: 2, 8: 3}[N_TILES]

    def make_tile_body(tile_idx_const: int):
        def tile_body(of_seq, of_partial, kernel_fn):
            for _ in range_(N_CHUNKS_PER_LAUNCH):
                elem_seq = of_seq.acquire(1)
                elem_partial = of_partial.acquire(1)
                kernel_fn(
                    elem_seq,
                    elem_partial,
                    seq_in_chunk_bytes,
                    tile_idx_const,
                    n_tiles_log2,
                )
                of_seq.release(1)
                of_partial.release(1)
        return tile_body

    workers = [
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
        rt.start(*workers)
        rt.fill(of_seq_in.prod(), S)
        rt.drain(of_sparse.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    global LAUNCH_CHUNKS, N_TILES, N_CHUNKS_PER_LAUNCH

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, choices=("npu", "npu2"))
    p.add_argument(
        "--launch-chunks",
        type=int,
        default=LAUNCH_CHUNKS,
        choices=(1, 2, 4, 8),
    )
    p.add_argument(
        "--n-chunks-per-launch",
        type=int,
        default=N_CHUNKS_PER_LAUNCH,
        choices=(1, 2, 4, 8),
    )
    opts = p.parse_args()

    LAUNCH_CHUNKS = int(opts.launch_chunks)
    N_TILES = LAUNCH_CHUNKS
    N_CHUNKS_PER_LAUNCH = int(opts.n_chunks_per_launch)
    dev = NPU2() if opts.dev == "npu2" else NPU1Col1()
    print(tandem_repeat(dev))


if __name__ == "__main__":
    main()
