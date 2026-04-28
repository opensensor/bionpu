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

# kmer_count.py — IRON lowering for the canonical k-mer counting kernel.
#
# Per state/kmer_count_interface_contract.md (T1) — symbols, ObjectFifo
# names, constants, streaming chunk size + overlap, and burst-length
# env-var policy are all pinned by that contract. The IRON Python here
# is the consumer of those pins; the external Kernel(...) declarations
# below MUST agree byte-for-byte with the C++ symbols T5 emits in
# kmer_count_tile.cc.
#
# ---------------------------------------------------------------------------
# Topology revision (2026-04-28, T11-driven; codified in T1 contract
# "Topology revision (2026-04-28)" section).
# ---------------------------------------------------------------------------
#
# The original T8 wired N_TILES per-tile partial ObjectFifos directly into
# a single aggregator CoreTile worker. T11's silicon build at n_tiles=4
# failed with:
#
#     'aie.tile' op number of input DMA channel exceeded!
#
# AM020 Ch. 2 p. 27 documents the AIE2P CoreTile DMA budget as 2 input
# (S2MM) channels per tile. Direct fan-in of >2 partial FIFOs to one
# CoreTile is therefore unbuildable. The contract now mandates routing
# the partial fan-in through the **memtile** instead — AM020 Ch. 5 p. 74
# documents memtile DMA as 6 S2MM + 6 MM2S channels with native 5D
# address-generation concatenation (Figure 22).
#
# The memtile-aggregated combine pattern is silicon-validated in
#   bionpu-public/src/bionpu/kernels/crispr/match_multitile_memtile/
#       multitile_memtile.py        (the .prod().join() call, lines 195-199)
#       DESIGN.md                   (the 4-into-1 budget rationale)
#
# We mirror it exactly here:
#
#   shim ──seq_in── (broadcast) ──→ tile_0 .. tile_{N_TILES-1}
#   tile_i ──sparse_out (slot i, named "partial_count_<i>")── memtile
#                                                                  │
#                                                                  ▼
#                                                     memtile reorganises
#                                                     via 5D addr-gen
#                                                                  │
#                                                                  ▼
#                                                              shim
#
# The memtile slot for tile i is named "partial_count_<i>" so the
# pinned ObjectFifo names in T1 §"ObjectFifo names" remain stable from
# the producer's perspective. The aggregator CoreTile (T6's
# kmer_count_aggregator.cc) is bypassed for v1 — its source file remains
# on disk for a possible v1.1 (memtile-resident table partition) but it
# is NOT linked into the v1 xclbin. Host runner T7 performs ALL
# host-side dedup-by-canonical_u64 across n_tiles partial slots via its
# existing reaggregate_records() pass.
#
# ---------------------------------------------------------------------------
# Per-k chunk alignment fix (T11-driven, 2026-04-28).
# ---------------------------------------------------------------------------
#
# The aiecc lowering enforces a 4-byte alignment on `aie.dma_bd` payload
# sizes. The original SEQ_IN_OVERLAP_K21=5 yielded a chunk size of
# 4096+5=4101 (not 4-byte aligned) and aiecc rejected it. Bumped to 8
# (which still covers all 20 needed overlap bases for k=21 with slack);
# k=15 stays at 4 (4096+4=4100, 4-aligned); k=31 stays at 8 (4096+8=4104,
# 4-aligned). All three chunk sizes now satisfy the alignment constraint.
#
# ---------------------------------------------------------------------------
# Per-tile count-table sizing fix (T11-driven, 2026-04-28).
# ---------------------------------------------------------------------------
#
# The original HASH_BUCKETS_PER_TILE=4096 produced a 49152-byte partial
# ObjectFifo element. With depth=2 ping-pong that is 96 KiB on the
# producer tile — exceeds the 64 KiB AIE2P CoreTile DM cap. Reduced to
# 1024 buckets * 12 B = 12 KiB; depth=2 ping-pong is 24 KiB; total tile
# DM ~50 KiB fits with comfortable headroom. The collision rate is
# higher at this sizing but the emit-on-evict overflow policy + host
# re-aggregation absorb it (T17 measures the realised rate).
#
# ---------------------------------------------------------------------------
# Burst-length monkey-patch.
# ---------------------------------------------------------------------------
#
# Activate by setting BIONPU_KMER_COUNT_SHIM_BURST_LENGTH in the
# environment to one of {0, 64, 128, 256, 512} (NOT 1024 — see memory
# `cascade-burst-length-falsified-2026-04-28`). Default behaviour
# preserves the IRON default.
#
# CLI: ``python3 kmer_count.py -d npu2 --k 21 --launch-chunks 4`` emits
# MLIR to stdout. The Makefile (T10) drives it for all 12 (k, n_tiles)
# cells.
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/crispr/pam_filter/pam_filter.py (multi-tile fan-out
# scaffold) and bionpu/kernels/crispr/match_multitile_memtile/
# multitile_memtile.py (memtile-aggregated combine; the proven 4-into-1
# fan-in pattern this file mirrors).

import argparse
import os as _os
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1


# ---------------------------------------------------------------------------
# Burst-length override (per pam_filter.py:187-233 + AM020 falsification).
# ---------------------------------------------------------------------------
def _maybe_install_burst_length_override() -> int:
    """Install a DMATask.resolve override per BIONPU_KMER_COUNT_SHIM_BURST_LENGTH.

    Returns the burst length in effect (0 = IRON default).

    Allowed values: {0, 64, 128, 256, 512}. 1024 is intentionally
    excluded per the AM020 falsification finding — see memory
    `cascade-burst-length-falsified-2026-04-28`.
    """
    val = _os.environ.get("BIONPU_KMER_COUNT_SHIM_BURST_LENGTH")
    if val is None:
        return 0
    try:
        bl = int(val)
    except ValueError as exc:
        raise ValueError(
            f"BIONPU_KMER_COUNT_SHIM_BURST_LENGTH={val!r} is not an integer"
        ) from exc
    if bl not in (0, 64, 128, 256, 512):
        raise ValueError(
            f"BIONPU_KMER_COUNT_SHIM_BURST_LENGTH={bl} not in "
            "{0, 64, 128, 256, 512} (1024 excluded per AM020 falsification)"
        )
    if bl == 0:
        return 0

    from aie.dialects._aiex_ops_gen import dma_start_task  # type: ignore
    from aie.dialects.aiex import shim_dma_single_bd_task
    from aie.iron.runtime import dmatask as _dmatask_mod  # type: ignore

    def _resolve_with_burst_length(self, loc=None, ip=None):
        self._task = shim_dma_single_bd_task(
            self._object_fifo.op,
            self._rt_data.op,
            tap=self._tap,
            issue_token=self._wait,
            burst_length=bl,
        )
        dma_start_task(self._task)

    _dmatask_mod.DMATask.resolve = _resolve_with_burst_length
    print(
        f"[kmer_count.py] BIONPU_KMER_COUNT_SHIM_BURST_LENGTH={bl}: "
        f"DMATask.resolve patched to inject burst_length={bl} on all shim BDs",
        file=sys.stderr,
    )
    return bl


# ---------------------------------------------------------------------------
# Pinned constants (per state/kmer_count_interface_contract.md REVISED 2026-04-28).
# ---------------------------------------------------------------------------

LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_KMER_COUNT_LAUNCH_CHUNKS", "4"))
if LAUNCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_KMER_COUNT_LAUNCH_CHUNKS={LAUNCH_CHUNKS} not in {{1, 2, 4, 8}}"
    )

# N_TILES is the per-launch number of match tiles (one CoreTile per tile,
# all writing into per-slot memtile buffers via the .join() fan-in).
N_TILES = LAUNCH_CHUNKS

# Streaming chunk + overlap protocol (REVISED 2026-04-28 for 4-byte
# aiecc alignment — k=21 was 5, now 8).
SEQ_CHUNK_BYTES = 4096       # base chunk (matches T1 + T7's runner)
SEQ_OVERLAP_K15 = 4          # 4096+4 = 4100 (4-byte aligned)
SEQ_OVERLAP_K21 = 8          # 4096+8 = 4104 (4-byte aligned, was 5 — bumped)
SEQ_OVERLAP_K31 = 8          # 4096+8 = 4104 (4-byte aligned)

_OVERLAP_BY_K = {
    15: SEQ_OVERLAP_K15,
    21: SEQ_OVERLAP_K21,
    31: SEQ_OVERLAP_K31,
}

# Per-tile count-table geometry (REVISED 2026-04-28: 4096 -> 1024
# buckets, see top-of-file note + T1 §"Per-tile element types and shapes").
# 1024 buckets * 12 B/record = 12 KiB primary table; with depth=2 ping-pong
# on the partial ObjectFifo (24 KiB) and seq_in chunk (~8 KiB) total tile
# DM is ~50 KiB — fits the 64 KiB AIE2P CoreTile DM cap with headroom for
# stack and code.
HASH_BUCKETS_PER_TILE = 1024
COUNT_RECORD_BYTES = 12
PARTIAL_OUT_BYTES = HASH_BUCKETS_PER_TILE * COUNT_RECORD_BYTES   # 12288

# Sparse-emit ring slot constants — preserved for documentation /
# host-side parity but NO LONGER USED in the v1 dataflow. The memtile
# join writes raw partials of PARTIAL_OUT_BYTES per slot directly to
# the shim; T7 host runner performs the dedup-by-canonical_u64 pass.
EMIT_RECORD_BYTES = 16
EMIT_SLOT_RECORDS = 1024
EMIT_SLOT_BYTES = EMIT_RECORD_BYTES * EMIT_SLOT_RECORDS          # 16384

# How many input chunks one xclbin launch processes.
N_CHUNKS_PER_LAUNCH = LAUNCH_CHUNKS


# ---------------------------------------------------------------------------
# IRON Program builder.
# ---------------------------------------------------------------------------

def kmer_count(dev, *, k: int = 21):
    """Build the IRON program for the k-mer counting kernel.

    Args:
        dev: AIE device (NPU2 for AIE2P; NPU1Col1 for AIE2).
        k: One of {15, 21, 31}. Selects the per-tile external Kernel
            symbol ``kmer_count_tile_k{k}``.

    Topology (memtile-aggregated combine — see top-of-file rationale):

        shim ──seq_in── (broadcast) ──→ tile_0 .. tile_{N_TILES-1}
        tile_i ──partial_count_<i> (slot i of joined sparse_out)── memtile
                                                                       │
                                                                       ▼
                                                                  shim drain

    Mirrors `crispr/match_multitile_memtile/multitile_memtile.py`'s
    `outC.prod().join(of_offsets, obj_types=[partial_ty]*n_cores,
    names=...)` call exactly.
    """
    if k not in (15, 21, 31):
        raise ValueError(f"k must be in {{15, 21, 31}}, got {k!r}")

    overlap_bytes = _OVERLAP_BY_K[k]
    seq_in_chunk_bytes = SEQ_CHUNK_BYTES + overlap_bytes  # 4100 / 4104 / 4104
    if seq_in_chunk_bytes % 4 != 0:
        raise AssertionError(
            f"seq_in chunk size {seq_in_chunk_bytes} not 4-byte aligned "
            f"(k={k}, overlap={overlap_bytes}); aiecc dma_bd will reject"
        )

    # ----- whole-runtime tensor types (host-visible flat buffers) -----
    # Per-launch input buffer holds N_CHUNKS_PER_LAUNCH chunks, each of
    # (SEQ_CHUNK_BYTES + overlap) bytes.
    seq_in_size = N_CHUNKS_PER_LAUNCH * seq_in_chunk_bytes

    # Sparse output is the joined memtile buffer — N_TILES partials of
    # PARTIAL_OUT_BYTES each, contiguous. The host runner reads this as
    # `n_tiles` discrete count-table blobs and performs the dedup-merge
    # by canonical_u64 host-side.
    joined_partial_bytes = N_TILES * PARTIAL_OUT_BYTES               # n_tiles * 12288
    sparse_out_size = N_CHUNKS_PER_LAUNCH * joined_partial_bytes

    seq_in_ty = np.ndarray[(seq_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    # ----- per-tile element types -----
    seq_chunk_ty = np.ndarray[(seq_in_chunk_bytes,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[(PARTIAL_OUT_BYTES,), np.dtype[np.uint8]]   # 12288
    joined_chunk_ty = np.ndarray[(joined_partial_bytes,), np.dtype[np.uint8]]  # n_tiles*12288

    # ----- external kernel function (compiled C++) -----
    # Per-tile kernel: one symbol per supported k. T1 contract pins
    # the signature as
    #   (uint8* packed_in, uint8* partial_out, int32 n_input_bytes,
    #    int32 bucket_lo, int32 bucket_hi)
    tile_symbol = f"kmer_count_tile_k{k}"
    tile_fn = Kernel(
        tile_symbol,
        "kmer_count_tile.o",
        [
            seq_chunk_ty,                  # packed_in (chunk + overlap bytes)
            partial_chunk_ty,              # partial_out (12 KiB count table)
            np.int32,                      # n_input_bytes
            np.int32,                      # bucket_lo
            np.int32,                      # bucket_hi
        ],
    )

    # NOTE: kmer_count_aggregator is intentionally NOT declared as a
    # Kernel here — the v1 build bypasses the aggregator CoreTile in
    # favour of memtile-aggregated combine. The C++ source remains on
    # disk for a possible v1.1 (memtile-resident table partition).

    # ----- ObjectFifos -----
    # seq_in: shim → broadcast to all N_TILES match tiles. depth=2 so
    # shim DMA can stay in flight while the tiles consume the previous
    # chunk.
    of_seq_in = ObjectFifo(seq_chunk_ty, name="seq_in", depth=2)

    # sparse_out: memtile-aggregated joiner FIFO. Memtile is the fan-in
    # target (NOT a CoreTile aggregator). `.prod().join(...)` splits the
    # producer side into N_TILES per-tile slots, each named
    # `partial_count_<i>`; memtile reorganises via 5D address generation
    # into a single contiguous joined_chunk_ty, which is MM2S'd to shim.
    of_sparse = ObjectFifo(joined_chunk_ty, name="sparse_out", depth=2)

    # Per-tile join offsets in bytes: tile i's PARTIAL_OUT_BYTES partial
    # lands at byte offset i * PARTIAL_OUT_BYTES in the joined buffer.
    join_offsets = [i * PARTIAL_OUT_BYTES for i in range(N_TILES)]

    partial_fifos = of_sparse.prod().join(
        join_offsets,
        obj_types=[partial_chunk_ty] * N_TILES,
        names=[f"partial_count_{i}" for i in range(N_TILES)],
    )

    # ----- per-tile worker bodies (one per match tile) -----
    # Each tile owns the full HASH_BUCKETS_PER_TILE bucket range for
    # its share of input chunks. The bucket_lo / bucket_hi args are
    # there for a future memtile-resident table partition (gaps.yaml
    # T18); for v1 each tile uses [0, HASH_BUCKETS_PER_TILE) and the
    # host runner dedups by canonical_u64.
    def tile_body(of_seq, of_partial, kernel_fn):
        for _ in range_(N_CHUNKS_PER_LAUNCH):
            elem_seq = of_seq.acquire(1)
            elem_partial = of_partial.acquire(1)
            kernel_fn(
                elem_seq,
                elem_partial,
                seq_in_chunk_bytes,         # n_input_bytes
                0,                          # bucket_lo
                HASH_BUCKETS_PER_TILE,      # bucket_hi
            )
            of_seq.release(1)
            of_partial.release(1)

    tile_workers = [
        Worker(
            tile_body,
            fn_args=[
                of_seq_in.cons(),
                partial_fifos[i].prod(),
                tile_fn,
            ],
        )
        for i in range(N_TILES)
    ]

    # ----- runtime sequence (shim DMA in/out) -----
    # Memtile MM2S → shim drains the joined buffer directly, mirroring
    # `multitile_memtile.py:236` (`rt.drain(of_out.cons(), Out, wait=True)`).
    rt = Runtime()
    with rt.sequence(seq_in_ty, sparse_out_ty) as (S, Out):  # noqa: N806
        rt.start(*tile_workers)
        rt.fill(of_seq_in.prod(), S)
        rt.drain(of_sparse.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    # Declared at top of function so any read of these names below
    # (e.g. in argparse default-help formatting) is unambiguous and
    # so the post-parse rebinding is legal Python.
    global LAUNCH_CHUNKS, N_TILES, N_CHUNKS_PER_LAUNCH

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--dev",
        required=True,
        dest="device",
        choices=("npu", "npu2"),
        help="AIE device target (npu = AIE2/NPU1Col1; npu2 = AIE2P/NPU2)",
    )
    p.add_argument(
        "--k",
        required=True,
        type=int,
        choices=(15, 21, 31),
        help="k-mer length. Selects the per-tile external Kernel symbol.",
    )
    p.add_argument(
        "--launch-chunks",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
        help=(
            "Multi-tile fan-out (n_tiles). Defaults to "
            "BIONPU_KMER_COUNT_LAUNCH_CHUNKS env var (= "
            f"{LAUNCH_CHUNKS}). Valid in {{1, 2, 4, 8}}."
        ),
    )
    opts = p.parse_args(sys.argv[1:])

    # CLI flag overrides the env-var-driven module-level default.
    if opts.launch_chunks is not None:
        LAUNCH_CHUNKS = opts.launch_chunks
        N_TILES = LAUNCH_CHUNKS
        N_CHUNKS_PER_LAUNCH = LAUNCH_CHUNKS

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    # Optional shim-DMA burst-length override.
    _maybe_install_burst_length_override()

    print(kmer_count(dev, k=opts.k))


if __name__ == "__main__":
    main()
