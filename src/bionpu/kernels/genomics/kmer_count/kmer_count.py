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
# kmer_count_tile.cc and T6 emits in kmer_count_aggregator.cc.
#
# Topology (mirrors CRISPR pam_filter's multi-tile fan-out, with
# kmer-counting-specific deviations spelled out in the contract's
# "Deviations from CRISPR pattern" section):
#
#   shim ──seq_in── (broadcast) ──→ tile_0 .. tile_{N-1}
#                                       │
#                                       ▼
#                                   partial_count_<i>
#                                       │
#                                       ▼
#                                   aggregator
#                                       │
#                                       ▼
#                                   sparse_out ── shim
#
# Per-tile body invokes external Kernel ``kmer_count_tile_k{K}`` (one
# of three k values — 15, 21, 31). Aggregator body invokes the
# k-independent external Kernel ``kmer_count_aggregator``.
#
# Multi-tile fan-out is governed by ``BIONPU_KMER_COUNT_LAUNCH_CHUNKS``
# (default 4, valid in {1, 2, 4, 8}) — same shape as
# pam_filter.py:235-240 but with a different default (kmer's production
# fan-out is 4-tile vs CRISPR's 1-tile per the contract's deviation
# table).
#
# Burst-length monkey-patch ``_maybe_install_burst_length_override()``
# mirrors pam_filter.py:187-233. The allowed-values set is
# {0, 64, 128, 256, 512} — note the exclusion of 1024 per the AM020
# falsification finding (memory: cascade-burst-length-falsified-2026-04-28).
#
# CLI: ``python3 kmer_count.py -d npu2 --k 21 --launch-chunks 4`` emits
# MLIR to stdout. The Makefile (T10) drives it for all 12 (k, n_tiles)
# cells.
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/crispr/pam_filter/pam_filter.py — same multi-tile
# dataflow shape, with PAM-filter and match-tile machinery replaced
# by the per-tile k-mer counting + aggregator pair.

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
#
# Item 3's dispatch_overhead_bisector found AIE2P shim DMA exhibits a
# ~2.37 s/launch firmware-TDR penalty per shim BD whose payload is
# <=256 B, additive across BDs in one launch. AM020 documents AIE-ML
# shim DMA burst lengths {64, 256, 512, 1024} B. The IRON
# `aiex.shim_dma_single_bd_task` wrapper hardcodes burst_length=0
# (firmware interprets as "highest available" = 1024 B). The cliff
# aligns with the 256/512-byte burst-length boundary.
#
# Activate by setting ``BIONPU_KMER_COUNT_SHIM_BURST_LENGTH`` in the
# environment to one of {0, 64, 128, 256, 512} (NOT 1024 — see memory
# `cascade-burst-length-falsified-2026-04-28` for the falsification
# log). Default behaviour (no env var, or =0) preserves the IRON
# default of 1024 B.
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

    # Monkey-patch DMATask.resolve. Identical structure to
    # pam_filter.py:_maybe_install_burst_length_override — the original
    # IRON implementation calls shim_dma_single_bd_task(...) with a
    # hardcoded burst_length=0; we replace it with one that forwards
    # the env-var value.
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
# Pinned constants (per state/kmer_count_interface_contract.md).
# ---------------------------------------------------------------------------

# Multi-tile fan-out. Default 4 (kmer's production fan-out per the
# contract's deviation table — CRISPR defaults to 1 because its
# downstream consumer is host-side; kmer's hash-table-bound per-tile
# work parallelises naturally).
LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_KMER_COUNT_LAUNCH_CHUNKS", "4"))
if LAUNCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_KMER_COUNT_LAUNCH_CHUNKS={LAUNCH_CHUNKS} not in {{1, 2, 4, 8}}"
    )

# N_TILES is the per-launch number of match tiles (the per-tile body
# pinned by `kmer_count_tile_k{K}`). Mirrors pam_filter.py's N_MATCH_TILES
# but here it is driven by the LAUNCH_CHUNKS env var so the same source
# emits all four artifact variants ({n1, n2, n4, n8}).
N_TILES = LAUNCH_CHUNKS

# Streaming chunk + overlap protocol.
SEQ_CHUNK_BYTES = 4096       # base chunk (matches T1 + T7's runner)
SEQ_OVERLAP_K15 = 4          # ceil((15-1)/4)
SEQ_OVERLAP_K21 = 5          # ceil((21-1)/4)
SEQ_OVERLAP_K31 = 8          # ceil((31-1)/4)

_OVERLAP_BY_K = {
    15: SEQ_OVERLAP_K15,
    21: SEQ_OVERLAP_K21,
    31: SEQ_OVERLAP_K31,
}

# Per-tile count-table geometry (matches kmer_count_constants.h).
# 4096 buckets * 12 B/record = 48 KiB. Drained per-chunk into the
# partial_count_<i> ObjectFifo; the aggregator merges + length-prefixes
# into the sparse_out ring slot.
HASH_BUCKETS_PER_TILE = 4096
COUNT_RECORD_BYTES = 12
PARTIAL_OUT_BYTES = HASH_BUCKETS_PER_TILE * COUNT_RECORD_BYTES   # 49152

# Sparse-emit ring slot (Tile Z output). 1024 records * 16 B/record =
# 16384 B per slot. Mirrors CRISPR post-fix EMIT_SLOT_RECORDS=1024.
EMIT_RECORD_BYTES = 16
EMIT_SLOT_RECORDS = 1024
EMIT_SLOT_BYTES = EMIT_RECORD_BYTES * EMIT_SLOT_RECORDS          # 16384

# How many input chunks one xclbin launch processes. The host runner
# (T7) submits one launch per (n_chunks_in_input // LAUNCH_CHUNKS)
# stride. Per-chunk we read SEQ_CHUNK_BYTES + overlap bytes from
# seq_in.
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

    Topology:

        shim ──seq_in── (broadcast) ──→ tile_0 .. tile_{N_TILES-1}
                                            │
                                            ▼
                                        partial_count_<i>
                                            │
                                            ▼
                                        aggregator
                                            │
                                            ▼
                                        sparse_out ── shim
    """
    if k not in (15, 21, 31):
        raise ValueError(f"k must be in {{15, 21, 31}}, got {k!r}")

    overlap_bytes = _OVERLAP_BY_K[k]
    seq_in_chunk_bytes = SEQ_CHUNK_BYTES + overlap_bytes

    # ----- whole-runtime tensor types (host-visible flat buffers) -----
    # Per-launch input buffer holds N_CHUNKS_PER_LAUNCH chunks, each of
    # (SEQ_CHUNK_BYTES + overlap) bytes. The host runner (T7) prepares
    # the overlap copies between adjacent chunks before submitting the
    # BO so no inter-chunk fixup is required on-tile.
    seq_in_size = N_CHUNKS_PER_LAUNCH * seq_in_chunk_bytes

    # Sparse output is drained as one fixed-size ring slot per
    # (tile, chunk) pair. The aggregator emits one slot per chunk that
    # carries up to EMIT_SLOT_RECORDS records merged across all
    # N_TILES partials.
    sparse_out_size = N_CHUNKS_PER_LAUNCH * EMIT_SLOT_BYTES

    seq_in_ty = np.ndarray[(seq_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    # ----- per-tile element types -----
    seq_chunk_ty = np.ndarray[(seq_in_chunk_bytes,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[(PARTIAL_OUT_BYTES,), np.dtype[np.uint8]]
    sparse_chunk_ty = np.ndarray[(EMIT_SLOT_BYTES,), np.dtype[np.uint8]]

    # ----- external kernel functions (compiled C++) -----
    # Per-tile kernel: one symbol per supported k. T1 contract pins
    # the signature as
    #   (uint8* packed_in, uint8* partial_out, int32 n_input_bytes,
    #    int32 bucket_lo, int32 bucket_hi)
    # which matches the IRON Kernel(...) declaration here byte-for-byte.
    tile_symbol = f"kmer_count_tile_k{k}"
    tile_fn = Kernel(
        tile_symbol,
        "kmer_count_tile.o",
        [
            seq_chunk_ty,                  # packed_in (chunk + overlap bytes)
            partial_chunk_ty,              # partial_out (48 KiB count table)
            np.int32,                      # n_input_bytes
            np.int32,                      # bucket_lo
            np.int32,                      # bucket_hi
        ],
    )

    # Aggregator kernel — k-independent. T1 contract pins the
    # signature as up to MAX_TILES=8 partial inputs + sparse_out + n_tiles_active.
    # We declare exactly N_TILES partial-input arguments here (the
    # aggregator's static signature widens at compile time per
    # the `partial_chunk_ty * N_TILES` expansion). Unused inputs
    # in <8-tile builds are not declared at this layer; the C++ side
    # has the wider 8-input signature and zeros unused inputs at
    # host-dispatch time per the contract.
    agg_args = [partial_chunk_ty] * N_TILES + [sparse_chunk_ty, np.int32]
    agg_fn = Kernel(
        "kmer_count_aggregator",
        "kmer_count_aggregator.o",
        agg_args,
    )

    # ----- ObjectFifos -----
    # seq_in: shim → broadcast to all N_TILES match tiles. depth=2 so
    # shim DMA can stay in flight while the tiles consume the previous
    # chunk.
    of_seq_in = ObjectFifo(seq_chunk_ty, name="seq_in", depth=2)

    # partial_count_<i>: per-tile → aggregator. Names pinned by T1.
    of_partials = [
        ObjectFifo(partial_chunk_ty, name=f"partial_count_{i}", depth=2)
        for i in range(N_TILES)
    ]

    # sparse_out: aggregator → shim. depth=2 to keep the drain BD
    # in flight.
    of_sparse = ObjectFifo(sparse_chunk_ty, name="sparse_out", depth=2)

    # ----- per-tile worker bodies (one per match tile) -----
    # Each tile owns the full HASH_BUCKETS_PER_TILE bucket range for
    # its share of input chunks. The bucket_lo / bucket_hi args are
    # there for a future memtile-resident table partition (gaps.yaml
    # T18); for v1 each tile uses [0, HASH_BUCKETS_PER_TILE) and the
    # aggregator dedups by canonical_u64.
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
                of_partials[i].prod(),
                tile_fn,
            ],
        )
        for i in range(N_TILES)
    ]

    # ----- aggregator worker body -----
    # Drains one partial-table-per-tile per chunk, dedup-merges by
    # canonical_u64, and emits a length-prefixed sparse ring slot to
    # the shim. The C++ aggregator handles eviction-flag passthrough
    # and host-side re-aggregation per the T1 contract.
    def aggregator_body(*args):
        # args layout (matches agg_args declaration order):
        #   partial_0..partial_{N_TILES-1}, sparse_out, agg_kernel
        partials_in = args[:N_TILES]
        of_out = args[N_TILES]
        kernel_fn = args[N_TILES + 1]
        for _ in range_(N_CHUNKS_PER_LAUNCH):
            partial_elems = [p.acquire(1) for p in partials_in]
            elem_out = of_out.acquire(1)
            kernel_fn(*partial_elems, elem_out, N_TILES)
            for p in partials_in:
                p.release(1)
            of_out.release(1)

    aggregator_worker = Worker(
        aggregator_body,
        fn_args=(
            [of_partials[i].cons() for i in range(N_TILES)]
            + [of_sparse.prod(), agg_fn]
        ),
    )

    # ----- runtime sequence (shim DMA in/out) -----
    rt = Runtime()
    with rt.sequence(seq_in_ty, sparse_out_ty) as (S, Out):  # noqa: N806
        rt.start(*tile_workers, aggregator_worker)
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

    # CLI flag overrides the env-var-driven module-level default. We
    # rebind the module-level globals so the IRON builder picks up the
    # CLI value. (This mirrors pam_filter.py's pattern of reading the
    # env var at import time but allowing the builder to consume the
    # current value.)
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
