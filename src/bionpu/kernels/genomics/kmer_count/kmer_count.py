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

# kmer_count.py — IRON lowering for the v0.5 streaming + multi-pass
# k-mer counting kernel.
#
# Per state/kmer_count_interface_contract.md (T1) v0.5 — symbols, ObjectFifo
# names, constants, and the streaming + multi-pass partition are pinned by
# the contract's "v0.5 REDESIGN" section.
#
# ---------------------------------------------------------------------------
# v0.5 design (supersedes the in-tile-counting design).
# ---------------------------------------------------------------------------
#
#   shim ──seq_in── (broadcast) ──→ tile_0 .. tile_{N_TILES-1}
#   tile_i ──partial_count_<i>── memtile (.join, N_TILES producers) ── shim
#
# The kernel is a streaming-emit only (no on-tile counting). For each
# chunk, each tile worker:
#   1. acquire seq_in elem (1 chunk + overlap bytes)
#   2. acquire partial_count_<i> elem (32 KiB pass-slot)
#   3. call kmer_count_tile_k{K}(elem_seq, elem_partial,
#                                n_input_bytes,
#                                pass_idx, n_passes_log2)
#      where pass_idx and n_passes_log2 are BAKED IN AT IRON-EMIT TIME
#      (compile-time arith.constants).
#   4. release both elems.
#
# The host runner dispatches N_PASSES separate xclbins per-chunk, one
# per pass_idx. Each xclbin has its (pass_idx, n_passes_log2) hardcoded
# at IRON-emit time. Per-chunk output blob = N_TILES × 32 KiB.
#
# Per-pass output blob shape: N_TILES × PARTIAL_OUT_BYTES_V05_PADDED =
# 4 × 32768 = 128 KiB at default n_tiles=4. The host parses N_TILES
# slots, accumulating canonical → count++ in a std::unordered_map<uint64_t,
# uint64_t> across ALL N_PASSES dispatches.
#
# ---------------------------------------------------------------------------
# Why pass_idx is BUILD-TIME (and host loops xclbins).
# ---------------------------------------------------------------------------
#
# IRON Python kernel-arg ABI accepts (numpy ndarray, numpy dtype, scalar)
# — where "scalar" means MLIR-time arith.constant. Loop indices from
# range_(N) lower to scf.for induction variables and CAN be passed to
# kernels via arith.index_cast(i32, idx), but a per-pass IRON loop with
# N_PASSES × N_TILES sub-fifos blew the memtile's 6 S2MM channel budget
# (16 producers > 6). Time-multiplexing N_PASSES through ONE per-tile
# fifo is possible but requires runtime-variable pass_idx INTO the kernel,
# which IRON doesn't support at the per-tile call site.
#
# The pragmatic resolution: **bake pass_idx + n_passes_log2 at IRON-emit
# time** and have the host runner switch xclbins per-pass. One xclbin
# per (k, n_tiles, pass_idx, n_passes), N_PASSES xclbins per
# (k, n_tiles, n_passes) artifact directory.
#
# ---------------------------------------------------------------------------
# Burst-length monkey-patch.
# ---------------------------------------------------------------------------
#
# Activate by setting BIONPU_KMER_COUNT_SHIM_BURST_LENGTH ∈ {0, 64, 128,
# 256, 512} (NOT 1024 — see memory `cascade-burst-length-falsified-2026
# -04-28`).
#
# CLI: ``python3 kmer_count.py -d npu2 --k 21 --launch-chunks 4
#       --n-passes 4 --pass-idx 0`` emits MLIR for one (pass_idx, n_passes)
# tuple to stdout. The Makefile drives the per-pass loop.
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/crispr/pam_filter/pam_filter.py (multi-tile fan-out
# scaffold) and bionpu/kernels/crispr/match_multitile_memtile/
# multitile_memtile.py (memtile-aggregated combine).

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
# Pinned constants (per state/kmer_count_interface_contract.md v0.5).
# ---------------------------------------------------------------------------

LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_KMER_COUNT_LAUNCH_CHUNKS", "4"))
if LAUNCH_CHUNKS not in (1, 2, 4, 8):
    raise ValueError(
        f"BIONPU_KMER_COUNT_LAUNCH_CHUNKS={LAUNCH_CHUNKS} not in {{1, 2, 4, 8}}"
    )

# N_TILES is the per-launch number of stream-emit tiles.
N_TILES = LAUNCH_CHUNKS

# N_PASSES — hash-slice partition count. log2 ∈ {0, 2, 4} → values
# {1, 4, 16}. Build-time constant; one xclbin per (k, n_tiles, n_passes,
# pass_idx).
N_PASSES = int(_os.environ.get("BIONPU_KMER_COUNT_N_PASSES", "4"))
if N_PASSES not in (1, 4, 16):
    raise ValueError(
        f"BIONPU_KMER_COUNT_N_PASSES={N_PASSES} not in {{1, 4, 16}}"
    )

# PASS_IDX — which slice this xclbin handles. 0..N_PASSES-1.
PASS_IDX = int(_os.environ.get("BIONPU_KMER_COUNT_PASS_IDX", "0"))


def _n_passes_log2(n: int) -> int:
    return {1: 0, 4: 2, 16: 4}[n]


# Streaming chunk + overlap protocol (4-byte aligned for aiecc dma_bd).
SEQ_CHUNK_BYTES = 4096
SEQ_OVERLAP_K15 = 4   # 4096+4 = 4100, aligned
SEQ_OVERLAP_K21 = 8   # 4096+8 = 4104, aligned (need 5, rounded up)
SEQ_OVERLAP_K31 = 8   # 4096+8 = 4104, aligned

_OVERLAP_BY_K = {
    15: SEQ_OVERLAP_K15,
    21: SEQ_OVERLAP_K21,
    31: SEQ_OVERLAP_K31,
}

# v0.5 partial-out element size: 32 KiB per pass-slot (4 byte uint32
# emit_idx prefix + up to 4095 × 8 byte canonicals + zero-pad).
PARTIAL_OUT_BYTES_V05_PADDED = 32768
MAX_EMIT_IDX_V05 = 4095

# How many input chunks one xclbin launch processes.
# v0.5: 1 chunk per dispatch — host loops dispatches per-chunk. The
# v1 design tied N_CHUNKS_PER_LAUNCH to LAUNCH_CHUNKS but that caused
# the host runner to under-fill the seq_in BO when only 1 chunk was
# needed (e.g. smoke 10 Kbp = 1 chunk total) and the kernel blocked
# acquiring chunks 2-4. Setting N_CHUNKS_PER_LAUNCH=1 makes each
# dispatch self-consistent: 1 chunk in, 1 set of N_TILES partials out.
N_CHUNKS_PER_LAUNCH = 1


# ---------------------------------------------------------------------------
# IRON Program builder.
# ---------------------------------------------------------------------------

def kmer_count(dev, *, k: int = 21, n_passes: int = 4, pass_idx: int = 0):
    """Build the IRON program for the v0.5 k-mer streaming-emit kernel.

    Per state/kmer_count_interface_contract.md v0.5.

    Args:
        dev: AIE device.
        k: k-mer length (15, 21, or 31).
        n_passes: hash-slice partition count (1, 4, or 16). BUILD-TIME.
        pass_idx: which slice this xclbin handles (0..n_passes-1).
            BUILD-TIME — the host runner dispatches one xclbin per
            pass_idx.
    """
    if k not in (15, 21, 31):
        raise ValueError(f"k must be in {{15, 21, 31}}, got {k!r}")
    if n_passes not in (1, 4, 16):
        raise ValueError(f"n_passes must be in {{1, 4, 16}}, got {n_passes!r}")
    if not (0 <= pass_idx < n_passes):
        raise ValueError(
            f"pass_idx must be in [0, {n_passes}), got {pass_idx!r}"
        )

    n_passes_log2 = _n_passes_log2(n_passes)

    overlap_bytes = _OVERLAP_BY_K[k]
    seq_in_chunk_bytes = SEQ_CHUNK_BYTES + overlap_bytes
    if seq_in_chunk_bytes % 4 != 0:
        raise AssertionError(
            f"seq_in chunk size {seq_in_chunk_bytes} not 4-byte aligned"
        )

    # ----- whole-runtime tensor types (host-visible flat buffers) -----
    seq_in_size = N_CHUNKS_PER_LAUNCH * seq_in_chunk_bytes

    # Sparse output: per chunk, N_TILES × 32 KiB pass-slots laid out
    # contiguously by the memtile joiner.
    joined_partial_bytes = N_TILES * PARTIAL_OUT_BYTES_V05_PADDED  # 128 KiB at n_tiles=4
    sparse_out_size = N_CHUNKS_PER_LAUNCH * joined_partial_bytes

    seq_in_ty = np.ndarray[(seq_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    # ----- per-tile element types -----
    seq_chunk_ty = np.ndarray[(seq_in_chunk_bytes,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[
        (PARTIAL_OUT_BYTES_V05_PADDED,), np.dtype[np.uint8]
    ]
    joined_chunk_ty = np.ndarray[
        (joined_partial_bytes,), np.dtype[np.uint8]
    ]

    # ----- external kernel function (compiled C++) -----
    # Per-tile v0.5 kernel: one symbol per supported k. T1 v0.5 contract pins
    # the signature as
    #   (uint8* packed_in, uint8* partial_out, int32 n_input_bytes,
    #    int32 pass_idx, int32 n_passes_log2)
    tile_symbol = f"kmer_count_tile_k{k}"
    tile_fn = Kernel(
        tile_symbol,
        "kmer_count_tile.o",
        [
            seq_chunk_ty,                  # packed_in (chunk + overlap bytes)
            partial_chunk_ty,              # partial_out (32 KiB pass-slot)
            np.int32,                      # n_input_bytes
            np.int32,                      # pass_idx
            np.int32,                      # n_passes_log2
            np.int32,                      # tile_idx
            np.int32,                      # n_tiles_log2
        ],
    )

    # ----- ObjectFifos -----
    # seq_in: shim → broadcast to all N_TILES tiles.
    of_seq_in = ObjectFifo(seq_chunk_ty, name="seq_in", depth=2)

    # sparse_out: memtile-aggregated joiner FIFO. One element per chunk
    # = N_TILES × 32 KiB.
    of_sparse = ObjectFifo(joined_chunk_ty, name="sparse_out", depth=2)

    # Per-tile join offset in the joined buffer.
    join_offsets = [i * PARTIAL_OUT_BYTES_V05_PADDED for i in range(N_TILES)]

    # depth=1 on the per-tile partial_count_<i> sub-FIFOs: at depth=2
    # the per-tile DM holds 2 × 32 KiB = 64 KiB of partials alone, which
    # combined with the depth=2 seq_in (~8 KiB) overflows the 64 KiB
    # AIE2P CoreTile DM cap. depth=1 means the kernel must finish writing
    # before the next chunk's DMA submit; this serialises slightly but
    # is acceptable per the v0.5 contract budget analysis.
    partial_fifos = of_sparse.prod().join(
        join_offsets,
        obj_types=[partial_chunk_ty] * N_TILES,
        names=[f"partial_count_{i}" for i in range(N_TILES)],
        depths=[1] * N_TILES,
    )

    # n_tiles_log2 — # bits used to pick the tile slot in the canonical's
    # low bits (above pass_idx's bits). Required for two-level partition.
    if N_TILES not in (1, 2, 4, 8):
        raise ValueError(f"N_TILES must be in {{1,2,4,8}}, got {N_TILES}")
    n_tiles_log2 = {1: 0, 2: 1, 4: 2, 8: 3}[N_TILES]

    # ----- per-tile worker bodies -----
    # Each tile worker is parameterised by its compile-time tile_idx so
    # the per-tile partition kernel arg is a baked-in IRON arith.constant.
    def make_tile_body(tile_idx_const: int):
        def tile_body(of_seq, of_partial, kernel_fn):
            for _ in range_(N_CHUNKS_PER_LAUNCH):
                elem_seq = of_seq.acquire(1)
                elem_partial = of_partial.acquire(1)
                kernel_fn(
                    elem_seq,
                    elem_partial,
                    seq_in_chunk_bytes,         # n_input_bytes
                    pass_idx,                   # build-time int
                    n_passes_log2,              # build-time int
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

    # ----- runtime sequence (shim DMA in/out) -----
    rt = Runtime()
    with rt.sequence(seq_in_ty, sparse_out_ty) as (S, Out):  # noqa: N806
        rt.start(*tile_workers)
        rt.fill(of_seq_in.prod(), S)
        rt.drain(of_sparse.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()


def main():
    global LAUNCH_CHUNKS, N_TILES, N_PASSES, PASS_IDX

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
        choices=(15, 21, 31),
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
        choices=(1, 4, 16),
        default=None,
        help=f"Hash-slice partition count (default {N_PASSES}). BUILD-TIME.",
    )
    p.add_argument(
        "--pass-idx",
        type=int,
        default=None,
        help=f"Which slice this xclbin handles (0..n_passes-1; default {PASS_IDX}).",
    )
    opts = p.parse_args(sys.argv[1:])

    if opts.launch_chunks is not None:
        LAUNCH_CHUNKS = opts.launch_chunks
        N_TILES = LAUNCH_CHUNKS
        # N_CHUNKS_PER_LAUNCH stays at module-level default (=1) per
        # v0.5; host loops dispatches per chunk.
    if opts.n_passes is not None:
        N_PASSES = opts.n_passes
    if opts.pass_idx is not None:
        PASS_IDX = opts.pass_idx
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

    _maybe_install_burst_length_override()

    print(kmer_count(dev, k=opts.k, n_passes=N_PASSES, pass_idx=PASS_IDX))


if __name__ == "__main__":
    main()
