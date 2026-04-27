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

# multitile_memtile.py — IRON lowering for the memtile-aggregated 4-into-1
# multi-tile streaming CRISPR mismatch kernel.
#
# **AM020 cross-walk follow-up to .** The original PRD §4.2 sketch was
# 4 match tiles × 32 guides each, joined to a full (n_windows × 128 guides)
# matrix. 's ship reduced to 2 match tiles × 64 guides because compute
# tiles only have 2 input DMA channels and a 4-into-1 joiner blew the
# budget. AM020 Ch. 5 p. 74 documents the canonical AIE-ML
# fix: aggregate via the memtile, which has 6 MM2S + 6 S2MM channels and
# native concatenation address generation. This file implements that path.
#
# # Topology (per AM020 Figure 22 + memtile join pattern in
# # programming_examples/basic/vector_reduce_max/single_column_designs/
# # vector_reduce_max_memtile.py — `outC.prod().join(of_offsets, ...)`):
#
#   shim ──guides──→ broadcast to 4 match tiles
#   shim ──windows──→ broadcast to 4 match tiles
#   match_0 ──memC0── memtile (slot 0)
#   match_1 ──memC1── memtile (slot 1) -- merged via .join() with offsets
#   match_2 ──memC2── memtile (slot 2)
#   match_3 ──memC3── memtile (slot 3)
#                      memtile ──out──→ shim
#
# IRON's `outC.prod().join(of_offsets, obj_types=...)` lowers each per-tile
# producer FIFO into a memtile S2MM channel; memtile reorganises the 4
# partials into a single contiguous (n_windows × N_GUIDES) buffer using
# 5D address generation (AM020 Ch. 5 p. 71). No joiner compute tile —
# memtile MM2S streams the assembled buffer to shim. The 4-into-1 fan-in
# budget is fully available because the memtile is the fan-in target, not
# a compute tile.
#
# # Tile budget math (per AM020 + reference)
#
# Per match tile (Tiles match_0..match_3):
#   - guides full batch (resident):         N_GUIDES * SPACER_BYTES = 640 B
#   - windows chunk (1 slot, dbl-buffered): 2 * 64 * 5     =     640 B
#   - partial out (1 slot, dbl-buffered):   2 * 64 * 32    =    4096 B
#   - subtotal:                                                ~5.4 KiB
# Comfortably under 64 KiB and ~half the per-match-tile footprint.
#
# Memtile (joiner is now the memtile, not a compute tile):
#   - 4 partials × 32 windows × 32 guides (× 2 dbl-buf) = 16384 B
#   - Joined out (32 windows × 128 guides × 2 dbl-buf)  = 16384 B
#   - subtotal:                                          ~32 KiB
# Memtile cap is 512 KiB per AM020 — utilisation < 7%.
#
# # Why mirror 's chunk geometry
#
# Same WINDOWS_PER_CHUNK=64 as + so the host-side window stream
# is byte-equivalent. The architectural change is **only** the fan-in
# width (4 vs 2), not the per-launch DMA pattern.
#
# # Naming convention for ObjectFifos
#
# guides       : shim → all 4 match tiles (broadcast, depth=1, fired once)
# windows      : shim → all 4 match tiles (broadcast, depth=2, fired N_CHUNKS)
# outC         : memtile-aggregated joiner FIFO (4 producers, 1 memtile, 1 shim consumer)
# memC<i>      : per-match-tile producer slot of outC (created by .join())
# of_out       : memtile → shim DMA (depth=2)
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/crispr/match_multitile/multitile.py and
# mlir-aie/programming_examples/basic/vector_reduce_max/single_column_designs/
# vector_reduce_max_memtile.py (memtile .join() pattern).

import argparse
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1

# Pinned shape — kept identical to / so byte-equality is direct.
N_GUIDES = 128
SPACER_BYTES = 5  # 20 nt × 2 bits / 8 = 5 bytes
N_WINDOWS = 4096

# Multi-tile fan-out: 4 match workers, each handles GUIDES_PER_TILE guides.
# Recovers the original PRD §4.2 sketch via memtile-mediated fan-in.
N_MATCH_TILES = 4
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 32

# Per-chunk geometry (matches / so the host-side window stream is
# byte-equivalent).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64 chunks

def crispr_match_multitile_memtile(dev):
    """Build the IRON program for the memtile-aggregated 4-into-1 multi-tile
    streaming CRISPR mismatch kernel.

    Topology:
        shim ──guides── (broadcast to all 4 match tiles, fired once)
        shim ──windows── (broadcast to all 4 match tiles, fired N_CHUNKS)
        match_0 ──memC0── memtile (slot 0 of outC)
        match_1 ──memC1── memtile (slot 1 of outC)
        match_2 ──memC2── memtile (slot 2 of outC)
        match_3 ──memC3── memtile (slot 3 of outC)
                            memtile ──of_out──→ shim

    Returns:
        Program — resolved IRON program targeting `dev`.
    """
    # ----- whole-runtime tensor types (host-visible flat buffers) -----
    guides_size = N_GUIDES * SPACER_BYTES                   # 640
    windows_size = N_WINDOWS * SPACER_BYTES                 # 20480
    out_size = N_WINDOWS * N_GUIDES                         # 524288 — window-major

    guides_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_ty = np.ndarray[(windows_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]

    # ----- per-tile types -----
    chunk_windows_size = WINDOWS_PER_CHUNK * SPACER_BYTES   # 320
    partial_chunk_size = WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 64*32 = 2048
    full_chunk_size = WINDOWS_PER_CHUNK * N_GUIDES          # 64*128 = 8192

    # All match tiles see the full 128-guide batch on input — they each pick
    # their slice based on a `guide_offset` parameter passed at call time.
    # This avoids needing to slice the DDR-side guides buffer per tile (the
    # split would otherwise require per-tile shim DMA programming, which is
    # a performance optimisation we explicitly defer past v1).
    guides_full_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_chunk_ty = np.ndarray[(chunk_windows_size,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[(partial_chunk_size,), np.dtype[np.uint8]]
    out_chunk_ty = np.ndarray[(full_chunk_size,), np.dtype[np.uint8]]

    # ----- external kernel function (compiled C++) -----
    # Single shared kernel (no joiner kernel — the join is fabric-side via
    # memtile DMA aggregation, not a compute kernel).
    match_fn = Kernel(
        "crispr_match_memtile_match",
        "match_kernel.o",
        [
            guides_full_ty,           # full 128-guide batch (resident)
            windows_chunk_ty,         # current 64-window chunk
            partial_chunk_ty,         # output: 64 windows × 32 guides
            np.int32,                 # n_windows in this chunk
            np.int32,                 # guide_offset (0, 32, 64, 96)
        ],
    )

    # ----- ObjectFifos -----
    # Guides: one producer (shim), 4 consumers (one per match tile, broadcast).
    of_guides = ObjectFifo(guides_full_ty, name="guides", depth=1)
    # Windows: one producer (shim), 4 consumers (one per match tile, broadcast).
    of_windows = ObjectFifo(windows_chunk_ty, name="windows", depth=2)

    # **The architectural pivot** — the memtile-aggregated joiner FIFO.
    # `of_out` is a single memtile ObjectFifo whose producer side is split
    # into N_MATCH_TILES per-tile producer slots via `.join(offsets, ...)`.
    # Each slot is a `partial_chunk_ty` (32 guides × 64 windows = 2048
    # bytes, **guide-major**); the memtile flat-concatenates them into an
    # `out_chunk_ty` (128 guides × 64 windows = 8192 bytes) per chunk.
    #
    # **Guide-major layout** is critical for flat-concat to produce the
    # right output. With each tile writing
    # `partial_out[g_local * n_windows + w]`, flat-concat into
    # `[0, 2048, 4096, 6144]` puts tile 0's 32 guides at rows 0..31, tile
    # 1's at rows 32..63, etc — natural guide-major layout. The host
    # then concatenates per-chunk (128 × WINDOWS_PER_CHUNK) buffers along
    # the window axis to form the final (N_GUIDES × N_WINDOWS) matrix.
    # **No transpose** required ( does a window-major → guide-major
    # transpose; this path skips that).
    #
    # This is the canonical AIE-ML memtile aggregation pattern documented
    # by AM020 Ch. 5 (memtile DMA) and exemplified in
    # mlir-aie/programming_examples/basic/vector_reduce_max/
    # single_column_designs/vector_reduce_max_memtile.py.
    of_out = ObjectFifo(out_chunk_ty, name="out", depth=2)

    # Flat-concat offsets in bytes: tile i's 2048-byte partial lands at
    # byte offset i * 2048 in the 8192-byte joined buffer.
    join_offsets = [i * partial_chunk_size for i in range(N_MATCH_TILES)]

    out_fifos = of_out.prod().join(
        join_offsets,
        obj_types=[partial_chunk_ty] * N_MATCH_TILES,
        names=[f"memC{i}" for i in range(N_MATCH_TILES)],
    )

    # ----- match worker body (one per match tile) -----
    def match_body(of_g, of_w, of_p, match_kernel, guide_offset):
        # Resident guide batch: acquire once, hold for the run.
        elem_g = of_g.acquire(1)
        for _ in range_(N_CHUNKS):
            elem_w = of_w.acquire(1)
            elem_p = of_p.acquire(1)
            match_kernel(elem_g, elem_w, elem_p, WINDOWS_PER_CHUNK, guide_offset)
            of_w.release(1)
            of_p.release(1)
        of_g.release(1)

    match_workers = [
        Worker(
            match_body,
            fn_args=[
                of_guides.cons(),
                of_windows.cons(),
                out_fifos[i].prod(),
                match_fn,
                int(i * GUIDES_PER_TILE),
            ],
        )
        for i in range(N_MATCH_TILES)
    ]

    # ----- runtime sequence (shim DMA in/out) -----
    # Memtile MM2S → shim is the canonical end of the memtile-aggregated
    # path: `rt.drain(of_out.cons(), Out)` reads the joined buffer
    # straight from memtile to the host's out tensor.
    rt = Runtime()
    with rt.sequence(guides_ty, windows_ty, out_ty) as (G, W, Out):  # noqa: N806
        rt.start(*match_workers)
        rt.fill(of_guides.prod(), G)
        rt.fill(of_windows.prod(), W)
        rt.drain(of_out.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device", help="AIE device")
    opts = p.parse_args(sys.argv[1:])

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    print(crispr_match_multitile_memtile(dev))

if __name__ == "__main__":
    main()
