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

# multitile.py — IRON lowering for the multi-tile streaming CRISPR mismatch
# kernel.
#
# Per CRISPR PRD §4.2 + plan ("the hard part"): take 's single-tile
# correctness foundation and scale it across multiple AIE2P tiles using a real
# DMA-driven dataflow. This is the C-M4 architecture milestone; layers
# the on-tile PAM filter on top, measures full GRCh38 throughput.
#
# # Working architecture (PRD §4.2 mapping)
#
# Tile A (window dispatcher / DMA in):
#   - Streams packed 2-bit windows from DDR via shim DMA.
# - For v1, Tile A is implicit: the windows ObjectFifo is broadcast
#     directly to all match tiles; the IRON / aie-rt placer fans the FIFO out
#     across compute tiles via shared mem-tile or DMA. Tile A emerges as the
#     placement pattern, not an explicit Worker.
#
# Tile B (guide-batch resident):
#   - 128 guides × 5 bytes = 640 B held resident on the match tiles. Broadcast
#     once per launch; lives across all chunk iterations.
#
# Tiles C..F (4 match tiles, parallel across guide-batch):
#   - Each tile owns a 32-guide slice of the 128-guide batch (128 / 4 = 32).
#   - Per chunk: read window slot, compute 32-guide × WINDOWS_PER_CHUNK
#     mismatch matrix, push partial output to Tile Z.
#
# Tile Z (output joiner / threshold):
#   - Collects 4 partial outputs (each 32 × WINDOWS_PER_CHUNK), concatenates
#     along the guide axis to produce the full 128 × WINDOWS_PER_CHUNK row,
#     emits to host via shim DMA out.
# - For v1, tile-side thresholding is staged through the dense-output
# join + a host-side sparse-extraction pass. will move the threshold
#     check (and PAM check) into Tile Z so only sparse hit records leave the
# NPU. This split keeps byte-equal to by construction (both
#     produce the same dense matrix), and isolates the dataflow correctness
#     gate from the threshold-emit-pipeline gate.
#
# # Tile budget math (the 64 KiB DM cap is the real constraint; PRD risk #3)
#
# Per match tile (Tiles C..F):
#   - guides slice (resident):              32 * 5         =     160 B
#   - windows chunk (1 slot, dbl-buffered): 2 * 64 * 5     =     640 B
#   - partial output (1 slot, dbl-buffered):2 * 64 * 32    =    4096 B
#   - subtotal:                                                 ~4.9 KiB
# Comfortably under 64 KiB (and under 16 KiB program memory once kernel + AIE
# control are linked).
#
# Tile Z (joiner):
#   - 4 input partials (1 slot each, dbl-buffered): 4 * 2 * 64 * 32 = 16384 B
#   - 1 output slot (dbl-buffered):                 2 * 64 * 128    = 16384 B
#   - subtotal:                                                      ~32 KiB
# Within budget.
#
# # Slide-by-1 windowing (PRD §4.2 — the dataflow novelty)
#
# Slide-by-1: every input nucleotide advance produces one new candidate window.
# v1 keeps host-side window enumeration (mirrors ): the host packs all
# overlapping 20-mer windows once and streams them through; the kernel does not
# slide — it consumes pre-cut windows. This is the "naive" slide-by-1
# implementation. 's PAM filter on Tile A is where the on-tile sliding
# window registers will land; for the C-M4 dataflow milestone we pin window
# enumeration on the host so byte-equality vs is direct.
#
# # Why mirror 's chunked launch model
#
# runs 64 chunks of 64 windows each through one tile. keeps the same
# chunk geometry (64 windows × 5 B) so the per-chunk DMA pattern is reusable;
# what changes is the fan-out across 4 match tiles + the join.
#
# # Naming convention for ObjectFifos
#
# The IRON pass picks tile placement from the ObjectFifo connectivity graph;
# we name FIFOs to expose the dataflow topology in the generated MLIR:
#   guides       : shim → all match tiles (broadcast, depth=1, fired once)
#   windows      : shim → all match tiles (broadcast, depth=2, fired N_CHUNKS)
#   partial_<i>  : match_i → joiner (depth=2, fired N_CHUNKS)
#   out          : joiner → shim (depth=2, fired N_CHUNKS)
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/crispr/match_singletile/match_singletile.py and
# mlir-aie/programming_examples/vision/edge_detect/edge_detect.py (multi-tile
# Worker pattern with ObjectFifo broadcast).

import argparse
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1

# Pinned shape — kept identical to so byte-equality is direct.
N_GUIDES = 128
SPACER_BYTES = 5  # 20 nt × 2 bits / 8 = 5 bytes
N_WINDOWS = 4096

# Multi-tile fan-out: 2 match workers, each handles GUIDES_PER_TILE guides.
#
# Why 2 (not 4 — the original PRD §4.2 sketch): AIE2P compute tiles have
# only 2 input + 2 output DMA channels each (verified empirically — see
# `` ). A 4-match-tile + 1-joiner topology asks the
# joiner to consume 4 input DMA streams, blowing the budget. Two match
# tiles + one join tile fits cleanly: joiner gets 2 input partials + 1
# guides broadcast (3 input flows; the `guides` ObjectFifo is a *broadcast*
# from shim, which IRON expands as memtile-buffered + per-tile-DMA-fanout
# rather than a per-tile shim DMA — so the input-DMA cost on the joiner is
# 2 (the partials), not 3).
#
# This still exercises the full multi-tile dataflow (broadcast guides +
# fan-out partials + join), and remains the architecturally-correct
# starting point for to layer on a memtile-aggregated fan-in if 4
# match tiles becomes profile-justified.
N_MATCH_TILES = 2
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 64

# Per-chunk geometry (matches so the host-side window stream is reusable).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64 chunks

def crispr_match_multitile(dev):
    """Build the IRON program for the multi-tile streaming CRISPR mismatch
    kernel.

    Topology:
        shim ──guides── (broadcast to all 4 match tiles, fired once)
        shim ──windows── (broadcast to all 4 match tiles, fired N_CHUNKS)
        match_0 ──partial_0── joiner
        match_1 ──partial_1── joiner
        match_2 ──partial_2── joiner
        match_3 ──partial_3── joiner
        joiner ──out── shim

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
    # (guides_per_tile bytes-on-tile = GUIDES_PER_TILE*SPACER_BYTES = 160 B
    # — kept implicit since each match tile receives the full 128-guide
    # batch via the broadcast guides ObjectFifo and slices in software via
    # the `guide_offset` parameter.)
    chunk_windows_size = WINDOWS_PER_CHUNK * SPACER_BYTES   # 320
    partial_chunk_size = WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 4096 (2 tiles × 64)
    full_chunk_size = WINDOWS_PER_CHUNK * N_GUIDES          # 8192

    # All match tiles see the full 128-guide batch on input — they each pick
    # their slice based on a `guide_offset` parameter passed at call time.
    # This avoids needing to slice the DDR-side guides buffer per tile (the
    # split would otherwise require per-tile shim DMA programming, which is a
    # performance optimisation we explicitly defer past v1).
    guides_full_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_chunk_ty = np.ndarray[(chunk_windows_size,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[(partial_chunk_size,), np.dtype[np.uint8]]
    out_chunk_ty = np.ndarray[(full_chunk_size,), np.dtype[np.uint8]]

    # ----- external kernel functions (compiled C++) -----
    # All match tiles share the same kernel symbol; it takes a `guide_offset`
    # int32 argument so each tile slices its 32-guide window of the 128-batch.
    match_fn = Kernel(
        "crispr_match_multitile_match",
        "match_kernel.o",
        [
            guides_full_ty,           # full 128-guide batch (resident)
            windows_chunk_ty,         # current 64-window chunk
            partial_chunk_ty,         # output: 64 windows × 32 guides
            np.int32,                 # n_windows in this chunk
            np.int32,                 # guide_offset (0, 32, 64, 96)
        ],
    )
    join_fn = Kernel(
        "crispr_match_multitile_join",
        "match_kernel.o",
        [
            partial_chunk_ty,         # partial from match_0 (GUIDES_PER_TILE cols)
            partial_chunk_ty,         # partial from match_1
            out_chunk_ty,             # joined 64 windows × 128 guides
            np.int32,                 # n_windows
        ],
    )

    # ----- ObjectFifos -----
    # Guides: one producer (shim), 4+1 consumers (4 match tiles + joiner).
    # The joiner doesn't need guides, but creating it as 4 cons-only handles
    # is the cleanest way to drive 4 broadcasts in IRON's current API.
    of_guides = ObjectFifo(guides_full_ty, name="guides", depth=1)
    # Windows: one producer (shim), 4 consumers (one per match tile).
    of_windows = ObjectFifo(windows_chunk_ty, name="windows", depth=2)
    # Partials: 4 separate FIFOs (one per match tile → joiner).
    of_partials = [
        ObjectFifo(partial_chunk_ty, name=f"partial_{i}", depth=2)
        for i in range(N_MATCH_TILES)
    ]
    # Joined output → shim.
    of_out = ObjectFifo(out_chunk_ty, name="out", depth=2)

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
                of_partials[i].prod(),
                match_fn,
                int(i * GUIDES_PER_TILE),
            ],
        )
        for i in range(N_MATCH_TILES)
    ]

    # ----- joiner worker body -----
    def join_body(of_p0, of_p1, of_o, join_kernel):
        for _ in range_(N_CHUNKS):
            e0 = of_p0.acquire(1)
            e1 = of_p1.acquire(1)
            eo = of_o.acquire(1)
            join_kernel(e0, e1, eo, WINDOWS_PER_CHUNK)
            of_p0.release(1)
            of_p1.release(1)
            of_o.release(1)

    join_worker = Worker(
        join_body,
        fn_args=[
            of_partials[0].cons(),
            of_partials[1].cons(),
            of_out.prod(),
            join_fn,
        ],
    )

    # ----- runtime sequence (shim DMA in/out) -----
    rt = Runtime()
    with rt.sequence(guides_ty, windows_ty, out_ty) as (G, W, Out):  # noqa: N806
        rt.start(*match_workers, join_worker)
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

    print(crispr_match_multitile(dev))

if __name__ == "__main__":
    main()
