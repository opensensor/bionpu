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

# pam_filter.py — IRON lowering for the PAM-filter + threshold + emit
# CRISPR streaming kernel.
#
# Per CRISPR PRD §4.3 + plan : layer (a) on-tile NGG PAM check ahead of
# the match tiles ("filter-early"), and (b) on-tile threshold + sparse-emit
# at Tile Z, on top of 's multi-tile dataflow.
#
# Two variants ship from this same Python lowering, selected by ``mode``:
#
#   filter-early : Tile A reads pre-cut windows (each window carries its
#                  3-nt PAM context as 4 trailing 2-bit bases), checks the
#                  NGG template, and only forwards windows that pass to
#                  the match tiles. ~7/8 of windows are dropped here.
#                  This is the production path.
#
#   filter-late  : Tile A passes every window through unconditionally;
#                  Tile Z (threshold + emit) is the *only* place PAM is
#                  checked, after the match arithmetic has been computed
#                  on every window. Used as a controlled comparison
#                  (same byte-equal output, different work distribution).
#
# Both variants produce **byte-identical** sparse hit-list output (after
# canonical normalization). They differ in (1) work done by the match
# tiles and (2) DMA volume out of Tile Z. The ratio of (filter-late
# match-tile cycles) / (filter-early match-tile cycles) is the headline
# C-M5 number in `results/crispr/c-m5/measurements.json`.
#
# # Working architecture (PRD §4.3 mapping; layered on 's dataflow)
#
# Tile A (NEW for — PAM filter / dispatcher):
#   - Streams 23-byte (sic — see byte layout below) windows from shim.
#   - Checks the 3-nt PAM context against the literal positions of the
#     PAM template (for NGG, positions 1 and 2 must be G; position 0 is
#     `N` and matches anything).
#   - filter-early: only forwards passing windows to the match tiles.
#                   Drops the rest (7/8 of windows on random ACGT).
#   - filter-late : forwards every window unconditionally.
#
#   Strand handling: the host pre-builds two passes through Tile A, one
#   per strand. Tile A treats forward-strand and reverse-strand window
#   streams identically; the strand label is host-side bookkeeping
#   attached to each contiguous launch (DESIGN.md §3 documents this
#   choice — keeping Tile A strand-agnostic saves ~25% of its program
#   memory vs an in-tile RC step).
#
# Tile B (guide-batch resident):
# - Unchanged from — N_GUIDES × SPACER_BYTES = 640 B resident.
#
# Tiles C, D (match tiles, parallel across guide-batch):
# - Unchanged kernel from . Now operating on a smaller,
#     PAM-filtered window stream in the filter-early case.
#
# Tile Z (joiner + threshold + sparse-emit, EXTENDED from ):
#   - Reads partials from match tiles, joins to (n_windows × N_GUIDES).
#   - Applies `mismatch_count <= max_mm` filter on-tile.
#   - filter-late: ALSO re-checks PAM (using the per-window PAM byte
#                  forwarded via the windows ObjectFifo).
#   - Emits sparse records (window_idx, guide_idx, mismatches) directly
#     via shim DMA-out. The host mapping table assigns (chrom, position,
#     strand) from window_idx by lookup table — Tile Z stays
#     genome-agnostic.
#
# # Tile budget math (the 64 KiB DM cap is still the real constraint)
#
# Tile A (PAM filter): the smallest tile by far.
#   - chunk_in (1 slot, dbl-buffered): 2 × 64 × WINDOW_BYTES_WITH_PAM
#                                       = 2 × 64 × 6 = 768 B
#   - chunk_out (1 slot, dbl-buffered): 2 × 64 × SPACER_BYTES
#                                        = 2 × 64 × 5 = 640 B
#   - peak: ~1.5 KiB, well under cap.
#
# Per match tile (Tiles C, D — unchanged from ):
#   - guides slice (resident):              640 B
#   - windows chunk (dbl-buffered):         640 B
#   - partial output (dbl-buffered):       8192 B
# - subtotal: ~9.5 KiB (same as ).
#
# Tile Z (joiner + threshold + emit, EXTENDED):
#   - 2 input partials (1 slot each, dbl-buffered):  16384 B
#   - PAM byte stream (dbl-buffered, filter-late):     128 B
#   - sparse-emit ring buffer (dbl-buffered):         4096 B
#                                                   (256 records × 8 B)
#   - subtotal:                                       ~20 KiB (down from
# 's 32 KiB because we no longer hold the dense
#     window-major output buffer — sparse emit replaces it).
#
# So the peak tile memory after 's additions is ~20 KiB on Tile Z
# vs 's 32 KiB — the on-tile threshold + sparse-emit *reduces* the
# joiner's peak DM usage. 's PAM filter on Tile A adds ~1.5 KiB on a
# dedicated tile that didn't exist in , but that tile's budget is
# nearly free (it doesn't hold guides or partials).
#
# # Strand handling — DESIGN.md §3
#
# Two passes through Tile A: one with forward-strand windows, one with
# reverse-complement windows. The host builds the RC window stream
# byte-pre-flipped (cheap, deterministic, parallelizable). This keeps
# Tile A strand-agnostic and the same xclbin handles both passes
# without per-strand kernel symbols.
#
# Why not in-tile RC? Two reasons:
#   1. RC requires per-base 2-bit complement (bitwise NOT then byte-
#      reverse) which doubles Tile A's program memory budget without
#      reducing DMA volume.
#   2. The sparse-emit at Tile Z stamps strand from the host-side
#      window-index table; mixing strands inside one Tile A run would
#      require per-window strand bits and bloat the DMA payload.
#
# # Filter-early vs filter-late — what's actually different in the IRON
#
# Both variants use **the same xclbin** but are dispatched with different
# host-side runner programs. The only difference at the Python lowering
# level is whether Tile A's body does the PAM check and conditionally
# acquires the output ObjectFifo, or always forwards. Build both .xclbin
# variants from this single source by passing ``mode`` at lowering time;
# the C++ kernel object exposes both ``crispr_pam_filter_early`` and
# ``crispr_pam_filter_late`` symbols and aiecc picks the right one based
# on the IRON ``Kernel`` reference.
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# bionpu/kernels/crispr/match_multitile/multitile.py — same
# multi-tile dataflow with two new tiles (Tile A filter, Tile Z extended).

import argparse
import os as _os
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1

# (followup-A, 2026-04-26) — Item 3's BD-cliff hypothesis test.
#
# Item 3's dispatch_overhead_bisector found that
# AIE2P shim DMA exhibits a ~2.37 s/launch firmware-TDR penalty per
# shim BD whose payload is <= 256 B, additive across BDs in one launch.
# AM020 documents AIE-ML shim DMA burst lengths {64, 256, 512, 1024} B;
# the IRON `aiex.shim_dma_single_bd_task` wrapper (called from
# `Runtime.fill / Runtime.drain` via `DMATask.resolve`) hardcodes
# `burst_length=0` (which the firmware interprets as "highest available"
# = 1024 B). The cliff aligns with the 256/512-byte burst-length
# boundary, suggesting the firmware-default burst path stalls when
# payload < single burst length.
#
# The IRON-Python `Runtime.fill / Runtime.drain` does NOT expose a
# `burst_length` knob — the value is hardcoded inside
# `aie.iron.runtime.dmatask.DMATask.resolve`. This module monkey-patches
# `DMATask.resolve` at lowering time to inject a non-zero
# `burst_length` so we can test (a) whether explicit annotation changes
# behavior at all, and (b) whether matching burst_length to a smaller
# value than the firmware default lifts the cliff. Default behavior
# (no env var) is unchanged.
#
# Activate by setting `BIONPU_PAM_FILTER_SHIM_BURST_LENGTH` in the
# environment to one of {64, 128, 256, 512, 1024} before invoking
# `python3 pam_filter.py -d npu2 --mode early`.
#
# Empirical baseline this experiment is testing against (T9, 2026-04-25):
#   per-launch wall on vendored xclbin sha 7057f848: 6.14393 s avg
# Item 3's prediction: lifting any sub-cliff shim BD across 512 B
# should drop per-launch wall from ~6 s to <1 ms.
#
# Note (BD audit, state/followup-a/bd-audit-20260426T011703Z.json):
# The CRISPR PAM filter's three shim BDs are guides=640 B,
# windows_in=24576 B, sparse_out=131072 B — ALL above the 512 B
# cliff. Item 3's hypothesis as stated does not predict an improvement
# here, but the experiment is run as a falsification test.
def _maybe_install_burst_length_override() -> int:
    """Install a DMATask.resolve override per BIONPU_PAM_FILTER_SHIM_BURST_LENGTH.

    Returns the burst length in effect (0 = IRON default).
    """
    val = _os.environ.get("BIONPU_PAM_FILTER_SHIM_BURST_LENGTH")
    if val is None:
        return 0
    try:
        bl = int(val)
    except ValueError as exc:
        raise ValueError(
            f"BIONPU_PAM_FILTER_SHIM_BURST_LENGTH={val!r} is not an integer"
        ) from exc
    if bl not in (0, 64, 128, 256, 512, 1024):
        raise ValueError(
            f"BIONPU_PAM_FILTER_SHIM_BURST_LENGTH={bl} not in "
            "{0, 64, 128, 256, 512, 1024}"
        )
    if bl == 0:
        return 0

    # Monkey-patch DMATask.resolve. The original implementation (see
    # third_party/mlir-aie/python/iron/runtime/dmatask.py) calls
    # `shim_dma_single_bd_task(...)` with a hardcoded burst_length=0.
    # We replace it with one that forwards `bl`.
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
        f"[pam_filter.py] BIONPU_PAM_FILTER_SHIM_BURST_LENGTH={bl}: "
        f"DMATask.resolve patched to inject burst_length={bl} on all shim BDs",
        file=sys.stderr,
    )
    return bl

# Pinned public shape is 4096 windows. BIONPU_PAM_FILTER_LAUNCH_CHUNKS
# builds an opt-in wider-launch xclbin that processes multiple public chunks
# per NPU invocation while preserving the host-facing 4096-window API.
LAUNCH_CHUNKS = int(_os.environ.get("BIONPU_PAM_FILTER_LAUNCH_CHUNKS", "1"))
if LAUNCH_CHUNKS <= 0:
    raise ValueError("BIONPU_PAM_FILTER_LAUNCH_CHUNKS must be > 0")

# Tile A "input" carries 1 byte of PAM context per window in addition to the
# 5-byte spacer (6 bytes total per Tile-A input slot).
N_GUIDES = 128
SPACER_BYTES = 5  # 20 nt × 2 bits / 8 = 5 bytes
N_WINDOWS_BASE = 4096
N_WINDOWS = N_WINDOWS_BASE * LAUNCH_CHUNKS
PAM_LEN = 3  # NGG
PAM_BYTES = 1  # 3-nt PAM packs into 6 bits → 1 byte (with 2 bits padding).

# Per Tile A input record: 5 bytes spacer + 1 byte PAM context = 6 bytes.
WINDOW_BYTES_IN = SPACER_BYTES + PAM_BYTES  # 6

# Multi-tile fan-out (unchanged from — DMA-channel constraint
# still applies until memtile-aggregated fan-in lands).
N_MATCH_TILES = 2
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 64

# Per-chunk geometry (mirrors so the host-side window stream is reusable).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64 chunks

# Sparse-emit ring buffer geometry (Tile Z). Each emit record is 8 bytes:
#   uint16 window_idx | uint8 guide_idx | uint8 mismatches | uint32 reserved
# A ring buffer of 256 records × 8 B = 2048 B per slot; double-buffered
# = 4096 B on Tile Z. Per chunk we worst-case emit 64 × 128 = 8192
# records — far more than the ring slot — so the host drains via shim
# DMA into a larger host-side buffer. will measure ring-buffer
# pressure on full chr22 + 10 guides.
EMIT_RECORD_BYTES = 8
# fix: bumped 256 -> 1024 (worst chr22 sub-chunk has 508
# host hits; 256 silently dropped 1004 records). Stays in lockstep
# with tile_a_filter.cc + runner.cpp + __init__.py.
EMIT_SLOT_RECORDS = 1024
EMIT_SLOT_BYTES = EMIT_RECORD_BYTES * EMIT_SLOT_RECORDS

def crispr_pam_filter(dev, *, mode: str = "early"):
    """Build the IRON program for the PAM-filter + threshold + emit kernel.

    Args:
        dev: AIE device (NPU2 for AIE2P; NPU1Col1 for AIE2).
        mode: ``"early"`` or ``"late"``. Selects which Tile A symbol the
            xclbin pulls in. Output bytes are identical between the two
            modes (the difference is only the work distribution).

    Topology (filter-early shown — filter-late changes only Tile A's body):

        shim ──windows_in── Tile A (PAM filter)
                              ├──windows_filtered── (broadcast) ──→ match_0, match_1
                              └──pam_meta── Tile Z
        match_0 ──partial_0── Tile Z
        match_1 ──partial_1── Tile Z
        Tile Z ──sparse_out── shim
    """
    if mode not in {"early", "late"}:
        raise ValueError(f"mode must be 'early' or 'late', got {mode!r}")

    # ----- whole-runtime tensor types (host-visible flat buffers) -----
    guides_size = N_GUIDES * SPACER_BYTES                         # 640
    windows_in_size = N_WINDOWS * WINDOW_BYTES_IN                 # 24576
    # Sparse output is drained as one fixed-size Tile-Z slot per chunk.
    # Each slot has its own uint32 count prefix plus records/padding.
    sparse_out_size = N_CHUNKS * EMIT_SLOT_BYTES

    guides_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_in_ty = np.ndarray[(windows_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    # ----- per-tile types -----
    chunk_windows_in_size = WINDOWS_PER_CHUNK * WINDOW_BYTES_IN     # 384
    chunk_windows_out_size = WINDOWS_PER_CHUNK * SPACER_BYTES       # 320
    chunk_pam_meta_size = WINDOWS_PER_CHUNK * PAM_BYTES             # 64
    partial_chunk_size = WINDOWS_PER_CHUNK * GUIDES_PER_TILE        # 4096
    sparse_chunk_size = EMIT_SLOT_BYTES                             # 2048

    guides_full_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_in_chunk_ty = np.ndarray[(chunk_windows_in_size,), np.dtype[np.uint8]]
    windows_out_chunk_ty = np.ndarray[(chunk_windows_out_size,), np.dtype[np.uint8]]
    pam_meta_chunk_ty = np.ndarray[(chunk_pam_meta_size,), np.dtype[np.uint8]]
    partial_chunk_ty = np.ndarray[(partial_chunk_size,), np.dtype[np.uint8]]
    sparse_chunk_ty = np.ndarray[(sparse_chunk_size,), np.dtype[np.uint8]]

    # ----- external kernel functions (compiled C++) -----
    # Tile A's symbol is mode-specific.
    filter_symbol = {
        "early": "crispr_pam_filter_tile_a_early",
        "late": "crispr_pam_filter_tile_a_late",
    }[mode]
    filter_fn = Kernel(
        filter_symbol,
        "tile_a_filter.o",
        [
            windows_in_chunk_ty,           # 64 input records (5+1 bytes each)
            windows_out_chunk_ty,          # filtered/forwarded windows
            pam_meta_chunk_ty,             # per-window PAM-pass byte (1=pass, 0=skip)
            np.int32,                      # n_windows in this chunk
        ],
    )

    # Match kernel — same C++ symbol as . The kernel is bit-equal.
    match_fn = Kernel(
        "crispr_match_multitile_match",
        "tile_a_filter.o",                 # symbol resolved from the same .o; aiecc links
        [
            guides_full_ty,
            windows_out_chunk_ty,
            partial_chunk_ty,
            np.int32,                      # n_windows
            np.int32,                      # guide_offset
        ],
    )

    # Tile Z — threshold + sparse-emit. Mode-specific symbol.
    emit_symbol = {
        "early": "crispr_pam_filter_tile_z_early",
        "late": "crispr_pam_filter_tile_z_late",
    }[mode]
    emit_fn = Kernel(
        emit_symbol,
        "tile_a_filter.o",
        [
            partial_chunk_ty,              # partial_0 (64 windows × 64 guides)
            partial_chunk_ty,              # partial_1
            pam_meta_chunk_ty,             # PAM-pass byte per window (filter-late only)
            sparse_chunk_ty,               # output ring slot
            np.int32,                      # n_windows
            np.int32,                      # max_mismatches
            np.int32,                      # chunk_base_window_idx (for record output)
        ],
    )

    # ----- ObjectFifos -----
    # Guides: shim → match tiles. Identical to .
    of_guides = ObjectFifo(guides_full_ty, name="guides", depth=1)
    # Windows in (with PAM context): shim → Tile A.
    of_windows_in = ObjectFifo(windows_in_chunk_ty, name="windows_in", depth=2)
    # Windows out (PAM-filtered or pass-through): Tile A → match tiles.
    #
    # T9 / dynamic-fifo half (composes with T3's vectorisation
    # half). T4's `VariableRateFifo` carries a discardable
    # `aie.variable_rate = true` attribute that the
    # `AIEObjectFifoStatefulTransformPass` consumes to:
    #   1. EXCLUDE this fifo from LCM-based loop unrolling on the producer
    #      (Tile A) side — the per-chunk loop is no longer unrolled on
    #      this fifo's account.
    #   2. ROUTE accesses through the `dynamicGlobalObjectFifos` runtime-
    #      counter machinery — counters advance only on actual releases
    #      (no static-rate assumption).
    #   3. PROPAGATE the marker through split-fifo paths to the consumer-
    #      side fifo (broadcast to match_0, match_1).
    #
    # Concrete cycle-time impact this delivers (orthogonal to T3's
    # vectorisation, which targeted per-window inner-loop compute):
    # the per-chunk shim-DMA / FIFO-sync overhead × 64 chunks dominates
    # the ~6.14 s/launch floor T3 measured. By relaxing the static-rate
    # invariant on the producer fifo, the BD chain on the producer side
    # is no longer locked to LCM(producer-fifo-size, consumer-fifo-size)
    # iterations, removing one source of dispatch overhead per chunk.
    #
    # Producer-side discard semantics (kernel-driven, not visible at the
    # IRON Python layer per T4's worked example): the kernel can branch
    # within an iteration to NOT actually consume a slot on this fifo
    # if all WINDOWS_PER_CHUNK windows in the chunk failed PAM. The
    # current C++ kernel (tile_a_filter.cc::crispr_pam_filter_tile_a_early)
    # always writes a chunk's worth of zero-filled spacers when PAM
    # fails per-window; per-chunk skip would be an incremental v2
    # optimisation gated on this primitive.
    # V1 emits exactly one fixed-size output chunk per input chunk. Keep
    # this as a static ObjectFifo; the variable-rate path is reserved for
    # the future per-chunk discard implementation.
    of_windows_out = ObjectFifo(windows_out_chunk_ty, name="windows_out", depth=2)
    # PAM-meta: Tile A → Tile Z (only consumed in filter-late mode).
    of_pam_meta = ObjectFifo(pam_meta_chunk_ty, name="pam_meta", depth=2)
    # Partials from match tiles → Tile Z.
    of_partials = [
        ObjectFifo(partial_chunk_ty, name=f"partial_{i}", depth=2)
        for i in range(N_MATCH_TILES)
    ]
    # Sparse-emit slot: Tile Z → shim (ring-buffer style).
    of_sparse = ObjectFifo(sparse_chunk_ty, name="sparse_out", depth=2)

    # ----- Tile A worker -----
    def tile_a_body(of_w_in, of_w_out, of_pam, filter_kernel):
        for _ in range_(N_CHUNKS):
            elem_in = of_w_in.acquire(1)
            elem_out = of_w_out.acquire(1)
            elem_pam = of_pam.acquire(1)
            filter_kernel(elem_in, elem_out, elem_pam, WINDOWS_PER_CHUNK)
            of_w_in.release(1)
            of_w_out.release(1)
            of_pam.release(1)

    tile_a_worker = Worker(
        tile_a_body,
        fn_args=[
            of_windows_in.cons(),
            of_windows_out.prod(),
            of_pam_meta.prod(),
            filter_fn,
        ],
    )

    # ----- match worker bodies (one per match tile) — same shape as -----
    def match_body(of_g, of_w, of_p, match_kernel, guide_offset):
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
                of_windows_out.cons(),
                of_partials[i].prod(),
                match_fn,
                int(i * GUIDES_PER_TILE),
            ],
        )
        for i in range(N_MATCH_TILES)
    ]

    # ----- Tile Z worker — threshold + sparse-emit -----
    # v1 caveat: IRON's `range_` yields an MLIR `index`, but the
    # emit kernel's third int32 arg is the chunk-base window index. We
    # avoid the type mismatch by tracking the chunk base in a separate
    # int counter (initialised at 0, incremented inside the body via a
    # helper kernel). For v1 we sidestep this by passing 0 — Tile Z
    # writes the local-chunk window_idx and the host applies the
    # chunk_base offset post-hoc when concatenating chunks. This is
    # equivalent to the on-tile chunk_base path but simpler to lower.
    def tile_z_body(of_p0, of_p1, of_pam, of_out, emit_kernel):
        for _ in range_(N_CHUNKS):
            e0 = of_p0.acquire(1)
            e1 = of_p1.acquire(1)
            ep = of_pam.acquire(1)
            eo = of_out.acquire(1)
            emit_kernel(
                e0, e1, ep, eo,
                WINDOWS_PER_CHUNK,
                4,                      # max_mismatches (host can override post-hoc)
                0,                      # chunk_base_window_idx (host applies offset)
            )
            of_p0.release(1)
            of_p1.release(1)
            of_pam.release(1)
            of_out.release(1)

    tile_z_worker = Worker(
        tile_z_body,
        fn_args=[
            of_partials[0].cons(),
            of_partials[1].cons(),
            of_pam_meta.cons(),
            of_sparse.prod(),
            emit_fn,
        ],
    )

    # ----- runtime sequence (shim DMA in/out) -----
    rt = Runtime()
    with rt.sequence(guides_ty, windows_in_ty, sparse_out_ty) as (G, W, Out):  # noqa: N806
        rt.start(tile_a_worker, *match_workers, tile_z_worker)
        rt.fill(of_guides.prod(), G)
        rt.fill(of_windows_in.prod(), W)
        rt.drain(of_sparse.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device", help="AIE device")
    p.add_argument(
        "--mode",
        required=True,
        choices=("early", "late"),
        help="filter-early (production) or filter-late (comparison).",
    )
    opts = p.parse_args(sys.argv[1:])

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    # followup-A: optionally inject burst_length on shim BDs
    # via the BIONPU_PAM_FILTER_SHIM_BURST_LENGTH env var. No-op when
    # the env var is unset (default IRON behavior preserved).
    _maybe_install_burst_length_override()

    print(crispr_pam_filter(dev, mode=opts.mode))

if __name__ == "__main__":
    main()
