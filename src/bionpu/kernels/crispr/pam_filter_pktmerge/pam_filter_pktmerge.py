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

# pam_filter_pktmerge.py — IRON lowering for PacketFifo retrofit
# of the PAM-filter / threshold / sparse-emit CRISPR streaming
# kernel (C-M5-pktmerge).
#
# Phase 2 retrofit using the fork's PacketFifo ( — variable-rate
# pktMerge / finish-on-TLAST / out-of-order BD): closes the OTHER HALF
# of (Phase 1 documented; Phase 2 closes via PacketFifo).
#
# # Honest-deviation note
#
# attempted to wire a PacketFifo edge beside the live ObjectFifo
# edge (Tile A -> match tiles). That was the wrong shape: it proved
# packetflow lowering, but it did not replace the data path and consumed
# the same switch/DMA resources as the ObjectFifo route. The fork's 
# PacketFifo remains useful as a topology declaration (see
# `tests/test_iron_packet_fifo.py`), but CRISPR must not emit it as a
# live sidecar.
#
# The retrofit therefore ships in two layers:
#
#   1. **Live IRON Program** (this file's ``crispr_pam_filter_pktmerge``
# function): the byte-equal twin of filter-early. Uses the
# same ObjectFifo wiring uses; produces a buildable xclbin
#      that runs on AIE2P silicon and produces byte-identical sparse-
# emit output to filter-early on the same input. The C++
#      kernel symbols are aliased to ``crispr_pam_filter_tile_a_pktmerge``
#      etc. so the IRON-level Kernel binding traces a separate code
#      path even though the math is the same.
#
#   2. **PacketFifo topology declaration** (``pktmerge_topology()``
# below): the canonical topology constructed at API-surface
#      level. The final speedup path is not a sidecar packetflow; it is
#      replacing `windows_out` with a packetized stream/DMA contract and
#      updating the match-tile kernel ABI accordingly. PacketFifo packet ids
#      are assigned per producer in the fork, not per window, so invalid
#      windows must not be emitted; valid packets carry original window_idx +
#      spacer bytes.
#
# Output bytes are byte-identical to 's filter-early + filter-late
# on the same input after canonical normalization. The same XOR +
# popcount + NGG arithmetic runs in the match kernel and the Tile Z
# emit kernel; only the IRON-level Kernel-symbol routing changes
# between and .
#
# Determinism: byte-identical across runs (the kernel arithmetic is
# bit-deterministic; ObjectFifo round-robin is order-deterministic
# for a single producer; PacketFifo's round-robin merge carries the
# same property at silicon level).
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Math kernels derived verbatim from
# bionpu/kernels/crispr/pam_filter/tile_a_filter.cc. Topology
# adapted via the fork's PacketFifo primitive.

import argparse
import sys

import numpy as np
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    PacketFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1, Tile

# Pinned shape — kept identical to / / so byte-equality
# tests + fixture tooling apply unchanged.
N_GUIDES = 128
SPACER_BYTES = 5  # 20 nt × 2 bits / 8 = 5 bytes
N_WINDOWS = 4096
PAM_LEN = 3  # NGG
PAM_BYTES = 1  # 3-nt PAM packs into 6 bits → 1 byte (with 2 bits padding).

# Per Tile A input record: 5 bytes spacer + 1 byte PAM context = 6 bytes
# (matches 's WINDOW_BYTES_IN exactly).
WINDOW_BYTES_IN = SPACER_BYTES + PAM_BYTES  # 6

# Multi-tile fan-out (unchanged from / — DMA-channel constraint
# still applies).
N_MATCH_TILES = 2
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 64

# Per-chunk geometry (mirrors / so the host-side window stream
# is reusable byte-for-byte).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64

# Sparse-emit ring buffer geometry (Tile Z).
EMIT_RECORD_BYTES = 8
EMIT_SLOT_RECORDS = 256
EMIT_SLOT_BYTES = EMIT_RECORD_BYTES * EMIT_SLOT_RECORDS

# packet-routing tags. The valid-bit gates pktMerge filtering.
PACKET_ID_VALID = 1    # PAM-passing windows
PACKET_ID_INVALID = 0  # legacy host metadata only; not a PacketFifo route
PACKETIZED_WINDOW_INDEX_BYTES = 4
PACKETIZED_WINDOW_PAD_BYTES = 3
PACKETIZED_WINDOW_PAYLOAD_BYTES = (
    PACKETIZED_WINDOW_INDEX_BYTES + SPACER_BYTES + PACKETIZED_WINDOW_PAD_BYTES
)
PACKETIZED_WINDOW_LOGICAL_WORDS = PACKETIZED_WINDOW_PAYLOAD_BYTES // 4
PACKETIZED_STREAM_WORD_BYTES = 8
PACKETIZED_STREAM_LOGICAL_WORDS_PER_CHUNK = (
    1 + WINDOWS_PER_CHUNK * PACKETIZED_WINDOW_LOGICAL_WORDS
)
PACKETIZED_STREAM_WORDS_PER_CHUNK = (
    PACKETIZED_STREAM_LOGICAL_WORDS_PER_CHUNK + 1
) // 2
PACKETIZED_STREAM_BEATS_PER_CHUNK = (PACKETIZED_STREAM_WORDS_PER_CHUNK + 1) // 2
COMPACT_PACKET_LOGICAL_WORDS_PER_WINDOW = 2
COMPACT_PACKET_LOGICAL_WORDS_PER_CHUNK = (
    1 + WINDOWS_PER_CHUNK * COMPACT_PACKET_LOGICAL_WORDS_PER_WINDOW
)
COMPACT_PACKET_WORDS_PER_CHUNK = (COMPACT_PACKET_LOGICAL_WORDS_PER_CHUNK + 1) // 2
COMPACT_MATCH_RECORD_BYTES = 1 + GUIDES_PER_TILE
COMPACT_MATCH_CHUNK_BYTES = WINDOWS_PER_CHUNK * COMPACT_MATCH_RECORD_BYTES
PACKETIZED_VALID_COUNT_BYTES = 4
PACKETIZED_MATCH_RECORD_BYTES = 4 + GUIDES_PER_TILE
PACKETIZED_MATCH_CHUNK_BYTES = WINDOWS_PER_CHUNK * PACKETIZED_MATCH_RECORD_BYTES

def pktmerge_topology():
    """Construct the canonical PacketFifo topology at API-surface level.

    Returns:
        A :class:`PacketFifo` instance: 1 producer (Tile A), N_MATCH_TILES
        consumers (the match tiles), packet id ``PACKET_ID_VALID``, a
        12-byte payload (uint32 original window_idx, 5 spacer bytes, 3 pad
        bytes), round-robin merge_strategy (pktMerge default), and
        keep_pkt_header=False (finish-on-TLAST mode).

    This function ratifies the corrected replacement contract. The fork's
    PacketFifo packetflow lowering routes by per-producer packet_id, so
    CRISPR cannot express "emit invalid packets and have fabric drop them"
    with a per-window header. Tile A must emit only PAM-passing windows,
    and each packet must carry the original window_idx so Tile Z can emit
    stable sparse coordinates after variable-rate compaction.
    """
    tile_a = Tile(0, 2)
    match_tiles = [Tile(0, 3 + i) for i in range(N_MATCH_TILES)]
    packetized_window_ty = np.ndarray[
        (PACKETIZED_WINDOW_PAYLOAD_BYTES,),
        np.dtype[np.uint8],
    ]
    return PacketFifo(
        producers=[tile_a],
        consumers=match_tiles,
        header_dtype="uint8",
        merge_strategy="round-robin",
        packet_ids=[PACKET_ID_VALID],
        obj_type=packetized_window_ty,
        keep_pkt_header=False,
        name="pf_windows_t33",
    )

def pktmerge_packetized_window_edge():
    """Construct the IRON stream edge for the real replacement path.

    This is the first-class IRON form of "replace ``of_windows_out``":
    a full core-to-core stream ObjectFifo carrying 12-byte valid-window
    packets. The old ObjectFifo memref path used 64 fixed slots per chunk;
    this stream edge carries only PAM-passing windows, with the original
    window index embedded in the payload.

    The edge intentionally has ``aie_stream=2`` (full stream). Workers
    attached to it must not call ObjectFifo ``acquire`` / ``release`` on
    this endpoint; the C++ kernels must use AIE stream intrinsics. The
    IRON wrapper now catches accidental acquire/release before aie-opt
    emits the lower-level "cannot acquire from objectfifo stream port"
    diagnostic.
    """
    packetized_window_ty = np.ndarray[
        (PACKETIZED_WINDOW_PAYLOAD_BYTES,),
        np.dtype[np.uint8],
    ]
    return ObjectFifo(
        packetized_window_ty,
        name="packetized_windows",
        depth=2,
        aie_stream=2,
        aie_stream_port=1,
    )

def pktmerge_direct_stream_edges():
    """Construct two direct stream route markers for valid-window broadcast.

    The ADF ``input_stream`` / ``output_stream`` C++ ABI is not available in
    the Peano-only toolchain. Pure MLIR ``aie.put_stream`` /
    ``aie.get_stream`` loops were also rejected for this kernel because the
    i32 form forced massive unrolling and the wider forms failed verifier or
    backend lowering. The buildable direct-stream path therefore emits only
    route markers here; packet words move inside Peano C++ kernels via
    ``put_ms`` / ``get_ss_int``.

    Two route declarations are needed because both 64-guide match tiles
    consume the same PAM-passing window packets. AIE2P's pathfinder only
    models Core stream channel 0 here, so the direct-stream prototype emits a
    single Tile-A put stream and lowers the two declarations to Core:0 fanout
    routes in ``_lower_direct_stream_flows``.
    """
    stream_route_marker_ty = np.ndarray[(1,), np.dtype[np.int32]]
    return [
        ObjectFifo(
            stream_route_marker_ty,
            name=f"packetized_words_stream_{i}",
            depth=2,
            aie_stream=2,
            aie_stream_port=0,
        )
        for i in range(N_MATCH_TILES)
    ]

def pktmerge_stream_sideband_edges():
    """Construct sideband ObjectFifos needed by the stream replacement.

    The packet stream is variable-rate, so match workers need a fixed-rate
    per-chunk count before they know how many 12-byte packets to consume.
    Each match tile then writes compact partial records keyed by original
    window index instead of the old fixed ``windows_out`` slot index.
    """
    valid_count_ty = np.ndarray[
        (PACKETIZED_VALID_COUNT_BYTES,), np.dtype[np.uint8]
    ]
    partial_packet_ty = np.ndarray[
        (PACKETIZED_MATCH_CHUNK_BYTES,), np.dtype[np.uint8]
    ]
    return (
        ObjectFifo(valid_count_ty, name="packetized_valid_count", depth=2),
        [
            ObjectFifo(partial_packet_ty, name=f"packetized_partial_{i}", depth=2)
            for i in range(N_MATCH_TILES)
        ],
    )

def _put_packet_words_direct_stream(packet_words, stream_port: int):
    """Experimental pure-MLIR stream emitter retained for backend probes.

    Do not use this for the buildable CRISPR direct-stream path. It expands
    packet loops into many in-core ``aie.put_stream`` ops; the production
    prototype keeps stream loops in Peano C++ intrinsics instead.
    """
    from aie.dialects import aie, arith, memref
    from aie.ir import IndexType, IntegerType

    i32 = IntegerType.get_signless(32)
    i128 = IntegerType.get_signless(128)
    index = IndexType.get()
    port = arith.constant(i32, stream_port)
    shift64 = arith.constant(i128, 64)
    packet_memref = getattr(packet_words, "op", packet_words)
    for beat_idx in range(PACKETIZED_STREAM_BEATS_PER_CHUNK):
        lo_idx = arith.constant(index, beat_idx * 2)
        hi_idx = arith.constant(index, beat_idx * 2 + 1)
        lo = arith.extui(i128, memref.load(packet_memref, [lo_idx]))
        hi = arith.extui(i128, memref.load(packet_memref, [hi_idx]))
        word = arith.ori(lo, arith.shli(hi, shift64))
        aie.put_stream(port, word)

def _get_packet_words_direct_stream(packet_words, stream_port: int):
    """Experimental pure-MLIR stream receiver retained for backend probes."""
    from aie.dialects import aie, arith, memref
    from aie.ir import IndexType, IntegerType

    i32 = IntegerType.get_signless(32)
    i64 = IntegerType.get_signless(64)
    i128 = IntegerType.get_signless(128)
    index = IndexType.get()
    port = arith.constant(i32, stream_port)
    shift64 = arith.constant(i128, 64)
    packet_memref = getattr(packet_words, "op", packet_words)
    for beat_idx in range(PACKETIZED_STREAM_BEATS_PER_CHUNK):
        lo_idx = arith.constant(index, beat_idx * 2)
        hi_idx = arith.constant(index, beat_idx * 2 + 1)
        word = aie.get_stream(i128, port)
        lo = arith.trunci(i64, word)
        hi = arith.trunci(i64, arith.shrui(word, shift64))
        memref.store(lo, packet_memref, [lo_idx])
        memref.store(hi, packet_memref, [hi_idx])

def _copy_valid_count(count_src, count_dst):
    from aie.dialects import arith, memref
    from aie.ir import IndexType

    src = getattr(count_src, "op", count_src)
    dst = getattr(count_dst, "op", count_dst)
    index = IndexType.get()
    for idx in range(PACKETIZED_VALID_COUNT_BYTES):
        idx_v = arith.constant(index, idx)
        memref.store(memref.load(src, [idx_v]), dst, [idx_v])

def _stream_port(edge_handle) -> int:
    return edge_handle._object_fifo.aie_stream_port

def _lower_direct_stream_flows(mlir: str) -> str:
    """Replace IRON stream ObjectFifo markers with explicit core flows.

    The upstream objectFifo stateful transform currently trips over IRON's
    logical-tile form for ``aie_stream=2`` endpoints before it can produce the
    simple routes we need. Emit those routes directly:

    Tile A Core:0 -> match tile 0 Core:0
    Tile A Core:0 -> match tile 1 Core:0

    That form gets past pathfinder and matches the single-port broadcast
    stream body below.
    """
    replacements = {
        (
            "    aie.objectfifo @packetized_words_stream_0(%logical_core, "
            "{%logical_core_0}, 2 : i32) {aie_stream = 2 : i32, "
            "aie_stream_port = 0 : i32} : !aie.objectfifo<memref<1xi32>> \n"
        ): "    aie.flow(%logical_core, Core : 0, %logical_core_0, Core : 0)\n",
        (
            "    aie.objectfifo @packetized_words_stream_1(%logical_core, "
            "{%logical_core_1}, 2 : i32) {aie_stream = 2 : i32, "
            "aie_stream_port = 0 : i32} : !aie.objectfifo<memref<1xi32>> \n"
        ): "    aie.flow(%logical_core, Core : 0, %logical_core_1, Core : 0)\n",
    }
    for old, new in replacements.items():
        if old not in mlir:
            raise RuntimeError(
                "direct-stream flow lowering did not find expected stream "
                "ObjectFifo marker"
            )
        mlir = mlir.replace(old, new)
    return mlir

def crispr_pam_filter_pktmerge(dev):
    """Build the IRON program for the PacketFifo PAM-filter kernel.

    **Honest deviation**: the live program ships as a
    byte-equal twin of filter-early (ObjectFifo wiring). A previous
    opt-in sidecar PacketFifo path emitted packetflow MLIR, but the data
    path still used ObjectFifo buffers because the current kernels
    consume memrefs and PacketFifo handles expose stream semantics. That
    sidecar is intentionally disabled; real pktMerge support must replace
    `windows_out` instead of coexisting with it.

    Output bytes are byte-identical to filter-early on the same
    input (same arithmetic, same chunk geometry). The xclbin is a
    distinct artifact (different IRON Kernel-symbol bindings) so
    operator scripts can dispatch ``crispr_pam_filter_pktmerge`` as a
    separate op for measurement.

    Args:
        dev: AIE device (NPU2 for AIE2P; NPU1Col1 for AIE2).

    Returns:
        Resolved IRON Program ready for aiecc.
    """
    # Construct the PacketFifo topology declaration for tests and design
    # ratification only. Do not emit it as a live sidecar: PacketFifoHandle
    # acquire() does not return a memref, so threading the handle through the
    # existing ObjectFifo kernels only creates a conflicting route while the
    # real data still moves through ObjectFifo.
    _ = pktmerge_topology()

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
    # Tile A — pktMerge variant. Same I/O signature as 's filter-
    # early. The C++ kernel sets per-window valid bits in pam_meta;
    # in the live (ObjectFifo-twin) path the per-window valid bits
    # gate Tile Z's emit (same as filter-early). On the silicon-
    # level pktMerge path (post-), the same per-window bits
    # become the AXI stream switch's per-packet routing tag.
    filter_fn = Kernel(
        "crispr_pam_filter_tile_a_pktmerge",
        "tile_a_pktmerge.o",
        [
            windows_in_chunk_ty,           # 64 input records (5+1 bytes each)
            windows_out_chunk_ty,          # filtered windows (compact)
            pam_meta_chunk_ty,             # per-window header (1=valid, 0=invalid)
            np.int32,                      # n_windows in this chunk
        ],
    )

    # Match kernel — same C++ symbol + arithmetic as / . Bit-equal.
    match_fn = Kernel(
        "crispr_match_multitile_match",
        "tile_a_pktmerge.o",
        [
            guides_full_ty,
            windows_out_chunk_ty,
            partial_chunk_ty,
            np.int32,                      # n_windows
            np.int32,                      # guide_offset
        ],
    )

    # Tile Z — same threshold + sparse-emit as filter-early.
    emit_fn = Kernel(
        "crispr_pam_filter_tile_z_pktmerge",
        "tile_a_pktmerge.o",
        [
            partial_chunk_ty,              # partial_0 (64 windows × 64 guides)
            partial_chunk_ty,              # partial_1
            pam_meta_chunk_ty,             # PAM-pass byte per window
            sparse_chunk_ty,               # output ring slot
            np.int32,                      # n_windows
            np.int32,                      # max_mismatches
            np.int32,                      # chunk_base_window_idx
        ],
    )

    # ----- ObjectFifos (live-program wiring; byte-equal twin of ) -----
    of_guides = ObjectFifo(guides_full_ty, name="guides", depth=1)
    of_windows_in = ObjectFifo(windows_in_chunk_ty, name="windows_in", depth=2)
    of_windows_out = ObjectFifo(windows_out_chunk_ty, name="windows_out", depth=2)
    of_pam_meta = ObjectFifo(pam_meta_chunk_ty, name="pam_meta", depth=2)
    of_partials = [
        ObjectFifo(partial_chunk_ty, name=f"partial_{i}", depth=2)
        for i in range(N_MATCH_TILES)
    ]
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

    tile_a_args = [
        of_windows_in.cons(),
        of_windows_out.prod(),
        of_pam_meta.prod(),
        filter_fn,
    ]
    tile_a_worker = Worker(
        tile_a_body,
        fn_args=tile_a_args,
    )

    # ----- match worker bodies — verbatim from filter-early -----
    def match_body(of_g, of_w, of_p, match_kernel, guide_offset):
        elem_g = of_g.acquire(1)
        for _ in range_(N_CHUNKS):
            elem_w = of_w.acquire(1)
            elem_p = of_p.acquire(1)
            match_kernel(elem_g, elem_w, elem_p, WINDOWS_PER_CHUNK, guide_offset)
            of_w.release(1)
            of_p.release(1)
        of_g.release(1)

    match_workers = []
    for i in range(N_MATCH_TILES):
        match_args = [
            of_guides.cons(),
            of_windows_out.cons(),
            of_partials[i].prod(),
            match_fn,
            int(i * GUIDES_PER_TILE),
        ]
        match_workers.append(
            Worker(
                match_body,
                fn_args=match_args,
            )
        )

    # ----- Tile Z worker — threshold + sparse-emit -----
    def tile_z_body(of_p0, of_p1, of_pam, of_out, emit_kernel):
        for _ in range_(N_CHUNKS):
            e0 = of_p0.acquire(1)
            e1 = of_p1.acquire(1)
            ep = of_pam.acquire(1)
            eo = of_out.acquire(1)
            emit_kernel(
                e0, e1, ep, eo,
                WINDOWS_PER_CHUNK,
                4,                      # max_mismatches
                0,                      # chunk_base_window_idx
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

def crispr_pam_filter_pktmerge_direct_stream(dev):
    """Build the Peano-only direct-stream PacketFifo replacement prototype.

    This variant avoids the unavailable ADF C++ stream-pointer ABI. C++
    packetizes valid windows and moves packet words with Peano's low-level
    AIE2P ``put_ms`` / ``get_ss_int`` intrinsics, while IRON emits only the
    explicit core stream routes and compact per-chunk worker loops.

    The default shipped artifact still uses :func:`crispr_pam_filter_pktmerge`
    until this direct-stream topology is silicon-ratified.
    """
    guides_size = N_GUIDES * SPACER_BYTES
    windows_in_size = N_WINDOWS * WINDOW_BYTES_IN
    sparse_out_size = N_CHUNKS * EMIT_SLOT_BYTES

    guides_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_in_ty = np.ndarray[(windows_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    chunk_windows_in_size = WINDOWS_PER_CHUNK * WINDOW_BYTES_IN
    sparse_chunk_size = EMIT_SLOT_BYTES

    guides_full_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_in_chunk_ty = np.ndarray[(chunk_windows_in_size,), np.dtype[np.uint8]]
    valid_count_ty = np.ndarray[(PACKETIZED_VALID_COUNT_BYTES,), np.dtype[np.uint8]]
    partial_packet_ty = np.ndarray[
        (PACKETIZED_MATCH_CHUNK_BYTES,), np.dtype[np.uint8]
    ]
    sparse_chunk_ty = np.ndarray[(sparse_chunk_size,), np.dtype[np.uint8]]

    packetize_fn = Kernel(
        "crispr_pam_filter_tile_a_pktmerge_stream_i32",
        "tile_a_pktmerge.o",
        [
            windows_in_chunk_ty,
            valid_count_ty,
            np.int32,
            np.int32,
        ],
    )
    match_packet_fn = Kernel(
        "crispr_match_multitile_match_packetized_stream_i32",
        "tile_a_pktmerge.o",
        [
            guides_full_ty,
            partial_packet_ty,
            valid_count_ty,
            np.int32,
        ],
    )
    emit_packet_fn = Kernel(
        "crispr_pam_filter_tile_z_pktmerge_packetized",
        "tile_a_pktmerge.o",
        [
            partial_packet_ty,
            partial_packet_ty,
            valid_count_ty,
            sparse_chunk_ty,
            np.int32,
        ],
    )

    of_guides = ObjectFifo(guides_full_ty, name="ds_guides", depth=1)
    of_windows_in = ObjectFifo(windows_in_chunk_ty, name="ds_windows_in", depth=2)
    of_counts = [
        ObjectFifo(valid_count_ty, name=f"ds_valid_count_{i}", depth=2)
        for i in range(N_MATCH_TILES + 1)
    ]
    of_partials = [
        ObjectFifo(partial_packet_ty, name=f"ds_partial_packet_{i}", depth=2)
        for i in range(N_MATCH_TILES)
    ]
    of_sparse = ObjectFifo(sparse_chunk_ty, name="ds_sparse_out", depth=2)
    stream_edges = pktmerge_direct_stream_edges()

    count_local = Buffer(valid_count_ty, name="ds_count_local")

    def tile_a_body(of_w_in, of_c0, of_c1, of_cz, _stream0, _stream1,
                    count_buf, packetize_kernel):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        chunk_size = arith.constant(i32, WINDOWS_PER_CHUNK)
        for chunk_idx in range_(N_CHUNKS):
            elem_in = of_w_in.acquire(1)
            count0 = of_c0.acquire(1)
            count1 = of_c1.acquire(1)
            countz = of_cz.acquire(1)
            chunk_i32 = arith.index_cast(i32, chunk_idx)
            base_idx = arith.muli(chunk_i32, chunk_size)
            packetize_kernel(
                elem_in,
                count_buf.op,
                chunk_size,
                base_idx,
            )
            _copy_valid_count(count_buf, count0)
            _copy_valid_count(count_buf, count1)
            _copy_valid_count(count_buf, countz)
            of_w_in.release(1)
            of_c0.release(1)
            of_c1.release(1)
            of_cz.release(1)

    tile_a_worker = Worker(
        tile_a_body,
        fn_args=[
            of_windows_in.cons(),
            of_counts[0].prod(),
            of_counts[1].prod(),
            of_counts[2].prod(),
            stream_edges[0].prod(),
            stream_edges[1].prod(),
            count_local,
            packetize_fn,
        ],
        tile=Tile(0, 2),
    )

    def match_body(of_g, of_count, _stream_edge, of_p, match_kernel, guide_offset):
        elem_g = of_g.acquire(1)
        for _ in range_(N_CHUNKS):
            elem_count = of_count.acquire(1)
            elem_p = of_p.acquire(1)
            match_kernel(elem_g, elem_p, elem_count, guide_offset)
            of_count.release(1)
            of_p.release(1)
        of_g.release(1)

    match_workers = []
    for i in range(N_MATCH_TILES):
        match_workers.append(
            Worker(
                match_body,
                fn_args=[
                    of_guides.cons(),
                    of_counts[i].cons(),
                    stream_edges[i].cons(),
                    of_partials[i].prod(),
                    match_packet_fn,
                    int(i * GUIDES_PER_TILE),
                ],
                tile=Tile(0, 3 + i),
            )
        )

    def tile_z_body(of_p0, of_p1, of_count, of_out, emit_kernel):
        for _ in range_(N_CHUNKS):
            e0 = of_p0.acquire(1)
            e1 = of_p1.acquire(1)
            ec = of_count.acquire(1)
            eo = of_out.acquire(1)
            emit_kernel(e0, e1, ec, eo, 4)
            of_p0.release(1)
            of_p1.release(1)
            of_count.release(1)
            of_out.release(1)

    tile_z_worker = Worker(
        tile_z_body,
        fn_args=[
            of_partials[0].cons(),
            of_partials[1].cons(),
            of_counts[2].cons(),
            of_sparse.prod(),
            emit_packet_fn,
        ],
        tile=Tile(0, 5),
    )

    rt = Runtime()
    with rt.sequence(guides_ty, windows_in_ty, sparse_out_ty) as (G, W, Out):  # noqa: N806
        rt.start(tile_a_worker, *match_workers, tile_z_worker)
        rt.fill(of_guides.prod(), G)
        rt.fill(of_windows_in.prod(), W)
        rt.drain(of_sparse.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program()

def crispr_pam_filter_pktmerge_compact_packets(dev):
    """Build compact packet ObjectFifo replacement for stream-fanout pressure.

    Tile A packetizes only PAM-passing windows into a compact memref. The
    compact packet object is multicast through normal ObjectFifo wiring to the
    two match tiles, avoiding the Core:0 stream fanout that stalls under denser
    valid-window traffic. The compact ObjectFifo ABI packs the local window
    index into the two-word spacer payload instead of sending a global index
    word.
    """
    guides_size = N_GUIDES * SPACER_BYTES
    windows_in_size = N_WINDOWS * WINDOW_BYTES_IN
    sparse_out_size = N_CHUNKS * EMIT_SLOT_BYTES

    guides_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_in_ty = np.ndarray[(windows_in_size,), np.dtype[np.uint8]]
    sparse_out_ty = np.ndarray[(sparse_out_size,), np.dtype[np.uint8]]

    chunk_windows_in_size = WINDOWS_PER_CHUNK * WINDOW_BYTES_IN
    packet_words_ty = np.ndarray[
        (COMPACT_PACKET_WORDS_PER_CHUNK,), np.dtype[np.int64]
    ]
    partial_packet_ty = np.ndarray[
        (COMPACT_MATCH_CHUNK_BYTES,), np.dtype[np.uint8]
    ]
    valid_count_ty = np.ndarray[(PACKETIZED_VALID_COUNT_BYTES,), np.dtype[np.uint8]]
    sparse_chunk_ty = np.ndarray[(EMIT_SLOT_BYTES,), np.dtype[np.uint8]]
    guides_full_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_in_chunk_ty = np.ndarray[(chunk_windows_in_size,), np.dtype[np.uint8]]

    packetize_fn = Kernel(
        "crispr_pam_filter_tile_a_pktmerge_packetize_spacers",
        "tile_a_pktmerge.o",
        [
            windows_in_chunk_ty,
            valid_count_ty,
            packet_words_ty,
            np.int32,
        ],
    )
    match_packet_fn = Kernel(
        "crispr_match_multitile_match_packetized_spacers_counted",
        "tile_a_pktmerge.o",
        [
            guides_full_ty,
            packet_words_ty,
            partial_packet_ty,
            np.int32,
        ],
    )
    emit_packet_fn = Kernel(
        "crispr_pam_filter_tile_z_pktmerge_packetized_indexed",
        "tile_a_pktmerge.o",
        [
            partial_packet_ty,
            partial_packet_ty,
            valid_count_ty,
            sparse_chunk_ty,
            np.int32,
        ],
    )

    of_guides = ObjectFifo(guides_full_ty, name="cp_guides", depth=1)
    of_windows_in = ObjectFifo(windows_in_chunk_ty, name="cp_windows_in", depth=2)
    of_packets = ObjectFifo(packet_words_ty, name="cp_packet_words", depth=2)
    of_count_z = ObjectFifo(valid_count_ty, name="cp_valid_count_z", depth=2)
    of_partials = [
        ObjectFifo(partial_packet_ty, name=f"cp_partial_packet_{i}", depth=2)
        for i in range(N_MATCH_TILES)
    ]
    of_sparse = ObjectFifo(sparse_chunk_ty, name="cp_sparse_out", depth=2)

    count_local = Buffer(valid_count_ty, name="cp_count_local")

    def tile_a_body(of_w_in, of_pkt, of_cz, count_buf, kernel):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        chunk_size = arith.constant(i32, WINDOWS_PER_CHUNK)
        for _ in range_(N_CHUNKS):
            elem_in = of_w_in.acquire(1)
            elem_pkt = of_pkt.acquire(1)
            countz = of_cz.acquire(1)
            kernel(elem_in, count_buf.op, elem_pkt, chunk_size)
            _copy_valid_count(count_buf, countz)
            of_w_in.release(1)
            of_pkt.release(1)
            of_cz.release(1)

    tile_a_worker = Worker(
        tile_a_body,
        fn_args=[
            of_windows_in.cons(),
            of_packets.prod(),
            of_count_z.prod(),
            count_local,
            packetize_fn,
        ],
        tile=Tile(0, 2),
    )

    def match_body(of_g, of_pkt, of_p, kernel, guide_offset):
        elem_g = of_g.acquire(1)
        for _ in range_(N_CHUNKS):
            elem_pkt = of_pkt.acquire(1)
            elem_p = of_p.acquire(1)
            kernel(elem_g, elem_pkt, elem_p, guide_offset)
            of_pkt.release(1)
            of_p.release(1)
        of_g.release(1)

    match_workers = []
    for i in range(N_MATCH_TILES):
        match_workers.append(
            Worker(
                match_body,
                fn_args=[
                    of_guides.cons(),
                    of_packets.cons(),
                    of_partials[i].prod(),
                    match_packet_fn,
                    int(i * GUIDES_PER_TILE),
                ],
                tile=Tile(0, 3 + i),
            )
        )

    def tile_z_body(of_p0, of_p1, of_count, of_out, kernel):
        for _ in range_(N_CHUNKS):
            e0 = of_p0.acquire(1)
            e1 = of_p1.acquire(1)
            ec = of_count.acquire(1)
            eo = of_out.acquire(1)
            kernel(e0, e1, ec, eo, 4)
            of_p0.release(1)
            of_p1.release(1)
            of_count.release(1)
            of_out.release(1)

    tile_z_worker = Worker(
        tile_z_body,
        fn_args=[
            of_partials[0].cons(),
            of_partials[1].cons(),
            of_count_z.cons(),
            of_sparse.prod(),
            emit_packet_fn,
        ],
        tile=Tile(0, 5),
    )

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
        "--direct-stream",
        action="store_true",
        help="emit the compact Peano-intrinsic direct stream prototype",
    )
    p.add_argument(
        "--compact-packets",
        action="store_true",
        help="emit the compact packet ObjectFifo prototype",
    )
    opts = p.parse_args(sys.argv[1:])

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    if opts.direct_stream and opts.compact_packets:
        raise ValueError("--direct-stream and --compact-packets are exclusive")
    if opts.direct_stream:
        print(_lower_direct_stream_flows(str(crispr_pam_filter_pktmerge_direct_stream(dev))))
    elif opts.compact_packets:
        print(crispr_pam_filter_pktmerge_compact_packets(dev))
    else:
        print(crispr_pam_filter_pktmerge(dev))

if __name__ == "__main__":
    main()
