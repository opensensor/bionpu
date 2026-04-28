// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// Per state/kmer_count_interface_contract.md (T1) — symbols, ObjectFifo
// names, and constants pinned there. This file implements the
// ``kmer_count_aggregator`` consumer tile (Tile Z in the topology) for
// the bionpu k-mer counter. The single ``extern "C"`` symbol is
// k-agnostic: by the time records reach this tile they are already
// canonical_u64 values in 16-byte ``EmitRecord`` form, so no per-k
// dispatch is required.
//
// Pattern reference (mirrored): the CRISPR pam_filter Tile-Z
// (``crispr_pam_filter_tile_z_early`` at
// bionpu-public/src/bionpu/kernels/crispr/pam_filter/tile_a_filter.cc:317).
// We widen the partial fan-in from 2 to MAX_TILES = 8, switch the
// 8-byte CRISPR record to a 16-byte ``EmitRecord(canonical, count,
// flags)``, drop the PAM/threshold filter, and pass through the
// ``EVICT_FLAG`` bit verbatim per acceptance criterion #7. The host
// runner (T7) re-aggregates duplicate ``canonical_u64`` entries by
// summing ``count_u32``; the aggregator does NOT dedupe — within-tile
// dedupe already happened in T5's open-addressed hash table, and
// cross-tile dedupe is host-side per the contract.
//
// Wire format on the partial_count_<i> ObjectFifo, per T1:
//   - First 4 bytes: little-endian uint32 ``n_records`` (count prefix).
//   - Next n_records * 16 bytes: packed ``EmitRecord`` entries.
// The buffer is sized to ``PARTIAL_OUT_BYTES = 49152`` (48 KiB) which
// caps the per-tile chunk at ~3071 records; the aggregator does NOT
// read beyond ``n_records`` records from any partial.
//
// Wire format on the sparse_out ObjectFifo (mirrors above):
//   - First 4 bytes: little-endian uint32 ``n_records`` count prefix.
//   - Next n_records * 16 bytes: packed ``EmitRecord`` entries (capped
//     at ``EMIT_SLOT_RECORDS = 1024``; any overflow is dropped at the
//     aggregator and the host runner is expected to re-launch with a
//     larger ring or accept truncation — this matches the post-fix
//     CRISPR pam_filter cap, which proved sufficient on chr22 once
//     bumped from 256 to 1024).

#include <stdint.h>
#include <stdio.h>

#include "kmer_count_constants.h"

// Compile-time sanity checks on the shared header. T5's tile kernel
// also asserts these; duplicating here makes the aggregator .o
// independently auditable.
static_assert(sizeof(EmitRecord) == 16,
              "EmitRecord must be 16 bytes packed");
static_assert(EMIT_RECORD_BYTES == 16,
              "EMIT_RECORD_BYTES must equal 16 (T1 contract)");
static_assert(EMIT_SLOT_BYTES == EMIT_SLOT_RECORDS * EMIT_RECORD_BYTES,
              "EMIT_SLOT_BYTES derivation drift vs T1 contract");
static_assert(MAX_TILES == 8,
              "MAX_TILES pinned at 8 by T1 contract");

// Length-prefixed read helper: extract the leading uint32 count prefix
// from a partial_count_<i> chunk. Bytes [0..3] are little-endian. We
// avoid a pointer-cast read so the kernel compiles cleanly under
// AIE2P clang++ with strict-aliasing discipline.
static inline uint32_t load_le_u32(const uint8_t* __restrict src) {
    uint32_t v0 = (uint32_t)src[0];
    uint32_t v1 = (uint32_t)src[1] << 8;
    uint32_t v2 = (uint32_t)src[2] << 16;
    uint32_t v3 = (uint32_t)src[3] << 24;
    return v0 | v1 | v2 | v3;
}

static inline void store_le_u32(uint8_t* __restrict dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xff);
    dst[1] = (uint8_t)((v >> 8) & 0xff);
    dst[2] = (uint8_t)((v >> 16) & 0xff);
    dst[3] = (uint8_t)((v >> 24) & 0xff);
}

// Byte-wise copy of a single 16-byte EmitRecord. Hand-unrolled so
// peano/aie2p does not lower to a memcpy intrinsic which (per the
// AIE2P shim BD payload cliff memory) can deterministically wedge on
// short payloads. 16 bytes inlined is well above the 256-byte BD
// payload cliff but we still avoid the libc symbol just in case the
// kernel ends up linked against a freestanding rt-lib.
static inline void copy_emit_record(uint8_t* __restrict dst,
                                     const uint8_t* __restrict src) {
    dst[0]  = src[0];   dst[1]  = src[1];
    dst[2]  = src[2];   dst[3]  = src[3];
    dst[4]  = src[4];   dst[5]  = src[5];
    dst[6]  = src[6];   dst[7]  = src[7];
    dst[8]  = src[8];   dst[9]  = src[9];
    dst[10] = src[10];  dst[11] = src[11];
    dst[12] = src[12];  dst[13] = src[13];
    dst[14] = src[14];  dst[15] = src[15];
}

extern "C" {

// ============================================================================
// kmer_count_aggregator — Tile Z
// ============================================================================
//
// Drains up to MAX_TILES = 8 partial_count_<i> ObjectFifo chunks into
// the single sparse_out ring slot. The signature widens the CRISPR
// Tile-Z fan-in from 2 to 8; unused inputs are NULL-able and skipped
// when ``n_tiles_active < MAX_TILES``.
//
// Behaviour:
//   1. Initialise the sparse_out length prefix to 0.
//   2. For each active partial input:
//        a. Read its 4-byte length prefix → n_partial_records.
//        b. Bounds-check against the per-partial cap
//           (PARTIAL_OUT_BYTES - 4) / EMIT_RECORD_BYTES.
//        c. For each record in the partial:
//             - If sparse_out is full (== EMIT_SLOT_RECORDS), break
//               out of all loops via the ``done`` label. Per T1 the
//               EVICT_FLAG passthrough policy says: do NOT silently
//               drop EVICT records preferentially; the host runner
//               re-aggregates duplicates regardless. Since we hit the
//               1024-record cap only on extremely k-mer-dense input,
//               truncating uniformly is the contract-correct choice.
//             - Copy the 16-byte EmitRecord verbatim into sparse_out
//               (preserves canonical, count, and the flags word
//               including EVICT_FLAG bit 0).
//             - Increment the running record count.
//   3. Write the final n_records count back into the sparse_out
//      length prefix.
//
// Note on optional in-tile sort: the T6 task description allows for
// "(b) optionally sort by count descending (within-tile is enough for
// v1; cross-tile sort happens host-side)". We DEFER that to the host
// runner — a 4096-record bubble/insertion sort on the AIE2P scalar
// core would dominate the per-launch budget, and the host already
// runs a full sort + dedupe pass to produce the Jellyfish-FASTA
// output. The aggregator stays as a strict pass-through with EVICT_FLAG
// preserved.
//
// Symbol must match the T1 contract literal: ``kmer_count_aggregator``
// (cited in IRON Python lowering, Makefile, and T9 NpuOp class).
void kmer_count_aggregator(uint8_t* __restrict partial_0,
                            uint8_t* __restrict partial_1,
                            uint8_t* __restrict partial_2,
                            uint8_t* __restrict partial_3,
                            uint8_t* __restrict partial_4,
                            uint8_t* __restrict partial_5,
                            uint8_t* __restrict partial_6,
                            uint8_t* __restrict partial_7,
                            uint8_t* __restrict sparse_out,
                            int32_t   n_tiles_active) {
    uint8_t* parts[MAX_TILES] = {
        partial_0, partial_1, partial_2, partial_3,
        partial_4, partial_5, partial_6, partial_7,
    };

    // Per-partial maximum record count (defensive bound; ignores
    // anything beyond the partial's allocation). PARTIAL_OUT_BYTES is
    // not in the constants header (T5 owns the per-chunk packing) so
    // we recompute it from the contract: HASH_BUCKETS_PER_TILE * 12.
    // Subtract 4 bytes for the leading length prefix, then divide by
    // 16-byte record size. = (49152 - 4) / 16 = 3071.
    constexpr int32_t PARTIAL_OUT_BYTES =
        HASH_BUCKETS_PER_TILE * (int32_t)sizeof(CountRecord);
    constexpr int32_t PARTIAL_RECORDS_MAX =
        (PARTIAL_OUT_BYTES - 4) / EMIT_RECORD_BYTES;

    // Clamp n_tiles_active to [0, MAX_TILES].
    int32_t n_active = n_tiles_active;
    if (n_active < 0) n_active = 0;
    if (n_active > MAX_TILES) n_active = MAX_TILES;

    uint32_t n_records = 0;
    uint8_t* dst = sparse_out + 4;  // first 4 bytes are the count prefix

    for (int32_t t = 0; t < n_active; t++) {
        uint8_t* part = parts[t];
        if (part == nullptr) continue;

        uint32_t n_partial = load_le_u32(part);
        if (n_partial > (uint32_t)PARTIAL_RECORDS_MAX) {
            // Defensive: malformed length prefix from the producer.
            // Cap rather than overrun; the host re-aggregation pass
            // will see only the records that actually fit.
            n_partial = (uint32_t)PARTIAL_RECORDS_MAX;
        }

        const uint8_t* rsrc = part + 4;
        for (uint32_t r = 0; r < n_partial; r++) {
            if (n_records >= (uint32_t)EMIT_SLOT_RECORDS) goto done;
            copy_emit_record(dst, rsrc);
            dst += EMIT_RECORD_BYTES;
            rsrc += EMIT_RECORD_BYTES;
            n_records++;
        }
    }

done:
    // Write the count prefix back to sparse_out little-endian.
    store_le_u32(sparse_out, n_records);
}

} // extern "C"
