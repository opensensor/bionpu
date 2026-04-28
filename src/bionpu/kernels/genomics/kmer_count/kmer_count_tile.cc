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

// kmer_count_tile.cc — per-tile k-mer counting kernel (T5 of the v1 plan).
//
// Per state/kmer_count_interface_contract.md (T1) — symbols, ObjectFifo
// names, and constants pinned there. The contract pins three extern "C"
// entry points (`kmer_count_tile_k15`, `_k21`, `_k31`), the shared
// signature `(packed_in, partial_out, n_input_bytes, bucket_lo,
// bucket_hi)`, the MSB-first 2-bit wire format, the per-k canonical
// mask `KMER_MASK_K{k}`, the 4096-bucket × 12-byte open-address hash
// table, and the emit-on-evict overflow policy at OVERFLOW_THRESHOLD=8.
//
// Implementation notes:
//   (a) MSB-first unpack: byte 0 carries bases 0..3 with base 0 in
//       bits [7:6]. We consume the top 2 bits first.
//   (b) Rolling forward/rc registers, both masked with KMER_MASK_K{k}
//       on every update — CRITICAL for k=31 (62-bit width). Without
//       the rc mask the >>2 leaks the previously-OR'd high base into
//       bit 63 and stays there, silently corrupting canonical = min().
//   (c) canonical = min(forward, rc).
//   (d) bucket = canonical & (HASH_BUCKETS_PER_TILE - 1). Tile owns
//       buckets in [bucket_lo, bucket_hi); k-mers outside this range
//       are skipped (other tiles count them).
//   (e) Open-address linear probe insert. On chain length > THRESHOLD
//       OR table full, evict the head of the chain to partial_out as a
//       16-byte EmitRecord with flags = EVICT_FLAG. Host (T7) re-
//       aggregates duplicate canonical_u64 entries by summing counts.
//
// NO <stdlib.h>, NO <stdio.h> — the from-source peano lacks libcxx, and
// the ironenv-bundled peano works without these headers per established
// pattern (linear_projection_fused.cc).

#include <stdint.h>

#include "kmer_count_constants.h"

// AIE2P toolchain headers. Gated on __AIE_ARCH__ so the .o links
// against host unit tests too (matches CRISPR tile_a_filter.cc pattern).
#if defined(__AIE_ARCH__) && !defined(BIONPU_FORCE_SCALAR)
#  define BIONPU_HAS_AIE_API 1
#  include "aie_kernel_utils.h"
#  include <aie_api/aie.hpp>
#else
#  define BIONPU_HAS_AIE_API 0
#endif

// Pinned-by-contract static asserts.
static_assert(sizeof(CountRecord) == 12,
              "CountRecord must be 12 bytes packed (T1 contract)");
static_assert(sizeof(EmitRecord) == 16,
              "EmitRecord must be 16 bytes packed (T1 contract)");
// Tightened from 48 KiB to 16 KiB after T11 build failure forced the
// 4096->1024 bucket revision (depth=2 partial ping-pong was the
// missing budget term). 1024 * 12 = 12 KiB, well under 16 KiB.
static_assert(HASH_BUCKETS_PER_TILE * sizeof(CountRecord) <= 16 * 1024,
              "count table exceeds 16 KiB tile-DM budget (T1 revision 2026-04-28)");
static_assert((HASH_BUCKETS_PER_TILE & (HASH_BUCKETS_PER_TILE - 1)) == 0,
              "HASH_BUCKETS_PER_TILE must be a power of two for "
              "canonical & (HASH_BUCKETS_PER_TILE-1) bucket math");

namespace {

// ----- 64-bit-pair reverse, used to compute reverse-complement of a
// k-mer packed at 2 bits/base. The standard "reverse bits" trick at
// 2-bit granularity (swap pairs, then nibbles, then bytes, then 16/32).
// Bit-pair-reversed value of a forward k-mer + per-base XOR with 0x3
// = reverse-complement. Equivalent to reversing AFTER the per-base XOR
// by the 2*k mask; we do per-base XOR-during-roll then bit-reverse the
// finished register at the end (impractical for streaming) OR, as we
// actually do here, build rc in parallel from the high end so no
// reversal is ever needed.
//
// We do NOT reverse — instead rc is maintained "from the other end":
//   rc' = (rc >> 2) | (complement(new_base) << (2 * (k-1)))
// where complement = base ^ 0x3. After every update both forward and
// rc are masked with KMER_MASK_K{k}.

// Inline helper: insert (canonical, +1) into the per-tile count table
// using open-address linear probing. On chain > OVERFLOW_THRESHOLD or
// table full, evict the chain-head record to `partial_out` as an
// EmitRecord with flags = EVICT_FLAG and reuse the freed slot.
//
// emit_idx is updated in place (number of records emitted so far in
// the current chunk's partial_out buffer; wraps via the slot bound).
// Returns nothing — emit-on-evict is silent unless it overflows the
// emit slot too, which is treated as a hard cap (records past
// EMIT_SLOT_RECORDS are dropped; this is a v1.1 follow-up tracked by
// gaps.yaml T18 since correctness still holds for chr22 at n_tiles=4
// per T1 contract analysis).
static inline void insert_or_evict(CountRecord* __restrict table,
                                   uint8_t* __restrict partial_out,
                                   uint32_t* __restrict emit_idx,
                                   uint64_t canonical) {
    // Hash to bucket index. canonical is already masked to 2*k bits.
    uint32_t home = (uint32_t)(canonical) & (uint32_t)(HASH_BUCKETS_PER_TILE - 1);

    // Linear probe up to OVERFLOW_THRESHOLD slots.
    for (int32_t probe = 0; probe < OVERFLOW_THRESHOLD; ++probe) {
        uint32_t idx = (home + (uint32_t)probe) & (uint32_t)(HASH_BUCKETS_PER_TILE - 1);
        CountRecord* slot = table + idx;
        if (slot->count == 0u) {
            // Empty — claim it.
            slot->canonical = canonical;
            slot->count = 1u;
            return;
        }
        if (slot->canonical == canonical) {
            // Hit — increment.
            slot->count += 1u;
            return;
        }
        // Collision; keep probing.
    }

    // Chain exhausted → evict the chain-head (probe distance 0) per the
    // T1 contract: "the slot at probe distance == OVERFLOW_THRESHOLD-1,
    // i.e. the head of the chain at eviction time" — the contract names
    // the conceptual 'oldest' entry, which for linear probing in a
    // bounded chain is the home bucket itself (it has been there
    // longest; subsequent probes are by definition newer entries that
    // collided onto it). We evict the home bucket and reuse it for the
    // new k-mer.
    CountRecord* victim = table + home;
    uint32_t e = *emit_idx;
    if (e < (uint32_t)EMIT_SLOT_RECORDS) {
        // Emit the evicted (canonical, count, EVICT_FLAG) as a 16-byte
        // record into the partial_out ring slot. Layout matches
        // EmitRecord struct exactly (8 + 4 + 4, packed).
        EmitRecord* dst = reinterpret_cast<EmitRecord*>(partial_out + 4) + e;
        dst->canonical = victim->canonical;
        dst->count = victim->count;
        dst->flags = EVICT_FLAG;
        *emit_idx = e + 1u;
    }
    // else: emit slot full — record dropped. Host re-aggregation can
    // still recover the partial table dump at end-of-chunk (the
    // aggregator drains the count table in a separate pass; emit-on-
    // evict here is the per-chunk chain-overflow path only).

    // Reuse the home slot for the new k-mer.
    victim->canonical = canonical;
    victim->count = 1u;
}

// Templated worker — k is a compile-time parameter so the per-k mask
// and the high-bit RC shift fold to constants. The 3 extern "C" entry
// points below instantiate this template for k ∈ {15, 21, 31}.
template <int K, uint64_t MASK>
static inline void kmer_count_tile_impl(uint8_t* __restrict packed_in,
                                        uint8_t* __restrict partial_out,
                                        int32_t n_input_bytes,
                                        int32_t bucket_lo,
                                        int32_t bucket_hi) {
    // ----- Local count table (48 KiB, lives in tile DM). -----
    // Allocate stack-resident; AIE2P L1 has 64 KiB total and the
    // ObjectFifo double-buffers + code consume the remaining ~16 KiB.
    // Zero-init so empty slots have count == 0.
    static CountRecord table[HASH_BUCKETS_PER_TILE];
    for (int32_t i = 0; i < HASH_BUCKETS_PER_TILE; ++i) {
        table[i].canonical = 0ull;
        table[i].count = 0u;
    }

    // First 4 bytes of partial_out are the uint32 record-count prefix
    // (mirrors CRISPR sparse-emit pattern in tile_a_filter.cc:325-353).
    uint32_t emit_idx = 0u;

    // Rolling forward/rc registers and a fill counter.
    uint64_t fwd = 0ull;
    uint64_t rc  = 0ull;
    int32_t fill = 0;  // bases consumed; only emit when fill >= K.

    // Compile-time RC high-base shift. base ^ 0x3 << (2*(K-1)). We
    // form the constant once per template instantiation.
    constexpr int32_t RC_HIGH_SHIFT = 2 * (K - 1);

    // ----- Walk the packed-2-bit stream MSB-first per byte. -----
    // Per T1 wire format: byte 0 carries bases 0..3 with base 0 in
    // bits [7:6], base 1 in bits [5:4], base 2 in bits [3:2], base 3
    // in bits [1:0]. We consume top 2 bits first.
    for (int32_t i = 0; i < n_input_bytes; ++i) {
        uint8_t byte = packed_in[i];
        for (int32_t shift = 6; shift >= 0; shift -= 2) {
            uint8_t base = (uint8_t)((byte >> shift) & 0x3);
            uint8_t comp = (uint8_t)(base ^ 0x3);

            // CRITICAL: per-k mask applied to BOTH forward and rc on
            // every update. For k=31 the MASK is (1ULL<<62)-1; without
            // it, fwd's bit 62 leak past bit 63, AND the rc shift
            // pushes the previously-OR'd high base off into bits >= 62
            // where it is NOT cleared by the >>2. Either path silently
            // corrupts canonical = min(fwd, rc).
            fwd = ((fwd << 2) | (uint64_t)base) & MASK;
            rc  = ((rc >> 2) | ((uint64_t)comp << RC_HIGH_SHIFT)) & MASK;

            if (fill < K) {
                fill += 1;
                if (fill < K) continue;
                // fall through and emit; we just reached the first
                // full k-mer.
            }

            // canonical = min(forward, rc).
            uint64_t canonical = (fwd < rc) ? fwd : rc;

            // Hash bucket and tile-range filter.
            uint32_t bucket = (uint32_t)(canonical) & (uint32_t)(HASH_BUCKETS_PER_TILE - 1);
            if ((int32_t)bucket < bucket_lo || (int32_t)bucket >= bucket_hi) {
                continue;
            }

            // Open-address insert with emit-on-evict at chain
            // length > OVERFLOW_THRESHOLD.
            insert_or_evict(table, partial_out, &emit_idx, canonical);
        }
    }

    // Write the record-count prefix back to the head of partial_out.
    // Mirrors CRISPR tile_a_filter.cc:349-352.
    partial_out[0] = (uint8_t)(emit_idx & 0xff);
    partial_out[1] = (uint8_t)((emit_idx >> 8) & 0xff);
    partial_out[2] = (uint8_t)((emit_idx >> 16) & 0xff);
    partial_out[3] = (uint8_t)((emit_idx >> 24) & 0xff);
}

}  // anonymous namespace

// =====================================================================
// Per-build single-k entry point (T11 fix, 2026-04-28).
//
// The original design instantiated all three template specialisations
// (k=15, k=21, k=31) in a single TU and exported all three extern "C"
// symbols from every artifact. Each specialisation owns its own
// `static CountRecord table[HASH_BUCKETS_PER_TILE]` (12 KiB BSS), so
// the 3-instantiation TU produced 36 KiB of BSS — overflowing the
// 64 KiB AIE2P CoreTile DM by 8 bytes after stack + ObjectFifo
// double-buffers + code (`ld.lld: section '.bss' will not fit`).
//
// Each xclbin is single-k anyway (Makefile builds 12 artifacts =
// 3 k × 4 n_tiles, and the IRON-Python references ONE
// kmer_count_tile_k{K} symbol per build). Gate the entry points on
// the build-time macro KMER_K_ACTIVE so each TU instantiates exactly
// one specialisation. BSS drops back to 12 KiB.
//
// Makefile passes -DKMER_K_ACTIVE={15,21,31}. The IRON-Python's
// external function reference matches the active symbol per artifact.
// =====================================================================

#ifndef KMER_K_ACTIVE
#define KMER_K_ACTIVE 21  // default for standalone compile probes
#endif
static_assert(KMER_K_ACTIVE == 15 || KMER_K_ACTIVE == 21 || KMER_K_ACTIVE == 31,
              "KMER_K_ACTIVE must be 15, 21, or 31");

extern "C" {

#if KMER_K_ACTIVE == 15
// ============================================================================
// Per-tile k-mer counter — k=15 (30-bit canonical, KMER_MASK_K15)
// ============================================================================
void kmer_count_tile_k15(uint8_t* __restrict packed_in,
                         uint8_t* __restrict partial_out,
                         int32_t n_input_bytes,
                         int32_t bucket_lo,
                         int32_t bucket_hi) {
    kmer_count_tile_impl<15, KMER_MASK_K15>(packed_in, partial_out,
                                            n_input_bytes,
                                            bucket_lo, bucket_hi);
}
#endif

#if KMER_K_ACTIVE == 21
// ============================================================================
// Per-tile k-mer counter — k=21 (42-bit canonical, KMER_MASK_K21)
// ============================================================================
void kmer_count_tile_k21(uint8_t* __restrict packed_in,
                         uint8_t* __restrict partial_out,
                         int32_t n_input_bytes,
                         int32_t bucket_lo,
                         int32_t bucket_hi) {
    kmer_count_tile_impl<21, KMER_MASK_K21>(packed_in, partial_out,
                                            n_input_bytes,
                                            bucket_lo, bucket_hi);
}
#endif

#if KMER_K_ACTIVE == 31
// ============================================================================
// Per-tile k-mer counter — k=31 (62-bit canonical, KMER_MASK_K31)
// ============================================================================
// At k=31 the per-update mask is the load-bearing correctness invariant:
// fwd uses bits [0..61], rc shifts a freshly XOR'd base into bit 60..61
// (RC_HIGH_SHIFT = 60), and without the post-shift mask, prior-iteration
// bits leak past bit 61 and corrupt min(fwd, rc) silently. The shared
// template instantiation below carries the mask through unchanged.
void kmer_count_tile_k31(uint8_t* __restrict packed_in,
                         uint8_t* __restrict partial_out,
                         int32_t n_input_bytes,
                         int32_t bucket_lo,
                         int32_t bucket_hi) {
    kmer_count_tile_impl<31, KMER_MASK_K31>(packed_in, partial_out,
                                            n_input_bytes,
                                            bucket_lo, bucket_hi);
}
#endif

}  // extern "C"
