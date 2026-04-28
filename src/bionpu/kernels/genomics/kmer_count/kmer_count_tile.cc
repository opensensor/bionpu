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

// kmer_count_tile.cc — per-tile k-mer streaming-emit kernel (T5 v0.5 + v1.2 (a) amendments).
//
// Per state/kmer_count_interface_contract.md (T1) v0.5 — symbols, ObjectFifo
// names, and constants pinned by the contract's "v0.5 REDESIGN" section.
//
// v0.5 supersedes the in-tile-counting design. This kernel just streams
// canonical k-mers and emits the ones that fall into the active hash slice.
// ALL counting happens host-side; no count_table, no insert_or_evict, no
// EMIT_FLAG.
//
// v1.2 (a) AMENDMENTS (closes kmer-chr22-canonical0-cap-fire +
//                       kmer-chunk-overlap-double-emit):
//   - In-band header expands from 4 bytes to 8 bytes. Bytes [0..3] still
//     carry the uint32 LE length prefix (actual_bytes); bytes [4..7] now
//     carry an int32 LE owned_start_offset_bases (per-chunk, host-set).
//     Payload starts at byte 8.
//   - Kernel tracks per-base position (`pos`) within the chunk and only
//     emits when k-mer start position (= pos - (K-1)) is in the chunk's
//     OWNED range (>= owned_start_offset_bases). Closes the chunk-overlap
//     double-emit gap without an IRON ABI change.
//   - canonical=0 (all-A k-mer) is redirected from the emit stream into
//     a per-chunk uint32 summary counter, written to the LAST 4 bytes of
//     partial_out. Only pass=0 increments this counter (canonical=0 was
//     already filtered to pass=0 by the slice mask). Closes the
//     chr22-canonical0-cap-fire gap.
//
// Three extern "C" entry points (one per supported k), gated on the
// build-time macro KMER_K_ACTIVE so each TU instantiates exactly one
// template specialisation:
//
//   extern "C" void kmer_count_tile_k{K}(
//       uint8_t*  __restrict packed_in,    // (chunk_bytes + overlap_bytes) bytes
//       uint8_t*  __restrict partial_out,  // PARTIAL_OUT_BYTES_V05_PADDED bytes
//       int32_t   n_input_bytes,           // size of packed_in
//       int32_t   pass_idx,                // 0..n_passes-1
//       int32_t   n_passes_log2,           // log2(N_PASSES) ∈ {0, 2, 4}
//       int32_t   tile_idx,                // 0..n_tiles-1 (per-tile partition)
//       int32_t   n_tiles_log2);           // log2(N_TILES) ∈ {0, 1, 2, 3}
//
// Algorithm (streaming, per chunk + per pass + per tile):
//   0. The chunk's first 8 bytes are an in-band header:
//        bytes [0..3]: uint32 LE actual_bytes (payload size in bytes)
//        bytes [4..7]: int32  LE owned_start_offset_bases (per-chunk;
//                       chunk 0 = 0; chunk i (i>0) = overlap_bases - (k-1)).
//      Payload starts at byte 8 and runs for `actual_bytes` bytes.
//   1. Walk the packed-2-bit stream MSB-first per byte (T1 wire format).
//      Track `pos` = base position within the chunk (0-indexed; 0 = first
//      base of payload).
//   2. Roll fwd/rc registers with per-k mask discipline (CRITICAL for
//      k=31; otherwise high bits leak and canonical = min(fwd, rc)
//      silently corrupts).
//   3. canonical = min(fwd, rc).
//   4. Owned-range gate: drop k-mers whose start position
//      (= pos - (K-1)) is < owned_start_offset_bases. Closes
//      kmer-chunk-overlap-double-emit.
//   5. canonical=0 special-case: increment per-chunk all_a_counter
//      (only in pass=0; canonical=0 is filtered to pass=0 by the slice
//      mask anyway). Closes kmer-chr22-canonical0-cap-fire.
//   6. Two-level hash-slice filter (canonical != 0 only):
//        emit IFF (canonical & ((1 << n_passes_log2) - 1)) == pass_idx
//             AND ((canonical >> n_passes_log2) & ((1 << n_tiles_log2) - 1)) == tile_idx
//      Combined: the LOW (n_passes_log2 + n_tiles_log2) bits of canonical
//      pick BOTH pass and tile. Bits [0 .. n_passes_log2-1] = pass slot;
//      bits [n_passes_log2 .. n_passes_log2 + n_tiles_log2 - 1] = tile slot.
//      Coverage = 100% partition (every canonical lands in exactly one
//      (tile, pass) cell across the full N_TILES × N_PASSES grid).
//   7. Write canonical_u64 to partial_out + 4 + 8*emit_idx; bump
//      emit_idx; cap at MAX_EMIT_IDX_V05.
//   8. After the input loop:
//        - write emit_idx as uint32 LE prefix to partial_out[0..3].
//        - write all_a_counter as uint32 LE to the LAST 4 bytes of
//          partial_out (offset = PARTIAL_OUT_BYTES_V05_PADDED - 4).
//      Host parses
//      [uint32 emit_idx][emit_idx × uint64 canonical] per slot,
//      then reads the trailing uint32 and adds to counts[0].

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

// Pinned-by-contract static asserts (v0.5).
static_assert(PARTIAL_OUT_BYTES_V05_PADDED == 32768,
              "partial_out element must be 32 KiB (T1 v0.5 contract)");
static_assert(MAX_EMIT_IDX_V05 == 4095,
              "max emit_idx caps at 4095 to fit in 32 KiB - 4 prefix");
static_assert(SLICE_HASH_SHIFT == 0,
              "v0.5 pins SLICE_HASH_SHIFT = 0 (low bits of canonical)");

namespace {

// Templated worker — k is a compile-time parameter so the per-k mask
// and the high-bit RC shift fold to constants. The 3 extern "C" entry
// points below instantiate this template for k ∈ {15, 21, 31}.
template <int K, uint64_t MASK>
static inline void kmer_count_tile_impl(uint8_t* __restrict packed_in,
                                        uint8_t* __restrict partial_out,
                                        int32_t n_input_bytes,
                                        int32_t pass_idx,
                                        int32_t n_passes_log2,
                                        int32_t tile_idx,
                                        int32_t n_tiles_log2) {
    // Two-level low-bits partition:
    //   pass_mask = low (n_passes_log2) bits of canonical
    //   tile_mask = next (n_tiles_log2) bits above pass_mask
    // The combined low (n_passes_log2 + n_tiles_log2) bits pick BOTH
    // pass and tile in one read. For (n_passes_log2 == 0,
    // n_tiles_log2 == 0) all bits are zero → single (pass=0, tile=0)
    // case (no partitioning).
    const uint64_t pass_mask = (1ULL << (uint64_t)n_passes_log2) - 1ULL;
    const uint64_t tile_mask = (1ULL << (uint64_t)n_tiles_log2) - 1ULL;
    const uint64_t pass_idx_u64 = (uint64_t)pass_idx;
    const uint64_t tile_idx_u64 = (uint64_t)tile_idx;

    // emit_idx tracks the number of canonicals already written to the
    // partial_out payload (post-prefix). Cap at MAX_EMIT_IDX_V05 to keep
    // total bytes ≤ PARTIAL_OUT_BYTES_V05_PADDED.
    uint32_t emit_idx = 0u;

    // v1.2 (a): per-chunk all-A (canonical=0) summary counter. Only the
    // pass=0 xclbin increments this; the slice mask filters canonical=0
    // to pass=0 by definition. Written to the LAST 4 bytes of partial_out
    // at chunk-end. Closes kmer-chr22-canonical0-cap-fire.
    uint32_t all_a_counter = 0u;

    // Rolling forward/rc registers + fill counter.
    uint64_t fwd = 0ull;
    uint64_t rc  = 0ull;
    int32_t fill = 0;  // bases consumed; only emit when fill >= K.

    // Compile-time RC high-base shift.
    constexpr int32_t RC_HIGH_SHIFT = 2 * (K - 1);

    // ----- In-band header decode. -----
    // v1.2 (a) header (8 bytes total):
    //   bytes [0..3]: uint32 LE actual_payload_bytes
    //   bytes [4..7]: int32  LE owned_start_offset_bases (per-chunk;
    //                  chunk 0 = 0; chunk i (i>0) = overlap_bases - (k-1)).
    // Payload starts at byte 8 and runs for `actual_bytes` bytes.
    // Compile-time `n_input_bytes` is the buffer CAPACITY (chunk size).
    int32_t actual_bytes = (int32_t)packed_in[0]
                         | ((int32_t)packed_in[1] << 8)
                         | ((int32_t)packed_in[2] << 16)
                         | ((int32_t)packed_in[3] << 24);
    // v1.2 (a): owned-range gate. K-mers whose start position is below
    // this offset are duplicates already emitted in the previous chunk's
    // tail (chunk overlap region). Drop them. Closes
    // kmer-chunk-overlap-double-emit.
    int32_t owned_start_offset_bases =
          (int32_t)packed_in[4]
        | ((int32_t)packed_in[5] << 8)
        | ((int32_t)packed_in[6] << 16)
        | ((int32_t)packed_in[7] << 24);
    if (owned_start_offset_bases < 0) owned_start_offset_bases = 0;
    // Defensive clamp: actual_bytes must be ≥ 0 and ≤ n_input_bytes - 8.
    if (actual_bytes < 0) actual_bytes = 0;
    if (actual_bytes > n_input_bytes - 8) actual_bytes = n_input_bytes - 8;

    // Track the position of the newest base (0-indexed; 0 = first base
    // of payload). The k-mer just completed by the current base has its
    // start position at (pos - (K - 1)).
    int32_t pos = 0;

    // ----- Walk the packed-2-bit stream MSB-first per byte. -----
    // Byte 0 of payload (= packed_in[8]) carries bases 0..3 with base 0
    // in bits [7:6] (MSB-first).
    for (int32_t i = 0; i < actual_bytes; ++i) {
        uint8_t byte = packed_in[8 + i];
        for (int32_t shift = 6; shift >= 0; shift -= 2) {
            uint8_t base = (uint8_t)((byte >> shift) & 0x3);
            uint8_t comp = (uint8_t)(base ^ 0x3);

            // CRITICAL: per-k mask applied to BOTH forward and rc on
            // every update. For k=31 the MASK is (1ULL<<62)-1; without
            // it, fwd's bit 62 leaks past bit 63, AND the rc shift
            // pushes the previously-OR'd high base off into bits >= 62
            // where it is NOT cleared by the >>2.
            fwd = ((fwd << 2) | (uint64_t)base) & MASK;
            rc  = ((rc >> 2) | ((uint64_t)comp << RC_HIGH_SHIFT)) & MASK;

            if (fill < K) {
                fill += 1;
                if (fill < K) {
                    pos += 1;
                    continue;
                }
                // fall through and emit; we just reached the first full k-mer.
            }

            // v1.2 (a) owned-range gate: drop k-mers whose start position
            // is in the overlap region (already emitted by prior chunk).
            // K-mer start position = pos - (K - 1).
            int32_t kmer_start = pos - (K - 1);
            if (kmer_start < owned_start_offset_bases) {
                pos += 1;
                continue;
            }

            // canonical = min(forward, rc).
            uint64_t canonical = (fwd < rc) ? fwd : rc;

            // v1.2 (a) canonical=0 special-case: redirect all-A k-mers
            // to a per-chunk summary counter. The slice mask filters
            // canonical=0 to (pass=0, tile=0) by definition (all bits
            // are zero). All N_TILES tiles see the same input chunk
            // (broadcast seq_in), so without the tile_idx==0 gate every
            // tile would increment its own counter and the host would
            // sum 4× the truth. Gate strictly by (pass_idx==0 &&
            // tile_idx==0). Closes kmer-chr22-canonical0-cap-fire.
            if (canonical == 0ull) {
                if (pass_idx == 0 && tile_idx == 0) {
                    all_a_counter += 1u;
                }
                pos += 1;
                continue;
            }

            // Two-level low-bits partition. Pass selector first (low
            // n_passes_log2 bits), tile selector next (n_tiles_log2 bits
            // above the pass field).
            uint64_t pass_slot = canonical & pass_mask;
            if (pass_slot != pass_idx_u64) {
                pos += 1;
                continue;  // wrong pass slice
            }
            uint64_t tile_slot =
                (canonical >> (uint64_t)n_passes_log2) & tile_mask;
            if (tile_slot != tile_idx_u64) {
                pos += 1;
                continue;  // wrong tile slice
            }

            // Emit canonical to partial_out (cap at MAX_EMIT_IDX_V05).
            if (emit_idx < (uint32_t)MAX_EMIT_IDX_V05) {
                // Layout: partial_out[0..3] = emit_idx prefix (written
                // at end of loop); partial_out[4..] = uint64 canonicals.
                uint8_t* dst = partial_out + 4 + (size_t)emit_idx * 8u;
                // Byte-wise LE store to avoid alignment assumptions.
                dst[0] = (uint8_t)(canonical >> 0);
                dst[1] = (uint8_t)(canonical >> 8);
                dst[2] = (uint8_t)(canonical >> 16);
                dst[3] = (uint8_t)(canonical >> 24);
                dst[4] = (uint8_t)(canonical >> 32);
                dst[5] = (uint8_t)(canonical >> 40);
                dst[6] = (uint8_t)(canonical >> 48);
                dst[7] = (uint8_t)(canonical >> 56);
                emit_idx += 1u;
            }
            // else: emit slot full at this pass — drop further canonicals.
            // Host detects by comparing emit_idx to MAX_EMIT_IDX_V05 and
            // re-dispatches with smaller chunks or larger N_PASSES.
            pos += 1;
        }
    }

    // Write the uint32 LE record-count prefix to the head of partial_out.
    partial_out[0] = (uint8_t)(emit_idx & 0xff);
    partial_out[1] = (uint8_t)((emit_idx >> 8) & 0xff);
    partial_out[2] = (uint8_t)((emit_idx >> 16) & 0xff);
    partial_out[3] = (uint8_t)((emit_idx >> 24) & 0xff);

    // v1.2 (a): write all_a_counter (canonical=0 summary) at the END of
    // partial_out. Host reads this offset after parsing the emit stream
    // and adds to counts[0]. Closes kmer-chr22-canonical0-cap-fire.
    // Layout: partial_out[PARTIAL_OUT_BYTES_V05_PADDED - 4 .. -1] = uint32 LE.
    uint8_t* tail = partial_out + (PARTIAL_OUT_BYTES_V05_PADDED - 4);
    tail[0] = (uint8_t)(all_a_counter & 0xff);
    tail[1] = (uint8_t)((all_a_counter >> 8) & 0xff);
    tail[2] = (uint8_t)((all_a_counter >> 16) & 0xff);
    tail[3] = (uint8_t)((all_a_counter >> 24) & 0xff);
}

}  // anonymous namespace

// =====================================================================
// Per-build single-k entry point.
//
// Each xclbin is single-k: the Makefile builds 12 (or more) artifacts =
// 3 k × N n_tiles × N n_passes, and the IRON-Python references ONE
// kmer_count_tile_k{K} symbol per build. Gate the entry points on the
// build-time macro KMER_K_ACTIVE so each TU instantiates exactly one
// specialisation (avoids redundant code bloat / .text overhead).
// =====================================================================

#ifndef KMER_K_ACTIVE
#define KMER_K_ACTIVE 21  // default for standalone compile probes
#endif
static_assert(KMER_K_ACTIVE == 15 || KMER_K_ACTIVE == 21 || KMER_K_ACTIVE == 31,
              "KMER_K_ACTIVE must be 15, 21, or 31");

extern "C" {

#if KMER_K_ACTIVE == 15
// ============================================================================
// Per-tile k-mer streaming-emit — k=15 (30-bit canonical, KMER_MASK_K15)
// ============================================================================
void kmer_count_tile_k15(uint8_t* __restrict packed_in,
                         uint8_t* __restrict partial_out,
                         int32_t n_input_bytes,
                         int32_t pass_idx,
                         int32_t n_passes_log2,
                         int32_t tile_idx,
                         int32_t n_tiles_log2) {
    kmer_count_tile_impl<15, KMER_MASK_K15>(
        packed_in, partial_out, n_input_bytes,
        pass_idx, n_passes_log2, tile_idx, n_tiles_log2);
}
#endif

#if KMER_K_ACTIVE == 21
// ============================================================================
// Per-tile k-mer streaming-emit — k=21 (42-bit canonical, KMER_MASK_K21)
// ============================================================================
void kmer_count_tile_k21(uint8_t* __restrict packed_in,
                         uint8_t* __restrict partial_out,
                         int32_t n_input_bytes,
                         int32_t pass_idx,
                         int32_t n_passes_log2,
                         int32_t tile_idx,
                         int32_t n_tiles_log2) {
    kmer_count_tile_impl<21, KMER_MASK_K21>(
        packed_in, partial_out, n_input_bytes,
        pass_idx, n_passes_log2, tile_idx, n_tiles_log2);
}
#endif

#if KMER_K_ACTIVE == 31
// ============================================================================
// Per-tile k-mer streaming-emit — k=31 (62-bit canonical, KMER_MASK_K31)
// ============================================================================
// At k=31 the per-update mask is the load-bearing correctness invariant:
// fwd uses bits [0..61], rc shifts a freshly XOR'd base into bit 60..61
// (RC_HIGH_SHIFT = 60), and without the post-shift mask, prior-iteration
// bits leak past bit 61 and corrupt min(fwd, rc) silently.
void kmer_count_tile_k31(uint8_t* __restrict packed_in,
                         uint8_t* __restrict partial_out,
                         int32_t n_input_bytes,
                         int32_t pass_idx,
                         int32_t n_passes_log2,
                         int32_t tile_idx,
                         int32_t n_tiles_log2) {
    kmer_count_tile_impl<31, KMER_MASK_K31>(
        packed_in, partial_out, n_input_bytes,
        pass_idx, n_passes_log2, tile_idx, n_tiles_log2);
}
#endif

}  // extern "C"
