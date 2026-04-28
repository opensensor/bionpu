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

// kmer_count_tile.cc — per-tile k-mer streaming-emit kernel (T5 v0.5).
//
// Per state/kmer_count_interface_contract.md (T1) v0.5 — symbols, ObjectFifo
// names, and constants pinned by the contract's "v0.5 REDESIGN" section.
//
// v0.5 supersedes the in-tile-counting design. This kernel just streams
// canonical k-mers and emits the ones that fall into the active hash slice.
// ALL counting happens host-side; no count_table, no insert_or_evict, no
// EMIT_FLAG.
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
//   0. The chunk's first 4 bytes are a uint32-LE length prefix giving
//      the actual number of payload bytes (≤ chunk_size - 4). The
//      kernel reads this length, then walks bytes 4..4+actual_bytes.
//      This works around IRON's lack of runtime-scalar args to per-tile
//      kernels: `n_input_bytes` is the COMPILE-TIME chunk capacity, but
//      the actual valid payload size is encoded in-band so the host
//      can vary it per dispatch.
//   1. Walk the packed-2-bit stream MSB-first per byte (T1 wire format).
//   2. Roll fwd/rc registers with per-k mask discipline (CRITICAL for
//      k=31; otherwise high bits leak and canonical = min(fwd, rc)
//      silently corrupts).
//   3. canonical = min(fwd, rc).
//   4. Two-level hash-slice filter:
//        emit IFF (canonical & ((1 << n_passes_log2) - 1)) == pass_idx
//             AND ((canonical >> n_passes_log2) & ((1 << n_tiles_log2) - 1)) == tile_idx
//      Combined: the LOW (n_passes_log2 + n_tiles_log2) bits of canonical
//      pick BOTH pass and tile. Bits [0 .. n_passes_log2-1] = pass slot;
//      bits [n_passes_log2 .. n_passes_log2 + n_tiles_log2 - 1] = tile slot.
//      Coverage = 100% partition (every canonical lands in exactly one
//      (tile, pass) cell across the full N_TILES × N_PASSES grid).
//   5. Write canonical_u64 to partial_out + 4 + 8*emit_idx; bump
//      emit_idx; cap at MAX_EMIT_IDX_V05.
//   6. After the input loop, write emit_idx as uint32 LE prefix to
//      partial_out[0..3]. Host parses
//      [uint32 emit_idx][emit_idx × uint64 canonical] per slot.

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

    // Rolling forward/rc registers + fill counter.
    uint64_t fwd = 0ull;
    uint64_t rc  = 0ull;
    int32_t fill = 0;  // bases consumed; only emit when fill >= K.

    // Compile-time RC high-base shift.
    constexpr int32_t RC_HIGH_SHIFT = 2 * (K - 1);

    // ----- In-band length prefix decode. -----
    // First 4 bytes of packed_in = uint32 LE actual_payload_bytes.
    // Payload starts at byte 4 and runs for `actual_bytes` bytes.
    // Compile-time `n_input_bytes` is the buffer CAPACITY (chunk size).
    int32_t actual_bytes = (int32_t)packed_in[0]
                         | ((int32_t)packed_in[1] << 8)
                         | ((int32_t)packed_in[2] << 16)
                         | ((int32_t)packed_in[3] << 24);
    // Defensive clamp: actual_bytes must be ≥ 0 and ≤ n_input_bytes - 4.
    if (actual_bytes < 0) actual_bytes = 0;
    if (actual_bytes > n_input_bytes - 4) actual_bytes = n_input_bytes - 4;

    // ----- Walk the packed-2-bit stream MSB-first per byte. -----
    // Byte 0 of payload (= packed_in[4]) carries bases 0..3 with base 0
    // in bits [7:6] (MSB-first).
    for (int32_t i = 0; i < actual_bytes; ++i) {
        uint8_t byte = packed_in[4 + i];
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
                if (fill < K) continue;
                // fall through and emit; we just reached the first full k-mer.
            }

            // canonical = min(forward, rc).
            uint64_t canonical = (fwd < rc) ? fwd : rc;

            // Two-level low-bits partition. Pass selector first (low
            // n_passes_log2 bits), tile selector next (n_tiles_log2 bits
            // above the pass field).
            uint64_t pass_slot = canonical & pass_mask;
            if (pass_slot != pass_idx_u64) {
                continue;  // wrong pass slice
            }
            uint64_t tile_slot =
                (canonical >> (uint64_t)n_passes_log2) & tile_mask;
            if (tile_slot != tile_idx_u64) {
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
        }
    }

    // Write the uint32 LE record-count prefix to the head of partial_out.
    partial_out[0] = (uint8_t)(emit_idx & 0xff);
    partial_out[1] = (uint8_t)((emit_idx >> 8) & 0xff);
    partial_out[2] = (uint8_t)((emit_idx >> 16) & 0xff);
    partial_out[3] = (uint8_t)((emit_idx >> 24) & 0xff);
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
