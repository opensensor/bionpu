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
// minimizer_tile.cc — per-tile sliding-window (w, k) minimizer kernel
//                     (v1: multi-pass hash-slice partitioning).
//
// Two extern "C" entry points (gated on -DKW_ACTIVE=KW{15_10,21_11}):
//
//   extern "C" void minimizer_tile_k{K}_w{W}(
//       uint8_t*  __restrict packed_in,    // chunk_bytes + overlap_bytes
//       uint8_t*  __restrict partial_out,  // MZ_PARTIAL_OUT_BYTES_PADDED
//       int32_t   n_input_bytes,           // BO capacity (chunk size + 8)
//       int32_t   pass_idx,                // 0..n_passes-1 (v1)
//       int32_t   n_passes_log2,           // log2(N_PASSES) (v1)
//       int32_t   tile_idx,                // unused for v1 (broadcast)
//       int32_t   n_tiles_log2);           // unused for v1
//
// Note 1: the IRON Python broadcasts seq_in to all N_TILES tiles (same
// fan-out topology as kmer_count). For v1 we still do NOT partition the
// minimizer output across tiles — every tile sees the same chunk and
// emits the SAME hash-slice-filtered minimizers. The host runner reads
// tile_0's slot only and ignores the others. The N_TILES fan-out exists
// so that the IRON memtile-join topology matches kmer_count's wide
// dispatch shape (and so that the same artifact dir naming convention
// works).
//
// v1 multi-pass: each pass emits only minimizers whose
// (position-within-chunk's low n_passes_log2 bits) equal pass_idx. The
// window-min logic is UNCHANGED across passes; only the emit decision
// filters by slice.
//
// Why slice on hashed POSITION rather than CANONICAL: chr22 contains
// long homopolymer-A regions in centromere/telomere that produce a high
// density of canonical=0 minimizers at distinct positions (~1 emit per
// W bases in steady state). Slicing on canonical's low bits would put
// every canonical=0 minimizer into pass 0 (since all bits of canonical
// are zero), blowing past MZ_MAX_EMIT_IDX in chunk 0. We slice on
// Fibonacci-hashed position so emits at any stride distribute uniformly
// across passes regardless of canonical value:
//
//   slice = (position * GOLDEN) >> (32 - n_passes_log2)
//
// where GOLDEN = 0x9E3779B9 is the 32-bit Fibonacci-hashing constant
// (closest integer to 2^32 * (sqrt(5)-1)/2). For n_passes_log2==0 we
// short-circuit to slice=0 (single-pass back-compat; avoids the
// shift-by-32 corner case).
//
// Algorithm — mirrors data/minimizer_oracle.py exactly except for the
// per-pass slice filter at emit time:
//
//   For each new ACGT base entering at index i:
//     1. fwd = ((fwd << 2) | base) & MASK
//     2. rc  = (rc >> 2) | ((base ^ 3) << (2*(k-1)))
//     3. valid_run += 1
//     4. If valid_run < k:  no canonical yet; continue.
//     5. canonical = min(fwd, rc); kmer_start = i - (k - 1).
//     6. slot = n_pushed % w; ring[slot] = (canonical, kmer_start).
//        n_pushed += 1.
//     7. If n_pushed < w: window not full; no emit.
//     8. If n_pushed == w: scan ring for oldest-on-tie min; emit if
//        owned (kmer_start >= owned_start_offset_bases on the SCAN
//        WINNER's position) AND in-slice.
//     9. Else (window has slid):
//        a. If cur_min_slot == slot: previous min was overwritten
//           → re-scan; emit if owned + in-slice.
//        b. Else if canonical < cur_min_canonical: strict improvement
//           → cur_min = (canonical, kmer_start, slot); emit if
//           owned + in-slice.
//        c. Else: no emit.
//
// In-slice predicate (per pass):
//   if (n_passes_log2 == 0) in_slice = true (single-pass back-compat)
//   else
//     hash = (uint32_t)position * 0x9E3779B9u
//     slice = hash >> (32 - n_passes_log2)
//     in_slice = (slice == pass_idx)
//
// Fibonacci hashing on chunk-local position spreads all stride patterns
// (including the W-stride pattern characteristic of homopolymer regions)
// uniformly across passes.
//
// Non-ACGT base resets the rolling state AND the ring buffer.
//
// Output record layout (16 bytes/record):
//   bytes [0..7]:   uint64 LE canonical
//   bytes [8..11]:  uint32 LE position (0-indexed kmer_start within
//                   chunk payload; host translates to global by adding
//                   chunk's source offset)
//   bytes [12..15]: uint32 _pad (zero)

#include <stdint.h>

#include "minimizer_constants.h"

#if defined(__AIE_ARCH__) && !defined(BIONPU_FORCE_SCALAR)
#  define BIONPU_HAS_AIE_API 1
#  include "aie_kernel_utils.h"
#  include <aie_api/aie.hpp>
#else
#  define BIONPU_HAS_AIE_API 0
#endif

static_assert(MZ_PARTIAL_OUT_BYTES_PADDED == 32768,
              "partial_out element must be 32 KiB");
static_assert(MZ_RECORD_BYTES == 16, "Record = uint64 + uint32 + uint32 = 16 B");
static_assert(MZ_MAX_EMIT_IDX == 2046,
              "max emit_idx caps at 2046 to fit in 32 KiB - 4 prefix");
static_assert(MZ_HEADER_BYTES == 8, "in-band header pinned at 8 bytes");

namespace {

template <int K, int W, uint64_t MASK>
static inline void minimizer_tile_impl(uint8_t* __restrict packed_in,
                                       uint8_t* __restrict partial_out,
                                       int32_t n_input_bytes,
                                       int32_t pass_idx,
                                       int32_t n_passes_log2,
                                       int32_t /*tile_idx*/,
                                       int32_t /*n_tiles_log2*/) {
    // Static window ring buffer in the tile DM.
    // Per-slot entry: (canonical_u64, kmer_start_pos_i32). w * 16 bytes
    // = 160 B at w=10, 176 B at w=11. Easy fit.
    uint64_t ring_canonical[W];
    int32_t  ring_position[W];
    bool     ring_valid[W];

    // We INITIALISE per-call (chunks are independent because the IRON
    // Python's per-tile call site re-enters the kernel for each chunk).
    for (int s = 0; s < W; ++s) {
        ring_canonical[s] = 0ull;
        ring_position[s]  = 0;
        ring_valid[s]     = false;
    }

    uint32_t emit_idx = 0u;

    // Rolling fwd / rc registers + fill counter.
    uint64_t fwd = 0ull;
    uint64_t rc  = 0ull;
    int32_t  valid_run = 0;
    int32_t  n_pushed = 0;

    // Current minimum tracking.
    uint64_t cur_min_canonical = 0ull;
    int32_t  cur_min_position  = 0;
    int32_t  cur_min_slot      = -1;
    bool     have_min          = false;

    constexpr int32_t RC_HIGH_SHIFT = 2 * (K - 1);

    // v1: Fibonacci-hash slice filter on chunk-local position. For
    // n_passes_log2 == 0 we short-circuit to in_slice=true (single-pass
    // back-compat). Otherwise compute slice = (pos * GOLDEN) >>
    // (32 - n_passes_log2). pass_idx must equal slice for emit.
    constexpr uint32_t FIB_GOLDEN = 0x9E3779B9u;
    const int32_t fib_shift = 32 - n_passes_log2;
    const uint32_t pass_idx_u32 = (uint32_t)pass_idx;

    // ----- In-band header decode. -----
    int32_t actual_bytes = (int32_t)packed_in[0]
                         | ((int32_t)packed_in[1] << 8)
                         | ((int32_t)packed_in[2] << 16)
                         | ((int32_t)packed_in[3] << 24);
    int32_t owned_start_offset_bases =
          (int32_t)packed_in[4]
        | ((int32_t)packed_in[5] << 8)
        | ((int32_t)packed_in[6] << 16)
        | ((int32_t)packed_in[7] << 24);
    if (owned_start_offset_bases < 0) owned_start_offset_bases = 0;
    if (actual_bytes < 0) actual_bytes = 0;
    if (actual_bytes > n_input_bytes - MZ_HEADER_BYTES) {
        actual_bytes = n_input_bytes - MZ_HEADER_BYTES;
    }

    // Track the position of the newest base (0-indexed; 0 = first base
    // of payload). The k-mer just completed by the current base has its
    // start at (pos - (K - 1)).
    int32_t pos = 0;

    // ----- Walk the packed-2-bit stream MSB-first per byte. -----
    for (int32_t i = 0; i < actual_bytes; ++i) {
        uint8_t byte = packed_in[MZ_HEADER_BYTES + i];
        for (int32_t shift = 6; shift >= 0; shift -= 2) {
            uint8_t base = (uint8_t)((byte >> shift) & 0x3);
            uint8_t comp = (uint8_t)(base ^ 0x3);

            // Roll fwd / rc with per-K mask discipline.
            fwd = ((fwd << 2) | (uint64_t)base) & MASK;
            rc  = ((rc >> 2) | ((uint64_t)comp << RC_HIGH_SHIFT)) & MASK;

            if (valid_run < K) {
                valid_run += 1;
                if (valid_run < K) {
                    pos += 1;
                    continue;
                }
                // Just reached the first full k-mer; fall through.
            }

            uint64_t canonical = (fwd < rc) ? fwd : rc;
            int32_t  kmer_start = pos - (K - 1);

            // Push into the ring.
            int32_t slot = n_pushed % W;
            ring_canonical[slot] = canonical;
            ring_position[slot]  = kmer_start;
            ring_valid[slot]     = true;
            n_pushed += 1;

            if (n_pushed < W) {
                pos += 1;
                continue;  // window not full
            }

            // Decide whether to (re)compute the min and emit.
            bool need_emit = false;
            uint64_t new_min_canonical = cur_min_canonical;
            int32_t  new_min_position  = cur_min_position;
            int32_t  new_min_slot      = cur_min_slot;

            if (n_pushed == W) {
                // First emit: scan the freshly-filled ring.
                int32_t oldest_slot = n_pushed % W;
                bool    found = false;
                for (int off = 0; off < W; ++off) {
                    int32_t s = (oldest_slot + off) % W;
                    if (!ring_valid[s]) continue;
                    if (!found || ring_canonical[s] < new_min_canonical) {
                        new_min_canonical = ring_canonical[s];
                        new_min_position  = ring_position[s];
                        new_min_slot      = s;
                        found = true;
                    }
                }
                if (found) {
                    need_emit = true;
                    have_min = true;
                }
            } else {
                // Window has slid by one.
                if (have_min && slot == cur_min_slot) {
                    // Previous min was overwritten by the new entry.
                    int32_t oldest_slot = n_pushed % W;
                    bool    found = false;
                    uint64_t scan_min_c = 0ull;
                    int32_t  scan_min_p = 0;
                    int32_t  scan_min_s = -1;
                    for (int off = 0; off < W; ++off) {
                        int32_t s = (oldest_slot + off) % W;
                        if (!ring_valid[s]) continue;
                        if (!found || ring_canonical[s] < scan_min_c) {
                            scan_min_c = ring_canonical[s];
                            scan_min_p = ring_position[s];
                            scan_min_s = s;
                            found = true;
                        }
                    }
                    if (found) {
                        new_min_canonical = scan_min_c;
                        new_min_position  = scan_min_p;
                        new_min_slot      = scan_min_s;
                        need_emit = true;
                    }
                } else if (have_min && canonical < cur_min_canonical) {
                    // Strict improvement.
                    new_min_canonical = canonical;
                    new_min_position  = kmer_start;
                    new_min_slot      = slot;
                    need_emit = true;
                }
            }

            if (need_emit) {
                cur_min_canonical = new_min_canonical;
                cur_min_position  = new_min_position;
                cur_min_slot      = new_min_slot;

                // Owned-range gate: drop emits whose minimizer-k-mer
                // start position is in the overlap region.
                bool owned = (cur_min_position >= owned_start_offset_bases);

                // v1: Fibonacci-hash slice filter on POSITION. n_passes_log2
                // == 0 is the single-pass back-compat case (every emit
                // passes). Otherwise slice = (pos * GOLDEN) >> (32 -
                // n_passes_log2); emit IFF slice == pass_idx.
                bool in_slice;
                if (n_passes_log2 == 0) {
                    in_slice = true;
                } else {
                    uint32_t hash = (uint32_t)cur_min_position * FIB_GOLDEN;
                    uint32_t slice = hash >> fib_shift;
                    in_slice = (slice == pass_idx_u32);
                }

                if (owned && in_slice && emit_idx < (uint32_t)MZ_MAX_EMIT_IDX) {
                    uint8_t* dst = partial_out + 4 +
                                   (size_t)emit_idx * (size_t)MZ_RECORD_BYTES;
                    // Bytes [0..7]: canonical LE.
                    dst[0] = (uint8_t)(cur_min_canonical >> 0);
                    dst[1] = (uint8_t)(cur_min_canonical >> 8);
                    dst[2] = (uint8_t)(cur_min_canonical >> 16);
                    dst[3] = (uint8_t)(cur_min_canonical >> 24);
                    dst[4] = (uint8_t)(cur_min_canonical >> 32);
                    dst[5] = (uint8_t)(cur_min_canonical >> 40);
                    dst[6] = (uint8_t)(cur_min_canonical >> 48);
                    dst[7] = (uint8_t)(cur_min_canonical >> 56);
                    // Bytes [8..11]: position LE (uint32).
                    uint32_t p_u32 = (uint32_t)cur_min_position;
                    dst[8]  = (uint8_t)(p_u32 >> 0);
                    dst[9]  = (uint8_t)(p_u32 >> 8);
                    dst[10] = (uint8_t)(p_u32 >> 16);
                    dst[11] = (uint8_t)(p_u32 >> 24);
                    // Bytes [12..15]: zero pad.
                    dst[12] = 0;
                    dst[13] = 0;
                    dst[14] = 0;
                    dst[15] = 0;
                    emit_idx += 1u;
                }
            }
            pos += 1;
        }
    }

    // Write emit_count prefix.
    partial_out[0] = (uint8_t)(emit_idx & 0xff);
    partial_out[1] = (uint8_t)((emit_idx >> 8) & 0xff);
    partial_out[2] = (uint8_t)((emit_idx >> 16) & 0xff);
    partial_out[3] = (uint8_t)((emit_idx >> 24) & 0xff);
}

}  // anonymous namespace

// =====================================================================
// Per-build single-(K, W) entry point.
//
// KW_ACTIVE values:
//   KW_15_10 = 1510 (k=15, w=10)
//   KW_21_11 = 2111 (k=21, w=11)
// =====================================================================

#ifndef KW_ACTIVE
#define KW_ACTIVE 1510  // default for standalone compile probes
#endif
static_assert(KW_ACTIVE == 1510 || KW_ACTIVE == 2111,
              "KW_ACTIVE must be 1510 (k15w10) or 2111 (k21w11)");

extern "C" {

#if KW_ACTIVE == 1510
void minimizer_tile_k15_w10(uint8_t* __restrict packed_in,
                            uint8_t* __restrict partial_out,
                            int32_t n_input_bytes,
                            int32_t pass_idx,
                            int32_t n_passes_log2,
                            int32_t tile_idx,
                            int32_t n_tiles_log2) {
    minimizer_tile_impl<15, 10, MINIMIZER_MASK_K15>(
        packed_in, partial_out, n_input_bytes,
        pass_idx, n_passes_log2, tile_idx, n_tiles_log2);
}
#endif

#if KW_ACTIVE == 2111
void minimizer_tile_k21_w11(uint8_t* __restrict packed_in,
                            uint8_t* __restrict partial_out,
                            int32_t n_input_bytes,
                            int32_t pass_idx,
                            int32_t n_passes_log2,
                            int32_t tile_idx,
                            int32_t n_tiles_log2) {
    minimizer_tile_impl<21, 11, MINIMIZER_MASK_K21>(
        packed_in, partial_out, n_input_bytes,
        pass_idx, n_passes_log2, tile_idx, n_tiles_log2);
}
#endif

}  // extern "C"
