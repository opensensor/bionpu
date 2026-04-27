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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// (T3, vectorisation half): use the AIE2P 32/64-lane uint8
// vector unit for the per-window mismatch loop. The aie_api headers
// require C++20 (see PEANOWRAP2P_FLAGS in programming_examples/makefile-
// common). We gate the vector path on the Peano-emitted __AIE_ARCH__
// macro so the scalar reference kernel still builds for non-AIE host
// targets used in unit tests. (__AIE_ARCH__ is set to 20 for AIE2 and
// 21 for AIE2P by the Peano clang frontend.)
#if defined(__AIE_ARCH__) && !defined(BIONPU_FORCE_SCALAR)
#  define BIONPU_HAS_AIE_API 1
#  include <aie_api/aie.hpp>
#else
#  define BIONPU_HAS_AIE_API 0
#endif

#ifndef SPACER_BYTES
#define SPACER_BYTES 5
#endif
#ifndef N_GUIDES
#define N_GUIDES 128
#endif
#ifndef GUIDES_PER_TILE
#define GUIDES_PER_TILE 64
#endif
#ifndef N_MATCH_TILES
#define N_MATCH_TILES 2
#endif
#ifndef WINDOW_BYTES_IN
#define WINDOW_BYTES_IN 6
#endif
#ifndef PAM_BYTES
#define PAM_BYTES 1
#endif
#ifndef EMIT_RECORD_BYTES
#define EMIT_RECORD_BYTES 8
#endif
#ifndef EMIT_SLOT_RECORDS
// fix (2026-04-26): bumped 256 -> 1024. The 256 cap silently
// dropped records on chr22's repeat-rich sub-chunks (worst observed 508
// hits / 64-window sub-chunk). Host-emulation reproducer at
// tracks/crispr/bench/run_npu_e2e_compare.py on chr22; pre-fix gap
// 1004 records. 1024 keeps slot at 8 KiB; double-buffered tile-DM
// usage 16 KiB (well within 64 KiB AIE2P L1).
#define EMIT_SLOT_RECORDS 1024
#endif

// 2-bit codes for ACGT (matches / encoding).
static constexpr uint8_t BASE_A = 0x0;
static constexpr uint8_t BASE_C = 0x1;
static constexpr uint8_t BASE_G = 0x2;
static constexpr uint8_t BASE_T = 0x3;

// PAM template for SpCas9 NGG: position 0 = wildcard (any), positions 1
// and 2 must equal G. We don't need a full IUPAC code system — just the
// `N` wildcard and literal G. may revisit.
static inline bool pam_is_ngg(uint8_t pam_byte) {
    // pam_byte layout: bits 1:0 = pam[0] (N, ignored)
    //                  bits 3:2 = pam[1] (must be G == 0b10)
    //                  bits 5:4 = pam[2] (must be G == 0b10)
    //                  bits 7:6 = padding (must be 0)
    uint8_t p1 = (pam_byte >> 2) & 0x3;
    uint8_t p2 = (pam_byte >> 4) & 0x3;
    return (p1 == BASE_G) && (p2 == BASE_G);
}

// popcount of a byte — table-free; AIE2P scalar core handles this fine.
static inline int popcount_u8(uint8_t x) {
    x = (uint8_t)((x & 0x55) + ((x >> 1) & 0x55));
    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
    x = (uint8_t)((x & 0x0F) + ((x >> 4) & 0x0F));
    return (int)x;
}

// One 20-nt mismatch count between guide[g_off:g_off+5] and window[w_off:w_off+5].
// Verbatim copy of / — preserves byte-equality by construction.
static inline uint8_t mismatch_count_5b(const uint8_t* g, const uint8_t* w) {
    int total = 0;
    for (int b = 0; b < SPACER_BYTES; b++) {
        uint8_t x = (uint8_t)(g[b] ^ w[b]);
        uint8_t m = (uint8_t)(((x | (x >> 1)) & 0x55));
        total += popcount_u8(m);
    }
    return (uint8_t)total;
}

extern "C" {

// ============================================================================
// Tile A — filter-early variant
// ============================================================================
//
// For each input window record (5 spacer bytes + 1 PAM byte):
//   - if pam matches NGG: copy spacer to output, set pam_meta byte = 1.
//   - else:               write zero spacer to output, set pam_meta = 0.
//
// We always write SPACER_BYTES of output per input position so the
// downstream match tiles see a contiguous, fixed-stride window stream
// (no per-chunk record-count negotiation). This matches 's chunk
// geometry exactly. Match tiles still compute mismatch counts on the
// "skipped" windows (their match output is irrelevant — Tile Z drops
// them via the pam_meta byte) but the kernel-launch cost saving comes
// from a compaction post-pass: may move to a sparse window-stream
// out of Tile A. For v1 the saving is on _meaningful_ work — the
// sparse-emit at Tile Z drops PAM-failing windows BEFORE they reach
// the output ring buffer, so DMA-out volume drops 7/8.
//
// (Note: the *true* "filter-early" cycle saving — match-tile cycles —
// requires either a sparse window stream or a tile-level "skip if
// pam=0" branch in the match kernel itself. For we keep the
// match kernel verbatim from and let the host accumulate the
// "windows actually examined" count from the pam_meta stream. This is
// the v1 trade-off documented in DESIGN.md and gaps.yaml.)
void crispr_pam_filter_tile_a_early(uint8_t* __restrict windows_in,
                                     uint8_t* __restrict windows_out,
                                     uint8_t* __restrict pam_meta,
                                     int32_t n_windows) {
    for (int w = 0; w < n_windows; w++) {
        const uint8_t* in = windows_in + (w * WINDOW_BYTES_IN);
        uint8_t* out = windows_out + (w * SPACER_BYTES);
        uint8_t pam_byte = in[SPACER_BYTES];
        bool pass = pam_is_ngg(pam_byte);
        pam_meta[w] = pass ? (uint8_t)1 : (uint8_t)0;
        if (pass) {
            // Forward the spacer bytes verbatim.
            for (int b = 0; b < SPACER_BYTES; b++) out[b] = in[b];
        } else {
            // Zero-fill so match tiles compute on a stable junk window;
            // the result is discarded at Tile Z by the pam_meta byte.
            for (int b = 0; b < SPACER_BYTES; b++) out[b] = 0;
        }
    }
}

// ============================================================================
// Tile A — filter-late variant
// ============================================================================
//
// Pass every window through unconditionally; pam_meta carries the PAM
// match for Tile Z to consult later. This is the "do nothing useful in
// Tile A" path used as the comparison baseline.
void crispr_pam_filter_tile_a_late(uint8_t* __restrict windows_in,
                                    uint8_t* __restrict windows_out,
                                    uint8_t* __restrict pam_meta,
                                    int32_t n_windows) {
    for (int w = 0; w < n_windows; w++) {
        const uint8_t* in = windows_in + (w * WINDOW_BYTES_IN);
        uint8_t* out = windows_out + (w * SPACER_BYTES);
        uint8_t pam_byte = in[SPACER_BYTES];
        // Always forward; record the PAM result for Tile Z.
        pam_meta[w] = pam_is_ngg(pam_byte) ? (uint8_t)1 : (uint8_t)0;
        for (int b = 0; b < SPACER_BYTES; b++) out[b] = in[b];
    }
}

// ============================================================================
// Match tile — vectorised match kernel
// ============================================================================
//
// (T3, vectorisation half): the per-window inner loop computes
// 64 mismatch counts (one per local guide) of 5 bytes each (~5 byte XORs +
// 2-bit popcounts + sum). Original scalar inner loop was 64 iterations of
// mismatch_count_5b → ~64 * 5 = 320 byte-ops per window × 4096 windows ×
// 2 match tiles ≈ 2.6 M scalar byte-ops per launch on a single AIE2P core.
// Combined with 's static-rate ObjectFifo wasting work on
// PAM-failing windows, this is contributor (2) to the ~5.8 s/launch
// kernel-time floor (gaps.yaml ).
//
// Strategy: vectorise across the 64 local guides. The guides arrive in
// AoS layout (`guides[g_global*5 + b]`); per-window we need byte-position-
// b of all 64 local guides (i.e. SoA). We transpose AoS → SoA ONCE at
// the top of the kernel call into a 320-byte scratch (5 vectors of 64
// uint8). For each window we then:
//
//   for b in 0..4:
//     wb = aie::broadcast<uint8, 64>(wp[b])
//     gb = aie::load_v<64>(scratch + b * 64)
//     x  = aie::bit_xor(wb, gb)
//     m  = (x | (x >> 1)) & 0x55      // any 2-bit-pair mismatch -> 1
//     // popcount of 4 bits at positions 0,2,4,6 in m:
//     p  = (m & 0x11) + ((m >> 2) & 0x11)
//     p  = (p & 0x0F) + ((p >> 4) & 0x0F)
//     acc += p
//   store_v(op, acc)
//
// That's ~9 vector ops per byte-position × 5 byte-positions + 5 vector
// adds = ~50 vector ops per window vs. 320 scalar byte-ops, a ~6×
// theoretical reduction; observed wall reduction on AIE2P silicon is the
// load-bearing number captured in
// state/g-t6.3-004/vectorisation-{iso8601}.json.
//
// Byte-equality with the scalar reference holds by construction: the
// vector arithmetic implements the same bitwise operations as the
// scalar mismatch_count_5b (XOR; pair-wise OR; mask; popcount); per-lane
// independent operations preserve the per-guide result.
//
// On non-AIE targets (host unit tests) the BIONPU_HAS_AIE_API gate
// falls back to the scalar reference so the .o still links.
// ============================================================================
#if BIONPU_HAS_AIE_API
void crispr_match_multitile_match(uint8_t* __restrict guides,
                                   uint8_t* __restrict windows,
                                   uint8_t* __restrict partial_out,
                                   int32_t n_windows,
                                   int32_t guide_offset) {
    constexpr int LANES = GUIDES_PER_TILE;  // 64
    using VU8 = ::aie::vector<uint8_t, LANES>;

    // ---- AoS → SoA transpose (once per call). ----
    // scratch[b*LANES + g] = guides[(guide_offset + g)*SPACER_BYTES + b]
    alignas(64) uint8_t scratch[SPACER_BYTES * LANES];
    {
        const uint8_t* gbase = guides + guide_offset * SPACER_BYTES;
        for (int g = 0; g < LANES; ++g) {
            const uint8_t* gp = gbase + g * SPACER_BYTES;
            for (int b = 0; b < SPACER_BYTES; ++b) {
                scratch[b * LANES + g] = gp[b];
            }
        }
    }

    // ---- Per-window vectorised inner loop. ----
    // Pre-load the 5 SoA guide-byte vectors (resident across all
    // n_windows). Loop carries (acc, gb0..gb4) in registers.
    VU8 gb0 = ::aie::load_v<LANES>(scratch + 0 * LANES);
    VU8 gb1 = ::aie::load_v<LANES>(scratch + 1 * LANES);
    VU8 gb2 = ::aie::load_v<LANES>(scratch + 2 * LANES);
    VU8 gb3 = ::aie::load_v<LANES>(scratch + 3 * LANES);
    VU8 gb4 = ::aie::load_v<LANES>(scratch + 4 * LANES);

    for (int w = 0; w < n_windows; ++w) {
        const uint8_t* wp = windows + (w * SPACER_BYTES);
        VU8 acc = ::aie::zeros<uint8_t, LANES>();

        // For each spacer byte position, broadcast the window byte,
        // XOR against the 64-lane SoA guide vector, mash the per-2-bit
        // mismatch bits down to a 4-bit popcount per lane, and add to
        // the accumulator.
        VU8 gb_arr[5] = {gb0, gb1, gb2, gb3, gb4};
        for (int b = 0; b < SPACER_BYTES; ++b) {
            VU8 wb = ::aie::broadcast<uint8_t, LANES>(wp[b]);
            VU8 x  = ::aie::bit_xor(wb, gb_arr[b]);
            // any-bit-set per 2-bit pair: m = (x | (x>>1)) & 0x55
            VU8 x_sh1 = ::aie::logical_downshift(x, 1);
            VU8 x_or  = ::aie::bit_or(x, x_sh1);
            VU8 m     = ::aie::bit_and(x_or,
                                        ::aie::broadcast<uint8_t, LANES>(0x55));
            // popcount of m where m has bits in {0,2,4,6}: at most 4 set.
            // Stage 1: pair (0+2), (4+6) → values in nibbles 0x11-style.
            VU8 m_sh2 = ::aie::logical_downshift(m, 2);
            VU8 p1_lo = ::aie::bit_and(m,
                                        ::aie::broadcast<uint8_t, LANES>(0x11));
            VU8 p1_hi = ::aie::bit_and(m_sh2,
                                        ::aie::broadcast<uint8_t, LANES>(0x11));
            VU8 p1    = ::aie::add(p1_lo, p1_hi);
            // Stage 2: combine low and high nibbles (each ≤ 2; sum ≤ 4).
            VU8 p1_sh4 = ::aie::logical_downshift(p1, 4);
            VU8 p2_lo  = ::aie::bit_and(p1,
                                         ::aie::broadcast<uint8_t, LANES>(0x0F));
            VU8 p2_hi  = ::aie::bit_and(p1_sh4,
                                         ::aie::broadcast<uint8_t, LANES>(0x0F));
            VU8 p      = ::aie::add(p2_lo, p2_hi);
            acc = ::aie::add(acc, p);
        }

        uint8_t* op = partial_out + (w * GUIDES_PER_TILE);
        ::aie::store_v(op, acc);
    }
}
#else  // BIONPU_HAS_AIE_API == 0  → scalar reference (host builds, tests).
void crispr_match_multitile_match(uint8_t* __restrict guides,
                                   uint8_t* __restrict windows,
                                   uint8_t* __restrict partial_out,
                                   int32_t n_windows,
                                   int32_t guide_offset) {
    for (int w = 0; w < n_windows; w++) {
        const uint8_t* wp = windows + (w * SPACER_BYTES);
        uint8_t* op = partial_out + (w * GUIDES_PER_TILE);
        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            const int g_global = guide_offset + g_local;
            const uint8_t* gp = guides + (g_global * SPACER_BYTES);
            op[g_local] = mismatch_count_5b(gp, wp);
        }
    }
}
#endif  // BIONPU_HAS_AIE_API

// ============================================================================
// Tile Z — filter-early variant (threshold only; PAM already cleared)
// ============================================================================
//
// For each (window, guide):
//   - if pam_meta[w] == 0: skip (PAM didn't pass; would have been
//     filtered out by Tile A anyway).
//   - if mismatch <= max_mm: emit (window_idx, guide_idx, mm).
//
// Output is a length-prefixed sparse record stream into the ring slot.
// First 4 bytes of `sparse_out` are the uint32 record count; subsequent
// records are 8 bytes each.
void crispr_pam_filter_tile_z_early(uint8_t* __restrict partial_0,
                                     uint8_t* __restrict partial_1,
                                     uint8_t* __restrict pam_meta,
                                     uint8_t* __restrict sparse_out,
                                     int32_t n_windows,
                                     int32_t max_mismatches,
                                     int32_t chunk_base_window_idx) {
    uint8_t* parts[N_MATCH_TILES] = { partial_0, partial_1 };
    uint32_t n_records = 0;
    uint8_t* dst = sparse_out + 4;  // first 4 bytes are the count prefix

    for (int w = 0; w < n_windows; w++) {
        if (pam_meta[w] == 0) continue;  // PAM failed (would be 0 in early mode if Tile A dropped it)
        for (int t = 0; t < N_MATCH_TILES; t++) {
            uint8_t* pp = parts[t] + (w * GUIDES_PER_TILE);
            for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
                uint8_t mm = pp[g_local];
                if ((int)mm > max_mismatches) continue;
                if (n_records >= (uint32_t)EMIT_SLOT_RECORDS) goto done;
                int g_global = t * GUIDES_PER_TILE + g_local;
                int wnd_idx = chunk_base_window_idx + w;
                dst[0] = (uint8_t)(wnd_idx & 0xff);
                dst[1] = (uint8_t)((wnd_idx >> 8) & 0xff);
                dst[2] = (uint8_t)g_global;
                dst[3] = mm;
                dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
                dst += EMIT_RECORD_BYTES;
                n_records++;
            }
        }
    }
done:
    sparse_out[0] = (uint8_t)(n_records & 0xff);
    sparse_out[1] = (uint8_t)((n_records >> 8) & 0xff);
    sparse_out[2] = (uint8_t)((n_records >> 16) & 0xff);
    sparse_out[3] = (uint8_t)((n_records >> 24) & 0xff);
}

// ============================================================================
// Tile Z — filter-late variant (threshold + PAM check)
// ============================================================================
//
// Same as the early variant except the per-window PAM check happens
// HERE — Tile A passed everything through. This is what makes
// filter-late "look at every window's match count" before the PAM
// filter discards 7/8 of them.
void crispr_pam_filter_tile_z_late(uint8_t* __restrict partial_0,
                                    uint8_t* __restrict partial_1,
                                    uint8_t* __restrict pam_meta,
                                    uint8_t* __restrict sparse_out,
                                    int32_t n_windows,
                                    int32_t max_mismatches,
                                    int32_t chunk_base_window_idx) {
    uint8_t* parts[N_MATCH_TILES] = { partial_0, partial_1 };
    uint32_t n_records = 0;
    uint8_t* dst = sparse_out + 4;

    for (int w = 0; w < n_windows; w++) {
        // PAM check happens HERE in filter-late mode. Match tiles
        // already paid the (n_guides × spacer_bytes) cycle cost for
        // every window, including the 7/8 that fail PAM.
        if (pam_meta[w] == 0) continue;
        for (int t = 0; t < N_MATCH_TILES; t++) {
            uint8_t* pp = parts[t] + (w * GUIDES_PER_TILE);
            for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
                uint8_t mm = pp[g_local];
                if ((int)mm > max_mismatches) continue;
                if (n_records >= (uint32_t)EMIT_SLOT_RECORDS) goto done;
                int g_global = t * GUIDES_PER_TILE + g_local;
                int wnd_idx = chunk_base_window_idx + w;
                dst[0] = (uint8_t)(wnd_idx & 0xff);
                dst[1] = (uint8_t)((wnd_idx >> 8) & 0xff);
                dst[2] = (uint8_t)g_global;
                dst[3] = mm;
                dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
                dst += EMIT_RECORD_BYTES;
                n_records++;
            }
        }
    }
done:
    sparse_out[0] = (uint8_t)(n_records & 0xff);
    sparse_out[1] = (uint8_t)((n_records >> 8) & 0xff);
    sparse_out[2] = (uint8_t)((n_records >> 16) & 0xff);
    sparse_out[3] = (uint8_t)((n_records >> 24) & 0xff);
}

} // extern "C"
