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

// Compile-time pin: 20 nt spacer = 5 packed bytes; 128 total guides; 32 per
// match tile (128 / 4 match tiles).
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

// popcount of a byte — table-free; AIE2P scalar core handles this fine.
static inline int popcount_u8(uint8_t x) {
    x = (uint8_t)((x & 0x55) + ((x >> 1) & 0x55));
    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
    x = (uint8_t)((x & 0x0F) + ((x >> 4) & 0x0F));
    return (int)x;
}

// One 20-nt mismatch count between guide[g_off:g_off+5] and window[w_off:w_off+5].
static inline uint8_t mismatch_count_5b(const uint8_t* g, const uint8_t* w) {
    int total = 0;
    for (int b = 0; b < SPACER_BYTES; b++) {
        uint8_t x = (uint8_t)(g[b] ^ w[b]);
        // collapse each 2-bit pair to a single bit in the low half:
        // pair == 00 → 0; any other → 1.
        uint8_t m = (uint8_t)(((x | (x >> 1)) & 0x55));
        total += popcount_u8(m);
    }
    return (uint8_t)total;
}

extern "C" {

// Process one chunk of windows against this match-tile's 32-guide slice of
// the full 128-guide batch.
//
// Args:
//   guides:        N_GUIDES * SPACER_BYTES bytes (full batch; resident).
//   windows:       n_windows * SPACER_BYTES bytes.
//   partial_out:   n_windows * GUIDES_PER_TILE bytes; row-major
//                  (window-major) — partial_out[w * GUIDES_PER_TILE + g_local]
//                  = mismatch_count(guides[guide_offset + g_local], windows[w]).
//   n_windows:     int32, number of 5-byte windows in this chunk.
//   guide_offset:  int32 ∈ {0, 32, 64, 96}; this tile's slice into the
//                  128-guide batch.
//
// Why we keep the full guides buffer resident on every match tile (instead
// of slicing it per-tile via a memtile DMA): IRON's broadcast ObjectFifo
// pattern delivers the same guides buffer to every match tile naturally.
// Each tile slices in software via the `guide_offset` parameter; the cost
// is 640 B - 160 B = 480 B of unused-but-resident guides per tile, which is
// negligible against the 64 KiB per-tile DM budget. may revisit this
// if profile data shows the broadcast saturating shim-DMA bandwidth.
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

// Joiner: take 2 partial outputs (each n_windows × GUIDES_PER_TILE=64) and
// concatenate along the guide axis to produce n_windows × N_GUIDES=128.
//
// Layout requirement (matches / single-tile kernel output):
//   out[w * N_GUIDES + g] = partial_<g/GUIDES_PER_TILE>[w * GUIDES_PER_TILE + (g%GUIDES_PER_TILE)]
//
// This is the byte-equality bridge: 's single-tile kernel writes
// out[w, g] in window-major directly; reconstructs the same matrix
// from 2 column slabs. No floating-point operations, no reorderings — the
// resulting bytes are identical to 's by construction.
//
// (Originally designed for 4 partials per CRISPR PRD §4.2 sketch; reduced
// to 2 due to AIE2P's 2-input-DMA-channels-per-tile budget — see
// gaps.yaml . The kernel scales to N_MATCH_TILES if a memtile
// aggregation stage is added in front of the joiner; left for .)
void crispr_match_multitile_join(uint8_t* __restrict p0,
                                  uint8_t* __restrict p1,
                                  uint8_t* __restrict out,
                                  int32_t n_windows) {
    uint8_t* parts[N_MATCH_TILES] = { p0, p1 };
    for (int w = 0; w < n_windows; w++) {
        uint8_t* op = out + (w * N_GUIDES);
        for (int t = 0; t < N_MATCH_TILES; t++) {
            uint8_t* pp = parts[t] + (w * GUIDES_PER_TILE);
            uint8_t* dst = op + (t * GUIDES_PER_TILE);
            for (int g = 0; g < GUIDES_PER_TILE; g++) {
                dst[g] = pp[g];
            }
        }
    }
}

} // extern "C"
