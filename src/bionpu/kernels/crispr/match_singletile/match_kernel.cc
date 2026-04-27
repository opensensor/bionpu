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

// Compile-time pin: 20 nt spacer = 5 packed bytes; 128 guides; window-chunk
// size adjustable via the IRON Python lowering.
#ifndef SPACER_BYTES
#define SPACER_BYTES 5
#endif

#ifndef N_GUIDES
#define N_GUIDES 128
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

// Process one chunk of windows against all N_GUIDES resident guides.
//
// Args:
//   guides:  N_GUIDES * SPACER_BYTES bytes, layout [guide0_byte0..byte4,
//            guide1_byte0..byte4, ...].
//   windows: n_windows * SPACER_BYTES bytes, same layout.
//   out:     n_windows * N_GUIDES bytes; row-major (window-major):
//            out[w * N_GUIDES + g] = mismatch_count(guide g, window w).
//   n_windows: int32, number of 5-byte windows in this chunk.
//
// Why window-major output: it matches the per-window streaming pattern
// (one window's row of 128 mismatch counts is fully resident at a time),
// which is what will need when it slots in tile-local thresholding.
// The host wrapper transposes to (n_guides, n_windows) at the boundary
// so the Python contract matches the oracle's natural shape.
void crispr_match_singletile(uint8_t* __restrict guides,
                             uint8_t* __restrict windows,
                             uint8_t* __restrict out,
                             int32_t n_windows) {
    for (int w = 0; w < n_windows; w++) {
        const uint8_t* wp = windows + (w * SPACER_BYTES);
        uint8_t* op = out + (w * N_GUIDES);
        for (int g = 0; g < N_GUIDES; g++) {
            const uint8_t* gp = guides + (g * SPACER_BYTES);
            op[g] = mismatch_count_5b(gp, wp);
        }
    }
}

} // extern "C"
