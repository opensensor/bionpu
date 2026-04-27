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
#define GUIDES_PER_TILE 32
#endif

#ifndef N_MATCH_TILES
#define N_MATCH_TILES 4
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
// **Guide-major partial output** — chosen to make IRON memtile flat-concat
// `.join()` work correctly. Each tile emits its 32 guides × n_windows in
// guide-major layout (`partial_out[g_local * n_windows + w]`); 4 tiles
// flat-concatenated by memtile yield a (128 guides × n_windows) guide-
// major buffer per chunk. The host then concatenates per-chunk buffers
// across the 64 chunks to form the full (N_GUIDES × N_WINDOWS) guide-major
// matrix — directly matching the registered-op output contract (no
// transpose required, unlike 's window-major path).
//
// Args:
//   guides:        N_GUIDES * SPACER_BYTES bytes (full batch; resident).
//   windows:       n_windows * SPACER_BYTES bytes.
//   partial_out:   GUIDES_PER_TILE * n_windows bytes; **guide-major**
//                  partial_out[g_local * n_windows + w]
//                  = mismatch_count(guides[guide_offset + g_local], windows[w]).
//   n_windows:     int32, number of 5-byte windows in this chunk.
//   guide_offset:  int32 ∈ {0, 32, 64, 96}; this tile's slice into the
//                  128-guide batch.
//
// Why we keep the full guides buffer resident on every match tile (instead
// of slicing it per-tile via a shim DMA): IRON's broadcast ObjectFifo
// pattern delivers the same guides buffer to every match tile naturally.
// Each tile slices in software via the `guide_offset` parameter; the cost
// is 640 B - 160 B = 480 B of unused-but-resident guides per tile, which
// is negligible against the 64 KiB per-tile DM budget.
void crispr_match_memtile_match(uint8_t* __restrict guides,
                                 uint8_t* __restrict windows,
                                 uint8_t* __restrict partial_out,
                                 int32_t n_windows,
                                 int32_t guide_offset) {
    for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
        const int g_global = guide_offset + g_local;
        const uint8_t* gp = guides + (g_global * SPACER_BYTES);
        uint8_t* op = partial_out + (g_local * n_windows);
        for (int w = 0; w < n_windows; w++) {
            const uint8_t* wp = windows + (w * SPACER_BYTES);
            op[w] = mismatch_count_5b(gp, wp);
        }
    }
}

} // extern "C"
