// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// cpg_island_tile.cc — per-tile CpG-island candidate scanner (v0).

#include <stdint.h>

#include "cpg_island_constants.h"

static_assert(CI_PARTIAL_OUT_BYTES_PADDED == 32768,
              "partial_out element must be 32 KiB");
static_assert(CI_RECORD_BYTES == 4, "Record = uint32 position");
static_assert(CI_HEADER_BYTES == 8, "in-band header pinned at 8 bytes");

namespace {

static inline void emit_pos(uint8_t* __restrict partial_out,
                            uint32_t emit_idx,
                            uint32_t pos) {
    uint8_t* dst = partial_out + 4u +
                   (uint32_t)CI_RECORD_BYTES * emit_idx;
    dst[0] = (uint8_t)(pos >> 0);
    dst[1] = (uint8_t)(pos >> 8);
    dst[2] = (uint8_t)(pos >> 16);
    dst[3] = (uint8_t)(pos >> 24);
}

static inline bool passes_thresholds(int32_t n_c,
                                     int32_t n_g,
                                     int32_t n_cg) {
    if (CI_GC_DEN * (n_c + n_g) < CI_GC_NUM * CI_W) {
        return false;
    }
    if (n_c == 0 || n_g == 0) {
        return false;
    }
    if (CI_OE_DEN * CI_W * n_cg < CI_OE_NUM * n_c * n_g) {
        return false;
    }
    return true;
}

static inline uint8_t packed_base_at(const uint8_t* packed_in, int32_t idx) {
    uint8_t byte = packed_in[CI_HEADER_BYTES + (idx >> 2)];
    int32_t shift = 6 - 2 * (idx & 3);
    return (uint8_t)((byte >> shift) & 0x3);
}

}  // anonymous namespace

extern "C" void cpg_island_tile(uint8_t* __restrict packed_in,
                                uint8_t* __restrict partial_out,
                                int32_t n_input_bytes,
                                int32_t /*tile_idx*/,
                                int32_t /*n_tiles_log2*/) {
    int32_t actual_bytes = (int32_t)packed_in[0]
                         | ((int32_t)packed_in[1] << 8)
                         | ((int32_t)packed_in[2] << 16)
                         | ((int32_t)packed_in[3] << 24);
    int32_t owned_start_offset_bases =
          (int32_t)packed_in[4]
        | ((int32_t)packed_in[5] << 8)
        | ((int32_t)packed_in[6] << 16)
        | ((int32_t)packed_in[7] << 24);
    if (owned_start_offset_bases < 0) {
        owned_start_offset_bases = 0;
    }
    if (actual_bytes < 0) {
        actual_bytes = 0;
    }
    if (actual_bytes > n_input_bytes - CI_HEADER_BYTES) {
        actual_bytes = n_input_bytes - CI_HEADER_BYTES;
    }

    int32_t n_bases = actual_bytes * 4;

    uint32_t emit_idx = 0u;
    if (n_bases < CI_W) {
        partial_out[0] = 0;
        partial_out[1] = 0;
        partial_out[2] = 0;
        partial_out[3] = 0;
        return;
    }

    int32_t n_c = 0;
    int32_t n_g = 0;
    int32_t n_cg = 0;
    for (int32_t i = 0; i < CI_W; ++i) {
        uint8_t b = packed_base_at(packed_in, i);
        if (b == CI_BASE_C) {
            n_c += 1;
        } else if (b == CI_BASE_G) {
            n_g += 1;
        }
        if (i >= 1 &&
            packed_base_at(packed_in, i - 1) == CI_BASE_C &&
            b == CI_BASE_G) {
            n_cg += 1;
        }
    }

    int32_t n_windows = n_bases - CI_W + 1;
    for (int32_t p = 0; p < n_windows; ++p) {
        if (p >= owned_start_offset_bases &&
            passes_thresholds(n_c, n_g, n_cg)) {
            if (emit_idx < (uint32_t)CI_MAX_EMIT_IDX) {
                emit_pos(partial_out, emit_idx, (uint32_t)p);
                emit_idx += 1u;
            }
        }

        if (p + 1 >= n_windows) {
            break;
        }

        uint8_t out_b = packed_base_at(packed_in, p);
        if (out_b == CI_BASE_C) {
            n_c -= 1;
        } else if (out_b == CI_BASE_G) {
            n_g -= 1;
        }
        if (packed_base_at(packed_in, p) == CI_BASE_C &&
            packed_base_at(packed_in, p + 1) == CI_BASE_G) {
            n_cg -= 1;
        }

        int32_t in_idx = p + CI_W;
        uint8_t in_b = packed_base_at(packed_in, in_idx);
        if (in_b == CI_BASE_C) {
            n_c += 1;
        } else if (in_b == CI_BASE_G) {
            n_g += 1;
        }
        if (packed_base_at(packed_in, in_idx - 1) == CI_BASE_C &&
            in_b == CI_BASE_G) {
            n_cg += 1;
        }
    }

    partial_out[0] = (uint8_t)(emit_idx & 0xff);
    partial_out[1] = (uint8_t)((emit_idx >> 8) & 0xff);
    partial_out[2] = (uint8_t)((emit_idx >> 16) & 0xff);
    partial_out[3] = (uint8_t)((emit_idx >> 24) & 0xff);
}
