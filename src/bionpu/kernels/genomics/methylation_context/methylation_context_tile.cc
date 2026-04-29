// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// methylation_context_tile.cc — per-tile CG/CHG/CHH sparse scanner (v0).

#include <stdint.h>

#include "methylation_context_constants.h"

static_assert(MC_PARTIAL_OUT_BYTES_PADDED == 32768,
              "partial_out element must be 32 KiB");
static_assert(MC_RECORD_BYTES == 8,
              "Record = uint32 pos | uint8 strand | uint8 context | pad");
static_assert(MC_HEADER_BYTES == 8, "in-band header pinned at 8 bytes");

namespace {

static inline uint8_t packed_base_at(const uint8_t* packed_in, int32_t idx) {
    uint8_t byte = packed_in[MC_HEADER_BYTES + (idx >> 2)];
    int32_t shift = 6 - 2 * (idx & 3);
    return (uint8_t)((byte >> shift) & 0x3);
}

static inline uint8_t comp(uint8_t b) {
    return (uint8_t)(b ^ 0x3u);
}

static inline bool is_h(uint8_t b) {
    return b == MC_BASE_A || b == MC_BASE_C || b == MC_BASE_T;
}

static inline void emit_record(uint8_t* __restrict partial_out,
                               uint32_t emit_idx,
                               uint32_t pos,
                               uint8_t strand,
                               uint8_t context) {
    uint8_t* dst = partial_out + 4u +
                   (uint32_t)MC_RECORD_BYTES * emit_idx;
    dst[0] = (uint8_t)(pos >> 0);
    dst[1] = (uint8_t)(pos >> 8);
    dst[2] = (uint8_t)(pos >> 16);
    dst[3] = (uint8_t)(pos >> 24);
    dst[4] = strand;
    dst[5] = context;
    dst[6] = 0;
    dst[7] = 0;
}

static inline bool classify_plus(const uint8_t* packed_in,
                                 int32_t n_bases,
                                 int32_t pos,
                                 uint8_t* context) {
    if (pos + 1 >= n_bases) {
        return false;
    }
    uint8_t b1 = packed_base_at(packed_in, pos + 1);
    if (b1 == MC_BASE_G) {
        *context = MC_CONTEXT_CG;
        return true;
    }
    if (!is_h(b1) || pos + 2 >= n_bases) {
        return false;
    }
    uint8_t b2 = packed_base_at(packed_in, pos + 2);
    if (b2 == MC_BASE_G) {
        *context = MC_CONTEXT_CHG;
        return true;
    }
    if (is_h(b2)) {
        *context = MC_CONTEXT_CHH;
        return true;
    }
    return false;
}

static inline bool classify_minus(const uint8_t* packed_in,
                                  int32_t pos,
                                  uint8_t* context) {
    if (pos < 1) {
        return false;
    }
    uint8_t b1 = comp(packed_base_at(packed_in, pos - 1));
    if (b1 == MC_BASE_G) {
        *context = MC_CONTEXT_CG;
        return true;
    }
    if (!is_h(b1) || pos < 2) {
        return false;
    }
    uint8_t b2 = comp(packed_base_at(packed_in, pos - 2));
    if (b2 == MC_BASE_G) {
        *context = MC_CONTEXT_CHG;
        return true;
    }
    if (is_h(b2)) {
        *context = MC_CONTEXT_CHH;
        return true;
    }
    return false;
}

}  // anonymous namespace

extern "C" void methylation_context_tile(uint8_t* __restrict packed_in,
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
    if (actual_bytes > n_input_bytes - MC_HEADER_BYTES) {
        actual_bytes = n_input_bytes - MC_HEADER_BYTES;
    }

    int32_t n_bases = actual_bytes * 4;
    uint32_t emit_idx = 0u;
    for (int32_t p = owned_start_offset_bases; p < n_bases; ++p) {
        uint8_t b = packed_base_at(packed_in, p);
        uint8_t context = 0;
        bool emit = false;
        uint8_t strand = MC_STRAND_PLUS;
        if (b == MC_BASE_C) {
            emit = classify_plus(packed_in, n_bases, p, &context);
            strand = MC_STRAND_PLUS;
        } else if (b == MC_BASE_G) {
            emit = classify_minus(packed_in, p, &context);
            strand = MC_STRAND_MINUS;
        }
        if (emit && emit_idx < (uint32_t)MC_MAX_EMIT_IDX) {
            emit_record(partial_out, emit_idx, (uint32_t)p, strand, context);
            emit_idx += 1u;
        }
    }

    partial_out[0] = (uint8_t)(emit_idx >> 0);
    partial_out[1] = (uint8_t)(emit_idx >> 8);
    partial_out[2] = (uint8_t)(emit_idx >> 16);
    partial_out[3] = (uint8_t)(emit_idx >> 24);
}
