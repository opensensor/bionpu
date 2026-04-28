// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// tandem_repeat_tile.cc — per-tile short tandem repeat (STR) scanner (v0).
//
// Algorithm (mirrors bionpu.data.tandem_repeat_oracle.find_tandem_repeats):
//
//   For each period q in [TR_MIN_PERIOD, TR_MAX_PERIOD]:
//     Walk p in [q, n_bases):
//       If seq[p] == seq[p - q]:
//         If streak == 0: streak_start = p - q  (first motif start)
//         streak += 1
//       Else:
//         If streak >= q * (TR_MIN_COPIES - 1):
//           n_copies = (streak + q) / q
//           emit (streak_start, streak_start + n_copies * q,
//                 q, motif=seq[streak_start..streak_start+q])
//         streak = 0
//     End-of-chunk: emit any pending streak that meets the threshold.
//
// The end position is q-ALIGNED to the streak_start: it's
// streak_start + n_copies * q where n_copies = floor((streak + q) / q).
// This matches the oracle's autocorrelation streak semantics exactly.
//
// Notes:
//
//   - No owned-range gate. The host runner's per-(period, motif)
//     overlap-merge pass handles chunk-overlap dedup uniformly,
//     including STRs that span chunk boundaries.
//
//   - Per CLAUDE.md: in-process pyxrt path takes only _dispatch_lock;
//     subprocess harness wraps npu_silicon_lock. Kernel itself is
//     hardware-agnostic.
//
//   - Base alphabet is the same MSB-first 2-bit code used by all other
//     genomics kernels (A=00, C=01, G=10, T=11).

#include <stdint.h>

#include "tandem_repeat_constants.h"

static_assert(TR_PARTIAL_OUT_BYTES_PADDED == 32768,
              "partial_out element must be 32 KiB");
static_assert(TR_RECORD_BYTES == 16, "Record = 16 bytes");
static_assert(TR_HEADER_BYTES == 8, "in-band header pinned at 8 bytes");
static_assert(TR_MAX_PERIOD == 6, "v0 fixes MAX_PERIOD = 6");
static_assert(TR_MIN_COPIES == 5, "v0 fixes MIN_COPIES = 5");
static_assert(TR_MIN_PERIOD == 1, "v0 fixes MIN_PERIOD = 1");

namespace {

static inline uint8_t packed_base_at(const uint8_t* packed_in, int32_t idx) {
    uint8_t byte = packed_in[TR_HEADER_BYTES + (idx >> 2)];
    int32_t shift = 6 - 2 * (idx & 3);
    return (uint8_t)((byte >> shift) & 0x3);
}

static inline uint32_t pack_motif(const uint8_t* packed_in,
                                  int32_t start, int32_t period) {
    // MSB-first packing, mirrors motif_to_canonical_u32 in the oracle.
    uint32_t v = 0u;
    for (int32_t i = 0; i < period; ++i) {
        v = (v << 2) | (uint32_t)packed_base_at(packed_in, start + i);
    }
    return v;
}

static inline void emit_record(uint8_t* __restrict partial_out,
                               uint32_t emit_idx,
                               uint32_t start, uint32_t end,
                               uint32_t period, uint32_t motif_canon) {
    uint8_t* dst = partial_out + 4u +
                   (uint32_t)TR_RECORD_BYTES * emit_idx;
    dst[0]  = (uint8_t)(start >> 0);
    dst[1]  = (uint8_t)(start >> 8);
    dst[2]  = (uint8_t)(start >> 16);
    dst[3]  = (uint8_t)(start >> 24);
    dst[4]  = (uint8_t)(end >> 0);
    dst[5]  = (uint8_t)(end >> 8);
    dst[6]  = (uint8_t)(end >> 16);
    dst[7]  = (uint8_t)(end >> 24);
    dst[8]  = (uint8_t)(period >> 0);
    dst[9]  = (uint8_t)(period >> 8);
    dst[10] = (uint8_t)(period >> 16);
    dst[11] = (uint8_t)(period >> 24);
    dst[12] = (uint8_t)(motif_canon >> 0);
    dst[13] = (uint8_t)(motif_canon >> 8);
    dst[14] = (uint8_t)(motif_canon >> 16);
    dst[15] = (uint8_t)(motif_canon >> 24);
}

}  // anonymous namespace

extern "C" void tandem_repeat_tile(uint8_t* __restrict packed_in,
                                   uint8_t* __restrict partial_out,
                                   int32_t n_input_bytes,
                                   int32_t /*tile_idx*/,
                                   int32_t /*n_tiles_log2*/) {
    int32_t actual_bytes = (int32_t)packed_in[0]
                         | ((int32_t)packed_in[1] << 8)
                         | ((int32_t)packed_in[2] << 16)
                         | ((int32_t)packed_in[3] << 24);
    // owned_start_offset_bases (bytes [4..7]) is reserved; v0 host
    // runner sets it to 0. The kernel emits all qualifying streaks;
    // host overlap-merge handles chunk-overlap dedup.
    if (actual_bytes < 0) {
        actual_bytes = 0;
    }
    if (actual_bytes > n_input_bytes - TR_HEADER_BYTES) {
        actual_bytes = n_input_bytes - TR_HEADER_BYTES;
    }

    int32_t n_bases = actual_bytes * 4;

    uint32_t emit_idx = 0u;
    if (n_bases < TR_MIN_PERIOD * TR_MIN_COPIES) {
        partial_out[0] = 0;
        partial_out[1] = 0;
        partial_out[2] = 0;
        partial_out[3] = 0;
        return;
    }

    // Per-period streak threshold: a tandem repeat with MIN_COPIES
    // copies of period q has length q * MIN_COPIES, of which
    // q * (MIN_COPIES - 1) bases match their predecessor at distance q.
    const int32_t threshold[TR_MAX_PERIOD + 1] = {
        0,                              // q=0 unused
        1 * (TR_MIN_COPIES - 1),        // q=1: 4
        2 * (TR_MIN_COPIES - 1),        // q=2: 8
        3 * (TR_MIN_COPIES - 1),        // q=3: 12
        4 * (TR_MIN_COPIES - 1),        // q=4: 16
        5 * (TR_MIN_COPIES - 1),        // q=5: 20
        6 * (TR_MIN_COPIES - 1),        // q=6: 24
    };

    // Per-period scan. Period-major loop keeps state minimal: a single
    // streak counter and streak_start position per period. The host
    // does cross-period dedup + chunk-spanning merge.
    for (int32_t q = TR_MIN_PERIOD; q <= TR_MAX_PERIOD; ++q) {
        int32_t streak = 0;
        int32_t streak_start = 0;

        for (int32_t p = q; p < n_bases; ++p) {
            uint8_t b_p = packed_base_at(packed_in, p);
            uint8_t b_pred = packed_base_at(packed_in, p - q);
            if (b_p == b_pred) {
                if (streak == 0) {
                    streak_start = p - q;
                }
                streak += 1;
            } else {
                if (streak >= threshold[q]) {
                    if (emit_idx < (uint32_t)TR_MAX_EMIT_IDX) {
                        int32_t n_copies = (streak + q) / q;
                        uint32_t end = (uint32_t)(streak_start + n_copies * q);
                        uint32_t motif_canon =
                            pack_motif(packed_in, streak_start, q);
                        emit_record(partial_out, emit_idx,
                                    (uint32_t)streak_start,
                                    end,
                                    (uint32_t)q,
                                    motif_canon);
                        emit_idx += 1u;
                    }
                }
                streak = 0;
            }
        }
        // End-of-chunk flush.
        if (streak >= threshold[q]) {
            if (emit_idx < (uint32_t)TR_MAX_EMIT_IDX) {
                int32_t n_copies = (streak + q) / q;
                uint32_t end = (uint32_t)(streak_start + n_copies * q);
                uint32_t motif_canon =
                    pack_motif(packed_in, streak_start, q);
                emit_record(partial_out, emit_idx,
                            (uint32_t)streak_start,
                            end,
                            (uint32_t)q,
                            motif_canon);
                emit_idx += 1u;
            }
        }
    }

    partial_out[0] = (uint8_t)(emit_idx & 0xff);
    partial_out[1] = (uint8_t)((emit_idx >> 8) & 0xff);
    partial_out[2] = (uint8_t)((emit_idx >> 16) & 0xff);
    partial_out[3] = (uint8_t)((emit_idx >> 24) & 0xff);
}
