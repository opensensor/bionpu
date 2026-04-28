// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// primer_scan_tile.cc — per-tile primer / adapter scan kernel (v0).
//
// Algorithm — simplest possible:
//
//   For each new ACGT base entering at index i:
//     1. fwd = ((fwd << 2) | base) & MASK
//     2. rc  = (rc >> 2) | ((base ^ 3) << (2*(P-1)))
//     3. valid_run += 1
//     4. If valid_run < P: continue.
//     5. kmer_start = i - (P - 1).
//     6. Owned-range gate: kmer_start >= owned_start_offset_bases.
//     7. If fwd == primer_fwd: emit (kmer_start, strand=0).
//     8. Else if rc == primer_fwd AND primer_fwd != primer_rc:
//                emit (kmer_start, strand=1).
//        (Palindromes — primer_fwd == primer_rc — emit only the
//         forward record per the oracle's documented contract.)
//
// Output record layout (16 bytes/record):
//   bytes [0..3]:   uint32 LE query_pos (kmer_start within chunk
//                   payload; host translates to global by adding
//                   chunk's source offset).
//   bytes [4]:      uint8 strand (0 = forward; 1 = reverse-complement).
//   bytes [5]:      uint8 primer_idx (zero for v0 single-primer).
//   bytes [6..7]:   uint16 _pad (zero).
//   bytes [8..15]:  uint64 _pad2 (zero).
//
// Path B (runtime primer): the primer's canonical pair lives in the
// chunk header (16 bytes after the 8-byte length+offset prefix). Each
// dispatch can scan a different primer without rebuilding the xclbin.

#include <stdint.h>

#include "primer_scan_constants.h"

#if defined(__AIE_ARCH__) && !defined(BIONPU_FORCE_SCALAR)
#  define BIONPU_HAS_AIE_API 1
#  include "aie_kernel_utils.h"
#  include <aie_api/aie.hpp>
#else
#  define BIONPU_HAS_AIE_API 0
#endif

static_assert(PS_PARTIAL_OUT_BYTES_PADDED == 32768,
              "partial_out element must be 32 KiB");
static_assert(PS_RECORD_BYTES == 16, "Record = 16 bytes");
static_assert(PS_MAX_EMIT_IDX == 2046,
              "max emit_idx caps at 2046 to fit in 32 KiB - 4 prefix");
static_assert(PS_HEADER_BYTES == 24,
              "in-band header pinned at 24 bytes (8 + 8 fwd + 8 rc)");

namespace {

// Decode a uint64 little-endian value from a byte buffer (no alignment
// assumption).
static inline uint64_t load_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    v |= ((uint64_t)p[0]) <<  0;
    v |= ((uint64_t)p[1]) <<  8;
    v |= ((uint64_t)p[2]) << 16;
    v |= ((uint64_t)p[3]) << 24;
    v |= ((uint64_t)p[4]) << 32;
    v |= ((uint64_t)p[5]) << 40;
    v |= ((uint64_t)p[6]) << 48;
    v |= ((uint64_t)p[7]) << 56;
    return v;
}

template <int P, uint64_t MASK>
static inline void primer_scan_tile_impl(
        uint8_t* __restrict packed_in,
        uint8_t* __restrict partial_out,
        int32_t n_input_bytes,
        int32_t /*tile_idx*/,
        int32_t /*n_tiles_log2*/) {
    uint32_t emit_idx = 0u;

    // Rolling fwd / rc registers + fill counter.
    uint64_t fwd = 0ull;
    uint64_t rc  = 0ull;
    int32_t  valid_run = 0;

    constexpr int32_t RC_HIGH_SHIFT = 2 * (P - 1);

    // ----- In-band header decode. -----
    // Bytes [0..3]: actual_payload_bytes
    // Bytes [4..7]: owned_start_offset_bases
    // Bytes [8..15]: primer_fwd_canonical (uint64 LE)
    // Bytes [16..23]: primer_rc_canonical  (uint64 LE)
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
    if (actual_bytes > n_input_bytes - PS_HEADER_BYTES) {
        actual_bytes = n_input_bytes - PS_HEADER_BYTES;
    }

    const uint64_t primer_fwd = load_u64_le(packed_in + 8) & MASK;
    const uint64_t primer_rc  = load_u64_le(packed_in + 16) & MASK;
    const bool is_palindrome = (primer_fwd == primer_rc);

    int32_t pos = 0;

    for (int32_t i = 0; i < actual_bytes; ++i) {
        uint8_t byte = packed_in[PS_HEADER_BYTES + i];
        for (int32_t shift = 6; shift >= 0; shift -= 2) {
            uint8_t base = (uint8_t)((byte >> shift) & 0x3);
            uint8_t comp = (uint8_t)(base ^ 0x3);

            fwd = ((fwd << 2) | (uint64_t)base) & MASK;
            rc  = ((rc >> 2) | ((uint64_t)comp << RC_HIGH_SHIFT)) & MASK;

            if (valid_run < P) {
                valid_run += 1;
                if (valid_run < P) {
                    pos += 1;
                    continue;
                }
                // Just reached the first full P-mer; fall through.
            }

            int32_t kmer_start = pos - (P - 1);
            // Owned-range gate.
            if (kmer_start < owned_start_offset_bases) {
                pos += 1;
                continue;
            }

            // Forward-strand match.
            bool fwd_match = (fwd == primer_fwd);
            // RC-strand match (only fire for non-palindromic primers
            // to avoid duplicate emit at palindrome positions).
            bool rc_match = (rc == primer_fwd) && !is_palindrome;

            if (fwd_match && emit_idx < (uint32_t)PS_MAX_EMIT_IDX) {
                uint8_t* dst = partial_out + 4 +
                               (size_t)emit_idx * (size_t)PS_RECORD_BYTES;
                uint32_t qp = (uint32_t)kmer_start;
                dst[0] = (uint8_t)(qp >> 0);
                dst[1] = (uint8_t)(qp >> 8);
                dst[2] = (uint8_t)(qp >> 16);
                dst[3] = (uint8_t)(qp >> 24);
                dst[4] = (uint8_t)0;  // strand = forward
                dst[5] = (uint8_t)0;  // primer_idx (single-primer v0)
                dst[6] = 0; dst[7] = 0;
                dst[8]  = 0; dst[9]  = 0; dst[10] = 0; dst[11] = 0;
                dst[12] = 0; dst[13] = 0; dst[14] = 0; dst[15] = 0;
                emit_idx += 1u;
            }
            if (rc_match && emit_idx < (uint32_t)PS_MAX_EMIT_IDX) {
                uint8_t* dst = partial_out + 4 +
                               (size_t)emit_idx * (size_t)PS_RECORD_BYTES;
                uint32_t qp = (uint32_t)kmer_start;
                dst[0] = (uint8_t)(qp >> 0);
                dst[1] = (uint8_t)(qp >> 8);
                dst[2] = (uint8_t)(qp >> 16);
                dst[3] = (uint8_t)(qp >> 24);
                dst[4] = (uint8_t)1;  // strand = rc
                dst[5] = (uint8_t)0;  // primer_idx
                dst[6] = 0; dst[7] = 0;
                dst[8]  = 0; dst[9]  = 0; dst[10] = 0; dst[11] = 0;
                dst[12] = 0; dst[13] = 0; dst[14] = 0; dst[15] = 0;
                emit_idx += 1u;
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
// Per-build single-P entry point.
// PRIMER_P_ACTIVE is set by the Makefile to one of {13, 20, 25}.
// =====================================================================

#ifndef PRIMER_P_ACTIVE
#define PRIMER_P_ACTIVE 13  // default for standalone compile probes
#endif
static_assert(PRIMER_P_ACTIVE == 13 ||
              PRIMER_P_ACTIVE == 20 ||
              PRIMER_P_ACTIVE == 25,
              "PRIMER_P_ACTIVE must be 13, 20, or 25");

extern "C" {

#if PRIMER_P_ACTIVE == 13
void primer_scan_tile_p13(uint8_t* __restrict packed_in,
                          uint8_t* __restrict partial_out,
                          int32_t n_input_bytes,
                          int32_t tile_idx,
                          int32_t n_tiles_log2) {
    primer_scan_tile_impl<13, PRIMER_MASK_P13>(
        packed_in, partial_out, n_input_bytes, tile_idx, n_tiles_log2);
}
#endif

#if PRIMER_P_ACTIVE == 20
void primer_scan_tile_p20(uint8_t* __restrict packed_in,
                          uint8_t* __restrict partial_out,
                          int32_t n_input_bytes,
                          int32_t tile_idx,
                          int32_t n_tiles_log2) {
    primer_scan_tile_impl<20, PRIMER_MASK_P20>(
        packed_in, partial_out, n_input_bytes, tile_idx, n_tiles_log2);
}
#endif

#if PRIMER_P_ACTIVE == 25
void primer_scan_tile_p25(uint8_t* __restrict packed_in,
                          uint8_t* __restrict partial_out,
                          int32_t n_input_bytes,
                          int32_t tile_idx,
                          int32_t n_tiles_log2) {
    primer_scan_tile_impl<25, PRIMER_MASK_P25>(
        packed_in, partial_out, n_input_bytes, tile_idx, n_tiles_log2);
}
#endif

}  // extern "C"
