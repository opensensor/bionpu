// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// pam_filter_iupac_tile.cc — per-tile multi-PAM IUPAC scan kernel (v0).
//
// Algorithm:
//
//   For each new ACGT base entering at index i:
//     1. Push base into a rolling PAM_LEN_MAX-base window (2 bits/base
//        in a uint16 register).
//     2. valid_run += 1.
//     3. If valid_run < pam_length: continue.
//     4. pam_start = i - (pam_length - 1).
//     5. Owned-range gate: pam_start >= owned_start_offset_bases.
//     6. For each position p in [0, pam_length):
//          base_at_p   = (window >> (2 * (pam_length - 1 - p))) & 0x3
//          base_onehot = 1u << base_at_p   (A=1, C=2, G=4, T=8)
//          pos_mask    = (pam_mask >> (4 * p)) & 0xF
//          if ((base_onehot & pos_mask) == 0): no match; break.
//     7. If matched: emit (pam_start, strand=0).
//
// v0 emits only the forward strand. Reverse-complement is delegated to
// the host (mirrors the locked crispr/pam_filter design — see
// crispr/pam_filter/pam_filter.py DESIGN.md §3 "strand handling").
//
// Output record layout (16 bytes/record):
//   bytes [0..3]:   uint32 LE query_pos (pam_start within chunk
//                   payload; host translates to global by adding
//                   chunk's source offset).
//   bytes [4]:      uint8 strand (0 = forward; v0 always 0).
//   bytes [5]:      uint8 _pad0.
//   bytes [6..7]:   uint16 _pad1.
//   bytes [8..15]:  uint64 _pad2.

#include <stdint.h>

#include "pam_filter_iupac_constants.h"

#if defined(__AIE_ARCH__) && !defined(BIONPU_FORCE_SCALAR)
#  define BIONPU_HAS_AIE_API 1
#  include "aie_kernel_utils.h"
#  include <aie_api/aie.hpp>
#else
#  define BIONPU_HAS_AIE_API 0
#endif

static_assert(PFI_PARTIAL_OUT_BYTES_PADDED == 32768,
              "partial_out element must be 32 KiB");
static_assert(PFI_RECORD_BYTES == 16, "Record = 16 bytes");
static_assert(PFI_MAX_EMIT_IDX == 2046,
              "max emit_idx caps at 2046 to fit in 32 KiB - 4 prefix");
static_assert(PFI_HEADER_BYTES == 24,
              "in-band header pinned at 24 bytes");
static_assert(PFI_PAM_LEN_MAX == 8,
              "PAM_LEN_MAX pinned to 8 (SaCas9-KKH NNNRRT + headroom)");

namespace {

// Rolling window stored as a uint32 to leave headroom for PAM_LEN_MAX up
// to 16 if a future variant needs it. Only the low 2*pam_length bits
// are actively compared.
static inline void primer_window_push(uint32_t &win, uint8_t base_2bit,
                                      uint32_t pam_mask_full) {
    win = ((win << 2) | (uint32_t)base_2bit) & pam_mask_full;
}

// pam_mask: 4 bits per position, position 0 in bits [0..3], position
// pam_length-1 in bits [4*(pam_length-1) .. 4*pam_length-1].
//
// Per-position check: base_at_p (2 bits) is the rolling window's 2-bit
// lane corresponding to position p. Position 0 is the FIRST PAM base
// (5'-most), so it lives at the highest 2-bit lane of the window:
//   base_at_p_2bit = (win >> (2 * (pam_length - 1 - p))) & 0x3
static inline bool pam_iupac_match(uint32_t win,
                                   uint32_t pam_mask,
                                   int32_t pam_length) {
    for (int32_t p = 0; p < pam_length; ++p) {
        uint8_t base_2bit = (uint8_t)((win >> (2 * (pam_length - 1 - p))) & 0x3);
        uint8_t base_onehot = (uint8_t)(1u << base_2bit);  // A=1 C=2 G=4 T=8
        uint8_t pos_mask = (uint8_t)((pam_mask >> (4 * p)) & 0xF);
        if ((base_onehot & pos_mask) == 0u) return false;
    }
    return true;
}

static inline uint32_t load_u32_le(const uint8_t* p) {
    uint32_t v = 0;
    v |= ((uint32_t)p[0]) <<  0;
    v |= ((uint32_t)p[1]) <<  8;
    v |= ((uint32_t)p[2]) << 16;
    v |= ((uint32_t)p[3]) << 24;
    return v;
}

}  // anonymous namespace

extern "C" {

// Single entry point. PAM length is a runtime header arg, not a build
// constant — one xclbin serves every Cas9 variant.
//
// Args (mirrors primer_scan):
//   packed_in       : 4096-byte chunk slot (24-byte header + payload)
//   partial_out     : 32 KiB output slot
//   n_input_bytes   : full BO size including header (informational)
//   tile_idx        : unused for v0 (broadcast topology)
//   n_tiles_log2    : unused for v0
void pam_filter_iupac_tile(uint8_t* __restrict packed_in,
                           uint8_t* __restrict partial_out,
                           int32_t n_input_bytes,
                           int32_t /*tile_idx*/,
                           int32_t /*n_tiles_log2*/) {
    uint32_t emit_idx = 0u;

    // ----- Header decode -----
    // Bytes [0..3]: pam_mask (uint32 LE)
    // Bytes [4]:    pam_length (uint8)
    // Bytes [5..15]: padding/reserved
    // Bytes [16..19]: actual_payload_bytes (uint32 LE)
    // Bytes [20..23]: owned_start_offset_bases (int32 LE)
    uint32_t pam_mask = load_u32_le(packed_in + 0);
    int32_t  pam_length = (int32_t)packed_in[4];

    int32_t actual_bytes = (int32_t)load_u32_le(packed_in + 16);
    int32_t owned_start_offset_bases = (int32_t)load_u32_le(packed_in + 20);

    if (pam_length < 1)            pam_length = 1;
    if (pam_length > PFI_PAM_LEN_MAX) pam_length = PFI_PAM_LEN_MAX;
    if (owned_start_offset_bases < 0) owned_start_offset_bases = 0;
    if (actual_bytes < 0) actual_bytes = 0;
    if (actual_bytes > n_input_bytes - PFI_HEADER_BYTES) {
        actual_bytes = n_input_bytes - PFI_HEADER_BYTES;
    }

    const uint32_t window_mask =
        (pam_length >= 16) ? 0xFFFFFFFFu
                           : (((uint32_t)1u << (2 * pam_length)) - 1u);

    // Mask the IUPAC mask to the active positions (defensive — host is
    // expected to zero unused nibbles, but we enforce it on tile so
    // wrapping-around bits can't accidentally match).
    {
        const uint32_t mask_active =
            (pam_length >= 8) ? 0xFFFFFFFFu
                              : (((uint32_t)1u << (4 * pam_length)) - 1u);
        pam_mask &= mask_active;
    }

    uint32_t win = 0u;
    int32_t  valid_run = 0;
    int32_t  pos = 0;

    for (int32_t i = 0; i < actual_bytes; ++i) {
        uint8_t byte = packed_in[PFI_HEADER_BYTES + i];
        for (int32_t shift = 6; shift >= 0; shift -= 2) {
            uint8_t base = (uint8_t)((byte >> shift) & 0x3);
            primer_window_push(win, base, window_mask);

            if (valid_run < pam_length) {
                valid_run += 1;
                if (valid_run < pam_length) {
                    pos += 1;
                    continue;
                }
                // Just reached the first full PAM-mer; fall through.
            }

            int32_t pam_start = pos - (pam_length - 1);
            // Owned-range gate.
            if (pam_start < owned_start_offset_bases) {
                pos += 1;
                continue;
            }

            if (pam_iupac_match(win, pam_mask, pam_length)
                && emit_idx < (uint32_t)PFI_MAX_EMIT_IDX) {
                uint8_t* dst = partial_out + 4 +
                               (size_t)emit_idx * (size_t)PFI_RECORD_BYTES;
                uint32_t qp = (uint32_t)pam_start;
                dst[0] = (uint8_t)(qp >> 0);
                dst[1] = (uint8_t)(qp >> 8);
                dst[2] = (uint8_t)(qp >> 16);
                dst[3] = (uint8_t)(qp >> 24);
                dst[4] = (uint8_t)0;  // strand = forward (v0 host does RC)
                dst[5] = (uint8_t)0;
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

}  // extern "C"
