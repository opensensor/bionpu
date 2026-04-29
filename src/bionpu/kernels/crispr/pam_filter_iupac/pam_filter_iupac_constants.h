// pam_filter_iupac_constants.h
//
// Track A v0 — Multi-PAM (IUPAC) filter for base editor design.
//
// Mirrors the primer_scan v0 wire format with one extension:
//
//   - Per-position 4-bit IUPAC mask (A=bit0, C=bit1, G=bit2, T=bit3).
//   - PAM length is variable (3..PAM_LEN_MAX); concrete PAMs span
//     SpCas9-NGG (3), SpCas9-NG (2; padded), SpRY-NRN (3), and
//     SaCas9-KKH-NNNRRT (6).
//   - Single xclbin handles every Cas9 variant via a runtime header
//     (pam_mask + pam_length).
//
// IUPAC encoding (4-bit one-hot per position):
//
//   N -> 0xF (any)
//   A -> 0x1
//   C -> 0x2
//   G -> 0x4
//   T -> 0x8
//   R -> 0x5 (A|G)
//   Y -> 0xA (C|T)
//   S -> 0x6 (G|C)
//   W -> 0x9 (A|T)
//   K -> 0xC (G|T)
//   M -> 0x3 (A|C)
//   B -> 0xE (C|G|T)
//   D -> 0xD (A|G|T)
//   H -> 0xB (A|C|T)
//   V -> 0x7 (A|C|G)
//
// pam_mask: uint32 -- packed 4-bit nibbles, position i in bits [4*i .. 4*i+3].
// pam_length: uint8 -- number of active PAM positions (3..PAM_LEN_MAX).
//
// Per-position match check (pseudo):
//
//     uint8_t base_2bit = (packed_input[byte_idx] >> shift) & 0x3;
//     uint8_t base_onehot = (uint8_t)(1u << base_2bit);  // A=1 C=2 G=4 T=8
//     uint8_t pos_mask = (pam_mask >> (4 * pos_idx)) & 0xF;
//     if ((base_onehot & pos_mask) == 0) { match = false; break; }

#ifndef BIONPU_PAM_FILTER_IUPAC_CONSTANTS_H
#define BIONPU_PAM_FILTER_IUPAC_CONSTANTS_H

#include <stdint.h>

// 2-bit base codes (matches kmer_count / minimizer / primer_scan).
static constexpr uint8_t PFI_BASE_A = 0x0;
static constexpr uint8_t PFI_BASE_C = 0x1;
static constexpr uint8_t PFI_BASE_G = 0x2;
static constexpr uint8_t PFI_BASE_T = 0x3;

// Maximum supported PAM length (build-time cap; runtime can use any
// pam_length in [1..PAM_LEN_MAX]). 8 covers SaCas9-KKH-NNNRRT (6) plus
// headroom for future variants.
static constexpr int32_t PFI_PAM_LEN_MAX = 8;

// Streaming chunk + overlap (4-byte aligned for aiecc dma_bd).
// Overlap covers the prior chunk's tail (PAM_LEN_MAX - 1) bases. We use
// 8 bytes for parity with primer_scan.
static constexpr int32_t PFI_SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t PFI_SEQ_IN_OVERLAP = 8;

// Per-tile output element size: 32 KiB (mirrors primer_scan).
//
// Layout per slot (16 bytes per record):
//   [0..3]:                                uint32 emit_count_LE prefix
//   [4 .. 4 + 16 * emit_count]:            emit_count × Record
//   [middle zero pad]
// where Record = { uint32 query_pos_LE; uint8 strand; uint8 _pad;
//                  uint16 _pad2; uint64 _pad3 }.
//
// emit cap: floor((32768 - 4) / 16) = 2047. Round to 2046 for safety.
static constexpr int32_t PFI_PARTIAL_OUT_BYTES_PADDED = 32768;
static constexpr int32_t PFI_RECORD_BYTES = 16;
static constexpr int32_t PFI_MAX_EMIT_IDX = 2046;

// In-band header is 24 bytes (mirrors primer_scan):
//   bytes [0..3]:   uint32 LE pam_mask  (packed 4-bit IUPAC nibbles)
//   bytes [4]:      uint8 pam_length    (1..PFI_PAM_LEN_MAX)
//   bytes [5]:      uint8 _pad0
//   bytes [6..7]:   uint16 _pad1
//   bytes [8..15]:  uint64 _reserved (zero; future flags)
//   bytes [16..19]: uint32 LE actual_payload_bytes
//   bytes [20..23]: int32  LE owned_start_offset_bases
// Payload starts at byte 24.
static constexpr int32_t PFI_HEADER_BYTES = 24;

#endif  // BIONPU_PAM_FILTER_IUPAC_CONSTANTS_H
