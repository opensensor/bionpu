// primer_scan_constants.h
//
// v0 design — exact-match scan for a single primer against a packed-2-bit
// query. Mirrors the wire format used by kmer_count and minimizer with
// one extension:
//
//   - MSB-first packed-2-bit DNA (A=00, C=01, G=10, T=11).
//   - 24-byte in-band chunk header (mirrors v1.2(a)'s 8-byte header
//     extended with the primer canonical:
//       bytes [0..3]:    uint32 actual_payload_bytes LE
//       bytes [4..7]:    int32  owned_start_offset_bases LE
//       bytes [8..15]:   uint64 primer_fwd_canonical LE
//       bytes [16..23]:  uint64 primer_rc_canonical  LE
//     Payload starts at byte 24).
//   - 4096-byte primary chunk + per-P overlap.
//
// Path B (runtime primer): the primer canonical lives in the chunk
// header, NOT in kernel scalar args, so each dispatch can scan a
// different primer without rebuilding the xclbin. The build-time
// constant is the primer LENGTH (P, from -DPRIMER_P_ACTIVE) — that
// folds the per-P mask and RC_HIGH_SHIFT into compile-time constants.
//
// v0 supported primer lengths:
//   P = 13 (Illumina TruSeq P5 adapter; default smoke target)
//   P = 20 (typical PCR primer)
//   P = 25 (qPCR primer)

#ifndef BIONPU_PRIMER_SCAN_CONSTANTS_H
#define BIONPU_PRIMER_SCAN_CONSTANTS_H

#include <stdint.h>

// 2-bit base codes (matches kmer_count and minimizer).
static constexpr uint8_t PS_BASE_A = 0x0;
static constexpr uint8_t PS_BASE_C = 0x1;
static constexpr uint8_t PS_BASE_G = 0x2;
static constexpr uint8_t PS_BASE_T = 0x3;

// Per-P mask (P*2 bits set). Build-time constants per active P.
static constexpr uint64_t PRIMER_MASK_P13 = (1ULL << 26) - 1ULL;
static constexpr uint64_t PRIMER_MASK_P20 = (1ULL << 40) - 1ULL;
static constexpr uint64_t PRIMER_MASK_P25 = (1ULL << 50) - 1ULL;

// Streaming chunk + overlap (4-byte aligned for aiecc dma_bd).
// Overlap covers the prior chunk's tail (P - 1) bases. 8 bytes covers
// up to P=33 -- enough for v0's P<=25.
//
// NOTE: the v0 kernel uses 8-byte overlap regardless of P; this gives
// uniform shim DMA payload sizing across all P-builds.
static constexpr int32_t PS_SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t PS_SEQ_IN_OVERLAP = 8;

// Per-tile output element size: 32 KiB (mirrors kmer_count and
// minimizer for shim DMA budget alignment).
//
// Layout per slot (16 bytes per record):
//   [0..3]:                                uint32 emit_count_LE prefix
//   [4 .. 4 + 16 * emit_count]:            emit_count × Record
//   [middle zero pad]
// where Record = { uint32 query_pos_LE; uint8 strand; uint8 primer_idx;
//                  uint16 _pad; uint64 _pad2 }.
//
// emit cap: floor((32768 - 4) / 16) = 2047. Round to 2046 for safety.
static constexpr int32_t PS_PARTIAL_OUT_BYTES_PADDED = 32768;
static constexpr int32_t PS_RECORD_BYTES = 16;
static constexpr int32_t PS_MAX_EMIT_IDX = 2046;

// In-band header is 24 bytes (mirrors kmer_count + minimizer's 8 bytes
// + 16 bytes primer canonical pair):
//   bytes [0..3]:   uint32 LE actual_payload_bytes
//   bytes [4..7]:   int32  LE owned_start_offset_bases (per-chunk;
//                   chunk 0 = 0; chunk i (i>0) = overlap_bases - (P-1)).
//   bytes [8..15]:  uint64 LE primer_fwd_canonical.
//   bytes [16..23]: uint64 LE primer_rc_canonical.
// Payload starts at byte 24.
static constexpr int32_t PS_HEADER_BYTES = 24;

#endif  // BIONPU_PRIMER_SCAN_CONSTANTS_H
