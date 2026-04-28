// minimizer_constants.h
//
// Per the v0 design doc — sliding-window canonical (w, k) minimizer on
// AIE2P. This header pins the wire format, per-(k, w) overlap, and
// output buffer geometry so the tile kernel, IRON Python lowering, host
// runner, and op-class agree byte-equal.
//
// Pinned configurations (matches silicon artifacts):
//   - (k=15, w=10) — short-read default.
//   - (k=21, w=11) — long-read default.

#ifndef BIONPU_MINIMIZER_CONSTANTS_H
#define BIONPU_MINIMIZER_CONSTANTS_H

#include <stdint.h>

// 2-bit base codes (matches kmer_count and pam_filter).
static constexpr uint8_t MZ_BASE_A = 0x0;
static constexpr uint8_t MZ_BASE_C = 0x1;
static constexpr uint8_t MZ_BASE_G = 0x2;
static constexpr uint8_t MZ_BASE_T = 0x3;

// Per-k canonical mask. Apply on EVERY rolling update so forward and
// reverse-complement uint64 registers stay within k*2 bits.
static constexpr uint64_t MINIMIZER_MASK_K15 = (1ULL << 30) - 1ULL;
static constexpr uint64_t MINIMIZER_MASK_K21 = (1ULL << 42) - 1ULL;

// Streaming chunk + per-(k,w) overlap. The overlap MUST cover the tail
// (w + k - 1) bases of the prior chunk so a minimizer whose k-mer
// straddles the boundary is recomputed correctly. The +HEADER_BYTES
// (8) is consumed by the in-band per-chunk header (uint32 actual_bytes
// + int32 owned_start_offset_bases).
//
// Total chunk size MUST be a multiple of 4 bytes so aiecc accepts the
// aie.dma_bd. We round overlap up to the nearest 4-byte boundary.
//
// (k=15, w=10): w + k - 1 = 24 bases = 6 bytes → round to 8.
// (k=21, w=11): w + k - 1 = 31 bases = 8 bytes → 8 (already aligned).
static constexpr int32_t MZ_SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t MZ_SEQ_IN_OVERLAP_K15_W10 = 8;  // covers 24 bases
static constexpr int32_t MZ_SEQ_IN_OVERLAP_K21_W11 = 8;  // covers 31 bases (32 bases ok)

// Per-tile output element size: 32 KiB (mirrors kmer_count's
// PARTIAL_OUT_BYTES_V05_PADDED for shim DMA budget alignment).
//
// Layout per slot (16 bytes per record):
//   [0..3]:                                uint32 emit_count_LE prefix
//   [4 .. 4 + 16 * emit_count]:            emit_count × Record
//   [middle zero pad]
// where Record = { uint64 canonical_LE; uint32 position_LE; uint32 _pad; }.
//
// emit_count cap: floor((32768 - 4) / 16) = 2047. Round down to 2046
// for safety (tail can carry future summary fields without breaking).
static constexpr int32_t MZ_PARTIAL_OUT_BYTES_PADDED = 32768;
static constexpr int32_t MZ_RECORD_BYTES = 16;
static constexpr int32_t MZ_MAX_EMIT_IDX = 2046;

// In-band header is 8 bytes (mirrors kmer_count v1.2 (a)):
//   bytes [0..3]: uint32 LE actual_payload_bytes
//   bytes [4..7]: int32  LE owned_start_offset_bases (per-chunk;
//                 chunk 0 = 0; chunk i (i>0) = overlap_bases - (k+w-2)).
// Payload starts at byte 8.
//
// owned-range gate semantics: a minimizer-emitting position is OWNED by
// the chunk if and only if the k-mer that triggered the emit (the
// CURRENT k-mer when emitting via strict-improvement OR the SCAN-WINNER
// k-mer when emitting via slide-out) has its start position
// >= owned_start_offset_bases. Without this gate, chunks would emit
// duplicate minimizers in the (w + k - 1) base overlap region.
static constexpr int32_t MZ_HEADER_BYTES = 8;

#endif  // BIONPU_MINIMIZER_CONSTANTS_H
