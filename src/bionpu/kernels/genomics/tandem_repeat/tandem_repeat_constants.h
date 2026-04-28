// tandem_repeat_constants.h
//
// v0 design — autocorrelation-based short tandem repeat (STR) detector.
// Mirrors the kmer_count + minimizer + primer_scan + cpg_island wire
// format with a per-period streak-counter emit.
//
//   - MSB-first packed-2-bit DNA (A=00, C=01, G=10, T=11).
//   - 8-byte in-band chunk header (mirrors kmer_count v1.2(a)):
//       bytes [0..3]:    uint32 actual_payload_bytes LE
//       bytes [4..7]:    int32  owned_start_offset_bases LE
//     Payload starts at byte 8.
//   - 4096-byte primary chunk + 12-byte overlap (covers MAX_PERIOD *
//     MIN_COPIES + 6 = 36 bases worst-case, rounded to 4-byte boundary).
//
// EMIT GRANULARITY: for each period in [MIN_PERIOD, MAX_PERIOD], the
// kernel maintains a running streak counter:
//
//   streak[q] = consecutive bases at the current write position whose
//               base equals the base `q` positions back.
//
// When the streak breaks (or end-of-chunk is reached), the kernel emits
// a record IFF the streak was >= q * (MIN_COPIES - 1) — i.e., enough
// bases matched their predecessor at distance q to imply MIN_COPIES
// consecutive identical motifs.
//
// Record layout (16 bytes / record):
//   bytes [0..3]:   uint32 LE start (chunk-local; host adds chunk
//                   global base to translate)
//   bytes [4..7]:   uint32 LE end (exclusive)
//   bytes [8..11]:  uint32 LE period (1..6)
//   bytes [12..15]: uint32 LE motif_canonical (period * 2 used bits,
//                   MSB-first)
//
// Host-side dedup: identical to the CPU oracle's "longer wins, then
// smaller period wins" greedy pass.

#ifndef BIONPU_TANDEM_REPEAT_CONSTANTS_H
#define BIONPU_TANDEM_REPEAT_CONSTANTS_H

#include <stdint.h>

// 2-bit base codes (matches sibling kernels).
static constexpr uint8_t TR_BASE_A = 0x0;
static constexpr uint8_t TR_BASE_C = 0x1;
static constexpr uint8_t TR_BASE_G = 0x2;
static constexpr uint8_t TR_BASE_T = 0x3;

// Period bounds.  v0 supports mono- through hexa-nucleotide repeats.
static constexpr int32_t TR_MIN_PERIOD = 1;
static constexpr int32_t TR_MAX_PERIOD = 6;

// Minimum copy count to emit a record (matches TRF default for short
// periods).
static constexpr int32_t TR_MIN_COPIES = 5;

// Streaming chunk + overlap.
// Worst-case STR detection latency: a streak of length
// (MAX_PERIOD * MIN_COPIES) - 1 = 29 bases must land entirely within
// one chunk's owned range to be detected. Overlap of 36 bases = 9 bytes
// rounded up to 12 bytes for 4-byte alignment.
//
// v0 scope: assume max STR length <= chunk payload (4096 B = 16384 bp);
// chr22 longest STR is ~10 kbp at the rare disease loci, which is
// covered by 16384 bp. Pathological STRs > chunk payload are a v1
// follow-on (chunk-spanning streak merging).
static constexpr int32_t TR_SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t TR_SEQ_IN_OVERLAP = 12;

// Per-tile output element size: 32 KiB (mirrors siblings for shim DMA
// budget alignment).
//
// Layout per slot:
//   [0..3]:                              uint32 emit_count_LE prefix
//   [4 .. 4 + 16 * emit_count]:          emit_count × Record (16 B)
//   [middle zero pad]
//
// emit cap: floor((32768 - 4) / 16) = 2047. Round to 2046 for safety.
//
// chr22 has ~1000 STR loci end-to-end; each chunk's STR density is way
// below 2046, so cap-fire is extremely unlikely. v1 follow-up if it
// fires: shrink chunk payload or widen partial_out_bytes to 64 KiB.
static constexpr int32_t TR_PARTIAL_OUT_BYTES_PADDED = 32768;
static constexpr int32_t TR_RECORD_BYTES = 16;
static constexpr int32_t TR_MAX_EMIT_IDX = 2046;

// In-band header is 8 bytes (mirrors kmer_count + minimizer).
//   bytes [0..3]: uint32 LE actual_payload_bytes
//   bytes [4..7]: int32  LE owned_start_offset_bases (per-chunk;
//                 chunk 0 = 0; chunk i (i>0) = overlap_bases - 1).
// Payload starts at byte 8.
static constexpr int32_t TR_HEADER_BYTES = 8;

#endif  // BIONPU_TANDEM_REPEAT_CONSTANTS_H
