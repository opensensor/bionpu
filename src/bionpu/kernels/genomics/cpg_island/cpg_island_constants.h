// cpg_island_constants.h
//
// v0 design — Gardiner-Garden & Frommer (1987) sliding-window CpG
// island detector. Mirrors the kmer_count + minimizer + primer_scan
// wire format with a counter-based per-position emit (as opposed to
// substring/canonical-match emits).
//
//   - MSB-first packed-2-bit DNA (A=00, C=01, G=10, T=11).
//   - 8-byte in-band chunk header (mirrors kmer_count v1.2(a) + minimizer):
//       bytes [0..3]:    uint32 actual_payload_bytes LE
//       bytes [4..7]:    int32  owned_start_offset_bases LE
//     Payload starts at byte 8.
//   - 4096-byte primary chunk + W-byte (= W/4 bytes, 50 for W=200) overlap.
//
// v0 EMIT GRANULARITY: per-base CANDIDATE record. The kernel emits one
// uint32 record per input position whose length-W window starting at
// that position satisfies BOTH thresholds. Host-side merges contiguous
// candidate run-lengths into (island_start, island_end) intervals and
// applies the run_len >= W island criterion.
//
// FIXED-POINT THRESHOLDS (silicon avoids float):
//   GC%    >= 0.5  <=>  2 * (n_C + n_G) >= W
//   obs/exp >= 0.6 <=>  5 * W * n_CG >= 3 * n_C * n_G
// These are pure integer comparisons; no float on silicon.

#ifndef BIONPU_CPG_ISLAND_CONSTANTS_H
#define BIONPU_CPG_ISLAND_CONSTANTS_H

#include <stdint.h>

// 2-bit base codes (matches kmer_count, minimizer, primer_scan).
static constexpr uint8_t CI_BASE_A = 0x0;
static constexpr uint8_t CI_BASE_C = 0x1;
static constexpr uint8_t CI_BASE_G = 0x2;
static constexpr uint8_t CI_BASE_T = 0x3;

// Gardiner-Garden window length (default = 200; UCSC convention).
static constexpr int32_t CI_W = 200;

// Fixed-point threshold numerators / denominators.
//   GC%    threshold = CI_GC_NUM / CI_GC_DEN     (default 1/2 = 0.5)
//   obs/exp threshold = CI_OE_NUM  / CI_OE_DEN    (default 3/5 = 0.6)
static constexpr int32_t CI_GC_NUM = 1;
static constexpr int32_t CI_GC_DEN = 2;
static constexpr int32_t CI_OE_NUM = 3;
static constexpr int32_t CI_OE_DEN = 5;

// Streaming chunk + overlap. Overlap MUST cover enough lookahead bases
// for windows starting near the end of a chunk's owned range. We round
// overlap up to a 4-byte boundary.
//
// W=200 -> ceil(W / 4) = 50 bytes -> rounded to 52. We use 52 directly
// as the overlap so the streaming chunk total (4096 + 52 = 4148) is
// 4-byte aligned.
static constexpr int32_t CI_SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t CI_SEQ_IN_OVERLAP = 52;  // ceil(200 / 4) rounded up to 4-byte multiple

// Per-tile output element size: 32 KiB (mirrors siblings for shim DMA
// budget alignment).
//
// Layout per slot:
//   [0..3]:                              uint32 emit_count_LE prefix
//   [4 .. 4 + 4 * emit_count]:           emit_count × Record
//   [middle zero pad]
// where Record = uint32 LE position (per-base CANDIDATE window-start
// in chunk-local coordinates).
//
// emit cap: floor((32768 - 4) / 4) = 8191. Round to 8190 for safety.
//
// Per-chunk max possible candidate records: a chunk has at most 4096
// payload bytes = 16384 bases, but the maximum number of valid
// length-W window-start positions in 4096 payload bytes is 16384 - W +
// 1 = 16185. That's well over the 8190 cap, so high-density chunks
// (e.g. CpG-island-rich regions) MIGHT cap-fire.
//
// However: chr22 has ~700 CpG islands averaging ~1 kb each = ~700 kb
// of candidate density distributed over 51 Mbp. Per 16384-bp chunk
// average ~225 candidate starts. Worst-case chunks fully inside the
// largest island (~3 kb) would have ~16384 on-streak bases — that
// exceeds the cap. v0 mitigation: the kernel sets a "hard cap" emit
// counter; if cap fires, the chunk's emit list is truncated and the
// runner logs a warning. v1 follow-up: widen the cap by reducing
// chunk size (4096 -> 1024) or increasing partial_out_bytes (32 KiB ->
// 64 KiB) — same lever as minimizer's emit-cap-saturation gap.
//
// Per the strategy: v0 caps at 8190 records/chunk — chr22 is expected
// to fit comfortably. If cap-fire is observed, a v1 gap is filed.
static constexpr int32_t CI_PARTIAL_OUT_BYTES_PADDED = 32768;
static constexpr int32_t CI_RECORD_BYTES = 4;
static constexpr int32_t CI_MAX_EMIT_IDX = 8190;

// In-band header is 8 bytes (mirrors kmer_count + minimizer):
//   bytes [0..3]: uint32 LE actual_payload_bytes
//   bytes [4..7]: int32  LE owned_start_offset_bases (per-chunk;
//                 chunk 0 = 0; chunk i (i>0) = overlap_bases).
// Payload starts at byte 8.
//
// owned-range gate semantics: a candidate window-start position is
// OWNED by the chunk if and only if window_start >=
// owned_start_offset_bases. This suppresses duplicate emits in the
// chunk overlap region.
static constexpr int32_t CI_HEADER_BYTES = 8;

#endif  // BIONPU_CPG_ISLAND_CONSTANTS_H
