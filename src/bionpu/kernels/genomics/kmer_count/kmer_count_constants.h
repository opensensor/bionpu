// kmer_count_constants.h
//
// Per state/kmer_count_interface_contract.md (T1) v0.5 — symbols,
// ObjectFifo names, and constants pinned by the contract's
// "v0.5 REDESIGN — Streaming + Multi-Pass" section.
//
// v0.5 supersedes the in-tile-counting design. Tile kernels just stream-
// emit canonical k-mers that fall into the active hash slice; ALL counting
// happens host-side (std::unordered_map<uint64_t, uint64_t>). N_PASSES
// partitions the canonical hash space across passes so per-pass output
// volume stays inside the shim DMA + tile DM budget.

#ifndef BIONPU_KMER_COUNT_CONSTANTS_H
#define BIONPU_KMER_COUNT_CONSTANTS_H

#include <stdint.h>

// 2-bit base codes (matches CRISPR pam_filter tile_a_filter.cc:66-69
// and the T2 numpy oracle's pack_dna_2bit).
static constexpr uint8_t KMER_BASE_A = 0x0;
static constexpr uint8_t KMER_BASE_C = 0x1;
static constexpr uint8_t KMER_BASE_G = 0x2;
static constexpr uint8_t KMER_BASE_T = 0x3;

// Per-k canonical mask. Apply on EVERY rolling update so forward and
// reverse-complement uint64 registers stay within k*2 bits. CRITICAL
// for k=31 (62-bit width) — without the mask the high bits leak and
// canonical = min(forward, rc) silently corrupts.
static constexpr uint64_t KMER_MASK_K15 = (1ULL << 30) - 1ULL;
static constexpr uint64_t KMER_MASK_K21 = (1ULL << 42) - 1ULL;
static constexpr uint64_t KMER_MASK_K31 = (1ULL << 62) - 1ULL;

// Streaming chunk + 4-byte alignment. The host (T7 runner) dispatches
// the input in 4096-byte chunks with overlap (per-k) so k-mers spanning
// chunk boundaries aren't lost. Total chunk_bytes + overlap_bytes MUST
// be a multiple of 4 because aiecc rejects non-4-aligned `aie.dma_bd`.
static constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t SEQ_IN_OVERLAP_K15 = 4;   // 4096+4 = 4100, aligned
static constexpr int32_t SEQ_IN_OVERLAP_K21 = 8;   // 4096+8 = 4104, aligned (need >=5, rounded up)
static constexpr int32_t SEQ_IN_OVERLAP_K31 = 8;   // 4096+8 = 4104, aligned

// =====================================================================
// v0.5 streaming-emit constants.
// =====================================================================
//
// Per-pass emit format (written by tile kernel into partial_out):
//   [uint32 emit_idx LE prefix]
//   [emit_idx × uint64 canonical]
//   [zero pad to PARTIAL_OUT_BYTES_V05_PADDED]
//
// MAX_EMIT_IDX_V05 caps the per-pass canonical count so the writeback
// stays within PARTIAL_OUT_BYTES_V05_PADDED (32 KiB). The kernel asserts
// emit_idx < MAX_EMIT_IDX_V05 and silently drops further emissions; this
// only fires at N_PASSES=1 (no slice partitioning) with chunks producing
// > MAX_EMIT_IDX_V05 k-mers. The host runner re-dispatches with smaller
// chunks or larger N_PASSES if the cap fires (runtime check in T7).
//
// 32768 = 32 KiB; (32768 - 4) / 8 = 4095.5 → cap at 4095.
static constexpr int32_t PARTIAL_OUT_BYTES_V05_PADDED = 32768;  // 32 KiB
static constexpr int32_t MAX_EMIT_IDX_V05 = 4095;               // (32768 - 4) / 8 floor

// =====================================================================
// Hash-slice partition: SLICE_HASH_SHIFT = 0 (low bits of canonical).
// =====================================================================
//
// We use the LOW bits of `canonical` for slice membership. This is
// uniform across k (k=15 has only 30 canonical bits, so a high-bit
// shift of 32 would produce 0 for ALL canonicals at k=15 and break
// the partition for the smallest k). Low-bits is a standard hash-table
// bucket-index trick; the canonical is already a near-uniform hash of
// the k-mer's identity.
//
// Slice membership:
//   slice = canonical & ((1 << n_passes_log2) - 1)
//   include this canonical IFF slice == pass_idx
//
// Coverage is exact: every canonical maps to exactly one slice in
// [0, N_PASSES); no double-counting, no dropped k-mers. The host's
// merged unordered_map after N_PASSES dispatches equals the dense
// k-mer count exactly.
static constexpr int32_t SLICE_HASH_SHIFT = 0;

// =====================================================================
// N_PASSES support set: {1, 4, 16}. log2 ∈ {0, 2, 4}.
// =====================================================================
//
//   N_PASSES = 1   (log2 = 0)  : single pass; only viable for fixtures
//                                where chunk's k-mers ≤ MAX_EMIT_IDX_V05.
//   N_PASSES = 4   (log2 = 2)  : default; balanced (per-pass ≈ 1/4
//                                of chunk's k-mers).
//   N_PASSES = 16  (log2 = 4)  : tight memory; per-pass ≈ 1/16, finer
//                                dispatch granularity.
//
// At smoke 10 Kbp / k=21, chunk is 2500 bytes producing ~9980 k-mers;
// N_PASSES=1 emits 9980 (exceeds 4095 cap by ~5885), N_PASSES=4 emits
// ~2495 per pass (within cap), so N_PASSES≥4 is required for correctness.
static constexpr int32_t N_PASSES_VALUES_LOG2[3] = {0, 2, 4};

#endif  // BIONPU_KMER_COUNT_CONSTANTS_H
