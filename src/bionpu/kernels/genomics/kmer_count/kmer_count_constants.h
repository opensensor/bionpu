// kmer_count_constants.h
// PINNED by state/kmer_count_interface_contract.md (T1) — DO NOT EDIT
// without updating the contract doc + every consumer in lockstep.
//
// Per state/kmer_count_interface_contract.md (T1) — symbols, ObjectFifo
// names, and constants pinned there. This header is included verbatim by
// kmer_count_tile.cc (T5) and kmer_count_aggregator.cc (T6).

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

// Reverse-complement XOR constant. RC of a 2-bit base = base ^ 0x3
// (A<->T = 0b00 <-> 0b11; C<->G = 0b01 <-> 0b10). For a packed k-mer
// we XOR with the per-k all-ones mask AND bit-reverse the 2-bit
// pairs. The packed RC mask (apply XOR before bit-reverse) for k bases
// is exactly the per-k mask above (every 2-bit pair flipped).

// Per-tile count-table geometry. 4096 buckets * 12 bytes = 48 KiB.
// Fits in 64 KiB AIE2P L1 with ~16 KiB headroom for stack + fifo
// double-buffers + code.
static constexpr int32_t HASH_BUCKETS_PER_TILE = 4096;

// Open-addressed linear probing — emit-on-evict overflow.
static constexpr int32_t OVERFLOW_THRESHOLD = 8;

// CountRecord layout (per-bucket entry in the on-tile count table).
// 12 bytes packed; static_assert in tile.cc.
struct CountRecord {
    uint64_t canonical;   // 8 bytes — canonical k-mer value
    uint32_t count;       // 4 bytes — observed count
};

// Sparse-emit record (length-prefixed payload of partial_count_<i>
// and sparse_out ObjectFifos). 16 bytes; the extra 4 bytes (vs. the
// CRISPR 8-byte record) are EVICT_FLAG + reserved padding for the
// host re-aggregation pass.
static constexpr int32_t EMIT_RECORD_BYTES = 16;

struct EmitRecord {
    uint64_t canonical;     // 8 bytes
    uint32_t count;         // 4 bytes
    uint32_t flags;         // 4 bytes — bit 0: EVICT_FLAG; bits 31:1 reserved (0)
};

static constexpr uint32_t EVICT_FLAG = 1u << 0;

// Sparse-emit ring slot — mirrors pam_filter post-fix EMIT_SLOT_RECORDS=1024.
// 1024 records * 16 bytes = 16 KiB per slot; double-buffered = 32 KiB
// on Tile Z. Tile Z sits below the 48 KiB count-table cap because it
// drains aggregator-side, not table-side.
static constexpr int32_t EMIT_SLOT_RECORDS = 1024;
static constexpr int32_t EMIT_SLOT_BYTES = EMIT_SLOT_RECORDS * EMIT_RECORD_BYTES; // 16384

// Streaming chunk + overlap protocol. The host (T7 runner) dispatches
// the input in 4096-byte chunks with (k-1)/4 (rounded up) bytes of
// overlap between consecutive chunks. Per-k overlap is also enumerated
// for include-time pin.
static constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t SEQ_IN_OVERLAP_K15 = 4;   // ceil((15-1)/4) = 4
static constexpr int32_t SEQ_IN_OVERLAP_K21 = 5;   // ceil((21-1)/4) = 5
static constexpr int32_t SEQ_IN_OVERLAP_K31 = 8;   // ceil((31-1)/4) = 8

// Aggregator fan-in cap. n_tiles is one of {1, 2, 4, 8}; aggregator
// signature widens to 8 partial inputs and zeros unused ones at host
// dispatch time.
static constexpr int32_t MAX_TILES = 8;

#endif // BIONPU_KMER_COUNT_CONSTANTS_H
