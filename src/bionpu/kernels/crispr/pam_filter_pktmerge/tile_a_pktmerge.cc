// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie2p/aie2p_streams.h>

#ifndef SPACER_BYTES
#define SPACER_BYTES 5
#endif
#ifndef N_GUIDES
#define N_GUIDES 128
#endif
#ifndef GUIDES_PER_TILE
#define GUIDES_PER_TILE 64
#endif
#ifndef N_MATCH_TILES
#define N_MATCH_TILES 2
#endif
#ifndef WINDOW_BYTES_IN
#define WINDOW_BYTES_IN 6
#endif
#ifndef WINDOWS_PER_CHUNK
#define WINDOWS_PER_CHUNK 64
#endif
#ifndef PAM_BYTES
#define PAM_BYTES 1
#endif
#ifndef EMIT_RECORD_BYTES
#define EMIT_RECORD_BYTES 8
#endif
#ifndef EMIT_SLOT_RECORDS
#define EMIT_SLOT_RECORDS 256
#endif
#ifndef PACKETIZED_WINDOW_WORDS
#define PACKETIZED_WINDOW_WORDS 3
#endif
#ifndef PACKETIZED_STREAM_WORDS_PER_CHUNK
#define PACKETIZED_STREAM_WORDS_PER_CHUNK \
    ((EMIT_SLOT_RECORDS * PACKETIZED_WINDOW_WORDS + 1) / 2)
#endif
#ifndef PACKETIZED_MATCH_RECORD_BYTES
#define PACKETIZED_MATCH_RECORD_BYTES 68
#endif
#ifndef COMPACT_PACKET_WINDOW_WORDS
#define COMPACT_PACKET_WINDOW_WORDS 2
#endif
#ifndef COMPACT_MATCH_RECORD_BYTES
#define COMPACT_MATCH_RECORD_BYTES 65
#endif

// Packet routing tags (1-bit valid + 7-bit reserved, packed in the uint8
// PacketFifo header — AM020 Ch. 2 p. 25's 5-bit pkt_id field carries
// PACKET_ID_VALID for surviving windows). Match the values declared in
// the IRON Python lowering.
static constexpr uint8_t PACKET_ID_VALID = 1;
static constexpr uint8_t PACKET_ID_INVALID = 0;
static constexpr uint32_t PACKETIZED_EMPTY_CHUNK_SENTINEL = 0xffffffffu;

// 2-bit codes for ACGT (matches / / encoding).
static constexpr uint8_t BASE_A = 0x0;
static constexpr uint8_t BASE_C = 0x1;
static constexpr uint8_t BASE_G = 0x2;
static constexpr uint8_t BASE_T = 0x3;

// PAM template for SpCas9 NGG: position 0 = wildcard (N, any), positions
// 1 and 2 must equal G (== 0b10 in our 2-bit code). Same as .
static inline bool pam_is_ngg(uint8_t pam_byte) {
    uint8_t p1 = (pam_byte >> 2) & 0x3;
    uint8_t p2 = (pam_byte >> 4) & 0x3;
    return (p1 == BASE_G) && (p2 == BASE_G);
}

// popcount of a byte — table-free; AIE2P scalar core handles this fine.
static inline int popcount_u8(uint8_t x) {
    x = (uint8_t)((x & 0x55) + ((x >> 1) & 0x55));
    x = (uint8_t)((x & 0x33) + ((x >> 2) & 0x33));
    x = (uint8_t)((x & 0x0F) + ((x >> 4) & 0x0F));
    return (int)x;
}

// One 20-nt mismatch count between guide[g_off:g_off+5] and window[w_off:w_off+5].
// Verbatim copy of / / — preserves byte-equality by construction.
static inline uint8_t mismatch_count_5b(const uint8_t* g, const uint8_t* w) {
    int total = 0;
    for (int b = 0; b < SPACER_BYTES; b++) {
        uint8_t x = (uint8_t)(g[b] ^ w[b]);
        uint8_t m = (uint8_t)(((x | (x >> 1)) & 0x55));
        total += popcount_u8(m);
    }
    return (uint8_t)total;
}

static inline uint32_t pack_u32_le(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static inline void unpack_spacer_words(uint32_t word1, uint32_t word2,
                                       uint8_t* spacer) {
    spacer[0] = (uint8_t)(word1 & 0xff);
    spacer[1] = (uint8_t)((word1 >> 8) & 0xff);
    spacer[2] = (uint8_t)((word1 >> 16) & 0xff);
    spacer[3] = (uint8_t)((word1 >> 24) & 0xff);
    spacer[4] = (uint8_t)(word2 & 0xff);
}

static inline void store_logical_word(int64_t* packed_words,
                                      int logical_idx,
                                      uint32_t value) {
    int packed_idx = logical_idx >> 1;
    uint64_t current = (uint64_t)packed_words[packed_idx];
    if ((logical_idx & 1) == 0) {
        current = (current & 0xffffffff00000000ULL) | (uint64_t)value;
    } else {
        current = (current & 0x00000000ffffffffULL) | ((uint64_t)value << 32);
    }
    packed_words[packed_idx] = (int64_t)current;
}

static inline uint32_t load_logical_word(const int64_t* packed_words,
                                         int logical_idx) {
    uint64_t value = (uint64_t)packed_words[logical_idx >> 1];
    if ((logical_idx & 1) == 0) return (uint32_t)(value & 0xffffffffULL);
    return (uint32_t)((value >> 32) & 0xffffffffULL);
}

extern "C" {

// ============================================================================
// Tile A — pktmerge variant
// ============================================================================
//
// For each input window record (5 spacer bytes + 1 PAM byte):
//   - if pam matches NGG:
//       * pam_meta[w] = 1 (PACKET_ID_VALID)
//       * windows_out[w*5..w*5+5] = spacer bytes (forwarded verbatim)
//       * (replacement PacketFifo path) emit a 12-byte packet:
//         uint32 window_idx + 5 spacer bytes + 3 pad bytes
//   - else:
//       * pam_meta[w] = 0 (PACKET_ID_INVALID)
//       * windows_out[w*5..w*5+5] = zero (the match tiles compute on
//         this stable junk window in v1; the per-window header in
//         pam_meta gates the match-tile loop body, and Tile Z's
//         sparse-emit drops the result downstream — same v1 approach
// as filter-early for byte-equality parity)
//       * (replacement PacketFifo path) emit nothing. PacketFifo packet
//         ids are per producer in the fork, not per packet, so invalid
//         windows are skipped rather than routed with an invalid header.
//
// The PacketFifo replacement path needs a stream/packet producer ABI.
// This v1 function keeps the byte-equality parity path through the compact
// ObjectFifo, so it does not deliver the pktMerge cycle saving yet.
//
// The optional ``windows_out`` argument is null when the IRON lowering
// elects the fully-variable-rate path; v1 always passes
// a non-null buffer.
void crispr_pam_filter_tile_a_pktmerge(uint8_t* __restrict windows_in,
                                         uint8_t* __restrict windows_out,
                                         uint8_t* __restrict pam_meta,
                                         int32_t n_windows) {
    for (int w = 0; w < n_windows; w++) {
        const uint8_t* in = windows_in + (w * WINDOW_BYTES_IN);
        uint8_t pam_byte = in[SPACER_BYTES];
        bool pass = pam_is_ngg(pam_byte);
        // pam_meta carries the legacy v1 validity byte for the ObjectFifo
        // twin. The replacement PacketFifo path skips invalid windows
        // instead of routing them by a per-packet drop header.
        pam_meta[w] = pass ? PACKET_ID_VALID : PACKET_ID_INVALID;

        // Compact-output path (byte-equality parity with filter-
        // early; see comment in tile_a_filter.cc):
        if (windows_out != nullptr) {
            uint8_t* out = windows_out + (w * SPACER_BYTES);
            if (pass) {
                for (int b = 0; b < SPACER_BYTES; b++) out[b] = in[b];
            } else {
                for (int b = 0; b < SPACER_BYTES; b++) out[b] = 0;
            }
        }
    }
}

// Direct-stream staging ABI for the real PacketFifo path.
//
// Peano does not ship the proprietary ADF <adf.h> stream-pointer ABI.
// Instead, Tile A packetizes valid/PAM-passing windows into a plain memref,
// the IRON/Python core body emits those i64 words with MLIR aie.put_stream,
// match tiles receive them with aie.get_stream into another plain memref,
// and this C++ match helper consumes the memref. This keeps C++ free of
// ADF pointer types while still lowering the transfer to the AIE dialect's
// native stream ops.
//
// Packet payload is exactly 12 bytes as three little-endian 32-bit stream
// words:
//   word0: original/global window_idx
//   word1: spacer bytes 0..3
//   word2: spacer byte 4 in bits 7:0, remaining bytes zero
void crispr_pam_filter_tile_a_pktmerge_packetize(
    uint8_t* __restrict windows_in,
    uint8_t* __restrict valid_count_out,
    int64_t* __restrict packet_words_out,
    int32_t n_windows,
    int32_t chunk_base_window_idx) {
    uint32_t n_valid = 0;
    const int max_words = (1 + n_windows * PACKETIZED_WINDOW_WORDS + 1) / 2;
    for (int i = 0; i < max_words; i++) packet_words_out[i] = 0;

    for (int w = 0; w < n_windows; w++) {
        const uint8_t* in = windows_in + (w * WINDOW_BYTES_IN);
        if (!pam_is_ngg(in[SPACER_BYTES])) continue;

        uint32_t word0 = (uint32_t)(chunk_base_window_idx + w);
        uint32_t word1 = pack_u32_le(in);
        uint32_t word2 = (uint32_t)in[4];
        int out_base = 1 + (int)n_valid * PACKETIZED_WINDOW_WORDS;

        store_logical_word(packet_words_out, out_base + 0, word0);
        store_logical_word(packet_words_out, out_base + 1, word1);
        store_logical_word(packet_words_out, out_base + 2, word2);
        n_valid++;
    }

    if (n_valid == 0) {
        store_logical_word(packet_words_out, 1, PACKETIZED_EMPTY_CHUNK_SENTINEL);
        store_logical_word(packet_words_out, 2, 0);
        store_logical_word(packet_words_out, 3, 0);
        n_valid = 1;
    }

    store_logical_word(packet_words_out, 0, n_valid);
    valid_count_out[0] = (uint8_t)(n_valid & 0xff);
    valid_count_out[1] = (uint8_t)((n_valid >> 8) & 0xff);
    valid_count_out[2] = (uint8_t)((n_valid >> 16) & 0xff);
    valid_count_out[3] = (uint8_t)((n_valid >> 24) & 0xff);
}

// Counted compact ObjectFifo ABI for the robust replacement path.
//
// Unlike the direct-stream ABI, the compact ObjectFifo path does not need a
// sentinel for zero-valid chunks because each worker still consumes one fixed
// ObjectFifo element per chunk. Match tiles only need spacer bytes; Tile Z gets
// the local window indices separately and reconstructs sparse coordinates.
void crispr_pam_filter_tile_a_pktmerge_packetize_spacers(
    uint8_t* __restrict windows_in,
    uint8_t* __restrict valid_count_out,
    int64_t* __restrict packet_words_out,
    int32_t n_windows) {
    uint32_t n_valid = 0;
    const int max_words = (1 + n_windows * COMPACT_PACKET_WINDOW_WORDS + 1) / 2;
    for (int i = 0; i < max_words; i++) packet_words_out[i] = 0;

    for (int w = 0; w < n_windows; w++) {
        const uint8_t* in = windows_in + (w * WINDOW_BYTES_IN);
        if (!pam_is_ngg(in[SPACER_BYTES])) continue;

        uint32_t word0 = pack_u32_le(in);
        uint32_t word1 = ((uint32_t)(uint8_t)w << 8) | (uint32_t)in[4];
        int out_base = 1 + (int)n_valid * COMPACT_PACKET_WINDOW_WORDS;

        store_logical_word(packet_words_out, out_base + 0, word0);
        store_logical_word(packet_words_out, out_base + 1, word1);
        n_valid++;
    }

    store_logical_word(packet_words_out, 0, n_valid);
    valid_count_out[0] = (uint8_t)(n_valid & 0xff);
    valid_count_out[1] = (uint8_t)((n_valid >> 8) & 0xff);
    valid_count_out[2] = (uint8_t)((n_valid >> 16) & 0xff);
    valid_count_out[3] = (uint8_t)((n_valid >> 24) & 0xff);
}

// Backend-legal direct-stream producer. AIE2P Peano declares only 32-bit
// master/slave stream intrinsics, so the compact buildable path keeps the loop
// in C++ instead of unrolling thousands of MLIR aie.put_stream ops.
void crispr_pam_filter_tile_a_pktmerge_stream_i32(
    uint8_t* __restrict windows_in,
    uint8_t* __restrict valid_count_out,
    int32_t n_windows,
    int32_t chunk_base_window_idx) {
    uint32_t n_valid = 0;

    for (int w = 0; w < n_windows; w++) {
        const uint8_t* in = windows_in + (w * WINDOW_BYTES_IN);
        if (!pam_is_ngg(in[SPACER_BYTES])) continue;

        uint32_t word0 = (uint32_t)(chunk_base_window_idx + w);
        uint32_t word1 = pack_u32_le(in);
        uint32_t word2 = (uint32_t)in[4];

        put_ms((int)word0, 0);
        put_ms((int)word1, 0);
        put_ms((int)word2, 0);
        n_valid++;
    }

    if (n_valid == 0) {
        put_ms((int)PACKETIZED_EMPTY_CHUNK_SENTINEL, 0);
        put_ms(0, 0);
        put_ms(0, 0);
        n_valid = 1;
    }

    valid_count_out[0] = (uint8_t)(n_valid & 0xff);
    valid_count_out[1] = (uint8_t)((n_valid >> 8) & 0xff);
    valid_count_out[2] = (uint8_t)((n_valid >> 16) & 0xff);
    valid_count_out[3] = (uint8_t)((n_valid >> 24) & 0xff);
}

// ============================================================================
// Match tile — VERBATIM copy of / 's symbol so aiecc resolves it
// from this same .o. Same arithmetic, same byte-for-byte output.
// ============================================================================
void crispr_match_multitile_match(uint8_t* __restrict guides,
                                   uint8_t* __restrict windows,
                                   uint8_t* __restrict partial_out,
                                   int32_t n_windows,
                                   int32_t guide_offset) {
    for (int w = 0; w < n_windows; w++) {
        const uint8_t* wp = windows + (w * SPACER_BYTES);
        uint8_t* op = partial_out + (w * GUIDES_PER_TILE);
        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            const int g_global = guide_offset + g_local;
            const uint8_t* gp = guides + (g_global * SPACER_BYTES);
            op[g_local] = mismatch_count_5b(gp, wp);
        }
    }
}

// Direct-stream consumer match kernel for the PacketFifo replacement path.
//
// Consumes valid_count packets previously received from aie.get_stream into
// packet_words and emits compact per-valid-window records:
//   uint32 window_idx + 64 mismatch bytes for this match tile.
// Tile Z can threshold these compact records without reconstructing the old
// fixed 64-window ObjectFifo chunk.
void crispr_match_multitile_match_packetized_memref(
    uint8_t* __restrict guides,
    int64_t* __restrict packet_words,
    uint8_t* __restrict partial_packets,
    uint8_t* __restrict valid_count_in,
    int32_t guide_offset) {
    uint32_t valid_count =
        ((uint32_t)valid_count_in[0]) |
        ((uint32_t)valid_count_in[1] << 8) |
        ((uint32_t)valid_count_in[2] << 16) |
        ((uint32_t)valid_count_in[3] << 24);
    for (int r = 0; r < valid_count; r++) {
        int in_base = r * PACKETIZED_WINDOW_WORDS;
        uint32_t word0 = load_logical_word(packet_words, in_base + 0);
        uint32_t word1 = load_logical_word(packet_words, in_base + 1);
        uint32_t word2 = load_logical_word(packet_words, in_base + 2);

        uint8_t spacer[SPACER_BYTES];
        unpack_spacer_words(word1, word2, spacer);

        uint8_t* out = partial_packets + (r * PACKETIZED_MATCH_RECORD_BYTES);
        out[0] = (uint8_t)(word0 & 0xff);
        out[1] = (uint8_t)((word0 >> 8) & 0xff);
        out[2] = (uint8_t)((word0 >> 16) & 0xff);
        out[3] = (uint8_t)((word0 >> 24) & 0xff);

        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            const int g_global = guide_offset + g_local;
            const uint8_t* gp = guides + (g_global * SPACER_BYTES);
            out[4 + g_local] = mismatch_count_5b(gp, spacer);
        }
    }
}

void crispr_match_multitile_match_packetized_memref_counted(
    uint8_t* __restrict guides,
    int64_t* __restrict packet_words,
    uint8_t* __restrict partial_packets,
    int32_t guide_offset) {
    uint32_t valid_count = load_logical_word(packet_words, 0);
    for (int r = 0; r < valid_count; r++) {
        int in_base = 1 + r * PACKETIZED_WINDOW_WORDS;
        uint32_t word0 = load_logical_word(packet_words, in_base + 0);
        uint32_t word1 = load_logical_word(packet_words, in_base + 1);
        uint32_t word2 = load_logical_word(packet_words, in_base + 2);

        uint8_t spacer[SPACER_BYTES];
        unpack_spacer_words(word1, word2, spacer);

        uint8_t* out = partial_packets + (r * PACKETIZED_MATCH_RECORD_BYTES);
        out[0] = (uint8_t)(word0 & 0xff);
        out[1] = (uint8_t)((word0 >> 8) & 0xff);
        out[2] = (uint8_t)((word0 >> 16) & 0xff);
        out[3] = (uint8_t)((word0 >> 24) & 0xff);

        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            const int g_global = guide_offset + g_local;
            const uint8_t* gp = guides + (g_global * SPACER_BYTES);
            out[4 + g_local] = mismatch_count_5b(gp, spacer);
        }
    }
}

void crispr_match_multitile_match_packetized_spacers_counted(
    uint8_t* __restrict guides,
    int64_t* __restrict packet_words,
    uint8_t* __restrict partial_packets,
    int32_t guide_offset) {
    uint32_t valid_count = load_logical_word(packet_words, 0);
    for (int r = 0; r < valid_count; r++) {
        int in_base = 1 + r * COMPACT_PACKET_WINDOW_WORDS;
        uint32_t word0 = load_logical_word(packet_words, in_base + 0);
        uint32_t word1 = load_logical_word(packet_words, in_base + 1);
        uint8_t local_window_idx = (uint8_t)((word1 >> 8) & 0xff);

        uint8_t spacer[SPACER_BYTES];
        unpack_spacer_words(word0, word1, spacer);

        uint8_t* out = partial_packets + (r * COMPACT_MATCH_RECORD_BYTES);
        out[0] = local_window_idx;
        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            const int g_global = guide_offset + g_local;
            const uint8_t* gp = guides + (g_global * SPACER_BYTES);
            out[1 + g_local] = mismatch_count_5b(gp, spacer);
        }
    }
}

// Backend-legal direct-stream consumer. Reads the same three logical i32 packet
// words emitted by crispr_pam_filter_tile_a_pktmerge_stream_i32.
void crispr_match_multitile_match_packetized_stream_i32(
    uint8_t* __restrict guides,
    uint8_t* __restrict partial_packets,
    uint8_t* __restrict valid_count_in,
    int32_t guide_offset) {
    uint32_t valid_count =
        ((uint32_t)valid_count_in[0]) |
        ((uint32_t)valid_count_in[1] << 8) |
        ((uint32_t)valid_count_in[2] << 16) |
        ((uint32_t)valid_count_in[3] << 24);

    for (int r = 0; r < valid_count; r++) {
        uint32_t word0 = (uint32_t)get_ss_int();
        uint32_t word1 = (uint32_t)get_ss_int();
        uint32_t word2 = (uint32_t)get_ss_int();

        uint8_t spacer[SPACER_BYTES];
        unpack_spacer_words(word1, word2, spacer);

        uint8_t* out = partial_packets + (r * PACKETIZED_MATCH_RECORD_BYTES);
        out[0] = (uint8_t)(word0 & 0xff);
        out[1] = (uint8_t)((word0 >> 8) & 0xff);
        out[2] = (uint8_t)((word0 >> 16) & 0xff);
        out[3] = (uint8_t)((word0 >> 24) & 0xff);

        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            const int g_global = guide_offset + g_local;
            const uint8_t* gp = guides + (g_global * SPACER_BYTES);
            out[4 + g_local] = mismatch_count_5b(gp, spacer);
        }
    }
}

void crispr_pam_filter_tile_z_pktmerge_packetized(
    uint8_t* __restrict partial_packets_0,
    uint8_t* __restrict partial_packets_1,
    uint8_t* __restrict valid_count_in,
    uint8_t* __restrict sparse_out,
    int32_t max_mismatches) {
    uint32_t valid_count =
        ((uint32_t)valid_count_in[0]) |
        ((uint32_t)valid_count_in[1] << 8) |
        ((uint32_t)valid_count_in[2] << 16) |
        ((uint32_t)valid_count_in[3] << 24);
    uint32_t emit_count = 0;
    uint8_t* out_records = sparse_out + 4;

    for (uint32_t r = 0; r < valid_count; r++) {
        uint8_t* rec0 = partial_packets_0 + (r * PACKETIZED_MATCH_RECORD_BYTES);
        uint8_t* rec1 = partial_packets_1 + (r * PACKETIZED_MATCH_RECORD_BYTES);
        uint32_t window_idx = pack_u32_le(rec0);
        if (window_idx == PACKETIZED_EMPTY_CHUNK_SENTINEL) continue;
        uint32_t slot_window_idx = window_idx & (WINDOWS_PER_CHUNK - 1);

        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            uint8_t mm = rec0[4 + g_local];
            if ((int32_t)mm <= max_mismatches && emit_count < EMIT_SLOT_RECORDS) {
                uint8_t* dst = out_records + emit_count * EMIT_RECORD_BYTES;
                dst[0] = (uint8_t)(slot_window_idx & 0xff);
                dst[1] = (uint8_t)((slot_window_idx >> 8) & 0xff);
                dst[2] = (uint8_t)(g_local & 0xff);
                dst[3] = mm;
                dst[4] = 0;
                dst[5] = 0;
                dst[6] = 0;
                dst[7] = 0;
                emit_count++;
            }
        }
        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            uint8_t mm = rec1[4 + g_local];
            if ((int32_t)mm <= max_mismatches && emit_count < EMIT_SLOT_RECORDS) {
                int guide_idx = GUIDES_PER_TILE + g_local;
                uint8_t* dst = out_records + emit_count * EMIT_RECORD_BYTES;
                dst[0] = (uint8_t)(slot_window_idx & 0xff);
                dst[1] = (uint8_t)((slot_window_idx >> 8) & 0xff);
                dst[2] = (uint8_t)(guide_idx & 0xff);
                dst[3] = mm;
                dst[4] = 0;
                dst[5] = 0;
                dst[6] = 0;
                dst[7] = 0;
                emit_count++;
            }
        }
    }

    sparse_out[0] = (uint8_t)(emit_count & 0xff);
    sparse_out[1] = (uint8_t)((emit_count >> 8) & 0xff);
    sparse_out[2] = (uint8_t)((emit_count >> 16) & 0xff);
    sparse_out[3] = (uint8_t)((emit_count >> 24) & 0xff);
}

void crispr_pam_filter_tile_z_pktmerge_packetized_indexed(
    uint8_t* __restrict partial_packets_0,
    uint8_t* __restrict partial_packets_1,
    uint8_t* __restrict valid_count_in,
    uint8_t* __restrict sparse_out,
    int32_t max_mismatches) {
    uint32_t valid_count =
        ((uint32_t)valid_count_in[0]) |
        ((uint32_t)valid_count_in[1] << 8) |
        ((uint32_t)valid_count_in[2] << 16) |
        ((uint32_t)valid_count_in[3] << 24);
    uint32_t emit_count = 0;
    uint8_t* out_records = sparse_out + 4;

    for (uint32_t r = 0; r < valid_count; r++) {
        uint8_t* rec0 = partial_packets_0 + (r * COMPACT_MATCH_RECORD_BYTES);
        uint8_t* rec1 = partial_packets_1 + (r * COMPACT_MATCH_RECORD_BYTES);
        uint32_t slot_window_idx = (uint32_t)rec0[0];

        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            uint8_t mm = rec0[1 + g_local];
            if ((int32_t)mm <= max_mismatches && emit_count < EMIT_SLOT_RECORDS) {
                uint8_t* dst = out_records + emit_count * EMIT_RECORD_BYTES;
                dst[0] = (uint8_t)(slot_window_idx & 0xff);
                dst[1] = (uint8_t)((slot_window_idx >> 8) & 0xff);
                dst[2] = (uint8_t)(g_local & 0xff);
                dst[3] = mm;
                dst[4] = 0;
                dst[5] = 0;
                dst[6] = 0;
                dst[7] = 0;
                emit_count++;
            }
        }
        for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
            uint8_t mm = rec1[1 + g_local];
            if ((int32_t)mm <= max_mismatches && emit_count < EMIT_SLOT_RECORDS) {
                int guide_idx = GUIDES_PER_TILE + g_local;
                uint8_t* dst = out_records + emit_count * EMIT_RECORD_BYTES;
                dst[0] = (uint8_t)(slot_window_idx & 0xff);
                dst[1] = (uint8_t)((slot_window_idx >> 8) & 0xff);
                dst[2] = (uint8_t)(guide_idx & 0xff);
                dst[3] = mm;
                dst[4] = 0;
                dst[5] = 0;
                dst[6] = 0;
                dst[7] = 0;
                emit_count++;
            }
        }
    }

    sparse_out[0] = (uint8_t)(emit_count & 0xff);
    sparse_out[1] = (uint8_t)((emit_count >> 8) & 0xff);
    sparse_out[2] = (uint8_t)((emit_count >> 16) & 0xff);
    sparse_out[3] = (uint8_t)((emit_count >> 24) & 0xff);
}

// ============================================================================
// Tile Z — pktmerge variant (threshold + sparse-emit; PAM-gate via header)
// ============================================================================
//
// For each (window, guide):
//   - if pam_meta[w] == 0: skip (PAM didn't pass — would have been
//     dropped by pktMerge in fabric on a fully variable-rate path).
//   - if mismatch <= max_mm: emit (window_idx, guide_idx, mm).
//
// Output is a length-prefixed sparse record stream into the ring slot.
// First 4 bytes of `sparse_out` are the uint32 record count; subsequent
// records are 8 bytes each. Identical layout to 's tile_z_early.
void crispr_pam_filter_tile_z_pktmerge(uint8_t* __restrict partial_0,
                                         uint8_t* __restrict partial_1,
                                         uint8_t* __restrict pam_meta,
                                         uint8_t* __restrict sparse_out,
                                         int32_t n_windows,
                                         int32_t max_mismatches,
                                         int32_t chunk_base_window_idx) {
    uint8_t* parts[N_MATCH_TILES] = { partial_0, partial_1 };
    uint32_t n_records = 0;
    uint8_t* dst = sparse_out + 4;  // first 4 bytes are the count prefix

    for (int w = 0; w < n_windows; w++) {
        if (pam_meta[w] == 0) continue;  // PAM failed (dropped by pktMerge)
        for (int t = 0; t < N_MATCH_TILES; t++) {
            uint8_t* pp = parts[t] + (w * GUIDES_PER_TILE);
            for (int g_local = 0; g_local < GUIDES_PER_TILE; g_local++) {
                uint8_t mm = pp[g_local];
                if ((int)mm > max_mismatches) continue;
                if (n_records >= (uint32_t)EMIT_SLOT_RECORDS) goto done;
                int g_global = t * GUIDES_PER_TILE + g_local;
                int wnd_idx = chunk_base_window_idx + w;
                dst[0] = (uint8_t)(wnd_idx & 0xff);
                dst[1] = (uint8_t)((wnd_idx >> 8) & 0xff);
                dst[2] = (uint8_t)g_global;
                dst[3] = mm;
                dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
                dst += EMIT_RECORD_BYTES;
                n_records++;
            }
        }
    }
done:
    sparse_out[0] = (uint8_t)(n_records & 0xff);
    sparse_out[1] = (uint8_t)((n_records >> 8) & 0xff);
    sparse_out[2] = (uint8_t)((n_records >> 16) & 0xff);
    sparse_out[3] = (uint8_t)((n_records >> 24) & 0xff);
}

} // extern "C"
