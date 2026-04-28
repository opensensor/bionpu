//===- bert_int8_matmul.cc -------------------------------------*- C++ -*-===//
//
// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AIE2P compute-tile kernel — INT8 × INT8 → INT8 matmul with fused
// FP32 per-output-channel scale and INT8 saturation. Two specializations:
//
// * `bert_int8_matmul_2`           — head v0.4-alpha. Single tile, full K
//                                     resident, full N=2 output. Single
//                                     entry point, scalar inner loop.
//
// * `bert_int8_matmul_qkvo_init`   — qkvo v0.4-beta init: zero the static
//                                     INT32 accumulator on each tile.
// * `bert_int8_matmul_qkvo_acc`    — qkvo v0.4-beta accumulate: i32 +=
//                                     i8 × i8 over a single K_CHUNK slice
//                                     of x and the per-tile w slice.
// * `bert_int8_matmul_qkvo_finalize` — qkvo v0.4-beta finalize: fused
//                                     FP32 scale + INT8 saturate of the
//                                     i32 accumulator into the int8
//                                     output partial.
//
// All four entry points live in the same .o; the IRON-Python topology
// links them in based on the variant flag at lower-time.

#include <cstdint>
#include <cstring>

// AIE-API would be included here for the v0.5 vectorized variant.
// v0 stays scalar so the kernel fits in <16 KiB program memory and
// the correctness path is reviewable without intrinsic knowledge.

#ifndef BIONPU_BERT_M
#define BIONPU_BERT_M 47
#endif
#ifndef BIONPU_BERT_K
#define BIONPU_BERT_K 768
#endif
#ifndef BIONPU_BERT_N
#define BIONPU_BERT_N 2
#endif
// qkvo per-tile slice sizes. With N=768 and N_TILES=4 → N_PER_TILE=192;
// K_CHUNK=64 → K_CHUNKS=12.
#ifndef BIONPU_BERT_N_TILES
#define BIONPU_BERT_N_TILES 4
#endif
#ifndef BIONPU_BERT_K_CHUNK
#define BIONPU_BERT_K_CHUNK 64
#endif

namespace {

// Saturating int8 cast. AIE2P has hardware saturate ops we'll use
// in v0.5; scalar version below is deliberately portable so the
// kernel can also build on host emulation.
inline int8_t saturate_int8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return static_cast<int8_t>(v);
}

// f32 round-half-away-from-zero, manual.
//
// The "obvious" `__builtin_nearbyintf` crashes the Peano AIE2P GISel
// Legalizer in this version of llvm-aie (memory: similar issues with
// scalar-FP32 round-to-int ops surface in BM-spill paths). Hand-rolled
// add-half-then-truncate sidesteps that legalization rule entirely
// and lowers cleanly.
//
// Tie-handling difference vs nearbyintf (round-half-to-even) is
// strictly bounded — at most 1 LSB of the i8 output, well inside the
// 1 ULP byte-equivalence tolerance the verify harness allows.
inline int32_t round_nearest_int32(float v) {
    return v >= 0.0f
        ? static_cast<int32_t>(v + 0.5f)
        : static_cast<int32_t>(v - 0.5f);
}

}  // namespace

extern "C" {

// ──────────────────────────────────────────────────────────────────────────
// v0.4-alpha — head specialization (M=47, K=768, N=2). Single tile.
// ──────────────────────────────────────────────────────────────────────────

// Single-tile INT8 matmul. The per-output `combined` scale already
// folds (input_scale * weight_scale[n]) / output_scale at calibration
// time on the host (see bionpu.scoring.quantize); the bias passes
// through unchanged.
//
// Inputs:
//   x:        M × K   int8 row-major
//   ws_buf:   N*K + (N+1)*4 bytes — int8 weights followed by float32 scales.
//             The 3-input-fifo design (x, w, scales separately) doesn't
//             fit AIE2P's 2-in / 2-out CoreTile DMA budget; w + scales
//             concatenate at the host into a single byte buffer.
//             Layout:
//               ws_buf[0 .. N*K - 1]                   = int8 weights (N×K)
//               ws_buf[N*K .. N*K + (N+1)*4 - 1]       = float32 scales (N+1)
// Output:
//   y:        M × N int8 row-major
void bert_int8_matmul_2(
    const int8_t * __restrict__ x,
    const int8_t * __restrict__ ws_buf,
          int8_t * __restrict__ y
) {
    constexpr int M = BIONPU_BERT_M;
    constexpr int K = BIONPU_BERT_K;
    constexpr int N = BIONPU_BERT_N;

    const int8_t *w      = ws_buf;
    const float  *scales = reinterpret_cast<const float *>(ws_buf + N * K);

    // For each output cell: i32 accumulator over the K reduction.
    // Inner loop is scalar; v0.5 swaps in 64-lane ::aie::vector.
    for (int m = 0; m < M; ++m) {
        const int8_t *x_row = x + m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t *w_row = w + n * K;
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(x_row[k])
                     * static_cast<int32_t>(w_row[k]);
            }
            // Fused scale + INT8 saturate. The host's
            // bionpu.scoring.quantize sets `combined[n]` such that
            // y_int8 ≈ round(W_fp · x_fp / output_scale), preserving
            // the byte-equivalence contract.
            const float fy = static_cast<float>(acc) * scales[n];
            y[m * N + n] = saturate_int8(round_nearest_int32(fy));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// v0.4-beta — qkvo specialization (M=47, K=768, N=768).
//
// Each compute tile owns N_PER_TILE = N/N_TILES output channels. The
// K loop runs in IRON-Python; this kernel's three entry points handle
// init / accumulate / finalize:
//
//   init()       — zero the static INT32 accumulator
//   acc()        — i32 += i8×i8 over one K_CHUNK slab
//   finalize()   — fused-scale + INT8 saturate of acc → y_part
//
// The accumulator is a tile-local static array (private to each tile's
// kernel image — IRON places the same .o on all four tiles, but each
// tile's static memory is independent).
// ──────────────────────────────────────────────────────────────────────────

namespace {

constexpr int QKVO_M           = BIONPU_BERT_M;                          // 47
constexpr int QKVO_K_CHUNK     = BIONPU_BERT_K_CHUNK;                    // 64
constexpr int QKVO_N           = BIONPU_BERT_N;                          // 768
constexpr int QKVO_N_PER_TILE  = BIONPU_BERT_N / BIONPU_BERT_N_TILES;    // 192
constexpr int QKVO_SCALES_N    = BIONPU_BERT_N + 1;                      // 769

}  // namespace

// Zero the i32 accumulator.
//
// Inputs:
//   acc_buf:    M × N_PER_TILE int32 — tile-local IRON Buffer
void bert_int8_matmul_qkvo_init(
          int32_t * __restrict__ acc_buf
) {
    for (int i = 0; i < QKVO_M * QKVO_N_PER_TILE; ++i) {
        acc_buf[i] = 0;
    }
}

// One K_CHUNK slab of the matmul.
//
// Inputs:
//   xs_chunk:   byte buffer of size M*K_CHUNK + (N+1)*4 bytes:
//                 [0          .. M*K_CHUNK - 1]            = x int8 (M × K_CHUNK)
//                 [M*K_CHUNK  .. M*K_CHUNK + (N+1)*4 - 1]  = scales fp32 (N+1)
//               The host duplicates the scales prefix on every chunk to
//               keep ObjectFifo elements uniform-size.
//   w_chunk:    N_PER_TILE × K_CHUNK int8 row-major (N-major, K_CHUNK-minor)
//   acc_buf:    M × N_PER_TILE int32 — tile-local IRON Buffer (read/write)
//   scales_buf: (N_PER_TILE + 1) float32 — tile-local IRON Buffer.
//               This kernel re-stashes this tile's per-tile slice of the
//               full scales array every call; only the LAST call's
//               write matters since finalize reads it after the K loop.
//   tile_idx:   0..N_TILES-1 — selects which N_PER_TILE slice of the
//               xs_chunk scales prefix to stash for this tile.
//
// Effect:
//   acc_buf[m * N_PER_TILE + n] += sum_{k=0..K_CHUNK-1} x_chunk[m, k] * w_chunk[n, k]
//   scales_buf[0..N_PER_TILE-1] = *(float*)(xs_chunk + M*K_CHUNK + tile_idx*N_PER_TILE*4)
//   scales_buf[N_PER_TILE]      = *(float*)(xs_chunk + M*K_CHUNK + N*4)  (bias slot)
void bert_int8_matmul_qkvo_acc(
    const int8_t * __restrict__ xs_chunk,
    const int8_t * __restrict__ w_chunk,
          int32_t * __restrict__ acc_buf,
          float   * __restrict__ scales_buf,
          int32_t                tile_idx
) {
    constexpr int M  = QKVO_M;
    constexpr int N  = QKVO_N;
    constexpr int Np = QKVO_N_PER_TILE;
    constexpr int Kc = QKVO_K_CHUNK;

    // Stash this tile's scales slice. Cost: 192*4 = 768 B per K_CHUNK ×
    // K_CHUNKS = 9 KB redundant fp32 copies per launch. Trivial vs.
    // 576 KB weight stream.
    {
        const float *full =
            reinterpret_cast<const float *>(xs_chunk + M * Kc);
        const float *src = full + tile_idx * Np;
        for (int i = 0; i < Np; ++i) {
            scales_buf[i] = src[i];
        }
        // Trailing bias slot (scale[N]) — same on all tiles.
        scales_buf[Np] = full[N];
    }

    const int8_t *x_chunk = xs_chunk;  // x lives at the head of xs_chunk

    for (int m = 0; m < M; ++m) {
        const int8_t *xr = x_chunk + m * Kc;
        int32_t      *ar = acc_buf + m * Np;
        for (int n = 0; n < Np; ++n) {
            const int8_t *wr = w_chunk + n * Kc;
            int32_t a = ar[n];
            for (int k = 0; k < Kc; ++k) {
                a += static_cast<int32_t>(xr[k]) * static_cast<int32_t>(wr[k]);
            }
            ar[n] = a;
        }
    }
}

// Fused-scale + INT8 saturate of the accumulator into the int8 output.
//
// Inputs:
//   acc_buf:    M × N_PER_TILE int32 — read-only at this stage
//   scales_buf: N_PER_TILE+1 float32 — this tile's slice (read-only)
//   y_part:     M × N_PER_TILE int8 row-major (output)
void bert_int8_matmul_qkvo_finalize(
    const int32_t * __restrict__ acc_buf,
    const float   * __restrict__ scales_buf,
          int8_t  * __restrict__ y_part
) {
    constexpr int M  = QKVO_M;
    constexpr int Np = QKVO_N_PER_TILE;

    for (int m = 0; m < M; ++m) {
        const int32_t *ar = acc_buf    + m * Np;
              int8_t  *yr = y_part     + m * Np;
        for (int n = 0; n < Np; ++n) {
            const float fy = static_cast<float>(ar[n]) * scales_buf[n];
            yr[n] = saturate_int8(round_nearest_int32(fy));
        }
    }
}

}  // extern "C"
