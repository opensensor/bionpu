//===- bert_int8_matmul.cc -------------------------------------*- C++ -*-===//
//
// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AIE2P compute-tile kernel — INT8 × INT8 → INT8 matmul with fused
// FP32 per-output-channel scale and INT8 saturation. Scalar inner
// loop in v0 (correctness path); the v0.5 perf pass switches to
// `::aie::vector` accumulators.
//
// Symbol naming convention: one C++ entry point per (M, K, N) tuple,
// resolved via the ObjectFifo `Kernel(symbol, …)` reference in the
// Python topology. v0 ships a single specialization for the
// classifier head (M=47, K=768, N=2). The other instances reuse
// this same .cc, picked up via different `-D` macros in the
// Makefile.

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

}  // extern "C"
