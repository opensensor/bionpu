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

// f32 round-to-nearest, ties-to-even. AIE2P FPU rounds the same way
// by default; we leave the rounding mode at FE_TONEAREST and use
// the standard library helper.
inline int32_t round_nearest_int32(float v) {
    // AIE2P clang lowers nearbyint to a single FPU op.
    return static_cast<int32_t>(__builtin_nearbyintf(v));
}

}  // namespace

extern "C" {

// Single-tile INT8 matmul. The per-output `combined` scale already
// folds (input_scale * weight_scale[n]) / output_scale at calibration
// time on the host (see bionpu.scoring.quantize); the bias passes
// through unchanged.
//
// Inputs:
//   x:        M × K  int8 row-major
//   w:        N × K  int8 row-major (each row = one output channel)
//   scales:   N + 1 float32   — first N entries are `combined[n]`,
//             trailing entry is reserved for an optional bias term.
// Output:
//   y:        M × N int8 row-major
//
// Note on memory ordering: x/w/y are passed as flat arrays via the
// IRON ObjectFifo lowering. The IRON Python topology elects to lay
// them out row-major; this matches how the host runner writes them
// from disk.
void bert_int8_matmul_2(
    const int8_t  * __restrict__ x,
    const int8_t  * __restrict__ w,
    const float   * __restrict__ scales,
          int8_t  * __restrict__ y
) {
    constexpr int M = BIONPU_BERT_M;
    constexpr int K = BIONPU_BERT_K;
    constexpr int N = BIONPU_BERT_N;

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
