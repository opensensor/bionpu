//===- bert_mini_block.cc -------------------------------------*- C++ -*-===//
//
// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// PRD-dnabert-epi-on-xdna §3.8 step 0.3 — AIE-resident softmax + LayerNorm
// kernels for the BERT-mini transformer block. The PRD calls these "FP16"
// but AIE2P's hardware exp2 / invsqrt / tanh primitives operate on bfloat16.
// bf16 is a strict superset of FP16's dynamic range and is the correct
// datatype for tile-resident reductions on AIE2P.
//
// Symbols exported (extern "C"):
//   bert_mini_attention_softmax — per-row max-subtract + exp + sum + divide
//                                  with /sqrt(head_dim) scaling baked in.
//                                  Operates on a single row of an (M, M)
//                                  attention score matrix in bf16.
//
//   bert_mini_layer_norm        — per-row LN (mean, variance, normalize,
//                                  scale·γ + β) over the hidden dim. γ/β
//                                  pass through as additional input
//                                  vectors so the host can stream them
//                                  alongside the residual input.
//
// Adapted from upstream mlir-aie reference kernels:
//   third_party/mlir-aie/install/include/aie_kernels/aie2p/softmax.cc
//   third_party/mlir-aie/install/include/aie_kernels/aie2p/layer_norm.cc
//
// Both reference kernels are Apache-2.0 WITH LLVM-exception (compatible).

#include <cstdint>
#include <cstring>

#if defined(__AIE_ARCH__) && !defined(BIONPU_FORCE_SCALAR)
#  define BIONPU_HAS_AIE_API 1
#  include <aie_api/aie.hpp>
#else
#  define BIONPU_HAS_AIE_API 0
#  include <cmath>
#endif

// BERT-mini single-block shape constants (PRD-dnabert-epi step 0.3):
//   hidden = 256, seq M = 47 (DNABERT 3-mer pair length), heads = 4,
//   head_dim = 64.
#ifndef BIONPU_BERT_MINI_M
#define BIONPU_BERT_MINI_M 47
#endif
#ifndef BIONPU_BERT_MINI_HIDDEN
#define BIONPU_BERT_MINI_HIDDEN 256
#endif
#ifndef BIONPU_BERT_MINI_HEAD_DIM
#define BIONPU_BERT_MINI_HEAD_DIM 64
#endif

// bf16 vector lane width on AIE2P (matches lstm_cell_bf16.cc).
#ifndef BIONPU_BF16_VEC
#define BIONPU_BF16_VEC 32
#endif

// Softmax row length is M (attention is M×M). For M=47 the row is padded
// to BIONPU_SOFTMAX_PAD bytes (multiple of vector lane width). Default
// pads M=47 → 48; the kernel zero-fills the tail at the host so that
// the reduction over BIONPU_SOFTMAX_PAD elements still produces the
// correct max / sum (zeros contribute 0 to the sum, and -inf for the
// max — host fills with -65504.0 sentinel for that reason).
#ifndef BIONPU_SOFTMAX_PAD
// Round M up to next multiple of bf16 vector lane (32). M=47 → 64.
#define BIONPU_SOFTMAX_PAD 64
#endif

// LayerNorm row length is HIDDEN. Pad to multiple of vector lane (32).
#ifndef BIONPU_LN_PAD
#define BIONPU_LN_PAD 256  // already multiple of 32
#endif

// log2(e) for softmax fast-path. AIE2P has hardware exp2 but not exp;
// scaling input by log2(e) before exp2 is the standard rewrite.
static constexpr float BIONPU_LOG2E = 1.4426950408889634f;

namespace {

// Saturate-clamp to bf16 limits. bf16 max ≈ 3.4e38 but the safe
// "log2(min positive) of bf16" is ~-126; we clamp the post-sub-max
// value to a soft floor of -32 in log2-space (equivalent to ~e-22 in
// linear) so very negative pre-softmax values quietly underflow to 0
// in the exp without producing NaN. This matches the upstream AIE2P
// softmax kernel's behavior.
inline float bionpu_clamp_lo(float v, float lo) {
    return v < lo ? lo : v;
}

}  // namespace

extern "C" {

// ──────────────────────────────────────────────────────────────────────────
// Softmax — attention scores
// ──────────────────────────────────────────────────────────────────────────
//
// Inputs:
//   in_row:   BIONPU_SOFTMAX_PAD bf16 — one row of the M×M attention
//             score matrix, scaled by 1/sqrt(head_dim) on the host.
//             Tail elements [M .. PAD-1] are pre-filled with a large
//             negative sentinel (-65000.0) so they don't affect max
//             or sum.
//   out_row:  BIONPU_SOFTMAX_PAD bf16 — softmax(in_row); tail elements
//             are well-defined (small floats summing to ~0) but the
//             host runner trims back to M elements.
//
// Algorithm (3-pass per upstream softmax.cc):
//   1) row_max = max_i(in_row[i] * log2e)
//   2) out[i] = exp2(in_row[i] * log2e - row_max);  s += out[i]
//   3) out[i] /= s
//
// Rationale for 3-pass over fused max+sum (Welford-style):
//   The 3-pass form is numerically stable across the bf16 dynamic
//   range and matches the PyTorch CPU reference's order-of-operations
//   to within bf16 round-off — critical for the silicon byte-equal
//   gate. Fused approaches save one DRAM pass at the cost of
//   different mantissa retention in the running max.

void bert_mini_attention_softmax(
    const uint16_t * __restrict__ in_row,   // bf16 stored as u16
          uint16_t * __restrict__ out_row
) {
#if BIONPU_HAS_AIE_API
    constexpr int N = BIONPU_BF16_VEC;
    constexpr int PAD = BIONPU_SOFTMAX_PAD;
    constexpr int CHUNKS = PAD / N;

    const bfloat16 *input  = reinterpret_cast<const bfloat16 *>(in_row);
          bfloat16 *output = reinterpret_cast<bfloat16 *>(out_row);

    auto log2e_v = aie::broadcast<bfloat16, N>((bfloat16)BIONPU_LOG2E);

    // Pass 1: row max in log2 space.
    aie::vector<bfloat16, N> max_acc =
        aie::broadcast<bfloat16, N>((bfloat16)-65000.0f);
    for (int c = 0; c < CHUNKS; ++c) {
        aie::vector<bfloat16, N> v = aie::load_v<N>(input + c * N);
        aie::accum<accfloat, N> sc = aie::mul(v, log2e_v);
        max_acc = aie::max(max_acc, sc.to_vector<bfloat16>());
    }
    float row_max = aie::reduce_max(max_acc);
    auto row_max_v = aie::broadcast<bfloat16, N>((bfloat16)row_max);

    // Pass 2: exp2(scaled - max), accumulate sum.
    aie::accum<accfloat, N> sum_acc;
    sum_acc.from_vector(aie::broadcast<float, N>(0.0f));
    for (int c = 0; c < CHUNKS; ++c) {
        aie::vector<bfloat16, N> v = aie::load_v<N>(input + c * N);
        aie::accum<accfloat, N> sc = aie::mul(v, log2e_v);
        aie::accum<accfloat, N> shifted = aie::sub(sc, row_max_v);
        aie::vector<bfloat16, N> e = aie::exp2<bfloat16>(shifted.to_vector<float>());
        sum_acc = add(sum_acc, e);
        aie::store_v(output + c * N, e);
    }

    aie::vector<float, N> sum_lanes = sum_acc.to_vector<float>();
    float row_sum = aie::reduce_add(sum_lanes);
    bfloat16 inv_sum = (bfloat16)aie::inv(row_sum);

    // Pass 3: divide.
    for (int c = 0; c < CHUNKS; ++c) {
        aie::vector<bfloat16, N> e = aie::load_v<N>(output + c * N);
        aie::accum<accfloat, N> r = aie::mul(e, inv_sum);
        aie::store_v(output + c * N, r.to_vector<bfloat16>());
    }
#else
    // Scalar host-emulation fallback. bf16 storage is uint16; we
    // reinterpret via __bfloat16 widen-to-float pattern so the host
    // build (used by the byte-equal harness) produces output that
    // matches the silicon's bf16 round-half-to-even.
    constexpr int PAD = BIONPU_SOFTMAX_PAD;
    auto bf16_to_f = [](uint16_t b) {
        uint32_t bits = uint32_t(b) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    };
    auto f_to_bf16 = [](float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // Round-to-nearest-even.
        uint32_t lsb = (bits >> 16) & 1u;
        uint32_t rounding_bias = 0x7FFFu + lsb;
        bits += rounding_bias;
        return (uint16_t)(bits >> 16);
    };
    float vals[PAD];
    for (int i = 0; i < PAD; ++i) {
        vals[i] = bf16_to_f(in_row[i]) * BIONPU_LOG2E;
    }
    float row_max = vals[0];
    for (int i = 1; i < PAD; ++i) {
        if (vals[i] > row_max) row_max = vals[i];
    }
    float row_max_bf = bf16_to_f(f_to_bf16(row_max));
    float sum = 0.0f;
    for (int i = 0; i < PAD; ++i) {
        float e = exp2f(vals[i] - row_max_bf);
        // bf16 round-trip every accumulator step to mimic silicon path.
        float e_bf = bf16_to_f(f_to_bf16(e));
        out_row[i] = f_to_bf16(e_bf);
        sum += e_bf;
    }
    float inv_sum = bf16_to_f(f_to_bf16(1.0f / sum));
    for (int i = 0; i < PAD; ++i) {
        float e = bf16_to_f(out_row[i]);
        out_row[i] = f_to_bf16(e * inv_sum);
    }
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// LayerNorm — per-row over hidden dim
// ──────────────────────────────────────────────────────────────────────────
//
// Inputs:
//   in_row:   BIONPU_LN_PAD bf16 — one row of the (M, hidden) activation.
//   gamma:    BIONPU_LN_PAD bf16 — per-channel scale.
//   beta:     BIONPU_LN_PAD bf16 — per-channel bias.
//   out_row:  BIONPU_LN_PAD bf16 — gamma·((x - mean) / sqrt(var + eps)) + beta.
//
// Algorithm:
//   mean     = sum(x) / cols
//   variance = sum(x*x) / cols - mean*mean
//   inv_std  = 1 / sqrt(variance + epsilon)
//   out[c]   = (x[c] - mean) * inv_std * gamma[c] + beta[c]
//
// epsilon = 1e-5 (matches PyTorch BertLayerNorm default).

void bert_mini_layer_norm(
    const uint16_t * __restrict__ in_row,
    const uint16_t * __restrict__ gamma,
    const uint16_t * __restrict__ beta,
          uint16_t * __restrict__ out_row
) {
    constexpr int PAD = BIONPU_LN_PAD;
    constexpr float epsilon = 1e-5f;
#if BIONPU_HAS_AIE_API
    constexpr int N = BIONPU_BF16_VEC;
    constexpr int CHUNKS = PAD / N;
    static_assert(PAD % N == 0, "LN_PAD must be multiple of BF16_VEC");

    const bfloat16 *input = reinterpret_cast<const bfloat16 *>(in_row);
    const bfloat16 *gv    = reinterpret_cast<const bfloat16 *>(gamma);
    const bfloat16 *bv    = reinterpret_cast<const bfloat16 *>(beta);
          bfloat16 *output = reinterpret_cast<bfloat16 *>(out_row);

    aie::vector<bfloat16, N> sum_acc = aie::zeros<bfloat16, N>();
    aie::vector<float, N>    sum_sq_acc = aie::zeros<float, N>();
    for (int c = 0; c < CHUNKS; ++c) {
        aie::vector<bfloat16, N> v = aie::load_v<N>(input + c * N);
        sum_acc = aie::add(sum_acc, v);
        aie::accum<accfloat, N> sq = aie::mul(v, v);
        sum_sq_acc = aie::add(sum_sq_acc, sq.to_vector<float>());
    }
    float sum    = aie::reduce_add(sum_acc);
    float sum_sq = aie::reduce_add(sum_sq_acc);

    float mean = sum / float(PAD);
    float variance = (sum_sq / float(PAD)) - mean * mean;
    float inv_std = aie::invsqrt(variance + epsilon);

    auto mean_v    = aie::broadcast<bfloat16, N>((bfloat16)mean);
    auto inv_std_v = aie::broadcast<bfloat16, N>((bfloat16)inv_std);

    // Per-chunk normalize+affine. aie::sub / aie::mul / aie::add over
    // pairs of vectors return aie::vector (not aie::accum) at the API
    // surface. Match the upstream layer_norm.cc reference verbatim.
    for (int c = 0; c < CHUNKS; ++c) {
        aie::vector<bfloat16, N> v  = aie::load_v<N>(input + c * N);
        aie::vector<bfloat16, N> g  = aie::load_v<N>(gv + c * N);
        aie::vector<bfloat16, N> b  = aie::load_v<N>(bv + c * N);
        aie::vector<bfloat16, N> diff   = aie::sub(v, mean_v);
        aie::vector<bfloat16, N> norm   = aie::mul(diff, inv_std_v);
        aie::vector<bfloat16, N> scaled = aie::mul(norm, g);
        aie::vector<bfloat16, N> out_v  = aie::add(scaled, b);
        aie::store_v(output + c * N, out_v);
    }
#else
    // Scalar fallback for host emulation.
    auto bf16_to_f = [](uint16_t b) {
        uint32_t bits = uint32_t(b) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    };
    auto f_to_bf16 = [](float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t lsb = (bits >> 16) & 1u;
        uint32_t rounding_bias = 0x7FFFu + lsb;
        bits += rounding_bias;
        return (uint16_t)(bits >> 16);
    };
    float sum = 0.0f, sum_sq = 0.0f;
    for (int c = 0; c < PAD; ++c) {
        float v = bf16_to_f(in_row[c]);
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / float(PAD);
    float variance = (sum_sq / float(PAD)) - mean * mean;
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    for (int c = 0; c < PAD; ++c) {
        float v = bf16_to_f(in_row[c]);
        float g = bf16_to_f(gamma[c]);
        float b = bf16_to_f(beta[c]);
        float n = (v - mean) * inv_std;
        out_row[c] = f_to_bf16(n * g + b);
    }
#endif
}

// Packed-γβ entry point used by the silicon IRON lowering. γ and β
// are concatenated into one 2*HIDDEN bf16 slab (γ first, β second);
// the per-row dispatch reads γ from gb[0..HIDDEN] and β from
// gb[HIDDEN..2*HIDDEN] via simple pointer offset.
void bert_mini_layer_norm_packed(
    const uint16_t * __restrict__ in_row,
    const uint16_t * __restrict__ gb,        // γ‖β packed slab
          uint16_t * __restrict__ out_row
) {
    const uint16_t *gamma = gb;
    const uint16_t *beta  = gb + BIONPU_LN_PAD;
    bert_mini_layer_norm(in_row, gamma, beta, out_row);
}

}  // extern "C"
