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

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int HIDDEN = 96;
constexpr int INPUT_DIM = 96;
constexpr int HALF_IN = INPUT_DIM / 2; // 48
constexpr int N_GATES = 4;

// Tile-resident persistent state (preserved across kernel calls within
// a single LSTM-cell program lifetime). Reset on t == 0 by the kernel.
static float h_state[HIDDEN];
static float c_state[HIDDEN];
// Per-timestep partial gate accumulators. Updated by gate index g.
static float gate_acc[N_GATES][HIDDEN];
// Cached biases. Loaded from the first weight chunk's bias prefix on
// (t == 0, g == 0). N_GATES * 2 (ih, hh) * HIDDEN = 768 floats.
constexpr int BIAS_LEN_K = N_GATES * 2 * HIDDEN; // 768
static float bias_cache[BIAS_LEN_K];
// Bias prefix length per chunk (768 floats); kernel skips this prefix
// in non-cache calls.
constexpr int BIAS_PREFIX = BIAS_LEN_K;

// Compact high-order rational tanh approximation. The coefficients are
// a [7/6] odd Padé-style form that is substantially tighter than the
// previous x*(27+x^2)/(27+9*x^2) approximation while staying small
// enough for AIE program memory. Clamp outside the useful gate range.
static inline float scalar_tanh_fp32(float x) {
  if (x > 5.0f) return 1.0f;
  if (x < -5.0f) return -1.0f;
  float x2 = x * x;
  float num = ((x2 + 378.0f) * x2 + 17325.0f) * x2 + 135135.0f;
  float den = ((28.0f * x2 + 3150.0f) * x2 + 62370.0f) * x2 + 135135.0f;
  return x * num / den;
}

static inline float scalar_sigmoid(float x) {
  return 0.5f * (scalar_tanh_fp32(0.5f * x) + 1.0f);
}

extern "C" {

// One per-gate-per-chunk call. The IRON lowering invokes this 4
// (chunks) * 4 (gates) = 16 times per timestep. See header comment for
// the math; this kernel handles ONE half-gate slab per call.
//
// Args:
//   x_t       : (HIDDEN,) FP32. Input frame at timestep t.
//   chunk_blk : (BIAS_PREFIX + HIDDEN*HALF_IN,) FP32 — bias prefix +
//               one half-gate weight slab. The bias prefix is the full
//               768-float bias slab (gates × ih/hh); the kernel caches
//               it on (t==0, g==0, chunk==0) and skips the prefix
//               on every other call. The post-prefix tail is the
//               half-gate weight slab whose role is determined by
//               `chunk_idx`:
//                 chunk_idx 0 : W_ih_h0 (first half along x_t input dim)
//                 chunk_idx 1 : W_ih_h1 (second half)
//                 chunk_idx 2 : W_hh_h0 (first half along h hidden dim)
//                 chunk_idx 3 : W_hh_h1 (second half)
//   y_t       : (HIDDEN,) FP32 output (only written when g == 3 and
//               chunk_idx == 3).
//   g         : int32 gate index (0..3 = i, f, g, o).
//   t         : int32 timestep (0..L-1). t==0 triggers state reset
//               (and chunk_idx==0, g==0 → bias caching).
//   chunk_idx : int32 within-gate chunk (0..3).
void dorado_fast_lstm_cell_fp32(float *restrict x_t,
                                float *restrict chunk_blk,
                                float *restrict y_t,
                                int32_t g,
                                int32_t t,
                                int32_t chunk_idx) {
  event0();

  // On (t==0, g==0, chunk==0): reset h, c, and cache biases.
  if (t == 0 && g == 0 && chunk_idx == 0) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int i = 0; i < HIDDEN; ++i) {
      h_state[i] = 0.0f;
      c_state[i] = 0.0f;
    }
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int i = 0; i < BIAS_LEN_K; ++i) {
      bias_cache[i] = chunk_blk[i];
    }
  }

  // First chunk of a (t, g) pair: initialize the gate accumulator with
  // bias values. Subsequent chunks accumulate into it.
  if (chunk_idx == 0) {
    float *restrict b_ih = bias_cache + (g * 2) * HIDDEN;
    float *restrict b_hh = bias_cache + (g * 2 + 1) * HIDDEN;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int oc = 0; oc < HIDDEN; ++oc) {
      gate_acc[g][oc] = b_ih[oc] + b_hh[oc];
    }
  }

  // The half-gate weight slab; skip the bias prefix.
  float *restrict W_half = chunk_blk + BIAS_PREFIX;

  // Pick the source vector for the dot product based on chunk_idx:
  //   chunk_idx 0 : W_ih_h0 @ x_t[0:48]
  //   chunk_idx 1 : W_ih_h1 @ x_t[48:96]
  //   chunk_idx 2 : W_hh_h0 @ h[0:48]
  //   chunk_idx 3 : W_hh_h1 @ h[48:96]
  float *src;
  if (chunk_idx == 0)      src = x_t;
  else if (chunk_idx == 1) src = x_t + HALF_IN;
  else if (chunk_idx == 2) src = h_state;
  else                     src = h_state + HALF_IN;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < HIDDEN; ++oc) {
    float s = gate_acc[g][oc];
    float *restrict w_row = W_half + oc * HALF_IN;
    for (int j = 0; j < HALF_IN; ++j) {
      s += w_row[j] * src[j];
    }
    gate_acc[g][oc] = s;
  }

  // After the last chunk of the last gate (g==3, chunk==3), apply
  // nonlinearities, update c and h, emit y_t.
  if (g == 3 && chunk_idx == 3) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int oc = 0; oc < HIDDEN; ++oc) {
      float i_g = scalar_sigmoid(gate_acc[0][oc]); // gate 0: input gate
      float f_g = scalar_sigmoid(gate_acc[1][oc]); // gate 1: forget gate
      float g_g = scalar_tanh_fp32(gate_acc[2][oc]); // gate 2: cell input
      float o_g = scalar_sigmoid(gate_acc[3][oc]); // gate 3: output gate
      float c_new = f_g * c_state[oc] + i_g * g_g;
      float h_new = o_g * scalar_tanh_fp32(c_new);
      c_state[oc] = c_new;
      h_state[oc] = h_new;
      y_t[oc] = h_new;
    }
  }

  event1();
}

} // extern "C"
