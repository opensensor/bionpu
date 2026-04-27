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
constexpr int VEC = 16; // bf16 vector lane width on AIE2P
constexpr int HIDDEN_VECS = HIDDEN / VEC;     // 6
constexpr int HALF_IN_VECS = HALF_IN / VEC;   // 3

// Tile-resident persistent state (mirrors 's bf16 cell exactly).
static bfloat16 h_state[HIDDEN];
static bfloat16 c_state[HIDDEN];
static bfloat16 gate_acc[N_GATES][HIDDEN];
constexpr int BIAS_LEN_K = N_GATES * 2 * HIDDEN; // 768
static bfloat16 bias_cache[BIAS_LEN_K];
constexpr int BIAS_PREFIX = BIAS_LEN_K;

static inline aie::vector<bfloat16, VEC>
vec_sigmoid_bf16(aie::vector<bfloat16, VEC> x) {
  aie::vector<bfloat16, VEC> half = aie::broadcast<bfloat16, VEC>(0.5f);
  aie::vector<bfloat16, VEC> one = aie::broadcast<bfloat16, VEC>(1.0f);
  auto half_x = aie::mul(x, half);
  auto t = aie::tanh<bfloat16>(half_x.to_vector<float>());
  auto t_plus_1 = aie::add(t, one);
  return aie::mul(t_plus_1, half).to_vector<bfloat16>();
}

static inline aie::vector<bfloat16, VEC>
vec_tanh_bf16(aie::vector<bfloat16, VEC> x) {
  aie::accum<accfloat, VEC> a;
  a.from_vector(x);
  return aie::tanh<bfloat16>(a.to_vector<float>());
}

extern "C" {

void dorado_fast_lstm_cell_bf16_compressed(bfloat16 *restrict x_t,
                                           bfloat16 *restrict chunk_blk,
                                           bfloat16 *restrict y_t,
                                           int32_t g,
                                           int32_t t,
                                           int32_t chunk_idx) {
  event0();

  // Reset h, c, and cache biases on (t==0, g==0, chunk==0).
  if (t == 0 && g == 0 && chunk_idx == 0) {
    aie::vector<bfloat16, VEC> z = aie::broadcast<bfloat16, VEC>(0.0f);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      aie::store_v(h_state + v * VEC, z);
      aie::store_v(c_state + v * VEC, z);
    }
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int i = 0; i < BIAS_LEN_K; ++i) {
      bias_cache[i] = chunk_blk[i];
    }
  }

  // First chunk of a (t, g) pair: initialize gate accumulator with
  // (b_ih + b_hh) for this gate.
  if (chunk_idx == 0) {
    bfloat16 *restrict b_ih = bias_cache + (g * 2) * HIDDEN;
    bfloat16 *restrict b_hh = bias_cache + (g * 2 + 1) * HIDDEN;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      auto bi = aie::load_v<VEC>(b_ih + v * VEC);
      auto bh = aie::load_v<VEC>(b_hh + v * VEC);
      auto sum = aie::add(bi, bh);
      aie::store_v(gate_acc[g] + v * VEC, sum);
    }
  }

  // The half-gate weight slab; skip the bias prefix.
  bfloat16 *restrict W_half = chunk_blk + BIAS_PREFIX;

  // Pick the source vector.
  bfloat16 *src;
  if (chunk_idx == 0)      src = x_t;
  else if (chunk_idx == 1) src = x_t + HALF_IN;
  else if (chunk_idx == 2) src = h_state;
  else                     src = h_state + HALF_IN;

  // For each output channel, accumulate W[oc, :] · src across HALF_IN
  // = 48 = 3 * VEC lanes via bf16 vector mac (fp32 accumulator).
  // N:M-pruned entries in W_half are exact zeros, so they contribute
  // 0 to the inner reduction — i.e. the math IS the sparse forward.
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < HIDDEN; ++oc) {
    aie::accum<accfloat, VEC> acc;
    acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
    bfloat16 *restrict w_row = W_half + oc * HALF_IN;
    for (int j = 0; j < HALF_IN_VECS; ++j) {
      aie::vector<bfloat16, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
      aie::vector<bfloat16, VEC> s_v = aie::load_v<VEC>(src + j * VEC);
      acc = aie::mac(acc, w_v, s_v);
    }
    auto out_vec = acc.to_vector<float>();
    float sum = 0.0f;
    for (int k = 0; k < VEC; ++k) sum += out_vec.get(k);
    bfloat16 prev = gate_acc[g][oc];
    gate_acc[g][oc] = (bfloat16)((float)prev + sum);
  }

  // After the last chunk of the last gate (g==3, chunk==3), apply
  // nonlinearities, update c and h, emit y_t.
  if (g == 3 && chunk_idx == 3) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      auto z_i = aie::load_v<VEC>(gate_acc[0] + v * VEC);
      auto z_f = aie::load_v<VEC>(gate_acc[1] + v * VEC);
      auto z_g = aie::load_v<VEC>(gate_acc[2] + v * VEC);
      auto z_o = aie::load_v<VEC>(gate_acc[3] + v * VEC);

      auto i_g = vec_sigmoid_bf16(z_i);
      auto f_g = vec_sigmoid_bf16(z_f);
      auto g_g = vec_tanh_bf16(z_g);
      auto o_g = vec_sigmoid_bf16(z_o);

      auto c_old = aie::load_v<VEC>(c_state + v * VEC);
      auto fc = aie::mul(f_g, c_old).to_vector<bfloat16>();
      auto ig = aie::mul(i_g, g_g).to_vector<bfloat16>();
      auto c_new = aie::add(fc, ig);
      aie::store_v(c_state + v * VEC, c_new);

      auto tanh_c = vec_tanh_bf16(c_new);
      auto h_new = aie::mul(o_g, tanh_c).to_vector<bfloat16>();
      aie::store_v(h_state + v * VEC, h_new);
      aie::store_v(y_t + v * VEC, h_new);
    }
  }

  event1();
}

} // extern "C"
