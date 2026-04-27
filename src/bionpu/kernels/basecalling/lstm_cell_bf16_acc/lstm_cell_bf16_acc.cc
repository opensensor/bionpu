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

// Tile-resident persistent state.
//
// h, c live as FP32 between timesteps (AM020 cross-walk: 23-mantissa-bit
// recurrent state preservation). The IRON Python lowering doesn't
// surface BM accumulator-register persistence directly, but FP32 static
// storage in tile DM is the precision-equivalent fallback — the
// accumulator-to-FP32-store path is lossless on AIE2P (the
// `acc.to_vector<float>()` conversion is hardware free), and FP32
// matches the accumulator's 23 mantissa bits.
//
// gate_acc is FP32 (per gate, per output channel) since the running
// pre-nonlinearity sum w·x + b is the natural width of the MAC's FP32
// accumulator. Holding gate_acc at FP32 (vs 's bf16 storage from
// ) is the load-bearing precision change.
//
// bias_cache stays bf16: the bias is a load-once value; bf16 width is
// sufficient (8 mantissa bits) for the bias which gets summed into the
// FP32 gate_acc each chunk.
static float h_state[HIDDEN];
static float c_state[HIDDEN];
static float gate_acc[N_GATES][HIDDEN];
constexpr int BIAS_LEN_K = N_GATES * 2 * HIDDEN; // 768
static bfloat16 bias_cache[BIAS_LEN_K];
constexpr int BIAS_PREFIX = BIAS_LEN_K;

// Vectorised sigmoid via tanh identity, using AIE2P hardware bf16 tanh:
//   sigmoid(x) = 0.5 * (tanh(0.5 * x) + 1)
static inline aie::vector<bfloat16, VEC>
vec_sigmoid_bf16(aie::vector<bfloat16, VEC> x) {
  aie::vector<bfloat16, VEC> half = aie::broadcast<bfloat16, VEC>(0.5f);
  aie::vector<bfloat16, VEC> one = aie::broadcast<bfloat16, VEC>(1.0f);
  auto half_x = aie::mul(x, half);
  auto t = aie::tanh<bfloat16>(half_x.to_vector<float>());
  auto t_plus_1 = aie::add(t, one);
  return aie::mul(t_plus_1, half).to_vector<bfloat16>();
}

// Vectorised hardware tanh.
static inline aie::vector<bfloat16, VEC>
vec_tanh_bf16(aie::vector<bfloat16, VEC> x) {
  aie::accum<accfloat, VEC> a;
  a.from_vector(x);
  return aie::tanh<bfloat16>(a.to_vector<float>());
}

// Helper: load a VEC-lane window of fp32 storage and narrow to bf16.
// Used to feed h_state (FP32 across timesteps) into the matmul input
// path (bf16). Per AM020 Ch. 4 p. 65, the bf16 multiplier-input
// narrowing is hardware free; this maps to the same lowering.
static inline aie::vector<bfloat16, VEC>
load_h_as_bf16(const float *restrict src) {
  aie::vector<float, VEC> vf = aie::load_v<VEC>(src);
  aie::accum<accfloat, VEC> a;
  a.from_vector(vf);
  return a.to_vector<bfloat16>();
}

extern "C" {

// Identical contract to 's lstm_cell_bf16: bf16 input/output on
// the wire, bf16 weight chunks. The internal precision discipline is
// the only thing that changes (gate_acc + h_state + c_state in FP32).
void dorado_fast_lstm_cell_bf16_acc(bfloat16 *restrict x_t,
                                    bfloat16 *restrict chunk_blk,
                                    bfloat16 *restrict y_t,
                                    int32_t g,
                                    int32_t t,
                                    int32_t chunk_idx) {
  event0();

  // Reset h, c, and cache biases on (t==0, g==0, chunk==0).
  if (t == 0 && g == 0 && chunk_idx == 0) {
    aie::vector<float, VEC> zf = aie::broadcast<float, VEC>(0.0f);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      aie::store_v(h_state + v * VEC, zf);
      aie::store_v(c_state + v * VEC, zf);
    }
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int i = 0; i < BIAS_LEN_K; ++i) {
      bias_cache[i] = chunk_blk[i];
    }
  }

  // First chunk of a (t, g) pair: initialize gate accumulator with
  // (b_ih + b_hh) for this gate. Promote bf16 bias to FP32 once at
  // gate_acc seed time; subsequent chunk MACs accumulate FP32-into-FP32.
  if (chunk_idx == 0) {
    bfloat16 *restrict b_ih = bias_cache + (g * 2) * HIDDEN;
    bfloat16 *restrict b_hh = bias_cache + (g * 2 + 1) * HIDDEN;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      auto bi = aie::load_v<VEC>(b_ih + v * VEC);
      auto bh = aie::load_v<VEC>(b_hh + v * VEC);
      auto sum_bf = aie::add(bi, bh);
      // Promote the (bf16) bias sum to FP32 for storage in gate_acc.
      aie::accum<accfloat, VEC> a;
      a.from_vector(sum_bf);
      aie::vector<float, VEC> sum_f = a.to_vector<float>();
      aie::store_v(gate_acc[g] + v * VEC, sum_f);
    }
  }

  // The half-gate weight slab; skip the bias prefix.
  bfloat16 *restrict W_half = chunk_blk + BIAS_PREFIX;

  // Pick the source vector. For chunks 0/1 (W_ih @ x_t halves) the
  // source is the bf16 input. For chunks 2/3 (W_hh @ h halves) the
  // source is **h_state which is FP32**; we narrow lane-wise to bf16
  // for the matmul (AM020 Ch. 4 p. 65 hardware-free narrowing).
  bool h_path = (chunk_idx == 2) || (chunk_idx == 3);
  bfloat16 *src_x = nullptr;
  const float *src_h = nullptr;
  if (chunk_idx == 0)      src_x = x_t;
  else if (chunk_idx == 1) src_x = x_t + HALF_IN;
  else if (chunk_idx == 2) src_h = h_state;
  else                     src_h = h_state + HALF_IN;

  // For each output channel, accumulate W[oc, :] · src across HALF_IN
  // = 48 = 3 * VEC lanes via bf16 vector mac (FP32 accumulator).
  // gate_acc stays in FP32 storage; the chunk MAC does
  //   gate_acc[oc] = (FP32) gate_acc[oc] + sum(bf16 mul).
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < HIDDEN; ++oc) {
    aie::accum<accfloat, VEC> acc;
    acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
    bfloat16 *restrict w_row = W_half + oc * HALF_IN;
    if (h_path) {
      for (int j = 0; j < HALF_IN_VECS; ++j) {
        aie::vector<bfloat16, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
        // h is FP32 in tile DM; narrow on read to bf16 lane.
        aie::vector<bfloat16, VEC> s_v = load_h_as_bf16(src_h + j * VEC);
        acc = aie::mac(acc, w_v, s_v);
      }
    } else {
      for (int j = 0; j < HALF_IN_VECS; ++j) {
        aie::vector<bfloat16, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
        aie::vector<bfloat16, VEC> s_v = aie::load_v<VEC>(src_x + j * VEC);
        acc = aie::mac(acc, w_v, s_v);
      }
    }
    // Reduce the lane FP32 accumulator to a scalar.
    auto out_vec = acc.to_vector<float>();
    float sum = 0.0f;
    for (int k = 0; k < VEC; ++k) sum += out_vec.get(k);
    // Add to the running gate accumulator (FP32 storage; preserves
    // 23 mantissa bits across all 4 chunks of a (t, g) pair).
    float prev = gate_acc[g][oc];
    gate_acc[g][oc] = prev + sum;
  }

  // After the last chunk of the last gate (g==3, chunk==3), apply
  // nonlinearities, update c and h, emit y_t.
  //
  // Critical: the FP32→bf16 narrowing for the activation inputs is
  // the smallest necessary. The gate_acc values themselves were FP32
  // throughout the chunk MACs, so no precision was lost during the
  // gate accumulation. We narrow z_i/z_f/z_g/z_o to bf16 only because
  // aie::tanh<bfloat16> takes a vector<bfloat16> argument — that
  // narrowing is the same one a single FP32→bf16 cast would do. The
  // **state update** (c_t, h_t storage) stays FP32.
  if (g == 3 && chunk_idx == 3) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      // Load FP32 gate sums.
      aie::vector<float, VEC> z_i_f = aie::load_v<VEC>(gate_acc[0] + v * VEC);
      aie::vector<float, VEC> z_f_f = aie::load_v<VEC>(gate_acc[1] + v * VEC);
      aie::vector<float, VEC> z_g_f = aie::load_v<VEC>(gate_acc[2] + v * VEC);
      aie::vector<float, VEC> z_o_f = aie::load_v<VEC>(gate_acc[3] + v * VEC);

      // Narrow to bf16 for the activations (sigmoid and tanh).
      aie::accum<accfloat, VEC> ai, af, ag, ao;
      ai.from_vector(z_i_f);
      af.from_vector(z_f_f);
      ag.from_vector(z_g_f);
      ao.from_vector(z_o_f);
      aie::vector<bfloat16, VEC> z_i = ai.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_f = af.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_g = ag.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_o = ao.to_vector<bfloat16>();

      auto i_g = vec_sigmoid_bf16(z_i);
      auto f_g = vec_sigmoid_bf16(z_f);
      auto g_g = vec_tanh_bf16(z_g);
      auto o_g = vec_sigmoid_bf16(z_o);

      // c_new = f * c_old + i * g, computed as FP32 accumulator path.
      // Load c_old as FP32 (the persistent state); bf16 is only the
      // multiplier input width.
      aie::vector<float, VEC> c_old_f = aie::load_v<VEC>(c_state + v * VEC);
      aie::accum<accfloat, VEC> c_old_acc;
      c_old_acc.from_vector(c_old_f);
      aie::vector<bfloat16, VEC> c_old_bf = c_old_acc.to_vector<bfloat16>();

      // Multiply f * c_old (bf16 inputs, FP32 accumulator).
      aie::accum<accfloat, VEC> fc_acc;
      fc_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
      fc_acc = aie::mac(fc_acc, f_g, c_old_bf);

      // Multiply i * g (bf16 inputs, FP32 accumulator).
      aie::accum<accfloat, VEC> ig_acc;
      ig_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
      ig_acc = aie::mac(ig_acc, i_g, g_g);

      // Sum the two FP32 accumulator vectors and store as FP32.
      // This is the cross-walk's load-bearing path: c_t stays in FP32
      // across timesteps (no bf16 writeback narrowing).
      aie::vector<float, VEC> fc_f = fc_acc.to_vector<float>();
      aie::vector<float, VEC> ig_f = ig_acc.to_vector<float>();
      aie::vector<float, VEC> c_new_f = aie::add(fc_f, ig_f);
      aie::store_v(c_state + v * VEC, c_new_f);

      // h_new = o * tanh(c_new). Narrow c_new to bf16 only for the
      // tanh input. The h_state writeback is FP32.
      aie::accum<accfloat, VEC> c_new_acc;
      c_new_acc.from_vector(c_new_f);
      aie::vector<bfloat16, VEC> c_new_bf = c_new_acc.to_vector<bfloat16>();

      auto tanh_c = vec_tanh_bf16(c_new_bf);
      aie::accum<accfloat, VEC> h_acc;
      h_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
      h_acc = aie::mac(h_acc, o_g, tanh_c);

      // Persist h as FP32 (the load-bearing recurrent-state-precision
      // change vs ). y_t is the output stream which the host
      // expects as bf16 — narrow on write.
      aie::vector<float, VEC> h_new_f = h_acc.to_vector<float>();
      aie::store_v(h_state + v * VEC, h_new_f);
      aie::vector<bfloat16, VEC> h_new_bf = h_acc.to_vector<bfloat16>();
      aie::store_v(y_t + v * VEC, h_new_bf);
    }
  }

  event1();
}

} // extern "C"
