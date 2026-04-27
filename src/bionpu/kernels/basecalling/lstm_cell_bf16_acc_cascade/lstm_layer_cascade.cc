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
#include <aie2p/aie2p_streams.h>

constexpr int HIDDEN = 96;
constexpr int INPUT_DIM = 96;
constexpr int HALF_IN = INPUT_DIM / 2;       // 48
constexpr int N_GATES = 4;
constexpr int N_LAYERS_K = 5;
constexpr int VEC = 16;                      // bf16 vector lane width on AIE2P
constexpr int HIDDEN_VECS = HIDDEN / VEC;    // 6
constexpr int HALF_IN_VECS = HALF_IN / VEC;  // 3

// ===== Per-tile persistent state =====
// Each cascade-stage layer has its own h_state, c_state, gate_acc, and
// bias_cache in tile DM. All FP32 storage (the load-bearing precision
// invariant from ); only the matmul / activation inputs are
// narrowed to bf16 at use time.
//
// Cascade transfers: the accumulator-to-accumulator stream is 512 bits
// per cycle; we cascade FP32 hidden state (16 floats per cascade word).
// h is the only payload — c is private to each layer (LSTMs don't
// share c across layers; preserves the exact math contract).

static float h_state[HIDDEN];
static float c_state[HIDDEN];
static float gate_acc[N_GATES][HIDDEN];
constexpr int BIAS_LEN_K = N_GATES * 2 * HIDDEN; // 768
static bfloat16 bias_cache[BIAS_LEN_K];
constexpr int BIAS_PREFIX = BIAS_LEN_K;
constexpr int CHUNK_LEN_K = BIAS_LEN_K + HIDDEN * HALF_IN; // 5376
constexpr int CHUNK_FRAME_LEN_K = N_LAYERS_K * CHUNK_LEN_K;

// ===== Activation helpers (verbatim from ) =====

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

static inline aie::vector<bfloat16, VEC>
load_h_as_bf16(const float *restrict src) {
  aie::vector<float, VEC> vf = aie::load_v<VEC>(src);
  aie::accum<accfloat, VEC> a;
  a.from_vector(vf);
  return a.to_vector<bfloat16>();
}

// ===== Shared LSTM-cell forward step =====
//
// Performs one chunk of one gate of one timestep: matmul, gate-acc
// update. Identical to except:
//   - ``x_t_bf16`` source: caller passes either an ObjectFifo input
//     (FIRST role) or a bf16-narrowed cascade-in payload (MIDDLE/LAST).
//   - The (g==3,chunk==3) post-step state update is now a SEPARATE
//     helper (`lstm_final_chunk_state_update`) called by the
//     `_putonly` / `_writeout` extern "C" variants only — so the
//     `_math` / `_getonly` variants never need a runtime branch over
//     the activations + state writeback.
//
// This helper takes the bf16 input pointer for the current half-pair
// (chunks 0/1 of W_ih @ x), which the caller selects from either x_t
// (ObjectFifo) or a tile-local bf16 buffer staged from cascade-in.
//
// Internal branches in this helper (init at (t==0,g==0,chunk==0); gate
// seed at chunk==0; h_path on chunks 2/3) do NOT involve cascade
// intrinsics, so per Followup E's silicon-validated invariant they do
// NOT trigger the AIE2P firmware-side cascade-port watchdog.

static inline void
lstm_chunk_step(const bfloat16 *restrict x_t_bf16,
                bfloat16 *restrict chunk_blk,
                int g, int t, int chunk_idx) {
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

  // First chunk of a (t, g) pair: seed gate_acc with (b_ih + b_hh).
  if (chunk_idx == 0) {
    bfloat16 *restrict b_ih = bias_cache + (g * 2) * HIDDEN;
    bfloat16 *restrict b_hh = bias_cache + (g * 2 + 1) * HIDDEN;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      auto bi = aie::load_v<VEC>(b_ih + v * VEC);
      auto bh = aie::load_v<VEC>(b_hh + v * VEC);
      auto sum_bf = aie::add(bi, bh);
      aie::accum<accfloat, VEC> a;
      a.from_vector(sum_bf);
      aie::vector<float, VEC> sum_f = a.to_vector<float>();
      aie::store_v(gate_acc[g] + v * VEC, sum_f);
    }
  }

  bfloat16 *restrict W_half = chunk_blk + BIAS_PREFIX;

  // Source vector: chunks 0/1 use x_t (bf16 input); chunks 2/3 use h
  // (FP32 in tile DM, narrowed lane-wise for the matmul).
  bool h_path = (chunk_idx == 2) || (chunk_idx == 3);
  const bfloat16 *src_x = nullptr;
  const float *src_h = nullptr;
  if (chunk_idx == 0)      src_x = x_t_bf16;
  else if (chunk_idx == 1) src_x = x_t_bf16 + HALF_IN;
  else if (chunk_idx == 2) src_h = h_state;
  else                     src_h = h_state + HALF_IN;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < HIDDEN; ++oc) {
    aie::accum<accfloat, VEC> acc;
    acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
    bfloat16 *restrict w_row = W_half + oc * HALF_IN;
    if (h_path) {
      for (int j = 0; j < HALF_IN_VECS; ++j) {
        aie::vector<bfloat16, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
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
    auto out_vec = acc.to_vector<float>();
    float sum = 0.0f;
    for (int k = 0; k < VEC; ++k) sum += out_vec.get(k);
    float prev = gate_acc[g][oc];
    gate_acc[g][oc] = prev + sum;
  }
}

// At the end of (g==3, chunk==3), apply nonlinearities, update c and h
// in FP32 storage. Returns nothing — the caller (FIRST/MIDDLE _putonly
// or LAST _writeout) cascades h_state or writes h_state-as-bf16 to y_t.
static inline void
lstm_final_chunk_state_update() {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::vector<float, VEC> z_i_f = aie::load_v<VEC>(gate_acc[0] + v * VEC);
    aie::vector<float, VEC> z_f_f = aie::load_v<VEC>(gate_acc[1] + v * VEC);
    aie::vector<float, VEC> z_g_f = aie::load_v<VEC>(gate_acc[2] + v * VEC);
    aie::vector<float, VEC> z_o_f = aie::load_v<VEC>(gate_acc[3] + v * VEC);

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

    aie::vector<float, VEC> c_old_f = aie::load_v<VEC>(c_state + v * VEC);
    aie::accum<accfloat, VEC> c_old_acc;
    c_old_acc.from_vector(c_old_f);
    aie::vector<bfloat16, VEC> c_old_bf = c_old_acc.to_vector<bfloat16>();

    aie::accum<accfloat, VEC> fc_acc;
    fc_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
    fc_acc = aie::mac(fc_acc, f_g, c_old_bf);

    aie::accum<accfloat, VEC> ig_acc;
    ig_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
    ig_acc = aie::mac(ig_acc, i_g, g_g);

    aie::vector<float, VEC> fc_f = fc_acc.to_vector<float>();
    aie::vector<float, VEC> ig_f = ig_acc.to_vector<float>();
    aie::vector<float, VEC> c_new_f = aie::add(fc_f, ig_f);
    aie::store_v(c_state + v * VEC, c_new_f);

    aie::accum<accfloat, VEC> c_new_acc;
    c_new_acc.from_vector(c_new_f);
    aie::vector<bfloat16, VEC> c_new_bf = c_new_acc.to_vector<bfloat16>();

    auto tanh_c = vec_tanh_bf16(c_new_bf);
    aie::accum<accfloat, VEC> h_acc;
    h_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
    h_acc = aie::mac(h_acc, o_g, tanh_c);

    aie::vector<float, VEC> h_new_f = h_acc.to_vector<float>();
    aie::store_v(h_state + v * VEC, h_new_f);
  }
}

// ===== Cascade I/O helpers (UNCHANGED from prior version) =====
//
// AM020 Ch. 4 p. 67: cascade transfer = 512 bits/cycle. 16 lanes of
// FP32 (accfloat) per word. To cascade HIDDEN=96 floats per timestep
// we issue HIDDEN_VECS=6 puts (or gets) at a final-chunk boundary.
//
// **Followup G note (Fix C invariant)**: these helpers MUST NOT be
// called from a runtime-conditional context in any extern "C" caller
// — they emit `vmov mcd` / `vmov scd` cascade-port intrinsics which
// the AIE2P firmware's cascade-port watchdog (Followup C/E hypothesis,
// silicon-confirmed) requires to be unconditional within their
// containing function.

static inline void cascade_put_h_state() {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::vector<float, VEC> h_lane = aie::load_v<VEC>(h_state + v * VEC);
    aie::accum<accfloat, VEC> a;
    a.from_vector(h_lane);
    // put_mcd expects the accumulator-typed value; the accfloat
    // accumulator is the cascade word width on AIE-ML/AIE2P.
    put_mcd(a);
  }
}

static inline void cascade_get_h_state(float *restrict dst) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    // Read one cascade word; reinterpret as FP32 lanes.
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    aie::vector<float, VEC> h_lane = a.to_vector<float>();
    aie::store_v(dst + v * VEC, h_lane);
  }
}

// Stage cascade-in into the layer's "x_t-equivalent" bf16 buffer for
// chunks 0/1 of the next timestep. The previous layer's h IS this
// layer's matmul input (Bonito's stack), narrowed bf16 once per cell.
// We reuse a per-tile bf16 staging buffer of length INPUT_DIM=96.
static bfloat16 cascade_in_x_bf16[INPUT_DIM];

static inline void cascade_in_to_bf16_x(const float *src) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::vector<float, VEC> vf = aie::load_v<VEC>(src + v * VEC);
    aie::accum<accfloat, VEC> a;
    a.from_vector(vf);
    aie::vector<bfloat16, VEC> v_bf = a.to_vector<bfloat16>();
    aie::store_v(cascade_in_x_bf16 + v * VEC, v_bf);
  }
}

// Helper: pure write-out of h_state (FP32) → y_t (bf16). Used by the
// LAST role's writeout variant after the final state update. NOT a
// cascade intrinsic — peano lowers this to plain DM stores.
static inline void write_h_state_as_bf16(bfloat16 *restrict y_t) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::vector<float, VEC> h_lane = aie::load_v<VEC>(h_state + v * VEC);
    aie::accum<accfloat, VEC> a;
    a.from_vector(h_lane);
    aie::vector<bfloat16, VEC> h_bf = a.to_vector<bfloat16>();
    aie::store_v(y_t + v * VEC, h_bf);
  }
}

extern "C" {

// =====================================================================
// FIRST role (Layer 0 / col 0 row 5) — bf16 input via ObjectFifo,
// FP32 h cascaded out via AccumFifo on (g==3,chunk==3).
// =====================================================================

// FIRST role math-only — called from IRON for 15 of the 16 (g, chunk)
// call-sites per ts. NO cascade intrinsic, NO state update.
void dorado_fast_lstm_layer_cascade_first_math(bfloat16 *restrict x_t,
                                                 bfloat16 *restrict chunk_blk,
                                                 int32_t g,
                                                 int32_t t,
                                                 int32_t chunk_idx) {
  event0();
  lstm_chunk_step(x_t, chunk_blk, g, t, chunk_idx);
  event1();
}

// FIRST role cascade-put — called from IRON ONLY at (g==3,chunk==3)
// per ts. Math + final-chunk state update + UNCONDITIONAL cascade put.
// NO runtime branch around the cascade ops.
void dorado_fast_lstm_layer_cascade_first_putonly(bfloat16 *restrict x_t,
                                                    bfloat16 *restrict chunk_blk,
                                                    int32_t g,
                                                    int32_t t,
                                                    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(x_t, chunk_blk, g, t, chunk_idx);
  // The IRON topology guarantees this function is only invoked at the
  // (g==3,chunk==3) call-site, so the state update + cascade put are
  // unconditional from peano's POV — no `jnz` around `vmov mcd`.
  lstm_final_chunk_state_update();
  cascade_put_h_state();
  event1();
}

// =====================================================================
// MIDDLE role (Layers 1-3 / col 0 rows 4..2) — FP32 h from AccumFifo
// upstream on (g==0,chunk==0), FP32 h cascaded out on (g==3,chunk==3).
// =====================================================================

// MIDDLE role math-only — 14 of the 16 call-sites per ts. NO cascade
// intrinsic, NO state update. Reads the per-tile bf16 staging buffer
// (cascade_in_x_bf16) populated by the prior call's _getonly variant
// at (g==0,chunk==0).
void dorado_fast_lstm_layer_cascade_middle_math(bfloat16 *restrict chunk_blk,
                                                  int32_t g,
                                                  int32_t t,
                                                  int32_t chunk_idx) {
  event0();
  lstm_chunk_step(cascade_in_x_bf16, chunk_blk, g, t, chunk_idx);
  event1();
}

// MIDDLE role cascade-get — called from IRON ONLY at (g==0,chunk==0)
// per ts. UNCONDITIONAL cascade get + math. NO `jnz` around `vmov scd`.
void dorado_fast_lstm_layer_cascade_middle_getonly(bfloat16 *restrict chunk_blk,
                                                     int32_t g,
                                                     int32_t t,
                                                     int32_t chunk_idx) {
  event0();
  // Cascade get UNCONDITIONAL at function entry. The IRON topology
  // guarantees this function is only invoked at (g==0,chunk==0).
  static float cascade_in_h[INPUT_DIM];
  cascade_get_h_state(cascade_in_h);
  cascade_in_to_bf16_x(cascade_in_h);
  lstm_chunk_step(cascade_in_x_bf16, chunk_blk, g, t, chunk_idx);
  event1();
}

// MIDDLE role cascade-put — called from IRON ONLY at (g==3,chunk==3)
// per ts. Math + final-chunk state update + UNCONDITIONAL cascade put.
void dorado_fast_lstm_layer_cascade_middle_putonly(bfloat16 *restrict chunk_blk,
                                                     int32_t g,
                                                     int32_t t,
                                                     int32_t chunk_idx) {
  event0();
  lstm_chunk_step(cascade_in_x_bf16, chunk_blk, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  cascade_put_h_state();
  event1();
}

// =====================================================================
// LAST role (Layer 4 / col 1 row 2) — FP32 h from AccumFifo upstream
// on (g==0,chunk==0); bf16 y_t output via ObjectFifo on (g==3,chunk==3).
// =====================================================================

// LAST role math-only — 14 of the 16 call-sites per ts. NO cascade,
// NO state update, NO y_out write. y_t arg is unused (kept for ABI
// uniformity with the other LAST variants so the IRON topology can
// pass the same args to all three).
void dorado_fast_lstm_layer_cascade_last_math(bfloat16 *restrict chunk_blk,
                                                bfloat16 *restrict y_t,
                                                int32_t g,
                                                int32_t t,
                                                int32_t chunk_idx) {
  event0();
  (void)y_t;
  lstm_chunk_step(cascade_in_x_bf16, chunk_blk, g, t, chunk_idx);
  event1();
}

// LAST role cascade-get — called from IRON ONLY at (g==0,chunk==0)
// per ts. UNCONDITIONAL cascade get + math.
void dorado_fast_lstm_layer_cascade_last_getonly(bfloat16 *restrict chunk_blk,
                                                   bfloat16 *restrict y_t,
                                                   int32_t g,
                                                   int32_t t,
                                                   int32_t chunk_idx) {
  event0();
  (void)y_t;
  static float cascade_in_h[INPUT_DIM];
  cascade_get_h_state(cascade_in_h);
  cascade_in_to_bf16_x(cascade_in_h);
  lstm_chunk_step(cascade_in_x_bf16, chunk_blk, g, t, chunk_idx);
  event1();
}

// LAST role write-out — called from IRON ONLY at (g==3,chunk==3) per
// ts. Math + final-chunk state update + UNCONDITIONAL bf16 write to
// y_t. NO cascade intrinsic — but the y_t write would otherwise need
// a runtime branch in the monolithic kernel, so we split it out here
// for hygiene + symmetry with the FIRST/MIDDLE _putonly variants.
void dorado_fast_lstm_layer_cascade_last_writeout(bfloat16 *restrict chunk_blk,
                                                    bfloat16 *restrict y_t,
                                                    int32_t g,
                                                    int32_t t,
                                                    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(cascade_in_x_bf16, chunk_blk, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  write_h_state_as_bf16(y_t);
  event1();
}

} // extern "C"

// ===== Dorado production-weight frame chain bisection =====
//
// This keeps the production consolidated weight-frame ABI:
// [L0_chunk, L1_chunk, L2_chunk, L3_chunk, L4_chunk], one frame per
// (t, gate, chunk). Unlike the production artifact it does not use
// ObjectFifoHandle.split/aie.objectfifo.link. Instead each core forwards
// the whole frame to the next core and consumes its own layer slice.
//
// The IRON side must forward/release all 16 frames for a timestep before
// waiting on cascade, otherwise downstream cannot see later weight tokens
// while it is blocked on the previous layer's final hidden state.

extern "C" {

void dorado_prod_weight_frame_forward(
    bfloat16 *restrict frame_in,
    bfloat16 *restrict frame_out) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int i = 0; i < CHUNK_FRAME_LEN_K; ++i) {
    frame_out[i] = frame_in[i];
  }
}

void dorado_prod_frame_first_math(
    bfloat16 *restrict x_t,
    bfloat16 *restrict frame,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(x_t, frame, g, t, chunk_idx);
  event1();
}

void dorado_prod_frame_first_putonly(
    bfloat16 *restrict x_t,
    bfloat16 *restrict frame,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(x_t, frame, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  cascade_put_h_state();
  event1();
}

void dorado_prod_frame_middle_math(
    bfloat16 *restrict frame,
    int32_t layer,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(
      cascade_in_x_bf16, frame + layer * CHUNK_LEN_K, g, t, chunk_idx);
  event1();
}

void dorado_prod_frame_middle_getonly(
    bfloat16 *restrict frame,
    int32_t layer,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  static float cascade_in_h[INPUT_DIM];
  cascade_get_h_state(cascade_in_h);
  cascade_in_to_bf16_x(cascade_in_h);
  lstm_chunk_step(
      cascade_in_x_bf16, frame + layer * CHUNK_LEN_K, g, t, chunk_idx);
  event1();
}

void dorado_prod_frame_middle_putonly(
    bfloat16 *restrict frame,
    int32_t layer,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(
      cascade_in_x_bf16, frame + layer * CHUNK_LEN_K, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  cascade_put_h_state();
  event1();
}

void dorado_prod_frame_last_math(
    bfloat16 *restrict frame,
    bfloat16 *restrict y_t,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  (void)y_t;
  lstm_chunk_step(
      cascade_in_x_bf16, frame + (N_LAYERS_K - 1) * CHUNK_LEN_K,
      g, t, chunk_idx);
  event1();
}

void dorado_prod_frame_last_getonly(
    bfloat16 *restrict frame,
    bfloat16 *restrict y_t,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  (void)y_t;
  static float cascade_in_h[INPUT_DIM];
  cascade_get_h_state(cascade_in_h);
  cascade_in_to_bf16_x(cascade_in_h);
  lstm_chunk_step(
      cascade_in_x_bf16, frame + (N_LAYERS_K - 1) * CHUNK_LEN_K,
      g, t, chunk_idx);
  event1();
}

void dorado_prod_frame_last_writeout(
    bfloat16 *restrict frame,
    bfloat16 *restrict y_t,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  lstm_chunk_step(
      cascade_in_x_bf16, frame + (N_LAYERS_K - 1) * CHUNK_LEN_K,
      g, t, chunk_idx);
  lstm_final_chunk_state_update();
  write_h_state_as_bf16(y_t);
  event1();
}

} // extern "C"

// ===== Dorado production-weight direct stream bisection =====
//
// The full production chunk frame is 5 * 5376 bf16 = 53,760 B. Sending that
// frame over ordinary inter-core ObjectFifos exceeds tile memory on middle
// tiles because they need both input and output frame buffers. This variant
// keeps the host-visible production frame ABI on the first tile, but forwards
// only the remaining suffix over Core stream 0. Middle tiles buffer just their
// own 5376-bf16 chunk, forward the tail, then run the normal LSTM math.

static bfloat16 prod_stream_chunk_buf[CHUNK_LEN_K];

static inline uint32_t pack_bf16_pair(const bfloat16 *restrict p) {
  uint16_t lo = *reinterpret_cast<const uint16_t *>(p);
  uint16_t hi = *reinterpret_cast<const uint16_t *>(p + 1);
  return ((uint32_t)hi << 16) | (uint32_t)lo;
}

static inline void unpack_bf16_pair(uint32_t word, bfloat16 *restrict p) {
  uint16_t lo = (uint16_t)(word & 0xffffu);
  uint16_t hi = (uint16_t)((word >> 16) & 0xffffu);
  p[0] = *reinterpret_cast<bfloat16 *>(&lo);
  p[1] = *reinterpret_cast<bfloat16 *>(&hi);
}

static inline void stream_send_bf16(const bfloat16 *restrict src,
                                    int32_t n_bf16) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int32_t i = 0; i < n_bf16; i += 2) {
    put_ms((int)pack_bf16_pair(src + i), 0);
  }
}

static inline void stream_receive_own_chunk_forward_tail(
    int32_t layers_remaining) {
  int32_t total_bf16 = layers_remaining * CHUNK_LEN_K;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int32_t i = 0; i < total_bf16; i += 2) {
    uint32_t word = (uint32_t)get_ss_int();
    if (i < CHUNK_LEN_K) {
      unpack_bf16_pair(word, prod_stream_chunk_buf + i);
    } else {
      put_ms((int)word, 0);
    }
  }
}

static inline void stream_receive_last_chunk() {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int32_t i = 0; i < CHUNK_LEN_K; i += 2) {
    uint32_t word = (uint32_t)get_ss_int();
    unpack_bf16_pair(word, prod_stream_chunk_buf + i);
  }
}

extern "C" {

void dorado_prod_stream_first_math(
    bfloat16 *restrict x_t,
    bfloat16 *restrict frame,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  stream_send_bf16(frame + CHUNK_LEN_K, (N_LAYERS_K - 1) * CHUNK_LEN_K);
  lstm_chunk_step(x_t, frame, g, t, chunk_idx);
  event1();
}

void dorado_prod_stream_first_putonly(
    bfloat16 *restrict x_t,
    bfloat16 *restrict frame,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  stream_send_bf16(frame + CHUNK_LEN_K, (N_LAYERS_K - 1) * CHUNK_LEN_K);
  lstm_chunk_step(x_t, frame, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  cascade_put_h_state();
  event1();
}

void dorado_prod_stream_middle_math(
    int32_t layers_remaining,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  stream_receive_own_chunk_forward_tail(layers_remaining);
  lstm_chunk_step(cascade_in_x_bf16, prod_stream_chunk_buf, g, t, chunk_idx);
  event1();
}

void dorado_prod_stream_middle_getonly(
    int32_t layers_remaining,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  stream_receive_own_chunk_forward_tail(layers_remaining);
  static float cascade_in_h[INPUT_DIM];
  cascade_get_h_state(cascade_in_h);
  cascade_in_to_bf16_x(cascade_in_h);
  lstm_chunk_step(cascade_in_x_bf16, prod_stream_chunk_buf, g, t, chunk_idx);
  event1();
}

void dorado_prod_stream_middle_putonly(
    int32_t layers_remaining,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  stream_receive_own_chunk_forward_tail(layers_remaining);
  lstm_chunk_step(cascade_in_x_bf16, prod_stream_chunk_buf, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  cascade_put_h_state();
  event1();
}

void dorado_prod_stream_last_math(
    bfloat16 *restrict y_t,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  (void)y_t;
  stream_receive_last_chunk();
  lstm_chunk_step(cascade_in_x_bf16, prod_stream_chunk_buf, g, t, chunk_idx);
  event1();
}

void dorado_prod_stream_last_getonly(
    bfloat16 *restrict y_t,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  (void)y_t;
  stream_receive_last_chunk();
  static float cascade_in_h[INPUT_DIM];
  cascade_get_h_state(cascade_in_h);
  cascade_in_to_bf16_x(cascade_in_h);
  lstm_chunk_step(cascade_in_x_bf16, prod_stream_chunk_buf, g, t, chunk_idx);
  event1();
}

void dorado_prod_stream_last_writeout(
    bfloat16 *restrict y_t,
    int32_t g,
    int32_t t,
    int32_t chunk_idx) {
  event0();
  stream_receive_last_chunk();
  lstm_chunk_step(cascade_in_x_bf16, prod_stream_chunk_buf, g, t, chunk_idx);
  lstm_final_chunk_state_update();
  write_h_state_as_bf16(y_t);
  event1();
}

void dorado_prod_stream_probe_first(bfloat16 *restrict frame) {
  event0();
  stream_send_bf16(frame + CHUNK_LEN_K, (N_LAYERS_K - 1) * CHUNK_LEN_K);
  event1();
}

void dorado_prod_stream_probe_middle(int32_t layers_remaining) {
  event0();
  stream_receive_own_chunk_forward_tail(layers_remaining);
  event1();
}

void dorado_prod_stream_probe_last(bfloat16 *restrict y_t) {
  event0();
  stream_receive_last_chunk();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto vals = aie::load_v<VEC>(prod_stream_chunk_buf + v * VEC);
    aie::store_v(y_t + v * VEC, vals);
  }
  event1();
}

void dorado_prod_direct_layer_probe_touch(bfloat16 *restrict chunk) {
  event0();
  volatile uint16_t sink = *reinterpret_cast<volatile uint16_t *>(chunk);
  (void)sink;
  event1();
}

void dorado_prod_direct_layer_probe_last(
    bfloat16 *restrict chunk,
    bfloat16 *restrict y_t) {
  event0();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto vals = aie::load_v<VEC>(chunk + v * VEC);
    aie::store_v(y_t + v * VEC, vals);
  }
  event1();
}

static inline void direct_group_math_pressure(bfloat16 *restrict chunk) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto wb = aie::load_v<VEC>(chunk + BIAS_PREFIX + v * VEC);
    auto hb = aie::load_v<VEC>(chunk + v * VEC);
    auto t = vec_tanh_bf16(wb);
    auto s = vec_sigmoid_bf16(hb);
    auto mixed = aie::add(t, s);
    aie::accum<accfloat, VEC> acc;
    acc.from_vector(mixed);
    auto f = acc.to_vector<float>();
    aie::store_v(gate_acc[v % N_GATES] + v * VEC, f);
    aie::store_v(h_state + v * VEC, f);
  }
}

void dorado_prod_direct_group_math_probe_touch(bfloat16 *restrict chunk) {
  event0();
  direct_group_math_pressure(chunk);
  event1();
}

void dorado_prod_direct_group_math_probe_last(
    bfloat16 *restrict chunk,
    bfloat16 *restrict y_t) {
  event0();
  direct_group_math_pressure(chunk);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto vals = aie::load_v<VEC>(chunk + v * VEC);
    aie::store_v(y_t + v * VEC, vals);
  }
  event1();
}

void dorado_prod_direct_group_cascade_probe_first(bfloat16 *restrict chunk) {
  event0();
  direct_group_math_pressure(chunk);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto h = aie::load_v<VEC>(h_state + v * VEC);
    aie::accum<accfloat, VEC> a;
    a.from_vector(h);
    put_mcd(a);
  }
  event1();
}

void dorado_prod_direct_group_cascade_probe_second(bfloat16 *restrict chunk) {
  event0();
  direct_group_math_pressure(chunk);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto bf = a.to_vector<bfloat16>();
    auto t = vec_tanh_bf16(bf);
    auto s = vec_sigmoid_bf16(bf);
    auto mixed = aie::add(t, s);
    aie::accum<accfloat, VEC> out;
    out.from_vector(mixed);
    auto f = out.to_vector<float>();
    aie::store_v(c_state + v * VEC, f);
  }
  event1();
}

void dorado_prod_direct_group_cascade_probe_middle(bfloat16 *restrict chunk) {
  event0();
  direct_group_math_pressure(chunk);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto bf = a.to_vector<bfloat16>();
    auto t = vec_tanh_bf16(bf);
    auto s = vec_sigmoid_bf16(bf);
    auto mixed = aie::add(t, s);
    aie::accum<accfloat, VEC> out;
    out.from_vector(mixed);
    put_mcd(out);
  }
  event1();
}

void dorado_prod_direct_group_cascade_probe_last(
    bfloat16 *restrict chunk,
    bfloat16 *restrict y_t) {
  event0();
  direct_group_math_pressure(chunk);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto bf = a.to_vector<bfloat16>();
    auto t = vec_tanh_bf16(bf);
    auto s = vec_sigmoid_bf16(bf);
    auto mixed = aie::add(t, s);
    aie::accum<accfloat, VEC> out;
    out.from_vector(mixed);
    auto f = out.to_vector<float>();
    aie::store_v(c_state + v * VEC, f);
    auto vals = aie::load_v<VEC>(chunk + v * VEC);
    aie::store_v(y_t + v * VEC, vals);
  }
  event1();
}

} // extern "C"

// ===== Dorado hidden-state cascade-only bisection =====
//
// This is the post-CRISPR "split the contract" probe. It keeps the
// production hidden-state payload shape (96 lanes = 6 cascade words),
// static tile state, and the 5-stage vertical+horizontal cascade chain,
// but removes the consolidated weight ObjectFifo/memtile split and the
// heavy LSTM gate math. A silicon PASS here says the next suspect is
// weight delivery or math pressure, not the hidden-state handoff itself.

extern "C" {

void dorado_hidden_stream_first(bfloat16 *restrict x_in) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto xb = aie::load_v<VEC>(x_in + v * VEC);
    aie::accum<accfloat, VEC> a;
    a.from_vector(xb);
    auto xf = a.to_vector<float>();
    aie::store_v(h_state + v * VEC, xf);
    put_mcd(a);
  }

  event1();
}

void dorado_hidden_stream_middle() {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto hf = a.to_vector<float>();
    aie::store_v(h_state + v * VEC, hf);
    put_mcd(a);
  }

  event1();
}

void dorado_hidden_stream_last(bfloat16 *restrict y_out) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto hf = a.to_vector<float>();
    aie::store_v(h_state + v * VEC, hf);
    aie::store_v(y_out + v * VEC, a.to_vector<bfloat16>());
  }

  event1();
}

} // extern "C"

// ===== Dorado hidden-state + math-pressure bisection =====
//
// Same no-weight ABI as the hidden-state-only probe, but adds tile-local
// static state updates and aie_api nonlinear math on every cascade word.
// This isolates "static state + heavy math + cascade" from weight delivery.

static inline aie::accum<accfloat, VEC>
hidden_math_transform(aie::accum<accfloat, VEC> in, int v) {
  auto bf = in.to_vector<bfloat16>();
  auto t = vec_tanh_bf16(bf);
  auto s = vec_sigmoid_bf16(bf);
  auto mixed = aie::add(t, s);
  aie::accum<accfloat, VEC> out;
  out.from_vector(mixed);
  auto f = out.to_vector<float>();
  aie::store_v(gate_acc[v % N_GATES] + v * VEC, f);
  aie::store_v(h_state + v * VEC, f);
  return out;
}

extern "C" {

void dorado_hidden_math_first(bfloat16 *restrict x_in) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto xb = aie::load_v<VEC>(x_in + v * VEC);
    aie::accum<accfloat, VEC> a;
    a.from_vector(xb);
    put_mcd(hidden_math_transform(a, v));
  }

  event1();
}

void dorado_hidden_math_middle() {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    put_mcd(hidden_math_transform(a, v));
  }

  event1();
}

void dorado_hidden_math_last(bfloat16 *restrict y_out) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto out = hidden_math_transform(a, v);
    aie::store_v(y_out + v * VEC, out.to_vector<bfloat16>());
  }

  event1();
}

} // extern "C"

// ===== Dorado hidden-state + simple-weight bisection =====
//
// Adds a single ordinary weight ObjectFifo into the FIRST stage only.
// There is no memtile split and no fanout. This isolates basic weight
// DMA/ObjectFifo pressure from the consolidated split topology.

extern "C" {

void dorado_hidden_weight_first(
    bfloat16 *restrict x_in,
    bfloat16 *restrict w_in) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto xb = aie::load_v<VEC>(x_in + v * VEC);
    auto wb = aie::load_v<VEC>(w_in + v * VEC);
    auto xw = aie::add(xb, wb);
    aie::accum<accfloat, VEC> a;
    a.from_vector(xw);
    put_mcd(hidden_math_transform(a, v));
  }

  event1();
}

void dorado_hidden_weight_chain_first(
    bfloat16 *restrict x_in,
    bfloat16 *restrict w_in,
    bfloat16 *restrict w_out) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto xb = aie::load_v<VEC>(x_in + v * VEC);
    auto wb = aie::load_v<VEC>(w_in + v * VEC);
    aie::store_v(w_out + v * VEC, wb);
    auto xw = aie::add(xb, wb);
    aie::accum<accfloat, VEC> a;
    a.from_vector(xw);
    put_mcd(hidden_math_transform(a, v));
  }

  event1();
}

void dorado_hidden_weight_forward(
    bfloat16 *restrict w_in,
    bfloat16 *restrict w_out) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    auto wb = aie::load_v<VEC>(w_in + v * VEC);
    aie::store_v(w_out + v * VEC, wb);
  }
}

void dorado_hidden_weight_chain_middle_compute(bfloat16 *restrict w_in) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto wb = aie::load_v<VEC>(w_in + v * VEC);
    aie::accum<accfloat, VEC> wa;
    wa.from_vector(wb);
    put_mcd(hidden_math_transform(aie::add(a, wa), v));
  }

  event1();
}

void dorado_hidden_weight_chain_middle(
    bfloat16 *restrict w_in,
    bfloat16 *restrict w_out) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto wb = aie::load_v<VEC>(w_in + v * VEC);
    aie::store_v(w_out + v * VEC, wb);
    aie::accum<accfloat, VEC> wa;
    wa.from_vector(wb);
    put_mcd(hidden_math_transform(aie::add(a, wa), v));
  }

  event1();
}

void dorado_hidden_weight_chain_last(
    bfloat16 *restrict w_in,
    bfloat16 *restrict y_out) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int v = 0; v < HIDDEN_VECS; ++v) {
    aie::accum<accfloat, VEC> a = get_scd_v16accfloat();
    auto wb = aie::load_v<VEC>(w_in + v * VEC);
    aie::accum<accfloat, VEC> wa;
    wa.from_vector(wb);
    auto out = hidden_math_transform(aie::add(a, wa), v);
    aie::store_v(y_out + v * VEC, out.to_vector<bfloat16>());
  }

  event1();
}

} // extern "C"
