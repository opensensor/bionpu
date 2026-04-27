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
// AIE2P INT8 vector lane width: 32 INT8 lanes per vector for the 256
// ops/tile/cycle path (AM020 Table 14). Inner matmul uses VEC_I8 lanes.
constexpr int VEC = 16; // bf16/FP32 lane width for activations
constexpr int VEC_I8 = 32; // int8 lane width for matmul
constexpr int HIDDEN_VECS = HIDDEN / VEC; // 6
constexpr int HALF_IN_VECS_I8 = HALF_IN / VEC_I8; // 1 (48/32 rounds — special)

// Tile-resident persistent state.
//
// h, c live as FP32 between timesteps:
// preserves 23-mantissa-bit precision across the recurrent step. The
// matmul input to W_hh @ h is re-quantized to INT8 on each timestep
// (lossy, but per-channel calibration absorbs most of the noise; 's
// honest deviation policy applies).
//
// gate_acc is INT32 (per gate, per output channel) since the INT8
// MAC chain accumulates in INT32 natively on AIE2P.
//
// bias_cache holds FP32 biases that the host PRE-MULTIPLIES with the
// per-gate scale chain (s_x * s_w_ih and s_h * s_w_hh) so the on-tile
// dequantization is a single FP32-multiply per gate (s_total[g]).

static float h_state[HIDDEN];
static float c_state[HIDDEN];
static int32_t gate_acc[N_GATES][HIDDEN];
constexpr int BIAS_LEN_K = N_GATES * 2 * HIDDEN; // 768
static float bias_cache[BIAS_LEN_K];
// Per-gate combined scale: applied to int32 gate_acc to yield FP32 z_g.
// per_gate_scale_x[g] = s_x * s_w_ih[g] (applied on chunks 0,1)
// per_gate_scale_h[g] = s_h * s_w_hh[g] (applied on chunks 2,3)
// In practice both are folded into one cached `per_gate_scale[g]`
// because chunks 0+1 vs 2+3 use DIFFERENT scale chains; the kernel
// stores 8 scale floats: 4 for x-path and 4 for h-path.
static float per_gate_scale_x[N_GATES];
static float per_gate_scale_h[N_GATES];
// Output requantization params (h -> INT8 wire format).
static float y_scale; // FP32 multiplier; y_q = round(h / y_scale)
// Re-quantization params for h-path matmul input (h_state -> int8).
static float h_scale;

// Bias prefix layout:
//   [0  ..  4  ): per_gate_scale_x[0..3]    (4 floats = 16 bytes)
//   [4  ..  8  ): per_gate_scale_h[0..3]    (4 floats = 16 bytes)
//   [8  ..  9  ): h_scale                   (1 float  =  4 bytes)
//   [9  .. 10  ): y_scale                   (1 float  =  4 bytes)
//   [10 .. 778 ): bias_cache[0..767]        (768 floats)
// Total prefix: 778 floats = 3112 bytes (rounded up to align with
// the int8 weight slab). The host pads to a 32-byte boundary so the
// int8 lane loads land aligned.
constexpr int SCALE_PREFIX = 4 + 4 + 1 + 1; // 10 floats
constexpr int FLOAT_PREFIX = SCALE_PREFIX + BIAS_LEN_K; // 778 floats
// Pad up so the int8 slab starts on a 32-byte boundary (VEC_I8 lanes).
// 778 floats = 3112 bytes → round up to 3136 = 784 floats = 8 extra
// floats of padding.
constexpr int FLOAT_PREFIX_ALIGNED = 784; // bytes 3136 = 32 * 98
constexpr int BIAS_PREFIX_BYTES = FLOAT_PREFIX_ALIGNED * 4; // 3136

// Half-gate weight slab: HIDDEN * HALF_IN INT8 = 4608 INT8 = 4608 bytes.
constexpr int WEIGHT_HALF_BYTES = HIDDEN * HALF_IN; // 4608

// The wire-format chunk is FLOAT_PREFIX_ALIGNED * 4 + WEIGHT_HALF_BYTES
// bytes. Host packs this; kernel reads via byte-pointer + casts.

// Vectorised sigmoid via tanh identity, using AIE2P hardware bf16 tanh:
//   sigmoid(x) = 0.5 * (tanh(0.5 * x) + 1)
// Uses bf16 hardware path — same as 's vec_sigmoid_bf16 — because
// AIE2P does not have a hardware INT8 transcendental. Promote int32
// → fp32 → bf16 once at activation time; saturating cast back to int8
// only on output writeback.
static inline aie::vector<bfloat16, VEC>
vec_sigmoid_bf16(aie::vector<bfloat16, VEC> x) {
  aie::vector<bfloat16, VEC> half = aie::broadcast<bfloat16, VEC>(0.5f);
  aie::vector<bfloat16, VEC> one = aie::broadcast<bfloat16, VEC>(1.0f);
  auto half_x = aie::mul(x, half);
  auto t = aie::tanh<bfloat16>(half_x.to_vector<float>());
  auto t_plus_1 = aie::add(t, one);
  return aie::mul(t_plus_1, half).to_vector<bfloat16>();
}

// Vectorised hardware tanh (bf16 path).
static inline aie::vector<bfloat16, VEC>
vec_tanh_bf16(aie::vector<bfloat16, VEC> x) {
  aie::accum<accfloat, VEC> a;
  a.from_vector(x);
  return aie::tanh<bfloat16>(a.to_vector<float>());
}

// Helper: requantize an FP32 lane to INT8 with saturation.
// Used for the h-path matmul input (h_state is FP32; the matmul
// expects INT8 lanes) AND for the y_t output writeback.
//
// AIE2P has no native FP32 vector multiply (AM020 §Appendix A);
// FP32×FP32 lowers to bf16 mantissa-narrowing emulation. The pattern
// here mirrors 's bf16_acc.cc:122-126 path: cast vector<float>
// to bf16 via accum round-trip, multiply two bf16 vectors with a zero
// accumulator MAC, then narrow accumulator -> int8 (saturating).
static inline aie::vector<int8, VEC>
fp32_to_int8_sat(aie::vector<float, VEC> x, float inv_scale) {
  // FP32 -> bf16 lane narrowing via accum cast.
  aie::accum<accfloat, VEC> ax;
  ax.from_vector(x);
  aie::vector<bfloat16, VEC> xbf = ax.to_vector<bfloat16>();
  // Build the inv_scale broadcast as bf16 so the multiply is bf16×bf16.
  aie::vector<bfloat16, VEC> sf_bf =
      aie::broadcast<bfloat16, VEC>((bfloat16)inv_scale);
  // Multiply: bf16 × bf16 with accumulator zero-init.
  aie::accum<accfloat, VEC> a;
  a.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
  a = aie::mac(a, xbf, sf_bf);
  // Narrow accumulator -> int8 with saturation (aie::set_saturation
  // mode is configured at kernel entry).
  return a.to_vector<int8>();
}

extern "C" {

// Wire contract:
//   x_t          : INT8[INPUT_DIM]              (per-tensor quantized)
//   chunk_blk    : byte stream — first chunk per (t==0) carries the
//                  scale + bias prefix (FLOAT_PREFIX_ALIGNED floats),
//                  every chunk carries WEIGHT_HALF_BYTES INT8 weights
//                  AFTER the prefix region (subsequent calls re-stream
//                  the same prefix; kernel ignores after the first
//                  cache load to amortize the bias prefix cost across
//                  all 5344 chunks).
//   y_t          : INT8[HIDDEN]                  (output INT8 wire)
//   g            : gate index 0..3
//   t            : timestep 0..L-1
//   chunk_idx    : 0..3 = (W_ih_h0, W_ih_h1, W_hh_h0, W_hh_h1)
//
// All per-channel scale handling lives in the host packing — see the
// runner's expand_wb. Kernel only consumes the combined per_gate
// scale floats from the prefix.
void dorado_fast_lstm_cell_int8(int8 *restrict x_t,
                                int8 *restrict chunk_blk,
                                int8 *restrict y_t,
                                int32_t g,
                                int32_t t,
                                int32_t chunk_idx) {
  event0();

  // Set saturating arithmetic for all narrowing casts (INT32 -> INT8
  // and FP32 -> INT8). Mirrors AIE2P aie_kernels/conv2dk1_i8.cc:101
  // pattern.
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::symmetric_inf);

  // Reset h, c, and cache the prefix on (t==0, g==0, chunk==0).
  // The prefix layout is documented above — interpret chunk_blk as a
  // float pointer for the prefix region, then byte pointer for the
  // weight slab.
  if (t == 0 && g == 0 && chunk_idx == 0) {
    // Zero-initialise persistent FP32 state.
    aie::vector<float, VEC> zf = aie::broadcast<float, VEC>(0.0f);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      aie::store_v(h_state + v * VEC, zf);
      aie::store_v(c_state + v * VEC, zf);
    }
    // Cache scale prefix + bias prefix.
    float *fp = (float *)chunk_blk;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < N_GATES; ++i) {
      per_gate_scale_x[i] = fp[i];
      per_gate_scale_h[i] = fp[N_GATES + i];
    }
    h_scale = fp[2 * N_GATES];
    y_scale = fp[2 * N_GATES + 1];
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int i = 0; i < BIAS_LEN_K; ++i) {
      bias_cache[i] = fp[SCALE_PREFIX + i];
    }
  }

  // First chunk of a (t, g) pair: initialize gate accumulator to ZERO
  // (the bias is applied after the int32 → fp32 dequantization at
  // gate-finalization time, since biases are FP32 and gate_acc is INT32
  // — adding FP32 biases to an INT32 accumulator at every chunk is
  // possible but adds dequant work; cleaner to dequant once at the end
  // of the 4-chunk gate cycle).
  if (chunk_idx == 0) {
    aie::vector<int32_t, VEC> z32 = aie::broadcast<int32_t, VEC>(0);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      aie::store_v(gate_acc[g] + v * VEC, z32);
    }
  }

  // Skip the prefix bytes; the int8 weight slab starts at byte
  // BIAS_PREFIX_BYTES into the chunk.
  int8 *restrict W_half = chunk_blk + BIAS_PREFIX_BYTES;

  // Pick the source vector. For chunks 0/1 (W_ih @ x_t halves) source
  // is the int8 input directly. For chunks 2/3 (W_hh @ h halves) the
  // source is **h_state which is FP32**; we requantize lane-wise to
  // INT8 using the cached h_scale (per-tensor activation scale).
  bool h_path = (chunk_idx == 2) || (chunk_idx == 3);
  int8 *src_x = nullptr;
  const float *src_h = nullptr;
  if (chunk_idx == 0)      src_x = x_t;
  else if (chunk_idx == 1) src_x = x_t + HALF_IN;
  else if (chunk_idx == 2) src_h = h_state;
  else                     src_h = h_state + HALF_IN;

  // For each output channel, accumulate W[oc, :] · src across HALF_IN
  // = 48 INT8 lanes via int8 vector mac (INT32 accumulator).
  // gate_acc stays in INT32 storage; the chunk MAC does
  //   gate_acc[oc] = (INT32) gate_acc[oc] + sum(int8 * int8 -> int32).
  //
  // HALF_IN = 48 = 32 + 16. We process the first 32 lanes via INT8
  // vector mac and the trailing 16 as a half-lane fallback (INT8
  // vector unit on AIE2P is 32-wide).
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < HIDDEN; ++oc) {
    aie::accum<acc32, VEC> acc;
    acc.from_vector(aie::broadcast<int32_t, VEC>(0));

    int8 *restrict w_row = W_half + oc * HALF_IN;

    if (h_path) {
      // h-path: requantize src_h to int8 once per (chunk, oc) — but
      // since the requantized lane is shared across all oc's reading
      // from the same src_h offset, we load src_h as fp32 inside the
      // inner loop; the AIE2P codegen factors common subexpressions.
      // For a clean lowering, requantize each VEC slice up front:
      // process HALF_IN/VEC = 3 chunks of 16 lanes each (matching the
      // bf16 path's lane width to keep the lowering symmetrical).
      AIE_PREPARE_FOR_PIPELINING
      for (int j = 0; j < HALF_IN / VEC; ++j) {
        // Load FP32 h slice.
        aie::vector<float, VEC> hf = aie::load_v<VEC>(src_h + j * VEC);
        // Requantize to INT8 with saturation.
        // inv_h_scale = 1 / h_scale; apply via fp32_to_int8_sat helper.
        // To save a divide per inner step, host pre-supplies h_scale
        // as the multiplicative inverse already (see runner.cpp).
        aie::vector<int8, VEC> s_v = fp32_to_int8_sat(hf, h_scale);
        // Load weight slice (int8) and MAC.
        aie::vector<int8, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
        // INT8 * INT8 → INT32 accumulator.
        acc = aie::mac(acc, w_v, s_v);
      }
    } else {
      AIE_PREPARE_FOR_PIPELINING
      for (int j = 0; j < HALF_IN / VEC; ++j) {
        aie::vector<int8, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
        aie::vector<int8, VEC> s_v = aie::load_v<VEC>(src_x + j * VEC);
        acc = aie::mac(acc, w_v, s_v);
      }
    }

    // Reduce the lane INT32 accumulator to a scalar.
    auto out_vec = acc.to_vector<int32_t>();
    int32_t sum = 0;
    for (int k = 0; k < VEC; ++k) sum += out_vec.get(k);

    // Add to the running gate accumulator (INT32 storage).
    int32_t prev = gate_acc[g][oc];
    gate_acc[g][oc] = prev + sum;
  }

  // After the last chunk of the last gate (g==3, chunk==3), apply
  // de-quantization, biases, nonlinearities, update c and h, emit y_t.
  //
  // Per-gate scale chain: for gate g, the dequant scale is
  //   s_x_path = per_gate_scale_x[g]   (applied to chunks 0+1 partial)
  //   s_h_path = per_gate_scale_h[g]   (applied to chunks 2+3 partial)
  //
  // Since we don't track per-half partial sums separately (we
  // accumulated into one INT32 gate_acc), we fold the host-side scale
  // via the constraint that the host sets s_x = s_h = unified_scale[g]
  // BEFORE this chunk gate_acc reduction. The runner enforces this by
  // requantizing W_hh weights against the h_scale's grid so that
  // s_x_path == s_h_path == per_gate_scale_x[g]. (Documented in
  // runner.cpp expand_wb.)
  if (g == 3 && chunk_idx == 3) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      // Load INT32 gate sums.
      aie::vector<int32_t, VEC> z_i_i = aie::load_v<VEC>(gate_acc[0] + v * VEC);
      aie::vector<int32_t, VEC> z_f_i = aie::load_v<VEC>(gate_acc[1] + v * VEC);
      aie::vector<int32_t, VEC> z_g_i = aie::load_v<VEC>(gate_acc[2] + v * VEC);
      aie::vector<int32_t, VEC> z_o_i = aie::load_v<VEC>(gate_acc[3] + v * VEC);

      // Convert INT32 -> FP32 via accum<accfloat>::from_vector(int_vec).
      // The framework handles the int->float lane conversion; per
      // aie_api/accum.hpp from_vector accepts vector<T> for any T.
      // INT32 fits in FP32 mantissa for typical Dorado gate sums
      // (|z| < 1e5 << 2^23 = 8.4e6).
      aie::accum<accfloat, VEC> ai_acc;
      aie::accum<accfloat, VEC> af_acc;
      aie::accum<accfloat, VEC> ag_acc;
      aie::accum<accfloat, VEC> ao_acc;
      ai_acc.from_vector(z_i_i);
      af_acc.from_vector(z_f_i);
      ag_acc.from_vector(z_g_i);
      ao_acc.from_vector(z_o_i);

      // Per-gate dequant + bias add fused into one bf16-emulated MAC:
      //   z_g_f = (FP32→bf16 narrow of int32 sum) * (bf16 broadcast scale) + bias
      // Pattern mirrors 's bf16_acc.cc: AIE2P has no native FP32
      // multiply; the FP32 chain emulates as bf16×bf16 with FP32
      // accumulator. The bias add lands as a from_vector seeding of
      // the accumulator (not a separate mul), saving one round-trip.
      aie::vector<bfloat16, VEC> z_i_bf = ai_acc.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_f_bf = af_acc.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_g_bf = ag_acc.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_o_bf = ao_acc.to_vector<bfloat16>();

      aie::vector<bfloat16, VEC> s_i =
          aie::broadcast<bfloat16, VEC>((bfloat16)per_gate_scale_x[0]);
      aie::vector<bfloat16, VEC> s_f =
          aie::broadcast<bfloat16, VEC>((bfloat16)per_gate_scale_x[1]);
      aie::vector<bfloat16, VEC> s_g =
          aie::broadcast<bfloat16, VEC>((bfloat16)per_gate_scale_x[2]);
      aie::vector<bfloat16, VEC> s_o =
          aie::broadcast<bfloat16, VEC>((bfloat16)per_gate_scale_x[3]);

      // Seed each gate's FP32 accumulator with the bias.
      // bias_cache layout: gate-major i/f/g/o pre-summed (b_ih + b_hh)
      // per gate, contiguous HIDDEN floats per gate.
      aie::vector<float, VEC> b_i = aie::load_v<VEC>(bias_cache + 0 * HIDDEN + v * VEC);
      aie::vector<float, VEC> b_f = aie::load_v<VEC>(bias_cache + 1 * HIDDEN + v * VEC);
      aie::vector<float, VEC> b_g = aie::load_v<VEC>(bias_cache + 2 * HIDDEN + v * VEC);
      aie::vector<float, VEC> b_o = aie::load_v<VEC>(bias_cache + 3 * HIDDEN + v * VEC);

      aie::accum<accfloat, VEC> zi_acc, zf_acc, zg_acc, zo_acc;
      zi_acc.from_vector(b_i);
      zf_acc.from_vector(b_f);
      zg_acc.from_vector(b_g);
      zo_acc.from_vector(b_o);
      // FP32 acc + bf16(z_*) * bf16(scale) -> FP32 accumulator (z_g w/ scale + bias).
      zi_acc = aie::mac(zi_acc, z_i_bf, s_i);
      zf_acc = aie::mac(zf_acc, z_f_bf, s_f);
      zg_acc = aie::mac(zg_acc, z_g_bf, s_g);
      zo_acc = aie::mac(zo_acc, z_o_bf, s_o);

      // Narrow to bf16 for the activations (sigmoid and tanh).
      aie::vector<bfloat16, VEC> z_i = zi_acc.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_f = zf_acc.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_g = zg_acc.to_vector<bfloat16>();
      aie::vector<bfloat16, VEC> z_o = zo_acc.to_vector<bfloat16>();

      auto i_g = vec_sigmoid_bf16(z_i);
      auto f_g = vec_sigmoid_bf16(z_f);
      auto g_g = vec_tanh_bf16(z_g);
      auto o_g = vec_sigmoid_bf16(z_o);

      // c_new = f * c_old + i * g, FP32 accumulator path.
      // Load c_old as FP32 (persistent state) and narrow to bf16 only
      // for the multiplier input.
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

      // Sum FP32 accumulators and store as FP32 (preserve recurrent
      // precision per / ).
      aie::vector<float, VEC> fc_f = fc_acc.to_vector<float>();
      aie::vector<float, VEC> ig_f = ig_acc.to_vector<float>();
      aie::vector<float, VEC> c_new_f = aie::add(fc_f, ig_f);
      aie::store_v(c_state + v * VEC, c_new_f);

      // h_new = o * tanh(c_new). Narrow c_new to bf16 only for the
      // tanh input. h_state writeback is FP32.
      aie::accum<accfloat, VEC> c_new_acc;
      c_new_acc.from_vector(c_new_f);
      aie::vector<bfloat16, VEC> c_new_bf = c_new_acc.to_vector<bfloat16>();

      auto tanh_c = vec_tanh_bf16(c_new_bf);
      aie::accum<accfloat, VEC> h_acc;
      h_acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
      h_acc = aie::mac(h_acc, o_g, tanh_c);

      // Persist h as FP32 (recurrent-state precision).
      aie::vector<float, VEC> h_new_f = h_acc.to_vector<float>();
      aie::store_v(h_state + v * VEC, h_new_f);

      // Quantize h_new to INT8 for the output wire (saturate to
      // [-128, 127]). y_scale is the inverse — host supplies
      // 1/output_activation_scale so the kernel does a single
      // multiply.
      aie::vector<int8, VEC> y_q = fp32_to_int8_sat(h_new_f, y_scale);
      aie::store_v(y_t + v * VEC, y_q);
    }
  }

  event1();
}

} // extern "C"
