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

// linear_projection_fused.cc — DESIGN-fusion.md stage 3.
//
// Per-timestep fused call: one kernel invocation per timestep that
// computes the full 256-output GEMM by looping over the 4 OC groups
// internally. Weight is bf16 (full 256 x 96 slab acquired once on the
// IRON side and passed in by pointer here every call); inputs and
// outputs are bf16 (precision-neutral on AIE2P per AM020 Appendix A —
// the multiplier already narrows fp32 inputs to bf16 in the per-group
// fp32 kernel; doing the narrowing host-side instead changes nothing
// numerically while halving the weight footprint).
//
// Math: y[oc] = clamp(sum_j W[oc, j] * x[j], -5, 5) for oc in 0..256.
//
// Loop nest mirrors the "weight-stationary" cleanup that makes scalar-
// FP32 reductions safe on AIE2P (G-T5.1-002 lesson; see
// conv_stem_layers_2_3/conv1d_layer2.cc): the FP32 accumulator lives
// in tile memory (an `acc` declared inside the oc body, materialised
// at vector lane width), not the BM register file. The bf16 multiplier
// inputs (weights + input) feed an aie::accum<accfloat, VEC>, and the
// final scalar reduction is one cross-lane sum per oc.

#define NOCPP

#include <stdint.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int HIDDEN = 96;
constexpr int OUT_DIM = 256;
constexpr int OC_GROUP_SIZE = 64;
constexpr int N_OC_GROUPS = OUT_DIM / OC_GROUP_SIZE; // 4
constexpr int VEC = 16;                              // bf16 vector lane width
constexpr int HIDDEN_VECS = HIDDEN / VEC;            // 6

extern "C" {

// Stage 3 entry point: one call per timestep, weight is the full
// (OUT_DIM=256, HIDDEN=96) bf16 slab in row-major order.
//
// W layout: W[oc, j] = W_full[oc * HIDDEN + j] for oc in 0..OUT_DIM,
//                                                    j  in 0..HIDDEN.
// This is the natural torch.nn.Linear weight layout (no transpose).
void dorado_fast_linear_projection_fused_perts(bfloat16 *restrict x_t,
                                               bfloat16 *restrict W_full,
                                               bfloat16 *restrict y_t) {
  event0();

  // PER-TIMESTEP loop sentinel — the regression-guard test searches
  // for this string to ensure the fused shape stays in place.
  // FUSED_PERTS_INNER_OC_LOOP

  // Walk all 4 OC groups internally (weight-stationary): outer oc
  // covers the full 256-output range, with the OC group as the
  // syntactic boundary the test sentinel pins. Loop count 4 is below
  // pipelining minimum, so hint nothing on the outer loop.
  for (int g = 0; g < N_OC_GROUPS; ++g) {
    bfloat16 *restrict W_group = W_full + g * OC_GROUP_SIZE * HIDDEN;
    bfloat16 *restrict y_group = y_t + g * OC_GROUP_SIZE;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int oc = 0; oc < OC_GROUP_SIZE; ++oc) {
      bfloat16 *restrict w_row = W_group + oc * HIDDEN;

      // FP32 accumulator at VEC lane width; bf16 multiplier inputs.
      // Following the LSTM bf16-mixed-fp32 precision contract: the
      // accfloat accumulator preserves the 23-mantissa-bit
      // accumulation regardless of the bf16 input width.
      aie::accum<accfloat, VEC> acc;
      acc.from_vector(aie::broadcast<bfloat16, VEC>(0.0f));
      for (int j = 0; j < HIDDEN_VECS; ++j) {
        aie::vector<bfloat16, VEC> w_v = aie::load_v<VEC>(w_row + j * VEC);
        aie::vector<bfloat16, VEC> x_v = aie::load_v<VEC>(x_t + j * VEC);
        acc = aie::mac(acc, w_v, x_v);
      }
      auto out_vec = acc.to_vector<float>();
      float s = 0.0f;
      for (int k = 0; k < VEC; ++k) s += out_vec.get(k);

      // Clamp to [-5, 5] per architecture.md §3 (kept inside the
      // oc loop; it's free on the FP32 scalar before bf16 narrow).
      if (s < -5.0f) s = -5.0f;
      if (s > 5.0f)  s = 5.0f;

      // Narrow FP32 -> bf16 for the output. Hardware-free on AIE2P.
      aie::vector<float, VEC> sv = aie::broadcast<float, VEC>(s);
      aie::accum<accfloat, VEC> sa;
      sa.from_vector(sv);
      bfloat16 s_bf = sa.to_vector<bfloat16>().get(0);
      y_group[oc] = s_bf;
    }
  }

  event1();
}

} // extern "C"
