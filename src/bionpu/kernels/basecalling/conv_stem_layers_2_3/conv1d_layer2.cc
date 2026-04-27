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

constexpr int CONV_K = 5;
constexpr int CONV_IN_CH = 16;
constexpr int CONV_OUT_CH = 16;
constexpr int W_PER_OC = CONV_IN_CH * CONV_K; // 80 weights per output channel
constexpr int BIAS_OFFSET = CONV_OUT_CH * W_PER_OC; // 1280

extern "C" {

// One conv1d slice: walk multi-channel `signal_padded` with a length-5
// kernel against 16 output channels. `signal_padded` length must be
// `CONV_IN_CH * (chunk_out_len + 4)` = 16 * (T_chunk + 4) floats.
// `wb` length = 16*16*5 + 16 = 1296 floats.
//
// Output: 16 contiguous rows of `chunk_out_len` floats.
void dorado_fast_conv_stem_layer2_fp32(float *restrict signal_padded,
                                       float *restrict wb,
                                       float *restrict output,
                                       int32_t chunk_out_len) {
  event0();

  float *restrict W = wb;
  float *restrict B = wb + BIAS_OFFSET;

  const int chunk_in_len = chunk_out_len + 4; // padding=2 each side, kernel=5.

  // Process each output channel independently. The inner work is a
  // weight-stationary strided FMA along the time axis (see header
  // comment for why this loop order avoids the miscompile).
  for (int oc = 0; oc < CONV_OUT_CH; ++oc) {
    float bias = B[oc];
    float *restrict out_row = output + oc * chunk_out_len;

    // Step 1: broadcast-init the output row with the bias. This
    // doubles as the FP32 accumulator for the (ic, k) reductions
    // that follow.
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int t = 0; t < chunk_out_len; ++t) {
      out_row[t] = bias;
    }

    // Step 2: weight-stationary 1D conv reduction. For each (ic, k)
    // pair, broadcast the scalar weight and FMA-into-place along t.
    for (int ic = 0; ic < CONV_IN_CH; ++ic) {
      float *restrict sig_row = signal_padded + ic * chunk_in_len;
      float *restrict w_row = W + (oc * CONV_IN_CH + ic) * CONV_K;

      // Hoist the 5 kernel weights as scalars (each becomes a
      // broadcast-multiply in the vectorized inner loop).
      const float w0 = w_row[0];
      const float w1 = w_row[1];
      const float w2 = w_row[2];
      const float w3 = w_row[3];
      const float w4 = w_row[4];

      // Inner-`t` strided FMA. Each kernel position contributes one
      // add-multiply per output sample. Splitting the kernel-axis
      // reduction across 5 separate t-loops keeps each loop a clean
      // 1D strided vector-FMA (no per-iteration multi-input fan-in
      // for Peano to fold into BM-spill emulation).
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int t = 0; t < chunk_out_len; ++t) {
        out_row[t] += sig_row[t + 0] * w0;
      }
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int t = 0; t < chunk_out_len; ++t) {
        out_row[t] += sig_row[t + 1] * w1;
      }
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int t = 0; t < chunk_out_len; ++t) {
        out_row[t] += sig_row[t + 2] * w2;
      }
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int t = 0; t < chunk_out_len; ++t) {
        out_row[t] += sig_row[t + 3] * w3;
      }
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for (int t = 0; t < chunk_out_len; ++t) {
        out_row[t] += sig_row[t + 4] * w4;
      }
    }
  }

  event1();
}

} // extern "C"
