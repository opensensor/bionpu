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

// kernel/conv geometry. These are compiled-in for the v1 path; 
// will lift the constants once we share this kernel across the 3-conv
// stem.
constexpr int CONV_K = 5;
constexpr int CONV_IN_CH = 1;
constexpr int CONV_OUT_CH = 16;
// chunk sizes are the *valid* output length for one core invocation.
// The host pads the input slice with CONV_K-1 zeros split half each
// side; the kernel walks the padded buffer with a fixed kernel window.
//
// We expose `chunk_out_len` as a runtime int32 so the IRON lowering can
// retune without a kernel rebuild.

extern "C" {

// One conv1d slice: walk `signal_padded` with a length-5 kernel against
// 16 output channels. `signal_padded` length must be `chunk_out_len + 4`.
// `wb` length = 16*5 + 16 = 96 floats.
//
// Output is laid out as 16 contiguous rows of `chunk_out_len`
// floats each (NCL with N=1).
void dorado_fast_conv_stem_layer1_fp32(float *restrict signal_padded,
                                       float *restrict wb,
                                       float *restrict output,
                                       int32_t chunk_out_len) {
  event0();

  // Weights live at the start of `wb`; bias is at offset 80 (16*5).
  float *restrict W = wb;
  float *restrict B = wb + (CONV_OUT_CH * CONV_K);

  // The scalar inner loop is intentionally simple for v1. AIE2P has bf16
  // vector intrinsics that would materially accelerate this (
  // research target); the scalar path is correct-first per umbrella PRD
  // §4.1's "v1-thin" mandate.
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < CONV_OUT_CH; ++oc) {
    float bias = B[oc];
    float w0 = W[oc * CONV_K + 0];
    float w1 = W[oc * CONV_K + 1];
    float w2 = W[oc * CONV_K + 2];
    float w3 = W[oc * CONV_K + 3];
    float w4 = W[oc * CONV_K + 4];

    float *restrict out_row = output + oc * chunk_out_len;

    for (int t = 0; t < chunk_out_len; ++t) {
      float s = bias;
      s += signal_padded[t + 0] * w0;
      s += signal_padded[t + 1] * w1;
      s += signal_padded[t + 2] * w2;
      s += signal_padded[t + 3] * w3;
      s += signal_padded[t + 4] * w4;
      out_row[t] = s;
    }
  }

  event1();
}

} // extern "C"
