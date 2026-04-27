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

constexpr int CONV_K = 19;
constexpr int CONV_IN_CH = 16;
constexpr int OC_GROUP_SIZE = 16;
constexpr int W_PER_OC = CONV_IN_CH * CONV_K; // 304 weights per output channel
constexpr int BIAS_OFFSET = OC_GROUP_SIZE * W_PER_OC; // 4864

extern "C" {

void dorado_fast_conv_stem_layer3_fp32(float *restrict signal_slice,
                                       float *restrict wb_slice,
                                       float *restrict output_slice) {
  event0();

  float *restrict W = wb_slice;
  float *restrict B = wb_slice + BIAS_OFFSET;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < OC_GROUP_SIZE; ++oc) {
    float s = B[oc];
    // Accumulate over 16 input channels × 19 kernel positions.
    for (int ic = 0; ic < CONV_IN_CH; ++ic) {
      // Per-channel input window (length 19) within the slice.
      float *restrict sig_row = signal_slice + ic * CONV_K;
      // Weight row for (oc, ic, *).
      float *restrict w_row = W + (oc * CONV_IN_CH + ic) * CONV_K;
      for (int k = 0; k < CONV_K; ++k) {
        s += sig_row[k] * w_row[k];
      }
    }
    output_slice[oc] = s;
  }

  event1();
}

} // extern "C"
