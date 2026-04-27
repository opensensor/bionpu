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
constexpr int OC_GROUP_SIZE = 64;

extern "C" {

void dorado_fast_linear_projection_fp32(float *restrict x_t,
                                        float *restrict W_group,
                                        float *restrict y_group) {
  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int oc = 0; oc < OC_GROUP_SIZE; ++oc) {
    float s = 0.0f;
    float *restrict w_row = W_group + oc * HIDDEN;
    for (int j = 0; j < HIDDEN; ++j) {
      s += w_row[j] * x_t[j];
    }
    // Clamp to [-5, 5] per architecture.md §3.
    if (s < -5.0f) s = -5.0f;
    if (s > 5.0f)  s = 5.0f;
    y_group[oc] = s;
  }

  event1();
}

} // extern "C"
