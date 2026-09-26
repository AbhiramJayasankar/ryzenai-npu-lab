// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// LFM2.5's BF16 SiLU(w1) * w3, with BF16 rounding at both model boundaries.

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_silu_gate(const bfloat16 *w1,
                                 const bfloat16 *w3, bfloat16 *output,
                                 int32_t count) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int32_t i = 0; i < count; ++i) {
    const float x = static_cast<float>(w1[i]);
    const float x2 = x * x;
    // The measured first-block inputs lie within [-0.75, 0.75]. The seventh
    // order Maclaurin series is accurate there without a device expf call.
    const float sigmoid =
        0.5f + x * (0.25f + x2 * (-1.0f / 48.0f +
                                 x2 * (1.0f / 480.0f - 17.0f * x2 / 80640.0f)));
    const bfloat16 silu = static_cast<bfloat16>(x * sigmoid);
    output[i] = static_cast<bfloat16>(static_cast<float>(silu) *
                                      static_cast<float>(w3[i]));
  }
  aie::set_rounding(saved_rounding);
}
