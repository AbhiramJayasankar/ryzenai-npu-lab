// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// LFM2.5's BF16 SiLU(w1) * w3, with BF16 rounding at both model boundaries.

#include <aie_api/aie.hpp>
#include <stdint.h>

static float exp_negative(float x) {
  if (x < -87.0f)
    return 0.0f;
  const int n = static_cast<int>(-x * 1.4426950408889634f);
  const float r = x + n * 0.6931471805599453f;
  const float polynomial = 1.0f + r * (1.0f + r * (0.5f + r *
      (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f +
      r * (1.0f / 720.0f + r / 5040.0f))))));
  const uint32_t bits = static_cast<uint32_t>(127 - n) << 23;
  float power_two;
  __builtin_memcpy(&power_two, &bits, sizeof(power_two));
  return polynomial * power_two;
}

extern "C" void lfm25_silu_gate(const bfloat16 *w1,
                                 const bfloat16 *w3, bfloat16 *output,
                                 int32_t count) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int32_t i = 0; i < count; ++i) {
    const float x = static_cast<float>(w1[i]);
    // Stable on either side of zero and accurate across the observed
    // late-layer range [-3.33, 3.73], unlike the local Maclaurin shortcut.
    const float exponential = exp_negative(x < 0.0f ? x : -x);
    const float sigmoid = x < 0.0f
                              ? exponential / (1.0f + exponential)
                              : 1.0f / (1.0f + exponential);
    const bfloat16 silu = static_cast<bfloat16>(x * sigmoid);
    output[i] = static_cast<bfloat16>(static_cast<float>(silu) *
                                      static_cast<float>(w3[i]));
  }
  aie::set_rounding(saved_rounding);
}
