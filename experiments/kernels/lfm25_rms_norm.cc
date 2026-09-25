// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Scalar correctness implementation for Phoenix AIE2. Vectorize after the
// model block is numerically validated.

#include <aie_api/aie.hpp>
#include <stdint.h>

static float invsqrt_approx(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits = 0x5f3759dfu - (bits >> 1);
  float result;
  __builtin_memcpy(&result, &bits, sizeof(result));
  const float half_value = 0.5f * value;
  for (int i = 0; i < 3; ++i)
    result *= 1.5f - half_value * result * result;
  return result;
}

extern "C" void lfm25_rms_norm(const bfloat16 *input,
                                const bfloat16 *gamma,
                                bfloat16 *output, int32_t cols) {
  const auto saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  float sum_sq = 0.0f;
  for (int32_t i = 0; i < cols; ++i) {
    const float x = static_cast<float>(input[i]);
    sum_sq += x * x;
  }
  const float inv_rms = invsqrt_approx(sum_sq / cols + 1.0e-5f);
  for (int32_t i = 0; i < cols; ++i) {
    // Transformers converts the normalized activation back to BF16 before
    // multiplying the learned BF16 weight.
    const bfloat16 normalized =
        static_cast<bfloat16>(static_cast<float>(input[i]) * inv_rms);
    output[i] = static_cast<bfloat16>(static_cast<float>(normalized) *
                                      static_cast<float>(gamma[i]));
  }
  aie::set_rounding(saved_rounding);
}
