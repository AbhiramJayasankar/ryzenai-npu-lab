// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Cast GEMV accumulations to the BF16 activations used by LFM2.5.

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_f32_to_bf16(const float *input, bfloat16 *output,
                                    int32_t count) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int32_t i = 0; i < count; ++i)
    output[i] = static_cast<bfloat16>(input[i]);
  aie::set_rounding(saved_rounding);
}
