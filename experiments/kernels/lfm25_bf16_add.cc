// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_bf16_add(const bfloat16 *left,
                                 const bfloat16 *right, bfloat16 *output,
                                 int32_t count) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int32_t i = 0; i < count; ++i)
    output[i] = static_cast<bfloat16>(static_cast<float>(left[i]) +
                                      static_cast<float>(right[i]));
  aie::set_rounding(saved_rounding);
}
