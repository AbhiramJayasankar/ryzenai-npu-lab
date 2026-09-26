// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_cast_store_1024(const float *input, bfloat16 *output,
                                        int32_t offset) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int32_t i = 0; i < 64; ++i)
    output[offset + i] = static_cast<bfloat16>(input[i]);
  aie::set_rounding(saved_rounding);
}
