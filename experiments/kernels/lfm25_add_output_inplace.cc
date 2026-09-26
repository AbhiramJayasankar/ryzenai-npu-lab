// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>

extern "C" void lfm25_add_output_inplace(bfloat16 *output,
                                            const bfloat16 *residual) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 1024; ++i)
    output[i] = static_cast<bfloat16>(static_cast<float>(output[i]) +
                                      static_cast<float>(residual[i]));
  aie::set_rounding(saved_rounding);
}
