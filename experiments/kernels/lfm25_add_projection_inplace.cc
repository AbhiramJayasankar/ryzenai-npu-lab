// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>

extern "C" void lfm25_add_projection_inplace(bfloat16 *residual,
                                                const bfloat16 *projection) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 1024; ++i)
    residual[i] = static_cast<bfloat16>(
        static_cast<float>(residual[i]) + static_cast<float>(projection[i]));
  aie::set_rounding(saved_rounding);
}
