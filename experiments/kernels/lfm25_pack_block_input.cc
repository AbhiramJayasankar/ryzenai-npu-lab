// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Pack the next block input without reading recurrent state back to the CPU.

#include <aie_api/aie.hpp>

extern "C" void lfm25_pack_block_input(const bfloat16 *hidden,
                                         const bfloat16 *state_and_weight,
                                         bfloat16 *packed) {
  for (int i = 0; i < 1024; ++i)
    packed[i] = hidden[i];
  for (int i = 1024; i < 6144; ++i)
    packed[i] = static_cast<bfloat16>(0.0f);
  for (int i = 0; i < 6144; ++i)
    packed[6144 + i] = state_and_weight[i];
}
