// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Assemble the recurrent convolution's two inputs in NPU memory.

#include <aie_api/aie.hpp>

extern "C" void lfm25_pack_conv(const bfloat16 *projection,
                                 const bfloat16 *weight, bfloat16 *packed) {
  for (int i = 0; i < 3072; ++i) {
    packed[i] = projection[i];
    packed[i + 3072] = weight[i];
  }
}
