// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_bf16_gemv_offset_3072(
    const bfloat16 *weights, const bfloat16 *activation, float *output,
    int32_t input_offset) {
  const bfloat16 *x = activation + input_offset;
  for (int row = 0; row < 64; ++row) {
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    for (int col = 0; col < 64; col += 16) {
      const aie::vector<bfloat16, 16> w =
          aie::load_v<16>(weights + row * 64 + col);
      const aie::vector<bfloat16, 16> v = aie::load_v<16>(x + col);
      acc = aie::mac(acc, w, v);
    }
    output[row] += aie::reduce_add(acc.to_vector<float>());
  }
}
