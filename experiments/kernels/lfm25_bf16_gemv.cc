// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// BF16 64x64 weight tile times BF16 64-vector, accumulated into FP32 outputs.

#include <aie_api/aie.hpp>

extern "C" void lfm25_bf16_gemv(const bfloat16 *weights,
                                const bfloat16 *activation, float *output) {
  for (int row = 0; row < 64; ++row) {
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    for (int col = 0; col < 64; col += 16) {
      const aie::vector<bfloat16, 16> w =
          aie::load_v<16>(weights + row * 64 + col);
      const aie::vector<bfloat16, 16> x = aie::load_v<16>(activation + col);
      acc = aie::mac(acc, w, x);
    }
    output[row] += aie::reduce_add(acc.to_vector<float>());
  }
}
