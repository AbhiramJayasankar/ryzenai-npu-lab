// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// One-token recurrent gate with persistent (state, weight) in NPU memory.

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_conv_gate_packed_state(
    const bfloat16 *projection, const bfloat16 *state_and_weight,
    bfloat16 *gated_output, bfloat16 *next_state_and_weight) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr int32_t width = 1024;
  const bfloat16 *previous_state = state_and_weight;
  const bfloat16 *depthwise_weight = state_and_weight + 3 * width;
  bfloat16 *next_state = next_state_and_weight;
  bfloat16 *next_weight = next_state_and_weight + 3 * width;
  for (int32_t i = 0; i < width; ++i) {
    const bfloat16 B = projection[i];
    const bfloat16 C = projection[width + i];
    const bfloat16 x = projection[2 * width + i];
    const bfloat16 Bx =
        static_cast<bfloat16>(static_cast<float>(B) * static_cast<float>(x));

    const bfloat16 previous1 = previous_state[3 * i + 1];
    const bfloat16 previous2 = previous_state[3 * i + 2];
    next_state[3 * i] = previous1;
    next_state[3 * i + 1] = previous2;
    next_state[3 * i + 2] = Bx;

    const bfloat16 product0 = static_cast<bfloat16>(
        static_cast<float>(previous1) * static_cast<float>(depthwise_weight[3 * i]));
    const bfloat16 product1 = static_cast<bfloat16>(
        static_cast<float>(previous2) * static_cast<float>(depthwise_weight[3 * i + 1]));
    const bfloat16 product2 = static_cast<bfloat16>(
        static_cast<float>(Bx) * static_cast<float>(depthwise_weight[3 * i + 2]));
    const bfloat16 conv = static_cast<bfloat16>(
        static_cast<float>(product0) + static_cast<float>(product1) +
        static_cast<float>(product2));
    gated_output[i] =
        static_cast<bfloat16>(static_cast<float>(C) * static_cast<float>(conv));
  }
  for (int32_t i = 0; i < 3 * width; ++i)
    next_weight[i] = depthwise_weight[i];
  aie::set_rounding(saved_rounding);
}
