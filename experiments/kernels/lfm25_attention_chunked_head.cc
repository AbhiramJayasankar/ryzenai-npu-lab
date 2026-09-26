// Stream 64-position KV blocks through one Phoenix compute tile.
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int width = 64;
constexpr int capacity = 64;
constexpr int head_elements = 2 + 2 * capacity * width;
constexpr int accumulator_stride = 2 + width;

float exp_negative(float x) {
  if (x < -87.0f)
    return 0.0f;
  const int n = static_cast<int>(-x * 1.4426950408889634f);
  const float r = x + n * 0.6931471805599453f;
  const float polynomial = 1.0f + r * (1.0f + r * (0.5f + r *
      (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f +
      r / 720.0f + r * r / 5040.0f)))));
  const uint32_t bits = static_cast<uint32_t>(127 - n) << 23;
  float power_two;
  __builtin_memcpy(&power_two, &bits, sizeof(power_two));
  return polynomial * power_two;
}
}

extern "C" void lfm25_attention_chunked_head(
    const bfloat16 *qkv, const bfloat16 *past, bfloat16 *next,
    bfloat16 *context, float *accumulator, int32_t kv_head,
    int32_t block_index, int32_t block_count) {
  const auto saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const int past_length = static_cast<int>(static_cast<float>(past[0]));
  const bool append = block_index == block_count - 1;
  const bfloat16 *past_keys = past + 2;
  const bfloat16 *past_values = past + 2 + capacity * width;
  const bfloat16 *new_key = qkv + 1024 + kv_head * width;
  const bfloat16 *new_value = qkv + 1536 + kv_head * width;

  for (int i = 0; i < head_elements; ++i)
    next[i] = past[i];
  if (append) {
    next[0] = static_cast<bfloat16>(static_cast<float>(past_length + 1));
    for (int dim = 0; dim < width; ++dim) {
      next[2 + past_length * width + dim] = new_key[dim];
      next[2 + capacity * width + past_length * width + dim] = new_value[dim];
    }
  }

  for (int group_head = 0; group_head < 2; ++group_head) {
    const int query_head = kv_head * 2 + group_head;
    const bfloat16 *query = qkv + query_head * width;
    float *state = accumulator + group_head * accumulator_stride;
    if (block_index == 0) {
      state[0] = -1.0e30f;
      state[1] = 0.0f;
      for (int dim = 0; dim < width; ++dim)
        state[2 + dim] = 0.0f;
    }
    const int length = past_length + (append ? 1 : 0);
    for (int token = 0; token < length; ++token) {
      const bfloat16 *key = token < past_length
          ? past_keys + token * width : new_key;
      const bfloat16 *value = token < past_length
          ? past_values + token * width : new_value;
      float dot = 0.0f;
      for (int dim = 0; dim < width; ++dim)
        dot += static_cast<float>(query[dim]) * static_cast<float>(key[dim]);
      const bfloat16 rounded_dot = static_cast<bfloat16>(dot);
      const bfloat16 scaled =
          static_cast<bfloat16>(static_cast<float>(rounded_dot) * 0.125f);
      const float score = static_cast<float>(scaled);
      if (score > state[0]) {
        const float scale = exp_negative(state[0] - score);
        state[1] *= scale;
        for (int dim = 0; dim < width; ++dim)
          state[2 + dim] *= scale;
        state[0] = score;
      }
      const float probability = exp_negative(score - state[0]);
      state[1] += probability;
      for (int dim = 0; dim < width; ++dim)
        state[2 + dim] += probability * static_cast<float>(value[dim]);
    }
    if (append) {
      for (int dim = 0; dim < width; ++dim)
        context[query_head * width + dim] =
            static_cast<bfloat16>(state[2 + dim] / state[1]);
    }
  }
  aie::set_rounding(saved_rounding);
}
