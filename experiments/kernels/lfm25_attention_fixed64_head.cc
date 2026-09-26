// One-token attention over a fixed-capacity KV cache on Phoenix AIE.
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int width = 64;
constexpr int capacity = 64;
constexpr int head_elements = 2 + 2 * capacity * width;

float exp_negative(float x) {
  if (x < -87.0f)
    return 0.0f;
  const int n = static_cast<int>(-x * 1.4426950408889634f);
  const float r = x + n * 0.6931471805599453f;
  const float polynomial = 1.0f + r * (1.0f + r * (0.5f + r *
      (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f +
      r * (1.0f / 720.0f + r / 5040.0f))))));
  const uint32_t bits = static_cast<uint32_t>(127 - n) << 23;
  float power_two;
  __builtin_memcpy(&power_two, &bits, sizeof(power_two));
  return polynomial * power_two;
}
}

extern "C" void lfm25_attention_fixed64_first_head(
    const bfloat16 *qkv, bfloat16 *next, bfloat16 *context, int32_t kv_head) {
  for (int i = 0; i < head_elements; ++i)
    next[i] = static_cast<bfloat16>(0.0f);
  next[0] = static_cast<bfloat16>(1.0f);
  const bfloat16 *key = qkv + 1024 + kv_head * width;
  const bfloat16 *value = qkv + 1536 + kv_head * width;
  for (int dim = 0; dim < width; ++dim) {
    next[2 + dim] = key[dim];
    next[2 + capacity * width + dim] = value[dim];
    context[(2 * kv_head) * width + dim] = value[dim];
    context[(2 * kv_head + 1) * width + dim] = value[dim];
  }
}

extern "C" void lfm25_attention_fixed64_head(
    const bfloat16 *qkv, const bfloat16 *past, bfloat16 *next,
    bfloat16 *context, int32_t kv_head) {
  const auto saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const int past_length = static_cast<int>(static_cast<float>(past[0]));
  const int total_length = past_length + 1;
  const bfloat16 *past_keys = past + 2;
  const bfloat16 *past_values = past + 2 + capacity * width;
  const bfloat16 *new_key = qkv + 1024 + kv_head * width;
  const bfloat16 *new_value = qkv + 1536 + kv_head * width;
  for (int i = 0; i < head_elements; ++i)
    next[i] = past[i];
  next[0] = static_cast<bfloat16>(static_cast<float>(total_length));
  for (int dim = 0; dim < width; ++dim) {
    next[2 + past_length * width + dim] = new_key[dim];
    next[2 + capacity * width + past_length * width + dim] = new_value[dim];
  }

  for (int group_head = 0; group_head < 2; ++group_head) {
    const int query_head = kv_head * 2 + group_head;
    const bfloat16 *query = qkv + query_head * width;
    float scores[capacity];
    float maximum = -1.0e30f;
    for (int token = 0; token < total_length; ++token) {
      const bfloat16 *key = token < past_length
                                ? past_keys + token * width : new_key;
      float dot = 0.0f;
      for (int dim = 0; dim < width; ++dim)
        dot += static_cast<float>(query[dim]) * static_cast<float>(key[dim]);
      const bfloat16 rounded_dot = static_cast<bfloat16>(dot);
      const bfloat16 scaled =
          static_cast<bfloat16>(static_cast<float>(rounded_dot) * 0.125f);
      scores[token] = static_cast<float>(scaled);
      if (scores[token] > maximum)
        maximum = scores[token];
    }
    float total = 0.0f;
    for (int token = 0; token < total_length; ++token) {
      scores[token] = exp_negative(scores[token] - maximum);
      total += scores[token];
    }
    bfloat16 probability[capacity];
    for (int token = 0; token < total_length; ++token)
      probability[token] = static_cast<bfloat16>(scores[token] / total);
    for (int dim = 0; dim < width; ++dim) {
      float weighted_sum = 0.0f;
      for (int token = 0; token < total_length; ++token) {
        const bfloat16 value = token < past_length
            ? past_values[token * width + dim] : new_value[dim];
        weighted_sum += static_cast<float>(probability[token]) *
                        static_cast<float>(value);
      }
      context[query_head * width + dim] = static_cast<bfloat16>(weighted_sum);
    }
  }
  aie::set_rounding(saved_rounding);
}

\n