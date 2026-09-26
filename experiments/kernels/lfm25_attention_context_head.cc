// One decode query per attention head; two query heads share each KV head.
#include <aie_api/aie.hpp>
#include <stdint.h>

static float exp_negative(float x) {
  if (x < -87.0f)
    return 0.0f;
  const int n = static_cast<int>(-x * 1.4426950408889634f);
  const float r = x + n * 0.6931471805599453f;
  float polynomial = 1.0f + r * (1.0f + r * (0.5f + r *
      (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f +
      r * (1.0f / 720.0f + r / 5040.0f))))));
  const uint32_t bits = static_cast<uint32_t>(127 - n) << 23;
  float power_two;
  __builtin_memcpy(&power_two, &bits, sizeof(power_two));
  return polynomial * power_two;
}

static void compute_attention_context_head(
    const bfloat16 *qkv, const bfloat16 *past_cache,
    bfloat16 *context, int32_t kv_head, int32_t past_length) {
  const auto saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const int total_length = past_length + 1;
  const bfloat16 *past_keys = past_cache;
  const bfloat16 *past_values = past_cache + past_length * 64;
  const bfloat16 *new_key = qkv + 1024 + kv_head * 64;
  const bfloat16 *new_value = qkv + 1536 + kv_head * 64;
  for (int group_head = 0; group_head < 2; ++group_head) {
    const int query_head = kv_head * 2 + group_head;
    const bfloat16 *query = qkv + query_head * 64;
    float scores[128];
    float maximum = -1.0e30f;
    for (int token = 0; token < total_length; ++token) {
      const bfloat16 *key = token < past_length
                                ? past_keys + token * 64
                                : new_key;
      float dot = 0.0f;
      for (int dim = 0; dim < 64; ++dim)
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
    bfloat16 probability[128];
    for (int token = 0; token < total_length; ++token)
      probability[token] = static_cast<bfloat16>(scores[token] / total);
    for (int dim = 0; dim < 64; ++dim) {
      float weighted_sum = 0.0f;
      for (int token = 0; token < total_length; ++token) {
        const bfloat16 value = token < past_length
                                   ? past_values[token * 64 + dim]
                                   : new_value[dim];
        weighted_sum += static_cast<float>(probability[token]) *
                        static_cast<float>(value);
      }
      context[query_head * 64 + dim] = static_cast<bfloat16>(weighted_sum);
    }
  }
  aie::set_rounding(saved_rounding);
}

extern "C" void lfm25_attention_context_head(
    const bfloat16 *qkv, const bfloat16 *past_cache,
    bfloat16 *context, int32_t kv_head) {
  compute_attention_context_head(qkv, past_cache, context, kv_head, 21);
}

extern "C" void lfm25_attention_context_head_dynamic(
    const bfloat16 *qkv, const bfloat16 *past_cache,
    bfloat16 *context, int32_t kv_head, int32_t past_length) {
  compute_attention_context_head(qkv, past_cache, context, kv_head,
                                 past_length);
}
