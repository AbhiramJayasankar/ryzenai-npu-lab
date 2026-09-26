// Append the NPU-generated key and value for one KV head to its cache.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_attention_append_cache(const bfloat16 *qkv,
                                                const bfloat16 *past,
                                                bfloat16 *next,
                                                int32_t kv_head) {
  constexpr int width = 64;
  constexpr int past_length = 21;
  constexpr int next_length = past_length + 1;
  const bfloat16 *old_keys = past;
  const bfloat16 *old_values = past + past_length * width;
  bfloat16 *new_keys = next;
  bfloat16 *new_values = next + next_length * width;
  for (int i = 0; i < past_length * width; ++i) {
    new_keys[i] = old_keys[i];
    new_values[i] = old_values[i];
  }
  const bfloat16 *key = qkv + 1024 + kv_head * width;
  const bfloat16 *value = qkv + 1536 + kv_head * width;
  for (int i = 0; i < width; ++i) {
    new_keys[past_length * width + i] = key[i];
    new_values[past_length * width + i] = value[i];
  }
}
