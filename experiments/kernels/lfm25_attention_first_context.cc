// Attention at sequence position zero: softmax over one key gives its value.
#include <aie_api/aie.hpp>

extern "C" void lfm25_attention_first_context(
    const bfloat16 *qkv_and_hidden, bfloat16 *tail_input,
    bfloat16 *first_cache) {
  const bfloat16 *keys = qkv_and_hidden + 1024;
  const bfloat16 *values = qkv_and_hidden + 1536;
  const bfloat16 *hidden = qkv_and_hidden + 2048;
  for (int kv_head = 0; kv_head < 8; ++kv_head) {
    for (int dim = 0; dim < 64; ++dim) {
      const bfloat16 key = keys[kv_head * 64 + dim];
      const bfloat16 value = values[kv_head * 64 + dim];
      first_cache[kv_head * 128 + dim] = key;
      first_cache[kv_head * 128 + 64 + dim] = value;
      tail_input[1024 + (2 * kv_head) * 64 + dim] = value;
      tail_input[1024 + (2 * kv_head + 1) * 64 + dim] = value;
    }
  }
  for (int i = 0; i < 1024; ++i)
    tail_input[i] = hidden[i];
}
