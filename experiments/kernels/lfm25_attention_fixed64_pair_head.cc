// Reuse the validated one-position head operation for a two-position QKV row.
#include "lfm25_attention_fixed64_head.cc"

extern "C" void lfm25_attention_fixed64_pair_head(
    const bfloat16 *qkv_pair, const bfloat16 *past, bfloat16 *next,
    bfloat16 *context, int32_t kv_head, int32_t position_in_pair) {
  lfm25_attention_fixed64_head(
      qkv_pair + position_in_pair * 3072, past, next, context, kv_head);
}
