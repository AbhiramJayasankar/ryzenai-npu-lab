// Update the maximum logit and first matching vocabulary index.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_argmax_chunk(const bfloat16 *logits,
                                     float *best_value,
                                     int32_t *best_index,
                                     int32_t base_index) {
  float best = best_value[0];
  int32_t index = best_index[0];
  for (int i = 0; i < 1024; ++i) {
    const float value = static_cast<float>(logits[i]);
    if (value > best) {
      best = value;
      index = base_index + i;
    }
  }
  best_value[0] = best;
  best_index[0] = index;
}
