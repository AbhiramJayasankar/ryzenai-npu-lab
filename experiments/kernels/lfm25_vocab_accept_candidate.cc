// Merge a neighboring worker's best row into the NPU-resident winner.
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void lfm25_vocab_accept_candidate(
    const bfloat16 *packed, bfloat16 *best_row,
    float *best_score, int32_t *best_index) {
  const float score = *reinterpret_cast<const float *>(packed + 1024);
  const int32_t index = *reinterpret_cast<const int32_t *>(packed + 1026);
  if (score > best_score[0] || (score == best_score[0] && index < best_index[0])) {
    best_score[0] = score;
    best_index[0] = index;
    for (int col = 0; col < 1024; ++col)
      best_row[col] = packed[col];
  }
}
