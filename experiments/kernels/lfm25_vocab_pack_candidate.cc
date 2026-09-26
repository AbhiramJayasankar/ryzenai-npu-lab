// Pack a BF16 embedding row and the exact score/index bits into one FIFO item.
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void lfm25_vocab_pack_candidate(
    const bfloat16 *best_row, const float *best_score,
    const int32_t *best_index, bfloat16 *packed) {
  for (int col = 0; col < 1024; ++col)
    packed[col] = best_row[col];
  *reinterpret_cast<float *>(packed + 1024) = best_score[0];
  *reinterpret_cast<int32_t *>(packed + 1026) = best_index[0];
}
