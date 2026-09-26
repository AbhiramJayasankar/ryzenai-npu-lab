// Score one tied vocabulary row and retain the winning embedding on AIE.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void lfm25_vocab_score_keep_row(
    const bfloat16 *weight_row, const bfloat16 *activation,
    bfloat16 *best_embedding, float *best_score,
    int32_t *best_index, int32_t row_index) {
  const auto saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int col = 0; col < 1024; col += 16) {
    const aie::vector<bfloat16, 16> w =
        aie::load_v<16>(weight_row + col);
    const aie::vector<bfloat16, 16> x =
        aie::load_v<16>(activation + col);
    acc = aie::mac(acc, w, x);
  }
  const float dot = aie::reduce_add(acc.to_vector<float>());
  const bfloat16 rounded = static_cast<bfloat16>(dot);
  const float score = static_cast<float>(rounded);
  if (score > best_score[0]) {
    best_score[0] = score;
    best_index[0] = row_index;
    for (int col = 0; col < 1024; ++col)
      best_embedding[col] = weight_row[col];
  }
  aie::set_rounding(saved_rounding);
}
