// BF16 per-head RMSNorm and rotary embedding for the first attention prefix.
#include <aie_api/aie.hpp>
#include <stdint.h>

static float invsqrt_approx(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits = 0x5f3759dfu - (bits >> 1);
  float result;
  __builtin_memcpy(&result, &bits, sizeof(result));
  const float half = 0.5f * value;
  for (int i = 0; i < 3; ++i)
    result *= 1.5f - half * result * result;
  return result;
}

extern "C" void lfm25_attention_norm_rope(bfloat16 *qkv,
                                            const bfloat16 *aux) {
  const auto saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *q_gamma = aux;
  const bfloat16 *k_gamma = aux + 64;
  const bfloat16 *cos = aux + 128;
  const bfloat16 *sin = aux + 192;
  for (int head = 0; head < 24; ++head) {
    bfloat16 *vector = qkv + (head < 16 ? head * 64 : 1024 + (head - 16) * 64);
    const bfloat16 *gamma = head < 16 ? q_gamma : k_gamma;
    float sum_sq = 0.0f;
    for (int d = 0; d < 64; ++d) {
      const float x = static_cast<float>(vector[d]);
      sum_sq += x * x;
    }
    const float inv_rms = invsqrt_approx(sum_sq / 64.0f + 1.0e-5f);
    bfloat16 normalized[64];
    for (int d = 0; d < 64; ++d) {
      const bfloat16 scaled =
          static_cast<bfloat16>(static_cast<float>(vector[d]) * inv_rms);
      normalized[d] = static_cast<bfloat16>(
          static_cast<float>(scaled) * static_cast<float>(gamma[d]));
    }
    for (int d = 0; d < 64; ++d) {
      const float rotated = d < 32
                                ? -static_cast<float>(normalized[d + 32])
                                : static_cast<float>(normalized[d - 32]);
      const bfloat16 part1 = static_cast<bfloat16>(
          static_cast<float>(normalized[d]) * static_cast<float>(cos[d]));
      const bfloat16 part2 =
          static_cast<bfloat16>(rotated * static_cast<float>(sin[d]));
      vector[d] = static_cast<bfloat16>(static_cast<float>(part1) +
                                       static_cast<float>(part2));
    }
  }
  aie::set_rounding(saved_rounding);
}
