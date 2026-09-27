// Shared BF16/FP32 math for the LFM2.5 Phoenix core programs (decode:
// x8_core.cc, batched prompt: x8p_core.cc). BF16 rounding at the model's
// boundaries with round-to-nearest-even (set by the caller) and FP32
// accumulation; FP32 scalars are split into BF16 pieces for exact products.
#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int OBJ = 512;

using bvec = aie::vector<bfloat16, 16>;
using facc = aie::accum<accfloat, 16>;

[[gnu::noinline]] int32_t read_i32(const bfloat16 *p) {
  int32_t v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
}

bvec to_bf16(const facc &a) { return a.to_vector<bfloat16>(); }
bvec ones() { return aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.0f)); }

[[gnu::noinline]] float invsqrt(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits = 0x5f3759dfu - (bits >> 1);
  float result;
  __builtin_memcpy(&result, &bits, sizeof(result));
  const float half_value = 0.5f * value;
  for (int i = 0; i < 3; ++i)
    result *= 1.5f - half_value * result * result;
  return result;
}

// Three BF16 pieces summing to an FP32 scalar, for exact vector scaling.
[[gnu::noinline]] void split3(float value, bfloat16 &p0, bfloat16 &p1, bfloat16 &p2) {
  p0 = static_cast<bfloat16>(value);
  const float r1 = value - static_cast<float>(p0);
  p1 = static_cast<bfloat16>(r1);
  p2 = static_cast<bfloat16>(r1 - static_cast<float>(p1));
}

using fvec = aie::vector<float, 16>;
using ivec = aie::vector<int32_t, 16>;

// FP32 vector products (emulated on AIE-ML) kept out of line to save
// program memory.
[[gnu::noinline]] fvec vmul(fvec a, fvec b) { return aie::mul(a, b).to_vector<float>(); }
[[gnu::noinline]] fvec vmul(fvec a, float b) { return aie::mul(a, b).to_vector<float>(); }

// exp(x) per lane for x <= 0 (clamped at -87) in FP32: x = -n*ln2 + r with
// n = round(-x*log2e), |r| <= ln2/2, degree-6 polynomial, 2^-n from bits.
[[gnu::noinline]] fvec exp16(fvec x) {
  x = aie::max(x, -87.0f);
  const ivec n = aie::to_fixed<int32_t>(vmul(x, -1.4426950408889634f), 0);
  const fvec r = aie::add(x, vmul(aie::to_float<float>(n, 0), 0.6931471805599453f));
  constexpr float c[6] = {1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f, 0.5f, 1.0f, 1.0f};
  fvec poly = aie::broadcast<float, 16>(1.0f / 720.0f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 6; ++i)
    poly = aie::add(vmul(poly, r), aie::broadcast<float, 16>(c[i]));
  const ivec bits = aie::upshift(aie::sub(aie::broadcast<int32_t, 16>(127), n), 23);
  return vmul(poly, aie::vector_cast<float>(bits));
}

// dot(w[0:512], x[0:512]) in FP32.
[[gnu::noinline]] float dot512(const bfloat16 *w, const bfloat16 *x) {
  facc a0 = aie::zeros<accfloat, 16>();
  facc a1 = aie::zeros<accfloat, 16>();
  facc a2 = aie::zeros<accfloat, 16>();
  facc a3 = aie::zeros<accfloat, 16>();
#pragma clang loop unroll_count(4)
  for (int c = 0; c < OBJ; c += 64) {
    a0 = aie::mac(a0, aie::load_v<16>(w + c), aie::load_v<16>(x + c));
    a1 = aie::mac(a1, aie::load_v<16>(w + c + 16), aie::load_v<16>(x + c + 16));
    a2 = aie::mac(a2, aie::load_v<16>(w + c + 32), aie::load_v<16>(x + c + 32));
    a3 = aie::mac(a3, aie::load_v<16>(w + c + 48), aie::load_v<16>(x + c + 48));
  }
  return aie::reduce_add(aie::add(aie::add(a0.to_vector<float>(), a1.to_vector<float>()),
                                  aie::add(a2.to_vector<float>(), a3.to_vector<float>())));
}

// v[0:64] *= s in FP32.
[[gnu::noinline]] void scale64(float *v, float s) {
  for (int c = 0; c < 64; c += 16)
    aie::store_v(v + c, vmul(aie::load_v<16>(v + c), s));
}

// 1/x for x > 0 from the inverse square root (avoids soft-float division).
[[gnu::noinline]] float reciprocal(float x) {
  const float r = invsqrt(x);
  return r * r;
}

[[gnu::noinline]] void copy(const bfloat16 *src, bfloat16 *dst, int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(dst + i, aie::load_v<16>(src + i));
}

[[gnu::noinline]] void zero_f(float *acc, int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(acc + i, aie::zeros<float, 16>());
}

// out[i] = bf16(acc[i]).
[[gnu::noinline]] void cast(const float *acc, bfloat16 *out, int n) {
  for (int i = 0; i < n; i += 16) {
    facc a;
    a.from_vector(aie::load_v<16>(acc + i));
    aie::store_v(out + i, to_bf16(a));
  }
}

// out[i] = bf16(residual[i] + proj[i]).
[[gnu::noinline]] void add(const bfloat16 *residual, const bfloat16 *proj, bfloat16 *out, int n) {
  const bvec one = ones();
  for (int i = 0; i < n; i += 16) {
    facc a = aie::mul(aie::load_v<16>(residual + i), one);
    a = aie::mac(a, aie::load_v<16>(proj + i), one);
    aie::store_v(out + i, to_bf16(a));
  }
}

// RMSNorm over 1024 values: normalized = bf16(x * inv_rms), out =
// bf16(normalized * gamma); x * inv_rms is formed exactly in FP32.
[[gnu::noinline]] void norm(const bfloat16 *input, const bfloat16 *gamma, bfloat16 *output) {
  facc sq0 = aie::zeros<accfloat, 16>();
  facc sq1 = aie::zeros<accfloat, 16>();
  for (int i = 0; i < 1024; i += 32) {
    const bvec a = aie::load_v<16>(input + i);
    const bvec b = aie::load_v<16>(input + i + 16);
    sq0 = aie::mac(sq0, a, a);
    sq1 = aie::mac(sq1, b, b);
  }
  const float sum_sq =
      aie::reduce_add(aie::add(sq0.to_vector<float>(), sq1.to_vector<float>()));
  bfloat16 p0, p1, p2;
  split3(invsqrt(sum_sq * (1.0f / 1024) + 1.0e-5f), p0, p1, p2);
  for (int i = 0; i < 1024; i += 16) {
    const bvec x = aie::load_v<16>(input + i);
    facc acc = aie::mul(x, p0);
    acc = aie::mac(acc, x, p1);
    acc = aie::mac(acc, x, p2);
    aie::store_v(output + i, to_bf16(aie::mul(to_bf16(acc), aie::load_v<16>(gamma + i))));
  }
}

[[gnu::noinline]] void norm_rope(bfloat16 *vector, const bfloat16 *gamma, const bfloat16 *cos,
               const bfloat16 *sin) {
  facc sq = aie::zeros<accfloat, 16>();
  for (int c = 0; c < 64; c += 16) {
    const bvec x = aie::load_v<16>(vector + c);
    sq = aie::mac(sq, x, x);
  }
  bfloat16 p0, p1, p2;
  split3(invsqrt(aie::reduce_add(sq.to_vector<float>()) * (1.0f / 64) + 1.0e-5f), p0, p1, p2);
  alignas(32) bfloat16 n[64];
  for (int c = 0; c < 64; c += 16) {
    const bvec x = aie::load_v<16>(vector + c);
    facc a = aie::mul(x, p0);
    a = aie::mac(a, x, p1);
    a = aie::mac(a, x, p2);
    aie::store_v(n + c, to_bf16(aie::mul(to_bf16(a), aie::load_v<16>(gamma + c))));
  }
  const bvec one = ones();
  for (int c = 0; c < 64; c += 16) {
    const bvec rotated = c < 32 ? aie::neg(aie::load_v<16>(n + c + 32))
                                : aie::load_v<16>(n + c - 32);
    const bvec part1 = to_bf16(aie::mul(aie::load_v<16>(n + c), aie::load_v<16>(cos + c)));
    const bvec part2 = to_bf16(aie::mul(rotated, aie::load_v<16>(sin + c)));
    facc sum = aie::mul(part1, one);
    sum = aie::mac(sum, part2, one);
    aie::store_v(vector + c, to_bf16(sum));
  }
}

// Online-softmax state per query head h at st[h*80]: [0] running max,
// [1] running sum, [16:80) weighted values. Probabilities are
// exp(score - running max) in FP32, normalized at the end.
// Up to four positions ([key 64 | value 64] each, n valid) for both query
// heads: BF16-rounded scaled scores, then one online-softmax step per head.
// Score lanes are h*4 + j; lanes 8 + h carry the running-max rescale.
[[gnu::noinline]] void attn_positions(const bfloat16 *kv, int n, const bfloat16 *query,
                                      float *st) {
  alignas(64) float raw[16];
  for (int l = 0; l < 16; ++l)
    raw[l] = 0.0f;
  for (int j = 0; j < n; ++j)
    for (int h = 0; h < 2; ++h) {
      facc d = aie::zeros<accfloat, 16>();
      for (int c = 0; c < 64; c += 16)
        d = aie::mac(d, aie::load_v<16>(query + h * 64 + c), aie::load_v<16>(kv + j * 128 + c));
      raw[h * 4 + j] = aie::reduce_add(d.to_vector<float>());
    }
  facc rounded;
  rounded.from_vector(aie::load_v<16>(raw));
  // bf16(dot), then * 0.125 (exact for BF16 values).
  alignas(64) float score[16];
  aie::store_v(score, aie::mul(to_bf16(rounded),
                               aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(0.125f)))
                          .to_vector<float>());
  alignas(64) float top[16];
  alignas(64) float arg[16];
  for (int h = 0; h < 2; ++h) {
    float m = st[h * 80];
    for (int j = 0; j < n; ++j)
      if (score[h * 4 + j] > m)
        m = score[h * 4 + j];
    for (int j = 0; j < 4; ++j) {
      top[h * 4 + j] = m;
      arg[h * 4 + j] = j < n ? score[h * 4 + j] : -1.0e30f;
    }
    top[8 + h] = m;
    arg[8 + h] = st[h * 80];
    st[h * 80] = m;
  }
  for (int l = 10; l < 16; ++l) {
    top[l] = 0.0f;
    arg[l] = 0.0f;
  }
  const fvec e = exp16(aie::sub(aie::load_v<16>(arg), aie::load_v<16>(top)));
  // Two BF16 pieces per probability for exact-enough FP32 accumulation.
  facc e_acc;
  e_acc.from_vector(e);
  const bvec hi = to_bf16(e_acc);
  facc residual;
  residual.from_vector(aie::sub(e, aie::mul(hi, ones()).to_vector<float>()));
  const bvec lo = to_bf16(residual);
  for (int h = 0; h < 2; ++h) {
    float *state = st + h * 80;
    const float scale = e[8 + h];
    float sum = state[1] * scale;
    for (int j = 0; j < n; ++j)
      sum += e[h * 4 + j];
    state[1] = sum;
    float *acc = state + 16;
    uint32_t bits;  // skip the rescale when the running max did not change
    __builtin_memcpy(&bits, &scale, sizeof(bits));
    const bool rescale = bits != 0x3f800000u;
    for (int c = 0; c < 64; c += 16) {
      facc a;
      const fvec current = aie::load_v<16>(acc + c);
      a.from_vector(rescale ? vmul(current, scale) : current);
      for (int j = 0; j < n; ++j) {
        const bvec v = aie::load_v<16>(kv + j * 128 + 64 + c);
        a = aie::mac(a, v, hi[h * 4 + j]);
        a = aie::mac(a, v, lo[h * 4 + j]);
      }
      aie::store_v(acc + c, a.to_vector<float>());
    }
  }
}

}  // namespace
