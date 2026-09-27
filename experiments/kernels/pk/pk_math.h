// Inline math and scratch layout shared by pk_core.cc (-Oz) and pk_vec.cc (-O2).
#pragma once
#include "pk_common.h"

namespace pk {

inline bvec ones() { return aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.0f)); }
inline bvec to_bf16(const facc &a) { return a.to_vector<bfloat16>(); }
inline bvec to_bf16(const fvec &v) {
  facc a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}
inline fvec to_f32(const bvec &v) { return aie::mul(v, ones()).to_vector<float>(); }

[[gnu::noinline]] static int32_t read_i32(const bfloat16 *p) {
  int32_t v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
}

fvec invsqrt16(fvec v);  // defined in pk_core.cc

// FP32 vector products (emulated on AIE-ML) kept out of line.
[[gnu::always_inline]] inline fvec vmul(fvec a, fvec b) { return aie::mul(a, b).to_vector<float>(); }
[[gnu::always_inline]] inline fvec vmul(fvec a, float b) { return aie::mul(a, b).to_vector<float>(); }





// Split an FP32 vector into two BF16 vectors whose sum approximates it to ~16 bits.
inline void split2(fvec v, bvec &hi, bvec &lo) {
  hi = to_bf16(v);
  lo = to_bf16(aie::sub(v, to_f32(hi)));
}

// Three BF16 pieces summing to an FP32 scalar (exact scaling).
[[gnu::noinline]] static void split3(float value, bfloat16 &p0, bfloat16 &p1, bfloat16 &p2) {
  p0 = static_cast<bfloat16>(value);
  const float r1 = value - static_cast<float>(p0);
  p1 = static_cast<bfloat16>(r1);
  p2 = static_cast<bfloat16>(r1 - static_cast<float>(p1));
}

[[gnu::noinline]] static void zero_f(float *p, int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(p + i, aie::zeros<float, 16>());
}

[[gnu::noinline]] static void copy_bf16(const bfloat16 *src, bfloat16 *dst, int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(dst + i, aie::load_v<16>(src + i));
}


// Scratch (bytes): qT bf16 [128][16] | bdT bf16 [TPAD][16] | accT f32 [128][16]
// | m f32 [16] | l f32 [16] | o bf16 [16][128]
constexpr int A_QT = 0;
constexpr int A_BDT = A_QT + 128 * 16 * 2;
constexpr int A_ACC = A_BDT + TPAD * 16 * 2;
constexpr int A_M = A_ACC + 128 * 16 * 4;
constexpr int A_L = A_M + 64;
constexpr int A_O = A_QT;  // o reuses qT: written by att_finish after the last key
constexpr int A_END = A_L + 64;

template <typename T> inline T *at(float *scr, int byte) {
  return reinterpret_cast<T *>(reinterpret_cast<char *>(scr) + byte);
}


// Scratch: ring bf16 [9][64] of GLU outputs | out bf16 [16F][64]
constexpr int C_RING = 0;
constexpr int C_OUT = 9 * 64 * 2;


// Out of line (one copy each), defined -O2 in pk_vec.cc.
fvec exp16(fvec x);
fvec sigmoid(fvec x);
fvec swish(fvec x);

// Hot routines compiled -O2 in pk_vec.cc.
void epilogue(float *acc, const bfloat16 *par, int32_t epi, int chunk);
void att_keys(const bfloat16 *obj, int j0, int32_t *ctl, float *scr);
void att_finish(float *scr);
void conv_frames(const bfloat16 *obj, const bfloat16 *par, int32_t *ctl, float *scr);

}  // namespace pk
