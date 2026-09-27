// Hot routines of the Parakeet encoder core program (experiment 012): the GEMM
// step, exp/sigmoid/Swish, the attention key loop and the convolution module,
// compiled -O2 (the rest of the program, pk_core.cc, is -Oz to fit 16 KB).
#include "pk_math.h"

using namespace pk;

namespace pk {

// Fast transcendental helpers built only from native operations: FP32
// products are formed from BF16 pieces with BF16 x BF16 -> FP32 MACs
// (~2^-16 relative), avoiding the emulated FP32 vector multiply (~45 cycles).

// a * b with a split into two BF16 pieces and b = (bh, bl), accumulated onto c.
inline fvec fma_split(fvec c, fvec a, bvec bh, bvec bl) {
  bvec ah, al;
  split2(a, ah, al);
  facc r;
  r.from_vector(c);
  r = aie::mac(r, ah, bh);
  r = aie::mac(r, ah, bl);
  r = aie::mac(r, al, bh);
  return r.to_vector<float>();
}

inline fvec fmul(fvec a, fvec b) {
  bvec bh, bl;
  split2(b, bh, bl);
  return fma_split(aie::zeros<float, 16>(), a, bh, bl);
}

// exp(x) per lane for x <= 0 (clamped at -87): 2^(x log2 e) = 2^n 2^f with
// n = round(t), f in [-1/2, 1/2], degree-4 polynomial (3.6e-6 relative).
[[gnu::noinline]] fvec exp16(fvec x) {
  x = aie::max(x, -87.0f);
  const bvec lh = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.4426950408889634f));
  const bvec ll = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(
      1.4426950408889634f - static_cast<float>(static_cast<bfloat16>(1.4426950408889634f))));
  const fvec t = fma_split(aie::zeros<float, 16>(), x, lh, ll);
  const ivec n = aie::to_fixed<int32_t>(t, 0);
  const fvec f = aie::sub(t, aie::to_float<float>(n, 0));
  bvec fh, fl;
  split2(f, fh, fl);
  constexpr float c[4] = {0.05592212f, 0.24022107f, 0.69312102f, 1.00000008f};
  fvec p = aie::broadcast<float, 16>(0.00967605f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 4; ++i)
    p = fma_split(aie::broadcast<float, 16>(c[i]), p, fh, fl);
  return aie::vector_cast<float>(
      aie::add(aie::vector_cast<int32_t>(p), aie::upshift(n, 23)));
}

// sigmoid(x) = 1 / (1 + e) for x >= 0, e / (1 + e) below, e = exp(-|x|);
// 1/d for d in (1, 2]: quadratic start (1.9%) and one Newton step (3.5e-4).
[[gnu::noinline]] fvec sigmoid(fvec x) {
  const fvec e = exp16(aie::min(x, aie::neg(x)));
  const fvec d = aie::add(e, 1.0f);
  bvec dh, dl;
  split2(d, dh, dl);
  fvec r = fma_split(aie::broadcast<float, 16>(-1.45898832f), aie::broadcast<float, 16>(0.32743468f),
                     dh, dl);
  r = fma_split(aie::broadcast<float, 16>(2.11761654f), r, dh, dl);
  const fvec dr = fma_split(aie::zeros<float, 16>(), r, dh, dl);
  r = fmul(r, aie::sub(aie::broadcast<float, 16>(2.0f), dr));
  return aie::select(fmul(e, r), r, aie::ge(x, aie::zeros<float, 16>()));
}

[[gnu::noinline]] fvec swish(fvec x) { return fmul(x, sigmoid(x)); }

// Eight keys ([k 128 | v 128] each) of the current head, key index j0..j0+7.
void att_keys(const bfloat16 *obj, int j0, int32_t *ctl, float *scr) {
  int nv = ctl[VALID] - j0;
  nv = nv > 8 ? 8 : nv;
  if (nv <= 0)
    return;
  const bfloat16 *qT = at<bfloat16>(scr, A_QT);
  const bfloat16 *bdT = at<bfloat16>(scr, A_BDT);
  float *accT = at<float>(scr, A_ACC);
  float *mp = at<float>(scr, A_M);
  float *lp = at<float>(scr, A_L);
  fvec s[8];
  fvec top = aie::load_v<16>(mp);
  for (int kk = 0; kk < nv; ++kk) {
    const bfloat16 *k = obj + kk * 256;
    facc a = aie::mul(aie::load_v<16>(bdT + (j0 + kk) * 16), ones());
    for (int d = 0; d < 128; ++d)
      a = aie::mac(a, aie::load_v<16>(qT + d * 16), k[d]);
    s[kk] = a.to_vector<float>();
    top = aie::max(top, s[kk]);
  }
  const fvec scale = exp16(aie::sub(aie::load_v<16>(mp), top));
  aie::store_v(mp, top);
  fvec lsum = vmul(aie::load_v<16>(lp), scale);
  bvec ph[8], pl[8];
  for (int kk = 0; kk < nv; ++kk) {
    const fvec p = exp16(aie::sub(s[kk], top));
    lsum = aie::add(lsum, p);
    split2(p, ph[kk], pl[kk]);
  }
  aie::store_v(lp, lsum);
  bvec sh, sl;
  split2(scale, sh, sl);
  for (int d = 0; d < 128; ++d) {
    bvec ah, al;
    split2(aie::load_v<16>(accT + d * 16), ah, al);
    facc a = aie::mul(ah, sh);
    a = aie::mac(a, ah, sl);
    a = aie::mac(a, al, sh);
    for (int kk = 0; kk < nv; ++kk) {
      const bfloat16 v = obj[kk * 256 + 128 + d];
      a = aie::mac(a, ph[kk], v);
      a = aie::mac(a, pl[kk], v);
    }
    aie::store_v(accT + d * 16, a.to_vector<float>());
  }
}

// Four input frames [a 256 | b 256] of the column's channels.
void conv_frames(const bfloat16 *obj, const bfloat16 *par, int32_t *ctl,
                                   float *scr) {
  bfloat16 *ring = at<bfloat16>(scr, C_RING);
  bfloat16 *out = at<bfloat16>(scr, C_OUT);
  const int ch0 = ctl[ROW] * 64;
  for (int f = 0; f < 4; ++f) {
    const int t = ctl[TCNT]++;
    const int slot = ctl[RPOS];
    ctl[RPOS] = slot == 8 ? 0 : slot + 1;
    bfloat16 *g = ring + slot * 64;
    for (int c = 0; c < 64; c += 16) {
      if (t < ctl[VALID]) {
        const fvec a = to_f32(aie::load_v<16>(obj + f * 512 + ch0 + c));
        const fvec b = to_f32(aie::load_v<16>(obj + f * 512 + 256 + ch0 + c));
        aie::store_v(g + c, to_bf16(fmul(a, sigmoid(b))));
      } else {
        aie::store_v(g + c, aie::zeros<bfloat16, 16>());
      }
    }
    if (t < 4)
      continue;
    bfloat16 *y = out + ctl[OROW] * 64;
    ctl[OROW] = ctl[OROW] + 1 == 16 * F ? 0 : ctl[OROW] + 1;
    const int oldest = slot == 8 ? 0 : slot + 1;  // frame t - 8
    for (int c = 0; c < 64; c += 16) {
      facc a = aie::mul(aie::load_v<16>(par + 9 * 256 + ch0 + c), ones());
      int s = oldest;
      for (int k = 0; k < 9; ++k) {
        a = aie::mac(a, aie::load_v<16>(par + k * 256 + ch0 + c),
                     aie::load_v<16>(ring + s * 64 + c));
        s = s == 8 ? 0 : s + 1;
      }
      aie::store_v(y + c, to_bf16(swish(a.to_vector<float>())));
    }
  }
}

}  // namespace pk


namespace {

// ---------------------------------------------------------------- GEMM
// C += A . B for A [rowA*4 x colA*8] and B [colA*8 x colB*4] in mmul block
// layout, C [rowA*4 x colB*4] FP32 in 4x4 blocks (row-major block order).
// 4x4 expansion of aie::mmul<4, 8, 4> after mlir-aie's aie_kernels/aie2/mm.cc
// (matmul_vectorized_4x4, Apache-2.0 WITH LLVM-exception).
template <unsigned rowA, unsigned colA, unsigned colB>
[[gnu::noinline]] void gemm_step(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto>;
  for (unsigned z = 0; z < rowA; z += 4) {
    float *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    float *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
    float *__restrict pC3 = pC + ((z + 2) * colB) * MMUL::size_C;
    float *__restrict pC4 = pC + ((z + 3) * colB) * MMUL::size_C;
    for (unsigned j = 0; j < colB; j += 4) {
      const bfloat16 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
      const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
      const bfloat16 *__restrict pA3 = pA + ((z + 2) * colA) * MMUL::size_A;
      const bfloat16 *__restrict pA4 = pA + ((z + 3) * colA) * MMUL::size_A;
      const bfloat16 *__restrict pB1 = pB + j * MMUL::size_B;
      const bfloat16 *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
      const bfloat16 *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
      const bfloat16 *__restrict pB4 = pB + (j + 3) * MMUL::size_B;
      MMUL C00(aie::load_v<16>(pC1)), C01(aie::load_v<16>(pC1 + 16)),
          C02(aie::load_v<16>(pC1 + 32)), C03(aie::load_v<16>(pC1 + 48));
      MMUL C10(aie::load_v<16>(pC2)), C11(aie::load_v<16>(pC2 + 16)),
          C12(aie::load_v<16>(pC2 + 32)), C13(aie::load_v<16>(pC2 + 48));
      MMUL C20(aie::load_v<16>(pC3)), C21(aie::load_v<16>(pC3 + 16)),
          C22(aie::load_v<16>(pC3 + 32)), C23(aie::load_v<16>(pC3 + 48));
      MMUL C30(aie::load_v<16>(pC4)), C31(aie::load_v<16>(pC4 + 16)),
          C32(aie::load_v<16>(pC4 + 32)), C33(aie::load_v<16>(pC4 + 48));
      for (unsigned i = 0; i < colA; ++i) {
        const auto A0 = aie::load_v<32>(pA1); pA1 += 32;
        const auto A1 = aie::load_v<32>(pA2); pA2 += 32;
        const auto A2 = aie::load_v<32>(pA3); pA3 += 32;
        const auto A3 = aie::load_v<32>(pA4); pA4 += 32;
        const auto B0 = aie::load_v<32>(pB1); pB1 += 32 * colB;
        const auto B1 = aie::load_v<32>(pB2); pB2 += 32 * colB;
        const auto B2 = aie::load_v<32>(pB3); pB3 += 32 * colB;
        const auto B3 = aie::load_v<32>(pB4); pB4 += 32 * colB;
        C00.mac(A0, B0); C01.mac(A0, B1); C02.mac(A0, B2); C03.mac(A0, B3);
        C10.mac(A1, B0); C11.mac(A1, B1); C12.mac(A1, B2); C13.mac(A1, B3);
        C20.mac(A2, B0); C21.mac(A2, B1); C22.mac(A2, B2); C23.mac(A2, B3);
        C30.mac(A3, B0); C31.mac(A3, B1); C32.mac(A3, B2); C33.mac(A3, B3);
      }
      aie::store_v(pC1, C00.template to_vector<float>()); aie::store_v(pC1 + 16, C01.template to_vector<float>());
      aie::store_v(pC1 + 32, C02.template to_vector<float>()); aie::store_v(pC1 + 48, C03.template to_vector<float>());
      aie::store_v(pC2, C10.template to_vector<float>()); aie::store_v(pC2 + 16, C11.template to_vector<float>());
      aie::store_v(pC2 + 32, C12.template to_vector<float>()); aie::store_v(pC2 + 48, C13.template to_vector<float>());
      aie::store_v(pC3, C20.template to_vector<float>()); aie::store_v(pC3 + 16, C21.template to_vector<float>());
      aie::store_v(pC3 + 32, C22.template to_vector<float>()); aie::store_v(pC3 + 48, C23.template to_vector<float>());
      aie::store_v(pC4, C30.template to_vector<float>()); aie::store_v(pC4 + 16, C31.template to_vector<float>());
      aie::store_v(pC4 + 32, C32.template to_vector<float>()); aie::store_v(pC4 + 48, C33.template to_vector<float>());
      pC1 += 64; pC2 += 64; pC3 += 64; pC4 += 64;
    }
  }
}

[[gnu::noinline]] void zero_acc(float *p, int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(p + i, aie::zeros<float, 16>());
}

}  // namespace

extern "C" void pk_wx(bfloat16 *w, bfloat16 *x, float *scr, bfloat16 *par, int32_t *ctl,
                      int32_t i) {
  if (i == 0)
    zero_acc(scr, TR * 32);
  gemm_step<TR / 4, 8, 8>(x, w, scr);
}
