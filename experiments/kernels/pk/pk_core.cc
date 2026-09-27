// Parakeet encoder core program for the Phoenix NPU (experiment 012).
//
// All 16 compute tiles run this program. Every phase of the encoder starts
// with a header object on the W stream (broadcast to the four cores of a
// column), which sets the object counts of the phase's segments:
//   prologue: NPRO W objects            (pk_hdr with op = 1)
//   NB blocks of: NWX (W, X) pairs      (pk_wx)
//                 NW W objects          (pk_w)
//                 NOUT output objects   (pk_out)
// Phases (see pk_engine.py for the data layout of each stream):
//   GEMM  C[frames, cols] = X[frames, K] . W[K, cols], optional per-column
//         bias and Swish. Core (c, r) owns frames [r*32F, (r+1)*32F) and
//         column group c; per 32-column chunk it takes K/64 steps of a 64x32
//         weight tile (mmul B blocks, host-packed) and a 32F x 64 activation
//         tile (mmul A blocks, re-blocked by the memory tile), FP32 accumulate.
//   LN    x' = x + s*y, a = LayerNorm(x') (or x' = LN_out(x + s*y) and a =
//         LN_next(x') at the end of a layer); frames in groups of F.
//   ATT   relative-position attention of one head per core quarter of the
//         queries, 16 queries per pass, online softmax over keys; the
//         position scores come precomputed from a GEMM phase.
//   CONV  GLU, depthwise convolution (kernel 9, batch norm folded) and Swish
//         on 64 channels per core.
#include "pk_math.h"

using namespace pk;

namespace {

// Four 4x4 blocks (columns 0-3, 4-7, 8-11, 12-15 of four rows) -> the four
// 16-column rows, via two rounds of chunk interleaving.
void rows16(const float *p, bfloat16 *out, int stride) {
  const auto [p0, p1] = aie::interleave_zip(to_bf16(aie::load_v<16>(p)),
                                            to_bf16(aie::load_v<16>(p + 16)), 4);
  const auto [q0, q1] = aie::interleave_zip(to_bf16(aie::load_v<16>(p + 32)),
                                            to_bf16(aie::load_v<16>(p + 48)), 4);
  const auto [r01a, r01b] = aie::interleave_zip(p0, q0, 8);
  const auto [r23a, r23b] = aie::interleave_zip(p1, q1, 8);
  aie::store_v(out, r01a);
  aie::store_v(out + stride, r01b);
  aie::store_v(out + 2 * stride, r23a);
  aie::store_v(out + 3 * stride, r23b);
}

// acc (32F x 32 in 4x4 FP32 blocks) -> out rows [32F][32] BF16.
[[gnu::noinline]] void store_rows(const float *acc, bfloat16 *out) {
  for (int rb = 0; rb < TR / 4; ++rb) {
    rows16(acc + rb * 128, out + rb * 128, 32);
    rows16(acc + rb * 128 + 64, out + rb * 128 + 16, 32);
  }
}

// ---------------------------------------------------------------- LayerNorm
// out = bf16((x - mean) * rstd * gamma + beta) over 1024 values, with the
// centred values and rstd kept to ~16 bits by BF16 splitting.
[[gnu::noinline]] void layer_norm(const bfloat16 *x, const bfloat16 *gb, bfloat16 *out) {
  const bvec one = ones();
  facc s = aie::zeros<accfloat, 16>();
  for (int i = 0; i < D; i += 16)
    s = aie::mac(s, aie::load_v<16>(x + i), one);
  const float mean = aie::reduce_add(vmul(s.to_vector<float>(), 1.0f / D));
  const fvec mv = aie::broadcast<float, 16>(mean);
  facc sq = aie::zeros<accfloat, 16>();
  for (int i = 0; i < D; i += 16) {
    bvec vh, vl;
    split2(aie::sub(to_f32(aie::load_v<16>(x + i)), mv), vh, vl);
    sq = aie::mac(sq, vh, vh);
    sq = aie::mac(sq, vh, vl);
    sq = aie::mac(sq, vh, vl);
  }
  const float var = aie::reduce_add(vmul(sq.to_vector<float>(), 1.0f / D));
  bfloat16 r0, r1, r2;
  split3(invsqrt16(aie::broadcast<float, 16>(var + 1.0e-5f))[0], r0, r1, r2);
  for (int i = 0; i < D; i += 16) {
    bvec vh, vl;
    split2(aie::sub(to_f32(aie::load_v<16>(x + i)), mv), vh, vl);
    facc n = aie::mul(vh, r0);
    n = aie::mac(n, vh, r1);
    n = aie::mac(n, vl, r0);
    bvec nh, nl;
    split2(n.to_vector<float>(), nh, nl);
    const bvec g = aie::load_v<16>(gb + i);
    facc y = aie::mul(aie::load_v<16>(gb + D + i), one);
    y = aie::mac(y, nh, g);
    y = aie::mac(y, nl, g);
    aie::store_v(out + i, to_bf16(y));
  }
}

// out = bf16(x + s * y)
[[gnu::noinline]] void residual(const bfloat16 *x, const bfloat16 *y, bfloat16 s, bfloat16 *out) {
  const bvec one = ones();
  for (int i = 0; i < D; i += 16) {
    facc a = aie::mul(aie::load_v<16>(x + i), one);
    a = aie::mac(a, aie::load_v<16>(y + i), s);
    aie::store_v(out + i, to_bf16(a));
  }
}

void ln_frame(const bfloat16 *obj, const bfloat16 *par, int32_t *ctl, bfloat16 *scr, int slot) {
  float sf;
  __builtin_memcpy(&sf, &ctl[SCALE], sizeof(sf));
  bfloat16 *xo = scr + slot * D;
  bfloat16 *ao = scr + (F + slot) * D;
  const int mode = ctl[MODE];
  if (mode == LN_SINGLE) {
    residual(obj, obj + D, static_cast<bfloat16>(sf), xo);
    layer_norm(xo, par, ao);
  } else {
    residual(obj, obj + D, static_cast<bfloat16>(sf), ao);  // ao as temporary
    layer_norm(ao, par, xo);
    if (mode == LN_DOUBLE)
      layer_norm(xo, par + 2 * D, ao);
    else
      copy_bf16(xo, ao, D);
  }
}

// ---------------------------------------------------------------- attention
void att_begin(float *scr) {
  aie::store_v(at<float>(scr, A_M), aie::broadcast<float, 16>(-1.0e30f));
  aie::store_v(at<float>(scr, A_L), aie::zeros<float, 16>());
  zero_f(at<float>(scr, A_ACC), 128 * 16);
}

// qT[d][q] = bf16(q[q][d] + delta[d]) for the core's 16 queries.
[[gnu::noinline]] void att_queries(const bfloat16 *q, const bfloat16 *delta, float *scr) {
  bfloat16 *qT = at<bfloat16>(scr, A_QT);
  for (int i = 0; i < 16; ++i)
    for (int d = 0; d < 128; ++d)
      qT[d * 16 + i] = static_cast<bfloat16>(static_cast<float>(q[i * 128 + d]) +
                                             static_cast<float>(delta[d]));
}

// Position scores: the BD stream carries, per query q, L values starting at
// the even address at or below its window; even queries skip one value.
[[gnu::noinline]] void att_bd(const bfloat16 *obj, int e0, int32_t *ctl, float *scr) {
  bfloat16 *bdT = at<bfloat16>(scr, A_BDT);
  const int L = ctl[BDL];
  const int base = ctl[ROW] * 16 * L;
  for (int q = 0; q < 16; ++q) {
    const int start = base + q * L + ((q & 1) ? 0 : 1);
    int lo = start > e0 ? start : e0;
    int hi = start + TPAD < e0 + WOBJ ? start + TPAD : e0 + WOBJ;
    for (int e = lo; e < hi; ++e)
      bdT[(e - start) * 16 + q] = obj[e - e0];
  }
}

// ---------------------------------------------------------------- conv module
}  // namespace

namespace pk {

// 1/sqrt(v) per lane (bit-trick start, three Newton steps), all in vector
// registers: scalar float multiplies would pull in soft-float code.
[[gnu::noinline]] fvec invsqrt16(fvec v) {
  const aie::vector<int32_t, 16> bits = aie::vector_cast<int32_t>(v);
  fvec y = aie::vector_cast<float>(
      aie::sub(aie::broadcast<int32_t, 16>(0x5f3759df), aie::downshift(bits, 1)));
  const fvec half = vmul(v, 0.5f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 3; ++i)
    y = vmul(y, aie::sub(aie::broadcast<float, 16>(1.5f), vmul(half, vmul(y, y))));
  return y;
}

// GEMM epilogue on the FP32 blocks: per-column bias (columns chunk*32 + ...,
// values in par) and/or Swish, in place.
void epilogue(float *acc, const bfloat16 *par, int32_t epi, int chunk) {
  alignas(32) float bias[8][16];
  if (epi & EPI_BIAS)
    for (int nb = 0; nb < 8; ++nb)
      for (int l = 0; l < 16; ++l)
        bias[nb][l] = static_cast<float>(par[chunk * 32 + nb * 4 + (l & 3)]);
  for (int b = 0; b < TR / 4 * 8; ++b) {
    fvec v = aie::load_v<16>(acc + b * 16);
    if (epi & EPI_BIAS)
      v = aie::add(v, aie::load_v<16>(bias[b & 7]));
    if (epi & EPI_SWISH)
      v = swish(v);
    aie::store_v(acc + b * 16, v);
  }
}

// o[q][d] = bf16(accT[d][q] / l[q])
void att_finish(float *scr) {
  const float *lp = at<float>(scr, A_L);
  const fvec r = invsqrt16(aie::load_v<16>(lp));
  bvec rh, rlo;
  split2(vmul(r, r), rh, rlo);
  const float *accT = at<float>(scr, A_ACC);
  bfloat16 *o = at<bfloat16>(scr, A_O);
  for (int d = 0; d < 128; ++d) {
    bvec ah, al;
    split2(aie::load_v<16>(accT + d * 16), ah, al);
    facc a = aie::mul(ah, rh);
    a = aie::mac(a, ah, rlo);
    a = aie::mac(a, al, rh);
    const bvec v = to_bf16(a);
    for (int q = 0; q < 16; ++q)
      o[q * 128 + d] = v[q];
  }
}

}  // namespace pk

extern "C" {

// Header (op == 0) or prologue object i (op == 1).
void pk_hdr(bfloat16 *w, float *scr, bfloat16 *par, int32_t *ctl, int32_t op, int32_t i) {
  if (op == 0) {
    // FP32 -> BF16 conversions round to nearest even (the default truncates);
    // the mode register persists, and every phase starts here.
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t h[16];
    __builtin_memcpy(h, w, sizeof(h));
    ctl[OP] = h[0];
    ctl[NPRO] = h[1];
    ctl[NB] = h[2];
    ctl[NWX] = h[3];
    ctl[NW] = h[4];
    ctl[NOUT] = h[5];
    ctl[EPI] = h[6];
    ctl[MODE] = h[7];
    ctl[SCALE] = h[8];
    ctl[PPH] = h[9];
    ctl[NQ] = h[10];
    ctl[NBD] = h[11];
    ctl[BDL] = h[12];
    ctl[CHUNK] = 0;
    ctl[NCH] = h[13];
    ctl[BLK] = 0;
    ctl[PASS] = 0;
    ctl[TCNT] = 0;
    ctl[RPOS] = 0;
    ctl[OROW] = 0;
    if (h[0] == OP_CONV)
      for (int k = 0; k < 9 * 64; k += 16)
        aie::store_v(at<bfloat16>(scr, C_RING) + k, aie::zeros<bfloat16, 16>());
    return;
  }
  const int o = ctl[OP];
  if (o == OP_GEMM || o == OP_LN) {
    copy_bf16(w, par + i * WOBJ, WOBJ);  // bias / LayerNorm parameters
  } else if (o == OP_ATT) {
    if (i == 0)
      ctl[VALID] = read_i32(w);
    else
      copy_bf16(w, par, 256);  // (bu - bv) / sqrt(128) of heads c and c + 4
  } else {  // OP_CONV
    if (i == 0)
      ctl[VALID] = read_i32(w);
    else if (i < 3)
      copy_bf16(w, par + (i - 1) * WOBJ, WOBJ);
    else
      conv_frames(w, par, ctl, scr);
  }
}

void pk_w(bfloat16 *w, float *scr, bfloat16 *par, int32_t *ctl, int32_t i) {
  const int o = ctl[OP];
  if (o == OP_LN) {
    const int slot = i - ctl[ROW] * F;
    if (slot >= 0 && slot < F)
      ln_frame(w, par, ctl, reinterpret_cast<bfloat16 *>(scr), slot);
  } else if (o == OP_ATT) {
    const int nq = ctl[NQ], nbd = ctl[NBD];
    if (i < nq) {
      if (i == 0)
        att_begin(scr);
      if (i == ctl[ROW])
        att_queries(w, par + (ctl[PASS] < ctl[PPH] ? 0 : 128), scr);
    } else if (i < nq + nbd) {
      att_bd(w, (i - nq) * WOBJ, ctl, scr);
    } else {
      att_keys(w, (i - nq - nbd) * 8, ctl, scr);
      if (i == ctl[NW] - 1) {
        att_finish(scr);
        ++ctl[PASS];
      }
    }
  } else {  // OP_CONV
    conv_frames(w, par, ctl, scr);
  }
}

void pk_out(bfloat16 *c, float *scr, bfloat16 *par, int32_t *ctl, int32_t j) {
  const int o = ctl[OP];
  if (o == OP_GEMM) {
    if (ctl[EPI])
      epilogue(scr, par, ctl[EPI], ctl[CHUNK]);
    store_rows(scr, c);
    ctl[CHUNK] = ctl[CHUNK] + 1 == ctl[NCH] ? 0 : ctl[CHUNK] + 1;  // chunks repeat per frame block
  } else if (o == OP_LN) {
    copy_bf16(reinterpret_cast<bfloat16 *>(scr) + j * F * D, c, F * D);
  } else if (o == OP_ATT) {
    copy_bf16(at<bfloat16>(scr, A_O) + j * COBJ, c, COBJ);
  } else {  // OP_CONV
    copy_bf16(at<bfloat16>(scr, C_OUT), c, COBJ);
  }
}

}  // extern "C"
