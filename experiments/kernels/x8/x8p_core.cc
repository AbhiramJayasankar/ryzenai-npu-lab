// Batched prompt processing: one Phoenix compute tile's share of four
// consecutive prompt tokens per weight pass (LFM2.5-230M layers only).
//
// Same object-driven state machine as x8_core.cc (five segments per
// stream: IN objects, OUT objects, NEXT), but every weight object is used
// for B = 4 tokens, so weights stream from DDR once per four positions.
//
// Segments of a layer stream (core k owns output slice k):
//   0: B input vectors (2 objects each), B token aux objects, RMSNorm gamma
//      (2), header (1)
//   1: projection rows (2 objects per row) then tail objects: conv state
//      (recurrent) or KV cache blocks (attention); outputs: B slice outputs,
//      then the new conv state or B KV entries
//   2: B gathered 1024-vectors + 128 output-projection rows; B outputs
//   3: B h2 vectors + FFN gamma (2) + 320 interleaved W1/W3 rows (4 objects)
//   4: B gated 2560-vectors + 128 W2 rows (5 objects each); B outputs
// All tokens of a batch share the cached context (past length P); within
// the batch, token t also attends to tokens 0..t (causal).
#include "x8_math.h"

namespace {

constexpr int B = 4;
constexpr int HIN = 0;                  // B x 1024: h, then h2
constexpr int ACT = HIN + B * 1024;     // B x 2560: normed / gathered inputs
constexpr int GAM = ACT + B * 2560;     // 1024
constexpr int PROJ = GAM + 1024;        // B x 640: projection outputs (BF16)
constexpr int HDR = PROJ + B * 640;     // 512: layer header
constexpr int AUX = HDR + 512;          // B x 512: token aux (cos, sin, ...)
constexpr int SH = AUX + B * 512;       // B x 512: staged outputs
constexpr int STATE = SH + B * 512;     // 2 x 512: conv state ping-pong
// acc (FP32): online-softmax state of the batch's 2B queries (query q =
// token q/2, head q%2): running max at [0:16), running sum at [16:32),
// weighted values at SOFT + q*64; projection row accumulators at ROWACC.
constexpr int QUERIES = 2 * B;
constexpr int SOFT = 32;
constexpr int ROWACC = SOFT + QUERIES * 64;
constexpr int MODE_ATTENTION = 1;
constexpr int OP_IN = 0, OP_OUT = 1;
enum Ctl { N_IN, N_OUT, SEG, MODE, ROWS, TAIL, PAST, CORE, ROW, COL };

// acc[t] += dot(w, x[t*stride : +512]) for the B tokens, one weight load.
[[gnu::noinline]] void dot4(const bfloat16 *w, const bfloat16 *x, int stride, float *acc) {
  facc a0 = aie::zeros<accfloat, 16>();
  facc a1 = aie::zeros<accfloat, 16>();
  facc a2 = aie::zeros<accfloat, 16>();
  facc a3 = aie::zeros<accfloat, 16>();
#pragma clang loop unroll_count(8)
  for (int c = 0; c < OBJ; c += 16) {
    const bvec wv = aie::load_v<16>(w + c);
    a0 = aie::mac(a0, wv, aie::load_v<16>(x + c));
    a1 = aie::mac(a1, wv, aie::load_v<16>(x + stride + c));
    a2 = aie::mac(a2, wv, aie::load_v<16>(x + 2 * stride + c));
    a3 = aie::mac(a3, wv, aie::load_v<16>(x + 3 * stride + c));
  }
  acc[0] += aie::reduce_add(a0.to_vector<float>());
  acc[1] += aie::reduce_add(a1.to_vector<float>());
  acc[2] += aie::reduce_add(a2.to_vector<float>());
  acc[3] += aie::reduce_add(a3.to_vector<float>());
}

// dst[t*stride + row] = bf16(acc[t]); acc[t] = 0.
[[gnu::noinline]] void finish_row(float *acc, bfloat16 *dst, int stride, int row) {
  for (int t = 0; t < B; ++t) {
    dst[t * stride + row] = static_cast<bfloat16>(acc[t]);
    acc[t] = 0.0f;
  }
}

// One weight object of a row-major GEMV whose rows span `cols` objects.
[[gnu::noinline]] void gemv_obj(const bfloat16 *w, const bfloat16 *x, int x_stride,
                                bfloat16 *dst, int dst_stride, float *acc, int32_t *ctl,
                                int cols) {
  const int col = ctl[COL];
  dot4(w, x + col * OBJ, x_stride, acc);
  if (col + 1 == cols) {
    finish_row(acc, dst, dst_stride, ctl[ROW]);
    ctl[COL] = 0;
    ++ctl[ROW];
  } else {
    ctl[COL] = col + 1;
  }
}

// 1/d per lane for d in [1, 2]: linear start, three Newton steps.
[[gnu::noinline]] fvec recip12(fvec d) {
  fvec r = aie::sub(aie::broadcast<float, 16>(24.0f / 17.0f), vmul(d, 8.0f / 17.0f));
#pragma clang loop unroll(disable)
  for (int i = 0; i < 3; ++i)
    r = vmul(r, aie::sub(aie::broadcast<float, 16>(2.0f), vmul(d, r)));
  return r;
}

// out[i] = bf16(bf16(x * sigmoid(x)) * u[i]) for x = g[i], sigmoid in FP32.
[[gnu::noinline]] void silu_gate(const bfloat16 *g, const bfloat16 *u, bfloat16 *out, int n) {
  const bvec one = ones();
  for (int i = 0; i < n; i += 16) {
    const fvec x = aie::mul(aie::load_v<16>(g + i), one).to_vector<float>();
    // sigmoid(x) = a / (a + b) with p = max(x, 0), a = exp(x - p),
    // b = exp(-p): both exponents are <= 0 and a + b lies in [1, 2].
    const fvec p = aie::max(x, 0.0f);
    const fvec a = exp16(aie::sub(x, p));
    const fvec b = exp16(aie::sub(aie::zeros<float, 16>(), p));
    const fvec sigmoid = vmul(a, recip12(aie::add(a, b)));

    facc s;
    s.from_vector(vmul(x, sigmoid));
    aie::store_v(out + i, to_bf16(aie::mul(to_bf16(s), aie::load_v<16>(u + i))));
  }
}

// Gated depthwise convolution for one token, planar [t*128 + ch] state and
// weights: y = C * conv(state, Bx); next state shifts in Bx.
[[gnu::noinline]] void conv_step(const bfloat16 *proj, const bfloat16 *weight,
                                 const bfloat16 *state, bfloat16 *y, bfloat16 *next_state) {
  const bvec one = ones();
  for (int i = 0; i < 128; i += 16) {
    const bvec Bv = aie::load_v<16>(proj + i);
    const bvec C = aie::load_v<16>(proj + 128 + i);
    const bvec x = aie::load_v<16>(proj + 256 + i);
    const bvec Bx = to_bf16(aie::mul(Bv, x));
    const bvec previous1 = aie::load_v<16>(state + 128 + i);
    const bvec previous2 = aie::load_v<16>(state + 256 + i);
    aie::store_v(next_state + i, previous1);
    aie::store_v(next_state + 128 + i, previous2);
    aie::store_v(next_state + 256 + i, Bx);
    const bvec product0 = to_bf16(aie::mul(previous1, aie::load_v<16>(weight + i)));
    const bvec product1 = to_bf16(aie::mul(previous2, aie::load_v<16>(weight + 128 + i)));
    const bvec product2 = to_bf16(aie::mul(Bx, aie::load_v<16>(weight + 256 + i)));
    facc sum = aie::mul(product0, one);
    sum = aie::mac(sum, product1, one);
    sum = aie::mac(sum, product2, one);
    aie::store_v(y + i, to_bf16(aie::mul(C, to_bf16(sum))));
  }
}

// Per-head norm + rotary for every token's q (2 heads) and k; reset the
// batch's online-softmax state.
[[gnu::noinline]] void attn_prep(bfloat16 *m, float *acc) {
  for (int t = 0; t < B; ++t) {
    bfloat16 *qkv = m + PROJ + t * 384;
    const bfloat16 *cos = m + AUX + t * OBJ;
    const bfloat16 *sin = cos + 64;
    norm_rope(qkv, m + HDR, cos, sin);
    norm_rope(qkv + 64, m + HDR, cos, sin);
    norm_rope(qkv + 128, m + HDR + 64, cos, sin);
  }
  aie::store_v(acc, aie::broadcast<float, 16>(-1.0e30f));
  zero_f(acc + 16, 16 + QUERIES * 64);
}

// One online-softmax step of all 2B queries (one per lane) over n <= 4
// positions, position j at kv + j*stride = [key 64 | value 64]. With
// `causal`, position j is visible to token t only when j <= t (the batch's
// own tokens). Scores are bf16(q.k) * 0.125 as in the single-token path.
[[gnu::noinline]] void attn_block4(const bfloat16 *kv, int stride, int n, bool causal,
                                   const bfloat16 *m, float *acc) {
  alignas(64) float raw[4][16];
  for (int j = 0; j < 4; ++j)
    for (int q = 0; q < 16; ++q) {
      float dot = -1.0e30f;
      if (q < QUERIES && j < n && (!causal || j <= q / 2)) {
        const bfloat16 *query = m + PROJ + (q / 2) * 384 + (q % 2) * 64;
        facc d = aie::zeros<accfloat, 16>();
        for (int c = 0; c < 64; c += 16)
          d = aie::mac(d, aie::load_v<16>(query + c), aie::load_v<16>(kv + j * stride + c));
        dot = aie::reduce_add(d.to_vector<float>());
      }
      raw[j][q] = dot;
    }
  const bvec eighth = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(0.125f));
  fvec score[4];
  fvec top = aie::load_v<16>(acc);
  for (int j = 0; j < 4; ++j) {
    facc r;
    r.from_vector(aie::load_v<16>(raw[j]));
    score[j] = aie::mul(to_bf16(r), eighth).to_vector<float>();
    top = aie::max(top, score[j]);
  }
  const fvec scale = exp16(aie::sub(aie::load_v<16>(acc), top));
  fvec sum = vmul(aie::load_v<16>(acc + 16), scale);
  // Probabilities of positions j and j+1 share one exp: queries use lanes
  // 0..7 of each score vector, so pack them as [j | j+1].
  alignas(64) bfloat16 hi[4][16];
  alignas(64) bfloat16 lo[4][16];
  const fvec top2 = aie::concat(top.extract<8>(0), top.extract<8>(0));
  for (int j = 0; j < 4; j += 2) {
    const fvec pair = exp16(aie::sub(
        aie::concat(score[j].extract<8>(0), score[j + 1].extract<8>(0)), top2));
    const fvec p0 = aie::concat(pair.extract<8>(0), aie::zeros<float, 8>());
    const fvec p1 = aie::concat(pair.extract<8>(1), aie::zeros<float, 8>());
    sum = aie::add(sum, aie::add(p0, p1));
    facc pa;
    pa.from_vector(pair);
    const bvec h = to_bf16(pa);
    facc residual;
    residual.from_vector(aie::sub(pair, aie::mul(h, ones()).to_vector<float>()));
    const bvec l = to_bf16(residual);
    aie::store_v(hi[j], h);
    aie::store_v(lo[j], l);
    aie::store_v(hi[j + 1], aie::shuffle_down_rotate(h, 8));
    aie::store_v(lo[j + 1], aie::shuffle_down_rotate(l, 8));
  }
  aie::store_v(acc, top);
  aie::store_v(acc + 16, sum);
  alignas(64) float scales[16];
  aie::store_v(scales, scale);
  for (int q = 0; q < QUERIES; ++q) {
    float *values = acc + SOFT + q * 64;
    // The running max rarely changes after the first blocks: skip the
    // rescale when this query's factor is exactly 1.
    uint32_t bits;
    __builtin_memcpy(&bits, &scales[q], sizeof(bits));
    const bool rescale = bits != 0x3f800000u;
    for (int c = 0; c < 64; c += 16) {
      facc a;
      const fvec current = aie::load_v<16>(values + c);
      a.from_vector(rescale ? vmul(current, scales[q]) : current);
      for (int j = 0; j < n; ++j) {
        const bvec v = aie::load_v<16>(kv + j * stride + 64 + c);
        a = aie::mac(a, v, hi[j][q]);
        a = aie::mac(a, v, lo[j][q]);
      }
      aie::store_v(values + c, a.to_vector<float>());
    }
  }
}

[[gnu::noinline]] void input(const bfloat16 *o, bfloat16 *m, float *acc, int32_t *ctl, int i) {
  float *rows = acc + ROWACC;
  switch (ctl[SEG]) {
  case 0:
    if (i < 2 * B)
      copy(o, m + HIN + i * OBJ, OBJ);
    else if (i < 3 * B)
      copy(o, m + AUX + (i - 2 * B) * OBJ, OBJ);
    else if (i < 3 * B + 2)
      copy(o, m + GAM + (i - 3 * B) * OBJ, OBJ);
    else
      copy(o, m + HDR, OBJ);
    return;
  case 1: {
    const int tail = i - 2 * ctl[ROWS];
    if (tail < 0) {
      gemv_obj(o, m + ACT, 1024, m + PROJ, 384, rows, ctl, 2);
      return;
    }
    if (ctl[MODE] == MODE_ATTENTION) {
      if (tail == 0)
        attn_prep(m, acc);
      const int n = ctl[PAST] - tail * 4;
      if (n > 0)
        attn_block4(o, 128, n < 4 ? n : 4, false, m, acc);
    } else {
      const bfloat16 *state = o;
      for (int t = 0; t < B; ++t) {
        bfloat16 *next_state = m + STATE + (t & 1) * OBJ;
        conv_step(m + PROJ + t * 384, m + HDR, state, m + SH + t * OBJ, next_state);
        state = next_state;
      }
    }
    return;
  }
  case 2:
    if (i < 2 * B)
      copy(o, m + ACT + i * OBJ, OBJ);
    else
      gemv_obj(o, m + ACT, 1024, m + PROJ, 384, rows, ctl, 2);
    return;
  case 3:
    if (i < 2 * B) {
      copy(o, m + HIN + i * OBJ, OBJ);
    } else if (i < 2 * B + 2) {
      copy(o, m + GAM + (i - 2 * B) * OBJ, OBJ);
    } else {
      if (i == 2 * B + 2)
        for (int t = 0; t < B; ++t)
          norm(m + HIN + t * 1024, m + GAM, m + ACT + t * 1024);
      // Row r of W1 (2 objects, rows[0:B]) then of W3 (rows[8:8+B]).
      const int col = ctl[COL];
      dot4(o, m + ACT + (col & 1) * OBJ, 1024, rows + (col < 2 ? 0 : 8));
      if (col == 3) {
        finish_row(rows, m + PROJ, 640, ctl[ROW]);
        finish_row(rows + 8, m + PROJ + 320, 640, ctl[ROW]);
        ctl[COL] = 0;
        ++ctl[ROW];
      } else {
        ctl[COL] = col + 1;
      }
    }
    return;
  default:
    if (i < 5 * B)
      copy(o, m + ACT + i * OBJ, OBJ);
    else
      gemv_obj(o, m + ACT, 2560, m + PROJ, 384, rows, ctl, 5);
    return;
  }
}

[[gnu::noinline]] void output(bfloat16 *o, bfloat16 *m, float *acc, int32_t *ctl, int j) {
  const int k_off = 128 * ctl[CORE];
  for (int i = 0; i < OBJ; i += 16)
    aie::store_v(o + i, aie::zeros<bfloat16, 16>());
  switch (ctl[SEG]) {
  case 1:
    if (ctl[MODE] == MODE_ATTENTION) {
      if (j < B) {
        // First output: causal attention to the batch's own tokens.
        if (j == 0)
          attn_block4(m + PROJ + 128, 384, B, true, m, acc);
        for (int h = 0; h < 2; ++h) {
          const int q = 2 * j + h;
          float *values = acc + SOFT + q * 64;
          scale64(values, reciprocal(acc[16 + q]));
          cast(values, o + h * 64, 64);
        }
      } else {
        copy(m + PROJ + (j - B) * 384 + 128, o, 128);  // [key | value]
      }
    } else if (j < B) {
      copy(m + SH + j * OBJ, o, 128);
    } else {
      copy(m + STATE + ((B - 1) & 1) * OBJ, o, 384);
    }
    return;
  case 3:
    silu_gate(m + PROJ + j * 640, m + PROJ + j * 640 + 320, o, 320);
    return;
  default:  // segments 2 and 4: residual add
    add(m + HIN + j * 1024 + k_off, m + PROJ + j * 384, o, 128);
    return;
  }
}

[[gnu::noinline]] void next(bfloat16 *m, float *acc, int32_t *ctl) {
  int32_t seg = ctl[SEG];
  int32_t n_in = 3 * B + 3, n_out = 0;
  if (seg == 0) {
    for (int t = 0; t < B; ++t)
      norm(m + HIN + t * 1024, m + GAM, m + ACT + t * 1024);
    const int32_t mode = read_i32(m + HDR + 500);
    const int32_t rows = read_i32(m + HDR + 504);
    const bool attention = mode == MODE_ATTENTION;
    ctl[MODE] = mode;
    ctl[ROWS] = rows;
    ctl[PAST] = read_i32(m + AUX + 128);
    ctl[TAIL] = attention ? read_i32(m + AUX + 130) : 1;
    n_in = 2 * rows + ctl[TAIL];
    n_out = attention ? 2 * B : B + 1;
  } else if (seg == 1) {
    n_in = 2 * B + 2 * 128;
    n_out = B;
  } else if (seg == 2) {
    n_in = 2 * B + 2 + 4 * 320;
    n_out = B;
  } else if (seg == 3) {
    n_in = 5 * B + 5 * 128;
    n_out = B;
  }
  zero_f(acc + ROWACC, 16);
  ctl[SEG] = seg == 4 ? 0 : seg + 1;
  ctl[N_IN] = n_in;
  ctl[N_OUT] = n_out;
  ctl[ROW] = 0;
  ctl[COL] = 0;
}

}  // namespace

extern "C" void x8p_do(bfloat16 *obj, bfloat16 *mem, float *acc, int32_t *ctl,
                       int32_t op, int32_t i) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (op == OP_IN)
    input(obj, mem, acc, ctl, i);
  else if (op == OP_OUT)
    output(obj, mem, acc, ctl, i);
  else
    next(mem, acc, ctl);
  aie::set_rounding(saved);
}
