// One Phoenix compute tile's share of LFM2.5-230M: every layer kind and the
// vocabulary head, as a state machine driven object by object.
//
// The IRON core loop runs five segments per stream. In each segment it feeds
// ctl[N_IN] input objects to x8_do(op=IN), then fills ctl[N_OUT] output
// objects with x8_do(op=OUT), then calls x8_do(op=NEXT), which finishes the
// segment and sets the next segment's counts. A header object inside the
// stream selects recurrent layer, attention layer or head work.
//
// Segments of a layer stream (core k owns output slice k):
//   0: input vector (2 objects), token aux (1), RMSNorm gamma (2), header (1)
//   1: projection rows (2 objects per row) then tail objects: conv state
//      (recurrent) or KV cache blocks (attention); outputs: slice output and
//      new state / KV entry
//   2: gathered 1024-vector (2) + 128 output-projection rows; output: slice
//   3: h2 (2) + FFN gamma (2) + 320 interleaved W1/W3 rows (4 objects each)
//   4: gated 2560-vector (5) + 128 W2 rows (5 objects each)
// Head stream: segment 0 as above, 1: 8192 vocabulary rows (2 objects
// each) -> candidate, 2: on core 0 only, pick among the 8 candidates.
//
// Numerics follow the experiment 006 kernels: BF16 rounding at the model's
// boundaries with round-to-nearest-even and FP32 accumulation.
#include "x8_math.h"

namespace {

constexpr int VEC = 2560;
constexpr int HIN = 0, ACT = VEC, GAM = 2 * VEC, WORK = 3 * VEC, SH = 4 * VEC;
// work layout: [0:384) projection outputs, [512:1024) header, [1024:1536) aux.
constexpr int HDR = 512, AUX = 1024;
constexpr int MODE_RECURRENT = 0, MODE_ATTENTION = 1, MODE_HEAD = 2;
constexpr int OP_IN = 0, OP_OUT = 1, OP_NEXT = 2;
enum Ctl { N_IN, N_OUT, SEG, MODE, ROWS, TAIL, BASE, CORE, PICK, ROW, COL };
constexpr int SILU_E0 = 112, SILU_E1 = 134;
constexpr int SILU_HALF = (SILU_E1 - SILU_E0) * 128;

// out[i] = bf16(bf16(silu(g[i])) * u[i]), silu from the host-computed table.
[[gnu::noinline]] void silu_gate(const bfloat16 *g, const bfloat16 *u, bfloat16 *out,
               const bfloat16 *table) {
  const uint16_t *in_bits = reinterpret_cast<const uint16_t *>(g);
  const uint16_t *tab = reinterpret_cast<const uint16_t *>(table);
  uint16_t *out_bits = reinterpret_cast<uint16_t *>(out);
  for (int i = 0; i < 320; ++i) {
    const uint16_t b = in_bits[i];
    const int e = (b >> 7) & 0xFF;
    const bool negative = b & 0x8000;
    uint16_t s;
    if (e < SILU_E0)
      s = e > 1 ? static_cast<uint16_t>(b - 0x80) : b;  // |x| < 2^-15: x/2
    else if (e >= SILU_E1)
      s = negative ? 0x8000 : b;  // |x| >= 128
    else
      s = tab[(negative ? SILU_HALF : 0) + ((e - SILU_E0) << 7) + (b & 0x7F)];
    out_bits[i] = s;
  }
  for (int i = 0; i < 320; i += 16)
    aie::store_v(out + i, to_bf16(aie::mul(aie::load_v<16>(out + i),
                                           aie::load_v<16>(u + i))));
}

// Gated depthwise convolution, planar state/weights [t*128 + ch].
[[gnu::noinline]] void conv_gate(const bfloat16 *work, const bfloat16 *state, bfloat16 *out) {
  const bvec one = ones();
  const bfloat16 *weight = work + HDR;
  bfloat16 *next_state = out + OBJ;
  for (int i = 0; i < 128; i += 16) {
    const bvec B = aie::load_v<16>(work + i);
    const bvec C = aie::load_v<16>(work + 128 + i);
    const bvec x = aie::load_v<16>(work + 256 + i);
    const bvec Bx = to_bf16(aie::mul(B, x));
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
    aie::store_v(out + i, to_bf16(aie::mul(C, to_bf16(sum))));
  }
}

// ------------------------------------------------------------- attention
// st[200] holds the past length (see x8_math.h for the softmax state).

[[gnu::noinline]] void attn_prep(bfloat16 *work, float *st) {
  const bfloat16 *cos = work + AUX;
  const bfloat16 *sin = work + AUX + 64;
  norm_rope(work, work + HDR, cos, sin);
  norm_rope(work + 64, work + HDR, cos, sin);
  norm_rope(work + 128, work + HDR + 64, cos, sin);
  for (int h = 0; h < 2; ++h) {
    float *state = st + h * 80;
    state[0] = -1.0e30f;
    state[1] = 0.0f;
    zero_f(state + 16, 64);
  }
  st[200] = static_cast<float>(read_i32(work + AUX + 128));
}

[[gnu::noinline]] void attn_block(const bfloat16 *kv, const bfloat16 *work, float *st,
                                  int block) {
  const int n = static_cast<int>(st[200]) - block * 4;
  if (n > 0)
    attn_positions(kv, n < 4 ? n : 4, work, st);
}

// Include the current position, write the 2-head context to out[0:128] and
// this position's [key | value] to out[512:640].
[[gnu::noinline]] void attn_final(const bfloat16 *work, float *st, bfloat16 *out) {
  attn_positions(work + 128, 1, work, st);  // work[128:256] = [key | value]
  for (int h = 0; h < 2; ++h) {
    float *state = st + h * 80;
    scale64(state + 16, reciprocal(state[1]));
    cast(state + 16, out + h * 64, 64);
  }
  copy(work + 128, out + OBJ, 128);
}

// ------------------------------------------------------------- head
// st[0] best BF16 logit, st[1] its row, st[3] first-half partial dot.

[[gnu::noinline]] void vscore(const bfloat16 *w, const bfloat16 *act, float *st, int row, int half) {
  const float dot = dot512(w, act + half * OBJ);
  if (half == 0) {
    st[3] = dot;
    return;
  }
  const float score = static_cast<float>(static_cast<bfloat16>(st[3] + dot));
  if (score > st[0]) {  // strict: the earliest maximum wins, like argmax
    st[0] = score;
    st[1] = static_cast<float>(row);
  }
}

[[gnu::noinline]] void pack_candidate(const float *st, bfloat16 *out) {
  const float score = st[0];
  const int32_t row = static_cast<int32_t>(st[1]);
  __builtin_memcpy(out, &score, sizeof(score));
  __builtin_memcpy(out + 2, &row, sizeof(row));
}

// Candidate c at cand[c*64]; ties keep the earlier core (lower rows).
[[gnu::noinline]] void pick(const bfloat16 *cand, bfloat16 *out) {
  float best;
  int32_t best_row;
  __builtin_memcpy(&best, cand, sizeof(best));
  __builtin_memcpy(&best_row, cand + 2, sizeof(best_row));
  for (int c = 1; c < 8; ++c) {
    float score;
    int32_t row;
    __builtin_memcpy(&score, cand + c * 64, sizeof(score));
    __builtin_memcpy(&row, cand + c * 64 + 2, sizeof(row));
    if (score > best) {
      best = score;
      best_row = row;
    }
  }
  __builtin_memcpy(out, &best_row, sizeof(best_row));
}

// ------------------------------------------------------------- segments

// One weight object of a row-major GEMV whose rows span `cols` objects.
[[gnu::noinline]] void gemv_obj(const bfloat16 *w, const bfloat16 *x, float *acc, int32_t *ctl, int cols) {
  const int col = ctl[COL];
  acc[ctl[ROW]] += dot512(w, x + col * OBJ);
  if (col + 1 == cols) {
    ctl[COL] = 0;
    ++ctl[ROW];
  } else {
    ctl[COL] = col + 1;
  }
}

[[gnu::noinline]] void input(const bfloat16 *o, bfloat16 *m, float *acc, int32_t *ctl, int i) {
  bfloat16 *work = m + WORK;
  switch (ctl[SEG]) {
  case 0:
    if (i < 2)
      copy(o, m + HIN + i * OBJ, OBJ);
    else if (i == 2)
      copy(o, work + AUX, OBJ);
    else if (i < 5)
      copy(o, m + GAM + (i - 3) * OBJ, OBJ);
    else
      copy(o, work + HDR, OBJ);
    return;
  case 1:
    if (ctl[MODE] == MODE_HEAD) {
      vscore(o, m + ACT, acc, ctl[BASE] + (i >> 1), i & 1);
      return;
    }
    if (i < 2 * ctl[ROWS]) {
      gemv_obj(o, m + ACT, acc, ctl, 2);
      return;
    }
    if (i == 2 * ctl[ROWS]) {  // first tail object: projections are complete
      cast(acc, work, 384);
      if (ctl[MODE] == MODE_ATTENTION)
        attn_prep(work, acc);
    }
    if (ctl[MODE] == MODE_ATTENTION)
      attn_block(o, work, acc, i - 2 * ctl[ROWS]);
    else
      conv_gate(work, o, m + SH);
    return;
  case 2:
    if (ctl[MODE] == MODE_HEAD)
      pick(o, m + SH);
    else if (i < 2)
      copy(o, m + ACT + i * OBJ, OBJ);
    else
      gemv_obj(o, m + ACT, acc, ctl, 2);
    return;
  case 3:
    if (i < 2) {
      copy(o, m + HIN + i * OBJ, OBJ);
    } else if (i < 4) {
      copy(o, m + GAM + (i - 2) * OBJ, OBJ);
    } else {
      if (i == 4)
        norm(m + HIN, m + GAM, m + ACT);
      // Row r of W1 (2 objects) then row r of W3 (2 objects).
      const int col = ctl[COL];
      acc[ctl[ROW] + (col < 2 ? 0 : 320)] += dot512(o, m + ACT + (col & 1) * OBJ);
      if (col == 3) {
        ctl[COL] = 0;
        ++ctl[ROW];
      } else {
        ctl[COL] = col + 1;
      }
    }
    return;
  default:
    if (i < 5)
      copy(o, m + ACT + i * OBJ, OBJ);
    else
      gemv_obj(o, m + ACT, acc, ctl, 5);
    return;
  }
}

[[gnu::noinline]] void output(bfloat16 *o, bfloat16 *m, float *acc, const bfloat16 *table,
            int32_t *ctl, int j) {
  bfloat16 *work = m + WORK;
  bfloat16 *sh = m + SH;
  const int k_off = 128 * ctl[CORE];
  for (int i = 0; i < OBJ; i += 16)
    aie::store_v(o + i, aie::zeros<bfloat16, 16>());
  switch (ctl[SEG]) {
  case 1:
    if (ctl[MODE] == MODE_HEAD) {
      pack_candidate(acc, o);
      return;
    }
    if (j == 0 && ctl[MODE] == MODE_ATTENTION)
      attn_final(work, acc, sh);
    copy(sh + j * OBJ, o, j == 0 ? 128 : OBJ);
    return;
  case 2:
    if (ctl[MODE] == MODE_HEAD) {
      copy(sh, o, 16);
      return;
    }
    cast(acc, work, 128);
    add(m + HIN + k_off, work, sh, 128);
    copy(sh, o, 128);
    return;
  case 3:
    cast(acc, work, 640);
    silu_gate(work, work + 320, sh, table);
    copy(sh, o, 320);
    return;
  default:
    cast(acc, work, 128);
    add(m + HIN + k_off, work, sh, 128);
    copy(sh, o, 128);
    return;
  }
}

// Per-segment input/output object counts of a layer stream (segments 1-4;
// segment 1's input count is filled in from the header).
constexpr int32_t LAYER_IN[5] = {6, 0, 2 + 2 * 128, 4 + 4 * 320, 5 + 5 * 128};
constexpr int32_t LAYER_OUT[5] = {0, 2, 1, 1, 1};
constexpr int32_t ACC_ZERO[5] = {0, 384, 128, 640, 128};

[[gnu::noinline]] void begin_stream(bfloat16 *m, float *acc, int32_t *ctl) {
  bfloat16 *work = m + WORK;
  norm(m + HIN, m + GAM, m + ACT);
  const int32_t mode = read_i32(work + HDR + 500);
  const int32_t rows = read_i32(work + HDR + 504);
  ctl[MODE] = mode;
  ctl[ROWS] = rows;
  ctl[BASE] = read_i32(work + HDR + 508);
  int32_t tail = 1;
  if (mode == MODE_ATTENTION)
    tail = read_i32(work + AUX + 130);
  acc[0] = -3.4028235e38f;  // head scan state; layers zero acc below
  acc[1] = 0.0f;
  ctl[TAIL] = tail;
}

[[gnu::noinline]] void next(bfloat16 *m, float *acc, int32_t *ctl) {
  int32_t seg = ctl[SEG];
  if (seg == 0)
    begin_stream(m, acc, ctl);
  seg = seg == 4 ? 0 : seg + 1;
  const bool head = ctl[MODE] == MODE_HEAD;
  int32_t n_in = LAYER_IN[seg];
  int32_t n_out = LAYER_OUT[seg];
  if (seg == 1) {
    n_in = 2 * ctl[ROWS] + (head ? 0 : ctl[TAIL]);
    n_out = head ? 1 : 2;
  } else if (head && seg == 2) {
    n_in = ctl[PICK];
    n_out = ctl[PICK];
  } else if (head && seg > 2) {
    n_in = 0;
    n_out = 0;
  }
  if (!(head && seg == 1))
    zero_f(acc, ACC_ZERO[seg]);
  ctl[SEG] = seg;
  ctl[N_IN] = n_in;
  ctl[N_OUT] = n_out;
  ctl[ROW] = 0;
  ctl[COL] = 0;
}

}  // namespace

extern "C" void x8_do(bfloat16 *obj, bfloat16 *mem, float *acc,
                      const bfloat16 *table, int32_t *ctl, int32_t op, int32_t i) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (op == OP_IN)
    input(obj, mem, acc, ctl, i);
  else if (op == OP_OUT)
    output(obj, mem, acc, table, ctl, i);
  else
    next(mem, acc, ctl);
  aie::set_rounding(saved);
}
