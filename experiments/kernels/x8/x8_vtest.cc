// Single-core test of the FP32 vector helpers used by x8p_core.cc.
// in[0:16] holds FP32 inputs; out receives 16-lane FP32 results per test.
#include "x8_math.h"

namespace {
[[gnu::noinline]] fvec recip12(fvec d) {
  fvec r = aie::sub(aie::broadcast<float, 16>(24.0f / 17.0f), vmul(d, 8.0f / 17.0f));
#pragma clang loop unroll(disable)
  for (int i = 0; i < 3; ++i)
    r = vmul(r, aie::sub(aie::broadcast<float, 16>(2.0f), vmul(d, r)));
  return r;
}
}  // namespace

extern "C" void x8_vtest(const float *in, float *out) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const fvec x = aie::load_v<16>(in);
  const fvec p = aie::max(x, 0.0f);
  const fvec a = exp16(aie::sub(x, p));
  const fvec b = exp16(aie::sub(aie::zeros<float, 16>(), p));
  const fvec d = aie::add(a, b);
  aie::store_v(out + 0, p);
  aie::store_v(out + 16, a);
  aie::store_v(out + 32, b);
  aie::store_v(out + 48, d);
  aie::store_v(out + 64, vmul(d, 8.0f / 17.0f));
  aie::store_v(out + 80, recip12(d));
  aie::store_v(out + 96, vmul(x, x));
  aie::store_v(out + 112, vmul(x, 2.0f));
  aie::set_rounding(saved);
}
