// Single-core timing/accuracy bench of the engine's vector math (experiment 012).
// in[0:1024] FP32 inputs, in[1024] = routine, in[1025] = repeats; out[0:1024].
#include "pk_fast.cc"

extern "C" void pk_vbench(float *in, float *out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int op = static_cast<int>(in[1024]);
  const int reps = static_cast<int>(in[1025]);
  for (int r = 0; r < reps; ++r)
    for (int v = 0; v < 1024; v += 16) {
      const fvec x = aie::load_v<16>(in + v);
      fvec y;
      if (op == 0)
        y = exp16(aie::min(x, aie::zeros<float, 16>()));
      else if (op == 1)
        y = sigmoid(x);
      else if (op == 2)
        y = swish(x);
      else if (op == 3)
        y = vmul(x, x);
      else
        y = to_f32(to_bf16(x));
      aie::store_v(out + v, y);
    }
}
