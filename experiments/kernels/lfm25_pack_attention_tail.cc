// Join hidden state with NPU attention context without a host round trip.
#include <aie_api/aie.hpp>

extern "C" void lfm25_pack_attention_tail(const bfloat16 *hidden,
                                            const bfloat16 *context,
                                            bfloat16 *packed) {
  for (int i = 0; i < 1024; ++i) {
    packed[i] = hidden[i];
    packed[1024 + i] = context[i];
  }
}
