// Experiment 015: add one to every value, then overwrite element 0 with the
// core's hardware ID so the host can see which physical tile ran the code.
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

void hc_id(int32_t *in, int32_t *out, int32_t n) {
  for (int32_t i = 0; i < n; i++)
    out[i] = in[i] + 1;
  out[0] = get_coreid();
}

}
