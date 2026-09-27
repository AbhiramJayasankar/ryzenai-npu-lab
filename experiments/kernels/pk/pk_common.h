// Shared definitions of the Parakeet encoder core program (experiment 012).
#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef PK_TPAD
#define PK_TPAD 128
#endif
#ifndef PK_QB
#define PK_QB 16
#endif

namespace pk {

constexpr int F = 1;               // 32-frame GEMM tiles per core and frame block
constexpr int TR = 32 * F;         // frames per core in a GEMM block
constexpr int TPAD = PK_TPAD;      // encoder frames handled by the program (128 per block)
constexpr int WOBJ = 2048;         // W stream object (BF16 elements)
constexpr int COBJ = 1024 * F;     // output object per core (BF16 elements)
constexpr int QB = PK_QB;          // attention queries per core per pass
constexpr int D = 1024;

// Control words shared by the IRON loop (first five) and the kernels.
enum Ctl {
  NPRO, NB, NWX, NW, NOUT,          // segment counts, read by the IRON loop
  OP, ROW, COL, EPI, CHUNK, BLK, MODE, VALID, PASS, PPH, NQ, NBD, BDL, TCNT, SCALE, RPOS, OROW, NCH,
  CTL_LEN = 32
};
enum Op { OP_GEMM = 0, OP_LN = 1, OP_ATT = 2, OP_CONV = 3 };
enum Epi { EPI_SWISH = 1, EPI_BIAS = 2 };
enum LnMode { LN_SINGLE = 0, LN_DOUBLE = 1, LN_FINAL = 2 };

using bvec = aie::vector<bfloat16, 16>;
using fvec = aie::vector<float, 16>;
using ivec = aie::vector<int32_t, 16>;
using facc = aie::accum<accfloat, 16>;

}  // namespace pk
