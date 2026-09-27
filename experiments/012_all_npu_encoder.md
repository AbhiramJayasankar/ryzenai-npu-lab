# Experiment 012: the Parakeet TDT 0.6B v2 encoder entirely on the Phoenix NPU (v1)

Follow-up to [experiment 011](011_parakeet_npu.md), where the Ryzen AI route
failed and a hybrid (NPU matmuls, CPU NumPy for the rest) only matched CPU
speed. Here all 24 Conformer layers of the encoder run on the NPU: 16 cores,
one custom IRON core program, no CPU work inside the encoder layers. State as
of 2026-09-27.

For continuing the work, read the [experiment 012 handoff](012_handoff.md).

## Status

**Works and is correct, but is not yet stable.** The encoder produces the
FP32 model's transcripts on the test clips at 2.2 to 2.6 times CPU speed, but
the NPU hangs intermittently (after 2 to 72 good submissions, independent of
the input). On this laptop two of those hangs were followed by Windows blue
screens (driver 32.0.203.280). Updating the NPU driver to 32.0.203.376
(32.0.20101.3760) did not stop the hangs. The WER evaluation on the
210-utterance set was therefore not run. The most likely cause is a
runtime-sequence ordering race (see the handoff); it is being investigated
separately.

## Results (1-block program: up to 128 encoder frames = 10.24 s of audio)

Test clips (test.wav and three LibriSpeech test-clean utterances, 73 to 123
encoder frames), [`012_encoder_test.py`](012_encoder_test.py): subsampling
prefix on the CPU (ONNX Runtime), 24 Conformer layers on the NPU, TDT decoding
on the CPU.

| | Value |
| --- | --- |
| Encoder output vs ONNX Runtime FP32 | cosine 0.99982 to 0.99994, relative RMS 1.1 to 1.9% |
| Transcripts vs FP32 | identical on all 4 clips |
| NPU time per 10 s window | 188 to 196 ms (two submissions of 12 layers) |
| For comparison, same 10 s window | CPU FP32 496 ms, CPU dynamic int8 411 ms, GPU (RTX 4060) 26 ms, exp-011 hybrid 520 ms |
| CPU use during the NPU encoder | the host thread waits; prefix and decoding stay on the CPU |

A 3-block program (384 frames, 30.7 s) gave identical outputs on the short
clips (600 ms per clip, since it always processes 384 frames); it hung on the
30.6 s clip, at a point where the intermittent hang was already occurring.

Layer 0 op by op against a NumPy mirror with BF16 rounding at the same points
([`012_layer_test.py`](012_layer_test.py)): LayerNorm, FFN with Swish, q/k/v
with bias, per-head position scores, attention, output projection, GLU +
depthwise convolution + Swish, and the final LayerNorm all match with cosine
>= 0.99993 and 0.4 to 1.2% relative RMS error.

## Design

* **Grid.** All 16 compute tiles. Weights stream from DDR once per column
  and are broadcast to that column's four cores; activation tiles are
  re-blocked into mmul layout by the memory tiles and broadcast along rows;
  outputs are joined per column in the memory tile.
* **One core program for every phase** ([`kernels/pk/`](kernels/pk)): each
  phase starts with a header object that sets the segment counts of the
  core's loop. Phases: GEMM (aie::mmul 4x8x4 BF16, FP32 accumulation,
  optional Swish or per-column bias), residual add + LayerNorm (single,
  double or final), relative-position attention (online softmax, 16 queries
  per pass, position scores from a per-head GEMM against the constant
  projected position table), and the convolution module (GLU, depthwise
  kernel 9 with batch norm folded, Swish). 22 phases per layer, 529 per
  encoder, issued as a few submissions.
* **Numerics.** BF16 activations between phases, FP32 accumulation inside.
  FP32 products in the vector math are built from BF16 pieces with native
  BF16 x BF16 -> FP32 MACs; exp uses a degree-4 polynomial for 2^f, sigmoid a
  Newton reciprocal (errors 7.7e-5 and 2.7e-4, below BF16 rounding).
  The 1/sqrt(128) attention scale is folded into Wq and the biases on the host.
* **Frame blocks.** Cores always work on 32-frame GEMM tiles and 1024-element
  outputs; a program for 128*n frames runs each GEMM once per 128-frame block
  (weights re-streamed per block, about 15% slower than one big tile at 384
  frames, which did not fit the 64 KB data memory).

## Measured along the way

| Step | Result |
| --- | --- |
| 8 GEMM phases of one layer, one submission | 3.7 ms at 128 frames (830 GMAC/s), 6.3 ms at 256, 9.6 ms at 384 (~970 GMAC/s); the hybrid needed ~9.2 ms per layer at 128 |
| First full 24-layer run | 450 ms per 10 s window, 6 to 16% relative error, transcripts correct |
| FP32 -> BF16 rounding mode set to round-to-nearest-even | error 6-16% -> 1.1-1.9% (the default truncates) |
| exp / sigmoid / Swish from native BF16 MACs instead of emulated FP32 multiplies | 513 / 882 / 933 -> 171 / 257 / 286 cycles per 16 values ([`012_vbench.py`](012_vbench.py)); encoder 450 -> 188 ms |
| Program memory | GEMM step and hot vector routines -O2 (pk_fast.cc, ~7 KB), the rest -Oz (pk_core.cc, ~7 KB); no soft-float or modulo library code (they pushed it over 16 KB) |

## Files

| File | Role |
| --- | --- |
| [`pk_engine.py`](pk_engine.py) | IRON design, data layout, phase classes, static program checker |
| [`pk_model.py`](pk_model.py) | Phase list for 24 layers, weight/position packing, `PkEncoder` |
| [`kernels/pk/`](kernels/pk) | Core program: `pk_core.cc` (-Oz), `pk_fast.cc` (-O2), `pk_math.h`, `pk_common.h`, `pk_vbench.cc` |
| [`012_layer_test.py`](012_layer_test.py) | Op-by-op check of layer 0 |
| [`012_encoder_test.py`](012_encoder_test.py) | Full encoder vs ONNX Runtime FP32 and transcripts |
| [`012_transcribe.py`](012_transcribe.py) | WER on the 210-utterance set (not yet run to completion) |
| [`012_vbench.py`](012_vbench.py) | Single-core timing and accuracy of the vector math |
| [`012_hang_probe.py`](012_hang_probe.py) | Runs the first k phases once (debugging) |

Setup is as in experiment 011 (weights from `011_export_weights.py`,
positions from `011_export_pos.py`, dynamic prefix from `011_split.py`).
