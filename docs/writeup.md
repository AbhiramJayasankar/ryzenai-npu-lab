# Running an LLM and speech recognition on a laptop NPU

A technical write-up of this repository: what was built, the measurements
that drove the design, and what did and did not work. The numbers come from
the experiment documents linked in each section, all measured on one ASUS
Zephyrus G14 with a Ryzen 9 8945HS.

## The problem

The Ryzen 9 8945HS has a first-generation XDNA NPU ("NPU1", Phoenix): a
grid of AI Engine tiles, each a small VLIW vector core with 64 KB of data
memory and 16 KB of program memory, fed by DMA engines from the laptop's
DDR. AMD ships two ways to use it. Neither worked for these models:

* AMD's LLM runtime (ONNX Runtime GenAI) supports only newer chips. Its
  documentation excludes Phoenix and Hawk Point from both NPU-only and hybrid
  LLM execution.
* The Ryzen AI compiler (Quark quantization plus the Vitis AI execution
  provider) handled vision models (experiments 001–004). For the Parakeet
  speech model it placed operators on the NPU only with XINT8 quantization.
  It split every Conformer layer into about ten NPU pieces with CPU work in
  between, ran slower than the CPU, and had 38% relative error after two of
  24 layers ([experiment 011](../experiments/011_parakeet_npu.md)).

So both models here run on custom programs. IRON (part of AMD's open
MLIR-AIE project) describes which tiles run which C++ kernel and how data
moves between DDR, memory tiles and compute tiles. Peano (llvm-aie) compiles
the kernels, and XRT submits the result to the NPU.

## Measure first

The first all-NPU LLM ([experiment 006](../experiments/006_lfm25_full_npu.md))
was correct and slow: ~450 ms per token for a 230M-parameter model that the
CPU runs at ~30 ms. It used one core per layer, about 3,000 small
synchronized transfers per layer and many program switches.

Before redesigning it, a set of probes measured what the hardware can do
([experiment 010](../experiments/010_x8_engine.md)):

* **DDR read bandwidth** was 14 GB/s with 4 DMA streams and ~27 GB/s with
  8 (4 columns × 2 shim channels). 16 streams do not fit the DMA channels.
* **Submitting a program** cost ~0.16 ms through raw XRT and ~1 ms through
  IRON's `iron.jit` wrapper.
* **Switching between compiled programs** cost ~0.5 ms each time.
* **A round trip through DDR** between stages inside one run cost only
  15–20 µs.

Decoding one token reads every weight once: 330 MB for the 14 layers and
134 MB for the tied vocabulary head, in BF16. At 27 GB/s that takes about
17 ms, so 17 ms per token is the floor for BF16 decode on this NPU. The
redesign aimed at that floor.

## The LLM engine

The eight-core engine gives each core one DMA stream and makes every stage
one long transfer:

* **Weights repacked per core.** Core *k* computes output slice *k* of every
  projection. The host packs each core's weight rows in the order it
  consumes them.
* **Partitioning that keeps work local.** Each core owns 128 channels of the
  gated convolution and one KV head of grouped-query attention. A layer then
  needs only four exchanges of full vectors, done through a DDR scratch
  buffer.
* **One submission per token.** All 14 layers, the head and the argmax run
  as one runtime sequence. The embedding lookup is a DMA from a sub-buffer at
  the token's row, and the NPU writes back the next token id.
* **One core program for everything.** A header object in each weight stream
  tells a small C++ state machine whether the next stage is a recurrent
  layer, an attention layer or the head. That avoids program switches. The
  decode program uses 16,228 of the 16,384 bytes of program memory.

The decode time fell in steps:

| Change | ms per token |
| --- | ---: |
| First version | ~450 |
| 8 cores, one stream each, raw XRT | 46 |
| Vectorized RMSNorm, SiLU and convolution | 32.6 |
| One core program for both layer types | 28 |
| Whole token in one submission | **17.2** |

The vectorization step matters because these cores have no scalar
floating-point unit. Scalar float code compiles to software-emulation calls
and cost about 1 ms per layer.

The result is 58 tokens/s with all arithmetic on the NPU. Validation scripts
compare the hidden state, every conv state and every KV cache with a CPU
BF16 reference after each position. Generated tokens match except where the
CPU's top two logits tie exactly in BF16. A second core program processes
prompts four tokens per weight pass at 4.6 ms per token.

**Compared honestly.** The engine beats PyTorch BF16 on 8 CPU threads
(~33 tok/s). llama.cpp on the same CPU is faster: 91 tok/s in F16 and
160 tok/s in Q8_0, because the CPU reads DDR faster than the NPU and
llama.cpp is heavily optimized. Faster NPU decode needs fewer bytes per
token, for example INT8 weights, which changes the model's numerics.

## Speech recognition

Parakeet TDT 0.6B v2 is a FastConformer encoder (24 layers) with a
transducer decoder. The encoder is about 95% of the arithmetic, so it is the
part worth moving.

The first custom attempt ([experiment 011](../experiments/011_parakeet_npu.md))
ran every linear layer as an NPU GEMM and the rest in NumPy. It matched the
FP32 word error rate but took 520 ms per 10 s of audio, no faster than the
CPU (496 ms).

The second ([experiment 012](../experiments/012_all_npu_encoder.md)) runs the
whole encoder on all 16 compute tiles with one core program and 22 phases
per layer: GEMMs on the matrix unit, LayerNorm, relative-position attention
with online softmax, and the convolution module. Weights stream once per
column and are broadcast to that column's four cores; memory tiles re-block
activations into the matrix unit's layout.

Two numerical details decided the result:

* **Rounding.** The first full run had 6–16% relative error. The hardware's
  FP32→BF16 conversion truncates by default. Switching it to
  round-to-nearest-even brought the error to 1.1–1.9%.
* **Building FP32 math from BF16 pieces.** exp, sigmoid and Swish were slow
  because FP32 multiplies are emulated. Splitting values into BF16 pieces
  and using native BF16×BF16→FP32 multiply-accumulates cut them from about
  900 to 260 cycles per 16 values. The encoder went from 450 ms to 188 ms.

On 210 LibriSpeech test-clean utterances the NPU encoder gives a 2.19% WER,
exactly the FP32 model's, and 205 of 210 transcripts are character-identical.
It takes 185 ms per 10 s window: 2.7× faster than the CPU in FP32, and 7×
slower than the RTX 4060.

## The hang

The first full encoder hung the NPU intermittently, after anywhere from 2 to
72 good runs. On this driver a hang makes Windows reset the device, and twice
that ended in a blue screen. So the investigation had to happen offline.

Decoding the compiled instruction binaries showed the cause. Attention steps
started up to nine DMA transfers on one shim channel before waiting for any.
Each channel's task queue holds four entries, a figure confirmed by AMD's
aie-rt driver source and its AM020 architecture manual. The MLIR-AIE guide
describes what happens past that: the push is dropped without an error, and
the core waiting for that data waits forever. Timing decided whether the
queue had drained in time, which explains why identical runs sometimes
passed.

The fix changed the runtime sequence: issue output drains before input
fills, and start at most four transfers per channel between awaited
completions. `pk_engine.check()` now verifies those rules for every task
group of every program before any XRT import, and offline regression tests
cover the old failing schedule. Since the fix, 566 submissions, the 2- and
3-block programs and the full 210-utterance evaluation have run with no
hang, and repeated runs are bit-identical.

## What the NPU is good for

On this laptop the NPU is not the fastest place to run either model. The CPU
with llama.cpp wins for the LLM, and the GPU wins for speech. In exchange,
the NPU leaves both free. During the experiment 011 hybrid, other work on
the laptop kept 87–96% of its idle speed, against 73–75% while the CPU ran
the FP32 encoder. A battery measurement in experiment 006 found ~2–3 W of
extra whole-laptop power for the NPU against 6 W for the CPU and 9 W for the
GPU, though that early version was slow enough to use more energy per
token.

The project also produced a practical app: [NPU Dictate](../apps/npu_dictate/README.md),
push-to-talk dictation for Windows that runs speech recognition on the NPU
and inserts the text into any window.

## What is left

* INT8 weights for the LLM, to get under the BF16 bandwidth floor.
* Power measured with an external meter instead of battery discharge.
* Programs for each 128-frame length step, so speech cost scales with the
  utterance instead of the program size.
* Porting to XDNA 2 (Ryzen AI 300), which has a larger array of tiles.
