# Experiment 014: three leads for more NPU speed, checked

## Question

Experiment [013](013_raw_capabilities.md) left three untested ideas for
making the two engines faster (the LFM2.5 LLM engine of experiment
[010](010_x8_engine.md) and the Parakeet encoder of experiment
[012](012_all_npu_encoder.md)):

1. **NPU power mode.** Every earlier measurement ran in the "Default" mode,
   while AMD's own `xrt-smi validate` switches to "Performance".
2. **A tuned matrix multiply.** Taka et al. report 1.57 TMAC/s BF16 and
   3.4 TMAC/s int8 on a Phoenix NPU, against 0.83-0.97 TMAC/s for the
   encoder's GEMM phases.
3. **8-bit weights.** int8 matrix multiply ran at 1.46 TMAC/s against 0.79
   for BF16, and int8 weights halve the bytes read from DDR.

Does each one give real speed?

## Short answer

| Lead | Verdict | In one line |
| --- | --- | --- |
| 1. Power mode | **not real** | Performance and Turbo leave the clock, the 27-29 GB/s DDR ceiling, the 17 ms LLM token and the encoder time where they were; Turbo only trims the command round trip from 88 to 80 µs |
| 2. Tuned GEMM (Taka et al.) | **not real as published; a smaller real lead found** | The paper uses the same 16 cores, BF16 partial sums and 4K-sized matrices. The encoder's GEMMs are limited by DDR because activations are re-read for every 32 output columns; reusing them could save an estimated 9-14% of the encoder time (not built: it changes the core program of all 529 phases) |
| 3. 8-bit weights | **real for accuracy, unproven for speed** | Parakeet: int8 weights (and even int8 weights × int8 activations) keep the WER at 2.15-2.26% against FP32's 2.19%. LFM2.5: int8 weights with a scale per 32 or 64 weights stay within BF16's own rounding noise. Estimated speed: LLM 17.2 → about 9.5-12 ms per token, encoder GEMMs 3.7 → 1.3-3 ms per layer. Needs new NPU kernels, not built |

The run-to-run spread on this laptop is about ±5% (two Default runs of the
same programs), so differences below that are not counted as gains.

## Machine and state

Same laptop and software as experiment 013: Ryzen 9 8945HS (Hawk Point),
NPU driver 32.0.20101.3760, firmware 1.5.5.391, XRT 2.21.0, IRON 1.4.3.
Runs on 2026-11-16 between 22:00 and 23:00 IST, on AC power (battery at 79%),
Windows Balanced plan. The CI runner containers were up but idle (0-2% CPU),
total CPU load about 3%, GPU idle. Free RAM was 1.3-3.3 GB (other
applications held most of it). The NPU Dictate tray app was running but
held no hardware context (`xrt-smi examine -r aie-partitions`: "No hardware
contexts running on device"). The CPU-only work of lead 3 ran after the NPU
timing, never at the same time.

## Lead 1: NPU power mode

### What was tried

`xrt-smi configure --help` lists five modes: `default`, `powersaver`,
`balanced`, `performance`, `turbo`. The current mode is shown by
`xrt-smi examine -r platform`. At the start it was **Default**.

[`014_power_modes.ps1`](014_power_modes.ps1) runs twelve unchanged programs
that had all run cleanly before, each in a fresh PowerShell process through
`scripts/run_npu_timeout.ps1`, and stops at the first error. It ran three
times: in Default, in Performance, and in Default again. The second Default
run shows how much the numbers move between runs of the same mode. The
programs:

* the experiment 010 LLM engine: `010_x8_validate.py --check` (the health
  check, with the fixture comparison) and `--repeats 3`, and
  `010_context_bench.py` (decode time from 32 to 4090 tokens of context);
* DDR bandwidth: the 010 bandwidth probe with 4 and 8 streams (called
  through `iron.jit`, so it includes about 0.75 ms of IRON overhead per call),
  and AMD's DMA-only 64 MB copy and one-core 16 MB copy from experiment 013
  (direct pyxrt, no such overhead);
* compute: AMD's 16-core BF16 → FP32 and int8 matrix multiplies at 1024
  (experiment 013), and the single-core vector-math timings of
  `012_vbench.py`, which serve as a clock check (they report nanoseconds, so
  a faster clock would show as fewer "cycles");
* `013_call_overhead.py direct` (submit and wait for a tiny program);
* the Parakeet encoder: `012_encoder_test.py 1 24 --clip 0 --repeats 3`
  (1-block program, 24 layers, error against ONNX Runtime FP32 and the
  transcript).

### Numbers

Medians; "LLM decode" is the median step over the 23-position fixture.

| | Default | Performance | Default again | Turbo |
| --- | ---: | ---: | ---: | ---: |
| LLM decode, ms per token (`--check`) | 17.00 | 18.36 | 17.33 | 18.03 |
| LLM decode, ms per token (3 repeats) | 16.95 / 16.99 / 16.97 | 18.19 / 18.41 / 18.37 | 18.18 / 17.92 / 18.17 | 17.62 / 17.61 / 17.68 |
| LLM decode at 32 / 1024 / 4090 tokens of context, ms | 17.3 / 23.7 / 38.5 | 18.0 / 23.8 / 41.4 | 17.4 / 24.3 / 37.6 | 17.2 / 21.8 / 37.2 |
| LLM fixture: max hidden / conv state / KV error | 0.25 / 0.094 / 0.156 | 0.25 / 0.094 / 0.156 | 0.25 / 0.094 / 0.156 | 0.25 / 0.094 / 0.156 |
| DMA-only copy 64 MB, ms (GB/s read + write) | 4.55 (29.5) | 4.91 (27.3) | 4.79 (28.0) | 4.61 (29.1) |
| 8-stream read probe, GB/s (with `iron.jit` overhead) | 17.4 | 17.5 | 17.4 | 18.7 |
| 4-stream read probe, GB/s (with `iron.jit` overhead) | 13.8 | 13.5 | 12.6 | 14.6 |
| One stream through one core, GB/s | 10.4 | 10.8 | 10.7 | 11.1 |
| BF16 → FP32 matmul 1024, ms | 1.38 | 1.27 | 1.33 | 1.24 |
| int8 matmul 1024, ms | 0.69 | 0.71 | 0.72 | 0.69 |
| exp16 / FP32 multiply, ns per 16 values (one core) | 171 / 65 | 172 / 64 | 169 / 60 | 170 / 61 |
| Tiny call: submit + wait, µs (wait alone) | 122 (95) | 130 (99) | 115 (88) | 108 (80) |
| Parakeet 1-block encoder, ms per run (98-frame clip) | 200 / 200 / 193 | 200 / 198 / 199 | 205 / 201 / 197 | 197 / 193 / 193 |
| Encoder cosine vs FP32 / transcript | 0.999939 / same | 0.999939 / same | 0.999939 / same | 0.999939 / same |

Every correctness check passed in every mode, with identical errors. The
LLM token mismatch at prompt position 7 is the known BF16 tie in the
reference (experiment 010).

### What it shows

* **Performance mode changes nothing measurable.** Every difference between
  Default and Performance is inside the spread between the two Default runs
  (LLM decode 17.0 to 18.2 ms, DMA copy 28.0 to 29.5 GB/s, BF16 matmul 1.33 to
  1.38 ms). The single-core vector timings are the same to within 5%, so the
  core clock did not change. The ~27-29 GB/s DDR ceiling and the 17 ms LLM
  token stay where they were.
* **Turbo needs administrator rights.** `xrt-smi configure --pmode turbo`
  from a normal shell printed
  `Failed to escape (0xc0000001): A device attached to the system is not functioning`,
  after which `examine` reported "Turbo" anyway. The next request, back to
  Default, failed with the same message. Switching to Performance and then to
  Default worked, and the device stayed OK; the repeat Default batch ran
  cleanly. The owner then set Turbo from an administrator shell and the same
  batch ran in Turbo (22:22 IST), with no error.
* **Turbo does not raise the clock either.** The single-core vector timings
  are unchanged (170 ns against 169-171), and the DDR ceiling stays put (DMA
  copy 29.1 GB/s against 28.0-29.5 in Default). Turbo's only consistent
  differences are small. The command round trip (`wait` for a tiny program) is
  80 µs against 88. That helps the probes called through `iron.jit` (18.7 against
  17.4 GB/s, from the shorter fixed cost) and is worth about 0.01 ms on a 17 ms
  token. The 16-core BF16 matmul is 1.24 ms against 1.33-1.38, long-context
  decode is 21.8 ms at 1024 tokens against 23.7-24.3, and the encoder is
  193-197 ms against 193-205. Those are 5-10% at most and inside, or at the
  edge of, the spread between the two Default runs. Short-context decode
  (17.6 ms) is no faster than Default.
* Taka et al. ran their GEMM results in Turbo mode ("NPUs are configured at
  their maximum performance level (i.e., turbo mode)") and report a 1 GHz
  clock for Phoenix. On this Hawk Point laptop Turbo does not change the
  vector timings, so their numbers are not explained by a power mode that
  these engines lack.
* The NPU was returned to **Default**, the mode found at the start
  (the owner switched it back from an administrator shell; `xrt-smi examine -r platform` at 22:27 IST: `Power Mode : Default`).

**Making a mode the default for the engines** would only need
`xrt-smi configure --pmode <mode>` once per boot (from an administrator
shell for Turbo). The engines' code does not set it, and nothing here was
changed. Since Turbo gives no
clear gain, the extra power and heat it may cost (not measured here) are not
worth it.

**Verdict: not real.** Neither Performance nor Turbo moves the DDR
ceiling, the 17 ms LLM token or the encoder time beyond run-to-run noise.
Turbo shaves about 8 µs off each command.

## Lead 2: the tuned matrix multiply

### What the paper does

E. Taka et al., "Striking the Balance: GEMM Performance Optimization Across
Generations of Ryzen AI NPUs" ([arXiv:2512.13282](https://arxiv.org/abs/2512.13282)),
measured on a Ryzen 9 7940HS (Phoenix) under Ubuntu with XRT and IRON. No
code is published with the paper. What matters for this repo:

| | Taka et al. (Phoenix) | This repo's encoder GEMM (`pk_engine.Gemm`) |
| --- | --- | --- |
| Cores | **16** (4 × 4). They leave the fifth column unused because it has no shim tile. | 16 (4 × 4) |
| Mapping | output-stationary, M and N split across cores, K summed in time on each core | the same: each core owns 32 frames × 32 output columns, K summed in time |
| Tile per core (m × k × n) | BF16: 96 × 56 × 96 (balanced), 64 × 104 × 64 (single-core best, 112.6 MACs per cycle) | 32 × 64 × 32 (32 frames, 64 K rows, 32 output columns) |
| Output and accumulation | BF16 output; the kernel reloads the BF16 partial results for every K tile and stores them back, so the running sum is rounded to BF16 every 56 values of K | FP32 accumulator kept on the core for the whole K, rounded to BF16 once |
| Memory tiles | A strips of up to 224 K values (BF16) held in each column's memory tile and broadcast along rows; B broadcast within a column; outputs gathered in the memory tile | X (activations) re-blocked and broadcast along rows; W broadcast within a column; outputs joined per column. Each memory tile holds only depth-2 object pools, nothing is kept for reuse |
| Matrix sizes | swept up to 8K; best results at about 4K × 4K × 4K | M = 128 frames per block (up to 384), K = 1024 or 4096, N = 1024 to 4096 |
| DDR bandwidth | measured "∼15 GB/s" | 27-29 GB/s measured here |
| Power mode, clock | Turbo, 1 GHz | Default; clock not published (≥ 1.33 GHz inferred in 013) |
| Result | BF16 3.12 TOPS = 1.56 TMAC/s at 4224 × 4032 × 4224; int8 6.52 TOPS = 3.26 TMAC/s | 0.83 TMAC/s (128 frames) to 0.97 (384) |

Experiment 013 called this a "tuned 20-core design". That was wrong: the
paper maps the GEMM onto the same 16 cores this repo uses. The hidden column 0
plays no part in their result.

### Why the encoder's GEMM runs at 0.83-0.97 TMAC/s

Counting the bytes read from DDR explains the encoder's number:

* Each column's shim streams that column's quarter of the weights (W), once per
  128-frame block. Each row's shim streams the activations (X) of that
  row's 32 frames, broadcast to the row's four cores, and it streams them
  again for every chunk of 32 output columns (`Gemm.groups`: the X fill
  repeats `chunks()` times). Per GEMM, the X bytes read therefore equal the
  W bytes: 32 frames × K × 2 bytes × (N / 128 chunks) per row, × 4 rows = 2·K·N bytes.
* The 8 weight GEMMs of a layer hold 24.1 M weights = 48 MB in BF16. At 128
  frames the layer reads **96 MB**. 96 MB in the measured 3.7 ms is **26 GB/s**.
  At 256 and 384 frames (weights re-streamed per block), 193 MB in 6.3 ms and
  289 MB in 9.6 ms are 30-31 GB/s.

That is the NPU's DDR read ceiling: experiment 010 measured 27.5 GB/s for
eight read streams in one submission and about 30 GB/s at the margin. So
the GEMM phases are limited by the bytes they read from DDR, and the TMAC/s
figure follows from those bytes, not from the compute.

The next limit is the stream into each core. A core receives a 4 KB W object
(64 × 32 BF16) on one input channel and a 4 KB X tile (32 × 64) on the other,
and does 65,536 multiply-accumulates with them. A stream port carries 4 bytes
per cycle, so receiving one object takes 1,024 cycles, while the matrix unit
(128 BF16 MACs per cycle) needs 512. Each core can therefore reach at most
64 MACs per cycle, 50% of its peak: 1.0 TMAC/s for 16 cores at 1 GHz, 1.36
at 1.33 GHz. For one layer at 128 frames (3.1 G multiply-accumulates) that is
2.3-3.0 ms against the 3.7 ms measured.

### What could apply here

| Difference | Applies to the encoder? | Effect |
| --- | --- | --- |
| Larger per-core tiles (96 × 96 outputs) | Only partly. m is the frames per core: 128 frames / 4 rows = 32. Bigger n needs bigger W objects or several W objects per X tile, and FP32 accumulators (needed to keep WER equal to FP32) take twice the space of their BF16 ones. A 32 × 64 FP32 tile (8 KB) fits the existing scratch buffer | raises MACs per streamed byte; see the next row |
| Reusing each X tile for more output columns (their A strips in the memory tile) | Yes, and it is the main lever. Reusing an X tile for two 32-column chunks halves the X bytes. Holding a whole 32-frame X strip (64-256 KB) in the memory tile and replaying it removes the re-reads entirely | DDR bytes per layer at 128 frames: 96 → 72 MB (two chunks) or ~50 MB (memory tile). The per-core stream limit (2.3-3.0 ms) then decides |
| BF16 partial sums rounded every K tile | No. Experiment 013 measured 3.1% error at K = 1024 for AMD's BF16-output setting, and the encoder needs FP32 sums to keep its WER equal to FP32 | none (not taken over) |
| Turbo mode | Unknown (lead 1: needs administrator rights) | unknown |
| 4K-sized matrices | No. The encoder has 128-384 frames per call | Their large M keeps reuse high; at M = 128, the weights can only be reused across 128 frames |
| Fifth column | No: their design does not use it either | none |

### Estimate for the encoder

The 8 weight GEMMs take 24 × 3.7 = 89 ms of the encoder's 185 ms per
10 s window (the rest is attention, position scores, LayerNorm and the
convolution module). If X were reused across two chunks, a layer would read 72 MB
(2.6 ms at 27.5 GB/s) and the per-core stream limit would give 2.3-3.0 ms.
A layer's GEMMs would then take about 2.6-3.0 ms instead of 3.7, saving
**17-26 ms per window (9-14%)**, so the encoder would take about 160-168 ms. Holding X in the
memory tile saves little more, because the stream into each core becomes
the limit. Getting past that limit needs more frames per core, and 32 frames is all a
128-frame block gives 4 rows. The paper's 1.56 TMAC/s is not reachable for
these shapes with FP32 accumulation.

### What it would take, and why it was not prototyped

The change is not contained. Reusing X means a new segment in the core
loop that every one of the 529 phases runs (`pk_engine._core`: hold one X
object while consuming two W objects into two accumulators). It also needs a new
GEMM step entry, an epilogue for two output tiles, a new weight packing
order, new runtime-sequence fills, and new counts in `check()`. The memory-tile
variant is a new FIFO topology. Either one is a new NPU design under the
safety rules, and the whole encoder would have to be validated again
(correctness, a soak of over 100 runs, the WER evaluation) for a 9-14%
gain. Per the task's rule (prototype only if worthwhile and contained), no
prototype was built.

**Verdict: the paper's design is not a real lead for this encoder as
published** (BF16 partial sums, large matrices). The analysis did find a
real but modest one: the encoder's GEMMs are limited by DDR because X is
re-read for every 32-column chunk. Reusing X could save about 9-14% of the
encoder time. That is a reasoned estimate; it was not measured.

## Lead 3: 8-bit weights

All of this lead ran on the CPU. No NPU program was built or run.

### Parakeet encoder

**What was tried.** [`014_int8_parakeet.py`](014_int8_parakeet.py) runs the
24 Conformer layers in NumPy (experiment 011's `conformer_layer`) on the
210-utterance LibriSpeech set (21,716 encoder frames). The ten linear layers
of every block are computed in each of the formats below. Everything else
(LayerNorm, attention scores and softmax, depthwise convolution, Swish, GLU,
residuals, position tables) stays FP32. The subsampling prefix and the TDT
decoder are ONNX Runtime FP32, as in experiments 011 and 012. Whole
utterances are run, unchunked, as in the 2.19% FP32 reference.

| Format | Weights | Activations into the linear layers | Sums |
| --- | --- | --- | --- |
| fp32 | FP32 | FP32 | FP32 |
| bf16 | BF16 | BF16 | FP32 (what the 012 engine computes) |
| W8 per channel | int8, one FP32 scale per output channel | BF16 | FP32, scale applied after the sum |
| W8 group 128 | int8, one scale per output channel and 128 inputs | BF16 | FP32 |
| W8A8 per frame | int8 per channel | int8, one scale per frame (row), computed from that frame | int32 |
| W8A8 per tensor | int8 per channel | int8, one scale for the whole matrix (from its own maximum) | int32 |

All scales are plain FP32 values (absolute maximum / 127), not powers of two.

**Numbers.** Encoder-output error is measured on the real frames against
the FP32 NumPy run. The FP32 NumPy run reproduces ONNX Runtime FP32's
transcripts on all 210 utterances.

| Format | WER (errors / 4566 words) | Transcripts identical to FP32 | Word changes vs FP32 | Error after 2 layers (rel. RMS) | Encoder output rel. RMS / cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 2.19% (100) | 210 / 210 | 0 | 0 | 0 / 1 |
| bf16 | 2.19% (100) | 208 | 0 | 0.17% | 0.8% / 0.99997 |
| W8 per channel | 2.15% (98) | 202 | 0.07% | 0.9% | 4.1% / 0.9991 |
| W8 group 128 | 2.19% (100) | 207 | 0 | 0.6% | 3.0% / 0.9996 |
| W8A8 per frame | 2.17% (99) | 189 | 0.15% | 3.7% | 10.6% / 0.9946 |
| W8A8 per tensor | 2.26% (103) | 171 | 0.33% | 5.0% | 17.7% / 0.985 |

For comparison, ONNX Runtime's dynamic int8 encoder (`encoder-model.int8.onnx`,
experiment 011) scored 2.26% (103 errors). The int8 weights themselves
have 0.8-1.2% relative RMS error per channel and 0.65-0.76% per group of 128.
The largest activations entering the linear layers are about 1 after LayerNorm (q, k, v),
7-17 for the FFN and conv inputs, and 90-127 for the inputs of the FFN
second linears and pointwise conv 2 (after Swish). Those large values are why
per-frame int8 activations lose more than int8 weights.

**How this differs from experiment 011's XINT8.** Experiment 011 measured
38% relative error after two layers with Quark's XINT8 (22% for A16W8, 34%
for A8W8). Here the error after two layers is 0.6-0.9% for int8 weights and
3.7-5.0% for int8 activations as well. The schemes differ in four ways:

* **Scales.** XINT8 forces power-of-two scales, which can waste up to half of
  the int8 range. These are exact FP32 maxima.
* **Granularity.** One weight scale per output channel (or per 128 inputs),
  and for activations one scale per frame. Quark used per-tensor scales.
* **What is quantized.** Only the inputs and weights of the linear layers.
  Quark quantized the residual stream (which has one outlier channel near
  2,000), LayerNorm, the attention matrix products and the softmax input, which
  shares a range with the -10000 mask constant.
* **No HardSigmoid.** XINT8 replaced Sigmoid by HardSigmoid; here the
  nonlinear functions are unchanged.

So the error is not a property of int8 on this model; it came from that
quantization recipe. The per-tensor activation scale here is taken from each matrix's
own maximum (dynamic), not calibrated in advance, so a static calibrated
scale could do somewhat worse.

### LFM2.5-230M

**What was tried.** The system Python has PyTorch but no `transformers`
(installing it would change an environment the owner uses), so
[`014_int8_lfm25.py`](014_int8_lfm25.py) implements the LFM2 forward pass
itself (about 120 lines: gated short convolution, grouped-query attention
with q/k norms and rotary embedding, SwiGLU, RMSNorm, tied head), reading
`model.safetensors` directly. Text for the loss test was tokenized with the
IRON venv's `tokenizers` ([`014_llm_tokenize.py`](014_llm_tokenize.py)):
7,703 predicted tokens from `docs/writeup.md` and the LibriSpeech
transcripts. All projections and the tied embedding table are quantized. The
embedding's per-row scale serves both the input lookup and the vocabulary
head. Norms and the 3-tap conv weights stay BF16. Matrix products use BF16
activations and FP32 sums with one BF16 rounding of each result, as PyTorch
BF16 and the 010 engine do.

**Is the reimplementation right?** Its BF16 run matches the experiment 006
fixture (teacher-forced `transformers` BF16, 23 positions). The maximum hidden
error is 0.125 (the NPU engine's is 0.25), and the argmax token agrees at all 23
positions. It also reproduces `010_cpu_greedy.json`'s `transformers` greedy
tokens: 48 of 48 for prompts 1 and 3, and prompt 2 up to token 14, the known
exact tie where the NPU engine also diverges.

**Numbers.** Agreement and KL are against this file's BF16 run. "fp32 act" keeps the
same BF16 weights but does not round activations to BF16. It shows how far
BF16 rounding alone moves the results.

| Format | Weight rel. RMS | Fixture hidden rel. RMS vs BF16 | Fixture argmax mismatches | Perplexity (7.7K tokens) | Top-1 agreement | Mean KL | Greedy: first token differing from BF16 (3 prompts × 48) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| bf16 | 0 | 0 | none | 119.40 | 100% | 0 | — |
| fp32 act (reference) | 0 | 1.46% | none | 119.32 | 98.8% | 0.0003 | none differ |
| W8 group 32 | 0.56% | 1.21% | none | 119.51 (+0.1%) | 97.5% | 0.0012 | none differ |
| W8 group 64 | 0.63% | 1.32% | none | 119.64 (+0.2%) | 97.3% | 0.0014 | 21 / none / none |
| W8 per row | 0.91% | 2.03% | positions 7, 8, 14 (7 and 14 are BF16 ties) | 120.65 (+1.0%) | 96.2% | 0.0027 | 21 / 5 / 31 |
| W4 group 32 (extra) | 10.1% | 16.6% | 8 of 23 | 141.8 (+19%) | 67% | 0.23 | 5 / 2 / 0 |

The perplexity is high in absolute terms because the text (markdown with
numbers, and lower-cased 19th-century prose) is far from what a 230M chat
model expects. Only the differences between rows matter.

What it shows:

* **int8 weights with a scale per 32 or 64 weights are as close to BF16
  as BF16 is to FP32 activations.** Hidden-state error and greedy output
  are the same, and the loss is 0.1-0.2% higher. Agreement is 97.5% against
  98.8%, so the remaining difference is small but measurable.
* **One scale per row costs more:** 1% higher perplexity, the greedy text
  changes after 5 to 31 tokens, and the KL is 9 times that of BF16 rounding.
  It is usable, but it is measurably worse.
* **int4 with groups of 32 is not usable** without better methods
  (calibration, outlier handling), which this experiment did not try.

### What the NPU would gain (estimates)

These use the measured DDR read ceiling (27.5 GB/s, experiment 010), the
per-core stream limit from lead 2 and experiment 013's matrix rates.
None of it is measured.

**LLM decode (W8).** Decode reads every weight once per token. The weights fall
from 457 MB (BF16) to 229 MB plus scales: 236 MB with a BF16 scale per 64
weights, 243 MB per 32. At 27.5 GB/s that is 8.6-8.8 ms of streaming, plus the
~0.5-1 ms of fixed per-token cost the 010 engine has, so **about 9.5-10 ms per
token (100-105 tok/s) instead of 17.2 ms (58 tok/s)**, if each core
keeps up with the stream. Whether it can is the open question:

* An offline compile with Peano (`llvm-aie`, no NPU) shows the int8 → BF16
  unpack is a native unpack on load plus about four vector instructions per
  16 weights (a magic-number integer-to-float trick). With per-row scales
  (applied once per row after the sum) the work is about 0.31 vector
  instructions per weight. Each of the 8 cores receives about 2.6 weights per
  cycle at 1.33 GHz, which leaves a budget of 0.39 cycles per weight. That
  fits with about 20% to spare at 1.33 GHz, but would not fit if the clock
  were 1 GHz.
* Group scales (needed for BF16-level quality) cannot be applied after the
  whole sum. Either each group's partial sum is scaled in FP32 (emulated and
  slow), or each weight is multiplied by its group scale in BF16 first
  (about 2 more instructions per 16 weights, plus one more BF16 rounding).
  That likely makes the cores slower than the stream: an estimated 10-12 ms per
  token (85-100 tok/s).

**Parakeet encoder.** Its weight GEMMs are DDR-bound (lead 2), so fewer bytes help
directly. Per layer at 128 frames (now 3.7 ms):

* **W8, BF16 activations.** With 64×64 int8 weight objects (the same 4 KB),
  each activation tile serves 64 output columns, so the activation re-reads
  halve too. DDR then reads about 48 MB (1.75 ms). The cores then do the unpack
  (one per weight, reused for 32 frames) as well as the matrix work. Depending
  on how well the two overlap: **1.8-3.0 ms per layer**, an encoder at about
  140-170 ms instead of 185 (9-25% faster).
* **W8A8, int8 activations.** The native int8 matrix mode (256 MACs per
  cycle) needs no unpack. The weight stream into each core becomes the limit
  (128 MACs per cycle), and DDR reads about 36 MB (1.3 ms). That gives
  **about 1.3-1.5 ms per layer**, an encoder at about 130 ms (about 30%
  faster), with the 2.17% WER measured above for per-frame scales.

### What would have to be built

LLM (experiment 010 engine):

* Host: int8 packing of all projections and the embedding table, with scales
  in the object stream; a dequantized embedding row for the input (one row
  per token, on the host).
* Core: an int8 dot product (unpack, BF16 multiply-accumulate, scale) in
  `x8_core.cc` and `x8p_core.cc`, and new object counts. A 1 KB object then
  holds 1024 weights instead of 512, which changes every row count and the
  header layout.
* Program memory: the decode program has **156 bytes free** and the prompt
  program 776. The BF16 dot product would have to go (every matrix int8) or
  other code would have to be trimmed first.
* Validation: a new fixture from this file's W8 emulation (BF16 is no
  longer the target), `010_x8_validate.py`-style checks, and greedy agreement.

Parakeet (experiment 012 engine):

* Host: int8 packing with per-channel scales (the epilogue already has a
  per-column bias path to extend).
* Core: an int8 GEMM step (`aie::mmul<4, 8, 8, int8, int8>` for W8A8, or
  an unpack step for W8). For W8A8, every phase that produces a GEMM input
  (LayerNorm, the Swish epilogue, attention output, convolution module)
  would also have to write int8 values and a scale per frame, and the
  position-score GEMMs, which multiply two activations, stay BF16. The encoder's two kernels
  leave only a few hundred bytes of the 16 KB program memory.
* Validation: a new NPU design under the safety rules, so offline checks
  (`check()`, `012_schedule_check.py`), approval, then the full correctness
  run, a soak and the WER evaluation.

**Verdict: real for accuracy, unproven for speed.** Both models keep their
accuracy with int8 weights (and Parakeet even with int8 activations). The
speed estimates (LLM about 1.7×, encoder GEMMs 1.2-2.8×) depend on new core
kernels that were not built in this session.

## Not done

* No NPU int8 kernels were built (per the task: only with a clearly good
  offline result *and* the owner's approval; the offline result is good,
  so this is the obvious next step).
* The X-reuse GEMM change of lead 2 was not prototyped (not contained).
* NPU power was not measured in any mode.
* W8A8 for the LLM, and calibrated (static) activation scales for Parakeet,
  were not tested.
* BF16-rounded scales (as an NPU kernel would store them) were not tested.
  The emulation used FP32 scales.

## Reproduce

From the repo root. NPU runs follow the safety rules in the
[experiment 012 handoff](012_handoff.md); the power-mode batch only runs
programs that ran cleanly before.

```powershell
# Lead 1: set a mode (Turbo needs an administrator shell), then run the batch
& 'C:\Windows\System32\AMD\xrt-smi.exe' configure --pmode performance
powershell -NoProfile -File experiments\014_power_modes.ps1 -Label performance
& 'C:\Windows\System32\AMD\xrt-smi.exe' configure --pmode default
python experiments\014_power_summary.py default performance default_after turbo
# Lead 3: CPU only
cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\014_int8_parakeet.py
cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\014_llm_tokenize.py
python experiments\014_int8_lfm25.py
# Gather
python experiments\014_collect.py
```

The 013 scripts in the power-mode batch append their rows to
`cache/013/*.jsonl`; for this experiment those rows were moved to
`cache/014/` so experiment 013's raw data stays as it was.

| File | What it does |
| --- | --- |
| [`014_power_modes.ps1`](014_power_modes.ps1) | Runs the known-stable NPU programs in the current power mode, one process each, stops on any error |
| [`014_power_summary.py`](014_power_summary.py) | Extracts the key numbers from the power-mode logs |
| [`014_int8_parakeet.py`](014_int8_parakeet.py) | Parakeet encoder with int8 linear layers emulated on the CPU, WER on 210 utterances |
| [`014_llm_tokenize.py`](014_llm_tokenize.py) | Tokenizes the LLM evaluation text |
| [`014_int8_lfm25.py`](014_int8_lfm25.py) | LFM2 forward pass in PyTorch, int8/int4 weight emulation, fixture, loss and greedy checks |
| [`014_collect.py`](014_collect.py) | Writes `014_results.json` |
| [`014_results.json`](014_results.json) | All raw numbers (per-utterance transcripts and token ids stay in `cache/014/`) |
