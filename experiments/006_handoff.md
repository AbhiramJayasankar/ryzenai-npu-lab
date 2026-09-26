# LFM2.5-230M Phoenix NPU: handoff

This is the current-state guide for a new task with no conversation history.
Read [the detailed experiment log](006_lfm25_full_npu.md) for the chronological
proofs and individual measurements. Read [experiment 005](005_low_level_access.md)
for the custom-kernel toolchain and its limitations.

## What works now

On the Ryzen 9 8945HS Phoenix NPU (`NPU1`),
[`006_lfm25_npu_prompt_sequence.py`](006_lfm25_npu_prompt_sequence.py) runs
**every LFM2.5-230M model calculation** for one fixed 21-token chat prompt
and two autoregressive decode positions. It starts with empty recurrent and
key/value (KV) states. Prompt IDs enter through NPU embedding DMA; later
tokens use the NPU-selected embedding and NPU-produced states. The NPU runs
normalization, projections, recurrent convolution, attention, feed-forward
layers, final normalization, vocabulary scoring, argmax, and embedding
selection. The host loads/tokenizes weights and the prompt, prepares rotary
constants, submits programs, and validates results. It does **not** compute
model activations or select tokens in the NPU inference path.

The fixed sequence selects token IDs **2797, 38785, 562**, identical to the
sequential CPU BF16 reference. The returned embedding row for each selection
matches exactly. Across all 23 positions, maximum absolute hidden error is
**0.25** and maximum recurrent/KV state error is **0.18359375**. These are
BF16 differences after 14 layers; the selected tokens remain identical. The
script checks every position and fails if its token or numerical thresholds
diverge. This is an **all-NPU arithmetic proof for this finite sequence**,
not a general text-generation API: the prompt is fixed at 21 tokens, decode
is limited to two positions, and several programs specialize on cache length
or token ID.

The implementation is much slower than the CPU and GPU for this workload.
On AC, the original full NPU run was about **11.1 s** versus matched
one-token-at-a-time PyTorch **0.995 s on CPU (8 threads)** and **0.670 s on
the RTX 4060 Laptop GPU**. The latest pair-batched weight DMA experiment
gave **10.08 to 10.87 s** across AC runs, with noticeable system variation.
The NPU result includes three vocabulary selections; CPU/GPU benchmark the
same 23 input positions and also report those selections. These are warmed
Python/XRT wall times, including submission, transfer and program switching;
they are not bare AI Engine execution times or hardware ceilings.

## Hardware and software to preserve

| Component | Verified local state |
| --- | --- |
| NPU | Phoenix/Hawk Point `PCI\\VEN_1022&DEV_1502`, XRT `NPU1`, five columns |
| OS | Windows 11 build 26200 |
| NPU driver / firmware | `32.0.203.280` / `1.5.5.391` |
| Ryzen AI | 1.7.0 at `C:\Program Files\RyzenAI\1.7.0`; separate Miniforge environment `ryzen-ai-1.7.0` |
| Direct kernels | IRON/MLIR-AIE 1.4.3, XRT Windows SDK 2.21.75, separate ignored `cache/iron/` environment |
| Reference runtime | Python environment `cache/lfm-env/`, PyTorch 2.8.0+cu126, Transformers 5.5.4, huggingface_hub 1.33.0 |

IRON compiles custom C++/AIE programs to NPU binaries and XRT submits them.
This bypasses the Ryzen AI ONNX compiler, **not** the NPU driver or firmware.
The old Ryzen AI installation and driver are working; do not upgrade them
merely to reproduce this result. Local environments, downloaded binaries,
model weights, fixtures, and generated NPU binaries are intentionally ignored
by Git.

The checkpoint is [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
at revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
`model.safetensors` SHA-256 is
`f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd`.
Its local files live under ignored `cache/lfm25-230m/` (including config,
tokenizer and chat template). The model has 229,693,184 BF16 parameters,
14 blocks, width 1024, FFN width 2560, and a tied 65,536-token vocabulary.

## Model path and what suits this NPU

The block order is convolution, convolution, attention, convolution,
attention, convolution, attention, convolution, attention, convolution,
attention, convolution, attention, convolution: **eight recurrent gated
convolution blocks and six grouped-query attention blocks**. Each recurrent
block has an input projection, depthwise state update/gate, output
projection, and gated FFN. Each attention block has Q/K/V projections,
per-head norm and rotary position, cached attention/softmax, output
projection, and gated FFN. The final head normalizes, scores the tied
embedding table, picks a token, and copies its embedding for the next step.

| Part | NPU evidence | Current practical fit |
| --- | --- | --- |
| BF16 GEMV/projections | Real 1024-to-3072 projection ran in 1.66 ms on four cores versus 3.89 ms on one; numerical error at most 0.00390. Full recurrent and attention block programs run on NPU. | **Compute is suitable**, especially when output channels are sharded. Current full model uses one core per block and streams all weights each token, so the implementation is still slow. |
| Recurrent depthwise convolution/state | NPU-produced state was consumed at the next token; isolated two-token state/output checks were exact or within small BF16 error. | **Suitable** as a small operation fused with projections. Its own dispatch is too costly as a standalone stage. |
| RMSNorm, residual, SiLU gate | Correct NPU kernels in the block programs; RMSNorm needed round-to-nearest-even to match PyTorch BF16. | **Suitable inside fused blocks**. Separate submissions are poor for such short vector work. |
| Grouped-query attention and KV cache | All six layers correctly build and consume NPU-produced KV state through the 23-position sequence. Softmax uses a bounded exponential approximation. | **Correct but currently awkward**: cache-length-specific programs and context switching add cost. |
| Vocabulary scoring and next embedding | Four-core fused head returns the correct argmax and exact winning BF16 embedding row. Isolated four-core head was 11.89 ms versus 39.47 ms on one core. | **Good parallel work**, but the full 134 MB tied table streams for each selection; full-chain head calls cost more. |
| Prompt embedding lookup | NPU DMA selects rows from the tied table; all prompt inputs enter NPU this way. | **Correct, low arithmetic intensity**; Python/XRT dispatch dominates. |
| Tiny SAXPY | 4096 BF16 values: NPU call 0.758 ms versus NumPy CPU 0.018 ms. | **Poor standalone fit** at this job size. |
| Larger INT16 matrix multiplication | One-core 256² and 512² examples passed; NPU beat the tested NumPy INT16 CPU implementation. | **Promising for larger dense jobs**, but that NumPy comparison is not an optimized CPU ceiling. |

Phoenix exposes only two input DMA channels per compute tile in this
configuration. A direct recurrent program with separate hidden, state and
weight input FIFOs failed compilation at **three inputs / one output**;
packing hidden and state into one input avoids that limit. An earlier
17-submission recurrent block was about 250-357 ms; composing it into one
program removed most switching cost (about 18 ms in its isolated first-block
test). XRT nominally has six hardware-context slots here. Distinct program
images above the effective cache capacity caused large latency cliffs; the
three-layer chain dropped from about 210 to 61 ms after reducing seven
programs to five. Reuse of the same compiled programs across layers matters.

## Latest transfer and timing findings

The full sequence now reports separate embedding, recurrent packing,
recurrent compute, attention prefix/context/tail, and head wall times. An
AC baseline before transfer batching (median of two warmed runs) was:

| Component over 23 positions | One weight tile awaited at a time | Pair-batched weight DMA (latest four-run median) |
| --- | ---: | ---: |
| Recurrent compute | 3.590 s | 2.851 s |
| Attention tail | 3.155 s | 2.666 s |
| Recurrent packing | 0.298 s | 0.336 s |
| Attention prefix | 1.479 s | 1.680 s |
| Attention context | 1.372 s | 1.601 s |
| Embedding DMA | 1.009 s | 0.962 s |
| Complete 23-position call | 11.078 s | 10.866 s |

Pairing weight transfers uses a two-element FIFO and waits for each second
tile before releasing both DMA tasks. The improvement in the directly changed
recurrent and attention-tail components is repeatable; total time varies with
embedding/context and system state. A separate two-run AC pass with the same
paired code gave 10.077 s total, 2.810 s recurrent compute and 2.804 s
attention tail. Four-element batching only marginally changed recurrent time;
preloading the 64-element activation into vector registers did not improve
full-run timing, so the latter was reverted. Pairing the attention-prefix
weight transfers showed no clear gain and was also reverted.

An optional `--fixed-cache` experiment uses one fixed 32-position KV layout
and one context program for all tested lengths. Its cache includes two BF16
metadata/padding values per head so each DMA slice is 4-byte aligned. It
passed all 23 positions with the same tokens and error limits, but a matched
pre-batching AC check was **11.419 s** versus **11.206 s** for variable-length
cache. It remains an optional research path; the faster variable-length
path is the default. Do not infer that one compiled context binary is faster
without considering the padded KV traffic. The fixed-cache option has not
been power-tested or benchmarked again after weight batching.

## Power result and limits

Phoenix `xrt-smi` reports `Power: N/A`, so there is **no NPU-only power
reading**. [`measure_battery_run.ps1`](../scripts/measure_battery_run.ps1)
sampled Windows battery discharge during the same 23-position workload on
2026-09-26. One unplugged session produced:

| Device | Idle W | Active W | Incremental whole-laptop W | Pass | Approx. incremental J/pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, 8 threads | 23.78 | 30.00 | 6.22 | 1.194 s | 7.43 |
| RTX 4060 GPU | 30.13 | 38.73 | 8.60 | 0.911 s | 7.83 |
| NPU prototype before paired DMA | 24.73 | 27.59 | 2.87 | 13.212 s | 37.85 |
| NPU with paired DMA, later unplugged session | 14.61 | 16.862 | 2.252 | 10.504 s | 23.65 |

In the first session, the NPU drew fewer incremental whole-laptop watts but used
roughly five times more estimated incremental energy per pass because it
ran much longer. These values are **exploratory whole-laptop estimates**:
battery readings update coarsely, idle levels drifted, and a first NPU trial
had to be discarded after GPU cooldown made its idle baseline too high.
The later NPU run used six measured passes after warm-up, 30 idle samples,
and 70 active samples. It selected the same tokens with the same maximum
hidden/state errors. Its pass was **20.5% faster** than the earlier battery
NPU pass. The reported charge capacity stayed at 58,145 mWh across this
short run, and the idle baseline was about 10 W lower than in the earlier
session. Thus **23.65 J is only a session-specific idle-adjusted estimate**;
the apparent energy improvement cannot be attributed confidently to the
code change. Keep the two battery sessions separate when interpreting power.
For stronger power evidence, randomize run order, repeat settled idle/load
trials, hold display state constant, or use an external meter. The power
script refuses to run on AC and keeps its raw output under ignored `cache/`.

## Reproduce and navigate

From this repository's root on the configured laptop, after ensuring the
ignored checkpoint and CPU fixtures are present:

```powershell
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_baseline.py --device cpu --tokens 24 --reference cache\lfm25-reference-cpu.npz
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_prompt_sequence_reference.py --decode 2
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_sequence_benchmark.py --device cpu --cpu-threads 8 --repeats 3
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_sequence_benchmark.py --device cuda --repeats 3

& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
. .\cache\iron\mlir-aie\iron_env.ps1
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_prompt_sequence.py --positions 21 --decode 2 --repeats 2
```

The NPU command needs the generated
`cache/lfm25-prompt-sequence-reference.npz` to check each position, and the
model files under `cache/lfm25-230m/`. Both are intentionally untracked.
Do not interpret the CPU fixture-generation step as CPU arithmetic inside
the NPU inference run: it is an independent reference saved for comparison.
Use `--fixed-cache` only to reproduce that optional cache experiment. The
NPU script prints JSON, including timing components and every position's
error. It exits with an error if token IDs or numerical thresholds fail.

Key code:

- [`006_lfm25_npu_prompt_sequence.py`](006_lfm25_npu_prompt_sequence.py): end-to-end fixed-sequence orchestration and checks.
- [`lfm25_single_program_block_kernel.py`](lfm25_single_program_block_kernel.py): recurrent block with two input streams and paired weight DMA.
- [`lfm25_attention_prefix_kernel.py`](lfm25_attention_prefix_kernel.py), [`lfm25_attention_context_cache_kernel.py`](lfm25_attention_context_cache_kernel.py), [`lfm25_attention_tail_kernel.py`](lfm25_attention_tail_kernel.py): attention stages.
- [`lfm25_fused_vocab_4core_kernel.py`](lfm25_fused_vocab_4core_kernel.py): NPU argmax and next embedding.
- [`lfm25_checkpoint.py`](lfm25_checkpoint.py): BF16 checkpoint loading and weight layouts.
- [`006_lfm25_full_npu.md`](006_lfm25_full_npu.md): full chronology, isolated experiments, numerical checks and earlier timings.

## Possible next experiments (not prerequisites for this proof)

1. Shard recurrent and attention GEMV output channels across several cores
   **inside the full block programs**. Four-core isolated projections and
   head kernels already show the most relevant speedups; preserve the model's
   BF16 rounding boundaries and NPU-resident state.
2. Reduce the many streamed 8 KB weight-tile transfers, Python/XRT program
   calls, and context switches. Test larger DMA batches or longer-lived
   programs against both latency and memory limits; pair batching helped
   compute-heavy block stages, while four-element batching barely helped.
3. Design variable-length KV cache and dynamic token embedding selection so
   one compiled program can handle arbitrary generation lengths without the
   fixed-cache variant's excessive padding. Verify real NPU execution and
   every position against a same-token CPU BF16 reference.
4. Repeat power trials with stable idle conditions and a battery meter that
   exposes reliable capacity changes. The key metric is joules per generated
   position for the **whole** NPU pipeline, not an isolated kernel's latency.

The immediate user request is to preserve this working result and its
findings. Further optimization is deferred until explicitly resumed.
