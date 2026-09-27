# Experiment 010: eight-core LFM2.5-230M engine on the Phoenix NPU

A rewrite of the all-NPU LFM2.5-230M path around the one measured limit
that matters for single-user decoding: how fast the NPU can read weights from
DDR. Every model calculation still runs on the NPU, in BF16, and matches the
CPU BF16 reference.

## Result

| Workload (AC power) | Old NPU path (006–009) | This engine | CPU, 8 threads, token-by-token |
| --- | ---: | ---: | ---: |
| Decode, one token (short context) | ~450 ms | **17.2 ms (58 tok/s)** | 29.7–30.9 ms |
| Prompt token, batched four per weight pass | ~450 ms | **4.6 ms** | ~30 ms |
| 34-token prompt, time to first token | 8.10 s (paired path) | **0.18 s** | **0.042 s**, whole-prompt BF16 |
| 21-token prompt + 2 decode steps (fixture) | 10.1–11.1 s | **0.30 s** (single-token path) | 0.995 s |
| 15 / 161-token prompt, time to first token | — | **0.10 / 0.82 s** | 0.53 / 4.83 s |
| Decode at 1024 / 4090 context | impractical | 22.0 / 37.1 ms | not measured |
| 1024 / 4000-token prompt, time to first token | impractical | 6.6 / 47.4 s | not measured |

For the exact 20-word experiment 009 prompt, the eight-core engine produced
`An NP` after two selected tokens and measured 0.18 s to the first token;
the separate BF16 CPU whole-prompt benchmark measured 42.3 ms. The NPU thus
still loses on this short prefill by about 4.3×. The CPU token-by-token column
above matches the NPU's streaming schedule, while the whole-prompt CPU
figure is the appropriate comparison for prefill latency.

The chat runner, [`010_lfm25_npu_chat.py`](010_lfm25_npu_chat.py), streams
replies at about 58 tokens/s with state reuse across turns and a 4096-position
context. The CPU column is PyTorch BF16 feeding one token at a time, the NPU's
schedule. PyTorch's whole-prompt batched prefill on the CPU is much faster
than token-by-token (42 ms for 34 tokens, experiment 009); the NPU's batched
prompt path reuses each weight object for four tokens.

## Correctness

* [`010_x8_validate.py --check`](010_x8_validate.py) replays the 23-position
  fixture on the single-token path and compares the final hidden state, all
  8 conv states and all 6 KV caches after every position: maximum absolute
  errors 0.25, 0.094 and 0.156 (the old path: 0.25 / 0.18). All generated-token
  selections match. Teacher-forced prompt positions 7 and 14 can pick another
  token where the reference logits tie or differ by one BF16 step (18.5 vs
  18.5; 16.125 vs 16.0).
* [`010_prefill_validate.py`](010_prefill_validate.py) runs positions 0–19 in
  batches of four and checks every token's final hidden state, the conv
  states and the KV caches after each batch (maximum errors 0.25, 0.11,
  0.13), then decodes positions 20–22: tokens 2797, 38785, 562 as on the CPU.
* [`010_compare.py`](010_compare.py) compares greedy generation (batched
  prompt, then decode) with [`010_cpu_greedy.py`](010_cpu_greedy.py): a
  15-token and a 161-token prompt (256-position KV tier) match the CPU for
  all 48 generated tokens. A third prompt diverges at token 14, where the
  CPU's top two logits are an exact BF16 tie (23.25 and 23.25).

## Measured limits that shaped the design

| Probe | Finding |
| --- | --- |
| [`010_bandwidth_probe.py`](010_bandwidth_probe.py) | DDR→NPU streaming: 4 streams 14 GB/s; 8 streams (4 columns × 2 shim channels) ~27 GB/s at 256 MB, ~30 GB/s marginal. 16 streams do not fit the shim/memory-tile DMA channels. 1 KB DMA objects stream as fast as 8 KB. |
| [`010_direct_overhead.py`](010_direct_overhead.py) | Direct pyxrt submission: ~0.16 ms fixed cost per run versus ~0.8–1.3 ms through `iron.jit`. XRT sub-buffers work as kernel arguments, giving DMA a runtime base address. |
| [`010_stage_probe.py`](010_stage_probe.py) | A DDR round trip between stages inside one run costs ~15–20 µs; four stages per layer are cheap. |
| Program switching | Alternating compiled programs (hardware contexts) costs ~0.5 ms per switch: layers measured 1.2 ms alone but 1.7 ms interleaved. Fixed by running everything on one core program. |
| Kernel arguments | A runtime sequence with 8 buffer arguments compiled but hung on Phoenix; 6 work. |
| Scalar float | Phoenix AIE cores have no scalar FPU; scalar RMSNorm/SiLU/conv cost ~1 ms per layer until vectorized. |
| Host overhead | 0.2 ms of the 17.1 ms decode step is Python/XRT; 16.9 ms is NPU time. |

At 27 GB/s the 14 layers (330 MB of BF16 weights) need 12.4 ms and the tied
vocabulary head (134 MB) about 5 ms: the measured 12.5 + 4.7 ms decode is at
that ceiling. BF16 decode on this NPU cannot get meaningfully faster without
reading fewer bytes per token.

## Design

* **Eight cores, one stream each.** Core *k* (column k/2, rows 2–3) owns
  output slice *k* of every projection and receives a private shim DMA
  stream: activations, then its pre-packed weight slice, then state. The
  weight packing ([`lfm25_x8.py`](lfm25_x8.py)) puts each core's rows in the
  order it consumes them, so each stage is one long DMA transfer.
* **Local work by partitioning.** Core *k* computes B, C and x for its 128
  conv channels (so the gated convolution and its state are local) and the
  q/k/v rows of query heads 2k, 2k+1 and KV head k (so grouped-query
  attention is local). Only four exchanges per layer need the full vector:
  outputs go to a DDR scratch buffer and the next stage gathers them.
* **One submission per token.** A single runtime sequence runs all 14
  layers and the head; the first layer's input is a DMA straight from the
  embedding table row (a sub-buffer at the token id), and the NPU writes the
  selected token id.
* **One core program for everything.** Each core runs a small IRON loop
  (consume objects, produce objects, advance) around a C++ state machine,
  [`kernels/x8/x8_core.cc`](kernels/x8/x8_core.cc). A header object in each
  weight stream selects recurrent layer, attention layer or head. This
  avoids program switches and fits the 16 KB program memory.
* **Batched prompts.** A second core program,
  [`kernels/x8/x8p_core.cc`](kernels/x8/x8p_core.cc), runs four prompt
  tokens per weight pass (`dot4` reuses each weight load for four
  activation vectors; attention handles the batch's eight queries in one
  vectorized online-softmax step, plus causal attention within the batch).
  It uses the same weights, caches and conv states; the host stages the four
  embedding rows (a copy, no arithmetic) because the prefill program cannot
  take more buffer arguments. The final prompt token runs on the decode
  program, which selects the first reply token.
* **Numerics.** BF16 rounding at the model's boundaries, FP32 accumulation,
  vectorized RMSNorm/residual/conv using exact FP32 products (scalars split
  into BF16 pieces), SiLU from a host-computed table of PyTorch's BF16
  results (decode) or a vectorized FP32 sigmoid (batched prompt), and a
  vectorized FP32 exp for online softmax. Shared code is in
  [`kernels/x8/x8_math.h`](kernels/x8/x8_math.h).
* **KV cache.** Per attention layer and core, position *p* holds
  [key | value] at p·128. Attention streams a capacity tier (64, 256, 1024 or
  4096 positions); tiers are separate instruction streams on the same
  program. Appends are DMA writes to a sub-buffer at the current position.

## Run it

```powershell
# Chat (first launch compiles 12 instruction streams, cached afterwards):
& .\scripts\iron_python.ps1 experiments\010_lfm25_npu_chat.py
# One prompt:
& .\scripts\iron_python.ps1 experiments\010_lfm25_npu_chat.py --prompt "Hello!" --json
# Checks and benchmarks:
& .\scripts\iron_python.ps1 experiments\010_x8_validate.py --check
& .\scripts\iron_python.ps1 experiments\010_prefill_validate.py
& .\cache\lfm-env\Scripts\python.exe experiments\010_cpu_greedy.py
& .\scripts\iron_python.ps1 experiments\010_compare.py
& .\scripts\iron_python.ps1 experiments\010_context_bench.py
& .\scripts\iron_python.ps1 experiments\010_prefill_bench.py
```

`scripts/iron_python.ps1` loads the Visual Studio and IRON environment; run
each script in a fresh PowerShell process (the Visual Studio shell grows
PATH on every load). The checkpoint, fixtures and compiled binaries stay in
ignored `cache/` and `~/.npu/cache`. The Ryzen AI installation and NPU driver
were not changed. [`010_vtest.py`](010_vtest.py) runs single-core unit tests
of the FP32 vector helpers.

## Toolchain notes

* IRON's disk cache keys on the generator and compile-time kwargs, not on
  kernel source contents or closure constants. Designs here carry a
  `CompileTime[str]` tag with a digest of the kernel sources.
* pyxrt objects can crash the process when destroyed late; scripts exit with
  `npu_direct.finish()` and keep programs alive. A hung NPU run blocks the
  Python process in `wait()` with the GIL held.
* Peano (llvm-aie) crashed ("virtual register defs don't dominate all
  uses") when the whole state machine was inlined into one function; keeping
  the larger helpers `noinline` avoids it.
* `-Oz` plus an unrolled dot product keeps GEMV at stream speed while
  fitting program memory (decode: 4× unroll; batched: 8×).
* `aie::abs` on FP32 vectors does not behave as a float absolute value on
  AIE2; the batched sigmoid avoids it.

## What is left

* **Fewer bytes per token.** Decode is at the DMA ceiling. Weight
  quantization (for example INT8 with per-row scales) would roughly halve
  decode time but changes the model's numerics; it was not attempted.
* **Batched prompt compute.** With `dot4` removed entirely the batched pass
  still takes 3.9 ms per token (stream floor ~3.1 ms), so the matrix unit
  (`mmul`) could save at most ~0.7 ms per prompt token and would need a
  second, tiled weight copy (330 MB). Longer prompts are dominated by
  attention (emulated FP32 vector math in the online softmax).
* **Long-context decode attention.** ~4.8 µs per cached position per token
  across the six attention layers; the decode program has almost no program
  memory left for a batched rewrite.
