# LFM2.5-230M on the Phoenix NPU: experiment 010 handoff

Start here if you are continuing this work without the original
conversation. [`010_x8_engine.md`](010_x8_engine.md) is the results write-up;
this file explains how the engine works, why it is built this way, the hard
limits you will run into, how to debug it, and what to try next. State as of
2026-09-27.

## 1. Status in one paragraph

Every LFM2.5-230M calculation runs on the Ryzen 9 8945HS NPU (Phoenix, XDNA,
AIE-ML/"AIE2" tiles) in BF16. Decode is **17.2 ms per token (58 tok/s)**
at short context, 22 ms at 1K and 37 ms at 4K. Prompts run four tokens per
weight pass at **4.6 ms per token**. Outputs match the CPU BF16 reference
token for token except at exact BF16 ties. Decode reads 464 MB of weights per
token at ~27.5 GB/s, which is the NPU's measured DDR read ceiling, so BF16
decode cannot get meaningfully faster. For comparison on the same laptop:
PyTorch BF16 on 8 CPU threads ~33 tok/s; llama.cpp CPU-only F16 91 tok/s,
Q8_0 160, Q4_0 254. The NPU's case is low power with the CPU left free,
not raw speed. Before this work the all-NPU path took ~400–470 ms per token.

## 2. File map

| File | Role |
| --- | --- |
| [`lfm25_x8.py`](lfm25_x8.py) | IRON program definitions (decode `token_design`, batched `prefill_design`), weight packing, `io`/KV layout constants |
| [`lfm25_x8_model.py`](lfm25_x8_model.py) | `X8Model`: compiles/loads programs, packs weights into XRT buffers, `step()` (one token), `prefill()` / `prefill_batch()` |
| [`npu_direct.py`](npu_direct.py) | Minimal pyxrt layer: `Buffer` (host-only BO + NumPy view + sub-buffer views), `Program` (one hardware context), `Entry` (one instruction stream), `compile_design`, `finish` |
| [`kernels/x8/x8_core.cc`](kernels/x8/x8_core.cc) | Decode core program: state machine for recurrent layer, attention layer and vocabulary head |
| [`kernels/x8/x8p_core.cc`](kernels/x8/x8p_core.cc) | Batched-prompt core program (4 tokens per weight pass) |
| [`kernels/x8/x8_math.h`](kernels/x8/x8_math.h) | Shared vector math: dot product, RMSNorm, split3, exp16, rotary, attention update |
| [`kernels/x8/x8_vtest.cc`](kernels/x8/x8_vtest.cc), [`010_vtest.py`](010_vtest.py) | Single-core unit test harness for vector-float helpers |
| [`010_lfm25_npu_chat.py`](010_lfm25_npu_chat.py), [`../scripts/chat_lfm25_x8.ps1`](../scripts/chat_lfm25_x8.ps1) | Interactive chat and its launcher |
| [`010_x8_validate.py`](010_x8_validate.py) | Decode path vs the 23-position CPU fixture (hidden, conv states, KV) |
| [`010_prefill_validate.py`](010_prefill_validate.py) | Batched prompt path vs the fixture, then decode |
| [`010_cpu_greedy.py`](010_cpu_greedy.py), [`010_compare.py`](010_compare.py) | CPU greedy reference for arbitrary prompts; NPU vs CPU token-for-token |
| [`010_context_bench.py`](010_context_bench.py), [`010_prefill_bench.py`](010_prefill_bench.py) | Decode vs context length; time to first token vs prompt length |
| [`010_bandwidth_probe.py`](010_bandwidth_probe.py), [`010_direct_overhead.py`](010_direct_overhead.py), [`010_stage_probe.py`](010_stage_probe.py) | Hardware probes (DDR bandwidth, submission overhead, stage barriers) |
| [`../scripts/iron_python.ps1`](../scripts/iron_python.ps1) | Runs a script with Visual Studio + IRON environment |
| [`../scripts/run_npu_timeout.ps1`](../scripts/run_npu_timeout.ps1) | Same with a wall-clock limit; use for anything that might hang the NPU |

Local assets (ignored by Git): `cache/lfm25-230m/` (checkpoint, tokenizer),
`cache/lfm25-prompt-sequence-reference.npz` (23-position CPU fixture from
`006_lfm25_prompt_sequence_reference.py`), `cache/010_cpu_greedy.json`,
`cache/lfm-env/` (PyTorch reference env), `cache/iron/` (IRON 1.4.3 + XRT
SDK), compiled programs in `~/.npu/cache/`.

## 3. Running things

```powershell
.\scripts\chat_lfm25_x8.ps1                         # interactive chat
.\scripts\chat_lfm25_x8.ps1 -Message "Hi" -MaxNewTokens 64
powershell -NoProfile -File .\scripts\iron_python.ps1 experiments\010_x8_validate.py --check
powershell -NoProfile -File .\scripts\run_npu_timeout.ps1 -Script experiments\010_prefill_validate.py -Seconds 900
```

* Start each IRON run in a **fresh PowerShell process**. The Visual Studio
  developer shell appends to PATH on every load and eventually fails with
  "The input line is too long".
* `iron_python.ps1` pipes output line by line; interactive programs need an
  unpiped launcher (`chat_lfm25_x8.ps1`).
* Never name a PowerShell script parameter `$Prompt`: the venv's
  `Activate.ps1`, loaded by `iron_env.ps1`, overwrites it.
* The first model load compiles 8 decode + 4 prefill instruction streams
  (a few minutes); later loads take ~2 s plus ~2 s of weight packing.

## 4. Hardware primer (why the design looks like this)

| | CPU (8 Zen 4 cores) | NPU (Phoenix, AIE-ML) |
| --- | --- | --- |
| Compute | 8 large cores, AVX-512 | 4 columns × 4 compute tiles = 16 small VLIW vector cores (rows 2–5), plus 4 memory tiles (row 1, 512 KB each) and 4 shim tiles (row 0, DDR access) |
| Memory per core | caches in front of all RAM | 64 KB data + **16 KB program** memory, no cache |
| Data movement | automatic (caches, prefetch) | explicit DMA; each compute tile has 2 input + 2 output DMA channels; each shim tile has 2 DDR→array and 2 array→DDR channels |
| Float math | hardware scalar + vector FP | **no scalar FPU** (scalar float is software-emulated); vector unit does BF16×BF16→FP32 (16 lanes for element-wise MAC, matrix unit for mmul), int8, int16, int8×int4; FP32×FP32 vector multiply is emulated with several BF16 multiplies |
| DDR bandwidth | enough for llama.cpp F16 at 91 tok/s | measured ~27 GB/s over all 8 shim read channels (~30 GB/s marginal, ~32 GB/s nominal) |
| Control | OS threads | host submits a static "runtime sequence" (DMA program) through XRT; cores run a fixed loop |

Decode reads every weight once per token, so it is DMA-bound. **Only 8 of
the 16 compute tiles are used**: there are exactly 8 DDR read channels, each
core's arithmetic takes a fraction of its stream time, and 16 separate
streams do not fit the shim/memory-tile channels (the probe failed to
compile). The spare cores and the memory tiles are unused; they would only
help compute-bound parts (batched prompts, long-context attention).

## 5. Engine design

### Partitioning
* Core *k* sits at column k/2, row 2 + k%2 and owns output slice *k* of
  every projection: 128 of 1024 hidden rows, 320 of 2560 FFN rows, 384 conv
  in-projection rows (B, C, x of its 128 channels), and q rows of query heads
  2k, 2k+1 plus k/v rows of KV head k (LFM2.5: 16 query heads, 8 KV heads,
  head dim 64).
* Conv channels and attention heads are therefore local to a core. Each
  layer has four stages; between stages each core drains its slice to a DDR
  scratch buffer (`io`) and every core gathers the full vector for the next
  stage (~15–20 µs per barrier).
* Stages: (1) RMSNorm + input projection + conv (or q/k/v + attention),
  (2) output projection + residual, (3) FFN RMSNorm + W1/W3 + SiLU gate,
  (4) W2 + residual.

### Streams and weight packing
Each core has one input stream (DDR → core, 1 KB objects of 512 BF16, depth
4) and one output stream. Per layer and core the weight buffer holds, in
consumption order: RMSNorm gamma (1024) | header object (512) | projection
rows | output-projection rows | FFN gamma | W1/W3 rows interleaved per row |
W2 rows (2560 wide). The header holds `mode` at [500] (0 recurrent,
1 attention, 2 head), `rows` at [504], the head's first vocabulary row at
[508], plus the planar conv weights (recurrent) or q/k norm gammas
(attention). All 14 layers and the head (gamma + header per core) live in one
packed buffer (`W_LEN`, 326 MB). See `pack_recurrent`, `pack_attention`,
`pack_head`, `pack_weights`.

### Core program: object-driven state machine
The IRON core loop ([`lfm25_x8._core`](lfm25_x8.py)) is tiny: for 5
segments, consume `ctl[0]` input objects with `x8_do(op=IN)`, fill `ctl[1]`
output objects with `x8_do(op=OUT)`, then `x8_do(op=NEXT)`. All logic is in
C++: segment 0 reads input vector, token aux, gamma and header; `next()`
decides the following segments' object counts from the header. This lets one
compiled program run every layer kind and the head, which matters because
switching programs (hardware contexts) costs ~0.5 ms each.

### One submission per token
`token_design(capacity, with_head)` emits the DMA program for all 14 layers
(+ head). Buffer arguments (Phoenix hangs with more than 6): weights,
embedding row (an XRT sub-buffer of the table at the token id), `io`, KV
caches, KV append view (a sub-buffer at the current position), embedding
table (for the head). The head scores 8192 vocabulary rows per core; core 0
picks the winner and writes the token id to `SLOT(6, 0)`. Instruction
streams for KV tiers 64/256/1024/4096, with and without the head, all run on
one hardware context.

### KV cache
Per attention layer and core, position *p* holds [key 64 | value 64] at
p·128 (`CACHE_STRIDE` per core, `MAX_CAPACITY` = 4096). A tier streams its
full capacity every token; the core skips positions ≥ the past length. The
new entry is written by a 512-element DMA drain at p·128 (it also writes
junk into the not-yet-used slots p+1..p+3, overwritten later).

### Batched prompts
`prefill_design(capacity)` + `x8p_core.cc`: 4 tokens per weight pass
(`dot4` shares each weight load across 4 activation vectors, rows are
finalized as they complete). Attention processes the batch's 8 queries in
one vector step per cached block, then causal attention within the batch.
Buffer arguments: weights, `io`, caches, append view (4 args; the host
copies the 4 embedding rows into `io` at `PEMB`). The last prompt token runs
on the decode program so the NPU selects the first reply token. Switching to
the prefill program costs one context switch per prompt.

### Numerics
BF16 at the model's rounding boundaries, round-to-nearest-even, FP32
accumulation. FP32 scalar × BF16 vector products are exact via `split3`
(three BF16 pieces). Decode SiLU uses a host-computed table of PyTorch's BF16
results for inputs with exponent field in [112, 134); the batched path uses a
vectorized FP32 sigmoid. Online softmax keeps FP32 running max/sum/values;
probabilities are split into two BF16 pieces for the value accumulation.
Expected differences from PyTorch: last-bit BF16 changes; token choices
differ only where the reference's top logits tie in BF16.

## 6. Optimization history

| Step | Decode per token | What changed |
| --- | ---: | --- |
| Start (Codex's path) | ~400–470 ms | 1 core per layer, ~3,000 synchronized 8 KB transfers per layer, many program switches |
| 8 cores, one stream each, direct XRT | 46 ms | measured DDR ceiling first; packed weights for one long DMA per stage |
| Vectorized norm/SiLU/conv | 32.6 ms | scalar float is emulated on AIE; cost ~1 ms per layer |
| One core program for both layer kinds | 28 ms | program switches were ~0.5 ms each |
| Whole token in one submission, head in same program | **17.2 ms** | ~15 submissions → 1; embedding via sub-buffer DMA |
| Vectorized attention, skip unchanged rescale | 17.2 ms short / 37 ms at 4K (from 77) | long-context only |
| Batched prompts | prompt 12.5 → 4.6 ms per token | weight reuse over 4 tokens, 8× unrolled `dot4`, batched attention |

Measured facts that drove the decisions are in the "Measured limits" table
of [`010_x8_engine.md`](010_x8_engine.md).

## 7. Hard constraints and budgets

| Budget | Now | Notes |
| --- | --- | --- |
| Program memory per core (16384 B) | decode **16228 B (156 free)**, prefill **15608 B (776 free)** | Both compiled `-Oz`; larger helpers `[[gnu::noinline]]`; no float division; FP32 vector multiply in shared `vmul` helpers. Anything new must free space first. |
| Data memory per core (64 KB) | decode ~48 KB (mem 25.6 KB, SiLU table 11 KB, FIFOs 6 KB, stack 2 KB); prefill ~59 KB | Prefill has no room for the SiLU table |
| Stack | 2048 B declared per worker | aiecc measures and refuses if too small |
| Buffer arguments per runtime sequence | ≤ 6 | 8 compiled but hung the NPU |
| DMA buffer descriptors per shim tile | 16 active | shared by the column's two cores; split task groups if exceeded |
| DDR read bandwidth | ~27 GB/s | 8 channels; decode is at it |
| Program (context) switch | ~0.5 ms | keep everything per token on one program |
| Per-submission overhead | ~0.16 ms direct; ~1 ms via `iron.jit` | never call `iron.jit` designs in the hot path |

## 8. Toolchain pitfalls and debugging

* **IRON compile cache** keys on the generator and `CompileTime` kwargs, not
  on kernel source or closure constants. Every design here carries a tag
  with `KVER` (digest of `kernels/x8/*`). Forgetting this silently runs stale
  binaries.
* **Compile without IRON's runtime**: `nd.compile_design(design)` uses
  `design.compilable.compile()`. Calling the `iron.jit` object also creates
  IRON-owned hardware contexts, and too many contexts fail with "Failed to
  create context". Do not mix IRON's run path and `npu_direct` on the same
  design in one process.
* **pyxrt teardown crashes** the process (access violation) when objects are
  collected late: keep programs/buffers referenced and exit with
  `npu_direct.finish()`. Output is lost on such crashes unless flushed.
* **Hangs**: a wrong object count, bad tap or too many buffer arguments hangs
  the NPU; `run.wait()` holds the GIL, so use `run_npu_timeout.ps1`. Bisect
  by building variants: `prefill_design(capacity, only=(layer,...))` runs a
  subset of layers.
* **Numerical bugs**: compare stage outputs (`io` slots) against a NumPy
  re-implementation with BF16 rounding (this found the prefill SiLU bug);
  test vector helpers on one core with `010_vtest.py`.
* **Peano** (llvm-aie) crashed ("Virtual register defs don't dominate all
  uses") when the whole state machine was inlined; `noinline` on the larger
  helpers avoids it. `aie::abs` on FP32 vectors is not a float abs on AIE2.
  `-Oz` without unrolling made the GEMV ~9% slower (compute-bound); the
  decode dot product uses a 4× unroll, the batched one 8×.
* **Scalar float anywhere** in a hot path compiles to soft-float calls
  (`__mulsf3`, `__gtsf2`, ...); check with `llvm-nm -u` on the object.
* Standalone kernel compile for size checks:
  `llvm-aie/bin/clang++ file.cc -c -Oz -std=c++20 --target=aie2-none-unknown-elf -I<mlir_aie>/include -D__AIE_API_AIE_ADF_HPP__`,
  then `llvm-size` / `llvm-nm --size-sort -S` / `llvm-objdump -d`.

## 9. Validating a change

1. `010_x8_validate.py --check`: generated tokens must match; max errors
   about hidden 0.25, conv state ≤ 0.12, KV ≤ 0.16. Prompt positions 7 and 14
   may differ (reference ties).
2. `010_prefill_validate.py`: same bounds after each batch; tokens 2797,
   38785, 562.
3. `010_compare.py` (after `010_cpu_greedy.py` once): prompts 1 and 3 must
   match all 48 tokens; prompt 2 diverges at token 14 on an exact tie.
4. Speed: `010_x8_validate.py --repeats 3`, `010_context_bench.py`,
   `010_prefill_bench.py`.

## 10. What to do next (ranked)

1. **INT8 weights (W8, BF16 activations)**, the only big decode lever:
   ~240 MB per token instead of 464 → ~100 tok/s estimated (not linear,
   ~1 ms fixed per-token costs remain). Store int8 weights with one scale
   per 32–64 weights; in the core unpack int8 → BF16 and reuse the BF16
   MAC path (fits in the per-object time budget, unmeasured). Quantize the
   tied embedding/head too (134 MB); keep norms and conv weights BF16.
   Needs a quality check against BF16 (token agreement, perplexity). Code
   space: decode has 156 B free, so trim first. Alternatives: W8A8 (int8 ×
   int8 native, more quality risk), W4A8 (int8 × int4 native mode on AIE2,
   ~130 MB/token, ~150+ tok/s, quality risk on a 230M model).
2. **Attention bookkeeping** (matters above ~1K context): ~4.9 µs per cached
   position per token. Process 8 positions per step (all 16 lanes), vector
   max/sum instead of scalar soft-float, keep value accumulators in
   registers, use `mmul` for q·k, or split a head's cache over two cores.
   Estimated 2–3× on attention (4K decode 37 → ~25 ms). Blocked mainly by
   decode program memory.
3. **Context beyond 4K**: raise `MAX_CAPACITY`, add 8K/16K/32K tiers (KV
   ~12 KB per position, 32K ≈ 400 MB). Tiers stream full capacity, so add
   finer tiers or stream only filled blocks. Check shim BD length limits for
   very long cache transfers. Expected ~16 tok/s at 8K, ~5 at 32K without (2).
4. **Batched prompt compute**: stubbing `dot4` entirely only saves 0.7 ms
   per prompt token (3.9 vs 4.6; floor ~3.1), so `mmul` needs a second tiled
   weight copy (+330 MB) for ≤ 15%. The rest is per-token SiLU/norm work
   (the SiLU table does not fit prefill data memory).
5. **Power**: the previous battery method is in
   [`006_handoff.md`](006_handoff.md) and `scripts/measure_battery_run.ps1`;
   the new engine has not been power-measured. Joules per token is the
   metric that could favour the NPU over the CPU.

## 11. Provenance

Experiments 001–009 were done by Codex (see [`006_handoff.md`](006_handoff.md)
and [`009_prefill.md`](009_prefill.md)). Experiment 010 was built in a Claude
Code session on 2026-09-27; Codex's scheduled run (automation
`explore-ryzen-npu-low-level-efficiency`, daily 10:00) then improved its own
paired prefill on the older path (13.7 → 8.1 s for 34 tokens), validated the
010 engine, fixed the chat runner's tokens/s count and committed and pushed
everything as 54f2040.
