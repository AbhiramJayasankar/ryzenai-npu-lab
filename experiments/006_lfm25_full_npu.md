# Experiment 006: LFM2.5-230M on the Phoenix NPU

For the tested final path, current setup, component suitability, power
interpretation, and future work, read the [current-state handoff](006_handoff.md).
This file preserves the chronological experiment record; intermediate sections
describe what was known at that stage.

## Objective

Run **all LFM2.5-230M model calculations** on the Ryzen 9 8945HS NPU, optimize
the resulting implementation, then compare CPU, GPU, and NPU latency and
whole-system power under the same workload. The CPU may tokenize, submit NPU
work, and read the result, but must not perform any model layer, attention,
normalization, logits, or token-selection arithmetic in the final NPU path.
Intermediate CPU references are fixtures for development, not part of inference.

## Model and reference

- Official checkpoint: [`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M), revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
- `model.safetensors` SHA-256: `f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd`.
- 229,693,184 parameters, BF16, 14 layers (8 recurrent gated convolutions and 6 grouped-query attention blocks), hidden width 1,024, FFN width 2,560, vocabulary 65,536.
- Downloaded files live in ignored `cache/lfm25-230m/`. The original weights and tokenization files are not committed.
- A separate ignored `cache/lfm-env/` overlays an existing CUDA-enabled PyTorch 2.8 environment with Transformers 5.5.4. The IRON and Ryzen AI environments are unchanged.
- [`006_lfm25_baseline.py`](006_lfm25_baseline.py) runs a 21-token chat prompt with a cached one-token decode loop. It can save logits, every layer's output, and the first block's projection inputs/outputs as reference fixtures.

The same prompt and first 24 generated positions gave exploratory BF16 PyTorch
results on this laptop:

| Device | Prefill | Decode median | Decode throughput |
| --- | ---: | ---: | ---: |
| Ryzen 9 8945HS CPU, 8 PyTorch threads | 38.9 ms | 29.4 ms/token | 34.0 tokens/s |
| GeForce RTX 4060 Laptop GPU | 16.9 ms | 14.4 ms/token | 66.8 tokens/s |

These timings exclude model loading and tokenization. They are PyTorch
baselines, not optimized hardware ceilings. BF16 CPU and GPU greedy generation
eventually produced different tokens, so future quality comparisons must feed
the **same token sequence** to both devices before comparing logits.

## NPU results so far

All runs used the already-installed IRON 1.4.3/XRT toolchain on `NPU1`.

1. **First decoder block's RMSNorm:**
   [`006_lfm25_npu_norm.py`](006_lfm25_npu_norm.py) runs a custom AIE2 kernel
   against a real decode activation and learned normalization weight. Its 1,024
   outputs match the CPU BF16 reference exactly. Median warm call: about
   **1.10 ms**. The AIE's default FP32-to-BF16 narrowing truncated; setting
   `aie::rounding_mode::conv_even` made it match PyTorch's rounding.
2. **First decoder block's convolution input projection:**
   [`006_lfm25_npu_projection.py`](006_lfm25_npu_projection.py) runs all
   3,072 outputs of the real `1024 -> 3072` BF16 weight projection. The custom
   [matrix kernel](lfm25_matmul_kernel.py) accumulates in FP32 and spreads
   output-channel strips over up to four AI Engine cores. In the latest warm
   runs, one core took **5.31 ms**, four cores **1.77 ms**. Maximum absolute
   difference against the CPU BF16 result was **0.00390**; mean absolute
   difference was about **0.000476**. The test's stated tolerances passed.
3. **First decoder block's recurrent convolution and gate:**
   [`006_lfm25_npu_conv_gate.py`](006_lfm25_npu_conv_gate.py) uses the real
   projection output, learned depthwise weights, and three-value state from a
   previous token. The custom NPU kernel updates the state and produces the
   gated vector. Two consecutive decode-token calls match the CPU BF16
   reference **exactly** across both 1,024-value outputs and both 3,072-value
   states. The second call consumes the state produced by the NPU's first
   call, without reading it back to the CPU. Median warm second call: about
   **1.18 ms**.
   Phoenix allows only two input DMA channels per compute tile, so the test
   packs projection activations and convolution weights into one input buffer.
4. **Decode-specific matrix-vector projection:**
   [`006_lfm25_npu_gemv.py`](006_lfm25_npu_gemv.py) runs the same 3,072-output
   real projection without the 16-row padding. The custom BF16/FP32 GEMV
   kernel took **3.89 ms on one core** and **1.66 ms on four cores**, with the
   same maximum reference error `0.00390`. Compared with the padded matrix
   kernel's latest 5.31/1.77 ms, removing arithmetic waste helps most on one
   core; four-core latency appears substantially influenced by dispatch and
   data movement. These are measured observations, not a separated overhead
   breakdown.
5. **Complete first recurrent block over two decode tokens:**
   [`006_lfm25_npu_recurrent_block.py`](006_lfm25_npu_recurrent_block.py)
   chains 17 NPU calls: input RMSNorm, two convolution projections, recurrent
   gate/state update, first residual, second RMSNorm, three feed-forward
   projections, SiLU gate, and final residual. New BF16 cast, convolution
   packing, BF16 addition, and SiLU kernels keep all intermediate arithmetic
   on the NPU. The CPU reference fixture supplies the block input, learned
   weights, and state before the first token. The second token consumes the
   first token's NPU-produced state. On the first token, **every recorded
   intermediate and all 1,024 final outputs matched BF16 PyTorch exactly**.
   On the second token, the state was exact and 96.0% of block outputs were
   bit-exact; maximum absolute output difference was **0.000244**. The first
   difference was one value after the convolution output GEMV; subsequent
   BF16 rounding propagated it through the feed-forward layers. The
   [`006_lfm25_npu_mlp.py`](006_lfm25_npu_mlp.py) standalone FFN check was
   also bit-exact for all four outputs on the first token.

   This chain proves one block's numerical path, **not** useful performance:
   17 separately submitted programs took about **250-357 ms per token**.
   Repeating five stages took about **6-7 ms total**; adding one more stage
   raised the repeated prefix to **96-153 ms**. The jump suggests costly
   switching among several NPU program images, but its exact runtime cause
   remains unverified. A composed NPU program or fused stages are necessary.
   The current SiLU kernel's polynomial was verified for the first block's
   measured input range of roughly `[-0.75, 0.75]`; it needs a wider-range
   implementation before reuse across all model layers.
6. **Reducing NPU context switches:**
   [`006_lfm25_npu_bf16_gemv.py`](006_lfm25_npu_bf16_gemv.py) verifies a
   four-core GEMV that rounds its FP32 accumulation to BF16 within the same
   NPU program: all 3,072 outputs were bit-exact, median **1.72 ms**.
   [`006_lfm25_npu_conv_gate_packed_state.py`](006_lfm25_npu_conv_gate_packed_state.py)
   keeps the depthwise weights beside the evolving convolution state in NPU
   memory. Its two token outputs, states, and retained weights were exact;
   the second standalone call took **1.32 ms**. Together these changes cut
   the first-block chain from 17 to 11 program submissions. Full-chain
   medians varied between **130 and 184 ms**, with numerical results unchanged.
   The installed XRT runtime reports a nominal six Phoenix hardware-context
   slots and implements cache eviction when that limited pool fills. A live
   [`--diagnose-cache`](006_lfm25_npu_recurrent_block.py) run observed five
   contexts in this process and a replacement on each miss. It explains the
   sudden ~20 ms stage costs when the chain uses too many distinct program
   images. A larger context cache cannot remove the systemwide hardware limit.
7. **Single-program composition proof:**
   [`006_lfm25_npu_norm_gemv.py`](006_lfm25_npu_norm_gemv.py) streams the real
   first-block normalization weight and 6 MB projection matrix through one
   input DMA channel while the original hidden vector uses the other. One
   Worker performs RMSNorm, FP32-accumulate GEMV, and BF16 rounding in a single
   Phoenix program. All 3,072 outputs matched the CPU BF16 reference exactly;
   a warmed one-core call took **4.16 ms**. It establishes a workable pattern
   for composing a whole block, though this one-core proof is not yet the
   optimized four-core path.
8. **Single-program convolution prefix:**
   [`006_lfm25_npu_norm_proj_conv.py`](006_lfm25_npu_norm_proj_conv.py)
   extends the same one-core program through the recurrent convolution gate
   and state update. Two input DMA streams carry the padded hidden/state
   data and grouped weight tiles; each weight group is explicitly awaited
   before its descriptor is recycled. The gated output, next convolution
   state, and retained depthwise weights matched the CPU BF16 fixture
   exactly. Median warmed call: **5.71 ms** for normalization, the full
   `1024 -> 3072` projection, and recurrent convolution together. This is
   a first-token correctness and latency check; the two-token test above
   remains the reference for NPU-produced state crossing token calls.
9. **Complete first recurrent block in one NPU program:**
   [`006_lfm25_npu_single_program_block.py`](006_lfm25_npu_single_program_block.py)
   streams all five real projection matrices, both RMSNorm scales, and the
   recurrent state through **two input DMA streams** into one AI Engine core.
   Its Worker performs the full normalization, convolution and state update,
   residual operations, and gated FFN without returning an intermediate to
   the CPU. It materializes BF16 values at the model's arithmetic boundaries.
   A warmed first-block invocation took a median **18.1 ms**. For that decode
   token, all 1,024 final hidden values and all 3,072 convolution state values
   matched the CPU BF16 reference exactly. The next token's hidden input came
   from the CPU reference fixture (the preceding model layers do not exist
   yet), while an **NPU packing kernel** combined it with the first token's
   NPU-produced convolution state. Its state was exact and 96.0% of final
   values were bit-exact, with maximum difference **0.000244**. No CPU model
   arithmetic was used between the two block invocations.

   The earlier 17-call path took roughly 322 ms in a later run and the 11-call
   path took 130-184 ms across runs. The one-program result removes the
   Phoenix context-cache cliff for this block, though it uses only one core
   and still streams roughly 24 MB of projection and scale data each call.
   It is a **single-block decode check**, not a full model speed or power
   result. The first token's initial convolution state still comes from a
   reference fixture; prompt prefill is not implemented.
10. **Two successive recurrent layers and NPU-resident state:**
    [`006_lfm25_npu_two_recurrent_layers.py`](006_lfm25_npu_two_recurrent_layers.py)
    loads the real BF16 checkpoint weights for layers 0 and 1 using
    [`lfm25_checkpoint.py`](lfm25_checkpoint.py). The *same compiled* recurrent
    program runs both layers with different weights; an NPU packing kernel
    connects layer 0's output to layer 1's input. Both final hidden vectors
    and both convolution states matched the CPU BF16 fixture **exactly** for
    the first decode token. A warmed layer 1 call took a median **17.8 ms**;
    the two-layer chain, including NPU packing, took **37.3 ms**. These are
    invocation times, not a whole-model throughput estimate.

    On the next token, each layer received its prior NPU-produced state.
    Its initial layer-0 hidden input came from the CPU reference fixture
    because embedding and earlier model work are not implemented yet.
    Layer 0's output was **96.0% bit-exact**, with maximum absolute error
    **0.000244**; layer 1's chained output was **53.9% bit-exact**, with
    maximum absolute error **0.000610**. The chained layer-1 convolution
    state differed by at most **0.0078125**. To separate state handling from
    propagated rounding, feeding layer 1 the fixture's second-token hidden
    vector plus its **NPU-produced** prior state gave an **exact state** and
    output error at most **0.000488**. This is evidence of small BF16
    differences accumulating across blocks, not evidence of CPU fallback.
11. **Matched two-layer CPU latency:**
    [`006_lfm25_cpu_two_recurrent_layers.py`](006_lfm25_cpu_two_recurrent_layers.py)
    executes the same two decode-layer formulas with the same BF16 checkpoint
    weights and initial fixture state using PyTorch CPU. Across 200 warmed
    calls, its median was **9.60 ms** with 1 thread, **4.58 ms** with 4,
    **3.65 ms** with 8, and **4.88 ms** with 16. The current one-core NPU
    chain is therefore about **10.2 times slower** than the best measured CPU
    setting for this limited workload (37.3 / 3.65 ms). All outputs and
    states of the corrected CPU implementation matched the full-model BF16
    fixture exactly. Its timing includes the same two recurrent layers;
    neither timing includes embedding, attention, logits, or tokenization.
    This comparison is latency only; no power result is implied.
12. **First attention-layer decode prefix and context:**
    [`006_lfm25_attention_prefix_reference.py`](006_lfm25_attention_prefix_reference.py)
    builds a BF16 CPU fixture for layer 2's operator norm, Q/K/V projections,
    per-head normalization, and rotary position at decode position 21.
    [`006_lfm25_npu_attention_prefix.py`](006_lfm25_npu_attention_prefix.py)
    runs those operations in one one-core NPU program. All **2,048** Q/K/V
    outputs matched the CPU fixture **exactly**; its warmed median was
    **4.44 ms**. [`006_lfm25_npu_attention_context.py`](006_lfm25_npu_attention_context.py)
    takes the NPU-generated Q/K/V and computes cached attention scores,
    softmax, and value mixing on the NPU. With 21 earlier cached positions
    supplied by the CPU fixture, the new key and value matched the full-model
    CPU cache exactly. The resulting 1,024-value context differed from the
    full-model CPU context by at most **0.000488**; its warmed call took
    about **5.1-5.3 ms**. The NPU softmax uses a bounded exponential approximation.
    An NPU cache-append program adds the new NPU-generated key and value
    to the 21-position cache. Its entire resulting 22-position KV cache
    matched the CPU full-model cache **exactly**, with a warmed median
    **1.08 ms** for the append call.
    These two calls do not yet include the attention output projection,
    residual, or FFN. The earlier KV cache still comes from CPU prompt
    prefill, and a next-token call consuming the NPU-updated cache is not
    yet implemented, so this remains a partial attention decode check.
13. **Complete first attention block over one decode token:**
    [`006_lfm25_npu_attention_block.py`](006_lfm25_npu_attention_block.py)
    combines the NPU Q/K/V prefix, cached score/softmax/value mixing,
    NPU-side packing, and a one-program output projection, residual, and
    gated FFN tail. The KV cache append also runs on the NPU. All model
    arithmetic in this block runs on NPU; the incoming hidden vector and
    21-token prior KV cache still come from the CPU reference fixture.
    The final hidden vector differed from the full-model CPU reference by
    at most **0.000977**, and the updated KV cache matched **exactly**.
    With the CPU reference context fed to the NPU tail to isolate its math,
    the final hidden error was at most **0.000122**. The warmed attention
    tail took a median **13.57 ms**, and the five-call block chain including
    cache append took **25.91 ms**. These timings are for a partial model.
14. **Three consecutive model layers on NPU:**
    [`006_lfm25_npu_three_layers.py`](006_lfm25_npu_three_layers.py)
    connects recurrent layers 0 and 1 directly to attention layer 2 without
    reading intermediate hidden vectors into CPU. Hidden outputs after
    layers 0 and 1 matched the CPU reference exactly; the final attention
    output differed by at most **0.000977**, and the updated KV cache was
    exact. A warmed median was about **210 ms**, much worse than the sum
    of isolated kernel calls. Per-stage timing showed roughly 19-35 ms for
    most calls, including NPU packing calls that take around 1 ms alone.
    This is the known Phoenix context-cache eviction cost with seven
    distinct compiled programs. A fused variant passes the original hidden
    vector through the Q/K/V program and combines context computation,
    KV append, and tail packing in one NPU program. It uses five distinct
    programs. [`006_lfm25_npu_three_layers.py --fused`](006_lfm25_npu_three_layers.py)
    preserved the same hidden and cache errors and reduced the warmed
    three-layer median to **61.4 ms** (about **3.4 times faster** than the
    seven-program chain). Stage medians were 18.3 ms and 18.0 ms for the two
    recurrent blocks, 1.4 ms for recurrent packing, 4.8 ms for attention
    Q/K/V, 5.5 ms for fused context/cache/packing, and 13.4 ms for the
    attention tail. The near-additive total confirms the context-cache
    eviction was the main extra cost. This remains slower than the CPU's
    full-model decode and is not yet an end-to-end model runtime.
15. **All 14 model blocks for one decode token:**
    [`006_lfm25_npu_layer_stack.py`](006_lfm25_npu_layer_stack.py)
    reuses the same five compiled NPU programs with each layer's own real
    checkpoint weights. Every intermediate hidden vector stays on NPU.
    The input embedding vector and the *prompt* convolution/KV states come
    from the CPU reference fixture. Layer 13's output is compared with a
    newly captured **pre-final-normalization** reference; the standard
    `hidden14` fixture is already normalized. With the corrected comparison
    and a SiLU approximation that covers measured later-layer inputs up to
    about ±3.7, all 14 block outputs stayed within **0.015625** maximum
    absolute error of the CPU BF16 reference. The warmed 14-block median
    was **333 ms**. This is a single decode position with fixture-provided
    prompt state, not an autonomous generation loop or tokens/s result.
16. **Final normalization, vocabulary logits, and NPU token selection:**
    [`006_lfm25_npu_output_head.py`](006_lfm25_npu_output_head.py)
    first validated final RMSNorm and the tied 65,536-row vocabulary GEMV
    with the CPU's raw last-block hidden vector. Final norm was exact;
    logits were 99.99% bit-exact with maximum error **0.000977**. A streamed
    NPU argmax selected the same next token, **38,785**, without CPU
    selection. When connected to all 14 NPU blocks with `--head`, the final
    normalized hidden vector differed by at most **0.125** and logits by
    **0.1875** because small BF16 block differences propagated. NPU argmax
    still selected **38,785**, matching the CPU. The warmed full chain
    median was about **498 ms** in that run, with pronounced context
    eviction after adding three more compiled programs. This verifies
    one token's model arithmetic after the reference embedding/state is on
    NPU. It does **not** yet prove NPU prompt prefill or a multi-token
    generation loop.
17. **Two decode tokens using NPU-produced recurrent and KV state:**
    [`006_lfm25_npu_attention_two_tokens.py`](006_lfm25_npu_attention_two_tokens.py)
    first verified layer 2 alone: its second call consumed the first call's
    NPU-produced 22-position KV cache. Both KV updates matched the CPU
    reference **exactly**; second-token hidden error was at most **0.000732**.
    The full [`006_lfm25_npu_layer_stack.py --head --tokens 2`](006_lfm25_npu_layer_stack.py)
    then chained all 14 layers and the output head for two decode steps.
    Token 2 consumed **every recurrent and attention state produced by token
    1 on the NPU**. Its raw last-block hidden error was **0.0234375** and
    maximum logit error **0.21875**. NPU argmax selected token **562**, the
    same as CPU; token 1 selected **38,785**, also matching CPU. The initial
    prompt states and each token's embedding vector still came from the CPU
    fixture. The two-token warmed total was about **1.11 s** in this run,
    with extra Phoenix context evictions from separate 21- and 22-position
    attention programs plus the output-head programs. This is a correctness
    experiment, not a practical generation throughput result.
18. **NPU-produced next-token embedding:**
    [`006_lfm25_npu_fused_vocab.py --cores 1`](006_lfm25_npu_fused_vocab.py)
    verifies a one-core IRON program that combines final RMSNorm, tied
    vocabulary scoring, argmax, and retention of the winning embedding row.
    In isolation it selected token **38,785** and returned the exact BF16
    embedding row, with a warmed median of **39.47 ms**. The full
    [`006_lfm25_npu_layer_stack.py --fused-head --fused-head-cores 1 --tokens 2`](006_lfm25_npu_layer_stack.py)
    then fed this NPU-produced embedding into token 2, together with all
    NPU-produced recurrent/KV states. The NPU selected tokens **38,785**
    and **562**; both selected embedding rows exactly matched the CPU
    reference. The second token's last-block hidden vector had maximum
    absolute error **0.0234375** versus CPU BF16. The warmed two-token
    chain took **1.07 s** median. Its two fused-head calls took about
    **71 ms** and **76 ms** inside the full chain. The larger full-chain
    values than the isolated head reflect context switching and cache
    pressure; they are not core-only kernel timings. The NPU performs all
    model arithmetic after the CPU-provided prompt state and initial token
    embedding. This is still a fixed two-step correctness experiment, not
    a self-contained prompt-to-generation runtime or an efficient decoder.
19. **Four-core fused output head:**
    [`lfm25_fused_vocab_4core_kernel.py`](lfm25_fused_vocab_4core_kernel.py)
    splits the tied vocabulary into four 16,384-row shards. Each AIE core
    scores its shard, retaining its best row; adjacent cores pass one
    candidate along a short chain so the final core returns the global
    token ID and embedding. This layout respects the Phoenix tile
    connection limits. The isolated warmed median was **11.89 ms** versus
    **39.47 ms** for the one-core version, a **3.3×** speedup. Both selected
    **38,785** with the exact embedding row. In the two-token full chain,
    both NPU-selected tokens remained **38,785** and **562**, the next
    embeddings were exact, and the final hidden error remained **0.0234375**.
    The warmed chain median was **955 ms** across three repeats, with fused
    head calls around **42 ms** each. This is a comparison of Python/XRT
    wall time including context switching, not isolated core execution.
    Synthetic one-hot probes forced each of the four shards to win in turn;
    every NPU token ID and returned embedding row matched exactly.
20. **First prompt token from empty state:**
    [`006_lfm25_npu_attention_first.py`](006_lfm25_npu_attention_first.py)
    validated a special first-position attention program against CPU BF16
    for all six attention layers. Each initial KV cache matched exactly
    except when upstream NPU hidden error had already accumulated; isolated
    per-layer errors were at most **0.03125** for cache and **0.001953125**
    for hidden output. The complete
    [`006_lfm25_npu_prompt_first_stack.py`](006_lfm25_npu_prompt_first_stack.py)
    starts all recurrent and attention states empty, selects prompt token
    **1** from the tied embedding table by NPU DMA, runs all 14 blocks, and
    applies the four-core fused output head. The embedding and returned
    winning row matched exactly, the NPU selected the CPU's token **1**,
    recurrent/KV state error was at most **0.0625**, and final raw hidden
    error was **0.25** in BF16 values. The warmed wall-time median was
    **553 ms** across three repeats, including a **42 ms** embedding call
    and **49 ms** head call. Those calls include Python/XRT dispatch and
    substantial NPU context switching. This covers position zero only;
    the other 20 prompt positions are not yet chained on NPU.
21. **Complete prompt and two generated tokens with NPU model arithmetic:**
    [`006_lfm25_prompt_sequence_reference.py`](006_lfm25_prompt_sequence_reference.py)
    captures a sequential CPU BF16 reference for all **21** prompt
    positions and two following decode positions. Sequential CPU and the
    earlier batched CPU prefill differ slightly in BF16 arithmetic, but
    select the same token IDs. The NPU
    [`006_lfm25_npu_prompt_sequence.py`](006_lfm25_npu_prompt_sequence.py)
    starts from prompt token IDs, uses NPU DMA for every prompt embedding,
    builds recurrent and KV state from empty buffers, and feeds its own
    selected embedding and state to each decode position. With all 21
    prompt positions and two decode positions, it selected **2,797 →
    38,785 → 562**, matching the sequential CPU reference. All three
    returned embedding rows matched the checkpoint exactly. Across the
    23 positions, maximum per-position hidden error was **0.25** and
    maximum recurrent/KV state error was **0.18359375** versus sequential
    CPU BF16. Warmed wall time was **10.08 s** for prompt plus first
    selection and **11.21 s** for prompt plus two decode positions and
    three selections; each timing came from one measured run after a warm
    run. The CPU provides token IDs, weights, and precomputed positional
    constants, and separately checks results; it performs no activation or
    model-layer arithmetic in the NPU execution path. This is a fixed
    21-token prompt and two-token generation proof, not a general or fast
    inference runtime.

The matched sequential 23-position benchmark
[`006_lfm25_sequence_benchmark.py`](006_lfm25_sequence_benchmark.py)
feeds the same 21 prompt IDs and two generated IDs to PyTorch CPU and GPU.
After one warm run, three measured runs gave:

| Device | Prompt 21 positions | Two decode positions | Total 23 positions |
| --- | ---: | ---: | ---: |
| CPU, 1 thread | 1,605 ms | 147 ms | 1,750 ms |
| CPU, 2 threads | 1,054 ms | 102 ms | 1,154 ms |
| CPU, 4 threads | 931 ms | 84 ms | 1,015 ms |
| CPU, 8 threads | 891 ms | 99 ms | 995 ms |
| RTX 4060 GPU | 609 ms | 61 ms | 670 ms |
| Phoenix NPU prototype | — | — | 11,208 ms |

The NPU row is one measured run after one warm run and includes its 21
prompt positions, two generated positions, and three vocabulary selections;
the CPU/GPU rows execute the same positions one token at a time and report
the final three selections without forcing their results into the model.
All three devices selected **2,797 → 38,785 → 562**. The NPU path is about
**11× slower than the best measured CPU total** and **17× slower than the
GPU total**. This is a Python/XRT prototype comparison, not a hardware
efficiency ceiling. Batched PyTorch CPU/GPU prefill is faster still, as
the earlier baseline table shows.

The initial matrix kernel pads one useful activation row to 16 matrix rows,
wasting compute, and streams weights from system memory for every invocation.
The newer GEMV kernel removes this padding but still streams weights and is
tested as a standalone call.
The initial BF16-output kernel accumulated across K tiles in BF16 and had a
much larger error (maximum `0.0391` on 64 outputs); FP32 accumulation fixed
that. A single DMA descriptor could not encode the full 6 MB projection
weight, so the current kernel schedules 64-output strips within one invocation.

The NPU timings include the Python/XRT call, dispatch, transfer, execution,
and completion, with compiled binaries and allocated input buffers warmed.
Summing standalone timings is not a valid full-model performance estimate.
The earlier fused-head two-token test uses CPU-created prompt fixtures. The
new 23-position sequence instead builds prompt state and all model
activations on NPU; the CPU reference is used only for validation. Host-side
tokenization, weight loading, positional-constant preparation, and XRT
dispatch remain.

### Follow-up transfer and cache checks

The full sequence now reports recurrent packing/compute and attention
prefix/context/tail separately. In a two-run AC baseline, recurrent compute
took **3.590 s** and attention tail **3.155 s** of an **11.078 s** total.
Grouping each pair of 8 KB weight tiles behind one DMA wait, with a
two-element weight FIFO in the recurrent and attention-tail programs,
reduced those directly changed components to **2.851 s** and **2.666 s**
in a later four-run AC median. Complete wall time was **10.866 s** in that
run, with separate same-code runs between **10.077 and 10.866 s** as system
timing varied. Tokens, embeddings, and per-position error limits remained
correct. Four-element recurrent batches brought only a small additional
change; preloading activation vectors and pairing the attention-prefix
weights showed no useful speed improvement, so those changes were reverted.

An optional fixed-capacity 32-position KV cache uses one attention context
program regardless of prior length. It passed all 23 positions with the
same tokens and errors, but a matched pre-batching AC test was **11.419 s**
versus **11.206 s** for the variable-length default. It is retained as
`--fixed-cache` for research, not selected as the faster path. The fixed
layout adds padded KV traffic. The earlier battery measurements below
predate the weight-transfer change and must not be combined with its newer
AC time to claim an updated energy result.

## Remaining runtime and efficiency work

The 23-position proof is still far slower than the CPU and GPU baselines.
The attention context program compiles separately for each prior cache
length (now tested from 1 through 22; local score storage bounds it to
127 prior positions). Embedding DMA also specializes to each token ID.
Efficient variable-length attention and KV storage, dynamic embedding
selection, and longer-generation support are needed for a general runtime.
Phoenix context-cache eviction is visible in the wall times. The four-core
fused head supplies the next embedding in one program but still streams
about 134 MB of tied weights per selection. Recurrent and attention
programs stream weights each call and use one compute core each. More
sharding, fusion, and persistent data should reduce latency and energy.

## Power measurement

The NPU's `xrt-smi examine --report platform` reports `Power: N/A` on this
Phoenix device. [AMD documents that Phoenix/Hawk Point lack NPU power
reporting](https://ryzenai.docs.amd.com/en/main/xrt_smi.html). Windows battery
discharge provides whole-laptop power while unplugged, not NPU-only power.

[`scripts/measure_battery_run.ps1`](../scripts/measure_battery_run.ps1)
was used for a battery-only comparison. It refuses an AC-powered run, samples
idle discharge before each benchmark, skips model load and warm-up time,
then records median active and idle-adjusted **whole-laptop watts**. It saves
each benchmark's raw output in ignored `cache/`. Its AC guard was verified.

One unplugged session on 2026-09-26 gave these exploratory results for the
same 21 prompt and two generated positions:

| Device | Idle W | Active W | Idle-adjusted W | Battery pass | Approx. J per 23-position pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, 8 threads | 23.78 | 30.00 | 6.22 | 1.194 s | 7.43 J |
| RTX 4060 GPU | 30.13 | 38.73 | 8.60 | 0.911 s | 7.83 J |
| Phoenix NPU prototype | 24.73 | 27.59 | 2.87 | 13.212 s | 37.85 J |

Joules here are idle-adjusted watts multiplied by the matching median pass
time, or about **0.323**, **0.341**, and **1.646 J per position** for CPU,
GPU, and NPU respectively. Runs used 65 CPU, 90 GPU, and 6 NPU measured
repeats after each script's warm-up; the battery sampler recorded 91, 96,
and 81 active readings. These are **whole-laptop incremental estimates**, not
device rail measurements. The battery sensor updates in coarse steps, idle
power drifted between devices (especially before GPU), and the pass times
changed on battery. A first NPU trial had a higher pre-run idle than active
reading because the previous GPU load had not settled; its negative
idle-adjusted result was discarded. The settled NPU trial used a 30-second
idle baseline. The NPU uses less incremental whole-laptop power in this
sample but about **five times more energy per 23-position pass** than the
measured CPU/GPU runs because its prototype is much slower. Repeat trials
with randomized order or an external meter are needed for tight power
confidence intervals.

The NPU battery run's timing breakdown was **12.01 s** for prompt positions
and **1.20 s** for two decode positions. Summed over all 23 positions,
attention calls took **7.65 s**, recurrent calls **4.29 s**, prompt embedding
DMA **1.01 s**, and the three output-head calls **0.19 s**. These are
Python/XRT wall times and expose attention/context switching as the main
optimization target.

For repeats, keep the same prompt, token IDs, display state, and power mode
and allow enough idle time between devices for temperature and power state
to settle. NVIDIA telemetry can additionally report GPU device power but
has no NPU equivalent on this Phoenix device.

## Reproduce the current checks

From the repo root, after downloading the pinned model and preparing the two
ignored environments:

```powershell
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_baseline.py --device cpu --tokens 24 --reference cache\lfm25-reference-cpu.npz
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_baseline.py --device cuda --tokens 24
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_cpu_two_recurrent_layers.py --threads 8 --repeats 200
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_attention_prefix_reference.py
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_prompt_first_reference.py
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_prompt_sequence_reference.py --decode 2
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_sequence_benchmark.py --device cpu --cpu-threads 8 --repeats 3
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_sequence_benchmark.py --device cuda --repeats 3
foreach ($layer in 4,6,8,10,12) { & .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_attention_prefix_reference.py --layer $layer }
foreach ($layer in 2,4,6,8,10,12) { & .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_attention_prefix_reference.py --layer $layer --step 2 }

& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
. .\cache\iron\mlir-aie\iron_env.ps1
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_norm.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_projection.py --outputs 3072 --cores 4
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_conv_gate.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_gemv.py --cores 4
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_mlp.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_recurrent_block.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_bf16_gemv.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_conv_gate_packed_state.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_norm_gemv.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_norm_proj_conv.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_single_program_block.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_two_recurrent_layers.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_attention_prefix.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_attention_context.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_attention_block.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_three_layers.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_three_layers.py --fused
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_layer_stack.py --head
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_attention_two_tokens.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_layer_stack.py --head --tokens 2
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_fused_vocab.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_fused_vocab.py --cores 4
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_fused_vocab.py --cores 4 --probe-shards
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_layer_stack.py --fused-head --tokens 2
foreach ($layer in 2,4,6,8,10,12) { & .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_attention_first.py --layer $layer }
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_prompt_first_stack.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_prompt_sequence.py --positions 21 --decode 2 --repeats 1
```
