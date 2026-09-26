# Experiment 006: LFM2.5-230M on the Phoenix NPU

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
These operations do not yet form a complete block, and summing their standalone
timings is not a valid full-model performance estimate. The current test scripts
use CPU-created reference fixtures as NPU inputs; no full-NPU decode runtime
exists yet.

## Remaining model computations

The first recurrent block needs a composed NPU program that avoids expensive
switches among the current kernels. Its first convolution state comes from a
CPU reference fixture; subsequent state remains on the NPU across calls.
Six attention blocks also need Q/K/V projections, per-head norms,
rotary position encoding, persistent KV cache, score reduction, softmax, value
mixing, and output projection. The complete path additionally needs embedding
lookup, final normalization, tied vocabulary projection, and NPU token
selection. Prompt prefill must initialize convolution and KV state correctly.

The next correctness gate is one attention block, followed by whole-model
logits under a fixed token sequence. A full decode loop also needs state
initialization from prompt prefill. Output-channel sharding, kernel fusion,
and persistent weights should reduce dispatch and transfer cost.

## Power measurement

The NPU's `xrt-smi examine --report platform` reports `Power: N/A` on this
Phoenix device. [AMD documents that Phoenix/Hawk Point lack NPU power
reporting](https://ryzenai.docs.amd.com/en/main/xrt_smi.html). This laptop is
currently on AC power, so Windows battery discharge is zero. No defensible
CPU/NPU/GPU power comparison has been measured yet.

For a comparable end-to-end result, run warmed CPU, GPU, and complete NPU
generation on battery with the same prompt, fixed token count, display/power
mode, and enough repetitions for the battery discharge sensor to stabilize.
Record idle-adjusted **whole-laptop watts**, tokens/s, and joules/token;
optionally record GPU device power through NVIDIA telemetry. Whole-laptop
discharge is not an isolated NPU power reading.

## Reproduce the current checks

From the repo root, after downloading the pinned model and preparing the two
ignored environments:

```powershell
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_baseline.py --device cpu --tokens 24 --reference cache\lfm25-reference-cpu.npz
& .\cache\lfm-env\Scripts\python.exe experiments\006_lfm25_baseline.py --device cuda --tokens 24

& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
. .\cache\iron\mlir-aie\iron_env.ps1
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_norm.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_projection.py --outputs 3072 --cores 4
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_conv_gate.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_gemv.py --cores 4
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_mlp.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\006_lfm25_npu_recurrent_block.py
```
