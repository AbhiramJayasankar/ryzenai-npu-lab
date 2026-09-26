# Experiment 008: streaming context exposes the practical limit

The [LFM2.5-230M model card](https://huggingface.co/LiquidAI/LFM2.5-230M)
lists **32,768 tokens** of context. Its config permits larger position IDs,
but that setting alone does not establish a larger supported context. Our
Phoenix NPU chat runner's previous 64/96-position ceiling was an
implementation limit: its attention cache had to fit within one compute
tile's local data memory. AMD describes XDNA as a tiled dataflow architecture
with explicit streams and local memories; the AIE-ML tile's data memory is
[64 KB](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Array-Features).

The experimental `chunked` mode streams the attention KV cache through that
tile in 64-token blocks. Attention scores, online softmax accumulation, KV
append, and context output are computed on the NPU. The host copies a cache
buffer only when it grows by another 64-token block; this is data handling,
not model arithmetic. The model still uses NPU recurrent and projection
programs, NPU embedding selection, and the NPU vocabulary head. There is no
CPU attention or CPU/NPU model-layer split.

## Run the experiment

From the repository root on the configured laptop:

```powershell
& .\scripts\chat_lfm25_npu.ps1 -CacheMode chunked
```

The normal launcher continues to default to `fixed64` because it is faster
for short chats. Chunked mode currently caps the transcript at **4,096
positions**. This is a software cap for the experiment, not a claim that
4,096-token end-to-end chat is practical or fully validated. Full model
inference was checked through 145 prompt tokens; a single attention program
was separately compiled and run at 4,096 tokens.

## Measured correctness and speed

| Check | Phoenix NPU | Matched CPU BF16 reference |
| --- | ---: | ---: |
| 73-token prompt, three selections | **34.45 s**; IDs `560, 985, 510` | **2.45 s**; same IDs. |
| 145-token prompt, three selections | **71.93 s**; IDs `560, 985, 510` | **4.29 s**, eight threads, one token at a time; same IDs. |
| Two turns crossing the old 64-token limit | First turn: 55 prompt tokens and four selections in **27.35 s**. Second turn: 74 prompt tokens total, only 16 newly processed, four selections in **11.94 s**. NPU state reuse succeeded. | Not timed for this two-turn transcript. |

The 145-token comparison includes prompt processing and three selections in
both implementations. It compares two different kernels with matched BF16
precision, input tokens, and sequential input schedule; it is not a general
CPU or NPU performance ceiling. The NPU's selected tokens match the CPU
reference on that transcript, but its intermediate tensors were not
exhaustively compared.

The isolated chunked-attention benchmark used a synthetic BF16 KV cache,
one query, a warmed compiled program, and three timed calls per size. Each
figure includes Python/XRT submission, data movement, kernel execution, and
output synchronization for **one of the model's six attention layers**:

| Active context | KV blocks | Median per attention layer |
| ---: | ---: | ---: |
| 64 | 1 | 10.64 ms |
| 128 | 2 | 19.71 ms |
| 256 | 4 | 37.55 ms |
| 512 | 8 | 73.34 ms |
| 1,024 | 16 | 145.15 ms |
| 2,048 | 32 | 289.10 ms |
| 4,096 | 64 | **576.26 ms** |

At 4,096 positions, six such calls would consume about **3.46 s for
attention alone per next token**, before the recurrent layers, projections,
embedding, and vocabulary head. Extrapolating the measured near-linear
per-token attention cost across a 4,096-token prompt gives **roughly two
hours of attention work for prefill alone** in this serial implementation.
That is an estimate, not a timed full-model 4K run. The cache itself is
about 8 MiB per attention layer at 4K, or 48 MiB across six layers; memory
capacity is not the immediate obstacle. Streaming and repeatedly scanning
that memory is.

## Why low watts do not guarantee useful chat efficiency

In the earlier matched 23-position battery experiment, the NPU prototype
drew about **2.87 W incremental whole-laptop power** versus **6.22 W** for
CPU, but took **13.21 s** versus **1.19 s**. That gave about **37.85 J** for
the NPU pass versus **7.43 J** for the CPU pass. These are exploratory
whole-laptop battery readings, not device-only rail measurements. No
long-context battery measurement was made. Lower instantaneous power can
still mean more energy when execution takes far longer; see
[experiment 006](006_lfm25_full_npu.md) for the raw method and caveats.

The current LFM2.5 implementation makes many small, sequential program
submissions. It streams large model weights and the tied vocabulary table
repeatedly, while six full-attention layers rescan a growing KV history.
The CPU handles small control-heavy steps more cheaply, and the RTX GPU can
hold the whole 230M model in its own memory and execute highly parallel
matrix operations. The NPU's dataflow design can be efficient when a
well-shaped, reused pipeline keeps its tiles busy, as the earlier matrix
experiments show. This chat path does not yet achieve that.

For a useful NPU-only long-chat implementation, the next work would be
batched prompt processing, parallel attention heads across tiles, fewer
program switches, better weight reuse or quantization, and a prompt
embedding lookup that does not scan the full table. Each needs correctness
and whole-chat speed checks. Reaching a larger *addressable* context does
not itself solve the latency or energy problem. Until then, CPU or GPU
inference is the practical way to chat with this model on this laptop.

Reproduce the isolated measurement with
[`008_chunked_attention_benchmark.py`](008_chunked_attention_benchmark.py),
the matched CPU transcript with
[`008_long_prompt_reference.py`](008_long_prompt_reference.py), and the
two-turn boundary check with
[`008_multi_turn_check.py`](008_multi_turn_check.py). All downloads and
compiled binaries remain in ignored `cache/`; the working Ryzen AI
installation and NPU driver were preserved.
