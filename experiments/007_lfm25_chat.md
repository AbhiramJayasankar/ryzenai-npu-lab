# Experiment 007: short interactive chat on the Phoenix NPU

[`007_lfm25_npu_chat.py`](007_lfm25_npu_chat.py) turns the fixed-sequence
[experiment 006](006_handoff.md) into a command-line chat runner. It accepts
new text prompts, renders the model's text chat format, tokenizes them,
computes rotary constants, and runs all LFM2.5-230M model calculations on
`NPU1`. It greedily selects and decodes reply tokens. A PowerShell launcher
loads the existing Visual Studio and IRON environment. It uses the local
checkpoint and tokenizer in ignored `cache/lfm25-230m/`; model weights are
not shipped in this repository.

## Run it on the configured laptop

From the repository root:

```powershell
# One-time dependency in the isolated IRON environment:
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe -m pip install -r experiments\007_requirements.txt

# Interactive text-only chat, with /quit to leave:
& .\scripts\chat_lfm25_npu.ps1

# One prompt with machine-readable output:
& .\scripts\chat_lfm25_npu.ps1 -Prompt 'Hello!' -MaxNewTokens 8 -Json
```

Default `fixed64` mode uses one reusable attention-cache program and can
process at most **64 total positions** per turn, including the prompt and
generated tokens already fed back. The runner rejects a too-long first
message and drops the oldest user/assistant pair when a later turn would
exceed that limit. In one interactive process, NPU state is reused when the
next transcript begins with the exact same token IDs already consumed.
Otherwise the retained conversation is rebuilt, including after old turns
are dropped. State does not persist across launches. For a longer but
slower and less tested path, use `-CacheMode variable`, which currently
allows **96 total positions**. Prepare its length-specific binaries once
with `& .\scripts\prepare_lfm25_chat.ps1 -Through 96`; compilation may take
several minutes and writes only to IRON's ignored disk cache. The fixed
cache is the practical default because previously unseen variable lengths
otherwise compile during a chat turn.

An experimental `-CacheMode chunked` path accepts longer transcripts by
streaming 64-token KV blocks through the NPU. Its measured latency grows
sharply with context, so it is not the default. See
[experiment 008](008_long_context_limits.md) for correctness and speed data.

This runner does **not** require CPU model reference files at inference time.
The CPU handles tokenizer work, rotary constant preparation, and NPU program
submission. A runtime token-ID tensor enters a custom NPU embedding program,
which selects the BF16 embedding row; no CPU/NPU layer split is used. On this
Phoenix static instruction path, a dynamic DMA row offset did not compile,
so the current embedding program streams the full table past an NPU core for
each prompt token. Warmed isolated calls took about **25 ms each** across four
different token IDs. This is correct but leaves substantial speed headroom.

## Hardware checks

| Check | Observed result |
| --- | --- |
| Known 21-token prompt, three selections | IDs **2797, 38785, 562**, matching the sequential CPU BF16 reference; output text `An NPU` after three tokens. The default 64-position path took **9.20 s** on a later run with binaries cached. |
| Different prompt `Hello!`, eight selections | NPU IDs **36309, 510, 2213, 1011, 859, 1801, 1010, 4008**; text `Hello! How can I help you today`. A separate sequential CPU BF16 run selected the exact same eight IDs. |
| Two interactive turns, four tokens per reply | `Hello!` followed by `What is an NPU?` produced `Hello! How can` and `An NPU (`. With NPU state reuse, the fixed64 turns took **5.99 s** and **7.88 s**; the second turn processed only 17 new prompt tokens. Before state reuse, it took 13.6 s. The variable-cache second turn took **60.7 s** when it had to compile unseen lengths. |
| Prompt near the fixed-cache limit | A 55-token prompt plus three selected tokens completed in **23.47 s**, returning `Sure!`. |
| Prompt beyond the fixed-cache limit | After preparing variable-length binaries through 96 positions, a 73-token prompt plus three selections completed on `NPU1` in **36.74 s**, returning `Sure!`. |

Times are Python/XRT wall times on AC and vary with compilation cache,
program context switching, system state, and length. The known prompt's
full-run numeric comparison remains in experiment 006; arbitrary prompts
above were checked by token selection, not by every hidden/KV value.
This is an interactive **short-context prototype**. It is not yet CPU/GPU
competitive, and it supports text, greedy decoding, and EOS stopping only.
The 64/96-position limits come from the present custom attention cache
implementations, not the model's advertised context length.
Preparing all 95 variable-length context programs took **370.5 s** on this
laptop; that compilation is a one-time local cache cost.

## Why these implementation choices matter

- The one-program recurrent block and fused attention stages keep model
  arithmetic and evolving state on the NPU. The four-core output head also
  returns the next embedding, so decode tokens feed back without CPU lookup.
- One attention program per cache length works through 96 positions, but
  compilation on first use pauses interaction. At 96 prior tokens, the
  current program exceeds Phoenix tile memory and fails to compile. A fixed
  64-position cache reuses one binary and removes that compilation cliff for
  short turns.
- The fixed cache transfers padded entries even early in the prompt; it is
  not a general long-context solution. Retained transcript tokens are
  replayed when the token prefix changes or the context window is trimmed.
- The next speed steps are multi-core recurrent/attention projection
  programs, less weight streaming and program switching, and a dynamic
  embedding selection that avoids scanning the entire tied table. Longer
  context needs chunked KV storage and attention rather than a larger
  single-tile fixed cache.

The working Ryzen AI installation, IRON/XRT toolchain and NPU driver were
kept in place. The chat runner and new kernels use the separate ignored IRON
environment; downloaded model assets and compiled binaries stay out of Git.
