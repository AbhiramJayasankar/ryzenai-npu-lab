# Experiment 009: can a short LFM2.5 prompt prefill beat the CPU on Phoenix NPU?

## Result so far

**No.** The correct all-NPU chat implementation selected the same first token as
the CPU, but its warm 34-token prefill took **13,695 ms**, compared with **42.3 ms**
for the fastest measured whole-prompt BF16 CPU run. This is a **324× latency
gap** for the current implementation. The input text is exactly 20 words, but
the chat template and tokenizer produce 34 model tokens.

I built a batched four-core NPU projection that processes all 34 useful
positions together. Reusing each BF16 weight tile across three 16-row prompt
tiles reduced one real `1024 → 3072` projection from **4.15 to 2.95 ms**. The
same projection took **0.683 ms** on the eight-thread CPU with BF16-rounded
inputs and FP32 output. The NPU result matched an FP32 reference within
`3.28e-7` maximum absolute error. This is a verified stage improvement, not
a faster full-model prefill.

| Operation | Device and implementation | Warm median |
| --- | --- | ---: |
| Full 34-token prefill and first token | BF16 PyTorch CPU, 8 threads | 42.3 ms |
| Full 34-token prefill and first token | Current all-NPU IRON chat | 13,695 ms |
| First layer input projection, 34 useful positions | BF16-rounded input, FP32 CPU, 8 threads | 0.683 ms |
| Same projection, 48 rows padded, four NPU tiles | BF16/FP32, weight repeated for each 16-row tile | 4.15 ms |
| Same projection, 48 rows padded, four NPU tiles | BF16/FP32, weight reused across all three tiles | 2.95 ms |

The projection comparison is deliberately narrow. The CPU executes 34 rows;
the NPU executes 48 because the current kernel uses 16-row matrix tiles. The
CPU stage converts BF16-rounded inputs and weights to FP32 to retain FP32
outputs. The full CPU model uses BF16 PyTorch operations. Both full-model
paths chose token ID `2797`. A separate tokenizer check confirmed that the
two prompt paths have identical 34-token ID sequences. Matching the first
token does not prove every intermediate is bit-identical.

## Why this is hard

The current NPU chat processes the prompt one token at a time. At each model
position it scans the 128 MiB embedding table, sends every recurrent or
attention block its weights again, and starts many separate NPU programs. A
whole-prompt CPU call instead batches matrix work across positions and can
reuse a streamed weight block for the batch. The current NPU path therefore
pays repeated memory traffic and program submission costs.

LFM2.5-230M has eight convolutional blocks and six attention blocks in this
implementation. Counting five dense projections in each convolutional block
and seven in each attention block gives 82 projection operations before the
final vocabulary head. Their shapes differ, so multiplying the measured
2.95 ms stage by 82 is **not** a valid prediction of the complete design. It
does show why replacing the current token loop with individually dispatched
batched projections is insufficient for a 42 ms target. The next design must
fuse work at least at the layer level, avoid rereading weights for each
position, and schedule multiple NPU tiles with shared DMA streams. Causal
convolution state and attention KV state must remain numerically correct on
the NPU. Only the final prompt position needs a vocabulary-head evaluation.

Naively assigning a separate input, weight, and output FIFO to each of eight
or sixteen cores did not compile: the Phoenix shim or memory tile DMA channel
budget was exhausted. More cores need data broadcast or fan-out through
memory tiles rather than one set of host-facing DMA streams per core. AMD's
[NPU1 device layout](https://github.com/Xilinx/mlir-aie/blob/main/docs/Devices.md)
shows four columns and four compute rows. The [IRON programming examples](https://github.com/Xilinx/mlir-aie/blob/main/programming_examples/README.md)
and [matmul example](https://github.com/Xilinx/mlir-aie/blob/main/programming_examples/basic/matrix_multiplication/single_core/single_core.py)
show the explicit FIFO and DMA approach used here.

## Reproduce

Run from the repository root. Local model weights and toolchain remain in
ignored `cache/`; no driver, firmware, or Ryzen AI installation was changed.

```powershell
& .\cache\lfm-env\Scripts\python.exe experiments\009_prefill_cpu_baseline.py
& .\cache\lfm-env\Scripts\python.exe experiments\009_prefill_projection_cpu.py
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 | Out-Null
. .\cache\iron\mlir-aie\iron_env.ps1
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\009_prefill_npu_baseline.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\009_prefill_projection_probe.py
& .\cache\iron\mlir-aie\ironenv\Scripts\python.exe experiments\009_prefill_projection_probe.py --reuse-weights
```

CPU whole-prompt result: 1/2/4/8/16 thread medians were
102.8/68.7/48.2/42.3/51.3 ms over five warmed calls each. NPU full-prompt
samples were 14,600/13,695/13,666 ms in one process. Four-core projection
samples were 4.153/4.223/4.044/3.874/4.564 ms without reuse and
3.153/2.909/2.952/2.941/2.955 ms with reuse. The machine was on AC at the
end of these measurements; power draw was not measured.

## Next experiment

Build one complete **layer-wise batched** NPU block for a 34-token prompt:
RMSNorm, all projections, recurrent convolution or causal attention, FFN, and
residual with the state retained on NPU. Share the weight stream across prompt
rows and across cores. Measure its total submission, transfer, and compute
latency against the equivalent CPU block. If that block cannot approach the
per-layer budget implied by 42 ms for fourteen layers, the full-model target
will require a different execution architecture or quantization strategy.
