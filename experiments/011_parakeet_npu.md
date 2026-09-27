# Experiment 011: Parakeet TDT 0.6B v2 speech-to-text on the Phoenix NPU

Can NVIDIA's Parakeet TDT 0.6B v2 (English ASR, FastConformer encoder + TDT
transducer decoder, ~600M parameters) run on the Ryzen 9 8945HS NPU, and how
does it compare with the RTX 4060 Laptop GPU and the CPU? State as of
2026-09-27.

## Short answer

* The Ryzen AI route (Quark quantization + Vitis AI execution provider) does
  not work for this model on Phoenix. Only the XINT8 scheme (power-of-two
  scales) puts any operator on the NPU. Even then LayerNorm, the convolution
  module and the attention score matmuls stay on the CPU, so each Conformer
  layer is split into about 10 NPU pieces. The NPU pieces are slower than the
  CPU running the FP32 model, and XINT8 loses too much accuracy (38%
  relative error after only two of 24 layers). A8W8 and A16W8 compile to zero
  NPU operators.
* The custom route works and is accurate. Every linear layer of the encoder
  (95% of its arithmetic) runs on the NPU through an IRON BF16 GEMM with
  FP32 accumulation, and the rest runs in NumPy on the CPU. On 210
  LibriSpeech test-clean utterances it gives the same WER as the FP32 model
  (2.19%, identical word sequences for all 210, 4 punctuation differences).
* It is not faster. The hybrid encoder takes 520 ms for 10 s of audio. The CPU
  runs the FP32 encoder in 496 ms and the dynamic-int8 one in 411 ms; the GPU
  takes 26 ms. Its advantage is that it keeps about 1.8 CPU cores busy instead
  of 8: other work on the laptop ran at 96% (1 thread) and 87% (8 threads) of
  idle speed during the NPU hybrid, versus 75% and 73% during CPU FP32.
* For actual use on this laptop, keep Parakeet on the GPU (50-110x real time
  end to end). The NPU only makes sense if the GPU is unavailable or must stay
  free, and then only after more engineering (see "What it would take").

## Results

All numbers on AC power, Windows 11 26200, NPU driver 32.0.203.280, Ryzen AI
1.7.0, IRON/MLIR-AIE 1.4.3, onnxruntime-gpu 1.30 (CUDA 13), onnxruntime 1.30
CPU. Transcription uses the same pipeline for every backend
([`parakeet_pipeline.py`](parakeet_pipeline.py)): nemo128 preprocessor on
the CPU, encoder on the backend under test, TDT greedy decoding on the CPU. It
reproduces onnx-asr 0.12 exactly (same transcript on `test.wav`).

### Accuracy and throughput on 210 LibriSpeech test-clean utterances

30 speakers, 28.8 minutes, 2.0-30.6 s per utterance. WER uses
LibriSpeech-style normalization (lower case, punctuation removed, digits
spelled out). "Real-time factor" is audio seconds per processing second.

| Backend | WER | Encoder time (all 210) | Encoder x real time | End to end x real time | Word diff vs CPU FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU FP32 (8 cores) | 2.19% | 90.9 s | 19 | 16 | reference |
| CPU dynamic int8 (`encoder-model.int8.onnx`) | 2.26% | 83.6 s | 21 | 17 | 0.39% |
| GPU FP32 (RTX 4060, exact lengths) | 2.19% | 21.5 s | 80 | 48 | 0 |
| GPU FP32, input padded to 31 s | 2.19% | 15.4 s | 112 | 60 | 0 |
| NPU hybrid (IRON BF16 GEMMs + CPU NumPy) | 2.19% | 120.2 s | 14 | 13 | 0 |

The published WER for this model on the full test-clean set is 1.69%; this
subset and the simple normalization give 2.19%. The comparison between
backends is what matters here. The GPU runs faster with one fixed padded shape
because every new input length costs the CUDA EP extra setup. Decoding (TDT
loop on the CPU) takes 7-10 s of the total for every backend.

### Encoder latency at fixed lengths

Same input for all (test.wav tiled, 10 warm calls, median).
[`011_encoder_bench.py`](011_encoder_bench.py), [`011_hybrid_bench.py`](011_hybrid_bench.py).

| Audio | CPU FP32 | CPU dynamic int8 | GPU FP32 | NPU hybrid |
| ---: | ---: | ---: | ---: | ---: |
| 10 s | 496 ms | 411 ms | 26 ms | 520 ms |
| 20 s | 1008 ms | 835 ms | 40 ms | 863 ms |
| 30 s | 1542 ms | 1290 ms | 57 ms | 1562 ms |
| CPU cores busy | 7.9-8.0 | 7.9-8.0 | 4.8-7.7 (ORT spin-waiting) | 1.8-1.9 |

Where the NPU hybrid's time goes:

| Audio | Total | NPU GEMMs (552 calls) | BF16 staging + sync | Readback | CPU NumPy | Prefix (ORT) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 s | 520 ms | 221 ms | 62 ms | 24 ms | 205 ms | 9 ms |
| 20 s | 863 ms | 308 ms | 79 ms | 63 ms | 396 ms | 18 ms |
| 30 s | 1562 ms | 397 ms | 104 ms | 204 ms | 830 ms | 27 ms |

### Effect on other work

[`011_contention_run.py`](011_contention_run.py) loops each backend on a 10 s
window and meanwhile measures SHA-256 throughput of a foreground workload
([`011_contention_probe.py`](011_contention_probe.py)), relative to idle.

| Backend running | 1-thread workload | 8-thread workload |
| --- | ---: | ---: |
| none (idle, after) | 99% | 99% |
| CPU FP32 | 75% | 73% |
| CPU dynamic int8 | 82% | 77% |
| GPU FP32 | 88% | 76% |
| NPU hybrid | 96% | 87% |

The GPU run still loads the CPU because ONNX Runtime's thread pool spins
between calls; turning spinning off would likely help it. Power was not
measured (the laptop was on AC; `scripts/measure_battery_run.ps1` from
experiment 006 is the method to use on battery).

## Model and data

* Model: `istupakov/parakeet-tdt-0.6b-v2-onnx`, loaded from the plain folder
  `projects/parakeet-stt/model` (HF-cache symlinks break ONNX Runtime's
  external-data check). Encoder `encoder-model.onnx` + 2.4 GB `.data` (FP32),
  `encoder-model.int8.onnx` (dynamic int8, 652 MB), `decoder_joint-model.onnx`
  (36 MB), `nemo128.onnx`, `vocab.txt`.
* Encoder: 128 mel features, 8x subsampling by three stride-2 convolutions
  (256 channels), then 24 Conformer layers with d=1024, 8 heads of 128,
  relative-position attention, FFN 1024-4096-1024 twice per layer (half-step
  residuals), a convolution module (pointwise 1024->2048, GLU, depthwise
  kernel 9 with batch norm folded in, Swish, pointwise 1024->1024). 10 s of
  audio is 1000 feature frames and 125 encoder frames. About 1 GMAC per
  encoder frame; 95% of it is in the linear layers.
* Data: LibriSpeech test-clean from openslr.org/12 (SHA-256
  `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`),
  converted by [`011_prepare_data.py`](011_prepare_data.py): 210 evaluation
  utterances from 30 speakers and 60 calibration utterances from 10 other
  speakers.

## Part 1: Ryzen AI / Vitis AI execution provider

Steps: make the encoder static ([`011_make_static.py`](011_make_static.py),
fixed 1000 frames, ORT basic optimizations fold the shape arithmetic),
split it into blocks ([`011_split.py`](011_split.py)), quantize with Quark
([`011_quantize.py`](011_quantize.py)), compile and run a block with the
operator report of experiment 004 ([`011_npu_block.py`](011_npu_block.py)).

Splitting was necessary. The FP32 encoder is 2.4 GB, and Quark calls
`ModelProto.ByteSize()`, which raises "Failed to serialize proto" above 2 GB
with this protobuf build, so the whole model cannot be quantized in one piece.
The tests below use layers 0-1 (195 MB), with 16 calibration utterances run
through the FP32 subsampling prefix.

Layers 0-1, 10 s input, error on the real frames against the FP32 block:

| Quark config | Nodes on NPU / CPU | CPU operators | Warm latency | Error of quantized model (CPU) | Error on NPU |
| --- | ---: | --- | ---: | ---: | ---: |
| FP32 on CPU (reference) | - | - | 36-38 ms | - | - |
| XINT8 (power-of-two scales, Sigmoid -> HardSigmoid) | 288 / 148, 20 NPU subgraphs | LayerNorm, Conv (pointwise + depthwise), attention MatMuls, Where, Q/DQ | 88-93 ms | cosine 0.927, rel. RMS 0.38 | cosine 0.881, rel. RMS 0.48 |
| A8W8 | 0 / 422 | everything | 48 ms | cosine 0.942, rel. RMS 0.34 | same (CPU) |
| A16W8 | 0 / 428 | everything | 60 ms | cosine 0.975, rel. RMS 0.22 | same (CPU) |

The ORT profile of the XINT8 block puts 79 of its 93 ms inside the NPU
partitions themselves, so the CPU fallbacks are not the main cost. A
single-MatMul probe ([`011_ep_matmul_probe.py`](011_ep_matmul_probe.py))
confirms it: [M x 1024] x [1024 x 4096] through the EP runs at 45-90 GMAC/s on
the NPU, while the CPU runs the same MatMul at 190-235 GMAC/s in FP32 and
760-845 GMAC/s as static int8. The Phoenix NPU compiler in Ryzen AI 1.7 is
built for INT8 CNNs (AMD's compatibility table lists nothing else for this
chip) and handles transformer-shaped MatMuls poorly.

The quantization error is a second, separate problem. Relative errors of
22-38% after two layers would compound over 24 layers. Likely causes are
per-tensor activation scales in a residual stream with large outliers and the
-10000 attention mask constant sharing a quantization range with the scores.
This was not pursued, because the speed result already rules the route out.
End-to-end transcription through the EP was not run for the same reason.

## Part 2: custom IRON GEMMs (hybrid encoder)

### NPU matmul throughput

[`011_iron_gemm_probe.py`](011_iron_gemm_probe.py) runs IRON's whole-array
GEMM (`programming_examples/basic/matrix_multiplication/whole_array`: 16
cores, `aie::mmul`) at Parakeet shapes through direct pyxrt submission.
Results were checked against NumPy (exact for int8; FP32 accumulation
matches to 1e-6).

| Shape (M x K x N) | Types | NPU time | GMAC/s |
| --- | --- | ---: | ---: |
| 128 x 1024 x 4096 (activations as A) | BF16 -> BF16 | 1.84 ms | 292 |
| 384 x 4096 x 1024 | BF16 -> BF16 | 1.69 ms | 954 |
| 384 x 4096 x 1024 | int8 -> int32 | 1.47 ms | 1093 |
| 1024 x 1024 x 128 (weights as A) | BF16 -> FP32 | 0.34 ms | 396 |
| 4096 x 1024 x 128 (weights as A) | BF16 -> FP32 | 1.12 ms | 480 |
| 1024 x 4096 x 128 (weights as A) | BF16 -> FP32 | 0.84 ms | 636 |

For reference, ONNX Runtime on the CPU reaches about 200 GMAC/s in FP32. Two
details mattered:

* Operand order. The design streams A once and re-streams B for every
  4-row-tile block of output rows. With the weights as B and 128 frames, each
  weight crossed DDR twice. Putting the weights in A (as W^T), the
  activations in a column-major B and writing C column-major gives Y in
  row-major order with every weight read once.
* Output type. BF16 output makes the kernel round the running sum to BF16
  after every 64-wide K step (errors up to 12% of the maximum). FP32 output
  removes that.

### Hybrid encoder

[`parakeet_hybrid.py`](parakeet_hybrid.py):

* The subsampling prefix (1% of the FLOPs) runs in ONNX Runtime with the
  exact input length (extracted from the dynamic export). A fixed-length
  prefix leaks the padding into the last frame through the convolution
  biases, so exact lengths are needed to match the reference.
* Each Conformer linear is cut into 1024x1024 blocks (23 per layer, 552 per
  encoder call). One compiled GEMM program [1024 out x 1024 in] x [1024 in x
  rows] serves all of them from one hardware context, avoiding the ~0.5 ms
  context switch measured in experiment 010. Programs for 128, 256 and 384
  frame rows (10.2, 20.4, 30.7 s) share one copy of the weights: 1.1 GB of
  BF16 in 24 XRT buffers, packed in about 5 s at load. FFN linear2 (K=4096) is
  four blocks whose partial sums the CPU adds.
* The CPU does LayerNorm, Swish, GLU, relative-position attention (scores,
  NeMo's rel_shift, softmax), the depthwise convolution and residuals in
  FP32 NumPy, spread over 8 threads. BLAS stays single-threaded because
  OpenBLAS with its default 16 threads was about 20x slower than with 8 on
  this CPU.
* Weights come from [`011_export_weights.py`](011_export_weights.py);
  relative-position tables from [`011_export_pos.py`](011_export_pos.py).

### Correctness

[`011_hybrid_validate.py`](011_hybrid_validate.py):

| Check | Cosine | Rel. RMS | Max abs |
| --- | ---: | ---: | ---: |
| Layers 0-1, NumPy FP32 vs ORT FP32 | 1.000000 | 0.0000 | 0.0001 |
| Layers 0-1, NumPy with BF16-rounded matmul inputs vs ORT | 0.999999 | 0.0017 | 0.16 |
| Layers 0-1, NPU hybrid vs ORT | 0.999999 | 0.0017 | 0.14 |
| Full encoder, 4 clips of 7-10 s, NPU hybrid vs ORT FP32 | 0.999992-0.999994 | 0.0034-0.0041 | 0.003 |
| Full encoder, 30.6 s clip | 0.999937 | 0.011 | 0.066 |

The NPU result matches the BF16-input emulation, so the NPU computes the
expected products and the remaining error is BF16 rounding of the inputs.
On the full evaluation set the transcripts give identical word sequences
to CPU FP32; 4 of 210 differ in one comma.

### Why it is not faster

* The GEMMs alone take 221 ms per 10 s window. Each call moves a 2 MB weight
  block at about 6 GB/s plus fixed submission cost. The whole-array design
  uses one shim DMA channel per column for weights (4 of the 8 channels
  measured at ~27 GB/s in experiment 010).
* Every encoder call streams all 1.1 GB of weights however short the audio,
  so a 2 s utterance costs about 430 ms. The CPU's time scales with length.
* The CPU half (NumPy) costs as much as the NPU half and grows with the
  square of the length in attention. NumPy creates a temporary array per
  operation; a C++ implementation or ONNX Runtime subgraphs would be several
  times faster.
* The NPU and CPU halves run one after the other, never overlapped.

## What it would take

Estimates from the measurements above, not built:

| Version | 10 s encoder (estimate) | CPU use |
| --- | ---: | --- |
| This hybrid | 520 ms (measured) | ~1.8 cores |
| Hybrid with CPU half in C++/ORT, larger GEMM calls (FFN projections in one call each, ~6.2 ms/layer of GEMM) | ~250-300 ms | ~1-2 cores during calls |
| All-NPU encoder in the style of experiment 010: one submission per layer, weights over all 8 DDR channels, LayerNorm/Swish/softmax/depthwise conv as AIE vector kernels | ~100-150 ms; floor 44 ms (1.2 GB BF16 at 27 GB/s) or 22 ms with int8 weights | nearly none |
| For comparison: CPU dynamic int8 / GPU | 411 ms / 26 ms | 8 cores / GPU |

An all-NPU encoder is a project of the size of experiment 010 (many custom
kernels: relative-position attention with rel_shift, depthwise convolution,
LayerNorm, per-layer orchestration within 16 KB of program memory per core).
At best it would make the NPU about 3-4x faster than the CPU, with the CPU
free, and still 4-6x slower than the GPU. Worth doing only if a GPU-free,
low-CPU speech-to-text path is a goal in itself; power would have to be
measured to make the energy case.

## Reproduce

Environments: the parakeet-stt venv (`..\parakeet-stt\.venv`, CUDA/CPU
baselines), the Ryzen AI Conda env
(`$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe`, Quark and the
Vitis AI EP), and the IRON env through `scripts\iron_python.ps1` /
`scripts\run_npu_timeout.ps1`. The IRON venv needs ONNX Runtime for the
prefix and decoder: `cache\iron\mlir-aie\ironenv\Scripts\python -m pip install onnxruntime`
(1.30.0 was installed for this experiment). From the repository root:

```powershell
$venv = "..\parakeet-stt\.venv\Scripts\python"
$ryzen = "$env:USERPROFILE\miniforge3\envs\ryzen-ai-1.7.0\python.exe"
# data: download test-clean.tar.gz from openslr.org/12 into cache\librispeech and extract
& $venv experiments\011_prepare_data.py
# baselines
& $venv experiments\011_transcribe_set.py --provider cpu --tag cpu_fp32
& $venv experiments\011_transcribe_set.py --provider cpu --encoder encoder-model.int8.onnx --tag cpu_int8dyn
& $venv experiments\011_transcribe_set.py --provider cuda --tag gpu_fp32
& $venv experiments\011_encoder_bench.py --provider cpu --seconds 10 20 30 --tag cpu_fp32
# Vitis AI EP route
& $ryzen experiments\011_make_static.py --frames 1000
& $ryzen experiments\011_split.py --model cache\parakeet\models\encoder_T1000.onnx --blocks prefix 0-1
& $ryzen experiments\011_quantize.py --model cache\parakeet\models\encoder_T1000_0_1.onnx --config XINT8 --calib 16 --frames 1000 --upstream cache\parakeet\models\encoder_T1000_prefix.onnx
& $ryzen experiments\011_npu_block.py --qmodel cache\parakeet\models\encoder_T1000_0_1_XINT8.onnx --fp32 cache\parakeet\models\encoder_T1000_0_1.onnx --upstream cache\parakeet\models\encoder_T1000_prefix.onnx --profile
& $ryzen experiments\011_ep_matmul_probe.py
# custom IRON route
& $ryzen experiments\011_export_weights.py --model cache\parakeet\models\encoder_T1000.onnx
& $ryzen experiments\011_export_pos.py --frames 3072
& $ryzen experiments\011_split.py --model ..\parakeet-stt\model\encoder-model.onnx --blocks prefix --out-stem cache\parakeet\models\encoder_dyn
& $ryzen experiments\011_reference_outputs.py --model cache\parakeet\models\encoder_T1000.onnx
& $ryzen experiments\011_reference_outputs.py --model encoder-model.onnx
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\011_iron_gemm_probe.py -ScriptArgs "10" -Seconds 900
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\011_hybrid_validate.py -Seconds 2400
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\011_hybrid_bench.py -Seconds 1200
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\011_hybrid_transcribe.py -Seconds 3600
& $venv experiments\011_contention_run.py
```

`011_iron_gemm_probe.py` takes one config index per process (0-13); running
many hardware contexts in one process crashed pyxrt. Local outputs go to the
ignored `cache/parakeet/` (models 2.7 GB, weights 2.3 GB); summaries of every
measurement are in [`011_results.json`](011_results.json).

## Files

| File | Role |
| --- | --- |
| [`parakeet_pipeline.py`](parakeet_pipeline.py) | Preprocessor, encoder (any EP or the hybrid), TDT greedy decoding, WER |
| [`parakeet_hybrid.py`](parakeet_hybrid.py) | Hybrid encoder: NumPy Conformer layer, `NpuWeights`, `NpuMatmul`, `HybridEncoder` |
| [`011_prepare_data.py`](011_prepare_data.py) | LibriSpeech subset selection and WAV conversion |
| [`011_transcribe_set.py`](011_transcribe_set.py), [`011_hybrid_transcribe.py`](011_hybrid_transcribe.py) | WER and speed on the evaluation set |
| [`011_encoder_bench.py`](011_encoder_bench.py), [`011_hybrid_bench.py`](011_hybrid_bench.py) | Encoder latency at 10/20/30 s |
| [`011_make_static.py`](011_make_static.py), [`011_split.py`](011_split.py), [`011_quantize.py`](011_quantize.py), [`011_npu_block.py`](011_npu_block.py), [`011_ep_matmul_probe.py`](011_ep_matmul_probe.py) | Vitis AI EP route |
| [`011_export_weights.py`](011_export_weights.py), [`011_export_pos.py`](011_export_pos.py), [`011_reference_outputs.py`](011_reference_outputs.py), [`011_iron_gemm_probe.py`](011_iron_gemm_probe.py), [`011_hybrid_validate.py`](011_hybrid_validate.py) | Custom route: weights, references, GEMM probe, validation |
| [`011_contention_probe.py`](011_contention_probe.py), [`011_contention_run.py`](011_contention_run.py) | Effect on other work |
| [`011_results.json`](011_results.json) | Summaries of all measurements |

## Limits

* One laptop, AC power, other apps open (Chrome, ChatGPT, WhatsApp and
  several Claude sessions); the Bonsai llama-server was stopped to free
  memory. CPU timings vary by about 5% between runs.
* WER on a 210-utterance subset with a simple normalizer; backend
  differences are the reliable part.
* The Vitis AI results are for Ryzen AI 1.7.0 with driver 32.0.203.280 on
  Phoenix. Newer software or an XDNA 2 (Strix) NPU may behave differently.
* Power and energy per utterance were not measured.
