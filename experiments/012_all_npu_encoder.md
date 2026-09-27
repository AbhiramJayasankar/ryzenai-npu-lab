# Experiment 012: the Parakeet TDT 0.6B v2 encoder entirely on the Phoenix NPU

Follow-up to [experiment 011](011_parakeet_npu.md), where the Ryzen AI route
failed and a hybrid (NPU matmuls, CPU NumPy for the rest) only matched CPU
speed. Here all 24 Conformer layers of the encoder run on the NPU: 16 cores,
one custom IRON core program, no CPU work inside the encoder layers. State as
of 2026-09-27; for continuing the work read the
[experiment 012 handoff](012_handoff.md).

## Result

**The whole encoder runs on the NPU with exactly the FP32 model's word error
rate, 2 to 2.7 times faster than the CPU, and ran every test without a hang**
after the task-queue fix described below. Three programs cover utterances up
to 10.2, 20.5 and 30.7 s (1, 2 and 3 blocks of 128 encoder frames); the mel
front end, subsampling prefix and TDT decoding stay on the CPU.

WER on the 210 LibriSpeech test-clean utterances of experiment 011, every
utterance transcribed whole by the smallest program that fits
([`012_transcribe.py`](012_transcribe.py), [`012_transcribe_long.py`](012_transcribe_long.py),
NPU driver 32.0.203.376), against the ONNX Runtime FP32 encoder on the CPU:

| Program (max. audio) | Utterances | NPU WER | CPU FP32 WER | NPU encoder, median | CPU FP32 encoder |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 block (10.2 s) | 152 | 2.57% (56 errors) | 2.57% (56) | 185 ms | 496 ms at 10 s |
| 2 blocks (20.5 s) | 47 | 1.82% (32) | 1.82% (32) | 428 ms | 1008 ms at 20 s |
| 3 blocks (30.7 s) | 11 | 1.92% (12) | 1.92% (12) | 765 ms | 1542 ms at 30 s |
| **All** | **210** | **2.19% (100 errors)** | **2.19% (100)** | | |

205 of 210 transcripts are character-identical to FP32; the word-level
difference between the two is 0.02%. Encoder time for all 28.8 minutes of
audio: 59.8 s on the NPU (29 times real time) versus 90.9 s on the CPU
(19x) and 15-22 s on the RTX 4060 GPU (80-112x).

| Encoder time per 10 s window | Time | vs NPU |
| --- | ---: | ---: |
| **NPU (this experiment)** | **185 ms** median (175-205, 277 windows) | 1x |
| CPU FP32 (ONNX Runtime, 8 cores) | 496 ms | 2.7x slower |
| CPU dynamic int8 | 411 ms | 2.2x slower |
| NPU + CPU hybrid (experiment 011) | 520 ms | 2.8x slower |
| RTX 4060 GPU (FP32) | 26 ms | 7x faster |

Longer programs always process their full frame count, so a 13 s utterance on
the 2-block program costs ~420 ms; a program per 128-frame step keeps the
cost proportional. Cutting long utterances into 10 s chunks without overlap
instead (first evaluation run, 1-block program only) raised the WER of those
58 utterances from 1.84% to 2.76%, exactly as with the FP32 encoder given
the same chunks.

Stability after the fix: 283 1-block encoder calls (566 submissions), 27
correctness runs of the 2- and 3-block programs on the test clips (including
the 30.6 s clip that hung before the fix; 815 ms, cosine 0.99991 to FP32,
identical transcript) and the 58 whole-utterance runs, all without a hang.
Repeated runs are bit-identical.

Not measured: power, and CPU load while the NPU runs (the process's CPU time
includes ONNX Runtime and XRT waiting threads).

## Live demo

`powershell -NoProfile -File scripts\parakeet_npu_demo.ps1` transcribes the
microphone live ([`012_live_demo.py`](012_live_demo.py); needs `sounddevice`
in the IRON venv). An energy VAD ends a phrase after 0.7 s of silence and cuts
speech longer than 10 s at the quietest point of its last 2 s, so every call
fits the 1-block program. Startup ~7 s (compiled program cached). Per phrase:
NPU encoder 175 to 200 ms, whole pipeline 205 to 270 ms. `--ptt` for push to
talk, `--cpu` to compare with the CPU FP32 encoder, file arguments to run
WAVs through the same segmenter. Tried by the owner live on the laptop mic.

## Root cause of the hangs: shim task-queue overflow

Before the fix the NPU hung intermittently (after 2 to 72 good submissions)
and two hangs were followed by Windows blue screens (see the handoff's safety
rules). The cause found offline: each task group started all input fills
before any output drain, and attention groups started 6 to 9 transfers on one
shim channel, whose task queue holds four. A push into a full queue is
dropped, and the core that waits for that transfer waits forever. The
investigation and fix below are by Codex (astra), and were validated on
hardware as described above: health check, then 12 correctness runs, then
the 210-utterance evaluation.

## Offline H1 investigation and candidate fix (2026-09-27)

**Found a concrete scheduling defect:** the previous default attention group
starts nine transfers on one W shim channel before issuing any output drain;
the following attention group starts six. NPU1 has a **four-entry task queue
per channel**, separate from its **16 shared shim buffer descriptors**.
The old `MAX_BDS = 10` check accepts the nine-W-plus-one-drain group and cannot
establish queue safety. The earlier 13-BD hang does not establish a 10-BD
hardware limit.

H1 needs one correction: a full queue is not documented as a blocking push.
MLIR-AIE's [DMA task-queue guide](https://xilinx.github.io/mlir-aie/dev/programming_guide/section-2/section-2d/DMATasks/#the-dma-task-queue)
describes a dropped push on overflow, followed by an indefinitely waiting
consumer or completion wait. That guide's measured example uses Strix; it
is supporting mechanism evidence, not a measurement of this Phoenix laptop.
The NPU1 depth itself is independently confirmed by aie-rt's AIE2 IPU shim
configuration and [AMD AM020's array-interface DMA specification](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Array-Interface-DMA-Memory-Mapped-AXI4-Master-Interface).
This is a strong explanation for the timing-sensitive attention hangs, but
does not prove that every historical timeout has the same cause. No hardware
overflow counter or live queue state was read in this investigation.

### Lowering evidence

Local mlir-aie checkout: `95b3d1ccc0bfe5183bae1fa9014cfdc1fb4d96c8`.
The installed `dmatask.py`, `taskgroup.py`, and `runtime.py` match that
checkout byte for byte. Paths below are relative to `cache/iron/mlir-aie/`:

* `python/iron/runtime/dmatask.py`, `DMATask.resolve`: a fill or drain emits
  `shim_dma_single_bd_task` followed immediately by `dma_start_task`.
* `python/dialects/aiex.py`, `shim_dma_single_bd_task`: emits
  `dma_configure_task_for` with one BD; the outer TAP dimension becomes
  `repeat_count = sizes[0] - 1`. Repeats are one queued task, not separate
  queue entries. `wait=True` requests a completion token.
* `python/iron/runtime/runtime.py`, `finish_task_group`: emits all requested
  awaits, then frees. It does not defer starts or insert queue-capacity waits.
* `lib/Dialect/AIEX/Transforms/AIEDMATasksToNPU.cpp`: start becomes
  `aiex.npu.push_queue`; await becomes `aiex.npu.sync` on the physical channel.
  `AIEAssignRuntimeSequenceBDIDs.cpp` recycles descriptor IDs at awaits/frees
  and erases frees; a free is not a hardware completion wait.
* `AIEDmaToNpu.cpp`, `PushQueuetoWrite32Pattern`: the push is a register
  `write32`, with BD ID, repeat count, and token bit. There is no queue-full
  poll in this lowering. `AIETargetModel.cpp`, `AIE2TargetModel`, supplies
  shim queue addresses: C/S2MM0 `0x1d204`, W/MM2S0 `0x1d214`, X/MM2S1
  `0x1d21c`, plus the column address (`col << 25`).
* `third_party/aie-rt/driver/src/global/xaie2ipugbl_reginit.c`,
  `AieMlShimDmaChProp`: `StartQSizeMax = 4U`; `AieMlShimDmaMod`: 16 BDs.
  `xaie_dma_aieml.c` counts the active task separately from the queued tasks.
  The fix conservatively allows at most four total starts between verified
  completions, without assuming an additional active slot or DMA progress.

Read-only audit of the two cached 1-block, 12-layer submissions:

| Cache directory under `~/.npu/cache` | Groups | Starts | Groups over four starts/channel | Fills before drains |
| --- | ---: | ---: | ---: | ---: |
| `3d103dd50a4dfc2325c85dc6` (first half, initial LN included) | 277 | 5,008 | 24 | 277 |
| `d950d66168aa07b3fff9eccb` (second half) | 276 | 4,988 | 24 | 276 |

In the first file, attention begins at `aie.mlir:2540`, group 13 (zero based).
Tasks `%212` through `%220` all start on `w3_0` before any drain: header,
AUX, delta, Q/BD/KV for pass 0, Q/BD/KV for pass 1. The compiler allocates
BD IDs 0 through 8 for these nine pushes. Both position-score transfers have
repeat count 3, but each is still just one queue push. The placed
`input_with_addresses.mlir` confirms W=MM2S0, X=MM2S1, C=S2MM0 on each column.

The actual cached `insts.bin` files were decoded with
`aie.dialects.aie.transaction_binary_to_mlir`. All 6,116 / 6,092 start-and-wait
events match their source MLIR in order, including the nine unchecked writes
to the same W queue. This is not only a Python-level scheduling inference.
The conservative counts are upper bounds on unretired tasks, not measurements
of actual occupancy at each instant.

### Implemented changes and offline validation

* `pk_engine.group_tasks()` is shared by emission and checking. Every column's
  drains are issued before any fills. All drains are awaited at group finish.
* Attention has a separate three-task header/AUX/delta group. Its last W task
  on each column issues a token and is awaited, so its BDs can be safely reused.
  Each later group handles one pass: three W tasks and one drain. The default
  `PK_ATT_GROUP` is now 1; larger unsafe groups fail static validation.
* `check()` enforces queue bounds per stream/column, drain ordering, prologue
  waits, matching column group counts, and exact cumulative input/output
  counts at each core-iteration boundary. The previous descriptor and buffer
  checks remain. Counts alone are not a general proof against H2/H3/H4.
* `PkEncoder` validates before importing `npu_direct` or allocating XRT buffers.
  Kernels and `LAYERS_PER_RUN` are unchanged.
* `012_encoder_test.py` supports `--clip` and `--repeats`, logs each run,
  and stops on nonfinite or nonidentical repeated outputs. An XRT failure
  propagates immediately; there is no retry handler.

Eight offline regression tests pass (`012_schedule_check.py`). They cover all
529 phases for each of 1/2/3 blocks and every submission slice, the old
attention overflow, oversized experimental groups, fills-first ordering,
missing prologue waits, incorrect intermediate group boundaries, repeat-count
semantics, and rejection before hardware import. The script blocks both
`pyxrt` and `npu_direct` imports. It uses weight metadata, not device buffers.

All 12 generated submission modules verify in MLIR. The two 1-block modules
also compile fully to xclbin + instruction binaries with an explicitly bound
NPU1 target and XRT imports blocked. Their decoded instructions match the new
source start/wait order exactly (6,356 / 6,332 events). No hardware execution
occurred, including no health check.

| Frame blocks | Submissions | Groups | Queue starts | Peak W / X / C starts per channel per group |
| --- | ---: | ---: | ---: | --- |
| 1 | 2 | 625 | 10,188 | 4 / 1 / 1 |
| 2 | 4 | 721 | 14,796 | 4 / 2 / 2 |
| 3 | 6 | 817 | 19,404 | 4 / 3 / 3 |

There are zero oversized or fills-before-drains groups in these modules.
At 1 block the new binaries are 667,584 / 665,008 bytes (old: 649,728 /
647,152); performance impact is unmeasured. Scratch MLIR, decoded instructions,
and offline build outputs are under
`C:/Users/abhir/Documents/Codex/2026-09-27/work-in-c-users-abhir-desktop/work/012_offline/`.

Reproduce the offline checks without opening the NPU:

```powershell
& scripts/iron_python.ps1 experiments/012_schedule_check.py
& cache/iron/mlir-aie/ironenv/Scripts/python.exe experiments/012_cache_audit.py C:/Users/abhir/.npu/cache/3d103dd50a4dfc2325c85dc6 C:/Users/abhir/.npu/cache/d950d66168aa07b3fff9eccb
```

The first hardware attempt of the fix failed before any submission at XRT
context creation (`0xc01e0009`, `STATUS_GRAPHICS_DRIVER_MISMATCH`); the NPU
had been through several hang resets and a live driver update since the last
boot. After a reboot, context creation worked and the known-stable
`010_x8_validate.py --check` passed, followed by the correctness test and the
evaluation above. The failed attempt's log is `single_test_20260927_2050.log`
in the scratch directory listed above. The timeout wrapper
`scripts
un_npu_timeout.ps1` returned exit code 0 despite the child's error:
judge results by the captured output.

## Before the fix: test clips and the 3-block program

Test clips (test.wav and three LibriSpeech test-clean utterances, 73 to 123
encoder frames), [`012_encoder_test.py`](012_encoder_test.py): subsampling
prefix on the CPU (ONNX Runtime), 24 Conformer layers on the NPU, TDT decoding
on the CPU.

| | Value |
| --- | --- |
| Encoder output vs ONNX Runtime FP32 | cosine 0.99982 to 0.99994, relative RMS 1.1 to 1.9% |
| Transcripts vs FP32 | identical on all 4 clips |
| NPU time per 10 s window | 188 to 196 ms (two submissions of 12 layers, schedule before the fix) |
| For comparison, same 10 s window | CPU FP32 496 ms, CPU dynamic int8 411 ms, GPU (RTX 4060) 26 ms, exp-011 hybrid 520 ms |
| CPU use during the NPU encoder | the host thread waits; prefix and decoding stay on the CPU |

A 3-block program (384 frames, 30.7 s) gave identical outputs on the short
clips (600 ms per clip, since it always processes 384 frames); it hung on the
30.6 s clip (the queue overflow). It has not been rerun with the fix.

Layer 0 op by op against a NumPy mirror with BF16 rounding at the same points
([`012_layer_test.py`](012_layer_test.py)): LayerNorm, FFN with Swish, q/k/v
with bias, per-head position scores, attention, output projection, GLU +
depthwise convolution + Swish, and the final LayerNorm all match with cosine
>= 0.99993 and 0.4 to 1.2% relative RMS error.

## Design

* **Grid.** All 16 compute tiles. Weights stream from DDR once per column
  and are broadcast to that column's four cores; activation tiles are
  re-blocked into mmul layout by the memory tiles and broadcast along rows;
  outputs are joined per column in the memory tile.
* **One core program for every phase** ([`kernels/pk/`](kernels/pk)): each
  phase starts with a header object that sets the segment counts of the
  core's loop. Phases: GEMM (aie::mmul 4x8x4 BF16, FP32 accumulation,
  optional Swish or per-column bias), residual add + LayerNorm (single,
  double or final), relative-position attention (online softmax, 16 queries
  per pass, position scores from a per-head GEMM against the constant
  projected position table), and the convolution module (GLU, depthwise
  kernel 9 with batch norm folded, Swish). 22 phases per layer, 529 per
  encoder, issued as a few submissions.
* **Numerics.** BF16 activations between phases, FP32 accumulation inside.
  FP32 products in the vector math are built from BF16 pieces with native
  BF16 x BF16 -> FP32 MACs; exp uses a degree-4 polynomial for 2^f, sigmoid a
  Newton reciprocal (errors 7.7e-5 and 2.7e-4, below BF16 rounding).
  The 1/sqrt(128) attention scale is folded into Wq and the biases on the host.
* **Frame blocks.** Cores always work on 32-frame GEMM tiles and 1024-element
  outputs; a program for 128*n frames runs each GEMM once per 128-frame block
  (weights re-streamed per block, about 15% slower than one big tile at 384
  frames, which did not fit the 64 KB data memory).

## Measured along the way

| Step | Result |
| --- | --- |
| 8 GEMM phases of one layer, one submission | 3.7 ms at 128 frames (830 GMAC/s), 6.3 ms at 256, 9.6 ms at 384 (~970 GMAC/s); the hybrid needed ~9.2 ms per layer at 128 |
| First full 24-layer run | 450 ms per 10 s window, 6 to 16% relative error, transcripts correct |
| FP32 -> BF16 rounding mode set to round-to-nearest-even | error 6-16% -> 1.1-1.9% (the default truncates) |
| exp / sigmoid / Swish from native BF16 MACs instead of emulated FP32 multiplies | 513 / 882 / 933 -> 171 / 257 / 286 cycles per 16 values ([`012_vbench.py`](012_vbench.py)); encoder 450 -> 188 ms |
| Program memory | GEMM step and hot vector routines -O2 (pk_fast.cc, ~7 KB), the rest -Oz (pk_core.cc, ~7 KB); no soft-float or modulo library code (they pushed it over 16 KB) |

## Files

| File | Role |
| --- | --- |
| [`pk_engine.py`](pk_engine.py) | IRON design, data layout, phase classes, static program checker |
| [`pk_model.py`](pk_model.py) | Phase list for 24 layers, weight/position packing, `PkEncoder` |
| [`kernels/pk/`](kernels/pk) | Core program: `pk_core.cc` (-Oz), `pk_fast.cc` (-O2), `pk_math.h`, `pk_common.h`, `pk_vbench.cc` |
| [`012_layer_test.py`](012_layer_test.py) | Op-by-op check of layer 0 |
| [`012_encoder_test.py`](012_encoder_test.py) | Full encoder vs ONNX Runtime FP32 and transcripts |
| [`012_transcribe.py`](012_transcribe.py) | WER with the 1-block program (long utterances chunked; CPU FP32 with the same chunks) |
| [`012_transcribe_long.py`](012_transcribe_long.py) | Long utterances whole with the 2- and 3-block programs; `--combine` builds the whole-utterance table |
| [`012_live_demo.py`](012_live_demo.py), [`../scripts/parakeet_npu_demo.ps1`](../scripts/parakeet_npu_demo.ps1) | Live microphone transcription on the NPU |
| [`012_results.json`](012_results.json) | Summaries of the evaluation and the correctness test |
| [`012_schedule_check.py`](012_schedule_check.py), [`012_cache_audit.py`](012_cache_audit.py) | Offline schedule regressions and audit of cached MLIR (queue limits) |
| [`012_vbench.py`](012_vbench.py) | Single-core timing and accuracy of the vector math |
| [`012_hang_probe.py`](012_hang_probe.py) | Runs the first k phases once (debugging) |

Setup is as in experiment 011 (weights from `011_export_weights.py`,
positions from `011_export_pos.py`, dynamic prefix from `011_split.py`).
