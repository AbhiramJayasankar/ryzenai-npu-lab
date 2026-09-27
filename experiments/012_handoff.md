# Experiment 012 handoff: all-NPU Parakeet encoder, and the intermittent hang

Results so far are in [`012_all_npu_encoder.md`](012_all_npu_encoder.md). This
file is for whoever continues the work. The main open problem is an
**intermittent NPU hang**. Read the safety rules first.

## 1. Safety rules (read before touching the NPU)

On 2026-09-27 NPU hangs caused **two Windows blue screens** on this laptop
(0x50 after ~47 hangs in a loop, 0x3B after only ~3). Every hang is an
`ERT_CMD_STATE_TIMEOUT`, and Windows then resets the device (LiveKernelEvent
0x141).

* Do not run anything on the NPU that might hang without asking the owner
  first. Never loop or bisect hang candidates on the hardware.
* After any hang, stop all NPU runs and analyze offline before the next run.
* Prefer offline work: static checks, reading the lowered MLIR, reasoning.
* Known-stable health check: `010_x8_validate.py --check` (experiment 010).
* Run NPU scripts through `scripts\run_npu_timeout.ps1` (kills the process on
  timeout), one at a time, in a fresh process.

## 2. The hang: what is known

The same program with the same inputs runs cleanly many times, then hangs on
a run that is no different from the runs before it.

| When (2026-09-27) | Program | Before the hang | Notes |
| --- | --- | --- | --- |
| 1-block, one 188 ms submission (earlier version) | encoder test | ~16 runs OK | no hang seen |
| 1-block, 2 submissions of 12 layers | `012_encoder_test.py 1` | 17 runs OK | no hang in that session |
| 3-block, 6 submissions of 4 layers | `012_encoder_test.py 3` | 12 runs (72 submissions) OK | hung on the next run |
| 1-block, 2 submissions | `012_transcribe.py` | 2 runs OK | hung on the next |
| 1-block, 2 submissions, **new driver 32.0.20101.3760** | `012_encoder_test.py 1` | 3 runs OK | hung on the 4th |
| attention phase, 256 frames, task groups of 3 or 4 passes | `012_hang_probe.py 2 14` | none | **hung every time**; groups of 1 or 2 passes ran |

Ruled out or unlikely:

* **Object counts.** `pk_engine.check()` statically verifies, per core and
  phase, that the objects delivered by the runtime sequence equal those the
  core program consumes and produces (from the headers). It also checks the
  descriptor count per shim tile per task group (`MAX_BDS = 10`) and buffer
  bounds. All 529 phases pass for 1, 2 and 3 blocks.
* **Input data.** The hangs occur on inputs that ran before. Prefix outputs
  of every utterance have the same range (max |x| ~1,900 to 2,100, one
  outlier channel).
* **Driver version.** It happens on both 32.0.203.280 and 32.0.203.376.
* **Power source changes.** The one near a hang came after it.
* The experiment 010 engine (8 cores, 17 ms submissions) is fine on both
  drivers.

## 3. Hypotheses, most likely first

**H1: runtime-sequence ordering deadlock (task queue full, drains not yet
issued).** In `pk_engine.design()` → `sequence()`, each task group issues all
fills of all four columns first (W stream, X stream) and only then the output
drains, then `g.finish()`.

Each `fill` / `drain` is pushed onto a shim DMA channel's task queue. If a
channel's queue holds fewer tasks than we push, the command processor blocks
on that push, so the drains later in the sequence are never issued. The cores
then fill their output buffers (C fifo depth 2 and memory-tile join depth 2)
and stop consuming inputs. The blocked fill queue never drains, and the
array deadlocks. Whether this happens depends on how far the DMAs have
progressed when the push is attempted, which would make it intermittent.

Attention with 3 or more passes per group pushes 3 + 3×passes tasks onto one
W channel and hangs every time. GEMM and LayerNorm phases push 3 or 4.

To check:

* How IRON lowers `fill`/`drain`: `dma_configure_task_for` plus
  `dma_start_task` (a queue push?), and `TaskGroup.finish` (awaits and frees).
* The shim task-queue depth on AIE-ML/NPU1: search mlir-aie for
  "queue"/"repeat_count" in the AIEX lowering and target model, and the AIE-ML
  DMA documentation.
* The lowered sequence: `~/.npu/cache/<hash>/aie.mlir` (and the insts) of the
  1-block program. Count outstanding tasks per channel at each point.

Fix candidates:

* Issue each task group's drains **before** its fills. A drain needs no data
  to be issued.
* Cap the pending tasks per channel below the queue depth (smaller groups,
  or `wait`/`finish` in between).
* Add both rules to `check()`.

For comparison, experiment 010's sequence (`lfm25_x8.py`) interleaves fill,
fill, fill, drain per core with few tasks per channel.

**H2: cross-core dependency through the broadcasts and joins.** W is broadcast
to a column's 4 cores, X along a row across 4 columns, and C joined per
column. Prove deadlock freedom per phase type for the given depths (W/X/C 2,
memory tile 2), especially GEMM (cores wait on both W and X).

**H3: a driver/firmware limit on long submissions.** The encoder submissions
take ~95 to 135 ms; experiment 010's take 17 ms. Splitting finer
(`LAYERS_PER_RUN` in `pk_model.py`) is cheap, but testing it needs hardware
runs.

**H4: core-side corruption.** Unlikely: all indices are static, the stack is
measured by aiecc (2,560 B), and the scratch aliases have been reviewed
(`A_O` reuses `qT` after the last key). Still worth a read of `pk_core.cc` and
`pk_fast.cc`.

## 4. How the engine works (for reading the code)

* `pk_engine.py`
  * `Layout(nblk)`: io layout. The AUX object holds the valid frame count.
    Frame records of S=8192 elements hold fields X, A, Y, BIG and O. Then
    the per-head position scores, [TPAD][2*TPAD].
  * Phase classes `Gemm`, `Ln`, `Att`, `Conv`. `groups(lay, c)` returns
    task groups of (W fills, X fills, drains) for column c.
  * `design()` builds the IRON program. The 16 workers run `_core`: a header
    object, NPRO prologue objects, then NB blocks of (NWX W+X pairs, NW W
    objects, NOUT outputs). The counts come from the header via `ctl`.
  * `check()` is the static checker.
* `pk_model.py`: `build_phases` (22 phases per layer: FFN1 W1+Swish, W2, LN,
  qkv+bias, 8 position-score GEMMs, attention, out, LN, pointwise1, conv
  module, pointwise2, LN, FFN2 W1+Swish, W2, LN), weight packing
  (`cache/parakeet/pk/weights.bin`, 1.2 GB), and `PkEncoder`. The encoder is
  split into `LAYERS_PER_RUN` layers per submission, sharing one hardware
  context.
* Kernels: `pk_core.cc` (-Oz: dispatch, LN, attention setup/finish,
  epilogue) and `pk_fast.cc` (-O2: GEMM step, exp/sigmoid/Swish, attention
  key loop, conv module). They must stay under 16 KB of program memory
  together (currently ~14 KB, plus ~1.6 KB of loop code).
* Pitfalls already hit:
  * The IRON compile cache keys only on the tag, so the tag includes digests
    of the kernels and of the whole runtime sequence.
  * The FP32→BF16 rounding mode must be set (in `pk_hdr`).
  * The FP32 vector multiply is emulated and slow.
  * OpenBLAS with 16 threads is slow on the host.

## 5. Reproduce (after the hang is fixed, and with the owner's OK)

```powershell
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\010_x8_validate.py -ScriptArgs "--check" -Seconds 900   # health check
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\012_encoder_test.py -ScriptArgs "1" -Seconds 1800
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\012_transcribe.py -Seconds 3600   # WER, 210 utterances
```

A stable fix should survive a soak of well over 100 consecutive encoder runs
(the longest clean streak so far was 72 submissions). Prerequisites are in
`cache/parakeet/`: weights and position tables (`011_export_weights.py`,
`011_export_pos.py`), `models/encoder_dyn_prefix.onnx` (`011_split.py`),
`reference_dyn.npz` (`011_reference_outputs.py`) and `eval.json`
(`011_prepare_data.py`). The IRON venv needs `onnxruntime` (pip).
