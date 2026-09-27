# Experiment 012 handoff: all-NPU Parakeet encoder, and the intermittent hang

Results so far are in [`012_all_npu_encoder.md`](012_all_npu_encoder.md). This
file is for whoever continues the work. The intermittent NPU hang
is fixed (below); read the safety rules before running new NPU programs.

**Status (2026-09-27, 22:00 IST): the hang is fixed.** Cause: shim task-queue
overflow (four entries per channel; attention groups pushed 6 to 9 transfers
before any drain). Codex's fix (drains before fills, at most four starts per
channel between verified completions, queue checks in `check()`) ran after a
reboot: health check, 12 correctness runs, then the 210-utterance WER
evaluation, 283 encoder calls / 566 submissions without a hang. Results in
`012_all_npu_encoder.md`. The hang history below is kept for reference; the
safety rules still apply to any new program. The 2- and 3-block programs
were then run with the fix too (27 correctness runs, 58 whole utterances):
WER equals FP32 on all 210 utterances (2.19%).

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

* **Total object counts.** The original `pk_engine.check()` statically verified, per core and
  phase, that the objects delivered by the runtime sequence equal those the
  core program consumes and produces (from the headers). It also checks the
  descriptor count per shim tile per task group (`MAX_BDS = 10`) and buffer
  bounds. All 529 phases passed for 1, 2 and 3 blocks. **This did not rule out
  task-queue overflow:** the hardware queue has four entries per channel,
  independent of the 16 shared BDs. The extended checker now rejects the old
  attention groups and checks intermediate completion boundaries too.
* **Input data.** The hangs occur on inputs that ran before. Prefix outputs
  of every utterance have the same range (max |x| ~1,900 to 2,100, one
  outlier channel).
* **Driver version.** It happens on both 32.0.203.280 and 32.0.203.376.
* **Power source changes.** The one near a hang came after it.
* The experiment 010 engine (8 cores, 17 ms submissions) is fine on both
  drivers.

## 3. Hypotheses, most likely first

**H1: task-queue overflow / lost transfer. Concrete defect found offline;
hardware causality and fix remain unverified.** The old sequence issued all
fills before drains, then `g.finish()`. The default attention groups started
nine W tasks (header + two prologues + two passes) or six W tasks (two passes)
on a four-entry channel queue. Even allowing a separate active slot does not
make these groups safe. The original theory of a blocking queue push needs
correction: the MLIR-AIE guide describes dropped pushes on queue overflow,
leaving downstream consumers or awaits waiting forever. Consumer timing
explains how the same schedule could sometimes succeed.

Offline findings:

* IRON fill/drain emits configure + immediate start. Finish emits requested
  awaits before frees. Start lowers to a queue-register write; repeat_count
  is encoded in that one write. Free only permits compile-time BD reuse.
* NPU1/AIE2 IPU shim `StartQSizeMax = 4U` in aie-rt; AMD AM020 independently
  confirms four tasks per channel and 16 shared BDs.
* Cached 1-block halves `3d103dd50a4dfc2325c85dc6` and
  `d950d66168aa07b3fff9eccb` have 24 oversized groups each. Decoding the actual
  `insts.bin` files confirms the same unchecked starts and wait order.

Implemented candidate:

* Issue **all columns' drains before any fill**.
* Separate attention header/AUX/delta into a three-W-task prologue group,
  waiting on the last W task of every column. Each later group is one pass
  (three W tasks + one drain); `PK_ATT_GROUP` defaults to 1.
* `check()` rejects more than four starts per channel between verified group
  completions without assuming asynchronous progress. It checks issue order,
  prologue waits and complete iteration boundaries as well as previous counts,
  bounds and BD limits. Validation runs before `PkEncoder` imports XRT.
* Eight offline regression tests pass. All 529 phases and submission slices
  pass for 1/2/3 blocks. All 12 generated modules verify; both 1-block halves
  compile offline, and their decoded instruction order matches the candidate.

This addresses H1's queue hazard, not a proof against H2/H3/H4. No kernel or
submission-splitting changes were made. The candidate has **zero encoder
submissions**: the first approved hardware attempt failed during setup at
`pyxrt.hw_context` with `0xc01e0009` (`STATUS_GRAPHICS_DRIVER_MISMATCH` in the
Windows SDK). The exact conflicting components are unknown. The two new
cache entries (`e6843726c9b28ee4cedcb39a`, `98a3f75e22a841471f124d6d`) have MLIR
and instruction binaries identical to the offline-validated candidate.

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

## 5. Hardware validation (owner approval required)

Do not run these merely because they are listed here. Approval for the one
encoder attempt was used; it failed during context creation, before any
submission. Do not retry under that approval. Proposed next diagnostic is
one known-stable health check, **pending fresh owner approval**, to distinguish
a general context-creation failure from this design's initialization. The
installed NPU driver is `32.0.20101.3760`; PnP reports OK. No reset, reboot, or
driver modification has been performed. After resolving initialization,
request approval for one encoder test again; if correct, ask for the soak.

```powershell
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\010_x8_validate.py -ScriptArgs "--check" -Seconds 900   # health check
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\012_encoder_test.py -ScriptArgs "1 24 --clip 0 --repeats 1" -Seconds 1800
# Only after correctness passes and the owner approves the soak:
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\012_encoder_test.py -ScriptArgs "1 24 --clip 0 --repeats 120" -Seconds 1800
# Only after the soak passes and the owner approves WER:
powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\012_transcribe.py -Seconds 3600   # WER, 210 utterances
```

The encoder test logs each start/completion and stops on any XRT error,
nonfinite result or changed output for identical input. There are no retries.
The timeout wrapper returned code 0 even though this attempt's Python child
exited 1, so always inspect its captured output before declaring success.
For the single correctness run, inspect cosine/relative RMS and transcript
against the historical ranges in `012_all_npu_encoder.md` before proceeding.
Never restart after a hang. Use `012_schedule_check.py` and
`012_cache_audit.py` for offline follow-up. Do not commit until asked; do not push.

A stable fix should survive a soak of well over 100 consecutive encoder runs
(the longest clean streak so far was 72 submissions). Prerequisites are in
`cache/parakeet/`: weights and position tables (`011_export_weights.py`,
`011_export_pos.py`), `models/encoder_dyn_prefix.onnx` (`011_split.py`),
`reference_dyn.npz` (`011_reference_outputs.py`) and `eval.json`
(`011_prepare_data.py`). The IRON venv needs `onnxruntime` (pip).
