"""Speech-to-text engine process: Parakeet TDT 0.6B v2 with the 24 encoder
layers on the NPU (experiment 012, 1-block program, <= 10.24 s per call),
loaded from an exported bundle (export_bundle.py, npu_runtime.py).

Runs in its own process so NPU calls never stall the hotkey, audio capture or
UI of the app. Protocol (multiprocessing queues, plus a stop event and an
npu_started event that the engine sets right before its first NPU submission):
  in:  ("audio", session, index, float32 16 kHz mono)   ("quit",)
  out: ("status", text)  ("npu",)  ("ready", load_s)  ("start_failed", reason, retry)
       ("fatal", message)  ("busy", session, index)  ("piece_failed", session, index, audio_s)
       ("text", session, index, text, audio_s, npu_s, total_s)
"start_failed" means nothing reached the NPU; retry says whether the cause is
one that clears by itself (memory, files), so the app may try again on its own.
After "npu", any NPU error ends the engine: the lab's safety rule is no retries.
Once the app sets the stop event, no further piece is started. While the
process lives it holds the named mutex MUTEX, so a later run of the app waits
for it (one engine on the NPU at a time, across runs too); an engine that finds
it held exits at once with "start_failed".
"""

import ctypes
import os
import sys
import time
import traceback

# Free physical RAM is not checked: Windows makes room for the 1.2 GB of NPU
# buffers by trimming other programs (measured: the model loaded in 4.1 s with
# 0.66 GB "available"), and if it cannot, the driver fails with its 0xc01e0200
# page-in error before any NPU work, which is caught and retried later. What
# Windows cannot make room for is commit (RAM + page file promised to programs):
# without ~1.2 GB of it left for the buffers plus ~0.3 GB for the ONNX parts and
# Python, the allocation would fail anyway, so loading is not attempted.
NEED_COMMIT_MB = 1536
MUTEX = "Local\\NPU-Dictate-engine"


def memory_mb():
    """(available physical, available commit) in MB."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys // 2**20, m.ullAvailPageFile // 2**20


class Encoder:
    """encoder-model.onnx interface: subsampling prefix on the CPU, 24 layers on the NPU."""

    def __init__(self, pp, np, npu_runtime, bundle):
        self.np = np
        self.prefix = pp.session(bundle / "encoder_prefix.onnx", ["CPUExecutionProvider"])
        self.npu = npu_runtime.NpuEncoder(bundle)
        self.frames = self.npu.TPAD
        self.npu_s = 0.0

    def run(self, names, feeds):
        hidden = self.prefix.run(["hidden"], feeds)[0][0]
        n = hidden.shape[0]
        if n > self.frames:
            raise ValueError(f"{n} encoder frames > {self.frames} (audio longer than 10.24 s)")
        x = self.npu.run(hidden, n)
        self.npu_s = self.npu.npu_s
        return [x.T[None], self.np.array([n], dtype=self.np.int64)]


def start_failure(e):
    """(reason, retry) for an exception raised before the first NPU submission."""
    text = str(e)
    first = text.splitlines()[0] if text else ""
    if "0xc01e0200" in text or "page-in" in text or isinstance(e, MemoryError):
        return "not enough free memory for the NPU model (it needs ~1.2 GB)", True
    if isinstance(e, ImportError) and (getattr(e, "name", "") == "pyxrt" or "xrt_coreutil" in text):
        return f"the AMD NPU driver was not found ({first})", False
    if type(e).__name__ == "UnsupportedNpu":
        return f"unsupported NPU: {first}", False
    # Anything else from the driver (a device that was reset, 0xc01e0009, ...) is not
    # retried automatically: a restart of Windows is the usual cure.
    return f"{type(e).__name__}: {first}", False


_outq = None
_mutex = None


def serve(inq, outq, log_path, bundle, stop=None, npu_started=None):
    import faulthandler
    import multiprocessing as mp
    from pathlib import Path

    import win32  # ctypes only: no window, thread or hook is created

    global _outq, _mutex
    _mutex = win32.create_mutex(MUTEX)  # released when this process ends, however it ends
    err = ctypes.get_last_error()  # win32's kernel32 keeps it (use_last_error)
    _outq = outq
    bundle = Path(bundle)
    die = os.environ.get("NPU_DICTATE_TEST_DIE_ONCE")  # test hook: die at startup if this file exists
    if die and os.path.exists(die):
        os.remove(die)
        os._exit(3)
    log = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    faulthandler.enable(log, all_threads=True)  # native crashes go to engine.log
    print(f"\n=== engine start {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}")
    if not _mutex:
        print(f"could not create the mutex {MUTEX} (error {err})")
    elif err == 183:  # ERROR_ALREADY_EXISTS: another engine process holds it; no XRT call is made
        print("another engine process holds the mutex; not starting")
        outq.put(("start_failed", "another NPU engine process is still running", True))
        return _exit(0)
    if os.environ.get("NPU_DICTATE_FAKE_ENGINE"):  # test hook: no NPU, no model
        return _fake(inq, outq, stop, npu_started)
    if not getattr(sys, "frozen", False):  # from the repository: parakeet_pipeline lives in experiments/
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
    try:
        t = time.perf_counter()
        outq.put(("status", "Loading the NPU encoder…"))
        import npu_runtime

        problems = npu_runtime.verify(bundle)
        if problems:
            outq.put(("start_failed", "model files missing or damaged: " + "; ".join(problems[:3]), True))
            return _exit(0)
        phys, commit = memory_mb()
        print(f"memory: {phys} MB available, {commit} MB commit available")
        if os.environ.get("NPU_DICTATE_TEST_START_FAIL") or commit < NEED_COMMIT_MB:  # test hook
            # Checked before the device is opened: a failed hw_context is avoided entirely.
            outq.put(("start_failed", f"not enough memory for the NPU model (Windows can promise only "
                                      f"{commit} MB more; it needs {NEED_COMMIT_MB} MB)", True))
            return _exit(0)
        import numpy as np
        import parakeet_pipeline as pp

        enc = Encoder(pp, np, npu_runtime, bundle)
        model = pp.Parakeet(enc, model_dir=bundle)
    except Exception as e:
        # Nothing has been submitted to the NPU yet (buffers, hardware context and
        # program setup only), so the app may start the engine again later.
        print(traceback.format_exc())
        outq.put(("start_failed", *start_failure(e)))
        return _exit(0)
    parent = mp.parent_process()
    if (stop is not None and stop.is_set()) or (parent is not None and not parent.is_alive()):
        print("closed by the app, or the app is gone, before the first NPU submission")
        return _exit(0)
    code = 0
    try:
        outq.put(("npu",))  # from here on a failure is an NPU failure: no retries
        if npu_started is not None:
            npu_started.set()  # read by the app directly, not through the queue
        model.transcribe(np.zeros(pp.SR, np.float32))  # warm-up (first NPU submissions)
        load_s = time.perf_counter() - t
        print(f"ready in {load_s:.1f} s")
        outq.put(("ready", load_s))
        while True:
            msg = _get(inq, stop)
            if msg is None or msg[0] == "quit":
                break
            _, session, index, audio = msg
            outq.put(("busy", session, index))  # the watchdog times this piece from here
            try:
                text, tt = model.transcribe(audio)
            except Exception:
                if enc.npu.in_npu:
                    raise  # an NPU (XRT) error: stop for good
                # A CPU-side failure (features, prefix, decoder) loses this piece only.
                print(traceback.format_exc())
                outq.put(("piece_failed", session, index, len(audio) / pp.SR))
                continue
            outq.put(("text", session, index, text, len(audio) / pp.SR, enc.npu_s, tt["total"]))
            print(f"{session}.{index} {len(audio) / pp.SR:.1f} s npu {1000 * enc.npu_s:.0f} ms "
                  f"total {1000 * tt['total']:.0f} ms, {len(text)} chars")  # no transcript text in the log
    except Exception:
        print(traceback.format_exc())
        outq.put(("fatal", "NPU engine error (stopped, no retries):\n" + traceback.format_exc(limit=3)))
        code = 1
    print("engine stopped")
    _exit(code)


def _get(inq, stop):
    """Next message; None once the app set the stop event or its process is gone
    (an orphaned engine would keep 1.2 GB and the NPU context until reboot)."""
    import multiprocessing as mp
    import queue

    parent = mp.parent_process()
    while True:
        if stop is not None and stop.is_set():
            print("stop requested")
            return None
        try:
            msg = inq.get(timeout=0.5)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                print("the app is gone; stopping")
                return None
            continue
        if stop is not None and stop.is_set():  # set while this piece was queued: never start it
            print("stop requested")
            return None
        return msg


def _exit(code):
    """Exit without Python teardown (pyxrt objects can crash when collected late),
    after the queue feeder thread has sent everything."""
    if _outq is not None:
        _outq.close()
        _outq.join_thread()
    sys.stdout.flush()
    os._exit(code)


def _fake(inq, outq, stop, npu_started=None):
    """Engine stand-in for testing the app without the NPU: "speech" is
    reported as its length. NPU_DICTATE_FAKE_ENGINE=<load seconds>[,<seconds per piece>]."""
    load, _, per = os.environ["NPU_DICTATE_FAKE_ENGINE"].partition(",")
    outq.put(("npu",))
    if npu_started is not None:
        npu_started.set()
    time.sleep(float(load or 0))
    outq.put(("ready", float(load or 0)))
    while True:
        msg = _get(inq, stop)
        if msg is None or msg[0] == "quit":
            return _exit(0)
        _, session, index, audio = msg
        outq.put(("busy", session, index))
        time.sleep(float(per or 0.2))
        outq.put(("text", session, index, f"piece {index} {len(audio) / 16000:.1f} s", len(audio) / 16000,
                  0.0, 0.0))
