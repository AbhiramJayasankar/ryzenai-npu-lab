"""Speech-to-text engine process: Parakeet TDT 0.6B v2 with the 24 encoder
layers on the NPU (experiment 012, 1-block program, <= 10.24 s per call).

Runs in its own process so NPU calls never stall the hotkey, audio capture or
UI of the app. Protocol (multiprocessing queues):
  in:  ("audio", session, index, float32 16 kHz mono)   ("quit",)
  out: ("status", text)  ("ready", load_s)  ("fatal", message)
       ("text", session, index, text, audio_s, npu_s, total_s)
Any NPU error ends the engine: the lab's safety rule is no retries.
"""

import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"


class NpuEncoder:
    """encoder-model.onnx interface: subsampling prefix on the CPU, 24 layers on the NPU."""

    def __init__(self, pp, pm, np):
        self.np = np
        self.prefix = pp.session(pp.CACHE / "models" / "encoder_dyn_prefix.onnx", ["CPUExecutionProvider"])
        self.npu = pm.PkEncoder(nblk=1)
        self.frames = self.npu.lay.TPAD
        self.npu_s = 0.0

    def run(self, names, feeds):
        hidden = self.prefix.run(["hidden"], feeds)[0][0]
        n = hidden.shape[0]
        if n > self.frames:
            raise ValueError(f"{n} encoder frames > {self.frames} (audio longer than 10.24 s)")
        x = self.npu.run(hidden, n)
        self.npu_s = self.npu.npu_s
        return [x.T[None], self.np.array([n], dtype=self.np.int64)]


def serve(inq, outq, log_path):
    log = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    print(f"\n=== engine start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.path.insert(0, str(EXPERIMENTS))
    nd = None
    try:
        t = time.perf_counter()
        outq.put(("status", "Loading the NPU encoder…"))
        import numpy as np

        import npu_direct as nd
        import parakeet_pipeline as pp
        import pk_model as pm

        enc = NpuEncoder(pp, pm, np)
        model = pp.Parakeet(enc)
        model.transcribe(np.zeros(pp.SR, np.float32))  # warm-up
        load_s = time.perf_counter() - t
        print(f"ready in {load_s:.1f} s")
        outq.put(("ready", load_s))
    except Exception:
        print(traceback.format_exc())
        outq.put(("fatal", "Engine failed to start:\n" + traceback.format_exc(limit=3)))
        return
    try:
        while True:
            msg = inq.get()
            if msg[0] == "quit":
                break
            _, session, index, audio = msg
            text, tt = model.transcribe(audio)
            outq.put(("text", session, index, text, len(audio) / pp.SR, enc.npu_s, tt["total"]))
            print(f"{session}.{index} {len(audio) / pp.SR:.1f} s npu {1000 * enc.npu_s:.0f} ms "
                  f"total {1000 * tt['total']:.0f} ms, {len(text)} chars")  # no transcript text in the log
    except Exception:
        print(traceback.format_exc())
        outq.put(("fatal", "NPU engine error (stopped, no retries):\n" + traceback.format_exc(limit=3)))
    finally:
        if nd is not None:
            nd.finish()
        print("engine stopped")
