"""Measure how much each Parakeet encoder backend slows other work on the laptop.

For each backend: start its encoder benchmark looping on a 10 s window in a
separate process, wait until it is warm, run 011_contention_probe.py (single-
and 8-thread SHA-256 throughput) for 10 s, stop the backend. Also runs the
probe idle before and after. Run from the repository root with the parakeet-stt venv:

  ..\\parakeet-stt\\.venv\\Scripts\\python experiments\\011_contention_run.py

Writes cache/parakeet/results/contention.json.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = str(ROOT.parent / "parakeet-stt" / ".venv" / "Scripts" / "python.exe")
BACKENDS = {
    "cpu_fp32": [VENV, "experiments/011_encoder_bench.py", "--provider", "cpu", "--seconds", "10",
                 "--repeats", "300", "--tag", "contention_cpu"],
    "cpu_int8dyn": [VENV, "experiments/011_encoder_bench.py", "--provider", "cpu", "--encoder",
                    "encoder-model.int8.onnx", "--seconds", "10", "--repeats", "300", "--tag", "contention_int8"],
    "gpu_fp32": [VENV, "experiments/011_encoder_bench.py", "--provider", "cuda", "--seconds", "10",
                 "--repeats", "3000", "--tag", "contention_gpu"],
    "npu_hybrid": ["powershell", "-NoProfile", "-File", "scripts/iron_python.ps1",
                   "experiments/011_hybrid_bench.py"] + ["10"] * 12,
}
READY = {"cpu_fp32": "load", "cpu_int8dyn": "load", "gpu_fp32": "load", "npu_hybrid": "ready"}


def probe(label):
    out = subprocess.run([VENV, "experiments/011_contention_probe.py", "--seconds", "10", "--label", label],
                         cwd=ROOT, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    results = [probe("idle")]
    print(results[-1], flush=True)
    for name, cmd in BACKENDS.items():
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ready = threading.Event()

        def pump():
            for line in proc.stdout:
                if READY[name] in line:
                    ready.set()
        threading.Thread(target=pump, daemon=True).start()
        if not ready.wait(300):
            raise RuntimeError(f"{name} did not start")
        time.sleep(8)  # past warm-up, into the timed loop
        results.append(probe(name))
        print(results[-1], flush=True)
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        proc.wait()
        time.sleep(3)
    results.append(probe("idle_after"))
    print(results[-1], flush=True)
    idle = results[0]
    for r in results:
        r["single_vs_idle"] = round(r["single_mb_s"] / idle["single_mb_s"], 3)
        r["multi8_vs_idle"] = round(r["multi8_mb_s"] / idle["multi8_mb_s"], 3)
    (ROOT / "cache" / "parakeet" / "results" / "contention.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
