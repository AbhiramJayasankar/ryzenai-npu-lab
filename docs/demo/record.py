"""Record a real NPU run as a timed transcript for docs/demo/render.py.

Run inside the IRON environment, one scenario per fresh process:
  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script docs\\demo\\record.py -ScriptArgs chat -Seconds 600
  powershell -NoProfile -File scripts\\run_npu_timeout.ps1 -Script docs\\demo\\record.py -ScriptArgs stt -Seconds 600

Writes docs/demo/<scenario>.json: a list of [seconds, kind, text] events,
where kind is "out" (program output, as it arrived) or "in" (what was typed).
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WAVS = REPO / "cache" / "parakeet" / "librispeech_wav"

SCENARIOS = {
    "chat": dict(
        cmd=["experiments/010_lfm25_npu_chat.py", "--max-new-tokens", "120"],
        inputs=["What is an NPU? Answer in two sentences.",
                "Write a haiku about a laptop that thinks.",
                "/quit"],
    ),
    "stt": dict(
        cmd=["experiments/012_live_demo.py"] + [str(WAVS / f"{n}.wav") for n in
                                                ("1089-134686-0021", "121-121726-0000", "1284-1180-0000")],
        inputs=[],
    ),
}


def main():
    name = sys.argv[1]
    sc = SCENARIOS[name]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(REPO / "experiments") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen([sys.executable, "-u"] + sc["cmd"], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    events, chunks = [], queue.Queue()

    def reader():
        while True:
            data = os.read(proc.stdout.fileno(), 4096)
            chunks.put((time.perf_counter() - t0, data))
            if not data:
                return

    threading.Thread(target=reader, daemon=True).start()
    inputs, tail, raw = list(sc["inputs"]), "", b""
    while True:
        t, data = chunks.get()
        if not data:
            break
        raw += data
        try:
            text, raw = raw.decode("utf-8"), b""
        except UnicodeDecodeError:
            continue  # wait for the rest of a multi-byte character
        events.append([round(t, 3), "out", text.replace(str(REPO) + os.sep, "")])
        sys.stdout.write(text)
        sys.stdout.flush()
        tail = (tail + text)[-16:]
        if inputs and tail.endswith("You: "):
            line = inputs.pop(0)
            events.append([round(time.perf_counter() - t0, 3), "in", line])
            proc.stdin.write((line + "\n").encode())
            proc.stdin.flush()
            tail = ""
    proc.wait()
    (Path(__file__).parent / f"{name}.json").write_text(json.dumps(events, ensure_ascii=False, indent=0),
                                                         encoding="utf-8")
    print(f"\nexit code {proc.returncode}, {len(events)} events")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
