"""Foreground-work probe: how much does a running speech-to-text backend slow other work?

Runs a fixed CPU workload for --seconds and prints its throughput as JSON:
  single: one thread hashing 4 MB buffers (SHA-256), like an app's UI/logic thread
  multi:  8 threads doing the same, like a build or another heavy app
Run it once idle and once while an encoder benchmark loops in another process
(see 011_parakeet_npu.md for the exact commands); the ratio is the slowdown.
Standard library only.
"""

import argparse
import hashlib
import json
import os
import threading
import time


def worker(stop, counter, idx):
    buf = os.urandom(4 << 20)
    n = 0
    while not stop.is_set():
        hashlib.sha256(buf).digest()
        n += 1
    counter[idx] = n


def rate(threads, seconds):
    stop, counter = threading.Event(), [0] * threads
    ts = [threading.Thread(target=worker, args=(stop, counter, i)) for i in range(threads)]
    t = time.perf_counter()
    for th in ts:
        th.start()
    time.sleep(seconds)
    stop.set()
    for th in ts:
        th.join()
    return 4 * sum(counter) / (time.perf_counter() - t)  # MB/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--label", default="idle")
    args = ap.parse_args()
    out = {"label": args.label, "single_mb_s": round(rate(1, args.seconds), 1),
           "multi8_mb_s": round(rate(8, args.seconds), 1)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
