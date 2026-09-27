"""Experiment 012 demo: live speech-to-text with the Parakeet encoder on the NPU.

Speak; each phrase is transcribed when you pause. Mel features, the
subsampling prefix and the TDT decoder run on the CPU, the 24 Conformer
layers on the NPU (the proven-stable 1-block program, <= 10.24 s per call).
Phrases longer than 10 s are cut at the quietest point of their last 2 s.

  powershell -NoProfile -File scripts\\parakeet_npu_demo.ps1              (microphone, pause-to-send)
  powershell -NoProfile -File scripts\\parakeet_npu_demo.ps1 --ptt        (Enter to start / stop)
  powershell -NoProfile -File scripts\\parakeet_npu_demo.ps1 --cpu        (also time the CPU FP32 encoder)
  powershell -NoProfile -File scripts\\parakeet_npu_demo.ps1 a.wav        (a file through the same segmenter)

Ctrl+C quits. Any NPU error ends the program (no retries).
"""

import argparse
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parakeet_pipeline as pp  # noqa: E402

BLOCK = 480  # 30 ms at 16 kHz
MAX_BLOCKS = int(10.0 * pp.SR / BLOCK)  # the 1-block program holds 10.24 s
CUT_BLOCKS = int(2.0 * pp.SR / BLOCK)  # search window for a forced cut
PRE_BLOCKS = 10  # 0.3 s kept before speech onset
END_BLOCKS = 23  # 0.7 s of silence ends a phrase
TAIL_BLOCKS = 7  # 0.2 s of that silence kept
MIN_VOICED = 8  # phrases with < 0.24 s of speech are dropped (clicks, bumps)
MIN_RMS = 0.004


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class Segmenter:
    """Energy VAD over 30 ms blocks with an adaptive noise floor."""

    def __init__(self):
        self.noise = None
        self.pre = deque(maxlen=PRE_BLOCKS)
        self.buf, self.levels = [], []
        self.active, self.silent, self.voiced = False, 0, 0
        self.level = 0.0

    def push(self, block):
        r = self.level = rms(block)
        if self.noise is None:
            self.noise = r
        speech = r > max(3.0 * self.noise, MIN_RMS)
        if r < self.noise:
            self.noise = 0.7 * self.noise + 0.3 * r  # follow quiet quickly
        elif not speech:
            self.noise = 0.98 * self.noise + 0.02 * r  # and louder rooms slowly
        if not self.active:
            self.pre.append(block)
            if speech:
                self.active, self.silent, self.voiced = True, 0, 1
                self.buf = list(self.pre)
                self.levels = [rms(b) for b in self.buf]
                self.pre.clear()
            return []
        self.buf.append(block)
        self.levels.append(r)
        self.silent = 0 if speech else self.silent + 1
        self.voiced += speech
        if self.silent >= END_BLOCKS:
            keep = len(self.buf) - self.silent + TAIL_BLOCKS
            out = self._emit(self.buf[:keep])
            self.active, self.buf, self.levels = False, [], []
            return out
        if len(self.buf) >= MAX_BLOCKS:  # cut at the quietest block of the last 2 s
            k = len(self.buf) - CUT_BLOCKS + int(np.argmin(self.levels[-CUT_BLOCKS:])) + 1
            out = self._emit(self.buf[:k])
            self.buf, self.levels = self.buf[k:], self.levels[k:]
            self.voiced = MIN_VOICED  # the continuation is speech too
            return out
        return []

    def flush(self):
        out = self._emit(self.buf) if self.active else []
        self.active, self.buf, self.levels = False, [], []
        return out

    def _emit(self, blocks):
        return [np.concatenate(blocks)] if blocks and self.voiced >= MIN_VOICED else []


class NpuEncoder:
    """encoder-model.onnx interface: subsampling prefix on the CPU, 24 layers on the NPU."""

    def __init__(self):
        import pk_model as pm

        self.prefix = pp.session(pp.CACHE / "models" / "encoder_dyn_prefix.onnx", ["CPUExecutionProvider"])
        self.npu = pm.PkEncoder(nblk=1)
        self.npu_s = 0.0

    def run(self, names, feeds):
        hidden = self.prefix.run(["hidden"], feeds)[0][0]
        n = hidden.shape[0]
        x = self.npu.run(hidden, n)
        self.npu_s = self.npu.npu_s
        return [x.T[None], np.array([n], dtype=np.int64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="transcribe these 16 kHz mono files instead of the microphone")
    parser.add_argument("--ptt", action="store_true", help="push to talk: Enter starts and stops a recording")
    parser.add_argument("--cpu", action="store_true", help="also run the ONNX Runtime FP32 encoder on the CPU")
    parser.add_argument("--device", help="input device (sounddevice name or index)")
    args = parser.parse_args()

    print("loading the NPU encoder (1.2 GB of weights; the first run compiles for ~1-2 min) ...", flush=True)
    t = time.perf_counter()
    enc = NpuEncoder()
    model = pp.Parakeet(enc)
    cpu = pp.Parakeet("encoder-model.onnx") if args.cpu else None
    model.transcribe(np.zeros(pp.SR, np.float32))  # warm-up
    print(f"ready in {time.perf_counter() - t:.1f} s", flush=True)

    def show(audio):
        text, tt = model.transcribe(audio)
        line = (f"[{len(audio) / pp.SR:4.1f} s audio | NPU {1000 * enc.npu_s:4.0f} ms"
                f" | total {1000 * tt['total']:4.0f} ms]")
        if cpu is not None:
            ctext, ct = cpu.transcribe(audio)
            same = "same text" if ctext == text else f"differs: {ctext!r}"
            line += f" [CPU FP32 encoder {1000 * ct['enc']:4.0f} ms, {same}]"
        print(f"\r{' ' * 60}\r{line} {text or '(nothing)'}", flush=True)

    try:
        if args.files:
            for path in args.files:
                print(f"== {path}")
                seg = Segmenter()
                audio = pp.load_audio(path)
                for i in range(0, len(audio) - BLOCK + 1, BLOCK):
                    for phrase in seg.push(audio[i:i + BLOCK]):
                        show(phrase)
                for phrase in seg.flush():
                    show(phrase)
        else:
            listen(args, show)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        import npu_direct as nd

        nd.finish()


def listen(args, show):
    import sounddevice as sd

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    name = sd.query_devices(device, "input")["name"]
    blocks = queue.Queue()
    stream = sd.InputStream(samplerate=pp.SR, channels=1, dtype="float32", blocksize=BLOCK, device=device,
                            callback=lambda d, *_: blocks.put(d[:, 0].copy()))
    if args.ptt:
        print(f"mic: {name}. Enter starts recording, Enter stops; Ctrl+C quits.", flush=True)
        while True:
            input("\n> press Enter and speak ...")
            while not blocks.empty():
                blocks.get_nowait()
            with stream:
                input("  recording, Enter to stop.")
            seg = Segmenter()
            seg.noise = 0.0  # every block of a push-to-talk recording counts
            audio = []
            while not blocks.empty():
                audio.append(blocks.get_nowait())
            for b in audio:
                for phrase in seg.push(b):
                    show(phrase)
            for phrase in seg.flush():
                show(phrase)
    print(f"mic: {name}. Speak; each phrase is transcribed when you pause. Ctrl+C quits.\n", flush=True)
    seg = Segmenter()
    with stream:
        while True:
            b = blocks.get()
            phrases = seg.push(b)
            bar = "#" * min(30, int(seg.level / 0.004))
            state = "speaking " if seg.active else "listening"
            print(f"\r  {state} {bar:<30}", end="", flush=True)
            for phrase in phrases:
                show(phrase)


if __name__ == "__main__":
    main()
