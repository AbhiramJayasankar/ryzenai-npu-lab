"""Minimal Parakeet TDT 0.6B v2 pipeline on ONNX Runtime (experiment 011).

Stages, each timed separately:
  1. nemo128.onnx preprocessor (waveform -> 128 log-mel features), CPU
  2. encoder (FastConformer, 24 layers, d=1024), any execution provider
  3. TDT greedy decode with decoder_joint-model.onnx, CPU

It reproduces onnx-asr 0.12's NemoConformerTdt decoding so that
transcripts match `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")`, and it
runs in both the parakeet-stt venv (CUDA/CPU) and the ryzen-ai-1.7.0 Conda
environment (VitisAI/CPU). Only numpy + onnxruntime are required.

Static-shape encoders (the NPU needs fixed shapes) are handled with
`fixed_frames`: features are zero-padded to that many 10 ms frames and the
true length is passed in `length`, so the encoder's masks ignore the padding.
"""

import os
import re
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("PARAKEET_MODEL_DIR", ROOT.parent / "parakeet-stt" / "model"))
CACHE = ROOT / "cache" / "parakeet"
SR = 16000
SUBSAMPLING = 8
MAX_TOKENS_PER_STEP = 10
_SPACE = re.compile(r"\A\s|\s\B|(\s)\b")


def load_audio(path):
    """16 kHz mono float32 in [-1, 1]. WAV via stdlib, other formats via soundfile."""
    path = str(path)
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError(f"{path}: expected 16 kHz mono PCM16")
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data.astype(np.float32) / 32768.0
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32")
    if rate != SR or data.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono")
    return data


def session(path, providers, provider_options=None, threads=None, log_level=3):
    opts = ort.SessionOptions()
    opts.log_severity_level = log_level
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers,
                                provider_options=provider_options)


class Parakeet:
    def __init__(self, encoder, providers=("CPUExecutionProvider",), provider_options=None,
                 fixed_frames=None, threads=None, model_dir=MODEL_DIR, decoder="decoder_joint-model.onnx"):
        model_dir = Path(model_dir)
        self.fixed_frames = fixed_frames
        self.pre = session(model_dir / "nemo128.onnx", ["CPUExecutionProvider"], threads=threads)
        if hasattr(encoder, "run"):  # any object with InferenceSession.run semantics (hybrid NPU encoder)
            self.enc = encoder
        else:
            encoder = Path(encoder)
            if not encoder.exists():  # bare file names refer to the model folder
                encoder = model_dir / encoder
            self.enc = session(encoder, list(providers), provider_options, threads=threads)
        self.dec = session(model_dir / decoder, ["CPUExecutionProvider"], threads=threads)
        with open(model_dir / "vocab.txt", encoding="utf-8") as f:
            pairs = (line.rstrip("\n").split(" ") for line in f)
            self.vocab = {int(i): tok.replace("▁", " ") for tok, i in pairs}
        self.blank = next(i for i, t in self.vocab.items() if t == "<blk>")
        self.vocab_size = len(self.vocab)
        shapes = {x.name: x.shape for x in self.dec.get_inputs()}
        self.state_shape = (shapes["input_states_1"][0], 1, shapes["input_states_1"][2])

    # -- stages -------------------------------------------------------------
    def features(self, audio):
        feats, lens = self.pre.run(["features", "features_lens"], {
            "waveforms": audio[None, :].astype(np.float32),
            "waveforms_lens": np.array([len(audio)], dtype=np.int64)})
        return feats, lens

    def pad(self, feats):
        if self.fixed_frames is None:
            return feats
        t = feats.shape[2]
        if t > self.fixed_frames:
            raise ValueError(f"{t} feature frames > fixed {self.fixed_frames}")
        out = np.zeros((1, feats.shape[1], self.fixed_frames), dtype=np.float32)
        out[:, :, :t] = feats
        return out

    def encode(self, feats, lens):
        out, out_lens = self.enc.run(["outputs", "encoded_lengths"],
                                     {"audio_signal": self.pad(feats), "length": lens})
        return out, out_lens

    def decode(self, enc, enc_len):
        """TDT greedy decoding, same loop as onnx-asr. enc: [1, 1024, T]."""
        frames = enc[0].T  # [T, 1024]
        state = (np.zeros(self.state_shape, np.float32), np.zeros(self.state_shape, np.float32))
        tokens, t, emitted = [], 0, 0
        n = int(enc_len[0])
        while t < n:
            out, s1, s2 = self.dec.run(["outputs", "output_states_1", "output_states_2"], {
                "encoder_outputs": frames[t][None, :, None],
                "targets": np.array([[tokens[-1] if tokens else self.blank]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": state[0], "input_states_2": state[1]})
            out = np.squeeze(out)
            token = int(out[:self.vocab_size].argmax())
            step = int(out[self.vocab_size:].argmax())
            if token != self.blank:
                state = (s1, s2)
                tokens.append(token)
                emitted += 1
            if step > 0:
                t += step
                emitted = 0
            elif token == self.blank or emitted == MAX_TOKENS_PER_STEP:
                t += 1
                emitted = 0
        text = "".join(self.vocab[i] for i in tokens)
        return _SPACE.sub(lambda m: " " if m.group(1) else "", text)

    def transcribe(self, audio):
        t0 = time.perf_counter()
        feats, lens = self.features(audio)
        t1 = time.perf_counter()
        enc, enc_len = self.encode(feats, lens)
        t2 = time.perf_counter()
        text = self.decode(enc, enc_len)
        t3 = time.perf_counter()
        return text, {"pre": t1 - t0, "enc": t2 - t1, "dec": t3 - t2, "total": t3 - t0}


# -- text normalization and WER ------------------------------------------------
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _num_words(rest))
    if n < 10000 and 1100 <= n < 2000 and n % 100:  # years like 1847
        return _num_words(n // 100) + " " + _num_words(n % 100)
    if n < 1_000_000:
        rest = n % 1000
        return _num_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + _num_words(rest))
    return str(n)


def normalize(text):
    """LibriSpeech-style: lower case, words and apostrophes only, digits spelled out."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"\d+", lambda m: " " + _num_words(int(m.group())) + " ", text)
    text = re.sub(r"[^a-z' ]", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs, hyps):
    """Corpus WER (errors / reference words) over normalized word lists."""
    errors = words = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r), normalize(h)
        errors += edit_distance(r, h)
        words += len(r)
    return errors / max(words, 1), errors, words
