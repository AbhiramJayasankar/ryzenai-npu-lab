"""Cuts a dictation into pieces of at most 10 s for the 1-block NPU program.

Pieces are sent to the engine while you are still speaking, so only the last
piece is left to transcribe when the recording stops. A piece ends at a pause
(>= 0.6 s of silence once it holds 6 s of audio), or, at 10 s, at the quietest
point of its last 3 s. Pieces without speech are dropped.
"""

import numpy as np

SR = 16000
BLOCK = 480  # 30 ms
SOFT_BLOCKS = int(6.0 * SR / BLOCK)
MAX_BLOCKS = int(10.0 * SR / BLOCK)
CUT_BLOCKS = int(3.0 * SR / BLOCK)
PAUSE_BLOCKS = 20  # 0.6 s
MIN_VOICED = 6  # 0.18 s of speech
MIN_RMS = 0.004
PAD = int(0.3 * SR)  # silence appended to every piece
LIMIT = int(10.2 * SR)  # the 1-block NPU program holds 10.24 s


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class Chunker:
    def __init__(self):
        self.blocks, self.levels, self.speech = [], [], []
        self.noise = None
        self.level = 0.0
        self.pending = np.zeros(0, np.float32)
        self.total = 0

    def push(self, audio):
        """Add captured samples (any length); returns finished pieces."""
        self.total += len(audio)
        data = np.concatenate([self.pending, audio]) if len(self.pending) else audio
        n = len(data) // BLOCK * BLOCK
        self.pending = data[n:].copy()
        out = []
        for i in range(0, n, BLOCK):
            out += self._block(data[i:i + BLOCK])
        return out

    def finish(self):
        if len(self.pending):
            self.blocks.append(self.pending)
            self.levels.append(rms(self.pending))
            self.speech.append(False)
            self.pending = np.zeros(0, np.float32)
        return self._emit(len(self.blocks))

    def _block(self, b):
        r = self.level = rms(b)
        if self.noise is None:
            self.noise = r
        voiced = r > max(3.0 * self.noise, MIN_RMS)
        if r < self.noise:
            self.noise = 0.7 * self.noise + 0.3 * r
        elif not voiced:
            self.noise = 0.98 * self.noise + 0.02 * r
        self.blocks.append(b)
        self.levels.append(r)
        self.speech.append(voiced)
        n = len(self.blocks)
        if n >= SOFT_BLOCKS and not any(self.speech[-PAUSE_BLOCKS:]):
            return self._emit(n - PAUSE_BLOCKS // 2)
        if n >= MAX_BLOCKS:
            return self._emit(n - CUT_BLOCKS + int(np.argmin(self.levels[-CUT_BLOCKS:])) + 1)
        return []

    def _emit(self, k):
        piece, voiced = self.blocks[:k], sum(self.speech[:k])
        self.blocks, self.levels, self.speech = self.blocks[k:], self.levels[k:], self.speech[k:]
        if not piece or voiced < MIN_VOICED:
            return []
        audio = np.concatenate(piece)
        # Trailing silence: without it Parakeet tends to invent an ending for a
        # sentence cut mid-word ("... suggested that the menu was a very good idea").
        pad = max(0, min(PAD, LIMIT - len(audio)))
        return [np.concatenate([audio, np.zeros(pad, np.float32)])]
