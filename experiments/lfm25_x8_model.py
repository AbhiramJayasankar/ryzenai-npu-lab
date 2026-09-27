"""LFM2.5-230M decode on the Phoenix NPU with eight-core programs.

The host tokenizes, prepares rotary constants, and submits one NPU program
run per token: all 14 layers and the vocabulary head. The token embedding
enters the NPU by DMA straight from the embedding table (an XRT sub-buffer at
the token's row), and the NPU selects the next token. No model arithmetic runs
on the CPU.
"""

import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import lfm25_x8 as x8
import npu_direct as nd
from lfm25_checkpoint import BF16Checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
TIERS = (64, 256, 1024, 4096)
_FREQUENCIES = np.float32(1) / np.power(
    np.float32(1_000_000), np.arange(0, 64, 2, dtype=np.float32) / np.float32(64))


def rotary(position):
    values = np.float32(position) * _FREQUENCIES
    phase = np.concatenate((values, values))
    return np.cos(phase).astype(bfloat16), np.sin(phase).astype(bfloat16)


class X8Model:
    def __init__(self, tiers=TIERS, log=print):
        self.tiers = tuple(sorted(tiers))
        self.capacity = self.tiers[-1]
        assert self.capacity <= x8.MAX_CAPACITY
        t0 = time.perf_counter()
        # Every variant has the same cores, FIFOs and routes; they differ only
        # in the runtime DMA sequence, so all run on one hardware context.
        paths = {(tier, head): nd.compile_design(x8.token_design(tier, head))
                 for tier in self.tiers for head in (True, False)}
        self.program = nd.Program(paths[(self.tiers[0], True)][0])
        self.entries = {key: self.program.entry(insts) for key, (_x, insts) in paths.items()}
        # Batched prompt processing is a second array configuration; switching
        # to it costs one context switch per prompt, not per token.
        prefill = {tier: nd.compile_design(x8.prefill_design(tier)) for tier in self.tiers}
        self.prefill_program = nd.Program(prefill[self.tiers[0]][0])
        self.prefill_entries = {tier: self.prefill_program.entry(insts)
                                for tier, (_x, insts) in prefill.items()}
        log(f"programs ready in {time.perf_counter() - t0:.1f} s")

        ck = BF16Checkpoint(MODEL / "model.safetensors")
        self.table = nd.Buffer(x8.VOCAB * x8.HIDDEN)
        self.table.write(ck.load("model.embed_tokens.weight"))
        self.weights = nd.Buffer(x8.W_LEN)
        x8.pack_weights(ck, self.weights.array)
        self.weights.to_device()
        self.io = nd.Buffer(x8.IO_LEN)
        self.caches = nd.Buffer(x8.CACHES_LEN)
        self.reset()
        log(f"weights packed in {time.perf_counter() - t0:.1f} s")

    def reset(self):
        self.position = 0
        self.io.write(np.zeros(x8.IO_LEN, dtype=bfloat16))

    def _set_token_aux(self, position, tier):
        cos, sin = rotary(position)
        aux = self.io.array
        aux[0:64] = cos
        aux[64:128] = sin
        aux[128:132] = np.array([position, tier // 4], dtype=np.int32).view(bfloat16)
        self.io.to_device(0, 132)

    def hidden(self):
        """Last layer output, gathered from the stage-4 slots."""
        raw = self.io.from_device(x8.SLOT(4, 0), x8.CORES * x8.OBJ)
        return np.asarray(raw).reshape(x8.CORES, x8.OBJ)[:, :128].reshape(-1)

    def conv_state(self, conv_index):
        raw = self.io.from_device(x8.STATE(conv_index, 0), x8.CORES * x8.OBJ)
        return x8.unpack_conv_state(raw)

    def kv(self, att_index, position):
        """(keys, values) of shape (8, position + 1, 64) for one attention layer."""
        base = att_index * x8.CACHE_LEN
        self.caches.from_device(base, x8.CACHE_LEN)
        region = self.caches.array[base:base + x8.CACHE_LEN]
        cores = np.stack([region[k * x8.CACHE_STRIDE:k * x8.CACHE_STRIDE + (position + 1) * 128]
                          for k in range(x8.CORES)]).reshape(x8.CORES, position + 1, 128)
        return cores[:, :, :64], cores[:, :, 64:]

    def prefill_batch(self, token_ids):
        """Run x8.PB prompt positions in one submission (no token selection)."""
        assert len(token_ids) == x8.PB
        position = self.position
        if position + x8.PB > self.capacity:
            raise ValueError(f"KV capacity {self.capacity} reached")
        tier = next(t for t in self.tiers if t >= position)
        aux = self.io.array
        for t in range(x8.PB):
            cos, sin = rotary(position + t)
            base = x8.PAUX(t)
            aux[base:base + 64] = cos
            aux[base + 64:base + 128] = sin
            aux[base + 128:base + 132] = np.array([position, tier // 4],
                                                  dtype=np.int32).view(bfloat16)
        self.io.to_device(x8.PAUX(0), x8.PB * x8.OBJ)
        # Stage the embedding rows (a copy, no arithmetic) for the NPU to read.
        rows = self.table.array.reshape(x8.VOCAB, x8.HIDDEN)[np.asarray(token_ids)]
        self.io.write(rows, x8.PEMB(0))
        append = self.caches.view(position * 128, x8.APPEND_LEN_P)
        self.prefill_entries[tier](self.weights, self.io, self.caches, append)
        self.position += x8.PB

    def prefill(self, token_ids):
        """Process prompt tokens; the last one also selects the next token.
        Returns the NPU-selected token id."""
        body = list(token_ids[:-1])
        full = len(body) - len(body) % x8.PB
        for start in range(0, full, x8.PB):
            self.prefill_batch(body[start:start + x8.PB])
        for token in body[full:]:
            self.step(token, select=False)
        return self.step(token_ids[-1])

    def prefill_hidden(self, token):
        """Last-layer output of token t of the most recent prompt batch."""
        raw = self.io.from_device(x8.PSLOT(4, token, 0), x8.CORES * x8.OBJ)
        return np.asarray(raw).reshape(x8.CORES, x8.OBJ)[:, :128].reshape(-1)

    def step(self, token_id, select=True):
        """Run one position on the NPU. Returns the NPU-selected next token id
        when `select` is true."""
        if self.position >= self.capacity:
            raise ValueError(f"KV capacity {self.capacity} reached")
        position = self.position
        tier = next(t for t in self.tiers if t >= position)
        self._set_token_aux(position, tier)
        row = self.table.view(int(token_id) * x8.HIDDEN, x8.HIDDEN)
        append = self.caches.view(position * 128, x8.APPEND_LEN)
        self.entries[(tier, select)](self.weights, row, self.io, self.caches, append,
                                     self.table)
        self.position += 1
        if not select:
            return None
        return int(self.io.from_device(x8.SLOT(6, 0), 2).view(np.int32)[0])
