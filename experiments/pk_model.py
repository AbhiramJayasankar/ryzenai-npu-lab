"""Host side of the all-NPU Parakeet encoder (experiment 012): builds the phase list
for the 24 Conformer layers, packs weights / position tables / headers, and
runs the one-submission encoder program.

Packed weights (1.2 GB BF16, bucket independent) are cached in
cache/parakeet/pk/weights.bin with their offsets; position tables and headers
are rebuilt per frame bucket (small).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import npu_direct as nd
import pk_engine as pk

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "parakeet"
WEIGHTS = CACHE / "weights"
PKDIR = CACHE / "pk"
SCALE = np.float32(1.0 / np.sqrt(128.0))
PACK_VERSION = 1
LAYERS_PER_RUN = {1: 12, 2: 6, 3: 4}  # encoder layers per NPU submission, by frame blocks


def load_layer(i):
    return dict(np.load(WEIGHTS / f"layer_{i:02d}.npz"))


def pos_tables(TPAD):
    """[layer][head] -> P^T [128, 2*TPAD] (last column zero), from the longest export."""
    f = WEIGHTS / "pos_T3072.npz"
    data = np.load(f)
    width = data["p0"].shape[-1]
    c = width // 2
    out = []
    for i in range(24):
        p = data[f"p{i}"][0, :, :, c - TPAD + 1:c + TPAD]  # [8, 128, 2T-1]
        out.append(np.concatenate([p, np.zeros((8, 128, 1), p.dtype)], axis=2))
    return out


class FilePacker:
    """Appends BF16 arrays to a file (64-byte aligned); returns element offsets."""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.size = 0

    def add(self, arr):
        arr = np.ascontiguousarray(np.asarray(arr).astype(bfloat16).reshape(-1))
        off = self.size
        self.f.write(arr.tobytes())
        self.size += arr.size
        pad = -self.size % 32
        if pad:
            self.f.write(np.zeros(pad, bfloat16).tobytes())
            self.size += pad
        return off

    def close(self):
        self.f.close()


def pack_weights(n_layers=24):
    """Pack every layer's weights in consumption order; returns the offset table."""
    PKDIR.mkdir(parents=True, exist_ok=True)
    fp = FilePacker(PKDIR / "weights.bin.tmp")
    offs = []
    w0 = load_layer(0)
    first = fp.add(pk.ln_params(w0, "norm_feed_forward1"))
    for i in range(n_layers):
        w = w0 if i == 0 else load_layer(i)
        o = {}
        o["ff1_w1"] = fp.add(pk.pack_gemm(w["feed_forward1.w1"]))
        o["ff1_w2"] = fp.add(pk.pack_gemm(w["feed_forward1.w2"]))
        o["ln_att"] = fp.add(pk.ln_params(w, "norm_self_att"))
        qkv = np.concatenate([w["att.wq"] * SCALE, w["att.wk"], w["att.wv"]], axis=1)
        o["qkv"] = fp.add(pk.pack_gemm(qkv))
        bias = np.concatenate([w["att.bias_v"].reshape(-1) * SCALE, np.zeros(2 * pk.D, np.float32)])
        cols = 3 * pk.D // pk.COLS
        o["qkv_bias"] = fp.add(np.concatenate([np.pad(bias[c * cols:(c + 1) * cols], (0, pk.WOBJ - cols))
                                               for c in range(pk.COLS)]))
        delta = (w["att.bias_u"] - w["att.bias_v"]) * SCALE
        o["att_delta"] = fp.add(np.concatenate([np.pad(np.concatenate([delta[c], delta[c + 4]]),
                                                       (0, pk.WOBJ - 256)) for c in range(pk.COLS)]))
        o["out"] = fp.add(pk.pack_gemm(w["att.wout"]))
        o["ln_conv"] = fp.add(pk.ln_params(w, "norm_conv"))
        o["pw1"] = fp.add(pk.pack_gemm(w["conv.pw1"]))
        conv = []
        for c in range(pk.COLS):
            ch = slice(256 * c, 256 * (c + 1))
            p = np.concatenate([w["conv.dw"][ch].T.reshape(-1), w["conv.dw_b"][ch]])
            conv.append(np.pad(p, (0, 2 * pk.WOBJ - p.size)))
        o["conv"] = fp.add(np.concatenate(conv))
        o["pw2"] = fp.add(pk.pack_gemm(w["conv.pw2"]))
        o["ln_ff2"] = fp.add(pk.ln_params(w, "norm_feed_forward2"))
        o["ff2_w1"] = fp.add(pk.pack_gemm(w["feed_forward2.w1"]))
        o["ff2_w2"] = fp.add(pk.pack_gemm(w["feed_forward2.w2"]))
        o["ln_out"] = fp.add(pk.ln_params(w, "norm_out"))
        if i + 1 < n_layers:
            w0 = load_layer(i + 1)
            o["ln_next"] = fp.add(pk.ln_params(w0, "norm_feed_forward1"))
        offs.append(o)
        print(f"packed layer {i}", flush=True)
    fp.close()
    table = {"version": pack_version(n_layers), "first": first, "layers": offs, "size": fp.size}
    os.replace(PKDIR / "weights.bin.tmp", PKDIR / "weights.bin")
    (PKDIR / "weights.json").write_text(json.dumps(table))
    return table


def pack_version(n_layers):
    return f"{PACK_VERSION}_{n_layers}"


def weight_table(n_layers=24):
    """Offsets of the packed weights; always packs all 24 layers (programs with
    fewer layers use the first ones)."""
    j = PKDIR / "weights.json"
    if j.exists():
        table = json.loads(j.read_text())
        if table["version"] == pack_version(24):
            return table
    return pack_weights(24)


def build_phases(lay, table, n_layers, pos_offs):
    """The encoder as a list of phases (see pk_engine)."""
    G, S = pk.Gemm, pk.S
    rec = lay.rec
    ph = [pk.Ln(pk.LN_SINGLE, 0.0, [table["first"]])]
    for i in range(n_layers):
        o = table["layers"][i]
        last = i == n_layers - 1
        ph.append(G(1024, 4096, rec(pk.FA), S, rec(pk.FBIG), S, "weights", o["ff1_w1"], epi=pk.EPI_SWISH))
        ph.append(G(4096, 1024, rec(pk.FBIG), S, rec(pk.FY), S, "weights", o["ff1_w2"]))
        ph.append(pk.Ln(pk.LN_SINGLE, 0.5, [o["ln_att"]]))
        ph.append(G(1024, 3072, rec(pk.FA), S, rec(pk.FBIG), S, "weights", o["qkv"],
                    bias_off=o["qkv_bias"]))
        for h in range(pk.HEADS):
            ph.append(G(128, 2 * lay.TPAD, rec(pk.FBIG + h * pk.DK), S, lay.bd(h), lay.R, "pos",
                        pos_offs[i][h]))
        ph.append(pk.Att(o["att_delta"]))
        ph.append(G(1024, 1024, rec(pk.FO), S, rec(pk.FY), S, "weights", o["out"]))
        ph.append(pk.Ln(pk.LN_SINGLE, 1.0, [o["ln_conv"]]))
        ph.append(G(1024, 2048, rec(pk.FA), S, rec(pk.FBIG), S, "weights", o["pw1"]))
        ph.append(pk.Conv(o["conv"]))
        ph.append(G(1024, 1024, rec(pk.FO), S, rec(pk.FY), S, "weights", o["pw2"]))
        ph.append(pk.Ln(pk.LN_SINGLE, 1.0, [o["ln_ff2"]]))
        ph.append(G(1024, 4096, rec(pk.FA), S, rec(pk.FBIG), S, "weights", o["ff2_w1"], epi=pk.EPI_SWISH))
        ph.append(G(4096, 1024, rec(pk.FBIG), S, rec(pk.FY), S, "weights", o["ff2_w2"]))
        if last:
            ph.append(pk.Ln(pk.LN_FINAL, 0.5, [o["ln_out"]]))
        else:
            ph.append(pk.Ln(pk.LN_DOUBLE, 0.5, [o["ln_out"], o["ln_next"]]))
    return ph


def plan(nblk, n_layers=24, max_phases=None):
    """Layout, phase list, packed position tiles and weight table (no NPU access)."""
    lay = pk.Layout(nblk)
    table = weight_table(n_layers)
    pos = pos_tables(lay.TPAD)
    pos_packed, pos_offs, off = [], [], 0
    for i in range(n_layers):
        row = []
        for h in range(pk.HEADS):
            a = pk.pack_gemm(pos[i][h])
            row.append(off)
            pos_packed.append(a)
            off += a.size
        pos_offs.append(row)
    phases = build_phases(lay, table, n_layers, pos_offs)[:max_phases]
    return lay, phases, pos_packed, table


def check_plan(nblk, n_layers=24):
    """Run pk_engine.check on the full encoder program without touching the NPU."""
    lay, phases, pos_packed, table = plan(nblk, n_layers)
    ctrl_len = pk.assign_headers(phases)
    lens = {"ctrl": ctrl_len, "weights": table["size"], "pos": sum(a.size for a in pos_packed),
            "io": lay.io_len}
    return pk.check(phases, lay, lens)


class PkEncoder:
    """The encoder layers on the NPU for up to TPAD = 128*nblk frames."""

    def __init__(self, nblk=1, n_layers=24, tag=None, max_phases=None):
        t0 = time.perf_counter()
        self.lay, self.phases, pos_packed, table = plan(nblk, n_layers, max_phases)
        lay = self.lay
        self.n_layers = n_layers
        off = sum(a.size for a in pos_packed)
        ctrl_len = pk.assign_headers(self.phases)
        self.lens = {"ctrl": ctrl_len, "weights": table["size"], "pos": off, "io": lay.io_len}
        self.ctrl = nd.Buffer(ctrl_len, bfloat16)
        self.ctrl.array[:] = pk.ctrl_array(self.phases, lay)
        self.ctrl.to_device()
        self.pos = nd.Buffer(off, bfloat16)
        self.pos.array[:] = np.concatenate(pos_packed)
        self.pos.to_device()
        self.weights = nd.Buffer(table["size"], bfloat16)
        mm = np.memmap(PKDIR / "weights.bin", dtype=np.uint16, mode="r")
        dst = self.weights.array.view(np.uint16)
        step = 64 << 20
        for s in range(0, table["size"], step):
            dst[s:s + step] = mm[s:s + step]
        del mm
        self.weights.to_device()
        self.io = nd.Buffer(lay.io_len, bfloat16)
        self.io.array[:] = 0
        self.io.to_device()
        self.pack_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        # Split into submissions of a few layers each (~150 ms of NPU work at
        # most): one 800 ms submission for 30 s of audio hung the NPU, while
        # every piece ran on its own. All parts share one hardware context
        # (same core program and data paths, different instruction streams).
        tag = tag or f"enc{n_layers}"
        per = LAYERS_PER_RUN[nblk]
        cuts = [0] + [1 + 22 * k for k in range(per, n_layers, per)] + [len(self.phases)]
        parts = [self.phases[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
        paths = [nd.compile_design(pk.design(nblk, ph, self.lens, f"{tag}_p{i}"))
                 for i, ph in enumerate(parts)]
        self.prog = nd.Program(*paths[0])
        self.entries = [self.prog.default] + [self.prog.entry(insts) for _, insts in paths[1:]]
        self.compile_s = time.perf_counter() - t0

    def run(self, hidden, valid):
        """hidden [T, 1024] FP32 prefix output (T <= TPAD), valid frames -> [valid, 1024]."""
        lay = self.lay
        a = self.io.array
        rec = a[lay.REC:lay.REC + (lay.TPAD + 8) * pk.S].reshape(lay.TPAD + 8, pk.S)
        rec[:, pk.FX:pk.FX + pk.D] = 0
        rec[:, pk.FY:pk.FY + pk.D] = 0
        rec[:hidden.shape[0], pk.FX:pk.FX + pk.D] = hidden.astype(bfloat16)
        a[lay.AUX:lay.AUX + 2] = np.array([valid], np.int32).view(bfloat16)
        self.io.to_device()
        t = time.perf_counter()
        for e in self.entries:
            e(self.ctrl, self.weights, self.pos, self.io)
        self.npu_s = time.perf_counter() - t
        self.io.from_device()
        return rec[:valid, pk.FX:pk.FX + pk.D].astype(np.float32)

    def bd(self):
        """Debug view of the position-score region: [8, TPAD, 2*TPAD]."""
        lay = self.lay
        return self.io.array[lay.BD:lay.BD + pk.HEADS * lay.TPAD * lay.R].astype(np.float32).reshape(
            pk.HEADS, lay.TPAD, lay.R)

    def field(self, name, n=None):
        """Debug view of an io record field: [TPAD, width] (BF16 -> FP32)."""
        lay = self.lay
        off, width = {"x": (pk.FX, 1024), "a": (pk.FA, 1024), "y": (pk.FY, 1024),
                      "big": (pk.FBIG, 4096), "o": (pk.FO, 1024)}[name]
        rec = self.io.array[lay.REC:lay.REC + (lay.TPAD + 8) * pk.S].reshape(lay.TPAD + 8, pk.S)
        return rec[:lay.TPAD, off:off + width].astype(np.float32)
