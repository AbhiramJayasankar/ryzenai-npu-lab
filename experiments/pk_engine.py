"""All-NPU Parakeet encoder engine for the Phoenix NPU (experiment 012): IRON design,
data layout and weight packing. The core program is kernels/pk/pk_core.cc.

Topology (4 columns x 4 rows of compute tiles, rows 2-5); core (c, r) = Tile(c, 2 + r):
  W stream  per column c: shim (c,0) -> memory tile -> broadcast to the 4
            cores of column c. Headers, prologue objects, weight tiles and
            the inputs of vector phases. Objects of WOBJ = 2048 BF16.
  X stream  per row r: shim (r,0) -> memory tile, which re-blocks 32 x 64
            activation tiles into mmul A blocks -> broadcast to the 4 cores
            of row r. GEMM activations only.
  C stream  per core -> memory-tile join of the column's 4 cores -> shim
            (c,0) -> DDR. Objects of 1024 BF16 per core.
One program handles TPAD = 128*nblk encoder frames (10.2 s of audio per block).

Runtime-sequence buffers: ctrl (phase headers, per bucket), weights (tiles
and small parameters, bucket independent), pos (projected relative-position
tables as GEMM tiles, per bucket), io (activations).

io layout (BF16 elements): AUX object (int32 valid frame count at [0]), then
TPAD + 8 frame records of S elements with fields
  X   residual stream (1024)      A   LayerNorm output (1024)
  Y   GEMM output for residuals   BIG FFN hidden (4096) / q|k|v (3072) /
                                      pointwise-1 output a|b (2048)
  O   attention output / convolution-module output (1024)
then the position scores of the 8 heads, [TPAD][2*TPAD] each.
"""

import hashlib
import struct
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, In, Kernel, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

KDIR = Path(__file__).resolve().parent / "kernels" / "pk"
KVER = hashlib.sha1(b"".join(p.read_bytes() for p in sorted(KDIR.glob("*.*")))).hexdigest()[:10]
COLS = ROWS = 4
WOBJ = 2048
KT, NT = 64, 32          # GEMM tile: 64 input rows x 32 output columns
D, HEADS, DK = 1024, 8, 128
OP_GEMM, OP_LN, OP_ATT, OP_CONV = 0, 1, 2, 3
EPI_SWISH, EPI_BIAS = 1, 2
LN_SINGLE, LN_DOUBLE, LN_FINAL = 0, 1, 2
CTL_LEN = 32
PAR_LEN = 4096
# Attention passes per task group: 3 or more (13+ buffer descriptors on the
# column's shim tile) hung the NPU at 256 frames; 1 and 2 work.
ATT_GROUP = int(__import__("os").environ.get("PK_ATT_GROUP", "2"))

# io record fields
S = 8192
FX, FA, FY, FBIG, FO = 0, 1024, 2048, 3072, 7168


class Layout:
    """Sizes and io offsets for a program handling nblk blocks of 128 encoder
    frames (TPAD = 128*nblk; 10.2 s of audio per block). Every core works on
    32-frame GEMM tiles and 1024-element output objects whatever TPAD is;
    GEMM phases run once per frame block, the other phases loop over all
    frames."""

    def __init__(self, nblk):
        self.nblk = nblk
        self.TR = 32
        self.TPAD = 128 * nblk
        self.AUX = 0
        self.REC = WOBJ
        self.BD = self.REC + (self.TPAD + 8) * S
        self.R = 2 * self.TPAD                      # BD row stride
        self.L = self.TPAD + 32                     # BD values streamed per query
        self.io_len = self.BD + HEADS * self.TPAD * self.R + 2 * self.L + 4096
        att_bytes = 128 * 16 * 2 + self.TPAD * 16 * 2 + 128 * 16 * 4 + 128
        self.scr = max(1024, att_bytes // 4, (9 * 128 + 16 * 128) // 4)

    def rec(self, field, frame=0):
        return self.REC + frame * S + field

    def bd(self, head):
        return self.BD + head * self.TPAD * self.R


def types(nblk):
    lay = Layout(nblk)
    return dict(
        W=np.ndarray[(WOBJ,), np.dtype[bfloat16]],
        X=np.ndarray[(lay.TR * KT,), np.dtype[bfloat16]],
        C=np.ndarray[(1024,), np.dtype[bfloat16]],
        CJ=np.ndarray[(ROWS * 1024,), np.dtype[bfloat16]],
        SCR=np.ndarray[(lay.scr,), np.dtype[np.float32]],
        PAR=np.ndarray[(PAR_LEN,), np.dtype[bfloat16]],
        CTL=np.ndarray[(CTL_LEN,), np.dtype[np.int32]],
    )


def kernels(nblk):
    """Core program: pk_core.cc (-Oz, headers and vector phases) and pk_fast.cc
    (-O2: GEMM step and the hot vector routines) linked into every core."""
    t = types(nblk)
    flags = [f"-DPK_TPAD={128 * nblk}"]
    inc = [config.cxx_header_path(), str(KDIR)]
    hdr = ExternalFunction(
        "pk_hdr", source_file=str(KDIR / "pk_core.cc"),
        arg_types=[t["W"], t["SCR"], t["PAR"], t["CTL"], np.int32, np.int32],
        include_dirs=inc, compile_flags=["-Oz"] + flags)
    wx = ExternalFunction(
        "pk_wx", source_file=str(KDIR / "pk_fast.cc"),
        arg_types=[t["W"], t["X"], t["SCR"], t["PAR"], t["CTL"], np.int32],
        include_dirs=inc, compile_flags=["-O2"] + flags)
    obj = hdr.object_file_name
    w = Kernel("pk_w", obj, [t["W"], t["SCR"], t["PAR"], t["CTL"], np.int32])
    out = Kernel("pk_out", obj, [t["C"], t["SCR"], t["PAR"], t["CTL"], np.int32])
    return hdr, wx, w, out


def a_dims():
    """Memory tile -> core: a 32 x 64 row-major tile as 4x8 mmul A blocks."""
    m, k, r, s = 32, KT, 4, 8
    return [(m // r, r * k), (k // s, s), (r, k), (s, 1)]


def _core(W, X, C, scr, par, ctl, hdr, wx, wk, out):
    w = W.acquire(1)
    hdr(w, scr, par, ctl, 0, 0)
    W.release(1)
    for i in range_(ctl[0]):
        w = W.acquire(1)
        hdr(w, scr, par, ctl, 1, i)
        W.release(1)
    for _ in range_(ctl[1]):
        for i in range_(ctl[2]):
            x = X.acquire(1)
            w = W.acquire(1)
            wx(w, x, scr, par, ctl, i)
            W.release(1)
            X.release(1)
        for i in range_(ctl[3]):
            w = W.acquire(1)
            wk(w, scr, par, ctl, i)
            W.release(1)
        for j in range_(ctl[4]):
            c = C.acquire(1)
            out(c, scr, par, ctl, j)
            C.release(1)


def _ctl(c, r):
    v = np.zeros(CTL_LEN, np.int32)
    v[6], v[7] = r, c
    return v


def tap(total, offset, sizes, strides):
    sizes, strides = list(sizes), list(strides)
    while len(sizes) < 4:
        sizes, strides = [1] + sizes, [0] + strides
    return TensorAccessPattern((total,), offset, sizes, strides)


def header(op, npro, nb, nwx, nw, nout, epi=0, mode=0, scale=0.0, pph=0, nq=0, nbd=0, bdl=0,
           nch=0):
    h = np.zeros(WOBJ // 2, np.int32)
    h[:14] = [op, npro, nb, nwx, nw, nout, epi, mode,
              struct.unpack("<i", struct.pack("<f", scale))[0], pph, nq, nbd, bdl, nch]
    return h.view(bfloat16)


# ---------------------------------------------------------------- phases
# Each phase knows its header (one per column), and issues its fills and
# drains: fills are (buffer name, offset, sizes, strides).

class Phase:
    hdr_off = None

    def headers(self, lay):
        h = self.header(lay)
        return [h] * COLS

    def groups(self, lay, c):
        """Per column c: list of task groups, each (w_fills, x_fills, drains)."""
        raise NotImplementedError


class Gemm(Phase):
    def __init__(self, K, N, x_off, x_stride, out_off, out_stride, wbuf, w_off,
                 epi=0, bias_off=None):
        self.__dict__.update(K=K, N=N, x_off=x_off, x_stride=x_stride, out_off=out_off,
                             out_stride=out_stride, wbuf=wbuf, w_off=w_off, epi=epi,
                             bias_off=bias_off)

    def chunks(self):
        return self.N // COLS // NT

    def header(self, lay):
        return header(OP_GEMM, int(self.bias_off is not None), self.chunks() * lay.nblk,
                      self.K // KT, 0, 1, self.epi | (EPI_BIAS if self.bias_off is not None else 0),
                      nch=self.chunks())

    def groups(self, lay, c):
        """Weights stream once per 128-frame block (repeat dimension); one
        activation fill and one drain per block."""
        w = [("ctrl", self.hdr_off + c * WOBJ, [WOBJ], [1])]
        if self.bias_off is not None:
            w.append(("weights", self.bias_off + c * WOBJ, [WOBJ], [1]))
        per_col = self.K * self.N // COLS
        w.append((self.wbuf, self.w_off + c * per_col, [lay.nblk, 1, 1, per_col], [0, 0, 0, 1]))
        x = [("io", self.x_off + (b * 128 + c * lay.TR) * self.x_stride,
              [self.chunks(), self.K // KT, lay.TR, KT], [0, KT, self.x_stride, 1])
             for b in range(lay.nblk)]
        d = [("io", self.out_off + b * 128 * self.out_stride + c * (self.N // COLS),
              [self.chunks(), 128, NT], [NT, self.out_stride, 1]) for b in range(lay.nblk)]
        return [(w, x, d)]


class Ln(Phase):
    """x += scale * y; a = LN(x). mode DOUBLE: x = LN_out(x + scale*y), a =
    LN_next(x); FINAL: x = LN_out(x + scale*y)."""

    def __init__(self, mode, scale, param_offs):
        self.mode, self.scale, self.param_offs = mode, scale, param_offs

    def header(self, lay):
        return header(OP_LN, len(self.param_offs), lay.TPAD // 16, 0, 4, 2, mode=self.mode,
                      scale=self.scale)

    def groups(self, lay, c):
        """Column c: frames [c*TPAD/4, (c+1)*TPAD/4) in blocks of 4 (one per core)."""
        n = lay.TPAD // 4
        base = lay.rec(FX, c * n)
        w = [("ctrl", self.hdr_off + c * WOBJ, [WOBJ], [1])]
        w += [("weights", off, [WOBJ], [1]) for off in self.param_offs]
        w.append(("io", base, [n, 2, D], [S, FY - FX, 1]))
        d = [("io", base, [n // 4, 2, 4, D], [4 * S, FA - FX, S, 1])]
        return [(w, [], d)]


class Att(Phase):
    """Heads c and c+4 on column c; the column's cores split each head's queries
    in quarters, 16 queries per pass."""

    def __init__(self, delta_off):
        self.delta_off = delta_off

    def header(self, lay):
        P = lay.TPAD // 4 // 16
        nq, nbd, nkv = 4, lay.L // 32, lay.TPAD // 8
        return header(OP_ATT, 2, 2 * P, 0, nq + nbd + nkv, 2, pph=P, nq=nq, nbd=nbd, bdl=lay.L)

    def groups(self, lay, c):
        """Task groups of at most 3 passes (DMA descriptor budget), each with
        the drain of its passes' outputs (2 objects of 8 rows per pass)."""
        P, T, R = lay.TPAD // 4 // 16, lay.TPAD, lay.R
        out = []
        for hi, h in enumerate((c, c + 4)):
            for p0 in range(0, P, ATT_GROUP):
                ps = range(p0, min(P, p0 + ATT_GROUP))
                w = []
                if hi == 0 and p0 == 0:
                    w += [("ctrl", self.hdr_off + c * WOBJ, [WOBJ], [1]),
                          ("io", lay.AUX, [WOBJ], [1]),
                          ("weights", self.delta_off + c * WOBJ, [WOBJ], [1])]
                for p in ps:
                    w.append(("io", lay.rec(FBIG + h * DK, p * 16), [4, 16, DK], [T // 4 * S, S, 1]))
                    w.append(("io", lay.bd(h) + T - 2 + p * 16 * (R - 1),
                              [4, 8, 2, lay.L], [T // 4 * (R - 1), 2 * R - 2, R, 1]))
                    w.append(("io", lay.rec(FBIG + D + h * DK), [T, 2, DK], [S, D, 1]))
                d = [("io", lay.rec(FO + h * DK, p0 * 16), [2 * len(ps), 4, 8, DK],
                      [8 * S, T // 4 * S, S, 1])]
                out.append((w, [], d))
        return out


class Conv(Phase):
    def __init__(self, param_off):
        self.param_off = param_off

    def header(self, lay):
        return header(OP_CONV, 4, lay.TPAD // 16, 0, 4, 1)

    def groups(self, lay, c):
        w = [("ctrl", self.hdr_off + c * WOBJ, [WOBJ], [1]),
             ("io", lay.AUX, [WOBJ], [1]),
             ("weights", self.param_off + c * 2 * WOBJ, [2 * WOBJ], [1]),
             ("io", lay.rec(FBIG + 256 * c), [lay.TPAD + 4, 2, 256], [S, D, 1])]
        d = [("io", lay.rec(FO + 256 * c), [lay.TPAD // 16, 4, 16, 64], [16 * S, 64, S, 1])]
        return [(w, [], d)]


# Shim-tile buffer descriptors per task group (W, X and drain tasks of one
# column's shim tile). 13 hung the NPU (attention at 256 frames); 10 ran.
MAX_BDS = 10


def _extent(off, sizes, strides):
    """Lowest and highest element index an access pattern touches."""
    lo = hi = off
    for n, st in zip(sizes, strides):
        if n > 1:
            if st >= 0:
                hi += (n - 1) * st
            else:
                lo += (n - 1) * st
    return lo, hi


def check(phases, lay, lens):
    """Static check before anything reaches the NPU: per core, the objects the
    core program consumes and produces (decoded from each phase header) must
    equal what the runtime sequence delivers and drains; every task group must
    stay within MAX_BDS descriptors per shim tile; every access pattern must
    stay inside its buffer. Returns a list of problems (empty = consistent)."""
    xobj, cj = lay.TR * KT, ROWS * 1024
    problems = []
    for pi, ph in enumerate(phases):
        h = ph.header(lay).view(np.int32)
        npro, nb, nwx, nw, nout = (int(v) for v in h[1:6])
        need = {"W": 1 + npro + nb * (nwx + nw), "X": nb * nwx, "C": nb * nout}
        for c in range(COLS):
            have = {"W": 0, "X": 0, "C": 0}
            for gi, (wf, xf, d) in enumerate(ph.groups(lay, c)):
                if len(wf) + len(xf) + len(d) > MAX_BDS:
                    problems.append(f"phase {pi} ({type(ph).__name__}) col {c} group {gi}: "
                                    f"{len(wf) + len(xf) + len(d)} descriptors > {MAX_BDS}")
                for kind, tasks, unit in (("W", wf, WOBJ), ("X", xf, xobj), ("C", d, cj)):
                    for name, off, sz, st in tasks:
                        n = int(np.prod(sz))
                        if n % unit:
                            problems.append(f"phase {pi} col {c}: {kind} task of {n} elements "
                                            f"is not whole objects of {unit}")
                        have[kind] += n // unit
                        lo, hi = _extent(off, sz, st)
                        if lo < 0 or hi >= lens[name]:
                            problems.append(f"phase {pi} col {c}: {kind} access [{lo}, {hi}] "
                                            f"outside {name} ({lens[name]})")
            for kind in need:
                if have[kind] != need[kind]:
                    problems.append(f"phase {pi} ({type(ph).__name__}) col {c}: {kind} objects "
                                    f"delivered {have[kind]}, core expects {need[kind]}")
    return problems


def assign_headers(phases):
    """Give each phase its place in ctrl; returns the ctrl array for a layout."""
    for i, ph in enumerate(phases):
        ph.hdr_off = i * COLS * WOBJ
    return len(phases) * COLS * WOBJ


def ctrl_array(phases, lay):
    out = np.zeros(len(phases) * COLS * WOBJ, bfloat16)
    for ph in phases:
        for c, h in enumerate(ph.headers(lay)):
            out[ph.hdr_off + c * WOBJ:ph.hdr_off + (c + 1) * WOBJ] = h
    return out


def design(nblk, phases, lens, tag):
    """Program running `phases` in one submission. lens: element counts of the
    ctrl, weights, pos and io buffers."""
    lay = Layout(nblk)
    t = types(nblk)
    names = ("ctrl", "weights", "pos", "io")
    problems = check(phases, lay, lens)
    if problems:
        raise ValueError("inconsistent NPU program (not built):\n  " + "\n  ".join(problems[:20]))
    # IRON's compile cache keys on the tag only: include a digest of the whole
    # runtime sequence (every fill and drain) and of this file.
    seq =repr([[ph.groups(lay, c) for c in range(COLS)] for ph in phases]) + repr(lens)
    sdig = hashlib.sha1(seq.encode() + Path(__file__).read_bytes()).hexdigest()[:10]
    tag = f"pk_{tag}_B{nblk}_{KVER}_{sdig}"

    @iron.jit(tag=tag)
    def pk(ctrl: In, weights: In, pos: In, io: Out, *, tag: CompileTime[str]):
        hdr, wx, wk, out = kernels(nblk)
        W3 = [ObjectFifo(t["W"], name=f"w3_{c}", depth=2) for c in range(COLS)]
        W2 = [W3[c].cons().forward(obj_type=t["W"], name=f"w2_{c}", depth=2, tile=Tile(c, 1))
              for c in range(COLS)]
        X3 = [ObjectFifo(t["X"], name=f"x3_{r}", depth=2) for r in range(ROWS)]
        X2 = [X3[r].cons().forward(obj_type=t["X"], name=f"x2_{r}", depth=2,
                                   dims_to_stream=a_dims(), tile=Tile(r, 1))
              for r in range(ROWS)]
        CJ = [ObjectFifo(t["CJ"], name=f"cj_{c}", depth=2) for c in range(COLS)]
        C1 = [CJ[c].prod().join([r * 1024 for r in range(ROWS)], tile=Tile(c, 1),
                                obj_types=[t["C"]] * ROWS,
                                names=[f"c1_{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
              for c in range(COLS)]
        workers = [Worker(_core, [W2[c].cons(), X2[r].cons(), C1[c][r].prod(),
                                  Buffer(t["SCR"], name=f"scr_{c}_{r}"),
                                  Buffer(t["PAR"], name=f"par_{c}_{r}"),
                                  Buffer(t["CTL"], initial_value=_ctl(c, r), name=f"ctl_{c}_{r}"),
                                  hdr, wx, wk, out],
                          tile=Tile(c, 2 + r), dynamic_objfifo_lowering=True, stack_size=2560)
                   for c in range(COLS) for r in range(ROWS)]

        def sequence(ctrl, w, pos, io, wp, xp, cc):
            bufs = dict(zip(names, (ctrl, w, pos, io)))
            for ph in phases:
                per_col = [ph.groups(lay, c) for c in range(COLS)]
                for gi in range(len(per_col[0])):
                    g = TaskGroup()
                    for c in range(COLS):
                        wf, xf, _ = per_col[c][gi]
                        for name, off, sz, st in wf:
                            wp[c].fill(bufs[name], tap=tap(lens[name], off, sz, st), group=g)
                        for name, off, sz, st in xf:
                            xp[c].fill(bufs[name], tap=tap(lens[name], off, sz, st), group=g)
                    for c in range(COLS):
                        for name, off, sz, st in per_col[c][gi][2]:
                            cc[c].drain(bufs[name], tap=tap(lens[name], off, sz, st), group=g,
                                        wait=True)
                    g.finish()

        rt = Runtime(sequence, [
            *[np.ndarray[(lens[n],), np.dtype[bfloat16]] for n in names],
            [W3[c].prod(tile=Tile(c, 0)) for c in range(COLS)],
            [X3[r].prod(tile=Tile(r, 0)) for r in range(ROWS)],
            [CJ[c].cons(tile=Tile(c, 0)) for c in range(COLS)],
        ])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return pk


# ---------------------------------------------------------------- packing

def pack_gemm(wmat):
    """W [K, N] (y = x . W) -> weight stream order: column group, chunk of 32
    columns, K step of 64 rows, then mmul B blocks [kb 8][nb 8][8][4]."""
    K, N = wmat.shape
    ng = N // COLS
    t = np.asarray(wmat, np.float32).reshape(K // KT, KT // 8, 8, COLS, ng // NT, NT // 4, 4)
    # axes: ks, kb, i, c, ch, nb, j -> c, ch, ks, kb, nb, i, j
    return np.ascontiguousarray(t.transpose(3, 4, 0, 1, 5, 2, 6)).reshape(-1).astype(bfloat16)


class Packer:
    """Appends BF16 arrays to one buffer, 64-byte aligned; returns offsets."""

    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr):
        arr = np.asarray(arr, bfloat16).reshape(-1)
        off = self.size
        self.parts.append(arr)
        self.size += arr.size
        pad = -self.size % 32
        if pad:
            self.parts.append(np.zeros(pad, bfloat16))
            self.size += pad
        return off

    def array(self):
        return np.concatenate(self.parts) if self.parts else np.zeros(32, bfloat16)


def ln_params(w, name):
    return np.concatenate([w[f"{name}.w"], w[f"{name}.b"]])
