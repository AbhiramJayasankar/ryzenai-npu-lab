"""Parakeet encoder with the Conformer linear layers on the NPU (custom IRON GEMM) and
everything else on the CPU in NumPy. Experiment 011.

Every linear layer of a Conformer block is cut into 1024x1024 weight blocks
(FFN linear1 = 4 column blocks, FFN linear2 = 4 row blocks whose partial
products the CPU sums, q/k/v/out/pointwise2 = 1 block, pointwise1 = 2 column
blocks), so a single compiled GEMM program [M x 1024] x [1024 x 1024] (BF16 in,
FP32 out, IRON whole-array design: 16 cores with aie::mmul) serves them all
from one hardware context. Weights are converted to BF16 once and kept in
XRT buffers; activations are rounded to BF16 on the way in.

CPU side (FP32 NumPy): LayerNorm, Swish, GLU, relative-position attention
scores + softmax, depthwise convolution (batch norm folded), residuals.
The prefix (subsampling convolutions + masks) runs in ONNX Runtime.
"""

import os

# All big matmuls run on the NPU; the CPU's small per-head attention matmuls are spread
# over our own thread pool, so BLAS itself stays single-threaded (OpenBLAS's default of 16
# SMT threads was ~20x slower than 8 on this laptop anyway).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "cache" / "parakeet" / "weights"
LAYERS = 24
D = 1024
HEADS, DK = 8, 128
EPS = 1e-5
BLOCK = 1024


_POOL = None
THREADS = 8


class _Pool:
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.ex = ThreadPoolExecutor(THREADS)

    def map_all(self, fn, items):
        list(self.ex.map(fn, items))


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = _Pool()
    return _POOL


def par_rows(fn, n, parts=THREADS):
    """Run fn(slice) over `parts` row ranges in threads (NumPy releases the GIL)."""
    step = -(-n // parts)
    _pool().map_all(fn, [slice(i, min(i + step, n)) for i in range(0, n, step)])


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    return xc / np.sqrt(var + EPS) * w + b


def sigmoid(x):
    out = np.negative(x)
    with np.errstate(over="ignore"):  # exp(-x) -> inf for very negative x gives sigmoid 0, as intended
        np.exp(out, out=out)
    out += 1.0
    return np.reciprocal(out, out=out)


def swish(x):
    """In place on x, in threads."""
    def f(rs):
        x[rs] *= sigmoid(x[rs])
    par_rows(f, x.shape[0])
    return x


def load_layer(i):
    return dict(np.load(WEIGHTS / f"layer_{i:02d}.npz"))


_POS = {}


def pos_for(frames):
    """Projected relative-position embeddings [8, 128, 2T-1] per layer for T encoder frames.
    RelPositionalEncoding takes a centred window of one long table, so shorter lengths
    are centred slices of the longest export."""
    files = sorted(WEIGHTS.glob("pos_T*.npz"), key=lambda p: int(p.stem[5:]))
    for f in files:
        if f not in _POS:
            data = np.load(f)
            _POS[f] = {k: data[k] for k in data.files}
        data = _POS[f]
        width = data["p0"].shape[-1]
        if width >= 2 * frames - 1:
            c = width // 2
            return [data[f"p{i}"][0, :, :, c - frames + 1:c + frames] for i in range(LAYERS)]
    raise ValueError(f"no position table covers {frames} frames")


# -- matmul backends ---------------------------------------------------------
class CpuMatmul:
    """FP32 NumPy reference (optionally emulating BF16 rounding of the inputs)."""

    def __init__(self, layers, bf16=False):
        self.layers = layers
        self.bf16 = bf16

    def __call__(self, layer, name, x):
        w = self.layers[layer][name]
        if self.bf16:
            from ml_dtypes import bfloat16
            x = x.astype(bfloat16).astype(np.float32)
            w = w.astype(bfloat16).astype(np.float32)
        return x @ w

    def multi(self, layer, names, x):
        return [self(layer, n, x) for n in names]


# (weight name, number of row blocks K/1024, number of column blocks N/1024)
LINEARS = [("feed_forward1.w1", 1, 4), ("feed_forward1.w2", 4, 1), ("att.wq", 1, 1), ("att.wk", 1, 1),
           ("att.wv", 1, 1), ("att.wout", 1, 1), ("conv.pw1", 1, 2), ("conv.pw2", 1, 1),
           ("feed_forward2.w1", 1, 4), ("feed_forward2.w2", 4, 1)]
BLOCKS_PER_LAYER = sum(r * c for _, r, c in LINEARS)  # 23


class NpuWeights:
    """All 24 layers' linear weights as BF16 [out, in] 1024x1024 blocks in XRT buffers
    (one buffer per layer, sub-buffer views per block), packed once and shared by every
    NpuMatmul program. Only the small CPU-side parameters stay in `small`."""

    def __init__(self, n_layers=LAYERS):
        import npu_direct as nd
        from ml_dtypes import bfloat16
        t = time.perf_counter()
        self.parents, self.views, self.small = [], [], []
        big = {n for n, _, _ in LINEARS}
        for li in range(n_layers):
            w = load_layer(li)
            parent = nd.Buffer(BLOCKS_PER_LAYER * BLOCK * BLOCK, bfloat16)
            dst = parent.array.reshape(BLOCKS_PER_LAYER, BLOCK, BLOCK)
            views, i = {}, 0
            for name, kb, nb in LINEARS:
                full = w[name]
                for r in range(kb):
                    for c in range(nb):
                        dst[i] = full[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK].T
                        views[(name, r, c)] = parent.view(i * BLOCK * BLOCK, BLOCK * BLOCK)
                        i += 1
            parent.to_device()
            self.parents.append(parent)
            self.views.append(views)
            self.small.append({k: v for k, v in w.items() if k not in big})
        self.pack_s = time.perf_counter() - t


class NpuMatmul:
    """Y[t, 1024] = X[t, 1024] @ W_block[1024, 1024] on the NPU for t <= rows frames:
    BF16 in, FP32 out, one compiled program (one hardware context) for every linear layer.

    Orientation: IRON's whole-array GEMM streams its A operand once and re-streams B
    for every block of output rows, so the weights go in as A (W^T, [out, in]) and
    the small activation matrix as B (column-major, i.e. X row-major [frames, in]);
    C is written column-major, which is Y row-major [frames, out]. Every weight
    byte crosses DDR once per call."""

    def __init__(self, weights, rows=128):
        import aie.iron as iron
        from aie.iron import CompileTime, In, Out
        from ml_dtypes import bfloat16

        import npu_direct as nd
        wa_dir = ROOT / "cache/iron/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
        sys.path.insert(0, str(wa_dir))
        import whole_array as wa

        assert rows % 128 == 0
        self.nd, self.bf16, self.rows = nd, bfloat16, rows
        self.views, self.small = weights.views, weights.small
        tag = f"p011gemm_{BLOCK}_{BLOCK}_{rows}_64_64_32_bf16_f32_b1c1"

        @iron.jit(tag=tag)
        def gemm(A: In, B: In, C: Out, *, tag: CompileTime[str]):
            return wa._build_design(iron.get_current_device(), BLOCK, BLOCK, rows, 64, 64, 32, 4,
                                    "bf16", "f32", 1, 1, False, False, False)

        t = time.perf_counter()
        self.prog = nd.Program.from_design(gemm)
        self.compile_s = time.perf_counter() - t
        # Activations staged block-major: block r = columns r*1024..(r+1)*1024, BF16,
        # converted and synced once however many weight blocks use them.
        self.X = nd.Buffer(4 * rows * BLOCK, bfloat16)
        self.X_views = [self.X.view(r * rows * BLOCK, rows * BLOCK) for r in range(4)]
        self.X.array[:] = 0
        self.X.to_device()
        self.Y = nd.Buffer(rows * BLOCK, np.float32)
        self.reset()

    def reset(self):
        self.calls = 0
        self.npu_s = self.stage_s = self.read_s = 0.0

    def stage(self, x):
        """x [t, K] FP32, K a multiple of 1024 -> BF16 blocks (rows >= t stay zero)."""
        s = time.perf_counter()
        t, k = x.shape
        a = self.X.array.reshape(4, self.rows, BLOCK)

        def cast(rs):
            for r in range(k // BLOCK):
                a[r, rs] = x[rs, r * BLOCK:(r + 1) * BLOCK]
        par_rows(cast, t)
        if t < self.rows:
            a[:, t:] = 0
        self.X.to_device(0, k // BLOCK * self.rows * BLOCK)
        self.stage_s += time.perf_counter() - s

    def gemm(self, r, view, t):
        s = time.perf_counter()
        self.prog(view, self.X_views[r], self.Y)
        s2 = time.perf_counter()
        self.npu_s += s2 - s
        self.calls += 1
        out = self.Y.from_device(0, t * BLOCK).reshape(t, BLOCK).copy()
        self.read_s += time.perf_counter() - s2
        return out

    def multi(self, layer, names, x):
        """Several weights applied to the same input (e.g. q, k, v): one staging."""
        self.stage(x)
        t = x.shape[0]
        views = self.views[layer]
        outs = []
        for name in names:
            kb, nb = next((k, n) for w, k, n in LINEARS if w == name)
            cols = []
            for c in range(nb):
                acc = self.gemm(0, views[(name, 0, c)], t)
                for r in range(1, kb):
                    acc += self.gemm(r, views[(name, r, c)], t)
                cols.append(acc)
            outs.append(cols[0] if nb == 1 else np.concatenate(cols, axis=1))
        return outs

    def __call__(self, layer, name, x):
        return self.multi(layer, [name], x)[0]


# -- one Conformer layer -----------------------------------------------------
def rel_shift(x):
    """NeMo RelPositionMultiHeadAttention.rel_shift for x [h, t, 2t-1]."""
    h, t, n = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (1, 0)))
    return x.reshape(h, n + 1, t)[:, 1:].reshape(h, t, n)


def conformer_layer(x, w, p, masked, valid, mm, layer):
    """x [T, 1024] FP32 (T = encoder frames incl. padding); masked [T, T] bool attention
    mask from the prefix (True = padded pair); valid = number of real frames."""
    t = x.shape[0]
    res = x
    h = swish(mm(layer, "feed_forward1.w1", layer_norm(res, w["norm_feed_forward1.w"], w["norm_feed_forward1.b"])))
    res = res + 0.5 * mm(layer, "feed_forward1.w2", h)

    a = layer_norm(res, w["norm_self_att.w"], w["norm_self_att.b"])
    q, k, v = mm.multi(layer, ["att.wq", "att.wk", "att.wv"], a)
    q = q.reshape(t, HEADS, DK)
    k = k.reshape(t, HEADS, DK).transpose(1, 2, 0)  # [h, dk, t]
    v = v.reshape(t, HEADS, DK).transpose(1, 0, 2)  # [h, t, dk]
    qu = (q + w["att.bias_u"]).transpose(1, 0, 2)
    qv = (q + w["att.bias_v"]).transpose(1, 0, 2)
    att = np.empty((t, HEADS, DK), np.float32)
    keep = ~masked
    scale = np.float32(1.0 / np.sqrt(DK))

    def head(hd):  # one attention head per thread
        bd = rel_shift((qv[hd] @ p[hd])[None])[0, :, :t]
        sc = (qu[hd] @ k[hd] + bd) * scale
        sc = np.where(masked, np.float32(-10000.0), sc)
        sc -= sc.max(-1, keepdims=True)
        pr = np.exp(sc)
        pr /= pr.sum(-1, keepdims=True)
        pr *= keep
        att[:, hd] = pr @ v[hd]
    _pool().map_all(head, range(HEADS))
    att = att.reshape(t, D)
    res = res + mm(layer, "att.wout", att)

    c = mm(layer, "conv.pw1", layer_norm(res, w["norm_conv.w"], w["norm_conv.b"]))  # [t, 2048]
    cp = np.zeros((t + 8, D), np.float32)  # GLU output, zero-padded by 4 frames each side

    def glu(rs):
        cp[4:t + 4][rs] = c[rs, :D] * sigmoid(c[rs, D:])
    par_rows(glu, t)
    cp[4 + valid:] = 0.0
    dw = w["conv.dw"]  # [1024, 9]
    c = np.empty((t, D), np.float32)

    def depthwise(cs):  # threads over channel ranges
        acc = cp[0:t, cs] * dw[cs, 0]
        for j in range(1, 9):
            acc += cp[j:j + t, cs] * dw[cs, j]
        c[:, cs] = acc + w["conv.dw_b"][cs]
    _pool().map_all(depthwise, [slice(i, i + 128) for i in range(0, D, 128)])
    res = res + mm(layer, "conv.pw2", swish(c))

    h = swish(mm(layer, "feed_forward2.w1", layer_norm(res, w["norm_feed_forward2.w"], w["norm_feed_forward2.b"])))
    res = res + 0.5 * mm(layer, "feed_forward2.w2", h)
    return layer_norm(res, w["norm_out.w"], w["norm_out.b"])


class HybridEncoder:
    """Drop-in for the encoder ONNX session: run(["outputs", "encoded_lengths"], feed).
    Prefix (subsampling) in ONNX Runtime on the CPU with the exact input length; the 24
    Conformer layers with their linears on the NPU. The GEMM program is picked by
    frame count: 128, 256 or 384 rows (10.2, 20.4, 30.7 s of audio)."""

    def __init__(self, prefix_path, weights, buckets=(128, 256, 384)):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = THREADS
        self.prefix = ort.InferenceSession(str(prefix_path), so, providers=["CPUExecutionProvider"])
        self.weights = weights
        self.mms = {rows: NpuMatmul(weights, rows) for rows in buckets}
        self.timing = {"prefix": 0.0, "layers": 0.0}

    def get_providers(self):
        return ["NPU(IRON GEMM)+CPU"]

    def run(self, names, feed):
        t0 = time.perf_counter()
        hidden, att_mask, pad_mask = self.prefix.run(["hidden", "att_mask", "pad_mask"], feed)
        t1 = time.perf_counter()
        frames = hidden.shape[1]
        valid = int((~pad_mask).sum())
        mm = self.mms[min(r for r in self.mms if r >= frames)]
        self.last_mm = mm
        pos = pos_for(frames)
        x = hidden[0]
        for i, w in enumerate(self.weights.small):
            x = conformer_layer(x, w, pos[i], att_mask[0, 0], valid, mm, i)
        t2 = time.perf_counter()
        self.timing["prefix"] += t1 - t0
        self.timing["layers"] += t2 - t1
        return [x.T[None].astype(np.float32), np.array([valid], np.int64)]
