"""Eight-core LFM2.5-230M programs for the Phoenix NPU.

Each program uses eight compute tiles (two per column, rows 2-3). Core k owns
output slice k of every projection and receives one private shim DMA stream:
activations, then its pre-packed weight slice, then any state. Recurrent
convolution channels and attention heads are partitioned the same way, so
most work stays local to a core. Where a full vector is needed, the stage
outputs go to a DDR scratch buffer (`io`) and the next stage gathers them.
Every layer and the vocabulary head run the same core program: a header
object at the start of each weight stream selects recurrent, attention or
head work. One XRT submission runs a whole token: 14 layers (four stages
each) and the head, sequenced by the runtime DMA program.

`io` layout, in BF16 elements:
  [0:512)       token aux: cos(64) | sin(64) | int32 past length |
                int32 KV block count
  SLOT(s, k)    stage s output of core k, 512 elements each
  SLOT(5, k)    vocabulary candidates, SLOT(6, 0) the selected id
  STATE(i, k)   conv state of recurrent layer i, core k
"""

import hashlib
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

KDIR = Path(__file__).resolve().parent / "kernels" / "x8"
# IRON's disk cache keys on the generator and compile kwargs, not on kernel
# source contents, so every design tag carries a digest of the sources.
KVER = hashlib.sha1(b"".join(p.read_bytes() for p in sorted(KDIR.glob("*.*")))).hexdigest()[:10]
CORES = 8
OBJ = 512
HIDDEN = 1024
FFN = 2560
VOCAB = 65536
VOCAB_ROWS = VOCAB // CORES
MAX_CAPACITY = 4096
MODE_RECURRENT, MODE_ATTENTION, MODE_HEAD = 0, 1, 2
KINDS = ("conv", "conv", "attention", "conv", "attention", "conv",
         "attention", "conv", "attention", "conv", "attention", "conv",
         "attention", "conv")
CONV_LAYERS = KINDS.count("conv")
ATT_LAYERS = KINDS.count("attention")


def SLOT(stage, core):
    return 512 + (stage - 1) * CORES * OBJ + core * OBJ


def STATE(conv_index, core):
    return SLOT(7, 0) + (conv_index * CORES + core) * OBJ


PB = 4  # prompt tokens per batched weight pass (x8p_core.cc)


def PAUX(token):
    """Batched prompt: aux object of token t of the batch."""
    return STATE(CONV_LAYERS, 0) + token * OBJ


def PSLOT(stage, token, core):
    """Batched prompt: stage output of token t, core k."""
    return PAUX(PB) + ((stage - 1) * PB + token) * CORES * OBJ + core * OBJ


def PEMB(token):
    """Batched prompt: the token's embedding row, staged by the host."""
    return PSLOT(5, 0, 0) + token * HIDDEN


IO_LEN = PEMB(PB)


# Per-core weight stream lengths: gamma, header object, projection rows, then
# the shared stage 2-4 weights.
REC_ROWS, ATT_ROWS = 384, 256
REC_S1 = HIDDEN + OBJ + REC_ROWS * HIDDEN
ATT_S1 = HIDDEN + OBJ + ATT_ROWS * HIDDEN
S2_LEN = 128 * HIDDEN
S3_LEN = HIDDEN + 320 * 2 * HIDDEN
S4_LEN = 128 * FFN
REC_CORE = REC_S1 + S2_LEN + S3_LEN + S4_LEN
ATT_CORE = ATT_S1 + S2_LEN + S3_LEN + S4_LEN
HEAD_CORE = HIDDEN + OBJ


def _weight_offsets():
    """Start of each layer's (and the head's) eight core slices in the packed
    weight buffer."""
    offsets, off = [], 0
    for kind in KINDS:
        offsets.append(off)
        off += CORES * (ATT_CORE if kind == "attention" else REC_CORE)
    return offsets, off, off + CORES * HEAD_CORE


LAYER_OFFSETS, HEAD_OFFSET, W_LEN = _weight_offsets()

# KV cache per attention layer: per core, position p at p*128 holds
# [key 64 | value 64]. A 512-element append at p also overwrites the unused
# slots p+1..p+3, so each core region carries one object of slack.
CACHE_STRIDE = MAX_CAPACITY * 128 + OBJ
CACHE_LEN = (CORES + 1) * CACHE_STRIDE
CACHES_LEN = ATT_LAYERS * CACHE_LEN
APPEND_LEN = (ATT_LAYERS - 1) * CACHE_LEN + (CORES - 1) * CACHE_STRIDE + OBJ
APPEND_LEN_P = APPEND_LEN + (PB - 1) * 128

OBJ_T = np.ndarray[(OBJ,), np.dtype[bfloat16]]
MEM_T = np.ndarray[(5 * FFN,), np.dtype[bfloat16]]
ACC_T = np.ndarray[(640,), np.dtype[np.float32]]
CTL_T = np.ndarray[(16,), np.dtype[np.int32]]
SILU_E0, SILU_E1 = 112, 134
TAB_T = np.ndarray[(2 * (SILU_E1 - SILU_E0) * 128,), np.dtype[bfloat16]]


def silu_table():
    """bf16(silu(x)) for BF16 inputs with exponent field in [E0, E1), computed
    in FP32 like PyTorch; positive half then negative half."""
    exps = np.arange(SILU_E0, SILU_E1, dtype=np.uint16)
    mant = np.arange(128, dtype=np.uint16)
    bits = (exps[:, None] << 7 | mant[None, :]).reshape(-1)
    bits = np.concatenate((bits, bits | np.uint16(0x8000)))
    x = bits.view(bfloat16).astype(np.float32)
    with np.errstate(over="ignore"):
        y = x / (np.float32(1) + np.exp(-x, dtype=np.float32))
    return y.astype(bfloat16)


def _kernel():
    """The whole per-core program body (kernels/x8/x8_core.cc)."""
    return ExternalFunction(
        "x8_do", source_file=str(KDIR / "x8_core.cc"),
        arg_types=[OBJ_T, MEM_T, ACC_T, TAB_T, CTL_T, np.int32, np.int32],
        include_dirs=[config.cxx_header_path(), str(KDIR)], compile_flags=["-Oz"])


MEMP_T = np.ndarray[(PB * 1024 + PB * 2560 + 1024 + PB * 640 + 512 + 2 * PB * 512 + 1024,),
                    np.dtype[bfloat16]]
ACCP_T = np.ndarray[(PB * 160 + 32,), np.dtype[np.float32]]


def _prefill_kernel():
    """Batched prompt per-core program body (kernels/x8/x8p_core.cc)."""
    return ExternalFunction(
        "x8p_do", source_file=str(KDIR / "x8p_core.cc"),
        arg_types=[OBJ_T, MEMP_T, ACCP_T, CTL_T, np.int32, np.int32],
        include_dirs=[config.cxx_header_path(), str(KDIR)],
        compile_flags=["-Oz"])


def _place(k):
    return k // 2, 2 + k % 2


def _fifos(prefix):
    ins = [ObjectFifo(OBJ_T, name=f"{prefix}_in{k}", depth=4) for k in range(CORES)]
    outs = [ObjectFifo(OBJ_T, name=f"{prefix}_out{k}", depth=2) for k in range(CORES)]
    return ins, outs


SEGMENTS = 5
OP_IN, OP_OUT, OP_NEXT = 0, 1, 2


def _core(inp, out, mem, acc, tab, ctl, dummy, do):
    """Feed every input object, fill every output object, then advance the
    state machine; five segments per stream (see x8_core.cc)."""
    for _ in range_(SEGMENTS):
        for i in range_(ctl[0]):
            o = inp.acquire(1)
            do(o, mem, acc, tab, ctl, OP_IN, i)
            inp.release(1)
        for j in range_(ctl[1]):
            o = out.acquire(1)
            do(o, mem, acc, tab, ctl, OP_OUT, j)
            out.release(1)
        do(dummy, mem, acc, tab, ctl, OP_NEXT, 0)


def _prefill_core(inp, out, mem, acc, ctl, dummy, do):
    for _ in range_(SEGMENTS):
        for i in range_(ctl[0]):
            o = inp.acquire(1)
            do(o, mem, acc, ctl, OP_IN, i)
            inp.release(1)
        for j in range_(ctl[1]):
            o = out.acquire(1)
            do(o, mem, acc, ctl, OP_OUT, j)
            out.release(1)
        do(dummy, mem, acc, ctl, OP_NEXT, 0)


def _control(k):
    """Initial state machine control words: segment 0 expects six objects."""
    ctl = np.zeros(16, dtype=np.int32)
    ctl[0] = 6
    ctl[7] = k
    ctl[8] = int(k == 0)
    return ctl


def _gather(total, stage, n):
    return TensorAccessPattern((total,), SLOT(stage, 0), [1, 1, CORES, n], [0, 0, OBJ, 1])


def _linear(total, offset, n):
    return TensorAccessPattern((total,), offset, [1, 1, 1, n], [0, 0, 0, 1])


def token_design(capacity, with_head=True):
    """One token as a single submission: all layers, then (optionally) the
    head. Attention layers stream `capacity` cached positions per core; the
    host writes capacity/4 into the token aux."""
    tag = f"x8token_{capacity}_{int(with_head)}_{KVER}"
    table_len = VOCAB * HIDDEN

    @iron.jit(tag=tag)
    def x8_token(weights: In, row: In, io: Out, caches: In, append: Out, table: In, *,
                 tag: CompileTime[str]):
        do = _kernel()
        ins, outs = _fifos("x8")
        workers = [Worker(_core, [
            ins[k].cons(), outs[k].prod(), Buffer(MEM_T, name=f"x8_mem{k}"),
            Buffer(ACC_T, name=f"x8_acc{k}"),
            Buffer(TAB_T, initial_value=silu_table(), name=f"x8_silu{k}"),
            Buffer(CTL_T, initial_value=_control(k), name=f"x8_ctl{k}"),
            Buffer(OBJ_T, name=f"x8_dummy{k}"), do],
            tile=Tile(*_place(k)), dynamic_objfifo_lowering=True, stack_size=2048)
            for k in range(CORES)]

        def layer(index, w, row, io, caches, append, prods, conss):
            attention = KINDS[index] == "attention"
            core_len = ATT_CORE if attention else REC_CORE
            s1_len = ATT_S1 if attention else REC_S1
            base = LAYER_OFFSETS[index]
            conv_index = KINDS[:index].count("conv")
            att_index = KINDS[:index].count("attention")
            g = TaskGroup()
            for k in range(CORES):
                if index == 0:
                    prods[k].fill(row, tap=_linear(HIDDEN, 0, HIDDEN), group=g)
                else:
                    prods[k].fill(io, tap=_gather(IO_LEN, 4, 128), group=g)
                prods[k].fill(io, tap=_linear(IO_LEN, 0, OBJ), group=g)
                prods[k].fill(w, tap=_linear(W_LEN, base + k * core_len, s1_len), group=g)
                if attention:
                    region = att_index * CACHE_LEN + k * CACHE_STRIDE
                    prods[k].fill(caches, tap=_linear(CACHES_LEN, region, capacity * 128),
                                  group=g)
                    conss[k].drain(io, tap=_linear(IO_LEN, SLOT(1, k), OBJ), group=g,
                                   wait=True)
                    conss[k].drain(append, tap=_linear(APPEND_LEN, region, OBJ), group=g,
                                   wait=True)
                else:
                    state = _linear(IO_LEN, STATE(conv_index, k), OBJ)
                    prods[k].fill(io, tap=state, group=g)
                    conss[k].drain(io, tap=_linear(IO_LEN, SLOT(1, k), OBJ), group=g,
                                   wait=True)
                    conss[k].drain(io, tap=state, group=g, wait=True)
            g.finish()
            for stage, gathered, part_off, part_len in (
                (2, (1, 128), s1_len, S2_LEN),
                (3, (2, 128), s1_len + S2_LEN, S3_LEN),
                (4, (3, 320), s1_len + S2_LEN + S3_LEN, S4_LEN),
            ):
                g = TaskGroup()
                for k in range(CORES):
                    prods[k].fill(io, tap=_gather(IO_LEN, *gathered), group=g)
                    prods[k].fill(w, tap=_linear(W_LEN, base + k * core_len + part_off,
                                                 part_len), group=g)
                    conss[k].drain(io, tap=_linear(IO_LEN, SLOT(stage, k), OBJ), group=g,
                                   wait=True)
                g.finish()

        def head(w, io, table, prods, conss):
            g = TaskGroup()
            for k in range(CORES):
                prods[k].fill(io, tap=_gather(IO_LEN, 4, 128), group=g)
                prods[k].fill(io, tap=_linear(IO_LEN, 0, OBJ), group=g)
                prods[k].fill(w, tap=_linear(W_LEN, HEAD_OFFSET + k * HEAD_CORE, HEAD_CORE),
                              group=g)
                prods[k].fill(table, tap=_linear(table_len, k * VOCAB_ROWS * HIDDEN,
                                                 VOCAB_ROWS * HIDDEN), group=g)
                conss[k].drain(io, tap=_linear(IO_LEN, SLOT(5, k), OBJ), group=g, wait=True)
            g.finish()
            g = TaskGroup()
            prods[0].fill(io, tap=TensorAccessPattern((IO_LEN,), SLOT(5, 0), [1, 1, CORES, 64],
                                                      [0, 0, OBJ, 1]), group=g)
            conss[0].drain(io, tap=_linear(IO_LEN, SLOT(6, 0), OBJ), group=g, wait=True)
            g.finish()

        def sequence(w, row, io, caches, append, table, prods, conss):
            for index in range(len(KINDS)):
                layer(index, w, row, io, caches, append, prods, conss)
            if with_head:
                head(w, io, table, prods, conss)

        rt = Runtime(sequence, [
            np.ndarray[(W_LEN,), np.dtype[bfloat16]],
            np.ndarray[(HIDDEN,), np.dtype[bfloat16]],
            np.ndarray[(IO_LEN,), np.dtype[bfloat16]],
            np.ndarray[(CACHES_LEN,), np.dtype[bfloat16]],
            np.ndarray[(APPEND_LEN,), np.dtype[bfloat16]],
            np.ndarray[(table_len,), np.dtype[bfloat16]],
            [ins[k].prod(tile=Tile(_place(k)[0], 0)) for k in range(CORES)],
            [outs[k].cons(tile=Tile(_place(k)[0], 0)) for k in range(CORES)],
        ])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return x8_token


def _tokens(total, offset, n, stride):
    """PB consecutive runs of n elements, `stride` apart."""
    return TensorAccessPattern((total,), offset, [1, 1, PB, n], [0, 0, stride, 1])


def _pgather(stage, n):
    """Gather PB tokens' stage outputs from all cores: token-major vectors."""
    return TensorAccessPattern((IO_LEN,), PSLOT(stage, 0, 0), [1, PB, CORES, n],
                               [0, CORES * OBJ, OBJ, 1])


def prefill_design(capacity, only=None):
    """PB prompt tokens through all layers in one submission, each weight
    object used for all PB tokens. No vocabulary head. Args: weights, io (with
    the PB embedding rows staged at PEMB), caches, append."""
    # `only` (debugging): run just these layer indices.
    tag = f"x8prefill_{capacity}_{PB}_{only}_{KVER}"

    @iron.jit(tag=tag)
    def x8_prefill(weights: In, io: Out, caches: In, append: Out, *, tag: CompileTime[str]):
        do = _prefill_kernel()
        ins, outs = _fifos("x8p")
        workers = [Worker(_prefill_core, [
            ins[k].cons(), outs[k].prod(), Buffer(MEMP_T, name=f"x8p_mem{k}"),
            Buffer(ACCP_T, name=f"x8p_acc{k}"),
            Buffer(CTL_T, initial_value=_prefill_control(k), name=f"x8p_ctl{k}"),
            Buffer(OBJ_T, name=f"x8p_dummy{k}"), do],
            tile=Tile(*_place(k)), dynamic_objfifo_lowering=True, stack_size=2048)
            for k in range(CORES)]

        def layer(index, w, io, caches, append, prods, conss):
            attention = KINDS[index] == "attention"
            core_len = ATT_CORE if attention else REC_CORE
            s1_len = ATT_S1 if attention else REC_S1
            base = LAYER_OFFSETS[index]
            conv_index = KINDS[:index].count("conv")
            att_index = KINDS[:index].count("attention")
            g = TaskGroup()
            for k in range(CORES):
                if index == 0:
                    prods[k].fill(io, tap=_linear(IO_LEN, PEMB(0), PB * HIDDEN), group=g)
                else:
                    prods[k].fill(io, tap=_pgather(4, 128), group=g)
                prods[k].fill(io, tap=_linear(IO_LEN, PAUX(0), PB * OBJ), group=g)
                prods[k].fill(w, tap=_linear(W_LEN, base + k * core_len, s1_len), group=g)
                conss[k].drain(io, tap=_tokens(IO_LEN, PSLOT(1, 0, k), OBJ, CORES * OBJ),
                               group=g, wait=True)
                if attention:
                    region = att_index * CACHE_LEN + k * CACHE_STRIDE
                    prods[k].fill(caches, tap=_linear(CACHES_LEN, region, capacity * 128),
                                  group=g)
                    conss[k].drain(append, tap=_tokens(APPEND_LEN_P, region, OBJ, 128),
                                   group=g, wait=True)
                else:
                    state = _linear(IO_LEN, STATE(conv_index, k), OBJ)
                    prods[k].fill(io, tap=state, group=g)
                    conss[k].drain(io, tap=state, group=g, wait=True)
            g.finish()
            for stage, gathered, part_off, part_len in (
                (2, (1, 128), s1_len, S2_LEN),
                (3, (2, 128), s1_len + S2_LEN, S3_LEN),
                (4, (3, 320), s1_len + S2_LEN + S3_LEN, S4_LEN),
            ):
                g = TaskGroup()
                for k in range(CORES):
                    prods[k].fill(io, tap=_pgather(*gathered), group=g)
                    prods[k].fill(w, tap=_linear(W_LEN, base + k * core_len + part_off,
                                                 part_len), group=g)
                    conss[k].drain(io, tap=_tokens(IO_LEN, PSLOT(stage, 0, k), OBJ,
                                                   CORES * OBJ), group=g, wait=True)
                g.finish()

        def sequence(w, io, caches, append, prods, conss):
            for index in (range(len(KINDS)) if only is None else only):
                layer(index, w, io, caches, append, prods, conss)

        rt = Runtime(sequence, [
            np.ndarray[(W_LEN,), np.dtype[bfloat16]],
            np.ndarray[(IO_LEN,), np.dtype[bfloat16]],
            np.ndarray[(CACHES_LEN,), np.dtype[bfloat16]],
            np.ndarray[(APPEND_LEN_P,), np.dtype[bfloat16]],
            [ins[k].prod(tile=Tile(_place(k)[0], 0)) for k in range(CORES)],
            [outs[k].cons(tile=Tile(_place(k)[0], 0)) for k in range(CORES)],
        ])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return x8_prefill


def _prefill_control(k):
    ctl = np.zeros(16, dtype=np.int32)
    ctl[0] = 3 * PB + 3
    ctl[7] = k
    return ctl


# ---------------------------------------------------------------- packing

def _header(mode, rows, payload):
    obj = np.zeros(OBJ, dtype=bfloat16)
    payload = np.asarray(payload, dtype=bfloat16).reshape(-1)
    obj[:payload.size] = payload
    obj[500:502] = np.array([mode], dtype=np.int32).view(bfloat16)
    obj[504:506] = np.array([rows], dtype=np.int32).view(bfloat16)
    return obj


def _ffn_parts(ck, prefix, k):
    gamma = ck.load(prefix + "ffn_norm.weight").reshape(-1)
    w1 = ck.load(prefix + "feed_forward.w1.weight")[320 * k:320 * (k + 1)]
    w3 = ck.load(prefix + "feed_forward.w3.weight")[320 * k:320 * (k + 1)]
    w2 = ck.load(prefix + "feed_forward.w2.weight")[128 * k:128 * (k + 1)]
    inter = np.stack((w1, w3), axis=1).reshape(-1)
    return [gamma, inter, w2.reshape(-1)]


def _join(parts, expected):
    core = np.concatenate([np.asarray(c, dtype=bfloat16).reshape(-1) for c in parts])
    assert core.size == expected, core.size
    return core


def pack_recurrent(ck, layer):
    p = f"model.layers.{layer}."
    gamma = ck.load(p + "operator_norm.weight").reshape(-1)
    inp = ck.load(p + "conv.in_proj.weight")
    out = ck.load(p + "conv.out_proj.weight")
    conv = ck.load(p + "conv.conv.weight")[:, 0, :]
    parts = []
    for k in range(CORES):
        ch = slice(128 * k, 128 * (k + 1))
        rows = np.concatenate((inp[0:1024][ch], inp[1024:2048][ch], inp[2048:3072][ch]))
        header = _header(MODE_RECURRENT, REC_ROWS, conv[ch].T.reshape(-1))
        parts.append(_join([gamma, header, rows, out[ch], *_ffn_parts(ck, p, k)], REC_CORE))
    return np.concatenate(parts)


def pack_attention(ck, layer):
    p = f"model.layers.{layer}."
    gamma = ck.load(p + "operator_norm.weight").reshape(-1)
    q = ck.load(p + "self_attn.q_proj.weight")
    kk = ck.load(p + "self_attn.k_proj.weight")
    v = ck.load(p + "self_attn.v_proj.weight")
    o = ck.load(p + "self_attn.out_proj.weight")
    qg = ck.load(p + "self_attn.q_layernorm.weight").reshape(-1)
    kg = ck.load(p + "self_attn.k_layernorm.weight").reshape(-1)
    parts = []
    for k in range(CORES):
        rows = np.concatenate((q[128 * k:128 * (k + 1)], kk[64 * k:64 * (k + 1)],
                               v[64 * k:64 * (k + 1)]))
        header = _header(MODE_ATTENTION, ATT_ROWS, np.concatenate((qg, kg)))
        parts.append(_join([gamma, header, rows, o[128 * k:128 * (k + 1)],
                            *_ffn_parts(ck, p, k)], ATT_CORE))
    return np.concatenate(parts)


def pack_head(ck):
    gamma = ck.load("model.embedding_norm.weight").reshape(-1)
    parts = []
    for k in range(CORES):
        header = _header(MODE_HEAD, VOCAB_ROWS, [])
        header[508:510] = np.array([VOCAB_ROWS * k], dtype=np.int32).view(bfloat16)
        parts.append(_join([gamma, header], HEAD_CORE))
    return np.concatenate(parts)


def pack_weights(ck, into):
    """Write every layer and the head into one packed weight array of W_LEN
    BF16 values."""
    for index, kind in enumerate(KINDS):
        packed = pack_attention(ck, index) if kind == "attention" else pack_recurrent(ck, index)
        into[LAYER_OFFSETS[index]:LAYER_OFFSETS[index] + packed.size] = packed
    into[HEAD_OFFSET:W_LEN] = pack_head(ck)


def pack_conv_state(state):
    """(1024, 3) HF conv state -> per-core planar [t*128 + ch] objects."""
    out = np.zeros((CORES, OBJ), dtype=bfloat16)
    s = np.asarray(state, dtype=bfloat16).reshape(1024, 3)
    for k in range(CORES):
        out[k, :384] = s[128 * k:128 * (k + 1)].T.reshape(-1)
    return out.reshape(-1)


def unpack_conv_state(packed):
    p = np.asarray(packed).reshape(CORES, OBJ)[:, :384].reshape(CORES, 3, 128)
    return p.transpose(0, 2, 1).reshape(1024, 3)
