"""NPU-side preparation of two recurrent-layer prompt positions."""

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16


@iron.jit
def concat_hidden_pair(first: In, second: In, pair: Out):
    hidden_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    pair_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    first_fifo = ObjectFifo(hidden_ty, name="pair_first", depth=1)
    second_fifo = ObjectFifo(hidden_ty, name="pair_second", depth=1)
    pair_fifo = ObjectFifo(pair_ty, name="pair_output", depth=1)

    def core_fn(first_in, second_in, out):
        a = first_in.acquire(1)
        b = second_in.acquire(1)
        y = out.acquire(1)
        for i in range_(1024):
            y[i] = a[i]
            y[1024 + i] = b[i]
        first_in.release(1)
        second_in.release(1)
        out.release(1)

    worker = Worker(core_fn, [first_fifo.cons(), second_fifo.cons(), pair_fifo.prod()])

    def sequence(a, b, y, a_prod, b_prod, y_cons):
        a_prod.fill(a)
        b_prod.fill(b)
        y_cons.drain(y, wait=True)

    runtime = Runtime(sequence, [hidden_ty, hidden_ty, pair_ty,
                                 first_fifo.prod(), second_fifo.prod(), pair_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


@iron.jit
def split_hidden_pair(pair: In, first: Out, second: Out):
    hidden_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    pair_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    pair_fifo = ObjectFifo(pair_ty, name="split_pair_input", depth=1)
    first_fifo = ObjectFifo(hidden_ty, name="split_first", depth=1)
    second_fifo = ObjectFifo(hidden_ty, name="split_second", depth=1)

    def core_fn(pair_in, first_out, second_out):
        x = pair_in.acquire(1)
        a = first_out.acquire(1)
        b = second_out.acquire(1)
        for i in range_(1024):
            a[i] = x[i]
            b[i] = x[1024 + i]
        pair_in.release(1)
        first_out.release(1)
        second_out.release(1)

    worker = Worker(core_fn, [pair_fifo.cons(), first_fifo.prod(), second_fifo.prod()])

    def sequence(x, a, b, x_prod, a_cons, b_cons):
        x_prod.fill(x)
        a_cons.drain(a, wait=True)
        b_cons.drain(b, wait=True)

    runtime = Runtime(sequence, [pair_ty, hidden_ty, hidden_ty,
                                 pair_fifo.prod(), first_fifo.cons(), second_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


@iron.jit
def pack_recurrent_pair(pair: In, state: In, packed: Out):
    pair_ty = np.ndarray[(2048,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(6144,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(18432,), np.dtype[bfloat16]]
    pair_fifo = ObjectFifo(pair_ty, name="pack_pair_hidden", depth=1)
    state_fifo = ObjectFifo(state_ty, name="pack_pair_state", depth=1)
    packed_fifo = ObjectFifo(packed_ty, name="pack_pair_output", depth=1)

    def core_fn(pair_in, state_in, out):
        h = pair_in.acquire(1)
        s = state_in.acquire(1)
        y = out.acquire(1)
        for i in range_(1024):
            y[i] = h[i]
            y[6144 + i] = h[1024 + i]
        for i in range_(1024, 6144):
            y[i] = 0
            y[6144 + i] = 0
        for i in range_(6144):
            y[12288 + i] = s[i]
        pair_in.release(1)
        state_in.release(1)
        out.release(1)

    worker = Worker(core_fn, [pair_fifo.cons(), state_fifo.cons(), packed_fifo.prod()])

    def sequence(h, s, y, h_prod, s_prod, y_cons):
        h_prod.fill(h)
        s_prod.fill(s)
        y_cons.drain(y, wait=True)

    runtime = Runtime(sequence, [pair_ty, state_ty, packed_ty,
                                 pair_fifo.prod(), state_fifo.prod(), packed_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()
