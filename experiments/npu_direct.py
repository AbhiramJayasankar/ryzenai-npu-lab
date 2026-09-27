"""Low-overhead XRT submission for IRON-compiled NPU programs.

IRON compiles a design once; afterwards this module keeps the hardware
context, instruction buffer and argument buffers alive and submits runs
directly through pyxrt. Buffers are XRT host-only BOs mapped as NumPy arrays.
"""

from pathlib import Path

import numpy as np
import pyxrt
from ml_dtypes import bfloat16

TO_DEVICE = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEVICE = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

_device = None


def device():
    global _device
    if _device is None:
        _device = pyxrt.device(0)
    return _device


class Buffer:
    """A host-only XRT buffer with a NumPy view."""

    def __init__(self, count, dtype=bfloat16, parent=None, offset_elems=0):
        self.dtype = np.dtype(dtype)
        self.count = int(count)
        nbytes = self.count * self.dtype.itemsize
        if parent is None:
            self.bo = pyxrt.bo(device(), nbytes, pyxrt.bo.host_only, 0)
            self.array = np.frombuffer(self.bo.map(), dtype=self.dtype, count=self.count)
        else:
            byte_offset = offset_elems * parent.dtype.itemsize
            self.bo = pyxrt.bo(parent.bo, nbytes, byte_offset)
            self.array = parent.array[offset_elems:offset_elems + self.count].view(self.dtype)
        self.parent = parent

    def write(self, values, offset=0):
        values = np.asarray(values).reshape(-1).astype(self.dtype, copy=False)
        self.array[offset:offset + values.size] = values
        self.to_device(offset, values.size)

    def to_device(self, offset=0, count=None):
        count = self.count - offset if count is None else count
        item = self.dtype.itemsize
        self.bo.sync(TO_DEVICE, count * item, offset * item)

    def from_device(self, offset=0, count=None):
        count = self.count - offset if count is None else count
        item = self.dtype.itemsize
        self.bo.sync(FROM_DEVICE, count * item, offset * item)
        return self.array[offset:offset + count]

    def view(self, offset, count):
        return Buffer(count, self.dtype, parent=self, offset_elems=offset)


class Program:
    """One compiled xclbin (hardware context) with one or more instruction
    streams. Designs whose cores, FIFOs and routes are identical and differ
    only in their runtime sequence can share the context: switching between
    their instruction streams avoids reloading the array."""

    def __init__(self, xclbin_path, insts_path=None, kernel_name="MLIR_AIE"):
        dev = device()
        self.xclbin = pyxrt.xclbin(str(xclbin_path))
        dev.register_xclbin(self.xclbin)
        self.context = pyxrt.hw_context(dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.context, kernel_name)
        self.default = self.entry(insts_path) if insts_path is not None else None

    def entry(self, insts_path):
        return Entry(self, insts_path)

    @classmethod
    def from_design(cls, design):
        """Compile an @iron.jit design (or reuse IRON's disk cache) and wrap it
        without creating an IRON runtime context."""
        xclbin_path, insts_path = compile_design(design)
        return cls(xclbin_path, insts_path)

    def start(self, *buffers):
        return self.default.start(*buffers)

    def __call__(self, *buffers):
        self.default(*buffers)


class Entry:
    """An instruction stream submitted on a Program's context."""

    def __init__(self, program, insts_path):
        self.program = program
        insts = np.fromfile(insts_path, dtype=np.uint32)
        self.insts_bytes = insts.nbytes
        self.insts_bo = pyxrt.bo(device(), insts.nbytes, pyxrt.bo.cacheable,
                                 program.kernel.group_id(1))
        self.insts_bo.write(insts.tobytes(), 0)
        self.insts_bo.sync(TO_DEVICE, insts.nbytes, 0)

    def start(self, *buffers):
        return self.program.kernel(3, self.insts_bo, self.insts_bytes,
                                   *[b.bo for b in buffers])

    def __call__(self, *buffers):
        run = self.start(*buffers)
        state = run.wait()
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"NPU run ended in state {state}")


def compile_design(design):
    """Return (xclbin, insts) paths for an @iron.jit design, compiling if needed."""
    xclbin_path, insts_path = design.compilable.compile()
    return Path(xclbin_path), Path(insts_path)


def finish(code=0):
    """Exit without Python teardown; pyxrt objects can crash when collected late."""
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
