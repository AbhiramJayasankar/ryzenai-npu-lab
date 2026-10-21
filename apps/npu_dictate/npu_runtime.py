"""The 24 Parakeet encoder layers on the NPU from an exported bundle
(export_bundle.py): pyxrt, NumPy and ml_dtypes only, no IRON and no compiler.

Same buffers, program and submissions as experiments/pk_model.PkEncoder(nblk=1),
which this replaces in the packaged app.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

AMD_DLLS = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "AMD"


def import_pyxrt():
    """pyxrt loads xrt_coreutil.dll, which the NPU driver installs in System32."""
    if AMD_DLLS.is_dir():
        os.environ["PATH"] = f"{AMD_DLLS};{os.environ.get('PATH', '')}"
    import pyxrt

    return pyxrt


BUNDLE_VERSION = 1
SUPPORTED_NPUS = ("Phoenix", "Hawk")  # NPU1; the program does not fit Strix and newer
LAYOUT_KEYS = ("TPAD", "REC", "AUX", "S", "FX", "FY", "D", "parts")


def check_manifest(m):
    """Problems with the manifest itself: version, layout and the files the runtime reads."""
    try:
        if m.get("version") != BUNDLE_VERSION:
            return [f"manifest version {m.get('version')}, this app reads {BUNDLE_VERSION}"]
        L, files = m["layout"], m["files"]
        if any(not isinstance(L.get(k), int) for k in LAYOUT_KEYS) or L["parts"] < 1 or L["TPAD"] != 128:
            return ["manifest layout is incomplete"]
        need = ["encoder.xclbin", "ctrl.bin", "pos.bin", "weights.bin", "encoder_prefix.onnx", "nemo128.onnx",
                "decoder_joint-model.onnx", "vocab.txt"] + [f"insts_{i}.bin" for i in range(L["parts"])]
        problems = [f"{n}: not in the manifest" for n in need if n not in files]
        for k in ("ctrl", "pos", "weights"):  # a buffer file must fill its buffer exactly
            if f"{k}.bin" in files and files[f"{k}.bin"]["bytes"] != 2 * L["lens"][k]:
                problems.append(f"{k}.bin does not match the buffer size in the layout")
        if L["lens"]["io"] < L["REC"] + (L["TPAD"] + 8) * L["S"]:
            problems.append("io buffer smaller than the layout")
        problems += [f"insts_{i}.bin: not whole instructions" for i in range(L["parts"])
                     if files.get(f"insts_{i}.bin", {}).get("bytes", 0) % 4]
        return problems
    except (AttributeError, KeyError, TypeError) as e:
        return [f"manifest is malformed ({type(e).__name__}: {e})"]


def verify(bundle, quick=True):
    """Problems with the bundle: the manifest, and files missing, of the wrong size or
    with the wrong SHA-256 (quick: the 1.2 GB weights file is only checked for its size)."""
    bundle = Path(bundle)
    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"manifest.json: {e}"]
    problems = check_manifest(manifest)
    if problems:
        return problems
    for name, info in manifest["files"].items():
        p = bundle / name
        if not p.is_file():
            problems.append(f"{name}: missing")
        elif p.stat().st_size != info["bytes"]:
            problems.append(f"{name}: {p.stat().st_size} bytes, expected {info['bytes']}")
        elif not quick or info["bytes"] < 64 << 20:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != info["sha256"]:
                problems.append(f"{name}: checksum mismatch")
    return problems


class UnsupportedNpu(RuntimeError):
    pass


class NpuEncoder:
    """Encoder layers for up to TPAD = 128 frames (10.24 s of audio)."""

    def __init__(self, bundle):
        self.pyxrt = xrt = import_pyxrt()
        bundle = Path(bundle)
        m = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.L = L = m["layout"]
        self.TPAD = L["TPAD"]
        self.dev = xrt.device(0)
        name = self.dev.get_info(xrt.xrt_info_device.name)
        if not any(n in name for n in SUPPORTED_NPUS):  # checked before any buffer or context exists
            raise UnsupportedNpu(f"{name}: this NPU program is for Phoenix / Hawk Point (Ryzen 7040/8040)")
        self._to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._from = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

        lens = L["lens"]
        self.ctrl = self._buffer(lens["ctrl"], bundle / "ctrl.bin")
        self.weights = self._buffer(lens["weights"], bundle / "weights.bin")
        self.pos = self._buffer(lens["pos"], bundle / "pos.bin")
        self.io_bo, self.io = self._buffer(lens["io"])

        self.xclbin = xrt.xclbin(str(bundle / "encoder.xclbin"))
        self.dev.register_xclbin(self.xclbin)
        self.context = xrt.hw_context(self.dev, self.xclbin.get_uuid())
        self.kernel = xrt.kernel(self.context, "MLIR_AIE")
        self.insts = []
        for i in range(L["parts"]):
            insts = np.fromfile(bundle / f"insts_{i}.bin", dtype=np.uint32)
            bo = xrt.bo(self.dev, insts.nbytes, xrt.bo.cacheable, self.kernel.group_id(1))
            bo.write(insts.tobytes(), 0)
            bo.sync(self._to, insts.nbytes, 0)
            self.insts.append((bo, insts.nbytes))
        self.npu_s = 0.0
        self.in_npu = False  # True from the first XRT call of run() until it returned (stays True after an error)

    def _buffer(self, count, src=None):
        """Host-only BF16 buffer of `count` elements, filled from `src` (raw BF16) or zeros."""
        xrt = self.pyxrt
        bo = xrt.bo(self.dev, 2 * count, xrt.bo.host_only, 0)
        arr = np.frombuffer(bo.map(), dtype=bfloat16, count=count)
        if src is None:
            arr[:] = 0
        else:
            mm = np.memmap(src, dtype=np.uint16, mode="r", shape=(count,))
            dst = arr.view(np.uint16)
            step = 64 << 20
            for s in range(0, count, step):
                dst[s:s + step] = mm[s:s + step]
            del mm
        bo.sync(self._to, 2 * count, 0)
        return (bo, arr) if src is None else bo

    def run(self, hidden, valid):
        """hidden [T, 1024] FP32 prefix output (T <= TPAD), valid frames -> [valid, 1024]."""
        import time

        self.in_npu = True  # any failure from here on (an XRT sync included) is an NPU failure
        L, a = self.L, self.io
        S, FX, FY, D = L["S"], L["FX"], L["FY"], L["D"]
        rec = a[L["REC"]:L["REC"] + (self.TPAD + 8) * S].reshape(self.TPAD + 8, S)
        rec[:, FX:FX + D] = 0
        rec[:, FY:FY + D] = 0
        rec[:hidden.shape[0], FX:FX + D] = hidden.astype(bfloat16)
        a[L["AUX"]:L["AUX"] + 2] = np.array([valid], np.int32).view(bfloat16)
        self.io_bo.sync(self._to, a.nbytes, 0)
        t = time.perf_counter()
        done = self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        for bo, nbytes in self.insts:
            state = self.kernel(3, bo, nbytes, self.ctrl, self.weights, self.pos, self.io_bo).wait()
            if state != done:
                raise RuntimeError(f"NPU run ended in state {state}")
        self.io_bo.sync(self._from, a.nbytes, 0)
        self.in_npu = False
        self.npu_s = time.perf_counter() - t
        return rec[:valid, FX:FX + D].astype(np.float32)
