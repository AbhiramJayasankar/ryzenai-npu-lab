"""Export everything the packaged app needs to run the encoder, so it never
imports IRON or compiles anything: the compiled NPU program (one xclbin and
the instruction stream of each submission), the ctrl and position buffers,
the buffer layout, the packed BF16 weights and the CPU ONNX parts.

Run once in the IRON environment (no NPU access; the program comes from
IRON's compile cache, or is compiled if it is missing):
  python apps\\npu_dictate\\export_bundle.py [out_dir]
Default out_dir: cache\\npu_dictate_bundle. Writes manifest.json with the
SHA-256 of every file.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

import npu_direct as nd  # noqa: E402  (pyxrt import only; no device is opened)
import parakeet_pipeline as pp  # noqa: E402
import pk_engine as pk  # noqa: E402
import pk_model as pm  # noqa: E402

BUNDLE_VERSION = 1
NBLK = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "cache" / "npu_dictate_bundle"
    out.mkdir(parents=True, exist_ok=True)

    # Same steps as pk_model.PkEncoder.__init__, without opening the device.
    lay, phases, pos_packed, table = pm.plan(NBLK, 24)
    ctrl_len = pk.assign_headers(phases)
    lens = {"ctrl": ctrl_len, "weights": table["size"], "pos": sum(a.size for a in pos_packed),
            "io": lay.io_len}
    problems = pk.check(phases, lay, lens)
    if problems:
        raise SystemExit("inconsistent NPU program:\n  " + "\n  ".join(problems[:20]))
    per = pm.LAYERS_PER_RUN[NBLK]
    cuts = [0] + [1 + 22 * k for k in range(per, 24, per)] + [len(phases)]
    parts = [phases[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
    paths = [nd.compile_design(pk.design(NBLK, ph, lens, f"enc24_p{i}")) for i, ph in enumerate(parts)]

    # All parts share the core program: like PkEncoder, the app loads the first
    # part's xclbin and runs every part's instruction stream on that context.
    files = {}

    def put(name, src=None, data=None):
        dst = out / name
        if src is not None:
            shutil.copyfile(src, dst)
        else:
            data.tofile(dst)
        files[name] = {"bytes": dst.stat().st_size, "sha256": sha256(dst)}

    put("encoder.xclbin", src=paths[0][0])
    for i, (_, insts) in enumerate(paths):
        put(f"insts_{i}.bin", src=insts)
    put("ctrl.bin", data=np.ascontiguousarray(pk.ctrl_array(phases, lay)).view(np.uint16))
    put("pos.bin", data=np.ascontiguousarray(np.concatenate(pos_packed)).view(np.uint16))
    put("weights.bin", src=pm.PKDIR / "weights.bin")
    put("encoder_prefix.onnx", src=pp.CACHE / "models" / "encoder_dyn_prefix.onnx")
    for name in ("nemo128.onnx", "decoder_joint-model.onnx", "vocab.txt"):
        put(name, src=pp.MODEL_DIR / name)

    layout = {"TPAD": lay.TPAD, "REC": lay.REC, "AUX": lay.AUX, "S": pk.S, "FX": pk.FX, "FY": pk.FY,
              "D": pk.D, "lens": lens, "parts": len(paths)}
    for k, v in lens.items():  # the files must hold exactly the buffer sizes
        if k in ("ctrl", "pos", "weights") and files[f"{k}.bin"]["bytes"] != 2 * v:
            raise SystemExit(f"{k}.bin has {files[f'{k}.bin']['bytes']} bytes, expected {2 * v}")
    manifest = {"version": BUNDLE_VERSION, "device": "npu1 (Phoenix / Hawk Point)", "layout": layout,
                "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(f["bytes"] for f in files.values())
    print(f"{out}: {len(files)} files, {total / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
