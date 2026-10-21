"""Where the model bundle lives, and the one-time download of it.

The packaged app carries manifest.json (sizes and SHA-256 of every bundle
file) and downloads the files themselves from the GitHub release `model-v1`
into %LOCALAPPDATA%\\NPU Dictate\\model-v1. In the repository the bundle made
by export_bundle.py (cache\\npu_dictate_bundle) is used directly.
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = getattr(sys, "frozen", False)
RESOURCES = Path(getattr(sys, "_MEIPASS", HERE))
REPO_BUNDLE = HERE.parents[1] / "cache" / "npu_dictate_bundle"
RELEASE = "https://github.com/AbhiramJayasankar/ryzenai-npu-lab/releases/download/model-v1/"
LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NPU Dictate" / "model-v1"


def manifest():
    """The manifest this app was built with."""
    p = RESOURCES / "manifest.json" if FROZEN else REPO_BUNDLE / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def bundle_dir():
    if os.environ.get("NPU_DICTATE_BUNDLE"):
        return Path(os.environ["NPU_DICTATE_BUNDLE"])
    return LOCAL if FROZEN else REPO_BUNDLE


HASH_BELOW = 64 << 20  # files smaller than this are hashed at every check (all but the weights, ~0.1 s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def missing(bundle=None):
    """Files to (re)download: absent, the wrong size, or with the wrong SHA-256
    (all but the weights are hashed), plus every file whose entry differs from the
    bundle's own manifest (a new model version). "manifest.json" is listed too when
    the bundle's copy is not the app's."""
    bundle = Path(bundle or bundle_dir())
    m = manifest()
    try:
        local = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        local = {}
    out = []
    for n, f in m["files"].items():
        p = bundle / n
        if (not p.is_file() or p.stat().st_size != f["bytes"]
                or local.get("files", {}).get(n) not in (None, f)
                or (f["bytes"] < HASH_BELOW and sha256(p) != f["sha256"])):
            out.append(n)
    return out if local == m else out + ["manifest.json"]


def download(progress, cancelled, bundle=None):
    """Fetch the missing files. progress(done_bytes, total_bytes, name) is called
    from this thread; cancelled() is polled. Raises on any failure; a file is only
    renamed into place after its SHA-256 matched."""
    bundle = Path(bundle or bundle_dir())
    bundle.mkdir(parents=True, exist_ok=True)
    m = manifest()
    names = [n for n in missing(bundle) if n != "manifest.json"]
    if (bundle / "manifest.json").exists():  # no longer complete until every file is right again
        os.replace(bundle / "manifest.json", bundle / "manifest.json.old")
    total = sum(m["files"][n]["bytes"] for n in names)
    done = 0
    for name in names:
        info = m["files"][name]
        part = bundle / (name + ".part")
        h = hashlib.sha256()
        req = urllib.request.Request(RELEASE + name, headers={"User-Agent": "npu-dictate"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            while True:
                if cancelled():
                    raise InterruptedError("download cancelled")
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                h.update(block)
                done += len(block)
                progress(done, total, name)
        if part.stat().st_size != info["bytes"] or h.hexdigest() != info["sha256"]:
            part.unlink()
            raise ValueError(f"{name}: downloaded file does not match the manifest")
        os.replace(part, bundle / name)
    tmp = bundle / "manifest.json.tmp"  # written last, atomically: it marks the bundle complete
    tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    os.replace(tmp, bundle / "manifest.json")
