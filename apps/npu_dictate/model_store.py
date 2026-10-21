"""Where the model bundle lives, and the one-time download of it.

The packaged app carries manifest.json (sizes and SHA-256 of every bundle
file) and downloads the files themselves from the GitHub release `model-v1`
into %LOCALAPPDATA%\\NPU Dictate\\model-v1. In the repository the bundle made
by export_bundle.py (cache\\npu_dictate_bundle) is used directly.
"""

import errno
import hashlib
import json
import os
import sys
import urllib.error
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


class NoAutoRetry(Exception):
    """A download failure that trying again by itself cannot fix (a file that does
    not match the manifest, a full disk): the app tries again only on a shortcut press."""


def download(progress, cancelled, bundle=None):
    """Fetch the missing files. progress(done_bytes, total_bytes, name) is called
    from this thread; cancelled() is polled. Raises on any failure, NoAutoRetry when
    retrying cannot help by itself; a file is only renamed into place after its
    SHA-256 matched. An interrupted file is resumed from its .part file; a resumed
    file that does not match is deleted and fetched whole by the next attempt."""
    bundle = Path(bundle or bundle_dir())
    m = manifest()
    total = done = 0
    try:
        bundle.mkdir(parents=True, exist_ok=True)
        names = [n for n in missing(bundle) if n != "manifest.json"]
        for n in names:  # the old copies go first: an interrupted update then fetches them again
            (bundle / n).unlink(missing_ok=True)
        if (bundle / "manifest.json").exists():  # no longer complete until every file is right again
            os.replace(bundle / "manifest.json", bundle / "manifest.json.old")
        total = sum(m["files"][n]["bytes"] for n in names)
        for name in names:
            info = m["files"][name]
            part = bundle / (name + ".part")
            h, left = hashlib.sha256(), 0
            have = part.stat().st_size if part.is_file() else 0  # left by an interrupted attempt
            if have > info["bytes"]:
                part.unlink()
                have = 0
            if have:
                with open(part, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                progress(done + have, total, name)
            if have < info["bytes"] or not part.is_file():
                headers = {"User-Agent": "npu-dictate"}
                if have:
                    headers["Range"] = f"bytes={have}-"
                req = urllib.request.Request(RELEASE + name, headers=headers)
                try:
                    r = urllib.request.urlopen(req, timeout=60)
                except urllib.error.HTTPError as e:
                    e.close()
                    if e.code != 416:
                        raise
                    part.unlink(missing_ok=True)  # the server's file ends before the bytes already here
                    raise NoAutoRetry(f"the downloaded {name} does not match the manifest (size)") from e
                with r:
                    if r.status != 206:  # the whole file, not the rest: start over
                        have, h = 0, hashlib.sha256()
                    done += have
                    with open(part, "ab" if have else "wb") as f:
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
                    left = r.length  # what the response still owed: > 0 after a dropped connection, None if unknown
            else:
                done += have
            size = part.stat().st_size
            if size < info["bytes"] and left != 0:  # the connection ended early: the next attempt resumes
                raise ConnectionError(f"{name}: the download stopped after {size} of {info['bytes']} bytes")
            if size != info["bytes"] or h.hexdigest() != info["sha256"]:
                part.unlink()
                if have:  # the bytes kept from an earlier attempt may be the bad ones
                    raise ValueError(f"the resumed {name} does not match the manifest: the next attempt fetches it whole")
                raise NoAutoRetry(f"the downloaded {name} does not match the manifest (size or SHA-256)")
            os.replace(part, bundle / name)
        tmp = bundle / "manifest.json.tmp"  # written last, atomically: it marks the bundle complete
        tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
        os.replace(tmp, bundle / "manifest.json")
    except OSError as e:
        if e.errno == errno.ENOSPC or getattr(e, "winerror", None) == 112:  # ERROR_DISK_FULL
            mb = -(-max(total - done, 1) // 2**20)
            raise NoAutoRetry(f"the disk is full ({mb} MB more free space is needed "
                              f"on {bundle.drive or bundle.anchor})") from e
        raise
