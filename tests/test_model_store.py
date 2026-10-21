"""Model download against a local HTTP server: resume, restart, failures that are
not retried automatically, and an interrupted in-place update (no network, no Windows)."""

import errno
import hashlib
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from apps.npu_dictate import model_store

MB = 1 << 20
WEIGHTS = bytes(range(256)) * (3 * MB // 256) + b"tail"  # several 1 MB blocks
SMALL = b'{"vocab": [1, 2, 3]}'


def entry(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def make_manifest(files):
    return {"version": 1, "files": {n: entry(d) for n, d in files.items()}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        s = self.server
        name = self.path.lstrip("/")
        data = s.files[name]
        rng = None if s.ignore_range else self.headers.get("Range")
        start = int(rng.removeprefix("bytes=").rstrip("-")) if rng else 0
        if start >= len(data) and rng:
            status, body = 416, b""
        else:
            status, body = (206 if rng else 200), data[start:]
        s.log.append((name, self.headers.get("Range"), status))
        self.send_response(status)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if s.cut_at is not None:  # the connection drops after cut_at bytes (once)
            body, s.cut_at = body[:s.cut_at], None
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.files = {"weights.bin": WEIGHTS, "small.json": SMALL}
    srv.ignore_range, srv.cut_at, srv.log = False, None, []
    threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True).start()
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    monkeypatch.setattr(model_store, "RELEASE", f"http://127.0.0.1:{srv.server_address[1]}/")
    monkeypatch.setattr(model_store, "manifest", lambda: make_manifest(srv.files))
    yield srv
    srv.shutdown()
    srv.server_close()
    urllib.request.install_opener(None)


def fetch(bundle, cancelled=lambda: False):
    model_store.download(lambda *a: None, cancelled, bundle)


def complete(bundle, files):
    assert model_store.missing(bundle) == []
    for n, d in files.items():
        assert (bundle / n).read_bytes() == d
        assert not (bundle / (n + ".part")).exists()


def test_fresh_download(server, tmp_path):
    fetch(tmp_path)
    complete(tmp_path, server.files)
    assert all(rng is None and status == 200 for _, rng, status in server.log)


def test_resume_with_206(server, tmp_path):
    cut = 1 * MB + 12345
    server.cut_at = cut
    with pytest.raises(OSError) as e:  # a network error: retried automatically
        fetch(tmp_path)
    assert not isinstance(e.value, model_store.NoAutoRetry)
    assert (tmp_path / "weights.bin.part").stat().st_size == cut
    server.log.clear()
    fetch(tmp_path)
    complete(tmp_path, server.files)
    assert ("weights.bin", f"bytes={cut}-", 206) in server.log


def test_resume_after_cancel(server, tmp_path):
    polls = []
    with pytest.raises(InterruptedError):
        fetch(tmp_path, lambda: polls.append(1) or len(polls) > 2)  # after two 1 MB blocks
    assert (tmp_path / "weights.bin.part").stat().st_size == 2 * MB
    server.log.clear()
    fetch(tmp_path)
    complete(tmp_path, server.files)
    assert ("weights.bin", f"bytes={2 * MB}-", 206) in server.log


def test_restart_on_200(server, tmp_path):
    (tmp_path / "weights.bin.part").write_bytes(b"x" * (MB + 7))  # not even the right bytes
    server.ignore_range = True
    fetch(tmp_path)
    complete(tmp_path, server.files)
    assert ("weights.bin", f"bytes={MB + 7}-", 200) in server.log


@pytest.mark.parametrize("served", [WEIGHTS[:-4] + b"TAIL", WEIGHTS + b"more"], ids=["sha256", "longer"])
def test_mismatch_not_retried(server, tmp_path, monkeypatch, served):
    m = make_manifest(server.files)
    server.files = {**server.files, "weights.bin": served}
    monkeypatch.setattr(model_store, "manifest", lambda: m)
    with pytest.raises(model_store.NoAutoRetry, match="weights.bin"):
        fetch(tmp_path)
    assert not (tmp_path / "weights.bin.part").exists()
    assert not (tmp_path / "weights.bin").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_shorter_file_not_retried(server, tmp_path, monkeypatch):
    """A file shorter than the manifest says looks like a dropped connection first
    (kept and resumed); the resume then shows it ends early."""
    m = make_manifest(server.files)
    server.files = {**server.files, "weights.bin": WEIGHTS[:2 * MB]}
    monkeypatch.setattr(model_store, "manifest", lambda: m)
    with pytest.raises(ConnectionError):
        fetch(tmp_path)
    with pytest.raises(model_store.NoAutoRetry, match="weights.bin"):
        fetch(tmp_path)
    assert not (tmp_path / "weights.bin.part").exists()


@pytest.mark.parametrize("winerror", [False, True])
def test_disk_full_not_retried(server, tmp_path, monkeypatch, winerror):
    real_open, full = open, [True]

    class Full:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

        def write(self, block):
            if full[0] and self.f.tell() >= MB:
                e = OSError(errno.EIO if winerror else errno.ENOSPC, "No space left on device")
                if winerror:
                    e.winerror = 112
                raise e
            return self.f.write(block)

    def fake_open(path, mode="r", *args, **kw):
        f = real_open(path, mode, *args, **kw)
        return Full(f) if "b" in mode and mode != "rb" else f

    monkeypatch.setattr(model_store, "open", fake_open, raising=False)
    with pytest.raises(model_store.NoAutoRetry, match=r"disk is full \(3 MB more free space"):
        fetch(tmp_path)
    assert (tmp_path / "weights.bin.part").stat().st_size == MB  # kept: resumed once there is room
    full[0] = False
    server.log.clear()
    fetch(tmp_path)
    complete(tmp_path, server.files)
    assert ("weights.bin", f"bytes={MB}-", 206) in server.log


def test_interrupted_update_refetches(server, tmp_path, monkeypatch):
    """A new manifest gives weights.bin a new SHA-256 with the same size. An update
    that stops early must not leave the old weights in a bundle marked complete."""
    monkeypatch.setattr(model_store, "HASH_BELOW", MB)  # weights.bin is not hashed, as in the app
    old = bytes(reversed(WEIGHTS))
    old_files = {"weights.bin": old, "small.json": SMALL}
    for n, d in old_files.items():
        (tmp_path / n).write_bytes(d)
    (tmp_path / "manifest.json").write_text(json.dumps(make_manifest(old_files), indent=1), encoding="utf-8")
    assert model_store.missing(tmp_path) == ["weights.bin", "manifest.json"]

    polls = []
    with pytest.raises(InterruptedError):
        fetch(tmp_path, lambda: polls.append(1) or len(polls) > 1)
    assert "weights.bin" in model_store.missing(tmp_path)

    server.log.clear()
    fetch(tmp_path)
    assert [n for n, _, _ in server.log] == ["weights.bin"]
    complete(tmp_path, server.files)
