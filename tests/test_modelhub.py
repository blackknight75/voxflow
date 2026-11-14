"""voxflow.modelhub: cache inspection and download progress, against a fake Hugging Face cache on disk.
Nothing here may touch the network."""
from __future__ import annotations

import os
import socket
import threading
import time

import pytest

from voxflow import modelhub as mh


@pytest.fixture
def cache(tmp_path, monkeypatch):
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    return root


def make_repo(root, repo_id, files=None, blobs=None, incomplete=None):
    base = root / ("models--" + repo_id.replace("/", "--"))
    snap = base / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (base / "blobs").mkdir()
    for name, size in (files or {}).items():
        (snap / name).write_bytes(b"x" * size)
    for name, size in (blobs or {}).items():
        (base / "blobs" / name).write_bytes(b"x" * size)
    for name, size in (incomplete or {}).items():
        (base / "blobs" / (name + ".incomplete")).write_bytes(b"x" * size)
    return base


def test_telemetry_is_switched_off_on_import():
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1" and os.environ["DO_NOT_TRACK"] == "1"


def test_cache_root_follows_the_hugging_face_environment(tmp_path, monkeypatch):
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mh.Path, "home", classmethod(lambda cls: tmp_path))
    assert mh.cache_root() == tmp_path / ".cache" / "huggingface" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert mh.cache_root() == tmp_path / "hfhome" / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert mh.cache_root() == tmp_path / "legacy"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert mh.cache_root() == tmp_path / "explicit"
    assert mh.repo_dir("org/name") == tmp_path / "explicit" / "models--org--name"


def test_model_names_map_to_repositories(tmp_path):
    assert mh.asr_repo("nemo-parakeet-tdt-0.6b-v3") == "istupakov/parakeet-tdt-0.6b-v3-onnx"
    assert mh.asr_repo("nemo-parakeet-tdt-0.6b-v2") == "istupakov/parakeet-tdt-0.6b-v2-onnx"
    assert mh.asr_repo("someone/custom-onnx") == "someone/custom-onnx"
    assert mh.asr_repo("unknown-model") == ""
    local = tmp_path / "models" / "parakeet"
    local.mkdir(parents=True)
    assert mh.asr_repo(str(local).replace("\\", "/")) == "", "an existing local folder is not a Hub repository"

    assert mh.cleanup_repo("bartowski/Qwen:Qwen-Q4.gguf") == ("bartowski/Qwen", "Qwen-Q4.gguf")
    assert mh.cleanup_repo("C:\\models\\q.gguf") == ("", "") and mh.cleanup_repo("C:/models/q.gguf") == ("", "")
    assert mh.cleanup_repo("no-colon") == ("", "")
    gguf = tmp_path / "local.gguf"
    gguf.write_bytes(b"x")
    assert mh.cleanup_repo(str(gguf)) == ("", "")
    from voxflow.cleanup import DEFAULT_MODEL
    repo, filename = mh.cleanup_repo(DEFAULT_MODEL)
    assert repo in mh.KNOWN_SIZES and filename.endswith(".gguf")
    assert mh.ASR_REPOS["nemo-parakeet-tdt-0.6b-v3"] in mh.KNOWN_SIZES


def test_is_cached(cache):
    assert mh.is_cached("") is True, "a local model needs no download"
    assert mh.is_cached("org/missing") is False
    make_repo(cache, "org/asr", files={"model.onnx": 10}, blobs={"aaa": 10})
    assert mh.is_cached("org/asr") is True
    assert mh.is_cached("org/asr", "model.onnx") is True and mh.is_cached("org/asr", "other.onnx") is False
    make_repo(cache, "org/partial", files={"config.json": 5}, blobs={"aaa": 5}, incomplete={"bbb": 100})
    assert mh.is_cached("org/partial") is False, "a download in progress is not a usable model"
    assert mh.is_cached("org/partial", "config.json") is True
    (cache / "models--org--empty" / "snapshots" / "x").mkdir(parents=True)
    assert mh.is_cached("org/empty") is False


def test_downloaded_bytes_counts_complete_and_partial_blobs(cache):
    assert mh.downloaded_bytes("org/missing") == 0
    make_repo(cache, "org/asr", blobs={"aaa": 1000, "bbb": 234}, incomplete={"ccc": 500})
    assert mh.downloaded_bytes("org/asr") == 1734


@pytest.mark.parametrize("done,total,text", [
    (312_000_000, 640_000_000, "Downloading… 312 MB of 640 MB"),
    (0, 2_549_805_955, "Downloading… 0 MB of 2,550 MB"),
    (5_400_000, None, "Downloading… 5 MB"),
    (1_234_000_000, 0, "Downloading… 1,234 MB"),
])
def test_progress_text(done, total, text):
    assert mh.progress_text(done, total) == text


def test_download_watcher_reports_growth_and_stops(cache):
    base = make_repo(cache, "istupakov/parakeet-tdt-0.6b-v3-onnx", blobs={"aaa": 100})
    seen = []
    got = threading.Event()

    def callback(done, total):
        seen.append((done, total))
        got.set()

    with mh.DownloadWatcher("istupakov/parakeet-tdt-0.6b-v3-onnx", callback, interval_s=0.01) as watcher:
        assert watcher.total == 2_549_805_955
        assert got.wait(2)
        got.clear()
        (base / "blobs" / "bbb.incomplete").write_bytes(b"x" * 400)
        assert got.wait(2)
    thread = watcher._thread
    assert not thread.is_alive() and thread.daemon and thread.name == "voxflow-download-watch"
    assert seen[0] == (100, 2_549_805_955) and seen[-1] == (500, 2_549_805_955)
    assert len(seen) == len(set(seen)), "the callback only fires when the byte count changes"
    count = len(seen)
    time.sleep(0.05)
    assert len(seen) == count, "nothing is reported after the watcher exits"


def test_download_watcher_for_unknown_or_local_models(cache):
    calls = []
    with mh.DownloadWatcher("", calls.append, interval_s=0.01) as w:
        time.sleep(0.05)
    assert w._thread is None and calls == [], "a local model has nothing to watch"
    with mh.DownloadWatcher("org/unknown-size", lambda done, total: calls.append(total), interval_s=0.01) as w:
        time.sleep(0.08)
    assert w.total is None and calls == [None], "unknown total: progress without a fraction"


def test_a_failing_progress_callback_never_breaks_model_loading(cache):
    make_repo(cache, "org/asr", blobs={"aaa": 100})
    with mh.DownloadWatcher("org/asr", lambda done, total: 1 / 0, interval_s=0.01) as w:
        time.sleep(0.05)
        assert w._thread.is_alive()


def test_modelhub_never_opens_a_socket(cache, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("modelhub must not use the network")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    make_repo(cache, "org/asr", files={"m.onnx": 1}, blobs={"a": 1})
    mh.is_cached("org/asr")
    mh.downloaded_bytes("org/asr")
    mh.asr_repo("nemo-parakeet-tdt-0.6b-v3")
    with mh.DownloadWatcher("org/asr", lambda d, t: None, interval_s=0.01):
        time.sleep(0.03)
