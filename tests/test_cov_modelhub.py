"""voxflow.modelhub: download progress must not trip over symlinks or files that vanish mid-scan."""
from __future__ import annotations

import os

from voxflow import modelhub as mh


def _repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    root = mh.repo_dir("org/model")
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "abc").mkdir(parents=True)
    return root


def test_symlinked_snapshot_files_are_not_counted_twice(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    (root / "blobs" / "b1").write_bytes(b"x" * 100)
    link = root / "snapshots" / "abc" / "model.onnx"
    link.write_bytes(b"x" * 100)                       # stands in for the symlink to blobs/b1
    real_islink = os.path.islink
    monkeypatch.setattr(mh.os.path, "islink", lambda p: os.path.basename(p) == "model.onnx" or real_islink(p))
    assert mh.downloaded_bytes("org/model") == 100


def test_a_file_that_disappears_during_the_scan_is_skipped(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    (root / "blobs" / "kept").write_bytes(b"x" * 40)
    (root / "blobs" / "gone.incomplete").write_bytes(b"x" * 60)
    real_getsize = os.path.getsize

    def getsize(path):
        if str(path).endswith("gone.incomplete"):
            raise FileNotFoundError(path)              # the downloader renamed it between listing and stat
        return real_getsize(path)

    monkeypatch.setattr(mh.os.path, "getsize", getsize)
    assert mh.downloaded_bytes("org/model") == 40
