"""Model cache helpers: is a model already downloaded, and how far along is a download?

Nothing here talks to the network. Progress is read from the Hugging Face cache folder on disk, so it works
no matter which library performs the download.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

# Never send usage telemetry to the Hub (must be set before huggingface_hub is imported).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# The CUDA kernels of llama.cpp are JIT-compiled for newer GPUs (about 25 s the first time, 1.5 s from the driver's
# cache afterwards). The driver cache is shared with every CUDA program and capped at 256 MiB by default, so other
# programs can evict VoxFlow's kernels and the next start costs 25 s again. Keep VoxFlow's own, large cache.
# Must be set before CUDA is initialised; an explicit user / test setting always wins.
_base = os.environ.get("APPDATA") or str(Path.home() / ".config")
os.environ.setdefault("CUDA_CACHE_PATH", str(Path(_base) / "VoxFlow" / "cuda-cache"))
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(1 << 30))
GPU_PREPARE_NOTE = "Preparing the GPU. This can take up to a minute the first time only."

ASR_REPOS = {
    "nemo-parakeet-tdt-0.6b-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
}
# Download sizes of the default models, to turn "bytes so far" into a fraction.
KNOWN_SIZES = {
    "istupakov/parakeet-tdt-0.6b-v3-onnx": 2_549_805_955,
    "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF": 2_497_280_736,
}


def cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def repo_dir(repo_id: str) -> Path:
    return cache_root() / ("models--" + repo_id.replace("/", "--"))


def asr_repo(model: str) -> str:
    """Hub repository behind an ASR model name ('' when the name is a local path or unknown)."""
    if model in ASR_REPOS:
        return ASR_REPOS[model]
    return model if "/" in model and not os.path.exists(model) else ""


def cleanup_repo(model: str) -> tuple[str, str]:
    """('repo', 'file.gguf') for a 'repo:file.gguf' spec; ('', '') for a local path."""
    if os.path.isfile(model) or ":" not in model or (len(model) > 2 and model[1] == ":" and model[2] in "\\/"):
        return "", ""
    repo, filename = model.split(":", 1)
    return repo, filename


def is_cached(repo_id: str, filename: str | None = None) -> bool:
    """True if a complete snapshot (containing ``filename`` when given) is in the local cache."""
    if not repo_id:
        return True
    snaps = repo_dir(repo_id) / "snapshots"
    try:
        for snap in snaps.iterdir():
            if filename:
                if (snap / filename).is_file():
                    return True
            elif any(p.is_file() for p in snap.iterdir()):
                return not any((repo_dir(repo_id) / "blobs").glob("*.incomplete"))
    except OSError:
        pass
    return False


def downloaded_bytes(repo_id: str) -> int:
    """Bytes of ``repo_id`` on disk, including partially downloaded files.

    Files being downloaded live in ``blobs/``. A finished file stays there with a symlink in ``snapshots/``,
    or - on Windows without symlink rights, the default - is moved into ``snapshots/`` as a real file. Both
    places are counted; symlinks are skipped so nothing is counted twice."""
    total = 0
    root = repo_dir(repo_id)
    for folder in (root / "blobs", root / "snapshots"):
        for base, _dirs, files in os.walk(folder):
            for name in files:
                path = os.path.join(base, name)
                try:
                    if not os.path.islink(path):
                        total += os.path.getsize(path)
                except OSError:
                    pass
    return total


def progress_text(done: int, total: int | None) -> str:
    mb = done / 1_000_000
    if total:
        return f"Downloading… {mb:,.0f} MB of {total / 1_000_000:,.0f} MB"
    return f"Downloading… {mb:,.0f} MB"


class DownloadWatcher:
    """Polls the cache folder while a model loads and reports ``callback(done_bytes, total_or_None)``."""

    def __init__(self, repo_id: str, callback, interval_s: float = 0.5):
        self.repo_id = repo_id
        self.total = KNOWN_SIZES.get(repo_id)
        self._callback = callback
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "DownloadWatcher":
        if self.repo_id:
            self._thread = threading.Thread(target=self._run, name="voxflow-download-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def _run(self) -> None:
        last = -1
        while not self._stop.wait(self._interval):
            done = max(last, downloaded_bytes(self.repo_id))     # progress never runs backwards
            if done != last:
                last = done
                try:
                    self._callback(done, self.total)
                except Exception:  # noqa: BLE001 - progress display must never break loading
                    pass
