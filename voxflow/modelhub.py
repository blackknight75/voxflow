"""Model cache helpers: is a model already downloaded, and how far along is a download?

Progress is read from the Hugging Face cache folder on disk, so it works no matter which library performs the
download. The default models are pinned to a commit and checked by size and hash (``PINNED``); only fetching a
missing file and checking an unpinned model's files reach the Hub.
"""
from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path

log = logging.getLogger("voxflow.modelhub")

# Never send usage telemetry to the Hub (must be set before huggingface_hub is imported).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# The CUDA kernels of llama.cpp are JIT-compiled for newer GPUs (about 25 s the first time, 1.5 s from the driver's
# cache afterwards). The driver cache is shared with every CUDA program and capped at 256 MiB by default, so other
# programs can evict VoxFlow's kernels and the next start costs 25 s again. Keep VoxFlow's own, large cache.
# Must be set before CUDA is initialised; an explicit user / test setting always wins. Local AppData, not Roaming: a
# 1 GiB cache of machine-specific kernels must not be copied around by a roaming profile at every sign-in (QA R16).
_local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home() / ".config")
os.environ.setdefault("CUDA_CACHE_PATH", str(Path(_local) / "VoxFlow" / "cuda-cache"))
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


def move_roaming_cuda_cache() -> str:
    r"""Versions up to 0.2.1 kept the CUDA cache in Roaming AppData (``%APPDATA%\VoxFlow\cuda-cache``): move it to
    the one in use (a rename on the same drive: the compiled kernels are kept, no 25 s recompile), or delete it
    when that is not possible. Call it before the models load. Returns "moved", "removed" or ""."""
    base = os.environ.get("APPDATA")
    if not base:
        return ""
    old, new = Path(base) / "VoxFlow" / "cuda-cache", Path(os.environ.get("CUDA_CACHE_PATH", ""))
    if not old.is_dir() or new == old:
        return ""
    if not new.exists():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, new)
            log.info("moved the GPU kernel cache from Roaming to Local AppData")
            return "moved"
        except OSError as exc:
            log.info("could not move the GPU kernel cache (%s); removing it", exc)
    shutil.rmtree(old, ignore_errors=True)
    return "removed"


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
    pinned = pinned_files(repo_id, filename)
    if pinned:                                  # the pinned commit, every file at its exact size
        return not _missing(repo_id, pinned)
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


# =========================================================================== pinned content (QA R7, R10, R18)
# The default models are fetched at a fixed commit, never "whatever main is today", and every file is known by
# size and hash: SHA-256 for files stored with Git LFS, the git blob SHA-1 ("git:") for small ones. A new model
# version is a reviewed change of these lines. Other models (a custom GGUF, Parakeet v2 without a pin) still come
# from ``main`` and are checked against the Hub's own file list when that can be reached.
PINNED: dict[str, tuple[str, dict[str, tuple[int, str]]]] = {
    "istupakov/parakeet-tdt-0.6b-v3-onnx": ("8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce", {
        "config.json": (97, "git:02a773005666393de591dfd55230b778da1653d8"),
        "decoder_joint-model.onnx": (72_520_893, "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1"),
        "encoder-model.onnx": (41_770_866, "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1"),
        "encoder-model.onnx.data": (2_435_420_160,
                                    "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36"),
        "vocab.txt": (93_939, "git:fc43e1c723e262df60b70e1919614417162d1fe2"),
    }),
    "istupakov/parakeet-tdt-0.6b-v2-onnx": ("0bbb45a3365852604aef28b538a8f066f4ccaa85", {
        "config.json": (97, "git:02a773005666393de591dfd55230b778da1653d8"),
        "decoder_joint-model.onnx": (35_792_059, "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191"),
        "encoder-model.onnx": (41_770_866, "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3"),
        "encoder-model.onnx.data": (2_435_420_160,
                                    "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41"),
        "vocab.txt": (9_384, "git:f0742785f6073e80b911964c455b05f3609bf23b"),
    }),
    "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF": ("ae44f08e1392f39c0e474af10c3ff8355c8b6688", {
        "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf": (
            2_497_280_736, "2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e"),
    }),
}
SPACE_MARGIN = 200_000_000       # free space kept beyond the download itself (temporary files, the cache's own)
_DISK_FULL_WINERRORS = (39, 112)  # ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL


class NotEnoughSpace(OSError):
    """There is not enough free space for a download. ``str()`` is a message for the user."""

    def __init__(self, needed: int, free: int, where: str):
        super().__init__(errno.ENOSPC, f"Needs {_gb(needed)} free on {where}, {_gb(free)} available")
        self.needed, self.free, self.where = needed, free, where

    def __str__(self) -> str:
        return str(self.strerror)


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def pinned_files(repo_id: str, filename: str | None = None) -> dict[str, tuple[int, str]]:
    """{file: (size, hash)} of the pinned files of ``repo_id`` (only ``filename`` when given); {} if not pinned."""
    files = PINNED.get(repo_id, ("", {}))[1]
    return {filename: files[filename]} if filename in files else ({} if filename else dict(files))


def snapshot_dir(repo_id: str) -> Path | None:
    """The cache folder of the pinned commit of ``repo_id`` (None when it is not pinned)."""
    pin = PINNED.get(repo_id)
    return repo_dir(repo_id) / "snapshots" / pin[0] if pin else None


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _missing(repo_id: str, files: dict[str, tuple[int, str]], remove: bool = False) -> dict[str, tuple[int, str]]:
    """The pinned ``files`` that are not in the cache with their exact size. A file of the wrong size is damaged
    (a cut-off copy); ``remove`` deletes it, so that the download replaces it."""
    folder = snapshot_dir(repo_id)
    out = {}
    for name, (size, digest) in files.items():
        have = _size(folder / name)
        if have != size:
            if have >= 0 and remove:
                remove_files(repo_id, [name])
            out[name] = (size, digest)
    return out


def check_space(repo_id: str, needed: int) -> None:
    """Raise NotEnoughSpace when the drive of the model cache has less than ``needed`` bytes (plus a margin)."""
    root = cache_root()
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < needed + SPACE_MARGIN:
        raise NotEnoughSpace(needed + SPACE_MARGIN, free, root.anchor or str(root))


def _fetch(repo_id: str, files: dict[str, tuple[int, str]], pattern: bool) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download
    revision = PINNED[repo_id][0]
    check_space(repo_id, sum(size for size, _ in files.values()))
    if pattern:
        got = Path(snapshot_download(repo_id, revision=revision, allow_patterns=sorted(files)))
    else:
        (name,) = files
        got = Path(hf_hub_download(repo_id, name, revision=revision))
    _verify_fetched(repo_id, files)
    return got


class ChecksumMismatch(OSError):
    """A file that was just downloaded is not the reviewed one. It has been deleted again."""


def _verify_fetched(repo_id: str, files: dict[str, tuple[int, str]]) -> None:
    """Check what was just downloaded against the pinned SHA-256, once, before it is ever loaded. Until now the
    hash was only looked at after a model had failed to load, so a file of the right size and the wrong content
    was loaded unchecked (security review S-13). Reads every byte: about two seconds per gigabyte, once."""
    folder = snapshot_dir(repo_id)
    bad = sorted(name for name, (size, digest) in files.items() if not _matches(folder / name, size, digest))
    if bad:
        remove_files(repo_id, bad)
        raise ChecksumMismatch(f"{', '.join(bad)} of {repo_id} did not arrive as published and was deleted again. "
                               "Try again; if it keeps happening, something between this PC and Hugging Face "
                               "changes the download.")


def model_folder(name: str) -> str | None:
    """The local folder with the pinned files of the speech model ``name``, downloading what is missing; None
    for a model without a pin (onnx-asr then finds or fetches it the way it always did)."""
    repo = ASR_REPOS.get(name, "")
    if repo not in PINNED:
        return None
    missing = _missing(repo, pinned_files(repo), remove=True)
    if missing:
        _fetch(repo, missing, pattern=True)
    return str(snapshot_dir(repo))


def model_file(repo_id: str, filename: str) -> str:
    """Path of ``filename`` from ``repo_id``: the pinned commit when there is one, else whatever the cache has
    (a download from ``main`` only when it has nothing)."""
    files = pinned_files(repo_id, filename)
    if files:
        if _missing(repo_id, files, remove=True):
            return str(_fetch(repo_id, files, pattern=False))
        return str(snapshot_dir(repo_id) / filename)
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    except Exception:  # noqa: BLE001 - not in the cache (any of the hub's "not found" errors)
        return hf_hub_download(repo_id, filename)


def disk_full(exc: BaseException | None) -> bool:
    """True when ``exc`` (or an exception it was raised from) says the disk is full."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and (exc.errno == errno.ENOSPC
                                         or getattr(exc, "winerror", None) in _DISK_FULL_WINERRORS):
            return True
        text = str(exc).lower()
        if "no space left" in text or ("space" in text and "disk" in text):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def remove_partial(repo_id: str) -> int:
    """Delete the unfinished downloads (``blobs/*.incomplete``) of ``repo_id``; returns how many."""
    removed = 0
    for part in (repo_dir(repo_id) / "blobs").glob("*.incomplete"):
        try:
            part.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def remove_files(repo_id: str, names) -> None:
    """Delete ``names`` from every cached snapshot of ``repo_id`` (and the blobs they link to), so that the
    next load downloads them again."""
    for snap in _snapshots(repo_id):
        for name in names:
            path = snap / name
            try:
                target = path.resolve() if path.is_symlink() else None
                path.unlink()
                if target is not None and target.is_file():
                    target.unlink()
            except OSError:
                pass


def _snapshots(repo_id: str) -> list[Path]:
    try:
        return [p for p in (repo_dir(repo_id) / "snapshots").iterdir() if p.is_dir()]
    except OSError:
        return []


def file_hash(path: Path, kind: str) -> str:
    """SHA-256 of the file, or its git blob SHA-1 for ``kind`` "git"."""
    h = hashlib.sha1(b"blob %d\0" % path.stat().st_size) if kind == "git" else hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path, size: int, digest: str) -> bool:
    if _size(path) != size:
        return False
    kind, _, value = digest.rpartition(":")
    try:
        return file_hash(path, kind or "sha256") == value
    except OSError:
        return False


def _hub_files(repo_id: str, revision: str) -> dict[str, tuple[int, str]]:
    """The Hub's own size and hash of each file of ``repo_id`` at ``revision`` ({} when it cannot be reached)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        return {s.rfilename: (int(s.size), s.lfs.sha256 if s.lfs else f"git:{s.blob_id}")
                for s in info.siblings or [] if s.size is not None}
    except Exception as exc:  # noqa: BLE001 - offline, or the repo is gone: the files cannot be checked
        log.info("could not read the file list of %s: %s", repo_id, type(exc).__name__)
        return {}


def damaged_files(repo_id: str, filename: str | None = None) -> list[str]:
    """The cached files of ``repo_id`` (only ``filename`` when given) whose content is not what was published:
    checked against the pin, else against the Hub's list (when online). Reads every byte: call it only after a
    cached model failed to load. [] when everything matches or nothing can be checked."""
    if not repo_id:
        return []
    if repo_id in PINNED:
        folder = snapshot_dir(repo_id)
        return sorted(name for name, (size, digest) in pinned_files(repo_id, filename).items()
                      if not _matches(folder / name, size, digest))
    bad: list[str] = []
    for snap in _snapshots(repo_id):
        local = [p.name for p in snap.iterdir() if p.is_file() and (filename is None or p.name == filename)]
        if not local:
            continue
        published = _hub_files(repo_id, snap.name)
        bad += [name for name in local if name in published and not _matches(snap / name, *published[name])]
    return sorted(set(bad))
