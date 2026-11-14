"""The default speech and language models for the real-model workflow (.github/workflows/models.yml).

    python .github/scripts/fetch_models.py --key      # cache key: changes when a default model changes
    python .github/scripts/fetch_models.py --status   # what is in the Hugging Face cache (exit 1 = something missing)
    python .github/scripts/fetch_models.py            # download what is missing, then check both are complete

The model names come from the app (voxflow.config, voxflow.cleanup), and the download runs through the app's own
loading code: the speech model is loaded on the CPU exactly as ``Transcriber.load`` does it (onnx-asr fetches the
files it needs), the language model file is resolved by ``Cleaner._resolve_path`` (``hf_hub_download``). So this
fetches precisely what a first run of VoxFlow fetches: anonymously, no token, into ``HF_HUB_CACHE`` / ``HF_HOME``.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def models() -> dict:
    from voxflow import cleanup, modelhub
    from voxflow.config import Config
    cfg = Config()
    cleanup_model = cfg.cleanup_model or cleanup.DEFAULT_MODEL
    repo, filename = modelhub.cleanup_repo(cleanup_model)
    return {"asr_model": cfg.asr_model, "asr_repo": modelhub.asr_repo(cfg.asr_model),
            "cleanup_model": cleanup_model, "cleanup_repo": repo, "cleanup_file": filename}


def cache_key(m: dict) -> str:
    """Readable and collision-safe: the model names, plus a hash of the full identifiers."""
    ident = "|".join((m["asr_repo"], m["cleanup_repo"], m["cleanup_file"]))
    names = "+".join((m["asr_repo"].rsplit("/", 1)[-1], Path(m["cleanup_file"]).stem))
    return re.sub(r"[^A-Za-z0-9._+-]", "_", names)[:120] + "-" + hashlib.sha256(ident.encode()).hexdigest()[:12]


def status(m: dict) -> dict:
    from voxflow import modelhub
    return {"speech": (m["asr_repo"], modelhub.is_cached(m["asr_repo"]), modelhub.downloaded_bytes(m["asr_repo"])),
            "language": (f"{m['cleanup_repo']}:{m['cleanup_file']}",
                         modelhub.is_cached(m["cleanup_repo"], m["cleanup_file"]),
                         modelhub.downloaded_bytes(m["cleanup_repo"]))}


def print_status(m: dict) -> bool:
    from voxflow import modelhub
    print(f"cache folder: {modelhub.cache_root()}")
    complete = True
    for what, (name, cached, size) in status(m).items():
        print(f"  {what:<9} {name}: {'cached' if cached else 'MISSING'}, {size / 1e9:.2f} GB on disk")
        complete &= cached
    return complete


def attempt(label: str, fn, tries: int = 3) -> None:
    for n in range(1, tries + 1):
        t0 = time.perf_counter()
        try:
            fn()
            print(f"{label}: done in {time.perf_counter() - t0:.0f} s", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - network trouble: say what happened and try again
            print(f"{label}: attempt {n} of {tries} failed after {time.perf_counter() - t0:.0f} s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if n == tries:
                raise
            time.sleep(20 * n)


def fetch(m: dict) -> None:
    from voxflow import modelhub
    from voxflow.asr import Transcriber
    from voxflow.cleanup import Cleaner
    if not modelhub.is_cached(m["asr_repo"]):
        attempt("speech model", lambda: Transcriber(m["asr_model"], "cpu").load())
    if not modelhub.is_cached(m["cleanup_repo"], m["cleanup_file"]):
        attempt("language model", lambda: Cleaner(m["cleanup_model"])._resolve_path())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", action="store_true", help="print the cache key and exit")
    ap.add_argument("--status", action="store_true", help="report what is cached and exit (1 = something missing)")
    args = ap.parse_args(argv)
    m = models()
    if args.key:
        print(cache_key(m))
        return 0
    if args.status:
        return 0 if print_status(m) else 1
    try:
        fetch(m)
    except Exception as exc:  # noqa: BLE001
        print(f"DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
        print_status(m)
        return 1
    if not print_status(m):
        print("DOWNLOAD INCOMPLETE")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
