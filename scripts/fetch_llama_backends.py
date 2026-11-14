r"""Fetch the extra llama.cpp builds VoxFlow chooses from at start: Vulkan (any GPU vendor) and plain CPU.

    .venv\Scripts\python.exe scripts\fetch_llama_backends.py            # once after installing requirements.txt
    .venv\Scripts\python.exe scripts\fetch_llama_backends.py --check    # no network: are they in place?

The installed llama-cpp-python wheel is the CUDA build: it loads only on a PC with an NVIDIA driver. This script
puts the DLLs of the Vulkan and CPU builds of the SAME llama-cpp-python release next to it:

    site-packages\llama_cpp\lib          the wheel's own (CUDA 12)
    site-packages\llama_cpp\lib_vulkan   AMD / Intel / NVIDIA through Vulkan
    site-packages\llama_cpp\lib_cpu      no GPU driver needed

``voxflow.cleanup`` picks one folder when the language model is loaded (see LLAMA_BACKENDS there), and
packaging/voxflow.spec bundles all of them into the installer; scripts/build_installer.py runs this first.

Where the files come from: the wheels the llama-cpp-python project publishes on its GitHub releases (the same
files its package indexes https://abetlen.github.io/llama-cpp-python/whl/vulkan and .../whl/cpu point at). Each
is pinned below by version and SHA-256; a download that does not match is refused. Only ``llama_cpp/lib/*.dll``
is taken from a wheel, and nothing is installed with pip. Folders that are already in place for the installed
version are left alone, so a second run needs no network.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RELEASES = "https://github.com/abetlen/llama-cpp-python/releases/download"
# llama-cpp-python version -> back-end name (voxflow.cleanup.LLAMA_BACKENDS) -> (wheel URL, SHA-256 of the wheel).
# When requirements.txt moves to another llama-cpp-python, add its row: the builds must be of the same release.
PINS = {
    "0.3.36": {
        "vulkan": (f"{_RELEASES}/v0.3.36-vulkan/llama_cpp_python-0.3.36-py3-none-win_amd64.whl",
                   "b3d58d70e9fa6216f3d1919bb6141a98eb22eb889e14a44d080dd250e969fab4"),
        "cpu": (f"{_RELEASES}/v0.3.36/llama_cpp_python-0.3.36-py3-none-win_amd64.whl",
                "ae5e88a2cf464e5a2dde7c8898c2d5c71cc24d03cac6396bf7509456bde0b9fb"),
    },
}
VERSION_FILE = "version.txt"


def backend_folders() -> dict[str, str]:
    """Back-end name -> folder name, for the builds that do not come with the installed wheel."""
    from voxflow.cleanup import LLAMA_BACKENDS
    return {b.name: b.folder for b in LLAMA_BACKENDS if b.folder != "lib"}


def package_dir() -> Path:
    from voxflow.cleanup import _llama_pkg_dir
    pkg = _llama_pkg_dir()
    if pkg is None:
        sys.exit("llama-cpp-python is not installed in this environment - install requirements.txt first.")
    return Path(pkg)


def in_place(folder: Path, version: str) -> bool:
    marker = folder / VERSION_FILE
    return (folder / "llama.dll").is_file() and marker.is_file() and marker.read_text("utf-8").strip() == version


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=120) as response:      # noqa: S310 - fixed https URL from PINS
        return response.read()


def install(folder: Path, wheel: bytes, sha256: str, version: str) -> list[str]:
    """Check ``wheel`` against its pin and unpack its llama_cpp/lib DLLs into ``folder``."""
    found = hashlib.sha256(wheel).hexdigest()
    if found != sha256:
        sys.exit(f"refusing the download for {folder.name}: SHA-256 is {found}, pinned is {sha256}")
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        names = [n for n in z.namelist() if n.startswith("llama_cpp/lib/") and n.lower().endswith(".dll")]
        if not any(n.endswith("/llama.dll") for n in names):
            sys.exit(f"the wheel for {folder.name} has no llama_cpp/lib/llama.dll")
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.dll"):
            old.unlink()                        # never a mix of two releases
        (folder / VERSION_FILE).unlink(missing_ok=True)
        for n in names:
            (folder / n.rsplit("/", 1)[1]).write_bytes(z.read(n))
    (folder / VERSION_FILE).write_text(version + "\n", "utf-8")       # last: marks the folder as complete
    return sorted(n.rsplit("/", 1)[1] for n in names)


def fetch(dest: Path | None = None, check_only: bool = False, fetcher=download) -> int:
    """Make every extra back-end folder match the installed llama-cpp-python. Returns 0, or 1 for a failed --check."""
    version = metadata.version("llama-cpp-python")
    pins = PINS.get(version)
    if pins is None:
        sys.exit(f"llama-cpp-python {version} is installed, but scripts/fetch_llama_backends.py pins builds only "
                 f"for {', '.join(PINS)}. Add the Vulkan and CPU wheels of {version} to PINS.")
    dest = dest or package_dir()
    missing = 0
    for name, folder_name in backend_folders().items():
        folder = dest / folder_name
        if in_place(folder, version):
            print(f"{name}: in place ({folder})")
            continue
        if check_only:
            print(f"{name}: MISSING ({folder})")
            missing += 1
            continue
        url, sha256 = pins[name]
        print(f"{name}: downloading {url}", flush=True)
        files = install(folder, fetcher(url), sha256, version)
        print(f"{name}: installed {', '.join(files)} into {folder}")
    return 1 if missing else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report whether the folders are in place (exit 1 if not)")
    ap.add_argument("--dest", type=Path, default=None, help="the llama_cpp package folder (default: the installed one)")
    args = ap.parse_args(argv)
    return fetch(args.dest, args.check)


if __name__ == "__main__":
    sys.exit(main())
