r"""Fetch the extra llama.cpp builds VoxFlow chooses from at start: native CUDA 13 (RTX 50xx), Vulkan, plain CPU.

    .venv\Scripts\python.exe scripts\fetch_llama_backends.py            # once after installing requirements.txt
    .venv\Scripts\python.exe scripts\fetch_llama_backends.py --check    # no network: are they in place?

The installed llama-cpp-python wheel is the CUDA 12 build: it loads only on a PC with an NVIDIA driver. This script
puts the DLLs of other builds of the SAME llama-cpp-python release next to it:

    site-packages\llama_cpp\lib          the wheel's own (CUDA 12: native code for sm_60-sm_90)
    site-packages\llama_cpp\lib_cuda13   the wheel's own llama / ggml DLLs + a CUDA 13 ggml-cuda.dll with native code
                                         for RTX 50xx (sm_120), which the CUDA 12 build has to JIT-compile first
    site-packages\llama_cpp\lib_vulkan   AMD / Intel / NVIDIA through Vulkan
    site-packages\llama_cpp\lib_cpu      no GPU driver needed

``voxflow.cleanup`` picks one folder when the language model is loaded (see LLAMA_BACKENDS there), and
packaging/voxflow.spec bundles all of them into the installer; scripts/build_installer.py runs this first.

Where the files come from:
* Vulkan and CPU: the wheels the llama-cpp-python project publishes on its GitHub releases (the same files its
  package indexes https://abetlen.github.io/llama-cpp-python/whl/vulkan and .../whl/cpu point at). Only
  ``llama_cpp/lib/*.dll`` is taken from a wheel, and nothing is installed with pip.
* CUDA 13: ``ggml-cuda.dll`` from the Windows CUDA 13 build that ggml-org publishes for the llama.cpp release that
  llama-cpp-python vendors (0.3.36 vendors llama.cpp commit 0c1e570 = release b11312). It is the only file taken
  from that archive. llama.dll, ggml.dll, ggml-base.dll, ggml-cpu.dll and mtmd.dll are copied from the installed
  wheel's own lib folder, so the Python bindings talk to the same llama.dll as in ``lib``. The CUDA 13 cuBLAS it
  needs comes with onnxruntime-gpu's requirements (site-packages\nvidia\cu13).
Each download is pinned below by version and SHA-256; one that does not match is refused. Folders that are already
in place for the installed version are left alone, so a second run needs no network.
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
_LLAMA_CPP = "https://github.com/ggml-org/llama.cpp/releases/download"
# The files of the installed CUDA wheel that lib_cuda13 shares: everything but ggml-cuda.dll.
WHEEL_SHARED = ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll")
# llama-cpp-python version -> folder (voxflow.cleanup.LLAMA_BACKENDS) -> (kind, URL, SHA-256 of the download):
#   "wheel"    the llama_cpp/lib/*.dll of that llama-cpp-python wheel
#   "cuda13"   ggml-cuda.dll of that ggml-org llama.cpp archive, plus WHEEL_SHARED from the installed wheel
# When requirements.txt moves to another llama-cpp-python, add its row: every build must be of the same release (for
# the archive: the llama.cpp release whose commit is that llama-cpp-python's vendor/llama.cpp).
PINS = {
    "0.3.36": {
        "lib_cuda13": ("cuda13", f"{_LLAMA_CPP}/b11312/llama-b11312-bin-win-cuda-13.4-x64.zip",
                       "1c545a1d67e0c66b8e7e3e31c81b6d6532af9ed94419b646426e5175d7a49e8e"),
        "lib_vulkan": ("wheel", f"{_RELEASES}/v0.3.36-vulkan/llama_cpp_python-0.3.36-py3-none-win_amd64.whl",
                       "b3d58d70e9fa6216f3d1919bb6141a98eb22eb889e14a44d080dd250e969fab4"),
        "lib_cpu": ("wheel", f"{_RELEASES}/v0.3.36/llama_cpp_python-0.3.36-py3-none-win_amd64.whl",
                    "ae5e88a2cf464e5a2dde7c8898c2d5c71cc24d03cac6396bf7509456bde0b9fb"),
    },
}
VERSION_FILE = "version.txt"


def backend_folders() -> list[str]:
    """The folders of the builds that do not come with the installed wheel, in LLAMA_BACKENDS order."""
    from voxflow.cleanup import LLAMA_BACKENDS
    return [b.folder for b in LLAMA_BACKENDS if b.folder != "lib"]


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


def _check(folder: Path, data: bytes, sha256: str) -> None:
    found = hashlib.sha256(data).hexdigest()
    if found != sha256:
        sys.exit(f"refusing the download for {folder.name}: SHA-256 is {found}, pinned is {sha256}")


def _replace(folder: Path, files: dict, version: str) -> list[str]:
    """Make ``folder`` hold exactly ``files`` (name -> bytes) and mark it complete for ``version``."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.dll"):
        old.unlink()                            # never a mix of two releases
    (folder / VERSION_FILE).unlink(missing_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    (folder / VERSION_FILE).write_text(version + "\n", "utf-8")       # last: marks the folder as complete
    return sorted(files)


def install(folder: Path, wheel: bytes, sha256: str, version: str) -> list[str]:
    """Check ``wheel`` against its pin and unpack its llama_cpp/lib DLLs into ``folder``."""
    _check(folder, wheel, sha256)
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        names = [n for n in z.namelist() if n.startswith("llama_cpp/lib/") and n.lower().endswith(".dll")]
        if not any(n.endswith("/llama.dll") for n in names):
            sys.exit(f"the wheel for {folder.name} has no llama_cpp/lib/llama.dll")
        return _replace(folder, {n.rsplit("/", 1)[1]: z.read(n) for n in names}, version)


def install_cuda13(folder: Path, archive: bytes, sha256: str, version: str, wheel_lib: Path) -> list[str]:
    """Check ``archive`` against its pin and put its ggml-cuda.dll into ``folder``, next to WHEEL_SHARED copied from
    ``wheel_lib`` (the installed wheel's own lib folder, which must be the CUDA build)."""
    _check(folder, archive, sha256)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        cuda = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == "ggml-cuda.dll"]
        if len(cuda) != 1:
            sys.exit(f"the archive for {folder.name} has no single ggml-cuda.dll")
        files = {"ggml-cuda.dll": z.read(cuda[0])}
    for name in WHEEL_SHARED:
        if not (wheel_lib / name).is_file():
            sys.exit(f"{wheel_lib / name} is missing: install requirements.txt (the CUDA wheel) first")
        files[name] = (wheel_lib / name).read_bytes()
    if b"ggml-cuda.dll" not in files["ggml.dll"]:
        sys.exit(f"{wheel_lib / 'ggml.dll'} is not from the CUDA wheel (it does not load ggml-cuda.dll): install "
                 "llama-cpp-python from the cu124 index named in requirements.txt")
    return _replace(folder, files, version)


def fetch(dest: Path | None = None, check_only: bool = False, fetcher=download) -> int:
    """Make every extra back-end folder match the installed llama-cpp-python. Returns 0, or 1 for a failed --check."""
    version = metadata.version("llama-cpp-python")
    pins = PINS.get(version)
    if pins is None:
        sys.exit(f"llama-cpp-python {version} is installed, but scripts/fetch_llama_backends.py pins builds only "
                 f"for {', '.join(PINS)}. Add the builds of {version} to PINS.")
    dest = dest or package_dir()
    missing = 0
    for folder_name in backend_folders():
        folder = dest / folder_name
        if in_place(folder, version):
            print(f"{folder_name}: in place ({folder})")
            continue
        if check_only:
            print(f"{folder_name}: MISSING ({folder})")
            missing += 1
            continue
        kind, url, sha256 = pins[folder_name]
        print(f"{folder_name}: downloading {url}", flush=True)
        if kind == "cuda13":
            files = install_cuda13(folder, fetcher(url), sha256, version, dest / "lib")
        else:
            files = install(folder, fetcher(url), sha256, version)
        print(f"{folder_name}: installed {', '.join(files)} into {folder}")
    return 1 if missing else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report whether the folders are in place (exit 1 if not)")
    ap.add_argument("--dest", type=Path, default=None, help="the llama_cpp package folder (default: the installed one)")
    args = ap.parse_args(argv)
    return fetch(args.dest, args.check)


if __name__ == "__main__":
    sys.exit(main())
