r"""Build VoxFlow-Setup-<version>.exe: PyInstaller (onedir) then Inno Setup.

    C:\...\.venv\Scripts\python.exe scripts\build_installer.py [--skip-freeze] [--skip-installer] [--clean]

Needs: pyinstaller (requirements-dev.txt) in the interpreter that runs this, and Inno Setup 6
(winget install --id JRSoftware.InnoSetup -e --scope user). The first freeze downloads the CUDA 13, Vulkan and CPU
builds of llama.cpp (about 200 MB, see scripts/fetch_llama_backends.py). Output: dist\VoxFlow\ (the app) and
dist\VoxFlow-Setup-<version>.exe. build\ and dist\ are git-ignored.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD, DIST = ROOT / "build", ROOT / "dist"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def version() -> str:
    from voxflow import __version__          # single source of truth
    return __version__


def find_iscc() -> str:
    found = shutil.which("ISCC") or shutil.which("iscc")
    candidates = [found] if found else []
    for base in (os.environ.get("LOCALAPPDATA", "") + r"\Programs", os.environ.get("ProgramFiles(x86)", ""),
                 os.environ.get("ProgramFiles", "")):
        candidates.append(str(Path(base) / "Inno Setup 6" / "ISCC.exe"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("Inno Setup 6 not found. Install: winget install --id JRSoftware.InnoSetup -e --scope user")


def write_version_file(path: Path, ver: str) -> None:
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)
    nums = tuple((int(x) for x in (ver.split(".") + ["0", "0", "0"])[:3])) + (0,)
    info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums, mask=0x3F, flags=0, OS=0x40004, fileType=1, subtype=0),
        kids=[StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", "VoxFlow"), StringStruct("FileDescription", "VoxFlow"),
            StringStruct("FileVersion", ver), StringStruct("InternalName", "VoxFlow"),
            StringStruct("OriginalFilename", "VoxFlow.exe"), StringStruct("ProductName", "VoxFlow"),
            StringStruct("ProductVersion", ver)])]),
              VarFileInfo([VarStruct("Translation", [0x409, 1200])])])
    path.write_text(str(info), encoding="utf-8")


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-freeze", action="store_true", help="reuse dist/VoxFlow")
    ap.add_argument("--skip-installer", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete build/ and dist/ first")
    args = ap.parse_args()
    t0 = time.time()
    ver = version()
    if args.clean:
        shutil.rmtree(BUILD, ignore_errors=True)
        shutil.rmtree(DIST, ignore_errors=True)
    BUILD.mkdir(exist_ok=True)
    DIST.mkdir(exist_ok=True)
    icon, vfile = BUILD / "voxflow.ico", BUILD / "version_info.txt"
    run([sys.executable, str(ROOT / "packaging" / "make_icon.py"), str(icon)])
    write_version_file(vfile, ver)
    if not args.skip_freeze:
        # the CUDA 13, Vulkan and CPU builds of llama.cpp the spec bundles next to the CUDA 12 one (pinned by SHA-256;
        # a no-op when they are already in site-packages)
        import fetch_llama_backends
        fetch_llama_backends.fetch()
        env = dict(os.environ, VOXFLOW_ICON=str(icon), VOXFLOW_VERSION_FILE=str(vfile))
        run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "voxflow.spec"), "--noconfirm",
             "--distpath", str(DIST), "--workpath", str(BUILD / "pyi")], cwd=str(ROOT), env=env)
    if not args.skip_installer:
        run([find_iscc(), f"/DAppVersion={ver}", f"/DSourceDir={DIST / 'VoxFlow'}", f"/DOutDir={DIST}",
             f"/DIconFile={icon}", str(ROOT / "packaging" / "voxflow.iss")])
    print(f"done in {time.time() - t0:.0f} s: {DIST / ('VoxFlow-Setup-' + ver + '.exe')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
