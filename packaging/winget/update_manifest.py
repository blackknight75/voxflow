r"""Fill the winget manifest templates in this folder for one release.

    python packaging\winget\update_manifest.py <version> <sha256> [--license MIT] [--out DIR] [--strict]

<version> is the released version without the "v" (it must be the one in the installer's file name) and <sha256>
the SHA-256 of VoxFlow-Setup-<version>.exe (the release's SHA256SUMS.txt has it). The three manifests are written to
--out, by default dist\winget\manifests\b\blackknight75\VoxFlow\<version>\ (the layout of microsoft/winget-pkgs).

The licence is the owner's choice: until it is given with --license (or written into the locale template) the
token __LICENSE__ stays in the output and the manifest must not be submitted; --strict turns that into an error.
Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PACKAGE_ID = "blackknight75.VoxFlow"
TEMPLATES = (f"{PACKAGE_ID}.yaml", f"{PACKAGE_ID}.installer.yaml", f"{PACKAGE_ID}.locale.en-US.yaml")
TOKEN = re.compile(r"__[A-Z0-9]+__")


def default_out(version: str) -> Path:
    publisher, name = PACKAGE_ID.split(".", 1)
    return ROOT / "dist" / "winget" / "manifests" / publisher[0].lower() / publisher / name / version


def render(version: str, sha256: str, license_id: str | None = None, templates: Path = HERE) -> dict[str, str]:
    """Return {file name: text} for the three manifests. Raises ValueError on a malformed version or hash."""
    version = version.strip()
    if version.lower().startswith("v"):
        version = version[1:]
    if not re.fullmatch(r"\d+(\.\d+){1,3}([-+.]?[0-9A-Za-z][0-9A-Za-z.-]*)?", version):
        raise ValueError(f"not a version: {version!r} (expected e.g. 1.2.3)")
    sha256 = sha256.strip()
    if not re.fullmatch(r"[0-9A-Fa-f]{64}", sha256):
        raise ValueError("the SHA-256 must be 64 hexadecimal characters")
    values = {"__VERSION__": version, "__SHA256__": sha256.upper()}
    if license_id and license_id.strip():
        values["__LICENSE__"] = license_id.strip()
    out = {}
    for name in TEMPLATES:
        text = (templates / name).read_text(encoding="utf-8")
        # the "# Template: ..." line describes the file in the repository, not the filled-in copy
        text = "".join(line for line in text.splitlines(keepends=True) if not line.startswith("# Template:"))
        for token, value in values.items():
            text = text.replace(token, value)
        out[name] = text
    return out


def unfilled(files: dict[str, str]) -> list[str]:
    return sorted({t for text in files.values() for t in TOKEN.findall(text)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("sha256")
    ap.add_argument("--license", dest="license_id", help="SPDX id or name of the licence, e.g. MIT")
    ap.add_argument("--out", type=Path, help="output folder (default: dist/winget/manifests/b/.../<version>)")
    ap.add_argument("--strict", action="store_true", help="fail if a placeholder (the licence) is still unfilled")
    args = ap.parse_args(argv)
    try:
        files = render(args.version, args.sha256, args.license_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    left = unfilled(files)
    if left and args.strict:
        print(f"error: unfilled placeholder(s) {', '.join(left)}: pass --license", file=sys.stderr)
        return 3
    version = args.version.strip().lstrip("vV")
    out = args.out or default_out(version)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    print(out)
    if left:
        print(f"warning: {', '.join(left)} is not filled in; this manifest cannot be submitted yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
