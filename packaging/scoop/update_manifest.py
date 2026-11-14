r"""Point the Scoop manifest (bucket/voxflow.json) at one published release.

    python packaging\scoop\update_manifest.py <version> <sha256> [--manifest FILE]

<version> is the released version without the "v" (it must be the one in the installer's file name) and <sha256>
the SHA-256 of VoxFlow-Setup-<version>.exe (the release's SHA256SUMS.txt has it). The manifest is rewritten in
place: "version", the 64-bit download URL (built from the "autoupdate" URL, so the two cannot drift apart) and its
"hash". Everything else is kept as it is. .github/workflows/scoop.yml runs this when a release is published.

PLACEHOLDER_HASH (64 zeros) is what the manifest carries before the first release exists: Scoop compares it with
the downloaded file and refuses the install ("Hash check failed!"). It is never accepted as input.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANIFEST = ROOT / "bucket" / "voxflow.json"
ARCH = "64bit"
PLACEHOLDER_HASH = "0" * 64


def normalise(version: str, sha256: str) -> tuple[str, str]:
    """Return (version, sha256) cleaned up. Raises ValueError on a malformed version or hash."""
    version = version.strip()
    if version.lower().startswith("v"):
        version = version[1:]
    if not re.fullmatch(r"\d+(\.\d+){1,3}([-+.]?[0-9A-Za-z][0-9A-Za-z.-]*)?", version):
        raise ValueError(f"not a version: {version!r} (expected e.g. 1.2.3)")
    sha256 = sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("the SHA-256 must be 64 hexadecimal characters")
    if sha256 == PLACEHOLDER_HASH:
        raise ValueError("that is the placeholder hash, not the installer's")
    return version, sha256


def render(manifest: dict, version: str, sha256: str) -> dict:
    """Return a copy of ``manifest`` for this release. Raises ValueError on a malformed version or hash."""
    version, sha256 = normalise(version, sha256)
    out = json.loads(json.dumps(manifest))
    template = out["autoupdate"]["architecture"][ARCH]["url"]
    out["version"] = version
    out["architecture"][ARCH]["url"] = template.replace("$version", version)
    out["architecture"][ARCH]["hash"] = sha256
    return out


def dumps(manifest: dict) -> str:
    """The manifest as it is committed: 4-space indent, key order kept, LF line ends, final newline."""
    return json.dumps(manifest, indent=4, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("sha256")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest to rewrite (default: bucket/voxflow.json)")
    args = ap.parse_args(argv)
    try:
        updated = render(json.loads(args.manifest.read_text(encoding="utf-8")), args.version, args.sha256)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.manifest.write_text(dumps(updated), encoding="utf-8", newline="\n")
    print(f"{args.manifest}: {updated['version']} {updated['architecture'][ARCH]['hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
