"""Print the CHANGELOG.md section of one version (the body of its GitHub release).

    python .github/scripts/release_notes.py <version> [--changelog CHANGELOG.md] [--out notes.md]

Accepts Keep-a-Changelog headings and their usual variants: "## [1.2.3] - 2026-01-31", "## 1.2.3", "## v1.2.3 (date)".
The section runs to the next heading of the same or a higher level; the link definitions at the foot of the file
("[1.2.3]: https://...") are left out. Exit code 0 = found, 1 = no file / no such section / empty section (the release
workflow then falls back to GitHub's generated notes). Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK_DEF = re.compile(r"^\[[^\]]+\]:\s+\S+")
FENCE = re.compile(r"^\s*(```|~~~)")


def _names_version(title: str, version: str) -> bool:
    # "[1.2.3] - date", "v1.2.3", "1.2.3 (date)", "[1.2.3](url) - date"; never 1.2.30 or 11.2.3
    return re.match(r"\[?v?" + re.escape(version) + r"(?![0-9A-Za-z.+-])", title, re.I) is not None


def extract(changelog: str, version: str) -> str | None:
    """Return the body of the section for ``version`` (without its heading), or None."""
    version = version.strip()
    if version[:1] in "vV":
        version = version[1:]
    if not version:
        return None
    body: list[str] | None = None
    level = 0
    fenced = False
    for line in changelog.splitlines():
        if FENCE.match(line):
            fenced = not fenced
        m = None if fenced else HEADING.match(line)
        if body is None:
            if m and _names_version(m.group(2), version):
                body, level = [], len(m.group(1))
            continue
        if m and len(m.group(1)) <= level:
            break
        if not fenced and LINK_DEF.match(line):
            continue
        body.append(line.rstrip())
    if body is None:
        return None
    text = "\n".join(body).strip("\n")
    return text if text.strip() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    ap.add_argument("--out", type=Path, help="write the notes here instead of stdout")
    args = ap.parse_args(argv)
    try:
        source = args.changelog.read_text(encoding="utf-8-sig")
    except OSError as exc:
        print(f"release notes: cannot read {args.changelog}: {exc}", file=sys.stderr)
        return 1
    notes = extract(source, args.version)
    if notes is None:
        print(f"release notes: no section for {args.version} in {args.changelog}", file=sys.stderr)
        return 1
    if args.out:
        args.out.write_text(notes + "\n", encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(notes + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
