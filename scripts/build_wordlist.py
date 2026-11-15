"""Rebuild voxflow/data/english_words.txt.gz and english_names.txt.gz from SCOWL.

The Dictionary's close-misspelling correction must never turn an ordinary English word into a name
("copper" -> "Cooper"), so it needs to know what an ordinary word is. SCOWL (Spell Checker Oriented Word Lists,
http://wordlist.aspell.net/, MIT-like licence, see voxflow/data/SCOWL-Copyright.txt) grades words by how common
they are; sizes 10-50 are the words in everyday dictionaries, with their inflections ("jerkins", "spender").
Proper names live in separate SCOWL lists, so the word list does not contain "jenkins" or "cooper" only because
they are surnames; the names list (sizes up to 60) is used to leave spelling variants of real names alone.

Usage: python scripts/build_wordlist.py <path to an extracted scowl-2020.12.07 folder>
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

WORD_SIZE, NAME_SIZE = 50, 60
VARIETIES = ("english", "american", "british", "british_z", "canadian", "australian")
OUT = Path(__file__).resolve().parent.parent / "voxflow" / "data"


def _read(final: Path, kinds: tuple[str, ...], size: int) -> set[str]:
    out: set[str] = set()
    for f in final.iterdir():
        name, _, level = f.name.rpartition(".")
        if not level.isdigit() or int(level) > size:
            continue
        kind = next((k for k in kinds if name.endswith("-" + k)), None)
        if kind is None or name[:-len(kind) - 1].split("_variant")[0] not in VARIETIES:
            continue
        for line in f.read_text(encoding="latin-1").splitlines():
            w = line.strip()
            if w and "'" not in w and " " not in w:
                out.add(w)
    return out


def main(scowl: str) -> None:
    final = Path(scowl) / "final"
    words = {w for w in _read(final, ("words",), WORD_SIZE) if w.islower()}
    names = {w.lower() for w in _read(final, ("proper-names", "upper"), NAME_SIZE)} - words
    OUT.mkdir(exist_ok=True)
    for fname, items in (("english_words.txt.gz", words), ("english_names.txt.gz", names)):
        data = ("\n".join(sorted(items)) + "\n").encode("utf-8")
        with gzip.GzipFile(OUT / fname, "wb", compresslevel=9, mtime=0) as fh:
            fh.write(data)
        print(fname, len(items), "entries", (OUT / fname).stat().st_size, "bytes")
    (OUT / "SCOWL-Copyright.txt").write_text((Path(scowl) / "Copyright").read_text(encoding="latin-1"),
                                             encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
