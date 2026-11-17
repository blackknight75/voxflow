"""JSON-backed, thread-safe stores: dictionary, snippets, history, stats and styles.

Every store
* lives in one JSON file (written atomically) inside the data directory, which callers may inject;
* never raises on I/O trouble: it logs, keeps working in memory and retries on the next change;
* hands out copies, so callers cannot corrupt its state;
* raises ``ValueError`` with a message fit for the user when given invalid input;
* notifies subscribers (``subscribe(callback)``; ``callback(topic)``) after each change, outside its lock.
"""
from __future__ import annotations

import atexit
import copy
import csv
import datetime as _dt
import difflib
import io
import json
import logging
import math
import re
import threading
import time
import uuid
import weakref
from pathlib import Path

from . import config as cfgmod

log = logging.getLogger("voxflow.store")

TYPING_WPM = 40.0           # the typing speed "time saved" is measured against
_MAX_TERM_LEN = 80
_MAX_SUGGESTIONS = 50


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _norm_spaces(s: str) -> str:
    return " ".join(str(s).split())


def clean_text(s) -> str:
    """Make text safe to store: lone surrogates become U+FFFD, NUL and other C0 controls (except tab / newline) go."""
    s = str(s)
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-8", "replace").decode("utf-8")
    return "".join(ch for ch in s if ch in "\t\n\r" or (ch >= " " and ch != "\x7f"))


def count_words(text: str) -> int:
    return len(text.split())


def _layout(data) -> dict | None:
    """A store file's top level: None when there is no file yet, else it must be a JSON object."""
    if data is not None and not isinstance(data, dict):
        raise ValueError("the file does not hold a JSON object")
    return data


def _rows(data: dict, key: str) -> list:
    """The list stored under ``key`` (missing: empty). Anything else is a layout this store does not know."""
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"“{key}” is not a list")
    return value


def _alias_list(value) -> list[str]:
    """Sounds-like spellings as a file holds them: a list, or one string of comma-separated spellings ("kube,
    cube net"), which is never taken letter by letter. Anything else holds none."""
    if isinstance(value, str):
        return [a for a in value.split(",") if a.strip()]
    if isinstance(value, (list, tuple)):
        return [a for a in value if isinstance(a, str)]
    return []


_STORES: "weakref.WeakSet" = weakref.WeakSet()


def flush_all() -> None:
    """Write every store's pending change (called at exit)."""
    for st in list(_STORES):
        try:
            st.flush()
        except Exception:  # noqa: BLE001
            pass


atexit.register(flush_all)


class _JsonStore:
    """Shared plumbing: load, atomic save, change notification."""

    filename = ""

    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else cfgmod.app_dir()
        self.path = path / self.filename if path.is_dir() or not path.suffix else path
        self._lock = threading.RLock()
        self._listeners: list = []
        _STORES.add(self)
        self._read()

    # ---- persistence
    def _read(self) -> None:
        for other in list(_STORES):               # a second store on the same file must see the first one's pending write
            if other is not self and getattr(other, 'path', None) == self.path:
                try:
                    other.flush()
                except Exception:  # noqa: BLE001
                    pass
        data = None
        self._skipped = 0                          # rows ``_load`` had to leave out
        broken = False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))    # tolerate an editor's BOM
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.warning("%s is unreadable (%s); starting empty and keeping a copy", self.path.name, exc)
            broken = True
        if not broken:
            try:
                self._load(data)
            except Exception as exc:  # noqa: BLE001 - a malformed file must never stop the app
                log.warning("%s has an unexpected layout (%s); starting empty and keeping a copy", self.path.name,
                            exc)
                broken = True
        if broken:
            self._skipped = 0
            self._dirty = False
            self._load(None)
        elif self._skipped:
            log.warning("%s: %d damaged entries were left out; keeping a copy", self.path.name, self._skipped)
        if broken or self._skipped:
            self._report_load_problem(self._keep_corrupt_copy(), self._skipped)
        if self._dirty:                            # the load changed the data (a migration): write it now,
            with self._lock:                       # after the copy above, before anything waits on the store
                self._write()

    # What the user is told when the file could not be read (completely, or some entries of it). The controller
    # shows it once at start-up with "Open data folder", where the untouched copy of the file is kept.
    load_problem = ""
    label = "this file"                        # "your dictionary": what the file holds, for that message

    def _report_load_problem(self, copy_name: str, skipped: int) -> None:
        kept = f" The original file was kept as {copy_name} in VoxFlow's data folder." if copy_name else ""
        if skipped:
            self.load_problem = (f"{skipped} entr{'y' if skipped == 1 else 'ies'} in {self.label} could not be read "
                                 f"and {'was' if skipped == 1 else 'were'} left out.{kept}")
        else:
            self.load_problem = f"{self.label[0].upper()}{self.label[1:]} could not be read, so it starts empty.{kept}"

    def _corrupt_copies(self) -> list[Path]:
        """The copies of unreadable files kept so far (``history.corrupt-20261117-072301.json``, and the
        older single ``history.corrupt.json``)."""
        return sorted(self.path.parent.glob(f"{self.path.stem}.corrupt*{self.path.suffix}"))

    def _keep_corrupt_copy(self) -> str:
        """Copy the unreadable file aside, under a name with the time, so a second bad file never overwrites the
        copy of the first. Returns the copy's file name ('' when there was nothing to copy or it failed)."""
        base = f"{self.path.stem}.corrupt-{_dt.datetime.now():%Y%m%d-%H%M%S}"
        target, n = self.path.with_name(base + self.path.suffix), 1
        while target.exists():                     # two bad files within one second
            n += 1
            target = self.path.with_name(f"{base}-{n}{self.path.suffix}")
        try:
            target.write_bytes(self.path.read_bytes())
        except OSError as exc:
            log.warning("could not keep a copy of %s: %s", self.path.name, exc)
            return ""
        return target.name

    BACKUP_TAGS: tuple[str, ...] = ()           # every ``_backup`` tag this store can write
    _purge_backups = False      # data was deleted: drop the backup copies after the next successful write
    # ... and the copies of unreadable files too (only when the user deleted everything: Clear all, a replacing
    # import). Retention, the size limit and single deletions keep them: such a copy holds data the user has not
    # seen in the app since, and may still want back.
    _purge_corrupt = False

    def _backup_path(self, tag: str) -> Path:
        return self.path.with_name(f"{self.path.stem}.{tag}{self.path.suffix}")

    def _backup(self, tag: str) -> None:
        """Keep a copy of the file as it is (``history.v1.json`` before a migration rewrites it)."""
        try:
            self._backup_path(tag).write_bytes(self.path.read_bytes())
        except OSError as exc:
            log.warning("could not keep a copy of %s: %s", self.path.name, exc)

    def _drop_backups(self, corrupt: bool = False) -> None:
        """Delete the copies ``_backup`` kept (``history.v1.json``): they hold the same data as the file, so
        deleted data must not survive in them. With ``corrupt`` also the copies of unreadable files. A copy that
        cannot be deleted is logged and is tried again after the next deletion."""
        paths = [self._backup_path(tag) for tag in self.BACKUP_TAGS] + (self._corrupt_copies() if corrupt else [])
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete %s: %s", path.name, exc.strerror or exc)

    def _load(self, data) -> None:
        raise NotImplementedError

    def _dump(self):
        raise NotImplementedError

    indent: int | None = 1          # small hand-editable files stay readable; the big history file is compact

    COALESCE_S = 0.2                 # saves closer together than this share one write (burst of adds / imports)
    # The history is written behind the caller, on the write timer's thread: a dictation never waits for it (at
    # 100,000 transcripts the file is about 40 MB and took about 165 ms per dictation to write).
    WRITE_BEHIND = False
    _dirty = False
    _last_write = 0.0
    _timer = None
    _batch = 0
    _changes = 0                     # bumped by every _save: did more come in while a write was being prepared?
    _seq = 0                         # bumped by every write: an older write never lands after a newer one

    def _save(self) -> None:
        """Ask for the file to be written. The first change after a quiet moment is written at once (so a single
        edit is on disk when the call returns; a ``WRITE_BEHIND`` store writes it on the timer thread right
        away); changes that follow within ``COALESCE_S`` are folded into one trailing write, so a bulk add costs a
        few writes, not one (with an fsync) per entry. ``flush()`` writes whatever is pending; ``with
        store.batch():`` holds every save back until the block ends."""
        with self._lock:
            self._changes += 1
            if self._batch:
                self._dirty = True
                return
            wait = self._last_write + self.COALESCE_S - time.monotonic()
            if wait <= 0 and self._timer is None and not self.WRITE_BEHIND:
                self._write()
                return
            self._dirty = True
            if self._timer is None:
                self._timer = threading.Timer(max(wait, 0.01), self._timer_fired)
                self._timer.daemon = True
                self._timer.name = "voxflow-store-write"
                self._timer.start()

    def _timer_fired(self) -> None:
        if self.WRITE_BEHIND:
            self._write_behind()
            return
        with self._lock:
            self._timer = None
            if self._dirty and not self._batch:
                self._write()

    def _write_behind(self) -> None:
        """The write of a ``WRITE_BEHIND`` store (timer thread). The data is turned into JSON outside the lock,
        so a page reading the history or the next dictation filing its transcript does not wait for that; only
        the file replace itself runs under the lock, where a newer write (``flush``) wins over this one."""
        with self._lock:
            self._timer = None
            if not self._dirty or self._batch:
                return
            self._seq += 1
            seq, changes = self._seq, self._changes
            purge = (self._purge_backups, self._purge_corrupt)
            payload = self._dump()                       # a snapshot: the store keeps changing meanwhile
        text = self._serialise(payload)
        with self._lock:
            if seq == self._seq:                         # else a flush() already wrote newer data
                self._commit(text, purge, current=changes == self._changes)

    def flush(self) -> None:
        """Write any pending change now (shutdown, export of the raw file, tests)."""
        with self._lock:
            timer, self._timer = self._timer, None
            if timer is not None:
                timer.cancel()
            if self._dirty:
                self._write()

    def retry_save(self) -> bool:
        """Try to write the file again after a failed save. True when the file is current."""
        with self._lock:
            self._dirty = True
            timer, self._timer = self._timer, None
            if timer is not None:
                timer.cancel()
            self._write()
            return not self.last_save_error

    class _Batch:
        def __init__(self, store):
            self.store = store

        def __enter__(self):
            with self.store._lock:
                self.store._batch += 1
            return self.store

        def __exit__(self, *exc):
            with self.store._lock:
                self.store._batch -= 1
            if not self.store._batch:
                self.store.flush()

    def batch(self):
        """Context manager: many changes, one write at the end."""
        return self._Batch(self)

    def _write(self) -> None:
        """Write the file atomically, now (call with the lock held). Never raises; a failure is remembered in
        ``last_save_error`` and announced once to the subscribers as the topic ``"save_failed:<file stem>"`` (and
        ``"save_ok:<stem>"`` when a later save works again)."""
        self._seq += 1
        self._commit(self._serialise(self._dump()), (self._purge_backups, self._purge_corrupt), current=True)

    def _serialise(self, payload):
        """The file's text, or the exception that stopped it (``_commit`` reports it)."""
        try:
            return json.dumps(payload, ensure_ascii=False, indent=self.indent)
        except (ValueError, TypeError) as exc:
            return exc

    def _commit(self, text, purge: tuple[bool, bool], current: bool) -> None:
        """Put ``text`` on disk (lock held). ``current``: it holds every change made so far. ``purge``: the backup
        purge that was pending when ``text`` was made (it may only run once a file without that data is on disk)."""
        if current:
            self._dirty = False
        self._last_write = time.monotonic()
        try:
            if isinstance(text, Exception):
                raise text
            cfgmod._atomic_write(self.path, text)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:   # lone surrogates raise UnicodeEncodeError
            log.warning("could not save %s: %s", self.path.name, exc)
            first = not self.last_save_error
            self.last_save_error = str(exc) or type(exc).__name__
            if first:
                self._notify("save_failed:" + self.path.stem)
            return                               # a pending backup purge waits for a write that works
        if purge[0]:                             # the file without the deleted data is on disk now
            if current:
                self._purge_backups = self._purge_corrupt = False
            self._drop_backups(purge[1])
        if self.last_save_error:
            self.last_save_error = ""
            self._notify("save_ok:" + self.path.stem)

    last_save_error = ""

    # ---- notifications
    def subscribe(self, callback) -> None:
        """``callback(topic: str)`` is called after every change (from the thread that made it)."""
        with self._lock:
            self._listeners.append(callback)

    def _notify(self, topic: str) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(topic)
            except Exception:  # noqa: BLE001
                log.exception("store listener failed")


# =========================================================================== dictionary
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#]*(?:[.\-'][A-Za-z0-9+#]+)*")


def _split_camel(term: str) -> str:
    """'VoxFlow' -> 'vox flow', 'PyTorch2' -> 'py torch 2' (how speech recognition tends to hear such names)."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[A-Z])(?=[A-Z][a-z])", " ", term)
    return _norm_spaces(s).lower()


def _match_key(text: str) -> str:
    """Lookup key for dictionary matching: case-insensitive, and a hyphen counts as a space
    ("wi-fi", "Wi Fi" and "wi  fi" are the same key)."""
    return " ".join(re.split(r"[\s\-]+", str(text).strip().lower())).strip()


def _interesting_casing(tok: str) -> bool:
    """Names the recogniser cannot guess: camelCase, ACRONYMS, letters mixed with digits."""
    core = tok.replace("'", "")
    if len(core) < 2:
        return False
    if any(c.isupper() for c in core[1:]):
        return True
    return any(c.isdigit() for c in core) and any(c.isalpha() for c in core)


# ---- close-misspelling correction (the recogniser cannot be told the user's words, so they are fixed afterwards)
#
# Precision first: a missed correction costs the user one edit, a corrupted sentence ("Red is my colour" ->
# "Redis my colour") costs their trust. So a span of words is only ever replaced when it is *not* made of real
# English words, judged by a bundled word list (SCOWL, see voxflow/data/SCOWL-Copyright.txt and
# scripts/build_wordlist.py). A "sounds like" alias the user typed is the only way to replace real words.
_FUZZY_MIN_LEN = 5                  # shorter terms are only matched exactly: "Dan" must never eat "can"
_FUZZY_MIN_WORD = 7                 # one heard word needs this many letters ("Monika" is not a typo of "Monica")
_FUZZY_MAX_WORDS = 4                # a name heard as at most this many words ("Supercaliphragolistus expialidocio")
_FUZZY_WORD_RE = re.compile(r"\w+(?:['’]\w+)*")
_FUZZY_GAP_RE = re.compile(r"[ \t]+|[ \t]*-[ \t]*")       # what may sit between the words of one spoken name
_POSSESSIVE_RE = re.compile(r"['’]s$", re.I)
_LETTERS_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
# a term is a word of prose: never touch it inside a URL, a path, an e-mail address or an identifier
# ("github.com/voxflow", "VOXFLOW_DEVICE", "C:\\voxflow\\logs"); a sentence's full stop after it is fine
_NOT_IN_CODE_BEFORE = r"(?<![\w./\\@])"
_NOT_IN_CODE_AFTER = r"(?![\w@/\\]|[.:]\w)"
_CODE_BEFORE_RE = re.compile(r"[\w./\\@]$")
_CODE_AFTER_RE = re.compile(r"^(?:[\w@/\\]|[.:]\w)")
_DATA_DIR = Path(__file__).resolve().parent / "data"
_WORDLISTS: tuple[frozenset, frozenset] | None = None
_WORDLISTS_LOCK = threading.Lock()


def _wordlists() -> tuple[frozenset, frozenset]:
    """(ordinary English words, given names and surnames), lower case, loaded once (about 30 ms, 2 MB).
    Both are empty when the files are missing: fuzzy correction is then switched off, never left unguarded."""
    global _WORDLISTS
    if _WORDLISTS is None:
        with _WORDLISTS_LOCK:
            if _WORDLISTS is None:
                import gzip

                try:
                    lists = tuple(frozenset(gzip.decompress((_DATA_DIR / name).read_bytes()).decode("utf-8").split())
                                  for name in ("english_words.txt.gz", "english_names.txt.gz"))
                except (OSError, ValueError, EOFError) as exc:
                    log.warning("word list unavailable, dictionary fuzzy correction is off: %s", exc)
                    lists = (frozenset(), frozenset())
                _WORDLISTS = lists                     # type: ignore[assignment]
    return _WORDLISTS                                  # type: ignore[return-value]


def _is_word(word: str, words: frozenset) -> bool:
    """Is ``word`` an ordinary English word ("copper", "jerkins", "Red", "isn't", "colour's")?"""
    w = word.lower().replace("’", "'")
    if w in words:
        return True
    if w.endswith("'s"):
        return w[:-2] in words
    if "'" in w:
        return w.replace("'", "") in words or w.split("'")[0] in words
    return False


def _all_words(text: str, words: frozenset) -> bool:
    """Is every word of ``text`` an ordinary English word ("red is", "Linear", "look her up")?"""
    if not words or re.search(r"[^\w\s'’\-]|[\d_]", text):
        return False                                   # "C++", "Node.js", "PyTorch2" are not prose
    parts = _LETTERS_RE.findall(text)
    return bool(parts) and all(_is_word(p, words) for p in parts)


def _looks_english(text: str, words: frozenset, names: frozenset) -> bool:
    """The word list only knows English: in another language every word would look like a misspelling."""
    parts = [p.lower() for p in _LETTERS_RE.findall(text)]
    if len(parts) < 3:
        return True
    known = sum(1 for p in parts if _is_word(p, words) or p in names)
    return known * 2 >= len(parts)


def _squash(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def _first_sound(key: str) -> str:
    c = key[:1]
    return "a" if c in "aeiou" else c                  # "Otto" / "auto": a vowel is a vowel


def _fuzzy_index(entries) -> dict:
    """What ``_fuzzy_fix`` matches against: every term and alias of ``_FUZZY_MIN_LEN`` letters or more."""
    from .cleanup import phonetic_key

    words, names = _wordlists() if entries else (frozenset(), frozenset())
    if not words:
        return {}
    by_sound: dict[str, list[tuple[str, str, str]]] = {}
    longest = 0
    for e in entries:
        for k in [e["term"]] + list(e["aliases"]):
            sq = _squash(k)
            if len(sq) < _FUZZY_MIN_LEN or sq.isdigit():
                continue
            ph = phonetic_key(sq)
            by_sound.setdefault(_first_sound(ph), []).append((sq, ph, e["term"]))
            longest = max(longest, len(sq))
    if not by_sound:
        return {}
    return {"by_sound": by_sound, "longest": longest, "words": words, "names": names,
            "terms": frozenset(e["term"] for e in entries)}


def _fuzzy_allowed(span_words: list[str], index: dict) -> bool:
    """May this span be replaced at all? Never when it is made of real English words: "red is", "copper",
    "is in", "look her", "spender" stay as they are, however much they sound like a term."""
    words, names = index["words"], index["names"]
    if len(span_words) == 1:
        w = span_words[0]
        return len(_squash(w)) >= _FUZZY_MIN_WORD and not _is_word(w, words) and w.lower() not in names
    # several words: at least half of the letters must be in words that are not English ("Go lang", "post
    # gress", "Supercaliphragolistus expialidocio"), so "dag stir" or "inter com" are left alone
    unknown = sum(len(_squash(w)) for w in span_words if not _is_word(w, words))
    return unknown * 2 >= sum(len(_squash(w)) for w in span_words)


def _fuzzy_score(cand: str, n_words: int, index: dict) -> tuple[float, str] | None:
    """(similarity, term) of the dictionary key closest to the squashed span ``cand``, or None."""
    from .cleanup import phonetic_key

    sound = phonetic_key(cand)
    best = None
    for sq, ph, term in index["by_sound"].get(_first_sound(sound), ()):
        if abs(len(cand) - len(sq)) > max(2, len(sq) // 4):
            continue
        if cand == sq:
            score = 1.0
        else:
            if n_words == 1 and (cand.startswith(sq) or sq.startswith(cand)):
                continue                       # the term plus / minus an ending ("Stripes")
            m = difflib.SequenceMatcher(None, cand, sq)
            if m.real_quick_ratio() < 0.75 or m.quick_ratio() < 0.75:
                continue
            spelt, heard = m.ratio(), difflib.SequenceMatcher(None, sound, ph).ratio()
            # it must both look and sound like the term: one word closely ("Kubernetis", "Graphana"), a name
            # split into words ("Supercaliphragolistus expialidocio", "Go lang") a little more loosely
            if n_words == 1:
                ok = heard >= 0.84 and spelt >= 0.78 and spelt + heard >= 1.72
            else:
                ok = spelt >= 0.8 and heard >= 0.8
            if not ok:
                continue
            score = (spelt + heard) / 2 - 0.02 * (n_words - 1)
        if best is None or score > best[0]:
            best = (score, term)
    return best


def _fuzzy_fix(text: str, index: dict) -> str:
    """Replace close misspellings of dictionary terms, left to right, never across punctuation."""
    if not _looks_english(text, index["words"], index["names"]):
        return text
    words = list(_FUZZY_WORD_RE.finditer(text))
    out: list[str] = []
    last = i = 0
    while i < len(words):
        best = None                                     # (score, n_words, replacement)
        for n in range(1, _FUZZY_MAX_WORDS + 1):
            if i + n > len(words):
                break
            if n > 1 and not _FUZZY_GAP_RE.fullmatch(text[words[i + n - 2].end():words[i + n - 1].start()]):
                break                                   # "Get. Hub" is two sentences, not one name
            if _CODE_BEFORE_RE.search(text[:words[i].start()][-1:]) or _CODE_AFTER_RE.match(
                    text[words[i + n - 1].end():words[i + n - 1].end() + 2]):
                continue                                # part of a URL, path or identifier
            span = text[words[i].start():words[i + n - 1].end()]
            suffix = ""
            if _POSSESSIVE_RE.search(span):
                span, suffix = span[:-2], span[-2:]
            cand = _squash(span)
            if len(cand) > index["longest"] + 3:
                break
            if span in index["terms"]:
                hit = (1.01, span)                      # already right: keep it, and keep the next word
            elif _fuzzy_allowed([w.group(0) for w in words[i:i + n]], index):
                hit = _fuzzy_score(cand, n, index)
            else:
                continue
            if hit is None:
                continue
            if best is None or hit[0] > best[0]:
                best = (hit[0], n, hit[1] + suffix)
        if best is None:
            i += 1
            continue
        _score, n, replacement = best
        out.append(text[last:words[i].start()])
        out.append(replacement)
        last = words[i + n - 1].end()
        i += n
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


def _read_import_text(path: Path) -> str:
    """A file the user picked to import: UTF-8 (with or without a BOM), else the Windows "ANSI" code page that
    Excel and Notepad use in Western Europe (cp1252), so "Müller" never stops an import with a codec error."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Couldn't read {path.name}: {exc.strerror or exc}") from None
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):        # Notepad's "Unicode" (UTF-16)
            return raw.decode("utf-16")
        if b"\x00" not in raw:
            try:
                return raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                return raw.decode("cp1252", errors="replace")
    except UnicodeDecodeError:
        pass
    raise ValueError(f"Couldn't read {path.name}: it is not a text file.")


# .txt dictionaries: "term: alias, alias". A ":" in the term, a "," in an alias, a backslash, and a "#" that starts
# the line are written with a backslash in front, so exporting and importing again gives the same entries.
_TXT_ESCAPABLE = "\\:,#"


def _txt_escape(text: str, chars: str) -> str:
    return "".join("\\" + c if c in chars else c for c in text)


def _txt_line(term: str, aliases: list[str]) -> str:
    head = _txt_escape(term, "\\:")
    if head.startswith("#"):
        head = "\\" + head                              # not a comment line
    return head + (": " + ", ".join(_txt_escape(a, "\\,") for a in aliases) if aliases else "")


def _txt_fields(line: str) -> tuple[str, list[str]]:
    """One line of a .txt dictionary: 'Ratio\\: 3: ratio three, a\\, b' -> ('Ratio: 3', ['ratio three', 'a, b']).
    A backslash before any other character is kept as it is."""
    term: str | None = None
    parts: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] in _TXT_ESCAPABLE:
            cur.append(line[i + 1])
            i += 2
            continue
        if c == ":" and term is None:
            term, cur = "".join(cur), []
        elif c == "," and term is not None:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if term is None:
        return "".join(cur), []
    parts.append("".join(cur))
    return term, [a for a in parts if a.strip()]


def _index_entry(entry: dict, taken: dict, alias_of: dict, remove: bool = False) -> None:
    """Add (or remove) an entry's term and aliases in the "who uses this spelling" maps of a bulk import."""
    keys = [(entry["term"].lower(), (taken,))] + [(a.lower(), (taken, alias_of)) for a in entry["aliases"]]
    for key, maps in keys:
        for m in maps:
            ids = m.setdefault(key, set())
            if remove:
                ids.discard(entry["id"])
            else:
                ids.add(entry["id"])


class Dictionary(_JsonStore):
    """Personal vocabulary: ``{id, term, aliases}`` entries plus an auto-learn suggestion queue."""

    filename = "dictionary.json"
    label = "your dictionary"

    def _load(self, data) -> None:
        self._entries: list[dict] = []
        self._by_key = None
        self._suggestions: list[dict] = []
        self._dismissed: set[str] = set()
        self._regex = None
        self._replacements: dict[str, str] = {}
        self._fuzzy: dict = {}
        self._exact: dict[str, str] = {}
        data = _layout(data)
        if data is None:
            return
        # one damaged entry costs that entry only, never the whole dictionary (it is counted and reported)
        seen: set[str] = set()
        for raw in _rows(data, "entries"):
            term = self._stored_term(raw)
            if term is None:
                self._skipped += 1
            elif _match_key(term) not in seen:
                seen.add(_match_key(term))
                self._entries.append({"id": str(raw.get("id") or _new_id()), "term": term,
                                      "aliases": self._clean_aliases(term, _alias_list(raw.get("aliases")))})
        for raw in _rows(data, "suggestions"):
            term = self._stored_term(raw)
            if term is None:
                self._skipped += 1
                continue
            self._suggestions.append({
                "id": str(raw.get("id") or _new_id()), "term": term, "aliases": _alias_list(raw.get("aliases")),
                "ts": _num(raw.get("ts")), "source": str(raw.get("source") or "cleanup")})
        self._dismissed = {d.lower() for d in _rows(data, "dismissed") if isinstance(d, str)}

    @staticmethod
    def _stored_term(raw) -> str | None:
        """The term of a stored row, or None when the row is damaged."""
        if isinstance(raw, dict) and isinstance(raw.get("term"), str) and raw["term"].strip():
            return _norm_spaces(raw["term"])
        return None

    def _dump(self):
        return {"version": 1, "entries": self._entries, "suggestions": self._suggestions,
                "dismissed": sorted(self._dismissed)}

    # ---- helpers (call with the lock held)
    _by_key: dict | None = None      # match key -> entry, built on first use, kept up to date (None: rebuild)

    def _keys(self) -> dict:
        if self._by_key is None:
            self._by_key = {}
            for e in self._entries:
                self._by_key.setdefault(_match_key(e["term"]), e)
        return self._by_key

    def _find_term(self, term: str) -> dict | None:
        """The entry for ``term`` as it is matched in dictation: any letter case, a hyphen = a space. A lookup
        in an index, so bulk changes (sync, imports) stay linear."""
        return self._keys().get(_match_key(term))

    def _find_id(self, entry_id: str) -> dict:
        entry = next((e for e in self._entries if e["id"] == entry_id), None)
        if entry is None:
            raise ValueError("That dictionary entry no longer exists.")
        return entry

    @staticmethod
    def _clean_term(term) -> str:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Type the word or name you want VoxFlow to learn.")
        term = _norm_spaces(clean_text(term))
        if not term:
            raise ValueError("Type the word or name you want VoxFlow to learn.")
        if len(term) > _MAX_TERM_LEN:
            raise ValueError(f"Dictionary terms can be at most {_MAX_TERM_LEN} characters long.")
        if ";" in term:                  # a spoken word never has one; a misread spreadsheet row does
            raise ValueError("A dictionary term can't contain “;”. Put each word in its own entry.")
        return term

    @staticmethod
    def _clean_aliases(term: str, aliases) -> list[str]:
        """Usable, distinct sounds-like spellings. One letter is never one: "e" or "k" would turn every lone
        letter of a sentence ("e-mail", "plan B") into the term."""
        out, seen = [], {term.lower()}
        for a in _alias_list(aliases):
            a = _norm_spaces(clean_text(a)).strip(",;")
            if len(a) >= 2 and a.lower() not in seen and len(a) <= _MAX_TERM_LEN:
                seen.add(a.lower())
                out.append(a)
        return out

    def _check_alias_conflicts(self, term: str, aliases: list[str], skip_id: str | None) -> None:
        for e in self._entries:
            if e["id"] == skip_id:
                continue
            taken = {e["term"].lower()} | {a.lower() for a in e["aliases"]}
            for a in aliases:
                if a.lower() in taken:
                    raise ValueError(f"“{a}” already belongs to “{e['term']}”.")
            if term.lower() in {a.lower() for a in e["aliases"]}:
                raise ValueError(f"“{term}” is already a sounds-like spelling of “{e['term']}”.")

    def _changed(self) -> None:
        self._regex = None
        self._save()

    # ---- entries
    def list(self) -> list[dict]:
        with self._lock:
            return sorted(copy.deepcopy(self._entries), key=lambda e: e["term"].lower())

    def terms(self, newest_first: bool = False) -> list[str]:
        """Every term, A to Z; or the most recently added first (what the clean-up's prompt should keep when
        there are more terms than it has room for)."""
        with self._lock:
            if newest_first:
                return [e["term"] for e in reversed(self._entries)]
            return sorted((e["term"] for e in self._entries), key=str.lower)

    def add(self, term: str, aliases: list[str] | None = None) -> dict:
        term = self._clean_term(term)
        with self._lock:
            if self._find_term(term) is not None:
                raise ValueError(f"“{term}” is already in your dictionary.")
            aliases = self._clean_aliases(term, aliases)
            self._check_alias_conflicts(term, aliases, None)
            entry = {"id": _new_id(), "term": term, "aliases": aliases}
            self._entries.append(entry)
            self._keys()[_match_key(term)] = entry
            dropped = self._drop_suggestions_for(term)
            self._changed()
            out = copy.deepcopy(entry)
        self._notify("dictionary")
        if dropped:
            self._notify("suggestions")
        return out

    def update(self, entry_id: str, term: str | None = None, aliases: list[str] | None = None) -> dict:
        with self._lock:
            entry = self._find_id(entry_id)
            new_term = self._clean_term(term) if term is not None else entry["term"]
            other = self._find_term(new_term)
            if other is not None and other is not entry:
                raise ValueError(f"“{new_term}” is already in your dictionary.")
            new_aliases = self._clean_aliases(new_term, entry["aliases"] if aliases is None else aliases)
            self._check_alias_conflicts(new_term, new_aliases, entry_id)
            self._keys().pop(_match_key(entry["term"]), None)
            entry["term"], entry["aliases"] = new_term, new_aliases
            self._keys()[_match_key(new_term)] = entry
            self._changed()
            out = copy.deepcopy(entry)
        self._notify("dictionary")
        return out

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e["id"] != entry_id]
            removed = len(self._entries) != before
            if removed:
                self._by_key = None
                self._changed()
        if removed:
            self._notify("dictionary")
        return removed

    def apply_changes(self, upserts=(), removals=(), conflicts: list | None = None) -> int:
        """Bulk change used by sync and profile import: ``upserts`` are ``{term, aliases}`` rows that replace
        the entry with the same term (any letter case) or are added; ``removals`` are terms to delete.
        Lenient on purpose (invalid rows are skipped, nothing raises) and saved once. Returns #changes.

        With a ``conflicts`` list (imports), a sounds-like spelling that another entry already uses is left out,
        as when it is typed by hand, and ``{"alias", "term", "owner"}`` is appended to the list for the user.
        Sync passes none: there the other device's entry wins as it is."""
        changed = 0
        with self._lock:
            for term in removals:
                entry = self._find_term(_norm_spaces(term)) if isinstance(term, str) else None
                if entry is not None:
                    self._entries.remove(entry)
                    self._keys().pop(_match_key(entry["term"]), None)
                    changed += 1
            taken: dict[str, set[str]] = {}
            alias_of: dict[str, set[str]] = {}
            if conflicts is not None:
                for e in self._entries:
                    _index_entry(e, taken, alias_of)
            for row in upserts:
                try:
                    term = self._clean_term(row.get("term"))
                except (ValueError, AttributeError):
                    continue
                aliases = self._clean_aliases(term, row.get("aliases") or [])
                entry = self._find_term(term)
                if conflicts is not None:
                    own = entry["id"] if entry else None
                    usable = []
                    for a in aliases:
                        others = taken.get(a.lower(), set()) - {own}
                        if others:
                            owner = next(e["term"] for e in self._entries if e["id"] in others)
                            conflicts.append({"alias": a, "term": term, "owner": owner})
                        else:
                            usable.append(a)
                    aliases = usable
                    if entry is not None:
                        _index_entry(entry, taken, alias_of, remove=True)
                if entry is None:
                    entry = {"id": _new_id(), "term": term, "aliases": aliases}
                    self._entries.append(entry)
                    self._keys()[_match_key(term)] = entry
                    self._drop_suggestions_for(term)
                elif entry["term"] != term or entry["aliases"] != aliases:
                    entry["term"], entry["aliases"] = term, aliases      # the same match key: the index holds
                else:
                    _index_entry(entry, taken, alias_of)
                    continue
                if conflicts is not None:
                    _index_entry(entry, taken, alias_of)
                changed += 1
            if changed:
                self._changed()
        if changed:
            self._notify("dictionary")
            self._notify("suggestions")
        return changed

    # ---- applying
    def _compile(self) -> None:
        words = _wordlists()[0] if self._entries else frozenset()   # an empty dictionary loads no word list
        repl: dict[str, str] = {}
        exact: dict[str, str] = {}                    # matched with the term's own capitals only
        for e in self._entries:
            term = e["term"]
            for k in e["aliases"]:                     # the user said so: always, in any letter case
                repl.setdefault(_match_key(k), term)
            # The term itself in another letter case. Not when it is an ordinary word: "Linear", "Hunter",
            # "Lattice" or "Gusto" must not capitalise "linear", "the hunter", "a lattice" or "with gusto";
            # the recogniser already writes a name with its capital.
            if not _all_words(term, words):
                repl.setdefault(_match_key(term), term)
            spoken = _split_camel(term)
            if " " in spoken:                          # "VoxFlow" said as "vox flow"
                if not _all_words(spoken, words):
                    repl.setdefault(_match_key(spoken), term)
                else:                                  # "GitHub": "Git Hub" yes, "git hub" / "One drive" no
                    exact.setdefault(re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[A-Z])(?=[A-Z][a-z])",
                                            " ", term), term)
        self._replacements = repl
        self._exact = exact
        self._fuzzy = _fuzzy_index(self._entries)
        alts = [r"[\s\-]+".join(re.escape(w) for w in k.split(" ")) for k in sorted(repl, key=len, reverse=True)]
        self._regex = re.compile(_NOT_IN_CODE_BEFORE + r"(?:" + "|".join(alts) + r")" + _NOT_IN_CODE_AFTER,
                                 re.I) if alts else False
        alts = [r"[ \-]".join(re.escape(w) for w in k.split(" ")) for k in sorted(exact, key=len, reverse=True)]
        self._exact_regex = re.compile(_NOT_IN_CODE_BEFORE + r"(?:" + "|".join(alts) + r")" + _NOT_IN_CODE_AFTER) \
            if alts else False

    def prepare(self) -> None:
        """Build the matchers now (the word list takes about 30 ms to load) instead of on the first dictation."""
        with self._lock:
            if self._regex is None:
                self._compile()

    def apply(self, text: str) -> str:
        """Write dictionary terms the user's way, whole words only:

        * a "sounds like" alias, in any letter case and with spaces or hyphens between the words, becomes the
          term. This is the only way ordinary words can turn into a term;
        * the term in another letter case ("kubernetes"), or the spoken form of a camel-case name ("vox flow"),
          becomes the term, unless that form is made of ordinary English words ("linear", "git hub"): those
          only match with the term's own capitals ("Git Hub");
        * a close misspelling of a term as the recogniser wrote it ("Kubernetis", "Graphana",
          "Supercaliphragolistus expialidocio") becomes the term too, but never a span of real English words
          ("copper", "red is", "look her up"). That is what makes a plain entry without aliases do something:
          the speech model cannot be told the user's words, so this is the step that corrects them
          (see ``_fuzzy_fix``)."""
        if not text:
            return text
        with self._lock:
            if self._regex is None:
                self._compile()
            rx, repl, fuzzy = self._regex, self._replacements, self._fuzzy
            exact_rx, exact = self._exact_regex, self._exact
        if rx:
            text = rx.sub(lambda m: repl.get(_match_key(m.group(0)), m.group(0)), text)
        if exact_rx:
            text = exact_rx.sub(lambda m: exact.get(m.group(0).replace("-", " "), m.group(0)), text)
        return _fuzzy_fix(text, fuzzy) if fuzzy else text

    # ---- import / export
    @staticmethod
    def _parse_import(path: Path) -> list[tuple[str, list[str]]]:
        text = _read_import_text(path)
        rows: list[tuple[str, list[str]]] = []
        ext = path.suffix.lower()
        if ext == ".json":
            try:
                data = json.loads(text)
            except ValueError:
                raise ValueError(f"{path.name} is not a valid JSON file.") from None
            if isinstance(data, dict):
                data = data.get("entries", [])
            if not isinstance(data, list):
                raise ValueError(f"{path.name} does not contain a list of dictionary entries.")
            for item in data:
                if isinstance(item, str):
                    rows.append((item, []))
                elif isinstance(item, dict) and isinstance(item.get("term"), str):
                    rows.append((item["term"], _alias_list(item.get("aliases"))))
        elif ext == ".csv":
            # Excel writes "CSV" with ";" (or a tab) in many locales (German, French, Dutch...)
            try:
                delimiter = csv.Sniffer().sniff(text[:4096], ",;\t").delimiter
            except csv.Error:
                delimiter = ","
            for i, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter)):
                cells = [c.strip() for c in row if c.strip()]
                if not cells or (i == 0 and cells[0].lower() == "term"):
                    continue
                rows.append((cells[0], cells[1:]))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(_txt_fields(line))
        return rows

    def import_(self, path: str) -> dict:
        """Add entries from a .json / .csv / .txt file; existing terms gain any new aliases.
        Returns ``{"added": new entries, "updated": existing entries that gained sounds-like spellings}``."""
        rows = self._parse_import(Path(path))
        added = 0
        changed: set[str] = set()
        with self._lock:
            # who uses which spelling, built once: checking every alias against every entry was O(n²) under the
            # lock (5,000 entries took 10 s, with the window frozen and dictations waiting for the lock)
            by_term = {_match_key(e["term"]): e for e in self._entries}
            taken: dict[str, set[str]] = {}            # term or alias (lower case) -> ids of the entries using it
            alias_of: dict[str, set[str]] = {}         # alias (lower case) -> ids
            for e in self._entries:
                _index_entry(e, taken, alias_of)
            for term, aliases in rows:
                try:
                    term = self._clean_term(term)
                except ValueError:
                    continue
                existing = by_term.get(_match_key(term))
                target_id = existing["id"] if existing else None
                if alias_of.get(term.lower(), set()) - {target_id}:
                    continue                           # the term is another entry's sounds-like spelling
                merged = self._clean_aliases(term, (existing["aliases"] if existing else []) + list(aliases))
                usable = [a for a in merged if not taken.get(a.lower(), set()) - {target_id}]
                if existing:
                    if usable != existing["aliases"]:
                        _index_entry(existing, taken, alias_of, remove=True)
                        existing["aliases"] = usable
                        _index_entry(existing, taken, alias_of)
                        changed.add(existing["id"])
                    continue
                entry = {"id": _new_id(), "term": term, "aliases": usable}
                self._entries.append(entry)
                by_term[_match_key(term)] = entry
                _index_entry(entry, taken, alias_of)
                self._drop_suggestions_for(term)
                added += 1
            self._by_key = None
            self._changed()
        self._notify("dictionary")
        self._notify("suggestions")
        return {"added": added, "updated": len(changed)}

    def export(self, path: str) -> int:
        """Write all entries to ``path`` (.json, .csv or .txt by extension). Returns the number written."""
        entries = self.list()
        target = Path(path)
        ext = target.suffix.lower()
        if ext == ".csv":
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerow(["term", "aliases"])
            for e in entries:
                writer.writerow([e["term"]] + e["aliases"])
            text = "\ufeff" + buf.getvalue()           # the BOM tells Excel it is UTF-8 ("Köln", not "KÃ¶ln")
        elif ext == ".txt":
            text = "".join(_txt_line(e["term"], e["aliases"]) + "\n" for e in entries)
        else:
            text = json.dumps({"entries": [{"term": e["term"], "aliases": e["aliases"]} for e in entries]},
                              ensure_ascii=False, indent=2)
        try:
            cfgmod._atomic_write(target, text)
        except OSError as exc:
            raise ValueError(f"Couldn't write {target.name}: {exc}") from None
        return len(entries)

    # ---- auto-learn
    def suggestions(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._suggestions)

    def _drop_suggestions_for(self, term: str) -> bool:
        before = len(self._suggestions)
        self._suggestions = [s for s in self._suggestions if s["term"].lower() != term.lower()]
        return len(self._suggestions) != before

    def suggest(self, term: str, aliases: list[str] | None = None, source: str = "cleanup") -> dict | None:
        """Queue a suggestion. Returns it, or None when it is already known, queued or was dismissed."""
        try:
            term = self._clean_term(term)
        except ValueError:
            return None
        with self._lock:
            aliases = self._clean_aliases(term, aliases)
            low = term.lower()
            if low in self._dismissed:
                return None
            existing = self._find_term(term)
            if existing is not None:
                known = {a.lower() for a in existing["aliases"]} | {_split_camel(existing["term"])}
                aliases = [a for a in aliases if a.lower() not in known]
                if not aliases:
                    return None
            queued = next((s for s in self._suggestions if s["term"].lower() == low), None)
            if queued is not None:
                fresh = [a for a in aliases if a.lower() not in {x.lower() for x in queued["aliases"]}]
                if not fresh:
                    return None
                queued["aliases"].extend(fresh)
                self._save()
                out = copy.deepcopy(queued)
            else:
                sug = {"id": _new_id(), "term": term, "aliases": aliases, "ts": time.time(), "source": source}
                self._suggestions.insert(0, sug)
                del self._suggestions[_MAX_SUGGESTIONS:]
                self._save()
                out = copy.deepcopy(sug)
        self._notify("suggestions")
        return out

    def learn(self, raw: str, text: str, limit: int = 3) -> list[dict]:
        """Compare what was heard (``raw``) with what was written (``text``) and queue suggestions for names
        the clean-up had to repair ("get hub" -> "GitHub"). Returns the newly queued suggestions."""
        if not raw or not text:
            return []
        from .cleanup import smart_format

        raw_words = [w.lower() for w in re.findall(r"[A-Za-z0-9+#']+", raw)]
        raw_tokens = set(_TOKEN_RE.findall(raw))
        # what smart formatting writes from the raw words on its own ("three pm" -> "3 PM") is not a name the
        # clean-up had to repair: never suggest it
        raw_tokens |= set(_TOKEN_RE.findall(smart_format(raw)))
        out: list[dict] = []
        seen: set[str] = set()
        for tok in _TOKEN_RE.findall(text):
            if len(out) >= limit:
                break
            if tok in raw_tokens or tok.lower() in seen or not _interesting_casing(tok):
                continue
            seen.add(tok.lower())
            alias = self._heard_as(tok, raw_words)
            if alias is None:
                continue
            sug = self.suggest(tok, [alias] if alias and alias != tok.lower() else [], "cleanup")
            if sug is not None:
                out.append(sug)
        return out

    @staticmethod
    def _heard_as(tok: str, raw_words: list[str]) -> str | None:
        """The run of raw words ``tok`` was written from; '' if it was only re-cased; None if not found."""
        key = re.sub(r"[^a-z0-9]", "", tok.lower())
        if not key:
            return None
        from .cleanup import phonetic_key

        key_sound = phonetic_key(key)
        best, best_ratio = None, 0.0
        for n in (3, 2, 1):                           # longer spans first: they win ties
            for i in range(len(raw_words) - n + 1):
                span = raw_words[i:i + n]
                cand = re.sub(r"[^a-z0-9]", "", "".join(span))
                if not cand:
                    continue
                if cand == key:
                    ratio = 1.0
                else:
                    ratio = max(difflib.SequenceMatcher(None, cand, key).ratio(),
                                difflib.SequenceMatcher(None, phonetic_key(cand), key_sound).ratio())
                    if n == 1 and cand in key:
                        # a fragment of the name ("torch" for PyTorch) is an ordinary word, never an alias
                        continue
                if ratio > best_ratio:
                    best, best_ratio = " ".join(span), ratio
        if best is None or best_ratio < 0.74:
            return None
        return "" if best == tok.lower() else best

    def accept_suggestion(self, suggestion_id: str) -> dict:
        with self._lock:
            sug = next((s for s in self._suggestions if s["id"] == suggestion_id), None)
            if sug is None:
                raise ValueError("That suggestion is no longer available.")
            self._suggestions.remove(sug)
            existing = self._find_term(sug["term"])
            if existing is not None:
                merged = self._clean_aliases(existing["term"], existing["aliases"] + sug["aliases"])
                existing["aliases"] = [a for a in merged if self._alias_free(existing["term"], a, existing["id"])]
                entry = existing
            else:
                aliases = [a for a in self._clean_aliases(sug["term"], sug["aliases"])
                           if self._alias_free(sug["term"], a, None)]
                entry = {"id": _new_id(), "term": sug["term"], "aliases": aliases}
                self._entries.append(entry)
                self._keys()[_match_key(entry["term"])] = entry
            self._changed()
            out = copy.deepcopy(entry)
        self._notify("dictionary")
        self._notify("suggestions")
        return out

    def _alias_free(self, term: str, alias: str, skip_id: str | None) -> bool:
        try:
            self._check_alias_conflicts(term, [alias], skip_id)
            return True
        except ValueError:
            return False

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        with self._lock:
            sug = next((s for s in self._suggestions if s["id"] == suggestion_id), None)
            if sug is None:
                return False
            self._suggestions.remove(sug)
            self._dismissed.add(sug["term"].lower())
            self._save()
        self._notify("suggestions")
        return True


# =========================================================================== snippets
_TRIGGER_WORD_RE = re.compile(r"[^\s,.;:!?\-–—\"“”()\[\]]+")      # "c++" and "#tag" keep their symbols
_NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                 "sixteen seventeen eighteen nineteen").split()
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_WORD_TO_DIGITS = {**{w: str(i) for i, w in enumerate(_NUMBER_WORDS)}, **{w: str(n) for w, n in _TENS.items()}}


def _number_words(n: int) -> list[str]:
    """0..99 as spoken: 21 -> ["twenty", "one"]."""
    if n < 20:
        return [_NUMBER_WORDS[n]]
    tens = next(w for w, v in _TENS.items() if v == n - n % 10)
    return [tens] + ([_NUMBER_WORDS[n % 10]] if n % 10 else [])
_SNIPPET_GAP = r"[\s,.;:!?\-–—]"            # what the recogniser may write between the words of a trigger
# A snippet fires only when it is asked for, never because its phrase happens to be in a sentence ("I will sign
# off on the budget", "send me an email"): when the whole dictation is the trigger, or when a lead word comes
# right before it ("insert my address", "Bitte füge meine Adresse ein"). Lead words of the seven languages with
# full clean-up; separable German / Dutch verbs ("füge ... ein", "voeg ... in") drop their particle as well.
_LEAD_VERBS = (
    "insert", "paste", "type",                                                              # en
    "inserta", "insertar", "inserte", "insertá", "pega", "pegar", "pegue", "escribe", "escribir", "escriba",  # es
    "insère", "insérer", "insérez", "insere", "inserer", "inserez", "colle", "coller", "collez", "tape",
    "taper", "tapez", "écris", "écrire", "écrivez",                                          # fr
    "einfügen", "tippe", "tipp", "schreibe", "schreib", "einsetzen",                         # de
    "inserisci", "inserire", "inserisca", "incolla", "incollare", "incolli", "scrivi", "scrivere", "digita",
    "digitare",                                                                              # it
    "insere", "insira", "inserir", "cola", "cole", "colar", "escreve", "escreva", "escrever", "digite",
    "digitar",                                                                               # pt
    "plak", "plakken", "typ", "typen", "invoegen", "schrijf",                                # nl
)
_SEPARABLE_LEADS = ("füge", "füg", "fuge", "setze", "setz", "voeg")                          # "füge X ein"
_SEPARABLE_PARTICLE = r"(?:[\s,]+(?:ein|in)(?![\w]))?"
_TRAILING_VERBS = ("einfügen", "eintippen", "einsetzen", "invoegen", "plakken", "intypen")    # "meine Adresse einfügen"
_PLEASE = r"(?:please|por\s+favor|s['’]il\s+(?:te|vous)\s+pla[iî]t|bitte|per\s+favore|alsjeblieft|alstublieft)"
_DETERMINER = r"(?:the|a|an|el|la|le|l['’]|den|die|das|il|lo|o|de|het)"
_LEAD = (r"(?:" + _PLEASE + r"[\s,]+)?(?:(?P<sep>" + "|".join(_SEPARABLE_LEADS) + r")|"
         + "|".join(sorted(_LEAD_VERBS, key=len, reverse=True)) + r")"
         r"\s*[,:]?\s+(?:" + _DETERMINER + r"\s+)?(?:snippet\s*[,:]?\s+)?")
# hesitations the recogniser may keep around a trigger said on its own ("Um, so long.")
_FILLER = r"(?:u+m+|u+h+m*|e+r+m*|a+h+|e+h+m*|h+m+|m+h*m+|ähm?|äh|euh|ehm)"
_SNIPPET_BEFORE = r"(?<![\w@/\\])"              # never inside a word, an e-mail address, a path or a URL
_SNIPPET_AFTER = r"(?![\w@/\\])"
_WHOLE_EDGE = r"(?:[\s\"'“”‘’()\[\].,;:!?…\-–—]|" + _FILLER + r"(?![\w]))*"


def _trigger_words(trigger: str) -> list[str]:
    return [w.replace("’", "'") for w in _TRIGGER_WORD_RE.findall(str(trigger).lower())]


def _trigger_tokens(trigger: str) -> list[str]:
    """The trigger's words, with numbers up to 99 as digits: "room twenty-one" -> ["room", "21"]."""
    words, out, i = _trigger_words(trigger), [], 0
    while i < len(words):
        w = words[i]
        unit = words[i + 1] if i + 1 < len(words) else ""
        if w in _TENS and unit in _NUMBER_WORDS[1:10]:
            out.append(str(_TENS[w] + _NUMBER_WORDS.index(unit)))
            i += 2
            continue
        out.append(_WORD_TO_DIGITS.get(w, w))
        i += 1
    return out


def _trigger_key(trigger: str) -> str:
    """How a trigger sounds: lower case, words only, numbers as digits ("Zoom link two" == "zoom-link 2",
    "room twenty one" == "room 21")."""
    return " ".join(_trigger_tokens(trigger))


def _spoken_number(token: str) -> list[str] | None:
    """The spoken words of a number token 0..99 ("21" -> ["twenty", "one"]), else None ("007" is not one)."""
    if token.isascii() and token.isdigit() and str(int(token)) == token and int(token) < 100:
        return _number_words(int(token))
    return None


def _trigger_word_pattern(token: str) -> str:
    spoken = _spoken_number(token)
    if spoken is not None:                   # "21", "twenty one", "twenty-one"
        return f"(?:{re.escape(token)}|{(_SNIPPET_GAP + '*').join(spoken)})"
    # "let's" == "lets" == "let’s", and a hyphen may split a word: "email" == "e-mail"
    return "['’]?".join("-?".join(re.escape(c) for c in part) for part in token.split("'") if part)


def _trigger_pattern(trigger: str) -> str:
    """Regex body for one trigger: its words with any recogniser punctuation between them; two words that make
    four letters or more may also be run together ("sign off" == "signoff", "e mail" == "email")."""
    tokens = _trigger_tokens(trigger)
    if not tokens:
        return ""
    size = lambda t: max(len(t), len(" ".join(_spoken_number(t) or [])))     # noqa: E731 - "2" is said "two"
    out = _trigger_word_pattern(tokens[0])
    for prev, token in zip(tokens, tokens[1:]):
        out += _SNIPPET_GAP + ("*" if size(prev) + size(token) >= 4 else "+") + _trigger_word_pattern(token)
    return out


class Snippets(_JsonStore):
    """Voice shortcuts: saying the trigger phrase types the saved text."""

    filename = "snippets.json"
    label = "your snippets"

    def _load(self, data) -> None:
        self._entries: list[dict] = []
        self._regex = None
        data = _layout(data)
        if data is None:
            return
        seen: set[str] = set()
        for raw in _rows(data, "entries"):
            if not (isinstance(raw, dict) and isinstance(raw.get("trigger"), str) and isinstance(raw.get("text"), str)
                    and self._norm_trigger(raw["trigger"]) and raw["text"]):
                self._skipped += 1                   # damaged: left out and reported, the others still load
                continue
            trigger = self._norm_trigger(raw["trigger"])
            if _trigger_key(trigger) not in seen:
                seen.add(_trigger_key(trigger))
                self._entries.append({"id": str(raw.get("id") or _new_id()), "trigger": trigger, "text": raw["text"]})

    def _dump(self):
        return {"version": 1, "entries": self._entries}

    @staticmethod
    def _norm_trigger(trigger: str) -> str:
        return _norm_spaces(re.sub(r"[.,!?;:]+", " ", trigger))

    def _find_trigger(self, trigger: str) -> dict | None:
        key = _trigger_key(trigger)                  # "sign-off" and "Sign off" are said the same way
        return next((e for e in self._entries if _trigger_key(e["trigger"]) == key), None)

    def _clean(self, trigger, text, skip: dict | None = None) -> tuple[str, str]:
        trigger, text = self._valid(trigger, text)
        other = self._find_trigger(trigger)
        if other is not None and other is not skip:
            raise ValueError(f"You already have a snippet for “{other['trigger']}”.")
        return trigger, text

    def _valid(self, trigger, text) -> tuple[str, str]:
        """The normalised (trigger, text), or ValueError with a message for the user. Duplicates are not checked."""
        if not isinstance(trigger, str) or not _trigger_words(trigger):
            raise ValueError("Type the phrase you will say to insert this snippet.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Type the text this snippet should insert.")
        trigger, text = self._norm_trigger(clean_text(trigger)), clean_text(text)
        if not trigger or not text.strip():
            raise ValueError("Type the phrase you will say to insert this snippet." if not trigger
                             else "Type the text this snippet should insert.")
        if len(trigger) > _MAX_TERM_LEN:
            raise ValueError(f"Trigger phrases can be at most {_MAX_TERM_LEN} characters long.")
        return trigger, text

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(copy.deepcopy(self._entries), key=lambda e: e["trigger"].lower())

    def add(self, trigger: str, text: str) -> dict:
        with self._lock:
            trigger, text = self._clean(trigger, text)
            entry = {"id": _new_id(), "trigger": trigger, "text": text}
            self._entries.append(entry)
            self._regex = None
            self._save()
            out = dict(entry)
        self._notify("snippets")
        return out

    def update(self, entry_id: str, trigger: str | None = None, text: str | None = None) -> dict:
        with self._lock:
            entry = next((e for e in self._entries if e["id"] == entry_id), None)
            if entry is None:
                raise ValueError("That snippet no longer exists.")
            new_trigger, new_text = self._clean(entry["trigger"] if trigger is None else trigger,
                                                entry["text"] if text is None else text, skip=entry)
            entry["trigger"], entry["text"] = new_trigger, new_text
            self._regex = None
            self._save()
            out = dict(entry)
        self._notify("snippets")
        return out

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e["id"] != entry_id]
            removed = len(self._entries) != before
            if removed:
                self._regex = None
                self._save()
        if removed:
            self._notify("snippets")
        return removed

    def apply_changes(self, upserts=(), removals=()) -> int:
        """Bulk change used by sync and profile import: ``upserts`` are ``{trigger, text}`` rows that replace
        the snippet with the same trigger or are added; ``removals`` are triggers to delete. Invalid rows
        are skipped, nothing raises, and the file is saved once. Returns the number of changes."""
        changed = 0
        with self._lock:
            for trigger in removals:
                entry = self._find_trigger(self._norm_trigger(trigger)) if isinstance(trigger, str) else None
                if entry is not None:
                    self._entries.remove(entry)
                    changed += 1
            for row in upserts:
                trigger, text = (row.get("trigger"), row.get("text")) if isinstance(row, dict) else (None, None)
                if not (isinstance(trigger, str) and isinstance(text, str) and text.strip()):
                    continue
                trigger = self._norm_trigger(trigger)
                if not trigger or len(trigger) > _MAX_TERM_LEN:
                    continue
                entry = self._find_trigger(trigger)
                if entry is None:
                    self._entries.append({"id": _new_id(), "trigger": trigger, "text": text})
                elif entry["trigger"] != trigger or entry["text"] != text:
                    entry["trigger"], entry["text"] = trigger, text
                else:
                    continue
                changed += 1
            if changed:
                self._regex = None
                self._save()
        if changed:
            self._notify("snippets")
        return changed

    def _compiled(self):
        """(whole-dictation regex, lead-word regex, trailing-verb regex, texts by group name), built once per
        change of the snippets."""
        with self._lock:
            if self._regex is None:
                texts: dict[str, str] = {}
                alts: list[str] = []
                for i, e in enumerate(sorted(self._entries, key=lambda e: len(_trigger_key(e["trigger"])),
                                             reverse=True)):
                    pattern = _trigger_pattern(e["trigger"])
                    if pattern:
                        texts[f"s{i}"] = e["text"]
                        alts.append(f"(?P<s{i}>{pattern})")
                if alts:
                    body = r"(?:" + "|".join(alts) + r")"
                    self._regex = (
                        re.compile(_WHOLE_EDGE + r"(?:" + _LEAD + r")?" + body + _SEPARABLE_PARTICLE + _WHOLE_EDGE,
                                   re.I),
                        re.compile(_SNIPPET_BEFORE + _LEAD + body + r"(?(sep)" + _SEPARABLE_PARTICLE + r")"
                                   + _SNIPPET_AFTER, re.I),
                        re.compile(_SNIPPET_BEFORE + body + r"[\s,]+(?:" + "|".join(_TRAILING_VERBS) + r")(?!\w)",
                                   re.I),
                        texts)
                else:
                    self._regex = (None, None, None, texts)
            return self._regex

    @staticmethod
    def _snippet_of(m, texts: dict[str, str]) -> str:
        return next(texts[k] for k in texts if m.group(k) is not None)

    def matches(self, text: str) -> bool:
        """Would ``expand`` insert a snippet into ``text``?"""
        whole_rx, lead_rx, trail_rx, _texts = self._compiled()
        return bool(text and whole_rx is not None
                    and (whole_rx.fullmatch(text) or lead_rx.search(text) or trail_rx.search(text)))

    def expand(self, text: str, transform=None) -> str:
        """Insert snippets where they were asked for, the way people say it and the recogniser writes it down:
        any letter case, with the punctuation it adds around and between the words ("Sign-off.", "My, address"),
        joined up or hyphenated ("signoff", "E-mail" for "email"), and with numbers as words or digits ("zoom
        link two" / "zoom link 2"). A snippet is asked for when

        * the whole dictation is its trigger, give or take a lead word, quotes, punctuation and hesitations
          ("My address.", "Um, sign off", "Insert my address."): the result is exactly the snippet text;
        * a lead word comes right before the trigger ("insert", "paste", "type", or the same in Spanish, French,
          German, Italian, Portuguese or Dutch): the lead word goes and the snippet takes the trigger's place
          ("Insert my address, please." -> "<address>, please."; "Bitte füge meine Adresse ein." -> "Bitte
          <address>.").

        A trigger inside an ordinary sentence ("I will sign off on the budget") is left alone. ``transform``
        (the dictionary) is applied to the text around the snippets only: a snippet is always typed exactly as
        it was saved."""
        if not text:
            return text
        whole_rx, lead_rx, trail_rx, texts = self._compiled()
        if whole_rx is None:
            return transform(text) if transform else text
        whole = whole_rx.fullmatch(text)
        if whole:
            return self._snippet_of(whole, texts)
        hits = sorted(list(lead_rx.finditer(text)) + list(trail_rx.finditer(text)), key=lambda m: m.start())
        out: list[str] = []
        last = 0
        for m in hits:
            if m.start() < last:
                continue                                # overlaps the previous one
            out.append(self._around(text[last:m.start()], transform))
            out.append(self._snippet_of(m, texts))
            last = m.end()
            # a snippet asked for as a sentence of its own ("Thanks. Insert sign off.") is the snippet: no full
            # stop after it. One that ends a longer sentence ("Send it to, insert my address.") keeps it.
            if not text[last:].strip(" \t.!?…") and text[:m.start()].strip(" \t\"“'‘")[-1:] in ("", ".", "!", "?",
                                                                                                 "…", "\n"):
                last = len(text)
        if not out:
            return transform(text) if transform else text
        out.append(self._around(text[last:], transform))
        return "".join(out)

    @staticmethod
    def _around(part: str, transform) -> str:
        return transform(part) if transform and part.strip() else part

    def import_(self, path: str) -> dict:
        """Add snippets from a JSON file (``[{trigger, text}]`` or an export). A snippet you already have (said
        the same way) takes the file's text, as a pack import does. Returns ``{"added", "updated"}``."""
        src = Path(path)
        try:
            data = json.loads(_read_import_text(src))
        except ValueError as exc:
            if str(exc).startswith("Couldn't read"):
                raise
            raise ValueError(f"{src.name} is not a valid JSON file.") from None
        if isinstance(data, dict):
            data = data.get("entries", [])
        if not isinstance(data, list):
            raise ValueError(f"{src.name} does not contain a list of snippets.")
        added: set[str] = set()
        updated: set[str] = set()
        with self._lock:
            by_key = {_trigger_key(e["trigger"]): e for e in self._entries}
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    trigger, text = self._valid(item.get("trigger"), item.get("text"))
                except ValueError:
                    continue
                existing = by_key.get(_trigger_key(trigger))
                if existing is None:
                    by_key[_trigger_key(trigger)] = entry = {"id": _new_id(), "trigger": trigger, "text": text}
                    self._entries.append(entry)
                    added.add(entry["id"])
                elif existing["text"] != text and existing["id"] not in added:   # in a file twice: the first wins
                    existing["text"] = text
                    updated.add(existing["id"])
            self._regex = None
            self._save()
        self._notify("snippets")
        return {"added": len(added), "updated": len(updated)}

    def export(self, path: str) -> int:
        entries = self.list()
        target = Path(path)
        payload = {"entries": [{"trigger": e["trigger"], "text": e["text"]} for e in entries]}
        try:
            cfgmod._atomic_write(target, json.dumps(payload, ensure_ascii=False, indent=2))
        except OSError as exc:
            raise ValueError(f"Couldn't write {target.name}: {exc}") from None
        return len(entries)


# =========================================================================== history
_HISTORY_FIELDS = ("id", "ts", "raw", "text", "app_exe", "app_title", "duration_s", "words", "wpm", "mode",
                   "latency_ms")


def make_entry(text: str, raw: str = "", *, app_exe: str = "", app_title: str = "", duration_s: float = 0.0,
               mode: str = "dictate", latency_ms: float = 0.0, ts: float | None = None,
               rules_only: bool = False, words: int | None = None) -> dict:
    """Build a history-shaped entry (also used for transcripts that are not stored).

    ``rules_only``: the language model timed out or was skipped for this dictation, so the text got the rule
    clean-up only. The key is present (and True) only on such entries; read it with ``entry.get("rules_only")``.
    ``words``: how many words were spoken, when that is not the text's own count (a snippet types many words for
    the two that were said; speed and time saved are measured on what was said)."""
    text, raw, app_title = clean_text(text), clean_text(raw or ""), clean_text(app_title or "")
    if words is None:
        words = count_words(text)
    else:                            # never more than was heard or written (a hand-edited file can say anything)
        words = min(max(0, int(words)), max(count_words(text), count_words(raw)))
    duration_s = max(0.0, float(duration_s or 0.0))
    wpm = round(words / (duration_s / 60.0), 1) if duration_s >= 1.0 and mode == "dictate" else 0.0
    entry = {"id": _new_id(), "ts": float(ts if ts is not None else time.time()), "raw": raw, "text": text,
            "app_exe": clean_text(app_exe or "").lower(), "app_title": app_title, "duration_s": round(duration_s, 2),
            "words": words, "wpm": wpm, "mode": mode if mode in ("dictate", "command") else "dictate",
            "latency_ms": int(round(latency_ms or 0))}
    if rules_only is True:
        entry["rules_only"] = True
    return entry


class History(_JsonStore):
    """Transcript history, newest first, with a size cap and optional retention in days."""

    filename = "history.json"
    label = "your history"
    BACKUP_TAGS = ("v1",)
    indent = None                   # compact: the file holds thousands of entries and is rewritten per dictation
    WRITE_BEHIND = True             # see _JsonStore: filing a transcript never waits for the file

    def __init__(self, path: str | Path | None = None, limit: int = 5000, retention_days: int = 0,
                 clock=time.time):
        self.limit = limit
        self.retention_days = retention_days
        self._clock = clock
        super().__init__(path)

    def _load(self, data) -> None:
        self._items: list[dict] = []
        migrated = False
        if isinstance(data, list):               # v1: a flat list of {ts, text, raw, ms}
            source, migrated = data, True
        else:
            data = _layout(data)
            source = [] if data is None else _rows(data, "items")
        for raw in source:
            if not (isinstance(raw, dict) and isinstance(raw.get("text"), str)):
                self._skipped += 1                   # damaged: left out and reported, the others still load
                continue
            entry = make_entry(
                raw["text"], raw.get("raw") if isinstance(raw.get("raw"), str) else "",
                app_exe=str(raw.get("app_exe") or ""), app_title=str(raw.get("app_title") or ""),
                duration_s=_num(raw.get("duration_s")), mode=str(raw.get("mode") or "dictate"),
                latency_ms=_num(raw.get("latency_ms", raw.get("ms"))),
                ts=_sane_ts(raw.get("ts"), self._clock()) or self._clock(), rules_only=raw.get("rules_only") is True,
                words=_stored_words(raw))
            if isinstance(raw.get("id"), str) and raw["id"]:
                entry["id"] = raw["id"]
            self._items.append(entry)
        self._items.sort(key=lambda e: e["ts"], reverse=True)
        pruned = self._prune()
        if migrated and self._items:
            self._backup("v1")
        if migrated or pruned:                   # pruned: the v1 copy goes once the migrated file is written
            self._dirty = True                   # written by _read, after a damaged file's copy is kept

    def _dump(self):
        return {"version": 2, "items": list(self._items)}     # a copy: it may be written outside the lock

    def _prune(self) -> bool:
        before = len(self._items)
        now = self._clock()
        for e in self._items:                       # a clock that ran ahead once must not pin entries for years
            if e["ts"] > now + 86_400.0:
                e["ts"] = now
        if self.retention_days and self.retention_days > 0:
            cutoff = now - self.retention_days * 86400.0
            self._items = [e for e in self._items if e["ts"] >= cutoff]
        del self._items[max(1, int(self.limit)):]
        removed = len(self._items) != before
        if removed:
            self._purge_backups = True           # see _write: deleted text must not survive in a backup copy
        return removed

    # ---- settings
    def configure(self, limit: int | None = None, retention_days: int | None = None) -> None:
        """Change the cap / retention and prune accordingly."""
        with self._lock:
            if limit is not None:
                self.limit = limit
            if retention_days is not None:
                self.retention_days = retention_days
            pruned = self._prune()
            if pruned:
                self._save()
        if pruned:
            self._notify("history")

    def prune(self) -> bool:
        """Apply retention now (called periodically). True if something was removed."""
        with self._lock:
            pruned = self._prune()
            if pruned:
                self._save()
        if pruned:
            self._notify("history")
        return pruned

    # ---- writing
    def add(self, text: str, raw: str = "", ms: float = 0.0, **meta) -> dict:
        """Store a transcript. ``meta``: app_exe, app_title, duration_s, mode, ts (``ms`` = latency)."""
        return self.add_entry(make_entry(text, raw, latency_ms=ms, **meta))

    def add_entry(self, entry: dict) -> dict:
        flagged = entry.get("rules_only") is True
        entry = {k: entry[k] for k in _HISTORY_FIELDS}
        if flagged:
            entry["rules_only"] = True
        with self._lock:
            if not self._items or entry["ts"] >= self._items[0]["ts"]:
                self._items.insert(0, entry)                    # the usual case: newest first, no sort needed
            else:
                self._items.insert(0, entry)
                self._items.sort(key=lambda e: e["ts"], reverse=True)
            self._prune()
            self._save()
        self._notify("history")
        return dict(entry)

    def import_entries(self, items, replace: bool = False) -> int:
        """Add history entries from a profile (entries whose id is already present are skipped); with
        ``replace`` the current history is dropped first. Malformed entries are skipped. Returns #added."""
        added = 0
        with self._lock:
            if replace:                          # the user chose to replace everything: no copy survives it
                self._items = []
                self._purge_backups = self._purge_corrupt = True
            known ={e["id"] for e in self._items}
            now = self._clock()
            for raw in items or []:
                if not (isinstance(raw, dict) and isinstance(raw.get("text"), str) and raw["text"].strip()):
                    continue
                entry = make_entry(
                    raw["text"], raw.get("raw") if isinstance(raw.get("raw"), str) else "",
                    app_exe=str(raw.get("app_exe") or ""), app_title=str(raw.get("app_title") or ""),
                    duration_s=_num(raw.get("duration_s")), mode=str(raw.get("mode") or "dictate"),
                    latency_ms=_num(raw.get("latency_ms")), ts=_sane_ts(raw.get("ts"), now) or now,
                    rules_only=raw.get("rules_only") is True, words=_stored_words(raw))
                if isinstance(raw.get("id"), str) and raw["id"]:
                    entry["id"] = raw["id"]
                if entry["id"] in known:
                    continue
                known.add(entry["id"])
                self._items.append(entry)
                added += 1
            self._items.sort(key=lambda e: e["ts"], reverse=True)
            self._prune()
            self._save()
        self._notify("history")
        return added

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [e for e in self._items if e["id"] != entry_id]
            removed = len(self._items) != before
            if removed:
                self._purge_backups = True
                self._save()
        if removed:
            self._notify("history")
        return removed

    def clear(self) -> None:
        """Delete every transcript: the file is rewritten at once, then the backup copies are deleted too, the
        copies of an unreadable history file included."""
        with self._lock:
            self._items = []
            self._purge_backups = self._purge_corrupt = True
            self._save()
            self.flush()
        self._notify("history")

    # ---- reading
    def list(self, limit: int | None = 50, offset: int = 0) -> list[dict]:
        with self._lock:
            offset = max(0, int(offset))
            end = None if limit is None else offset + max(0, int(limit))
            return [dict(e) for e in self._items[offset:end]]

    def snapshot(self) -> list[dict]:
        """Every entry, newest first."""
        return self.list(limit=None)

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def count_older_than(self, days: int) -> int:
        """How many transcripts a retention of ``days`` days would delete now (0 for "forever")."""
        if not days or days <= 0:
            return 0
        with self._lock:
            cutoff = self._clock() - days * 86400.0
            return sum(1 for e in self._items if e["ts"] < cutoff)

    def get(self, entry_id: str) -> dict | None:
        with self._lock:
            return next((dict(e) for e in self._items if e["id"] == entry_id), None)

    def search(self, q: str, limit: int | None = None) -> list[dict]:
        """Entries containing every word of ``q`` (case-insensitive) in the text, raw text or app."""
        needles = [n for n in (q or "").lower().split() if n]
        limit = None if limit is None else max(0, int(limit))
        found: list[dict] = []
        with self._lock:
            for e in self._items:
                if limit is not None and len(found) >= limit:
                    break                         # enough: a page asks for 201, not for all 100,000
                if needles:
                    hay = " ".join((e["text"], e["raw"], e["app_exe"], e["app_title"])).lower()
                    if not all(n in hay for n in needles):
                        continue
                found.append(dict(e))
        return found

    def export(self, path: str, fmt: str | None = None) -> int:
        """Write the history to ``path`` as txt, json or csv (default: by extension). Returns #entries."""
        target = Path(path)
        fmt = (fmt or target.suffix.lstrip(".") or "txt").lower()
        if fmt not in ("txt", "json", "csv"):
            raise ValueError("History can be exported as .txt, .json or .csv.")
        items = self.snapshot()
        if fmt == "json":
            text = json.dumps(items, ensure_ascii=False, indent=2)
        elif fmt == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerow(["time", "app", "window", "mode", "words", "wpm", "duration_s", "text", "raw"])
            for e in items:
                writer.writerow([_stamp(e["ts"]), _csv_safe(e["app_exe"]), _csv_safe(e["app_title"]), e["mode"],
                                 e["words"], e["wpm"], e["duration_s"], _csv_safe(e["text"]), _csv_safe(e["raw"])])
            text = "\ufeff" + buf.getvalue()           # the BOM tells Excel it is UTF-8 ("Köln", not "KÃ¶ln")
        else:
            text = "".join(f"[{_stamp(e['ts'])}]" + (f" {e['app_exe']}" if e["app_exe"] else "")
                           + f"\n{e['text']}\n\n" for e in items)
        try:
            cfgmod._atomic_write(target, text)
        except OSError as exc:
            raise ValueError(f"Couldn't write {target.name}: {exc}") from None
        return len(items)


def _stored_words(raw: dict) -> int | None:
    """The word count a stored entry carries (the spoken words, see ``make_entry``); None to count the text."""
    words = raw.get("words")
    return words if isinstance(words, int) and not isinstance(words, bool) and words >= 0 else None


def _num(value) -> float:
    """A finite number, else 0.0 (hand-edited files can hold NaN, Infinity or strings)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return 0.0


def _csv_safe(value: str) -> str:
    """Spreadsheets run a cell that starts with = + - @ as a formula: prefix such cells with an apostrophe."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def _sane_ts(value, now: float) -> float | None:
    """A timestamp between 2000 and one day after ``now``, else None (the caller falls back to "now")."""
    ts = _num(value)
    return ts if 946_684_800.0 <= ts <= now + 86_400.0 else None


def _stamp(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# =========================================================================== stats
class Stats(_JsonStore):
    """Per-day usage totals. Independent of the history, so clearing or disabling history keeps them."""

    filename = "stats.json"
    label = "your usage statistics"

    def __init__(self, path: str | Path | None = None, clock=time.time):
        self._clock = clock
        super().__init__(path)

    def _load(self, data) -> None:
        self._days: dict[str, dict] = {}
        data = _layout(data)
        self.is_new = data is None
        if not self.is_new:
            days = data.get("days") or {}
            if not isinstance(days, dict):
                raise ValueError("the per-day totals are not a dictionary")     # -> backed up and reset
            self._days = self.parse_days(days)

    @staticmethod
    def parse_days(raw_days) -> dict[str, dict]:
        """Validated ``{iso date: totals}`` from stored or imported data (anything malformed is dropped)."""
        days: dict[str, dict] = {}
        for day, raw in raw_days.items() if isinstance(raw_days, dict) else ():
            if not (isinstance(day, str) and isinstance(raw, dict)):
                continue
            try:
                _dt.date.fromisoformat(day)
            except ValueError:
                continue
            apps = {}
            for exe, a in (raw.get("apps") or {}).items() if isinstance(raw.get("apps"), dict) else ():
                if isinstance(exe, str) and isinstance(a, dict):
                    apps[exe] = {"words": int(_num(a.get("words"))), "sessions": int(_num(a.get("sessions")))}
            days[day] = {"words": int(_num(raw.get("words"))), "sessions": int(_num(raw.get("sessions"))),
                         "seconds": _num(raw.get("seconds")), "timed_words": int(_num(raw.get("timed_words"))),
                         "apps": apps}
        return days

    def export_days(self) -> dict[str, dict]:
        """A copy of the per-day totals (for a backup)."""
        with self._lock:
            return copy.deepcopy(self._days)

    def import_days(self, days, replace: bool = False) -> int:
        """Take per-day totals from a backup: days that have no totals here are added; with ``replace``
        the imported days become the statistics. Returns the number of days taken."""
        days = self.parse_days(days)
        with self._lock:
            if replace:
                taken, self._days = len(days), days
            else:
                fresh = {day: totals for day, totals in days.items() if day not in self._days}
                taken = len(fresh)
                self._days.update(fresh)
            self.is_new = False
            self._save()
        self._notify("stats")
        return taken

    def _dump(self):
        return {"version": 1, "days": self._days}

    def _today(self) -> _dt.date:
        return _dt.date.fromtimestamp(self._clock())

    def _add(self, words: int, seconds: float, app_exe: str, ts: float) -> None:
        day = self._days.setdefault(_dt.date.fromtimestamp(ts).isoformat(),
                                    {"words": 0, "sessions": 0, "seconds": 0.0, "timed_words": 0, "apps": {}})
        day["words"] += words
        day["sessions"] += 1
        if seconds > 0:
            day["seconds"] = round(day["seconds"] + seconds, 2)
            day["timed_words"] += words
        if app_exe:
            app = day["apps"].setdefault(app_exe.lower(), {"words": 0, "sessions": 0})
            app["words"] += words
            app["sessions"] += 1

    def record(self, words: int, seconds: float = 0.0, app_exe: str = "", ts: float | None = None) -> None:
        """Count one finished dictation of ``words`` words spoken in ``seconds``."""
        if words <= 0:
            return
        with self._lock:
            now = self._clock()
            ts = _sane_ts(ts, now) or now                # a dead-battery clock (1601, year 9999) must not raise
            self._add(int(words), max(0.0, float(seconds)), app_exe, ts)
            self._save()
        self._notify("stats")

    def seed_from_history(self, items: list[dict]) -> None:
        """First run after an upgrade: derive totals from the transcripts that already exist."""
        with self._lock:
            if self._days:
                return
            for e in items:
                if e.get("mode", "dictate") == "dictate" and e.get("words"):
                    self._add(int(e["words"]), float(e.get("duration_s") or 0.0), e.get("app_exe") or "", e["ts"])
            self.is_new = False
            self._save()
        self._notify("stats")

    def summary(self) -> dict:
        with self._lock:
            days = self._days
            total_words = sum(d["words"] for d in days.values())
            sessions = sum(d["sessions"] for d in days.values())
            seconds = sum(d["seconds"] for d in days.values())
            timed_words = sum(d["timed_words"] for d in days.values())
            today = self._today()
            today_words = days.get(today.isoformat(), {}).get("words", 0)
            streak, day = 0, today
            if not days.get(day.isoformat(), {}).get("sessions"):
                day -= _dt.timedelta(days=1)          # today not used yet: the streak is still alive
            while days.get(day.isoformat(), {}).get("sessions"):
                streak += 1
                day -= _dt.timedelta(days=1)
        avg_wpm = round(timed_words / (seconds / 60.0), 1) if seconds >= 1.0 else 0.0
        time_saved = max(0.0, timed_words / TYPING_WPM * 60.0 - seconds)
        return {"total_words": total_words, "sessions": sessions, "seconds_spoken": round(seconds, 1),
                "avg_wpm": avg_wpm, "time_saved_s": round(time_saved, 1), "streak_days": streak,
                "today_words": today_words}

    def daily(self, n_days: int = 14) -> list[dict]:
        """One row per day for the last ``n_days`` days (oldest first), including empty days."""
        n_days = max(1, int(n_days))
        with self._lock:
            today = self._today()
            out = []
            for back in range(n_days - 1, -1, -1):
                key = (today - _dt.timedelta(days=back)).isoformat()
                d = self._days.get(key, {})
                out.append({"date": key, "words": d.get("words", 0), "sessions": d.get("sessions", 0),
                            "seconds": d.get("seconds", 0.0)})
        return out

    def top_apps(self, n: int = 5) -> list[dict]:
        totals: dict[str, dict] = {}
        with self._lock:
            for d in self._days.values():
                for exe, a in d["apps"].items():
                    t = totals.setdefault(exe, {"exe": exe, "words": 0, "sessions": 0})
                    t["words"] += a["words"]
                    t["sessions"] += a["sessions"]
        ranked = sorted(totals.values(), key=lambda t: (-t["words"], t["exe"]))
        return ranked[:max(0, int(n))]

    def reset(self) -> None:
        with self._lock:
            self._days = {}
            self._save()
        self._notify("stats")


# =========================================================================== styles
CATEGORIES = ("personal", "work", "email", "other")
TONES = ("formal", "casual", "very_casual")
CATEGORY_LABELS = {"personal": "Personal messages", "work": "Work messages", "email": "Email", "other": "Other"}
TONE_LABELS = {"formal": "Formal", "casual": "Casual", "very_casual": "Very casual"}
DEFAULT_TONES = {"personal": "casual", "work": "casual", "email": "formal", "other": "formal"}

BUILTIN_APPS: dict[str, str] = {
    # personal messengers
    "whatsapp.exe": "personal", "whatsapp.root.exe": "personal", "telegram.exe": "personal",
    "signal.exe": "personal", "discord.exe": "personal", "messenger.exe": "personal", "wechat.exe": "personal",
    "line.exe": "personal", "viber.exe": "personal", "phoneexperiencehost.exe": "personal",
    # work chat and collaboration
    "slack.exe": "work", "teams.exe": "work", "ms-teams.exe": "work", "zoom.exe": "work", "webex.exe": "work",
    "notion.exe": "work", "linear.exe": "work", "mattermost.exe": "work", "lark.exe": "work",
    "feishu.exe": "work", "asana.exe": "work", "clickup.exe": "work",
    # mail clients
    "outlook.exe": "email", "olk.exe": "email", "hxoutlook.exe": "email", "thunderbird.exe": "email",
    "mailspring.exe": "email", "em client.exe": "email", "mailbird.exe": "email", "proton mail.exe": "email",
    "spark.exe": "email", "superhuman.exe": "email",
}
BROWSERS = frozenset({"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe",
                      "arc.exe", "zen.exe", "librewolf.exe", "chromium.exe", "waterfox.exe"})
# For browsers the window title decides. Web apps put their name in its own segment of the title
# ("Inbox (3) - me@example.com - Gmail - Google Chrome", "general | Acme | Slack"), so whole segments are
# matched; a page that merely mentions a word ("Signal processing - Wikipedia") is not that app.
_SITE_CATEGORIES: dict[str, str] = {
    **dict.fromkeys(("gmail", "outlook", "proton mail", "protonmail", "yahoo mail", "fastmail", "icloud mail",
                     "zoho mail", "hey", "superhuman", "mail", "inbox", "aol mail", "tuta mail"), "email"),
    **dict.fromkeys(("whatsapp", "messenger", "telegram", "discord", "instagram", "google messages",
                     "messages for web", "snapchat", "facebook"), "personal"),
    **dict.fromkeys(("slack", "microsoft teams", "linear", "jira", "confluence", "notion", "asana", "clickup",
                     "trello", "basecamp", "google chat", "mattermost", "zoom", "github", "gitlab"), "work"),
}
_TITLE_SPLIT_RE = re.compile(r"\s+[-|–—·:]\s+")
_SEGMENT_NOISE_RE = re.compile(r"^\(\d+\+?\)\s*|\s*\(\d+\+?\)$|\s+web$", re.I)


def title_category(title: str) -> str | None:
    """Category of the web app shown in a browser window title, or None if it is just some page."""
    for segment in _TITLE_SPLIT_RE.split(title or ""):
        name = segment.strip().lower()
        for _ in range(2):                      # "(2) WhatsApp Web" -> "whatsapp"
            name = _SEGMENT_NOISE_RE.sub("", name).strip()
        if name in _SITE_CATEGORIES:
            return _SITE_CATEGORIES[name]
    return None


def normalize_exe(exe: str) -> str:
    """'C:\\Apps\\Slack.EXE' / 'Slack' -> 'slack.exe'."""
    name = re.split(r"[\\/]", str(exe or "").strip())[-1].lower()
    if name and not name.endswith(".exe"):
        name += ".exe"
    return name


class Styles(_JsonStore):
    """Writing tone per context: each app belongs to a category, each category has a tone."""

    filename = "styles.json"
    label = "your style settings"
    CATEGORIES = CATEGORIES
    TONES = TONES

    def _load(self, data) -> None:
        self._tones: dict[str, str] = dict(DEFAULT_TONES)
        self._apps: dict[str, str] = {}
        data = _layout(data)
        if data is None:
            return
        tones, apps = data.get("tones"), data.get("apps")
        for cat, tone in tones.items() if isinstance(tones, dict) else ():
            if cat in CATEGORIES and tone in TONES:
                self._tones[cat] = tone
        for exe, cat in apps.items() if isinstance(apps, dict) else ():
            if isinstance(exe, str) and normalize_exe(exe) and cat in CATEGORIES:
                self._apps[normalize_exe(exe)] = cat

    def _dump(self):
        return {"version": 1, "tones": self._tones, "apps": self._apps}

    def tones(self) -> dict[str, str]:
        with self._lock:
            return dict(self._tones)

    def set_tone(self, category: str, tone: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"“{category}” is not a style category.")
        if tone not in TONES:
            raise ValueError(f"“{tone}” is not a writing tone.")
        with self._lock:
            self._tones[category] = tone
            self._save()
        self._notify("styles")

    def app_categories(self) -> list[dict]:
        """Built-in and user app mappings, sorted by exe."""
        with self._lock:
            merged = {exe: {"exe": exe, "category": cat, "builtin": True, "overridden": False}
                      for exe, cat in BUILTIN_APPS.items()}
            for exe, cat in self._apps.items():
                row = merged.setdefault(exe, {"exe": exe, "category": cat, "builtin": False, "overridden": False})
                row["overridden"] = row["builtin"]
                row["category"] = cat
        return sorted(merged.values(), key=lambda r: r["exe"])

    def user_apps(self) -> dict[str, str]:
        """The app mappings the user made (``exe -> category``), without the built-in ones."""
        with self._lock:
            return dict(self._apps)

    def apply_changes(self, tones=None, apps=None, removed_apps=()) -> int:
        """Bulk change used by sync and profile import. Unknown categories / tones are skipped, nothing
        raises, and the file is saved once. Returns the number of changes."""
        changed = 0
        with self._lock:
            for cat, tone in (tones or {}).items():
                if cat in CATEGORIES and tone in TONES and self._tones[cat] != tone:
                    self._tones[cat] = tone
                    changed += 1
            for exe in removed_apps:
                if self._apps.pop(normalize_exe(exe), None) is not None:
                    changed += 1
            for exe, cat in (apps or {}).items():
                name = normalize_exe(exe) if isinstance(exe, str) else ""
                if name and name != ".exe" and cat in CATEGORIES and self._apps.get(name) != cat:
                    self._apps[name] = cat
                    changed += 1
            if changed:
                self._save()
        if changed:
            self._notify("styles")
        return changed

    def set_app_category(self, exe: str, category: str) -> None:
        name = normalize_exe(exe)
        if not name or name == ".exe":
            raise ValueError("Type the app's program name, for example slack.exe.")
        if category not in CATEGORIES:
            raise ValueError(f"“{category}” is not a style category.")
        with self._lock:
            self._apps[name] = category
            self._save()
        self._notify("styles")

    def remove_app_category(self, exe: str) -> bool:
        name = normalize_exe(exe)
        with self._lock:
            removed = self._apps.pop(name, None) is not None
            if removed:
                self._save()
        if removed:
            self._notify("styles")
        return removed

    def category_for(self, exe: str, title: str = "") -> str:
        name = normalize_exe(exe)
        with self._lock:
            if name in self._apps:
                return self._apps[name]
        if name in BROWSERS:
            return title_category(title) or "other"
        return BUILTIN_APPS.get(name, "other")

    def tone_for(self, exe: str, title: str = "") -> str:
        cat = self.category_for(exe, title)
        with self._lock:
            return self._tones[cat]


__all__ = ["Dictionary", "Snippets", "History", "Stats", "Styles", "make_entry", "count_words", "normalize_exe",
           "CATEGORIES", "TONES", "CATEGORY_LABELS", "TONE_LABELS", "DEFAULT_TONES", "BUILTIN_APPS", "TYPING_WPM"]
