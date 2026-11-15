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
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))    # tolerate an editor's BOM
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.warning("%s is unreadable (%s); starting empty and keeping a copy", self.path.name, exc)
            self._backup("corrupt")
        try:
            self._load(data)
        except Exception as exc:  # noqa: BLE001 - a malformed file must never stop the app
            log.warning("%s has an unexpected layout (%s); starting empty and keeping a copy", self.path.name, exc)
            self._backup("corrupt")
            self._load(None)

    BACKUP_TAGS: tuple[str, ...] = ("corrupt",)    # every ``_backup`` tag this store can write
    _purge_backups = False      # data was deleted: drop the backup copies after the next successful write

    def _backup_path(self, tag: str) -> Path:
        return self.path.with_name(f"{self.path.stem}.{tag}{self.path.suffix}")

    def _backup(self, tag: str) -> None:
        try:
            if self.path.exists():
                self._backup_path(tag).write_bytes(self.path.read_bytes())
        except OSError:
            pass

    def _drop_backups(self) -> None:
        """Delete the copies ``_backup`` kept (``history.v1.json``, ``history.corrupt.json``): they hold the
        same data as the file, so deleted data must not survive in them. A copy that cannot be deleted is logged
        and is tried again after the next deletion."""
        for tag in self.BACKUP_TAGS:
            try:
                self._backup_path(tag).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete %s: %s", self._backup_path(tag).name, exc.strerror or exc)

    def _load(self, data) -> None:
        raise NotImplementedError

    def _dump(self):
        raise NotImplementedError

    indent: int | None = 1          # small hand-editable files stay readable; the big history file is compact

    COALESCE_S = 0.2                 # saves closer together than this share one write (burst of adds / imports)
    _dirty = False
    _last_write = 0.0
    _timer = None
    _batch = 0

    def _save(self) -> None:
        """Ask for the file to be written. The first change after a quiet moment is written at once (so a single
        edit is on disk when the call returns); changes that follow within ``COALESCE_S`` are folded into one
        trailing write, so a bulk add costs a few writes, not one (with an fsync) per entry. ``flush()`` writes
        whatever is pending; ``with store.batch():`` holds every save back until the block ends."""
        with self._lock:
            if self._batch:
                self._dirty = True
                return
            wait = self._last_write + self.COALESCE_S - time.monotonic()
            if wait <= 0 and self._timer is None:
                self._write()
                return
            self._dirty = True
            if self._timer is None:
                self._timer = threading.Timer(max(wait, 0.01), self._timer_fired)
                self._timer.daemon = True
                self._timer.name = "voxflow-store-write"
                self._timer.start()

    def _timer_fired(self) -> None:
        with self._lock:
            self._timer = None
            if self._dirty and not self._batch:
                self._write()

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
        """Write the file atomically. Never raises; a failure is remembered in ``last_save_error`` and announced
        once to the subscribers as the topic ``"save_failed:<file stem>"`` (and ``"save_ok:<stem>"`` when a later save works again)."""
        self._dirty = False
        self._last_write = time.monotonic()
        try:
            cfgmod._atomic_write(self.path, json.dumps(self._dump(), ensure_ascii=False, indent=self.indent))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:   # lone surrogates raise UnicodeEncodeError
            log.warning("could not save %s: %s", self.path.name, exc)
            first = not self.last_save_error
            self.last_save_error = str(exc) or type(exc).__name__
            if first:
                self._notify("save_failed:" + self.path.stem)
            return                               # a pending backup purge waits for a write that works
        if self._purge_backups:                  # the file without the deleted data is on disk now
            self._purge_backups = False
            self._drop_backups()
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


class Dictionary(_JsonStore):
    """Personal vocabulary: ``{id, term, aliases}`` entries plus an auto-learn suggestion queue."""

    filename = "dictionary.json"

    def _load(self, data) -> None:
        self._entries: list[dict] = []
        self._suggestions: list[dict] = []
        self._dismissed: set[str] = set()
        self._regex = None
        self._replacements: dict[str, str] = {}
        if not isinstance(data, dict):
            return
        for raw in data.get("entries") or []:
            if isinstance(raw, dict) and isinstance(raw.get("term"), str) and raw["term"].strip():
                term = _norm_spaces(raw["term"])
                if self._find_term(term) is None:
                    aliases = [a for a in raw.get("aliases") or [] if isinstance(a, str)]
                    self._entries.append({"id": str(raw.get("id") or _new_id()), "term": term,
                                          "aliases": self._clean_aliases(term, aliases)})
        for raw in data.get("suggestions") or []:
            if isinstance(raw, dict) and isinstance(raw.get("term"), str) and raw["term"].strip():
                self._suggestions.append({
                    "id": str(raw.get("id") or _new_id()), "term": _norm_spaces(raw["term"]),
                    "aliases": [a for a in raw.get("aliases") or [] if isinstance(a, str)],
                    "ts": float(raw.get("ts") or 0.0), "source": str(raw.get("source") or "cleanup")})
        self._dismissed = {d.lower() for d in data.get("dismissed") or [] if isinstance(d, str)}

    def _dump(self):
        return {"version": 1, "entries": self._entries, "suggestions": self._suggestions,
                "dismissed": sorted(self._dismissed)}

    # ---- helpers (call with the lock held)
    def _find_term(self, term: str) -> dict | None:
        low = term.lower()
        return next((e for e in self._entries if e["term"].lower() == low), None)

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
        return term

    @staticmethod
    def _clean_aliases(term: str, aliases) -> list[str]:
        out, seen = [], {term.lower()}
        for a in aliases or []:
            if not isinstance(a, str):
                continue
            a = _norm_spaces(clean_text(a)).strip(",;")
            if a and a.lower() not in seen and len(a) <= _MAX_TERM_LEN:
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

    def terms(self) -> list[str]:
        with self._lock:
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
            entry["term"], entry["aliases"] = new_term, new_aliases
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
                self._changed()
        if removed:
            self._notify("dictionary")
        return removed

    def apply_changes(self, upserts=(), removals=()) -> int:
        """Bulk change used by sync and profile import: ``upserts`` are ``{term, aliases}`` rows that replace
        the entry with the same term (any letter case) or are added; ``removals`` are terms to delete.
        Lenient on purpose (invalid rows are skipped, nothing raises) and saved once. Returns #changes."""
        changed = 0
        with self._lock:
            for term in removals:
                entry = self._find_term(_norm_spaces(term)) if isinstance(term, str) else None
                if entry is not None:
                    self._entries.remove(entry)
                    changed += 1
            for row in upserts:
                try:
                    term = self._clean_term(row.get("term"))
                except (ValueError, AttributeError):
                    continue
                aliases = self._clean_aliases(term, row.get("aliases") or [])
                entry = self._find_term(term)
                if entry is None:
                    self._entries.append({"id": _new_id(), "term": term, "aliases": aliases})
                    self._drop_suggestions_for(term)
                elif entry["term"] != term or entry["aliases"] != aliases:
                    entry["term"], entry["aliases"] = term, aliases
                else:
                    continue
                changed += 1
            if changed:
                self._changed()
        if changed:
            self._notify("dictionary")
            self._notify("suggestions")
        return changed

    # ---- applying
    def _compile(self) -> None:
        repl: dict[str, str] = {}
        for e in self._entries:
            keys = [e["term"]] + e["aliases"]
            spoken = _split_camel(e["term"])
            if " " in spoken:
                keys.append(spoken)
            for k in keys:
                repl.setdefault(_match_key(k), e["term"])
        self._replacements = repl
        if not repl:
            self._regex = False
            return
        alts = [r"[\s\-]+".join(re.escape(w) for w in k.split(" "))
                for k in sorted(repl, key=len, reverse=True)]
        self._regex = re.compile(r"(?<![\w])(?:" + "|".join(alts) + r")(?![\w])", re.I)

    def apply(self, text: str) -> str:
        """Replace sounds-like spellings (and wrongly cased terms) with the dictionary term, whole words only."""
        if not text:
            return text
        with self._lock:
            if self._regex is None:
                self._compile()
            rx, repl = self._regex, self._replacements
        if not rx:
            return text

        def sub(m: re.Match) -> str:
            return repl.get(_match_key(m.group(0)), m.group(0))

        return rx.sub(sub, text)

    # ---- import / export
    @staticmethod
    def _parse_import(path: Path) -> list[tuple[str, list[str]]]:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Couldn't read {path.name}: {exc}") from None
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
                    rows.append((item["term"], [a for a in item.get("aliases") or [] if isinstance(a, str)]))
        elif ext == ".csv":
            for i, row in enumerate(csv.reader(io.StringIO(text))):
                cells = [c.strip() for c in row if c.strip()]
                if not cells or (i == 0 and cells[0].lower() == "term"):
                    continue
                rows.append((cells[0], cells[1:]))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                term, _, rest = line.partition(":")
                rows.append((term, [a for a in rest.split(",") if a.strip()]))
        return rows

    def import_(self, path: str) -> int:
        """Add entries from a .json / .csv / .txt file; existing terms gain any new aliases. Returns #added."""
        rows = self._parse_import(Path(path))
        added = 0
        with self._lock:
            for term, aliases in rows:
                try:
                    term = self._clean_term(term)
                except ValueError:
                    continue
                existing = self._find_term(term)
                target_id = existing["id"] if existing else None
                merged = self._clean_aliases(term, (existing["aliases"] if existing else []) + list(aliases))
                usable = []
                for a in merged:
                    try:
                        self._check_alias_conflicts(term, [a], target_id)
                        usable.append(a)
                    except ValueError:
                        continue
                if existing:
                    existing["aliases"] = usable
                    continue
                try:
                    self._check_alias_conflicts(term, [], None)
                except ValueError:
                    continue
                self._entries.append({"id": _new_id(), "term": term, "aliases": usable})
                self._drop_suggestions_for(term)
                added += 1
            self._changed()
        self._notify("dictionary")
        self._notify("suggestions")
        return added

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
            text = buf.getvalue()
        elif ext == ".txt":
            text = "".join(e["term"] + (": " + ", ".join(e["aliases"]) if e["aliases"] else "") + "\n"
                           for e in entries)
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
        raw_words = [w.lower() for w in re.findall(r"[A-Za-z0-9+#']+", raw)]
        raw_tokens = set(_TOKEN_RE.findall(raw))
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
class Snippets(_JsonStore):
    """Voice shortcuts: saying the trigger phrase types the saved text."""

    filename = "snippets.json"

    def _load(self, data) -> None:
        self._entries: list[dict] = []
        self._regex = None
        if not isinstance(data, dict):
            return
        for raw in data.get("entries") or []:
            if not (isinstance(raw, dict) and isinstance(raw.get("trigger"), str) and isinstance(raw.get("text"), str)):
                continue
            trigger = self._norm_trigger(raw["trigger"])
            if trigger and raw["text"] and self._find_trigger(trigger) is None:
                self._entries.append({"id": str(raw.get("id") or _new_id()), "trigger": trigger, "text": raw["text"]})

    def _dump(self):
        return {"version": 1, "entries": self._entries}

    @staticmethod
    def _norm_trigger(trigger: str) -> str:
        return _norm_spaces(re.sub(r"[.,!?;:]+", " ", trigger))

    def _find_trigger(self, trigger: str) -> dict | None:
        low = trigger.lower()
        return next((e for e in self._entries if e["trigger"].lower() == low), None)

    def _clean(self, trigger, text, skip: dict | None = None) -> tuple[str, str]:
        if not isinstance(trigger, str) or not self._norm_trigger(trigger):
            raise ValueError("Type the phrase you will say to insert this snippet.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Type the text this snippet should insert.")
        trigger, text = self._norm_trigger(clean_text(trigger)), clean_text(text)
        if not trigger or not text.strip():
            raise ValueError("Type the phrase you will say to insert this snippet." if not trigger
                             else "Type the text this snippet should insert.")
        if len(trigger) > _MAX_TERM_LEN:
            raise ValueError(f"Trigger phrases can be at most {_MAX_TERM_LEN} characters long.")
        other = self._find_trigger(trigger)
        if other is not None and other is not skip:
            raise ValueError(f"You already have a snippet for “{other['trigger']}”.")
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

    def expand(self, text: str) -> str:
        """Replace spoken trigger phrases with their snippet text.

        A dictation that is nothing but the trigger becomes exactly the snippet (no added full stop)."""
        if not text:
            return text
        with self._lock:
            if self._regex is None:
                table = {e["trigger"].lower(): e["text"] for e in self._entries}
                if table:
                    alts = [r"[\s,]+".join(re.escape(w) for w in k.split(" "))
                            for k in sorted(table, key=len, reverse=True)]
                    self._regex = (re.compile(r"(?<![\w])(?:" + "|".join(alts) + r")(?![\w])", re.I), table)
                else:
                    self._regex = (None, table)
            rx, table = self._regex
        if rx is None:
            return text
        whole = rx.fullmatch(text.strip().rstrip(".!?"))
        if whole:
            return table.get(re.sub(r"[\s,]+", " ", whole.group(0)).lower(), text)
        return rx.sub(lambda m: table.get(re.sub(r"[\s,]+", " ", m.group(0)).lower(), m.group(0)), text)

    def import_(self, path: str) -> int:
        """Add snippets from a JSON file (``[{trigger, text}]`` or an export). Returns the number added."""
        src = Path(path)
        try:
            data = json.loads(src.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Couldn't read {src.name}: {exc}") from None
        except ValueError:
            raise ValueError(f"{src.name} is not a valid JSON file.") from None
        if isinstance(data, dict):
            data = data.get("entries", [])
        if not isinstance(data, list):
            raise ValueError(f"{src.name} does not contain a list of snippets.")
        added = 0
        with self._lock:
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    trigger, text = self._clean(item.get("trigger"), item.get("text"))
                except ValueError:
                    continue
                self._entries.append({"id": _new_id(), "trigger": trigger, "text": text})
                added += 1
            self._regex = None
            self._save()
        self._notify("snippets")
        return added

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
               rules_only: bool = False) -> dict:
    """Build a history-shaped entry (also used for transcripts that are not stored).

    ``rules_only``: the language model timed out or was skipped for this dictation, so the text got the rule
    clean-up only. The key is present (and True) only on such entries; read it with ``entry.get("rules_only")``."""
    text, raw, app_title = clean_text(text), clean_text(raw or ""), clean_text(app_title or "")
    words = count_words(text)
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
    BACKUP_TAGS = ("corrupt", "v1")
    indent = None                   # compact: the file holds thousands of entries and is rewritten per dictation

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
        elif isinstance(data, dict):
            source = data.get("items") or []
        else:
            source = []
        for raw in source:
            if not (isinstance(raw, dict) and isinstance(raw.get("text"), str)):
                continue
            entry = make_entry(
                raw["text"], raw.get("raw") if isinstance(raw.get("raw"), str) else "",
                app_exe=str(raw.get("app_exe") or ""), app_title=str(raw.get("app_title") or ""),
                duration_s=_num(raw.get("duration_s")), mode=str(raw.get("mode") or "dictate"),
                latency_ms=_num(raw.get("latency_ms", raw.get("ms"))),
                ts=_sane_ts(raw.get("ts"), self._clock()) or self._clock(), rules_only=raw.get("rules_only") is True)
            if isinstance(raw.get("id"), str) and raw["id"]:
                entry["id"] = raw["id"]
            self._items.append(entry)
        self._items.sort(key=lambda e: e["ts"], reverse=True)
        pruned = self._prune()
        if migrated and self._items:
            self._backup("v1")
        if migrated or pruned:                   # pruned: the v1 copy goes once the migrated file is written
            self._save()

    def _dump(self):
        return {"version": 2, "items": self._items}

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
            if replace and self._items:
                self._items = []
                self._purge_backups = True
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
                    rules_only=raw.get("rules_only") is True)
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
        """Delete every transcript: the file is rewritten at once, then the backup copies are deleted too."""
        with self._lock:
            self._items = []
            self._purge_backups = True
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

    def get(self, entry_id: str) -> dict | None:
        with self._lock:
            return next((dict(e) for e in self._items if e["id"] == entry_id), None)

    def search(self, q: str, limit: int | None = None) -> list[dict]:
        """Entries containing every word of ``q`` (case-insensitive) in the text, raw text or app."""
        needles = [n for n in (q or "").lower().split() if n]
        with self._lock:
            if not needles:
                found = [dict(e) for e in self._items]
            else:
                found = []
                for e in self._items:
                    hay = " ".join((e["text"], e["raw"], e["app_exe"], e["app_title"])).lower()
                    if all(n in hay for n in needles):
                        found.append(dict(e))
        return found if limit is None else found[:max(0, int(limit))]

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
            text = buf.getvalue()
        else:
            text = "".join(f"[{_stamp(e['ts'])}]" + (f" {e['app_exe']}" if e["app_exe"] else "")
                           + f"\n{e['text']}\n\n" for e in items)
        try:
            cfgmod._atomic_write(target, text)
        except OSError as exc:
            raise ValueError(f"Couldn't write {target.name}: {exc}") from None
        return len(items)


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

    def __init__(self, path: str | Path | None = None, clock=time.time):
        self._clock = clock
        super().__init__(path)

    def _load(self, data) -> None:
        self._days: dict[str, dict] = {}
        self.is_new = not isinstance(data, dict)
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
    CATEGORIES = CATEGORIES
    TONES = TONES

    def _load(self, data) -> None:
        self._tones: dict[str, str] = dict(DEFAULT_TONES)
        self._apps: dict[str, str] = {}
        if not isinstance(data, dict):
            return
        for cat, tone in (data.get("tones") or {}).items():
            if cat in CATEGORIES and tone in TONES:
                self._tones[cat] = tone
        for exe, cat in (data.get("apps") or {}).items():
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
