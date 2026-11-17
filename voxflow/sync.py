"""Local sync and sharing: profile bundles, shareable packs and a sync folder. No server, no account.

* A **backup** (``.voxbackup``, JSON) is the whole profile in one file: settings, dictionary, snippets and
  styles, optionally the history with its statistics. It can be imported again (merge or replace).
* A **pack** (``.voxpack``, JSON) is a named list of dictionary terms or snippets to hand to someone else.
* The **sync folder** is any folder the user picks (OneDrive, Syncthing, a network share...). Every device
  writes one file of its own there and reads the files of the others, so two devices never write the same
  file and the folder's own sync tool has nothing to conflict about. Entries are merged one by one: the
  last writer wins per entry, deletions travel as tombstones, and a folder that is offline only delays things.

Nothing here talks to the network: files are the only transport. The history is never synced.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import re
import threading
import time
import uuid
from pathlib import Path

from . import __version__, config as cfgmod, store, system

log = logging.getLogger("voxflow.sync")

PROFILE_FORMAT, PACK_FORMAT, SYNC_FORMAT = "voxflow-profile", "voxflow-pack", "voxflow-sync"
VERSION = 1
PROFILE_EXT, PACK_EXT = ".voxbackup", ".voxpack"
KINDS = ("settings", "dictionary", "snippets", "styles")
PARTS = KINDS + ("stats", "history")
PACK_KINDS = ("dictionary", "snippets")
# Settings that describe this computer rather than the person: never exported, never synced.
LOCAL_SETTINGS = frozenset({
    "config_version", "mic_device", "asr_model", "asr_device", "cleanup_model", "cleanup_device",
    "cleanup_timeout_s", "command_timeout_s", "restore_delay_s", "launch_at_login", "start_minimized",
    "overlay_position", "ui_state", "onboarding_done", "update_mode", "update_install_when", "update_time"})
# Interface state: never written to a backup either.
_NEVER_EXPORTED = frozenset({"config_version", "ui_state", "overlay_position", "onboarding_done"})
_MAX_FILE_BYTES = 50 * 1024 * 1024               # packs and sync files
# A backup can hold the whole history: 100,000 transcripts (the largest history_limit) make 60-80 MB, more with
# long dictations. Exporting must never write a file that importing then refuses.
_MAX_BACKUP_BYTES = 1024 * 1024 * 1024
_DELETED = object()

BASELINE = 1.0                       # timestamp of entries a device had before it joined a sync folder
TOMBSTONE_DAYS = 180                 # how long a deletion is remembered
_CLOCK_SKEW_S = 86400.0              # records stamped further in the future than this are ignored
_SYNC_FILE_RE = re.compile(r"^voxflow-sync-.+\.json$", re.I)


def portable_settings() -> tuple[str, ...]:
    """The setting keys that travel in profiles and through the sync folder."""
    return tuple(k for k in cfgmod.defaults() if k not in LOCAL_SETTINGS)


class DataSet:
    """What profiles, packs and sync read and write: the three stores, the history and the settings.

    ``get_settings()`` returns the current value of every setting; ``set_settings(values)`` applies a
    ``{key: value}`` batch and returns the keys it could not apply."""

    def __init__(self, dictionary, snippets, styles, history, get_settings, set_settings, stats=None):
        self.dictionary, self.snippets, self.styles, self.history = dictionary, snippets, styles, history
        self.get_settings, self.set_settings, self.stats = get_settings, set_settings, stats


# =========================================================================== canonical state
def snapshot(ds: DataSet, every_setting: bool = False) -> dict:
    """``{kind: {key: value}}`` for everything portable. Keys identify an entry on every device.
    ``every_setting`` adds the settings that belong to this computer (for a backup)."""
    settings = ds.get_settings()
    keys = [k for k in settings if k not in _NEVER_EXPORTED] if every_setting else portable_settings()
    styles: dict = {f"tone:{cat}": tone for cat, tone in ds.styles.tones().items()}
    styles.update({f"app:{exe}": cat for exe, cat in ds.styles.user_apps().items()})
    return {
        "settings": {k: settings[k] for k in keys if k in settings},
        # keyed the way an entry is said, as the stores match them: "Wi-Fi" == "wi fi", "zoom link two" ==
        # "Zoom link 2", so a backup or pack never adds a second copy of something that is already there
        "dictionary": {_dict_key(e["term"]): {"term": e["term"], "aliases": list(e["aliases"])}
                       for e in ds.dictionary.list()},
        "snippets": {_snippet_key(e["trigger"]): {"trigger": e["trigger"], "text": e["text"]}
                     for e in ds.snippets.list()},
        "styles": styles,
    }


def _dict_key(term: str) -> str:
    return store._match_key(term)


def _snippet_key(trigger: str) -> str:
    return store._trigger_key(trigger)


def _norm(kind: str, key, value, every_setting: bool = False):
    """Validate one foreign entry. Returns ``(key, value)`` in canonical form, or None if it is unusable."""
    try:
        if kind == "settings":
            allowed = (key in cfgmod.defaults() and key not in _NEVER_EXPORTED) if every_setting else (
                key in portable_settings())
            if not allowed:
                return None
            return key, cfgmod.Config.validate(key, value)
        if kind == "dictionary":
            term = store.Dictionary._clean_term(value.get("term"))
            aliases = store.Dictionary._clean_aliases(term, value.get("aliases"))   # "kube, cube" is two
            return _dict_key(term), {"term": term, "aliases": aliases}
        if kind == "snippets":
            trigger, text = value.get("trigger"), value.get("text")
            if not (isinstance(trigger, str) and isinstance(text, str) and text.strip()):
                return None
            trigger = store.Snippets._norm_trigger(trigger)
            if not _snippet_key(trigger) or len(trigger) > store._MAX_TERM_LEN:
                return None
            return _snippet_key(trigger), {"trigger": trigger, "text": text}
        if kind == "styles" and isinstance(key, str):
            if key.startswith("tone:") and key[5:] in store.CATEGORIES and value in store.TONES:
                return key, value
            if key.startswith("app:") and value in store.CATEGORIES:
                exe = store.normalize_exe(key[4:])
                return (f"app:{exe}", value) if exe and exe != ".exe" else None
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None
    return None


def _deletable(kind: str, key: str) -> bool:
    return kind in ("dictionary", "snippets") or (kind == "styles" and key.startswith("app:"))


def _apply(ds: DataSet, changes: dict, conflicts: list | None = None) -> list[str]:
    """Apply ``{kind: {key: value | _DELETED}}``. Returns the setting keys that could not be applied.
    ``conflicts``: see ``Dictionary.apply_changes`` (imports pass a list, sync does not)."""
    for kind, st in (("dictionary", ds.dictionary), ("snippets", ds.snippets)):
        part = changes.get(kind) or {}
        if part:
            upserts = [v for v in part.values() if v is not _DELETED]
            removals = [k for k, v in part.items() if v is _DELETED]
            if kind == "dictionary" and conflicts is not None:
                st.apply_changes(upserts, removals, conflicts=conflicts)
            else:
                st.apply_changes(upserts, removals)
    part = changes.get("styles") or {}
    if part:
        ds.styles.apply_changes(
            tones={k[5:]: v for k, v in part.items() if k.startswith("tone:") and v is not _DELETED},
            apps={k[4:]: v for k, v in part.items() if k.startswith("app:") and v is not _DELETED},
            removed_apps=[k[4:] for k, v in part.items() if k.startswith("app:") and v is _DELETED])
    part = {k: v for k, v in (changes.get("settings") or {}).items() if v is not _DELETED}
    return list(ds.set_settings(part)) if part else []


def _diff(current: dict, target: dict) -> tuple[dict, dict]:
    """Changes that turn ``current`` into ``target`` plus their counts."""
    changes, counts = {}, {"added": 0, "updated": 0, "removed": 0}
    for key, value in target.items():
        if key not in current:
            changes[key] = value
            counts["added"] += 1
        elif current[key] != value:
            changes[key] = value
            counts["updated"] += 1
    for key in current:
        if key not in target:
            changes[key] = _DELETED
            counts["removed"] += 1
    return changes, counts


# =========================================================================== files
def _read_json(path: Path, what: str, limit: int | None = None):
    try:
        if path.stat().st_size > (_MAX_FILE_BYTES if limit is None else limit):
            raise ValueError(f"{path.name} is too large to be a VoxFlow {what}.")
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ValueError(f"{path.name} was not found.") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Couldn't read {path.name}: {exc}") from None
    try:
        return json.loads(text)
    except ValueError:
        raise ValueError(f"{path.name} is not a VoxFlow {what} (it is not valid JSON).") from None


def _check_version(data: dict, path: Path, what: str) -> None:
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{path.name} is not a valid VoxFlow {what} (it has no version).")
    if version > VERSION:
        raise ValueError(f"{path.name} was made by a newer version of VoxFlow. Update VoxFlow to open it.")


def _write(path: Path, payload: dict) -> None:
    try:
        cfgmod._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=1))
    except OSError as exc:
        raise ValueError(f"Couldn't write {path.name}: {exc}") from None


def device_name() -> str:
    return (platform.node() or "This computer").strip()[:60] or "This computer"


def _text(value, limit: int = 120) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _norm_list(kind: str, rows) -> tuple[dict, int]:
    """Canonical ``{key: value}`` of a list of pack / profile rows, plus how many rows were unusable."""
    out, skipped = {}, 0
    for row in rows if isinstance(rows, list) else []:
        if kind == "dictionary" and isinstance(row, str):
            row = {"term": row}
        got = _norm(kind, None, row) if isinstance(row, dict) else None
        if got is None:
            skipped += 1
        else:
            out.setdefault(got[0], got[1])
    return out, skipped


# =========================================================================== backups (whole profile)
def build_profile(ds: DataSet, include_history: bool = True, device_id: str = "", clock=time.time) -> dict:
    snap = snapshot(ds, every_setting=True)
    profile = {
        "format": PROFILE_FORMAT, "version": VERSION, "app_version": __version__, "created": float(clock()),
        "device": device_name(), "device_id": device_id,
        "settings": snap["settings"],
        "dictionary": sorted(snap["dictionary"].values(), key=lambda e: e["term"].lower()),
        "snippets": sorted(snap["snippets"].values(), key=lambda e: e["trigger"].lower()),
        "styles": {"tones": ds.styles.tones(), "apps": ds.styles.user_apps()},
    }
    if include_history:
        profile["history"] = ds.history.snapshot()
        if ds.stats is not None:
            profile["stats"] = ds.stats.export_days()
    return profile


def _profile_counts(data: dict, history, stats) -> dict:
    return {"settings": len(data.get("settings", {})), "dictionary": len(data.get("dictionary", {})),
            "snippets": len(data.get("snippets", {})),
            "styles": sum(1 for k in data.get("styles", {}) if k.startswith("app:")),
            "stats": len(stats) if stats is not None else 0, "history": len(history) if history is not None else 0}


def export_profile(ds: DataSet, path: str, include_history: bool = True, device_id: str = "") -> dict:
    """Write a backup to ``path``. Returns ``{path, counts, includes}``: how many entries of each part went
    in (``styles`` counts the user's app rules, ``stats`` the days) and which parts the file contains."""
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(PROFILE_EXT)
    profile = build_profile(ds, include_history, device_id)
    _write(target, profile)
    if target.stat().st_size > _MAX_BACKUP_BYTES:       # never hand the user a backup that cannot be restored
        target.unlink(missing_ok=True)
        raise ValueError(f"The backup would be larger than {_MAX_BACKUP_BYTES // 1024 ** 3} GB, too large to "
                         "import again. Leave out the history, or lower the history size limit.")
    counts = {"settings": len(profile["settings"]), "dictionary": len(profile["dictionary"]),
              "snippets": len(profile["snippets"]), "styles": len(profile["styles"]["apps"]),
              "stats": len(profile.get("stats", {})), "history": len(profile.get("history", []))}
    return {"path": str(target), "counts": counts, "includes": [p for p in PARTS if p in profile]}


def read_profile(path: str) -> dict:
    """Load and validate a backup. Raises ValueError (human message) when the file is not one.

    Returns ``{"meta", "data": {kind: {key: value}}, "history": list | None, "stats": dict | None,
    "skipped": int}`` with only the sections the file contains; unusable entries are dropped and counted."""
    src = Path(path)
    data = _read_json(src, "backup", _MAX_BACKUP_BYTES)
    if not isinstance(data, dict) or data.get("format") != PROFILE_FORMAT:
        if isinstance(data, dict) and data.get("format") == PACK_FORMAT:
            raise ValueError(f"{src.name} is a shared pack, not a backup. Import it on the Dictionary or Snippets page.")
        raise ValueError(f"{src.name} is not a VoxFlow backup.")
    _check_version(data, src, "backup")
    created = data.get("created")
    out: dict = {"data": {}, "history": None, "stats": None, "skipped": 0,
                 "meta": {"created": float(created) if isinstance(created, (int, float)) and not isinstance(
                              created, bool) and math.isfinite(created) else None,
                          "device": _text(data.get("device"), 60), "device_id": _text(data.get("device_id"), 64),
                          "app_version": _text(data.get("app_version"), 20), "version": data["version"]}}
    if isinstance(data.get("settings"), dict):
        part = {}
        for key, value in data["settings"].items():
            got = _norm("settings", key, value, every_setting=True)
            if got is None:
                out["skipped"] += 1
            else:
                part[got[0]] = got[1]
        out["data"]["settings"] = part
    for kind in PACK_KINDS:
        if isinstance(data.get(kind), list):
            out["data"][kind], skipped = _norm_list(kind, data[kind])
            out["skipped"] += skipped
    styles = data.get("styles")
    if isinstance(styles, dict):
        part = {}
        rows = [(f"tone:{k}", v) for k, v in styles["tones"].items()] if isinstance(styles.get("tones"), dict) else []
        rows += [(f"app:{k}", v) for k, v in styles["apps"].items()] if isinstance(styles.get("apps"), dict) else []
        for key, value in rows:
            got = _norm("styles", key, value)
            if got is None:
                out["skipped"] += 1
            else:
                part[got[0]] = got[1]
        out["data"]["styles"] = part
    if isinstance(data.get("history"), list):
        out["history"] = [e for e in data["history"] if isinstance(e, dict) and isinstance(e.get("text"), str)]
        out["skipped"] += len(data["history"]) - len(out["history"])
    if isinstance(data.get("stats"), dict):
        out["stats"] = store.Stats.parse_days(data["stats"])
    if not out["data"] and out["history"] is None and out["stats"] is None:
        raise ValueError(f"{src.name} is a VoxFlow backup, but it is empty.")
    return out


def describe_profile(path: str, device_id: str = "") -> dict:
    """What a backup file contains, for a confirmation dialog (nothing is changed):
    ``{created, version, device, same_device, counts, includes, skipped}``."""
    prof = read_profile(path)
    data, meta = prof["data"], prof["meta"]
    includes = [p for p in PARTS if p in data or (p == "history" and prof["history"] is not None)
                or (p == "stats" and prof["stats"] is not None)]
    return {"created": meta["created"], "version": meta["app_version"], "device": meta["device"],
            "same_device": bool(device_id) and meta["device_id"] == device_id,
            "counts": _profile_counts(data, prof["history"], prof["stats"]), "includes": includes,
            "skipped": prof["skipped"]}


def _merge_target(kind: str, current: dict, incoming: dict, mode: str, overwrite: bool = False) -> dict:
    """The state ``kind`` should have after the import.

    replace          exactly the incoming entries (settings: incoming values; tones always exist)
    merge            only additions: entries that are missing here; a dictionary term both sides have
                     gains the incoming sounds-like spellings. Nothing existing is changed or removed.
    merge+overwrite  additions, and incoming entries replace existing ones with the same key (packs)."""
    if mode == "replace":
        if kind in PACK_KINDS:
            return dict(incoming)
        if kind == "styles":
            return {**{k: v for k, v in current.items() if k.startswith("tone:")}, **incoming}
        return {**current, **incoming}
    if kind == "settings":
        return dict(current)
    target = dict(current)
    for key, value in incoming.items():
        if key not in current:
            target[key] = value
        elif kind == "dictionary":
            term = value["term"] if overwrite else current[key]["term"]
            target[key] = {"term": term, "aliases": store.Dictionary._clean_aliases(
                term, current[key]["aliases"] + value["aliases"])}
        elif overwrite:
            target[key] = value
    return target


def _import(ds: DataSet, incoming: dict, history, stats, mode: str, overwrite: bool = False) -> dict:
    """Shared by backups and packs: apply ``incoming`` ({kind: {key: value}}) all or nothing."""
    if mode not in ("merge", "replace"):
        raise ValueError("Import mode must be “merge” or “replace”.")
    before = snapshot(ds, every_setting=True)
    history_before = ds.history.snapshot() if history is not None else None
    stats_before = ds.stats.export_days() if stats is not None and ds.stats is not None else None
    details: dict = {}
    counts = dict.fromkeys(PARTS, 0)
    skipped_settings: list[str] = []
    conflicts: list[dict] = []
    try:
        changes = {}
        for kind in KINDS:
            if kind in incoming:
                target = _merge_target(kind, before[kind], incoming[kind], mode, overwrite)
                changes[kind], details[kind] = _diff(before[kind], target)
                counts[kind] = len(changes[kind])
        skipped_settings = sorted(_apply(ds, changes, conflicts))
        if conflicts:                            # spellings were left out: count what really changed
            details["dictionary"] = _diff(before["dictionary"], snapshot(ds)["dictionary"])[1]
            counts["dictionary"] = sum(details["dictionary"].values())
        counts["settings"] = max(0, counts["settings"] - len(skipped_settings))
        if history is not None:
            counts["history"] = ds.history.import_entries(history, replace=(mode == "replace"))
        if stats_before is not None:
            counts["stats"] = ds.stats.import_days(stats, replace=(mode == "replace"))
    except Exception as exc:  # noqa: BLE001 - all or nothing: put everything back the way it was
        log.error("import failed; restoring the previous state\n%s", system.redacted_traceback(exc))
        try:
            now = snapshot(ds, every_setting=True)
            _apply(ds, {kind: _diff(now[kind], before[kind])[0] for kind in KINDS})
            if history_before is not None:
                ds.history.import_entries(history_before, replace=True)
            if stats_before is not None:
                ds.stats.import_days(stats_before, replace=True)
        except Exception as exc2:  # noqa: BLE001
            log.error("restoring after a failed import also failed\n%s", system.redacted_traceback(exc2))
        raise ValueError("The import could not be completed, so nothing was changed.") from None
    return {"mode": mode, "counts": counts, "details": details, "settings_skipped": skipped_settings,
            "aliases_skipped": conflicts}


def import_profile(ds: DataSet, path: str, mode: str = "merge", parts=None, device_id: str = "") -> dict:
    """Import a backup.

    ``mode`` "merge" only adds what is missing here (terms, sounds-like spellings, snippets, app rules,
    history entries, statistics for days this profile has none for) and changes nothing that exists, so
    settings and tones are left alone. "replace" makes this profile equal to the file: dictionary,
    snippets and app rules exactly, tones and settings from the file, history and statistics replaced.
    Settings that describe a computer (microphone, models, start with Windows...) are only taken from a
    backup made on this same device (``device_id``). ``parts`` limits the import to some of ``PARTS``.

    The file is validated completely before anything changes; if applying it fails, everything is put back.
    Returns ``{mode, counts: {part: entries added or changed}, details: {kind: {added, updated, removed}},
    settings_skipped: [keys], skipped: unusable entries in the file}``."""
    prof = read_profile(path)
    wanted = set(PARTS if parts is None else parts)
    unknown = wanted - set(PARTS)
    if unknown:
        raise ValueError(f"“{sorted(unknown)[0]}” is not a part of a backup.")
    incoming = {k: dict(v) for k, v in prof["data"].items() if k in wanted}
    if "settings" in incoming and not (device_id and prof["meta"]["device_id"] == device_id):
        portable = set(portable_settings())
        incoming["settings"] = {k: v for k, v in incoming["settings"].items() if k in portable}
    history = prof["history"] if "history" in wanted else None
    stats = prof["stats"] if "stats" in wanted else None
    if not incoming and history is None and stats is None:
        raise ValueError("That backup does not contain any of the parts you picked.")
    summary = _import(ds, incoming, history, stats, mode)
    summary["skipped"] = prof["skipped"]
    return summary


# =========================================================================== packs
def export_pack(ds: DataSet, kind: str, path: str, name: str = "", description: str = "", ids=None,
                clock=time.time) -> dict:
    """Write a shareable pack of dictionary terms or snippets (all of them, or the entries in ``ids``).
    Returns ``{path, kind, name, count}``."""
    if kind not in PACK_KINDS:
        raise ValueError("A pack holds either dictionary terms or snippets.")
    st = ds.dictionary if kind == "dictionary" else ds.snippets
    rows = st.list()
    if ids is not None:
        picked = set(ids)
        rows = [r for r in rows if r["id"] in picked]
    if not rows:
        raise ValueError("Pick at least one entry to share." if ids is not None
                         else f"There are no {'terms' if kind == 'dictionary' else 'snippets'} to share yet.")
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(PACK_EXT)
    name = _text(name, 80) or target.stem
    entries = [{k: v for k, v in r.items() if k != "id"} for r in rows]
    _write(target, {"format": PACK_FORMAT, "version": VERSION, "kind": kind, "name": name,
                    "description": _text(description, 400), "app_version": __version__,
                    "created": float(clock()), "entries": entries})
    return {"path": str(target), "kind": kind, "name": name, "count": len(entries)}


def read_pack(path: str) -> dict:
    """Load and validate a pack. Plain exports (``{"entries": [...]}`` or a bare list) are accepted too.
    Returns ``{kind, name, description, created, entries: {key: value}, skipped}``."""
    src = Path(path)
    data = _read_json(src, "pack")
    if isinstance(data, dict) and data.get("format") == PROFILE_FORMAT:
        raise ValueError(f"{src.name} is a whole backup, not a pack. Use “Import backup” for it.")
    meta = data if isinstance(data, dict) else {}
    if meta.get("format") == PACK_FORMAT:
        _check_version(meta, src, "pack")
    rows = meta.get("entries") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{src.name} is not a VoxFlow pack.")
    kind = meta.get("kind")
    if kind not in PACK_KINDS:
        first = next((r for r in rows if isinstance(r, (dict, str))), None)
        kind = "snippets" if isinstance(first, dict) and "trigger" in first else "dictionary"
    entries, skipped = _norm_list(kind, rows)
    if not entries:
        raise ValueError(f"{src.name} does not contain any usable {'terms' if kind == 'dictionary' else 'snippets'}.")
    created = meta.get("created")
    return {"kind": kind, "name": _text(meta.get("name"), 80) or src.stem,
            "description": _text(meta.get("description"), 400),
            "created": float(created) if isinstance(created, (int, float)) and not isinstance(created, bool)
            and math.isfinite(created) else None,
            "entries": entries, "skipped": skipped}


def describe_pack(ds: DataSet, path: str) -> dict:
    """What a pack contains and how much of it is new here (nothing is changed):
    ``{kind, name, description, count, created, new, existing, skipped}``."""
    pack = read_pack(path)
    have = snapshot(ds)[pack["kind"]]
    new = sum(1 for key in pack["entries"] if key not in have)
    return {"kind": pack["kind"], "name": pack["name"], "description": pack["description"],
            "count": len(pack["entries"]), "created": pack["created"], "new": new,
            "existing": len(pack["entries"]) - new, "skipped": pack["skipped"]}


def import_pack(ds: DataSet, path: str, mode: str = "merge") -> dict:
    """Bring a pack into the dictionary or the snippets. "merge": new entries are added and entries that
    exist here are updated from the pack (a dictionary term keeps its own sounds-like spellings too);
    "replace": the pack becomes the whole list. All or nothing.
    Returns ``{kind, name, added, updated, removed, skipped}``."""
    pack = read_pack(path)
    summary = _import(ds, {pack["kind"]: pack["entries"]}, None, None, mode, overwrite=True)
    return {"kind": pack["kind"], "name": pack["name"], **summary["details"][pack["kind"]],
            "skipped": pack["skipped"], "aliases_skipped": summary["aliases_skipped"]}


# =========================================================================== sync folder
class SyncUnavailable(Exception):
    """The sync folder cannot be read or written right now."""


def _wins(a: dict, b: dict) -> bool:
    """Last writer wins; a tie is broken by the device id, so every device picks the same winner."""
    return (a["t"], a["d"]) > (b["t"], b["d"])


class SyncEngine:
    """Keeps the portable data in step with the other devices that use the same folder.

    State kept in ``state_path`` (inside the data directory): this device's id, the folder, and one record
    per entry: ``{"v": value, "t": timestamp, "d": device id}`` or ``{"x": 1, "t", "d"}`` for a deletion.
    A sync round is ``read_remote()`` (file I/O), ``merge()`` (memory + local stores) and ``write_own()``
    (file I/O); ``sync()`` runs the three in a row. The split lets a caller keep slow folders off its thread.
    """

    def __init__(self, ds: DataSet, state_path: str | Path, clock=time.time, name: str | None = None):
        self._ds = ds
        self._path = Path(state_path)
        self._clock = clock
        self._lock = threading.RLock()
        self._state = self._load()
        if name:
            self._state["name"] = _text(name, 60) or self._state["name"]
        self._phase = "idle"
        self._problem = ""
        self._devices: list[dict] = []
        self._notes = {"unreadable": 0, "newer": 0}

    # ---- state file
    def _load(self) -> dict:
        state = {"version": 1, "device": uuid.uuid4().hex[:12], "name": device_name(), "folder": "",
                 "joined": False, "settled": False, "records": {}, "last_sync": 0.0, "written": ""}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return state
        except (OSError, ValueError) as exc:
            log.warning("sync state is unreadable (%s); this device will join its folder again", exc)
            return state
        if not isinstance(data, dict):
            return state
        for key in ("device", "name", "folder", "written"):
            if isinstance(data.get(key), str) and (data[key] or key in ("folder", "written")):
                state[key] = data[key]
        state["joined"] = bool(data.get("joined"))
        state["settled"] = bool(data.get("settled"))
        if isinstance(data.get("last_sync"), (int, float)) and math.isfinite(data["last_sync"]):
            state["last_sync"] = float(data["last_sync"])
        records = data.get("records") if isinstance(data.get("records"), dict) else {}
        now = self._clock()
        for kind in KINDS:
            part = {}
            for key, rec in (records.get(kind) or {}).items() if isinstance(records.get(kind), dict) else ():
                got = self._valid(kind, key, rec, now + 10 * 365 * 86400.0)
                if got is not None:
                    part[got[0]] = got[1]
            state["records"][kind] = part
        return state

    _saved_sig = ""
    _saved_at = 0.0
    _pending_last_sync = False

    def _save(self) -> None:
        self._pending_last_sync = False
        self._saved_sig, self._saved_at = self._sig(), self._clock()
        try:
            cfgmod._atomic_write(self._path, json.dumps(self._state, ensure_ascii=False))
        except OSError as exc:
            log.warning("could not save the sync state: %s", exc)

    # ---- configuration
    @property
    def folder(self) -> str:
        return self._state["folder"]

    @property
    def enabled(self) -> bool:
        return bool(self._state["folder"])

    @property
    def device_id(self) -> str:
        return self._state["device"]

    @property
    def file_name(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self._state["name"]).strip("-")[:30] or "device"
        return f"voxflow-sync-{slug}-{self._state['device'][:8]}.json"

    def set_folder(self, path: str | None) -> None:
        """Turn sync on (an existing, writable folder) or off (''). Raises ValueError with a human message.

        Pointing at a different folder makes this device join it afresh: what the folder already holds
        wins over this device's entries where both have the same one, and nothing is deleted."""
        path = (path or "").strip() if isinstance(path, str) or path is None else None
        if path is None:
            raise ValueError("The sync folder must be a folder path.")
        with self._lock:
            if not path:
                self._state.update(folder="", joined=False, settled=False, records={}, written="", last_sync=0.0)
                self._devices, self._problem, self._phase = [], "", "idle"
                self._save()
                return
            folder = Path(path).expanduser()
            if not folder.is_dir():
                raise ValueError("That folder doesn't exist. Pick an existing folder, for example one inside "
                                 "OneDrive, Dropbox or Syncthing.")
            try:
                folder = folder.resolve()
                if folder == self._path.parent.resolve():
                    raise ValueError("Pick a folder other than VoxFlow's own data folder, for example one inside "
                                     "OneDrive, Dropbox or Syncthing.")
                probe = folder / f".voxflow-write-test-{uuid.uuid4().hex[:8]}"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError:
                raise ValueError("VoxFlow can't save files in that folder. Pick a folder you are allowed to "
                                 "write to.") from None
            if str(folder) != self._state["folder"]:
                self._state.update(folder=str(folder), joined=False, settled=False, records={}, written="",
                                   last_sync=0.0)
                self._devices, self._problem = [], ""
            self._save()

    # ---- records
    @staticmethod
    def _valid(kind: str, key, rec, latest: float):
        """A record read from a file in canonical form ``(key, record)``, or None."""
        if not isinstance(rec, dict) or not isinstance(key, str):
            return None
        t, d = rec.get("t"), rec.get("d")
        if (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 or t > latest
                or not isinstance(d, str) or not 0 < len(d) <= 64):
            return None
        if rec.get("x"):
            if kind == "styles":
                key = f"app:{store.normalize_exe(key[4:])}" if key.startswith("app:") else key
            elif kind == "snippets":
                key = _snippet_key(key)
            else:
                key = _dict_key(key)
            return (key, {"x": 1, "t": float(t), "d": d}) if key and _deletable(kind, key) else None
        got = _norm(kind, key, rec.get("v"))
        return None if got is None else (got[0], {"v": got[1], "t": float(t), "d": d})

    def stamp_local(self) -> bool:
        """Notice what changed on this device since the last look and timestamp it. No folder access.
        Returns True when something was stamped."""
        with self._lock:
            if not self.enabled:
                return False
            local, now, me = snapshot(self._ds), self._clock(), self._state["device"]
            joining = not self._state["joined"]
            changed = False
            for kind in KINDS:
                recs = self._state["records"].setdefault(kind, {})
                for key, value in local[kind].items():
                    rec = recs.get(key)
                    if rec is None or rec.get("x") or rec["v"] != value:
                        stamp = BASELINE if joining else max(now, (rec["t"] if rec else 0.0) + 0.001)
                        recs[key] = {"v": value, "t": stamp, "d": me}
                        changed = True
                for key, rec in list(recs.items()):
                    if rec.get("x") or key in local[kind]:
                        continue
                    if _deletable(kind, key):
                        recs[key] = {"x": 1, "t": max(now, rec["t"] + 0.001), "d": me}
                    else:
                        del recs[key]                    # a setting this version no longer has
                    changed = True
            if joining or changed:
                self._state["joined"] = True
                self._save()
            return changed

    # ---- one round
    def read_remote(self) -> dict:
        """Read the other devices' files. File I/O only; safe to call from a worker thread.
        Raises SyncUnavailable when the folder cannot be listed."""
        with self._lock:
            folder, own = self._state["folder"], self.file_name
        if not folder:
            raise SyncUnavailable("sync is off")
        try:
            names = sorted(n for n in os.listdir(folder) if _SYNC_FILE_RE.match(n))
        except OSError as exc:
            raise SyncUnavailable(str(exc)) from None
        out = {"docs": [], "unreadable": 0, "newer": 0, "own_present": own in names}
        for name in names:
            if name == own:
                continue
            path = Path(folder) / name
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    raise ValueError("too large")
                doc = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(doc, dict) or doc.get("format") != SYNC_FORMAT:
                    raise ValueError("not a sync file")
                version = doc.get("version")
                if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                    raise ValueError("no version")
            except (OSError, ValueError) as exc:        # half-copied by the sync tool, or not ours: next round
                log.info("sync file %s skipped: %s", name, exc)
                out["unreadable"] += 1
                continue
            if version > VERSION:
                out["newer"] += 1
                continue
            if isinstance(doc.get("records"), dict):
                out["docs"].append(doc)
        return out

    def merge(self, remote: dict) -> tuple[str, str] | None:
        """Merge what the other devices wrote, apply the winners here, and return ``(text, digest)`` for
        this device's own file when it needs (re)writing, else None. Memory and local stores only."""
        with self._lock:
            if not self.enabled:
                return None
            self.stamp_local()
            now, records, me = self._clock(), self._state["records"], self._state["device"]
            # First merge after joining a folder: what the folder holds wins over what this device brought
            # along, except that an old deletion does not remove an entry this device still has.
            joining = not self._state["settled"]
            cutoff = now - TOMBSTONE_DAYS * 86400.0
            taken: dict = {kind: {} for kind in KINDS}
            devices: dict[str, dict] = {}
            for doc in remote["docs"]:
                dev = doc.get("device")
                if isinstance(dev, str) and dev and dev != self._state["device"]:
                    updated = doc.get("updated")
                    seen = devices.setdefault(dev, {"id": dev, "name": _text(doc.get("name"), 60) or "Another device",
                                                    "updated": 0.0})
                    if isinstance(updated, (int, float)) and math.isfinite(updated):
                        seen["updated"] = max(seen["updated"], float(updated))
                for kind in KINDS:
                    part = doc["records"].get(kind)
                    recs = records.setdefault(kind, {})
                    for key, raw in part.items() if isinstance(part, dict) else ():
                        got = self._valid(kind, key, raw, now + _CLOCK_SKEW_S)
                        if got is None:
                            continue
                        key, rec = got
                        mine = recs.get(key)
                        brought = (joining and mine is not None and not mine.get("x")
                                   and mine["t"] == BASELINE and mine["d"] == me)
                        if mine is None:
                            if rec.get("x") and rec["t"] < cutoff:
                                continue                 # a deletion too old to matter
                        elif brought and rec.get("x"):
                            recs[key] = {"v": mine["v"], "t": now, "d": me}     # counts as added again, now
                            continue
                        elif not brought and not _wins(rec, mine):
                            continue
                        recs[key] = rec
                        taken[kind][key] = _DELETED if rec.get("x") else rec["v"]
            local = snapshot(self._ds)
            changes = {kind: {k: v for k, v in part.items()
                              if (v is _DELETED and k in local[kind]) or (v is not _DELETED and local[kind].get(k) != v)}
                       for kind, part in taken.items()}
            if any(changes.values()):
                _apply(self._ds, changes)
                self.stamp_local()                       # whatever could not be applied counts as a local edit
            for recs in records.values():
                for key in [k for k, r in recs.items() if r.get("x") and r["t"] < cutoff]:
                    del recs[key]
            self._state["settled"] = True
            self._devices = sorted(devices.values(), key=lambda d: d["name"].lower())
            self._notes = {"unreadable": remote["unreadable"], "newer": remote["newer"]}
            body = json.dumps(records, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
            if digest == self._state["written"] and remote["own_present"]:
                self._finish_round(now, defer=True)       # nothing changed: no disk write for a mere timestamp
                return None
            self._save()
            doc = {"format": SYNC_FORMAT, "version": VERSION, "device": self._state["device"],
                   "name": self._state["name"], "app_version": __version__, "updated": now, "records": records}
            return json.dumps(doc, ensure_ascii=False, sort_keys=True), digest

    def write_own(self, payload: tuple[str, str]) -> None:
        """Write this device's file (atomically). File I/O; safe from a worker thread.
        Raises SyncUnavailable when the folder cannot be written."""
        text, digest = payload
        with self._lock:
            folder, name = self._state["folder"], self.file_name
        if not folder:
            return
        try:
            cfgmod._atomic_write(Path(folder) / name, text)
        except OSError as exc:
            raise SyncUnavailable(str(exc)) from None
        with self._lock:
            if self._state["folder"] == folder:
                self._state["written"] = digest
                self._finish_round(self._clock())

    SAVE_EVERY_S = 600.0              # an unchanged round writes the state file at most this often

    def _sig(self) -> str:
        return json.dumps({k: v for k, v in self._state.items() if k != "last_sync"}, sort_keys=True,
                          ensure_ascii=False)

    def _finish_round(self, now: float, defer: bool = False) -> None:
        self._state["last_sync"] = now
        self._phase, self._problem = "idle", ""
        if defer and self._sig() == self._saved_sig and now - self._saved_at < self.SAVE_EVERY_S:
            self._pending_last_sync = True
            return
        self._save()

    def flush(self) -> None:
        """Write the state file if only ``last_sync`` is waiting (shutdown): the time survives a restart."""
        with self._lock:
            if self._pending_last_sync:
                self._save()

    def note_unavailable(self, exc: BaseException) -> None:
        with self._lock:
            self._phase = "idle"
            self._problem = str(exc) or "unreachable"
            log.info("sync folder unavailable: %s", exc)

    def note_syncing(self) -> None:
        with self._lock:
            self._phase = "syncing"

    def note_idle(self) -> None:
        """A round was put off (nothing failed)."""
        with self._lock:
            self._phase = "idle"

    def sync(self) -> dict:
        """One complete round, synchronously. Never raises; the outcome is in the returned ``status()``."""
        if self.enabled:
            self.note_syncing()
            try:
                self.stamp_local()                   # before the folder is touched: an offline edit keeps its time
                payload = self.merge(self.read_remote())
                if payload is not None:
                    self.write_own(payload)
            except SyncUnavailable as exc:
                self.note_unavailable(exc)
        return self.status()

    # ---- reporting
    def status(self) -> dict:
        """``{state, folder, last_sync, error, message, enabled, device, devices, unreadable, newer}``.

        ``state`` is "off", "syncing", "idle" (in step) or "error" (the folder cannot be reached; ``error``
        then holds a sentence for the user, otherwise it is ''). ``message`` is one line for a status row."""
        with self._lock:
            enabled, error = self.enabled, ""
            if not enabled:
                state, message = "off", "Sync is off."
            elif self._problem:
                state = "error"
                error = message = ("The sync folder can't be reached right now. Your changes are kept and will "
                                   "sync when it is back.")
            elif self._phase == "syncing":
                state, message = "syncing", "Syncing…"
            else:
                state = "idle"
                n = len(self._devices)
                if not self._state["last_sync"]:
                    message = "Waiting for the first sync."
                elif n == 0:
                    message = "Up to date. No other device has used this folder yet."
                else:
                    message = f"Up to date with {n} other device{'s' if n != 1 else ''}."
                if self._notes["newer"]:
                    message += " A device with a newer VoxFlow was skipped: update this one to sync with it."
                elif self._notes["unreadable"]:
                    message += " A file in the folder could not be read yet and was skipped."
            return {"state": state, "folder": self._state["folder"], "last_sync": self._state["last_sync"] or None,
                    "error": error, "message": message, "enabled": enabled, "device": self._state["name"],
                    "devices": [dict(d) for d in self._devices] if enabled else [],
                    "unreadable": self._notes["unreadable"] if enabled else 0,
                    "newer": self._notes["newer"] if enabled else 0}


__all__ = ["DataSet", "SyncEngine", "SyncUnavailable", "export_profile", "import_profile", "describe_profile",
           "read_profile", "export_pack", "import_pack", "describe_pack", "read_pack", "portable_settings",
           "snapshot", "PARTS", "KINDS", "PACK_KINDS", "LOCAL_SETTINGS", "PROFILE_EXT", "PACK_EXT"]
