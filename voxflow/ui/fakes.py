"""UI preview stand-ins: ``FakeRecorder`` (a synthetic speech-like signal) and ``FakeController``, an
in-memory twin of ``voxflow.controller.Controller`` (same signals, properties and methods)
seeded with realistic data. The screenshot script uses it so every page can be rendered deterministically
without models, a microphone, the keyboard hook or any file on disk.
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import json
import math
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

SAMPLE_RATE = 16000


class FakeRecorder:
    """Synthesises a speech-like signal (syllable bursts) so the UI can be shown without a mic."""

    hit_cap = False

    def __init__(self, device=None, max_seconds: float = 120.0):
        self._rec = False
        self._t0 = 0.0
        self.max_seconds = max_seconds
        self.active_voice = True

    def _env(self, t: np.ndarray) -> np.ndarray:
        syl = 0.5 + 0.5 * np.sin(2 * math.pi * 3.1 * t + 1.3 * np.sin(2 * math.pi * 0.7 * t))
        phr = np.clip(np.sin(2 * math.pi * 0.45 * t) * 1.6, 0, 1) ** 0.7
        return (syl ** 1.4) * (0.15 + 0.85 * phr) * 0.32

    def start(self) -> None:
        self._rec, self._t0 = True, time.perf_counter()

    def stop(self) -> np.ndarray:
        if not self._rec:
            return np.zeros(0, dtype=np.float32)
        self._rec = False
        n = int(min(self.max_seconds, time.perf_counter() - self._t0) * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        return (self._env(t) * np.sin(2 * math.pi * 160 * t)).astype(np.float32)

    def close(self) -> None:
        self._rec = False

    def last_speech_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self._t0) if self._t0 else 0.0

    @property
    def is_recording(self) -> bool:
        return self._rec

    def recent_samples(self, n: int = 1024) -> np.ndarray:
        if not self.active_voice:
            return np.zeros(n, dtype=np.float32)
        t = time.perf_counter() - np.arange(n)[::-1] / SAMPLE_RATE
        f0 = 140 + 40 * np.sin(t * 2.1)
        sig = np.sin(2 * math.pi * f0 * t) + 0.5 * np.sin(2 * math.pi * 2 * f0 * t)
        return (self._env(t) * sig * 0.8).astype(np.float32)

    def level(self) -> float:
        if not self.active_voice:
            return 0.0
        s = self.recent_samples(800)
        return float(min(1.0, np.sqrt((s ** 2).mean()) * 3.2))


# --------------------------------------------------------------------------- in-memory stores
def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _clean_aliases(aliases) -> list[str]:
    out: list[str] = []
    for a in aliases or []:
        a = str(a).strip()
        if a and a.lower() not in [o.lower() for o in out]:
            out.append(a)
    return out


class FakeDictionary:
    def __init__(self, changed, suggestions_changed):
        self._changed, self._sugg_changed = changed, suggestions_changed
        self._items: list[dict] = []
        self._sugg: list[dict] = []

    def list(self) -> list[dict]:
        return copy.deepcopy(sorted(self._items, key=lambda e: e["term"].lower()))

    def terms(self) -> list[str]:
        return [e["term"] for e in self.list()]

    def _check(self, term: str, skip_id: str | None = None) -> str:
        term = (term or "").strip()
        if not term:
            raise ValueError("Enter the word or phrase you want VoxFlow to spell correctly.")
        if any(e["term"].lower() == term.lower() and e["id"] != skip_id for e in self._items):
            raise ValueError(f"“{term}” is already in your dictionary.")
        return term

    def add(self, term: str, aliases=None) -> dict:
        entry = {"id": _new_id(), "term": self._check(term), "aliases": _clean_aliases(aliases)}
        self._items.append(entry)
        self._changed.emit()
        return copy.deepcopy(entry)

    def update(self, entry_id: str, term: str | None = None, aliases=None) -> dict:
        for e in self._items:
            if e["id"] == entry_id:
                if term is not None:
                    e["term"] = self._check(term, entry_id)
                if aliases is not None:
                    e["aliases"] = _clean_aliases(aliases)
                self._changed.emit()
                return copy.deepcopy(e)
        raise ValueError("That dictionary entry no longer exists.")

    def remove(self, entry_id: str) -> bool:
        n = len(self._items)
        self._items = [e for e in self._items if e["id"] != entry_id]
        if len(self._items) != n:
            self._changed.emit()
        return len(self._items) != n

    def apply(self, text: str) -> str:
        return text

    def import_(self, path: str) -> int:
        added = 0
        p = Path(path)
        try:
            raw = p.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ValueError(f"Could not read {p.name}: {exc.strerror or exc}") from exc
        rows: list[tuple[str, list[str]]] = []
        if p.suffix.lower() == ".json":
            try:
                data = json.loads(raw)
            except ValueError as exc:
                raise ValueError(f"{p.name} is not valid JSON.") from exc
            for d in data if isinstance(data, list) else []:
                if isinstance(d, dict) and d.get("term"):
                    rows.append((str(d["term"]), list(d.get("aliases") or [])))
        else:
            for line in raw.splitlines():
                term, _, rest = line.partition(":" if p.suffix.lower() != ".csv" else ",")
                if term.strip():
                    rows.append((term, rest.split(",")))
        for term, aliases in rows:
            try:
                self.add(term, aliases)
                added += 1
            except ValueError:
                pass
        return added

    def export(self, path: str) -> int:
        items = self.list()
        p = Path(path)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps([{"term": e["term"], "aliases": e["aliases"]} for e in items], indent=1,
                                    ensure_ascii=False), encoding="utf-8")
        else:
            sep = "," if p.suffix.lower() == ".csv" else ": "
            p.write_text("\n".join(e["term"] + (sep + ", ".join(e["aliases"]) if e["aliases"] else "")
                                   for e in items) + "\n", encoding="utf-8")
        return len(items)

    def suggestions(self) -> list[dict]:
        return copy.deepcopy(self._sugg)

    def accept_suggestion(self, suggestion_id: str) -> dict:
        for s in self._sugg:
            if s["id"] == suggestion_id:
                self._sugg.remove(s)
                self._sugg_changed.emit()
                for e in self._items:
                    if e["term"].lower() == s["term"].lower():
                        return self.update(e["id"], aliases=e["aliases"] + s["aliases"])
                return self.add(s["term"], s["aliases"])
        raise ValueError("That suggestion is no longer available.")

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        n = len(self._sugg)
        self._sugg = [s for s in self._sugg if s["id"] != suggestion_id]
        if len(self._sugg) != n:
            self._sugg_changed.emit()
        return len(self._sugg) != n


class FakeSnippets:
    def __init__(self, changed):
        self._changed = changed
        self._items: list[dict] = []

    def list(self) -> list[dict]:
        return copy.deepcopy(sorted(self._items, key=lambda e: e["trigger"].lower()))

    def _check(self, trigger: str | None, text: str | None, skip_id: str | None = None):
        if trigger is not None:
            trigger = trigger.strip()
            if not trigger:
                raise ValueError("Enter the phrase you will say to insert this snippet.")
            if any(e["trigger"].lower() == trigger.lower() and e["id"] != skip_id for e in self._items):
                raise ValueError(f"You already have a snippet for “{trigger}”.")
        if text is not None and not text.strip():
            raise ValueError("Enter the text this snippet should insert.")
        return trigger, text

    def add(self, trigger: str, text: str) -> dict:
        trigger, text = self._check(trigger, text)
        entry = {"id": _new_id(), "trigger": trigger, "text": text}
        self._items.append(entry)
        self._changed.emit()
        return copy.deepcopy(entry)

    def update(self, entry_id: str, trigger: str | None = None, text: str | None = None) -> dict:
        for e in self._items:
            if e["id"] == entry_id:
                trigger, text = self._check(trigger, text, entry_id)
                if trigger is not None:
                    e["trigger"] = trigger
                if text is not None:
                    e["text"] = text
                self._changed.emit()
                return copy.deepcopy(e)
        raise ValueError("That snippet no longer exists.")

    def remove(self, entry_id: str) -> bool:
        n = len(self._items)
        self._items = [e for e in self._items if e["id"] != entry_id]
        if len(self._items) != n:
            self._changed.emit()
        return len(self._items) != n

    def expand(self, text: str) -> str:
        return text

    def import_(self, path: str) -> int:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read snippets from {Path(path).name}.") from exc
        added = 0
        for d in data if isinstance(data, list) else []:
            try:
                self.add(str(d.get("trigger", "")), str(d.get("text", "")))
                added += 1
            except (ValueError, AttributeError):
                pass
        return added

    def export(self, path: str) -> int:
        items = self.list()
        Path(path).write_text(json.dumps([{"trigger": e["trigger"], "text": e["text"]} for e in items], indent=1,
                                         ensure_ascii=False), encoding="utf-8")
        return len(items)


class FakeHistory:
    def __init__(self, changed):
        self._changed = changed
        self._items: list[dict] = []          # newest first

    def list(self, limit: int | None = 50, offset: int = 0) -> list[dict]:
        end = None if limit is None else offset + limit
        return copy.deepcopy(self._items[offset:end])

    def search(self, q: str, limit: int | None = None) -> list[dict]:
        q = (q or "").strip().lower()
        hits = [e for e in self._items if not q or q in e["text"].lower() or q in e["raw"].lower()
                or q in e["app_exe"].lower() or q in e["app_title"].lower()]
        return copy.deepcopy(hits[:limit])

    def get(self, entry_id: str) -> dict | None:
        return next((copy.deepcopy(e) for e in self._items if e["id"] == entry_id), None)

    def count(self) -> int:
        return len(self._items)

    def count_older_than(self, days: int) -> int:
        cutoff = time.time() - days * 86400.0
        return sum(1 for e in self._items if e["ts"] < cutoff) if days > 0 else 0

    def delete(self, entry_id: str) -> bool:
        n = len(self._items)
        self._items = [e for e in self._items if e["id"] != entry_id]
        if len(self._items) != n:
            self._changed.emit()
        return len(self._items) != n

    def restore(self, entry: dict) -> bool:
        if any(e["id"] == entry["id"] for e in self._items):
            return False
        self._items.append(copy.deepcopy(entry))
        self._items.sort(key=lambda e: e["ts"], reverse=True)
        self._changed.emit()
        return True

    def clear(self) -> None:
        self._items = []
        self._changed.emit()

    def export(self, path: str, fmt: str | None = None) -> int:
        p = Path(path)
        fmt = (fmt or p.suffix.lstrip(".") or "txt").lower()
        items = self.list(None)
        if fmt == "json":
            p.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        elif fmt == "csv":
            with p.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["time", "app", "words", "text", "raw"])
                for e in items:
                    w.writerow([_dt.datetime.fromtimestamp(e["ts"]).isoformat(timespec="seconds"), e["app_exe"],
                                e["words"], e["text"], e["raw"]])
        else:
            p.write_text("\n\n".join(f"[{_dt.datetime.fromtimestamp(e['ts']):%Y-%m-%d %H:%M}] {e['text']}"
                                     for e in items) + "\n", encoding="utf-8")
        return len(items)


class FakeStats:
    def __init__(self, changed):
        self._changed = changed
        self._daily: list[dict] = []
        self._apps: list[dict] = []

    def summary(self) -> dict:
        words = sum(d["words"] for d in self._daily)
        secs = sum(d["seconds"] for d in self._daily)
        streak = 0
        for d in reversed(self._daily):
            if d["words"] <= 0:
                break
            streak += 1
        return {"total_words": words, "sessions": sum(d["sessions"] for d in self._daily), "seconds_spoken": secs,
                "avg_wpm": (words / secs * 60) if secs else 0.0, "time_saved_s": max(0.0, words / 40 * 60 - secs),
                "streak_days": streak, "today_words": self._daily[-1]["words"] if self._daily else 0}

    def daily(self, n_days: int = 14) -> list[dict]:
        today = _dt.date.today()
        by_date = {d["date"]: d for d in self._daily}
        out = []
        for i in range(n_days - 1, -1, -1):
            key = (today - _dt.timedelta(days=i)).isoformat()
            out.append(dict(by_date.get(key, {"date": key, "words": 0, "sessions": 0, "seconds": 0.0})))
        return out

    def top_apps(self, n: int = 5) -> list[dict]:
        return copy.deepcopy(sorted(self._apps, key=lambda a: -a["words"])[:n])

    def reset(self) -> None:
        self._daily, self._apps = [], []
        self._changed.emit()


_BUILTIN_APPS = {"slack.exe": "work", "teams.exe": "work", "discord.exe": "personal", "outlook.exe": "email",
                 "whatsapp.exe": "personal", "telegram.exe": "personal", "thunderbird.exe": "email",
                 "code.exe": "work", "notion.exe": "work", "winword.exe": "other", "chrome.exe": "other"}


class FakeStyles:
    CATEGORIES = ("personal", "work", "email", "other")
    TONES = ("formal", "casual", "very_casual")

    def __init__(self, changed):
        self._changed = changed
        self._tones = {"personal": "casual", "work": "casual", "email": "formal", "other": "formal"}
        self._overrides: dict[str, str] = {}

    def tones(self) -> dict[str, str]:
        return dict(self._tones)

    def set_tone(self, category: str, tone: str) -> None:
        if category not in self.CATEGORIES or tone not in self.TONES:
            raise ValueError("Unknown style.")
        self._tones[category] = tone
        self._changed.emit()

    @staticmethod
    def _norm(exe: str) -> str:
        exe = (exe or "").strip().lower().replace("\\", "/").rsplit("/", 1)[-1]
        if not exe:
            raise ValueError("Enter an app name such as slack.exe.")
        return exe if exe.endswith(".exe") else exe + ".exe"

    def app_categories(self) -> list[dict]:
        rows = []
        for exe in sorted(set(_BUILTIN_APPS) | set(self._overrides)):
            builtin = exe in _BUILTIN_APPS          # like the real store: "overridden" only ever describes built-ins
            rows.append({"exe": exe, "category": self._overrides.get(exe, _BUILTIN_APPS.get(exe, "other")),
                         "builtin": builtin, "overridden": builtin and exe in self._overrides})
        return rows

    def set_app_category(self, exe: str, category: str) -> None:
        if category not in self.CATEGORIES:
            raise ValueError("Unknown category.")
        self._overrides[self._norm(exe)] = category
        self._changed.emit()

    def remove_app_category(self, exe: str) -> bool:
        had = self._overrides.pop(self._norm(exe), None) is not None
        if had:
            self._changed.emit()
        return had

    def category_for(self, exe: str, title: str = "") -> str:
        exe = self._norm(exe)
        return self._overrides.get(exe, _BUILTIN_APPS.get(exe, "other"))

    def tone_for(self, exe: str, title: str = "") -> str:
        return self._tones[self.category_for(exe, title)]


# --------------------------------------------------------------------------- the controller twin
_DEFAULTS = dict(
    hotkey="ctrl+win", command_hotkey="alt+win", paste_last_hotkey="alt+shift+z", undo_hotkey="",
    double_tap_lock=True, cleanup_enabled=True, smart_formatting=True, auto_learn=True, language="auto",
    whisper_mode=False, mic_device=None, sounds_enabled=True, sound_volume=0.5, mute_media=False,
    launch_at_login=False, start_minimized=False, save_history=True, history_retention_days=0,
    history_limit=5000, restore_clipboard=True, restore_delay_s=0.5, show_idle_pill=False, live_preview=True, phrase_streaming=False, theme="system",
    reduced_motion=False, overlay_position=None, ui_state={}, onboarding_done=False, sync_folder="",
    update_mode="notify", update_install_when="quit", update_time="03:00",
    asr_model="nemo-parakeet-tdt-0.6b-v3", asr_device="auto", cleanup_model=None, cleanup_device="auto",
    cleanup_timeout_s=4.0, command_timeout_s=20.0, min_record_s=0.3, max_record_s=120.0, config_version=2)
# the release the preview "finds" (see Controller.update_info)
_DEMO_RELEASE = {
    "version": "0.3.0", "tag": "v0.3.0", "title": "VoxFlow 0.3.0", "published": "2026-12-01T10:00:00Z",
    "page": "https://github.com/blackknight75/voxflow/releases/tag/v0.3.0", "installer": "VoxFlow-Setup-0.3.0.exe",
    "installer_url": "https://github.com/blackknight75/voxflow/releases/download/v0.3.0/VoxFlow-Setup-0.3.0.exe",
    "size": 1_800_000_000, "sums_url": "https://github.com/blackknight75/voxflow/releases/download/v0.3.0/SHA256SUMS.txt",
    "notes": "Added\n• App updates: install automatically, or be told and choose when.\n\nFixed\n"
             "• The setup guide waits for the speech model before the hotkey practice."}
_UPDATE_MANAGED = {"scoop": "Updates are managed by Scoop: run scoop update voxflow.",
                   "winget": "Updates are managed by winget: run winget upgrade blackknight75.VoxFlow.",
                   "source": "VoxFlow is running from source: update it with git pull.",
                   "allusers": "VoxFlow is installed for all users, so it cannot update itself without administrator "
                               "rights. Download the new version from the releases page.",
                   "portable": "This copy was not installed with the VoxFlow installer, so it cannot update itself. "
                               "Get the new version from the releases page."}
# what the speed check measured on the preview's PC (an NVIDIA desktop); see Controller.device_info
_DEVICE_BACKENDS = {"asr": ("cuda", "webgpu", "cpu"), "cleanup": ("cuda", "vulkan", "cpu")}
_DEVICE_RESULTS = {"asr": {"cuda": {"ms": 29.0}, "webgpu": {"ms": 65.0}, "cpu": {"ms": 257.0}},
                   "cleanup": {"cuda": {"ms": 186.0}, "vulkan": {"ms": 241.0}, "cpu": {"ms": 1930.0}}}
_DEVICE_TEXT = {"cuda": "Running on the GPU (CUDA).", "webgpu": "Running on the GPU (WebGPU).",
                "vulkan": "Running on the GPU (Vulkan).", "cpu": "Running on the CPU."}
_HOTKEY_LABELS = {"hotkey": "Dictation", "command_hotkey": "Command mode", "paste_last_hotkey": "Paste last",
                  "undo_hotkey": "Undo last paste"}
_MODS = ("ctrl", "alt", "shift", "win")

_SAMPLE_TEXTS = [
    ("slack.exe", "Slack | #launch", "Hey Sam, can you move our sync to Thursday at 3 PM? I'll send the updated agenda tonight.",
     "hey sam um can you move our sync to thursday at 3 pm i'll send the updated agenda tonight", "dictate"),
    ("outlook.exe", "RE: Q3 board update - Outlook", "Hi Priya,\n\nThe quarterly numbers look strong: revenue is up 12%, churn is down, and the new onboarding flow cut time-to-value in half. Let's highlight that in the board update.\n\nBest,\nAlex",
     "hi priya the quarterly numbers look strong revenue is up twelve percent churn is down and the new onboarding flow cut time to value in half let's highlight that in the board update best alex", "dictate"),
    ("code.exe", "pipeline.py - voxflow - Visual Studio Code", "Refactor the retry loop so it backs off exponentially and logs the attempt number.",
     "refactor the retry loop so it backs off exponentially and uh logs the attempt number", "dictate"),
    ("chrome.exe", "Google Docs - Launch plan", "- Finalise pricing page\n- Record the demo video\n- Email the beta list",
     "turn into bullets", "command"),
    ("whatsapp.exe", "WhatsApp", "running 10 min late, save me a seat", "running ten minutes late save me a seat", "dictate"),
    ("notion.exe", "Meeting notes", "Remember to book the flights before Friday.", "remember to book the flights before friday", "dictate"),
    ("slack.exe", "Slack | DM", "Sounds good, shipping it after lunch.", "sounds good shipping it after lunch", "dictate"),
    ("chrome.exe", "Gmail", "What is the capital of France?", "what is the capital of france", "dictate"),
    ("teams.exe", "Design review | Microsoft Teams", "Can everyone add their comments to the Figma file by end of day?",
     "can everyone add their comments to the figma file by end of day", "dictate"),
]


class FakeController(QObject):
    """In-memory implementation of the Controller facade."""

    state_changed = Signal(str, str, str)
    partial_text = Signal(str, str)            # live preview: (committed, tentative) words so far
    level_changed = Signal(float)
    transcript_ready = Signal(dict)
    history_changed = Signal()
    stats_changed = Signal()
    dictionary_changed = Signal()
    suggestions_changed = Signal()
    dictionary_suggestion = Signal(dict)
    snippets_changed = Signal()
    styles_changed = Signal()
    config_changed = Signal(str, object)
    status_changed = Signal(dict)
    model_status_changed = Signal(str, dict)
    microphones_changed = Signal(list)
    hotkey_captured = Signal(str)
    notice = Signal(str, str)
    notice_action = Signal(str, str, str)      # message, action id, button label
    error_info_changed = Signal(dict)
    activate_requested = Signal()
    sync_status_changed = Signal(dict)
    devices_changed = Signal(dict)
    update_changed = Signal(dict)
    update_available = Signal(dict)
    restart_requested = Signal()

    is_demo = True
    version = "0.2.0"

    def __init__(self, *, empty: bool = False, parent=None):
        super().__init__(parent)
        self.config = SimpleNamespace(**copy.deepcopy(_DEFAULTS))
        self.state, self.mode, self.message = "idle", "dictate", ""
        self.paused = False
        self.paused_until = None
        self.data_dir = Path(tempfile.gettempdir()) / "voxflow-preview"
        self.log_path = self.data_dir / "voxflow.log"
        self.dictionary = FakeDictionary(self.dictionary_changed, self.suggestions_changed)
        self.snippets = FakeSnippets(self.snippets_changed)
        self.history = FakeHistory(self.history_changed)
        self.stats = FakeStats(self.stats_changed)
        self.styles = FakeStyles(self.styles_changed)
        self._recorder = FakeRecorder()
        self._models = {"asr": {"state": "ready", "detail": "Running on GPU (CUDA)", "progress": None, "backend": "cuda"},
                        "cleanup": {"state": "ready", "detail": "Running on GPU (CUDA)", "progress": None,
                                    "backend": "cuda"}}
        self._mics = [{"id": None, "name": "System default", "is_default": True},
                      {"id": "Microphone (Razer Seiren V2 X)", "name": "Microphone (Razer Seiren V2 X)", "is_default": False},
                      {"id": "Headset (WH-1000XM5)", "name": "Headset (WH-1000XM5)", "is_default": False}]
        self._capturing = False
        self._monitor = False
        self.rules_only_count = 0              # dictations of this session cleaned by rules only
        self._sync_state, self._sync_error, self._last_sync = "idle", "", None
        self._sync_devices: list[dict] = []
        self._device_results = copy.deepcopy(_DEVICE_RESULTS)
        self._device_auto = {"asr": "cuda", "cleanup": "cuda"}
        self._forced = ""
        self._speed = {"state": "done", "done": 0, "total": 0, "model": "", "backend": "", "available": True,
                       "measured_at": time.time() - 3 * 86400}
        self._update = {"kind": "installer", "phase": "idle", "progress": None, "error": "",
                        # a minute ago: "today" at any hour, even just after midnight
                        "last_check": time.time() - 60, "last_ok": True, "last_result": "VoxFlow is up to date.",
                        "available": None, "skipped": False, "install_on_quit": False, "downloaded": False}
        self.quit_prepared = False
        self._speed_timer = QTimer(self)
        self._speed_timer.setInterval(250)
        self._speed_timer.timeout.connect(self._speed_step)
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(lambda: self.demo_set_state("idle"))
        self.played: list[str] = []            # play_sound() calls, for tests
        if not empty:
            self._seed()

    # ---- seed data
    def _seed(self) -> None:
        now = time.time()
        offsets = [40, 1500, 5400, 9000, 86400 + 3000, 86400 + 9000, 2 * 86400 + 4000, 4 * 86400 + 800, 6 * 86400 + 5000]
        for (exe, title, text, raw, mode), off in zip(_SAMPLE_TEXTS, offsets):
            words = len(text.split())
            dur = max(1.2, words / 2.4)
            self.history._items.append({
                "id": _new_id(), "ts": now - off, "raw": raw, "text": text, "app_exe": exe, "app_title": title,
                "duration_s": round(dur, 1), "words": words, "wpm": round(words / dur * 60, 1), "mode": mode,
                "latency_ms": 420 + (words * 7) % 380})
        self.history._items[2]["rules_only"] = True          # one dictation the language model was too slow for
        today = _dt.date.today()
        pattern = [320, 0, 540, 880, 410, 0, 0, 760, 1240, 930, 610, 1480, 1120, 870]
        for i, words in enumerate(pattern):
            day = today - _dt.timedelta(days=len(pattern) - 1 - i)
            self.stats._daily.append({"date": day.isoformat(), "words": words, "sessions": words // 45,
                                      "seconds": round(words / 148 * 60, 1)})
        self.stats._apps = [{"exe": "slack.exe", "words": 4120, "sessions": 96},
                            {"exe": "outlook.exe", "words": 2890, "sessions": 41},
                            {"exe": "code.exe", "words": 1540, "sessions": 37},
                            {"exe": "chrome.exe", "words": 1010, "sessions": 22},
                            {"exe": "whatsapp.exe", "words": 400, "sessions": 18}]
        for term, aliases in (("Kubernetes", ["cube are netties", "kubernetties"]), ("VoxFlow", ["vox flow", "box flow"]),
                              ("Priya Raghunathan", ["pria ragunathan"]), ("PostgreSQL", ["post gress q l", "postgres sequel"]),
                              ("OAuth", ["oh auth"]), ("Figma", [])):
            self.dictionary._items.append({"id": _new_id(), "term": term, "aliases": aliases})
        self.dictionary._sugg = [
            {"id": _new_id(), "term": "GitHub", "aliases": ["get hub"], "ts": now - 600, "source": "cleanup"},
            {"id": _new_id(), "term": "Anthropic", "aliases": ["and tropic"], "ts": now - 4000, "source": "cleanup"}]
        for trig, text in (("my address", "221B Baker Street\nLondon NW1 6XE"),
                           ("calendar link", "https://cal.example.com/alex/30min"),
                           ("sign off", "Thanks so much,\nAlex"),
                           ("standup template", "Yesterday:\n- \n\nToday:\n- \n\nBlockers:\n- none")):
            self.snippets._items.append({"id": _new_id(), "trigger": trig, "text": text})

    # ---- settings
    @property
    def last_transcript(self) -> dict | None:
        items = self.history.list(1)
        return items[0] if items else None

    def get(self, key: str):
        return getattr(self.config, key)

    def set(self, key: str, value) -> None:
        if key not in _DEFAULTS:
            raise ValueError(f"Unknown setting “{key}”.")
        if key in _HOTKEY_LABELS and value:
            problem = self.validate_hotkey(key, value)
            if problem:
                raise ValueError(problem)
        if key == "sound_volume" and not (0.0 <= float(value) <= 1.0):
            raise ValueError("Volume must be between 0 and 100%.")
        if key == "history_retention_days" and not (0 <= int(value) <= 3650):
            raise ValueError("History can be kept for at most 10 years.")
        if key == "theme" and value not in ("system", "dark", "light"):
            raise ValueError("Unknown theme.")
        if key in ("update_mode", "update_install_when", "update_time"):
            from .. import config as cfgmod
            value = cfgmod.Config.validate(key, value)       # the same sentences as the real controller
        if key == "sync_folder" and value and not Path(str(value)).is_dir():
            raise ValueError("That folder does not exist any more. Choose another one.")
        if key in ("asr_device", "cleanup_device") and value not in ("auto",) + _DEVICE_BACKENDS[key[:-7]]:
            raise ValueError(f"“{value}” is not a valid choice for {key.replace('_', ' ')}.")
        if getattr(self.config, key) == value:
            return
        setattr(self.config, key, value)
        self.config_changed.emit(key, value)
        if key in ("hotkey", "cleanup_enabled"):
            self.status_changed.emit(self.status())
        if key == "asr_device":                  # speech recognition moves at once, like the real one
            backend = self._device_auto["asr"] if value == "auto" else value
            self.demo_set_model("asr", "ready", _DEVICE_TEXT[backend], None, backend)
        if key in ("asr_device", "cleanup_device"):
            self.devices_changed.emit(self.device_info())
        if key == "sync_folder":
            self._sync_state, self._sync_error = "idle", ""
            self._last_sync = time.time() if value else None
            self.sync_status_changed.emit(self.sync_status())

    def reset_settings(self) -> None:
        for key, default in _DEFAULTS.items():
            if key in ("onboarding_done", "ui_state", "config_version"):
                continue
            if getattr(self.config, key) != default:
                setattr(self.config, key, copy.deepcopy(default))
                self.config_changed.emit(key, default)
        self.status_changed.emit(self.status())

    def validate_hotkey(self, key: str, combo: str) -> str:
        parts = [p for p in (combo or "").lower().split("+") if p]
        if not parts:
            return "" if key != "hotkey" else "The dictation hotkey cannot be empty."
        mods = [p for p in parts if p in _MODS]
        others = [p for p in parts if p not in _MODS]
        fkey = len(others) == 1 and others[0].startswith("f") and others[0][1:].isdigit()
        if not ((mods and others) or len(mods) >= 2 or (fkey and not mods)):
            return "Use at least one modifier (Ctrl, Alt, Shift, Win) plus a key, or two modifiers."
        if len(others) > 1:
            return "Use only one non-modifier key."
        for other, label in _HOTKEY_LABELS.items():
            if other != key and set(str(getattr(self.config, other)).split("+")) == set(parts):
                return f"Already used for {label}."
        if set(parts) in ({"ctrl", "c"}, {"ctrl", "v"}, {"ctrl", "x"}, {"ctrl", "z"}, {"alt", "tab"}, {"alt", "f4"}):
            return f"{self.pretty_hotkey(combo)} is a system shortcut. Pick something else."
        return ""

    def pretty_hotkey(self, combo: str) -> str:
        names = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win", "space": "Space"}
        parts = [p for p in (combo or "").split("+") if p]
        return " + ".join(names.get(p, p.upper() if len(p) <= 3 else p.title()) for p in parts) or "Not set"

    def begin_hotkey_capture(self) -> None:
        self._capturing = True

    def cancel_hotkey_capture(self) -> None:
        if self._capturing:
            self._capturing = False
            self.hotkey_captured.emit("")

    def languages(self) -> list[tuple[str, str]]:
        return [("auto", "Auto-detect"), ("en", "English"), ("de", "German"), ("fr", "French"), ("es", "Spanish"),
                ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"), ("sv", "Swedish"),
                ("da", "Danish"), ("fi", "Finnish"), ("cs", "Czech"), ("el", "Greek"), ("hu", "Hungarian"),
                ("ro", "Romanian"), ("bg", "Bulgarian"), ("hr", "Croatian"), ("sk", "Slovak"), ("sl", "Slovenian"),
                ("et", "Estonian"), ("lv", "Latvian"), ("lt", "Lithuanian"), ("mt", "Maltese"), ("ru", "Russian"),
                ("uk", "Ukrainian")]

    def tones(self) -> list[tuple[str, str]]:
        return [("formal", "Formal"), ("casual", "Casual"), ("very_casual", "Very casual")]

    def categories(self) -> list[tuple[str, str]]:
        return [("personal", "Personal messages"), ("work", "Work messages"), ("email", "Email"), ("other", "Other")]

    FULL_LANGUAGES = ("en", "es", "fr", "de", "it", "pt", "nl")

    def language_support(self, code: str) -> str:
        return "auto" if code == "auto" else "full" if code in self.FULL_LANGUAGES else "basic"

    def preview_tone(self, text: str, tone: str, language: str | None = None) -> str:
        """The real tone pass, exactly as ``Controller.preview_tone`` calls it. A hand-written imitation here
        drifted from it twice (contractions, then accented and hyphenated sentence starts), which made the
        screenshots show behaviour the app does not have."""
        from ..cleanup import apply_tone
        return apply_tone(text, tone, None if language in (None, "auto", "en") else language)

    # ---- status
    def status(self) -> dict:
        if self.paused:
            if self.paused_until is not None:
                at = time.strftime("%H:%M", time.localtime(self.paused_until))
                return {"level": "paused", "title": f"Paused until {at}", "detail": f"Dictation resumes at {at}."}
            return {"level": "paused", "title": "Paused", "detail": "Dictation hotkeys are off until you resume."}
        for name, label in (("asr", "speech"), ("cleanup", "cleanup")):
            st = self._models[name]["state"]
            if st in ("pending", "downloading", "loading"):
                return {"level": "loading", "title": f"Loading {label} model…", "detail": self._models[name]["detail"]}
            if st == "failed" and name == "asr":
                return {"level": "error", "title": "Speech model failed", "detail": self._models[name]["detail"]}
        return {"level": "demo", "title": "Ready", "detail": f"Hold {self.pretty_hotkey(self.config.hotkey)} to dictate"}

    def model_status(self, name: str) -> dict:
        return dict(self._models[name])

    def reload_models(self, which=None) -> None:
        for name in self._models:
            if which is None or name in which:
                self.demo_set_model(name, "ready", "Running on GPU (CUDA)", None, "cuda")

    # ---- devices and the speed check (Controller.device_info)
    def device_info(self) -> dict:
        out = {"forced": self._forced, "check": dict(self._speed)}
        for name, key in (("asr", "asr_device"), ("cleanup", "cleanup_device")):
            results, setting = self._device_results[name], getattr(self.config, key)
            current = self._models[name].get("backend", "")
            target = setting if setting != "auto" else self._device_auto[name]
            timed = {b: r["ms"] for b, r in results.items() if r.get("ms") is not None}
            options = [{"id": b, "measured": b in results,
                        "available": results.get(b, {}).get("available", True),
                        "reason": results.get(b, {}).get("reason", ""), "ms": results.get(b, {}).get("ms"),
                        "failed": bool(results.get(b, {}).get("failed"))} for b in _DEVICE_BACKENDS[name]]
            out[name] = {"setting": setting, "current": current, "next": target, "auto": self._device_auto[name],
                         "fastest": min(timed, key=timed.get) if timed else None,
                         "restart": name == "cleanup" and current in _DEVICE_BACKENDS[name] and target != current,
                         "options": options}
        return out

    def measure_devices(self) -> None:
        if self._speed["state"] == "running":
            return
        self._speed.update(state="running", done=0, total=6, model="asr", backend="cuda")
        self._speed_timer.start()
        self.devices_changed.emit(self.device_info())

    def _speed_step(self) -> None:
        plan = [(m, b) for m in ("asr", "cleanup") for b in _DEVICE_BACKENDS[m]]
        done = self._speed["done"] + 1
        if done >= len(plan):
            self._speed_timer.stop()
            self._speed.update(state="done", done=0, total=0, model="", backend="", measured_at=time.time())
        else:
            self._speed.update(done=done, model=plan[done][0], backend=plan[done][1])
        self.devices_changed.emit(self.device_info())

    def demo_set_devices(self, results=None, auto=None, forced=None, **check) -> None:
        """Pose the device menus: ``results`` {model: {backend: {"ms", "available", "reason", "failed"}}},
        ``auto`` {model: backend}, ``forced`` (a VOXFLOW_DEVICE value) and speed-check fields (``state``...)."""
        if results is not None:
            self._device_results = copy.deepcopy(results)
        if auto is not None:
            self._device_auto.update(auto)
        if forced is not None:
            self._forced = forced
        self._speed.update(check)
        self.devices_changed.emit(self.device_info())

    # ---- audio
    def microphones(self) -> list[dict]:
        return copy.deepcopy(self._mics)

    def refresh_microphones(self) -> None:
        self.microphones_changed.emit(self.microphones())

    def set_level_monitor(self, on: bool) -> None:
        self._monitor = bool(on)

    def level(self) -> float:
        return self._recorder.level() if (self._monitor or self.state in ("listening", "locked")) else 0.0

    def recent_samples(self, n: int = 1024) -> np.ndarray:
        if self.state in ("listening", "locked") or self._monitor:
            return self._recorder.recent_samples(n)
        return np.zeros(n, dtype=np.float32)

    def play_sound(self, name: str) -> None:
        self.played.append(name)

    # ---- actions
    retried = 0

    def retry_saves(self) -> dict:
        """Demo twin of the controller's retry after failed saves."""
        self.retried += 1
        self.notice.emit("success", "Saved.")
        return {}

    def error_info(self) -> dict:
        return dict(getattr(self, "_error_info", {"kind": "", "action": "", "hint": "", "message": ""}))

    def demo_error(self, kind: str = "silence") -> None:
        """Flow-bar error plus the matching machine-readable reason (as the real controller emits)."""
        table = {"silence": ("microphone", "No speech heard. Check your microphone."),
                 "mic": ("microphone", "Microphone disconnected. Pick another one in Settings."),
                 "model": ("models", "The speech model is not ready. Check the models in Settings.")}
        action, hint = table.get(kind, ("", ""))
        self._error_info = {"kind": kind, "action": action, "hint": hint, "message": hint}
        self.error_info_changed.emit(dict(self._error_info))
        self.demo_set_state("error", "dictate", hint.split(".")[0])
        if action:                      # like the controller: the notice first, then its next step
            self.notice.emit("warning", hint)
            self.notice_action.emit(hint, "open_settings:" + action,
                                    "Open microphone settings" if action == "microphone" else "Open model settings")

    def set_paused(self, paused: bool, for_s: float | None = None) -> None:
        until = time.time() + for_s if paused and for_s else None
        if self.paused != bool(paused) or until != self.paused_until:
            self.paused, self.paused_until = bool(paused), until
            self.status_changed.emit(self.status())

    def cancel(self) -> None:
        self.demo_set_state("idle")

    def paste_last(self, entry_id: str | None = None) -> bool:
        entry = self.history.get(entry_id) if entry_id else self.last_transcript
        if not entry:
            self.notice.emit("warning", "There is nothing to paste yet.")
            return False
        return True

    def undo_last_paste(self) -> bool:
        self.notice.emit("info", "Nothing to undo in the preview.")
        return False

    def copy_text(self, text: str) -> bool:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(text)
        return True

    def open_data_folder(self) -> None:
        self.notice.emit("info", f"Your data lives in {self.data_dir}.")

    def open_log_folder(self) -> None:
        self.notice.emit("info", f"Logs are written to {self.log_path.parent}.")

    def start(self) -> None:
        self.status_changed.emit(self.status())

    # ---- app updates (Controller.update_info); nothing is downloaded or installed in the preview
    def update_info(self) -> dict:
        u = self._update
        kind = u["kind"]
        rel = copy.deepcopy(u["available"]) if u["available"] else None
        return {"version": self.version, "mode": self.config.update_mode, "when": self.config.update_install_when,
                "time": self.config.update_time, "kind": kind, "can_install": kind == "installer",
                "managed": _UPDATE_MANAGED.get(kind, ""), "phase": u["phase"], "progress": u["progress"],
                "error": u["error"], "last_check": u["last_check"], "last_ok": u["last_ok"],
                "last_result": u["last_result"], "available": rel,
                "installable": bool(rel and kind == "installer" and rel.get("installer_url")),
                "skipped": bool(rel and u["skipped"]), "install_on_quit": bool(rel and u["install_on_quit"]),
                "downloaded": bool(rel and u["downloaded"]),
                "releases_page": rel["page"] if rel else "https://github.com/blackknight75/voxflow/releases"}

    def _update_emit(self) -> None:
        self.update_changed.emit(self.update_info())

    def check_updates(self) -> bool:
        u = self._update
        u.update(last_check=time.time(), last_ok=True, available=copy.deepcopy(_DEMO_RELEASE), error="",
                 last_result=f"VoxFlow {_DEMO_RELEASE['version']} is available.")
        self._update_emit()
        self.update_available.emit(self.update_info())
        return True

    def install_update(self, when: str = "now") -> None:
        info = self.update_info()
        if when not in ("now", "quit"):
            raise ValueError("Choose to install now or when you quit VoxFlow.")
        if info["available"] is None:
            raise ValueError("There is no newer version to install.")
        if not info["can_install"]:
            raise ValueError(info["managed"])
        self._update.update(skipped=False, downloaded=True, install_on_quit=when == "quit",
                            phase="waiting" if when == "now" else "idle")
        self._update_emit()
        self.notice.emit("info", "The preview does not install anything.")

    def remind_update_later(self) -> None:
        self._update_emit()

    def skip_update(self) -> None:
        if self._update["available"]:
            self._update.update(skipped=True, install_on_quit=False, downloaded=False)
            self._update_emit()

    def prepare_quit(self) -> None:
        self.quit_prepared = True

    def demo_set_update(self, available: bool | dict = True, **fields) -> None:
        """Pose Settings > Updates: ``available`` (True = the demo release, a dict = that release, False = none) and
        any update_info field (``kind``, ``phase``, ``progress``, ``error``, ``last_ok``, ``last_result``...)."""
        rel = copy.deepcopy(_DEMO_RELEASE) if available is True else (dict(available) if available else None)
        self._update.update(available=rel, **fields)
        self._update_emit()

    def shutdown(self) -> None:
        self._idle_timer.stop()
        self._speed_timer.stop()

    # ---- backup bundle, shareable packs, sync folder
    BACKUP_SUFFIX = ".voxbackup"
    PACK_SUFFIX = ".voxpack"

    @staticmethod
    def _read_json(path: str, what: str) -> dict:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except OSError as exc:
            raise ValueError(f"{Path(path).name} could not be read: {exc.strerror or exc}.") from exc
        except ValueError as exc:
            raise ValueError(f"{Path(path).name} is not a VoxFlow {what}.") from exc
        if not isinstance(data, dict) or data.get("voxflow") != what:
            raise ValueError(f"{Path(path).name} is not a VoxFlow {what}.")
        return data

    @staticmethod
    def _write_json(path: str, data: dict) -> None:
        try:
            Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{Path(path).name} could not be written: {exc.strerror or exc}.") from exc

    def _counts(self) -> dict:
        return {"settings": len(_DEFAULTS), "dictionary": len(self.dictionary._items),
                "snippets": len(self.snippets._items), "styles": len(self.styles._overrides),
                "stats": len(self.stats._daily), "history": len(self.history._items)}

    def export_backup(self, path: str, include_history: bool = True) -> dict:
        if not Path(path).suffix:
            path += self.BACKUP_SUFFIX
        self._write_json(path, {
            "voxflow": "backup", "version": self.version, "created": time.time(),
            "settings": {k: getattr(self.config, k) for k in _DEFAULTS if k not in ("ui_state", "overlay_position")},
            "device": "this-pc", "dictionary": self.dictionary.list(), "snippets": self.snippets.list(),
            "styles": {"tones": self.styles.tones(), "apps": dict(self.styles._overrides)},
            **({"history": self.history.list(None),
                "stats": {"daily": copy.deepcopy(self.stats._daily), "apps": copy.deepcopy(self.stats._apps)}}
               if include_history else {})})
        counts = self._counts()
        if not include_history:
            counts["history"] = counts["stats"] = 0
        return {"path": path, "counts": counts,
                "includes": ["settings", "dictionary", "snippets", "styles"] + (["stats", "history"] if include_history else [])}

    def inspect_backup(self, path: str) -> dict:
        d = self._read_json(path, "backup")
        return {"created": float(d.get("created", 0)), "version": str(d.get("version", "")),
                "device": str(d.get("device", "")), "same_device": d.get("device") == "this-pc",
                "counts": {"settings": len(d.get("settings") or {}), "dictionary": len(d.get("dictionary") or []),
                           "snippets": len(d.get("snippets") or []),
                           "styles": len((d.get("styles") or {}).get("apps") or {}),
                           "stats": len((d.get("stats") or {}).get("daily") or []),
                           "history": len(d.get("history") or [])},
                "includes": [k for k in ("settings", "dictionary", "snippets", "styles", "stats", "history") if k in d],
                "skipped": 0}

    def import_backup(self, path: str, mode: str = "merge", parts=None) -> dict:
        if mode not in ("merge", "replace"):
            raise ValueError("Choose whether to merge the backup or replace your profile.")
        d = self._read_json(path, "backup")
        if parts is not None:                    # only these parts of the file count
            d = {k: v for k, v in d.items() if k in ("voxflow", "version", "created", "device") or k in parts}
        counts = {"settings": 0, "dictionary": 0, "snippets": 0, "styles": 0, "stats": 0, "history": 0}
        if mode == "replace":
            for store, key in ((self.dictionary, "dictionary"), (self.snippets, "snippets"), (self.history, "history")):
                if key in d:
                    store._items = []
        have = {e["term"].lower() for e in self.dictionary._items}
        for e in d.get("dictionary") or []:
            if e.get("term") and e["term"].lower() not in have:
                self.dictionary._items.append({"id": _new_id(), "term": e["term"], "aliases": list(e.get("aliases") or [])})
                counts["dictionary"] += 1
        have = {e["trigger"].lower() for e in self.snippets._items}
        for e in d.get("snippets") or []:
            if e.get("trigger") and e["trigger"].lower() not in have:
                self.snippets._items.append({"id": _new_id(), "trigger": e["trigger"], "text": e.get("text", "")})
                counts["snippets"] += 1
        have = {e["id"] for e in self.history._items}
        for e in d.get("history") or []:
            if e.get("id") not in have:
                self.history._items.append(dict(e))
                counts["history"] += 1
        self.history._items.sort(key=lambda e: -float(e.get("ts", 0)))
        if mode == "replace":
            for key, value in (d.get("settings") or {}).items():
                if key in _DEFAULTS and key != "onboarding_done" and getattr(self.config, key) != value:
                    setattr(self.config, key, value)
                    self.config_changed.emit(key, value)
            if "styles" in d:
                self.styles._tones.update(d["styles"].get("tones") or {})
                self.styles._overrides = dict(d["styles"].get("apps") or {})
                self.styles_changed.emit()
        for sig in (self.dictionary_changed, self.snippets_changed, self.history_changed, self.stats_changed):
            sig.emit()
        return {"mode": mode, "counts": counts, "settings_skipped": [], "skipped": 0}

    def export_pack(self, kind: str, path: str, name: str = "", description: str = "", ids=None) -> int:
        if kind not in ("dictionary", "snippets"):
            raise ValueError("Only the dictionary and snippets can be shared as a pack.")
        wanted = None if ids is None else set(ids)
        items = ([{"term": e["term"], "aliases": e["aliases"]} for e in self.dictionary.list()
                  if wanted is None or e["id"] in wanted] if kind == "dictionary"
                 else [{"trigger": e["trigger"], "text": e["text"]} for e in self.snippets.list()
                       if wanted is None or e["id"] in wanted])
        if not items:
            raise ValueError("There is nothing to share yet.")
        if not Path(path).suffix:
            path += self.PACK_SUFFIX
        self._write_json(path, {"voxflow": "pack", "kind": kind, "name": name.strip(), "created": time.time(),
                                "description": description.strip(), "items": items})
        return len(items)

    def inspect_pack(self, path: str) -> dict:
        d = self._read_json(path, "pack")
        kind, items = str(d.get("kind", "")), list(d.get("items") or [])
        store, key = (self.dictionary, "term") if kind == "dictionary" else (self.snippets, "trigger")
        have = {e[key].lower() for e in store._items}
        new = sum(1 for e in items if str(e.get(key, "")).lower() not in have)
        return {"kind": kind, "name": str(d.get("name", "")), "description": str(d.get("description", "")),
                "count": len(items), "created": float(d.get("created", 0)), "new": new,
                "existing": len(items) - new, "skipped": 0}

    def import_pack(self, path: str, mode: str = "merge") -> dict:
        if mode not in ("merge", "replace"):
            raise ValueError("Choose whether to merge the pack or replace your list.")
        d = self._read_json(path, "pack")
        kind = d.get("kind")
        store, key = (self.dictionary, "term") if kind == "dictionary" else (self.snippets, "trigger")
        if kind not in ("dictionary", "snippets"):
            raise ValueError(f"{Path(path).name} is not a dictionary or snippet pack.")
        removed = 0
        if mode == "replace":
            keep = {str(e.get(key, "")).lower() for e in d.get("items") or []}
            removed = sum(1 for e in store._items if e[key].lower() not in keep)
            store._items = [e for e in store._items if e[key].lower() in keep]
        added = skipped = 0
        have = {e[key].lower() for e in store._items}
        for e in d.get("items") or []:
            if not e.get(key) or e[key].lower() in have:
                skipped += 1
                continue
            have.add(e[key].lower())
            store._items.append({"id": _new_id(), **({"term": e["term"], "aliases": list(e.get("aliases") or [])}
                                                    if kind == "dictionary" else
                                                    {"trigger": e["trigger"], "text": e.get("text", "")})})
            added += 1
        (self.dictionary_changed if kind == "dictionary" else self.snippets_changed).emit()
        return {"kind": kind, "name": str(d.get("name", "")), "added": added, "updated": 0, "removed": removed,
                "skipped": 0}

    def sync_status(self) -> dict:
        folder = str(self.config.sync_folder or "")
        base = {"enabled": bool(folder), "device": "this-pc", "devices": self._sync_devices if folder else [],
                "unreadable": 0, "newer": 0}
        if not folder:
            return {"state": "off", "folder": "", "last_sync": None, "error": "", "message": "Sync is off.", **base}
        n = len(self._sync_devices)
        message = {"syncing": "Syncing…", "error": self._sync_error}.get(
            self._sync_state, f"Up to date with {n} other device{'s' if n != 1 else ''}." if n else
            "Up to date. No other device has used this folder yet.")
        return {"state": self._sync_state, "folder": folder, "last_sync": self._last_sync, "error": self._sync_error,
                "message": message, **base}

    def sync_now(self) -> None:
        if not self.config.sync_folder:
            self.notice.emit("info", "Choose a sync folder first.")
            return
        self._sync_state, self._sync_error, self._last_sync = "idle", "", time.time()
        self.sync_status_changed.emit(self.sync_status())

    def demo_set_sync(self, state: str, error: str = "", last_sync: float | None = None, devices=None) -> None:
        self._sync_state, self._sync_error = state, error
        if devices is not None:
            self._sync_devices = [{"name": n} for n in devices]
        if last_sync is not None:
            self._last_sync = last_sync
        self.sync_status_changed.emit(self.sync_status())

    # ---- demo helpers
    def demo_set_state(self, state: str, mode: str = "dictate", message: str = "") -> None:
        self._idle_timer.stop()
        self.state, self.mode, self.message = state, mode, message
        self.state_changed.emit(state, mode, message)

    def demo_set_model(self, name: str, state: str, detail: str = "", progress: float | None = None,
                       backend: str = "") -> None:
        self._models[name] = {"state": state, "detail": detail, "progress": progress, "backend": backend}
        self.model_status_changed.emit(name, dict(self._models[name]))
        self.status_changed.emit(self.status())
        self.devices_changed.emit(self.device_info())

    def demo_add_transcript(self, text: str = "Let's ship the new build on Friday.", rules_only: bool = False) -> dict:
        words = len(text.split())
        entry = {"id": _new_id(), "ts": time.time(), "raw": text.lower().rstrip("."), "text": text,
                 "app_exe": "slack.exe", "app_title": "Slack | #general", "duration_s": round(words / 2.4, 1),
                 "words": words, "wpm": 144.0, "mode": "dictate", "latency_ms": 512}
        if rules_only:                           # like the real store: the key exists only when it is True
            entry["rules_only"] = True
        if self.config.save_history:
            self.history._items.insert(0, entry)
            self.history_changed.emit()
        if self.stats._daily:
            self.stats._daily[-1]["words"] += words
            self.stats._daily[-1]["sessions"] += 1
        self.stats_changed.emit()
        self.transcript_ready.emit(dict(entry))
        return entry
