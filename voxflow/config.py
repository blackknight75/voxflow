"""Persistent configuration (JSON under %APPDATA%/VoxFlow): defaults, validation and migration."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger("voxflow.config")

CONFIG_VERSION = 2

# Parakeet TDT 0.6b v3 covers these 25 European languages (it detects the language by itself; the setting
# tells the clean-up pass which language rules apply).
LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Auto-detect"), ("en", "English"), ("bg", "Bulgarian"), ("hr", "Croatian"), ("cs", "Czech"),
    ("da", "Danish"), ("nl", "Dutch"), ("et", "Estonian"), ("fi", "Finnish"), ("fr", "French"),
    ("de", "German"), ("el", "Greek"), ("hu", "Hungarian"), ("it", "Italian"), ("lv", "Latvian"),
    ("lt", "Lithuanian"), ("mt", "Maltese"), ("pl", "Polish"), ("pt", "Portuguese"), ("ro", "Romanian"),
    ("ru", "Russian"), ("sk", "Slovak"), ("sl", "Slovenian"), ("es", "Spanish"), ("sv", "Swedish"),
    ("uk", "Ukrainian"),
]
LANGUAGE_CODES = tuple(code for code, _ in LANGUAGES)
THEMES = ("system", "dark", "light")
HOTKEY_KEYS = ("hotkey", "command_hotkey", "paste_last_hotkey", "undo_hotkey")
# App updates (voxflow/updater.py): install automatically, notify and let the user choose, or never check.
UPDATE_MODES = ("auto", "notify", "off")
DEFAULT_UPDATE_MODE = "notify"      # confirmed by the maintainer; the one place the default is decided
UPDATE_WHEN = ("quit", "time")      # automatic installs: when VoxFlow is quit, or at update_time when idle
HOTKEY_LABELS = {"hotkey": "Dictation", "command_hotkey": "Command mode",
                 "paste_last_hotkey": "Paste last transcript", "undo_hotkey": "Undo last paste"}


def app_dir_path() -> Path:
    """Where VoxFlow keeps its data (``%APPDATA%\\VoxFlow``), without creating it."""
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "VoxFlow"


def app_dir() -> Path:
    p = app_dir_path()
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_path() -> Path:
    return app_dir() / "config.json"


def history_path() -> Path:
    return app_dir() / "history.json"


@dataclass
class Config:
    # ---- hotkeys
    hotkey: str = "ctrl+win"
    command_hotkey: str = "alt+win"
    paste_last_hotkey: str = "alt+shift+z"
    undo_hotkey: str = ""
    double_tap_lock: bool = True
    # ---- dictation
    cleanup_enabled: bool = True
    smart_formatting: bool = True
    auto_learn: bool = True
    language: str = "auto"
    whisper_mode: bool = False
    mic_device: Any = None            # None = system default, device name (str) or legacy int index
    # ---- sound
    sounds_enabled: bool = True
    sound_volume: float = 0.5
    mute_media: bool = False
    # ---- system
    launch_at_login: bool = False
    start_minimized: bool = False
    # ---- history
    save_history: bool = True
    history_retention_days: int = 0   # 0 = keep forever
    history_limit: int = 5000
    # ---- pasting
    restore_clipboard: bool = True
    restore_delay_s: float = 0.5      # wait after Ctrl+V before the old clipboard is put back
    # ---- interface (stored for the UI)
    show_idle_pill: bool = False      # the Flow bar only appears while a hotkey is in use
    live_preview: bool = True         # the Flow bar shows the words recognised so far while the key is held
    theme: str = "system"
    reduced_motion: bool = False
    overlay_position: Any = None      # None or [x, y]
    ui_state: dict = field(default_factory=dict)
    onboarding_done: bool = False
    # ---- updates (voxflow/updater.py)
    update_mode: str = DEFAULT_UPDATE_MODE
    update_install_when: str = "quit"
    update_time: str = "03:00"        # HH:MM, local time
    # ---- advanced
    asr_model: str = "nemo-parakeet-tdt-0.6b-v3"
    asr_device: str = "auto"          # auto = the speed check's measured choice, else the fixed order
    cleanup_model: Any = None
    cleanup_device: str = "auto"      # read once at start: llama.cpp picks its build once per process
    cleanup_timeout_s: float = 4.0
    command_timeout_s: float = 20.0
    min_record_s: float = 0.3
    max_record_s: float = 120.0       # hard cap per recording (lost key-up / forgotten hands-free lock)
    config_version: int = CONFIG_VERSION

    # ---- validation --------------------------------------------------------
    @classmethod
    def validate(cls, key: str, value: Any) -> Any:
        """Return the normalised value for ``key`` or raise ValueError with a human-readable message."""
        defaults = _DEFAULTS
        if key not in defaults:
            raise ValueError(f"Unknown setting “{key}”.")
        default = defaults[key]
        if key in HOTKEY_KEYS:
            return _valid_hotkey(key, value)
        if key == "mic_device":
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                return value
            if isinstance(value, str):
                return value.strip() or None
            raise ValueError("The microphone must be a device name.")
        if key == "cleanup_model":
            if value is None or (isinstance(value, str) and value.strip()):
                return value.strip() if isinstance(value, str) else None
            raise ValueError("The cleanup model must be a “repo:file.gguf” name or a file path.")
        if key == "overlay_position":
            if value is None:
                return None
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                            for v in value)):
                return [int(value[0]), int(value[1])]
            raise ValueError("The overlay position must be two numbers.")
        if key == "ui_state":
            if not isinstance(value, dict):
                raise ValueError("The interface state must be a dictionary.")
            try:
                return json.loads(json.dumps(value))
            except (TypeError, ValueError):
                raise ValueError("The interface state must contain only plain values.") from None
        if key == "update_time":
            m = re.fullmatch(r"\s*([01]?\d|2[0-3]):([0-5]\d)\s*", value) if isinstance(value, str) else None
            if not m:
                raise ValueError("The install time must be a time of day such as 03:00.")
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        if key in _CHOICES:
            if isinstance(value, str) and value.strip().lower() in _CHOICES[key]:
                return value.strip().lower()
            raise ValueError(f"“{value}” is not a valid choice for {key.replace('_', ' ')}.")
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be on or off.")
        if isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a number.")
            lo, hi = _RANGES.get(key, (None, None))
            if lo is not None:
                value = min(max(value, lo), hi)        # also brings +-Infinity back into range
            elif not math.isfinite(value):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a number.")
            return int(value) if isinstance(default, int) else float(value)
        if isinstance(default, str):
            if isinstance(value, str) and value.strip():
                return value.strip()
            raise ValueError(f"{key.replace('_', ' ').capitalize()} cannot be empty.")
        return value

    # ---- (de)serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict | None, rejected: list | None = None) -> "Config":
        """Build a Config from stored JSON: migrates old layouts and keeps the default for an invalid value (its
        key is appended to ``rejected``). "false" / "off" and "100" written by hand or by a deployment script are
        read as the setting's type. Keys this version does not know are kept for ``save_config`` (a newer
        VoxFlow wrote them), and ``config_version`` is never lowered."""
        cfg = cls()
        if not isinstance(data, dict):
            return cfg
        data = migrate(data)
        known = {f.name for f in fields(cls)}
        cfg._extras = {k: v for k, v in data.items() if k not in known}
        cfg.config_version = max(CONFIG_VERSION, data["config_version"])
        for f in fields(cls):
            if f.name not in data or f.name == "config_version":
                continue
            try:
                setattr(cfg, f.name, cls.validate(f.name, _from_text(f.name, data[f.name])))
            except (ValueError, TypeError, OverflowError):
                if rejected is not None:
                    rejected.append(f.name)      # keep the default for anything unusable, and say so
        _drop_duplicate_hotkeys(cfg)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


_DEFAULTS = asdict(Config())
_CHOICES = {"language": LANGUAGE_CODES, "theme": THEMES, "asr_device": ("auto", "cuda", "webgpu", "cpu"),
            "cleanup_device": ("auto", "cuda", "vulkan", "cpu"), "update_mode": UPDATE_MODES,
            "update_install_when": UPDATE_WHEN}
_RANGES = {
    "sound_volume": (0.0, 1.0), "history_retention_days": (0, 3650), "history_limit": (10, 100000),
    "restore_delay_s": (0.05, 5.0), "cleanup_timeout_s": (0.5, 30.0), "command_timeout_s": (2.0, 120.0),
    "min_record_s": (0.05, 5.0), "max_record_s": (5.0, 3600.0),
}


_TRUE, _FALSE = ("true", "on", "yes", "1"), ("false", "off", "no", "0")


def _from_text(key: str, value: Any) -> Any:
    """A boolean or a number written as text in the file ("false", "100", "0.8") as the setting's type."""
    default = _DEFAULTS.get(key)
    if not isinstance(value, str) or key in _CHOICES or not isinstance(default, (bool, int, float)):
        return value
    text = value.strip().lower()
    if isinstance(default, bool):
        return True if text in _TRUE else False if text in _FALSE else value
    try:
        return float(text)
    except ValueError:
        return value


def defaults() -> dict:
    """A fresh copy of every default value."""
    return asdict(Config())


def _valid_hotkey(key: str, value: Any) -> str:
    from .hotkey import hotkey_problem, normalize_hotkey

    if not isinstance(value, str):
        raise ValueError("A shortcut must be a key combination such as Ctrl + Win.")
    if not value.strip():
        if key == "hotkey":
            raise ValueError("The dictation shortcut cannot be empty.")
        return ""
    problem = hotkey_problem(value)
    if problem:
        raise ValueError(problem)
    return normalize_hotkey(value)


def _drop_duplicate_hotkeys(cfg: Config) -> None:
    """Two actions can never share a shortcut: later duplicates fall back to their default, else to off."""
    from .hotkey import combos_clash

    taken: list[str] = []
    for key in HOTKEY_KEYS:
        combo = getattr(cfg, key)
        if combo and any(combos_clash(combo, other) for other in taken):
            fallback = _DEFAULTS[key]
            combo = fallback if fallback and not any(combos_clash(fallback, o) for o in taken) else ""
            setattr(cfg, key, combo)
        if combo:
            taken.append(combo)


def migrate(data: dict) -> dict:
    """Upgrade a stored config dict to the current layout (pure; returns a new dict)."""
    out = dict(data)
    version = out.get("config_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        version = 1
    if version < 2:
        # v1 kept at most 200 transcripts because history was a flat list shown in one small window.
        if out.get("history_limit") == 200:
            out["history_limit"] = _DEFAULTS["history_limit"]
        version = 2
    out["config_version"] = version
    return out


_SHARING_ERRORS = (32, 33)        # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION: someone has the file open


def write_problem(path: Path) -> str:
    """Why ``path`` cannot be written at all ("" if it can, or if it does not exist): a rename onto it would keep
    failing, so waiting only freezes the caller. Checks the read-only attribute and, on Windows, whether the
    file's permissions allow writing (opening it for writing, sharing everything, changes nothing)."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    if getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_READONLY:
        return f"{Path(path).name} is read-only"
    if sys.platform != "win32":
        return ""
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)          # a private copy: argtypes set here stay here
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    # GENERIC_WRITE, FILE_SHARE_READ | WRITE | DELETE, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL
    handle = k32.CreateFileW(str(path), 0x40000000, 7, None, 3, 0x80, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        return f"Windows does not allow VoxFlow to change {Path(path).name}" if ctypes.get_last_error() == 5 else ""
    k32.CloseHandle(handle)
    return ""


def _replace(tmp: str, path: Path) -> None:
    """os.replace that waits out a reader. On Windows the rename fails with PermissionError while another
    thread, a sync tool, a backup program or a virus scanner has the target open for reading; that is over
    in milliseconds, so retry for up to ~2 s instead of losing the save. A file that cannot be written at all
    (read-only, permissions) fails at once: waiting for it froze the window for 3.6 s per change (QA R6)."""
    delay = 0.005
    for attempt in range(40):  # pragma: no branch - the last attempt returns or re-raises
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            if attempt == 39:
                raise
            if getattr(exc, "winerror", None) not in _SHARING_ERRORS and write_problem(path):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())               # a power cut must not leave a zero-length file after the rename
            except OSError:
                pass
        _replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


last_load_problem = ""   # human message when the last load_config() fell back to defaults; "" otherwise


def load_config(path: Path | None = None) -> Config:
    global last_load_problem
    path = Path(path) if path else config_path()
    last_load_problem = ""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))     # tolerate a BOM added by an editor
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        cfg = Config()
        if not path.exists():
            try:
                save_config(cfg, path)
            except OSError:
                pass
        else:
            # keep the unreadable file: the next save would otherwise overwrite the user's settings for good
            log.warning("config file %s is unreadable (%s); using defaults, copy kept as %s.bad",
                        path, exc, path.name)
            try:
                path.with_name(path.name + ".bad").write_bytes(path.read_bytes())
            except OSError:
                pass
            last_load_problem = ("Your settings file could not be read, so defaults are in use. "
                                 f"The old file was kept as {path.name}.bad.")
        return cfg
    rejected: list[str] = []
    cfg = Config.from_dict(data, rejected)
    if rejected:
        log.warning("config file %s: invalid values reset to their defaults: %s", path, ", ".join(rejected))
        last_load_problem = ("Some settings in your settings file were not valid and are back to their defaults: "
                             + ", ".join(k.replace("_", " ") for k in rejected) + ".")
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = Path(path) if path else config_path()
    # settings a newer VoxFlow wrote are kept as they are, so going back to it loses nothing
    _atomic_write(path, json.dumps({**getattr(cfg, "_extras", {}), **cfg.to_dict()}, indent=2))


def __getattr__(name: str):
    # The transcript history moved to voxflow.store; keep ``voxflow.config.History`` importable.
    if name == "History":
        from .store import History
        return History
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
