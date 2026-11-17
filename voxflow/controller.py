"""The single façade between the user interface and everything else.

``Controller`` owns the configuration, the stores, the recorder, the models, the hotkeys and the pipeline
worker. The UI connects to its signals and calls its methods; it never reaches into the pipeline.
The public signals, properties and methods of ``Controller`` are the contract.

Threads: the controller lives on the Qt (UI) thread. Hotkey events arrive from the hook thread and results
from the pipeline worker; both are forwarded through private queued signals, so every public signal is
emitted on the UI thread.
"""
from __future__ import annotations

import copy
import logging
import queue
import shutil
import tempfile
import os
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from . import __version__, audio as audiomod, config as cfgmod, fakes, langs, modelhub, speedcheck, store, system
from . import sync as syncmod
from . import updater as updatermod
from .command import CommandError, CommandMode
from .context import AppContext, foreground_app, foreground_hwnd
from .hotkey import HotkeyListener, hotkey_problem, normalize_hotkey, pretty_hotkey
from .sounds import MediaMuter, SoundPlayer

log = logging.getLogger("voxflow.controller")

STATES = ("idle", "listening", "locked", "processing", "done", "error")
MODES = ("dictate", "command")
DONE_MS, ERROR_MS = 1100, 2200
_MODEL_KEYS = ("asr_model", "asr_device", "cleanup_model")
_KEPT_ON_RESET = ("config_version", "onboarding_done", "ui_state", "overlay_position")
SYNC_INTERVAL_MS, SYNC_DEBOUNCE_MS = 30_000, 2_000
SPEED_DELAY_MS = 30_000       # idle time after both models are ready before the automatic speed check starts
SPEED_RETRY_MS = 15_000       # how soon a speed check that had to wait for a dictation looks again
_DIAG = os.environ.get("VOXFLOW_DIAG") == "1"      # extra once-a-minute CPU line in the log, for idle-cost hunts


class _RealAutostart:
    def enabled(self) -> bool:
        return system.autostart_enabled()

    def set(self, enabled: bool) -> None:
        system.set_autostart(enabled)


class Controller(QObject):
    BACKUP_SUFFIX, PACK_SUFFIX = syncmod.PROFILE_EXT, syncmod.PACK_EXT      # for file dialogs

    # ---- public signals (UI thread)
    state_changed = Signal(str, str, str)          # state, mode, message
    level_changed = Signal(float)
    notice_action = Signal(str, str, str)   # (message, action_id, label): sent right after the `notice` with that message
    error_info_changed = Signal(dict)       # {"kind", "action", "hint", "message"} just before state "error"
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
    notice = Signal(str, str)                       # level, message
    activate_requested = Signal()
    sync_status_changed = Signal(dict)              # sync_status()
    devices_changed = Signal(dict)                  # device_info()
    update_changed = Signal(dict)                   # update_info()
    update_available = Signal(dict)                 # update_info(), when a newer version should be shown to the user
    restart_requested = Signal()                    # quit now: the update installer runs once VoxFlow has shut down

    # ---- private cross-thread plumbing
    _sig_hotkey = Signal(str, str)                  # binding, event
    _sig_captured = Signal(str)
    _sig_store = Signal(str)
    _sig_suggestion = Signal(dict)
    _sig_model = Signal(str, dict)
    _sig_done = Signal(dict)
    _sig_failed = Signal(str, int)
    _sig_notice = Signal(str, str)
    _sig_sync = Signal(str, object)                 # step ("read" | "written"), result or exception
    _sig_speed = Signal(dict)                       # an event of the speed check (voxflow.speedcheck.SpeedCheck)

    def __init__(self, *, demo: bool = False, fake: bool = False, data_dir: str | Path | None = None,
                 install_hook: bool = True, parent=None, modules=None, injector=None, sound_backend=None,
                 mute_backend=None, context_fn=None, autostart=None, speed_check: bool | None = None,
                 updates=None):
        """``modules`` = (RecorderClass, transcriber, cleaner) overrides model/mic construction; ``injector``,
        ``sound_backend``, ``mute_backend``, ``context_fn`` and ``autostart`` replace the desktop back-ends.
        With ``demo`` or ``fake`` every desktop back-end that was not passed in is a harmless fake.
        ``speed_check``: run the automatic speed check (default: only with the real models, and not when
        ``VOXFLOW_SPEED_CHECK=0``). ``updates``: an ``updater.Updater`` (or a factory ``(data_dir, get, busy)``
        returning one) to use, or a bool for the daily update
        check (default: only the real app, and not when ``VOXFLOW_UPDATES=0``). A fake or demo copy never
        reaches the network for updates, not even through "Check now"."""
        super().__init__(parent)
        self.is_demo = bool(demo)
        self._fake = bool(fake or demo)
        self.version = __version__
        self._tmp_dir: Path | None = None
        if demo and data_dir is None:
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="voxflow-demo-"))
            data_dir = self._tmp_dir
        self.data_dir = Path(data_dir) if data_dir is not None else cfgmod.app_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = system.log_dir(self.data_dir) / system.LOG_NAME

        self._config_path = self.data_dir / "config.json"
        self.config = cfgmod.load_config(self._config_path)
        self._startup_notices: list[tuple[str, str]] = []
        if getattr(cfgmod, "last_load_problem", ""):
            self._startup_notices.append(("warning", cfgmod.last_load_problem))
        cfg = self.config

        # ---- stores
        self.dictionary = store.Dictionary(self.data_dir)
        self.snippets = store.Snippets(self.data_dir)
        self._rule_cleaner = None                     # model-free clean-up, for a trigger the full one reworded
        self.history = store.History(self.data_dir, limit=cfg.history_limit,
                                     retention_days=cfg.history_retention_days)
        self.stats = store.Stats(self.data_dir)
        self.styles = store.Styles(self.data_dir)
        if self.stats.is_new and self.history.count():
            self.stats.seed_from_history(self.history.snapshot())
        if demo:
            _seed_demo(self)
        for st in (self.dictionary, self.snippets, self.history, self.stats, self.styles):
            st.subscribe(self._sig_store.emit)
        self.last_transcript: dict | None = next(iter(self.history.list(limit=1)), None)

        # ---- profiles, packs and the sync folder
        self._portable = syncmod.DataSet(self.dictionary, self.snippets, self.styles, self.history,
                                         lambda: self.config.to_dict(), self._apply_settings, self.stats)
        self.sync = syncmod.SyncEngine(self._portable, self.data_dir / "sync_state.json")
        self._sync_busy = False
        self._sync_applying = False
        self._sync_again = False

        # ---- desktop back-ends
        self._autostart = autostart if autostart is not None else (
            fakes.FakeAutostart(cfg.launch_at_login) if self._fake else _RealAutostart())
        if not self._fake:
            cfg.launch_at_login = self._autostart.enabled()     # the registry is the source of truth
        self._context_fn = context_fn if context_fn is not None else (
            (lambda: AppContext("notepad.exe", "Untitled - Notepad", 1)) if self._fake else foreground_app)
        self._hwnd_fn = (lambda: self._context_fn().hwnd) if (context_fn is not None or self._fake) else foreground_hwnd
        if injector is None:
            if self._fake:
                injector = fakes.FakeInjector()
            else:
                from .inject import Injector
                injector = Injector(restore=cfg.restore_clipboard, restore_delay=cfg.restore_delay_s)
        self.injector = injector
        self.injector.restore = cfg.restore_clipboard
        self.injector.restore_delay = cfg.restore_delay_s
        if sound_backend is None and self._fake:
            sound_backend = lambda wav: None      # noqa: E731 - silent
        self.sounds = SoundPlayer(cfg.sounds_enabled, cfg.sound_volume, backend=sound_backend)
        if mute_backend is None and self._fake:
            mute_backend = fakes.FakeMuteBackend()
        self.muter = MediaMuter(self.data_dir, backend=mute_backend)

        # ---- speed check: where each model runs fastest on this PC (voxflow.speedcheck); read before the
        # models are built, because "auto" uses its stored result
        if speed_check is None:
            speed_check = modules is None and not self._fake and os.environ.get("VOXFLOW_SPEED_CHECK") != "0"
        self._speed_auto = bool(speed_check)
        self._speed_manual = self._speed_auto or self.is_demo          # "Measure again" (demo: simulated)
        self._speed_machine = {"demo": True} if demo else speedcheck.machine() if self._speed_manual else {}
        self._speed_record = speedcheck.load(self.data_dir) if self._speed_manual and not demo else None
        if demo:
            self._speed_record = speedcheck.make_record(self._speed_key(), self._speed_machine,
                                                        self._speed_models(), speedcheck.demo_results(), 41.0)
        self._speed = speedcheck.SpeedCheck(self._sig_speed.emit,
                                            speedcheck.demo_child if demo else speedcheck.run_child)
        self._speed_state = {"state": "idle", "done": 0, "total": 0, "model": "", "backend": ""}
        self._speed_wanted = False                     # the user asked for a measurement
        self._speed_run: tuple[str, dict] = ("", {})   # fingerprint and model names of the run under way
        for name, device in self._measured_devices().items():
            if device:
                log.info("speed check: %s uses %s on this PC (measured faster than the automatic choice)",
                         "speech recognition" if name == "asr" else "the language model", device)

        # ---- app updates (voxflow/updater.py); the setting "Off" means no update traffic at all
        self._quit_by_user = False
        if isinstance(updates, updatermod.Updater):
            self.updates = updates
        elif callable(updates):                        # a factory (tests): (data_dir, get, busy) -> Updater
            self.updates = updates(self.data_dir, self.get, self._update_busy)
        else:
            if updates is None:
                updates = modules is None and not self._fake and os.environ.get("VOXFLOW_UPDATES") != "0"
            self.updates = updatermod.Updater(
                self.data_dir, self.get, self._update_busy, automatic=bool(updates),
                http=None if updates else updatermod.OfflineHttp(),
                kind=None if updates else "source",
                folder=None if updates else self.data_dir / "update-offline", parent=self)
        self.updates.changed.connect(self.update_changed)
        self.updates.available.connect(self.update_available)
        self.updates.notice.connect(self.notice)
        self.updates.restart_requested.connect(self._on_update_restart)

        # ---- models and microphone
        self._modules_override = modules
        self._Rec = self._asr = self._cleaner = None
        self._stub_notes: list[str] = []
        self._build_modules()
        self._recorder = self._make_recorder(cfg.mic_device)
        self._mic_count = -1
        self.command = CommandMode(self._cleaner, self.injector, cfg.command_timeout_s)
        self._models = {
            "asr": {"state": "pending", "detail": "Waiting to load", "progress": None, "backend": ""},
            "cleanup": {"state": "pending", "detail": "Waiting to load", "progress": None, "backend": ""},
        }
        self._load_generation = 0

        # ---- session state
        self.state, self.mode, self.message = "idle", "dictate", ""
        self.paused = False
        self.paused_until: float | None = None         # wall-clock time a timed pause ends (set_paused(for_s=...))
        self._problems: dict[str, str] = {}            # subsystem -> human message (hook, asr, mic)
        self._warnings: dict[str, str] = {}
        self._gen = 0
        self._active = False
        self._t_start = 0.0
        self._context = AppContext()
        self._last_paste: dict | None = None
        self.rules_only_count = 0                      # dictations of this session cleaned by rules only
        self._rules_only_times: list[float] = []
        self._rules_only_told: float | None = None
        self._mono = time.monotonic
        self._monitor = False
        self._started = False
        self._closed = False
        self._capturing = False

        # ---- hotkeys
        self._install_hook = bool(install_hook) and not demo
        self.listener = HotkeyListener(cfg.hotkey, lambda ev: self._sig_hotkey.emit("dictate", ev),
                                       lambda combo: self._sig_captured.emit(combo), cfg.double_tap_lock)
        self.listener.add_binding("command", cfg.command_hotkey, lambda ev: self._sig_hotkey.emit("command", ev))
        self.listener.add_binding("paste_last", cfg.paste_last_hotkey,
                                  lambda ev: self._sig_hotkey.emit("paste_last", ev), mode="press")
        self.listener.add_binding("undo", cfg.undo_hotkey, lambda ev: self._sig_hotkey.emit("undo", ev),
                                  mode="press")
        self._apply_hold_limits()
        self.injector.settle = self.listener.settle     # never paste while a menu-focus check is outstanding

        # ---- worker + timers
        self._work: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, name="voxflow-pipeline", daemon=True)
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._back_to_idle)
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(33)
        self._level_timer.timeout.connect(lambda: self.level_changed.emit(self.level()))
        self._cap_timer = QTimer(self)
        self._cap_timer.setInterval(100)
        self._cap_timer.timeout.connect(self._poll_cap)
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(2000)
        self._health_timer.timeout.connect(self._health_tick)
        self._failure_told: dict = {}
        self._settings_unsaved = False
        self._error_info: dict = {"kind": "", "action": "", "hint": "", "message": ""}
        self._health_ticks = 0
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(SYNC_INTERVAL_MS)
        self._sync_timer.timeout.connect(self._sync_round)
        self._sync_soon = QTimer(self)
        self._sync_soon.setSingleShot(True)
        self._sync_soon.setInterval(SYNC_DEBOUNCE_MS)
        self._sync_soon.timeout.connect(self._sync_round)
        self._speed_timer = QTimer(self)
        self._speed_timer.setSingleShot(True)
        self._speed_timer.timeout.connect(self._speed_tick)
        self._unpause_timer = QTimer(self)             # ends a timed pause ("Pause for 1 hour")
        self._unpause_timer.setSingleShot(True)
        self._unpause_timer.timeout.connect(lambda: self.set_paused(False))

        self._sig_hotkey.connect(self._on_hotkey)
        self._sig_speed.connect(self._on_speed)
        self._sig_captured.connect(self._on_captured)
        self._sig_store.connect(self._on_store_event)
        self._sig_suggestion.connect(self.dictionary_suggestion)
        self._sig_model.connect(self._on_model_event)
        self._sig_done.connect(self._on_done)
        self._sig_failed.connect(self._on_failed)
        self._sig_notice.connect(self.notice)
        self._sig_sync.connect(self._on_sync_step)
        self.config_changed.connect(
            lambda key, _value: None if key in syncmod.LOCAL_SETTINGS or key == "sync_folder"
            else self._sync_touch())

    # ================================================================== construction helpers
    def _build_modules(self) -> None:
        if self._modules_override is not None:
            self._Rec, self._asr, self._cleaner = self._modules_override
            self._stub_notes = []
            return
        from . import app as appmod
        self._Rec, self._asr, self._cleaner, self._stub_notes = appmod.make_modules(self.config, self._fake,
                                                                                    self._measured_devices())

    def _make_recorder(self, device):
        from . import app as appmod
        rec = appmod.make_recorder(self._Rec, device, self.config)
        try:
            rec.level_boost = 3.0 if self.config.whisper_mode else 1.0
        except Exception:  # noqa: BLE001 - foreign recorder classes may not allow it
            pass
        return rec

    def _apply_hold_limits(self) -> None:
        # watchdog slightly above the Recorder's own cap, so the (cleaner) hit_cap path normally fires first
        limit = self.config.max_record_s + 5
        self.listener.machine.max_hold_s = limit
        cmd = self.listener.binding("command")
        if cmd is not None:
            cmd.max_hold_s = limit

    # ================================================================== lifecycle
    def start(self) -> None:
        """Load the models, install the keyboard hook, start timers. Call once, after the UI is connected."""
        if self._started:
            return
        self._started = True
        self._worker.start()
        self.muter.recover()
        if self._install_hook:
            if not self.listener.start():
                self._problems["hook"] = ("The global shortcuts could not be registered "
                                          f"({self.listener.error}). Restart VoxFlow to try again.")
        self._load_models()
        if not self._fake:
            self._mic_count = audiomod.input_device_count()
            self._check_mic(initial=True)
        self._health_timer.start()
        self.updates.start()
        if self.sync.enabled:
            self._sync_timer.start()
            self._sync_soon.start()
        for level, message in self._startup_notices:
            self.notice.emit(level, message)
        self._startup_notices = []
        if self._stub_notes and not self._fake:
            log.warning("running with stand-ins: %s", ", ".join(self._stub_notes))
        self._emit_status()
        self.state_changed.emit(self.state, self.mode, self.message)

    def shutdown(self) -> None:
        """Idempotent: release the keyboard, the microphone, other apps' audio, the clipboard and the worker."""
        if self._closed:
            return
        self._closed = True
        for timer in (self._idle_timer, self._level_timer, self._cap_timer, self._health_timer,
                      self._sync_timer, self._sync_soon, self._speed_timer):
            timer.stop()
        self.updates.close()
        self._speed.cancel()
        self._speed.join(3.0)                          # its child is killed: nothing keeps running after VoxFlow
        try:
            self.sync.stamp_local()                    # edits made just now keep their time; no folder access
            self.sync.flush()
        except Exception:  # noqa: BLE001
            log.exception("could not record pending sync changes")
        self._flush_stores()
        self.listener.stop()
        self._active = False
        try:
            if self._recorder.is_recording:
                self._recorder.stop()
        except Exception:  # noqa: BLE001
            pass
        self.muter.close()
        if self._worker.is_alive():
            self._work.put(None)
            self._worker.join(3.0)
        self.injector.close()
        self.sounds.close()
        from . import app as appmod
        appmod.close_recorder(self._recorder)
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        try:                                           # last: everything is saved and released before Setup runs
            self.updates.on_exit(self._quit_by_user)
        except Exception:  # noqa: BLE001 - a failed update must never stop VoxFlow from closing
            log.exception("could not start the update installer")

    def _flush_stores(self) -> None:
        for st in (self.dictionary, self.snippets, self.history, self.stats, self.styles):
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                log.exception("could not flush %s", getattr(st, "filename", "a store"))

    def emergency_restore(self) -> None:
        """For the crash handler: undo global side effects without touching Qt."""
        self._flush_stores()
        try:
            self.muter.restore_now()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.injector.close()
        except Exception:  # noqa: BLE001
            pass

    def request_activate(self) -> None:
        """Another launch asked this instance to show itself."""
        self.activate_requested.emit()

    # ================================================================== settings
    def get(self, key: str):
        if key == "sync_folder":                       # kept with the sync state, not in config.json
            return self.sync.folder
        if key not in cfgmod.defaults():
            raise ValueError(f"Unknown setting “{key}”.")
        return getattr(self.config, key)

    def set(self, key: str, value) -> None:
        """Validate, apply live, persist, announce. Raises ValueError (human message) and changes nothing
        when the value is not acceptable."""
        if key == "sync_folder":
            self.set_sync_folder(value)
            return
        self._set(key, value, check_clash=True)

    def _set(self, key: str, value, check_clash: bool) -> None:
        cfg = self.config
        value = cfgmod.Config.validate(key, value)
        if key == "config_version":
            raise ValueError("The settings version cannot be changed.")
        if check_clash and key in cfgmod.HOTKEY_KEYS and value:
            problem = self.validate_hotkey(key, value)
            if problem:
                raise ValueError(problem)
        if getattr(cfg, key) == value:
            return
        if key == "launch_at_login":
            self._autostart.set(value)                 # raises ValueError if Windows refuses
        if key in ("hotkey", "command_hotkey") or key == "mic_device" or key in _MODEL_KEYS:
            self._abort_recording()                    # never re-bind / swap devices under a live recording
        setattr(cfg, key, value)
        self._apply(key, value)
        self._save_config()
        self.config_changed.emit(key, value)
        self._emit_status()

    def _save_config(self) -> None:
        try:
            cfgmod.save_config(self.config, self._config_path)
        except OSError as exc:
            log.warning("could not save settings: %s", exc)
            self._settings_unsaved = True
            self._notice_action("warning", "Your change is active, but it could not be saved to disk and will be "
                                           "lost when VoxFlow closes.", "retry_save", "Try again")
            return
        self._settings_unsaved = False

    def _apply(self, key: str, value) -> None:
        cfg = self.config
        if key == "hotkey":
            self.listener.set_combo(value)
        elif key == "command_hotkey":
            self.listener.set_binding("command", value)
            self._apply_hold_limits()
        elif key == "paste_last_hotkey":
            self.listener.set_binding("paste_last", value)
        elif key == "undo_hotkey":
            self.listener.set_binding("undo", value)
        elif key == "double_tap_lock":
            self.listener.machine.double_tap = bool(value)
        elif key == "restore_clipboard":
            self.injector.restore = bool(value)
        elif key == "restore_delay_s":
            self.injector.restore_delay = float(value)
        elif key == "mic_device":
            self._switch_recorder(value)
        elif key == "max_record_s":
            self._apply_hold_limits()
            if not self._active:
                self._switch_recorder(cfg.mic_device)
        elif key == "whisper_mode":
            try:
                self._recorder.level_boost = 3.0 if value else 1.0
            except Exception:  # noqa: BLE001
                pass
        elif key == "sounds_enabled":
            self.sounds.enabled = bool(value)
        elif key == "sound_volume":
            self.sounds.volume = float(value)
        elif key == "mute_media":
            if not value:
                self.muter.unmute()
        elif key in ("history_limit", "history_retention_days"):
            self.history.configure(limit=cfg.history_limit, retention_days=cfg.history_retention_days)
        elif key == "cleanup_timeout_s":
            try:
                self._cleaner.timeout_s = float(value)
            except Exception:  # noqa: BLE001
                pass
        elif key == "command_timeout_s":
            self.command.timeout_s = float(value)
        elif key in _MODEL_KEYS:
            self.reload_models()
        elif key in ("update_mode", "update_install_when", "update_time"):
            self.updates.settings_changed(key)

    def reset_settings(self) -> None:
        """Every setting back to its default (stores, onboarding state and window layout are kept)."""
        # The default shortcuts do not clash with each other, but one of them may clash with a shortcut that
        # is still at its old value half-way through; so the cross-check is skipped here. Each setting is
        # applied, saved and announced once, and only if its value actually changes.
        for key, value in cfgmod.defaults().items():
            if key in _KEPT_ON_RESET or getattr(self.config, key) == value:
                continue
            try:
                self._set(key, value, check_clash=False)
            except ValueError as exc:
                log.warning("reset: %s not restored (%s)", key, exc)

    def validate_hotkey(self, key: str, combo: str) -> str:
        """'' if ``combo`` can be assigned to ``key``; otherwise why not (a sentence for the user)."""
        if key not in cfgmod.HOTKEY_KEYS:
            return "That is not a shortcut setting."
        if not isinstance(combo, str) or not combo.strip():
            return "The dictation shortcut cannot be empty." if key == "hotkey" else ""
        others = {cfgmod.HOTKEY_LABELS[k]: getattr(self.config, k) for k in cfgmod.HOTKEY_KEYS if k != key}
        return hotkey_problem(combo, others)

    @staticmethod
    def pretty_hotkey(combo: str) -> str:
        return pretty_hotkey(combo) if combo else "Not set"

    def begin_hotkey_capture(self) -> None:
        """The next key combination pressed anywhere is reported through ``hotkey_captured``."""
        if not self._install_hook or self.listener.error:
            self.notice.emit("info", "Shortcuts can't be recorded right now because the keyboard hook is not "
                                     "running.")
            self.hotkey_captured.emit("")
            return
        self._abort_recording()
        self._capturing = True
        self.listener.begin_capture()

    def cancel_hotkey_capture(self) -> None:
        self.listener.cancel_capture()
        if self._capturing:
            self._capturing = False
            self.hotkey_captured.emit("")

    def _on_captured(self, combo: str) -> None:
        if not self._capturing:
            return
        self._capturing = False
        if combo:
            problem = hotkey_problem(combo)
            if problem:
                self.notice.emit("warning", problem)
                combo = ""
            else:
                combo = normalize_hotkey(combo)
        self.hotkey_captured.emit(combo)

    @staticmethod
    def languages() -> list[tuple[str, str]]:
        return list(cfgmod.LANGUAGES)

    @staticmethod
    def tones() -> list[tuple[str, str]]:
        return [(t, store.TONE_LABELS[t]) for t in store.TONES]

    @staticmethod
    def categories() -> list[tuple[str, str]]:
        return [(c, store.CATEGORY_LABELS[c]) for c in store.CATEGORIES]

    @staticmethod
    def preview_tone(text: str, tone: str, language: str | None = None) -> str:
        """``text`` as it would be pasted with ``tone``. ``language``: a code from ``languages()`` when the
        sample is not English ("auto" / None / "en" = English)."""
        from .cleanup import apply_tone
        return apply_tone(text, tone, None if language in (None, "auto", "en") else language)

    @staticmethod
    def language_support(code: str) -> str:
        """How much clean-up a dictation language gets: "full" (own rules: fillers, spoken commands,
        self-corrections, smart formatting), "basic" (tidy-up plus the language model with a neutral prompt)
        or "auto" (decided per dictation from the words that were spoken)."""
        if code == "auto":
            return "auto"
        return "full" if code == "en" or code in langs.SUPPORTED else "basic"

    # ================================================================== profiles and packs
    def _apply_settings(self, values: dict) -> list[str]:
        """Apply a batch of settings from a profile or another device. Returns the keys that were not
        applied as given. Shortcuts are checked against each other once the whole batch is in."""
        skipped = []
        for key, value in values.items():
            try:
                self._set(key, value, check_clash=False)
            except ValueError as exc:
                log.info("setting %s not applied (%s)", key, exc)
                skipped.append(key)
        probe = copy.copy(self.config)
        cfgmod._drop_duplicate_hotkeys(probe)           # two actions can never share a shortcut
        for key in cfgmod.HOTKEY_KEYS:
            if getattr(probe, key) != getattr(self.config, key):
                self._set(key, getattr(probe, key), check_clash=False)
                if key not in skipped:
                    skipped.append(key)
        return skipped

    def export_backup(self, path: str, include_history: bool = True) -> dict:
        """Write the whole profile to one file: settings, dictionary, snippets and styles, plus the history
        and its statistics unless ``include_history`` is False. Returns ``{path, counts, includes}``.
        Raises ValueError (human message) when the file cannot be written."""
        return syncmod.export_profile(self._portable, path, include_history, self.sync.device_id)

    def inspect_backup(self, path: str) -> dict:
        """What a backup file holds, without changing anything: ``{created, version, device, same_device,
        counts, includes, skipped}``. Raises ValueError when the file is not a usable backup."""
        return syncmod.describe_profile(path, self.sync.device_id)

    def import_backup(self, path: str, mode: str = "merge", parts=None) -> dict:
        """Import a backup. ``mode`` "merge" adds what is missing and keeps what is there; "replace" makes
        this profile equal to the file. ``parts`` limits it to some of ``backup_parts()``. All or nothing.
        Returns ``{mode, counts, details, settings_skipped, skipped}``; raises ValueError on failure."""
        self._abort_recording()
        self._sync_applying = True
        try:
            return syncmod.import_profile(self._portable, path, mode, parts, self.sync.device_id)
        finally:
            self._sync_applying = False
            self._sync_touch()

    @staticmethod
    def backup_parts() -> list[tuple[str, str]]:
        return [("settings", "Settings"), ("dictionary", "Dictionary"), ("snippets", "Snippets"),
                ("styles", "Styles"), ("stats", "Statistics"), ("history", "History")]

    def export_pack(self, kind: str, path: str, name: str = "", description: str = "", ids=None) -> int:
        """Write a shareable pack of dictionary terms or snippets (``kind``: "dictionary" | "snippets";
        ``ids`` = only these entries, default all). Returns the number of entries written; the file gets
        ``PACK_SUFFIX`` when ``path`` has no extension. Raises ValueError on failure."""
        return syncmod.export_pack(self._portable, kind, path, name, description, ids)["count"]

    def inspect_pack(self, path: str) -> dict:
        """``{kind, name, description, count, created, new, existing, skipped}`` for a pack file; changes
        nothing. Raises ValueError when the file is not a pack."""
        return syncmod.describe_pack(self._portable, path)

    def import_pack(self, path: str, mode: str = "merge") -> dict:
        """Bring a pack in: "merge" adds new entries and updates the ones that exist here; "replace" makes
        the pack the whole list. Returns ``{kind, name, added, updated, removed, skipped}``."""
        try:
            return syncmod.import_pack(self._portable, path, mode)
        finally:
            self._sync_touch()

    # ================================================================== sync folder
    @property
    def sync_folder(self) -> str:
        return self.sync.folder

    def sync_status(self) -> dict:
        """``{state, folder, last_sync, error, message, enabled, device, devices, unreadable, newer}``;
        ``state``: "off" | "idle" | "syncing" | "error" (see ``voxflow.sync.SyncEngine.status``)."""
        return self.sync.status()

    def set_sync_folder(self, path: str | None) -> None:
        """Keep dictionary, snippets, styles and portable settings in step with other devices through files
        in ``path`` ('' or None turns it off). Raises ValueError (human message) for an unusable folder."""
        before = self.sync.folder
        self.sync.set_folder(path)
        if self.sync.folder != before:
            self.config_changed.emit("sync_folder", self.sync.folder)
        if self.sync.enabled:
            if self._started and not self._closed:
                self._sync_timer.start()
            self._sync_round()
        else:
            self._sync_timer.stop()
            self._sync_soon.stop()
        self.sync_status_changed.emit(self.sync_status())

    def sync_now(self) -> None:
        """Start a sync round now (coalesced when one is already running; a notice when sync is off)."""
        if not self.sync.enabled:
            self.notice.emit("info", "Sync is off. Choose a sync folder first.")
            return
        self._sync_round()

    def _sync_touch(self) -> None:
        """Something portable changed here: note the time now, sync shortly."""
        if self._sync_applying or self._closed or not self.sync.enabled:
            return
        if self._sync_busy:
            self._sync_again = True
        elif self._started:
            self._sync_soon.start()

    def _sync_round(self) -> None:
        if self._closed or not self.sync.enabled:
            return
        if self._sync_busy or self._active:
            self._sync_again = True                    # one round at a time, and never under a recording
            if not self._sync_busy:
                self._sync_soon.start()
            return
        self._sync_busy, self._sync_again = True, False
        self.sync.stamp_local()
        self.sync.note_syncing()
        self.sync_status_changed.emit(self.sync_status())
        self._sync_thread("read", self.sync.read_remote)

    def _sync_thread(self, step: str, job) -> None:
        def run():
            try:
                result = job()
            except Exception as exc:  # noqa: BLE001 - reported on the UI thread
                result = exc
            try:
                self._sig_sync.emit(step, result)
            except RuntimeError:                        # the controller is gone (shutdown)
                pass

        threading.Thread(target=run, name=f"voxflow-sync-{step}", daemon=True).start()

    def _on_sync_step(self, step: str, result) -> None:
        if self._closed:
            return
        if isinstance(result, Exception):
            if not isinstance(result, syncmod.SyncUnavailable):
                log.error("sync %s failed: %r", step, result)
            self.sync.note_unavailable(result)
        elif step == "read":
            if self._active:                            # a recording started meanwhile: merge on the next round
                self._sync_again = True
                self.sync.note_idle()
            else:
                self._sync_applying = True
                try:
                    payload = self.sync.merge(result)
                except Exception as exc:  # noqa: BLE001
                    log.exception("sync merge failed")
                    self.sync.note_unavailable(exc)
                    payload = None
                finally:
                    self._sync_applying = False
                if payload is not None:
                    self._sync_thread("written", lambda: self.sync.write_own(payload))
                    return
        self._sync_busy = False
        self.sync_status_changed.emit(self.sync_status())
        if self._sync_again:
            self._sync_again = False
            self._sync_soon.start()

    # ================================================================== status
    def status(self) -> dict:
        cfg = self.config
        hk = pretty_hotkey(cfg.hotkey)
        if self._problems:
            key = next((k for k in ("hook", "asr", "mic") if k in self._problems), next(iter(self._problems)))
            title = {"hook": "Shortcuts unavailable", "asr": "Speech model unavailable",
                     "mic": "Microphone problem"}.get(key, "Problem")
            return {"level": "error", "title": title, "detail": self._problems[key]}
        if self.paused:
            if self.paused_until is not None:
                at = time.strftime("%H:%M", time.localtime(self.paused_until))
                return {"level": "paused", "title": f"Paused until {at}",
                        "detail": f"Dictation resumes by itself at {at}. Resume it sooner from the tray or the "
                                  "main window."}
            return {"level": "paused", "title": "Paused", "detail": "Dictation is paused. Resume it from the tray "
                                                                    "or the main window."}
        asr = self._models["asr"]
        if asr["state"] in ("pending", "loading", "downloading"):
            title = "Downloading speech model…" if asr["state"] == "downloading" else "Loading speech model…"
            return {"level": "loading", "title": title, "detail": asr["detail"]}
        if self.is_demo:
            return {"level": "demo", "title": "Demo mode", "detail": "Simulated: no models, mic or hotkeys."}
        if self._warnings:
            return {"level": "warning", "title": "Ready", "detail": next(iter(self._warnings.values()))}
        cleanup = self._models["cleanup"]
        if cfg.cleanup_enabled and cleanup["state"] in ("pending", "loading", "downloading"):
            return {"level": "ok", "title": "Ready", "detail": f"Hold {hk} to dictate. The cleanup model is "
                                                               "still loading."}
        return {"level": "ok", "title": "Ready", "detail": f"Hold {hk} to dictate."}

    def _emit_status(self) -> None:
        self.status_changed.emit(self.status())

    def model_status(self, name: str) -> dict:
        if name not in self._models:
            raise ValueError(f"Unknown model “{name}”.")
        return dict(self._models[name])

    def _on_model_event(self, name: str, status: dict) -> None:
        self._models[name] = status
        if name == "asr":
            if status["state"] == "failed":
                self._problems["asr"] = status["detail"]
            elif status["state"] == "ready":
                self._problems.pop("asr", None)
        if name == "cleanup":
            if status["state"] in ("degraded", "failed"):
                self._warnings["cleanup"] = status["detail"]
            else:
                self._warnings.pop("cleanup", None)
        self.model_status_changed.emit(name, dict(status))
        self._emit_status()
        self._speed_consider()
        self.devices_changed.emit(self.device_info())

    def reload_models(self, which=None) -> None:
        """(Re)load the models in the background, e.g. after a failure or a model setting change.
        ``which``: model names ("asr", "cleanup"), default both."""
        names = ("asr", "cleanup") if which is None else tuple(n for n in ("asr", "cleanup") if n in which)
        self._abort_recording()
        self._speed.cancel()                           # a measurement must not compete with the load
        if self._modules_override is None:
            old_asr, old_cleaner = self._asr, self._cleaner
            rec = self._Rec
            self._build_modules()
            self._Rec = rec
            if "asr" not in names:
                self._asr = old_asr                    # a model that is not reloaded stays loaded as it is
            if "cleanup" not in names:
                self._cleaner = old_cleaner
            self.command.cleaner = self._cleaner
            del old_asr, old_cleaner
        self._load_models(names)

    def _load_models(self, names=("asr", "cleanup")) -> None:
        """Load ``names`` on background threads. Loading only one model is for when the other one is settled:
        a newer load generation drops the reports of a load still under way."""
        self._load_generation += 1
        generation = self._load_generation
        asr, cleaner, cfg = self._asr, self._cleaner, self.config

        def report(name: str, state: str, detail: str, progress=None, backend: str = "") -> None:
            if generation == self._load_generation:
                self._sig_model.emit(name, {"state": state, "detail": detail, "progress": progress,
                                            "backend": backend})

        def load_asr() -> None:
            try:
                self.dictionary.prepare()          # off the dictation path: the word list loads in ~30 ms
            except Exception:  # noqa: BLE001 - it is built on first use instead
                log.exception("dictionary matchers could not be prepared")
            repo = "" if self._fake else modelhub.asr_repo(cfg.asr_model)
            cached = modelhub.is_cached(repo)
            report("asr", "loading" if cached else "downloading",
                   "Loading the speech model…" if cached else "Downloading the speech model (one time)…")

            def progress(done: int, total) -> None:
                report("asr", "downloading", modelhub.progress_text(done, total),
                       min(0.99, done / total) if total else None)

            try:
                if cached:
                    asr.load()
                else:
                    with modelhub.DownloadWatcher(repo, progress):
                        asr.load()
            except Exception as exc:  # noqa: BLE001
                log.exception("speech model failed to load")
                report("asr", "failed", _model_failure("speech", exc))
                return
            backend = str(getattr(asr, "backend", ""))
            report("asr", "ready", _backend_text(backend), None, backend)

        def load_cleanup() -> None:
            model = cfg.cleanup_model or getattr(cleaner, "model", "") or ""
            repo, filename = ("", "") if self._fake else modelhub.cleanup_repo(str(model))
            cached = modelhub.is_cached(repo, filename or None)
            report("cleanup", "loading" if cached else "downloading",
                   "Loading the cleanup model…" if cached else "Downloading the cleanup model (one time)…")
            # a first GPU start compiles kernels for ~25 s: say so instead of looking frozen
            slow_hint = threading.Timer(6.0, lambda: report("cleanup", "loading", modelhub.GPU_PREPARE_NOTE))
            slow_hint.daemon = True
            slow_hint.start()

            def progress(done: int, total) -> None:
                report("cleanup", "downloading", modelhub.progress_text(done, total),
                       min(0.99, done / total) if total else None)

            try:
                if cached:
                    cleaner.load()
                else:
                    with modelhub.DownloadWatcher(repo, progress):
                        cleaner.load()
            except Exception as exc:  # noqa: BLE001
                log.exception("cleanup model failed to load")
                slow_hint.cancel()
                report("cleanup", "failed", "The cleanup model could not be loaded, so dictation is pasted as "
                                            f"heard. ({_short(exc)})")
                return
            slow_hint.cancel()
            backend = str(getattr(cleaner, "backend", ""))
            if not getattr(cleaner, "ready", True):
                report("cleanup", "failed", "The cleanup model could not be loaded, so dictation is pasted as heard.")
            elif backend == "rules":
                report("cleanup", "degraded", "The language model is unavailable, so only basic cleanup rules "
                                              "run and command mode is off.", None, backend)
            else:
                report("cleanup", "ready", _backend_text(backend), None, backend)

        if "asr" in names:
            threading.Thread(target=load_asr, daemon=True, name="voxflow-load-asr").start()
        if "cleanup" in names:
            threading.Thread(target=load_cleanup, daemon=True, name="voxflow-load-cleanup").start()

    # ================================================================== speed check (voxflow.speedcheck)
    def _speed_models(self) -> dict:
        return {"asr": self.config.asr_model, "cleanup": self.config.cleanup_model or ""}

    def _speed_key(self) -> str:
        return speedcheck.fingerprint(self._speed_machine, self._speed_models())

    def _speed_valid(self) -> dict | None:
        """The stored result if it was measured on this hardware and software with these models, else None."""
        rec = self._speed_record
        return rec if rec is not None and rec.get("fingerprint") == self._speed_key() else None

    def _measured_devices(self) -> dict:
        rec = self._speed_valid()
        return dict(rec["choice"]) if rec else {}

    def _models_settled(self) -> bool:
        return (self._models["asr"]["state"] == "ready"
                and self._models["cleanup"]["state"] in ("ready", "degraded", "failed"))

    def _busy(self) -> bool:
        """A dictation is being recorded or processed (the speed check and a device switch wait for it)."""
        return self._active or self.state in ("listening", "locked", "processing") or not self._work.empty()

    def _speed_needed(self) -> bool:
        """The user asked, or (automatic) this PC has no valid result and VOXFLOW_DEVICE does not decide."""
        from . import app as appmod
        return self._speed_wanted or (self._speed_auto and self._speed_valid() is None
                                      and not appmod.forced_device())

    def _speed_consider(self) -> None:
        """Arm the speed check once both models are ready, when it is needed."""
        if (self._closed or self._speed.running or self._speed_timer.isActive() or not self._models_settled()
                or not self._speed_needed()):
            return
        self._set_speed_state("waiting")
        self._speed_timer.start(0 if self._speed_wanted else SPEED_DELAY_MS)

    def _speed_tick(self) -> None:
        if self._closed or self._speed.running or not self._models_settled() or not self._speed_needed():
            return                                     # a reload is under way: _speed_consider arms it again
        if self._busy():
            self._set_speed_state("waiting")
            self._speed_timer.start(SPEED_RETRY_MS)
            return
        self._speed_timer.stop()
        self._speed_wanted = False
        self._speed_run = (self._speed_key(), self._speed_models())
        log.info("speed check started (each back-end is measured in a child process)")
        self._speed.start(self._speed_run[1], self.config.cleanup_timeout_s)
        self._set_speed_state("running")

    def _set_speed_state(self, state: str, done: int = 0, total: int = 0, model: str = "", backend: str = "") -> None:
        self._speed_state = {"state": state, "done": done, "total": total, "model": model, "backend": backend}
        self.devices_changed.emit(self.device_info())

    def _on_speed(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "progress":
            self._set_speed_state("running", ev["done"], ev["total"], ev["model"], ev["backend"])
        elif kind == "cancelled":
            log.info("speed check postponed (VoxFlow is busy); it starts again when it is idle")
            if self._closed:
                return
            self._speed_wanted = self._speed_wanted or self._speed_valid() is not None   # a manual run resumes
            self._set_speed_state("waiting")
            self._speed_timer.start(SPEED_RETRY_MS)
        else:
            key, models = self._speed_run
            record = speedcheck.make_record(key, self._speed_machine, models, ev["results"], ev["took_s"])
            if not self.is_demo:
                try:
                    speedcheck.save(self.data_dir, record)
                except OSError as exc:
                    log.warning("could not save the speed check result (%s); it is used until VoxFlow closes", exc)
            self._speed_record = record
            log.info("speed check finished in %.1f s", record["took_s"])
            for line in speedcheck.describe(record):
                log.info("speed check: %s", line)
            self._set_speed_state("done")
            self._apply_measured()

    def _apply_measured(self) -> None:
        """Move speech recognition to a newly measured device now. The language model's choice applies at the
        next start: llama.cpp loads its build once per process."""
        from . import app as appmod
        if self._closed:
            return
        want = appmod.model_devices(self.config, self._measured_devices())[0]
        built_for = getattr(self._asr, "device", None)
        if built_for is None or built_for == want:
            return
        if self._busy() or not self._models_settled():
            QTimer.singleShot(SPEED_RETRY_MS, self._apply_measured)
            return
        log.info("speech recognition moves to %s (the speed check measured it faster)", want)
        self.reload_models(("asr",))

    def measure_devices(self) -> None:
        """Run the speed check now (or as soon as VoxFlow is idle), also when this PC was measured before.
        Raises ValueError when it cannot run (stand-in models)."""
        if not self._speed_manual:
            raise ValueError("The speed check needs the real models, which are not in use.")
        if self._speed.running:
            return
        self._speed_wanted = True
        self._set_speed_state("waiting")
        self._speed_timer.stop()
        self._speed_consider()

    def device_info(self) -> dict:
        """What the device menus in Settings show. Per model ("asr", "cleanup"): ``setting`` (the config value),
        ``current`` (where it runs now), ``auto`` (what Automatic picks here: the measured choice, else the
        fixed order), ``fastest``, ``next`` (the back-end the next start is expected to use), ``restart``
        (the language model moves to ``next`` at the next start) and ``options``:
        one ``{id, measured, available, reason, ms, failed}`` per back-end. ``check``: the speed check's
        ``state`` (idle | waiting | running | done), progress, ``measured_at`` and ``available``.
        ``forced``: the VOXFLOW_DEVICE value that overrides everything ("" if unset)."""
        from . import app as appmod
        rec = self._speed_valid()
        next_start = appmod.model_devices(self.config, rec["choice"] if rec else None)
        out = {"forced": appmod.forced_device(),
               "check": dict(self._speed_state, available=self._speed_manual,
                             measured_at=rec["measured_at"] if rec else None)}
        for i, (name, key) in enumerate((("asr", "asr_device"), ("cleanup", "cleanup_device"))):
            names = speedcheck.backends(name)
            results = rec["results"][name] if rec else {}
            static = rec["static"][name] if rec else None
            current = str(self._models[name].get("backend", ""))
            target = next_start[i] if next_start[i] != "auto" else (static or current)
            options = []
            for b in names:
                r = results.get(b)
                options.append({"id": b, "measured": r is not None,
                                "available": r is None or r["available"],
                                "reason": speedcheck.short_reason(r["reason"]) if r and not r["available"] else "",
                                "ms": r["median_ms"] if r and r["ok"] else None,
                                "failed": bool(r and r["available"] and not r["ok"])})
            out[name] = {"setting": getattr(self.config, key), "current": current, "next": target,
                         "auto": (rec["choice"][name] or static) if rec else None,
                         "fastest": speedcheck.fastest(results, names),
                         "restart": name == "cleanup" and current in names and target in names and target != current,
                         "options": options}
        return out

    # ================================================================== app updates
    def update_info(self) -> dict:
        """Settings > Updates: version, mode, how this copy was installed, last check, the newer release (if any)
        and what is under way (``phase``: idle, checking, downloading, waiting, restarting)."""
        return self.updates.info()

    def check_updates(self) -> bool:
        """"Check now": one check, in every mode. With the setting Off this click is the only way VoxFlow ever asks
        GitHub (no startup or daily check). False when a check or download is already under way."""
        return self.updates.check(user=True)

    def install_update(self, when: str = "now") -> None:
        """``"now"``: download, then restart into the installer as soon as no dictation or command is under way.
        ``"quit"``: install when VoxFlow is quit. Raises ValueError when this copy cannot update itself."""
        self.updates.install(when)

    def remind_update_later(self) -> None:
        self.updates.remind_later()

    def skip_update(self) -> None:
        self.updates.skip()

    def prepare_quit(self) -> None:
        """The user chose Quit: an update waiting for "when I quit" may run (not when Windows ends the session)."""
        self._quit_by_user = True

    def _update_busy(self) -> bool:
        return self._closed or self._busy() or self._capturing

    def _on_update_restart(self) -> None:
        self._quit_by_user = True
        self.restart_requested.emit()

    # ================================================================== audio
    def level(self) -> float:
        try:
            return float(self._recorder.level())
        except Exception:  # noqa: BLE001
            return 0.0

    def recent_samples(self, n: int = 1024) -> np.ndarray:
        try:
            return self._recorder.recent_samples(n)
        except Exception:  # noqa: BLE001
            return np.zeros(n, dtype=np.float32)

    def set_level_monitor(self, on: bool) -> None:
        self._monitor = bool(on)
        self._sync_level_timer()

    def _sync_level_timer(self) -> None:
        # The Flow bar polls the recorder itself; the only consumers of ``level_changed`` are the meters that call
        # ``set_level_monitor(True)`` (mic test in Settings, onboarding). No monitor, no 30 Hz wake-ups.
        want = self._monitor
        if want and not self._level_timer.isActive():
            self._level_timer.start()
        elif not want and self._level_timer.isActive():
            self._level_timer.stop()
            self.level_changed.emit(0.0)

    def microphones(self) -> list[dict]:
        if self._fake:
            names = list(fakes.FAKE_MICROPHONES)
            default = names[0]
        else:
            names = [name for _idx, name in audiomod.list_input_devices()]
            default = audiomod.default_input_name()
        label = f"System default ({default})" if default else "System default"
        out = [{"id": None, "name": label, "is_default": True}]
        out += [{"id": name, "name": name, "is_default": False} for name in names]
        chosen = self.config.mic_device
        if isinstance(chosen, str) and chosen not in names:
            out.append({"id": chosen, "name": f"{chosen} (not connected)", "is_default": False})
        return out

    def refresh_microphones(self) -> None:
        """Re-enumerate audio devices (PortAudio caches them) and announce the list."""
        if not self._fake and not self._active:
            try:
                self._recorder.close()
                audiomod.rescan_devices()
                self._recorder.reopen(rescan=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("microphone refresh: %s", exc)
            self._check_mic()
        self.microphones_changed.emit(self.microphones())

    def play_sound(self, name: str) -> None:
        self.sounds.play(name, force=True)

    def _switch_recorder(self, device) -> None:
        from . import app as appmod
        old = self._recorder
        appmod.close_recorder(old)                     # release the previous input stream / mic indicator
        try:
            self._recorder = self._make_recorder(device)
        except Exception as exc:  # noqa: BLE001
            log.warning("recorder could not be created: %s", exc)
            self._problems["mic"] = audiomod.describe_mic_error(exc)
            return
        self._check_mic()

    def _check_mic(self, initial: bool = False) -> None:
        """Translate the recorder's condition into a problem / warning for the status line."""
        rec = self._recorder
        err = getattr(rec, "error", None)
        if err is not None:
            log.warning("microphone unavailable: %s", err)
            self._problems["mic"] = audiomod.describe_mic_error(err)
        else:
            self._problems.pop("mic", None)
        if getattr(rec, "using_fallback", False) and err is None:
            self._warnings["mic"] = (f"“{self.config.mic_device}” is not connected, so the system default "
                                     "microphone is being used.")
        else:
            self._warnings.pop("mic", None)
        if not initial:
            self._emit_status()

    def _health_tick(self) -> None:
        self._health_ticks += 1
        if _DIAG and self._health_ticks % 30 == 0:           # VOXFLOW_DIAG=1: one log line a minute (P6)
            now, cpu = time.perf_counter(), time.process_time()
            last = getattr(self, "_diag_last", None)
            if last is not None:
                log.info("diag: process cpu %.2f%% of one core over %.0f s, %d python threads, state %s",
                         (cpu - last[1]) / max(now - last[0], 1e-6) * 100, now - last[0],
                         threading.active_count(), self.state)
            self._diag_last = (now, cpu)
        if self._health_ticks % 1800 == 0:             # hourly: enforce history retention
            self.history.prune()
        if self._fake or self._closed:
            return
        count = audiomod.input_device_count()
        if count != self._mic_count:
            self._mic_count = count
            if not self._active:
                self.refresh_microphones()
            return
        alive = getattr(self._recorder, "alive", True)
        if alive:
            if "mic" in self._problems and getattr(self._recorder, "error", None) is None:
                self._check_mic()
            return
        if self._active:
            # the device vanished mid-sentence: keep what was said
            self.notice.emit("warning", "The microphone was disconnected. What you said so far is being used.")
            self.listener.force_idle()
            self._stop_recording()
            return
        try:
            self._recorder.reopen()
            if "mic" in self._problems:
                self.notice.emit("success", "Microphone connected again.")
        except Exception as exc:  # noqa: BLE001
            log.debug("microphone still unavailable: %s", exc)
        self._check_mic()

    # ================================================================== flow-bar state
    def _set_state(self, state: str, message: str = "", mode: str | None = None) -> None:
        if mode is not None:
            self.mode = mode
        self.state, self.message = state, message
        self._idle_timer.stop()
        if state == "done":
            self._idle_timer.start(DONE_MS)
        elif state == "error":
            self._idle_timer.start(ERROR_MS)
        self._sync_level_timer()
        if state in ("listening", "locked"):
            self._cap_timer.start()
        else:
            self._cap_timer.stop()
        self.state_changed.emit(self.state, self.mode, self.message)

    def _back_to_idle(self) -> None:
        if self.state in ("done", "error"):
            self._set_state("idle")

    def _fail(self, message: str, sound: bool = True, reason: str | None = None) -> None:
        """``reason``: the full sentence behind a short bar ``message``, used to classify the failure."""
        if sound:
            self.sounds.play("error")
        self._error_info = dict(classify_error(reason or message), message=message)
        self.error_info_changed.emit(dict(self._error_info))
        self._announce_failure(self._error_info)
        self._set_state("error", message)

    _FAILURE_LABELS = {"microphone": "Open microphone settings", "models": "Open model settings"}

    def _announce_failure(self, info: dict) -> None:
        """The Flow bar is click-through, so a reason that has a next step also goes out as a notice with an action
        id the window / tray can turn into a button (``open_settings:microphone`` / ``open_settings:models``).
        At most one per reason every 30 s."""
        action = info.get("action")
        if not action:
            return
        now = self._mono()
        last = self._failure_told.get(info["kind"])
        if last is not None and now - last < 30.0:
            return
        self._failure_told[info["kind"]] = now
        text = f"{info['message'].rstrip('.')}. {info['hint']}"
        self.notice.emit("warning", text)
        self.notice_action.emit(text, "open_settings:" + action, self._FAILURE_LABELS.get(action, "Open settings"))

    def retry_saves(self) -> dict:
        """Try again to write whatever could not be saved (the five stores and settings.json).
        Returns ``{name: {"ok": bool, "error": str}}`` for ``history, dictionary, snippets, stats, styles,
        settings``; ``ok`` is True for everything that was fine all along. Success is reported as a notice."""
        out: dict = {}
        failed_before = False
        for name, st in (("history", self.history), ("dictionary", self.dictionary), ("snippets", self.snippets),
                         ("stats", self.stats), ("styles", self.styles)):
            if st.last_save_error:
                failed_before = True
                ok = st.retry_save()
                out[name] = {"ok": ok, "error": st.last_save_error}
            else:
                out[name] = {"ok": True, "error": ""}
        if self._settings_unsaved:
            failed_before = True
            try:
                cfgmod.save_config(self.config, self._config_path)
                self._settings_unsaved = False
                out["settings"] = {"ok": True, "error": ""}
            except OSError as exc:
                out["settings"] = {"ok": False, "error": str(exc) or type(exc).__name__}
        else:
            out["settings"] = {"ok": True, "error": ""}
        if failed_before and all(v["ok"] for v in out.values()):
            self.notice.emit("success", "Everything is saved now.")
        elif failed_before:
            bad = ", ".join(k for k, v in out.items() if not v["ok"])
            self._notice_action("error", f"Still couldn't save {bad}. Check that the disk has room and the "
                                         "VoxFlow folder is writable.", "retry_save", "Try again")
        return out

    def _notice_action(self, level: str, message: str, action_id: str, label: str) -> None:
        self.notice.emit(level, message)
        self.notice_action.emit(message, action_id, label)

    def error_info(self) -> dict:
        """Why the Flow bar is (or last was) in the error state: ``kind`` (silence, no_speech, mic, model, paste,
        nothing, other), ``action`` (the Settings area that helps: "microphone", "models" or ""), a one-sentence
        ``hint`` for the user and the bar ``message``. Empty ``kind`` when there was no error yet."""
        return dict(self._error_info)

    def demo_set_state(self, state: str, mode: str = "dictate", message: str = "") -> None:
        """Demo / screenshot helper: show a flow-bar state without recording anything."""
        if not self.is_demo:
            return
        if state not in STATES or mode not in MODES:
            raise ValueError(f"Unknown state “{state}” or mode “{mode}”.")
        self._set_state(state, message, mode)
        if state in ("done", "error"):
            self._idle_timer.stop()                    # screenshots decide when to move on

    def demo_set_model(self, name: str, state: str, detail: str = "", progress: float | None = None,
                       backend: str = "") -> None:
        """Demo / screenshot helper: show a model in a given load state (download progress, failure ...)."""
        if not self.is_demo:
            return
        if name not in self._models:
            raise ValueError(f"Unknown model “{name}”.")
        self._load_generation += 1                     # a real (simulated) load must not overwrite the pose
        self._on_model_event(name, {"state": state, "detail": detail, "progress": progress, "backend": backend})

    def demo_add_transcript(self, text: str | None = None) -> dict | None:
        """Demo helper: file one more sample dictation (history, stats, signals). Returns the entry."""
        if not self.is_demo:
            return None
        raw, sample, exe = _DEMO_LINES[self.history.count() % len(_DEMO_LINES)]
        if text is not None:
            raw, sample = text.lower().rstrip("."), text
        entry = store.make_entry(sample, raw, app_exe=exe, app_title="", duration_s=len(sample) / 14.0,
                                 latency_ms=640)
        if self.config.save_history:
            self.history.add_entry(entry)
        self.stats.record(entry["words"], entry["duration_s"], entry["app_exe"])
        self.last_transcript = entry
        self.transcript_ready.emit(dict(entry))
        return dict(entry)

    # ================================================================== hotkey events (UI thread)
    def _on_hotkey(self, binding: str, ev: str) -> None:
        if self._closed:
            return
        if binding == "paste_last":
            self.paste_last()
            return
        if binding == "undo":
            self.undo_last_paste()
            return
        if ev == "start":
            self._start_recording(binding)
        elif ev == "lock":
            if self._active:
                self._set_state("locked")
        elif ev == "cancel":
            self.cancel()
        elif ev == "stop":
            self._stop_recording()

    def _start_recording(self, mode: str) -> None:
        if self._active or self.paused:
            return
        if self._speed.running:
            self._speed.cancel()                       # its child is killed at once: the dictation gets the PC
        asr_state = self._models["asr"]["state"]
        if asr_state != "ready":
            self.listener.force_idle()
            self.mode = mode
            self._fail("Speech model unavailable" if asr_state == "failed" else "Speech model still loading…")
            return
        if mode == "command" and not self.command.available:
            self.listener.force_idle()
            self.mode = mode
            loading = self._models["cleanup"]["state"] in ("pending", "loading", "downloading")
            self._fail("Language model still loading…" if loading else "Command mode needs the language model")
            return
        self._context = self._safe_context()
        try:
            self._recorder.start()
        except Exception as exc:  # noqa: BLE001
            log.warning("recording could not start: %s", exc)
            self.listener.force_idle()
            self._problems["mic"] = audiomod.describe_mic_error(exc)
            self._emit_status()
            self.mode = mode
            self._fail("Microphone unavailable")
            return
        if "mic" in self._problems:
            self._problems.pop("mic")
            self._emit_status()
        self._gen += 1
        self._t_start = time.perf_counter()
        self._active = True
        self.sounds.play("command" if mode == "command" else "start")
        if self.config.mute_media:
            self.muter.mute()
        self._set_state("listening", "", mode)

    def _safe_context(self) -> AppContext:
        try:
            ctx = self._context_fn()
            return ctx if isinstance(ctx, AppContext) else AppContext()
        except Exception as exc:  # noqa: BLE001
            log.debug("context capture failed: %s", exc)
            return AppContext()

    def cancel(self) -> None:
        """Abort the current recording (same as pressing Esc while dictating)."""
        if not self._active:
            return
        self._abort_recording()

    def _abort_recording(self) -> None:
        if not self._active:
            return
        self._active = False
        self.listener.force_idle()
        try:
            if self._recorder.is_recording:
                self._recorder.stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("recorder stop failed while cancelling: %s", exc)
        self.muter.unmute()
        self._set_state("idle")

    def _stop_recording(self) -> None:
        if not self._active:
            return
        self._active = False
        t_release = time.perf_counter()
        self.muter.unmute()
        try:
            audio = self._recorder.stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("recorder stop failed: %s", exc)
            self._fail("Recording failed")
            return
        from . import app as appmod
        if appmod.too_short(audio, self._recorder, t_release - self._t_start, self.config.min_record_s):
            self._set_state("idle")
            return
        self.sounds.play("stop")
        speech_s = getattr(self._recorder, "last_speech_seconds", None)
        if not isinstance(speech_s, (int, float)) or isinstance(speech_s, bool):
            speech_s = len(audio) / audiomod.SAMPLE_RATE
        self._set_state("processing")
        self._work.put({"kind": self.mode, "audio": audio, "t_release": t_release, "gen": self._gen,
                        "context": self._context, "speech_s": float(speech_s)})

    def _poll_cap(self) -> None:
        """Recorder hit its max_seconds cap: end the recording exactly like a hotkey release (also when locked)."""
        from . import app as appmod
        if self._active and appmod.recorder_hit_cap(self._recorder):
            self.listener.force_idle()
            self._stop_recording()

    def set_paused(self, paused: bool, for_s: float | None = None) -> None:
        """Pause / resume dictation. While paused the shortcuts are ignored and keys pass through untouched.
        ``for_s`` pauses for that many seconds and then resumes by itself. Every call replaces the previous one, so
        resuming (or pausing again) by any route cancels a pending timed resume."""
        paused = bool(paused)
        until = time.time() + for_s if paused and for_s else None
        self._unpause_timer.stop()
        if until is not None:
            self._unpause_timer.start(max(1, round(for_s * 1000)))
        if paused == self.paused and until is None and self.paused_until is None:
            return
        if paused and not self.paused:
            self._abort_recording()
        self.paused, self.paused_until = paused, until
        self.listener.enabled = not paused
        self._emit_status()

    # ================================================================== actions
    def paste_last(self, entry_id: str | None = None) -> bool:
        """Paste the most recent transcript (or the history entry ``entry_id``) into the focused app."""
        entry = self.history.get(entry_id) if entry_id else self.last_transcript
        if not entry or not entry.get("text"):
            self.notice.emit("info", "There is nothing to paste yet." if not entry_id
                             else "That transcript is no longer in your history.")
            return False
        self._work.put({"kind": "paste", "text": entry["text"], "context": self._safe_context()})
        return True

    def undo_last_paste(self) -> bool:
        """Undo the last paste with Ctrl+Z, but only in the window that received it."""
        last = self._last_paste
        if not last:
            self.notice.emit("info", "There is no paste to undo.")
            return False
        try:
            here = int(self._hwnd_fn() or 0)
        except Exception:  # noqa: BLE001
            here = 0
        if last["hwnd"] and here != last["hwnd"]:
            self.notice.emit("warning", "Undo only works in the window you last dictated into. Switch back to "
                                        "it and try again.")
            return False
        self._last_paste = None
        self._work.put({"kind": "undo"})
        return True

    def copy_text(self, text: str) -> bool:
        try:
            ok = bool(text) and bool(self.injector.copy_text(text))
        except Exception as exc:  # noqa: BLE001
            log.warning("copy failed: %s", exc)
            ok = False
        if not ok:
            self.notice.emit("error", "Couldn't copy: the clipboard is in use by another app. Try again.")
        return ok

    def open_data_folder(self) -> None:
        if not system.open_folder(self.data_dir):
            self.notice.emit("error", f"Couldn't open {self.data_dir}.")

    def open_log_folder(self) -> None:
        if not system.open_folder(system.log_dir(self.data_dir)):
            self.notice.emit("error", f"Couldn't open {system.log_dir(self.data_dir)}.")

    # ================================================================== pipeline worker
    def _worker_loop(self) -> None:
        while True:
            job = self._work.get()
            if job is None:
                return
            try:
                kind = job["kind"]
                if kind == "dictate":
                    self._job_dictate(job)
                elif kind == "command":
                    self._job_command(job)
                elif kind == "paste":
                    self._job_paste(job)
                elif kind == "undo":
                    if not self.injector.undo():
                        self._sig_notice.emit("warning", "Couldn't send the undo because a modifier key is "
                                                         "still held down.")
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                # the details belong in the log (the stack, never the message: it can quote the dictated words);
                # the user gets plain words and a next step
                log.error("pipeline job '%s' failed\n%s", job.get("kind"), system.redacted_traceback(exc))
                message = _JOB_FAILURES.get(job.get("kind"), _JOB_FAILURES["dictate"])
                if "gen" in job:
                    self._sig_failed.emit(message, job["gen"])
                else:                                   # paste / undo: no flow-bar turn of their own to fail
                    self._sig_notice.emit("error", message)

    def _clean_kwargs(self, context: AppContext) -> dict:
        cfg = self.config
        return {"tone": self.styles.tone_for(context.exe, context.title), "terms": self.dictionary.terms(),
                "language": cfg.language, "smart": cfg.smart_formatting}

    def _personalise(self, text: str, raw: str, context: AppContext) -> str:
        """The user's snippets and dictionary, applied to the cleaned dictation.

        Snippets go first and the dictionary only corrects the words around them, so a snippet is typed exactly
        as saved. When the clean-up reworded a trigger phrase that the recogniser did hear (the language model
        or smart formatting rewrote "sig dot" as "sig."), the dictation is rule-cleaned again without those
        steps, so the snippet still fires: saying the trigger must always work."""
        if raw and raw != text and not self.snippets.matches(text) and self.snippets.matches(raw):
            kwargs = dict(self._clean_kwargs(context), smart=False)
            try:
                if self._rule_cleaner is None:
                    from .cleanup import Cleaner
                    self._rule_cleaner = Cleaner(use_llm=False)
                    self._rule_cleaner.load()
                again = self._rule_cleaner.clean(raw, **kwargs)
            except Exception:  # noqa: BLE001 - the rules never fail a dictation
                log.exception("rule clean-up for a snippet failed")
                again = ""
            text = again if again and self.snippets.matches(again) else raw
        return self.snippets.expand(text, transform=self.dictionary.apply)

    def _paste(self, text: str) -> tuple[bool, str]:
        """Paste on the worker thread. Returns (text reached the app or the clipboard, flow-bar message)."""
        self.injector.begin(text)
        self.injector.finish_later()
        outcome = getattr(self.injector, "last_outcome", "pasted")
        if outcome == "pasted":
            return True, "Pasted"
        if outcome == "typed":
            return True, "Typed"
        if outcome == "copied":
            return True, "Copied. Press Ctrl+V"
        return False, "Couldn't paste"

    def _job_dictate(self, job: dict) -> None:
        from . import app as appmod
        cfg, ctx, gen = self.config, job["context"], job["gen"]
        audio = job["audio"]
        silent = audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4
        if cfg.whisper_mode and not silent:
            audio = audiomod.normalize_gain(audio)
        try:
            res = appmod.run_pipeline(audio, self._asr, self._cleaner, cfg.cleanup_enabled,
                                      **self._clean_kwargs(ctx))
        except Exception:  # noqa: BLE001 - only the recogniser can raise here (cleanup problems are absorbed)
            log.exception("speech recognition failed")
            self._sig_failed.emit(_SPEECH_FAILURE, gen)
            return
        text = res["text"]
        if not text:
            if silent:
                msg = "Heard only silence. Is the microphone muted?"
            else:
                msg = "Nothing to paste" if res["dropped"] else "Didn't catch that"
            self._sig_failed.emit(msg, gen)
            return
        if text.strip():
            cleaned = text
            text = self._personalise(text, res["raw"], ctx)
            if cfg.auto_learn and cfg.save_history and res["cleaned"]:     # history off: learn nothing from it
                for suggestion in self.dictionary.learn(res["raw"], cleaned):
                    self._sig_suggestion.emit(suggestion)
        ok, message = self._paste(text)
        latency = time.perf_counter() - job["t_release"]
        rules_only = bool(res.get("rules_only"))
        entry = store.make_entry(text, res["raw"], app_exe=ctx.exe, app_title=ctx.title,
                                 duration_s=job["speech_s"], mode="dictate", latency_ms=latency * 1000,
                                 rules_only=rules_only)
        if ok and rules_only:
            message += RULES_ONLY_SUFFIX                 # "Pasted · basic clean-up": the quiet hint on the bar
        self._finish(entry, ctx, gen, message, res, pasted=ok)

    def _job_command(self, job: dict) -> None:
        ctx, gen, cfg = job["context"], job["gen"], self.config
        try:
            selection = self.command.grab_selection()
            audio = job["audio"]
            if cfg.whisper_mode:
                audio = audiomod.normalize_gain(audio)
            try:
                instruction = (self._asr.transcribe(audio) or "").strip()
            except Exception:  # noqa: BLE001
                log.exception("speech recognition failed")
                self._sig_failed.emit(_SPEECH_FAILURE, gen)
                return
            result = self.command.run(instruction, selection, tone=self.styles.tone_for(ctx.exe, ctx.title),
                                      terms=self.dictionary.terms(), language=cfg.language)
        except CommandError as exc:
            self._sig_failed.emit(str(exc), gen)
            return
        result = self.dictionary.apply(result)
        if ctx.hwnd:
            try:
                moved = int(self._hwnd_fn() or 0) != ctx.hwnd
            except Exception:  # noqa: BLE001
                moved = False
            if moved:
                # the user switched windows while the model was working: never paste into the wrong place
                copied = bool(self.injector.copy_text(result))
                self._sig_failed.emit("Window changed. Result copied: press Ctrl+V" if copied
                                      else "Window changed, so nothing was pasted", gen)
                return
        ok, message = self._paste(result)
        latency = time.perf_counter() - job["t_release"]
        entry = store.make_entry(result, instruction, app_exe=ctx.exe, app_title=ctx.title,
                                 duration_s=job["speech_s"], mode="command", latency_ms=latency * 1000)
        if ok and message == "Pasted":
            message = "Rewritten" if selection.strip() else "Written"
        self._finish(entry, ctx, gen, message, None, pasted=ok)

    def _job_paste(self, job: dict) -> None:
        ok, message = self._paste(job["text"])
        if ok:
            self._last_paste = {"hwnd": job["context"].hwnd, "text": job["text"], "t": time.time()}
        if not ok or message != "Pasted":
            self._sig_notice.emit("info" if ok else "error",
                                  "The text is on the clipboard. Press Ctrl+V to paste it." if ok
                                  else "Couldn't paste the last transcript.")

    def _finish(self, entry: dict, ctx: AppContext, gen: int, message: str, res: dict | None,
                pasted: bool = True) -> None:
        """File the transcript and report the outcome. A transcript that could not be pasted is kept all the
        same (history, ``last_transcript``), so the words are never lost: the user can copy it or paste it again."""
        if pasted:
            self._last_paste = {"hwnd": ctx.hwnd, "text": entry["text"], "t": time.time()}
        elif entry["text"].strip():
            message = ("Couldn't paste. The text is saved in your History." if self.config.save_history
                       else "Couldn't paste. Use Paste last transcript to try again.")
        if res is not None:
            log.info("asr %.0f ms + cleanup %.0f ms, release->paste %d ms, %d words (%s)",
                     res["t_asr"] * 1000, res["t_clean"] * 1000, entry["latency_ms"], entry["words"],
                     ctx.exe or "unknown app")
        # the bar says "Pasted" first; the (slower) file writes follow and can never turn a paste into an error
        self._sig_done.emit({"entry": entry, "gen": gen, "message": message, "pasted": pasted})
        if entry["text"].strip():                      # a lone line break is pasted but is not a transcript
            try:
                if self.config.save_history:
                    self.history.add_entry(entry)
                if entry["mode"] == "dictate":
                    self.stats.record(entry["words"], entry["duration_s"], entry["app_exe"], entry["ts"])
            except Exception as exc:  # noqa: BLE001
                log.error("could not file the transcript\n%s", system.redacted_traceback(exc))

    def _owns_bar(self, gen: int) -> bool:
        """False when a newer recording started (or is running): a stale job must not repaint the flow bar."""
        return gen == self._gen and not self._active

    def _note_rules_only(self) -> None:
        """The language model was too slow for a dictation. One slow answer is only marked (bar, history);
        when it keeps happening the user is told once, and not again for a while."""
        now = self._mono()
        self.rules_only_count += 1
        self._rules_only_times = [t for t in self._rules_only_times if now - t <= RULES_ONLY_WINDOW_S] + [now]
        if (len(self._rules_only_times) >= RULES_ONLY_BURST
                and (self._rules_only_told is None or now - self._rules_only_told >= RULES_ONLY_QUIET_S)):
            self._rules_only_told = now
            self.notice.emit("info", RULES_ONLY_NOTICE)

    def _on_done(self, payload: dict) -> None:
        entry = payload["entry"]
        if entry["text"].strip():
            self.last_transcript = entry
            if entry.get("rules_only"):
                self._note_rules_only()
            self.transcript_ready.emit(dict(entry))
        if not payload.get("pasted", True):
            self._on_failed(payload["message"], payload["gen"])
        elif self._owns_bar(payload["gen"]):
            self._set_state("done", payload["message"])

    def _on_failed(self, message: str, gen: int) -> None:
        if not self._owns_bar(gen):
            return
        # the flow bar has room for one short sentence; anything longer also goes out as a notice
        first = message.split(". ")[0].rstrip(".")
        bar = first if len(first) <= 48 else "Something went wrong"
        if bar.rstrip(".") != message.rstrip("."):
            self.notice.emit("error", message)
        self._fail(bar, reason=message)

    # ================================================================== store events
    _SAVE_NAMES = {"history": "your history", "dictionary": "your dictionary", "snippets": "your snippets",
                   "stats": "your usage statistics", "styles": "your style settings"}

    def _on_store_event(self, topic: str) -> None:
        if topic.startswith("save_failed:"):
            what = self._SAVE_NAMES.get(topic.split(":", 1)[1], "your data")
            self._notice_action("warning", f"VoxFlow couldn't save {what} (the disk may be full or read-only). "
                                           "It is kept for now but will be lost when VoxFlow closes.",
                                "retry_save", "Try again")
            return
        if topic.startswith("save_ok:"):
            return
        signal = {"dictionary": self.dictionary_changed, "suggestions": self.suggestions_changed,
                  "snippets": self.snippets_changed, "history": self.history_changed,
                  "stats": self.stats_changed, "styles": self.styles_changed}.get(topic)
        if signal is not None:
            signal.emit()
        if topic in ("dictionary", "snippets", "styles"):
            self._sync_touch()


# =========================================================================== helpers
RULES_ONLY_SUFFIX = " · basic clean-up"          # appended to the flow-bar message of such a dictation
RULES_ONLY_BURST = 3                             # this many within RULES_ONLY_WINDOW_S earn one notice ...
RULES_ONLY_WINDOW_S = 300.0
RULES_ONLY_QUIET_S = 1800.0                      # ... and then none for this long
RULES_ONLY_NOTICE = ("The language model has been too slow several times, so those dictations got the basic "
                     "clean-up only. They are marked in your History. This usually passes when the computer "
                     "is less busy.")
_ERROR_KINDS = (                                 # (message starts with, kind, action, hint)
    ("Heard only silence", "silence", "microphone", "Check that your microphone isn't muted and is the one selected."),
    ("Didn't catch", "no_speech", "", "Speak a little closer to the microphone and try again."),
    ("Microphone unavailable", "mic", "microphone", "Check that a microphone is connected, then choose it in Settings."),
    ("Recording failed", "mic", "microphone", "Check that a microphone is connected, then choose it in Settings."),
    ("speech model", "model", "models", "Open Settings > On-device models to see what is wrong."),
    ("language model", "model", "models", "Open Settings > On-device models to see what is wrong."),
    ("command mode needs", "model", "models", "Open Settings > On-device models to see what is wrong."),
    ("Couldn't paste", "paste", "", "The text is saved in your History; use Paste last transcript to try again."),
    ("Window changed", "paste", "", "The result is on the clipboard: press Ctrl+V where you want it."),
    ("Nothing to paste", "nothing", "", "That was only filler words, so nothing was pasted."),
)


def classify_error(message: str) -> dict:
    """Map a Flow-bar error message to a machine-readable reason (see ``Controller.error_info``)."""
    low = message.lower()
    for prefix, kind, action, hint in _ERROR_KINDS:
        if prefix.lower() in low:                  # also matches the long forms ("The speech model hit an error ...")
            return {"kind": kind, "action": action, "hint": hint, "message": message}
    return {"kind": "other", "action": "", "hint": "Try again; if it keeps happening, restart VoxFlow.",
            "message": message}


_SPEECH_FAILURE = "The speech model hit an error. Try again; if it keeps happening, restart VoxFlow."
_JOB_FAILURES = {
    "dictate": "Something went wrong with that dictation. Try again; if it keeps happening, restart VoxFlow.",
    "command": "Something went wrong in command mode, so nothing was changed. Try again; if it keeps happening, "
               "restart VoxFlow.",
    "paste": "The text could not be pasted. Try again; if it keeps happening, restart VoxFlow.",
    "undo": "The undo could not be sent. Press Ctrl+Z in the app instead.",
}


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip().rstrip(":")
    return text if len(text) <= 120 else text[:117] + "…"


def _backend_text(backend: str) -> str:
    return {"cuda": "Running on the GPU (CUDA).", "vulkan": "Running on the GPU (Vulkan).",
            "webgpu": "Running on the GPU (WebGPU).", "cpu": "Running on the CPU.",
            "rules": "Basic rules only.", "demo": "Simulated (demo)."}.get(backend, "Ready.")


def _model_failure(what: str, exc: BaseException) -> str:
    text = str(exc).lower()
    if any(k in text for k in ("connection", "timed out", "offline", "name resolution", "max retries", "network")):
        return (f"The {what} model could not be downloaded. Connect to the internet once so VoxFlow can fetch "
                "it, then choose Retry.")
    if "space" in text and "disk" in text:
        return f"There is not enough disk space for the {what} model."
    return f"The {what} model could not be loaded ({_short(exc)}). Choose Retry, or see the log for details."


_DEMO_LINES = [
    ("um so can we move the stand up to ten thirty tomorrow", "Can we move the stand-up to 10:30 tomorrow?", "slack.exe"),
    ("hey are you free for lunch on friday", "hey are you free for lunch on friday", "whatsapp.exe"),
    ("thanks for the update i will review the pull request this afternoon",
     "Thanks for the update. I'll review the pull request this afternoon.", "outlook.exe"),
    ("add a retry with exponential back off to the upload client",
     "Add a retry with exponential backoff to the upload client.", "code.exe"),
    ("the quarterly numbers look good revenue is up twelve percent",
     "The quarterly numbers look good. Revenue is up 12%.", "winword.exe"),
    ("remind me to call the dentist on monday no tuesday", "Remind me to call the dentist on Tuesday.", "notepad.exe"),
]


def _seed_demo(controller: "Controller") -> None:
    """Fill the throw-away demo profile with realistic content so every page has something to show."""
    now = time.time()
    c = controller
    for term, aliases in (("VoxFlow", ["vox flow"]), ("Kubernetes", ["cube are netties"]),
                          ("PostgreSQL", ["post gres", "postgres"]), ("Anthropic", []), ("Dr. Okafor", ["doctor okafor"])):
        c.dictionary.add(term, aliases)
    c.dictionary.suggest("TensorFlow", ["tensor flow"])
    c.dictionary.suggest("GitHub", ["get hub"])
    c.snippets.add("my address", "221B Baker Street\nLondon NW1 6XE")
    c.snippets.add("meeting link", "https://meet.example.com/voxflow-weekly")
    c.snippets.add("sign off", "Best regards,\nAlex")
    c.styles.set_app_category("code.exe", "work")
    k = 0
    for day in range(13, -1, -1):
        for n in range(0 if day in (5, 9) else 1 + (day * 7) % 4):
            raw, text, exe = _DEMO_LINES[k % len(_DEMO_LINES)]
            k += 1
            ts = now - day * 86400 - n * 2700 - 600
            entry = store.make_entry(text, raw, app_exe=exe, app_title="", duration_s=len(text) / 14.0,
                                     latency_ms=520 + 40 * n, ts=ts)
            c.history.add_entry(entry)
            c.stats.record(entry["words"] * (3 + (k % 5)), entry["duration_s"] * (3 + (k % 5)), exe, ts)
