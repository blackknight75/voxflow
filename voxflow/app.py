"""VoxFlow application entry: process set-up, the pipeline helpers, self-test.

The running app is three objects: ``Controller`` (voxflow.controller: models, hotkeys, pipeline, stores),
the UI built by ``voxflow.ui.shell.build_ui(controller)``, and the Qt event loop started here.
"""
from __future__ import annotations

import atexit
import getpass
import hashlib
import importlib
import inspect
import logging
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

from . import config as cfgmod, fakes, system

log = logging.getLogger("voxflow.app")

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- module loading w/ fallbacks
_IMPORT_ERRORS: dict[str, BaseException] = {}       # voxflow.<modname> -> why it could not be imported


def _real(modname: str, attr: str):
    """Import voxflow.<modname>.<attr> lazily; return None if the module is missing or its dependency is broken."""
    try:
        found = getattr(importlib.import_module(f"voxflow.{modname}"), attr)
    except Exception as exc:  # ImportError, AttributeError, or a failing dependency import
        log.warning("voxflow.%s.%s unavailable (%s: %s): using a stand-in", modname, attr, type(exc).__name__, exc)
        _IMPORT_ERRORS[modname] = exc
        return None
    _IMPORT_ERRORS.pop(modname, None)
    return found


class DamagedInstall(RuntimeError):
    """Part of VoxFlow's own code could not be imported: only reinstalling helps. The message is for the user."""


def damaged_install(modname: str) -> DamagedInstall:
    exc = _IMPORT_ERRORS.get(modname)
    why = f"{type(exc).__name__} in {modname}.py" if exc is not None else f"{modname}.py is missing"
    return DamagedInstall(f"VoxFlow's installation is damaged ({why}). Reinstall VoxFlow.")


class BrokenModel:
    """Takes the place of a model whose code could not be imported, in the real app (never the demo stand-ins:
    they would paste a canned sentence into the user's window). ``load()`` raises, so the model reports
    "failed" with a message that says to reinstall; nothing else is ever called on a model that did not load."""

    backend = ""
    has_llm = False
    ready = False
    n_ctx = 4096

    def __init__(self, error: DamagedInstall):
        self.error = error

    def load(self) -> None:
        raise self.error

    def transcribe(self, audio) -> str:
        raise self.error

    def clean(self, text: str, **_options) -> str:
        return text

    def generate(self, *_a, **_k) -> str:
        raise self.error


class BrokenRecorder:
    """Takes the place of ``voxflow.audio.Recorder`` when it could not be imported, in the real app: no stream,
    ``error`` says to reinstall, every attempt to record fails."""

    alive = False
    using_fallback = False
    is_recording = False
    hit_cap = False
    last_speech_seconds = 0.0

    def __init__(self, device=None, max_seconds: float = 300.0):
        self.device = device
        self.level_boost = 1.0
        self.error = damaged_install("audio")

    def start(self) -> None:
        raise self.error

    def reopen(self, rescan: bool = True) -> None:
        raise self.error

    def stop(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def level(self) -> float:
        return 0.0

    def recent_samples(self, n: int = 1024) -> np.ndarray:
        return np.zeros(n, dtype=np.float32)

    def set_device(self, device) -> None:
        self.device = device

    def close(self) -> None:
        pass


# VOXFLOW_DEVICE -> (speech recognition back-end: voxflow.asr.ASR_BACKENDS,
#                    language model back-end:     voxflow.cleanup.LLAMA_BACKENDS).
# The one place where the names are mapped. The two models use different GPU libraries, so "the GPU of any vendor"
# has two names: WebGPU for speech (onnxruntime), Vulkan for the language model (llama.cpp). Either name asks for
# both. A named back-end that does not work on this PC ends on the CPU, as in the automatic choice.
DEVICE_NAMES = {
    "cpu": ("cpu", "cpu"),
    "cuda": ("cuda", "cuda"),            # NVIDIA only
    "vulkan": ("webgpu", "vulkan"),
    "webgpu": ("webgpu", "vulkan"),
}


def forced_device() -> str:
    """VOXFLOW_DEVICE=cpu | cuda | vulkan | webgpu: where the models must run instead of the automatic choice
    ("" if unset). The way out when a graphics driver misbehaves, and how the fallbacks are tested on a PC that
    has a GPU."""
    forced = os.environ.get("VOXFLOW_DEVICE", "").strip().lower()
    return forced if forced in DEVICE_NAMES else ""


def forced_devices() -> tuple[str, str]:
    """(speech recognition device, language model device) named by VOXFLOW_DEVICE; ("", "") if unset."""
    return DEVICE_NAMES.get(forced_device(), ("", ""))


def model_devices(cfg, measured: dict | None = None) -> tuple[str, str]:
    """(speech device, language model device) to build the models with. Precedence: VOXFLOW_DEVICE, then a
    device named in Settings (``asr_device`` / ``cleanup_device``), then the speed check's measured choice
    (``measured``: {"asr": name | None, "cleanup": name | None}), then "auto" (the fixed order)."""
    measured = measured or {}
    forced_asr, forced_llm = forced_devices()
    asr_device = forced_asr or (cfg.asr_device if cfg.asr_device != "auto" else measured.get("asr") or "auto")
    llm_device = forced_llm or (cfg.cleanup_device if cfg.cleanup_device != "auto"
                                else measured.get("cleanup") or "auto")
    return asr_device, llm_device


def make_modules(cfg, fake: bool = False, measured: dict | None = None):
    """Return (recorder_factory, transcriber, cleaner, notes). ``measured``: see ``model_devices``."""
    notes = []
    Rec = None if fake else _real("audio", "Recorder")
    Asr = None if fake else _real("asr", "Transcriber")
    Cln = None if fake else _real("cleanup", "Cleaner")
    # the stand-ins only with ``fake``: in the real app a module that cannot be imported is a damaged install, and
    # the demo models would paste a canned sentence into the user's window (QA R1)
    if Rec is None:
        Rec = fakes.FakeRecorder if fake else BrokenRecorder
        notes.append("audio stub" if fake else "audio broken")
    asr_device, llm_device = model_devices(cfg, measured)
    if Asr is None:
        asr = fakes.FakeTranscriber() if fake else BrokenModel(damaged_install("asr"))
        notes.append("asr stub" if fake else "asr broken")
    else:
        asr = Asr(cfg.asr_model, asr_device)
    if Cln is None:
        cleaner = fakes.FakeCleaner() if fake else BrokenModel(damaged_install("cleanup"))
        notes.append("cleanup stub" if fake else "cleanup broken")
    else:
        cleaner = Cln(cfg.cleanup_model, llm_device, cfg.cleanup_timeout_s)
    return Rec, asr, cleaner, notes


_KW_CACHE: dict = {}


def _accepts_keywords(fn, names) -> bool:
    """True if ``fn`` can be called with all of the keyword arguments ``names`` (the answer is cached per function)."""
    key = (getattr(fn, "__func__", fn), tuple(names))
    try:
        return _KW_CACHE[key]
    except (KeyError, TypeError):
        pass
    answer = _accepts_keywords_uncached(fn, names)
    try:
        _KW_CACHE[key] = answer
    except TypeError:
        pass
    return answer


def _accepts_keywords_uncached(fn, names) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(n in params for n in names)


def run_pipeline(audio, asr, cleaner, cleanup_enabled: bool = True, **clean_options) -> dict:
    """audio -> raw transcript -> cleaned text. Never raises for cleanup problems.

    ``clean_options`` (tone, terms, language, smart) are passed to ``cleaner.clean`` when it supports them.
    ``rules_only`` in the result is True when the language model was wanted for this utterance but timed out or
    was skipped for lack of time, so the text got the rule clean-up only; ``model_skipped`` says it was skipped
    only because an earlier, abandoned generation was still running (this utterance's model call never ran)."""
    out = {"raw": "", "text": "", "t_asr": 0.0, "t_clean": 0.0, "cleaned": False, "dropped": False,
           "rules_only": False, "model_skipped": False}
    t0 = time.perf_counter()
    out["raw"] = (asr.transcribe(audio) or "").strip()
    out["t_asr"] = time.perf_counter() - t0
    out["text"] = out["raw"]
    if out["raw"] and cleanup_enabled and getattr(cleaner, "ready", False):
        t1 = time.perf_counter()
        try:
            # contract: a ready Cleaner returns "" on purpose for filler-only / command-only speech
            # ("um", "new line") -> paste nothing; on failure it returns the (non-empty) input.
            # NOT stripped: a lone "new line" command legitimately returns whitespace (a real line break),
            # which must be pasted. Only the empty string means "paste nothing".
            if clean_options and _accepts_keywords(cleaner.clean, clean_options):
                cleaned = cleaner.clean(out["raw"], **clean_options) or ""
            else:
                cleaned = cleaner.clean(out["raw"]) or ""
            out["text"], out["cleaned"] = cleaned, cleaned != out["raw"]
            out["dropped"] = cleaned == ""
            info = getattr(cleaner, "last_info", None)
            # the language model was wanted but timed out or had no time left: the text is rule-cleaned only
            out["rules_only"] = bool(cleaned) and isinstance(info, dict) and bool(info.get("rules_only"))
            # ... only because an EARLIER generation was still stuck: this one never ran (QA C-01 / C-02)
            out["model_skipped"] = (out["rules_only"] and bool(info.get("model_skipped"))
                                    and not info.get("timeout"))
        except Exception as exc:  # cleanup failed: keep the raw transcript
            log.warning("cleanup raised, pasting the raw transcript: %s", system.redacted_exc(exc))
        out["t_clean"] = time.perf_counter() - t1
    return out


def read_wav_16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        x = np.arange(int(len(a) * SAMPLE_RATE / sr)) * (sr / SAMPLE_RATE)
        a = np.interp(x, np.arange(len(a)), a).astype(np.float32)
    return a.astype(np.float32)


def too_short(audio, recorder, held_s: float, min_s: float) -> bool:
    """True if the recording holds less than ``min_s`` of real speech time.

    Recorder prepends a pre-roll (PREROLL_S) to every recording, so raw length alone is never short.
    Prefer the recorder's own ``last_speech_seconds`` when it provides one; else subtract the pre-roll."""
    if held_s < min_s:               # hotkey held for less than the minimum: accidental tap
        return True
    speech = getattr(recorder, "last_speech_seconds", None)
    if callable(speech):
        try:
            speech = speech()
        except Exception:
            speech = None
    if isinstance(speech, (int, float)) and not isinstance(speech, bool):
        return float(speech) < min_s
    preroll = 0.0
    try:
        preroll = float(getattr(importlib.import_module("voxflow.audio"), "PREROLL_S", 0.0))
    except Exception:
        pass
    return len(audio) / SAMPLE_RATE - preroll < min_s


def recorder_hit_cap(rec) -> bool:
    """Recorder.hit_cap (True once the current recording reached its max_seconds cap); False if absent."""
    try:
        return getattr(rec, "hit_cap", False) is True
    except Exception:
        return False


def make_recorder(Rec, device, cfg):
    """Build a Recorder whose hard cap is the configured max_record_s (classes without that option still work)."""
    if _accepts_keywords(Rec, ("max_seconds",)):
        return Rec(device, max_seconds=cfg.max_record_s)
    return Rec(device)


def close_recorder(rec) -> None:
    closer = getattr(rec, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


# --------------------------------------------------------------------------- selftest
def selftest(wav_path: str | None, fake: bool = False) -> int:
    """Load everything, run the pipeline on a WAV, print timings. Every step reports pass / fail; exit 0 = all ok."""
    print("== VoxFlow selftest ==")
    cfg = cfgmod.Config()
    ok = True

    def step(label: str, fn) -> bool:
        t = time.perf_counter()
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            return False
        print(f"[ok] {label} ({time.perf_counter() - t:.2f}s){': ' + detail if detail else ''}")
        return True

    def ui_imports() -> str:
        import PySide6  # noqa: F401
        from . import controller  # noqa: F401
        from .ui import shell  # noqa: F401
        return ""

    def hook() -> str:
        from .hotkey import HotkeyListener
        hl = HotkeyListener("ctrl+win", lambda e: None)
        started = hl.start()
        hl.stop()
        if not started:
            raise RuntimeError(hl.error or "hook did not start")
        return "installed + removed"

    def clipboard() -> str:
        from .inject import Win32Clipboard
        cb = Win32Clipboard()
        snap = cb.snapshot()
        if snap is None:                       # we could not give the user's clipboard back: do not touch it
            return "skipped (clipboard is in use or holds content that cannot be restored)"
        try:
            if cb.set_text("voxflow-selftest") is None or cb.get_text() != "voxflow-selftest":
                raise RuntimeError("text did not round-trip")
        finally:
            cb.restore(snap)
        return "round-trip, previous content restored"

    ok &= step("UI imports", ui_imports)
    ok &= step("keyboard hook", hook)
    ok &= step("clipboard", clipboard)

    def hub() -> str:
        # VOXFLOW_SELFTEST_HUB=1: fetch one tiny file the way a first-run model download does (TLS, cache folder)
        from huggingface_hub import hf_hub_download
        from . import modelhub
        path = hf_hub_download(modelhub.ASR_REPOS[cfg.asr_model], "config.json")
        return f"downloaded {os.path.getsize(path)} bytes into {modelhub.cache_root()}"

    if os.environ.get("VOXFLOW_SELFTEST_HUB") == "1":
        ok &= step("model hub download", hub)


    forced = forced_device()                  # VOXFLOW_DEVICE=cpu: test the no-GPU fallback
    Rec, asr, cleaner, notes = make_modules(cfg, fake)       # make_modules applies the override to both models
    if forced:
        print(f"[note] VOXFLOW_DEVICE={forced}")
    if notes:
        print(f"[note] using stand-ins: {', '.join(notes)}")

    def load_asr() -> str:
        asr.load()
        return f"backend {getattr(asr, 'backend', '?')}"

    def load_cleanup() -> str:
        cleaner.load()
        return f"ready={getattr(cleaner, 'ready', False)}, backend {getattr(cleaner, 'backend', '?')}"

    if not step("speech model load", load_asr):
        print("SELFTEST FAILED")
        return 1
    ok &= step("cleanup model load", load_cleanup)

    if wav_path:
        try:
            audio = read_wav_16k(wav_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] reading {wav_path}: {type(exc).__name__}: {exc}")
            print("SELFTEST FAILED")
            return 1
    else:
        print("[note] no wav given - using 2 s of silence")
        audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    dur = len(audio) / SAMPLE_RATE
    res = None
    try:
        for i in range(2):  # 1st = cold-ish, 2nd = warm
            res = run_pipeline(audio, asr, cleaner, True)
            total = res["t_asr"] + res["t_clean"]
            print(f"run {i + 1}: audio {dur:.2f}s | asr {res['t_asr'] * 1000:.0f} ms | cleanup "
                  f"{res['t_clean'] * 1000:.0f} ms | total {total * 1000:.0f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] pipeline: {type(exc).__name__}: {exc}")
        ok = False
    if res is not None:
        print(f"raw  : {res['raw']!r}")
        print(f"final: {res['text']!r}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- the Qt application
_DEMO_CYCLE = [
    ("listening", "dictate", "", 3.4), ("processing", "dictate", "", 1.4), ("done", "dictate", "Pasted", 1.4),
    ("idle", "dictate", "", 1.2), ("locked", "dictate", "", 2.6), ("processing", "dictate", "", 1.2),
    ("error", "dictate", "Didn't catch that", 1.6), ("idle", "dictate", "", 1.2),
    ("listening", "command", "", 2.6), ("processing", "command", "", 1.6), ("done", "command", "Rewritten", 1.4),
    ("idle", "dictate", "", 1.6),
]


def _version() -> str:
    from . import __version__
    return __version__


def _instance_name() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "user"
    # the profile folder is part of the name: a second copy with another APPDATA (portable, test run) must not
    # take over the pipe of the real instance
    where = hashlib.sha1(str(cfgmod.app_dir()).lower().encode("utf-8", "replace")).hexdigest()[:8]
    return f"VoxFlow-instance-{''.join(ch for ch in user if ch.isalnum())}-{where}"


def show_start_error(text: str) -> None:
    """Tell the user why VoxFlow cannot start (the windowed exe has no console). Replaced in tests."""
    print(text, file=sys.stderr)
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.critical(None, "VoxFlow can't start", text)


def _folder_problem(folder: Path, reason: str) -> int:
    log.error("the data folder %s cannot be used: %s", folder, reason)
    show_start_error(f"VoxFlow can't use its data folder:\n{folder}\n\n{reason}\n\nCheck that the folder (or the "
                     "drive it is on) is available and that you are allowed to change it: it may be read-only, "
                     "offline, full or blocked by a sync or security program. Then start VoxFlow again.")
    return 1


QUIT_COMMAND = b"quit"


def request_quit(timeout_s: float = 5.0) -> int:
    """``VoxFlow.exe --quit``: ask the VoxFlow of this user and data folder to quit the normal way (pending saves
    are written, muted apps get their sound back) and wait until it has. The uninstaller and Setup use it before
    they replace or remove files, instead of ending the process. Exit code 0 when no VoxFlow is running any more."""
    from PySide6.QtCore import QCoreApplication, QLockFile
    from PySide6.QtNetwork import QLocalSocket

    QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    try:
        data_dir = cfgmod.app_dir()
    except OSError:
        return 0                               # no data folder: no VoxFlow can be running with it
    lock = QLockFile(str(data_dir / "voxflow.lock"))

    def stopped() -> bool:
        if lock.tryLock(0):
            lock.unlock()
            return True
        return False

    if stopped():
        return 0
    sock = QLocalSocket()
    sock.connectToServer(_instance_name())
    if sock.waitForConnected(1000):
        sock.write(QUIT_COMMAND)
        sock.waitForBytesWritten(1000)
        sock.disconnectFromServer()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stopped():
            return 0
        time.sleep(0.1)
    return 1


def run_app(demo: bool = False, fake: bool = False, open_settings: bool = False, no_hook: bool = False,
            autostart: bool = False) -> int:
    from PySide6.QtCore import QLockFile, QThread, QTimer
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtWidgets import QApplication

    from .controller import Controller

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("VoxFlow")

    # ---- one instance per user; a second launch brings the first one's window up and exits
    lock = server = None
    data_dir = None
    if not demo:
        try:
            data_dir = cfgmod.app_dir()
        except OSError as exc:                # APPDATA on a missing drive, a folder that cannot be created
            return _folder_problem(cfgmod.app_dir_path(), exc.strerror or type(exc).__name__)
        lock = QLockFile(str(data_dir / "voxflow.lock"))
        if not lock.tryLock(100):
            if lock.error() != QLockFile.LockError.LockFailedError:   # not "held by another instance" (QA R2)
                reason = ("Windows did not allow VoxFlow to write there."
                          if lock.error() == QLockFile.LockError.PermissionError
                          else "VoxFlow could not create a file there.")
                return _folder_problem(data_dir, reason)
            sock = QLocalSocket()
            sock.connectToServer(_instance_name())
            if sock.waitForConnected(500):
                sock.write(b"show")
                sock.waitForBytesWritten(500)
                sock.disconnectFromServer()
            print("VoxFlow is already running; its window has been brought to the front.")
            return 0

    # ---- the log and the crash handler first, so that whatever happens while the controller is built (an unreadable
    # settings file, a microphone that does not open, an exception) is in voxflow.log (QA R9)
    controller = None

    def on_crash() -> None:
        if controller is not None:
            controller.emergency_restore()

    def on_error() -> None:
        if controller is not None:
            controller.recover_after_error()

    def ends_app() -> bool:
        # an exception in a Qt slot is reported while the event loop keeps running: VoxFlow carries on (QA R4)
        return QThread.currentThread().loopLevel() == 0

    if data_dir is not None:
        system.setup_logging(data_dir)
        system.install_crash_handler(data_dir, on_crash, on_error, ends_app)
        log.info("VoxFlow %s starting (demo=False, fake=%s, data=%s)", _version(), fake, data_dir)
        from . import modelhub
        modelhub.move_roaming_cuda_cache()     # before any model starts CUDA (QA R16)

    injector = context_fn = None
    if fake and not demo:
        # --fake: stand-in audio and models, but the real hotkeys, paste and app detection
        from .context import foreground_app
        from .inject import Injector
        injector, context_fn = Injector(), foreground_app
    try:
        controller = Controller(demo=demo, fake=fake, install_hook=not no_hook, injector=injector,
                                context_fn=context_fn)
    except Exception as exc:  # noqa: BLE001 - logged without its message; the user gets a box, not silence
        log.critical("VoxFlow could not start\n%s", system.redacted_traceback(exc))
        if lock is not None:
            lock.unlock()
        show_start_error("VoxFlow could not start because of an unexpected error. Details are in the log folder:\n"
                         f"{system.log_dir(data_dir or cfgmod.app_dir_path())}")
        system.shutdown_logging()
        return 1
    if data_dir is None:                      # demo: a throw-away profile made by the controller
        system.setup_logging(controller.data_dir)
        system.install_crash_handler(controller.data_dir, on_crash, on_error, ends_app)
        log.info("VoxFlow %s starting (demo=True, data=%s)", controller.version, controller.data_dir)
    atexit.register(controller.emergency_restore)

    if not demo:
        server = QLocalServer()
        QLocalServer.removeServer(_instance_name())
        if server.listen(_instance_name()):
            def on_connection() -> None:
                conn = server.nextPendingConnection()
                command = b""
                if conn is not None:
                    if not conn.bytesAvailable():
                        conn.waitForReadyRead(500)
                    command = bytes(conn.readAll().data()).strip()
                    conn.disconnectFromServer()
                if command == QUIT_COMMAND:    # VoxFlow.exe --quit (the uninstaller, an upgrade): save and close
                    log.info("asked to quit by another process")
                    app.quit()
                else:
                    controller.request_activate()
            server.newConnection.connect(on_connection)
        else:
            log.warning("single-instance channel unavailable: %s", server.errorString())

    from .ui.shell import build_ui
    show_window = open_settings or demo or not (controller.config.start_minimized or autostart)
    ui = build_ui(controller, show_window=show_window)
    app.aboutToQuit.connect(controller.shutdown)
    controller.start()

    if demo:
        step_no = [0]

        def step() -> None:
            state, mode, message, seconds = _DEMO_CYCLE[step_no[0] % len(_DEMO_CYCLE)]
            controller.demo_set_state(state, mode, message)
            if state == "done" and step_no[0] < 3 * len(_DEMO_CYCLE):
                controller.demo_add_transcript()
            step_no[0] += 1
            QTimer.singleShot(int(seconds * 1000), step)

        QTimer.singleShot(600, step)

    code = app.exec()
    controller.shutdown()
    closer = getattr(ui, "close", None)
    if callable(closer):
        closer()
    if server is not None:
        server.close()
    if lock is not None:
        lock.unlock()
    log.info("VoxFlow stopped (exit code %s)", code)
    system.shutdown_logging()
    return code
