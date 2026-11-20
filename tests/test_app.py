"""voxflow.app and the ``python -m voxflow`` entry point: module loading with stand-ins, the WAV reader,
``--selftest`` (every step reports pass/fail), and the Qt application start-up paths (demo, fake, single
instance, start minimised). Qt runs offscreen; the hook, clipboard and microphone are fakes."""
from __future__ import annotations

import logging
import sys
import threading
import wave

import numpy as np
import pytest

from voxflow import app, fakes
from voxflow.config import Config


# =========================================================================== module loading
def test_real_returns_none_for_missing_modules_and_attributes(caplog):
    with caplog.at_level("WARNING"):
        assert app._real("does_not_exist", "Thing") is None
        assert app._real("audio", "NoSuchClass") is None
    assert caplog.text.count("using a stand-in") == 2
    from voxflow.audio import Recorder
    assert app._real("audio", "Recorder") is Recorder


def test_make_modules_fake_mode_uses_stand_ins_without_importing_the_real_ones():
    Rec, asr, cleaner, notes = app.make_modules(Config(), fake=True)
    assert Rec is fakes.FakeRecorder and isinstance(asr, fakes.FakeTranscriber) and isinstance(cleaner, fakes.FakeCleaner)
    assert notes == ["audio stub", "asr stub", "cleanup stub"]


def test_make_modules_builds_the_real_classes_from_the_config(monkeypatch):
    built = {}

    class Rec:
        pass

    class Asr:
        def __init__(self, model, device):
            built["asr"] = (model, device)

    class Cln:
        def __init__(self, model, device, timeout_s):
            built["cleanup"] = (model, device, timeout_s)

    monkeypatch.setattr(app, "_real", lambda mod, attr: {"audio": Rec, "asr": Asr, "cleanup": Cln}[mod])
    cfg = Config(asr_model="my-asr", asr_device="cpu", cleanup_model="repo:file.gguf", cleanup_timeout_s=7.0)
    R, asr, cleaner, notes = app.make_modules(cfg)
    assert R is Rec and notes == []
    assert built == {"asr": ("my-asr", "cpu"), "cleanup": ("repo:file.gguf", "auto", 7.0)}


def test_make_modules_degrades_per_module_when_a_dependency_is_broken(monkeypatch):
    # the real app never falls back to the demo models (QA R1): a module that cannot be imported fails to load
    from voxflow.audio import Recorder
    monkeypatch.setattr(app, "_real", lambda mod, attr: Recorder if mod == "audio" else None)
    Rec, asr, cleaner, notes = app.make_modules(Config())
    assert Rec is Recorder and isinstance(asr, app.BrokenModel) and isinstance(cleaner, app.BrokenModel)
    assert notes == ["asr broken", "cleanup broken"]


def test_accepts_keywords():
    def plain(text):
        return text

    def rich(text, *, tone=None, terms=None):
        return text

    def anything(text, **kw):
        return text

    assert app._accepts_keywords(rich, ("tone", "terms")) and not app._accepts_keywords(rich, ("tone", "smart"))
    assert not app._accepts_keywords(plain, ("tone",)) and app._accepts_keywords(anything, ("whatever",))
    assert app._accepts_keywords(plain, ()) is True
    assert app._accepts_keywords(len, ("x",)) is False             # builtins without a signature


def test_run_pipeline_passes_options_only_to_cleaners_that_take_them():
    class Asr:
        def transcribe(self, audio):
            return " hello there "

    class Modern:
        ready = True

        def clean(self, text, *, tone=None, terms=None, language=None, smart=False):
            return f"{text}|{tone}|{terms}|{language}|{smart}"

    class Legacy:
        ready = True

        def clean(self, text):
            return text.upper()

    audio = np.zeros(10, dtype=np.float32)
    out = app.run_pipeline(audio, Asr(), Modern(), True, tone="casual", terms=["X"], language="en", smart=True)
    assert out["text"] == "hello there|casual|['X']|en|True" and out["cleaned"] and not out["dropped"]
    assert out["raw"] == "hello there" and out["t_asr"] >= 0 and out["t_clean"] >= 0
    assert app.run_pipeline(audio, Asr(), Legacy(), True, tone="casual")["text"] == "HELLO THERE"
    assert app.run_pipeline(audio, Asr(), Legacy(), False, tone="casual")["text"] == "hello there"


# =========================================================================== WAV reader
def write_wav(path, samples, rate=16000, width=2, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return str(path)


def tone(rate, seconds=0.5, freq=440.0):
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t)


def test_read_wav_16_bit_mono(tmp_path):
    x = tone(16000)
    a = app.read_wav_16k(write_wav(tmp_path / "a.wav", (x * 32767).astype("<i2")))
    assert a.dtype == np.float32 and a.shape == (8000,) and np.allclose(a, x, atol=1e-3)


def test_read_wav_stereo_is_downmixed(tmp_path):
    left, right = tone(16000, freq=440), np.zeros(8000)
    stereo = np.stack([left, right], axis=1)
    a = app.read_wav_16k(write_wav(tmp_path / "s.wav", (stereo * 32767).astype("<i2"), channels=2))
    assert a.shape == (8000,) and np.allclose(a, left / 2, atol=1e-3)


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_read_wav_resamples_to_16k_keeping_duration_and_pitch(tmp_path, rate):
    x = tone(rate, seconds=1.0, freq=300.0)
    a = app.read_wav_16k(write_wav(tmp_path / "r.wav", (x * 32767).astype("<i2"), rate=rate))
    assert abs(a.size - 16000) <= 1
    spectrum = np.abs(np.fft.rfft(a))
    assert abs(np.fft.rfftfreq(a.size, 1 / 16000)[int(np.argmax(spectrum))] - 300.0) < 3.0


def test_read_wav_8_bit_and_32_bit(tmp_path):
    x = tone(16000)
    eight = app.read_wav_16k(write_wav(tmp_path / "8.wav", (x * 127 + 128).astype(np.uint8), width=1))
    assert np.allclose(eight, x, atol=0.02) and eight.min() >= -1.0 and eight.max() <= 1.0
    wide = app.read_wav_16k(write_wav(tmp_path / "32.wav", (x * 2147483647).astype("<i4"), width=4))
    assert np.allclose(wide, x, atol=1e-3)


def test_read_wav_rejects_unsupported_files(tmp_path):
    x = (tone(16000) * 8388607).astype("<i4")
    packed = b"".join(int(v).to_bytes(3, "little", signed=True) for v in x)
    with wave.open(str(tmp_path / "24.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(16000)
        w.writeframes(packed)
    with pytest.raises(ValueError, match="unsupported sample width 3"):
        app.read_wav_16k(str(tmp_path / "24.wav"))
    with pytest.raises(FileNotFoundError):
        app.read_wav_16k(str(tmp_path / "missing.wav"))
    (tmp_path / "junk.wav").write_bytes(b"not a wave file")
    with pytest.raises(Exception):
        app.read_wav_16k(str(tmp_path / "junk.wav"))


def test_the_bundled_sample_is_readable_speech_length_audio():
    from pathlib import Path
    a = app.read_wav_16k(str(Path(__file__).with_name("tts_sample.wav")))
    assert a.dtype == np.float32 and 3.0 < a.size / 16000 < 30.0 and float(np.abs(a).max()) > 0.05


# =========================================================================== selftest
@pytest.fixture
def desktop(monkeypatch):
    """A working fake desktop for --selftest: hook start succeeds, clipboard is in-memory."""
    from tests.win32_fakes import install
    from voxflow import hotkey
    user32, _kernel32 = install(monkeypatch)
    user32.put_text("what the user had copied")
    monkeypatch.setattr(hotkey.HotkeyListener, "start", lambda self: True)
    monkeypatch.setattr(hotkey.HotkeyListener, "stop", lambda self: None)
    return user32


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_passes_with_stand_in_models_and_leaves_the_clipboard_alone(desktop, capsys):
    from pathlib import Path
    code = app.selftest(str(Path(__file__).with_name("tts_sample.wav")), fake=True)
    out = capsys.readouterr().out
    assert code == 0 and out.rstrip().endswith("SELFTEST PASSED")
    for line in ("[ok] UI imports", "[ok] keyboard hook", "installed + removed", "[ok] clipboard",
                 "previous content restored", "[note] using stand-ins: audio stub, asr stub, cleanup stub",
                 "[ok] speech model load", "backend demo", "[ok] cleanup model load", "run 1: audio", "run 2: audio",
                 "raw  : 'um so this is a demo transcript", "final: 'So this is a demo transcript"):
        assert line in out, line
    assert desktop.text() == "what the user had copied"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_without_a_wav_uses_silence(desktop, capsys):
    assert app.selftest(None, fake=True) == 0
    assert "no wav given - using 2 s of silence" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_never_touches_a_clipboard_it_cannot_restore(desktop, capsys):
    desktop.busy_opens = 10 ** 6                                  # another app holds the clipboard
    assert app.selftest(None, fake=True) == 0
    assert "skipped (clipboard is in use" in capsys.readouterr().out
    assert desktop.text() == "what the user had copied"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_reports_each_failing_step_and_exits_1(desktop, capsys, monkeypatch):
    from voxflow import hotkey
    monkeypatch.setattr(hotkey.HotkeyListener, "start", lambda self: setattr(self, "error", "access denied") or False)
    desktop.fail_next_sets = 1
    assert app.selftest(None, fake=True) == 1
    out = capsys.readouterr().out
    assert "[FAIL] keyboard hook: RuntimeError: access denied" in out
    assert "[FAIL] clipboard: RuntimeError: text did not round-trip" in out
    assert out.rstrip().endswith("SELFTEST FAILED")
    assert desktop.text() == "what the user had copied", "the clipboard is restored even when the check fails"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_with_a_bad_wav_or_a_model_that_will_not_load(desktop, capsys, monkeypatch, tmp_path):
    assert app.selftest(str(tmp_path / "missing.wav"), fake=True) == 1
    out = capsys.readouterr().out
    assert "[FAIL] reading" in out and "FileNotFoundError" in out and out.rstrip().endswith("SELFTEST FAILED")
    assert "Traceback" not in out

    monkeypatch.setattr(fakes.FakeTranscriber, "load", lambda self: (_ for _ in ()).throw(OSError("model missing")))
    assert app.selftest(None, fake=True) == 1
    out = capsys.readouterr().out
    assert "[FAIL] speech model load: OSError: model missing" in out and "run 1" not in out


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_cleanup_load_failure_and_pipeline_crash_are_failures_not_tracebacks(desktop, capsys, monkeypatch):
    monkeypatch.setattr(fakes.FakeCleaner, "load", lambda self: (_ for _ in ()).throw(OSError("gguf missing")))
    monkeypatch.setattr(fakes.FakeTranscriber, "transcribe", lambda self, audio: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app.selftest(None, fake=True) == 1
    out = capsys.readouterr().out
    assert "[FAIL] cleanup model load: OSError: gguf missing" in out
    assert "[FAIL] pipeline: RuntimeError: boom" in out and out.rstrip().endswith("SELFTEST FAILED")


def test_selftest_under_the_harness_guard_fails_cleanly(capsys):
    """Without the fake desktop the harness refuses hook + clipboard: selftest must report, not crash."""
    assert app.selftest(None, fake=True) == 1
    out = capsys.readouterr().out
    assert "[FAIL] keyboard hook" in out and "[FAIL] clipboard" in out and "SELFTEST FAILED" in out


# =========================================================================== entry point
def test_main_dispatches_arguments(monkeypatch):
    import voxflow.__main__ as entry
    calls = []
    monkeypatch.setattr(app, "run_app", lambda **kw: calls.append(("run", kw)) or 0)
    monkeypatch.setattr(app, "selftest", lambda wav, fake=False: calls.append(("selftest", wav, fake)) or 3)
    defaults = dict(demo=False, fake=False, open_settings=False, no_hook=False, autostart=False)
    assert entry.main([]) == 0 and calls[-1] == ("run", defaults)
    entry.main(["--demo", "--settings", "--no-hook"])
    assert calls[-1] == ("run", dict(defaults, demo=True, open_settings=True, no_hook=True))
    entry.main(["--fake", "--autostart"])
    assert calls[-1] == ("run", dict(defaults, fake=True, autostart=True))
    assert entry.main(["--selftest"]) == 3 and calls[-1] == ("selftest", None, False)
    assert entry.main(["--selftest", "clip.wav", "--fake"]) == 3 and calls[-1] == ("selftest", "clip.wav", True)
    with pytest.raises(SystemExit) as exc:
        entry.main(["--no-such-flag"])
    assert exc.value.code == 2


def test_help_documents_every_mode(capsys):
    import voxflow.__main__ as entry
    with pytest.raises(SystemExit) as exc:
        entry.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--demo", "--fake", "--selftest", "--settings", "--no-hook", "--autostart"):
        assert flag in out


# =========================================================================== the Qt application
@pytest.fixture
def clean_process_state(monkeypatch):
    """run_app installs logging handlers and exception hooks and builds a whole UI: put everything back afterwards."""
    from voxflow import controller as ctl_mod
    from voxflow import system
    from voxflow.ui import shell as shell_mod
    root = logging.getLogger("voxflow")
    handlers, level = list(root.handlers), root.level
    hooks = (sys.excepthook, threading.excepthook)
    built: list = []                                       # every Controller and Shell that run_app creates
    for cls in (ctl_mod.Controller, shell_mod.Shell):
        original = cls.__init__

        def recording_init(self, *args, _original=original, **kwargs):
            _original(self, *args, **kwargs)
            built.append(self)

        monkeypatch.setattr(cls, "__init__", recording_init)

    # run_app's demo cycle re-arms itself with QTimer.singleShot(callable), which cannot be cancelled: after the test
    # the chain kept firing inside whichever later test ran next, on a Controller that was already deleted
    # ("QTimer already deleted", found by shuffling the suite). Give those timers an owner the fixture can stop.
    from PySide6.QtCore import QTimer
    singles: list = []
    real_single_shot = QTimer.singleShot

    def tracked_single_shot(msec, *args):
        if len(args) == 1 and callable(args[0]):                  # the (msec, callable) form
            t = QTimer()
            t.setSingleShot(True)
            t.timeout.connect(args[0])
            singles.append(t)
            t.start(int(msec))
            return None
        return real_single_shot(msec, *args)

    monkeypatch.setattr(QTimer, "singleShot", staticmethod(tracked_single_shot))
    yield
    for t in singles:                                             # nothing scheduled by the app outlives the test
        t.stop()
        t.deleteLater()
    singles.clear()
    system.shutdown_logging()
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    sys.excepthook, threading.excepthook = hooks
    # run_app keeps its UI and controller for the life of the process. In the test process they must not outlive
    # the test: their timers and late signals would otherwise fire inside later, unrelated tests.
    from PySide6.QtCore import QCoreApplication, QEvent
    for obj in built:
        if isinstance(obj, shell_mod.Shell):
            obj.close()
            for part in (obj.onboarding, obj.window, obj.overlay, obj.tray.menu, obj.tray.icon, obj.tray):
                if part is not None:
                    part.deleteLater()
        else:
            obj.shutdown()
        obj.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


REAL_INSTANCE_NAME = app._instance_name


@pytest.fixture(autouse=True)
def _private_instance_channel(monkeypatch):
    """The single-instance channel is a per-user named pipe. Tests must never use the real name: a VoxFlow that is
    really running on this machine (or another pytest process) would answer on it."""
    import os
    import uuid
    name = f"VoxFlow-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(app, "_instance_name", lambda: name)


_quit_timers: list = []


@pytest.fixture(autouse=True)
def _stop_quit_timers():
    yield
    for timer in _quit_timers:
        timer.stop()
        timer.deleteLater()
    _quit_timers.clear()


def quit_after(ms, extra=None):
    """End the app's event loop about ``ms`` after it starts processing events.

    ``QApplication.quit()`` is a no-op while no loop is running (e.g. when a timer fires inside a nested
    ``processEvents`` during start-up), so the request is repeated every 50 ms until ``exec()`` returns."""
    import time

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    state = {"first": None, "ran_extra": False}

    def tick():
        now = time.monotonic()
        if state["first"] is None:
            state["first"] = now
        if (now - state["first"]) * 1000 < ms:
            return
        if extra is not None and not state["ran_extra"]:
            state["ran_extra"] = True
            extra()
        QApplication.instance().quit()

    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(tick)
    timer.start()
    _quit_timers.append(timer)


def voxflow_threads():
    return {t.name for t in threading.enumerate() if t.name.startswith("voxflow")}


def test_demo_app_starts_runs_and_stops_cleanly(qapp, clean_process_state, qt_messages, capsys):
    before = voxflow_threads()
    seen = {}

    def inspect():
        from PySide6.QtWidgets import QApplication
        seen["windows"] = [w.objectName() or type(w).__name__ for w in QApplication.topLevelWidgets() if w.isVisible()]
        seen["quit_on_close"] = QApplication.instance().quitOnLastWindowClosed()

    quit_after(900, inspect)
    assert app.run_app(demo=True) == 0
    assert seen["windows"], "demo mode must show its window"
    assert seen["quit_on_close"] is False, "closing the window must leave the tray app running"
    import time
    deadline = time.time() + 3
    while voxflow_threads() - before and time.time() < deadline:
        time.sleep(0.02)
    assert voxflow_threads() - before == set(), "threads left behind after the app exited"
    assert qt_messages == [], f"Qt warnings during a demo run: {qt_messages}"
    assert "Traceback" not in capsys.readouterr().err


def test_fake_app_takes_the_instance_lock_writes_a_log_and_releases_both(qapp, clean_process_state, data_dir):
    quit_after(500)
    assert app.run_app(fake=True, no_hook=True) == 0
    log_text = (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")
    assert "starting (demo=False, fake=True" in log_text and "VoxFlow stopped (exit code 0)" in log_text
    from PySide6.QtCore import QLockFile
    lock = QLockFile(str(data_dir / "voxflow.lock"))
    assert lock.tryLock(100), "the single-instance lock was not released on exit"
    lock.unlock()
    assert (data_dir / "config.json").exists()


def test_second_launch_wakes_the_first_instance_and_exits(qapp, clean_process_state, data_dir, capsys):
    from PySide6.QtCore import QLockFile
    from PySide6.QtNetwork import QLocalServer
    lock = QLockFile(str(data_dir / "voxflow.lock"))                # "the first instance"
    assert lock.tryLock(100)
    server = QLocalServer()
    QLocalServer.removeServer(app._instance_name())
    assert server.listen(app._instance_name())
    woken = []
    server.newConnection.connect(lambda: woken.append(server.nextPendingConnection()))
    try:
        assert app.run_app(fake=True, no_hook=True) == 0             # returns at once: no event loop is started
        qapp.processEvents()
        assert "already running" in capsys.readouterr().out
        assert len(woken) == 1, "the running instance was not asked to show its window"
        assert not (data_dir / "logs").exists(), "the second launch must not start a controller"
    finally:
        server.close()
        lock.unlock()


def test_running_instance_shows_its_window_when_a_second_launch_connects(qapp, clean_process_state, data_dir):
    from PySide6.QtCore import QTimer
    from PySide6.QtNetwork import QLocalSocket
    from voxflow import controller as ctl
    activations = []
    real_request = ctl.Controller.request_activate

    def spy(self):
        activations.append(True)
        real_request(self)

    ctl.Controller.request_activate = spy
    try:
        def knock():
            sock = QLocalSocket()
            sock.connectToServer(app._instance_name())
            assert sock.waitForConnected(1000)
            sock.write(b"show")
            sock.waitForBytesWritten(500)
            sock.disconnectFromServer()

        quit_after(900)
        QTimer.singleShot(300, knock)
        assert app.run_app(fake=True, no_hook=True) == 0
    finally:
        ctl.Controller.request_activate = real_request
    assert activations == [True]


@pytest.mark.parametrize("kwargs,config,expect_window", [
    (dict(), {}, True),
    (dict(autostart=True), {}, False),                                  # started by Windows at login
    (dict(), {"start_minimized": True}, False),
    (dict(open_settings=True), {"start_minimized": True}, True),        # --settings wins
])
def test_window_visibility_at_start(qapp, clean_process_state, data_dir, monkeypatch, kwargs, config, expect_window):
    import json
    (data_dir / "config.json").write_text(json.dumps(dict(config, onboarding_done=True)), encoding="utf-8")
    from voxflow.ui import shell
    shown = {}
    real_build = shell.build_ui

    def build(controller, *, show_window=False):
        shown["show_window"] = show_window
        return real_build(controller, show_window=show_window)

    monkeypatch.setattr(shell, "build_ui", build)
    quit_after(300)
    assert app.run_app(fake=True, no_hook=True, **kwargs) == 0
    assert shown["show_window"] is expect_window


def test_quitting_hides_everything_before_the_shutdown_that_may_take_seconds(qapp, clean_process_state, data_dir,
                                                                            monkeypatch):
    """QA C-02: with a dictation still being worked on the shutdown waits for it; the window, the tray icon and the
    Flow bar stayed on screen meanwhile and looked frozen."""
    import json
    (data_dir / "config.json").write_text(json.dumps({"onboarding_done": True}), encoding="utf-8")
    from voxflow import controller as ctl
    from voxflow.ui import shell
    order, built = [], {}
    real_build, real_shutdown = shell.build_ui, ctl.Controller.shutdown

    def build(controller, *, show_window=False):
        ui = real_build(controller, show_window=show_window)
        real_close = ui.close
        ui.close = lambda: (order.append("ui hidden"), real_close())[1]
        built["ui"] = ui
        return ui

    def shutdown(self):
        if not self._closed:
            order.append(("shutdown", built["ui"].window.isVisible(), built["ui"].tray.icon.isVisible()))
        real_shutdown(self)

    monkeypatch.setattr(shell, "build_ui", build)
    monkeypatch.setattr(ctl.Controller, "shutdown", shutdown)
    quit_after(300)
    assert app.run_app(fake=True, no_hook=True) == 0
    assert order[0] == "ui hidden" and ("shutdown", False, False) in order, order
    assert not [step for step in order if isinstance(step, tuple) and (step[1] or step[2])], "nothing visible by then"


def test_instance_name_is_per_user_and_safe(monkeypatch):
    import getpass
    monkeypatch.setattr(getpass, "getuser", lambda: "Dan O'Brien-Smith")
    name = REAL_INSTANCE_NAME()
    assert name.startswith("VoxFlow-instance-DanOBrienSmith") and name.replace("-", "").isalnum()
    assert name == REAL_INSTANCE_NAME(), "stable for one profile"
    monkeypatch.setattr(getpass, "getuser", lambda: (_ for _ in ()).throw(OSError("no user")))
    assert REAL_INSTANCE_NAME().startswith("VoxFlow-instance-user")


def test_demo_cycle_covers_every_flow_bar_state_in_both_modes():
    from voxflow.controller import MODES, STATES
    assert {step[0] for step in app._DEMO_CYCLE} == set(STATES)
    assert {step[1] for step in app._DEMO_CYCLE} == set(MODES)
    assert all(step[3] > 0 for step in app._DEMO_CYCLE)
