"""Regression tests for the robustness QA findings (R1-R19, see CHANGELOG): a damaged install, an unwritable
data folder, log redaction, the crash handler, a PC without a microphone, a read-only settings file, damaged
model files, a full disk, the CUDA cache location, crash reports and launch at login."""
from __future__ import annotations

import logging
import re
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.controller_rig import TAutostart, TCleaner, TInjector, TMixer, TRecorder
from tests.test_cov_controller import patch_os
from voxflow import app, controller as ctl_mod
from voxflow.config import Config
from voxflow.context import AppContext
from voxflow.controller import Controller

_REAL_INSTANCE_NAME = app._instance_name        # before tests/test_app.py's fixture gives each test a private one


def _real_controller(tmp_path, **kw):
    kw.setdefault("injector", TInjector())
    return Controller(fake=False, data_dir=tmp_path, install_hook=False, sound_backend=lambda wav: None,
                      mute_backend=TMixer(), context_fn=lambda: AppContext(), autostart=TAutostart(),
                      speed_check=False, updates=False, **kw)


# =========================================================================== R1: no demo stand-ins in the real app
def test_r1_a_broken_speech_module_fails_to_load_instead_of_pasting_the_demo_sentence(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch)                                           # says the model is cached...
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))       # ...but the cache is empty, as on the CI runner
    TRecorder.reset()
    real = app._real
    monkeypatch.setitem(sys.modules, "voxflow.asr", None)          # "import voxflow.asr" raises ImportError
    # only the speech module is really imported (and fails); never the real cleanup model
    monkeypatch.setattr(app, "_real", lambda mod, attr: {"audio": TRecorder, "cleanup": lambda *a: TCleaner()}.get(mod)
                        or real(mod, attr))
    c = _real_controller(tmp_path)
    try:
        c.start()
        qtbot.waitUntil(lambda: c.model_status("asr")["state"] == "failed", timeout=5000)
        detail = c.model_status("asr")["detail"]
        assert detail == "VoxFlow's installation is damaged (ModuleNotFoundError in asr.py). Reinstall VoxFlow."
        assert c._damaged == {}, "broken code is not blamed on the model files (no re-download on Retry)"
        assert c.status()["level"] == "error" and c.status()["title"] == "Speech model unavailable"
        assert c.model_status("asr")["backend"] != "demo"
        with pytest.raises(app.DamagedInstall):
            c._asr.transcribe(np.zeros(16000, dtype=np.float32))
    finally:
        c.shutdown()


def test_r1_a_broken_cleanup_module_pastes_as_heard_and_says_to_reinstall(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch)
    monkeypatch.setitem(sys.modules, "voxflow.cleanup", None)
    cfg = Config()
    Rec, asr, cleaner, notes = app.make_modules(cfg)
    assert isinstance(cleaner, app.BrokenModel) and notes == ["cleanup broken"]
    assert not cleaner.ready and not cleaner.has_llm and cleaner.clean("text") == "text"
    with pytest.raises(app.DamagedInstall, match="Error in cleanup.py"):
        cleaner.generate([], max_tokens=5, timeout_s=1)
    # the pipeline skips a cleaner that is not ready: the words are pasted as heard
    out = app.run_pipeline(np.zeros(16000, dtype=np.float32), type("A", (), {"transcribe": lambda s, a: "hi"})(),
                           cleaner)
    assert out["text"] == "hi"

    TRecorder.reset()
    monkeypatch.setattr(app, "make_modules", lambda c, fake, measured=None: (TRecorder, _ReadyAsr(), cleaner, []))
    c = _real_controller(tmp_path)
    try:
        c.start()
        qtbot.waitUntil(lambda: c.model_status("cleanup")["state"] == "failed", timeout=5000)
        assert "Reinstall VoxFlow" in c.model_status("cleanup")["detail"]
        assert "DamagedInstall" not in c.model_status("cleanup")["detail"]
    finally:
        c.shutdown()


class _ReadyAsr:
    backend = "cpu"

    def load(self):
        pass

    def transcribe(self, audio):
        return "hello"


def test_r1_a_broken_audio_module_gives_a_recorder_that_never_records(monkeypatch):
    monkeypatch.setattr(app, "_IMPORT_ERRORS", {})
    real = app._real
    monkeypatch.setattr(app, "_real", lambda mod, attr: real(mod, "NoSuchRecorder") if mod == "audio" else None)
    Rec, *_rest, notes = app.make_modules(Config())
    assert Rec is app.BrokenRecorder and notes[0] == "audio broken"
    rec = Rec("mic", max_seconds=3)
    assert "AttributeError in audio.py" in str(rec.error) and rec.alive is False
    for call in (rec.start, rec.reopen):
        with pytest.raises(app.DamagedInstall):
            call()
    assert rec.stop().size == 0 and rec.level() == 0.0 and rec.recent_samples(4).tolist() == [0, 0, 0, 0]
    rec.set_device("other")
    rec.close()
    assert rec.device == "other" and not rec.is_recording and not rec.hit_cap and rec.last_speech_seconds == 0
    assert str(app.damaged_install("nothing")) == "VoxFlow's installation is damaged (nothing.py is missing). " \
                                                  "Reinstall VoxFlow."


def test_r1_fake_mode_still_uses_the_stand_ins(monkeypatch):
    Rec, asr, cleaner, notes = app.make_modules(Config(), fake=True)
    assert notes == ["audio stub", "asr stub", "cleanup stub"]


# =========================================================================== R15: OpenBLAS thread buffers
@pytest.mark.parametrize("preset, expected", [(None, "1"), ("4", "4")])
def test_r15_openblas_uses_one_thread_unless_told_otherwise(preset, expected):
    import os
    import subprocess
    from pathlib import Path
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    if preset:
        env["OPENBLAS_NUM_THREADS"] = preset
    root = Path(__file__).resolve().parent.parent
    code = "import sys, os, voxflow.__main__; print(os.environ['OPENBLAS_NUM_THREADS'], 'numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == [expected, "False"], "set before numpy is imported"
    entry = (root / "packaging" / "voxflow_entry.py").read_text(encoding="utf-8")
    assert entry.index('setdefault("OPENBLAS_NUM_THREADS", "1")') < entry.index("from voxflow")


# =========================================================================== R2 / R9: start-up
from tests.test_app import _private_instance_channel, _stop_quit_timers, clean_process_state, quit_after  # noqa: E402,F401


@pytest.fixture
def start_errors(monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(app, "show_start_error", shown.append)
    return shown


def test_r2_a_data_folder_on_a_missing_drive_is_explained_not_a_traceback(qapp, clean_process_state, tmp_path,
                                                                          monkeypatch, start_errors):
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setenv("APPDATA", str(blocker / "nope"))          # like APPDATA=Q:\nope: the folder cannot exist
    assert app.run_app(fake=True, no_hook=True) == 1
    assert len(start_errors) == 1 and str(blocker / "nope" / "VoxFlow") in start_errors[0]
    assert start_errors[0].startswith("VoxFlow can't use its data folder")


@pytest.mark.parametrize("error, reason", [("PermissionError", "Windows did not allow VoxFlow to write there."),
                                           ("UnknownError", "VoxFlow could not create a file there.")])
def test_r2_a_lock_that_fails_for_another_reason_than_a_running_copy_says_so(qapp, clean_process_state, data_dir,
                                                                            monkeypatch, start_errors, capsys,
                                                                            error, reason):
    from PySide6.QtCore import QLockFile
    monkeypatch.setattr(QLockFile, "tryLock", lambda self, timeout=0: False)
    monkeypatch.setattr(QLockFile, "error", lambda self: getattr(QLockFile.LockError, error))
    assert app.run_app(fake=True, no_hook=True) == 1
    assert "already running" not in capsys.readouterr().out
    assert len(start_errors) == 1 and str(data_dir) in start_errors[0] and reason in start_errors[0]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
def test_r2_a_read_only_data_folder_is_not_mistaken_for_a_running_copy(qapp, clean_process_state, data_dir,
                                                                       start_errors, capsys):
    import getpass
    import subprocess
    user = getpass.getuser()
    subprocess.run(["icacls", str(data_dir), "/deny", f"{user}:(W,AD)"], check=True, capture_output=True)
    try:
        assert app.run_app(fake=True, no_hook=True) == 1
    finally:
        subprocess.run(["icacls", str(data_dir), "/remove:d", user], check=True, capture_output=True)
    assert "already running" not in capsys.readouterr().out
    assert len(start_errors) == 1 and "Windows did not allow VoxFlow to write there." in start_errors[0]


def test_r2_the_message_box_is_a_real_qt_box(monkeypatch, capsys):
    from PySide6.QtWidgets import QMessageBox
    seen = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text: seen.append((title, text)))
    app.show_start_error("why")
    assert seen == [("VoxFlow can't start", "why")] and "why" in capsys.readouterr().err


def test_r9_problems_while_the_controller_is_built_reach_voxflow_log(qapp, clean_process_state, data_dir):
    (data_dir / "config.json").write_text("{ not json", encoding="utf-8")
    (data_dir / "history.json").write_text("{ not json either", encoding="utf-8")
    quit_after(300)
    assert app.run_app(fake=True, no_hook=True) == 0
    log_text = (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")
    assert "config file" in log_text and "is unreadable" in log_text
    assert "history.json is unreadable" in log_text
    assert log_text.index("starting") < log_text.index("config file"), "the start line comes first"


def test_r9_an_exception_while_the_controller_is_built_is_logged_and_shown(qapp, clean_process_state, data_dir,
                                                                          monkeypatch, start_errors):
    words = "secret dictated words"

    def broken(*a, **k):
        raise RuntimeError(words)              # the source line shows the name, never the words

    monkeypatch.setattr(ctl_mod.Controller, "__init__", broken)
    assert app.run_app(fake=True, no_hook=True) == 1
    log_text = (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")
    assert "VoxFlow could not start" in log_text and "RuntimeError at test_robust_fixes.py" in log_text
    assert "secret dictated words" not in log_text
    assert len(start_errors) == 1 and str(data_dir / "logs") in start_errors[0]
    from PySide6.QtCore import QLockFile
    lock = QLockFile(str(data_dir / "voxflow.lock"))
    assert lock.tryLock(100), "the lock is released, so the next start is not taken for a running copy"
    lock.unlock()


# =========================================================================== R3: dictated text never in the log
SECRET = "insert my address secret medical diagnosis"


@pytest.fixture
def log_file(data_dir):
    from voxflow import system
    root = logging.getLogger("voxflow")
    handlers, level = list(root.handlers), root.level
    path = system.setup_logging(data_dir)
    yield path
    system.shutdown_logging()
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
    root.setLevel(level)


def test_r3_log_exception_never_writes_the_message(log_file, capsys):
    try:
        {}[SECRET]
    except KeyError:
        logging.getLogger("voxflow.anything").exception("something failed")
    text = log_file.read_text(encoding="utf-8")
    assert "something failed" in text and "KeyError at test_robust_fixes.py" in text and SECRET not in text
    assert SECRET not in capsys.readouterr().err, "nor on the console"
    from voxflow.system import RedactingFormatter
    plain = logging.Formatter().formatException((None, None, None))
    assert RedactingFormatter().formatException((None, None, None)) == plain     # nothing to redact


def test_r3_a_failing_rule_clean_up_does_not_log_the_dictation(tmp_path, qtbot, monkeypatch, log_file):
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        rig.c.snippets.matches = lambda text: text == "raw"
        rig.c._rule_cleaner = type("Broken", (), {"clean": lambda self, raw, **k: {}[SECRET]})()
        assert rig.c._personalise("cleaned", "raw", AppContext()) == "raw"
    finally:
        rig.close()
    text = log_file.read_text(encoding="utf-8")
    assert "rule clean-up for a snippet failed" in text and "KeyError" in text and SECRET not in text


def test_r3_store_listener_and_layout_failures_do_not_log_entries(data_dir, log_file):
    from voxflow import store
    st = store.Dictionary(data_dir)
    st.subscribe(lambda topic: {}[SECRET])
    st._notify("dictionary")
    (data_dir / "history.json").write_text('{"version": 2, "entries": "x"}', encoding="utf-8")

    class Picky(store.History):
        def _load(self, data):
            if data is not None:
                raise ValueError(SECRET)
            super()._load(data)

    Picky(data_dir)
    text = log_file.read_text(encoding="utf-8")
    assert "store listener failed" in text and "has an unexpected layout (ValueError at" in text
    assert SECRET not in text


def test_r3_a_failed_import_does_not_log_the_imported_text(tmp_path, qtbot, monkeypatch, log_file):
    from tests.controller_rig import make_rig
    from voxflow import sync as syncmod
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        path = tmp_path / "backup.json"
        rig.c.export_backup(str(path))

        def boom(*a, **k):
            raise ValueError(SECRET)

        monkeypatch.setattr(syncmod, "_apply", boom)
        with pytest.raises(ValueError, match="nothing was changed"):
            rig.c.import_backup(str(path))
    finally:
        rig.close()
    text = log_file.read_text(encoding="utf-8")
    assert "import failed; restoring the previous state" in text
    assert "restoring after a failed import also failed" in text and SECRET not in text


def test_r3_a_failed_sync_merge_does_not_log_the_merged_text(tmp_path, qtbot, monkeypatch, log_file):
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        monkeypatch.setattr(rig.c.sync, "merge", lambda result: {}[SECRET])
        monkeypatch.setattr(rig.c.sync, "note_unavailable", lambda exc: None)
        rig.c._on_sync_step("read", {"peers": []})
    finally:
        rig.close()
    text = log_file.read_text(encoding="utf-8")
    assert "sync merge failed" in text and "KeyError" in text and SECRET not in text


# =========================================================================== R4: errors VoxFlow survives
def test_r4_a_slot_error_keeps_muting_working_and_writes_no_crash_report(qapp, clean_process_state, data_dir,
                                                                        monkeypatch):
    from PySide6.QtCore import QTimer
    from voxflow import controller as ctl
    seen = {}

    def broken_slot():
        raise ZeroDivisionError("in a slot")

    def check():
        c = seen["controller"]
        c.config.mute_media = True
        c.muter._threaded = False
        c.muter.mute()
        seen["calls"] = list(c.muter._backend.calls)
        seen["notices"] = list(notices)

    notices = []
    real_start = ctl.Controller.start

    def start(self):
        seen["controller"] = self
        self.notice_action.connect(lambda m, a, label: notices.append(a))
        real_start(self)

    monkeypatch.setattr(ctl.Controller, "start", start)
    QTimer.singleShot(150, broken_slot)
    quit_after(700, check)
    assert app.run_app(fake=True, no_hook=True) == 0
    assert list((data_dir / "logs").glob("crash-*.log")) == []
    log_text = (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")
    assert "uncaught exception in main thread (VoxFlow keeps running)" in log_text
    assert seen["calls"][-1] == "mute", "muting still works after the error"
    assert seen["notices"] == ["open_log_folder"]


def test_r4_recover_after_error_unmutes_without_latching_and_tells_once(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        c = rig.c
        c.muter._threaded = False
        told = []
        c.notice_action.connect(lambda m, a, label: told.append((a, label)))
        c.muter.mute()
        c.recover_after_error()
        c.recover_after_error()
        qtbot.waitUntil(lambda: len(told) >= 1, timeout=2000)
        assert told == [("open_log_folder", "Open log folder")]
        assert c.muter._closing is False
        c.muter.mute()
        assert c.muter.active, "the next dictation still mutes other apps"
        monkeypatch.setattr(c.muter, "unmute_now", lambda: 1 / 0)
        c.recover_after_error()                                 # a failing unmute never raises from the hook
    finally:
        rig.close()


def test_r4_a_fatal_main_thread_error_still_runs_the_full_restore(tmp_path, monkeypatch):
    import threading
    from voxflow import system
    hooks = (sys.excepthook, threading.excepthook)
    calls = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)
    try:
        system.install_crash_handler(tmp_path, on_crash=lambda: calls.append("crash"),
                                     on_error=lambda: calls.append("error"), fatal=lambda: True)
        try:
            raise RuntimeError("x")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook, threading.excepthook = hooks
        system.shutdown_logging()
    assert calls == ["crash"] and len(list((tmp_path / "logs").glob("crash-*.log"))) == 1


# =========================================================================== R17: crash reports
def test_r17_only_the_ten_newest_crash_reports_are_kept(tmp_path):
    from voxflow import system
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(15):
        (logs / f"crash-20260101-0000{i:02d}.log").write_text("old", encoding="utf-8")
    try:
        raise RuntimeError("x")
    except RuntimeError:
        path = system.write_crash_report(tmp_path, *sys.exc_info())
    names = sorted(p.name for p in logs.glob("crash-*.log"))
    assert len(names) == system.CRASH_KEEP and path.name in names and "crash-20260101-000005.log" not in names


def test_r17_fault_log_is_started_afresh_above_a_megabyte(tmp_path, monkeypatch):
    import threading
    from voxflow import system
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "fault.log").write_bytes(b"x" * (system.FAULT_LOG_MAX + 1))
    hooks = (sys.excepthook, threading.excepthook)
    monkeypatch.setattr(system, "_fault_file", None)
    try:
        system.install_crash_handler(tmp_path)
    finally:
        sys.excepthook, threading.excepthook = hooks
        system.shutdown_logging()
    assert (logs / "fault.old.log").stat().st_size == system.FAULT_LOG_MAX + 1
    assert (logs / "fault.log").stat().st_size == 0
    system._cap_fault_log(logs / "missing.log")                   # nothing there: nothing to do


def test_r17_a_crash_of_the_previous_run_is_reported_once(tmp_path):
    import os
    import time
    from voxflow import system
    logs = tmp_path / "logs"
    assert system.new_crash_report(tmp_path) is None, "first run: nothing to compare with"
    assert system.new_crash_report(tmp_path) is None
    report = logs / "crash-20990101-000000.log"
    report.write_text("crash", encoding="utf-8")
    os.utime(report, (time.time() + 5, time.time() + 5))
    assert system.new_crash_report(tmp_path) == report
    assert system.new_crash_report(tmp_path) is None, "told once"
    # a native crash: new text in fault.log naming a fatal error; other new text is not a crash
    (logs / "fault.log").write_text("Windows fatal exception: access violation\n", encoding="utf-8")
    assert system.new_crash_report(tmp_path) == logs / "fault.log"
    with (logs / "fault.log").open("a", encoding="utf-8") as fh:
        fh.write("Thread 0x1 (most recent call first):\n")
    assert system.new_crash_report(tmp_path) is None
    (logs / "fault.log").write_text("Fatal Python error\n", encoding="utf-8")     # started afresh, smaller
    assert system.new_crash_report(tmp_path) == logs / "fault.log"
    (logs / "reports-seen.json").write_text("{broken", encoding="utf-8")
    assert system.new_crash_report(tmp_path) is None, "an unreadable record starts over quietly"


def test_r17_unreadable_report_folders_never_break_the_start(tmp_path, monkeypatch):
    from pathlib import Path
    from voxflow import system
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert system.new_crash_report(blocker) is None              # logs/ cannot even be created
    logs = tmp_path / "d" / "logs"
    logs.mkdir(parents=True)
    (logs / "fault.log").write_text("fatal", encoding="utf-8")
    (logs / "reports-seen.json").write_text('{"time": 0, "fault": 0}', encoding="utf-8")
    real_open = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked"))
                        if self.name == "fault.log" else real_open(self, *a, **k))
    assert system.new_crash_report(tmp_path / "d") is None


def test_r17_the_next_start_says_voxflow_closed_unexpectedly(tmp_path, qtbot, monkeypatch):
    import os
    import time
    from voxflow import system
    from tests.controller_rig import make_rig
    data = tmp_path / "data"
    system.new_crash_report(data)                                 # a profile that has run before
    report = data / "logs" / "crash-20990101-000000.log"
    report.write_text("crash", encoding="utf-8")
    os.utime(report, (time.time() + 5, time.time() + 5))
    told = []
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    rig.c.notice_action.connect(lambda m, a, label: told.append((m, a)))
    rig.start()
    try:
        assert told == [("VoxFlow closed unexpectedly last time. The details are in the log folder.",
                         "open_log_folder")]
    finally:
        rig.close()


def test_r17_the_open_log_folder_button_opens_it(qapp, monkeypatch):
    from voxflow.ui import shell
    opened = []
    fake = type("Shell", (), {"ctl": type("C", (), {"open_log_folder": lambda self: opened.append("log"),
                                                     "open_data_folder": lambda self: opened.append("data"),
                                                     "autostart_here": lambda self: opened.append("here")})()})()
    shell.Shell.run_action(fake, "open_log_folder")
    shell.Shell.run_action(fake, "open_data_folder")
    shell.Shell.run_action(fake, "autostart_here")
    fake.ctl = object()
    shell.Shell.run_action(fake, "open_log_folder")               # a controller without it: ignored
    assert opened == ["log", "data", "here"]


def test_r4_the_crash_callbacks_follow_the_controller_and_the_event_loop(qapp, clean_process_state, data_dir,
                                                                        monkeypatch, start_errors):
    import threading
    restored = []
    monkeypatch.setattr(ctl_mod.Controller, "emergency_restore", lambda self: restored.append(self))
    quit_after(200)
    assert app.run_app(fake=True, no_hook=True) == 0
    try:
        raise RuntimeError("after the loop ended")
    except RuntimeError:
        monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)
        sys.excepthook(*sys.exc_info())                          # no event loop running: this one ends the app
    assert len(restored) == 1 and len(list((data_dir / "logs").glob("crash-*.log"))) == 1

    def broken(self, *a, **k):                                   # errors before the controller exists
        t = threading.Thread(target=lambda: 1 / 0, name="voxflow-early")
        t.start()
        t.join()
        try:
            raise KeyError("x")
        except KeyError:
            sys.excepthook(*sys.exc_info())
        raise RuntimeError("no controller")

    monkeypatch.setattr(ctl_mod.Controller, "__init__", broken)
    assert app.run_app(fake=True, no_hook=True) == 1
    assert len(restored) == 1, "no controller yet: nothing to restore"
    assert app.run_app(demo=True) == 1                            # demo: no lock to release
    assert start_errors[-1].startswith("VoxFlow could not start because of an unexpected error")


def test_r17_pruning_and_capping_tolerate_what_they_cannot_change(tmp_path, monkeypatch):
    from pathlib import Path
    from voxflow import system
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(system.CRASH_KEEP + 2):
        (logs / f"crash-20260101-0000{i:02d}.log").write_text("old", encoding="utf-8")
    (logs / "fault.log").write_text("small", encoding="utf-8")
    system._cap_fault_log(logs / "fault.log")
    assert (logs / "fault.log").read_text(encoding="utf-8") == "small" and not (logs / "fault.old.log").exists()
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(PermissionError()))
    system.prune_crash_reports(tmp_path)
    assert len(list(logs.glob("crash-*.log"))) == system.CRASH_KEEP + 2, "a locked report is left for next time"


# =========================================================================== R5: a PC without a microphone
def test_r5_no_microphone_is_logged_once_and_retried_every_minute_not_every_tick(tmp_path, qtbot, monkeypatch, caplog):
    from tests.controller_rig import make_rig
    patch_os(monkeypatch)
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 0)
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    try:
        rec = rig.rec
        rec.alive = False
        rec.error = OSError("Error querying device -1")
        with caplog.at_level(logging.DEBUG, logger="voxflow.controller"):
            for _ in range(30):                                 # one minute of health ticks
                rig.c._health_tick()
        assert rec.reopened == 1, "PortAudio is not restarted every 2 s"
        assert caplog.text.count("microphone unavailable") == 1
        assert rig.c.status()["title"] == "Microphone problem"
        rig.c._health_tick()                                    # the minute is over: one more try
        assert rec.reopened == 2 and caplog.text.count("microphone unavailable") == 1
        rec.error = OSError("another reason")
        rig.c._check_mic()
        assert caplog.text.count("microphone unavailable") == 2, "a different error is logged again"
    finally:
        rig.close()


def test_r5_a_device_change_retries_at_once(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    patch_os(monkeypatch)
    count = {"n": 0}
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: count["n"])
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    try:
        rec = rig.rec
        rec.alive, rec.error = False, OSError("Error querying device -1")
        rig.c._health_tick()
        rig.c._health_tick()
        assert rec.reopened == 1
        count["n"] = 1                                          # a microphone was plugged in
        rec.error = None
        rig.c._health_tick()
        assert rec.reopened == 2 and "mic" not in rig.c._problems
    finally:
        rig.close()


def test_r5_a_failed_refresh_also_waits(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    patch_os(monkeypatch)
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    try:
        rig.rec.error = OSError("Error querying device -1")
        rig.c.refresh_microphones()
        assert rig.c._mic_retry_tick == rig.c._health_ticks + ctl_mod.MIC_RETRY_TICKS
    finally:
        rig.close()


def test_r1_a_recorder_of_a_damaged_install_says_to_reinstall(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    patch_os(monkeypatch)
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    try:
        rig.rec.error = app.damaged_install("audio")
        rig.c._check_mic()
        assert rig.c.status()["detail"].startswith("VoxFlow's installation is damaged")
    finally:
        rig.close()


# =========================================================================== R6: a read-only config.json
def test_r6_a_read_only_settings_file_fails_at_once_and_says_why(tmp_path, qtbot, monkeypatch):
    import os
    import stat
    import time
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    cfg = rig.data_dir / "config.json"
    os.chmod(cfg, stat.S_IREAD)                                   # the read-only attribute
    try:
        t = time.perf_counter()
        rig.c.set("sounds_enabled", not rig.c.get("sounds_enabled"))
        took = time.perf_counter() - t
        assert took < 0.5, f"the window froze for {took:.2f} s"
        notices = [m for lvl, m in rig.spy.notices() if lvl == "warning"]
        assert notices and "config.json is read-only, in " in notices[-1] and str(rig.data_dir) in notices[-1]
    finally:
        os.chmod(cfg, stat.S_IREAD | stat.S_IWRITE)
        rig.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs and sharing")
def test_r6_permissions_that_deny_writing_fail_at_once(tmp_path):
    import getpass
    import subprocess
    import time
    from voxflow import config as cfgmod
    path = tmp_path / "config.json"
    cfgmod.save_config(Config(), path)
    user = getpass.getuser()
    subprocess.run(["icacls", str(path), "/deny", f"{user}:(W,D)"], check=True, capture_output=True)
    real = cfgmod.os.replace

    def denied(src, dst):
        err = PermissionError(13, "Access is denied")
        err.winerror = 5
        raise err

    try:
        assert cfgmod.write_problem(path) == "Windows does not allow VoxFlow to change config.json"
        cfgmod.os.replace = denied
        t = time.perf_counter()
        with pytest.raises(PermissionError):
            cfgmod.save_config(Config(theme="dark"), path)
        assert time.perf_counter() - t < 0.5
    finally:
        cfgmod.os.replace = real
        subprocess.run(["icacls", str(path), "/remove:d", user], check=True, capture_output=True)
    assert cfgmod.write_problem(tmp_path / "missing.json") == ""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing rules")
def test_r6_a_reader_holding_the_file_is_still_waited_out(tmp_path):
    import threading
    from voxflow import config as cfgmod
    path = tmp_path / "config.json"
    cfgmod.save_config(Config(), path)
    fh = open(path, "rb")                                        # a reader: the rename fails with "access denied"
    assert cfgmod.write_problem(path) == "", "the file itself can be written: wait for the reader"
    threading.Timer(0.15, fh.close).start()
    cfgmod.save_config(Config(theme="dark"), path)
    assert cfgmod.load_config(path).theme == "dark"


def test_r6_sharing_violations_are_retried_without_probing(tmp_path, monkeypatch):
    import os
    from voxflow import config as cfgmod
    path = tmp_path / "config.json"
    cfgmod.save_config(Config(), path)
    real, fails = os.replace, [2]

    def flaky(src, dst):
        if fails[0]:
            fails[0] -= 1
            err = PermissionError(13, "sharing violation")
            err.winerror = 32
            raise err
        real(src, dst)

    monkeypatch.setattr(cfgmod.os, "replace", flaky)
    monkeypatch.setattr(cfgmod, "write_problem", lambda p: pytest.fail("not asked for a sharing violation"))
    cfgmod.save_config(Config(theme="light"), path)
    assert fails == [0] and cfgmod.load_config(path).theme == "light"


def test_r6_off_windows_only_the_read_only_attribute_is_checked(tmp_path, monkeypatch):
    from voxflow import config as cfgmod
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfgmod.sys, "platform", "linux")
    assert cfgmod.write_problem(path) == ""


def test_r6_a_save_failure_that_is_not_about_the_file_keeps_the_general_message(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    from voxflow import config as cfgmod
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        monkeypatch.setattr(cfgmod, "save_config", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "disk full")))
        rig.c.set("sounds_enabled", not rig.c.get("sounds_enabled"))
        notices = [m for lvl, m in rig.spy.notices() if lvl == "warning"]
        assert notices[-1] == ("Your change is active, but it could not be saved to disk and will be lost when "
                               "VoxFlow closes.")
    finally:
        rig.close()


# =========================================================================== R7 / R10 / R18: model files
import hashlib  # noqa: E402

from voxflow import modelhub  # noqa: E402

REPO = "org/tiny-onnx"
REV = "a" * 40
FILES = {"config.json": b'{"model_type": "x"}', "encoder-model.onnx": b"\x08\x01" * 500, "vocab.txt": b"a\nb\n"}


def _git(data: bytes) -> str:
    return "git:" + hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A Hugging Face cache in a temp folder with one pinned repo, fully downloaded; nothing reaches the network."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    manifest = {name: (len(data), _git(data) if name.endswith((".json", ".txt"))
                       else hashlib.sha256(data).hexdigest()) for name, data in FILES.items()}
    monkeypatch.setitem(modelhub.PINNED, REPO, (REV, manifest))
    monkeypatch.setitem(modelhub.ASR_REPOS, "tiny", REPO)
    snap = modelhub.snapshot_dir(REPO)
    snap.mkdir(parents=True)
    for name, data in FILES.items():
        (snap / name).write_bytes(data)
    fetched = []

    def snapshot_download(repo, revision=None, allow_patterns=None, **kw):
        fetched.append(("snapshot", repo, revision, tuple(allow_patterns)))
        for name in allow_patterns:
            (snap / name).write_bytes(FILES[name])
        return str(snap)

    def hf_hub_download(repo, filename, revision=None, local_files_only=False, **kw):
        fetched.append(("file", repo, revision, filename, local_files_only))
        if repo != REPO:
            if local_files_only:
                raise FileNotFoundError(filename)
            return f"/downloaded/{repo}/{filename}"
        (snap / filename).write_bytes(FILES[filename])
        return str(snap / filename)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(modelhub.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 ** 12})())
    return type("Hub", (), {"snap": snap, "fetched": fetched, "manifest": manifest})


def test_r18_the_default_models_are_pinned_to_a_commit_with_every_file_hashed():
    for repo in (modelhub.ASR_REPOS["nemo-parakeet-tdt-0.6b-v3"], "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"):
        rev, files = modelhub.PINNED[repo]
        assert len(rev) == 40 and files
        for size, digest in files.values():
            assert size > 0 and len(digest.rpartition(":")[2]) in (40, 64)
    v3 = modelhub.ASR_REPOS["nemo-parakeet-tdt-0.6b-v3"]
    assert sum(s for s, _ in modelhub.PINNED[v3][1].values()) == modelhub.KNOWN_SIZES[v3]
    from voxflow import cleanup
    repo, filename = modelhub.cleanup_repo(cleanup.DEFAULT_MODEL)
    assert filename in modelhub.PINNED[repo][1], "the default cleanup model is pinned"


def test_r18_a_pinned_model_is_cached_only_with_every_file_at_its_size(hub):
    assert modelhub.is_cached(REPO) and modelhub.is_cached(REPO, "vocab.txt")
    (hub.snap / "encoder-model.onnx").write_bytes(b"cut")
    assert not modelhub.is_cached(REPO) and modelhub.is_cached(REPO, "vocab.txt")
    assert (hub.snap / "encoder-model.onnx").exists(), "a question never deletes anything"
    assert not modelhub.is_cached(REPO, "other.gguf"), "a file outside the pin: the plain cache check"


def test_r18_model_folder_fetches_the_pinned_commit_and_replaces_a_cut_off_file(hub):
    assert modelhub.model_folder("not-pinned") is None
    assert modelhub.model_folder("tiny") == str(hub.snap) and hub.fetched == []
    (hub.snap / "encoder-model.onnx").write_bytes(b"cut")           # truncated: removed and fetched again
    (hub.snap / "vocab.txt").unlink()
    assert modelhub.model_folder("tiny") == str(hub.snap)
    assert hub.fetched == [("snapshot", REPO, REV, ("encoder-model.onnx", "vocab.txt"))]
    assert (hub.snap / "encoder-model.onnx").read_bytes() == FILES["encoder-model.onnx"]


def test_r18_model_file_uses_the_pin_and_falls_back_to_main_for_other_files(hub):
    assert modelhub.model_file(REPO, "vocab.txt") == str(hub.snap / "vocab.txt") and hub.fetched == []
    (hub.snap / "vocab.txt").unlink()
    assert modelhub.model_file(REPO, "vocab.txt") == str(hub.snap / "vocab.txt")
    assert hub.fetched == [("file", REPO, REV, "vocab.txt", False)]
    hub.fetched.clear()
    assert modelhub.model_file("org/other", "x.gguf") == "/downloaded/org/other/x.gguf"
    assert hub.fetched == [("file", "org/other", None, "x.gguf", True), ("file", "org/other", None, "x.gguf", False)]


def test_r10_a_download_that_does_not_fit_is_refused_with_the_numbers(hub, monkeypatch, tmp_path):
    monkeypatch.setattr(modelhub.shutil, "disk_usage", lambda p: type("U", (), {"free": 1_200_000_000})())
    monkeypatch.setitem(modelhub.PINNED, REPO, (REV, dict(hub.manifest, **{"big.bin": (2_500_000_000, "0" * 64)})))
    with pytest.raises(modelhub.NotEnoughSpace) as err:
        modelhub.model_folder("tiny")
    assert str(err.value) == f"Needs 2.7 GB free on {tmp_path.anchor}, 1.2 GB available" and err.value.errno == 28
    assert hub.fetched == []
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "not" / "yet" / "there"))   # measured on the parent
    with pytest.raises(modelhub.NotEnoughSpace):
        modelhub.check_space(REPO, 2_000_000_000)
    monkeypatch.setattr(modelhub.shutil, "disk_usage", lambda p: (_ for _ in ()).throw(OSError("no drive")))
    modelhub.check_space(REPO, 2_000_000_000)                       # cannot tell: let the download try


def test_r10_a_full_disk_is_recognised_by_its_error_code(monkeypatch):
    assert modelhub.disk_full(OSError(28, os.strerror(28)))
    full = OSError(0, "x")
    full.winerror = 112
    assert modelhub.disk_full(full)
    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as inner:
            raise RuntimeError("download failed") from inner
    except RuntimeError as outer:
        assert modelhub.disk_full(outer)
    assert modelhub.disk_full(RuntimeError("not enough space on the disk"))
    assert not modelhub.disk_full(RuntimeError("other")) and not modelhub.disk_full(None)
    from voxflow.controller import _model_failure
    assert _model_failure("speech", OSError(28, os.strerror(28))) == "There is not enough disk space for the speech model."


def test_r10_unfinished_downloads_and_damaged_files_can_be_removed(hub, monkeypatch):
    blobs = modelhub.repo_dir(REPO) / "blobs"
    blobs.mkdir()
    (blobs / "abc.incomplete").write_bytes(b"x")
    (blobs / "keep").write_bytes(b"x")
    real_unlink = Path.unlink
    assert modelhub.remove_partial(REPO) == 1 and sorted(p.name for p in blobs.iterdir()) == ["keep"]
    (blobs / "locked.incomplete").write_bytes(b"x")
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(PermissionError()))
    assert modelhub.remove_partial(REPO) == 0
    modelhub.remove_files(REPO, ["vocab.txt"])                      # a locked file stays; no error
    monkeypatch.setattr(Path, "unlink", real_unlink)
    # a snapshot entry that links to its blob: both go
    link = hub.snap / "vocab.txt"
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == link)
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: blobs / "keep" if self == link else self)
    modelhub.remove_files(REPO, ["vocab.txt", "config.json"])
    assert not link.exists() and not (blobs / "keep").exists() and not (hub.snap / "config.json").exists()
    assert modelhub._snapshots("org/none") == []


def test_r7_damaged_files_are_found_by_their_hash(hub, monkeypatch):
    assert modelhub.damaged_files("") == [] and modelhub.damaged_files(REPO) == []
    data = bytearray(FILES["encoder-model.onnx"])
    data[10] ^= 0xFF                                                # same size, different bytes
    (hub.snap / "encoder-model.onnx").write_bytes(bytes(data))
    (hub.snap / "vocab.txt").write_bytes(b"a\nc\n")
    assert modelhub.damaged_files(REPO) == ["encoder-model.onnx", "vocab.txt"]
    assert modelhub.damaged_files(REPO, "vocab.txt") == ["vocab.txt"]
    monkeypatch.setattr(modelhub, "file_hash", lambda path, kind: (_ for _ in ()).throw(OSError("locked")))
    assert modelhub.damaged_files(REPO, "config.json") == ["config.json"], "a file that cannot be read is damaged"


def test_r7_an_unpinned_model_is_checked_against_the_hubs_list(hub, monkeypatch):
    other = "org/custom-gguf"
    snap = modelhub.repo_dir(other) / "snapshots" / ("b" * 40)
    snap.mkdir(parents=True)
    (snap / "good.gguf").write_bytes(b"good")
    (snap / "bad.gguf").write_bytes(b"bad!")
    (snap / "local-only.txt").write_bytes(b"x")
    (modelhub.repo_dir(other) / "snapshots" / ("c" * 40)).mkdir()      # an empty snapshot is skipped
    asked = []

    class Api:
        def model_info(self, repo, revision=None, files_metadata=False):
            asked.append((repo, revision, files_metadata))
            sib = lambda name, data, lfs: type("S", (), {           # noqa: E731
                "rfilename": name, "size": len(data), "blob_id": _git(data)[4:],
                "lfs": type("L", (), {"sha256": hashlib.sha256(data).hexdigest()})() if lfs else None})()
            return type("I", (), {"siblings": [sib("good.gguf", b"good", True), sib("bad.gguf", b"BAD!", True),
                                                sib("README.md", b"r", False),
                                                type("S", (), {"rfilename": "x", "size": None})()]})()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert modelhub.damaged_files(other) == ["bad.gguf"]
    assert ("org/custom-gguf", "b" * 40, True) in asked
    assert modelhub.damaged_files(other, "good.gguf") == []

    class Offline:
        def model_info(self, *a, **k):
            raise ConnectionError("offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", Offline)
    assert modelhub.damaged_files(other) == [], "offline: nothing can be checked, nothing is called damaged"


# ---- the controller: messages and Retry
class _FailingAsr:
    backend = ""

    def __init__(self, exc):
        self.exc = exc

    def load(self):
        raise self.exc


def _model_rig(tmp_path, qtbot, monkeypatch, asr=None, cleaner=None):
    from tests.controller_rig import TCleaner, make_rig
    monkeypatch.setattr(ctl_mod.modelhub, "asr_repo", lambda model: REPO)
    monkeypatch.setattr(ctl_mod.modelhub, "cleanup_repo", lambda model: (REPO, "vocab.txt"))
    return make_rig(tmp_path, qtbot, monkeypatch, fake=False, asr=asr, cleaner=cleaner or TCleaner())


def test_r7_damaged_speech_model_files_are_named_and_retry_downloads_them_again(hub, tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 1)
    (hub.snap / "vocab.txt").write_bytes(b"a\nc\n")                 # damaged, same size: "cached"
    rig = _model_rig(tmp_path, qtbot, monkeypatch, asr=_FailingAsr(RuntimeError("INVALID_PROTOBUF : Load model "
                                                                                "from C:\\Users\\someone\\x")))
    try:
        st = rig.c.model_status("asr")
        assert st["state"] == "failed"
        assert st["detail"] == "The speech model files are damaged. Choose Retry to download them again."
        rig.c.reload_models(["asr"])                                # Retry
        assert not (hub.snap / "vocab.txt").exists(), "the damaged file is removed, so it is downloaded again"
        rig.wait(lambda: rig.c.model_status("asr")["state"] == "failed")
        assert "asr" not in rig.c._damaged
    finally:
        rig.close()


def test_r7_a_load_failure_with_intact_files_keeps_the_usual_message(hub, tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 1)
    rig = _model_rig(tmp_path, qtbot, monkeypatch, asr=_FailingAsr(RuntimeError("driver")))
    try:
        assert rig.c.model_status("asr")["detail"].startswith("The speech model could not be loaded (RuntimeError")
        assert rig.c._damaged == {}
    finally:
        rig.close()


def test_r10_a_full_disk_while_downloading_says_so_and_cleans_up(hub, tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 1)
    (hub.snap / "vocab.txt").unlink()                               # not cached: a download
    blobs = modelhub.repo_dir(REPO) / "blobs"
    blobs.mkdir()
    (blobs / "x.incomplete").write_bytes(b"x")
    rig = _model_rig(tmp_path, qtbot, monkeypatch, asr=_FailingAsr(modelhub.NotEnoughSpace(3_000_000_000,
                                                                                         1_000_000_000, "D:\\")))
    try:
        assert rig.c.model_status("asr")["detail"] == ("There is not enough disk space for the speech model. Needs "
                                                       "3.0 GB free on D:\\, 1.0 GB available. Free some space, "
                                                       "then choose Retry.")
        assert not (blobs / "x.incomplete").exists()
    finally:
        rig.close()


def test_r7_r10_cleanup_model_file_problems_are_reported_not_hidden_behind_basic_rules(hub, tmp_path, qtbot,
                                                                                      monkeypatch):
    from tests.controller_rig import TAsr, TCleaner
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 1)

    class RulesOnly(TCleaner):
        def __init__(self, error):
            super().__init__()
            self.error = error

        def load(self):
            self.backend, self.ready, self.load_error = "rules", True, self.error

    (hub.snap / "vocab.txt").write_bytes(b"a\nc\n")
    rig = _model_rig(tmp_path, qtbot, monkeypatch, asr=TAsr(), cleaner=RulesOnly(RuntimeError("bad magic")))
    try:
        assert rig.c.model_status("cleanup") == {
            "state": "failed", "detail": "The cleanup model files are damaged. Choose Retry to download them again.",
            "progress": None, "backend": ""}
    finally:
        rig.close()
    rig = _model_rig(tmp_path, qtbot, monkeypatch, asr=TAsr(), cleaner=RulesOnly(OSError(28, "No space")))
    try:
        assert rig.c.model_status("cleanup")["detail"].startswith("There is not enough disk space for the cleanup")
    finally:
        rig.close()
    rig = _model_rig(tmp_path, qtbot, monkeypatch, asr=TAsr(), cleaner=RulesOnly(None))
    try:
        assert rig.c.model_status("cleanup")["state"] == "degraded", "no error: the usual basic-rules notice"
    finally:
        rig.close()


def test_r7_the_cleaner_remembers_why_the_language_model_did_not_load(monkeypatch):
    from voxflow import cleanup
    c = cleanup.Cleaner(model="org/repo:file.gguf", device="cpu")
    monkeypatch.setattr(cleanup, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(c, "_resolve_path", lambda: (_ for _ in ()).throw(OSError(28, "No space left on device")))
    c.load()
    assert c.backend == "rules" and isinstance(c.load_error, OSError)
    monkeypatch.setattr(c, "_resolve_path", lambda: "x.gguf")
    monkeypatch.setattr(cleanup, "_select_llama_libs", lambda device: "cpu")
    monkeypatch.setattr(c, "_build", lambda path, gpu: (_ for _ in ()).throw(ValueError("bad magic")))
    c.load()
    assert isinstance(c.load_error, RuntimeError) and isinstance(c.load_error.__cause__, ValueError)
    c.use_llm = False
    c.load()
    assert c.load_error is None


def test_r7_a_cut_off_file_is_damaged_without_reading_it(hub, monkeypatch):
    (hub.snap / "encoder-model.onnx").write_bytes(b"cut")
    monkeypatch.setattr(modelhub, "file_hash", lambda path, kind: pytest.fail("the size already tells"))
    assert modelhub.damaged_files(REPO, "encoder-model.onnx") == ["encoder-model.onnx"]


def test_r18_the_speech_model_is_loaded_from_the_pinned_folder(monkeypatch):
    import types
    from voxflow import asr
    seen = []
    fake = types.ModuleType("onnx_asr")
    fake.load_model = lambda name, path=None, **kw: seen.append((name, path)) or "model"
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)
    monkeypatch.setattr(asr.modelhub, "model_folder", lambda name: f"C:/cache/{name}")
    asr._build_cpu("nemo-parakeet-tdt-0.6b-v3")
    assert seen == [("nemo-parakeet-tdt-0.6b-v3", "C:/cache/nemo-parakeet-tdt-0.6b-v3")]


# =========================================================================== R18: dependencies pinned by hash
def _pins(path: Path) -> dict[str, str]:
    from packaging.utils import canonicalize_name
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\;#]+)", line.strip())
        if m:
            out[canonicalize_name(m.group(1))] = m.group(2)
    return out


ROOT = Path(__file__).resolve().parent.parent


def test_r18_the_release_installs_every_dependency_pinned_by_hash():
    lock = (ROOT / "requirements-release.txt").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[a-z0-9])", lock.split("\n", 4)[-1].strip())
    for entry in entries:
        assert "--hash=sha256:" in entry, entry.splitlines()[0]
    pinned = _pins(ROOT / "requirements-release.txt")
    for req in ("requirements.txt", "requirements-dev.txt"):
        for name, version in _pins(ROOT / req).items():
            assert pinned.get(name) == version, f"{name} {version} in {req} but {pinned.get(name)} in the lock"
    assert "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124" in lock
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "pip install --require-hashes --only-binary=:all: -r requirements-release.txt" in release
    assert "-r requirements-dev.txt" not in release


def test_r18_the_lock_script_lists_the_whole_dependency_tree_and_only_wheels_for_this_platform(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lock_requirements", ROOT / "scripts" / "lock_requirements.py")
    lock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lock)
    assert "pytest" in lock.roots(ROOT / "requirements-dev.txt") and "numpy" in lock.roots(ROOT / "requirements-dev.txt")
    tree = lock.closure(["pytest"])
    assert {"pytest", "pluggy", "iniconfig", "packaging"} <= set(tree)
    pypi = {"urls": [{"filename": "pkg-1.0-py3-none-any.whl", "digests": {"sha256": "a" * 64}},
                     {"filename": "pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl", "digests": {"sha256": "b" * 64}},
                     {"filename": "pkg-1.0.tar.gz", "digests": {"sha256": "c" * 64}}]}
    gh = {"assets": [{"name": "llama_cpp_python-0.3.36-py3-none-win_amd64.whl", "digest": "sha256:" + "d" * 64},
                     {"name": "notes.txt"}]}
    monkeypatch.setattr(lock, "_get", lambda url: gh if "github" in url else pypi)
    assert lock.wheel_hashes("pkg", "1.0") == ["a" * 64]
    assert lock.wheel_hashes("llama-cpp-python", "0.3.36") == ["d" * 64]
    pypi["urls"] = pypi["urls"][1:]
    with pytest.raises(SystemExit, match="no wheel of pkg==1.0"):
        lock.wheel_hashes("pkg", "1.0")


# =========================================================================== R16: the CUDA cache in Local AppData
def test_r16_the_roaming_cuda_cache_is_moved_once_keeping_the_kernels(tmp_path, monkeypatch):
    roaming, local = tmp_path / "Roaming", tmp_path / "Local"
    old = roaming / "VoxFlow" / "cuda-cache"
    (old / "ComputeCache").mkdir(parents=True)
    (old / "ComputeCache" / "index").write_bytes(b"kernels")
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("CUDA_CACHE_PATH", str(local / "VoxFlow" / "cuda-cache"))
    assert modelhub.move_roaming_cuda_cache() == "moved"
    assert not old.exists() and (local / "VoxFlow" / "cuda-cache" / "ComputeCache" / "index").read_bytes() == b"kernels"
    assert modelhub.move_roaming_cuda_cache() == "", "once"
    old.mkdir(parents=True)                                      # the new one exists already: the old one goes
    assert modelhub.move_roaming_cuda_cache() == "removed" and not old.exists()
    old.mkdir(parents=True)
    monkeypatch.setenv("CUDA_CACHE_PATH", str(old))             # a setting that points there: left alone
    assert modelhub.move_roaming_cuda_cache() == "" and old.exists()
    monkeypatch.setenv("CUDA_CACHE_PATH", str(local / "elsewhere" / "cache"))
    monkeypatch.setattr(modelhub.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError(17, "other drive")))
    assert modelhub.move_roaming_cuda_cache() == "removed" and not old.exists()
    monkeypatch.delenv("APPDATA")
    assert modelhub.move_roaming_cuda_cache() == ""


def test_r16_the_app_moves_it_before_the_models_load(qapp, clean_process_state, data_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(modelhub, "move_roaming_cuda_cache", lambda: calls.append("moved") or "")
    real_init = ctl_mod.Controller.__init__
    monkeypatch.setattr(ctl_mod.Controller, "__init__",
                        lambda self, *a, **k: calls.append("controller") or real_init(self, *a, **k))
    quit_after(200)
    assert app.run_app(fake=True, no_hook=True) == 0
    assert calls == ["moved", "controller"]
    iss = (ROOT / "packaging" / "voxflow.iss").read_text(encoding="utf-8")
    assert r'Type: filesandordirs; Name: "{localappdata}\VoxFlow\cuda-cache"' in iss


# =========================================================================== R12 / R13: installer and uninstaller
def test_r12_voxflow_quit_closes_a_running_copy_the_normal_way(qapp, clean_process_state, data_dir, monkeypatch):
    import subprocess
    # the child names the channel the real way; with this test's own data folder in it, that name is this test's
    monkeypatch.setattr(app, "_instance_name", _REAL_INSTANCE_NAME)
    from PySide6.QtCore import QTimer
    child = {}

    def ask():
        child["p"] = subprocess.Popen([sys.executable, "-m", "voxflow", "--quit"], cwd=ROOT,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    QTimer.singleShot(300, ask)
    quit_after(20000)                                               # safety net only
    assert app.run_app(fake=True, no_hook=True) == 0
    _out, err = child["p"].communicate(timeout=20)
    assert child["p"].returncode == 0, err
    log_text = (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")
    assert "asked to quit by another process" in log_text and "VoxFlow stopped (exit code 0)" in log_text


def test_r12_quit_without_a_running_copy_and_with_one_that_does_not_answer(data_dir, monkeypatch):
    from PySide6.QtCore import QLockFile
    assert app.request_quit(0.2) == 0                               # nothing running
    lock = QLockFile(str(data_dir / "voxflow.lock"))
    assert lock.tryLock(100)
    try:
        assert app.request_quit(0.3) == 1, "an old version ignores the request: the caller ends it instead"
    finally:
        lock.unlock()
    monkeypatch.setattr(app.cfgmod, "app_dir", lambda: (_ for _ in ()).throw(FileNotFoundError(3, "no drive")))
    assert app.request_quit(0.2) == 0


def test_r12_main_routes_quit_before_anything_else(monkeypatch):
    from voxflow import __main__ as main_mod, system
    monkeypatch.setattr(app, "request_quit", lambda: 7)
    monkeypatch.setattr(system, "update_setup_running", lambda: pytest.fail("--quit must work while Setup runs"))
    assert main_mod.main(["--quit"]) == 7


def test_r12_r13_the_installer_asks_voxflow_to_quit_and_starts_from_an_empty_runtime_folder():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text(encoding="utf-8")
    assert "taskkill" not in iss.lower(), "no longer every VoxFlow.exe of every copy, by force"
    code = iss.split("\n[Code]\n", 1)[1]
    stop = code.split("procedure StopRunningCopy;", 1)[1].split("\nend;", 1)[0]
    assert "'--quit'" in stop and "ewWaitUntilTerminated" in stop
    assert "StartsWith(''' + Folder + '''" in stop and "StringChangeEx(Folder, '''', '''''', True)" in stop
    prepare = code.split("function PrepareToInstall", 1)[1].split("\nend;", 1)[0]
    uninstall = code.split("procedure CurUninstallStepChanged", 1)[1]
    assert "StopRunningCopy;" in prepare and "StopRunningCopy;" in uninstall
    deletes = iss.split("\n[InstallDelete]\n", 1)[1].split("\n[", 1)[0]
    assert [ln for ln in deletes.splitlines() if ln and not ln.startswith(";")] == [
        r'Type: filesandordirs; Name: "{app}\_internal"']
    assert iss.index("[InstallDelete]") < iss.index("[Files]")


# =========================================================================== R19: launch at login names this copy
@pytest.fixture
def run_key(monkeypatch):
    from tests.test_system import FakeRegistry
    from voxflow import system
    fake = FakeRegistry()
    monkeypatch.setattr(system, "registry", fake)
    return fake


def test_r19_the_run_entry_is_compared_with_this_copy(run_key, tmp_path, monkeypatch):
    from voxflow import system
    assert system.autostart_state() == ("off", "")
    system.set_autostart(True)
    assert system.autostart_state() == ("this", "")
    other = tmp_path / "Other" / "VoxFlow.exe"
    other.parent.mkdir()
    other.write_text("")
    run_key.values[system.RUN_VALUE] = f'"{other}" --autostart'
    assert system.autostart_state() == ("other", str(other))
    run_key.values[system.RUN_VALUE] = f'"{tmp_path / "Moved" / "VoxFlow.exe"}" --autostart'
    assert system.autostart_state() == ("missing", str(tmp_path / "Moved" / "VoxFlow.exe"))
    # a copy run from source: the folder in its -c code counts too
    gone = system.launch_command(sys.executable, tmp_path / "deleted checkout")
    run_key.values[system.RUN_VALUE] = gone
    assert system.autostart_state() == ("missing", str(tmp_path / "deleted checkout"))
    (tmp_path / "other checkout").mkdir()
    run_key.values[system.RUN_VALUE] = system.launch_command(sys.executable, tmp_path / "other checkout")
    assert system.autostart_state()[0] == "other"
    run_key.values[system.RUN_VALUE] = "plain.exe --autostart"
    assert system.autostart_target("plain.exe --autostart") == ("plain.exe", "")
    monkeypatch.setattr(system, "launch_command", lambda: (_ for _ in ()).throw(ValueError("quotes")))
    assert system.autostart_state()[0] == "missing"
    monkeypatch.setattr(run_key, "get", lambda name: (_ for _ in ()).throw(OSError("denied")))
    assert system.autostart_state() == ("off", "")
    monkeypatch.setattr(system, "registry", None)
    assert system.autostart_state() == ("off", "")


def _autostart_controller(tmp_path, qtbot, monkeypatch):
    from tests import controller_rig
    patch_os(monkeypatch)
    monkeypatch.setattr(controller_rig, "TAutostart", ctl_mod._RealAutostart)     # the registry (a fake key here)
    return controller_rig.make_rig(tmp_path, qtbot, monkeypatch, fake=False, start=False)


def test_r19_an_entry_for_a_moved_install_is_rewritten_for_this_copy(run_key, tmp_path, qtbot, monkeypatch):
    from voxflow import system
    run_key.values[system.RUN_VALUE] = f'"{tmp_path / "old place" / "VoxFlow.exe"}" --autostart'
    rig = _autostart_controller(tmp_path, qtbot, monkeypatch)
    try:
        assert run_key.values[system.RUN_VALUE] == system.launch_command() and rig.c.get("launch_at_login")
    finally:
        rig.close()


def test_r19_an_entry_for_another_copy_offers_use_this_copy(run_key, tmp_path, qtbot, monkeypatch):
    from voxflow import system
    other = tmp_path / "Scoop" / "VoxFlow.exe"
    other.parent.mkdir()
    other.write_text("")
    run_key.values[system.RUN_VALUE] = f'"{other}" --autostart'
    rig = _autostart_controller(tmp_path, qtbot, monkeypatch)
    actions = []
    rig.c.notice_action.connect(lambda m, a, label: actions.append((m, a, label)))
    try:
        rig.start()
        assert actions == [(f"Launch at login starts another copy of VoxFlow ({other}).", "autostart_here",
                            "Use this copy")]
        assert run_key.values[system.RUN_VALUE] == f'"{other}" --autostart', "not changed without asking"
        rig.c.config.launch_at_login = False
        rig.c.autostart_here()
        assert run_key.values[system.RUN_VALUE] == system.launch_command() and rig.c.get("launch_at_login")
        assert ("success", "Launch at login now starts this copy of VoxFlow.") in rig.spy.notices()
        rig.c.autostart_here()                                      # already on: still fine
    finally:
        rig.close()


def test_r19_when_windows_refuses_the_user_is_told(run_key, tmp_path, qtbot, monkeypatch, caplog):
    from voxflow import system
    run_key.values[system.RUN_VALUE] = f'"{tmp_path / "gone.exe"}" --autostart'
    monkeypatch.setattr(run_key, "set", lambda name, value: (_ for _ in ()).throw(PermissionError("denied")))
    with caplog.at_level(logging.WARNING):
        rig = _autostart_controller(tmp_path, qtbot, monkeypatch)
    try:
        assert "launch at login could not be repaired" in caplog.text
        rig.c.autostart_here()
        assert ("error", "Windows did not let VoxFlow change the launch-at-login setting.") in rig.spy.notices()
    finally:
        rig.close()


def test_r12_quit_asks_over_the_channel_and_waits_for_the_lock(qapp, data_dir):
    import threading
    from PySide6.QtCore import QLockFile
    from PySide6.QtNetwork import QLocalServer
    lock = QLockFile(str(data_dir / "voxflow.lock"))
    assert lock.tryLock(100)
    server = QLocalServer()
    QLocalServer.removeServer(app._instance_name())
    assert server.listen(app._instance_name())
    threading.Timer(0.3, lock.unlock).start()                      # the running copy finishes quitting
    try:
        assert app.request_quit(5) == 0
    finally:
        server.close()


def test_r12_a_quit_request_already_read_is_not_waited_for(qapp, clean_process_state, data_dir, monkeypatch):
    from PySide6.QtCore import QByteArray, QTimer
    from PySide6.QtNetwork import QLocalSocket
    kept = []

    def knock():
        # the command is already in the connection's buffer when it is accepted (as with a slower UI thread)
        monkeypatch.setattr(QLocalSocket, "bytesAvailable", lambda self: 5)
        monkeypatch.setattr(QLocalSocket, "readAll", lambda self: QByteArray(b"quit\r\n"))
        monkeypatch.setattr(QLocalSocket, "waitForReadyRead", lambda self, ms=0: pytest.fail("no wait"))
        sock = QLocalSocket()
        kept.append(sock)
        sock.connectToServer(app._instance_name())
        assert sock.waitForConnected(1000)

    QTimer.singleShot(300, knock)
    quit_after(10000)                                               # safety net only
    assert app.run_app(fake=True, no_hook=True) == 0
    assert "asked to quit by another process" in (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")


def test_r19_an_entry_for_this_copy_is_left_alone(run_key, tmp_path, qtbot, monkeypatch):
    from voxflow import system
    system.set_autostart(True)
    before = dict(run_key.values)
    rig = _autostart_controller(tmp_path, qtbot, monkeypatch)
    try:
        rig.start()
        assert run_key.values == before and rig.c.get("launch_at_login")
        assert not any(a for a in rig.spy.of("notice_action"))
    finally:
        rig.close()
