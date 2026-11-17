"""Regression tests for the robustness QA findings (R1-R19, see CHANGELOG): a damaged install, an unwritable
data folder, log redaction, the crash handler, a PC without a microphone, a read-only settings file, damaged
model files, a full disk, the CUDA cache location, crash reports and launch at login."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from tests.controller_rig import TAutostart, TCleaner, TInjector, TMixer, TRecorder
from tests.test_cov_controller import patch_os
from voxflow import app, controller as ctl_mod
from voxflow.config import Config
from voxflow.context import AppContext
from voxflow.controller import Controller


def _real_controller(tmp_path, **kw):
    kw.setdefault("injector", TInjector())
    return Controller(fake=False, data_dir=tmp_path, install_hook=False, sound_backend=lambda wav: None,
                      mute_backend=TMixer(), context_fn=lambda: AppContext(), autostart=TAutostart(),
                      speed_check=False, updates=False, **kw)


# =========================================================================== R1: no demo stand-ins in the real app
def test_r1_a_broken_speech_module_fails_to_load_instead_of_pasting_the_demo_sentence(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch)
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
                                                     "open_data_folder": lambda self: opened.append("data")})()})()
    shell.Shell.run_action(fake, "open_log_folder")
    shell.Shell.run_action(fake, "open_data_folder")
    fake.ctl = object()
    shell.Shell.run_action(fake, "open_log_folder")               # a controller without it: ignored
    assert opened == ["log", "data"]


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
