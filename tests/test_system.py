"""voxflow.system: launch at login (fake registry), rotating log, crash handler, folder opening.

The real ``HKCU\\...\\Run`` key is never written: tests replace ``voxflow.system.registry``; the harness
additionally blocks real writes to that key.
"""
from __future__ import annotations

import logging
import sys
import threading

import pytest

from voxflow import system as sysmod


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


class FakeRegistry:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.fail: Exception | None = None

    def get(self, name):
        if self.fail:
            raise self.fail
        return self.values.get(name)

    def set(self, name, value):
        if self.fail:
            raise self.fail
        self.values[name] = value

    def delete(self, name):
        if self.fail:
            raise self.fail
        self.values.pop(name, None)


@pytest.fixture
def reg(monkeypatch):
    fake = FakeRegistry()
    monkeypatch.setattr(sysmod, "registry", fake)
    return fake


@pytest.fixture
def clean_logging():
    """Leave the process-wide logging / excepthook state as we found it."""
    root = logging.getLogger("voxflow")
    handlers, level = list(root.handlers), root.level
    hooks = (sys.excepthook, threading.excepthook)
    yield
    sysmod.shutdown_logging()
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    sys.excepthook, threading.excepthook = hooks


# ---------------------------------------------------------------- launch at login
def test_autostart_roundtrip(reg):
    assert sysmod.autostart_enabled() is False
    sysmod.set_autostart(True)
    assert list(reg.values) == [sysmod.RUN_VALUE] == ["VoxFlow"]
    assert sysmod.autostart_enabled() is True
    sysmod.set_autostart(True)                                  # idempotent
    assert len(reg.values) == 1
    sysmod.set_autostart(False)
    assert reg.values == {} and sysmod.autostart_enabled() is False
    sysmod.set_autostart(False)                                 # removing a missing entry is fine


def test_autostart_command_starts_this_install_without_a_console(reg, tmp_path):
    exe_dir = tmp_path / "venv" / "Scripts"
    exe_dir.mkdir(parents=True)
    (exe_dir / "python.exe").write_text("")
    (exe_dir / "pythonw.exe").write_text("")
    cmd = sysmod.launch_command(str(exe_dir / "python.exe"), tmp_path / "app")
    assert cmd.startswith(f'"{exe_dir / "pythonw.exe"}" -c "'), "the window-less interpreter must be used at login"
    assert str(tmp_path / "app") in cmd and "run_module('voxflow'" in cmd and sysmod.AUTOSTART_FLAG in cmd
    assert cmd.count('"') == 4, "exactly two quoted arguments, or Windows will split the command wrongly"
    (exe_dir / "pythonw.exe").unlink()
    assert sysmod.launch_command(str(exe_dir / "python.exe"), tmp_path / "app").startswith(
        f'"{exe_dir / "python.exe"}"')


def test_autostart_command_is_valid_python_that_runs_the_package(tmp_path, monkeypatch):
    """Execute the generated ``-c`` program against a stub package: it must run ``voxflow`` as __main__ with the flag."""
    pkg = tmp_path / "it's root" if False else tmp_path / "root dir"
    (pkg / "voxflow").mkdir(parents=True)
    (pkg / "voxflow" / "__init__.py").write_text("")
    (pkg / "voxflow" / "__main__.py").write_text(
        "import sys, pathlib\npathlib.Path(sys.argv[0]).parent  # argv[0] is replaced by runpy\n"
        f"pathlib.Path(r'{tmp_path / 'ran.txt'}').write_text(' '.join(sys.argv[1:]))\n")
    cmd = sysmod.launch_command(sys.executable, pkg)
    code = cmd.split(' -c "', 1)[1][:-1]
    import subprocess
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "ran.txt").read_text() == sysmod.AUTOSTART_FLAG


def test_autostart_command_for_a_frozen_build(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\VoxFlow\VoxFlow.exe")
    assert sysmod.launch_command() == r'"C:\Program Files\VoxFlow\VoxFlow.exe" --autostart'


@pytest.mark.parametrize("root", ['C:\\my "quoted" dir', "C:\\dan's files"])
def test_autostart_refuses_install_paths_that_cannot_be_quoted(root):
    with pytest.raises(ValueError, match="quotes"):
        sysmod.launch_command(sys.executable, root)


def test_c50_autostart_command_works_from_a_drive_root(tmp_path):
    cmd = sysmod.launch_command(sys.executable, "C:\\")
    code = cmd.split(' -c "', 1)[1][:-1]
    compile(code, "<autostart>", "exec")            # r'C:\' is not a valid string literal


def test_c50_autostart_command_fits_the_run_key_limit_for_a_deep_install_path():
    exe = r"C:\Users\some.long.username\AppData\Local\Programs\VoxFlow Dictation\.venv\Scripts\pythonw.exe"
    root = r"C:\Users\some.long.username\AppData\Local\Programs\VoxFlow Dictation"
    assert len(sysmod.launch_command(exe, root)) <= 260, "Windows ignores Run entries longer than 260 characters"


def test_c51_the_package_entry_point_accepts_the_autostart_flag(monkeypatch):
    """The command registered at login must be parseable by ``python -m voxflow``."""
    import voxflow.__main__ as entry
    import voxflow.app as app
    called = {}
    monkeypatch.setattr(app, "run_app", lambda *a, **k: called.update(k) or 0)
    assert entry.main([sysmod.AUTOSTART_FLAG]) == 0
    assert called, "run_app was not reached"


def test_autostart_errors_are_human_and_reading_never_raises(reg, caplog):
    reg.fail = PermissionError("Access is denied")
    with caplog.at_level("WARNING"):
        assert sysmod.autostart_enabled() is False
    for enabled in (True, False):
        with pytest.raises(ValueError, match="Windows did not let VoxFlow change"):
            sysmod.set_autostart(enabled)
    assert "Access is denied" in caplog.text


def test_autostart_off_windows(monkeypatch):
    monkeypatch.setattr(sysmod, "registry", None)
    assert sysmod.autostart_enabled() is False
    with pytest.raises(ValueError, match="only available on Windows"):
        sysmod.set_autostart(True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_real_run_key_wrapper_reads_but_cannot_write_in_tests():
    """``RunKey`` itself: reading a missing value is None; writes are stopped by the harness, not by luck."""
    key = sysmod.RunKey()
    assert key.get("VoxFlow-QA-value-that-does-not-exist") is None
    with pytest.raises(AssertionError, match="blocked in tests"):
        key.set("VoxFlow-QA-test", "x")
    with pytest.raises(AssertionError, match="blocked in tests"):
        key.delete("VoxFlow-QA-test")
    assert key.get("VoxFlow-QA-test") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_run_key_wrapper_against_a_fake_winreg(monkeypatch):
    """Every winreg call RunKey makes, checked against an in-memory registry (key path, access rights, type)."""
    import winreg
    store: dict[str, tuple[int, str]] = {}
    opened = []

    class Key:
        def __init__(self, access):
            self.access = access

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_key(root, sub_key, reserved=0, access=winreg.KEY_READ):
        opened.append((root, sub_key, access))
        return Key(access)

    def query(key, name):
        if name not in store:
            raise FileNotFoundError(name)
        kind, value = store[name]
        return value, kind

    def set_value(key, name, reserved, kind, value):
        assert key.access & winreg.KEY_SET_VALUE
        store[name] = (kind, value)

    def delete_value(key, name):
        assert key.access & winreg.KEY_SET_VALUE
        if name not in store:
            raise FileNotFoundError(name)
        del store[name]

    monkeypatch.setattr(winreg, "OpenKey", open_key)
    monkeypatch.setattr(winreg, "CreateKeyEx", open_key)
    monkeypatch.setattr(winreg, "QueryValueEx", query)
    monkeypatch.setattr(winreg, "SetValueEx", set_value)
    monkeypatch.setattr(winreg, "DeleteValue", delete_value)

    key = sysmod.RunKey()
    assert key.get("VoxFlow") is None
    command = r'"C:\apps\pythonw.exe" -c "..."'
    key.set("VoxFlow", command)
    assert store["VoxFlow"] == (winreg.REG_SZ, command)
    assert key.get("VoxFlow") == command
    key.delete("VoxFlow")
    key.delete("VoxFlow")                                       # already gone: not an error
    assert store == {}
    assert {(root, sub) for root, sub, _a in opened} == {(winreg.HKEY_CURRENT_USER, sysmod.RUN_KEY)}
    assert sysmod.RUN_KEY == r"Software\Microsoft\Windows\CurrentVersion\Run"


def test_shutdown_logging_tolerates_an_already_closed_fault_file(tmp_path, clean_logging, monkeypatch):
    sysmod.install_crash_handler(tmp_path)
    monkeypatch.setattr(sysmod.faulthandler, "disable", lambda: (_ for _ in ()).throw(ValueError("closed")))
    sysmod.shutdown_logging()
    assert sysmod._fault_file is None


# ---------------------------------------------------------------- logging
def test_setup_logging_writes_a_rotating_file(tmp_path, clean_logging):
    path = sysmod.setup_logging(tmp_path)
    assert path == tmp_path / "logs" / "voxflow.log" == sysmod.log_dir(tmp_path) / sysmod.LOG_NAME
    logging.getLogger("voxflow.test").info("hello %s", "log")
    logging.getLogger("voxflow.test").debug("too detailed")
    logging.getLogger("other.library").warning("not ours")
    sysmod.shutdown_logging()
    text = path.read_text(encoding="utf-8")
    assert "INFO" in text and "voxflow.test" in text and "hello log" in text and "MainThread" in text
    assert "too detailed" not in text and "not ours" not in text
    handler_types = [type(h).__name__ for h in logging.getLogger("voxflow").handlers]
    assert "RotatingFileHandler" not in handler_types


def test_setup_logging_is_idempotent_and_can_move(tmp_path, clean_logging):
    sysmod.setup_logging(tmp_path / "a")
    sysmod.setup_logging(tmp_path / "a")
    second = sysmod.setup_logging(tmp_path / "b")
    root = logging.getLogger("voxflow")
    files = [h for h in root.handlers if type(h).__name__ == "RotatingFileHandler"]
    consoles = [h for h in root.handlers if getattr(h, "_voxflow_console", False)]
    assert len(files) == 1 and len(consoles) <= 1
    assert files[0].maxBytes == 1_000_000 and files[0].backupCount == 3
    logging.getLogger("voxflow.test").info("once")
    sysmod.shutdown_logging()
    assert second.read_text(encoding="utf-8").count("once") == 1
    assert not (tmp_path / "a" / "logs" / "voxflow.log").exists(), "nothing was logged to the first location"
    sysmod.shutdown_logging()                                   # idempotent


def test_setup_logging_survives_an_unwritable_folder(tmp_path, clean_logging, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    path = sysmod.setup_logging(blocker)                        # logs/ cannot be created below a file
    logging.getLogger("voxflow.test").warning("still alive")
    assert path == blocker / "logs" / "voxflow.log"
    assert "log file unavailable" in capsys.readouterr().err


def test_log_rotation_keeps_three_backups(tmp_path, clean_logging):
    path = sysmod.setup_logging(tmp_path)
    handler = next(h for h in logging.getLogger("voxflow").handlers if type(h).__name__ == "RotatingFileHandler")
    handler.maxBytes = 2000
    for i in range(400):
        logging.getLogger("voxflow.test").info("line %04d %s", i, "x" * 60)
    sysmod.shutdown_logging()
    names = sorted(p.name for p in path.parent.iterdir())
    assert names == ["voxflow.log", "voxflow.log.1", "voxflow.log.2", "voxflow.log.3"]
    assert "line 0399" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- crash handling
def _raise(msg="boom"):
    try:
        raise RuntimeError(msg)
    except RuntimeError:
        return sys.exc_info()


def test_write_crash_report(tmp_path):
    path = sysmod.write_crash_report(tmp_path, *_raise("kaboom"), where="thread pipeline")
    assert path.parent == tmp_path / "logs" and path.name.startswith("crash-") and path.suffix == ".log"
    text = path.read_text(encoding="utf-8")
    assert "VoxFlow crash report" in text and "where: thread pipeline" in text
    assert "RuntimeError: kaboom" in text and "Traceback (most recent call last)" in text and "python: " in text
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert sysmod.write_crash_report(blocker, *_raise()) is None, "an unwritable folder must not raise from a crash"


def test_crash_handler_reports_main_thread_exceptions_and_runs_cleanup(tmp_path, clean_logging, monkeypatch, caplog):
    cleaned, chained = [], []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: chained.append(a[0]))
    sysmod.install_crash_handler(tmp_path, on_crash=lambda: cleaned.append("unmuted"))
    with caplog.at_level("CRITICAL"):
        sys.excepthook(*_raise("main thread died"))
    reports = list((tmp_path / "logs").glob("crash-*.log"))
    assert len(reports) == 1 and "main thread died" in reports[0].read_text(encoding="utf-8")
    assert cleaned == ["unmuted"] and chained == [RuntimeError]
    assert "uncaught exception in main thread" in caplog.text and "crash report written to" in caplog.text
    assert (tmp_path / "logs" / "fault.log").exists(), "faulthandler must be armed for native crashes"


def test_crash_handler_reports_worker_thread_exceptions(tmp_path, clean_logging):
    cleaned = []
    sysmod.install_crash_handler(tmp_path, on_crash=lambda: cleaned.append(1))
    th = threading.Thread(target=lambda: 1 / 0, name="pipeline-worker")
    th.start()
    th.join()
    reports = list((tmp_path / "logs").glob("crash-*.log"))
    assert len(reports) == 1 and cleaned == [1]
    text = reports[0].read_text(encoding="utf-8")
    assert "where: thread pipeline-worker" in text and "ZeroDivisionError" in text


def test_crash_handler_ignores_ctrl_c_and_thread_exit_and_survives_a_failing_cleanup(tmp_path, clean_logging,
                                                                                    monkeypatch, caplog):
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)
    sysmod.install_crash_handler(tmp_path, on_crash=lambda: 1 / 0)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())
    th = threading.Thread(target=lambda: sys.exit(0))
    th.start()
    th.join()
    assert list((tmp_path / "logs").glob("crash-*.log")) == [], "Ctrl+C and SystemExit are not crashes"
    with caplog.at_level("ERROR"):
        sys.excepthook(*_raise())                               # cleanup raises ZeroDivisionError: must be contained
    assert "crash clean-up failed" in caplog.text
    assert len(list((tmp_path / "logs").glob("crash-*.log"))) == 1


def test_crash_handler_without_cleanup_and_in_an_unwritable_folder(tmp_path, clean_logging, monkeypatch):
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    sysmod.install_crash_handler(blocker)                       # no fault.log, no crash file: still no exception
    sys.excepthook(*_raise())
    args = type("A", (), {"exc_type": RuntimeError, "exc_value": RuntimeError("x"), "exc_traceback": None,
                          "thread": None})()
    threading.excepthook(args)                                  # thread object already gone


# ---------------------------------------------------------------- folders
def test_open_folder_creates_and_opens(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(sysmod.os, "startfile", opened.append, raising=False)
    monkeypatch.setattr(sysmod.sys, "platform", "win32")
    target = tmp_path / "new" / "logs"
    assert sysmod.open_folder(target) is True
    assert target.is_dir() and opened == [str(target)]


def test_open_folder_failures_return_false(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sysmod.sys, "platform", "win32")
    monkeypatch.setattr(sysmod.os, "startfile", lambda p: (_ for _ in ()).throw(OSError("no shell")), raising=False)
    with caplog.at_level("WARNING"):
        assert sysmod.open_folder(tmp_path) is False
    assert "could not open" in caplog.text
    monkeypatch.setattr(sysmod.sys, "platform", "linux")
    assert sysmod.open_folder(tmp_path) is False
