"""Regression tests for the independent review of the update feature (scratchpad update-review.md, items 1-12).
Each test is named after its item. Hermetic like tests/test_updater.py, whose fakes it reuses."""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_updater import (API, INSTALLER, REPO, FakeHttp, Gate, Resp, _raiser, github, make,  # noqa: F401
                                no_sockets, settle)
from tests.cov_ui_kit import make_demo  # noqa: F401 - fixture
from voxflow import system as sysmod
from voxflow import updater as U

ROOT = Path(__file__).resolve().parent.parent
SETUP_URL = f"{REPO}/releases/download/v0.3.0/VoxFlow-Setup-0.3.0.exe"
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing / mutex semantics")


def downloaded(tmp_path, qtbot, **kw):
    """An Updater whose 0.3.0 installer is downloaded and verified (through "Install when I quit" unless auto)."""
    up = make(tmp_path, qtbot, **kw)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.phase != "checking", timeout=3000)
    if kw.get("mode") != "auto":
        up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    return up


# =========================================================================== 1: no install on quit without consent
def test_r1_leaving_automatic_mode_drops_its_install_on_quit(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot, mode="auto")
    assert up.info()["install_on_quit"] and up.state.quit_by == "auto"
    up.settings["update_mode"] = "notify"
    up.settings_changed("update_mode")
    assert not up.info()["install_on_quit"], "the user never chose it"
    assert up.on_exit(user_quit=True) is False and up.launched == []
    up.close()


def test_r1_a_download_that_ends_after_leaving_automatic_mode_arms_nothing(tmp_path, qtbot):
    http = github()
    gate = Gate()
    http.routes[SETUP_URL] = gate
    up = make(tmp_path, qtbot, http=http, mode="auto")
    up.check(user=False)
    qtbot.waitUntil(lambda: up.phase == "downloading", timeout=3000)
    up.settings["update_mode"] = "notify"
    up.settings_changed("update_mode")
    gate.release()
    qtbot.waitUntil(lambda: up.phase == "idle", timeout=3000)
    assert up.info()["downloaded"] and not up.info()["install_on_quit"]
    assert not any("will be installed" in m for _l, m in up.events["notice"])
    assert up.on_exit(user_quit=True) is False
    up.close()


def test_r1_the_users_own_install_on_quit_survives_a_mode_change(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)                       # chosen in the dialog
    assert up.state.quit_by == "user"
    up.settings["update_mode"] = "auto"
    up.settings_changed("update_mode")
    up.settings["update_mode"] = "notify"
    up.settings_changed("update_mode")
    assert up.info()["install_on_quit"] and up.on_exit(user_quit=True) is True
    up.close()


def test_r1_switching_automatic_mode_to_a_set_time_drops_the_quit_install(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot, mode="auto")
    up.settings["update_install_when"] = "time"
    up.settings_changed("update_install_when")
    assert not up.info()["install_on_quit"] and up._slot_timer.isActive()
    up.close()


# =========================================================================== 2: the scheduled install re-validates
def test_r2_the_set_time_never_installs_while_a_newer_download_runs(tmp_path, qtbot):
    moment = _dt.datetime(2026, 11, 15, 3, 10).astimezone()
    up = downloaded(tmp_path, qtbot, mode="auto", when="time", idle=None, now=lambda: moment)
    up.state.slot_done = ""
    http = github("0.4.0")
    gate = Gate()
    http.routes[f"{REPO}/releases/download/v0.4.0/VoxFlow-Setup-0.4.0.exe"] = gate
    up._http = http
    up.check(user=False)                                   # 0.4.0 is out: its download starts
    qtbot.waitUntil(lambda: up.phase == "downloading", timeout=3000)
    up._slot_tick()                                        # 03:10: the slot is open
    assert up.events["restart"] == 0 and up.phase == "downloading", "the download is left alone"
    gate.release()
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    assert sorted(p.name for p in (tmp_path / "dl").iterdir()) == ["VoxFlow-Setup-0.4.0.exe"]
    up._slot_tick()
    assert up.events["restart"] == 1, "then 0.4.0 installs in the same slot"
    up.close()


def test_r2_a_skipped_version_is_never_installed_at_the_set_time(tmp_path, qtbot, monkeypatch):
    moment = _dt.datetime(2026, 11, 15, 3, 10).astimezone()
    up = downloaded(tmp_path, qtbot, mode="auto", when="time", idle=None, now=lambda: moment)
    assert up._slot_timer.isActive()
    monkeypatch.setattr(U, "_remove", lambda path: False)  # e.g. a virus scanner holds the file
    up.skip()
    assert not up._slot_timer.isActive(), "skip disarms the slot"
    up._slot_timer.start()                                 # even a tick that comes anyway
    up._slot_tick()
    assert up.events["restart"] == 0 and not up._slot_timer.isActive()
    up.close()


def test_r2_a_changed_installer_is_not_installed_at_the_set_time(tmp_path, qtbot):
    moment = _dt.datetime(2026, 11, 15, 3, 10).astimezone()
    up = downloaded(tmp_path, qtbot, mode="auto", when="time", idle=None, now=lambda: moment)
    (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").write_bytes(b"swapped")
    up._slot_tick()
    assert up.events["restart"] == 0, "VoxFlow is not quit for a file that would be refused"
    assert "gone or changed" in up.error and not up.info()["downloaded"] and not up._slot_timer.isActive()
    up.close()


def test_r2_install_now_with_a_changed_file_does_not_quit(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").write_bytes(b"swapped")
    up.install("now")                                      # the file "is there": no download, straight to restart
    assert up.events["restart"] == 0 and up.phase == "idle" and "gone or changed" in up.error
    up.close()


# =========================================================================== 3: normalised paths only
@pytest.mark.parametrize("url", [
    f"{REPO}/releases/download/../../../evil/x/releases/download/v1/f",
    "https://github.com/blackknight75/voxflow/releases/download/%2e%2e/%2E%2E/evil/f",
    "https://github.com/blackknight75/voxflow/releases/download/v1%2f..%2fx",
    "https://github.com/blackknight75/voxflow/releases/download/v1%5cx",
    "https://github.com/blackknight75/voxflow/releases/download/v1\\..\\x",
    "https://github.com/blackknight75/voxflow/releases/download//v1/x",
    "https://github.com/blackknight75/voxflow/releases/download/./v1/x",
    "https://objects.githubusercontent.com/a/../b",
    "https://api.github.com/repos/blackknight75/voxflow/../../other/repo/releases/latest",
])
def test_r3_paths_with_dot_segments_or_encoded_separators_are_refused(url):
    assert U.allowed_url(url, "asset") is False and U.allowed_url(url, "api") is False


def test_r3_ordinary_paths_still_pass():
    assert U.allowed_url(f"{REPO}/releases/download/v1.2.3/VoxFlow-Setup-1.2.3.exe")
    assert U.allowed_url("https://objects.githubusercontent.com/github-production-release-asset-2e65be/1/2?x=%2F")
    assert U.allowed_url(API, "api")


def test_r3_a_stored_release_is_held_to_the_same_rules_as_the_api_answer():
    rel = U.parse_release({**__import__("tests.test_updater", fromlist=["release_json"]).release_json()}).to_dict()
    cdn = dict(rel, installer_url="https://objects.githubusercontent.com/x/VoxFlow-Setup-0.3.0.exe")
    assert U.Release.from_dict(cdn).installer_url == "", "only github.com links, as parse_release requires"
    renamed = dict(rel, installer="evil.exe")
    assert U.Release.from_dict(renamed).installer == "" and not U.Release.from_dict(renamed).installable
    dots = dict(rel, sums_url=f"{REPO}/releases/download/../../x/SHA256SUMS.txt")
    assert U.Release.from_dict(dots).sums_url == ""


# =========================================================================== 4: no start on half-replaced files
@windows_only
def test_r4_the_setup_mutex_is_seen_while_setup_runs():
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    assert sysmod.update_setup_running() is False
    handle = k32.CreateMutexW(None, False, sysmod.SETUP_MUTEX)      # what Setup does (SetupMutex)
    try:
        assert sysmod.update_setup_running() is True
    finally:
        k32.CloseHandle(handle)
    assert sysmod.update_setup_running() is False


def test_r4_voxflow_exits_with_a_message_while_its_setup_runs(monkeypatch):
    from voxflow import __main__ as entry
    from voxflow import app
    started, told = [], []
    monkeypatch.setattr(app, "run_app", lambda **kw: started.append(kw) or 0)
    monkeypatch.setattr(sysmod, "update_setup_running", lambda: True)
    monkeypatch.setattr(sysmod, "tell_update_running", lambda: told.append(1))
    assert entry.main([]) == 0 and told == [1] and started == []
    assert entry.main(["--after-update"]) == 0 and len(started) == 1, "Setup's own relaunch starts"
    assert entry.main(["--demo"]) == 0 and len(started) == 2
    monkeypatch.setattr(sysmod, "update_setup_running", lambda: False)
    assert entry.main([]) == 0 and len(started) == 3


def test_r4_the_message_never_raises(monkeypatch, capsys):
    monkeypatch.setattr(sysmod.sys, "platform", "linux")
    sysmod.tell_update_running()
    assert "Finishing the update" in capsys.readouterr().out
    assert sysmod.update_setup_running() is False


def test_r4_the_message_box_comes_from_a_helper_so_voxflow_exits_at_once(monkeypatch):
    """A VoxFlow.exe left open in its own message box keeps files in the install folder locked, and a silent Setup
    then aborts and rolls the update back (tests/e2e_update; Inno Setup 6.7.3, exit code 5). The box is shown by
    PowerShell from System32 and VoxFlow returns without waiting."""
    import subprocess
    started = []
    monkeypatch.setattr(sysmod.sys, "platform", "win32")
    monkeypatch.setenv("_MEIPASS2", "x")
    sysmod.tell_update_running(popen=lambda cmd, **kw: started.append((cmd, kw)))
    (cmd, kw), = started
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    assert Path(cmd[0]) == Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    assert kw["cwd"] == root, "nothing of the install folder is held, not even as the working folder"
    assert kw["env"][sysmod.UPDATE_MESSAGE_VAR].startswith("Finishing the update")
    assert "_MEIPASS2" not in kw["env"] and "MessageBox" in cmd[-1] and sysmod.UPDATE_MESSAGE_VAR in cmd[-1]
    assert kw["creationflags"] & subprocess.CREATE_NO_WINDOW


def test_r4_no_powershell_no_message_and_no_error(monkeypatch):
    monkeypatch.setattr(sysmod.sys, "platform", "win32")

    def missing(*_a, **_k):
        raise FileNotFoundError(2, "not found")

    sysmod.tell_update_running(popen=missing)               # logged, not raised


def test_r4_setup_holds_the_mutex_and_relaunches_with_after_update_but_not_on_quit():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    assert f"SetupMutex={sysmod.SETUP_MUTEX},Global\\{sysmod.SETUP_MUTEX}" in iss
    assert 'Parameters: "--after-update"; Flags: nowait; Check: RelaunchAfterUpdate' in iss
    assert U.installer_command(Path("s.exe"), relaunch=False)[-1] == "/RELAUNCH=0"


def test_r4_install_on_quit_runs_setup_without_relaunch(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    assert up.on_exit(user_quit=True) and up.launched[0][1] is False
    up.close()


# =========================================================================== 5: 416 restarts once, not forever
def test_r5_a_416_restarts_the_download_once_only(tmp_path):
    http = github()
    calls = []
    http.routes[SETUP_URL] = lambda h: calls.append(dict(h)) or Resp(416, b"")
    rel = U.fetch_latest(http)
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / (rel.installer + ".part")).write_bytes(INSTALLER[:10])
    with pytest.raises(U.UpdateError, match="HTTP 416"):
        U.download_installer(http, rel, folder)
    assert calls == [{"Range": "bytes=10-"}, {}], "one restart from 0, then an error"
    calls.clear()
    with pytest.raises(U.UpdateError, match="HTTP 416"):
        U.download_installer(http, rel, folder)                 # no part at all: no restart
    assert calls == [{}]


# =========================================================================== 6: metered in automatic mode is told
def test_r6_automatic_mode_on_a_metered_connection_tells_the_user(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto", metered=True)
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 1 and "metered" in up.events["available"][0]["error"]
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 1, "once per version, like any notification"
    up.close()


# =========================================================================== 7: Off while a check runs
def test_r7_a_check_that_ends_after_switching_to_off_does_not_notify(tmp_path, qtbot):
    http = github()
    gate = Gate()
    answer = http.routes[API]
    http.routes[API] = lambda h: (gate.event.wait(5), answer(h))[1]
    up = make(tmp_path, qtbot, http=http)
    up.check(user=False)
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    gate.release()
    qtbot.waitUntil(lambda: up.phase == "idle", timeout=3000)
    assert up.events["available"] == [] and up.info()["available"]["version"] == "0.3.0", "recorded, not shown"
    assert [u for u in http.urls() if "Setup" in u] == []
    up.close()


def test_r7_check_now_while_off_still_shows_the_result(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="off")
    up.check(user=True)
    settle(up)
    assert len(up.events["available"]) == 1
    up.close()


# =========================================================================== 8: no change between hash and launch
@windows_only
def test_r8_the_held_installer_can_be_run_but_not_changed(tmp_path):
    exe = tmp_path / "VoxFlow-Setup-9.9.9.exe"
    shutil.copy(Path(os.environ["SystemRoot"]) / "System32" / "whoami.exe", exe)
    with U.open_exclusive(exe) as held:
        digest = U.sha256_stream(held)
        with pytest.raises(PermissionError):
            exe.write_bytes(b"evil")
        with pytest.raises(PermissionError):
            exe.unlink()
        with pytest.raises(PermissionError):
            os.replace(exe, tmp_path / "moved.exe")
        out = subprocess.run([str(exe)], capture_output=True, timeout=30)   # Setup can still start from it
        assert out.returncode == 0
    assert U.sha256_file(exe) == digest
    exe.write_bytes(b"free again")                         # released afterwards


@windows_only
def test_r8_on_exit_keeps_the_file_locked_until_setup_has_started(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    attempts = []

    def launcher(path, relaunch, log):
        try:
            path.write_bytes(b"swapped at the last moment")
        except PermissionError:
            attempts.append("refused")
        return True

    up._launcher = launcher
    assert up.on_exit(user_quit=True) is True and attempts == ["refused"]
    assert (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").read_bytes() == INSTALLER
    up.close()


def test_r8_open_exclusive_elsewhere_is_a_plain_read(tmp_path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    monkeypatch.setattr(U.sys, "platform", "linux")
    with U.open_exclusive(f) as fh:
        assert fh.read() == b"abc"


@windows_only
def test_r8_open_exclusive_reports_a_missing_file(tmp_path):
    with pytest.raises(OSError):
        U.open_exclusive(tmp_path / "missing.exe")


# =========================================================================== 9: notices once per version
def test_r9_automatic_install_notices_are_shown_once_per_version_and_plan(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot, mode="auto")
    for _ in range(3):                                     # daily checks, settings touched
        up.check(user=False)
        settle(up)
        up.settings_changed("update_time")
    notices = [m for _l, m in up.events["notice"]]
    assert notices == ["VoxFlow 0.3.0 will be installed when you quit VoxFlow."]
    up.settings["update_install_when"] = "time"
    up.settings_changed("update_install_when")
    up.settings_changed("update_install_when")
    assert [m for _l, m in up.events["notice"]][1:] == [
        "VoxFlow 0.3.0 will be installed at 03:00, when VoxFlow and the PC are idle."], "a new plan is said once"
    again = make(tmp_path, qtbot, mode="auto", when="time")   # and not again after a restart
    again._auto_next()
    assert again.events["notice"] == []
    up.close()
    again.close()


# =========================================================================== 10: all-users installs only notify
def test_r10_an_install_under_program_files_does_not_update_itself(tmp_path, monkeypatch):
    pf = tmp_path / "Program Files"
    app = pf / "VoxFlow"
    app.mkdir(parents=True)
    (app / "VoxFlow.exe").write_bytes(b"")
    (app / "unins000.exe").write_bytes(b"")
    monkeypatch.setenv("ProgramFiles", str(pf))
    assert U.install_kind(app / "VoxFlow.exe", frozen=True) == "allusers"
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("ProgramW6432", raising=False)
    assert U.install_kind(app / "VoxFlow.exe", frozen=True) == "installer"
    monkeypatch.setattr(U.os, "access", lambda path, mode: False)     # a folder this user cannot write
    assert U.install_kind(app / "VoxFlow.exe", frozen=True) == "allusers"
    assert "administrator" in U.managed_text("allusers")


def test_r10_an_all_users_copy_notifies_and_offers_the_download_page(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto", kind="allusers")
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 1 and not up.info()["installable"]
    assert [u for u in up._http.urls() if "Setup" in u or "SHA256" in u] == []
    with pytest.raises(ValueError, match="administrator"):
        up.install("now")
    up.close()


def test_r10_the_dialog_says_download_for_an_all_users_copy(make_demo):
    from tests.cov_ui_kit import flush
    d = make_demo()
    d.c.demo_set_update(kind="allusers")
    page = d.page("settings")
    page.open_update()
    flush()
    assert d.find("update_dialog_page").text() == "Download"
    assert "administrator" in d.find("update_hint").text()


# =========================================================================== 11: the winget marker goes
def test_r11_a_manual_install_removes_the_winget_marker():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    code = iss.split("\n[Code]\n", 1)[1]
    proc = code.split("procedure CurStepChanged", 1)[1].split("\nend;", 1)[0]
    assert "SaveStringToFile(ExpandConstant('{app}\\managed-by.txt'), 'winget', False)\n  else\n" in proc
    assert "DeleteFile(ExpandConstant('{app}\\managed-by.txt'))" in proc


# =========================================================================== 12: cleanup that meets Setup retries
@windows_only
def test_r12_a_file_still_held_by_setup_is_deleted_on_a_later_try(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    up.close()
    log = tmp_path / "dl" / "install.log"
    log.write_text("Setup is still writing this", encoding="utf-8")
    held = U.open_exclusive(log)                           # Setup has it open
    try:
        after = make(tmp_path, qtbot, version="0.3.0")     # the relaunched VoxFlow
        assert after._tidy_timer.isActive() and log.exists()
        assert not (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").exists(), "the rest is cleaned already"
    finally:
        held.close()
    after._tidy_timer.stop()                               # a few minutes later
    after._tidy_again()
    assert list((tmp_path / "dl").iterdir()) == [] and not after._tidy_timer.isActive()
    after.close()


def test_r12_cleanup_gives_up_quietly_after_a_few_tries(tmp_path, qtbot, monkeypatch):
    (tmp_path / "dl").mkdir()
    (tmp_path / "dl" / "stuck.log").write_text("x", encoding="utf-8")
    monkeypatch.setattr(U, "_remove", lambda path: False)
    up = make(tmp_path, qtbot)
    for _ in range(U.TIDY_RETRIES + 2):
        if up._tidy_timer.isActive():
            up._tidy_timer.stop()
            up._tidy_again()
    assert up._tidy_tries == U.TIDY_RETRIES and not up._tidy_timer.isActive()
    up.close()


def test_r12_remove_reports_instead_of_raising(tmp_path, monkeypatch):
    f = tmp_path / "busy.exe"
    f.write_bytes(b"x")
    monkeypatch.setattr(Path, "unlink", _raiser(PermissionError(32, "in use")))
    assert U._remove(f) is False and U.clean_downloads(tmp_path) is False
    monkeypatch.undo()
    assert U._remove(f) is True and U.clean_downloads(tmp_path) is True
