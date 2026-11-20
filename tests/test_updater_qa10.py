"""Regression tests for the updater / installer findings of the 1.0 QA pass (UP-02 ... UP-12, D-05, R-A). Each test
is named after its finding and fails without the fix. Hermetic like tests/test_updater.py, whose fakes it reuses:
no network, no Setup, no real profile."""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.test_updater import (API, INSTALLER, REPO, FakeHttp, Resp, _exe, _raiser, github,  # noqa: F401
                                installed_version_is_older, make, release_json, settle)
from voxflow import system as sysmod
from voxflow import updater as U
from voxflow.ui import updates as UP

ROOT = Path(__file__).resolve().parent.parent
SETUP = "VoxFlow-Setup-0.3.0.exe"
SETUP_URL = f"{REPO}/releases/download/v0.3.0/{SETUP}"
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing and permissions")


def downloaded(tmp_path, qtbot, **kw):
    """An Updater whose 0.3.0 installer is downloaded and verified ("Install when I quit" unless automatic)."""
    up = make(tmp_path, qtbot, **kw)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.phase != "checking", timeout=3000)
    if kw.get("mode") != "auto":
        up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    return up


def scheduled(tmp_path, qtbot, day=0, **kw):
    """Automatic mode with a set time, at 03:05 on a December day, nobody at the PC."""
    moment = _dt.datetime(2026, 12, 1 + day, 3, 5).astimezone()
    return make(tmp_path, qtbot, mode="auto", when="time", idle=3600, now=lambda: moment, **kw)


def warnings(up):
    return [m for level, m in up.events["notice"] if level == "warning"]


def install_now(up):
    up.install("now")
    up.qtbot.waitUntil(lambda: up.phase != "downloading", timeout=3000)


# =========================================================================== UP-02: a Setup run that changes nothing
def test_up02_a_setup_run_that_changes_nothing_is_said_once_its_log_kept_and_not_repeated_for_ever(tmp_path, qtbot):
    log = tmp_path / "dl" / U.SETUP_LOG
    up = scheduled(tmp_path, qtbot)
    up.start()
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    up._slot_tick()
    assert up.events["restart"] == 1 and up.on_exit(user_quit=True) is True
    st = up.state
    assert (st.attempted, st.attempts, st.attempt_tray, st.attempt_told) == ("0.3.0", 1, True, False)
    assert st.attempt_time == up.clock.t
    assert json.loads((tmp_path / "data" / U.STATE_FILE).read_text("utf-8"))["attempted"] == "0.3.0", "saved"
    up.close()
    log.write_text("Setup failed: exit code 5", encoding="utf-8")       # Setup's /LOG; the version did not change

    day1 = scheduled(tmp_path, qtbot, day=1)
    assert log.exists(), "the log of the run that did not take is kept"
    day1.start()
    said = warnings(day1)
    assert len(said) == 1 and "VoxFlow 0.3.0 was not installed" in said[0] and "still VoxFlow 0.2.0" in said[0]
    assert str(log) in said[0] and "will not try again" not in said[0] and day1.info()["error"] == said[0]
    assert not day1.info()["auto_stopped"]
    day1.close()
    again = scheduled(tmp_path, qtbot, day=1)
    again.start()
    assert warnings(again) == [], "said once per run of Setup"
    again._slot_tick()                                                   # one more automatic try
    assert again.events["restart"] == 1 and again.on_exit(user_quit=True) is True
    assert again.state.attempts == 2 and again.launched[0][0].name == SETUP
    again.close()

    day2 = scheduled(tmp_path, qtbot, day=2)
    day2.start()
    said = warnings(day2)
    assert len(said) == 1 and "will not try again by itself" in said[0] and "Check now" in said[0]
    assert not day2._slot_timer.isActive(), "the set time no longer quits VoxFlow for this version"
    day2._slot_tick()
    day2.check(user=False)                                               # the daily check: nothing automatic either
    settle(day2)
    assert day2.events["restart"] == 0 and day2.launched == [] and day2.phase == "idle"
    assert day2.info()["auto_stopped"] and log.exists()
    assert "will not try again by itself" in UP.status_text(day2.info())[0]
    assert "installed at 03:00" not in UP.status_text(day2.info())[0]
    install_now(day2)                                                    # the user still can
    assert day2.events["restart"] == 1 and day2.on_exit(user_quit=True) is True
    assert day2.state.attempts == 3 and day2.launched[0][1] is True
    day2.close()

    done = scheduled(tmp_path, qtbot, day=3, version="0.3.0")            # this time Setup did its work
    assert done.state.attempted == "" and done.state.attempts == 0 and not log.exists()
    done.start()
    assert warnings(done) == []
    done.close()


def test_up02_automatic_install_on_quit_also_stops_after_two_runs(tmp_path, qtbot):
    for run in (1, 2):
        up = downloaded(tmp_path, qtbot, mode="auto")
        assert up.info()["install_on_quit"], f"run {run}: automatic mode plans it again"
        assert up.on_exit(user_quit=True) is True and up.state.attempts == run
        up.close()
    up = make(tmp_path, qtbot, mode="auto")
    up.start()
    assert "will not try again by itself" in warnings(up)[0], "(no log file: none is named)"
    assert "log" not in warnings(up)[0]
    up.check(user=False)
    settle(up)
    assert not up.info()["install_on_quit"] and up.on_exit(user_quit=True) is False and up.launched == []
    up.install("quit")                                                   # the user's own choice still counts
    assert up.on_exit(user_quit=True) is True
    up.close()


def test_up02_a_newer_release_starts_counting_again_and_notify_mode_promises_nothing(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    up.state.attempted, up.state.attempts = "0.2.9", 2                   # an older release that never installed
    assert not up._gave_up(up.latest())
    assert up.on_exit(user_quit=True) is True
    assert (up.state.attempted, up.state.attempts) == ("0.3.0", 1)
    up.close()
    later = make(tmp_path, qtbot)                                        # notify mode: nothing automatic to stop
    later.state.attempts = 5
    later.start()
    assert "was not installed" in warnings(later)[0] and "by itself" not in warnings(later)[0]
    later.close()


def test_up02_the_attempt_record_survives_a_damaged_state_file(tmp_path):
    path = tmp_path / U.STATE_FILE
    for bad in ("three", -1, True, 2.5, None):
        path.write_text(json.dumps({"attempted": "0.3.0", "attempts": bad, "attempt_tray": "yes"}), encoding="utf-8")
        st = U.UpdateState.load(path)
        assert (st.attempted, st.attempts, st.attempt_tray) == ("0.3.0", 0, False)
    path.write_text(json.dumps({"attempts": 2, "attempt_tray": True, "exit_error": "x"}), encoding="utf-8")
    st = U.UpdateState.load(path)
    assert (st.attempts, st.attempt_tray, st.exit_error) == (2, True, "x")


# =========================================================================== UP-04: Install now never leaves it closed
def test_up04_an_installer_that_cannot_be_opened_does_not_quit_voxflow(tmp_path, qtbot, monkeypatch):
    up = downloaded(tmp_path, qtbot)
    monkeypatch.setattr(U, "open_exclusive", _raiser(PermissionError("held by another program")))
    install_now(up)
    assert up.events["restart"] == 0 and up.phase == "idle" and up._pending_launch is None
    assert warnings(up) == ["The downloaded installer is gone or changed. Try again."] == [up.error]
    assert up.on_exit(user_quit=True) is False and up.launched == []
    up.close()


@windows_only
def test_up04_a_file_another_program_writes_to_is_found_before_the_restart_not_after_it(tmp_path, qtbot):
    """QA case G with a real handle: the exclusive open used to fail in on_exit, after VoxFlow had quit."""
    up = downloaded(tmp_path, qtbot)
    with open(tmp_path / "dl" / SETUP, "r+b"):                           # e.g. a sync tool with write access
        install_now(up)
        assert up.events["restart"] == 0 and up.phase == "idle" and warnings(up)
    up.close()


@windows_only
def test_up04_the_verified_file_is_held_from_the_restart_request_until_setup_starts(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    path = tmp_path / "dl" / SETUP
    install_now(up)
    assert up.events["restart"] == 1 and up.phase == "restarting"
    with pytest.raises(PermissionError):
        open(path, "r+b").close()                                        # nobody can change it now
    with pytest.raises(PermissionError):
        path.unlink()
    up.close()                                                           # controller.shutdown: close, then on_exit
    seen = []
    up._launcher = lambda p, relaunch, log: seen.append((p.read_bytes() == INSTALLER, relaunch, log.name)) or True
    assert up.on_exit(user_quit=False) is True
    assert seen == [(True, True, U.SETUP_LOG)], "Setup can still be started (read) while the file is held"
    open(path, "r+b").close()                                            # released once Setup has started
    assert up.on_exit(user_quit=True) is False, "once"


def test_up04_a_setup_that_does_not_start_is_said_at_the_next_start_and_voxflow_comes_back(tmp_path, qtbot):
    """QA case H: CreateProcess refused. VoxFlow had quit for it and nothing started it again."""
    up = downloaded(tmp_path, qtbot)
    restarted = []
    up._launcher = lambda *a: False
    up._restarter = lambda tray: restarted.append(tray)
    install_now(up)
    assert up.events["restart"] == 1
    assert up.on_exit(user_quit=True) is False
    assert restarted == [False], "started again, with its window (the user clicked Install now)"
    assert up.state.attempted == "" and up.state.attempts == 0, "Setup never ran: not a failed install"
    up.close()
    nxt = make(tmp_path, qtbot)
    nxt.start()
    assert warnings(nxt) == ["VoxFlow 0.3.0 was not installed: Windows did not start its installer. Security "
                             "software may have blocked it."]
    assert nxt.info()["error"] == warnings(nxt)[0] and nxt.info()["downloaded"], "the download is kept for a retry"
    nxt.close()
    third = make(tmp_path, qtbot)
    third.start()
    assert warnings(third) == [], "said once"
    third.close()


def test_up04_a_failed_scheduled_start_comes_back_in_the_tray_and_keeps_earlier_attempts(tmp_path, qtbot):
    up = scheduled(tmp_path, qtbot)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    up.state.attempted, up.state.attempts = "0.3.0", 1                   # one earlier run of Setup
    restarted = []
    up._launcher = lambda *a: False
    up._restarter = lambda tray: restarted.append(tray)
    up._slot_tick()
    assert up.on_exit(user_quit=True) is False and restarted == [True]
    assert (up.state.attempted, up.state.attempts) == ("0.3.0", 1)
    up.close()


def test_up04_install_when_i_quit_that_cannot_run_is_said_at_the_next_start(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    restarted = []
    up._restarter = lambda tray: restarted.append(tray)
    up._launcher = lambda *a: False
    assert up.on_exit(user_quit=True) is False and restarted == [], "the user quit: VoxFlow stays closed"
    assert "did not start its installer" in up.state.exit_error
    up.close()
    changed = downloaded(tmp_path / "b", qtbot)
    (tmp_path / "b" / "dl" / SETUP).write_bytes(b"MZ something else")
    assert changed.on_exit(user_quit=True) is False and changed.launched == []
    assert not changed.info()["install_on_quit"] and not changed.info()["downloaded"]
    changed.close()
    nxt = make(tmp_path / "b", qtbot)
    nxt.start()
    assert warnings(nxt) == ["VoxFlow 0.3.0 was not installed when you quit: the downloaded installer was gone or "
                             "changed."]
    nxt.close()


def test_up04_switching_to_off_while_restarting_lets_go_of_the_installer(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    install_now(up)
    held = up._pending_launch[1]
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    assert held.closed and up._pending_launch is None and up.on_exit(user_quit=True) is False
    up.close()


@windows_only
def test_up04_voxflow_is_started_again_by_a_helper_that_waits_for_this_copy_to_end(tmp_path, monkeypatch):
    calls = []

    def popen(cmd, **kw):
        calls.append((cmd, kw))

    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", "x")
    assert U.restart_app(False, popen=popen) is False and calls == [], "python -m voxflow: nothing to start"
    exe = tmp_path / "it's here" / "VoxFlow.exe"
    assert U.restart_app(True, popen=popen, exe=exe) is True
    cmd, kw = calls[-1]
    script = cmd[-1]
    assert Path(cmd[0]).name == "powershell.exe" and "System32" in cmd[0] and cmd[-2] == "-Command"
    assert script.startswith(f"Wait-Process -Id $env:{U.RESTART_PID_VAR} ") and "Start-Process" in script
    assert script.endswith("-ArgumentList '--autostart'") and str(exe) not in script, "the path travels in env"
    assert kw["env"][U.RESTART_EXE_VAR] == str(exe) and kw["env"][U.RESTART_PID_VAR] == str(os.getpid())
    assert kw["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1" and "_PYI_APPLICATION_HOME_DIR" not in kw["env"]
    assert kw["creationflags"] & subprocess.CREATE_NO_WINDOW and kw["close_fds"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert U.restart_app(False, popen=popen) is True
    assert "--autostart" not in calls[-1][0][-1] and calls[-1][1]["env"][U.RESTART_EXE_VAR] == sys.executable
    assert U.restart_app(False, popen=_raiser(FileNotFoundError(2, "no PowerShell"))) is False
    monkeypatch.setattr(U.sys, "platform", "linux")
    assert U.restart_app(False, popen=popen, exe=exe) is False


# =========================================================================== UP-05: "Remind me later" is tomorrow
def test_up05_remind_me_later_reminds_after_a_day_without_waiting_for_a_check(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 1 and not up._remind_timer.isActive()
    up.clock.t += 60                                                     # the user reads the dialog for a minute
    up.remind_later()
    assert up._remind_timer.isActive() and up._remind_timer.interval() == U.REMIND_AFTER_S * 1000
    requests = len(up._http.requests)
    up.clock.t += U.REMIND_AFTER_S
    up._remind()                                                         # what the timer does
    assert len(up.events["available"]) == 2 and len(up._http.requests) == requests, "the known release, no check"
    assert up.state.remind_after == 0.0 and up.state.notified == "0.3.0"
    up._remind()
    assert len(up.events["available"]) == 2, "once"
    up.close()


def test_up05_the_reminder_timer_fires(tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(U, "REMIND_AFTER_S", 0.05)
    up = make(tmp_path, qtbot)
    up.check(user=False)
    settle(up)
    up.remind_later()
    qtbot.waitUntil(lambda: len(up.events["available"]) == 2, timeout=3000)
    up.close()


def test_up05_a_pending_reminder_is_armed_again_at_the_next_start(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check(user=False)
    settle(up)
    up.remind_later()
    up.close()
    assert not up._remind_timer.isActive()
    nxt = make(tmp_path, qtbot)
    nxt.clock.t += 6 * 3600                                              # six hours later
    nxt.start()
    assert nxt._remind_timer.isActive() and nxt._remind_timer.interval() == 18 * 3600 * 1000
    nxt.close()
    late = make(tmp_path, qtbot)
    late.clock.t += 3 * U.REMIND_AFTER_S                                 # it fell due while VoxFlow was closed
    late.start()
    assert late._remind_timer.interval() == U.STARTUP_DELAY_S[0] * 1000, "soon, not on top of the starting app"
    late.close()
    odd = make(tmp_path, qtbot)
    odd.state.remind_after = odd.clock.t + 400 * U.REMIND_AFTER_S        # the clock was wrong when it was clicked
    odd._arm_remind()
    assert odd._remind_timer.interval() == U.REMIND_AFTER_S * 1000, "never longer than a day"
    odd.close()


def test_up05_no_reminder_when_it_is_off_skipped_already_said_or_already_decided(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.start()
    assert not up._remind_timer.isActive(), "nothing known, nothing pending"
    up.check(user=False)
    settle(up)
    up.remind_later()
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    assert not up._remind_timer.isActive()
    up._remind()
    assert len(up.events["available"]) == 1, "Off: VoxFlow says nothing by itself"
    up.settings["update_mode"] = "notify"
    up.remind_later()
    up.settings_changed("update_mode")
    assert up._remind_timer.isActive(), "back on: the reminder is armed again"
    up.skip()
    assert not up._remind_timer.isActive()
    up.state.skipped = ""
    up.remind_later()
    up.install("quit")                                                   # the user decided meanwhile
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    up._remind()
    assert len(up.events["available"]) == 1
    up.remind_later()
    up.close()
    up._arm_remind()
    assert not up._remind_timer.isActive(), "closed"


# =========================================================================== UP-10: a withdrawn release
def _not_found(_headers):
    problem = U.UpdateError(U._http_problem(404))
    problem.status = 404
    return problem


def test_up10_a_release_that_was_withdrawn_is_not_installed_on_quit(tmp_path, qtbot, caplog):
    up = downloaded(tmp_path, qtbot, mode="auto")
    assert up.info()["install_on_quit"] and (tmp_path / "dl" / SETUP).exists()
    up._http = FakeHttp({API: _not_found})                               # /releases/latest answers 404
    up.clock.t += U.CHECK_EVERY_S
    up.check(user=False)
    settle(up)
    info = up.info()
    assert info["available"] is None and not info["install_on_quit"] and not info["downloaded"]
    assert up.state.latest is None and list((tmp_path / "dl").iterdir()) == []
    assert info["last_result"] == "No published release was found on GitHub." and info["last_ok"]
    assert up._check_timer.interval() == U.CHECK_EVERY_S * 1000, "GitHub answered: tomorrow, not in an hour"
    assert "no longer published" in caplog.text
    assert up.on_exit(user_quit=True) is False and up.launched == []
    up.check(user=False)                                                 # still 404, nothing known: quiet
    settle(up)
    assert caplog.text.count("no longer published") == 1
    up.close()


def test_up10_the_scheduled_install_of_a_withdrawn_release_is_dropped_too(tmp_path, qtbot):
    up = scheduled(tmp_path, qtbot)
    up.start()
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    assert up._slot_timer.isActive()
    up._http = FakeHttp({API: _not_found})
    up.check(user=True)
    settle(up)
    assert not up._slot_timer.isActive()
    up._slot_tick()
    assert up.events["restart"] == 0
    up.close()


def test_up10_the_http_layer_says_which_status_github_answered_with(monkeypatch):
    class Opener:
        code = 404

        def open(self, req, timeout):
            raise urllib.error.HTTPError(req.full_url, self.code, "x", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(U.NoRelease, match="No published release") as caught:
        U.fetch_latest(U.Http())
    assert caught.value.status == 404
    Opener.code = 403                                                    # rate limited: a failure, not "no release"
    with pytest.raises(U.UpdateError, match="limiting") as caught:
        U.fetch_latest(U.Http())
    assert caught.value.status == 403 and not isinstance(caught.value, U.NoRelease)
    assert U.UpdateError("x").status == 0


# =========================================================================== D-05: nothing to offer is not a failure
@pytest.mark.parametrize("change,text", [
    ({"tag_name": "nightly"}, "The newest release on GitHub has no usable version number."),
    ({"prerelease": True}, "The newest release on GitHub is not a final release."),
    ({"draft": True}, "The newest release on GitHub is not a final release."),
])
def test_d05_a_newest_release_that_cannot_be_offered_is_checked_daily_not_hourly(tmp_path, qtbot, caplog, change, text):
    up = downloaded(tmp_path, qtbot)                                     # 0.3.0 is known and waits for the quit
    up._http = github(payload={**release_json("0.4.0"), **change})
    up.clock.t += U.CHECK_EVERY_S
    up.check(user=False)
    settle(up)
    info = up.info()
    assert info["last_ok"] is True and info["last_result"] == text
    assert up._check_timer.interval() == U.CHECK_EVERY_S * 1000, "tomorrow: asking again in an hour changes nothing"
    assert "update check failed" not in caplog.text and warnings(up) == []
    assert info["available"]["version"] == "0.3.0" and info["install_on_quit"], "the release known before stays"
    up.close()


def test_d05_a_check_that_did_not_get_through_is_still_tried_again_in_an_hour(tmp_path, qtbot):
    up = make(tmp_path, qtbot, http=FakeHttp({API: lambda h: Resp(200, b"<html>captive portal</html>")}))
    up.check(user=False)
    settle(up)
    assert up.info()["last_ok"] is False and up._check_timer.interval() == U.RETRY_FAILED_S * 1000
    up.close()


# =========================================================================== UP-07 / UP-11: how this copy was installed
def test_up07_a_folder_that_refuses_a_new_file_is_not_updated_in_place(tmp_path, monkeypatch):
    exe = _exe(tmp_path / "Programs" / "VoxFlow")
    (exe.parent / "unins000.exe").write_bytes(b"")
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        monkeypatch.delenv(var, raising=False)
    assert U.install_kind(exe, frozen=True) == "installer"
    assert sorted(p.name for p in exe.parent.iterdir()) == ["VoxFlow.exe", "unins000.exe"], "the probe leaves nothing"
    with monkeypatch.context() as m:                                     # what Windows answers for a denied folder
        m.setattr(U.os, "open", _raiser(PermissionError(13, "Permission denied")))
        assert U.install_kind(exe, frozen=True) == "allusers" and U.managed_text("allusers")
    with monkeypatch.context() as m:
        m.setattr(U.os, "access", lambda path, mode: False)
        assert U._writable(exe.parent) is False


@windows_only
def test_up07_a_real_folder_whose_permissions_deny_writing_is_detected(tmp_path, monkeypatch):
    """QA up04_kind.py: icacls denies this user writing; os.access(W_OK) still says True."""
    exe = _exe(tmp_path / "locked" / "VoxFlow")
    (exe.parent / "unins000.exe").write_bytes(b"")
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        monkeypatch.delenv(var, raising=False)
    user = os.environ.get("USERNAME", "")
    # WD, AD: no new files, no new folders (nothing else, so the folder can still be read and its entry removed)
    denied = subprocess.run(["icacls", str(exe.parent), "/deny", f"{user}:(WD,AD)"], capture_output=True, timeout=30,
                            stdin=subprocess.DEVNULL)
    if denied.returncode != 0:
        pytest.skip("icacls could not change the permissions of the test folder")
    try:
        assert os.access(exe.parent, os.W_OK), "what the old check trusted"
        assert U._writable(exe.parent) is False and U.install_kind(exe, frozen=True) == "allusers"
    finally:
        subprocess.run(["icacls", str(exe.parent), "/remove:d", user], capture_output=True, timeout=30,
                       stdin=subprocess.DEVNULL)
    assert U.install_kind(exe, frozen=True) == "installer"


def test_up11_an_installer_copy_in_a_folder_that_looks_like_scoops_updates_itself(tmp_path, monkeypatch):
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        monkeypatch.delenv(var, raising=False)
    exe = _exe(tmp_path / "apps" / "voxflow" / "app")                    # D:\apps\voxflow\app, chosen in Setup
    assert U.install_kind(exe, frozen=True) == "scoop", "no uninstaller: Scoop unpacked it"
    (exe.parent / "unins000.exe").write_bytes(b"")
    assert U.install_kind(exe, frozen=True) == "installer"
    (exe.parent / U.MANAGED_MARKER).write_text("winget", encoding="utf-8")
    assert U.install_kind(exe, frozen=True) == "winget"


# =========================================================================== UP-03: free disk space
def test_up03_a_download_that_does_not_fit_is_refused_before_it_starts(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    asked = []

    def free(folder):
        asked.append(folder)
        return 50 * 1024 ** 2

    with pytest.raises(U.UpdateError) as caught:
        U.download_installer(http, rel, tmp_path / "dl", free=free)
    assert str(caught.value) == (f"There is not enough free disk space to download the update: 100 MB is needed in "
                                 f"{tmp_path / 'dl'}, 50 MB is free.")
    assert asked == [tmp_path / "dl"] and SETUP_URL not in http.urls(), "not one byte was fetched"
    (tmp_path / "dl" / (SETUP + ".part")).write_bytes(INSTALLER[:4096])   # a resumed download needs only the rest
    need = len(INSTALLER) - 4096 + U.DOWNLOAD_SPARE_BYTES
    with pytest.raises(U.UpdateError):
        U.download_installer(http, rel, tmp_path / "dl", free=lambda f: need - 1)
    path, _digest = U.download_installer(http, rel, tmp_path / "dl", free=lambda f: need)
    assert path.read_bytes() == INSTALLER
    path.unlink()
    U.download_installer(http, rel, tmp_path / "dl", free=lambda f: None)   # cannot be told: not refused
    unsized = github(size=0)                                             # a release that does not say its size
    U.download_installer(unsized, U.fetch_latest(unsized), tmp_path / "dl2", free=lambda f: 0)


def test_up03_free_space_is_asked_of_the_nearest_folder_that_exists(tmp_path, monkeypatch):
    free = U.free_bytes(tmp_path / "not" / "there" / "yet")
    assert isinstance(free, int) and free == U.free_bytes(tmp_path)
    monkeypatch.setattr(U.shutil, "disk_usage", _raiser(OSError("no such drive")))
    assert U.free_bytes(tmp_path) is None
    assert U.space_problem("install", 10, tmp_path) == "", "unknown is not a refusal"
    assert U._size(int(1.8 * 1024 ** 3)) == "1.8 GB" and U._size(5 * 1024 ** 2) == "5 MB" and U._size(10) == "1 MB"
    text = U.space_problem("install", 5 * 1024 ** 3, tmp_path, free=lambda f: 1024 ** 3)
    assert text == f"There is not enough free disk space to install the update: 5.0 GB is needed in {tmp_path}, " \
                   "1.0 GB is free."


def test_up03_the_updater_says_when_the_download_does_not_fit(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up._free = lambda folder: 1024
    up.check(user=False)
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: bool(warnings(up)), timeout=3000)
    assert "not enough free disk space to download" in warnings(up)[0] and up.info()["error"] == warnings(up)[0]
    assert SETUP_URL not in up._http.urls() and not up.info()["downloaded"]
    up.close()


def test_up03_install_now_without_room_to_unpack_does_not_quit_voxflow(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    up._app_dir = tmp_path / "Programs" / "VoxFlow"
    asked = []

    def free(folder):
        asked.append(folder)
        return int(len(INSTALLER) * U.UNPACKED_FACTOR) - 1

    up._free = free
    install_now(up)
    assert up.events["restart"] == 0 and up.phase == "idle" and asked == [up._app_dir]
    assert len(warnings(up)) == 1 and "not enough free disk space to install" in warnings(up)[0]
    assert str(up._app_dir) in warnings(up)[0] and up.info()["error"] == warnings(up)[0]
    assert up.info()["downloaded"] and up.info()["install_on_quit"], "the download and the plan are kept"
    up._free = lambda folder: int(len(INSTALLER) * U.UNPACKED_FACTOR)    # room was made
    install_now(up)
    assert up.events["restart"] == 1 and up.on_exit(user_quit=True) is True
    up.close()


def test_up03_install_on_quit_without_room_runs_no_setup_and_says_so_at_the_next_start(tmp_path, qtbot):
    up = downloaded(tmp_path, qtbot)
    up._free = lambda folder: 0
    assert up.on_exit(user_quit=True) is False and up.launched == []
    assert up.info()["install_on_quit"] and up.info()["downloaded"] and up.state.attempted == ""
    up.close()
    nxt = make(tmp_path, qtbot)
    nxt.start()
    assert "not enough free disk space to install" in warnings(nxt)[0]
    assert nxt.on_exit(user_quit=True) is True, "with room again the same download installs"
    nxt.close()


def test_up03_a_scheduled_install_without_room_is_tried_once_a_day_not_every_minute(tmp_path, qtbot):
    up = scheduled(tmp_path, qtbot)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    up._free = lambda folder: 0
    up._slot_tick()
    up._slot_tick()
    assert up.events["restart"] == 0 and len(warnings(up)) == 1 and up.state.slot_done
    assert up._slot_timer.isActive(), "tomorrow's slot may find room"
    up.close()


# =========================================================================== UP-09: back in the tray after a set-time update
def test_up09_the_relaunch_after_a_scheduled_update_stays_in_the_tray(tmp_path, qtbot, monkeypatch, data_dir):
    from voxflow import __main__ as entry
    from voxflow import app
    started = []
    monkeypatch.setattr(app, "run_app", lambda **kw: started.append(kw["autostart"]) or 0)
    monkeypatch.setattr(sysmod, "update_setup_running", lambda: True)    # Setup still holds its mutex
    assert U.relaunch_in_tray() is False and U.relaunch_in_tray(tmp_path / "nothing") is False
    assert entry.main(["--after-update"]) == 0 and started == [False], "no record: the window opens as before"

    up = scheduled(tmp_path, qtbot)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    up._slot_tick()
    assert up.on_exit(user_quit=True) is True and up.launched[0][1] is True, "Setup is asked to start VoxFlow again"
    up.close()
    assert U.relaunch_in_tray(tmp_path / "data") is True
    (data_dir / U.STATE_FILE).write_bytes((tmp_path / "data" / U.STATE_FILE).read_bytes())
    assert entry.main(["--after-update"]) == 0 and started[-1] is True, "hidden, like --autostart"
    assert entry.main(["--after-update", "--autostart"]) == 0 and started[-1] is True
    monkeypatch.setattr(sysmod, "update_setup_running", lambda: False)
    assert entry.main([]) == 0 and started[-1] is False, "a start by hand opens the window"

    again = downloaded(tmp_path / "b", qtbot)                            # Install now: someone is at the PC
    install_now(again)
    assert again.on_exit(user_quit=True) is True
    again.close()
    assert U.relaunch_in_tray(tmp_path / "b" / "data") is False


# =========================================================================== the installer script
def _iss() -> tuple[str, str]:
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    return iss, iss.split("\n[Code]\n", 1)[1]


def _body(code: str, name: str) -> str:
    return code.split(name, 1)[1].split("\nend;", 1)[0]


def test_up12_uninstall_removes_the_update_download_folder():
    iss, _code = _iss()
    deletes = iss.split("\n[UninstallDelete]\n", 1)[1].split("\n[", 1)[0]
    assert r'Type: filesandordirs; Name: "{%TEMP|{localappdata}\Temp}\VoxFlow-update"' in deletes
    assert U.download_folder().name == "VoxFlow-update"
    assert r'Type: filesandordirs; Name: "{app}\previous"' in deletes
    assert "{userappdata}" not in iss, "never the profile"


def test_up03_setup_keeps_the_old_program_files_until_the_new_ones_are_complete():
    iss, code = _iss()
    assert r'Type: filesandordirs; Name: "{app}\_internal"' in iss.split("\n[InstallDelete]\n", 1)[1].split("\n[", 1)[0]
    prepare = _body(code, "function PrepareToInstall")
    assert prepare.index("StopRunningCopy;") < prepare.index("KeepOldCopy;") < prepare.index("Prepared := True;")
    keep = code.split("procedure KeepOldCopy;", 1)[1].split("procedure RestoreOldCopy;", 1)[0]
    assert r"RenameFile(App + '\_internal', Prev + '\_internal')" in keep, "moved, not copied: no extra space"
    assert r"FileCopy(App + '\VoxFlow.exe', Prev + '\VoxFlow.exe', False)" in keep
    assert r"DirExists(Prev + '\_internal') or not DirExists(App + '\_internal')" in keep
    restore = code.split("procedure RestoreOldCopy;", 1)[1].split("procedure RemoveOldCopy;", 1)[0]
    assert restore.index(r"DelTree(App + '\_internal'") < restore.index(r"RenameFile(Prev + '\_internal', App")
    assert r"FileCopy(Prev + '\VoxFlow.exe', App + '\VoxFlow.exe', False)" in restore
    done = _body(code, "procedure CurStepChanged")
    assert done.index("CurStep <> ssPostInstall") < done.index("Installed := True;") < done.index("RemoveOldCopy;")
    end = _body(code, "procedure DeinitializeSetup;")
    assert "if Installed or not Prepared then\n    Exit;\n  RestoreOldCopy;" in end
    # a VoxFlow that quit for an update that then failed is started again (the old version), only for the updater
    assert "if RelaunchAfterUpdate and FileExists(ExpandConstant('{app}\\VoxFlow.exe')) then" in end
    assert "'--after-update', '', SW_SHOWNORMAL, ewNoWait, ResultCode" in end
    events = re.findall(r"^(?:procedure|function)\s+(\w+)", code, re.M)
    assert events == ["StopRunningCopy", "KeepOldCopy", "RestoreOldCopy", "RemoveOldCopy", "PrepareToInstall",
                      "RelaunchAfterUpdate", "CurStepChanged", "DeinitializeSetup", "CurUninstallStepChanged"]


# =========================================================================== R-A: nothing of HTTP at import
def test_ra_importing_the_updater_does_not_import_the_http_stack():
    code = ("import sys; import voxflow.updater; "
            "print([m for m in ('urllib.request', 'http.client', 'email', 'ssl') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stdout + out.stderr
    assert U.checked_redirect() is U.checked_redirect(), "the redirect handler class is built once, on first use"
