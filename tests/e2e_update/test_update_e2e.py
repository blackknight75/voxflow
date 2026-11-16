"""The updater the way a user meets it: real Controller, real Updater, real urllib/TLS against a local fake GitHub,
real files in %TEMP%\\VoxFlow-update, a real exe started by the real launch_installer. See harness.py for exactly
what is faked. Run: ``set VOXFLOW_UPDATE_E2E=1`` then ``python -m pytest tests/e2e_update -q``."""
from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import time

import hashlib
import pytest
from PySide6.QtWidgets import QApplication

import voxflow
from tests.e2e_update import harness as H
from voxflow import updater as U
from voxflow.ui import updates as UP
from voxflow.ui.widgets import Button

pytestmark = pytest.mark.update_e2e

SWITCHES = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]


def api_calls(env) -> int:
    return env.sc.count(H.API_HOST, "/releases/latest")


def available(app) -> list[dict]:
    return app.rig.spy.of("update_available")


def notices(app) -> list[tuple[str, str]]:
    return app.rig.spy.notices()


def dialog(app):
    """Open the "new version" dialog the way "See update" does; returns (modal, notes text, buttons by name)."""
    app.shell.run_action("update")
    QApplication.processEvents()
    page = app.shell.window.pages["settings"]
    modal = page.open_update()
    assert modal is not None
    buttons = {b.objectName(): b for b in modal.findChildren(Button) if b.objectName().startswith("update_dialog")}
    from voxflow.ui.widgets import TextArea
    notes = modal.findChild(TextArea, "update_notes") or next(iter(modal.findChildren(TextArea)))
    return modal, notes, buttons


def assert_installer_args(record: dict, folder, relaunch: bool):
    assert record["argv"][:4] == SWITCHES + [f"/RELAUNCH={1 if relaunch else 0}"], record["argv"]
    assert record["argv"][4].lower() == f"/LOG={folder / 'install.log'}".lower(), record["argv"]
    assert record["meipass2"] is None, "the installer must not inherit this process's PyInstaller state"
    assert record["e2e_canary"] == "1", "the rest of the environment is passed on"
    assert record["exe"].lower().startswith(str(folder).lower()), record["exe"]


# =========================================================================== 1. Off
def test_1_off_sends_nothing_on_start_or_on_the_daily_timer_and_check_now_asks_once(env):
    app = env.app(mode="off", ui=True)
    env.qtbot.wait(1500)                                   # the first automatic check would come after 0.3 s
    assert env.sc.requests == [] and env.gh.dns_lookups == [], "Off must not touch the network"
    assert not app.u._check_timer.isActive()
    app.u._arm_check(first=True)                           # what the daily timer does: it stays unarmed
    assert not app.u._check_timer.isActive()
    sec = app.shell.window.pages["settings"].updates
    text = sec.status_row.sub.text()
    assert "Automatic checks are off" in text and "Check now" in text
    assert sec.btn_check.isEnabled(), "Check now works while Off"
    sec.btn_check.click()                                  # the user presses Check now
    app.wait(lambda: api_calls(env) == 1 and app.u.phase == "idle")
    assert available(app), "a check the user asked for shows what it found"
    env.qtbot.wait(1500)
    assert api_calls(env) == 1, "exactly one request, and Off arms nothing afterwards"
    assert not app.u._check_timer.isActive()
    assert "VoxFlow 0.3.0 is available." in sec.status_row.sub.text()
    env.note("off/check-now", f"one request; Settings: {sec.status_row.sub.text()!r}")
    # Notify a day later: the daily check comes back (right after Check now it is correctly due in 24 h)
    env.clock.advance(24 * 3600 + 1)
    app.c.set("update_mode", "notify")
    before = api_calls(env)
    app.wait(lambda: api_calls(env) == before + 1 and app.u.phase == "idle")
    env.qtbot.wait(800)
    assert api_calls(env) == before + 1
    app.c.set("update_mode", "off")
    env.qtbot.wait(800)
    assert api_calls(env) == before + 1
    env.note("off", f"{len(env.sc.requests)} requests in total: one Check now while Off, one after Notify")


# =========================================================================== 2. Notify
def test_2_notify_shows_a_notification_and_a_plain_text_dialog(env, caplog):
    caplog.set_level(logging.INFO, "voxflow.updater")
    app = env.app(mode="notify", ui=True)
    app.wait(lambda: available(app))                       # the automatic check, 0.3 s after start
    info = available(app)[0][0]
    assert info["available"]["version"] == "0.3.0" and info["installable"]
    # the window is up: a toast with "See update" (tray otherwise; see the next assertion)
    req = [r for r in env.sc.requests if r["host"] == H.API_HOST][0]
    assert req["headers"]["User-Agent"] == f"VoxFlow/{voxflow.__version__}" and req["tls"]   # the package version
    assert "Cookie" not in req["headers"] and "Authorization" not in req["headers"]
    modal, notes, buttons = dialog(app)
    text = notes.toPlainText()
    assert "VoxFlow 0.3.0 is available" in modal.findChild(type(modal.title), None).text() \
        if hasattr(modal, "title") else True
    assert "• A new thing" in text and "See the docs." in text
    assert "evil" not in text and "<img" not in text and "pixel" not in text and "![" not in text
    assert notes.isReadOnly() and type(notes).__mro__[1].__name__ == "QPlainTextEdit"
    assert set(buttons) == {"update_dialog_skip", "update_dialog_later", "update_dialog_quit", "update_dialog_now"}
    assert all(h != H.EVIL_HOST for h, _p in env.gh.dns_lookups), "nothing in the notes was fetched"
    # a second automatic check does not notify again for the same version
    app.check(user=False)
    assert len(available(app)) == 1
    env.note("notify/dialog", f"notes={text!r}")


def test_2_install_now_waits_for_the_dictation_then_restarts_into_the_installer(env):
    app = env.app(mode="notify", ui=True)
    app.wait(lambda: available(app))
    app.rig.asr.delay_s = 3.0                              # processing takes 3 s (longer than the 2 s poll)
    app.rig.pause()
    app.rig.press()                                        # the user is dictating
    assert app.c.state == "listening"
    _m, _n, buttons = dialog(app)
    buttons["update_dialog_now"].click()
    app.wait(lambda: app.u.phase == "waiting")             # downloaded and verified, now waiting for the dictation
    assert app.files() == ["VoxFlow-Setup-0.3.0.exe"]
    env.qtbot.wait(2500)
    assert app.u.phase == "waiting" and not app.restarts, "never while recording"
    app.rig.release()
    app.wait(lambda: app.c.state == "processing", 3000)
    env.qtbot.wait(2200)
    assert app.u.phase == "waiting" and not app.restarts, "never while processing"
    app.wait(lambda: app.restarts, 15000)
    assert app.u.phase == "restarting"
    assert app.c.state in ("done", "idle") and app.rig.last_entry() is not None, "the dictation finished first"
    assert not env.record.exists(), "the installer starts only once VoxFlow has shut down"
    app.exit(user_quit=False)                              # QApplication.quit -> shutdown (restart counts as quit)
    rec = env.ran()
    assert_installer_args(rec, app.u.folder, relaunch=True)
    env.note("notify/install-now", f"argv={rec['argv']}")


def test_2_install_when_i_quit_runs_the_installer_on_quit_without_a_relaunch(env):
    app = env.app(mode="notify", ui=True)
    app.wait(lambda: available(app))
    _m, _n, buttons = dialog(app)
    buttons["update_dialog_quit"].click()
    app.wait(lambda: app.u.state.downloaded == "0.3.0" and app.u.phase == "idle")
    assert "when you quit VoxFlow" in UP.status_text(app.c.update_info())[0]
    env.qtbot.wait(2500)
    assert not app.restarts and not env.record.exists()
    app.shell.quit()                                       # the tray's Quit: prepare_quit + QApplication.quit
    app.exit(user_quit=False)                              # (prepare_quit already done by the shell)
    rec = env.ran()
    assert_installer_args(rec, app.u.folder, relaunch=False)
    env.note("notify/install-on-quit", f"argv={rec['argv']}")


def test_2_install_when_i_quit_does_not_run_when_windows_ends_the_session(env):
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("quit")
    app.wait(lambda: app.u.state.downloaded == "0.3.0")
    app.exit(user_quit=False)                              # logoff / shutdown: no prepare_quit
    time.sleep(3)
    assert not env.record.exists()


def test_2_remind_me_later_is_quiet_until_the_next_day(env):
    app = env.app(mode="notify", ui=True)
    app.wait(lambda: available(app))
    _m, _n, buttons = dialog(app)
    buttons["update_dialog_later"].click()
    n = len(available(app))
    app.check(user=False)                                  # the daily check, an hour later
    env.clock.advance(3600)
    app.check(user=False)
    assert len(available(app)) == n, "no prompt before the remind time"
    env.clock.advance(23 * 3600 + 1)                       # time travel: past the 24 h
    app.check(user=False)
    assert len(available(app)) == n + 1
    env.note("notify/remind-later", "quiet at +1 h, prompted at +24 h")


def test_2_skip_this_version_is_quiet_until_a_newer_one(env):
    app = env.app(mode="notify", ui=True)
    app.wait(lambda: available(app))
    app.c.install_update("quit")                           # something is downloaded: skip must remove it
    app.wait(lambda: app.u.state.downloaded == "0.3.0")
    assert app.files() == ["VoxFlow-Setup-0.3.0.exe"]
    _m, _n, buttons = dialog(app)
    buttons["update_dialog_skip"].click()
    assert app.files() == [], "skip deletes the download"
    n = len(available(app))
    env.clock.advance(3 * 86400)
    app.check(user=False)
    assert len(available(app)) == n
    assert "You skipped this version." in UP.status_text(app.c.update_info())[0]
    env.sc.version = "0.4.0"
    app.check(user=False)
    assert len(available(app)) == n + 1 and available(app)[-1][0]["available"]["version"] == "0.4.0"
    app.exit()
    time.sleep(2)
    assert not env.record.exists(), "a skipped version is never installed on quit"


# =========================================================================== 3. Automatic
def test_3_auto_when_i_quit(env):
    app = env.app(mode="auto", when="quit", metered=lambda: False)
    app.wait(lambda: app.u.state.downloaded == "0.3.0" and app.u.phase == "idle")
    assert ("info", "VoxFlow 0.3.0 will be installed when you quit VoxFlow.") in notices(app)
    assert not available(app), "automatic mode does not ask"
    env.qtbot.wait(2500)
    assert not app.restarts
    app.exit(user_quit=True)
    rec = env.ran()
    assert_installer_args(rec, app.u.folder, relaunch=False)
    env.note("auto/quit", f"argv={rec['argv']}")


def test_3_auto_on_a_metered_connection_waits(env):
    app = env.app(mode="auto", when="quit", metered=lambda: True)
    app.wait(lambda: api_calls(env) == 1 and app.u.phase == "idle")
    env.qtbot.wait(500)
    assert env.sc.count(H.ASSET_HOST) == 0 and app.files() == []
    assert "metered" in app.c.update_info()["error"]


def test_3_auto_at_a_set_time_only_when_idle_and_not_dictating(env, monkeypatch):
    monkeypatch.setattr(U, "SCHEDULE_POLL_MS", 200)        # the slot is looked at every minute; 0.2 s here
    today = dt.datetime.now().astimezone()
    env.now_value = today.replace(hour=5, minute=0, second=0, microsecond=0)     # outside the 03:00 slot
    env.idle_value = 3600.0
    app = env.app(mode="auto", when="time", at="03:00", metered=lambda: False)
    app.wait(lambda: app.u.state.downloaded == "0.3.0" and app.u.phase == "idle")
    assert any("installed at 03:00" in m for _l, m in notices(app))
    env.qtbot.wait(800)
    assert not app.restarts, "05:00 is outside the slot"
    env.now_value = today.replace(hour=3, minute=10, second=0, microsecond=0)
    env.idle_value = 60.0                                  # someone used the PC a minute ago
    env.qtbot.wait(800)
    assert not app.restarts, "not while someone is at the PC"
    env.idle_value = 700.0
    app.rig.pause()
    app.rig.press()                                        # ...and not while dictating (the hotkey, not input)
    env.qtbot.wait(800)
    assert not app.restarts, "not while dictating"
    app.rig.release()
    app.wait(lambda: app.restarts, 10000)
    assert app.u.state.slot_done == env.now_value.date().isoformat()
    app.exit(user_quit=False)
    rec = env.ran()
    assert_installer_args(rec, app.u.folder, relaunch=True)
    env.note("auto/time", f"argv={rec['argv']}")


def test_3_the_real_idle_and_metered_apis_answer(env):
    idle = U.idle_seconds()
    metered = U.connection_metered()
    assert idle is not None and idle >= 0
    assert metered in (True, False, None)
    env.note("real windows apis", f"GetLastInputInfo idle={idle:.1f}s, INetworkCostManager metered={metered}")


# =========================================================================== 4. versions
@pytest.mark.parametrize("tag,result", [
    ("v0.2.0", "VoxFlow is up to date."), ("v0.1.9", "VoxFlow is up to date."),
    ("0.2.0", "VoxFlow is up to date."), ("v0.2.0+build.7", "VoxFlow is up to date."),
    ("v0.2.1", "VoxFlow 0.2.1 is available."), ("v0.10.0", "VoxFlow 0.10.0 is available."),
    ("V1.0.0", "VoxFlow 1.0.0 is available."),
])
def test_4_versions_from_a_real_answer(env, tag, result):
    env.sc.tag = tag
    env.sc.version = U.parse_version(tag) and str(U.parse_version(tag))
    app = env.app(mode="notify", version="0.2.0")
    app.wait(lambda: api_calls(env) == 1 and app.u.phase == "idle")
    info = app.c.update_info()
    assert info["last_result"] == result and info["last_ok"]
    assert bool(available(app)) == ("available" in result)
    if "up to date" in result:
        assert UP.status_text(info)[0].startswith("VoxFlow is up to date.")


@pytest.mark.parametrize("change,text", [
    ({"draft": True}, "not a final release"), ({"prerelease": True}, "not a final release"),
    ({"tag": "v0.3.0-rc.1"}, "no usable version"), ({"tag": "latest"}, "no usable version"),
    ({"api_raw": b"<html>rate limited</html>"}, "does not understand"),
])
def test_4_drafts_prereleases_and_odd_answers_are_never_offered(env, change, text):
    for k, v in change.items():
        setattr(env.sc, k, v)
    app = env.app(mode="notify")
    app.wait(lambda: api_calls(env) == 1 and app.u.phase == "idle")
    info = app.c.update_info()
    assert text in info["last_result"] and not info["last_ok"] and info["available"] is None
    assert not available(app) and not notices(app)


# =========================================================================== 5. refusals and failures
def test_5_a_hash_mismatch_is_refused_deleted_and_reported(env):
    env.sc.sums_override = f"{'ab' * 32}  VoxFlow-Setup-0.3.0.exe\n"
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("now")
    app.wait(lambda: notices(app))
    level, message = notices(app)[-1]
    assert level == "warning" and "does not match its published SHA-256" in message
    assert app.files() == [], "the file is deleted"
    assert not app.restarts and app.u.state.downloaded == ""
    assert "does not match" in UP.status_text(app.c.update_info())[0]
    app.exit()
    time.sleep(2)
    assert not env.record.exists()
    env.note("hash mismatch", message)


def test_5_no_sha256sums_means_no_install(env):
    env.sc.with_sums = False
    app = env.app(mode="notify", ui=True)
    app.wait(lambda: available(app))
    assert not app.c.update_info()["installable"]
    _m, _n, buttons = dialog(app)
    assert "update_dialog_page" in buttons and "update_dialog_now" not in buttons
    with pytest.raises(ValueError, match="no Windows installer"):
        app.c.install_update("now")
    assert env.sc.count(H.ASSET_HOST) == 0


def test_5_sums_without_the_installer_line_is_refused(env):
    env.sc.sums_override = f"{'0' * 64}  other.zip\n"
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("now")
    app.wait(lambda: notices(app))
    assert "does not list the installer" in notices(app)[-1][1]
    assert env.sc.count(H.ASSET_HOST, "/exe") == 0, "the installer is not even downloaded"


@pytest.mark.parametrize("target", [
    f"https://{H.EVIL_HOST}/github-production-release-asset/1/exe",
    f"http://{H.ASSET_HOST}/github-production-release-asset/1/exe",
    f"https://{H.ASSET_HOST}:8443/github-production-release-asset/1/exe",
])
def test_5_a_redirect_to_an_untrusted_place_is_refused(env, target):
    env.sc.asset_redirect = target
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("now")
    app.wait(lambda: notices(app))
    assert "redirected to a server VoxFlow does not trust" in notices(app)[-1][1]
    assert H.EVIL_HOST not in [h for h, _p in env.gh.dns_lookups], "the untrusted host was never contacted"
    assert not any(p == 80 for _h, p in env.gh.dns_lookups), "no plain-HTTP connection"
    assert [n for n in app.files() if n.endswith(".exe")] == []


def test_5_an_interrupted_download_is_resumed_and_cleaned_up(env):
    env.sc.truncate_once = len(env.sc.installer) // 3
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("quit")
    app.wait(lambda: notices(app))
    first = notices(app)[-1][1]
    assert app.files() == ["VoxFlow-Setup-0.3.0.exe.part"], app.files()
    part = (app.u.folder / "VoxFlow-Setup-0.3.0.exe.part").stat().st_size
    assert part == env.sc.truncate_once or part == len(env.sc.installer) // 3
    app.c.install_update("quit")                           # the user tries again (or the next automatic try)
    app.wait(lambda: app.u.state.downloaded == "0.3.0")
    ranged = [r for r in env.sc.requests if "/exe?" in r["path"] and "Range" in r["headers"]]
    assert ranged and ranged[-1]["headers"]["Range"] == f"bytes={part}-"
    assert app.files() == ["VoxFlow-Setup-0.3.0.exe"], "the .part is gone"
    assert U.sha256_file(app.u.folder / "VoxFlow-Setup-0.3.0.exe") == hashlib.sha256(env.sc.installer).hexdigest()
    env.note("interrupted download", f"first try: {first!r}; resumed with Range bytes={part}-")


@pytest.mark.parametrize("setup,text", [
    (lambda sc, gh: setattr(sc, "api_status", 404), "No published release"),
    (lambda sc, gh: setattr(sc, "api_status", 500), "HTTP 500"),
    (lambda sc, gh: setattr(sc, "api_status", 403), "limiting requests"),
    (lambda sc, gh: setattr(sc, "api_delay", 2.5), "could not be reached"),
    (lambda sc, gh: setattr(gh, "network", False), "could not be reached"),
])
def test_5_server_failures_are_quiet_and_shown_in_settings_and_the_log(env, caplog, setup, text):
    caplog.set_level(logging.INFO, "voxflow.updater")
    setup(env.sc, env.gh)
    app = env.app(mode="notify", timeout=1.0, ui=True)
    app.wait(lambda: app.u.state.last_check and app.u.phase == "idle", 20000)
    info = app.c.update_info()
    assert text in info["last_result"] and not info["last_ok"]
    sentence, role = UP.status_text(info)
    assert text in sentence and role == "AMBER"
    assert app.shell.window.pages["settings"].updates.status_row.sub.text() == sentence
    assert not notices(app) and not available(app), "no popup, no tray message"
    assert any("update check failed" in r.getMessage() for r in caplog.records)
    env.note(f"failure {text}", info["last_result"])


def test_5_a_failed_download_the_user_asked_for_is_reported(env):
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    env.sc.asset_status = 500
    app.c.install_update("now")
    app.wait(lambda: notices(app))
    assert "HTTP 500" in notices(app)[-1][1]
    assert not app.restarts


# =========================================================================== 6. managed copies
def _layout(root, kind, stub):
    if kind == "scoop":
        app = root / "scoop" / "apps" / "voxflow" / "current"
    else:
        app = root / kind / "VoxFlow"
    app.mkdir(parents=True)
    shutil.copy(stub, app / "VoxFlow.exe")
    if kind in ("installer", "winget"):
        (app / "unins000.exe").write_bytes(b"MZ")
    if kind == "winget":
        (app / U.MANAGED_MARKER).write_text("winget\n", encoding="utf-8")
    return app / "VoxFlow.exe"


@pytest.mark.parametrize("kind,message", [
    ("scoop", "scoop update voxflow"), ("winget", "winget upgrade blackknight75.VoxFlow"),
    ("portable", "cannot update itself"), ("source", "git pull"),
])
def test_6_scoop_winget_portable_and_source_copies_never_update_themselves(env, stub_exe, kind, message):
    if kind == "source":
        detected = U.install_kind()                        # this test process: python, not frozen
    else:
        detected = U.install_kind(_layout(env.tmp / "layouts", kind, stub_exe), frozen=True)
    assert detected == kind
    app = env.app(mode="auto", kind=detected, ui=True, metered=lambda: False)
    app.wait(lambda: api_calls(env) == 1 and app.u.phase == "idle")
    env.qtbot.wait(500)
    assert env.sc.count(H.ASSET_HOST) == 0 and app.files() == [], "no download"
    info = app.c.update_info()
    assert message in info["managed"] and not info["installable"]
    assert message in UP.status_text(info)[0]
    with pytest.raises(ValueError, match=message.split()[0]):
        app.c.install_update("now")
    _m, _n, buttons = dialog(app)
    assert "update_dialog_page" in buttons and "update_dialog_now" not in buttons
    env.note(f"kind {kind}", info["managed"])


def test_6_an_inno_install_is_detected_as_one(env, stub_exe):
    assert U.install_kind(_layout(env.tmp / "layouts", "installer", stub_exe), frozen=True) == "installer"


# =========================================================================== 7. the download folder
def test_7_the_download_folder_holds_one_installer_and_is_emptied_after_the_update(env, monkeypatch):
    monkeypatch.setattr(U, "TIDY_RETRY_MS", 300)
    app = env.app(mode="notify")
    assert app.u.folder == U.download_folder() and app.u.folder.parent == H.Path(H.tempfile.gettempdir()) \
        if hasattr(H, "tempfile") else True
    app.wait(lambda: available(app))
    app.c.install_update("quit")
    app.wait(lambda: app.u.state.downloaded == "0.3.0")
    (app.u.folder / "stray.tmp").write_bytes(b"x")
    env.sc.version = "0.4.0"
    app.check(user=True)
    app.c.install_update("quit")
    app.wait(lambda: app.u.state.downloaded == "0.4.0")
    assert app.files() == ["VoxFlow-Setup-0.4.0.exe"], app.files()
    app.exit(user_quit=True)
    rec = env.ran()
    assert rec["argv"][3] == "/RELAUNCH=0"
    (app.u.folder / "install.log").write_text("Inno Setup log", encoding="utf-8")   # what Setup leaves
    # the new version starts at once, while the installer may still be running (Inno Setup's [Run] nowait entry
    # starts VoxFlow before Setup exits, and Windows keeps a running exe locked): it is removed as soon as it can be
    after = env.app(mode="notify", version="0.4.0")
    locked_at_start = after.files()
    after.wait(lambda: after.files() == [], 20000)
    assert after.u.state.downloaded == "" and after.u.state.install_on_quit == ""
    env.note("download folder", f"one installer at a time; at the new version's start: {locked_at_start}, then []")


def test_7_switching_off_removes_the_download(env):
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("quit")
    app.wait(lambda: app.u.state.downloaded == "0.3.0")
    assert app.files() == ["VoxFlow-Setup-0.3.0.exe"]
    app.c.set("update_mode", "off")
    assert app.files() == [], "deleted at once"
    app.exit(user_quit=True)
    time.sleep(2)
    assert not env.record.exists(), "Off cancels an install on quit"
    assert app.files() == [], f"Off deletes the download: {app.files()}"


# =========================================================================== 8. persistence
def test_8_settings_and_choices_survive_a_restart(env):
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.set("update_mode", "auto")
    app.c.set("update_install_when", "time")
    app.c.set("update_time", "04:00")
    app.c.skip_update()
    app.exit()
    cfg = json.loads((env.tmp / "data" / "config.json").read_text(encoding="utf-8"))
    assert (cfg["update_mode"], cfg["update_install_when"], cfg["update_time"]) == ("auto", "time", "04:00")
    again = env.app(config=False)
    assert again.c.get("update_mode") == "auto" and again.c.get("update_install_when") == "time"
    assert again.c.get("update_time") == "04:00"
    assert again.u.state.skipped == "0.3.0" and again.u.state.latest["version"] == "0.3.0"
    info = again.c.update_info()
    assert info["skipped"] and info["mode"] == "auto" and info["time"] == "04:00"
    env.note("persistence", f"config={ {k: cfg[k] for k in ('update_mode', 'update_install_when', 'update_time')} }")


# =========================================================================== 9. new on fix/update (merged)
def test_9_an_all_users_install_only_notifies_and_offers_download(env, stub_exe, monkeypatch):
    pf = env.tmp / "Program Files"
    monkeypatch.setenv("ProgramFiles", str(pf))            # where an /ALLUSERS install lives
    app_dir = pf / "VoxFlow"
    app_dir.mkdir(parents=True)
    shutil.copy(stub_exe, app_dir / "VoxFlow.exe")
    (app_dir / "unins000.exe").write_bytes(b"MZ")
    kind = U.install_kind(app_dir / "VoxFlow.exe", frozen=True)
    assert kind == "allusers"
    app = env.app(mode="auto", kind=kind, ui=True, metered=lambda: False)
    app.wait(lambda: api_calls(env) == 1 and app.u.phase == "idle")
    env.qtbot.wait(500)
    assert env.sc.count(H.ASSET_HOST) == 0 and app.files() == [], "auto mode downloads nothing"
    info = app.c.update_info()
    assert not info["installable"] and "installed for all users" in info["managed"]
    with pytest.raises(ValueError, match="all users"):
        app.c.install_update("now")
    _m, _n, buttons = dialog(app)
    assert set(buttons) >= {"update_dialog_page"} and "update_dialog_now" not in buttons
    assert buttons["update_dialog_page"].text() == "Download"
    env.note("all users", f"kind={kind}; button {buttons['update_dialog_page'].text()!r}; {info['managed']}")


_ENTRY = """
import sys, time
sys.path.insert(0, {root!r})
from voxflow import system, app
from voxflow import __main__ as entry
system.tell_update_running = lambda *a, **k: print("TOLD: Finishing the update", flush=True)
app.run_app = lambda **kw: print("STARTED", sorted(k for k, v in kw.items() if v), flush=True) or 0
t = time.perf_counter()
code = entry.main(sys.argv[1:])
print("EXIT", code, round((time.perf_counter() - t) * 1000), "ms", flush=True)
"""


def _mutex(name):
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k32.CreateMutexW(None, False, name)
    assert handle
    return lambda: k32.CloseHandle(handle)


@pytest.mark.parametrize("name", ["VoxFlowSetupRunning", "Global\\VoxFlowSetupRunning"])
def test_9_a_voxflow_started_while_setup_runs_says_finishing_the_update_and_exits(env, name):
    """The real entry point in a separate process, the real named mutex as Inno's SetupMutex creates it (only the
    message box and the app itself are replaced by prints)."""
    import subprocess
    import sys
    root = str(H.HERE.parent.parent)
    run = lambda *args: subprocess.run([sys.executable, "-c", _ENTRY.format(root=root), *args],
                                       capture_output=True, text=True, timeout=120, cwd=root)
    release = _mutex(name)
    try:
        plain = run()
        after = run("--after-update")
    finally:
        release()
    assert "TOLD: Finishing the update" in plain.stdout and "STARTED" not in plain.stdout, plain.stdout + plain.stderr
    assert "EXIT 0" in plain.stdout
    assert "STARTED" in after.stdout and "TOLD" not in after.stdout, "Setup's own relaunch is not refused"
    free = run()
    assert "STARTED" in free.stdout and "TOLD" not in free.stdout, "no Setup running: VoxFlow starts"
    env.note(f"setup mutex {name}", plain.stdout.strip().replace("\n", " | "))


def test_9_install_on_quit_never_asks_setup_to_relaunch(env):
    app = env.app(mode="notify")
    app.wait(lambda: available(app))
    app.c.install_update("quit")
    app.wait(lambda: app.u.state.downloaded == "0.3.0")
    app.exit(user_quit=True)
    rec = env.ran()
    assert "/RELAUNCH=0" in rec["argv"] and "/RELAUNCH=1" not in rec["argv"]
    assert not any(a.startswith("--after-update") for a in rec["argv"])
