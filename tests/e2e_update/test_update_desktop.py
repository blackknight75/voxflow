"""The update UI on the REAL Windows desktop: the ``windows`` Qt platform, the real system tray and notifications,
driven through Windows UI Automation (what a screen reader uses) and in-process Qt events; the real Controller and
Updater against the local fake GitHub (conftest.py), with the recording stub as the installer.

Run (one process at a time; add QT_SCALE_FACTOR=1.5 for the high-DPI pass):

    set VOXFLOW_UPDATE_E2E=1& set VOXFLOW_DESKTOP_TESTS=1& python -m pytest -q tests/e2e_update/test_update_desktop.py

Rules these tests keep (another app, possibly a game with an anti-cheat, may own the foreground):
* NO synthetic OS input: no SendInput / mouse_event / keybd_event / SendKeys; the cursor never moves. Buttons are
  pressed with UIA InvokePattern; keys and menu-row clicks are Qt events sent to VoxFlow's own widgets (QTest).
* Windows are shown without activating them (``Qt.WA_ShowWithoutActivating``; the app's own bring-to-front is
  replaced by that), so the foreground app keeps the keyboard.
* Screenshots are ``QWidget.grab()`` of VoxFlow's own window: never the screen.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import time

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from tests.e2e_update import desktop as D
from tests.e2e_update import harness as H
from voxflow import updater as U

pytestmark = [pytest.mark.update_e2e, pytest.mark.desktop]

SCALE = os.environ.get("QT_SCALE_FACTOR", "")
TAG = f"@{SCALE}x" if SCALE else ""
FG: list[str] = []


# --------------------------------------------------------------------------------------------------- helpers
def foreground_pid() -> int:
    u32 = ctypes.windll.user32
    pid = ctypes.c_ulong()
    u32.GetWindowThreadProcessId(u32.GetForegroundWindow(), ctypes.byref(pid))
    return pid.value


@pytest.fixture
def quiet(monkeypatch):
    """The app's bring_to_front (present(): the tray, a notification click) shows the window without activating
    it; every call is recorded so a test can check the app did ask to come forward."""
    calls = []

    def show_quietly(window):
        calls.append(window.objectName() or type(window).__name__)
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        window.show()
        window.raise_()

    import voxflow.ui.shell as shell_mod
    import voxflow.ui.window as window_mod
    monkeypatch.setattr(window_mod, "bring_to_front", show_quietly)
    monkeypatch.setattr(shell_mod, "bring_to_front", show_quietly)
    before = foreground_pid()
    yield calls
    after = foreground_pid()
    FG.append(f"foreground pid before {before}, after {after}, VoxFlow (this process) {os.getpid()}")


def start(env, **kw):
    """VoxFlow in the tray (as after login), the window shown without activation on Settings > Updates."""
    app = env.app(ui="tray", **kw)
    w = app.shell.window
    w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    return app, w


def open_updates(app):
    w = app.shell.window
    w.open_settings("updates")
    D.pump(500)
    sec = w.pages["settings"].updates
    w.pages["settings"].scroll.ensureWidgetVisible(sec, 0, 0)
    D.pump(200)
    return sec


def choose(choice, text: str) -> dict:
    """Pick ``text`` in a dropdown the way a keyboard / screen-reader user does: UIA Invoke opens it, arrow keys
    (Qt key events to the menu) move, UIA reads the highlighted row's name, Enter picks it."""
    r = D.uia("invoke", choice.accessibleName(), control="Button")
    assert r["found"] and r["done"] == "InvokePattern", r
    D.pump(300)
    menu = choice._menu
    assert menu is not None and menu.isVisible(), "the dropdown opened"
    texts = [o[1] for o in choice.options]
    cur = texts.index(choice.label()) if choice.label() in texts else 0
    step = Qt.Key.Key_Down if texts.index(text) > cur else Qt.Key.Key_Up
    for _ in range(abs(texts.index(text) - cur)):
        QTest.keyClick(menu, step)
    D.pump(100)
    seen = D.uia("windows")
    names = [i["name"] for i in seen["items"]]
    assert any(n.startswith(text) for n in names), f"UIA reads the highlighted choice: {names}"
    QTest.keyClick(menu, Qt.Key.Key_Return)
    D.pump(200)
    assert not menu.isVisible()
    return {"uia_menu_names": names}


def click_row(choice, text: str) -> None:
    """Pick ``text`` with a mouse click on its row: a Qt mouse event on VoxFlow's own popup (not an OS click)."""
    assert D.uia("invoke", choice.accessibleName(), control="Button")["done"] == "InvokePattern"
    D.pump(300)
    menu = choice._menu
    i = [it.get("text") for it in menu.items()].index(text) if hasattr(menu, "items") else None
    rect = menu._rects[i]
    pos = QPoint(rect.center().x(), rect.center().y() - menu._scroll)
    QTest.mouseMove(menu, pos)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    D.pump(200)


def closed(modal) -> bool:
    """The dialog was closed (close_modal deletes it later)."""
    import shiboken6
    return not shiboken6.isValid(modal) or not modal.isVisible()


def status(sec) -> str:
    return sec.status_row.sub.text()


def uia_has(name: str, control: str = "") -> dict:
    r = D.uia("find", name, control=control)
    assert r["found"], f"UIA does not see {name!r}"
    return r["element"]


def check_now(app, sec) -> None:
    before = app.env.sc.count(H.API_HOST, "/releases/latest")
    r = D.uia("invoke", "Check now", control="Button")
    assert r["done"] == "InvokePattern", r
    app.wait(lambda: app.env.sc.count(H.API_HOST, "/releases/latest") == before + 1 and app.u.phase == "idle")
    D.pump(300)


# --------------------------------------------------------------------------------------------------- 1. Settings
def test_1_settings_updates_modes_when_time_and_check_now_on_screen(env, quiet):
    app, w = start(env, mode="notify", metered=lambda: False)
    sec = open_updates(app)
    assert quiet, "Settings came forward through the app's own present()"
    assert w.isVisible() and w.windowHandle() is not None
    assert QApplication.platformName() == "windows"
    shots, results = [], []
    mode_btn = uia_has("Updates", "Button")
    assert mode_btn["description"] == "Notify me, I choose", f"UIA reads the current mode: {mode_btn}"

    def pose(result: str):
        sc = env.sc
        sc.api_status, sc.version = (500, "0.3.0") if result == "error" else (200, "0.2.0" if result == "up to date"
                                                                                else "0.3.0")

    for mode, label in (("off", "Off (never check)"), ("notify", "Notify me, I choose"),
                        ("auto", "Install automatically")):
        how = choose(sec.mode, label)
        app.wait(lambda: app.c.get("update_mode") == mode)
        assert uia_has("Updates", "Button")["description"] == label
        for result in ("up to date", "error", "available"):
            pose(result)
            check_now(app, sec)
            text = status(sec)
            uia_has(text)                                    # the status line is readable by UIA, verbatim
            if result == "up to date":
                assert "VoxFlow is up to date." in text, text
            elif result == "error":
                assert "available" not in text and sec.status_row.sub.property("role") in (None, "AMBER") \
                    or "did not work" in text or "GitHub" in text, text
            else:
                assert "VoxFlow 0.3.0 is available." in text, text
                uia_has("See update", "Button")
            if mode == "off":
                assert "Automatic checks are off" in text
            results.append(f"{mode}/{result}: {text!r}")
            if mode == "auto" and result == "available":
                app.wait(lambda: app.u.phase == "idle" and app.u.state.downloaded == "0.3.0", 30000)
        D.pump(300)
        shots.append(D.shot(f"settings-updates-{mode}{TAG}", w))
        assert not D.clipped(sec), D.clipped(sec)
        env.note(f"desktop/settings/{mode}", f"menu via UIA: {how['uia_menu_names']}")

    # auto: both "when" options and the time picker
    assert sec.when_row.isVisible()
    uia_has("Install automatically", "Button")
    choose(sec.when, "At a set time")
    app.wait(lambda: app.c.get("update_install_when") == "time")
    D.pump(200)
    assert sec.at.isVisible()
    assert uia_has("Install time", "Button")["description"] == app.c.get("update_time")
    choose(sec.at, "04:00")
    app.wait(lambda: app.c.get("update_time") == "04:00")
    assert "installed at 04:00" in status(sec), status(sec)
    shots.append(D.shot(f"settings-updates-auto-at-time{TAG}", w))
    assert not D.clipped(sec), D.clipped(sec)
    click_row(sec.when, "When I quit VoxFlow")             # the mouse: a Qt click on the popup's row
    app.wait(lambda: app.c.get("update_install_when") == "quit")
    D.pump(200)
    assert not sec.at.isVisible()
    shots.append(D.shot(f"settings-updates-auto-on-quit{TAG}", w))
    env.note("desktop/settings", " | ".join(results) + f" | shots: {[str(s) for s in shots]}")


# --------------------------------------------------------------------------------------------------- 2. dialog
def tray_dialog(env, app, w, quiet):
    """The automatic check finds 0.3.0 while the window is hidden: the real tray notification. It is clicked through
    UIA when Windows exposes it, else messageClicked is emitted (reported). Returns (modal, how it was opened)."""
    tray, shown = app.shell.tray, app.balloons
    app.wait(lambda: shown, 15000)
    assert tray.icon.isVisible() and QSystemTrayIcon.isSystemTrayAvailable() and QSystemTrayIcon.supportsMessages()
    title, message = shown[0][0], shown[0][1]
    assert title == "VoxFlow" and message.startswith("VoxFlow 0.3.0 is available.")
    assert not w.isVisible(), "the window was hidden: the notification went to the tray"
    code, state = D.notification_state()
    found = {}
    if code == 5:                                          # QUNS_ACCEPTS_NOTIFICATIONS: the shell shows it
        found = D.uia("toast", "VoxFlow 0.3.0 is available", timeout_ms=7000, invoke=False)
        if found.get("found"):
            rect = found["element"].get("rect")
            pic = D.shot_rect(f"tray-notification{TAG}", rect) if rect else None
            found = D.uia("toast", "VoxFlow 0.3.0 is available", timeout_ms=3000, invoke=True)
            found["shot"] = str(pic)
    if found.get("done"):
        how = f"UIA Invoke on the Windows notification ({found['element'].get('name')!r}; shot {found.get('shot')})"
        app.wait(lambda: w._modal is not None, 8000)
    elif code != 5:
        how = f"messageClicked emitted: Windows is not showing notifications now: {state}"
        tray.icon.messageClicked.emit()
    else:
        how = (f"messageClicked emitted ({state}, but no notification was exposed to UIA; windows scanned: "
               f"{found.get('scanned_classes')}; found={found.get('found')})")
        tray.icon.messageClicked.emit()
    app.wait(lambda: w._modal is not None and w._modal.isVisible())
    D.pump(300)
    assert w.isVisible() and quiet, "the click brought Settings forward with the dialog"
    return w._modal, how, shown[0]


@pytest.fixture
def stub_browser(monkeypatch):
    opened = []
    import voxflow.ui.updates as upd
    monkeypatch.setattr(upd.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    return opened


def open_dialog(env, quiet, **kw):
    app, w = start(env, mode="notify", **kw)       # the first automatic check comes 0.3 s after the start
    tray = app.shell.tray
    app.balloons = []
    real = tray.icon.showMessage
    tray.icon.showMessage = lambda *a: (app.balloons.append(a), real(*a))[-1]     # recorded, then really shown
    m, how, balloon = tray_dialog(env, app, w, quiet)
    focus = w.focusWidget()
    assert focus is not None and focus.objectName() == "update_dialog_later", \
        f"the dialog opens on Remind me later, not {focus and focus.objectName()}"
    return app, w, m, how, balloon


def test_2_dialog_from_the_tray_install_now_waits_for_the_dictation(env, quiet):
    app, w, m, how, balloon = open_dialog(env, quiet)
    for name in ("Skip this version", "Remind me later", "Install when I quit", "Install now"):
        uia_has(name, "Button")
    uia_has("Release notes")
    shot = D.shot(f"update-dialog{TAG}", w)
    assert not D.clipped(m), D.clipped(m)
    app.rig.asr.delay_s = 2.0
    app.rig.pause()
    app.rig.press()                                        # a dictation is running
    assert app.c.state == "listening"
    assert D.uia("invoke", "Install now", control="Button")["done"] == "InvokePattern"
    app.wait(lambda: app.u.phase == "waiting", 30000)
    D.pump(300)
    assert "as soon as you finish dictating" in w.pages["settings"].updates.status_row.sub.text()
    D.shot(f"settings-updates-waiting{TAG}", w)
    D.pump(2500)
    assert not app.restarts and not env.record.exists(), "never while dictating"
    app.rig.release()
    app.wait(lambda: app.restarts, 20000)
    assert app.c.state in ("done", "idle") and app.rig.last_entry() is not None
    app.exit(user_quit=False)
    rec = env.ran()
    assert "/RELAUNCH=1" in rec["argv"] and rec["exe"].lower().endswith("voxflow-setup-0.3.0.exe")
    env.note("desktop/dialog/install-now", f"opened by: {how}; balloon {balloon[:2]}; stub argv {rec['argv']}; "
                                           f"shot {shot}")


def test_2_dialog_install_when_i_quit(env, quiet):
    app, w, m, how, _b = open_dialog(env, quiet)
    assert D.uia("invoke", "Install when I quit", control="Button")["done"] == "InvokePattern"
    app.wait(lambda: app.u.state.downloaded == "0.3.0" and app.u.phase == "idle", 30000)
    assert closed(m) and not app.restarts
    app.exit(user_quit=True)
    rec = env.ran()
    assert "/RELAUNCH=0" in rec["argv"]
    env.note("desktop/dialog/on-quit", f"opened by: {how}; stub argv {rec['argv']}")


def test_2_dialog_remind_me_later(env, quiet):
    app, w, m, how, _b = open_dialog(env, quiet)
    assert D.uia("invoke", "Remind me later", control="Button")["done"] == "InvokePattern"
    app.wait(lambda: closed(m))
    assert app.u.state.remind_after > env.clock() and app.u.state.notified == ""
    assert app.files() == [] and not env.record.exists()
    env.note("desktop/dialog/later", f"opened by: {how}; remind_after +{app.u.state.remind_after - env.clock():.0f} s")


def test_2_dialog_skip_this_version(env, quiet):
    app, w, m, how, _b = open_dialog(env, quiet)
    assert D.uia("invoke", "Skip this version", control="Button")["done"] == "InvokePattern"
    app.wait(lambda: closed(m))
    assert app.u.state.skipped == "0.3.0" and app.files() == [] and not env.record.exists()
    D.pump(300)
    assert "You skipped this version." in w.pages["settings"].updates.status_row.sub.text()
    env.note("desktop/dialog/skip", f"opened by: {how}")


def test_2_all_users_dialog_download_opens_the_release_page(env, quiet, stub_exe, stub_browser, monkeypatch):
    pf = env.tmp / "Program Files"
    monkeypatch.setenv("ProgramFiles", str(pf))
    (pf / "VoxFlow").mkdir(parents=True)
    shutil.copy(stub_exe, pf / "VoxFlow" / "VoxFlow.exe")
    (pf / "VoxFlow" / "unins000.exe").write_bytes(b"MZ")
    kind = U.install_kind(pf / "VoxFlow" / "VoxFlow.exe", frozen=True)
    assert kind == "allusers"
    app, w, m, how, _b = open_dialog(env, quiet, kind=kind)
    uia_has("Download", "Button")
    assert D.uia("find", "Install now", control="Button", timeout_ms=500)["found"] is False
    shot = D.shot(f"update-dialog-all-users{TAG}", w)
    assert not D.clipped(m), D.clipped(m)
    assert D.uia("invoke", "Download", control="Button")["done"] == "InvokePattern"
    app.wait(lambda: stub_browser and closed(m))
    assert stub_browser == [f"https://{H.WEB_HOST}/{H.REPO}/releases/tag/v0.3.0"], stub_browser
    assert app.files() == [] and not env.record.exists()
    env.note("desktop/dialog/all-users", f"opened by: {how}; browser stub got {stub_browser}; shot {shot}")


# --------------------------------------------------------------------------------------------------- 3. keyboard
def test_3_keyboard_tab_order_enter_and_esc(env, quiet):
    app, w, m, how, _b = open_dialog(env, quiet)
    order = [w.focusWidget().objectName()]
    for _ in range(5):
        QTest.keyClick(w.focusWidget(), Qt.Key.Key_Tab)
        order.append(w.focusWidget().objectName())
    assert order == ["update_dialog_later", "update_dialog_quit", "update_dialog_now", "update_notes",
                     "update_dialog_skip", "update_dialog_later"], order
    back = [w.focusWidget().objectName()]
    for _ in range(2):
        QTest.keyClick(w.focusWidget(), Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
        back.append(w.focusWidget().objectName())
    assert back == ["update_dialog_later", "update_dialog_skip", "update_notes"], back
    # Enter where no button has the focus (the release notes): the default, Remind me later, not Install now
    QTest.keyClick(w.focusWidget(), Qt.Key.Key_Return)
    app.wait(lambda: closed(m))
    assert app.u.phase == "idle" and app.files() == [] and app.u.state.remind_after > env.clock(), \
        "Enter did not start the install"
    # Esc on a fresh dialog: Remind me later as well
    app.u.state.remind_after = 0
    m2 = w.pages["settings"].open_update()
    D.pump(300)
    assert w.focusWidget().objectName() == "update_dialog_later"
    QTest.keyClick(w.focusWidget(), Qt.Key.Key_Escape)
    app.wait(lambda: closed(m2))
    assert app.u.state.remind_after > env.clock() and app.u.phase == "idle"
    # Enter on a focused button presses that button
    m3 = w.pages["settings"].open_update()
    D.pump(300)
    QTest.keyClick(w.focusWidget(), Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    assert w.focusWidget().objectName() == "update_dialog_skip"
    QTest.keyClick(w.focusWidget(), Qt.Key.Key_Return)
    app.wait(lambda: closed(m3))
    assert app.u.state.skipped == "0.3.0"
    env.note("desktop/keyboard", f"tab {order}; shift+tab {back}; Enter on notes = remind later; Esc = remind later; "
                                 f"Enter on Skip = skip; opened by: {how}")


# --------------------------------------------------------------------------------------------------- 4. human click
CLICK_BANNER = """
##########################################################################
##                                                                      ##
##            CLICK THE VOXFLOW NOTIFICATION NOW                        ##
##                                                                      ##
##   (the banner bottom right; if none, press Win+N and click the       ##
##    "VoxFlow 0.3.0 is available" notification there; not the icon)   ##
##   waiting up to {s} s                                                ##
##########################################################################
"""


@pytest.mark.skipif(os.environ.get("VOXFLOW_HUMAN_CLICK") != "1",
                    reason="manual: a person clicks the real notification (set VOXFLOW_HUMAN_CLICK=1, run with -s)")
def test_4_human_clicks_the_real_notification(env, quiet, capsys):
    """The real update notification from the real tray icon; NOT emitted by the test: a person clicks it (the banner,
    or the entry in the notification centre: Qt reports both as messageClicked) and the update dialog must open."""
    wait_s = int(os.environ.get("VOXFLOW_HUMAN_CLICK_WAIT", "180"))
    app, w = start(env, mode="notify")
    tray = app.shell.tray
    shown, clicks, icon_clicks = [], [], []
    real = tray.icon.showMessage
    tray.icon.showMessage = lambda *a: (shown.append(a), real(*a))[-1]
    tray.icon.messageClicked.connect(lambda: clicks.append(time.monotonic()))
    tray.icon.activated.connect(lambda r: icon_clicks.append(str(r)))
    app.wait(lambda: shown, 15000)
    t0 = time.monotonic()
    assert shown[0][1].startswith("VoxFlow 0.3.0 is available.") and not w.isVisible()
    code, state = D.notification_state()
    with capsys.disabled():
        print(CLICK_BANNER.format(s=wait_s), flush=True)
    found = D.uia("toast", "VoxFlow 0.3.0 is available", timeout_ms=4000) if code == 5 else {}
    pic = None
    if found.get("found") and found["element"].get("rect"):
        pic = D.shot_rect(f"tray-notification-human{TAG}", found["element"]["rect"])
    end = t0 + wait_s
    while not clicks and time.monotonic() < end:
        D.pump(100)
    banner = (f"banner exposed to UIA ({found['element'].get('name')!r}, rect {found['element'].get('rect')}, "
              f"shot {pic})" if found.get("found") else f"banner NOT exposed to UIA ({state})")
    env.note("desktop/human-click", f"{banner}; messageClicked from a person after "
                                    f"{(clicks[0] - t0) if clicks else None} s; tray icon activations {icon_clicks}")
    assert clicks, f"nobody clicked the notification within {wait_s} s ({banner}; icon activations {icon_clicks})"
    app.wait(lambda: w._modal is not None and w._modal.isVisible(), 8000)
    focus = w.focusWidget()
    assert focus is not None and focus.objectName() == "update_dialog_later"
    with capsys.disabled():
        print(f"\nOK: the update dialog opened after a human click ({clicks[0] - t0:.1f} s). {banner}\n", flush=True)
    D.shot(f"update-dialog-human{TAG}", w)


# --------------------------------------------------------------------------------------------------- 6. metered
def test_6_connection_metered_on_this_pc(env):
    value = U.connection_metered()
    env.note("desktop/metered", f"connection_metered() = {value!r} (read only; no network setting was changed)")
    assert value in (True, False, None)


def test_zz_foreground_report(env):
    env.note("desktop/foreground", " | ".join(FG) or "no data")
