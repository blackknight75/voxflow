"""``voxflow.ui.tray``: the tray icon and its menu, on the in-memory demo controller. The operating system's
notification call is replaced by a recorder (the offscreen platform has no system tray)."""
from __future__ import annotations

import time

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QSystemTrayIcon

from tests.cov_ui_kit import Signals, demo, grab_bytes, make_demo  # noqa: F401 - fixtures
from voxflow.ui import tray as tray_mod


def by_text(items, text) -> dict:
    return next(it for it in items if it.get("text") == text)


@pytest.fixture
def notifications(monkeypatch):
    """The messages handed to the OS, as (title, message, icon kind, ms)."""
    def record(tray, visible=True, supported=True) -> list:
        shown: list = []
        tray.icon.isVisible = lambda: visible
        tray.icon.showMessage = lambda *a: shown.append(a)
        monkeypatch.setattr(QSystemTrayIcon, "supportsMessages", staticmethod(lambda: supported))
        return shown
    return record


# =========================================================================== icon
def test_every_look_has_its_own_cached_icon(qapp):
    icons = {look: tray_mod.make_icon(look) for look in tray_mod.ICON_STATES}
    pictures = {look: grab_bytes(ic.pixmap(32, 32)) for look, ic in icons.items()}
    assert len(set(pictures.values())) == len(tray_mod.ICON_STATES) >= 7
    assert all(tray_mod.make_icon(look) is ic for look, ic in icons.items()), "rendered once, then reused"
    assert tray_mod.make_icon("no-such-look") is tray_mod.make_icon("ready")
    assert sorted(s.width() for s in icons["error"].availableSizes()) == [16, 20, 24, 32, 48, 64]
    assert tray_mod._short("Microphone (Razer Seiren V2 X)") == "Microphone (Razer S…" and tray_mod._short("USB") == "USB"


def test_the_icon_and_tooltip_follow_the_controller(demo):
    tray, c = demo.shell.tray, demo.c
    assert tray.look == "ready" and tray.icon.toolTip() == "VoxFlow: Ready\nHold Ctrl + Win to dictate"
    for change, look in ((lambda: c.demo_set_state("listening"), "listening"),
                         (lambda: c.demo_set_state("locked", "command"), "command"),
                         (lambda: c.demo_set_state("processing"), "processing"),
                         (lambda: c.demo_set_state("idle"), "ready"),
                         (lambda: c.demo_set_model("asr", "loading", "Warming up"), "loading"),
                         (lambda: c.demo_set_model("asr", "failed", "Out of memory"), "error"),
                         (lambda: c.demo_set_model("asr", "ready"), "ready"),
                         (lambda: c.set_paused(True), "paused")):
        change()
        assert tray.look == tray.current_look() == look
    assert tray.icon.toolTip() == "VoxFlow: Paused\nDictation hotkeys are off until you resume."
    c.set_paused(False)
    c.status = lambda: {"level": "warning", "title": "Microphone problem"}
    tray.refresh()
    assert tray.look == "warning" and tray.icon.toolTip() == "VoxFlow: Microphone problem", "no detail, no second line"
    c.status = lambda: None
    tray.refresh()
    assert tray.look == "ready" and tray.icon.toolTip() == "VoxFlow: "


# =========================================================================== activation + menu
def test_clicks_on_the_icon_open_the_window_or_the_menu(demo):
    tray = demo.shell.tray
    seen = Signals(show=tray.show_window)
    reason = QSystemTrayIcon.ActivationReason
    tray._activated(reason.Trigger)
    tray._activated(reason.DoubleClick)
    assert seen.show == [("",), ("",)] and not tray.menu.isVisible()
    tray._activated(reason.Unknown)
    assert len(seen.show) == 2 and not tray.menu.isVisible()
    tray._activated(reason.Context)
    assert tray.menu.isVisible() and tray._page == "main"
    tray.menu.close()
    tray._activated(reason.MiddleClick)
    assert tray.menu.isVisible()
    tray.hide()
    assert not tray.menu.isVisible()


def test_main_menu_rows_act_on_the_controller(demo):
    tray, c = demo.shell.tray, demo.c
    seen = Signals(show=tray.show_window, quit=tray.quit_requested)
    menu = tray.show_menu(QPoint(500, 500))
    items = menu.items()
    assert [it["kind"] for it in items[:3]] == ["header", "status", "separator"]
    assert items[0]["sub"] == f"Local dictation  ·  v{c.version}" and items[1]["text"] == "Ready"
    assert by_text(items, "Paste last transcript")["sub"] == "Alt + Shift + Z"
    assert by_text(items, "Microphone")["sub"] == "System default  ›"

    by_text(items, "Smart cleanup")["on_click"](False)
    assert c.get("cleanup_enabled") is False
    assert by_text(menu.items(), "Smart cleanup")["checked"] is False, "the open menu is rebuilt from the new state"
    by_text(menu.items(), "Sound cues")["on_click"](False)
    assert c.get("sounds_enabled") is False
    by_text(menu.items(), "Open VoxFlow")["on_click"]()
    by_text(menu.items(), "Settings")["on_click"]()
    by_text(menu.items(), "Quit VoxFlow")["on_click"]()
    assert seen.show == [("home",), ("settings",)] and len(seen.quit) == 1
    notices = Signals(notice=c.notice)
    by_text(menu.items(), "Paste last transcript")["on_click"]()
    assert notices.notice == [], "there is a transcript to paste"
    c.history.clear()
    c.set("paste_last_hotkey", "")
    tray.menu.set_items(tray.build_items())
    paste = by_text(menu.items(), "Paste last transcript")
    assert paste["enabled"] is False and paste["sub"] == ""
    menu.close()
    c.set("cleanup_enabled", True)                              # menu closed: nothing to rebuild, nothing breaks
    assert not menu.isVisible()


def test_pause_for_an_hour_hands_the_deadline_to_the_controller(demo):
    tray, c = demo.shell.tray, demo.c
    menu = tray.show_menu(QPoint(500, 500))
    by_text(menu.items(), "Pause for 1 hour")["on_click"]()
    assert c.paused and c.paused_until is not None and 3590 < c.paused_until - time.time() <= 3600
    assert by_text(menu.items(), "Pause for 1 hour")["enabled"] is False
    assert by_text(menu.items(), "Pause dictation")["checked"] is True
    assert by_text(menu.items(), c.status()["title"])["text"].startswith("Paused until "), "the menu says when"
    assert tray.icon.toolTip().startswith("VoxFlow: Paused until ")
    by_text(menu.items(), "Pause dictation")["on_click"](False)
    assert not c.paused and c.paused_until is None
    tray.pause_for_hour()
    by_text(menu.items(), "Pause dictation")["on_click"](True)
    assert c.paused and c.paused_until is None, "pausing by hand replaces the timed pause"
    menu.close()


def test_microphone_submenu(demo, notifications):
    tray, c = demo.shell.tray, demo.c
    refreshed = Signals(mics=c.microphones_changed)
    menu = tray.show_menu(QPoint(500, 500))
    by_text(menu.items(), "Microphone")["on_click"]()
    assert tray._page == "mics" and menu.isVisible() and len(refreshed.mics) == 1
    items = menu.items()
    assert items[0]["text"] == "Back" and items[1]["kind"] == "separator"
    choices = [it for it in items if it["kind"] == "choice"]
    assert [it["checked"] for it in choices] == [True, False, False]
    assert choices[1]["text"] == choices[1]["tip"] == "Microphone (Razer Seiren V2 X)"
    choices[2]["on_click"]()
    assert c.get("mic_device") == "Headset (WH-1000XM5)"
    assert [it["checked"] for it in menu.items() if it["kind"] == "choice"] == [False, False, True]
    menu.items()[0]["on_click"]()                              # Back
    assert tray._page == "main" and by_text(menu.items(), "Microphone")["sub"] == "Headset (WH-1000XM5)  ›"
    menu.close()

    def unplugged():
        raise RuntimeError("audio service not running")

    c.refresh_microphones = unplugged                          # must not keep the menu from opening
    c._mics = []
    menu = tray.show_menu(QPoint(500, 500), page="mics")
    assert [it["kind"] for it in menu.items()] == ["action", "separator", "info"]
    assert menu.items()[2]["text"] == "No microphones found"
    tray._goto("main")
    assert by_text(menu.items(), "Microphone")["sub"] == "Headset (WH-1000XM5)  ›", "a device that is gone keeps its name"
    tray._goto("mics")
    assert tray._page == "mics"
    c.config.mic_device = None
    assert tray._mic_name() == "System default"

    shown = notifications(tray)

    def refuse(key, value):
        raise ValueError("That microphone is not connected.")

    c.set = refuse
    tray._set("mic_device", "Ghost")
    assert [(a[1], a[2]) for a in shown] == [("That microphone is not connected.", QSystemTrayIcon.MessageIcon.Critical)]
    menu.close()


# =========================================================================== notifications
def test_notifications_go_to_the_os_only_when_the_icon_can_show_them(demo, notifications):
    tray = demo.shell.tray
    shown = notifications(tray)
    tray.notify("error", "The speech model failed.")
    tray.notify("warning", "No speech heard.", "open_settings:microphone")
    tray.notify("info", "Nothing to undo.")
    kinds = QSystemTrayIcon.MessageIcon
    assert shown == [("VoxFlow", "The speech model failed.", kinds.Critical, 6000),
                     ("VoxFlow", "No speech heard.", kinds.Warning, 6000),
                     ("VoxFlow", "Nothing to undo.", kinds.Information, 6000)]
    assert tray._pending_action == "", "a notification without a next step clears the previous one"
    for visible, supported in ((False, True), (True, False)):
        quiet = notifications(tray, visible, supported)
        tray.notify("error", "Not shown.", "retry_save")
        assert quiet == [] and tray._pending_action == "retry_save"


def test_clicking_a_notification_runs_its_next_step_once(demo):
    tray = demo.shell.tray
    seen = Signals(action=tray.action_requested)
    tray._message_clicked()
    assert seen.action == [], "no next step: a click does nothing"
    tray.set_pending_action("open_settings:models")
    tray.icon.messageClicked.emit()
    tray.icon.messageClicked.emit()
    assert seen.action == [("open_settings:models",)]
