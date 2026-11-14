"""``voxflow.ui.widgets``: the controls on their own (no window, no controller): pointer hover, press and
keyboard paths, reduced motion, disabled looks, the popup menu with the mouse, hotkey recording corner cases and
the progress / level animations on a scripted clock."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from tests.cov_ui_kit import Signals, hover, pixels, show, unhover  # noqa: F401 - fixture
from tests.helpers import fake_time
from voxflow.ui import theme as T
from voxflow.ui import widgets as W
from voxflow.ui.widgets import (Button, ChoiceButton, HotkeyRecorder, IconButton, KeyCaps, LevelMeter, PopupMenu,
                                ProgressBar, ScrollArea, Segmented, Slider, Spinner, Toggle)

LEFT, RIGHT, NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton


def mouse(widget, kind, pos, button=NONE, buttons=NONE) -> None:
    pos = QPointF(pos)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, QPointF(widget.mapToGlobal(pos.toPoint())), button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


def key(widget, kind, code, auto_repeat=False) -> bool:
    return QApplication.sendEvent(widget, QKeyEvent(kind, code, Qt.KeyboardModifier.NoModifier, "", auto_repeat))


def focus_by_keyboard(widget) -> None:
    widget.activateWindow()
    widget.clearFocus()
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert widget.hasFocus()


# =========================================================================== window helper
def test_bring_to_front_raises_the_window_on_a_real_desktop_only(show, monkeypatch):
    win = show(QWidget())
    win.hide()
    calls: list[str] = []
    win.raise_ = lambda: calls.append("raise")
    win.activateWindow = lambda: calls.append("activate")
    W.bring_to_front(win)
    assert win.isVisible() and calls == ["activate"], "the offscreen platform cannot raise"
    monkeypatch.setattr(W.QGuiApplication, "platformName", staticmethod(lambda: "windows"))
    W.bring_to_front(win)
    assert calls == ["activate", "raise", "activate"]


# =========================================================================== hover base + buttons
@pytest.mark.parametrize("kind", ["primary", "ghost", "secondary", "danger"])
def test_button_hover_animates_in_and_out_and_changes_the_picture(show, qtbot, kind):
    b = show(Button("Save", kind))
    plain = pixels(b)
    hover(b)
    qtbot.waitUntil(lambda: b._hover == 1.0, timeout=3000)
    hot = pixels(b)
    assert hot != plain
    b.setDown(True)
    assert pixels(b) != hot, "a pressed button is darker"
    b.setDown(False)
    unhover(b)
    qtbot.waitUntil(lambda: b._hover == 0.0, timeout=3000)
    assert pixels(b) == plain


def test_hover_jumps_with_reduced_motion_and_is_ignored_while_disabled(show):
    T.apply_theme("dark", True)
    b = show(Button("Save"))
    hover(b)
    assert b._hover == 1.0 and b._anim.state() == b._anim.State.Stopped, "no animation with reduced motion"
    unhover(b)
    assert b._hover == 0.0
    hover(b)
    b.setEnabled(False)
    assert b._hover == 0.0 and b.cursor().shape() == Qt.CursorShape.ArrowCursor
    hover(b)
    assert b._hover == 0.0, "a disabled button does not light up"
    b.setEnabled(True)
    assert b.cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_button_without_text_is_square_and_a_pressed_light_button_still_differs(show):
    icon_only = show(Button("", icon="plus"))
    assert icon_only.sizeHint().width() == icon_only.height() == icon_only.minimumSizeHint().height()
    assert Button("Wide label").sizeHint().width() > icon_only.sizeHint().width()
    assert Button("Save", icon="check").sizeHint().width() == Button("Save").sizeHint().width() + 22
    T.apply_theme("light", False)
    b = show(Button("Save"))
    up = pixels(b)
    b.setDown(True)
    assert pixels(b) != up


def test_button_flash_shows_the_confirmation_and_restores_the_label(show):
    b = show(Button("Copy"))
    plain = pixels(b)
    b._end_flash()                                             # nothing flashed: nothing to restore
    assert b.text() == "Copy"
    b.flash("Copied")
    b.flash("Copied!")                                         # a second flash keeps the ORIGINAL label to go back to
    assert b.text() == "Copied!" and b._flash == 1.0 and pixels(b) != plain
    b._fa.setCurrentTime(500)
    assert 0.0 < b._flash < 1.0
    b._fa.setCurrentTime(b._fa.duration())
    assert b.text() == "Copy" and b._flash == 0.0 and b._base_text is None
    b._fa.stop()


def test_icon_button_states(show, qtbot):
    b = show(IconButton("copy", "Copy"))
    assert b.accessibleName() == b.toolTip() == "Copy"
    plain = pixels(b)
    b.confirm()
    assert b._ok == 1.0
    confirmed = pixels(b)
    assert confirmed != plain, "the icon turns into a green check"
    b._oa.setCurrentTime(1000)                                 # near the end the normal icon is back
    assert b._ok < 0.25 and pixels(b) == plain
    b._oa.stop()
    hover(b)
    qtbot.waitUntil(lambda: b._hover == 1.0, timeout=3000)
    hot = pixels(b)
    assert hot != plain
    b.setDown(True)
    assert pixels(b) != hot
    b.setDown(False)
    b.setEnabled(False)
    assert pixels(b) not in (plain, hot), "disabled is dimmed"
    danger = show(IconButton("trash", "Delete", danger=True))
    T.apply_theme("dark", True)
    calm = pixels(danger)
    hover(danger)
    assert danger._hover == 1.0 and pixels(danger) != calm


def test_toggle_slides_or_jumps_and_dims_when_disabled(show, qtbot):
    t = show(Toggle())
    assert t.accessibleName() == "" and Toggle(name="Sound cues").accessibleName() == "Sound cues"
    off = pixels(t)
    t.setChecked(True)
    qtbot.waitUntil(lambda: t._pos == 1.0, timeout=3000)
    on = pixels(t)
    assert on != off
    T.apply_theme("dark", True)
    t.setChecked(False)
    assert t._pos == 0.0 and t._slide.state() == t._slide.State.Stopped, "reduced motion: no slide"
    assert pixels(t) == off
    t.setEnabled(False)
    assert pixels(t) != off
    t.set_checked_silent(True)
    assert t.isChecked() and t._pos == 1.0


# =========================================================================== segmented
def test_segmented_mouse_keyboard_and_focus(show, qtbot):
    seg = show(Segmented([("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")], "a", name="Letters"), 300)
    seen = Signals(changed=seg.changed)
    third = seg.width() / 3
    plain = pixels(seg)
    mouse(seg, QEvent.Type.MouseMove, QPoint(int(third * 2.5), 10))
    assert seg._hover == 2 and pixels(seg) != plain, "the hovered label brightens"
    mouse(seg, QEvent.Type.MouseMove, QPoint(int(third * 2.6), 12))
    assert seg._hover == 2
    QApplication.sendEvent(seg, QEvent(QEvent.Type.Leave))
    assert seg._hover == -1
    mouse(seg, QEvent.Type.MouseButtonPress, QPoint(int(third * 1.5), 10), RIGHT, RIGHT)
    assert seg.value == "a" and seen.changed == [], "only the left button chooses"
    mouse(seg, QEvent.Type.MouseButtonPress, QPoint(int(third * 1.5), 10), LEFT, LEFT)
    assert seg.value == "b" and seen.changed == [("b",)]
    qtbot.waitUntil(lambda: seg._pos == 1.0, timeout=3000)
    mouse(seg, QEvent.Type.MouseButtonPress, QPoint(5000, 10), LEFT, LEFT)       # beyond the last segment
    assert seg.value == "c"
    qtbot.waitUntil(lambda: seg._pos == 2.0, timeout=3000)

    T.apply_theme("dark", True)
    for code, value in ((Qt.Key.Key_Right, "c"), (Qt.Key.Key_Left, "b"), (Qt.Key.Key_Up, "a"), (Qt.Key.Key_Up, "a"),
                        (Qt.Key.Key_Down, "b")):
        qtbot.keyClick(seg, code)
        assert seg.value == value and seg._pos == float(seg._index())
    assert [v for (v,) in seen.changed] == ["b", "c", "b", "a", "b"], "keys at the end of the track change nothing"
    qtbot.keyClick(seg, Qt.Key.Key_X)
    assert seg.value == "b"

    seg.clearFocus()
    unfocused = pixels(seg)
    focus_by_keyboard(seg)
    assert seg._kbd and pixels(seg) != unfocused, "keyboard focus draws the ring"
    seg.clearFocus()
    QApplication.processEvents()
    assert not seg._kbd and pixels(seg) == unfocused


def test_segmented_with_an_unknown_value_or_no_options(show, qtbot):
    seg = show(Segmented([("a", "Alpha"), ("b", "Beta")], "zzz"))
    assert seg._index() == 0 and seg._pos == 0.0, "an unknown value shows the first option"
    empty = show(Segmented([], None))
    seen = Signals(changed=empty.changed)
    mouse(empty, QEvent.Type.MouseButtonPress, QPoint(10, 10), LEFT, LEFT)
    qtbot.keyClick(empty, Qt.Key.Key_Right)
    assert seen.changed == [] and empty.sizeHint().width() >= 60
    empty.setEnabled(False)
    assert pixels(empty)


# =========================================================================== slider
def test_slider_jumps_drags_and_shows_keyboard_focus(show):
    s = show(Slider(0, 100, 0, name="Volume"), 220)
    seen = Signals(released=s.sliderReleased)
    at_zero = pixels(s)
    mouse(s, QEvent.Type.MouseButtonPress, QPoint(110, 14), RIGHT, RIGHT)
    assert s.value() == 0 and not s.isSliderDown(), "the right button does nothing"
    mouse(s, QEvent.Type.MouseButtonPress, QPoint(110, 14), LEFT, LEFT)
    assert s.value() == 50 and s.isSliderDown() and pixels(s) != at_zero
    s.setMouseTracking(True)                                   # so a move without a button reaches the slider at all
    mouse(s, QEvent.Type.MouseMove, QPoint(160, 14), NONE, NONE)
    assert s.value() == 50, "moving without the button held is not a drag"
    mouse(s, QEvent.Type.MouseMove, QPoint(160, 14), NONE, LEFT)
    assert s.value() == 75
    mouse(s, QEvent.Type.MouseMove, QPoint(9999, 14), NONE, LEFT)
    assert s.value() == 100
    mouse(s, QEvent.Type.MouseButtonRelease, QPoint(160, 14), LEFT, NONE)
    assert not s.isSliderDown() and len(seen.released) == 1, "one release, one signal (it plays the preview tone)"
    mouse(s, QEvent.Type.MouseButtonRelease, QPoint(160, 14), RIGHT, NONE)       # a release that had no press
    assert len(seen.released) == 2
    unfocused = pixels(s)
    focus_by_keyboard(s)
    assert s._kbd and pixels(s) != unfocused
    s.clearFocus()
    QApplication.processEvents()
    assert not s._kbd and pixels(s) == unfocused
    s.setEnabled(False)
    assert pixels(s) != unfocused


# =========================================================================== key caps + hotkey recorder
def test_keycaps_show_the_combo_a_hint_or_not_set(show):
    caps = show(KeyCaps("ctrl+win"))
    combo = pixels(caps)
    width = caps.sizeHint().width()
    caps.set_combo("ctrl+win", "Press the new combo…")
    caps.resize(caps.sizeHint())
    assert caps.sizeHint().width() != width and pixels(caps) != combo
    caps.set_combo("")
    assert caps._labels() == [] and caps.sizeHint().width() < width
    T.apply_theme("light", False)
    caps.set_combo("alt+shift+z")
    caps.resize(caps.sizeHint())
    assert caps._labels() == ["Alt", "Shift", "Z"] and pixels(caps)


def test_qt_keys_map_to_hotkey_tokens():
    assert W.qt_key_token(Qt.Key.Key_F1) == "f1" and W.qt_key_token(Qt.Key.Key_F24) == "f24"
    assert W.qt_key_token(Qt.Key.Key_A) == "a" and W.qt_key_token(Qt.Key.Key_7) == "7"
    assert W.qt_key_token(Qt.Key.Key_Meta) == "win" and W.qt_key_token(Qt.Key.Key_AltGr) == "alt"
    assert W.qt_key_token(Qt.Key.Key_MediaPlay) is None
    assert W.combo_from_tokens(["k", "win", "ctrl", "k"]) == "ctrl+win+k"
    assert W.pretty_hotkey("ctrl+esc") == "Ctrl + Esc" and W.pretty_hotkey("") == ""


def test_hotkey_recorder_records_a_combo_from_qt_keys(show):
    rec = show(HotkeyRecorder("ctrl+win", name="Dictation hotkey"))
    seen = Signals(changed=rec.changed, requested=rec.capture_requested, cancelled=rec.capture_cancelled)
    press, release = QEvent.Type.KeyPress, QEvent.Type.KeyRelease
    assert rec.eventFilter(rec.btn, QKeyEvent(press, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)) is False, \
        "keys are left alone while nothing is being recorded"
    rec.captured("ctrl+k")
    assert rec.combo == "ctrl+win" and seen.changed == [], "a late result from the hook is ignored"

    rec.toggle_capture()
    assert rec.capturing and rec.btn.text() == "Cancel" and rec.caps.hint == "Press the new combo…"
    assert len(seen.requested) == 1
    rec.set_combo("alt+win")                                   # a config change arrives while recording
    assert rec.combo == "alt+win" and rec.caps.hint, "the prompt stays up"
    assert rec.eventFilter(rec, QKeyEvent(press, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)) is False
    assert rec.eventFilter(rec.btn, QEvent(QEvent.Type.Paint)) is False

    assert key(rec.btn, press, Qt.Key.Key_Control)
    assert key(rec.btn, press, Qt.Key.Key_Control, auto_repeat=True)
    assert rec._held == ["ctrl"] and rec.caps.combo == "ctrl"
    key(rec.btn, press, Qt.Key.Key_Control)                    # a second Ctrl key: still one token
    key(rec.btn, press, Qt.Key.Key_MediaPlay)                  # not a hotkey key: swallowed
    key(rec.btn, release, Qt.Key.Key_K)                        # released without having been pressed
    assert rec._held == ["ctrl"] and rec._seen == ["ctrl"] and rec.capturing
    key(rec.btn, press, Qt.Key.Key_K)
    assert rec.caps.combo == "ctrl+k"
    key(rec.btn, release, Qt.Key.Key_K)
    assert rec.capturing, "still holding Ctrl"
    key(rec.btn, release, Qt.Key.Key_Control)
    assert not rec.capturing and rec.combo == "ctrl+k" and seen.changed == [("ctrl+k",)]
    assert rec.btn.text() == "Change" and rec.caps.hint == "" and seen.cancelled == []


def test_hotkey_recorder_validation_escape_and_hook_mode(show):
    rec = show(HotkeyRecorder("ctrl+win", validator=lambda c: "Already used." if c == "ctrl+k" else None,
                              allow_clear=True))
    rec.setObjectName("rec")
    assert rec.btn.objectName() == "rec_change" and rec.btn_clear.objectName() == "rec_clear"
    seen = Signals(changed=rec.changed, cancelled=rec.capture_cancelled)
    rec.toggle_capture()
    assert not rec.btn_clear.isVisible()
    rec.captured("ctrl+k")
    assert rec.combo == "ctrl+win" and rec.msg.isVisible() and rec.msg.text() == "Already used."
    assert rec._msg_timer.isActive() and seen.changed == []
    rec.toggle_capture()
    assert not rec.msg.isVisible(), "starting again clears the old complaint"
    key(rec.btn, QEvent.Type.KeyPress, Qt.Key.Key_Escape)
    assert not rec.capturing and len(seen.cancelled) == 1
    rec.use_qt_keys = False                                    # the global hook delivers the keys
    rec.toggle_capture()
    assert key(rec.btn, QEvent.Type.KeyPress, Qt.Key.Key_J) and rec._seen == []
    ev = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier)
    ev.ignore()
    assert rec.eventFilter(rec.btn, ev) is True and ev.isAccepted(), "shortcuts must not fire while recording"
    rec.captured("ctrl+j")
    assert rec.combo == "ctrl+j" and seen.changed == [("ctrl+j",)] and rec.btn_clear.isVisible()
    rec.toggle_capture()
    rec.toggle_capture()
    assert len(seen.cancelled) == 2 and rec.combo == "ctrl+j"
    rec.btn_clear.click()
    assert rec.combo == "" and seen.changed[-1] == ("",) and not rec.btn_clear.isVisible()
    rec.show_message("Saved", "GREEN", 60_000)
    assert rec.msg.text() == "Saved"
    rec._msg_timer.stop()


# =========================================================================== popup menu
def _menu(show, items, width=240) -> PopupMenu:
    m = PopupMenu(width)
    m.set_items(items)
    show(m)
    m.popup_at(QPoint(400, 400))
    m._anim.setCurrentTime(m._anim.duration())               # skip the fade-in: the pictures below compare states
    m._anim.stop()
    QApplication.processEvents()
    return m


def test_popup_menu_with_the_mouse(show):
    picked: list = []
    items = [{"kind": "header", "text": "VoxFlow", "sub": "v1"},
             {"kind": "action", "text": "Open", "icon": "window", "on_click": lambda: picked.append("open")},
             {"kind": "toggle", "text": "Sounds", "checked": False, "on_click": lambda on: picked.append(("sounds", on))},
             {"kind": "action", "text": "Stay", "keep_open": True, "on_click": lambda: picked.append("stay")},
             {"kind": "action", "text": "Off", "enabled": False, "on_click": lambda: picked.append("off")},
             {"kind": "action", "text": "No callback"}]
    m = _menu(show, items)
    seen = Signals(closed=m.closed)
    centre = [r.center() for r in m._rects]
    plain = pixels(m)
    mouse(m, QEvent.Type.MouseMove, centre[1])
    assert m._hover == 1 and m.cursor().shape() == Qt.CursorShape.PointingHandCursor and pixels(m) != plain
    mouse(m, QEvent.Type.MouseMove, centre[0])
    assert m._hover == -1 and m.cursor().shape() == Qt.CursorShape.ArrowCursor, "the heading is not clickable"
    mouse(m, QEvent.Type.MouseMove, centre[4])
    assert m._hover == -1, "a disabled row is not hoverable"
    mouse(m, QEvent.Type.MouseMove, centre[2])
    QApplication.sendEvent(m, QEvent(QEvent.Type.Leave))
    assert m._hover == -1

    mouse(m, QEvent.Type.MouseButtonRelease, centre[2], LEFT)
    assert picked == [("sounds", True)] and items[2]["checked"] is True and m.isVisible(), "a toggle keeps the menu open"
    mouse(m, QEvent.Type.MouseButtonRelease, centre[3], LEFT)
    assert picked[-1] == "stay" and m.isVisible()
    mouse(m, QEvent.Type.MouseButtonRelease, centre[4], LEFT)
    mouse(m, QEvent.Type.MouseButtonRelease, centre[0], LEFT)
    assert len(picked) == 2 and m.isVisible(), "disabled rows and headings do nothing"
    mouse(m, QEvent.Type.MouseButtonRelease, centre[5], LEFT)
    assert not m.isVisible() and len(seen.closed) == 1 and len(picked) == 2
    m.popup_at(QPoint(400, 400))
    mouse(m, QEvent.Type.MouseButtonRelease, centre[1], LEFT)
    assert picked[-1] == "open" and not m.isVisible()


def test_popup_menu_scrolls_with_the_wheel_and_ignores_it_when_everything_fits(show):
    m = _menu(show, [{"kind": "choice", "text": f"Row {i}", "on_click": lambda: None} for i in range(80)])
    assert m._content_h > m.height()

    def wheel(delta: int) -> None:
        pos = QPointF(20, 20)
        QApplication.sendEvent(m, QWheelEvent(pos, QPointF(m.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, delta),
                                              NONE, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))

    top = pixels(m)
    wheel(-240)
    assert m._scroll == 120 and pixels(m) != top
    mouse(m, QEvent.Type.MouseMove, QPoint(60, 60))
    assert m._hover == m._index_at(QPoint(60, 60)) > 2, "rows are hit-tested at their scrolled position"
    wheel(-10 ** 6)
    assert m._scroll == m._content_h - m.height()
    wheel(10 ** 6)
    assert m._scroll == 0
    m._ensure_visible(999)                                     # not a row: nothing moves
    m._ensure_visible(-1)
    assert m._scroll == 0
    m.close()
    short = _menu(show, [{"kind": "action", "text": "Only", "on_click": lambda: None}])
    wheel_target = short
    QApplication.sendEvent(wheel_target, QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -240), NONE,
                                                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    assert short._scroll == 0
    short.close()


def test_popup_menu_keyboard_corners_and_reduced_motion(show, qtbot):
    T.apply_theme("dark", True)
    info_only = _menu(show, [{"kind": "info", "text": "Nothing to choose from"}, {"kind": "separator"}])
    assert info_only._appear == 1.0 and info_only._anim.state() == info_only._anim.State.Stopped
    qtbot.keyClick(info_only, Qt.Key.Key_Down)
    assert info_only._hover == -1, "no selectable row"
    qtbot.keyClick(info_only, Qt.Key.Key_Return)
    assert info_only.isVisible(), "Enter with nothing selected does nothing"
    qtbot.keyClick(info_only, Qt.Key.Key_A)
    assert info_only.isVisible()
    qtbot.keyClick(info_only, Qt.Key.Key_Escape)
    assert not info_only.isVisible()

    flipped: list = []
    m = _menu(show, [{"kind": "toggle", "text": "Pause", "checked": True, "on_click": flipped.append},
                     {"kind": "status", "text": "Ready", "sub": "Hold Ctrl + Win", "dot": T.GREEN}])
    qtbot.keyClick(m, Qt.Key.Key_Up)                           # from "nothing selected" Up goes to the last usable row
    assert m._hover == 0
    qtbot.keyClick(m, Qt.Key.Key_Space)
    assert flipped == [False] and m.isVisible()
    m.close()


# =========================================================================== dropdown
def test_choice_button_opens_with_arrow_keys_and_reuses_its_menu(show, qtbot):
    c = show(ChoiceButton(name="Microphone"))
    assert c.label() == "System default" and c.sizeHint().width() == 150
    c.set_options([(None, "System default"), ("usb", "USB microphone with a rather long name")], "usb")
    assert c.label().startswith("USB") and c.sizeHint().width() > 150
    seen = Signals(chosen=c.chosen)
    qtbot.keyClick(c, Qt.Key.Key_Down)
    first = c._menu
    assert first is not None and first.isVisible() and first._hover == 1, "the current choice is highlighted"
    first.close()
    qtbot.keyClick(c, Qt.Key.Key_Up)
    assert c._menu is first and first.isVisible(), "one menu object, reused"
    qtbot.keyClick(first, Qt.Key.Key_Home)
    qtbot.keyClick(first, Qt.Key.Key_Return)
    assert seen.chosen == [(None,)] and c.value is None and not first.isVisible()
    qtbot.keyClick(c, Qt.Key.Key_A)
    assert not first.isVisible(), "other keys do not open it"
    qtbot.keyClick(c, Qt.Key.Key_Return)                       # Enter activates like a click
    assert first.isVisible()
    first.close()
    c.set_value("usb")
    enabled = pixels(c)
    c.setEnabled(False)
    assert pixels(c) != enabled
    empty = show(ChoiceButton())
    menu = empty.open_menu()
    assert [it["kind"] for it in menu.items()] == ["info"]
    menu.close()


# =========================================================================== meters + progress
def test_level_meter_follows_its_source_and_survives_a_broken_one(show, monkeypatch):
    clock = {"t": 50.0}
    monkeypatch.setattr(W, "time", fake_time(perf_counter=lambda: clock["t"]))
    meter = show(LevelMeter(lambda: 0.4), 200)
    assert meter._timer.isActive()
    silent = pixels(meter)
    clock["t"] += 0.05
    meter._tick()
    assert 0.5 < meter._level < 0.88 and meter._peak == meter._level and pixels(meter) != silent
    loud = meter._level

    def broken():
        raise RuntimeError("stream died")

    meter.set_source(broken)
    clock["t"] += 0.05
    meter._tick()
    assert meter._level < loud and meter._peak > meter._level, "a failing source reads as silence; the peak decays slowly"
    meter.set_source(None)
    clock["t"] += 5
    meter._tick()
    assert meter._level < 0.4 * loud
    meter.set_level(2.0)
    assert meter._level == 1.0 == meter._peak
    meter.hide()
    assert not meter._timer.isActive()


def test_spinner_spins_only_with_motion_allowed(show):
    T.apply_theme("dark", True)
    still = show(Spinner())
    assert not still._timer.isActive() and pixels(still) == pixels(still)
    T.apply_theme("dark", False)
    moving = show(Spinner())
    assert moving._timer.isActive()
    moving.hide()
    assert not moving._timer.isActive()


def test_progress_bar_eases_towards_its_value(show, monkeypatch):
    clock = {"t": 10.0}
    monkeypatch.setattr(W, "time", fake_time(perf_counter=lambda: clock["t"]))
    bar = show(ProgressBar(), 200)
    empty = pixels(bar)
    bar._tick()
    assert bar._shown == 0.0, "already there: nothing to animate"
    bar.set_value(0.8)
    assert bar.accessibleDescription() == "80 percent" and bar._shown == 0.0
    clock["t"] += 0.05
    bar._tick()
    part = bar._shown
    assert 0.0 < part < 0.8 and pixels(bar) != empty
    for _ in range(200):
        clock["t"] += 0.05
        bar._tick()
    assert bar._shown == pytest.approx(0.8, abs=0.002)
    bar.set_value(7)
    assert bar.value == 1.0
    bar.set_value(None)
    assert bar.accessibleDescription() == "Working"
    first = pixels(bar)
    clock["t"] += 0.4
    bar._tick()
    assert pixels(bar) != first, "the indeterminate block travels"


def test_progress_bar_with_reduced_motion_jumps_and_stands_still(show, monkeypatch):
    clock = {"t": 10.0}
    monkeypatch.setattr(W, "time", fake_time(perf_counter=lambda: clock["t"]))
    T.apply_theme("dark", True)
    bar = show(ProgressBar(), 200)
    bar.set_value(0.5)
    assert bar._shown == 0.5
    half = pixels(bar)
    bar.value = 0.9                                            # e.g. set while motion was still allowed
    clock["t"] += 0.02
    bar._tick()
    assert bar._shown == 0.9 and pixels(bar) != half
    bar.set_value(None)
    still = pixels(bar)
    clock["t"] += 0.4
    bar._tick()
    assert pixels(bar) == still, "no travelling block with reduced motion"
    bar.hide()
    assert not bar._timer.isActive()


def test_scroll_area_with_and_without_content(show):
    bare = show(ScrollArea())
    assert bare.widget() is None and "background: transparent" in bare.styleSheet()
    inner = QWidget()
    inner.setAutoFillBackground(True)
    filled = show(ScrollArea(inner))
    assert filled.widget() is inner and not inner.autoFillBackground()
