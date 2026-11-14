"""``voxflow.ui.components``: the composite pieces on their own: rows and sections with icons, the stat tile, the
activity chart with pointer and keyboard, toasts and their host, the in-window dialog and the transcript list's
pointer, keyboard and painting corners. No controller is involved."""
from __future__ import annotations

import datetime as dt

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from tests.cov_ui_kit import Signals, flush, hover, pixels, show, unhover  # noqa: F401 - fixture
from tests.helpers import fake_time
from voxflow.ui import components as C
from voxflow.ui import theme as T
from voxflow.ui.components import (BarChart, EmptyState, Modal, NavButton, Row, Section, StatCard, ToastHost,
                                   TranscriptList, WindowButton, confirm)
from voxflow.ui.widgets import Button, LineEdit, ScrollArea, Toggle

LEFT, RIGHT, NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton


def mouse(widget, kind, pos, button=NONE, buttons=NONE) -> None:
    pos = QPointF(pos)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, QPointF(widget.mapToGlobal(pos.toPoint())), button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


def focus_by_keyboard(widget) -> None:
    widget.activateWindow()
    widget.clearFocus()
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert widget.hasFocus()


@pytest.fixture
def host(show):
    return show(QWidget(), 760, 520)


# =========================================================================== formatting
def test_formatting_helpers():
    assert [C.fmt_count(n) for n in (0, 999, 9_999, 10_000, 1_250_000)] == ["0", "999", "9,999", "10.0k", "1.2M"]
    assert [C.fmt_duration(s) for s in (-5, 59, 60, 3599, 3600, 3660)] == ["0 sec", "59 sec", "1 min", "59 min", "1 h",
                                                                           "1 h 1 min"]
    now = dt.datetime(2025, 3, 12, 12, 0).timestamp()
    day = 86400
    assert [C.day_label(now - k * day, now) for k in (0, 1, 3)] == ["Today", "Yesterday", "Sunday"]
    assert C.day_label(now - 10 * day, now) == "March 2" and C.day_label(now - 400 * day, now) == "February 6, 2024"
    assert C.day_label("soon", now) == "Earlier" and C.clock_label(None) == ""
    assert C.clock_label(dt.datetime(2025, 3, 12, 9, 5).timestamp()) == "09:05"
    assert [C.app_label(e) for e in ("Code.exe", "my_cool-app.exe", "", None)] == ["VS Code", "My Cool App", "", ""]


# =========================================================================== rows + sections
def test_row_paints_its_icon_and_a_section_indents_every_row_once_one_has_an_icon(show):
    plain = show(Row("Sound cues", "Soft tones"), 400)
    with_icon = show(Row("Sound cues", "Soft tones", icon="sound"), 400)
    assert with_icon._outer.contentsMargins().left() == plain._outer.contentsMargins().left() + 30
    assert pixels(with_icon) != pixels(plain)
    toggle = Toggle()
    labelled = Row("Sound cues", "Soft tones", toggle)
    assert toggle.parent() is labelled
    assert toggle.accessibleName() == "Sound cues" and toggle.accessibleDescription() == "Soft tones"

    sec = show(Section("Audio", icons=True), 420)
    first = sec.add(Row("No icon here"))
    assert first._outer.contentsMargins().left() == T.S4
    second = sec.add(Row("Microphone", icon="mic"))
    assert second.icon_name == "mic"
    assert first._outer.contentsMargins().left() == second._outer.contentsMargins().left() == T.S4 + 30, \
        "titles line up once any row carries an icon"
    third = sec.add(Row("Volume"))
    assert third._outer.contentsMargins().left() == T.S4 + 30
    assert (first.divider, second.divider, third.divider) == (True, True, False)
    sec.clear()
    assert sec._rows == [] and sec.add(Row("Fresh"))._outer.contentsMargins().left() == T.S4

    quiet = Section("Audio")                                   # icons off (the default): the hint is kept, not shown
    row = quiet.add(Row("Microphone", icon="mic"))
    assert row.icon_name is None and row.icon_hint == "mic"
    row.set_sub("Not connected", "RED")
    assert row.sub.isVisibleTo(row) and row.sub.text() == "Not connected"
    row.set_sub("")
    assert not row.sub.isVisibleTo(row)


def test_stat_card_shrinks_a_long_value_and_handles_a_missing_caption(show):
    card = show(StatCard("text", "Words dictated", name="stat_words"), 150, 116)
    card.set_value("12", "3 today")
    assert card.accessibleName() == "Words dictated: 12" and card.accessibleDescription() == "3 today"
    with_caption = pixels(card)
    card.set_value("12")
    assert pixels(card) != with_caption
    card.set_value("1,234,567,890,123,456")                    # wider than the tile at the headline size
    assert pixels(card) and card.sizeHint().height() == 116


# =========================================================================== bar chart
def _series(n=7, words=(120, 0, 300, 80, 0, 950, 410)):
    today = dt.date.today()
    return [{"date": (today - dt.timedelta(days=n - 1 - i)).isoformat(), "words": words[i % len(words)]}
            for i in range(n)]


def test_bar_chart_hover_and_keyboard_reveal_a_day(show, qtbot):
    chart = show(BarChart(), 520, 200)
    chart.set_series(_series())
    assert chart.accessibleDescription() == "1860 words over the last 7 days"
    chart.clearFocus()
    calm = pixels(chart)
    slot = chart._slot(5)
    mouse(chart, QEvent.Type.MouseMove, slot.center())
    assert chart._hover == 5
    hot = pixels(chart)
    assert hot != calm, "the hovered day shows its value"
    mouse(chart, QEvent.Type.MouseMove, slot.center() + QPointF(1, 0))
    assert chart._hover == 5
    mouse(chart, QEvent.Type.MouseMove, QPoint(2, 50))         # left of the plot
    assert chart._hover == -1
    mouse(chart, QEvent.Type.MouseMove, chart._slot(0).center())
    QApplication.sendEvent(chart, QEvent(QEvent.Type.Leave))
    assert chart._hover == -1 and pixels(chart) == calm

    focus_by_keyboard(chart)
    assert chart._kbd and chart._hover == 6, "keyboard focus starts on today"
    ringed = pixels(chart)
    assert ringed != calm
    qtbot.keyClick(chart, Qt.Key.Key_Left)
    assert chart._hover == 5
    qtbot.keyClick(chart, Qt.Key.Key_Right)
    qtbot.keyClick(chart, Qt.Key.Key_Right)
    assert chart._hover == 6, "stops at the last day"
    qtbot.keyClick(chart, Qt.Key.Key_A)
    assert chart._hover == 6
    QApplication.sendEvent(chart, QEvent(QEvent.Type.Leave))
    assert chart._hover == 6, "the pointer leaving does not take the keyboard selection away"
    chart.clearFocus()
    mouse(chart, QEvent.Type.MouseMove, chart._slot(2).center())
    chart.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert chart.hasFocus() and chart._hover == 2, "focus arriving keeps the day the pointer is on"
    chart.clearFocus()
    QApplication.processEvents()
    assert not chart._kbd and chart._hover == -1


def test_bar_chart_grows_in_when_animated_and_tolerates_odd_data(show, qtbot):
    chart = show(BarChart(), 520, 200)
    chart.set_series(_series(), animate=True)
    assert chart._anim.state() == chart._anim.State.Running
    qtbot.waitUntil(lambda: chart._grow == 1.0 and chart._anim.state() == chart._anim.State.Stopped, timeout=4000)
    T.apply_theme("dark", True)
    chart._grow = 0.3
    chart.set_series(_series(), animate=True)
    assert chart._grow == 1.0 and chart._anim.state() == chart._anim.State.Stopped, "reduced motion: no growing"

    empty = pixels(BarChart())
    chart.set_series([])
    qtbot.keyClick(chart, Qt.Key.Key_Right)                    # nothing to select
    assert chart._hover == -1
    mouse(chart, QEvent.Type.MouseMove, QPoint(200, 60))
    assert chart._hover == -1 and pixels(chart) != empty
    chart.set_series([{"date": "not-a-date", "words": 40}, {"words": 15}])
    chart._set_hover(0)
    assert pixels(chart), "a broken date paints an empty label instead of raising"
    chart.set_series(_series(30))                              # a month: every fifth day is labelled
    chart._set_hover(29)
    assert pixels(chart)


# =========================================================================== empty state
def test_empty_state_button_runs_the_action_or_nothing(show):
    clicks: list = []
    state = show(EmptyState("book", "Your dictionary is empty", "Add names.", "Add a word", lambda: clicks.append(1)),
                 500, 400)
    state.button.click()
    assert clicks == [1] and state.button.isVisible() and state.accessibleName() == "Your dictionary is empty"
    empty = pixels(state)
    state.show_state("error", "warning", "Could not load", "Disk says no.")
    assert not state.button.isVisible() and state.kind == "error" and pixels(state) != empty
    state.button.click()                                       # no action: clicking the hidden button is harmless
    assert clicks == [1]
    state.resize(900, 400)
    QApplication.processEvents()
    assert state.sub.width() == 440, "the explanation keeps a readable measure however wide the page is"
    assert state.sub.height() == state.sub.heightForWidth(440)


# =========================================================================== toasts
def test_toast_pauses_while_hovered_and_goes_away_on_click(host):
    toasts = ToastHost(host)
    toasts.show_toast("info", "Exported 3 transcripts.")
    toast = toasts._toasts[0]
    toast._anim.setCurrentTime(toast._anim.duration())
    assert toast._life.isActive() and toasts.isVisible() and toast._appear == 1.0
    calm = pixels(toast)
    hover(toast)
    assert not toast._life.isActive(), "reading a toast must not race its timer"
    assert pixels(toast) != calm, "the dismiss cross shows while hovered"
    unhover(toast)
    assert toast._life.isActive() and toast._life.interval() == 2200 and pixels(toast) == calm
    seen = Signals(closed=toast.closed)
    mouse(toast, QEvent.Type.MouseButtonRelease, QPoint(20, 20), LEFT)
    assert toasts.messages() == [] and not toasts.isVisible() and not toast._life.isActive()
    toast.dismiss()                                            # already closing: nothing more happens
    assert len(seen.closed) == 1
    QApplication.sendEvent(toast, QEvent(QEvent.Type.Leave))
    assert not toast._life.isActive(), "a closing toast is not revived by the pointer leaving"
    flush()


def test_toast_actions_levels_and_the_stack_limit(host):
    T.apply_theme("dark", True)
    toasts = ToastHost(host)
    toasts.left_inset = 200
    ran: list = []
    toasts.show_toast("success", "Removed “Kubernetes”.", "Undo", lambda: ran.append("undo"))
    toast = toasts._toasts[0]
    assert toast._appear == 1.0 and toast._anim.state() == toast._anim.State.Stopped, "reduced motion: no fade"
    assert toast.btn.text() == "Undo" and toast._life.interval() == 6500
    assert toasts.attach_action("Removed “Kubernetes”.", "Put back", lambda: ran.append("back")) is True
    assert toast.btn.text() == "Put back" and toast._life.interval() == 8000
    assert toasts.attach_action("No such toast", "x", lambda: None) is False
    hover(toast)
    assert pixels(toast), "a toast with a button shows no dismiss cross"
    toast.btn.click()
    flush()
    assert ran == ["back"] and toasts.messages() == []

    toasts.show_toast("warning", "Careful.")
    plain = toasts._toasts[0]
    assert toasts.attach_action("Careful.", "Open settings", None) is True
    assert plain.btn is not None and plain.btn.isVisibleTo(plain)
    plain.btn.click()                                          # no callback: the toast just closes
    flush()
    assert toasts.messages() == []

    toasts.show_toast("no-such-level", "Odd level.")
    assert toasts._toasts[0].level == "info"
    toasts.show_toast("info", "Odd level.", ms=9000)           # the same message again only extends it
    assert toasts.messages() == ["Odd level."] and toasts._toasts[0]._life.interval() == 9000
    toasts.show_toast("info", "")
    for i in range(4):
        toasts.show_toast("error", f"Problem {i}")
    assert toasts.messages() == ["Problem 1", "Problem 2", "Problem 3"], "at most three; the oldest goes first"
    assert toasts.geometry().left() >= 200, "the stack never covers the sidebar"
    host.resize(900, 600)
    QApplication.processEvents()
    assert toasts.geometry().bottom() <= 600 and toasts.geometry().center().x() == pytest.approx(200 + 350, abs=2)
    stray = toasts._toasts[0]
    toasts._drop(stray)
    toasts._drop(stray)                                        # dropping it twice is harmless
    assert toasts.messages() == ["Problem 2", "Problem 3"]
    for t in list(toasts._toasts):
        t.dismiss()
    flush()
    assert not toasts.isVisible()


def test_toast_host_without_a_parent_does_not_lay_out(host):
    toasts = ToastHost(host)
    toasts.show_toast("info", "Hello.")
    before = toasts.geometry()
    toasts.setParent(None)
    toasts._relayout()
    assert toasts.geometry().size() == before.size()
    toasts._toasts[0].dismiss()
    toasts.deleteLater()
    flush()


# =========================================================================== modal
def test_modal_keeps_focus_inside_handles_keys_and_follows_its_host(host, qtbot):
    outside = LineEdit("", "outside", host)
    outside.show()
    outside.activateWindow()
    outside.setFocus(Qt.FocusReason.OtherFocusReason)
    assert outside.hasFocus()
    saved: list = []
    m = Modal(host, "Add a word", "VoxFlow will always write this exactly as you type it.", name="editor")
    seen = Signals(closed=m.closed)
    field = LineEdit("", "word")
    m.body.addWidget(field)
    extra = Button("Help", "ghost")
    ok = m.add_buttons("Save", lambda: saved.append(field.text()), extra=extra)
    assert ok.objectName() == "editor_ok" and m.btn_cancel.objectName() == "editor_cancel" and extra.parent() is m.card
    m.open(field)
    m._anim.setCurrentTime(m._anim.duration())
    assert m.isVisible() and field.hasFocus() and m.geometry() == host.rect()

    order = [field, extra, m.btn_cancel, ok]
    for expected in order[1:] + [field]:
        assert m.focusNextPrevChild(True) and expected.hasFocus(), "Tab walks the dialog and wraps"
    assert m.focusNextPrevChild(False) and ok.hasFocus(), "Shift+Tab goes backwards"
    m.setFocus()
    assert m.focusNextPrevChild(True) and field.hasFocus(), "from the dialog itself Tab goes to the first control"
    m.setFocus()
    assert m.focusNextPrevChild(False) and ok.hasFocus()

    field.setText("Kubernetes")
    qtbot.keyClick(m, Qt.Key.Key_A)                            # an ordinary key is not swallowed or acted on
    assert saved == [] and m.isVisible()
    QApplication.sendEvent(m, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier))
    assert saved == [], "Shift+Enter is for line breaks, not for saving"
    qtbot.keyClick(m, Qt.Key.Key_Return)
    assert saved == ["Kubernetes"]
    m.show_error("That word is already there.")
    assert m.error.isVisibleTo(m) and m.error.text() == "That word is already there."
    mouse(m, QEvent.Type.MouseButtonPress, m.card.geometry().center(), LEFT, LEFT)
    assert m.isVisible(), "a click inside the card does not dismiss"
    host.resize(900, 640)
    QApplication.processEvents()
    assert m.geometry() == host.rect() and m.card.geometry().center().x() == pytest.approx(450, abs=1)
    mouse(m, QEvent.Type.MouseButtonPress, QPoint(5, 5), LEFT, LEFT)
    assert not m.isVisible() and len(seen.closed) == 1 and outside.hasFocus(), "focus returns to where it was"
    flush()


def test_modal_without_controls_and_with_a_vanished_previous_focus(host, qtbot):
    T.apply_theme("light", True)
    outside = LineEdit("", "outside", host)
    outside.show()
    outside.activateWindow()
    outside.setFocus(Qt.FocusReason.OtherFocusReason)
    bare = Modal(host, "Just so you know")
    bare.open()
    assert bare._appear == 1.0 and bare._anim.state() == bare._anim.State.Stopped and bare.hasFocus()
    assert bare.focusNextPrevChild(True) is True and bare.hasFocus(), "nothing to tab to: focus stays put"
    qtbot.keyClick(bare, Qt.Key.Key_Return)                    # no primary button: Enter does nothing
    assert bare.isVisible() and pixels(bare)
    shiboken6.delete(outside)                                  # the widget that had focus is gone meanwhile
    qtbot.keyClick(bare, Qt.Key.Key_Escape)
    assert not bare.isVisible()

    hidden = LineEdit("", "hidden", host)
    hidden.show()
    hidden.setFocus(Qt.FocusReason.OtherFocusReason)
    second = Modal(host, "Second")
    hidden.hide()
    second.open()
    second.close_modal()
    assert not hidden.hasFocus(), "a hidden widget is not given the focus back"
    flush()


def test_confirm_runs_the_action_after_closing(host):
    done: list = []
    m = confirm(host, "Clear all history?", "This cannot be undone.", "Delete everything",
                lambda: done.append(m.isVisible()), name="really")
    m.open(m.btn_cancel)
    assert m.btn_primary.kind == "danger" and m.btn_cancel.hasFocus()
    m.btn_primary.click()
    assert done == [False], "the dialog is closed before the action runs"
    calm = confirm(host, "Reset?", "Back to defaults.", "Reset", lambda: None, danger=False)
    assert calm.btn_primary.kind == "primary"
    flush()


# =========================================================================== chrome
def test_nav_and_window_buttons_react_to_the_pointer(show):
    T.apply_theme("dark", True)
    nav = show(NavButton("book", "Dictionary"), 190)
    calm = pixels(nav)
    hover(nav)
    assert pixels(nav) != calm
    unhover(nav)
    nav.set_badge(3)
    assert nav.accessibleDescription() == "3 new"
    three = pixels(nav)
    nav.set_badge(250)
    assert pixels(nav) not in (calm, three)
    nav.set_badge(0)
    assert nav.accessibleDescription() == "" and pixels(nav) == calm
    nav.setChecked(True)
    T.apply_theme("light", True)
    assert pixels(nav)

    pictures = {}
    for kind in ("min", "max", "restore", "close"):
        b = show(WindowButton(kind))
        idle = pixels(b)
        hover(b)
        pictures[kind] = pixels(b)
        assert pictures[kind] != idle
    assert len(set(pictures.values())) == 4
    b.set_kind("restore")
    assert b.toolTip() == b.accessibleName() == "Restore"


# =========================================================================== transcript list
def _entries(n=3, **extra):
    base = dt.datetime.now().timestamp()
    return [{"id": f"e{i}", "ts": base - i * 60, "text": f"Transcript number {i}.", "raw": f"transcript number {i}",
             "app_exe": "slack.exe", "words": 3, **extra} for i in range(n)]


def test_transcript_list_pointer_hover_tooltips_and_clicks(show):
    lst = show(TranscriptList(), 560, 400)
    lst.set_items(_entries(2, rules_only=True) + ["not a dict"])
    assert len(lst.items) == 2
    seen = Signals(copy=lst.copy_requested, paste=lst.paste_requested, delete=lst.delete_requested)
    calm = pixels(lst)
    card = lst._card_rect(0)
    copy_rect, paste_rect, trash_rect = lst._action_rects(card)
    mouse(lst, QEvent.Type.MouseMove, copy_rect.center())
    assert lst._hover == (0, 0) and lst.toolTip() == "Copy" and lst.cursor().shape() == Qt.CursorShape.PointingHandCursor
    hot = pixels(lst)
    assert hot != calm
    mouse(lst, QEvent.Type.MouseMove, copy_rect.center() + QPoint(1, 1))
    assert lst._hover == (0, 0), "still on the same button: nothing to update"
    mouse(lst, QEvent.Type.MouseMove, trash_rect.center())
    assert lst.toolTip() == "Delete" and pixels(lst) not in (calm, hot), "the delete button warns in red"
    mouse(lst, QEvent.Type.MouseMove, lst._tags[0].center())
    assert lst.toolTip() == lst.RULES_ONLY_TIP and lst.cursor().shape() == Qt.CursorShape.ArrowCursor
    mouse(lst, QEvent.Type.MouseMove, QPoint(card.left() + 30, card.bottom() - 6))
    assert lst.toolTip() == "" and lst._hover == (0, -1)
    mouse(lst, QEvent.Type.MouseMove, QPoint(10, 4))           # on the day heading
    assert lst._hover == (-1, -1) and lst._hit(QPoint(10, 4)) == (-1, -1)
    mouse(lst, QEvent.Type.MouseMove, card.center())
    QApplication.sendEvent(lst, QEvent(QEvent.Type.Leave))
    assert lst._hover == (-1, -1)

    mouse(lst, QEvent.Type.MouseButtonPress, QPoint(10, 4), LEFT, LEFT)
    assert lst.selected() is None, "a click on a heading selects nothing"
    mouse(lst, QEvent.Type.MouseButtonPress, lst._card_rect(1).center(), LEFT, LEFT)
    assert lst.selected()["id"] == "e1" and "basic clean-up only" in lst.accessibleDescription()
    mouse(lst, QEvent.Type.MouseButtonRelease, lst._card_rect(1).center(), LEFT)
    mouse(lst, QEvent.Type.MouseButtonRelease, QPoint(10, 4), LEFT)
    mouse(lst, QEvent.Type.MouseButtonRelease, paste_rect.center(), RIGHT)
    assert seen.copy == seen.paste == seen.delete == [], "only a left click on a button acts"
    mouse(lst, QEvent.Type.MouseButtonRelease, paste_rect.center(), LEFT)
    mouse(lst, QEvent.Type.MouseButtonRelease, trash_rect.center(), LEFT)
    assert [e[0]["id"] for e in seen.paste] == ["e0"] and [e[0]["id"] for e in seen.delete] == ["e0"]
    assert list(lst._flash) == [("e0", "paste")], "deleting is not confirmed with a tick"
    lst._tick.stop()


def test_transcript_list_confirmation_tick_expires(show, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(C, "time", fake_time(time=lambda: clock["t"]))
    lst = show(TranscriptList(show_delete=False), 560, 300)
    lst.set_items(_entries(2))
    assert [a[0] for a in lst.actions] == ["copy", "paste"]
    calm = pixels(lst)
    lst._run(0, "copy")
    assert lst._tick.isActive() and pixels(lst) != calm, "a green tick replaces the icon"
    lst._run(99, "copy")                                       # no such row
    assert list(lst._flash) == [("e0", "copy")]
    clock["t"] += 1.0
    lst._run(1, "paste")
    lst._expire_flash()
    assert len(lst._flash) == 2 and lst._tick.isActive()
    clock["t"] += 0.5
    lst._expire_flash()
    assert list(lst._flash) == [("e1", "paste")] and lst._tick.isActive()
    clock["t"] += 2.0
    lst._expire_flash()
    assert lst._flash == {} and not lst._tick.isActive() and pixels(lst) == calm


def test_transcript_list_keyboard(show, qtbot):
    scroll = show(ScrollArea(), 560, 260)
    inner = QWidget()
    box = QVBoxLayout(inner)
    lst = TranscriptList()
    box.addWidget(lst)
    scroll.setWidget(inner)
    lst.set_items(_entries(12))
    QApplication.processEvents()
    seen = Signals(copy=lst.copy_requested, paste=lst.paste_requested, delete=lst.delete_requested)
    focus_by_keyboard(lst)
    assert lst._kbd and lst.selected()["id"] == "e0", "keyboard focus selects the first card"
    ringed = pixels(lst)
    qtbot.keyClick(lst, Qt.Key.Key_End)
    at_end = scroll.verticalScrollBar().value()
    assert lst.selected()["id"] == "e11" and at_end > 0, "the selection is scrolled into view"
    assert pixels(lst) != ringed
    qtbot.keyClick(lst, Qt.Key.Key_Home)
    assert lst.selected()["id"] == "e0" and scroll.verticalScrollBar().value() < at_end / 4
    qtbot.keyClick(lst, Qt.Key.Key_P)
    qtbot.keyClick(lst, Qt.Key.Key_Return)
    qtbot.keyClick(lst, Qt.Key.Key_Delete)
    qtbot.keyClick(lst, Qt.Key.Key_X)                          # not a list key
    assert [e[0]["id"] for e in seen.paste] == ["e0"] and len(seen.copy) == 1 and len(seen.delete) == 1
    lst._tick.stop()
    lst._sel = 3
    focus_by_keyboard(lst)
    assert lst.selected()["id"] == "e3", "an existing selection survives focus coming back"
    lst.set_items(_entries(12)[2:])
    assert lst.selected()["id"] == "e3", "and survives the list being refreshed"
    lst.set_items([])
    assert lst.selected() is None and lst.minimumHeight() == 0 and lst.accessibleDescription() != ""
    qtbot.keyClick(lst, Qt.Key.Key_Down)                       # an empty list ignores the keys
    assert lst.selected() is None

    no_delete = show(TranscriptList(show_delete=False), 400, 200)
    no_delete.set_items(_entries(1))
    gone = Signals(delete=no_delete.delete_requested)
    focus_by_keyboard(no_delete)
    qtbot.keyClick(no_delete, Qt.Key.Key_Down)                 # no scroll area around it: nothing to reveal
    qtbot.keyClick(no_delete, Qt.Key.Key_Delete)
    assert gone.delete == [] and no_delete._card_rect(5) is None


def test_transcript_list_painting_variants(show):
    lst = show(TranscriptList(group_by_day=False), 560, 420)
    base = dt.datetime.now().timestamp()
    lst.set_items([
        {"id": "a", "ts": base, "text": "Plain dictation.", "raw": "plain dictation um", "app_exe": "slack.exe", "words": 2},
        {"id": "b", "ts": base, "text": "No app and no word count.", "app_exe": "", "words": 0},
        {"id": "c", "ts": base, "text": "- bullets", "raw": "turn into bullets", "mode": "command", "rules_only": True,
         "app_exe": "chrome.exe", "words": 2},
    ])
    assert [r[0] for r in lst._rows] == ["card", "card", "card"], "no day headings when not grouped"
    clean = pixels(lst)
    assert 2 in lst._tags and 0 not in lst._tags
    lst.set_show_raw(True)
    assert lst.body_text(lst.items[0]) == "plain dictation um" and lst.body_text(lst.items[1]) == "No app and no word count."
    assert pixels(lst) != clean
    lst.set_show_raw(False)
    lst.resize(170, 600)                                       # too narrow for the tags: they are left out, not clipped
    QApplication.processEvents()
    lst._tags.clear()
    pixels(lst)
    assert lst._tags == {}
    lst.resize(560, 420)
    QApplication.processEvents()
    lst._tags.clear()
    lst.repaint(QRect(0, 0, 560, 20))                          # a partial repaint skips the cards outside it
    assert 2 not in lst._tags
    assert lst.sizeHint().height() == lst.minimumHeight() > 0
