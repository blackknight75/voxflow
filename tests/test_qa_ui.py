"""QA pass on the UI (QA-ui.md, UI-01..UI-16): one test (or more) per finding, in-process only."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QAccessibleEvent
from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget

from tests.cov_ui_kit import Signals, demo, make_demo, show  # noqa: F401 - fixtures
from voxflow.ui import a11y
from voxflow.ui.components import NavButton, Row, TranscriptList
from voxflow.ui.widgets import Button, HotkeyRecorder, Segmented


def iface(widget):
    return QAccessible.queryAccessibleInterface(widget)


@pytest.fixture
def spoken(monkeypatch):
    """Accessibility events the widgets send, as if a screen reader were listening."""
    sent: list = []
    monkeypatch.setattr(QAccessible, "isActive", staticmethod(lambda: True))
    monkeypatch.setattr(a11y, "_send", sent.append)
    return sent


def said(sent) -> list[str]:
    return [ev.message() for ev in sent if ev.type() == QAccessible.Event.Announcement]


# =========================================================================== voxflow.ui.a11y
def test_a11y_factory_only_serves_widgets_that_ask_for_it(qapp, spoken):
    a11y.install()
    a11y.install()                                             # idempotent
    plain = QWidget()
    assert iface(plain).role() == QAccessible.Role.Client
    assert a11y._factory("QObject", qapp) is None, "a non-widget never gets one"
    a11y.announce(plain, "")
    assert spoken == [], "nothing to say, nothing sent"
    a11y.announce(plain, "Saved")
    a11y.announce(plain, "Failed", assertive=True)
    assert said(spoken) == ["Saved", "Failed"]
    assert [ev.politeness() for ev in spoken] == [QAccessible.AnnouncementPoliteness.Polite,
                                                 QAccessible.AnnouncementPoliteness.Assertive]
    plain.deleteLater()


def test_a11y_sends_nothing_while_no_screen_reader_listens(qapp, monkeypatch):
    sent: list = []
    monkeypatch.setattr(QAccessible, "isActive", staticmethod(lambda: False))
    monkeypatch.setattr(a11y, "_send", sent.append)
    w = QWidget()
    a11y.announce(w, "Hello")
    a11y.value_changed(w, "Dark")
    assert sent == []
    w.deleteLater()


def test_a11y_send_reaches_qt(qapp, monkeypatch):
    got: list = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", staticmethod(got.append))
    w = QWidget()
    ev = QAccessibleEvent(w, QAccessible.Event.Focus)
    a11y._send(ev)
    assert got == [ev]
    w.deleteLater()


# =========================================================================== UI-03 segmented controls
def test_ui03_segmented_reads_as_a_combo_box_with_its_value(show, spoken):
    seg = show(Segmented([("system", "System"), ("light", "Light"), ("dark", "Dark")], "dark", name="Theme"))
    i = iface(seg)
    assert i.role() == QAccessible.Role.ComboBox and i.text(QAccessible.Text.Name) == "Theme"
    assert i.text(QAccessible.Text.Value) == "Dark"
    assert seg.accessibleDescription().startswith("Option 3 of 3: System, Light, Dark.")
    seg.setFocus()
    QApplication.processEvents()
    seg.keyPressEvent(_key(Qt.Key.Key_Left))
    assert i.text(QAccessible.Text.Value) == "Light" and "Option 2 of 3" in seg.accessibleDescription()
    changes = [ev for ev in spoken if ev.type() == QAccessible.Event.ValueChanged]
    assert [ev.value() for ev in changes] == ["Light"], "the arrow key is no longer silent"
    assert not i.state().selected, "the state is Qt's own"
    empty = show(Segmented([], name="Nothing"))
    assert iface(empty).text(QAccessible.Text.Value) == "" and empty.accessibleDescription() == ""


def _key(key):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


# =========================================================================== UI-05 button names follow the label
def test_ui05_a_button_is_announced_by_its_current_label(show):
    b = show(Button("Measure now"))
    b.setText("Measure again")
    assert b.accessibleName() == "Measure again"
    b.flash("Copied")
    assert b.accessibleName() == "Copied"
    b._fa.stop()
    b._end_flash()
    assert b.text() == b.accessibleName() == "Measure again"
    named = show(Button("Go"))
    named.setAccessibleName("Start the speed check")
    named.setText("Stop")
    assert named.accessibleName() == "Start the speed check", "an explicit name is kept"


def test_ui05_settings_speed_check_button_reads_its_label(demo):
    btn = demo.find("settings_speed_measure") if demo.find_all("settings_speed_measure") else None
    for b in demo.win.findChildren(Button):
        assert not b._auto_name or b.accessibleName() == b.text(), b.objectName()
    assert btn is None or btn.accessibleName() == btn.text()


# =========================================================================== UI-07 names with context
def test_ui07_row_buttons_carry_the_row_as_description(show):
    holder = QWidget()
    lay = QHBoxLayout(holder)
    a, b = Button("Export…"), Button("Import…")
    lay.addWidget(a)
    lay.addWidget(b)
    row = show(Row("Backup", "Your dictionary, snippets and styles", holder))
    assert a.accessibleDescription() == b.accessibleDescription() == "Backup. Your dictionary, snippets and styles"
    one = Button("Open")
    row2 = show(Row("Logs", "", one))
    assert one.accessibleName() == "Open" and one.accessibleDescription() == "Logs"
    kept = Button("Test")
    kept.setAccessibleDescription("Plays a short sound")
    icon_only = Button("")
    rows = [show(Row("Sounds", "Cues", kept)), show(Row("Clear", "Wipe it", icon_only))]
    assert kept.accessibleDescription() == "Plays a short sound" and icon_only.accessibleName() == "Clear"
    assert row.isVisible() and row2.isVisible() and all(r.isVisible() for r in rows)


def test_ui07_hotkey_change_button_says_the_current_combo(show):
    rec = show(HotkeyRecorder("ctrl+win", name="Dictation hotkey", allow_clear=True))
    assert rec.btn.accessibleName() == "Change dictation hotkey (now Ctrl + Win)"
    rec.set_combo("ctrl+alt+space")
    assert rec.btn.accessibleName() == "Change dictation hotkey (now Ctrl + Alt + Space)"
    rec.toggle_capture()
    assert rec.btn.accessibleName() == "Cancel recording the dictation hotkey"
    rec.set_combo("ctrl+shift")                                # arrives while recording: the name waits
    assert rec.btn.accessibleName() == "Cancel recording the dictation hotkey"
    rec.toggle_capture()
    assert rec.btn.accessibleName() == "Change dictation hotkey (now Ctrl + Shift)"
    rec._clear()
    assert rec.btn.accessibleName() == "Change dictation hotkey (now not set)"


# =========================================================================== UI-08 transcript list
def _entries(n):
    return [{"id": f"e{i}", "text": f"Entry number {i}", "raw": f"entry number {i}", "ts": 1_700_000_000 - i * 60,
             "words": 3} for i in range(n)]


def test_ui08_transcript_list_is_a_list_that_speaks_the_chosen_entry(show, qtbot, spoken):
    lst = show(TranscriptList(), 500, 400)
    lst.set_items(_entries(3))
    i = iface(lst)
    assert i.role() == QAccessible.Role.List and i.text(QAccessible.Text.Value) == ""
    assert "Enter or C copies" in lst.accessibleDescription() and "Delete" in lst.toolTip()
    lst.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    qtbot.keyClick(lst, Qt.Key.Key_Home)
    qtbot.keyClick(lst, Qt.Key.Key_Down)
    assert "Entry number 1" in i.text(QAccessible.Text.Value)
    assert said(spoken)[-1].endswith("Entry number 1. 2 of 3")
    n = len(spoken)
    qtbot.keyClick(lst, Qt.Key.Key_End)
    qtbot.keyClick(lst, Qt.Key.Key_End)                        # already there: said once
    assert len(said(spoken[n:])) == 1 and not i.state().selected
    lst.clearFocus()
    lst._select(0)
    assert len(said(spoken[n:])) == 1, "no focus, nothing said"


# =========================================================================== UI-11 sidebar items are tabs
def test_ui11_nav_items_are_page_tabs_not_check_boxes(show):
    home, hist = NavButton("home", "Home"), NavButton("history", "History")
    show(home)
    show(hist)
    home.setChecked(True)
    i, j = iface(home), iface(hist)
    assert i.role() == j.role() == QAccessible.Role.PageTab
    assert i.text(QAccessible.Text.Value) == ""
    si, sj = i.state(), j.state()
    assert si.selected and si.selectable and not si.checked and not si.checkable
    assert not sj.selected and sj.selectable


def test_ui11_the_window_sidebar_uses_tabs(demo):
    navs = demo.win.findChildren(NavButton)
    assert navs and all(iface(n).role() == QAccessible.Role.PageTab for n in navs)
    assert sum(bool(iface(n).state().selected) for n in navs) == 1
    Signals()


# =========================================================================== UI-10 Flow bar announcements
def test_ui10_flow_bar_results_are_announced(qapp, spoken):
    from voxflow.ui.overlay import HINT_SEP, OverlayPill
    bar = OverlayPill(source=None, show_idle=False)
    try:
        bar.set_state("listening")
        bar.set_state("processing")
        assert said(spoken) == [], "nothing is spoken while the microphone is open"
        bar.set_state("done")
        bar.set_state("done")                                  # same message again: said once
        bar.set_state("error", "No speech heard. Check your microphone")
        bar.set_state("done", f"Window changed. Result copied: press Ctrl+V{HINT_SEP}Paste it yourself")
        bar.set_state("loading", "Downloading speech model", progress=0.1)
        bar.set_state("loading", "Downloading speech model", progress=0.2)
        bar.set_state("loading", "", progress=0.3)
        bar.set_state("idle")
        bar.set_state("loading")
        assert said(spoken) == ["Pasted", "No speech heard. Check your microphone",
                                "Window changed. Result copied: press Ctrl+V. Paste it yourself",
                                "Downloading speech model", "Getting ready…"]
        polite = QAccessible.AnnouncementPoliteness
        kinds = [ev.politeness() for ev in spoken if ev.type() == QAccessible.Event.Announcement]
        assert kinds[1] == polite.Assertive and kinds[0] == kinds[2] == polite.Polite
    finally:
        bar._timer.stop()
        bar._follow.stop()
        bar.deleteLater()
