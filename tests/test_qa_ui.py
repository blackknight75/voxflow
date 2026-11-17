"""QA pass on the UI (QA-ui.md, UI-01..UI-16): one test (or more) per finding, in-process only."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAccessible, QAccessibleEvent, QKeyEvent
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


# =========================================================================== UI-04 settings search
def _visible_titles(page) -> set[str]:
    return {row.title.text() for sec in page._sections() if not sec.isHidden() for row in sec._rows if not row.isHidden()}


def test_ui04_settings_search_matches_sections_and_everyday_words(demo):
    page = demo.page("settings")
    page.find.setText("hotkey")
    assert {"Dictation", "Command mode", "Paste last transcript", "Undo last paste",
            "Hands-free lock"} <= _visible_titles(page)
    page.find.setText("shortcut")
    assert "Dictation" in _visible_titles(page) and not page.no_match.isVisible()
    page.find.setText("gpu")
    assert {"Speech recognition device", "Language model device", "Speed check"} <= _visible_titles(page)
    page.find.setText("startup")
    assert _visible_titles(page) == {"Launch at login", "Start in the tray"}
    page.find.setText("privacy")                               # a section name shows the whole section
    assert "Keep transcripts for" in _visible_titles(page)
    page.find.setText("model")
    assert {"Speech recognition", "Smart cleanup and Command mode"} <= _visible_titles(page)
    page.find.setText("zzz-nothing")
    assert page.no_match.isVisible() and not _visible_titles(page)
    page.find.setText("")
    assert not page.no_match.isVisible()


# =========================================================================== UI-06 settings tab order
def _tab_stops(win) -> list[str]:
    out, x = [], win.nextInFocusChain()
    while x is not win:
        if x.isVisible() and x.isEnabled() and x.focusPolicy() & Qt.FocusPolicy.TabFocus and x.objectName():
            out.append(x.objectName())
        x = x.nextInFocusChain()
    return out


def test_ui06_search_and_jump_come_first_in_settings(demo):
    page = demo.page("settings")
    stops = _tab_stops(demo.win)
    i = stops.index("settings_search")
    assert stops[i + 1] == "settings_jump" and stops[i + 2] == "settings_hotkey_recorder_change"
    assert stops.index("settings_reset") > i
    page.hide()
    page.show()                                                # shown again: the order is fixed only once
    assert _tab_stops(demo.win) == stops


# =========================================================================== UI-16 duplicates and alignment
def test_ui16_the_version_heads_one_row_only(demo):
    page = demo.page("settings")
    titles = [row.title.text() for sec in page._sections() for row in sec._rows]
    assert sum(t.startswith("VoxFlow ") for t in titles) == 0
    assert f"Version {demo.c.version}" in titles and "About VoxFlow" in titles


def test_ui16_a_hotkey_that_cannot_be_cleared_ends_at_the_edge(show):
    fixed = show(HotkeyRecorder("ctrl+win", name="Dictation hotkey"), 400, 60)
    clearable = show(HotkeyRecorder("", name="Undo hotkey", allow_clear=True), 400, 60)
    QApplication.processEvents()
    assert fixed.btn.geometry().right() == fixed.width() - 1, "no empty x column"
    assert clearable.btn.geometry().right() < clearable.width() - 30, "a clearable one keeps the slot for its x"


# =========================================================================== UI-09 undo a deleted transcript
def test_ui09_deleting_a_transcript_can_be_undone(demo):
    page, c = demo.page("history"), demo.c
    total = c.history.count()
    entry = page.list.items[1]
    page.list.delete_requested.emit(entry)
    assert c.history.count() == total - 1 and demo.toasts() == ["Transcript deleted."]
    toast = demo.win.toasts._toasts[-1]
    assert toast.btn.text() == "Undo"
    toast.btn.click()
    assert c.history.count() == total and c.history.get(entry["id"])["text"] == entry["text"]
    assert c.history.restore(entry) is False, "a second undo adds nothing"
    page.list._tick.stop()


# =========================================================================== UI-12 status chip
def test_ui12_status_chip_wraps_to_three_lines_and_can_be_opened(show, qtbot):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent
    from voxflow.ui.components import StatusChip
    chip = show(StatusChip(), 188, 76)
    pressed = Signals(go=chip.activated)
    chip.set_status({"level": "error", "title": "Microphone problem",
                     "detail": "The microphone “Yeti” stopped sending sound. Check the cable, or choose another "
                               "microphone in Settings."})
    assert 2 <= len(chip._lines) <= 3 and chip._lines[0].startswith("The microphone")
    chip.set_status({"level": "error", "title": "Problem", "detail": " ".join(["word"] * 80)})
    assert len(chip._lines) == 3 and chip._lines[-1].endswith("…"), "the rest is elided on the last line"
    chip.set_status({"level": "ok", "title": "Ready", "detail": "Hold Ctrl + Win."})
    assert chip._lines == ["Hold Ctrl + Win."] and chip._dy() > 0
    chip.set_status({"level": "loading", "title": "Loading", "detail": ""})
    assert chip._lines == [] and chip._dy() == (chip.height() - 30) // 2
    assert chip.focusPolicy() & Qt.FocusPolicy.TabFocus
    i = iface(chip)
    assert i.role() == QAccessible.Role.StatusBar and i.text(QAccessible.Text.Value) == ""
    assert not i.state().selected
    chip.activateWindow()
    chip.clearFocus()
    chip.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert chip.hasFocus() and chip._kbd
    chip.repaint()                                             # paints the focus ring
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Space, Qt.Key.Key_A):
        qtbot.keyClick(chip, key)
    centre = QPointF(chip.width() / 2, chip.height() / 2)
    for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
        QApplication.sendEvent(chip, QMouseEvent(QEvent.Type.MouseButtonRelease, centre, centre, button,
                                                 Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    assert len(pressed.go) == 3, "Enter, Space and a left click"
    chip.clearFocus()
    QApplication.processEvents()
    assert not chip._kbd


def test_ui12_pressing_the_chip_shows_the_whole_status(demo):
    chip = demo.win.status
    for level, kind in (("error", "error"), ("warning", "warning"), ("ok", "info")):
        chip.set_status({"level": level, "title": "Microphone problem", "detail": "Check the cable."})
        chip.activated.emit()
        assert demo.toasts()[-1] == "Microphone problem. Check the cable."
        assert demo.win.toasts._toasts[-1].level == kind
    chip.set_status({"level": "ok", "title": "Ready", "detail": ""})
    chip.activated.emit()
    assert demo.toasts()[-1] == "Ready"


def test_ui12_tray_menu_status_gets_a_second_line_when_needed(show):
    from voxflow.ui.widgets import PopupMenu
    menu = PopupMenu(296)
    short = [{"kind": "status", "text": "Ready", "sub": "Hold Ctrl + Win to dictate."}]
    long = [{"kind": "status", "text": "Microphone problem",
             "sub": "The microphone “Yeti” stopped sending sound. Check the cable, or choose another one."}]
    menu.set_items(short)
    h1 = menu.height()
    menu.set_items(long)
    assert menu.height() == h1 + menu.STATUS_LINE_H and len(menu._status_lines(long[0])) == 2
    show(menu)
    menu.grab()                                                # paints both lines
    menu.close()
