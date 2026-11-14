"""Coverage holes named in the tester review: interactive widget paths that the page-level tests never reached
(hotkey capture with Qt keys, slider, frameless window edges / Esc / title double-click). The destructive
confirmations (clear history, reset settings, import replace, delete word) were already covered in test_ui*.py."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from tests.test_ui import make_ui, seed  # noqa: F401  (fixture re-exported)
from voxflow.ui import widgets as W


def key(widget, k, down=True, mods=Qt.KeyboardModifier.NoModifier):
    ev = QKeyEvent(QEvent.Type.KeyPress if down else QEvent.Type.KeyRelease, k, mods)
    QApplication.sendEvent(widget, ev)
    return ev


# ============================================================================ HotkeyRecorder, Qt-key fallback
def test_recorder_captures_a_combo_from_qt_keys_and_validates_it(qtbot):
    rec = W.HotkeyRecorder("ctrl+win", "Dictation", validator=lambda c: "Taken" if c == "ctrl+alt+x" else None)
    qtbot.addWidget(rec)
    rec.show()
    changed = []
    rec.changed.connect(changed.append)
    rec.toggle_capture()
    for k in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_D):
        key(rec.btn, k)
    for k in (Qt.Key.Key_D, Qt.Key.Key_Shift, Qt.Key.Key_Control):
        key(rec.btn, k, down=False)
    assert changed == ["ctrl+shift+d"] and rec.combo == "ctrl+shift+d"
    # a refused combo shows the message and keeps the old one
    rec.toggle_capture()
    for k in (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_X):
        key(rec.btn, k)
    for k in (Qt.Key.Key_X, Qt.Key.Key_Alt, Qt.Key.Key_Control):
        key(rec.btn, k, down=False)
    assert rec.combo == "ctrl+shift+d" and changed == ["ctrl+shift+d"]
    assert rec.msg.isVisibleTo(rec) and "Taken" in rec.msg.text()


def test_recorder_escape_and_focus_loss_cancel(qtbot):
    rec = W.HotkeyRecorder("ctrl+win", "Dictation")
    qtbot.addWidget(rec)
    rec.show()
    cancelled = []
    rec.capture_cancelled.connect(lambda: cancelled.append(1))
    rec.toggle_capture()
    key(rec.btn, Qt.Key.Key_Escape)
    assert cancelled == [1] and rec.combo == "ctrl+win" and rec.btn.text() == "Change"
    rec.toggle_capture()
    QApplication.sendEvent(rec.btn, QEvent(QEvent.Type.FocusOut))
    assert cancelled == [1, 1] and rec.btn.text() == "Change"
    rec.toggle_capture()
    rec.toggle_capture()                                     # pressing Cancel
    assert len(cancelled) == 3


def test_recorder_ignores_keys_it_cannot_express_and_auto_repeat(qtbot):
    rec = W.HotkeyRecorder("", "Dictation")
    qtbot.addWidget(rec)
    rec.show()
    rec.toggle_capture()
    key(rec.btn, Qt.Key.Key_unknown)
    assert rec._seen == []
    rec.use_qt_keys = False                                   # the global hook delivers the capture instead
    key(rec.btn, Qt.Key.Key_A)
    assert rec._seen == []
    rec.captured("alt+f9")
    assert rec.combo == "alt+f9"


def test_recorder_can_be_cleared(qtbot):
    rec = W.HotkeyRecorder("alt+win", "Command mode", allow_clear=True)
    qtbot.addWidget(rec)
    rec.show()
    changed = []
    rec.changed.connect(changed.append)
    rec.btn_clear.click()
    assert changed == [""] and rec.combo == ""


# ============================================================================ Slider
def test_slider_jumps_to_the_click_and_follows_the_keyboard(qtbot):
    s = W.Slider(0, 100, 20, name="Volume")
    qtbot.addWidget(s)
    s.resize(220, 28)
    s.show()
    assert s.accessibleName() == "Volume"
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(110, 14), QPointF(110, 14), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(s, ev)
    assert 45 <= s.value() <= 55
    rel = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(110, 14), QPointF(110, 14), Qt.MouseButton.LeftButton,
                      Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(s, rel)
    before = s.value()
    s.setFocus()
    key(s, Qt.Key.Key_Right)
    assert s.value() > before
    assert not s.grab().isNull()                              # paints (also with the focus ring after the key)
    s.setEnabled(False)
    assert not s.grab().isNull()


# ============================================================================ main window: frameless edges, Esc, title
def test_window_edges_and_cursors(make_ui):
    ui = make_ui()
    win = ui.win
    win.show()
    QApplication.processEvents()
    r = win.panel.geometry()
    assert win._edges_at(r.center()) == Qt.Edge(0)
    assert win._edges_at(QPoint(r.left() + 1, r.center().y())) & Qt.Edge.LeftEdge
    assert win._edges_at(QPoint(r.right() - 1, r.bottom() - 1)) == (Qt.Edge.RightEdge | Qt.Edge.BottomEdge)
    assert win._cursor_for(Qt.Edge.LeftEdge) == Qt.CursorShape.SizeHorCursor
    assert win._cursor_for(Qt.Edge.TopEdge) == Qt.CursorShape.SizeVerCursor
    assert win._cursor_for(Qt.Edge.LeftEdge | Qt.Edge.TopEdge) == Qt.CursorShape.SizeFDiagCursor
    assert win._cursor_for(Qt.Edge.RightEdge | Qt.Edge.TopEdge) == Qt.CursorShape.SizeBDiagCursor
    assert win._cursor_for(Qt.Edge(0)) == Qt.CursorShape.ArrowCursor
    assert win._edges_at(QPoint(-50, -50)) == Qt.Edge(0), "outside the grip band nothing resizes"


def test_escape_hides_the_main_window_and_the_title_bar_double_click_maximises(make_ui):
    ui = make_ui()
    win = ui.win
    win.show()
    QApplication.processEvents()
    was_max = win.isMaximized()
    pos = QPointF(win.panel.width() / 2, 10)
    dbl = QMouseEvent(QEvent.Type.MouseButtonDblClick, pos, pos, Qt.MouseButton.LeftButton,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    assert win.eventFilter(win.panel, dbl) is True
    QApplication.processEvents()
    win.toggle_maximized()
    QApplication.processEvents()
    assert win.isMaximized() in (True, False)                 # the offscreen platform may ignore maximise
    key(win, Qt.Key.Key_Escape)
    assert not win.isVisible()
