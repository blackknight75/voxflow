"""``voxflow.ui.window``: the frameless main window on the demo controller: page cross-fade, search shortcut,
one dialog at a time, restoring a remembered position, and the frameless move / resize handling. The native
``startSystemMove`` / ``startSystemResize`` calls are replaced by recorders, so no window is dragged."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent, QWindow
from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import demo, flush, make_demo  # noqa: F401 - fixtures
from voxflow.ui import theme as T
from voxflow.ui.components import Modal
from voxflow.ui.window import MainWindow, _Panel

LEFT, RIGHT, NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton
EDGE = Qt.Edge


def mouse(widget, kind, pos, button=NONE, buttons=NONE) -> bool:
    pos = QPointF(pos)
    return QApplication.sendEvent(widget, QMouseEvent(kind, pos, QPointF(widget.mapToGlobal(pos.toPoint())), button,
                                                      buttons, Qt.KeyboardModifier.NoModifier))


@pytest.fixture
def native(monkeypatch):
    """Record the native move / resize requests instead of starting them."""
    calls: list = []
    monkeypatch.setattr(QWindow, "startSystemMove", lambda self: calls.append("move") or True)
    monkeypatch.setattr(QWindow, "startSystemResize", lambda self, edges: calls.append(("resize", edges)) or True)
    return calls


# =========================================================================== navigation
def test_switching_pages_cross_fades_unless_motion_is_reduced(demo):
    win = demo.win
    win.show_page("history")
    page = win.pages["history"]
    assert page.graphicsEffect() is not None and win.nav["history"].isChecked() and not win.nav["home"].isChecked()
    demo.wait(lambda: page.graphicsEffect() is None)           # the effect is removed when the fade ends
    win.show_page("history")
    assert page.graphicsEffect() is None, "showing the current page again does not fade"
    win.show_page("no-such-page")
    assert win.current == "history"
    T.apply_theme("dark", True)
    win.show_page("style")
    assert win.pages["style"].graphicsEffect() is None and win.current == "style"
    assert win._save_timer.isActive()
    win._save_timer.stop()


def test_find_shortcut_goes_to_the_history_search(demo):
    demo.win._focus_search()
    QApplication.processEvents()
    assert demo.win.current == "history" and demo.win.pages["history"].search.hasFocus()


def test_only_one_dialog_at_a_time_and_a_page_switch_closes_it(demo):
    win = demo.win
    first = win.open_modal(Modal(win.panel, "First", name="first"))
    assert first.isVisible() and win._modal is first
    second = Modal(win.panel, "Second", name="second")
    second.add_buttons("OK", second.close_modal)
    win.open_modal(second)
    assert not first.isVisible() and second.isVisible() and win._modal is second and second.btn_cancel.hasFocus()
    win.show_page("snippets")
    assert not second.isVisible() and win._modal is None
    flush()


def test_controller_events_reach_the_sidebar_and_the_toasts(demo):
    win, c = demo.win, demo.c
    assert win.nav["dictionary"].badge == len(c.dictionary.suggestions()) == 2
    c.dictionary.dismiss_suggestion(c.dictionary.suggestions()[0]["id"])
    assert win.nav["dictionary"].badge == 1

    def broken():
        raise RuntimeError("dictionary file is locked")

    c.dictionary.suggestions = broken
    win._update_badges()
    assert win.nav["dictionary"].badge == 0, "a store problem only costs the badge"

    win.btn_pause.click()
    assert c.paused and win.btn_pause.text() == win.btn_pause.accessibleName() == "Resume dictation"
    assert win.btn_pause.icon_name == "play" and win.status.level == "paused"
    win.btn_pause.click()
    assert not c.paused and win.btn_pause.text() == "Pause dictation" and win.btn_pause.icon_name == "pause"
    win._on_status(None)
    assert win.status.title == ""

    c.dictionary_suggestion.emit({"term": "GitHub"})
    assert demo.toasts() == ["New dictionary suggestion: “GitHub”."]
    win.toasts._toasts[0].btn.click()
    flush()
    assert win.current == "dictionary"
    c.dictionary_suggestion.emit({"term": "Qt"})
    assert demo.toasts() == [], "already on the Dictionary page: the inbox is in front of the user"
    win.hide()
    c.notice.emit("info", "Nobody is looking.")
    win.show_page("home")
    c.dictionary_suggestion.emit({"term": "Qt"})
    assert demo.toasts() == []


# =========================================================================== geometry
def test_window_opens_centred_and_remembers_where_it_was_left(make_demo):
    demo = make_demo()
    win, area = demo.win, QGuiApplication.primaryScreen().availableGeometry()
    assert win.geometry().center() == area.center() or abs(win.geometry().center().x() - area.center().x()) <= 1
    win.setGeometry(40, 30, 828, 620)
    win._save_state()
    assert demo.c.get("ui_state") == {"page": "home", "maximized": False, "window": [40, 30, 828, 620]}
    stored = dict(demo.c.get("ui_state"))
    win._save_state()
    assert demo.c.get("ui_state") == stored

    again = make_demo(ui_state={"page": "style", "maximized": False, "window": [40, 30, 828, 620]})
    assert again.win.current == "style"
    assert (again.win.x(), again.win.y(), again.win.width(), again.win.height()) == (40, 30, 828, 620)


@pytest.mark.parametrize("stored", [
    [10 ** 6, 10 ** 6, 900, 640],            # a monitor that is no longer there
    [40, 30, 900],                           # damaged
    ["a", "b", "c", "d"],
    None,
])
def test_an_unusable_remembered_position_falls_back_to_the_centre(make_demo, stored):
    demo = make_demo(ui_state={"page": "nope", "window": stored})
    win, area = demo.win, QGuiApplication.primaryScreen().availableGeometry()
    assert win.current == "home"
    assert abs(win.geometry().center().x() - area.center().x()) <= 1
    assert abs(win.geometry().center().y() - area.center().y()) <= 1
    assert win.width() <= area.width() + 2 * win.M and win.height() <= area.height() + 2 * win.M


def test_a_remembered_size_larger_than_the_screen_is_cut_down_and_maximized_is_restored(make_demo):
    area = QGuiApplication.primaryScreen().availableGeometry()
    demo = make_demo(ui_state={"window": [0, 0, 10 ** 5, 10 ** 5], "maximized": True})
    win = demo.win
    QApplication.processEvents()
    assert win.isMaximized() and win.btn_max.kind == "restore" and win.btn_max.toolTip() == "Restore"
    assert win._root.contentsMargins().left() == 0 and win.panel.corner_radius == 0
    assert win._edges_at(QPoint(1, 300)) == EDGE(0), "a maximized window has no resize band"
    assert demo.render()
    win.btn_max.click()
    QApplication.processEvents()
    assert not win.isMaximized() and win.btn_max.kind == "max" and win._root.contentsMargins().left() == win.M
    assert win.width() <= area.width() + 2 * win.M and win.height() <= area.height() + 2 * win.M


def test_remembering_the_layout_never_breaks_hiding(demo):
    def refuse(key, value):
        raise RuntimeError("config file is read-only")

    demo.c.set = refuse
    demo.win.hide()
    assert not demo.win.isVisible() and not demo.win._save_timer.isActive()


def test_moving_and_resizing_schedule_a_save_and_minimising_pauses_the_page(demo):
    win = demo.win
    win._save_timer.stop()
    win.resize(win.width() + 20, win.height())
    QApplication.processEvents()
    assert win._save_timer.isActive()
    win._save_timer.stop()
    win.move(win.x() + 15, win.y() + 5)
    QApplication.processEvents()
    assert win._save_timer.isActive()
    win._save_timer.stop()
    page = demo.page("settings")
    assert page._sync_timer.isActive()
    win.btn_min.click()
    QApplication.processEvents()
    assert win.isMinimized() and not page._sync_timer.isActive(), "nothing is visible: page timers stop"
    demo.c.notice.emit("info", "Not shown while minimised.")
    assert demo.toasts() == []
    win.present()
    QApplication.processEvents()
    assert not win.isMinimized() and page._sync_timer.isActive()
    win.hide()
    win.setWindowState(Qt.WindowState.WindowMaximized)         # a state change while hidden enters nothing
    QApplication.processEvents()
    assert not page._sync_timer.isActive()
    win.setWindowState(Qt.WindowState.WindowNoState)


def test_closing_hides_to_the_tray(demo):
    win = demo.win
    assert win.close() is False and not win.isVisible(), "the app lives in the tray"
    win.present("history")
    assert win.isVisible() and win.current == "history"
    demo.qtbot.keyClick(win, Qt.Key.Key_A)
    assert win.isVisible()
    demo.qtbot.keyClick(win, Qt.Key.Key_Escape)
    assert not win.isVisible()
    win.present()
    win.btn_close.click()
    assert not win.isVisible()


# =========================================================================== frameless move / resize
def test_resize_band_and_cursors(demo):
    win = demo.win
    r = win.panel.geometry()
    mid_x, mid_y = r.center().x(), r.center().y()
    cases = {
        (r.left() - 2, mid_y): (EDGE.LeftEdge, Qt.CursorShape.SizeHorCursor),
        (r.right() + 2, mid_y): (EDGE.RightEdge, Qt.CursorShape.SizeHorCursor),
        (mid_x, r.top() - 2): (EDGE.TopEdge, Qt.CursorShape.SizeVerCursor),
        (mid_x, r.bottom() + 2): (EDGE.BottomEdge, Qt.CursorShape.SizeVerCursor),
        (r.left() - 1, r.top() - 1): (EDGE.LeftEdge | EDGE.TopEdge, Qt.CursorShape.SizeFDiagCursor),
        (r.right() + 1, r.bottom() + 1): (EDGE.RightEdge | EDGE.BottomEdge, Qt.CursorShape.SizeFDiagCursor),
        (r.right() + 1, r.top() - 1): (EDGE.RightEdge | EDGE.TopEdge, Qt.CursorShape.SizeBDiagCursor),
        (r.left() - 1, r.bottom() + 1): (EDGE.LeftEdge | EDGE.BottomEdge, Qt.CursorShape.SizeBDiagCursor),
        (mid_x, mid_y): (EDGE(0), Qt.CursorShape.ArrowCursor),                 # inside the panel
        (1, 1): (EDGE(0), Qt.CursorShape.ArrowCursor),                         # far out in the shadow
    }
    for (x, y), (edges, cursor) in cases.items():
        assert win._edges_at(QPoint(x, y)) == edges, (x, y)
        assert win._cursor_for(edges) == cursor
        mouse(win, QEvent.Type.MouseMove, QPoint(x, y))
        assert win.cursor().shape() == cursor, (x, y)
    mouse(win, QEvent.Type.MouseMove, QPoint(r.left() - 2, mid_y))
    QApplication.sendEvent(win, QEvent(QEvent.Type.Leave))
    assert win.cursor().shape() == Qt.CursorShape.ArrowCursor, "leaving the window drops the resize cursor"


def test_pressing_the_frame_starts_a_native_resize_or_move(demo, native):
    win = demo.win
    r = win.panel.geometry()
    mouse(win, QEvent.Type.MouseButtonPress, QPoint(r.left() - 2, r.center().y()), LEFT, LEFT)
    assert native == [("resize", EDGE.LeftEdge)]
    mouse(win, QEvent.Type.MouseButtonPress, QPoint(r.left() - 2, r.center().y()), RIGHT, RIGHT)
    mouse(win, QEvent.Type.MouseButtonPress, QPoint(2, 2), LEFT, LEFT)                   # in the shadow
    assert len(native) == 1, "only the left button on the band resizes"
    assert win._press(QPoint(r.center().x(), r.top() + 20), LEFT) is True and native[-1] == "move"
    assert win._press(QPoint(r.center().x(), r.top() + _Panel.TITLE_H + 5), LEFT) is False
    assert win._press(QPoint(r.right() + 2, r.bottom() + 2), LEFT) is True
    assert native[-1] == ("resize", EDGE.RightEdge | EDGE.BottomEdge)

    # the same through the panel, whose events the window filters (positions are panel-relative there)
    native.clear()
    title = QPoint(win.panel.width() // 2, 20)
    assert mouse(win.panel, QEvent.Type.MouseButtonPress, title, LEFT, LEFT) and native == ["move"]
    mouse(win.panel, QEvent.Type.MouseButtonPress, QPoint(1, win.panel.height() // 2), LEFT, LEFT)
    assert native[-1] == ("resize", EDGE.LeftEdge)
    mouse(win.panel, QEvent.Type.MouseMove, QPoint(1, win.panel.height() // 2))
    assert win.cursor().shape() == Qt.CursorShape.SizeHorCursor
    mouse(win.panel, QEvent.Type.MouseMove, QPoint(300, 300))
    assert win.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert win.eventFilter(win.stack, QEvent(QEvent.Type.MouseButtonPress)) is False, "other widgets are not filtered"


def test_double_click_on_the_title_band_toggles_maximized(demo, native):
    win = demo.win
    mouse(win.panel, QEvent.Type.MouseButtonDblClick, QPoint(win.panel.width() // 2, 200), LEFT, LEFT)
    QApplication.processEvents()
    assert not win.isMaximized(), "only the title band reacts"
    mouse(win.panel, QEvent.Type.MouseButtonDblClick, QPoint(win.panel.width() // 2, 20), LEFT, LEFT)
    QApplication.processEvents()
    assert win.isMaximized()
    mouse(win.panel, QEvent.Type.MouseButtonDblClick, QPoint(win.panel.width() // 2, 20), LEFT, LEFT)
    QApplication.processEvents()
    assert not win.isMaximized()


def test_a_window_that_was_never_shown_has_nothing_to_drag(qapp, native):
    from voxflow.ui.fakes import FakeController
    c = FakeController(empty=True)
    win = MainWindow(c)
    try:
        assert win.windowHandle() is None
        assert win._press(QPoint(100, 30), LEFT) is False and native == []
        assert not win._placed
        win._save_state()
        assert "window" not in c.get("ui_state"), "no position is stored before the window was ever placed"
    finally:
        win._save_timer.stop()
        win.deleteLater()
        c.deleteLater()
        flush()
        T.apply_theme("dark", False)


# =========================================================================== QA-ui fixes
class _Screen:
    def __init__(self, w, h):
        from PySide6.QtCore import QRect
        self.rect = QRect(0, 0, w, h)

    def availableGeometry(self):
        return self.rect


def test_ui02_the_window_fits_a_small_work_area(make_demo, monkeypatch):
    from voxflow.ui import overlay as overlay_mod
    from voxflow.ui import window as window_mod
    small = _Screen(960, 492)                                  # 1920x1080 at 200 %
    demo = make_demo(show_window=False)
    win, m = demo.win, demo.win.M
    monkeypatch.setattr(window_mod.QGuiApplication, "screens", staticmethod(lambda: [small]))
    monkeypatch.setattr(overlay_mod, "active_screen", lambda: small)
    win.show()
    QApplication.processEvents()
    assert win.minimumHeight() == 492 + 2 * m and win.minimumWidth() == 800 + 2 * m
    assert win.y() + m >= 0 and win.y() + win.height() - m <= 492, "the bottom of the window is on the screen"
    win._fit_minimum(_Screen(600, 300))                        # tinier than the floor: the floor holds
    assert (win.minimumWidth(), win.minimumHeight()) == (640 + 2 * m, 460 + 2 * m)
    win._fit_minimum(_Screen(2560, 1400))                      # moved to a big monitor: the usual minimum
    assert (win.minimumWidth(), win.minimumHeight()) == (800 + 2 * m, 580 + 2 * m)


def test_ui02_a_saved_size_is_fitted_to_the_small_screen_it_is_on(make_demo, monkeypatch):
    from voxflow.ui import window as window_mod
    small = _Screen(1097, 569)                                 # 1920x1080 at 175 %
    demo = make_demo(show_window=False, ui_state={"page": "home", "maximized": False,
                                                  "window": [0, 0, 1108, 748]})
    monkeypatch.setattr(window_mod.QGuiApplication, "screens", staticmethod(lambda: [small]))
    demo.win.show()
    QApplication.processEvents()
    m = demo.win.M
    assert demo.win.minimumHeight() == 569 + 2 * m and demo.win.height() == 569 + 2 * m
