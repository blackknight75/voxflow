"""Shared fixtures and helpers of the ``tests/test_cov_ui_*.py`` files (the UI coverage pass).

* ``show``      : put a bare widget on the (offscreen) screen; closed again by pytest-qt, Qt warnings fail the test.
* ``make_demo`` : the whole UI (``build_ui``) on the in-memory ``FakeController`` that the demo script uses.
* small event helpers (``hover``, ``unhover``, ``pixels``, ``flush``, ``Signals``).

Nothing here starts a timer that outlives its test: every widget is hidden/closed and deleted in the fixture
teardown, and the theme is put back to dark / full motion.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF
from PySide6.QtGui import QColor, QEnterEvent, QImage, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from voxflow.ui import theme as T
from voxflow.ui.fakes import FakeController
from voxflow.ui.shell import build_ui


def flush() -> None:
    """Run pending ``deleteLater`` calls and queued events."""
    QApplication.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def hover(widget: QWidget) -> None:
    """Deliver the pointer-entered event (the offscreen platform has no real pointer)."""
    c = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(widget, QEnterEvent(c, c, QPointF(widget.mapToGlobal(c.toPoint()))))


def unhover(widget: QWidget) -> None:
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


def pixels(widget: QWidget) -> bytes:
    """The painted picture of a widget on a fixed backdrop, comparable between two states. (``grab()`` would mix in
    the window background, which changes with the active / inactive palette group.)"""
    img = QImage(widget.size(), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(18, 20, 30))
    widget.render(img, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    return bytes(img.constBits())


def image_bytes(image: QImage) -> bytes:
    """The pixels of an image. The image is held in a name while its buffer is read: reading the buffer of a
    temporary (``x.toImage().constBits()``) can crash."""
    return bytes(image.constBits())


def grab_bytes(widget_or_pixmap) -> bytes:
    """``grab()`` of a top-level widget (or a pixmap) as comparable bytes."""
    pm = widget_or_pixmap.grab() if isinstance(widget_or_pixmap, QWidget) else widget_or_pixmap
    assert not pm.isNull()
    return image_bytes(pm.toImage())


class Signals:
    """Record the emissions of some signals: ``Signals(a=obj.sig_a).a`` -> list of argument tuples."""

    def __init__(self, **signals):
        for name, sig in signals.items():
            seen: list = []
            setattr(self, name, seen)
            sig.connect(lambda *a, _seen=seen: _seen.append(a))


@pytest.fixture
def show(qtbot, qt_messages):
    """``show(widget, width=None, height=None)`` -> the widget, visible. Teardown restores the theme."""

    def _show(widget: QWidget, width: int | None = None, height: int | None = None) -> QWidget:
        qtbot.addWidget(widget)
        if width or height:
            widget.resize(width or widget.width(), height or widget.height())
        widget.move(260, 190)            # away from the offscreen pointer at (0, 0): no stray hover events
        widget.show()
        QApplication.processEvents()
        return widget

    yield _show
    T.apply_theme("dark", False)
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


class Demo:
    def __init__(self, ctl: FakeController, shell, qtbot):
        self.c, self.shell, self.qtbot = ctl, shell, qtbot
        self.win = shell.window

    def find(self, name: str, root=None) -> QWidget:
        found = self.find_all(name, root)
        assert found, f"no widget named {name!r}"
        return found[0]

    def find_all(self, name: str, root=None) -> list[QWidget]:
        flush()
        return list((root or self.win).findChildren(QWidget, name))

    def page(self, key: str):
        self.win.show_page(key)
        QApplication.processEvents()
        return self.win.pages[key]

    def toasts(self) -> list[str]:
        return self.win.toasts.messages()

    def render(self, widget=None):
        pm = (widget or self.win).grab()
        assert not pm.isNull()
        return pm

    def wait(self, cond, timeout=4000):
        self.qtbot.waitUntil(lambda: bool(cond()), timeout=timeout)


@pytest.fixture
def make_demo(qtbot, qt_messages):
    """Factory: ``make_demo(empty=False, show_window=True, **config)`` -> Demo (whole UI on a FakeController)."""
    made: list[Demo] = []

    def build(empty: bool = False, show_window: bool = True, prepare=None, **config) -> Demo:
        ctl = FakeController(empty=empty)
        ctl.config.onboarding_done = True
        ctl.config.theme = "dark"
        for key, value in config.items():
            setattr(ctl.config, key, value)
        if prepare is not None:
            prepare(ctl)
        shell = build_ui(ctl, show_window=show_window)
        QApplication.processEvents()
        demo = Demo(ctl, shell, qtbot)
        made.append(demo)
        return demo

    yield build
    for demo in made:
        shell = demo.shell
        shell.close()
        demo.c.shutdown()
        shell.overlay._timer.stop()
        shell.overlay._follow.stop()
        for widget in (shell.onboarding, shell.window, shell.overlay, shell.tray.menu):
            if widget is not None:
                widget.deleteLater()
        shell.tray.icon.deleteLater()
        shell.tray.deleteLater()
        shell.deleteLater()
        demo.c.deleteLater()
    flush()
    T.apply_theme("dark", False)
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


@pytest.fixture
def demo(make_demo) -> Demo:
    return make_demo()
