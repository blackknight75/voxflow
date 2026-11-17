"""Screen-reader plumbing for the self-painted controls.

Qt describes a plain ``QWidget`` as a nameless "client" with no value. A widget that defines ``a11y_role()`` gets an
accessible interface from :func:`install` that reports that role, the text of ``a11y_value()`` as its value (UIA
Value pattern) and the state changes of ``a11y_state(state)``. :func:`announce` speaks a message without moving
focus (UIA notification), for things that only show visually, such as the Flow bar's states.
"""
from __future__ import annotations

from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent, QAccessibleValueChangeEvent
from PySide6.QtWidgets import QAccessibleWidget

Role = QAccessible.Role


class _Iface(QAccessibleWidget):
    def __init__(self, widget):
        super().__init__(widget, widget.a11y_role())

    def text(self, t):
        if t == QAccessible.Text.Value:
            return str(self.widget().a11y_value())
        return super().text(t)

    def state(self):
        return self.widget().a11y_state(super().state())


def _factory(_key: str, obj):
    return _Iface(obj) if hasattr(obj, "a11y_role") and hasattr(obj, "isWidgetType") and obj.isWidgetType() else None


_installed = False


def install() -> None:
    """Register the interface factory once per process (idempotent)."""
    global _installed
    if not _installed:
        QAccessible.installFactory(_factory)
        _installed = True


def _send(ev) -> None:
    QAccessible.updateAccessibility(ev)


def value_changed(widget, value: str) -> None:
    """Tell assistive technology that ``widget``'s value now reads ``value`` (no-op when none is listening)."""
    if QAccessible.isActive():
        _send(QAccessibleValueChangeEvent(widget, value))


def announce(widget, text: str, assertive: bool = False) -> None:
    """Have a screen reader say ``text`` (Narrator / NVDA via a UIA notification); focus does not move."""
    if text and QAccessible.isActive():
        ev = QAccessibleAnnouncementEvent(widget, text)
        if assertive:
            ev.setPoliteness(QAccessible.AnnouncementPoliteness.Assertive)
        _send(ev)


install()
