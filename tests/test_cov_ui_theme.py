"""``voxflow.ui.theme``: following the system colour scheme, reduced motion, and the corners that need a Qt
without ``colorScheme`` (older than 6.5) or no application at all. The application object is replaced by a
stand-in, so the real style hints are never touched."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from tests.cov_ui_kit import Signals
from voxflow.ui import theme as T


@pytest.fixture(autouse=True)
def _restore_theme(qapp):
    yield
    T.apply_theme("dark", False)


class _Hints:
    def __init__(self, scheme):
        self._scheme = scheme
        self.connected: list = []
        self.colorSchemeChanged = self

    def colorScheme(self):
        return self._scheme

    def connect(self, slot):
        self.connected.append(slot)


class _OldHints:
    """Qt before 6.5: no ``colorScheme`` and no ``colorSchemeChanged``."""


def _fake_app(monkeypatch, hints):
    class App:
        @staticmethod
        def instance():
            return None if hints is None else App

        @staticmethod
        def styleHints():
            return hints

    monkeypatch.setattr(T, "QGuiApplication", App)
    return App


@pytest.mark.parametrize("scheme, resolved", [(Qt.ColorScheme.Light, "light"), (Qt.ColorScheme.Dark, "dark"),
                                              (Qt.ColorScheme.Unknown, "dark")])
def test_system_mode_resolves_to_what_the_desktop_reports(monkeypatch, scheme, resolved):
    _fake_app(monkeypatch, _Hints(scheme))
    T.apply_theme("system")
    assert (T.MODE, T.RESOLVED) == ("system", resolved) and T.is_light() == (resolved == "light")
    assert T.TEXT == (T.LIGHT if resolved == "light" else T.DARK).TEXT


def test_system_mode_is_dark_without_an_application_or_on_an_old_qt(monkeypatch):
    _fake_app(monkeypatch, None)
    assert T._system_scheme() == "dark"
    T.apply_theme("system")
    assert T.RESOLVED == "dark"
    _fake_app(monkeypatch, _OldHints())
    assert T._system_scheme() == "dark"


def test_the_system_scheme_signal_is_hooked_once_and_survives_an_old_qt(monkeypatch):
    hints = _Hints(Qt.ColorScheme.Dark)
    _fake_app(monkeypatch, hints)
    monkeypatch.setattr(T, "_hooked_system", False)
    T.apply_theme("dark")
    T.apply_theme("light")
    assert hints.connected == [T._on_system_scheme] and T._hooked_system is True
    monkeypatch.setattr(T, "_hooked_system", False)
    _fake_app(monkeypatch, _OldHints())
    T.apply_theme("dark")                                    # must not raise
    assert T._hooked_system is False


def test_a_desktop_theme_change_is_followed_only_in_system_mode(monkeypatch):
    hints = _Hints(Qt.ColorScheme.Dark)
    _fake_app(monkeypatch, hints)
    seen = Signals(changed=T.bus().changed)
    T.apply_theme("system")
    assert T.RESOLVED == "dark" and len(seen.changed) == 1
    hints._scheme = Qt.ColorScheme.Light
    T._on_system_scheme(Qt.ColorScheme.Light)
    assert T.RESOLVED == "light" and T.MODE == "system" and len(seen.changed) == 2
    T.apply_theme("dark")
    T._on_system_scheme(Qt.ColorScheme.Light)
    assert T.RESOLVED == "dark" and len(seen.changed) == 3, "an explicit choice ignores the desktop"


def test_unknown_mode_falls_back_to_dark():
    T.apply_theme("sepia")
    assert (T.MODE, T.RESOLVED) == ("dark", "dark")


def test_reduced_motion_zeroes_durations_and_announces_real_changes_only(monkeypatch):
    seen = Signals(changed=T.bus().changed)
    assert T.dur(180) == 180
    T.set_reduced_motion(True)
    assert T.REDUCED_MOTION and T.dur(180) == 0 and len(seen.changed) == 1
    T.set_reduced_motion(True)
    assert len(seen.changed) == 1
    T.apply_theme("dark", reduced_motion=False)
    assert T.dur(180) == 180
    monkeypatch.setattr(T, "_bus", None)                     # before any widget asked for the bus
    T.set_reduced_motion(True)
    T.apply_theme("light")
    assert T.REDUCED_MOTION and T.RESOLVED == "light" and len(seen.changed) == 2


def test_colour_helpers():
    assert T.alpha(T.DARK.CYAN, 2.0).alpha() == 255 and T.alpha(T.DARK.CYAN, -1).alpha() == 0
    mid = T.mix(T.DARK.BG0, T.DARK.TEXT, 0.5)
    assert T.DARK.BG0.red() < mid.red() < T.DARK.TEXT.red()
    assert T.mix(T.DARK.BG0, T.DARK.TEXT, 7) == T.DARK.TEXT and T.mix(T.DARK.BG0, T.DARK.TEXT, -7) == T.DARK.BG0
    assert T.css(T.DARK.SCRIM) == "rgba(4,5,10,170)"
    assert T.token("TEXT_DIM") == T.TEXT_DIM
    assert T.ease_out(-1) == 0 and T.ease_out(2) == 1 and 0.5 < T.ease_out(0.5) < 1
    assert T.smooth(0.0, 1.0, 0.0, 10) == 0.0 and T.smooth(0.0, 1.0, 100.0, 10) == pytest.approx(1.0)
    assert 'QLineEdit[invalid="true"]' in T.qss() and T.css(T.RED) in T.qss()
