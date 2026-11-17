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


# =========================================================================== QA UI-14: Windows accessibility settings
class _Access:
    def __init__(self, motion=Qt.MotionPreference.NoPreference, contrast=Qt.ContrastPreference.NoPreference):
        self.motion, self.contrast = motion, contrast
        self.connected: list = []
        self.motionPreferenceChanged = self.contrastPreferenceChanged = self

    def motionPreference(self):
        return self.motion

    def contrastPreference(self):
        return self.contrast

    def connect(self, slot):
        self.connected.append(slot)


class _NewHints(_Hints):
    def __init__(self, scheme, access):
        super().__init__(scheme)
        self._access = access

    def accessibility(self):
        return self._access


def _new_app(monkeypatch, access, platform="offscreen", palette=None):
    from PySide6.QtGui import QPalette
    hints = _NewHints(Qt.ColorScheme.Dark, access)
    app = _fake_app(monkeypatch, hints)
    app.platformName = staticmethod(lambda: platform)
    app.palette = staticmethod(lambda: palette or QPalette())
    app.applicationStateChanged = hints
    return app, hints


def test_ui14_a_windows_contrast_theme_replaces_the_palette(monkeypatch):
    from PySide6.QtGui import QColor, QPalette
    pal = QPalette()
    role = QPalette.ColorRole
    for r, c in ((role.Window, QColor(0, 0, 0)), (role.WindowText, QColor(255, 255, 255)),
                 (role.Highlight, QColor(26, 235, 255)), (role.HighlightedText, QColor(0, 0, 0))):
        pal.setColor(r, c)
    access = _Access(contrast=Qt.ContrastPreference.HighContrast)
    _new_app(monkeypatch, access, palette=pal)
    seen = Signals(changed=T.bus().changed)
    T.apply_theme("light")
    assert T.HIGH_CONTRAST and T.RESOLVED == "dark", "the contrast theme wins over the app's own choice"
    assert T.PANEL == T.BG0 == QColor(0, 0, 0) and T.TEXT == T.TEXT_DIM == T.STROKE == QColor(255, 255, 255)
    assert T.CYAN == T.VIOLET == QColor(26, 235, 255) and T.ON_ACCENT == QColor(0, 0, 0) and T.BLOOM == 0
    assert T.SCRIM.alpha() == 110 and seen.changed
    pal.setColor(role.Window, QColor(255, 255, 255))
    T.apply_theme("dark")
    assert T.RESOLVED == "light", "a white contrast theme counts as light"
    access.contrast = Qt.ContrastPreference.NoPreference
    T._on_system_prefs()                                       # Windows switched the contrast theme off
    assert not T.HIGH_CONTRAST and T.TEXT == T.DARK.TEXT


def test_ui14_windows_reduced_motion_is_followed_and_the_switch_cannot_undo_it(monkeypatch):
    access = _Access(motion=Qt.MotionPreference.ReducedMotion)
    _, hints = _new_app(monkeypatch, access)
    monkeypatch.setattr(T, "_hooked_system", False)
    T.apply_theme("dark", False)
    assert T.REDUCED_MOTION and T.dur(200) == 0
    assert hints.connected == [T._on_system_scheme, T._on_system_prefs]
    assert access.connected == [T._on_system_prefs, T._on_system_prefs]
    T.set_reduced_motion(False)
    assert T.REDUCED_MOTION, "Windows' setting keeps motion reduced"
    seen = Signals(changed=T.bus().changed)
    T._on_system_prefs()                                       # nothing changed: nothing redrawn
    assert seen.changed == []
    access.motion = Qt.MotionPreference.NoPreference
    T._on_system_prefs()
    assert not T.REDUCED_MOTION and len(seen.changed) == 1
    T.set_reduced_motion(True)
    assert T.REDUCED_MOTION


@pytest.mark.parametrize("value, ok, reduced", [(0, 1, True), (1, 1, False), (0, 0, False)])
def test_ui14_the_animation_effects_setting_is_read_on_the_real_desktop(monkeypatch, value, ok, reduced):
    calls: list = []

    class User32:
        @staticmethod
        def SystemParametersInfoW(action, _p, ref, _w):
            calls.append(action)
            ref._obj.value = value
            return ok

    monkeypatch.setattr(T, "_user32", lambda: User32)
    _new_app(monkeypatch, _Access(), platform="windows")
    T.apply_theme("dark", False)
    assert T.REDUCED_MOTION is reduced and calls == [T.SPI_GETCLIENTAREAANIMATION]


def test_ui14_user32_is_a_private_copy():
    import ctypes
    assert isinstance(T._user32(), ctypes.WinDLL)
