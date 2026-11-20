"""Design system: colour tokens (dark / light), type and spacing scale, elevation, motion.

Colour tokens are module-level names (``T.TEXT``, ``T.PANEL`` ...). Painters must read them at paint time
(``T.TEXT``, never ``from theme import TEXT``) because :func:`apply_theme` rebinds them and then emits
``bus().changed`` so every widget repaints. ``T.DARK`` is the fixed dark palette for surfaces that float over
arbitrary desktop content (the Flow bar) and therefore never follow the app theme.

All sizes are logical pixels: Qt 6 scales them per monitor, so 100 / 150 / 200 % need no special casing.
"""
from __future__ import annotations

import ctypes
import math
from types import SimpleNamespace

from PySide6.QtCore import QEasingCurve, QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPalette, QPen


# --------------------------------------------------------------------------- palettes
class _Dark:
    name = "dark"
    CYAN = QColor(34, 228, 255)
    VIOLET = QColor(139, 92, 255)
    MAGENTA = QColor(255, 79, 216)
    GREEN = QColor(53, 242, 176)
    RED = QColor(255, 92, 122)
    AMBER = QColor(255, 191, 71)
    TEXT = QColor(234, 239, 255)
    TEXT_DIM = QColor(178, 187, 216)
    TEXT_FAINT = QColor(140, 149, 184)
    BG0 = QColor(8, 10, 17)             # window base, bottom
    BG1 = QColor(16, 19, 33)            # window base, top
    SIDEBAR = QColor(255, 255, 255, 7)
    PANEL = QColor(26, 30, 50)          # cards
    PANEL_HI = QColor(36, 42, 68)       # hovered card / raised control
    SUNK = QColor(11, 13, 23)           # inputs, wells
    CONTROL = QColor(34, 39, 64)        # neutral button fill
    CONTROL_HI = QColor(48, 55, 88)
    STROKE = QColor(255, 255, 255, 24)
    STROKE_STRONG = QColor(255, 255, 255, 46)
    ON_ACCENT = QColor(5, 8, 18)        # text on the accent gradient
    SHADOW = QColor(0, 0, 0)
    SCRIM = QColor(4, 5, 10, 170)
    BLOOM = 0.13                        # strength of the ambient colour blooms


class _Light:
    name = "light"
    CYAN = QColor(0, 106, 138)
    VIOLET = QColor(109, 56, 226)
    MAGENTA = QColor(176, 32, 134)
    GREEN = QColor(0, 112, 76)
    RED = QColor(190, 34, 66)
    AMBER = QColor(140, 84, 0)
    TEXT = QColor(20, 24, 44)
    TEXT_DIM = QColor(66, 74, 106)
    TEXT_FAINT = QColor(92, 100, 132)
    BG0 = QColor(236, 239, 248)
    BG1 = QColor(247, 249, 254)
    SIDEBAR = QColor(20, 30, 80, 10)
    PANEL = QColor(255, 255, 255)
    PANEL_HI = QColor(243, 246, 253)
    SUNK = QColor(240, 243, 250)
    CONTROL = QColor(232, 236, 247)
    CONTROL_HI = QColor(219, 225, 242)
    STROKE = QColor(20, 30, 80, 30)
    STROKE_STRONG = QColor(20, 30, 80, 64)
    ON_ACCENT = QColor(255, 255, 255)
    SHADOW = QColor(30, 40, 90)
    SCRIM = QColor(20, 26, 50, 110)
    BLOOM = 0.09


DARK, LIGHT = _Dark, _Light
_TOKENS = [k for k in vars(_Dark) if k.isupper()]

# current palette, rebound by apply_theme()
CYAN = VIOLET = MAGENTA = GREEN = RED = AMBER = TEXT = TEXT_DIM = TEXT_FAINT = BG0 = BG1 = SIDEBAR = PANEL = \
    PANEL_HI = SUNK = CONTROL = CONTROL_HI = STROKE = STROKE_STRONG = ON_ACCENT = SHADOW = SCRIM = QColor()
BLOOM = 0.13
MODE = "dark"            # the requested mode: dark | light | system
RESOLVED = "dark"        # what is actually showing
REDUCED_MOTION = False   # in effect: the app's own switch, or Windows' "Animation effects" being off
HIGH_CONTRAST = False    # a Windows contrast theme is on: the tokens are its system colours
_user_reduced = False    # the app's own "Reduce motion" switch
_system = (False, False)  # (reduced motion, high contrast) as Windows last asked for them

# --------------------------------------------------------------------------- scale
S1, S2, S3, S4, S5, S6, S8, S10 = 4, 8, 12, 16, 20, 24, 32, 40        # spacing
R_SM, R_MD, R_LG, R_XL = 8, 11, 15, 20                                # corner radii
CONTROL_H = 34
FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial"]
FONT_FAMILY = "Segoe UI"
# type scale (pt)
F_CAPTION, F_SMALL, F_BODY, F_BODY_LG, F_TITLE, F_H2, F_H1, F_DISPLAY = 9.0, 9.0, 9.75, 10.5, 12.5, 15.0, 19.0, 25.0

W_NORMAL, W_MEDIUM, W_SEMI, W_BOLD = (QFont.Weight.Normal, QFont.Weight.Medium, QFont.Weight.DemiBold,
                                      QFont.Weight.Bold)


class _Bus(QObject):
    changed = Signal()           # palette or motion preference changed: restyle + repaint


_bus: _Bus | None = None
_hooked_system = False


def bus() -> _Bus:
    global _bus
    if _bus is None:
        _bus = _Bus()
    return _bus


def _system_scheme() -> str:
    app = QGuiApplication.instance()
    if app is None:
        return "dark"
    try:
        return "light" if app.styleHints().colorScheme() == Qt.ColorScheme.Light else "dark"
    except AttributeError:      # Qt < 6.5
        return "dark"


def _on_system_scheme(*_a) -> None:
    if MODE == "system":
        apply_theme("system")


SPI_GETCLIENTAREAANIMATION = 0x1042


def _user32():
    return ctypes.WinDLL("user32", use_last_error=True)


def _windows_animations_off(app) -> bool:
    """Windows Settings > Accessibility > Visual effects > Animation effects is off (read on the real desktop
    only; Qt's own motion preference may not report it)."""
    if app.platformName() != "windows":
        return False
    on = ctypes.c_int(1)
    return bool(_user32().SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(on), 0)) and not on.value


def _system_prefs() -> tuple[bool, bool]:
    """(reduced motion, high contrast) as Windows asks for them."""
    app = QGuiApplication.instance()
    if app is None:
        return False, False
    style = app.styleHints()                       # held: the accessibility hints live as long as it does
    try:
        hints = style.accessibility()
    except AttributeError:                         # Qt < 6.10
        return False, False
    reduced = hints.motionPreference() == Qt.MotionPreference.ReducedMotion or _windows_animations_off(app)
    return reduced, hints.contrastPreference() == Qt.ContrastPreference.HighContrast


def _contrast_palette() -> SimpleNamespace:
    """Every token from the system colours of the active Windows contrast theme: solid surfaces, solid strokes,
    the theme's highlight for every accent, and no decorative bloom or shadow."""
    pal = QGuiApplication.palette()
    role = QPalette.ColorRole
    bg, fg = pal.color(role.Window), pal.color(role.WindowText)
    hi, on_hi = pal.color(role.Highlight), pal.color(role.HighlightedText)
    grey = pal.color(QPalette.ColorGroup.Disabled, role.WindowText)
    scrim = QColor(fg)
    scrim.setAlpha(110)
    tokens = dict.fromkeys(("BG0", "BG1", "SIDEBAR", "PANEL", "PANEL_HI", "SUNK", "CONTROL", "CONTROL_HI",
                            "SHADOW"), bg)
    tokens.update(dict.fromkeys(("TEXT", "TEXT_DIM", "STROKE", "STROKE_STRONG", "GREEN", "RED", "AMBER"), fg))
    tokens.update(dict.fromkeys(("CYAN", "VIOLET", "MAGENTA"), hi))
    tokens.update(TEXT_FAINT=grey, ON_ACCENT=on_hi, SCRIM=scrim, BLOOM=0.0,
                  name="light" if bg.lightness() >= 128 else "dark")
    return SimpleNamespace(**tokens)


def _on_system_prefs(*_a) -> None:
    if _system_prefs() != _system:
        apply_theme(MODE)


def apply_theme(mode: str = "dark", reduced_motion: bool | None = None) -> None:
    """Switch palette (``dark`` | ``light`` | ``system``) and optionally the app's reduced-motion switch. Windows'
    contrast themes and its "Animation effects" setting are followed as well."""
    global MODE, RESOLVED, REDUCED_MOTION, HIGH_CONTRAST, _user_reduced, _system, _hooked_system
    mode = mode if mode in ("dark", "light", "system") else "dark"
    if reduced_motion is not None:
        _user_reduced = bool(reduced_motion)
    _system = _system_prefs()
    HIGH_CONTRAST = _system[1]
    if HIGH_CONTRAST:
        pal = _contrast_palette()
        resolved = pal.name
    else:
        resolved = _system_scheme() if mode == "system" else mode
        pal = LIGHT if resolved == "light" else DARK
    g = globals()
    for k in _TOKENS:
        g[k] = getattr(pal, k)
    MODE, RESOLVED = mode, resolved
    REDUCED_MOTION = _user_reduced or _system[0]
    app = QGuiApplication.instance()
    if app is not None and not _hooked_system:
        try:
            style = app.styleHints()
            style.colorSchemeChanged.connect(_on_system_scheme)
            _hooked_system = True
            hints = style.accessibility()
            hints.motionPreferenceChanged.connect(_on_system_prefs)
            hints.contrastPreferenceChanged.connect(_on_system_prefs)
            app.applicationStateChanged.connect(_on_system_prefs)     # back to VoxFlow: Windows may have changed
        except AttributeError:
            pass
    if _bus is not None:
        _bus.changed.emit()


def set_reduced_motion(on: bool) -> None:
    """The app's own switch. Windows' "Animation effects" off keeps motion reduced either way."""
    global REDUCED_MOTION, _user_reduced
    _user_reduced = bool(on)
    effective = _user_reduced or _system[0]
    if REDUCED_MOTION != effective:
        REDUCED_MOTION = effective
        if _bus is not None:
            _bus.changed.emit()


def is_light() -> bool:
    return RESOLVED == "light"


def dur(ms: int) -> int:
    """Animation duration honouring reduced motion (0 = jump straight to the end value)."""
    return 0 if REDUCED_MOTION else ms


EASE = QEasingCurve.Type.OutCubic
EASE_IN_OUT = QEasingCurve.Type.InOutCubic


# --------------------------------------------------------------------------- helpers
def font(pt: float = F_BODY, weight=QFont.Weight.Normal, spacing: float = 0) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPointSizeF(pt)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


def mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(int(a.red() + (b.red() - a.red()) * t), int(a.green() + (b.green() - a.green()) * t),
                  int(a.blue() + (b.blue() - a.blue()) * t), int(a.alpha() + (b.alpha() - a.alpha()) * t))


def css(c: QColor) -> str:
    return f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})"


def token(name: str) -> QColor:
    """Current colour for a token name (``'TEXT_DIM'``); lets widgets store a role instead of a colour."""
    return globals()[name]


def neon_gradient(x0: float, x1: float, a: float = 1.0, c0: QColor | None = None,
                  c1: QColor | None = None) -> QLinearGradient:
    g = QLinearGradient(x0, 0, x1, 0)
    g.setColorAt(0.0, alpha(c0 or CYAN, a))
    g.setColorAt(1.0, alpha(c1 or VIOLET, a))
    return g


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def smooth(cur: float, target: float, dt: float, speed: float) -> float:
    """Frame-rate independent exponential smoothing."""
    return target + (cur - target) * math.exp(-speed * dt)


def rounded(r: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    return path


def paint_shadow(p: QPainter, r: QRectF, radius: float, spread: int = 14, strength: float = 1.0,
                 dy: float = 4.0) -> None:
    """Soft drop shadow built from stacked translucent rounded rects (no QGraphicsEffect, so it is cheap)."""
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    base = 9.0 if not is_light() else 5.0
    for i in range(spread, 0, -1):
        a = base * strength * (1 - i / (spread + 1)) ** 2 + 0.6
        c = QColor(SHADOW)
        c.setAlpha(int(min(255, a)))
        p.setBrush(c)
        p.drawRoundedRect(r.adjusted(-i, -i + dy * 0.4, i, i + dy), radius + i, radius + i)
    p.restore()


def paint_card(p: QPainter, r: QRectF, radius: float = R_LG, hover: float = 0.0, accent: float = 0.0,
               fill: QColor | None = None) -> None:
    """Standard raised surface: subtle vertical gradient + hairline border (+ accent border when selected)."""
    path = rounded(r, radius)
    base = fill or mix(PANEL, PANEL_HI, hover)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0, alpha(base, 0.92 if not is_light() else 1.0))
    g.setColorAt(1, alpha(mix(base, BG0, 0.22 if not is_light() else 0.0), 0.92 if not is_light() else 1.0))
    p.fillPath(path, g)
    if accent > 0.01:
        p.setPen(QPen(neon_gradient(r.left(), r.right(), 0.35 + 0.5 * accent), 1.2))
    else:
        p.setPen(QPen(mix(STROKE, STROKE_STRONG, hover), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


def paint_focus_ring(p: QPainter, r: QRectF, radius: float) -> None:
    """Keyboard focus indicator: 2 px accent ring just outside the control, with a soft halo."""
    p.save()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(alpha(CYAN, 0.28), 4.0))
    p.drawRoundedRect(r, radius, radius)
    p.setPen(QPen(alpha(CYAN, 0.95), 1.6))
    p.drawRoundedRect(r, radius, radius)
    p.restore()


def glass_panel(p: QPainter, r: QRectF, radius: float, accent: QColor | None = None, glow: float = 0.0) -> None:
    """Floating glass panel (menus, dialogs): themed fill with a hairline gradient border."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = rounded(r, radius)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    if is_light():
        g.setColorAt(0, QColor(255, 255, 255, 252))
        g.setColorAt(1, QColor(244, 246, 253, 252))
    else:
        g.setColorAt(0, QColor(27, 31, 52, 250))
        g.setColorAt(1, QColor(13, 15, 26, 252))
    p.fillPath(path, g)
    bg = QLinearGradient(r.topLeft(), r.bottomRight())
    bg.setColorAt(0, alpha(accent or CYAN, 0.50 + glow * 0.3))
    bg.setColorAt(0.5, STROKE)
    bg.setColorAt(1, alpha(VIOLET, 0.45 + glow * 0.3))
    p.setPen(QPen(bg, 1.0))
    p.drawPath(path)


def scheme() -> str:
    """The palette in use, as a word a stylesheet selector can test: ``dark``, ``light`` or ``contrast``."""
    return "contrast" if HIGH_CONTRAST else RESOLVED


_TIP_BOX = "padding: 5px 8px; border-radius: 6px;"


def _tooltip_css(panel: QColor, text: QColor, stroke: QColor) -> str:
    return f"background: {css(panel)}; color: {css(text)}; border: 1px solid {css(stroke)}; {_TIP_BOX}"


def scroll_bar_qss() -> str:
    """Stylesheet of a scroll bar, in the current theme."""
    return f"""
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 4px 3px 4px 3px; }}
QScrollBar::handle:vertical {{ background: {css(alpha(TEXT_DIM, 0.35))}; border-radius: 3px; min-height: 36px; }}
QScrollBar::handle:vertical:hover {{ background: {css(alpha(CYAN, 0.75))}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 3px 4px 3px 4px; }}
QScrollBar::handle:horizontal {{ background: {css(alpha(TEXT_DIM, 0.35))}; border-radius: 3px; min-width: 36px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; width: 0; height: 0; }}
"""


def scroll_area_qss() -> str:
    """Stylesheet of a scrolling area (``widgets.ScrollArea``): the same text whatever the theme, so it is set
    once and a theme switch does not make Qt polish every row below the area again.

    The tooltips of the widgets inside take their colours from the rule for the area's ``scheme`` property; Qt
    works a tooltip's style out anew each time one is shown, so they follow a live switch. The contrast rule names
    palette roles because those colours come from Windows (see ``_contrast_palette``)."""
    tips = "".join(f'QScrollArea[scheme="{pal.name}"] QToolTip '
                   f'{{ {_tooltip_css(pal.PANEL_HI, pal.TEXT, pal.STROKE_STRONG)} }}\n' for pal in (DARK, LIGHT))
    return (tips + 'QScrollArea[scheme="contrast"] QToolTip { background: palette(window); '
                   f'color: palette(window-text); border: 1px solid palette(window-text); {_TIP_BOX} }}\n'
                   "QScrollArea { background: transparent; border: none; }\n"
                   "QScrollArea > QWidget > QWidget { background: transparent; }\n")


def qss() -> str:
    """Stylesheet for the stock text inputs (with their own scroll bars and tooltips), in the current theme."""
    sel = css(alpha(CYAN, 0.32))
    return f"""
QToolTip {{ {_tooltip_css(PANEL_HI, TEXT, STROKE_STRONG)} }}
QLineEdit, QPlainTextEdit {{ background: {css(SUNK)}; color: {css(TEXT)}; border: 1px solid {css(STROKE)};
    border-radius: {R_MD}px; padding: 7px 11px; selection-background-color: {sel};
    selection-color: {css(TEXT)}; placeholder-text-color: {css(TEXT_FAINT)}; }}
QLineEdit:hover, QPlainTextEdit:hover {{ border-color: {css(STROKE_STRONG)}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border: 1.5px solid {css(CYAN)}; }}
QLineEdit:disabled, QPlainTextEdit:disabled {{ color: {css(TEXT_FAINT)}; }}
QLineEdit[invalid="true"] {{ border: 1.5px solid {css(RED)}; }}
""" + scroll_bar_qss()


apply_theme("dark")
