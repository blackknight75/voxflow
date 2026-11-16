"""Core controls of the design system. Everything is custom painted from theme tokens, keyboard reachable,
carries an accessible name, shows a focus ring for keyboard focus and honours reduced motion.

Button / IconButton / Toggle / Segmented / Slider / ChoiceButton + PopupMenu / KeyCaps / HotkeyRecorder /
LevelMeter / Spinner / ProgressBar / LineEdit / SearchField / TextArea / ScrollArea / Label.
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QGuiApplication, QLinearGradient, QPainter, QPainterPath,
                           QPalette, QPen)
from PySide6.QtWidgets import (QAbstractButton, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QScrollArea,
                               QSizePolicy, QSlider, QToolTip, QVBoxLayout, QWidget)

from . import theme as T
from .icons import icon as make_qicon
from .icons import paint_icon

_KBD_REASONS = (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason, Qt.FocusReason.ShortcutFocusReason)


def painter(widget: QWidget) -> QPainter:
    p = QPainter(widget)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    return p


def bring_to_front(window: QWidget) -> None:
    """Show, raise and focus a top-level window (the offscreen test platform cannot raise; skip it there)."""
    window.show()
    if QGuiApplication.platformName() != "offscreen":
        window.raise_()
    window.activateWindow()


def style_tooltips() -> None:
    """Tooltips follow the theme (called by the shell on start and on every theme change)."""
    pal = QToolTip.palette()
    pal.setColor(QPalette.ColorRole.ToolTipBase, T.PANEL_HI)
    pal.setColor(QPalette.ColorRole.ToolTipText, T.TEXT)
    QToolTip.setPalette(pal)
    QToolTip.setFont(T.font(T.F_SMALL))


# --------------------------------------------------------------------------- text
class Label(QLabel):
    """Themed text. ``role`` is a colour token name so the label follows theme switches."""

    def __init__(self, text: str = "", pt: float = T.F_BODY, role: str = "TEXT", weight=T.W_NORMAL,
                 spacing: float = 0, wrap: bool = True, parent=None, selectable: bool = False):
        super().__init__(text, parent)
        self._role = role
        self.setFont(T.font(pt, weight, spacing))
        self.setWordWrap(wrap)
        self.setTextFormat(Qt.TextFormat.PlainText)
        if selectable:
            self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                         | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self._apply()
        T.bus().changed.connect(self._apply)

    def set_role(self, role: str) -> None:
        self._role = role
        self._apply()

    def _apply(self) -> None:
        pal = self.palette()
        c = T.token(self._role)
        pal.setColor(QPalette.ColorRole.WindowText, c)
        pal.setColor(QPalette.ColorRole.Text, c)
        pal.setColor(QPalette.ColorRole.Highlight, T.alpha(T.CYAN, 0.32))
        pal.setColor(QPalette.ColorRole.HighlightedText, T.TEXT)
        self.setPalette(pal)


class Caption(Label):
    """Small uppercase section caption."""

    def __init__(self, text: str, parent=None):
        super().__init__(text.upper(), T.F_CAPTION, "TEXT_DIM", T.W_BOLD, 1.2, False, parent)


# --------------------------------------------------------------------------- button base
class _Hover(QAbstractButton):
    """Button base: animated 0..1 hover value, keyboard-focus tracking, Enter/Return activation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._hover = 0.0
        self._kbd = False
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(T.EASE)
        self._anim.valueChanged.connect(self._set_hover)
        T.bus().changed.connect(self.update)

    def _set_hover(self, v) -> None:
        self._hover = float(v)
        self.update()

    def _go(self, target: float) -> None:
        self._anim.stop()
        if T.REDUCED_MOTION:
            self._set_hover(target)
            return
        self._anim.setDuration(130)
        self._anim.setStartValue(self._hover)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, e):
        if self.isEnabled():
            self._go(1.0)

    def leaveEvent(self, e):
        self._go(0.0)

    def focusInEvent(self, e):
        self._kbd = e.reason() in _KBD_REASONS
        super().focusInEvent(e)
        self.update()

    def focusOutEvent(self, e):
        self._kbd = False
        super().focusOutEvent(e)
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.isEnabled():
            self.click()
            return
        super().keyPressEvent(e)

    def changeEvent(self, e):
        if e.type() == QEvent.Type.EnabledChange:
            if not self.isEnabled():
                self._hover = 0.0
            self.setCursor(Qt.CursorShape.PointingHandCursor if self.isEnabled() else Qt.CursorShape.ArrowCursor)
            self.update()
        super().changeEvent(e)

    def _ring(self, p: QPainter, r: QRectF, radius: float) -> None:
        if self._kbd and self.hasFocus():
            T.paint_focus_ring(p, r, radius)


# --------------------------------------------------------------------------- buttons
class Button(_Hover):
    """Text button. kind: primary (accent gradient) | secondary | ghost | danger."""

    def __init__(self, text: str = "", kind: str = "secondary", icon: str | None = None, parent=None,
                 height: int = T.CONTROL_H, accent: bool = False):
        super().__init__(parent)
        self.kind = "primary" if accent else kind
        self.icon_name = icon
        self.setText(text)
        self.setAccessibleName(text)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._font = T.font(T.F_SMALL + 0.3, T.W_SEMI, 0.15)
        self._flash = 0.0
        self._base_text: str | None = None
        self._fa = QVariantAnimation(self)
        self._fa.setDuration(1000)
        self._fa.setStartValue(1.0)
        self._fa.setEndValue(0.0)
        self._fa.valueChanged.connect(self._set_flash)
        self._fa.finished.connect(self._end_flash)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.updateGeometry()
        self.update()

    def _set_flash(self, v) -> None:
        self._flash = float(v)
        self.update()

    def _end_flash(self) -> None:
        self._flash = 0.0
        if self._base_text is not None:
            base, self._base_text = self._base_text, None
            self.setText(base)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        w = fm.horizontalAdvance(self.text()) + 30 + (22 if self.icon_name else 0)
        if not self.text():
            w = self.height()
        return QSize(int(math.ceil(w)), self.height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def flash(self, text: str) -> None:
        """Briefly show a confirmation (e.g. 'Copied') with a green wash, then restore the label."""
        if self._base_text is None:
            self._base_text = self.text()
        super().setText(text)
        self._fa.stop()
        self._flash = 1.0
        self._fa.start()

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.42)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        rad = min(T.R_MD, r.height() / 2)
        path = T.rounded(r, rad)
        down = self.isDown()
        h = self._hover
        if self.kind == "primary":
            p.fillPath(path, T.neon_gradient(r.left(), r.right(), 0.86 + 0.14 * h))
            if h > 0:
                p.fillPath(path, QColor(255, 255, 255, int(26 * h)))
            col = T.ON_ACCENT
        elif self.kind == "danger":
            p.fillPath(path, T.alpha(T.RED, 0.12 + 0.12 * h))
            p.setPen(QPen(T.alpha(T.RED, 0.45 + 0.4 * h), 1))
            p.drawPath(path)
            col = T.RED
        elif self.kind == "ghost":
            if h > 0:
                p.fillPath(path, T.alpha(T.CONTROL_HI, 0.75 * h))
            col = T.mix(T.TEXT_DIM, T.TEXT, h)
        else:
            p.fillPath(path, T.mix(T.CONTROL, T.CONTROL_HI, h))
            p.setPen(QPen(T.mix(T.STROKE, T.STROKE_STRONG, h), 1))
            p.drawPath(path)
            col = T.TEXT
        if down:
            p.fillPath(path, QColor(0, 0, 0, 46) if not T.is_light() else QColor(20, 30, 80, 30))
        if self._flash > 0:
            p.fillPath(path, T.alpha(T.GREEN, 0.26 * self._flash))
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        tw = fm.horizontalAdvance(self.text())
        iw = 16 if self.icon_name else 0
        gap = 7 if (self.icon_name and self.text()) else 0
        x = r.center().x() - (tw + iw + gap) / 2
        if self.icon_name:
            paint_icon(p, self.icon_name, QRectF(x, r.center().y() - 8, 16, 16), col)
            x += iw + gap
        p.setPen(col)
        p.drawText(QRectF(x, r.top(), tw + 2, r.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        self._ring(p, r, rad)


GlassButton = Button     # historic name


class IconButton(_Hover):
    """Square icon-only button with a tooltip (which doubles as its accessible name)."""

    def __init__(self, icon: str, tooltip: str, parent=None, size: int = 32, role: str = "TEXT_DIM",
                 danger: bool = False):
        super().__init__(parent)
        self.icon_name, self.role, self.danger = icon, role, danger
        self.setFixedSize(size, size)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self._ok = 0.0
        self._oa = QVariantAnimation(self)
        self._oa.setDuration(1100)
        self._oa.setStartValue(1.0)
        self._oa.setEndValue(0.0)
        self._oa.valueChanged.connect(self._set_ok)

    def _set_ok(self, v) -> None:
        self._ok = float(v)
        self.update()

    def confirm(self) -> None:
        """Flash a green check in place of the icon (copy / paste feedback)."""
        self._oa.stop()
        self._ok = 1.0
        self._oa.start()

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.4)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        path = T.rounded(r, T.R_SM)
        h = self._hover
        hi = T.RED if self.danger else T.TEXT
        if h > 0:
            p.fillPath(path, T.alpha(T.RED, 0.14 * h) if self.danger else T.alpha(T.CONTROL_HI, 0.9 * h))
        if self.isDown():
            p.fillPath(path, QColor(0, 0, 0, 40))
        s = min(18.0, r.width() - 10)
        ir = QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s)
        if self._ok > 0.25:
            paint_icon(p, "check", ir, T.GREEN, 2.0)
        else:
            paint_icon(p, self.icon_name, ir, T.mix(T.token(self.role), hi, h))
        self._ring(p, r, T.R_SM)


class Toggle(_Hover):
    def __init__(self, checked: bool = False, parent=None, name: str = ""):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(46, 28)
        if name:
            self.setAccessibleName(name)
        self._pos = 1.0 if checked else 0.0
        self._slide = QVariantAnimation(self)
        self._slide.setEasingCurve(T.EASE)
        self._slide.valueChanged.connect(self._set_pos)
        self.toggled.connect(self._on_toggled)

    def _set_pos(self, v) -> None:
        self._pos = float(v)
        self.update()

    def _on_toggled(self, on: bool) -> None:
        self._slide.stop()
        if T.REDUCED_MOTION:
            self._set_pos(1.0 if on else 0.0)
            return
        self._slide.setDuration(170)
        self._slide.setStartValue(self._pos)
        self._slide.setEndValue(1.0 if on else 0.0)
        self._slide.start()

    def set_checked_silent(self, on: bool) -> None:
        self._slide.stop()
        self.blockSignals(True)
        self.setChecked(bool(on))
        self.blockSignals(False)
        self._pos = 1.0 if on else 0.0
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.42)
        r = QRectF(2.5, 3.5, self.width() - 5, self.height() - 7)
        path = T.rounded(r, r.height() / 2)
        p.fillPath(path, T.mix(T.CONTROL, T.CONTROL_HI, self._hover))
        if self._pos > 0.01:
            p.fillPath(path, T.neon_gradient(r.left(), r.right(), self._pos))
        p.setPen(QPen(T.STROKE_STRONG if self._pos < 0.5 else T.alpha(T.CYAN, 0.25), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        d = r.height() - 6
        x = r.left() + 3 + (r.width() - d - 6) * self._pos
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(QRectF(x, r.top() + 4, d, d))
        p.setBrush(T.mix(T.TEXT_DIM if not T.is_light() else QColor(255, 255, 255), QColor(255, 255, 255), self._pos))
        p.drawEllipse(QRectF(x, r.top() + 3, d, d))
        self._ring(p, r.adjusted(-1.5, -1.5, 1.5, 1.5), r.height() / 2 + 1.5)


class CheckBox(_Hover):
    """Tick box with a label (and an optional quieter note after it). Space or Enter toggles."""

    def __init__(self, text: str, checked: bool = True, note: str = "", parent=None):
        super().__init__(parent)
        self.note = note
        self.setCheckable(True)
        self.setChecked(checked)
        self.setText(text)
        self.setAccessibleName(text)
        self.setAccessibleDescription(note)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._font = T.font(T.F_BODY)
        self._note_font = T.font(T.F_SMALL)
        self.toggled.connect(self.update)

    def sizeHint(self) -> QSize:
        w = 28 + QFontMetricsF(self._font).horizontalAdvance(self.text())
        if self.note:
            w += 8 + QFontMetricsF(self._note_font).horizontalAdvance(self.note)
        return QSize(int(math.ceil(w)) + 6, 28)

    def minimumSizeHint(self) -> QSize:
        return QSize(60, 28)

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.42)
        box = QRectF(2.5, self.height() / 2 - 8.5, 17, 17)
        path = T.rounded(box, 5)
        if self.isChecked():
            p.fillPath(path, T.neon_gradient(box.left(), box.right()))
            paint_icon(p, "check", box.adjusted(2.5, 2.5, -2.5, -2.5), T.ON_ACCENT, 2.2)
        else:
            p.fillPath(path, T.mix(T.SUNK, T.CONTROL, self._hover))
            p.setPen(QPen(T.mix(T.STROKE_STRONG, T.CYAN, self._hover), 1.2))
            p.drawPath(path)
        x = 28.0
        avail = self.width() - x - 2
        fm = QFontMetricsF(self._font)
        text = fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, avail)
        p.setFont(self._font)
        p.setPen(T.TEXT)
        p.drawText(QRectF(x, 0, avail, self.height()), Qt.AlignmentFlag.AlignVCenter, text)
        used = fm.horizontalAdvance(text) + 8
        if self.note and avail - used > 30:
            nf = QFontMetricsF(self._note_font)
            p.setFont(self._note_font)
            p.setPen(T.TEXT_DIM)
            p.drawText(QRectF(x + used, 0, avail - used, self.height()), Qt.AlignmentFlag.AlignVCenter,
                       nf.elidedText(self.note, Qt.TextElideMode.ElideRight, avail - used))
        self._ring(p, box.adjusted(-1.5, -1.5, 1.5, 1.5), 6)


# --------------------------------------------------------------------------- segmented control
class Segmented(QWidget):
    """Mutually exclusive options in one track. Left/Right arrows move the selection."""

    changed = Signal(object)

    def __init__(self, options: list[tuple[object, str]], value=None, parent=None, name: str = ""):
        super().__init__(parent)
        self.options = list(options)
        self.value = value if value is not None or not options else options[0][0]
        self.setFixedHeight(T.CONTROL_H)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setAccessibleName(name)
        self._font = T.font(T.F_SMALL, T.W_SEMI, 0.1)
        self._kbd = False
        self._hover = -1
        self._pos = float(self._index())
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(T.EASE)
        self._anim.valueChanged.connect(self._set_pos)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        T.bus().changed.connect(self.update)

    def _index(self) -> int:
        for i, (v, _t) in enumerate(self.options):
            if v == self.value:
                return i
        return 0

    def _set_pos(self, v) -> None:
        self._pos = float(v)
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        seg = max([fm.horizontalAdvance(t) + 26 for _v, t in self.options] or [60])
        return QSize(int(seg * max(1, len(self.options)) + 6), T.CONTROL_H)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def set_value(self, value, emit: bool = False) -> None:
        if value == self.value:
            return
        self.value = value
        self._anim.stop()
        if T.REDUCED_MOTION or not self.isVisible():
            self._set_pos(float(self._index()))
        else:
            self._anim.setDuration(180)
            self._anim.setStartValue(self._pos)
            self._anim.setEndValue(float(self._index()))
            self._anim.start()
        if emit:
            self.changed.emit(value)

    def _seg_at(self, x: float) -> int:
        n = max(1, len(self.options))
        return max(0, min(n - 1, int((x - 3) / ((self.width() - 6) / n))))

    def mouseMoveEvent(self, e):
        i = self._seg_at(e.position().x())
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, e):
        self._hover = -1
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.options:
            self.set_value(self.options[self._seg_at(e.position().x())][0], emit=True)

    def keyPressEvent(self, e):
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1, Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1}.get(e.key())
        if step is None or not self.options:
            return super().keyPressEvent(e)
        i = max(0, min(len(self.options) - 1, self._index() + step))
        self.set_value(self.options[i][0], emit=True)

    def focusInEvent(self, e):
        self._kbd = e.reason() in _KBD_REASONS
        self.update()

    def focusOutEvent(self, e):
        self._kbd = False
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.42)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        p.fillPath(T.rounded(r, T.R_MD), T.SUNK)
        p.setPen(QPen(T.STROKE, 1))
        p.drawRoundedRect(r, T.R_MD, T.R_MD)
        n = max(1, len(self.options))
        sw = (r.width() - 4) / n
        thumb = QRectF(r.left() + 2 + sw * self._pos, r.top() + 2, sw, r.height() - 4)
        tp = T.rounded(thumb, T.R_MD - 2)
        p.fillPath(tp, T.CONTROL_HI if not T.is_light() else T.PANEL)
        p.setPen(QPen(T.neon_gradient(thumb.left(), thumb.right(), 0.75), 1.1))
        p.drawPath(tp)
        p.setFont(self._font)
        cur = self._index()
        for i, (_v, text) in enumerate(self.options):
            sr = QRectF(r.left() + 2 + sw * i, r.top(), sw, r.height())
            p.setPen(T.TEXT if i == cur else (T.TEXT if i == self._hover else T.TEXT_DIM))
            p.drawText(sr, Qt.AlignmentFlag.AlignCenter, text)
        if self._kbd and self.hasFocus():
            T.paint_focus_ring(p, r, T.R_MD)


# --------------------------------------------------------------------------- slider
class Slider(QSlider):
    """Horizontal slider with an accent-filled track. Click anywhere on the track to jump there."""

    def __init__(self, lo: int = 0, hi: int = 100, value: int = 50, parent=None, name: str = ""):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(lo, hi)
        self.setValue(value)
        self.setFixedHeight(28)
        self.setMinimumWidth(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(name)
        self._kbd = False
        T.bus().changed.connect(self.update)

    def _value_at(self, x: float) -> int:
        span = max(1.0, self.width() - 20.0)
        f = max(0.0, min(1.0, (x - 10.0) / span))
        return int(round(self.minimum() + f * (self.maximum() - self.minimum())))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.setValue(self._value_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.setValue(self._value_at(e.position().x()))

    def mouseReleaseEvent(self, e):
        was_down = self.isSliderDown()
        self.setSliderDown(False)               # Qt emits sliderReleased itself when the slider was down
        if not was_down:
            self.sliderReleased.emit()

    def focusInEvent(self, e):
        self._kbd = e.reason() in _KBD_REASONS
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self._kbd = False
        super().focusOutEvent(e)

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.42)
        cy = self.height() / 2
        x0, x1 = 10.0, self.width() - 10.0
        f = (self.value() - self.minimum()) / max(1, self.maximum() - self.minimum())
        xv = x0 + (x1 - x0) * f
        track = QRectF(x0, cy - 2.5, x1 - x0, 5)
        p.fillPath(T.rounded(track, 2.5), T.CONTROL)
        if xv > x0 + 0.5:
            p.fillPath(T.rounded(QRectF(x0, cy - 2.5, xv - x0, 5), 2.5), T.neon_gradient(x0, x1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawEllipse(QPointF(xv, cy + 1), 8, 8)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QPointF(xv, cy), 7.5, 7.5)
        p.setBrush(T.mix(T.CYAN, T.VIOLET, f))
        p.drawEllipse(QPointF(xv, cy), 3.2, 3.2)
        if self._kbd and self.hasFocus():
            T.paint_focus_ring(p, QRectF(xv - 10, cy - 10, 20, 20), 10)


# --------------------------------------------------------------------------- logo + keycaps
def paint_logo(p: QPainter, r: QRectF, c0: QColor | None = None, c1: QColor | None = None) -> None:
    """The VoxFlow mark: rounded gradient tile with a mirrored mini waveform."""
    p.fillPath(T.rounded(r, r.width() * 0.29), T.neon_gradient(r.left(), r.right(), 1.0, c0 or T.DARK.CYAN,
                                                               c1 or T.DARK.VIOLET))
    heights = [0.22, 0.5, 0.82, 0.5, 0.3]
    bw = r.width() * 0.085
    gap = r.width() * 0.135
    total = len(heights) * bw + (len(heights) - 1) * gap
    x = r.center().x() - total / 2
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(8, 10, 20))
    for h in heights:
        hh = r.height() * 0.6 * h
        p.drawRoundedRect(QRectF(x, r.center().y() - hh / 2, bw, hh), bw / 2, bw / 2)
        x += bw + gap


def pretty_hotkey(combo: str) -> str:
    names = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win", "space": "Space", "esc": "Esc"}
    return " + ".join(names.get(k, k.upper() if len(k) <= 3 else k.title()) for k in (combo or "").split("+") if k)


class KeyCaps(QWidget):
    """Renders a hotkey combo as keycaps, or a hint text while a new combo is being recorded."""

    H = 32

    def __init__(self, combo: str = "ctrl+win", parent=None):
        super().__init__(parent)
        self.combo, self.hint = combo, ""
        self._font = T.font(T.F_SMALL + 0.2, T.W_SEMI)
        self.setFixedHeight(self.H)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        T.bus().changed.connect(self.update)

    def set_combo(self, combo: str, hint: str = "") -> None:
        self.combo, self.hint = combo, hint
        self.updateGeometry()
        self.update()

    def _labels(self) -> list[str]:
        return [s for s in pretty_hotkey(self.combo).split(" + ") if s]

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        if self.hint:
            return QSize(int(fm.horizontalAdvance(self.hint)) + 8, self.H)
        labels = self._labels()
        if not labels:
            return QSize(int(fm.horizontalAdvance("Not set")) + 8, self.H)
        return QSize(int(sum(max(30.0, fm.horizontalAdvance(s) + 20) + 6 for s in labels)) - 3, self.H)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, _):
        p = painter(self)
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        full = QRectF(0, 0, self.width(), self.height())
        if self.hint:
            p.setPen(T.CYAN)
            p.drawText(full, Qt.AlignmentFlag.AlignVCenter, self.hint)
            return
        labels = self._labels()
        if not labels:
            p.setPen(T.TEXT_FAINT)
            p.drawText(full, Qt.AlignmentFlag.AlignVCenter, "Not set")
            return
        x = 1.0
        for s in labels:
            w = max(30.0, fm.horizontalAdvance(s) + 20)
            r = QRectF(x, 3.5, w, self.height() - 9)
            path = T.rounded(r, 7)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T.alpha(T.SHADOW, 0.35 if not T.is_light() else 0.08))
            p.drawRoundedRect(r.translated(0, 2), 7, 7)
            g = QLinearGradient(r.topLeft(), r.bottomLeft())
            g.setColorAt(0, T.PANEL if T.is_light() else T.CONTROL_HI)      # light theme: white key, not a lavender blob
            g.setColorAt(1, T.CONTROL)
            p.fillPath(path, g)
            p.setPen(QPen(T.STROKE_STRONG, 1))
            p.drawPath(path)
            p.setPen(T.TEXT)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, s)
            x += w + 6


_QT_KEY_NAMES = {
    Qt.Key.Key_Control: "ctrl", Qt.Key.Key_Alt: "alt", Qt.Key.Key_AltGr: "alt", Qt.Key.Key_Shift: "shift",
    Qt.Key.Key_Meta: "win", Qt.Key.Key_Super_L: "win", Qt.Key.Key_Super_R: "win", Qt.Key.Key_Space: "space",
    Qt.Key.Key_Return: "enter", Qt.Key.Key_Enter: "enter", Qt.Key.Key_Tab: "tab", Qt.Key.Key_Backspace: "backspace",
    Qt.Key.Key_CapsLock: "capslock", Qt.Key.Key_Insert: "insert", Qt.Key.Key_Delete: "delete",
    Qt.Key.Key_Home: "home", Qt.Key.Key_End: "end", Qt.Key.Key_PageUp: "pageup", Qt.Key.Key_PageDown: "pagedown",
    Qt.Key.Key_Left: "left", Qt.Key.Key_Up: "up", Qt.Key.Key_Right: "right", Qt.Key.Key_Down: "down",
    Qt.Key.Key_QuoteLeft: "`", Qt.Key.Key_Minus: "-", Qt.Key.Key_Equal: "=", Qt.Key.Key_BracketLeft: "[",
    Qt.Key.Key_BracketRight: "]", Qt.Key.Key_Semicolon: ";", Qt.Key.Key_Apostrophe: "'", Qt.Key.Key_Comma: ",",
    Qt.Key.Key_Period: ".", Qt.Key.Key_Slash: "/", Qt.Key.Key_Backslash: "\\",
}
_MOD_ORDER = ["ctrl", "alt", "shift", "win"]


def qt_key_token(key: int) -> str | None:
    """Qt key code -> hotkey token understood by voxflow.hotkey.parse_hotkey (None if unsupported)."""
    if key in _QT_KEY_NAMES:
        return _QT_KEY_NAMES[key]
    if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F24:
        return f"f{key - Qt.Key.Key_F1 + 1}"
    if Qt.Key.Key_A <= key <= Qt.Key.Key_Z or Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        return chr(key).lower()
    return None


def combo_from_tokens(tokens) -> str:
    mods = sorted({t for t in tokens if t in _MOD_ORDER}, key=_MOD_ORDER.index)
    others = [t for t in dict.fromkeys(tokens) if t not in _MOD_ORDER]
    return "+".join(mods + others)


class HotkeyRecorder(QWidget):
    """Keycaps + Change/Cancel + inline validation message.

    Recording works two ways: the app's global hook (``capture_requested`` -> the owner later calls
    :meth:`captured`) and, as a fallback that also works without the hook, plain Qt key events while this
    widget has focus. ``validator(combo) -> str | None`` returns a human message for a conflicting combo.
    """

    capture_requested = Signal()
    capture_cancelled = Signal()
    changed = Signal(str)

    def __init__(self, combo: str = "", name: str = "Hotkey", validator=None, allow_clear: bool = False,
                 parent=None):
        super().__init__(parent)
        self.combo = combo
        self.validator = validator
        self.use_qt_keys = True           # set False when a global hook delivers the capture instead
        self._capturing = False
        self._held: list[str] = []
        self._seen: list[str] = []
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName(name)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.caps = KeyCaps(combo)
        self.btn = Button("Change", height=30)
        self.btn.setAccessibleName(f"Change {name.lower()}")
        self.btn.clicked.connect(self.toggle_capture)
        self.btn_clear = IconButton("close", f"Clear {name.lower()}", size=30)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_clear.setVisible(allow_clear and bool(combo))
        self._allow_clear = allow_clear
        row.addStretch(1)
        policy = self.btn_clear.sizePolicy()
        policy.setRetainSizeWhenHidden(True)      # the slot stays even when empty: keycaps, Change and x line up per row
        self.btn_clear.setSizePolicy(policy)
        row.addWidget(self.caps)
        row.addWidget(self.btn)
        row.addWidget(self.btn_clear)            # fixed slot after Change, so the x column is aligned in every row
        col.addLayout(row)
        self.msg = Label("", T.F_CAPTION, "RED", wrap=False)
        self.msg.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.msg.hide()
        col.addWidget(self.msg)
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(self.msg.hide)

    def setObjectName(self, name: str) -> None:
        super().setObjectName(name)
        self.btn.setObjectName(f"{name}_change")
        self.btn_clear.setObjectName(f"{name}_clear")

    @property
    def capturing(self) -> bool:
        return self._capturing

    def set_combo(self, combo: str) -> None:
        self.combo = combo or ""
        if not self._capturing:
            self.caps.set_combo(self.combo)
        self.btn_clear.setVisible(self._allow_clear and bool(self.combo) and not self._capturing)

    def show_message(self, text: str, role: str = "RED", ms: int = 5000) -> None:
        self.msg.set_role(role)
        self.msg.setText(text)
        self.msg.show()
        self._msg_timer.start(ms)

    def toggle_capture(self) -> None:
        if self._capturing:
            self._finish(None)
            self.capture_cancelled.emit()
            return
        self._capturing = True
        self._held, self._seen = [], []
        self.msg.hide()
        self.btn.setText("Cancel")
        self.btn_clear.hide()
        self.caps.set_combo(self.combo, "Press the new combo…")
        self.btn.setFocus(Qt.FocusReason.OtherFocusReason)
        self.btn.installEventFilter(self)
        self.capture_requested.emit()

    def captured(self, combo: str | None) -> None:
        """Result from the global hook (None = nothing captured / cancelled)."""
        if self._capturing:
            self._finish(combo)

    def _clear(self) -> None:
        self.combo = ""
        self.caps.set_combo("")
        self.btn_clear.hide()
        self.changed.emit("")

    def _finish(self, combo: str | None) -> None:
        self._capturing = False
        self.btn.removeEventFilter(self)
        self.btn.setText("Change")
        if combo:
            problem = self.validator(combo) if self.validator else None
            if problem:
                self.show_message(problem)
            else:
                self.combo = combo
                self.msg.hide()
                self.changed.emit(combo)
        self.caps.set_combo(self.combo)
        self.btn_clear.setVisible(self._allow_clear and bool(self.combo))

    def eventFilter(self, obj, ev):
        if not self._capturing or obj is not self.btn:
            return False
        t = ev.type()
        if t == QEvent.Type.FocusOut:
            self._finish(None)
            self.capture_cancelled.emit()
            return False
        if t not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.ShortcutOverride):
            return False
        if t == QEvent.Type.ShortcutOverride:
            ev.accept()
            return True
        if ev.isAutoRepeat():
            return True
        if t == QEvent.Type.KeyPress and ev.key() == Qt.Key.Key_Escape:
            self._finish(None)
            self.capture_cancelled.emit()
            return True
        if not self.use_qt_keys:
            return True
        tok = qt_key_token(ev.key())
        if tok is None:
            return True
        if t == QEvent.Type.KeyPress:
            if tok not in self._held:
                self._held.append(tok)
            if tok not in self._seen:
                self._seen.append(tok)
            self.caps.set_combo(combo_from_tokens(self._seen))
        else:
            if tok in self._held:
                self._held.remove(tok)
            if not self._held and self._seen:
                self._finish(combo_from_tokens(self._seen))
        return True


# --------------------------------------------------------------------------- popup menu + dropdown
class PopupMenu(QWidget):
    """Custom-painted popup list used for the tray menu and dropdowns. Keyboard: arrows, Home/End, Enter, Esc.

    items: dicts with ``kind`` in {header, status, action, toggle, choice, separator, info} and optional
    ``text, sub, checked, enabled, on_click, dot (QColor), keep_open, icon``.
    """

    closed = Signal()

    ROW_H = {"header": 54, "status": 48, "action": 36, "toggle": 38, "choice": 34, "separator": 11, "info": 30}
    _INTERACTIVE = ("action", "toggle", "choice")

    def __init__(self, width: int = 272):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setAccessibleName("Menu")
        self._w = width
        self._items: list[dict] = []
        self._rects: list[QRect] = []
        self._hover = -1
        self._scroll = 0
        self._content_h = 0
        self._max_h = 0
        self._appear = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(T.EASE)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._on_anim)
        self._margin = 14

    def _on_anim(self, v) -> None:
        self._appear = float(v)
        self.update()

    def items(self) -> list[dict]:
        return self._items

    def set_width(self, width: int) -> None:
        self._w = int(width)
        self._layout()

    def set_items(self, items: list[dict]) -> None:
        self._items = items
        self._hover = -1
        self.setAccessibleName("Menu")
        self._scroll = 0
        self._layout()

    def _layout(self) -> None:
        m, y = self._margin, self._margin
        self._rects = []
        for it in self._items:
            h = self.ROW_H.get(it.get("kind"), self.ROW_H["info"])
            self._rects.append(QRect(m, y, self._w - 2 * m, h))
            y += h
        self._content_h = y + m
        cap = self._max_h or self._content_h
        self.setFixedSize(self._w, min(self._content_h, cap))
        self.update()

    def _fit_screen(self, pos: QPoint):
        screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        g = screen.availableGeometry()
        self._max_h = max(160, int(g.height() * 0.7))
        self._layout()
        return g

    def _show(self) -> None:
        self._appear = 1.0 if T.REDUCED_MOTION else 0.0
        self.show()
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        if not T.REDUCED_MOTION:
            self._anim.setDuration(140)
            self._anim.start()

    def popup_at(self, pos: QPoint, anchor_bottom: bool = True) -> None:
        """Centre horizontally on ``pos``; above it (tray) or below it."""
        g = self._fit_screen(pos)
        x = min(max(g.left() + 4, pos.x() - self.width() // 2), g.right() - self.width() - 4)
        y = pos.y() - self.height() - 6 if anchor_bottom else pos.y() + 6
        self.move(x, min(max(g.top() + 4, y), g.bottom() - self.height() - 4))
        self._show()

    def popup_below(self, widget: QWidget) -> None:
        """Left-align under ``widget`` (dropdowns); flips above when there is no room below."""
        top_left = widget.mapToGlobal(QPoint(0, 0))
        g = self._fit_screen(top_left)
        x = min(max(g.left() + 4, top_left.x() - self._margin + 4), g.right() - self.width() - 4)
        y = top_left.y() + widget.height() - self._margin + 8
        if y + self.height() > g.bottom() - 4:
            y = top_left.y() - self.height() + self._margin - 8
        self.move(x, max(g.top() + 4, y))
        self._show()

    # ---- interaction
    def _usable(self, i: int) -> bool:
        return 0 <= i < len(self._items) and self._items[i]["kind"] in self._INTERACTIVE \
            and self._items[i].get("enabled", True)

    def _index_at(self, pos: QPoint) -> int:
        pos = QPoint(pos.x(), pos.y() + self._scroll)
        for i, r in enumerate(self._rects):
            if r.contains(pos) and self._usable(i):
                return i
        return -1

    def _set_hover(self, i: int) -> None:
        if i != self._hover:
            self._hover = i
            self.update()
            # The rows are painted, not widgets: the menu's accessible name follows the highlighted row, so UI
            # Automation (a screen reader) reports the choice the arrow keys moved to.
            it = self._items[i] if 0 <= i < len(self._items) else None
            text = str(it.get("text") or "") if it else ""
            if text and it.get("checked"):
                text += ", selected"
            self.setAccessibleName(text or "Menu")

    def _ensure_visible(self, i: int) -> None:
        if not (0 <= i < len(self._rects)):
            return
        r = self._rects[i]
        view_h = self.height()
        if r.top() - self._scroll < self._margin:
            self._scroll = max(0, r.top() - self._margin)
        elif r.bottom() - self._scroll > view_h - self._margin:
            self._scroll = min(self._content_h - view_h, r.bottom() - view_h + self._margin)

    def _move(self, step: int, start: int | None = None) -> None:
        n = len(self._items)
        i = self._hover if start is None else start
        for _ in range(n):
            i = (i + step) % n
            if self._usable(i):
                self._ensure_visible(i)
                self._set_hover(i)
                self.update()
                return

    def _activate(self, i: int) -> None:
        if not self._usable(i):
            return
        it = self._items[i]
        if it["kind"] == "toggle":
            it["checked"] = not it.get("checked", False)
            self.update()
        cb = it.get("on_click")
        if not it.get("keep_open", it["kind"] == "toggle"):
            self.close()
        if cb:
            cb(it.get("checked")) if it["kind"] == "toggle" else cb()

    def mouseMoveEvent(self, e):
        i = self._index_at(e.position().toPoint())
        self.setCursor(Qt.CursorShape.PointingHandCursor if i >= 0 else Qt.CursorShape.ArrowCursor)
        self._set_hover(i)

    def leaveEvent(self, e):
        self._set_hover(-1)

    def mouseReleaseEvent(self, e):
        self._activate(self._index_at(e.position().toPoint()))

    def wheelEvent(self, e):
        if self._content_h > self.height():
            self._scroll = max(0, min(self._content_h - self.height(), self._scroll - e.angleDelta().y() // 2))
            self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Down:
            self._move(1)
        elif k == Qt.Key.Key_Up:
            self._move(-1, None if self._hover >= 0 else 0)
        elif k == Qt.Key.Key_Home:
            self._move(1, -1)
        elif k == Qt.Key.Key_End:
            self._move(-1, 0)
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._activate(self._hover)
        elif k == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.closed.emit()
        super().closeEvent(e)

    # ---- painting
    def paintEvent(self, _):
        p = painter(self)
        p.setOpacity(self._appear)
        p.translate(0, (1 - self._appear) * 6)
        inset = self._margin - 4
        panel = QRectF(inset, inset, self.width() - 2 * inset, self.height() - 2 * inset)
        T.paint_shadow(p, panel, 16, spread=9, strength=1.1, dy=3)
        T.glass_panel(p, panel, 16)
        p.save()
        p.setClipPath(T.rounded(panel.adjusted(1, 1, -1, -1), 15))
        p.translate(0, -self._scroll)
        for i, (it, r) in enumerate(zip(self._items, self._rects)):
            if r.bottom() - self._scroll < 0 or r.top() - self._scroll > self.height():
                continue
            self._paint_row(p, it, QRectF(r), i == self._hover)
        p.restore()
        if self._content_h > self.height():      # scroll affordances: fade the edge and show a chevron
            base = QColor(244, 246, 253) if T.is_light() else QColor(15, 17, 29)
            for at_top in (True, False):
                more = self._scroll > 0 if at_top else self._scroll < self._content_h - self.height()
                if more:
                    band = QRectF(panel.left() + 1, panel.top() + 1 if at_top else panel.bottom() - 31,
                                  panel.width() - 2, 30)
                    fade = QLinearGradient(0, band.top(), 0, band.bottom())
                    fade.setColorAt(0.0 if at_top else 1.0, base)
                    fade.setColorAt(0.45 if at_top else 0.55, T.alpha(base, 0.92))
                    fade.setColorAt(1.0 if at_top else 0.0, T.alpha(base, 0.0))
                    p.save()
                    p.setClipPath(T.rounded(panel.adjusted(1, 1, -1, -1), 15))
                    p.fillRect(band, fade)
                    p.restore()
                    cx, cy = panel.center().x(), (panel.top() + 9 if at_top else panel.bottom() - 9)
                    d = -2.5 if at_top else 2.5
                    p.setPen(QPen(T.TEXT_DIM, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                    p.drawLine(QPointF(cx - 5, cy - d), QPointF(cx, cy + d))
                    p.drawLine(QPointF(cx, cy + d), QPointF(cx + 5, cy - d))

    def _paint_row(self, p: QPainter, it: dict, r: QRectF, hover: bool) -> None:
        k = it["kind"]
        enabled = it.get("enabled", True)
        if k == "separator":
            p.setPen(QPen(T.STROKE, 1))
            p.drawLine(QPointF(r.left() + 6, r.center().y()), QPointF(r.right() - 6, r.center().y()))
            return
        if hover:
            g = QLinearGradient(r.topLeft(), r.topRight())
            g.setColorAt(0, T.alpha(T.CYAN, 0.16))
            g.setColorAt(1, T.alpha(T.VIOLET, 0.16))
            p.fillPath(T.rounded(r.adjusted(-2, 1, 2, -1), 9), g)
        if k == "header":
            paint_logo(p, QRectF(r.left() + 4, r.center().y() - 15, 30, 30))
            p.setPen(T.TEXT)
            p.setFont(T.font(12, T.W_BOLD, 0.3))
            p.drawText(QRectF(r.left() + 44, r.top() + 6, r.width() - 50, 22), Qt.AlignmentFlag.AlignVCenter,
                       it.get("text", "VoxFlow"))
            p.setPen(T.TEXT_DIM)
            p.setFont(T.font(T.F_CAPTION))
            p.drawText(QRectF(r.left() + 44, r.top() + 27, r.width() - 50, 18), Qt.AlignmentFlag.AlignVCenter,
                       it.get("sub", ""))
            return
        if k == "status":
            c = it.get("dot", T.GREEN)
            cx, cy = r.left() + 14, r.center().y()
            p.setPen(Qt.PenStyle.NoPen)
            for rad, a in ((8, 0.12), (5.5, 0.24)):
                p.setBrush(T.alpha(c, a))
                p.drawEllipse(QPointF(cx, cy), rad, rad)
            p.setBrush(c)
            p.drawEllipse(QPointF(cx, cy), 3.2, 3.2)
            p.setPen(T.TEXT)
            p.setFont(T.font(T.F_BODY, T.W_SEMI))
            fm = p.fontMetrics()
            w = int(r.width() - 36)
            p.drawText(QRectF(r.left() + 32, r.top() + 5, w, 20), Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(it["text"], Qt.TextElideMode.ElideRight, w))
            p.setPen(T.TEXT_DIM)
            p.setFont(T.font(T.F_CAPTION))
            fm = p.fontMetrics()
            p.drawText(QRectF(r.left() + 32, r.top() + 25, w, 18), Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(it.get("sub", ""), Qt.TextElideMode.ElideRight, w))
            return
        if k == "info":
            p.setPen(T.TEXT_DIM)
            p.setFont(T.font(T.F_SMALL))
            p.drawText(r.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter, it["text"])
            return
        text_col = T.TEXT if enabled else T.TEXT_FAINT
        left = r.left() + 10
        if it.get("icon"):
            paint_icon(p, it["icon"], QRectF(left, r.center().y() - 8, 16, 16), T.TEXT_DIM if enabled else T.TEXT_FAINT)
            left += 26
        if k == "choice":
            if it.get("checked"):
                paint_icon(p, "check", QRectF(left, r.center().y() - 8, 16, 16), T.CYAN, 2.0)
            left += 24
        p.setFont(T.font(T.F_BODY, T.W_MEDIUM))
        fm = p.fontMetrics()
        sub = it.get("sub", "") if k in ("action", "choice") else ""
        sub_font = T.font(T.F_SMALL)
        sub_w = QFontMetricsF(sub_font).horizontalAdvance(sub) + 12 if sub else 0
        avail = int(r.right() - left - (52 if k == "toggle" else 8) - sub_w)
        p.setPen(text_col)
        p.drawText(QRectF(left, r.top(), avail, r.height()), Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(it["text"], Qt.TextElideMode.ElideRight, avail))
        if sub:
            p.setPen(T.TEXT_DIM)
            p.setFont(sub_font)
            p.drawText(QRectF(r.left(), r.top(), r.width() - 10, r.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, sub)
        if k == "toggle":
            on = it.get("checked", False)
            tr = QRectF(r.right() - 42, r.center().y() - 9, 34, 18)
            p.fillPath(T.rounded(tr, 9), T.neon_gradient(tr.left(), tr.right(), 1.0) if on else T.CONTROL_HI)
            d = 12
            x = tr.right() - d - 3 if on else tr.left() + 3
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255) if on else T.TEXT_DIM)
            p.drawEllipse(QRectF(x, tr.center().y() - d / 2, d, d))


class ChoiceButton(_Hover):
    """Dropdown: shows the current option and opens a PopupMenu of choices.

    ``options``: ``(value, text)`` pairs, or ``(value, text, note, enabled)``: the note is shown dimmed at the
    right of the choice, and a choice that is not enabled cannot be picked."""

    chosen = Signal(object)

    def __init__(self, parent=None, name: str = "", fallback: str = "System default"):
        super().__init__(parent)
        self.setFixedHeight(T.CONTROL_H)
        self.setMinimumWidth(150)
        self.setAccessibleName(name)
        self.fallback = fallback
        self.options: list[tuple[object, str]] = []
        self.value = None
        self._menu: PopupMenu | None = None
        self._font = T.font(T.F_BODY - 0.2)
        self.clicked.connect(self.open_menu)

    def set_options(self, options, value) -> None:
        self.options, self.value = list(options), value
        self._describe()
        self.updateGeometry()
        self.update()

    def set_value(self, value) -> None:
        self.value = value
        self._describe()
        self.update()

    def _describe(self) -> None:
        """The current choice is painted; UI Automation (a screen reader) gets it as the description (HelpText)."""
        self.setAccessibleDescription(self.label())

    def label(self) -> str:
        for o in self.options:
            if o[0] == self.value:
                return o[1]
        return self.fallback

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        w = max([fm.horizontalAdvance(o[1]) for o in self.options] or [90]) + 50
        return QSize(int(min(360, max(150, w))), T.CONTROL_H)

    def open_menu(self) -> PopupMenu:
        if self._menu is None:
            self._menu = PopupMenu()
        m = self._menu
        m.set_width(max(self.width() + 20, 220))
        items = []
        for o in self.options:
            v, t = o[0], o[1]
            note, enabled = (o[2], bool(o[3])) if len(o) >= 4 else ("", True)
            items.append({"kind": "choice", "text": t, "sub": note, "enabled": enabled, "checked": v == self.value,
                          "on_click": (lambda v=v: self._pick(v))})
        m.set_items(items or [{"kind": "info", "text": "Nothing to choose from"}])
        m.popup_below(self)
        for i, o in enumerate(self.options):
            if o[0] == self.value:
                m._ensure_visible(i)
                m._set_hover(i)
        return m

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            self.open_menu()
            return
        super().keyPressEvent(e)

    def _pick(self, v) -> None:
        self.value = v
        self._describe()
        self.update()
        self.chosen.emit(v)

    def paintEvent(self, _):
        p = painter(self)
        if not self.isEnabled():
            p.setOpacity(0.42)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        path = T.rounded(r, T.R_MD)
        p.fillPath(path, T.mix(T.SUNK, T.CONTROL, self._hover))
        p.setPen(QPen(T.mix(T.STROKE, T.STROKE_STRONG, self._hover), 1))
        p.drawPath(path)
        p.setPen(T.TEXT)
        p.setFont(self._font)
        avail = int(r.width() - 42)
        p.drawText(QRectF(r.left() + 12, r.top(), avail, r.height()), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(self.label(), Qt.TextElideMode.ElideRight, avail))
        paint_icon(p, "chevron-down", QRectF(r.right() - 26, r.center().y() - 8, 16, 16),
                   T.mix(T.TEXT_DIM, T.CYAN, self._hover))
        self._ring(p, r, T.R_MD)


# --------------------------------------------------------------------------- meters + progress
class LevelMeter(QWidget):
    """Live input level. ``source`` is a callable returning 0..1; polled only while visible."""

    def __init__(self, source=None, parent=None, height: int = 10):
        super().__init__(parent)
        self.source = source
        self.setFixedHeight(height)
        self.setMinimumWidth(120)
        self.setAccessibleName("Microphone input level")
        self._level = 0.0
        self._peak = 0.0
        self._last = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        T.bus().changed.connect(self.update)

    def set_source(self, source) -> None:
        self.source = source

    def set_level(self, v: float) -> None:
        self._level = max(0.0, min(1.0, float(v)))
        self._peak = max(self._peak, self._level)
        self.update()

    def showEvent(self, e):
        self._last = time.perf_counter()
        self._timer.start()

    def hideEvent(self, e):
        self._timer.stop()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(0.1, now - self._last)
        self._last = now
        target = 0.0
        if self.source is not None:
            try:
                target = max(0.0, min(1.0, float(self.source()) * 2.2))
            except Exception:
                target = 0.0
        self._level = T.smooth(self._level, target, dt, 24 if target > self._level else 9)
        self._peak = max(self._level, self._peak - dt * 0.35)
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        rad = r.height() / 2
        p.fillPath(T.rounded(r, rad), T.SUNK)
        p.setPen(QPen(T.STROKE, 1))
        p.drawRoundedRect(r, rad, rad)
        w = r.width() * self._level
        if w > 2:
            p.save()
            p.setClipPath(T.rounded(r, rad))
            g = QLinearGradient(r.left(), 0, r.right(), 0)
            g.setColorAt(0.0, T.CYAN)
            g.setColorAt(0.7, T.VIOLET)
            g.setColorAt(1.0, T.MAGENTA)
            p.fillRect(QRectF(r.left(), r.top(), w, r.height()), g)
            p.restore()
        if self._peak > 0.03:
            x = r.left() + (r.width() - 3) * self._peak
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T.TEXT)
            p.drawRoundedRect(QRectF(x, r.top() + 1.5, 2.2, r.height() - 3), 1, 1)


class Spinner(QWidget):
    def __init__(self, size: int = 20, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._t0 = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        T.bus().changed.connect(self.update)

    def showEvent(self, e):
        if not T.REDUCED_MOTION:
            self._timer.start()

    def hideEvent(self, e):
        self._timer.stop()

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(2.5, 2.5, self.width() - 5, self.height() - 5)
        p.setPen(QPen(T.CONTROL_HI, 2.4))
        p.drawEllipse(r)
        a = 0.0 if T.REDUCED_MOTION else (time.perf_counter() - self._t0) * 330
        p.setPen(QPen(T.CYAN, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(r, int(-a * 16) + 90 * 16, -100 * 16)


class ProgressBar(QWidget):
    """Determinate (0..1) or indeterminate (``None``) progress."""

    def __init__(self, parent=None, height: int = 8):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setMinimumWidth(120)
        self.value: float | None = 0.0
        self._shown = 0.0
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setAccessibleName("Progress")
        T.bus().changed.connect(self.update)

    def set_value(self, v: float | None) -> None:
        self.value = None if v is None else max(0.0, min(1.0, float(v)))
        if T.REDUCED_MOTION and self.value is not None:
            self._shown = self.value
        self.setAccessibleDescription("Working" if self.value is None else f"{int(self.value * 100)} percent")
        self.update()

    def showEvent(self, e):
        self._last = time.perf_counter()
        self._timer.start()

    def hideEvent(self, e):
        self._timer.stop()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt, self._last = min(0.1, now - self._last), now
        if self.value is not None:
            if abs(self._shown - self.value) < 0.001:
                return
            self._shown = self.value if T.REDUCED_MOTION else T.smooth(self._shown, self.value, dt, 9)
        elif T.REDUCED_MOTION:
            return
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        rad = r.height() / 2
        clip = T.rounded(r, rad)
        p.fillPath(clip, T.CONTROL)
        p.setClipPath(clip)
        g = T.neon_gradient(r.left(), r.right())
        if self.value is None:
            if T.REDUCED_MOTION:
                p.fillRect(QRectF(r.left() + r.width() * 0.3, r.top(), r.width() * 0.4, r.height()), g)
            else:
                k = ((time.perf_counter() - self._t0) * 0.75) % 1.0
                w = r.width() * 0.34
                x = r.left() - w + (r.width() + w) * k
                p.fillPath(T.rounded(QRectF(x, r.top(), w, r.height()), rad), g)
        elif self._shown > 0.002:
            p.fillPath(T.rounded(QRectF(r.left(), r.top(), max(r.height(), r.width() * self._shown), r.height()),
                                 rad), g)


# --------------------------------------------------------------------------- stock inputs, themed
class _Styled:
    """Mixin for stock Qt widgets: (re)applies the theme stylesheet on theme change."""

    def _init_style(self) -> None:
        self._restyle()
        T.bus().changed.connect(self._restyle)

    def _restyle(self) -> None:
        self.setStyleSheet(T.qss())


class LineEdit(QLineEdit, _Styled):
    def __init__(self, text: str = "", placeholder: str = "", parent=None, name: str = ""):
        super().__init__(text, parent)
        self.setPlaceholderText(placeholder)
        self.setAccessibleName(name or placeholder)
        self.setFont(T.font(T.F_BODY))
        self.setFixedHeight(T.CONTROL_H + 2)
        self._init_style()

    def set_invalid(self, bad: bool) -> None:
        if bool(self.property("invalid")) != bool(bad):
            self.setProperty("invalid", bool(bad))
            self._restyle()

    def _restyle(self) -> None:
        super()._restyle()
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, T.TEXT_FAINT)
        self.setPalette(pal)


class SearchField(LineEdit):
    def __init__(self, placeholder: str = "Search", parent=None):
        self._lead = None
        super().__init__("", placeholder, parent, name=placeholder)
        self.setClearButtonEnabled(True)
        self._lead = self.addAction(make_qicon("search", T.TEXT_DIM, 16), QLineEdit.ActionPosition.LeadingPosition)

    def _restyle(self) -> None:
        super()._restyle()
        if self._lead is not None:
            self._lead.setIcon(make_qicon("search", T.TEXT_DIM, 16))


class TextArea(QPlainTextEdit, _Styled):
    def __init__(self, text: str = "", placeholder: str = "", parent=None, name: str = ""):
        super().__init__(text, parent)
        self.setPlaceholderText(placeholder)
        self.setAccessibleName(name or placeholder)
        self.setFont(T.font(T.F_BODY))
        self.setTabChangesFocus(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._init_style()

    def _restyle(self) -> None:
        super()._restyle()
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, T.TEXT_FAINT)
        self.setPalette(pal)


class ScrollArea(QScrollArea, _Styled):
    """Transparent, frameless vertical scroller holding one content widget."""

    def __init__(self, content: QWidget | None = None, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._init_style()
        if content is not None:
            self.setWidget(content)

    def setWidget(self, w: QWidget) -> None:
        w.setAutoFillBackground(False)
        super().setWidget(w)

    def _restyle(self) -> None:
        self.setStyleSheet(T.qss() + "QScrollArea > QWidget > QWidget { background: transparent; }")
