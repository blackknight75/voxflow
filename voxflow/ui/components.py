"""Composite building blocks shared by the pages: cards, setting rows, stat tiles, the activity chart,
empty/loading/error states, toasts, the in-window dialog, navigation buttons and the transcript list."""
from __future__ import annotations

import datetime as _dt
import time

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from . import theme as T
from .icons import paint_icon
from .widgets import Button, Caption, IconButton, Label, Spinner, _Hover, _KBD_REASONS, painter


# --------------------------------------------------------------------------- formatting helpers
def fmt_count(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    return f"{int(n):,}"


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s} sec"
    if s < 3600:
        return f"{s // 60} min"
    h, m = s // 3600, (s % 3600) // 60
    return f"{h} h {m} min" if m else f"{h} h"


def day_label(ts: float, now: float | None = None) -> str:
    """'Today' / 'Yesterday' / 'Monday' / 'March 4' / 'March 4, 2024' for the day holding ``ts``."""
    try:
        d = _dt.date.fromtimestamp(float(ts))
    except (TypeError, ValueError, OSError, OverflowError):
        return "Earlier"
    today = _dt.date.fromtimestamp(now if now is not None else time.time())
    delta = (today - d).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 1 < delta < 7:
        return d.strftime("%A")
    return f"{d.strftime('%B')} {d.day}" + (f", {d.year}" if d.year != today.year else "")


def clock_label(ts: float) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def app_label(exe: str) -> str:
    """'slack.exe' -> 'Slack'; friendly names for the common ones."""
    known = {"code": "VS Code", "winword": "Word", "excel": "Excel", "powerpnt": "PowerPoint", "msedge": "Edge",
             "chrome": "Chrome", "firefox": "Firefox", "outlook": "Outlook", "teams": "Teams", "ms-teams": "Teams",
             "whatsapp": "WhatsApp", "notepad++": "Notepad++", "explorer": "Explorer", "windowsterminal": "Terminal"}
    base = (exe or "").lower().removesuffix(".exe")
    if not base:
        return ""
    return known.get(base, base.replace("_", " ").replace("-", " ").title())


# --------------------------------------------------------------------------- containers
class Card(QWidget):
    """Raised rounded surface. Add children through ``card.body`` (a QVBoxLayout)."""

    def __init__(self, parent=None, padding: int = T.S4, radius: int = T.R_LG, spacing: int = T.S3):
        super().__init__(parent)
        self.radius = radius
        self.accent = 0.0
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)
        T.bus().changed.connect(self.update)

    def paintEvent(self, _):
        p = painter(self)
        T.paint_card(p, QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), self.radius, accent=self.accent)


class Row(QWidget):
    """One setting: optional icon, title + description on the left, control on the right, hairline below."""

    def __init__(self, title: str, sub: str = "", control: QWidget | None = None, icon: str | None = None,
                 parent=None, below: QWidget | None = None):
        super().__init__(parent)
        self.icon_name = icon
        self.divider = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(T.S4 + (30 if icon else 0), 13, T.S4, 13)
        outer.setSpacing(10)
        self._outer = outer
        lay = QHBoxLayout()
        lay.setSpacing(T.S4)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = Label(title, T.F_BODY, "TEXT", T.W_SEMI)
        col.addWidget(self.title)
        self.sub = Label(sub, T.F_SMALL - 0.2, "TEXT_DIM")
        self.sub.setVisible(bool(sub))
        col.addWidget(self.sub)
        lay.addLayout(col, 1)
        if control is not None:
            lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            if not control.accessibleName():
                control.setAccessibleName(title)
            if sub and not control.accessibleDescription():
                control.setAccessibleDescription(sub)
        outer.addLayout(lay)
        if below is not None:
            outer.addWidget(below)
        T.bus().changed.connect(self.update)

    def set_indent(self, on: bool) -> None:
        """Reserve the icon column even without an icon, so titles line up within a section."""
        self._outer.setContentsMargins(T.S4 + (30 if (on or self.icon_name) else 0), 13, T.S4, 13)

    def set_sub(self, text: str, role: str = "TEXT_DIM") -> None:
        self.sub.set_role(role)
        self.sub.setText(text)
        self.sub.setVisible(bool(text))

    def paintEvent(self, _):
        p = painter(self)
        if self.icon_name:
            paint_icon(p, self.icon_name, QRectF(T.S4, 13, 20, 20), T.TEXT_DIM)
        if self.divider:
            p.setPen(QPen(T.STROKE, 1))
            y = self.height() - 0.5
            p.drawLine(QPointF(T.S4, y), QPointF(self.width() - T.S4, y))


class Section(QWidget):
    """Caption + a card of rows (the last row loses its divider)."""

    def __init__(self, title: str, parent=None, icons: bool = False):
        super().__init__(parent)
        self.icons = icons                      # per-row icons only when every row can have one (D-07): off by default
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(T.S2)
        self.caption = Caption(title)
        self.caption.setContentsMargins(4, 0, 0, 0)
        lay.addWidget(self.caption)
        self.card = Card(padding=0, spacing=0)
        self.card.body.setContentsMargins(0, 2, 0, 2)
        lay.addWidget(self.card)
        self._rows: list[Row] = []
        self._indent = False

    def add(self, row: Row) -> Row:
        for r in self._rows:
            r.divider = True
        row.divider = False
        self._rows.append(row)
        if not self.icons:
            row.icon_hint, row.icon_name = row.icon_name, None
        if row.icon_name and not self._indent:
            self._indent = True
            for r in self._rows:
                r.set_indent(True)
        row.set_indent(self._indent)
        self.card.body.addWidget(row)
        return row

    def clear(self) -> None:
        for r in self._rows:
            self.card.body.removeWidget(r)
            r.hide()
            r.deleteLater()
        self._rows = []
        self._indent = False


# --------------------------------------------------------------------------- stats
class StatCard(QWidget):
    """One headline number: icon + label, the value (auto-shrunk to fit), and a quiet caption underneath."""

    def __init__(self, icon: str, label: str, parent=None, name: str = ""):
        super().__init__(parent)
        self.icon_name, self.label = icon, label
        self.value, self.caption = "0", ""
        self.setMinimumSize(140, 116)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setObjectName(name)
        T.bus().changed.connect(self.update)

    def sizeHint(self) -> QSize:
        return QSize(190, 116)

    def set_value(self, value: str, caption: str = "") -> None:
        self.value, self.caption = value, caption
        self.setAccessibleName(f"{self.label}: {value}")
        self.setAccessibleDescription(caption)
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        T.paint_card(p, r, T.R_LG)
        chip = QRectF(r.left() + 14, r.top() + 14, 28, 28)
        p.fillPath(T.rounded(chip, 8), T.neon_gradient(chip.left(), chip.right(), 0.16))
        paint_icon(p, self.icon_name, chip.adjusted(6, 6, -6, -6), T.CYAN)
        inner_w = r.width() - 30
        p.setPen(T.TEXT_DIM)
        lf = T.font(T.F_SMALL, T.W_MEDIUM)
        p.setFont(lf)
        lw = inner_w - 36
        p.drawText(QRectF(chip.right() + 9, chip.top(), lw, 28), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetricsF(lf).elidedText(self.label, Qt.TextElideMode.ElideRight, lw))
        pt = T.F_H1 + 1
        vf = T.font(pt, T.W_BOLD, -0.2)
        while pt > 11 and QFontMetricsF(vf).horizontalAdvance(self.value) > inner_w:
            pt -= 0.75
            vf = T.font(pt, T.W_BOLD, -0.2)
        p.setFont(vf)
        p.setPen(T.TEXT)
        p.drawText(QRectF(r.left() + 15, r.top() + 50, inner_w, 34),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.value)
        if self.caption:
            cf = T.font(T.F_CAPTION)
            p.setFont(cf)
            p.setPen(T.TEXT_DIM)
            p.drawText(QRectF(r.left() + 15, r.top() + 84, inner_w, 20), Qt.AlignmentFlag.AlignVCenter,
                       QFontMetricsF(cf).elidedText(self.caption, Qt.TextElideMode.ElideRight, inner_w))


class BarChart(QWidget):
    """Daily words for the last N days. Hover or arrow keys reveal the exact value of a day."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series: list[dict] = []
        self.setMinimumHeight(170)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Words dictated per day")
        self._hover = -1
        self._kbd = False
        self._grow = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(T.EASE)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._set_grow)
        T.bus().changed.connect(self.update)

    def _set_grow(self, v) -> None:
        self._grow = float(v)
        self.update()

    def set_series(self, series: list[dict], animate: bool = False) -> None:
        self.series = list(series)
        total = sum(int(d.get("words", 0)) for d in self.series)
        self.setAccessibleDescription(f"{total} words over the last {len(self.series)} days")
        if animate and not T.REDUCED_MOTION and self.isVisible():
            self._anim.stop()
            self._anim.setDuration(520)
            self._anim.start()
        else:
            self._grow = 1.0
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(40, 12, self.width() - 52, self.height() - 44)

    def _slot(self, i: int) -> QRectF:
        pl = self._plot()
        n = max(1, len(self.series))
        w = pl.width() / n
        return QRectF(pl.left() + w * i, pl.top(), w, pl.height())

    def _set_hover(self, i: int) -> None:
        if i != self._hover:
            self._hover = i
            self.update()

    def mouseMoveEvent(self, e):
        pl = self._plot()
        x = e.position().x()
        n = len(self.series)
        self._set_hover(int((x - pl.left()) / (pl.width() / n)) if n and pl.left() <= x < pl.right() else -1)

    def leaveEvent(self, e):
        if not self.hasFocus():
            self._set_hover(-1)

    def focusInEvent(self, e):
        self._kbd = e.reason() in _KBD_REASONS
        if self._hover < 0 and self.series:
            self._hover = len(self.series) - 1
        self.update()

    def focusOutEvent(self, e):
        self._kbd = False
        self._set_hover(-1)

    def keyPressEvent(self, e):
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(e.key())
        if step is None or not self.series:
            return super().keyPressEvent(e)
        self._set_hover(max(0, min(len(self.series) - 1, self._hover + step)))

    def paintEvent(self, _):
        p = painter(self)
        pl = self._plot()
        n = len(self.series)
        peak = max([int(d.get("words", 0)) for d in self.series] or [0])
        if not n or peak <= 0:
            p.setPen(T.TEXT_FAINT)
            p.setFont(T.font(T.F_BODY))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, "Your daily activity will appear here.")
            return
        # nice axis maximum + two gridlines
        mag = 10 ** max(0, len(str(peak)) - 1)
        top = max(10, ((peak + mag // 2) // max(1, mag // 2) + 1) * max(1, mag // 2))
        p.setFont(T.font(T.F_CAPTION - 0.6))
        for frac in (0.0, 0.5, 1.0):
            y = pl.bottom() - pl.height() * frac
            p.setPen(QPen(T.STROKE, 1))
            p.drawLine(QPointF(pl.left(), y), QPointF(pl.right(), y))
            p.setPen(T.TEXT_FAINT)
            p.drawText(QRectF(0, y - 9, pl.left() - 8, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       fmt_count(top * frac))
        label_every = 1 if n <= 8 else 2 if n <= 16 else 5
        for i, d in enumerate(self.series):
            slot = self._slot(i)
            words = int(d.get("words", 0))
            bw = max(4.0, min(26.0, slot.width() * 0.56))
            h = pl.height() * (words / top) * self._grow
            bar = QRectF(slot.center().x() - bw / 2, pl.bottom() - max(h, 2.0), bw, max(h, 2.0))
            hot = i == self._hover
            if words <= 0:
                p.fillPath(T.rounded(bar, 1), T.CONTROL_HI)
            else:
                g = QLinearGradient(0, bar.top(), 0, pl.bottom())
                g.setColorAt(0, T.CYAN if not hot else T.mix(T.CYAN, QColor(255, 255, 255), 0.25))
                g.setColorAt(1, T.alpha(T.VIOLET, 0.85))
                p.fillPath(T.rounded(bar, min(5.0, bw / 2)), g)
            is_last = i == n - 1
            if i % label_every == (n - 1) % label_every:
                try:
                    day = _dt.date.fromisoformat(str(d.get("date", "")))
                    text = "Today" if is_last else (day.strftime("%a") if n <= 8 else f"{day.day} {day.strftime('%b')}")
                except ValueError:
                    text = ""
                p.setPen(T.TEXT if (hot or is_last) else T.TEXT_FAINT)
                p.setFont(T.font(T.F_CAPTION - 0.4, T.W_SEMI if is_last else T.W_NORMAL))
                p.drawText(QRectF(slot.center().x() - 30, pl.bottom() + 8, 60, 18), Qt.AlignmentFlag.AlignCenter, text)
        if 0 <= self._hover < n:
            d = self.series[self._hover]
            slot = self._slot(self._hover)
            words = int(d.get("words", 0))
            try:
                when = day_label(_dt.datetime.fromisoformat(str(d.get("date"))).timestamp())
            except ValueError:
                when = ""
            text = f"{when}  ·  {words:,} words"
            f = T.font(T.F_SMALL, T.W_SEMI)
            tw = QFontMetricsF(f).horizontalAdvance(text) + 20
            y = pl.bottom() - pl.height() * (words / top) - 34
            tip = QRectF(min(max(2.0, slot.center().x() - tw / 2), self.width() - tw - 2), max(0.0, y), tw, 26)
            p.fillPath(T.rounded(tip, 8), T.PANEL_HI)
            p.setPen(QPen(T.STROKE_STRONG, 1))
            p.drawRoundedRect(tip, 8, 8)
            p.setPen(T.TEXT)
            p.setFont(f)
            p.drawText(tip, Qt.AlignmentFlag.AlignCenter, text)
        if self._kbd and self.hasFocus():
            T.paint_focus_ring(p, QRectF(1.5, 1.5, self.width() - 3, self.height() - 3), T.R_MD)


# --------------------------------------------------------------------------- states
class EmptyState(QWidget):
    """Centred icon tile + title + explanation + optional action. ``kind``: "empty" (accent) or "error" (red).

    One instance per page is re-used for every such situation through :meth:`show_state`.
    """

    def __init__(self, icon: str = "info", title: str = "", sub: str = "", action: str = "", on_action=None,
                 parent=None, kind: str = "empty"):
        super().__init__(parent)
        self.icon_name, self.kind = icon, kind
        self._on_action = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(T.S6, 92, T.S6, T.S6)
        lay.setSpacing(6)
        self.title = Label("", T.F_TITLE, "TEXT", T.W_SEMI)
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.sub = Label("", T.F_BODY, "TEXT_DIM")
        self.sub.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.title)
        lay.addWidget(self.sub, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(8)
        self.button = Button("", "secondary")
        self.button.clicked.connect(self._clicked)
        lay.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(1)
        self.show_state(kind, icon, title, sub, action, on_action)
        T.bus().changed.connect(self.update)

    def show_state(self, kind: str, icon: str, title: str, sub: str = "", action: str = "", on_action=None) -> None:
        self.kind, self.icon_name = kind, icon
        self._on_action = on_action
        self.title.setText(title)
        self.sub.setText(sub)
        self.button.setText(action)
        self.button.setAccessibleName(action)
        self.button.setVisible(bool(action))
        self.setAccessibleName(title)
        self.setAccessibleDescription(sub)
        self._fit()
        self.update()

    def _clicked(self) -> None:
        if self._on_action is not None:
            self._on_action()

    def resizeEvent(self, e):
        if e.size().width() != e.oldSize().width():
            self._fit()

    def _fit(self) -> None:
        """Give the wrapped explanation an explicit, readable measure and the height that goes with it."""
        w = max(200, min(440, self.width() - 2 * T.S6))
        h = self.sub.heightForWidth(w) if self.sub.text() else 0
        self.sub.setFixedSize(w, max(0, h))
        self.setMinimumHeight(150 + h + (56 if self.button.text() else 0))

    def paintEvent(self, _):
        p = painter(self)
        c = QPointF(self.width() / 2, 52)
        col = T.RED if self.kind == "error" else T.CYAN
        tile = QRectF(c.x() - 26, c.y() - 26, 52, 52)
        p.fillPath(T.rounded(tile, 16), T.alpha(col, 0.12))
        p.setPen(QPen(T.alpha(col, 0.35), 1))
        p.drawRoundedRect(tile, 16, 16)
        paint_icon(p, self.icon_name, tile.adjusted(14, 14, -14, -14), col, 1.8)


class StatusChip(QWidget):
    """Status dot (a spinner while something is loading) + title + detail, in the sidebar footer."""

    LEVEL_ROLES = {"ok": "GREEN", "demo": "GREEN", "loading": "AMBER", "warning": "AMBER", "error": "RED",
                   "paused": "TEXT_DIM"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.level, self.title, self.detail = "loading", "Starting…", ""
        self._two_lines = False
        self.setFixedHeight(60)          # constant: room for a two-line detail without ever resizing the window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spinner = Spinner(14, self)
        T.bus().changed.connect(self.update)

    def set_status(self, status: dict) -> None:
        self.level = str(status.get("level", "ok"))
        self.title = str(status.get("title", ""))
        self.detail = str(status.get("detail", ""))
        self.setAccessibleName(f"Status: {self.title}")
        self.setAccessibleDescription(self.detail)
        self.setToolTip(self.detail)
        self.spinner.setVisible(self.level == "loading")
        self._fit()
        self.update()

    def resizeEvent(self, e):
        self._fit()

    def _fit(self) -> None:
        """One line of detail when it fits, two when it does not; a single line sits vertically centred."""
        f = T.font(T.F_CAPTION - 0.3)
        self._two_lines = QFontMetricsF(f).horizontalAdvance(self.detail) > self.width() - 37
        self.spinner.move(9, 9 + self._dy())

    def _dy(self) -> int:
        return 0 if self._two_lines else 7

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        col = T.token(self.LEVEL_ROLES.get(self.level, "GREEN"))
        p.fillPath(T.rounded(r, T.R_MD), T.alpha(col, 0.09))
        p.setPen(QPen(T.alpha(col, 0.32), 1))
        p.drawRoundedRect(r, T.R_MD, T.R_MD)
        r = r.translated(0, self._dy())               # text block origin (the frame above is already painted)
        if self.level != "loading":
            cx, cy = r.left() + 16, r.top() + 16
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T.alpha(col, 0.25))
            p.drawEllipse(QPointF(cx, cy), 6, 6)
            p.setBrush(col)
            p.drawEllipse(QPointF(cx, cy), 3.2, 3.2)
        w = int(r.width() - 36)
        p.setPen(T.TEXT)
        f = T.font(T.F_SMALL, T.W_SEMI)
        p.setFont(f)
        p.drawText(QRectF(r.left() + 28, r.top() + 6, w, 20), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight, w))
        p.setPen(T.TEXT_DIM)
        df = T.font(T.F_CAPTION - 0.3)
        p.setFont(df)
        if self._two_lines:
            # second line elided by hand: Qt cannot elide the last line of wrapped text
            words, first = self.detail.split(), ""
            fm = QFontMetricsF(df)
            while words and fm.horizontalAdvance((first + " " + words[0]).strip()) <= w:
                first = (first + " " + words.pop(0)).strip()
            p.drawText(QRectF(r.left() + 28, r.top() + 24, w, 16), Qt.AlignmentFlag.AlignVCenter, first)
            p.drawText(QRectF(r.left() + 28, r.top() + 39, w, 16), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(" ".join(words), Qt.TextElideMode.ElideRight, w))
        else:
            p.drawText(QRectF(r.left() + 28, r.top() + 24, w, 18), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(self.detail, Qt.TextElideMode.ElideRight, w))


# --------------------------------------------------------------------------- toasts
class _Toast(QWidget):
    closed = Signal(object)
    PAD = 7                     # room around the body for its drop shadow
    ICONS = {"info": ("info", "CYAN"), "success": ("check", "GREEN"), "warning": ("warning", "AMBER"),
             "error": ("warning", "RED")}

    def __init__(self, level: str, message: str, action: str, on_action, parent):
        super().__init__(parent)
        self.level, self.message = level if level in self.ICONS else "info", message
        self._on_action = on_action
        self._font = T.font(T.F_BODY, T.W_MEDIUM)
        self._appear = 1.0 if T.REDUCED_MOTION else 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(f"{self.level}: {message}")
        self.setToolTip("Click to dismiss")
        self.setMouseTracking(True)
        self._hover = False
        self.btn = None
        if action:
            self.btn = Button(action, "ghost", parent=self, height=28)
            self.btn.clicked.connect(self._act)
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(T.EASE)
        self._anim.valueChanged.connect(self._set_appear)
        self._life = QTimer(self)
        self._life.setSingleShot(True)
        self._life.timeout.connect(self.dismiss)
        self._closing = False

    def _set_appear(self, v) -> None:
        self._appear = float(v)
        self.update()

    def layout_for(self, max_w: int) -> QSize:
        fm = QFontMetricsF(self._font)
        bw = (self.btn.sizeHint().width() + 8) if self.btn else 0
        text_w = min(max_w - 62 - bw, fm.horizontalAdvance(self.message) + 4)
        br = fm.boundingRect(QRectF(0, 0, text_w, 1000), int(Qt.TextFlag.TextWordWrap), self.message)
        h = max(44, int(br.height()) + 24) + 2 * self.PAD
        size = QSize(int(text_w + 62 + bw) + 2 * self.PAD, h)
        self.resize(size)
        if self.btn:
            self.btn.move(size.width() - bw - 4 - self.PAD, (h - 28) // 2)
        return size

    def set_action(self, label: str, on_action) -> None:
        if self.btn is None:
            self.btn = Button(label, "ghost", parent=self, height=28)
            self.btn.clicked.connect(self._act)
            self.btn.show()
        self.btn.setText(label)
        self._on_action = on_action

    def fit_width(self, w: int) -> None:
        """All toasts in a stack share one width (set by the host) so the edges line up."""
        self.resize(w, self.height())
        if self.btn:
            self.btn.move(w - self.btn.sizeHint().width() - 12 - self.PAD, (self.height() - 28) // 2)

    def enterEvent(self, e):
        self._hover = True
        self._life.stop()                 # reading a toast must not race its timer
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        if not self._closing:
            self._life.start(2200)
        self.update()

    def start(self, ms: int) -> None:
        self.show()
        if not T.REDUCED_MOTION:
            self._anim.setDuration(200)
            self._anim.setStartValue(0.0)
            self._anim.setEndValue(1.0)
            self._anim.start()
        self._life.start(ms)

    def _act(self) -> None:
        cb, self._on_action = self._on_action, None
        self.dismiss()
        if cb:
            cb()

    def dismiss(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._life.stop()
        self.closed.emit(self)

    def mouseReleaseEvent(self, e):
        self.dismiss()

    def paintEvent(self, _):
        p = painter(self)
        p.setOpacity(self._appear)
        p.translate(0, (1 - self._appear) * 10)
        r = QRectF(self.PAD, self.PAD, self.width() - 2 * self.PAD, self.height() - 2 * self.PAD)
        icon, role = self.ICONS[self.level]
        col = T.token(role)
        T.paint_shadow(p, r, T.R_MD + 1, spread=self.PAD - 1, strength=1.6, dy=2)
        p.fillPath(T.rounded(r, T.R_MD + 1), T.PANEL_HI)
        p.setPen(QPen(T.alpha(col, 0.55), 1.1))
        p.drawRoundedRect(r, T.R_MD + 1, T.R_MD + 1)
        paint_icon(p, icon, QRectF(r.left() + 14, r.center().y() - 9, 18, 18), col, 1.9)
        p.setPen(T.TEXT)
        p.setFont(self._font)
        bw = (self.btn.width() + 8) if self.btn else 0
        p.drawText(QRectF(r.left() + 42, r.top(), r.width() - 56 - bw - (14 if self._hover and not self.btn else 0),
                          r.height()), int(Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap), self.message)
        if self._hover and not self.btn:               # dismiss affordance, only while the pointer is on it
            paint_icon(p, "close", QRectF(r.right() - 26, r.center().y() - 6, 12, 12), T.TEXT_DIM, 1.8)


class ToastHost(QWidget):
    """Bottom-centre stack of transient messages inside a window. Transparent to the mouse except the toasts."""

    MAX = 3

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._toasts: list[_Toast] = []
        self.bottom_margin = 22
        self.left_inset = 0          # width of a sidebar the stack must never cover: it centres on what is right of it
        parent.installEventFilter(self)
        self.hide()

    def attach_action(self, message: str, label: str, on_action) -> bool:
        """Give the toast showing ``message`` an action button (the controller announces the next step right after
        the notice itself). Returns False when no such toast is up."""
        for t in self._toasts:
            if t.message == message and not t._closing:
                t.set_action(label, on_action)
                self._relayout()
                t._life.start(8000)
                return True
        return False

    def messages(self) -> list[str]:
        return [t.message for t in self._toasts]

    def show_toast(self, level: str, message: str, action: str = "", on_action=None, ms: int | None = None) -> None:
        if not message:
            return
        for t in self._toasts:                      # identical message already showing: just extend it
            if t.message == message and t.level == level and not t._closing:
                t._life.start(ms or 4000)
                return
        while len(self._toasts) >= self.MAX:
            self._drop(self._toasts[0])
        toast = _Toast(level, message, action, on_action, self)
        toast.closed.connect(self._drop)
        self._toasts.append(toast)
        self._relayout()
        default = 6500 if (action or level in ("error", "warning")) else 3600
        toast.start(ms or default)
        self.show()
        self.raise_()

    def _drop(self, toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
            toast.hide()
            toast.deleteLater()
            self._relayout()

    def _relayout(self) -> None:
        par = self.parentWidget()
        if par is None:
            return
        if not self._toasts:
            self.hide()
            return
        room = par.width() - self.left_inset
        max_w = max(260, min(560, room - 80))
        sizes = [t.layout_for(max_w) for t in self._toasts]
        w = max(s.width() for s in sizes)
        h = sum(s.height() for s in sizes) + (8 - 2 * _Toast.PAD) * (len(sizes) - 1)
        self.setGeometry(self.left_inset + (room - w) // 2, par.height() - h - self.bottom_margin + _Toast.PAD, w, h)
        y = 0
        for t, s in zip(self._toasts, sizes):
            t.fit_width(w)
            t.move(0, y)
            y += s.height() + 8 - 2 * _Toast.PAD

    def eventFilter(self, obj, ev):
        if obj is self.parentWidget() and ev.type() == QEvent.Type.Resize:
            self._relayout()
        return False


# --------------------------------------------------------------------------- in-window dialog
class Modal(QWidget):
    """Scrim + centred card inside the main window. Esc / click outside cancels; focus stays inside.

    Build the content by adding widgets to ``modal.body``; ``add_buttons`` adds the footer.
    """

    closed = Signal()

    def __init__(self, host: QWidget, title: str, sub: str = "", width: int = 440, name: str = "modal"):
        super().__init__(host)
        self.setObjectName(name)
        self.setAccessibleName(title)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._host = host
        self._prev_focus = host.window().focusWidget() if host.window() else None
        self._appear = 1.0 if T.REDUCED_MOTION else 0.0
        self.card = QWidget(self)
        self.card.setFixedWidth(width)
        self.body = QVBoxLayout(self.card)
        self.body.setContentsMargins(T.S6, T.S5, T.S6, T.S5)
        self.body.setSpacing(T.S3)
        self.title = Label(title, T.F_H2, "TEXT", T.W_BOLD)
        self.body.addWidget(self.title)
        if sub:
            lead = Label(sub, T.F_BODY, "TEXT_DIM")
            lead.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            lead.setFixedHeight(lead.heightForWidth(width - 2 * T.S6))    # exact: wrapped labels over-ask in layouts
            self.body.addWidget(lead)
            self.body.addSpacing(T.S1)
        self.error = Label("", T.F_SMALL, "RED")
        self.error.hide()
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(T.EASE)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._set_appear)
        host.installEventFilter(self)
        T.bus().changed.connect(self.update)

    def _set_appear(self, v) -> None:
        self._appear = float(v)
        self.update()

    def add_buttons(self, primary: str, on_primary, cancel: str = "Cancel", danger: bool = False,
                    extra: QWidget | None = None) -> Button:
        self.body.addWidget(self.error)
        row = QHBoxLayout()
        row.setSpacing(T.S2)
        if extra is not None:
            row.addWidget(extra)
        row.addStretch(1)
        self.btn_cancel = Button(cancel, "ghost")
        self.btn_cancel.setObjectName(f"{self.objectName()}_cancel")
        self.btn_cancel.clicked.connect(self.close_modal)
        self.btn_primary = Button(primary, "danger" if danger else "primary")
        self.btn_primary.setObjectName(f"{self.objectName()}_ok")
        self.btn_primary.clicked.connect(on_primary)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_primary)
        self.body.addSpacing(T.S1)
        self.body.addLayout(row)
        return self.btn_primary

    def show_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(bool(message))
        self._fit()

    def open(self, focus: QWidget | None = None) -> None:
        self._fit()
        self.show()
        self.raise_()
        (focus or self).setFocus(Qt.FocusReason.OtherFocusReason)
        if not T.REDUCED_MOTION:
            self._anim.setDuration(170)
            self._anim.start()

    def close_modal(self) -> None:
        self._host.removeEventFilter(self)
        self.hide()
        self.closed.emit()
        prev = self._prev_focus
        try:
            if prev is not None and prev.isVisible():
                prev.setFocus(Qt.FocusReason.OtherFocusReason)
        except RuntimeError:
            pass
        self.deleteLater()

    def _fit(self) -> None:
        self.setGeometry(self._host.rect())
        self.card.adjustSize()
        h = min(self.card.sizeHint().height(), self.height() - 40)
        self.card.setGeometry((self.width() - self.card.width()) // 2, max(20, (self.height() - h) // 2 - 10),
                              self.card.width(), h)

    def eventFilter(self, obj, ev):
        if obj is self._host and ev.type() == QEvent.Type.Resize:
            self._fit()
        return False

    def mousePressEvent(self, e):
        if not self.card.geometry().contains(e.position().toPoint()):
            self.close_modal()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close_modal()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and hasattr(self, "btn_primary") \
                and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.btn_primary.click()
        else:
            super().keyPressEvent(e)

    def focusNextPrevChild(self, forward: bool) -> bool:
        """Keep Tab inside the dialog."""
        chain = [w for w in self.card.findChildren(QWidget)
                 if w.focusPolicy() & Qt.FocusPolicy.TabFocus and w.isVisibleTo(self) and w.isEnabled()]
        if not chain:
            return True
        cur = self.window().focusWidget()
        i = chain.index(cur) if cur in chain else (-1 if forward else 0)
        chain[(i + (1 if forward else -1)) % len(chain)].setFocus(
            Qt.FocusReason.TabFocusReason if forward else Qt.FocusReason.BacktabFocusReason)
        return True

    def paintEvent(self, _):
        p = painter(self)
        p.setOpacity(self._appear)
        p.fillPath(T.rounded(QRectF(self.rect()), getattr(self._host, "corner_radius", 0)), T.SCRIM)
        r = QRectF(self.card.geometry()).adjusted(0.5, 0.5, -0.5, -0.5)
        T.paint_shadow(p, r, T.R_XL, spread=16, strength=1.4, dy=6)
        T.glass_panel(p, r, T.R_XL)


def confirm(host: QWidget, title: str, message: str, action: str, on_confirm, danger: bool = True,
            name: str = "confirm") -> Modal:
    """Build an "are you sure?" dialog for a destructive action (the window opens it, focused on Cancel)."""
    m = Modal(host, title, message, width=420, name=name)

    def go():
        m.close_modal()
        on_confirm()

    m.add_buttons(action, go, danger=danger)
    return m


# --------------------------------------------------------------------------- chrome
class NavButton(_Hover):
    """Sidebar navigation item (checkable, exclusive via the window)."""

    def __init__(self, icon: str, text: str, parent=None):
        super().__init__(parent)
        self.icon_name = icon
        self.badge = 0
        self.setText(text)
        self.setAccessibleName(text)
        self.setCheckable(True)
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_badge(self, n: int) -> None:
        self.badge = int(n)
        self.setAccessibleDescription(f"{n} new" if n else "")
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        on = self.isChecked()
        if on:
            g = QLinearGradient(r.left(), 0, r.right(), 0)
            g.setColorAt(0, T.alpha(T.CYAN, 0.26 if T.is_light() else 0.20))
            g.setColorAt(1, T.alpha(T.VIOLET, 0.16 if T.is_light() else 0.10))
            p.fillPath(T.rounded(r, T.R_MD), g)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T.CYAN)
            p.drawRoundedRect(QRectF(r.left() + 1, r.center().y() - 8, 3, 16), 1.5, 1.5)
        elif self._hover > 0:
            p.fillPath(T.rounded(r, T.R_MD), T.alpha(T.CONTROL_HI, 0.7 * self._hover))
        col = T.TEXT if on else T.mix(T.TEXT_DIM, T.TEXT, self._hover)
        paint_icon(p, self.icon_name, QRectF(r.left() + 13, r.center().y() - 9, 18, 18), T.CYAN if on else col)
        p.setPen(col)
        p.setFont(T.font(T.F_BODY, T.W_BOLD if on else T.W_MEDIUM))
        p.drawText(QRectF(r.left() + 42, r.top(), r.width() - 76, r.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        if self.badge:
            text = str(self.badge) if self.badge < 100 else "99+"
            f = T.font(T.F_CAPTION - 0.6, T.W_BOLD)
            w = max(18.0, QFontMetricsF(f).horizontalAdvance(text) + 10)
            b = QRectF(r.right() - w - 8, r.center().y() - 9, w, 18)
            p.fillPath(T.rounded(b, 9), T.neon_gradient(b.left(), b.right()))
            p.setPen(T.ON_ACCENT)
            p.setFont(f)
            p.drawText(b, Qt.AlignmentFlag.AlignCenter, text)
        self._ring(p, r, T.R_MD)


class WindowButton(_Hover):
    """Title-bar button: min | max | restore | close."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(40, 30)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.set_kind(kind)

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        tip = {"min": "Minimise", "max": "Maximise", "restore": "Restore", "close": "Close to tray"}[kind]
        self.setToolTip(tip)
        self.setAccessibleName(tip)
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(2, 2, self.width() - 4, self.height() - 4)
        if self._hover > 0:
            col = T.RED if self.kind == "close" else T.TEXT
            p.fillPath(T.rounded(r, T.R_SM), T.alpha(col, (0.85 if self.kind == "close" else 0.10) * self._hover))
        ink = T.mix(T.TEXT_DIM, QColor(255, 255, 255) if self.kind == "close" else T.TEXT, self._hover)
        paint_icon(p, self.kind, QRectF(r.center().x() - 8, r.center().y() - 8, 16, 16), ink, 1.5)


# --------------------------------------------------------------------------- transcript list
class TranscriptList(QWidget):
    """Virtualised, self-painted list of history entries grouped by day.

    Mouse: hover a card for its actions. Keyboard: Up/Down select, Enter or C copy, P paste again, Delete remove.
    """

    copy_requested = Signal(dict)
    paste_requested = Signal(dict)
    delete_requested = Signal(dict)

    PAD, GAP, HEAD_H = 0, 8, 34
    RULES_ONLY_TAG = "Basic clean-up"
    RULES_ONLY_TIP = ("The language model was too slow for this one, so only the basic clean-up was applied. "
                      "The words are as you said them.")
    ACTIONS = (("copy", "Copy"), ("paste", "Paste again"), ("trash", "Delete"))

    def __init__(self, parent=None, group_by_day: bool = True, show_delete: bool = True):
        super().__init__(parent)
        self.items: list[dict] = []
        self.group_by_day = group_by_day
        self.actions = [a for a in self.ACTIONS if show_delete or a[0] != "trash"]
        self.show_raw = False
        self._rows: list[tuple] = []           # ("head", QRect, label) | ("card", QRect, index, text QRect)
        self._tags: dict[int, QRect] = {}      # item index -> rect of its painted "basic clean-up" marker
        self._hover: tuple[int, int] = (-1, -1)        # (item index, action index)
        self._sel = -1
        self._kbd = False
        self._flash: dict[tuple[str, str], float] = {}  # (entry id, action) -> time of confirmation
        self._font = T.font(T.F_BODY_LG)
        self._meta_font = T.font(T.F_CAPTION - 0.2, T.W_SEMI, 0.2)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Transcripts")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._tick = QTimer(self)
        self._tick.setInterval(120)
        self._tick.timeout.connect(self._expire_flash)
        T.bus().changed.connect(self.update)

    # ---- data
    def set_items(self, items: list[dict]) -> None:
        keep = self.items[self._sel].get("id") if 0 <= self._sel < len(self.items) else None
        self.items = [it for it in items if isinstance(it, dict)]
        self._sel = next((i for i, it in enumerate(self.items) if keep is not None and it.get("id") == keep), -1)
        self._hover = (-1, -1)
        self._relayout()

    def set_show_raw(self, raw: bool) -> None:
        self.show_raw = bool(raw)
        self._relayout()

    def body_text(self, it: dict) -> str:
        text = it.get("raw") if self.show_raw else it.get("text")
        return str(text or it.get("text") or it.get("raw") or "")

    def selected(self) -> dict | None:
        return self.items[self._sel] if 0 <= self._sel < len(self.items) else None

    def _select(self, index: int) -> None:
        self._sel = index
        it = self.selected()
        self.setAccessibleDescription(
            f"{day_label(it.get('ts', 0))} {clock_label(it.get('ts', 0))}"
            f"{', basic clean-up only' if it.get('rules_only') else ''}: {self.body_text(it)}" if it else "")
        self.update()

    # ---- layout
    def resizeEvent(self, e):
        if e.size().width() != e.oldSize().width():
            self._relayout()

    def _relayout(self) -> None:
        fm = QFontMetricsF(self._font)
        w = max(120, self.width())
        y = self.PAD
        self._rows = []
        self._tags = {}
        last_day = None
        text_w = w - 2 * self.PAD - 36
        for i, it in enumerate(self.items):
            if self.group_by_day:
                day = day_label(it.get("ts", 0))
                if day != last_day:
                    last_day = day
                    self._rows.append(("head", QRect(self.PAD, y, w - 2 * self.PAD, self.HEAD_H), day))
                    y += self.HEAD_H
            br = fm.boundingRect(QRectF(0, 0, max(60, text_w), 100000), int(Qt.TextFlag.TextWordWrap), self.body_text(it))
            th = int(br.height()) + 2
            h = 40 + th + 14
            card = QRect(self.PAD, y, w - 2 * self.PAD, h)
            self._rows.append(("card", card, i, QRect(card.left() + 18, card.top() + 38, int(text_w), th)))
            y += h + self.GAP
        self.setMinimumHeight(max(0, y - self.GAP + self.PAD) if self.items else 0)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(400, self.minimumHeight())

    def _card_rect(self, index: int) -> QRect | None:
        for row in self._rows:
            if row[0] == "card" and row[2] == index:
                return row[1]
        return None

    def _action_rects(self, card: QRect) -> list[QRect]:
        n = len(self.actions)
        return [QRect(card.right() - 10 - (n - k) * 32, card.top() + 7, 28, 28) for k in range(n)]

    def _hit(self, pos) -> tuple[int, int]:
        for row in self._rows:
            if row[0] == "card" and row[1].contains(pos):
                for k, r in enumerate(self._action_rects(row[1])):
                    if r.contains(pos):
                        return row[2], k
                return row[2], -1
        return -1, -1

    # ---- interaction
    def _run(self, index: int, action: str) -> None:
        if not (0 <= index < len(self.items)):
            return
        it = self.items[index]
        if action != "trash":
            self._flash[(str(it.get("id")), action)] = time.time()
            self._tick.start()
        {"copy": self.copy_requested, "paste": self.paste_requested, "trash": self.delete_requested}[action].emit(dict(it))
        self.update()

    def _expire_flash(self) -> None:
        now = time.time()
        self._flash = {k: v for k, v in self._flash.items() if now - v < 1.3}
        if not self._flash:
            self._tick.stop()
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        hit = self._hit(pos)
        on_tag = hit[0] >= 0 and hit[1] < 0 and self._tags.get(hit[0]) is not None and self._tags[hit[0]].contains(pos)
        tip = self.actions[hit[1]][1] if hit[1] >= 0 else (self.RULES_ONLY_TIP if on_tag else "")
        if tip != self.toolTip():
            self.setToolTip(tip)
        if hit != self._hover:
            self._hover = hit
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit[1] >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, e):
        self._hover = (-1, -1)
        self.update()

    def mousePressEvent(self, e):
        i, _k = self._hit(e.position().toPoint())
        if i >= 0:
            self._select(i)

    def mouseReleaseEvent(self, e):
        i, k = self._hit(e.position().toPoint())
        if i >= 0 and k >= 0 and e.button() == Qt.MouseButton.LeftButton:
            self._run(i, self.actions[k][0])

    def focusInEvent(self, e):
        self._kbd = e.reason() in _KBD_REASONS
        if self._sel < 0 and self.items:
            self._select(0)
        self.update()

    def focusOutEvent(self, e):
        self._kbd = False
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        n = len(self.items)
        if not n:
            return super().keyPressEvent(e)
        if k in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_Home, Qt.Key.Key_End):
            self._kbd = True
            self._select({Qt.Key.Key_Down: min(n - 1, self._sel + 1), Qt.Key.Key_Up: max(0, self._sel - 1),
                          Qt.Key.Key_Home: 0, Qt.Key.Key_End: n - 1}[k])
            self._reveal(self._sel)
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_C):
            self._run(self._sel, "copy")
        elif k == Qt.Key.Key_P:
            self._run(self._sel, "paste")
        elif k == Qt.Key.Key_Delete and any(a[0] == "trash" for a in self.actions):
            self._run(self._sel, "trash")
        else:
            super().keyPressEvent(e)

    def _reveal(self, index: int) -> None:
        card = self._card_rect(index)
        par = self.parentWidget()
        while par is not None and not hasattr(par, "ensureVisible"):
            par = par.parentWidget()
        if card is not None and par is not None:
            pt = self.mapTo(par.widget(), card.center())
            par.ensureVisible(pt.x(), pt.y(), 10, card.height() // 2 + 24)

    # ---- painting
    def paintEvent(self, ev):
        p = painter(self)
        vis = ev.rect()
        now = time.time()
        for row in self._rows:
            if not row[1].intersects(vis):
                continue
            if row[0] == "head":
                p.setPen(T.TEXT_DIM)
                p.setFont(T.font(T.F_SMALL, T.W_SEMI))
                p.drawText(QRectF(row[1]).adjusted(6, 8, 0, 0), Qt.AlignmentFlag.AlignVCenter, row[2])
                continue
            _kind, card, i, text_rect = row
            it = self.items[i]
            r = QRectF(card).adjusted(0.5, 0.5, -0.5, -0.5)
            hovered = self._hover[0] == i
            selected = i == self._sel and self.hasFocus()
            T.paint_card(p, r, T.R_MD + 2, hover=1.0 if hovered else 0.0)
            command = it.get("mode") == "command"
            # meta line: time · app · words
            parts = [clock_label(it.get("ts", 0))] if self.group_by_day else \
                [f"{day_label(it.get('ts', 0))} {clock_label(it.get('ts', 0))}".strip()]
            if app_label(str(it.get("app_exe") or "")):
                parts.append(app_label(str(it.get("app_exe"))))
            if it.get("words"):
                parts.append(f"{int(it['words'])} words")
            x = r.left() + 18
            p.setFont(self._meta_font)
            p.setPen(T.TEXT_DIM)
            meta_w = r.width() - (x - r.left()) - 16 - len(self.actions) * 32
            meta = "  ·  ".join(s for s in parts if s)          # the meta column always starts with the time (D-06)
            shown = p.fontMetrics().elidedText(meta, Qt.TextElideMode.ElideRight, int(meta_w))
            p.drawText(QRectF(x, r.top() + 9, meta_w, 20), Qt.AlignmentFlag.AlignVCenter, shown)
            # trailing tags: what kind of result this was. Muted on purpose: information, not a warning
            tag_x = x + p.fontMetrics().horizontalAdvance(shown) + 12
            for label, ink, is_rules in (("Command", T.MAGENTA, False),
                                         (self.RULES_ONLY_TAG, T.TEXT_DIM, True)):
                if (is_rules and not it.get("rules_only")) or (not is_rules and not command):
                    continue
                tag_w = p.fontMetrics().horizontalAdvance(label) + 16
                if tag_x + tag_w - x > meta_w + 2:
                    continue
                tag = QRectF(tag_x, r.top() + 10, tag_w, 18)
                p.setPen(QPen(T.alpha(ink, 0.55) if not is_rules else T.STROKE_STRONG, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(tag, 9, 9)
                p.setPen(ink)
                p.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)
                if is_rules:
                    self._tags[i] = tag.toAlignedRect()
                tag_x += tag_w + 8
            # body
            p.setPen(T.TEXT if not self.show_raw else T.TEXT_DIM)
            p.setFont(self._font)
            p.drawText(QRectF(text_rect), int(Qt.TextFlag.TextWordWrap), self.body_text(it))
            # actions: visible on hover / selection, otherwise faint
            for k, (ar, (icon, _tip)) in enumerate(zip(self._action_rects(card), self.actions)):
                arf = QRectF(ar)
                hot = self._hover == (i, k)
                flashed = now - self._flash.get((str(it.get("id")), icon), 0) < 1.3
                if hot:
                    p.fillPath(T.rounded(arf, T.R_SM), T.alpha(T.RED, 0.16) if icon == "trash" else T.CONTROL_HI)
                if flashed:
                    paint_icon(p, "check", arf.adjusted(6, 6, -6, -6), T.GREEN, 2.0)
                else:
                    base = T.TEXT_DIM if (hovered or selected) else T.TEXT_FAINT
                    ink = (T.RED if icon == "trash" else T.TEXT) if hot else base
                    paint_icon(p, icon, arf.adjusted(6, 6, -6, -6), ink)
            if selected and self._kbd:
                T.paint_focus_ring(p, r.adjusted(1, 1, -1, -1), T.R_MD + 1)
