"""Vector icon set, painted with QPainter on a 24 x 24 grid (crisp at any scale factor, themable)."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _line(p: QPainter, *pts: float) -> None:
    path = QPainterPath(QPointF(pts[0], pts[1]))
    for i in range(2, len(pts), 2):
        path.lineTo(pts[i], pts[i + 1])
    p.drawPath(path)


def _home(p):
    _line(p, 4, 11, 12, 4.5, 20, 11)
    _line(p, 6, 10, 6, 19.5, 18, 19.5, 18, 10)
    _line(p, 10, 19.5, 10, 14, 14, 14, 14, 19.5)


def _history(p):
    p.drawEllipse(QPointF(12, 12), 8, 8)
    _line(p, 12, 7.5, 12, 12, 15.2, 14)


def _book(p):
    _line(p, 5, 5.5, 5, 19, 19, 19, 19, 4.5, 7, 4.5)
    p.drawArc(QRectF(5, 4.5, 4, 3), 90 * 16, 90 * 16)
    _line(p, 9, 9, 15, 9)
    _line(p, 9, 12.5, 15, 12.5)
    _line(p, 5, 16, 19, 16)


def _snippet(p):
    _line(p, 13.2, 3.5, 6, 13.2, 11.6, 13.2, 10.6, 20.5, 18, 10.6, 12.4, 10.6, 13.2, 3.5)


def _style(p):
    _line(p, 4, 18.5, 8.5, 6, 13, 18.5)
    _line(p, 5.6, 14.5, 11.4, 14.5)
    p.drawEllipse(QPointF(17.2, 15.4), 2.9, 3.1)
    _line(p, 20.1, 12, 20.1, 18.5)


def _settings(p):
    for y, x in ((7, 15), (12, 9), (17, 14)):
        _line(p, 4, y, x - 2.2, y)
        _line(p, x + 2.2, y, 20, y)
        p.drawEllipse(QPointF(x, y), 2.2, 2.2)


def _search(p):
    p.drawEllipse(QPointF(10.5, 10.5), 6, 6)
    _line(p, 15, 15, 20, 20)


def _copy(p):
    p.drawRoundedRect(QRectF(8.5, 8.5, 11, 11), 2.4, 2.4)
    path = QPainterPath(QPointF(5.5, 15.5))
    path.lineTo(5.5, 6.8)
    path.quadTo(5.5, 4.5, 7.8, 4.5)
    path.lineTo(15.5, 4.5)
    p.drawPath(path)


def _paste(p):
    p.drawRoundedRect(QRectF(5.5, 5.5, 13, 15), 2.4, 2.4)
    p.drawRoundedRect(QRectF(9, 3.5, 6, 4), 1.4, 1.4)
    _line(p, 12, 11, 12, 16.5)
    _line(p, 9.5, 14.2, 12, 16.7, 14.5, 14.2)


def _trash(p):
    _line(p, 4.5, 7, 19.5, 7)
    _line(p, 9.5, 7, 9.5, 4.5, 14.5, 4.5, 14.5, 7)
    _line(p, 6.5, 7, 7.4, 19.5, 16.6, 19.5, 17.5, 7)
    _line(p, 10.2, 10.5, 10.2, 16)
    _line(p, 13.8, 10.5, 13.8, 16)


def _export(p):
    _line(p, 12, 15, 12, 4.5)
    _line(p, 8, 8.3, 12, 4.3, 16, 8.3)
    _line(p, 5, 13.5, 5, 19.5, 19, 19.5, 19, 13.5)


def _import(p):
    _line(p, 12, 4.5, 12, 15)
    _line(p, 8, 11.2, 12, 15.2, 16, 11.2)
    _line(p, 5, 13.5, 5, 19.5, 19, 19.5, 19, 13.5)


def _plus(p):
    _line(p, 12, 5.5, 12, 18.5)
    _line(p, 5.5, 12, 18.5, 12)


def _check(p):
    _line(p, 5.5, 12.5, 10, 17, 18.5, 7.5)


def _close(p):
    _line(p, 6.5, 6.5, 17.5, 17.5)
    _line(p, 17.5, 6.5, 6.5, 17.5)


def _mic(p):
    p.drawRoundedRect(QRectF(9, 3.5, 6, 11), 3, 3)
    p.drawArc(QRectF(5.5, 6, 13, 12), 180 * 16, 180 * 16)
    _line(p, 12, 18, 12, 20.5)
    _line(p, 8.8, 20.5, 15.2, 20.5)


def _keyboard(p):
    p.drawRoundedRect(QRectF(3.5, 6.5, 17, 11), 2.4, 2.4)
    for x in (7, 10.3, 13.7, 17):
        _line(p, x, 10.2, x + 0.01, 10.2)
    _line(p, 8, 14, 16, 14)


def _sound(p):
    _line(p, 4.5, 9.8, 8, 9.8, 12.2, 6, 12.2, 18, 8, 14.2, 4.5, 14.2, 4.5, 9.8)
    p.drawArc(QRectF(10.5, 8.2, 6, 7.6), -55 * 16, 110 * 16)
    p.drawArc(QRectF(9.5, 5.2, 10.5, 13.6), -55 * 16, 110 * 16)


def _moon(p):
    path = QPainterPath(QPointF(19.5, 13.6))
    path.cubicTo(17, 20.5, 7, 20.6, 4.9, 13.4)
    path.cubicTo(3.6, 8.6, 7, 4.6, 11, 4.5)
    path.cubicTo(8.6, 9.5, 13.5, 15.4, 19.5, 13.6)
    p.drawPath(path)


def _sun(p):
    p.drawEllipse(QPointF(12, 12), 3.8, 3.8)
    for i in range(8):
        a = i * math.pi / 4
        _line(p, 12 + 6.6 * math.cos(a), 12 + 6.6 * math.sin(a), 12 + 8.6 * math.cos(a), 12 + 8.6 * math.sin(a))


def _chevron_right(p):
    _line(p, 9.5, 6, 15.5, 12, 9.5, 18)


def _chevron_left(p):
    _line(p, 14.5, 6, 8.5, 12, 14.5, 18)


def _chevron_down(p):
    _line(p, 6.5, 9.5, 12, 15, 17.5, 9.5)


def _edit(p):
    _line(p, 5, 19, 5.9, 15.2, 15.8, 5.3, 18.7, 8.2, 8.8, 18.1, 5, 19)
    _line(p, 13.6, 7.5, 16.5, 10.4)


def _pause(p):
    _line(p, 9, 6, 9, 18)
    _line(p, 15, 6, 15, 18)


def _play(p):
    _line(p, 8, 5.5, 18, 12, 8, 18.5, 8, 5.5)


def _info(p):
    p.drawEllipse(QPointF(12, 12), 8, 8)
    _line(p, 12, 11.2, 12, 16.2)
    _line(p, 12, 8, 12.01, 8)


def _warning(p):
    _line(p, 12, 4.5, 20.5, 19, 3.5, 19, 12, 4.5)
    _line(p, 12, 10, 12, 14.2)
    _line(p, 12, 16.6, 12.01, 16.6)


def _sparkle(p):
    path = QPainterPath(QPointF(10.5, 4.5))
    path.quadTo(11.2, 11.3, 18, 12)
    path.quadTo(11.2, 12.7, 10.5, 19.5)
    path.quadTo(9.8, 12.7, 3, 12)
    path.quadTo(9.8, 11.3, 10.5, 4.5)
    p.drawPath(path)
    _line(p, 18.5, 4.5, 18.5, 8)
    _line(p, 16.75, 6.25, 20.25, 6.25)


def _lock(p):
    p.drawRoundedRect(QRectF(5.5, 10.5, 13, 9.5), 2.4, 2.4)
    p.drawArc(QRectF(8, 4.5, 8, 12), 0, 180 * 16)
    _line(p, 12, 14.2, 12, 16.4)


def _folder(p):
    _line(p, 3.5, 18.5, 3.5, 6, 9.5, 6, 11.5, 8.5, 20.5, 8.5, 20.5, 18.5, 3.5, 18.5)


def _refresh(p):
    p.drawArc(QRectF(5, 5, 14, 14), 35 * 16, 290 * 16)
    _line(p, 18.6, 4.4, 18.2, 8.7, 13.9, 8.2)


def _min(p):
    _line(p, 6.5, 12.5, 17.5, 12.5)


def _max(p):
    p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 1.8, 1.8)


def _restore(p):
    p.drawRoundedRect(QRectF(5.5, 8.5, 10, 10), 1.8, 1.8)
    _line(p, 8.5, 8.3, 8.5, 5.5, 18.5, 5.5, 18.5, 15.5, 15.7, 15.5)


def _more(p):
    for x in (6, 12, 18):
        _line(p, x, 12, x + 0.01, 12)


def _globe(p):
    p.drawEllipse(QPointF(12, 12), 8, 8)
    p.drawEllipse(QPointF(12, 12), 3.4, 8)
    _line(p, 4, 12, 20, 12)


def _clock_fast(p):
    p.drawEllipse(QPointF(13, 12.5), 7, 7)
    _line(p, 13, 8.5, 13, 12.5, 15.8, 14.2)
    _line(p, 2.5, 9, 5, 9)
    _line(p, 2, 12.5, 4, 12.5)
    _line(p, 2.5, 16, 5, 16)


def _flame(p):
    path = QPainterPath(QPointF(12, 3.5))
    path.cubicTo(13, 8, 18, 9.5, 18, 14.5)
    path.cubicTo(18, 18, 15.3, 20.5, 12, 20.5)
    path.cubicTo(8.7, 20.5, 6, 18, 6, 14.5)
    path.cubicTo(6, 12.2, 7.2, 10.6, 8.6, 9.4)
    path.cubicTo(8.8, 11.4, 9.8, 12.2, 10.6, 12.2)
    path.cubicTo(10, 9, 10.5, 5.6, 12, 3.5)
    p.drawPath(path)


def _text(p):
    _line(p, 5, 7, 19, 7)
    _line(p, 5, 12, 19, 12)
    _line(p, 5, 17, 13.5, 17)


def _gauge(p):
    p.drawArc(QRectF(4, 6, 16, 16), 0, 180 * 16)
    _line(p, 12, 14, 16, 9.2)
    _line(p, 4, 14, 4.01, 14)
    _line(p, 20, 14, 20.01, 14)


def _power(p):
    p.drawArc(QRectF(5, 5.5, 14, 14), 125 * 16, 290 * 16)
    _line(p, 12, 3.5, 12, 11.5)


def _window(p):
    p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2.4, 2.4)
    _line(p, 3.5, 9, 20.5, 9)


def _undo(p):
    _line(p, 8.5, 5.5, 4.5, 9.5, 8.5, 13.5)
    path = QPainterPath(QPointF(4.8, 9.5))
    path.lineTo(14, 9.5)
    path.cubicTo(21, 9.5, 21, 18.5, 14, 18.5)
    path.lineTo(9, 18.5)
    p.drawPath(path)


def _share(p):
    for x, y in ((6.5, 12), (17, 6.5), (17, 17.5)):
        p.drawEllipse(QPointF(x, y), 2.4, 2.4)
    _line(p, 8.7, 10.9, 14.8, 7.6)
    _line(p, 8.7, 13.1, 14.8, 16.4)


def _sync(p):
    p.drawArc(QRectF(5, 5, 14, 14), 20 * 16, 140 * 16)
    p.drawArc(QRectF(5, 5, 14, 14), 200 * 16, 140 * 16)
    _line(p, 19.6, 5.6, 18.9, 9.9, 14.7, 9.2)
    _line(p, 4.4, 18.4, 5.1, 14.1, 9.3, 14.8)


def _archive(p):
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 5), 1.6, 1.6)
    _line(p, 5.2, 9.5, 5.2, 19.5, 18.8, 19.5, 18.8, 9.5)
    _line(p, 10, 13.2, 14, 13.2)


_ICONS = {
    "home": _home, "history": _history, "book": _book, "snippet": _snippet, "style": _style,
    "settings": _settings, "search": _search, "copy": _copy, "paste": _paste, "trash": _trash,
    "export": _export, "import": _import, "plus": _plus, "check": _check, "close": _close, "mic": _mic,
    "keyboard": _keyboard, "sound": _sound, "moon": _moon, "sun": _sun, "chevron-right": _chevron_right,
    "chevron-left": _chevron_left, "chevron-down": _chevron_down, "edit": _edit, "pause": _pause,
    "play": _play, "info": _info, "warning": _warning, "sparkle": _sparkle, "lock": _lock,
    "folder": _folder, "refresh": _refresh, "min": _min, "max": _max, "restore": _restore, "more": _more,
    "globe": _globe, "clock-fast": _clock_fast, "flame": _flame, "text": _text, "gauge": _gauge,
    "power": _power, "window": _window, "undo": _undo, "share": _share, "sync": _sync, "archive": _archive,
}
NAMES = tuple(_ICONS)


def paint_icon(p: QPainter, name: str, rect: QRectF, color: QColor, width: float = 1.7) -> None:
    """Paint icon ``name`` fitted into ``rect`` (unknown names paint nothing rather than raising)."""
    fn = _ICONS.get(name)
    if fn is None:
        return
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = min(rect.width(), rect.height()) / 24.0
    p.translate(rect.center().x() - 12 * s, rect.center().y() - 12 * s)
    p.scale(s, s)
    p.setPen(QPen(color, width / max(s, 0.01), Qt.PenStyle.SolidLine,   # constant on-screen stroke weight
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    fn(p)
    p.restore()


def icon(name: str, color: QColor, size: int = 20) -> QIcon:
    """A QIcon for stock-widget slots (rendered at 1x, 2x and 3x so it stays sharp on scaled monitors)."""
    ic = QIcon()
    for k in (1, 2, 3):
        pm = QPixmap(size * k, size * k)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        paint_icon(p, name, QRectF(0, 0, size * k, size * k), color)
        p.end()
        pm.setDevicePixelRatio(k)
        ic.addPixmap(pm)
    return ic
