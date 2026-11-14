"""``voxflow.ui.icons``: every icon of the set paints something, in the asked colour, inside its box; an unknown
name paints nothing instead of raising."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from tests.cov_ui_kit import image_bytes
from voxflow.ui import icons

INK = QColor(255, 0, 0)


def _render(name: str, size: int = 48, box: QRectF | None = None, width: float = 1.7) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    icons.paint_icon(p, name, box or QRectF(0, 0, size, size), INK, width)
    p.end()
    return img


def _inked(img: QImage) -> list[tuple[int, int]]:
    return [(x, y) for y in range(img.height()) for x in range(img.width()) if img.pixelColor(x, y).alpha() > 0]


def test_the_set_has_no_duplicate_painters_and_names_are_stable(qapp):
    assert len(icons.NAMES) == len(set(icons.NAMES)) == len(icons._ICONS) >= 47
    for name in ("home", "history", "book", "snippet", "style", "settings", "min", "max", "restore", "close",
                 "moon", "sun", "chevron-right", "chevron-left", "more", "globe", "undo", "archive"):
        assert name in icons.NAMES


@pytest.mark.parametrize("name", icons.NAMES)
def test_every_icon_paints_only_in_its_colour_and_inside_its_box(qapp, name):
    img = _render(name)
    inked = _inked(img)
    assert len(inked) >= 10, f"{name} painted (almost) nothing"
    for x, y in inked[:: max(1, len(inked) // 60)]:
        c = img.pixelColor(x, y)
        assert c.red() > 0 and c.green() == 0 and c.blue() == 0, f"{name} must use only the colour it was given"
    boxed = _render(name, 64, QRectF(16, 16, 32, 32))
    xs, ys = zip(*_inked(boxed))
    assert min(xs) >= 14 and max(xs) <= 50 and min(ys) >= 14 and max(ys) <= 50, f"{name} leaves its rectangle"


def test_icons_differ_from_each_other(qapp):
    pictures = {name: image_bytes(_render(name, 32)) for name in icons.NAMES}
    assert len(set(pictures.values())) == len(pictures), "two names paint the same picture"


def test_an_unknown_name_paints_nothing(qapp):
    assert _inked(_render("no-such-icon")) == []


def test_stroke_weight_stays_constant_on_screen_and_the_painter_state_is_restored(qapp):
    thin, thick = len(_inked(_render("min", 48, width=1.0))), len(_inked(_render("min", 48, width=4.0)))
    assert thick > thin * 2
    img = QImage(24, 24, QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(img)
    before = (p.pen().color().name(), p.transform())
    icons.paint_icon(p, "sun", QRectF(0, 0, 24, 24), INK)
    assert (p.pen().color().name(), p.transform()) == before
    p.end()


def test_qicon_carries_three_scale_factors(qapp):
    ic = icons.icon("search", INK, 16)
    assert not ic.isNull()
    sizes = sorted({(s.width(), s.height()) for s in ic.availableSizes()})
    assert sizes == [(16, 16), (32, 32), (48, 48)], "rendered at 1x, 2x and 3x"
    pm = ic.pixmap(16, 16)
    assert not pm.isNull() and _inked(pm.toImage())
    assert icons.icon("no-such-icon", INK, 16).pixmap(16, 16).toImage().pixelColor(8, 8).alpha() == 0
