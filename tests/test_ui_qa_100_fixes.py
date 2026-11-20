"""The remaining UI findings of the 1.0 QA pass, each with the fix it guards: dialogs in a low window (Q-04), the cost
of a theme switch (Q-14), placeholder colours (Q-16), menu keys (Q-15), Ctrl+F (Q-18), rows kept across a change
(U-05), the Style page's "Add an app" row (Q-13), focus after Undo and Add (Q-02), the level meter's clock (U-09) and
long words that widened a page. Offscreen, on the demo controller."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QKeyEvent, QPalette, QTextDocument
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

from tests.cov_ui_kit import flush, make_demo, show  # noqa: F401  (fixtures re-exported)
from tests.helpers import fake_time
from voxflow.ui import dialogs
from voxflow.ui import theme as T
from voxflow.ui import widgets as W
from voxflow.ui.components import Modal, Row, Section, StatusChip, TranscriptList, _Toast
from voxflow.ui.widgets import (Button, HotkeyRecorder, IconButton, Label, LevelMeter, LineEdit, PopupMenu,
                                ScrollArea, TextArea, WrapLabel, breakable, might_be_rich_text, plain_tip)

LONG_WORD = "Pneumonoultramicroscopicsilicovolcanoconiosis" * 3
LONG_LINK = ("https://intranet.example.com/a/very/long/path/that/keeps/going/and/going/without/any/spaces/at/all/"
             "index.html?query=1&more=2")


def _panel_size(demo, width: int, height: int) -> None:
    """Make the window's panel this size (smaller than the usual minimum: a small or scaled screen)."""
    win = demo.win
    win.setMinimumSize(1, 1)
    win.resize(width + 2 * win.M, height + 2 * win.M)
    QApplication.processEvents()
    assert (win.panel.width(), win.panel.height()) == (width, height)


def _top_bottom(widget: QWidget, inside: QWidget) -> tuple[int, int]:
    top = widget.mapTo(inside, QPoint(0, 0)).y()
    return top, top + widget.height()


def _activate(demo) -> None:
    demo.win.activateWindow()
    QApplication.processEvents()


# ============================================================================ Q-04: dialogs in a low window
def _restore_dialog(demo, tmp_path, monkeypatch) -> Modal:
    backup = tmp_path / "my-backup.voxbackup"
    demo.c.export_backup(str(backup))
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: str(backup))
    page = demo.page("settings")
    m = page.import_backup()
    m.mode.set_value("replace", emit=True)
    m.show_error("Tick at least one part to import.")
    QApplication.processEvents()
    return m


def test_a_dialog_too_tall_for_the_window_scrolls_its_body_between_the_title_and_the_buttons(make_demo, tmp_path,
                                                                                            monkeypatch):
    demo = make_demo()
    _panel_size(demo, 800, 500)                                 # 1920x1080 at 200 %
    m = _restore_dialog(demo, tmp_path, monkeypatch)
    card, panel = m.card, demo.win.panel
    assert m.scrolling() and m.scroll.verticalScrollBar().maximum() > 0
    assert _top_bottom(card, panel)[0] >= 20 and _top_bottom(card, panel)[1] <= panel.height() - 20
    # the title above the body, the error and the buttons below it: all of them whole, inside the card
    body = _top_bottom(m.scroll, card)
    assert 0 < _top_bottom(m.title, card)[1] <= body[0]
    for w in (m.error, m.btn_cancel, m.btn_primary):
        top, bottom = _top_bottom(w, card)
        assert body[1] <= top and bottom <= card.height(), f"{w.objectName()} is at {top}..{bottom}"
    # nothing in the body is squeezed: every line has its own place (they were printed on top of each other)
    lines = sorted(_top_bottom(w, m._inner) for w in m._inner.findChildren(Label) if w.isVisibleTo(m._inner))
    assert len(lines) >= 7
    assert all(a[1] <= b[0] for a, b in zip(lines, lines[1:])), lines
    assert all(w.height() >= w.heightForWidth(w.width()) for w in m._inner.findChildren(WrapLabel))
    assert m._inner.width() == m.scroll.viewport().width(), "and nothing is cut off at the side"

    _panel_size(demo, 1080, 720)                                # room again: the whole dialog, no scroll bar
    assert not m.scrolling() and m.scroll.verticalScrollBar().maximum() == 0
    assert m.body.contentsMargins().right() == T.S6 and m.scroll.height() == m._inner.height()
    m.close_modal()


def test_tab_in_a_scrolling_dialog_brings_the_focused_control_into_view_and_stays_inside(make_demo, tmp_path,
                                                                                        monkeypatch):
    demo = make_demo()
    _panel_size(demo, 800, 460)
    _activate(demo)
    m = _restore_dialog(demo, tmp_path, monkeypatch)
    view = m.scroll.viewport()
    seen = []
    for _ in range(12):                                         # once round the dialog and on
        assert m.focusNextPrevChild(True)
        w = demo.win.focusWidget()
        assert m.card.isAncestorOf(w), "the focus stays in the dialog"
        seen.append(w.objectName())
        if m._inner.isAncestorOf(w):
            top, bottom = _top_bottom(w, view)
            assert 0 <= top and bottom <= view.height(), f"{w.objectName()} is at {top}..{bottom} of {view.height()}"
    assert {"backup_import_part_history", "backup_import_mode", "backup_import_cancel", "backup_import_ok"} <= set(seen)
    assert m.focusNextPrevChild(False) and m.card.isAncestorOf(demo.win.focusWidget())

    m.btn_cancel.setFocus()                                     # from the footer, Page Down / Up turn the body
    bar = m.scroll.verticalScrollBar()
    bar.setValue(0)
    QTest.keyClick(m.btn_cancel, Qt.Key.Key_PageDown)
    assert bar.value() > 0
    QTest.keyClick(m.btn_cancel, Qt.Key.Key_PageUp)
    assert bar.value() == 0
    m.close_modal()


def test_the_snippet_editor_and_the_pack_dialog_fit_a_low_window(make_demo):
    def many(ctl):
        for i in range(12):
            ctl.snippets.add(f"phrase number {i:02d}", f"text {i}")
    demo = make_demo(prepare=many)
    _panel_size(demo, 640, 460)                                 # the smallest window there is
    _activate(demo)
    page = demo.page("snippets")
    m = page.edit(demo.c.snippets.list()[0])
    m.show_error("A snippet with that phrase already exists.")
    QApplication.processEvents()
    text = demo.find("snippet_text", m)
    assert m.scrolling() and text.height() == 150, "the text box keeps its height"
    error = _top_bottom(m.error, m.card)
    assert error[0] >= _top_bottom(m.scroll, m.card)[1], "the error is below the body, not printed inside the text box"
    assert error[1] <= _top_bottom(m.btn_primary, m.card)[0]
    m.close_modal()

    m = page.share_pack()
    QApplication.processEvents()
    assert m.scrolling() and _top_bottom(m.btn_primary, m.card)[1] <= m.card.height()
    last = m.entries[-1][1]                                     # a tick box far down the list inside the body
    last.setFocus()
    m._reveal(last)
    listing = demo.find("pack_entries", m)
    for view in (listing.viewport(), m.scroll.viewport()):      # the list scrolled first, then the body
        top, bottom = _top_bottom(last, view)
        assert 0 <= top and bottom <= view.height()
    m._reveal(None)                                             # nothing to bring into view
    m._reveal(m.btn_primary)                                    # the footer never scrolls
    m.close_modal()


def test_a_dialog_in_a_window_far_too_low_keeps_a_usable_body(show):
    host = show(QWidget(), 700, 230)
    m = Modal(host, "Pick the parts", "Everything ticked is imported.", name="tiny")
    for i in range(8):
        m.body.addWidget(Label(f"Line {i}", wrap=False))
    m.add_buttons("Import", lambda: None)
    m.open()
    assert m.scrolling() and m.scroll.height() == Modal.MIN_BODY, "never squeezed to nothing"
    QTest.keyClick(m, Qt.Key.Key_PageDown)
    assert m.scroll.verticalScrollBar().value() > 0
    m.close_modal()

    short = Modal(host, "Sure?", name="short")                  # a body smaller than the least it is given
    short.body.addWidget(Label("One line.", wrap=False))
    short.add_buttons("Yes", lambda: None)
    host.resize(700, 120)
    short.open()
    assert not short.scrolling() and short.scroll.height() == short._inner.sizeHint().height() < Modal.MIN_BODY
    QTest.keyClick(short, Qt.Key.Key_PageDown)                  # nothing to scroll: the key is left alone
    assert short.scroll.verticalScrollBar().value() == 0
    short.close_modal()
    flush()


def test_a_dialog_is_as_tall_as_its_content_at_the_width_it_has(make_demo):
    """The card used to be sized from size hints, which are for another width: up to 36 px too tall."""
    demo = make_demo()
    page = demo.page("dictionary")
    m = page.edit(demo.c.dictionary.list()[0])
    QApplication.processEvents()
    assert not m.scrolling() and m.card.height() == m.card.heightForWidth(m.card.width())
    before = m.card.height()
    m.show_error("That word is already in your dictionary.")
    QApplication.processEvents()
    assert m.card.height() > before and _top_bottom(m.btn_primary, m.card)[1] <= m.card.height() - T.S5 + 1
    m.close_modal()


# ============================================================================ Q-14: a theme switch and the rows
class _StyleChanges(QObject):
    """Counts the "your style changed" events Qt sends to a widget whenever it polishes it again."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def eventFilter(self, obj, ev):
        self.count += ev.type() == QEvent.Type.StyleChange
        return False


def test_a_theme_switch_does_not_polish_the_rows_of_a_page_again(make_demo):
    demo = make_demo()
    page = demo.page("dictionary")
    watch = _StyleChanges()
    rows = demo.find_all("dict_edit", page) + demo.find_all("dict_remove", page) + list(page.entries._rows)
    for w in rows:
        w.installEventFilter(watch)
    area = page.scroll
    sheet = area.styleSheet()
    bar_before = area.verticalScrollBar().styleSheet()
    demo.c.set("theme", "light")
    QApplication.processEvents()
    assert watch.count == 0, "the cost of a theme switch must not grow with the number of rows"
    assert area.styleSheet() == sheet and sheet == T.scroll_area_qss(), "the area's stylesheet is not set again"
    assert area.property("scheme") == "light"
    assert area.verticalScrollBar().styleSheet() == T.scroll_bar_qss() != bar_before, "the scroll bars follow"
    assert T.css(T.alpha(T.LIGHT.TEXT_DIM, 0.35)) in area.verticalScrollBar().styleSheet()


def _tooltip_colours(widget: QWidget) -> tuple:
    QToolTip.showText(widget.mapToGlobal(QPoint(4, 4)), "Edit this word", widget)
    QApplication.processEvents()
    tip = next(w for w in QApplication.topLevelWidgets() if w.inherits("QTipLabel") and w.isVisible())
    image = tip.grab().toImage()
    out = (tip.height(), image.pixelColor(4, tip.height() // 2).getRgb()[:3])
    QToolTip.hideText()
    tip.hide()
    QApplication.processEvents()
    return out


def test_tooltips_inside_a_page_follow_a_live_theme_switch(make_demo):
    demo = make_demo()
    page = demo.page("dictionary")
    button = demo.find("dict_edit", page)
    height, colour = _tooltip_colours(button)
    assert colour == T.DARK.PANEL_HI.getRgb()[:3] and height > 30, "the padded tooltip, in the dark colours"
    demo.c.set("theme", "light")
    QApplication.processEvents()
    assert _tooltip_colours(button) == (height, T.LIGHT.PANEL_HI.getRgb()[:3]), "the same tooltip, now light"


def test_a_scrolling_area_names_the_contrast_scheme_to_its_stylesheet(show, monkeypatch):
    area = show(ScrollArea(QWidget()), 200, 100)
    assert T.scheme() == "dark" and area.property("scheme") == "dark"
    monkeypatch.setattr(T, "HIGH_CONTRAST", True)
    area._restyle()
    assert T.scheme() == "contrast" and area.property("scheme") == "contrast"
    assert 'QScrollArea[scheme="contrast"] QToolTip { background: palette(window)' in area.styleSheet()


def test_a_label_that_is_not_on_screen_takes_the_new_colours_when_it_is_shown(show):
    host = show(QWidget(), 300, 100)
    seen, hidden = Label("seen", parent=host), Label("hidden", role="TEXT_DIM", parent=host)
    seen.show()
    colour = lambda w: w.palette().color(QPalette.ColorRole.WindowText)  # noqa: E731
    T.apply_theme("light", False)
    assert colour(seen) == T.LIGHT.TEXT, "on screen: at once"
    assert colour(hidden) == T.DARK.TEXT_DIM and hidden._stale, "not on screen: not yet (hundreds of rows are not)"
    hidden.show()
    assert colour(hidden) == T.LIGHT.TEXT_DIM and not hidden._stale
    hidden.hide()
    T.apply_theme("dark", False)
    hidden.set_role("RED")                                      # a new role does not wait
    assert colour(hidden) == T.DARK.RED and not hidden._stale


def test_the_pages_that_were_not_open_during_a_theme_switch_have_the_new_colours_when_opened(make_demo):
    demo = make_demo()
    for key in demo.win.pages:
        demo.page(key)
    demo.page("settings")
    demo.c.set("theme", "light")
    page = demo.page("dictionary")
    row = page.entries._rows[0]
    assert row.title.palette().color(QPalette.ColorRole.WindowText) == T.LIGHT.TEXT
    assert row.sub.palette().color(QPalette.ColorRole.WindowText) == T.LIGHT.TEXT_DIM
    assert page.title.palette().color(QPalette.ColorRole.WindowText) == T.LIGHT.TEXT


# ============================================================================ Q-16: placeholder colour
def _placeholder(widget: QWidget) -> tuple:
    QApplication.processEvents()
    return widget.palette().color(QPalette.ColorRole.PlaceholderText).getRgb()


def test_placeholder_text_has_the_same_colour_after_a_fresh_start_and_after_a_live_theme_switch(make_demo):
    fresh = make_demo(theme="light")
    boxes = lambda demo: [demo.page("history").search, demo.page("settings").find, demo.page("home").practice]  # noqa: E731
    want = T.LIGHT.TEXT_FAINT.getRgb()
    assert [_placeholder(w) for w in boxes(fresh)] == [want] * 3, "a fresh start in the light theme"
    live = make_demo()                                          # dark, then switched while it is open
    shown = boxes(live)
    assert [_placeholder(w) for w in shown] == [T.DARK.TEXT_FAINT.getRgb()] * 3
    live.c.set("theme", "light")
    assert [_placeholder(w) for w in shown] == [want] * 3, "switched live: the same colour"
    live.c.set("theme", "dark")
    assert [_placeholder(w) for w in shown] == [T.DARK.TEXT_FAINT.getRgb()] * 3


def test_a_text_box_made_later_has_the_placeholder_colour_of_the_theme(show):
    T.apply_theme("light", True)
    line, area = show(LineEdit("", "name")), show(TextArea("", "text"))
    assert _placeholder(line) == _placeholder(area) == T.LIGHT.TEXT_FAINT.getRgb()
    line.set_invalid(True)                                      # the red border re-applies the stylesheet
    assert _placeholder(line) == T.LIGHT.TEXT_FAINT.getRgb()


# ============================================================================ Q-15: menu keys
LANGUAGES = ["Auto-detect", "English", "German", "French", "Spanish", "Italian", "Portuguese", "Dutch", "Polish",
             "Swedish", "Danish", "Finnish", "Czech", "Greek", "Hungarian", "Romanian", "Bulgarian", "Croatian",
             "Slovak", "Slovenian", "Estonian", "Latvian", "Lithuanian", "Maltese", "Russian", "Ukrainian"]


@pytest.fixture
def menu(qtbot, qt_messages):
    m = PopupMenu(240)
    qtbot.addWidget(m)
    m.set_items([{"kind": "choice", "text": name, "enabled": name != "Greek"} for name in LANGUAGES])
    m.popup_at(QPoint(400, 600), anchor_bottom=False)
    QApplication.processEvents()
    yield m
    m.close()
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


def _at(menu: PopupMenu) -> str:
    return menu.items()[menu._hover]["text"] if menu._hover >= 0 else ""


def _row_in_view(menu: PopupMenu) -> bool:
    r = menu._rects[menu._hover]
    return r.top() - menu._scroll >= menu._margin and r.bottom() - menu._scroll <= menu.height() - menu._margin


def test_page_down_and_page_up_move_a_view_at_a_time_in_a_long_menu(menu):
    assert menu._content_h > menu.height(), "the list is longer than the menu is tall"
    QTest.keyClick(menu, Qt.Key.Key_PageDown)
    assert _at(menu) == "Auto-detect", "nothing highlighted yet: the first row"
    QTest.keyClick(menu, Qt.Key.Key_PageDown)
    first_page = menu._hover
    per_page = (menu.height() - 2 * menu._margin) // menu.ROW_H["choice"]
    assert first_page == per_page - 1 and _row_in_view(menu)
    assert menu.accessibleName() == _at(menu), "announced like a move with the arrow keys"
    assert 2 * first_page > len(LANGUAGES) - 1, "the next page is the end of this list"
    QTest.keyClick(menu, Qt.Key.Key_PageDown)
    assert _at(menu) == "Ukrainian" and _row_in_view(menu) and menu._scroll > 0, "it stops on the last row"
    QTest.keyClick(menu, Qt.Key.Key_PageDown)
    assert _at(menu) == "Ukrainian"
    QTest.keyClick(menu, Qt.Key.Key_PageUp)
    assert menu._hover == len(LANGUAGES) - 1 - first_page and _row_in_view(menu)
    for _ in range(6):
        QTest.keyClick(menu, Qt.Key.Key_PageUp)
    assert _at(menu) == "Auto-detect" and menu._scroll == 0
    menu._set_hover(-1)
    QTest.keyClick(menu, Qt.Key.Key_PageUp)
    assert _at(menu) == "Ukrainian", "nothing highlighted: Page Up starts from the end"


def test_page_keys_cope_with_a_menu_lower_than_a_row_and_with_nothing_to_choose(menu, qtbot):
    menu._max_h = 60                                            # lower than two rows
    menu._layout()
    menu._go(3)
    menu._page(1)
    assert menu._hover == 4, "at least one row further"
    menu._page(-1)
    assert menu._hover == 3
    info = PopupMenu()
    qtbot.addWidget(info)
    info.set_items([{"kind": "info", "text": "Nothing to choose from"}])
    info._page(1)
    info._type("n")
    assert info._hover == -1


def test_typing_in_a_menu_goes_to_the_row_that_starts_with_it(menu, monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(W, "time", fake_time(monotonic=lambda: clock["t"]))

    def key(char: str, wait: float = 0.2) -> str:
        clock["t"] += wait
        QTest.keyClicks(menu, char)
        return _at(menu)

    assert key("s") == "Spanish" and _row_in_view(menu) and menu.accessibleName() == "Spanish"
    assert key("s") == "Swedish", "the same letter again: the next row with that letter"
    assert key("s") == "Slovak" and key("s") == "Slovenian"
    assert key("s") == "Spanish", "and round again"
    assert key("P", wait=2) == "Portuguese", "after a pause a new word starts; capitals do not matter"
    assert key("o") == "Portuguese" and key("l") == "Polish", "letters in quick succession are the start of a name"
    assert key("z") == "Polish", "nothing starts with 'polz': it stays where it is"
    assert key("g", wait=2) == "German" and key("g") == "German", "a row that cannot be chosen (Greek) is passed over"
    assert key("d", wait=2) == "Dutch" and key("a") == "Danish"
    clock["t"] += 2
    QTest.keyClick(menu, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
    assert _at(menu) == "Danish", "Ctrl+E is not typing"
    picked = []
    menu.items()[menu._hover]["on_click"] = lambda: picked.append(_at(menu))
    QTest.keyClick(menu, Qt.Key.Key_Space)
    assert picked == ["Danish"], "Space still chooses the row"
    menu.set_items([{"kind": "choice", "text": "One"}])         # another list: the typed letters are forgotten
    assert menu._typed == ""


# ============================================================================ Q-18: Ctrl+F
def test_ctrl_f_goes_to_the_search_box_of_the_page_that_is_open(make_demo):
    def more_words(ctl):
        for i in range(7):
            ctl.dictionary.add(f"Extra{i}")
    demo = make_demo(prepare=more_words)
    _activate(demo)
    win = demo.win
    for key, name in (("dictionary", "dict_search"), ("style", "style_app_search"), ("settings", "settings_search"),
                      ("history", "history_search")):
        page = demo.page(key)
        box = demo.find(name, page)
        box.setText("abc")
        win.nav[key].setFocus()
        page.scroll.verticalScrollBar().setValue(0)
        QApplication.processEvents()
        win._focus_search()
        QApplication.processEvents()
        assert win.current == key and win.focusWidget() is box, f"{key}: its own search box"
        assert box.selectedText() == "abc", "with the text selected, ready to be typed over"
        top, bottom = _top_bottom(box, page)
        assert 0 <= top and bottom <= page.height(), f"{key}: on screen (on Style it is half-way down the page)"
        if key == "style":
            assert page.scroll.verticalScrollBar().value() > 0, "the page was scrolled to it"
        box.setText("")


def test_ctrl_f_goes_to_history_from_a_page_without_a_usable_search_box(make_demo):
    demo = make_demo()                                          # six words: the Dictionary shows no search box
    _activate(demo)
    win = demo.win
    for key in ("dictionary", "snippets", "home"):
        page = demo.page(key)
        assert page.search_box() is None
        win._focus_search()
        assert win.current == "history" and win.focusWidget() is win.pages["history"].search
    settings = demo.page("settings")
    settings.find.setEnabled(False)
    assert settings.search_box() is None, "a box that cannot be typed in does not count"


def test_ctrl_f_does_nothing_under_an_open_dialog(make_demo):
    demo = make_demo()
    _activate(demo)
    page = demo.page("dictionary")
    m = page.edit(None)
    term = demo.find("dict_term", m)
    term.setText("half-typed")
    demo.win._focus_search()
    assert demo.win.current == "dictionary" and demo.win._modal is m and term.text() == "half-typed"
    m.close_modal()


# ============================================================================ U-05: rows are kept across a change
def _many_words(ctl) -> None:
    for i in range(40):
        ctl.dictionary.add(f"Word{i:02d}", [f"word {i}"])
        ctl.snippets.add(f"phrase number {i:02d}", f"text {i}")


def _next_stop(widget: QWidget) -> QWidget:
    """The widget Tab goes to from ``widget`` (the focus chain also holds the widgets Tab passes over)."""
    w = widget.nextInFocusChain()
    while w.focusPolicy() == Qt.FocusPolicy.NoFocus:
        w = w.nextInFocusChain()
    return w


def _tab_order(page, name_a: str, name_b: str) -> list[str]:
    """The accessible names of the row buttons in the order Tab visits them."""
    first = page.findChildren(QWidget, name_a)[0]
    out, w = [], first
    while w.objectName() in (name_a, name_b):
        out.append(w.accessibleName())
        w = _next_stop(w)
    return out


def test_a_change_to_the_dictionary_touches_only_the_rows_that_changed(make_demo):
    demo = make_demo(prepare=_many_words)
    page, c = demo.page("dictionary"), demo.c
    rows = list(page.entries._rows)
    built = []
    real = page._entry_row
    page._entry_row = lambda e: (built.append(e["term"]), real(e))[1]
    titles = lambda: [r.title.text() for r in page.entries._rows]  # noqa: E731
    expected = lambda: [e["term"] for e in c.dictionary.list()]  # noqa: E731

    c.dictionary.add("Aardvark", ["hard park"])                 # a new word, wherever the store puts it
    at = expected().index("Aardvark")
    assert built == ["Aardvark"] and titles() == expected()
    assert [r for r in page.entries._rows if r.title.text() != "Aardvark"] == rows, "the other rows are the same widgets"
    entry = c.dictionary.list()[5]
    c.dictionary.update(entry["id"], entry["term"], ["new alias"])
    assert built == ["Aardvark", entry["term"]] and titles() == expected()
    assert page.entries._rows[5].sub.text() == "Heard as: new alias"
    gone = c.dictionary.list()[9]
    c.dictionary.remove(gone["id"])
    assert len(built) == 2 and titles() == expected() and gone["term"] not in titles()
    flush()
    # one list still: a line below every row but the last, and the buttons in the order of the rows, for
    # ``findChildren`` (the page looks a button up by its place) and for Tab
    assert [r.divider for r in page.entries._rows] == [True] * (len(titles()) - 1) + [False]
    assert [b.accessibleName() for b in demo.find_all("dict_edit", page)] == [f"Edit {t}" for t in titles()]
    assert _tab_order(page, "dict_edit", "dict_remove") == [f"{verb} {t}" for t in titles() for verb in ("Edit", "Remove")]
    assert at >= 0


def test_accepting_a_suggestion_builds_one_row_and_keeps_the_rest(make_demo):
    demo = make_demo(prepare=_many_words)
    page = demo.page("dictionary")
    words, suggestions = list(page.entries._rows), list(page.inbox._rows)
    assert len(suggestions) == 2
    built = []
    real_word, real_suggestion = page._entry_row, page._suggestion_row
    page._entry_row = lambda e: (built.append(e["term"]), real_word(e))[1]
    page._suggestion_row = lambda s: (built.append("suggestion"), real_suggestion(s))[1]
    refreshes = []
    real_refresh = page.refresh
    page.refresh = lambda: (refreshes.append(1), real_refresh())[1]
    term = suggestions[0].title.text()
    demo.find_all("dict_suggestion_accept", page)[0].click()
    assert len(refreshes) == 1, "two signals arrive (the suggestion is gone, the word is there): one refresh"
    assert built == [term], "and in it one row is built"
    assert page.inbox._rows == suggestions[1:] and not page.inbox._rows[0].divider
    assert [r for r in page.entries._rows if r.title.text() != term] == words
    assert f"Added “{term}”" in demo.toasts()[-1]


def test_a_page_refreshes_once_for_everything_one_call_changed(make_demo):
    demo = make_demo()
    page, c = demo.page("dictionary"), demo.c
    refreshes = []
    real_refresh = page.refresh
    page.refresh = lambda: (refreshes.append(len(c.dictionary.list())), real_refresh())[1]

    def three_changes():
        c.dictionary.add("Alpha")
        page.run(c.dictionary.add, "Beta")                      # a call inside a call: still held
        assert refreshes == []
        c.dictionary.add("Gamma")
    assert page.run(three_changes) is True
    assert refreshes == [9], "one refresh, when everything is in"
    assert {"Alpha", "Beta", "Gamma"} <= {r.title.text() for r in page.entries._rows}

    def fails():
        c.dictionary.add("Delta")
        raise ValueError("That did not go through.")
    assert page.run(fails) is None and refreshes == [9, 10], "what changed before the failure is shown as well"
    assert demo.toasts()[-1] == "That did not go through."
    assert page.run(lambda: None) is True and len(refreshes) == 2, "nothing changed: nothing refreshed"
    demo.page("home")
    page.run(c.dictionary.add, "Epsilon")                       # not the page on screen: refreshed when it is shown
    assert len(refreshes) == 2 and page._dirty
    demo.page("dictionary")
    assert len(refreshes) == 3


def test_a_change_to_the_snippets_and_show_more_keep_the_rows(make_demo):
    def many(ctl):
        for i in range(130):
            ctl.snippets.add(f"phrase number {i:03d}", f"text {i}")
    demo = make_demo(prepare=many)
    page, c = demo.page("snippets"), demo.c
    rows = list(page.rows._rows)
    assert len(rows) == 100
    first = c.snippets.list()[0]
    c.snippets.update(first["id"], first["trigger"], "another text")
    assert page.rows._rows[1:] == rows[1:] and page.rows._rows[0] is not rows[0]
    assert page.rows._rows[0].sub.text() == "another text"
    kept = list(page.rows._rows)
    page.btn_more.click()
    assert page.rows._rows[:100] == kept and len(page.rows._rows) == len(c.snippets.list())
    assert not page.rows.card.isHidden()
    names = [b.accessibleName() for b in demo.find_all("snippets_edit", page)]
    assert names == [f"Edit the snippet {s['trigger']}" for s in c.snippets.list()]


def test_section_sync_keeps_moves_adds_and_removes_rows(show):
    host = show(QWidget(), 420, 600)
    before = Button("Before", parent=host)
    sec = Section("Rows", host)
    sec.setGeometry(0, 40, 420, 500)
    before.show()
    sec.show()
    made = []

    def build(item: str) -> Row:
        made.append(item)
        box = QWidget()
        IconButton("edit", f"Edit {item}", box).setObjectName("edit")
        return Row(item, "", box, icon="mic" if item == "with icon" else None)

    order = lambda: [w.accessibleName() for w in sec.findChildren(IconButton)]  # noqa: E731
    sync = lambda items: sec.sync(items, lambda it: it, build)  # noqa: E731
    sync(["b"])
    assert made == ["b"] and not sec._rows[0].divider and not sec._rows[0].isHidden()
    row_b = sec._rows[0]
    sync(["a", "b"])                                            # one new row, above the one that is there
    assert made == ["b", "a"] and sec._rows[1] is row_b and order() == ["Edit a", "Edit b"]
    assert [r.divider for r in sec._rows] == [True, False]
    edit_a, edit_b = sec.findChildren(IconButton)
    assert _next_stop(before) is edit_a and _next_stop(edit_a) is edit_b, "Tab: Before, a, b"
    sync(["a", "b", "c", "d", "b"])                             # several new ones at once, and one name twice
    assert made == ["b", "a", "c", "d", "b"] and len(sec._rows) == 5 and sec._rows[4] is not row_b
    assert not sec.card.isHidden() and all(not r.isHidden() for r in sec._rows)
    assert order() == ["Edit a", "Edit b", "Edit c", "Edit d", "Edit b"]
    row_a, row_c, row_d = sec._rows[0], sec._rows[2], sec._rows[3]
    sync(["d", "a", "c"])                                       # two removed, one moved to the front
    flush()
    assert made == ["b", "a", "c", "d", "b"] and sec._rows == [row_d, row_a, row_c]
    assert order() == ["Edit d", "Edit a", "Edit c"] and [r.divider for r in sec._rows] == [True, True, False]
    assert [w.accessibleName() for w in (_next_stop(before), _next_stop(_next_stop(before)))] == ["Edit d", "Edit a"]
    sync(["a", "c", "d"])                                       # the first row moved to the end
    assert sec._rows == [row_a, row_c, row_d] and order() == ["Edit a", "Edit c", "Edit d"]
    sync([])
    flush()
    assert sec._rows == [] and sec.findChildren(Row) == []

    sec.card.hide()                                             # a card that is hidden stays hidden
    sync(["x", "y", "z"])
    assert sec.card.isHidden() and len(sec._rows) == 3
    sec.clear()
    flush()
    assert sec._rows == [] and sec._keyed == {}

    icons = Section("Icons", host, icons=True)                  # one row with an icon: every title is indented
    icons.sync(["plain", "with icon"], lambda it: it, build)
    assert all(r._outer.contentsMargins().left() == T.S4 + 30 for r in icons._rows) and icons._indent
    icons.sync(["plain"], lambda it: it, build)
    assert icons._rows[0]._outer.contentsMargins().left() == T.S4 and not icons._indent
    bare = Section("No controls", host)                         # rows with nothing to tab to
    bare.sync(["one", "two"], lambda it: it, lambda it: Row(it))
    bare.sync(["zero", "one", "two"], lambda it: it, lambda it: Row(it))
    assert [r.title.text() for r in bare._rows] == ["zero", "one", "two"]


# ============================================================================ Q-13: the Style page's "Add an app" row
def test_a_half_typed_app_name_survives_a_change_to_the_styles(make_demo):
    demo = make_demo()
    page = demo.page("style")
    box, add_row = page.new_exe, page.add_row
    assert demo.find("style_new_exe", page) is box and demo.find("style_add_app", page).isVisibleTo(page)
    assert demo.find("style_new_category", page) is page.new_cat and page.apps._rows[0] is add_row
    box.setText("obsid")
    page.new_cat.set_value("email")
    demo.c.styles.set_tone("work", "casual")                    # a tone above, or a change that came in by sync
    demo.c.styles.set_app_category("zed.exe", "work")
    page.app_search.setText("zed")
    page.app_search.setText("")
    assert page.new_exe is box and box.text() == "obsid" and page.new_cat.value == "email"
    assert page.apps._rows[0] is add_row and demo.find_all("style_new_exe", page) == [box]


def test_a_style_search_without_a_match_says_so(make_demo):
    demo = make_demo()
    page = demo.page("style")
    line = demo.find("style_no_match", page)
    assert not line.isVisibleTo(page)
    page.app_search.setText("zzzz")
    assert line.isVisibleTo(page) and line.text().startswith("No app in the list matches “zzzz”.")
    assert [r.title.text() for r in page.apps._rows] == ["Add an app"], "the way to add it is still there"
    page.app_search.setText("slack")
    assert not line.isVisibleTo(page) and len(page.apps._rows) == 2
    page.app_search.setText("")
    assert not line.isVisibleTo(page) and len(page.apps._rows) > 5


def test_app_rows_are_kept_when_another_app_changes(make_demo):
    demo = make_demo()
    page = demo.page("style")
    rows = {r.title.text(): r for r in page.apps._rows}
    exe = next(r["exe"] for r in demo.c.styles.app_categories() if r.get("builtin"))
    demo.c.styles.set_app_category(exe, "email")                # now "your choice": that row is new, and first
    after = {r.title.text(): r for r in page.apps._rows}
    changed = [name for name in after if after[name] is not rows.get(name)]
    assert len(changed) == 1 and page.apps._rows[1] is after[changed[0]] and page.apps._rows[0] is page.add_row
    assert after[changed[0]].sub.text() == f"{exe}  ·  your choice"


# ============================================================================ Q-02: focus after Undo and after Add
def test_undo_of_a_deleted_transcript_selects_it_again(make_demo):
    demo = make_demo()
    page = demo.page("history")
    _activate(demo)
    lst = page.list
    lst.setFocus()
    QTest.keyClick(lst, Qt.Key.Key_Down)
    QTest.keyClick(lst, Qt.Key.Key_Down)
    entry = dict(lst.selected())
    QTest.keyClick(lst, Qt.Key.Key_Delete)
    assert lst.selected()["id"] != entry["id"] and demo.toasts() == ["Transcript deleted."]
    demo.win.nav["home"].setFocus()                             # the focus is somewhere else by now
    demo.win.toasts._toasts[0].btn.click()                      # Undo
    assert lst.selected()["id"] == entry["id"], "the transcript that came back is the chosen one"
    assert lst.hasFocus(), "and the keyboard is back in the list"
    assert not lst.select_id("no-such-id") and lst.selected()["id"] == entry["id"]


def test_undo_of_a_transcript_that_is_not_listed_leaves_the_focus_alone(make_demo):
    demo = make_demo()
    page = demo.page("history")
    _activate(demo)
    entry = dict(page.list.items[0])
    page._delete(entry)
    page.search.setText("no transcript says this")
    page._debounce.stop()
    page._search_changed()
    demo.win.nav["home"].setFocus()
    demo.win.toasts._toasts[0].btn.click()                      # Undo while a search hides the transcript
    assert demo.c.history.count() == 9 and demo.win.nav["home"].hasFocus()


def test_adding_an_app_leaves_the_focus_in_the_name_box(make_demo, monkeypatch):
    demo = make_demo()
    page = demo.page("style")
    _activate(demo)
    add = demo.find("style_add_app", page)
    add.setFocus()
    add.click()                                                 # nothing typed
    assert page.new_exe.hasFocus() and page.new_exe.property("invalid") is True
    page.new_exe.setText("signal.exe")
    add.setFocus()
    add.click()
    assert demo.toasts()[-1] == "Signal added." and page.new_exe.hasFocus() and page.new_exe.text() == ""
    assert any(r.title.text() == "Signal" for r in page.apps._rows)
    page.new_exe.setText("broken.exe")

    def refuse(*_a):
        raise ValueError("That app cannot be added.")
    monkeypatch.setattr(demo.c.styles, "set_app_category", refuse)
    QTest.keyClick(page.new_exe, Qt.Key.Key_Return)
    assert page.new_exe.text() == "broken.exe" and page.new_exe.hasFocus(), "refused: the name is still there to fix"
    assert demo.toasts()[-1] == "That app cannot be added."


# ============================================================================ U-09: the level meter's clock
def test_the_level_meter_has_no_clock_running_without_a_source(show, monkeypatch):
    clock = {"t": 10.0}
    monkeypatch.setattr(W, "time", fake_time(perf_counter=lambda: clock["t"]))
    meter = show(LevelMeter(None), 200)
    assert not meter._timer.isActive(), "shown, nothing to follow: no clock"
    meter.set_source(lambda: 0.4)
    assert meter._timer.isActive()
    clock["t"] += 0.05
    meter._tick()
    assert meter._level > 0 and meter._timer.isActive()
    meter.set_source(None)                                      # the test ended: the bar falls back, then rests
    assert meter._timer.isActive(), "still falling"
    for _ in range(200):
        clock["t"] += 0.1
        meter._tick()
    assert meter._level == 0.0 == meter._peak and not meter._timer.isActive(), "at rest: the clock stops"
    meter.set_level(0.5)                                        # a level pushed in by hand falls back the same way
    assert meter._timer.isActive()
    meter.set_level(0.6)
    assert meter._timer.isActive()
    meter.hide()
    meter.set_source(lambda: 0.2)
    assert not meter._timer.isActive(), "hidden: never"
    meter.show()
    assert meter._timer.isActive()


def test_the_audio_settings_run_the_meter_only_during_a_microphone_test(make_demo):
    demo = make_demo()
    page = demo.page("settings")
    page.show_tab("audio")
    QApplication.processEvents()
    meter = page.meter
    assert meter.isVisible() and not meter._timer.isActive(), "the group is shown, no test is running"
    page.btn_test.click()
    assert meter._timer.isActive()
    meter.set_level(0.5)
    page.btn_test.click()                                       # Stop
    assert meter.source is None and meter._timer.isActive(), "the peak mark is still falling"
    demo.wait(lambda: not meter._timer.isActive(), timeout=6000)
    assert meter._level == 0.0 == meter._peak


# ============================================================================ long words do not widen anything
def test_breakable_text_and_the_label_that_shows_it():
    assert breakable("short words stay as they are") == "short words stay as they are"
    assert breakable("x" * 24) == "x" * 24
    assert breakable("a" * 60) == "a" * 24 + "​" + "a" * 24 + "​" + "a" * 12
    assert breakable(f"see {LONG_LINK} now").replace("​", "") == f"see {LONG_LINK} now"


def test_a_wrap_label_breaks_a_long_word_and_reports_the_text_as_given(show):
    plain, wrapping = Label(LONG_WORD), WrapLabel(LONG_WORD, weight=T.W_SEMI)
    assert wrapping.text() == LONG_WORD and "​" in wrapping.accessibleName() + Label.text(wrapping)
    assert wrapping.minimumSizeHint().width() < 260 < plain.minimumSizeHint().width()
    show(wrapping, 300)
    assert wrapping.heightForWidth(300) > 2 * plain.sizeHint().height(), "it takes more lines instead"
    wrapping.setText(None)
    assert wrapping.text() == ""


def _fits(page) -> None:
    QApplication.processEvents()
    view = page.scroll.viewport()
    assert page.content.width() <= view.width(), f"the page is {page.content.width()} px wide in a {view.width()} px view"


def test_long_words_in_user_data_do_not_widen_a_page(make_demo, tmp_path):
    def long_data(ctl):
        ctl.dictionary.add(LONG_WORD, [LONG_WORD.lower() + "x"])
        ctl.snippets.add(LONG_WORD.lower()[:90], LONG_LINK)
        ctl.styles.set_app_category(LONG_WORD.lower()[:80] + "_x64.exe", "work")
        ctl.demo_add_transcript(LONG_LINK + " " + LONG_WORD)
    demo = make_demo(prepare=long_data)
    demo.win.resize(demo.win.minimumSize())
    for key in ("home", "history", "dictionary", "snippets", "style"):
        _fits(demo.page(key))
    style = demo.page("style")
    style.app_search.setText("q" * 120)                         # the "no match" line quotes the search
    _fits(style)
    style.app_search.setText("")
    dictionary = demo.page("dictionary")
    row = next(r for r in dictionary.entries._rows if r.title.text() == LONG_WORD)
    button = row.findChild(QWidget, "dict_remove")
    assert button.mapTo(dictionary, QPoint(0, 0)).x() + button.width() <= dictionary.width(), "its buttons are on screen"
    pack = tmp_path / (LONG_WORD[:100] + ".voxpack")
    demo.c.export_pack("dictionary", str(pack), LONG_WORD[:80], LONG_LINK)
    m = dictionary.import_pack(str(pack))
    QApplication.processEvents()
    assert m._inner.width() == m.scroll.viewport().width() <= m.card.width(), "the facts stay inside the dialog"
    m.show_error(f"“{LONG_WORD}” is already in your dictionary.")
    QApplication.processEvents()
    assert m.error.width() <= m.card.width() - 2 * T.S6 and m.error.text().endswith("dictionary.")
    m.close_modal()
    history = demo.page("history")
    card = history.list._rows[1]
    assert card[0] == "card" and card[3].height() > 3 * 20, "the long word in the transcript wraps instead of being cut"


def test_a_long_hotkey_message_wraps_instead_of_widening_the_settings(make_demo):
    demo = make_demo()
    demo.win.resize(demo.win.minimumSize())
    page = demo.page("settings")
    rec = page.recorders["command_hotkey"]
    rec.show_message("A single modifier key would trigger all the time. Combine two, such as Ctrl + Win.")
    _fits(page)
    assert rec.msg.isVisibleTo(page) and rec.msg.height() > rec.msg.fontMetrics().height() * 1.5, "on two lines"
    rec.show_message("Saved", "GREEN", 1800)
    demo.wait(lambda: rec.msg.height() < rec.msg.fontMetrics().height() * 1.5)
    rec._msg_timer.stop()


def test_a_long_word_in_a_toast_or_a_transcript_wraps(show):
    host = show(QWidget(), 700, 400)
    toast = _Toast("success", f"Removed “{LONG_WORD}”.", "", None, host)
    size = toast.layout_for(420)
    assert toast.message == f"Removed “{LONG_WORD}”." and size.width() <= 420 + 2 * _Toast.PAD
    assert size.height() > 44 + 2 * _Toast.PAD + 20, "several lines"
    toast.show()
    assert not toast.grab().isNull()
    lst = show(TranscriptList(group_by_day=False), 400)
    lst.set_items([{"id": "1", "ts": 0, "text": LONG_WORD}])
    assert lst.body_text(lst.items[0]) == LONG_WORD, "what is copied has no break marks in it"
    assert lst._rows[0][3].height() > 2 * 20 and not lst.grab().isNull()


def test_a_recorder_message_still_fits_one_line_when_it_is_short(show):
    rec = show(HotkeyRecorder("ctrl+alt+win", name="Dictation"))
    rec.show_message("Alt + Win is already used for Dictation.")
    QApplication.processEvents()
    assert rec.msg.wordWrap() and rec.msg.heightForWidth(rec.width()) < rec.msg.fontMetrics().height() * 1.5
    rec._msg_timer.stop()


def test_enter_in_a_dialog_still_reaches_the_default_button_when_the_body_scrolls(show):
    host = show(QWidget(), 700, 260)
    done = []
    m = Modal(host, "Name it", name="named")
    field = LineEdit("", "name")
    m.body.addWidget(field)
    for i in range(6):
        m.body.addWidget(Label(f"Line {i}", wrap=False))
    m.add_buttons("Save", lambda: done.append(field.text()))
    m.open(field)
    assert m.scrolling() and field.hasFocus()
    field.setText("x")
    QApplication.sendEvent(field, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert done == ["x"], "Enter typed in a field of the scrolling body still saves"
    m.close_modal()
    flush()


# ============================================================================ S-02: tooltips are never HTML
MARKUP = '<b>x</b> <img src="//host/share/a.png">'


def _as_shown(tip: str) -> str:
    """What Qt puts on screen for a tooltip text (rich text when it looks like HTML), and no image in it."""
    if not might_be_rich_text(tip):
        return tip
    doc = QTextDocument()
    doc.setHtml(tip)
    assert "<img" not in tip and "<b>" not in tip, "the markup is escaped, so nothing is rendered or fetched"
    return doc.toPlainText()


def test_a_tooltip_shows_markup_in_user_data_as_typed(make_demo):
    assert might_be_rich_text(MARKUP), "Qt would render this as rich text, and load the picture"
    assert plain_tip("Edit Kubernetes") == "Edit Kubernetes" and plain_tip(None) == ""
    assert _as_shown(plain_tip(f"two  spaces\n{MARKUP}")) == f"two  spaces\n{MARKUP}"

    def markup(ctl):
        ctl.dictionary.add(MARKUP, ["x"])
        ctl.snippets.add(MARKUP.lower(), "text")
    demo = make_demo(prepare=markup)
    page = demo.page("dictionary")
    row = next(r for r in page.entries._rows if r.title.text() == MARKUP)
    for name, verb in (("dict_edit", "Edit"), ("dict_remove", "Remove")):
        button = row.findChild(QWidget, name)
        assert _as_shown(button.toolTip()) == f"{verb} {MARKUP}" and button.toolTip() != f"{verb} {MARKUP}"
        assert button.accessibleName() == f"{verb} {MARKUP}", "a screen reader gets the words themselves"
    for name in ("snippets_edit", "snippets_remove"):
        tips = [w.toolTip() for w in demo.find_all(name, demo.page("snippets"))]
        marked = [t for t in tips if "&lt;" in t]
        assert marked and all(not might_be_rich_text(t) or "<b>" not in t for t in tips), (name, tips)
    settings = demo.page("settings")
    folder = "C:/Sync/<b>team</b>"
    settings.sync_folder.setToolTip(folder)
    assert _as_shown(settings.sync_folder.toolTip()) == folder
    assert "<b>" not in settings.sync_folder.toolTip()


def test_the_status_tooltip_shows_a_detail_with_markup_as_text(show):
    chip = show(StatusChip(), 200)
    chip.set_status({"level": "error", "title": "Microphone", "detail": f"{MARKUP} is not connected"})
    assert _as_shown(chip.toolTip()) == f"{MARKUP} is not connected" and "<img" not in chip.toolTip()
    assert chip.accessibleDescription() == f"{MARKUP} is not connected"
