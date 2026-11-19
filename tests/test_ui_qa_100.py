"""UI findings of the 1.0 QA pass (Q-01 ... Q-17), each with the fix it guards. Offscreen, on the demo controller."""
from __future__ import annotations

import time

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from tests.cov_ui_kit import flush, make_demo, show  # noqa: F401  (fixtures re-exported)
from voxflow.ui.widgets import ElidedLabel, Segmented

LONG_FOLDER = r"C:\Users\maria.hernandez\OneDrive - Contoso Ltd\Documents\Team shared\VoxFlow sync folder for all my PCs"


def _inside_viewport(page, *names) -> None:
    QApplication.processEvents()
    vp = page.scroll.viewport()
    assert page.content.width() <= vp.width(), f"the page is {page.content.width()} px wide in a {vp.width()} px view"
    for name in names:
        w = page.findChild(QWidget, name)
        right = w.mapTo(vp, QPoint(0, 0)).x() + w.width()
        assert right <= vp.width(), f"{name} ends at x={right}, the view at {vp.width()}"


# ============================================================================ ElidedLabel
def test_an_elided_label_cuts_what_does_not_fit_and_still_reports_the_whole_text(show):
    label = show(ElidedLabel("a fairly long sentence that will not fit into eighty pixels"), width=80)
    assert label.text() == "a fairly long sentence that will not fit into eighty pixels"
    assert label.shown_text().endswith("…") and len(label.shown_text()) < len(label.text())
    assert label.minimumSizeHint().width() <= 80 and label.sizePolicy().horizontalPolicy().name == "Ignored"
    label.resize(900, label.height())
    QApplication.processEvents()
    assert label.shown_text() == label.text(), "room again: the whole text"
    label.setText("")
    assert label.text() == "" and label.shown_text() == ""
    middle = show(ElidedLabel(LONG_FOLDER, mode=Qt.TextElideMode.ElideMiddle), width=200)
    assert middle.shown_text().startswith("C:") and middle.shown_text().endswith("PCs") and "…" in middle.shown_text()


# ============================================================================ Q-01 / Q-09 / Q-10: nothing widens a page
def test_a_long_sync_folder_does_not_push_the_settings_controls_out_of_the_window(make_demo):
    demo = make_demo()
    page = demo.page("settings")
    page.show_tab("data")
    demo.c.config.sync_folder = LONG_FOLDER
    demo.c.demo_set_sync("idle", "", time.time() - 260, ["DESKTOP-7QK2M4P", "LAPTOP-HR3V9T1C", "SURFACE-PRO-9"])
    page.refresh()
    assert LONG_FOLDER in page.sync_folder.text() and "SURFACE-PRO-9" in page.sync_folder.text()
    assert page.sync_folder.toolTip() == LONG_FOLDER, "the whole path is one hover away"
    _inside_viewport(page, "settings_save_history", "settings_backup_export", "settings_sync_choose")
    demo.win.resize(demo.win.minimumSize())
    _inside_viewport(page, "settings_save_history", "settings_backup_export", "settings_sync_choose")
    assert "…" in page.sync_folder.shown_text()


def test_a_long_history_search_does_not_widen_the_page(make_demo):
    demo = make_demo()
    page = demo.page("history")
    demo.win.resize(demo.win.minimumSize())
    word = page.list.items[0]["text"].split()[0]
    query = word + " " * 150                                    # matches, and is far longer than the window is wide
    page.search.setText(query)
    demo.wait(lambda: "for “" in page.count.text())
    assert page.list.items and page.count.text().endswith("”")
    _inside_viewport(page)


def test_the_home_hint_is_cut_not_clipped_for_a_long_shortcut_in_a_small_window(make_demo):
    demo = make_demo(hotkey="ctrl+alt+shift+pagedown")
    page = demo.page("home")
    demo.win.resize(demo.win.minimumSize())
    QApplication.processEvents()
    hint = next(w for w in page.findChildren(ElidedLabel) if w.text().startswith("in any app"))
    right = hint.mapTo(page, QPoint(0, 0)).x() + hint.width()
    assert right <= page.width() and hint.shown_text().endswith("…")
    assert page.caps.mapTo(page, QPoint(0, 0)).x() + page.caps.width() <= page.width(), "the keys are shown whole"


# ============================================================================ Q-03: the practice box stays
def test_the_practice_box_stays_when_the_first_dictation_lands_in_it(make_demo):
    demo = make_demo(empty=True)
    page = demo.page("home")
    assert page.hero.isVisibleTo(page)
    page.practice.setPlainText("My very first dictation.")      # what the paste does
    demo.c.demo_add_transcript("My very first dictation.")
    assert page.hero.isVisibleTo(page), "the box that just received the text is still there"
    assert page.recent.isVisibleTo(page) and page.stats_box.isVisibleTo(page), "and the page moves on around it"
    page.practice.setPlainText("")
    page.practice.clearFocus()
    page.refresh()
    assert not page.hero.isVisibleTo(page), "emptied and left: the first-run card has done its job"
    page.refresh()
    assert not page.hero.isVisibleTo(page), "and it does not come back"


def test_the_practice_box_stays_while_it_has_the_focus(make_demo):
    demo = make_demo(empty=True)
    page = demo.page("home")
    demo.win.activateWindow()
    page.practice.setFocus()
    demo.wait(page.practice.hasFocus)
    demo.c.demo_add_transcript("Pasted somewhere else.")
    assert page.hero.isVisibleTo(page)


# ============================================================================ Q-06: nothing keeps running out of sight
def test_the_microphone_test_ends_when_its_button_leaves_the_screen(make_demo):
    demo = make_demo()
    page = demo.page("settings")
    page.show_tab("audio")
    page.btn_test.click()
    assert page._testing and demo.c._monitor is True
    page.find.setText("zzz no such setting")
    assert not page._testing and demo.c._monitor is False, "hidden by a search: stopped"
    page.show_tab("audio")
    page.btn_test.click()
    page.tabs.set_value("appearance", emit=True)
    assert not page._testing, "another group: stopped"
    page.show_tab("audio")
    page.btn_test.click()
    page.find.setText(page.test_row.title.text())               # a search that still shows the row
    assert page._testing
    page._set_testing(False)


def test_moving_the_flow_bar_ends_when_its_button_leaves_the_screen(make_demo):
    demo = make_demo()
    page = demo.page("settings")
    page.show_tab("appearance")
    page.btn_move.click()
    assert page._arranging and demo.shell.overlay.arranging
    page.tabs.set_value("audio", emit=True)
    assert not page._arranging and not demo.shell.overlay.arranging


# ============================================================================ Q-07: the chosen group ends a search
def test_clicking_the_highlighted_group_ends_a_settings_search(make_demo):
    demo = make_demo()
    page = demo.page("settings")
    page.show_tab("audio")
    page.find.setText("theme")
    assert any(not s.isHidden() and s.tab != "audio" for s in page._sections()), "the search shows other groups"
    tabs = page.tabs
    index = [v for v, _t in tabs.options].index("audio")
    x = int(3 + (tabs.width() - 6) / len(tabs.options) * (index + 0.5))
    QTest.mouseClick(tabs, Qt.MouseButton.LeftButton, pos=QPoint(x, tabs.height() // 2))
    assert page.find.text() == "" and tabs.value == "audio"
    assert all(s.isHidden() == (s.tab != "audio") for s in page._sections())


def test_a_segmented_control_reports_a_click_on_its_chosen_segment_separately(show):
    seg = show(Segmented([(1, "One"), (2, "Two")], 1), width=200)
    changed, again = [], []
    seg.changed.connect(changed.append)
    seg.reselected.connect(again.append)
    QTest.mouseClick(seg, Qt.MouseButton.LeftButton, pos=QPoint(20, 10))
    assert (changed, again) == ([], [1]), "no change, so 'changed' stays silent"
    QTest.mouseClick(seg, Qt.MouseButton.LeftButton, pos=QPoint(180, 10))
    assert (changed, again) == ([2], [1])


# ============================================================================ Q-11: a reset while the bar is being moved
def test_a_position_reset_while_the_flow_bar_is_being_moved_applies_when_moving_ends(make_demo):
    demo = make_demo(overlay_position=[300, 300])
    page = demo.page("settings")
    overlay = demo.shell.overlay
    assert overlay._anchor == QPoint(300, 300)
    page.show_tab("appearance")
    page.btn_move.click()
    demo.c.set("overlay_position", None)                        # Reset settings, or a synced change
    assert overlay._anchor == QPoint(300, 300), "not under the user's hand"
    page.btn_move.click()                                       # Done
    assert not overlay.arranging and overlay._anchor is None, "the default place, as the setting says"


# ============================================================================ Q-12: Tab follows the layout
def test_tab_reaches_share_before_add_in_the_dictionary_and_snippets_headers(make_demo):
    demo = make_demo()
    for key in ("dictionary", "snippets"):
        page = demo.page(key)
        order = [page.actions.itemAt(i).widget() for i in range(page.actions.count())]
        at = order.index(page.btn_share)
        assert order[at + 1] is page.btn_add
        assert order[at - 1].nextInFocusChain() is page.btn_share, "the button left of Share leads to Share"
        assert page.btn_share.nextInFocusChain() is page.btn_add


# ============================================================================ Q-17: no timer for a window never shown
def test_starting_in_the_tray_on_the_settings_page_starts_no_sync_clock(make_demo):
    demo = make_demo(show_window=False, ui_state={"page": "settings"})
    page = demo.win.pages["settings"]
    assert demo.win.current == "settings" and not demo.win.isVisible()
    assert not page._sync_timer.isActive()
    demo.win.present()
    assert page._sync_timer.isActive()
    demo.win.hide()
    assert not page._sync_timer.isActive()


# ============================================================================ Q-02: the keyboard stays in the list
def _activate(demo):
    demo.win.activateWindow()
    QApplication.processEvents()


def test_removing_a_word_with_the_keyboard_leaves_the_focus_on_the_next_rows_button(make_demo):
    demo = make_demo()
    page = demo.page("dictionary")
    _activate(demo)
    buttons = demo.find_all("dict_remove", page)
    assert len(buttons) >= 3
    buttons[1].setFocus()
    demo.wait(buttons[1].hasFocus)
    following = buttons[2].accessibleName()
    QTest.keyClick(buttons[1], Qt.Key.Key_Return)
    flush()
    now = demo.win.focusWidget()
    assert now is not None and now.objectName() == "dict_remove" and now.accessibleName() == following
    last = demo.find_all("dict_remove", page)[-1]
    last.setFocus()
    QTest.keyClick(last, Qt.Key.Key_Return)                     # the last row: the one before it
    flush()
    assert demo.win.focusWidget() is demo.find_all("dict_remove", page)[-1]


def test_removing_the_only_snippet_puts_the_focus_on_the_add_button(make_demo):
    def one(ctl):
        for s in ctl.snippets.list()[1:]:
            ctl.snippets.remove(s["id"])
    demo = make_demo(prepare=one)
    page = demo.page("snippets")
    _activate(demo)
    rm = demo.find("snippets_remove", page)
    rm.setFocus()
    demo.wait(rm.hasFocus)
    QTest.keyClick(rm, Qt.Key.Key_Return)
    flush()
    assert demo.win.focusWidget() is page.btn_add


def test_saving_an_edited_word_returns_the_focus_to_its_row(make_demo):
    demo = make_demo()
    page = demo.page("dictionary")
    _activate(demo)
    edit = demo.find_all("dict_edit", page)[2]
    edit.setFocus()
    demo.wait(edit.hasFocus)
    QTest.keyClick(edit, Qt.Key.Key_Return)
    modal = demo.find("dict_editor")
    term = demo.find("dict_term", modal)
    term.setText(term.text() + "X")
    modal.btn_primary.click()
    flush()
    assert demo.win.focusWidget().objectName() == "dict_edit" and demo.win.focusWidget().isVisibleTo(page)
    assert demo.win.focusWidget() is demo.find_all("dict_edit", page)[2]


def test_focus_that_never_left_or_was_never_on_the_page_is_not_moved(make_demo):
    demo = make_demo()
    page = demo.page("dictionary")
    _activate(demo)
    page.restore_focus(None)                                    # nothing to give back
    page.btn_more.setFocus()                                    # in the page's content, but hidden: no slot
    assert page.focus_slot() is None
    edit = demo.find_all("dict_edit", page)[0]
    edit.setFocus()
    demo.wait(edit.hasFocus)
    slot = page.focus_slot()
    assert slot[:2] == ("dict_edit", 0)
    page.restore_focus(slot)                                    # still there: untouched
    assert demo.win.focusWidget() is edit
    demo.win.nav["home"].setFocus()                             # outside the page's content: no slot
    demo.wait(demo.win.nav["home"].hasFocus)
    assert page.focus_slot() is None


# ============================================================================ Q-08: the tray menu keeps its place
def test_the_tray_menu_keeps_its_highlight_when_a_toggle_or_the_status_refills_it(make_demo):
    demo = make_demo(show_window=False)
    menu = demo.shell.tray.show_menu(QPoint(400, 700))
    QApplication.processEvents()
    for _ in range(3):
        QTest.keyClick(menu, Qt.Key.Key_Down)
    at, was = menu._hover, demo.c.get("cleanup_enabled")
    assert at >= 0
    QTest.keyClick(menu, Qt.Key.Key_Space)
    QApplication.processEvents()
    assert menu.isVisible() and demo.c.get("cleanup_enabled") is (not was)
    assert menu._hover == at, "still on the row that was just used"
    QTest.keyClick(menu, Qt.Key.Key_Down)
    assert menu._hover > at, "and Down goes on from there, not from the top"
    here = menu._hover
    demo.c.demo_set_state("processing")                         # something unrelated happens meanwhile
    QApplication.processEvents()
    assert menu._hover == here and not demo.c.paused
    menu.close()
    menu.set_items(menu.items()[:2])                            # a closed menu, or another list: from the top
    assert menu._hover == -1


# ============================================================================ Q-13: typing clears the red border
def test_typing_an_app_name_clears_the_invalid_mark(make_demo):
    demo = make_demo()
    page = demo.page("style")
    demo.find("style_add_app", page).click()
    assert page.new_exe.property("invalid") is True
    page.new_exe.setText("s")
    assert not page.new_exe.property("invalid")


# ============================================================================ U-05: "Show more" adds, it does not rebuild
def test_show_more_adds_rows_below_the_ones_that_are_there(make_demo):
    def many(ctl):
        for i in range(250):
            ctl.dictionary.add(f"Zeta{i:03d}term")
            ctl.snippets.add(f"phrase number {i:03d}", f"text {i}")
    demo = make_demo(prepare=many)
    for key, edit, noun in (("dictionary", "dict_edit", "words"), ("snippets", "snippets_edit", "snippets")):
        page = demo.page(key)
        section = page.entries if key == "dictionary" else page.rows
        total = len(page._store().list())
        first = list(section._rows)
        assert len(first) == 100 and f"({total - 100:,} not shown)" in page.btn_more.text()
        page.btn_more.click()
        assert section._rows[:100] == first, "the rows that were there are the same widgets"
        assert len(section._rows) == 200 and len(demo.find_all(edit, page)) == 200
        assert page.btn_more.isVisibleTo(page) and f"Show more {noun} ({total - 200:,} not shown)" == page.btn_more.text()
        page.btn_more.click()
        assert len(section._rows) == total and not page.btn_more.isVisibleTo(page)
        assert section._rows[-2].divider and not section._rows[-1].divider, "one list: only the last row has no line"
        page.refresh()
        assert len(section._rows) == total, "a later change keeps what was shown"


def test_show_more_snippets_falls_back_to_the_full_refresh_when_the_list_cannot_be_read(make_demo, monkeypatch):
    def many(ctl):
        for i in range(150):
            ctl.snippets.add(f"phrase number {i:03d}", f"text {i}")
    demo = make_demo(prepare=many)
    page = demo.page("snippets")
    monkeypatch.setattr(demo.c.snippets, "list", lambda: (_ for _ in ()).throw(OSError("disk")))
    page.btn_more.click()
    assert page.empty.isVisibleTo(page) and "could not be loaded" in page.empty.title.text()


def test_a_page_without_rows_or_an_add_button_has_nowhere_to_put_the_focus_and_leaves_it(make_demo):
    demo = make_demo()
    page = demo.page("home")
    _activate(demo)
    page.restore_focus(("no_such_widget", 0, page.practice))    # nothing of that name, and Home has no Add button
    assert demo.win.focusWidget() is None or demo.win.focusWidget().objectName() != "no_such_widget"
