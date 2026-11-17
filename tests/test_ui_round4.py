"""Closing pass UI (ui b309e12, cf0caee on core ff7a04c): choosing the parts of a backup to import, choosing the
entries of a shared pack, the CheckBox control, the quiet "Basic clean-up" tag on history rows and the dimmed
secondary hint on the Flow bar. Offscreen, against the REAL Controller; every file in ``tmp_path``; every file
dialog replaced.
"""
from __future__ import annotations

import json

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import QApplication, QWidget

from tests.test_ui import UI, make_ui, seed, ui  # noqa: F401 - fixtures
from tests.test_ui_backup import make_backup, open_from, other, populate, save_to, texts  # noqa: F401
from voxflow import controller as ctl_mod
from voxflow.ui import overlay as overlay_mod
from voxflow.ui import theme as T
from voxflow.ui.components import TranscriptList as HistoryList
from voxflow.ui.widgets import CheckBox

PART_KEYS = ["settings", "dictionary", "snippets", "styles", "history"]


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


def terms(c) -> set:
    return {e["term"] for e in c.dictionary.list()}


def triggers(c) -> set:
    return {e["trigger"] for e in c.snippets.list()}


def open_import(ui, monkeypatch, path):
    ui.page("settings")
    open_from(monkeypatch, path)
    ui.find("settings_backup_import").click()
    return ui.find("backup_import")


def part(ui, key: str) -> CheckBox:
    return ui.find(f"backup_import_part_{key}")


# =========================================================================== 1. parts of a backup
def test_import_dialog_offers_every_part_the_file_holds(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui()
    dialog = open_import(ui, monkeypatch, path)
    assert list(dialog.parts) == PART_KEYS
    info = ui.c.inspect_backup(str(path))
    offered = {key for key, _label in ui.c.backup_parts()}
    assert set(PART_KEYS) | {"stats"} == offered, "statistics travel with the history; every other part has a box"
    for key in PART_KEYS:
        box = part(ui, key)
        assert box.isVisibleTo(dialog) and box.accessibleName() == box.text() and box.text()
        assert (key in info["includes"]) is True
    assert part(ui, "dictionary").note == "2 words" and part(ui, "snippets").note == "1 snippet"
    assert part(ui, "history").note == "2 transcripts and statistics"
    assert "WHAT TO IMPORT" in texts(dialog)
    ui.render(dialog)


def test_merge_shows_settings_unticked_and_says_why_and_replace_gives_the_choice_back(other, make_ui, monkeypatch,
                                                                                      tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui()
    open_import(ui, monkeypatch, path)
    mode, settings = ui.find("backup_import_mode"), part(ui, "settings")
    assert mode.value == "merge"
    assert not settings.isChecked() and not settings.isEnabled()
    assert "Merge never changes this" in settings.toolTip() and "Replace" in settings.toolTip()
    for key in PART_KEYS[1:]:
        assert part(ui, key).isChecked() and part(ui, key).isEnabled()

    mode.set_value("replace", emit=True)
    assert settings.isChecked() and settings.isEnabled() and settings.toolTip() == ""
    settings.click()                                              # the user does not want the settings
    assert not settings.isChecked()
    mode.set_value("merge", emit=True)
    assert not settings.isChecked() and not settings.isEnabled()
    mode.set_value("replace", emit=True)
    assert not settings.isChecked(), "the user's own choice for Replace is remembered across a look at Merge"
    settings.click()
    mode.set_value("merge", emit=True)
    mode.set_value("replace", emit=True)
    assert settings.isChecked()


def test_parts_missing_from_the_file_are_shown_off_and_cannot_be_ticked(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path, history=False)
    ui = make_ui()
    seed(ui.rig, 2)
    open_import(ui, monkeypatch, path)
    box = part(ui, "history")
    assert not box.isEnabled() and not box.isChecked() and box.toolTip() == "Not in this backup"
    box.click()
    assert not box.isChecked()
    ui.find("backup_import_mode").set_value("replace", emit=True)
    assert not box.isEnabled() and not box.isChecked(), "Replace does not make a missing part available"
    ui.find("backup_import_ok").click()
    assert ui.c.history.count() == 2, "a replace from a file without history leaves the history alone"
    assert "Your profile was replaced with the backup." in ui.toasts(), \
        "every part the file holds was ticked: that is the plain, whole import"


def test_merge_with_only_the_dictionary_ticked_adds_only_words(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "dark"})
    populate(ui.c)
    seed(ui.rig, 1)
    dialog = open_import(ui, monkeypatch, path)
    for key in ("snippets", "styles", "history"):
        part(ui, key).click()
    calls = []
    real = ui.c.import_backup
    monkeypatch.setattr(ui.c, "import_backup", lambda *a, **k: calls.append((a, k)) or real(*a, **k))
    ui.find("backup_import_ok").click()
    assert not dialog.isVisible()
    assert calls[0][0][1:] == ("merge", ["dictionary"])
    assert terms(ui.c) == {"Kubernetes", "VoxFlow", "Anthropic"}
    assert triggers(ui.c) == {"my address"} and ui.c.history.count() == 1 and ui.c.get("theme") == "dark"
    assert "Backup merged: 1 word added." in ui.toasts()


def test_replace_with_some_parts_unticked_leaves_those_alone(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "dark"})
    populate(ui.c)
    seed(ui.rig, 4)
    open_import(ui, monkeypatch, path)
    ui.find("backup_import_mode").set_value("replace", emit=True)
    part(ui, "settings").click()
    part(ui, "history").click()
    part(ui, "dictionary").click()
    assert "ticked parts" in ui.find("backup_import_explain").text()
    ui.find("backup_import_ok").click()
    assert triggers(ui.c) == {"sig"}, "snippets were ticked: replaced"
    assert terms(ui.c) == {"Kubernetes", "VoxFlow"}, "dictionary was not ticked: untouched"
    assert ui.c.history.count() == 4 and ui.c.get("theme") == "dark"
    assert "The parts you ticked were replaced with the backup's." in ui.toasts()
    assert "Your profile was replaced with the backup." not in ui.toasts()


def test_replace_with_history_ticked_also_takes_the_statistics(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui()
    seed(ui.rig, 5)
    words_before = ui.c.stats.summary()["total_words"]
    open_import(ui, monkeypatch, path)
    ui.find("backup_import_mode").set_value("replace", emit=True)
    for key in ("settings", "dictionary", "snippets", "styles"):
        part(ui, key).click()
    calls = []
    real = ui.c.import_backup
    monkeypatch.setattr(ui.c, "import_backup", lambda *a, **k: calls.append(a) or real(*a, **k))
    ui.find("backup_import_ok").click()
    assert calls[0][1:] == ("replace", ["history", "stats"])
    assert ui.c.history.count() == 2
    assert ui.c.stats.summary()["total_words"] != words_before, "the statistics describe the imported history"


def test_all_parts_ticked_is_the_plain_whole_import(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui()
    open_import(ui, monkeypatch, path)
    calls = []
    real = ui.c.import_backup
    monkeypatch.setattr(ui.c, "import_backup", lambda *a, **k: calls.append(a) or real(*a, **k))
    ui.find("backup_import_ok").click()
    assert calls[0][1:] == ("merge", None), "nothing unticked: no part list is passed"


@pytest.mark.parametrize("mode", ["merge", "replace"])
def test_nothing_ticked_asks_for_a_part_and_imports_nothing(other, make_ui, monkeypatch, tmp_path, mode):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "dark"})
    populate(ui.c)
    dialog = open_import(ui, monkeypatch, path)
    ui.find("backup_import_mode").set_value(mode, emit=True)
    for key in PART_KEYS:
        if part(ui, key).isChecked():
            part(ui, key).click()
    assert not any(part(ui, key).isChecked() for key in PART_KEYS)
    toasts = list(ui.toasts())
    ui.find("backup_import_ok").click()
    assert dialog.isVisible() and dialog.error.text() == "Tick at least one part to import."
    assert terms(ui.c) == {"Kubernetes", "VoxFlow"} and ui.toasts() == toasts and ui.c.get("theme") == "dark"
    part(ui, "dictionary").click()
    assert dialog.error.text() == "", "ticking a part clears the message"
    ui.find("backup_import_ok").click()
    assert not dialog.isVisible() and "Anthropic" in terms(ui.c)


def test_merge_never_passes_settings_even_when_the_box_was_ticked_for_replace(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "dark"})
    open_import(ui, monkeypatch, path)
    mode = ui.find("backup_import_mode")
    mode.set_value("replace", emit=True)
    assert part(ui, "settings").isChecked()
    mode.set_value("merge", emit=True)
    ui.find("backup_import_ok").click()
    assert ui.c.get("theme") == "dark" and "Backup merged" in " ".join(ui.toasts())


def test_parts_dialog_renders_in_both_themes_and_boxes_are_reachable_by_keyboard(other, make_ui, monkeypatch,
                                                                                 tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "light"})
    dialog = open_import(ui, monkeypatch, path)
    ui.render(dialog)
    ui.c.set("theme", "dark")
    ui.render(dialog)
    box = part(ui, "snippets")
    assert box.focusPolicy() & Qt.FocusPolicy.TabFocus
    box.setFocus(Qt.FocusReason.TabFocusReason)
    ui.qtbot.keyClick(box, Qt.Key.Key_Space)
    assert not box.isChecked()
    for key in PART_KEYS:                                         # no label is cut off in the two-column grid
        b = part(ui, key)
        need = 28 + QFontMetricsF(b._font).horizontalAdvance(b.text())
        assert b.width() >= need, f"{key}: label {b.text()!r} does not fit ({b.width()} < {need:.0f})"


# =========================================================================== 2. entries of a shared pack
def many_terms(c, n: int) -> list[str]:
    names = [f"Term{i:02d}" for i in range(n)]
    for name in names:
        c.dictionary.add(name, [f"alias {name.lower()}"])
    return names


def open_share(ui, page: str):
    ui.page(page)
    ui.find("dict_share" if page == "dictionary" else "snippets_share").click()
    return ui.find("pack_export")


def test_share_dialog_lists_every_entry_ticked_with_a_count(ui):
    populate(ui.c)
    dialog = open_share(ui, "dictionary")
    boxes = ui.find_all("pack_entry", dialog)
    assert [b.text() for b in boxes] == [e["term"] for e in ui.c.dictionary.list()]
    assert all(b.isChecked() for b in boxes) and len(dialog.entries) == 2
    assert ui.find("pack_selected_count").text() == "IN THE PACK  ·  2 OF 2"
    kube = next(b for b in boxes if b.text() == "Kubernetes")
    assert kube.note == "cooper netties" and kube.accessibleDescription() == "cooper netties"
    assert "you tick below" in texts(dialog)
    assert ui.find("pack_select_all").accessibleName() and ui.find("pack_select_none").accessibleName()
    ui.render(dialog)


def test_snippet_share_dialog_shows_the_snippet_text_on_one_line(ui):
    ui.c.snippets.add("sig", "Best regards,\nDan")
    dialog = open_share(ui, "snippets")
    box = ui.find("pack_entry", dialog)
    assert box.text() == "sig" and box.note == "Best regards, Dan"


def test_only_ticked_terms_end_up_in_the_pack(ui, monkeypatch, tmp_path):
    names = many_terms(ui.c, 6)
    dialog = open_share(ui, "dictionary")
    boxes = {b.text(): b for b in ui.find_all("pack_entry", dialog)}
    for name in names[:4]:
        boxes[name].click()
    assert ui.find("pack_selected_count").text() == "IN THE PACK  ·  2 OF 6"
    target = tmp_path / "two.voxpack"
    save_to(monkeypatch, target)
    ui.find("pack_name").setText("Two")
    ui.find("pack_export_ok").click()
    assert not dialog.isVisible() and "Pack saved: 2 words in two.voxpack." in ui.toasts()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert sorted(e["term"] for e in data["entries"]) == names[4:]
    assert ui.c.inspect_pack(str(target))["count"] == 2
    assert len(ui.c.dictionary.list()) == 6, "sharing never changes the list"


def test_only_ticked_snippets_end_up_in_the_pack(ui, monkeypatch, tmp_path):
    for trig in ("alpha", "beta", "gamma"):
        ui.c.snippets.add(trig, f"text of {trig}")
    dialog = open_share(ui, "snippets")
    next(b for b in ui.find_all("pack_entry", dialog) if b.text() == "beta").click()
    target = tmp_path / "s.voxpack"
    save_to(monkeypatch, target)
    ui.find("pack_name").setText("S")
    ui.find("pack_export_ok").click()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["kind"] == "snippets" and sorted(e["trigger"] for e in data["entries"]) == ["alpha", "gamma"]
    assert "Pack saved: 2 snippets in s.voxpack." in ui.toasts()


def test_all_ticked_exports_the_whole_list_without_an_id_list(ui, monkeypatch, tmp_path):
    populate(ui.c)
    open_share(ui, "dictionary")
    calls = []
    real = ui.c.export_pack
    monkeypatch.setattr(ui.c, "export_pack", lambda *a, **k: calls.append(a) or real(*a, **k))
    save_to(monkeypatch, tmp_path / "all.voxpack")
    ui.find("pack_name").setText("All")
    ui.find("pack_export_ok").click()
    assert calls[0][4] is None and ui.c.inspect_pack(str(tmp_path / "all.voxpack"))["count"] == 2


def test_select_none_and_all_buttons_and_the_zero_case(ui, monkeypatch, tmp_path):
    many_terms(ui.c, 5)
    dialog = open_share(ui, "dictionary")
    boxes = ui.find_all("pack_entry", dialog)
    target = tmp_path / "none.voxpack"
    asked = save_to(monkeypatch, target)
    ui.find("pack_name").setText("Nothing")
    ui.find("pack_select_none").click()
    assert not any(b.isChecked() for b in boxes)
    assert ui.find("pack_selected_count").text() == "IN THE PACK  ·  0 OF 5"
    ui.find("pack_export_ok").click()
    assert dialog.isVisible() and dialog.error.text() == "Tick at least one word to put in the pack."
    assert not asked and not target.exists(), "no file dialog and no file for an empty selection"
    boxes[2].click()
    assert dialog.error.text() == "" and ui.find("pack_selected_count").text() == "IN THE PACK  ·  1 OF 5"
    ui.find("pack_select_all").click()
    assert all(b.isChecked() for b in boxes) and ui.find("pack_selected_count").text() == "IN THE PACK  ·  5 OF 5"
    ui.find("pack_export_ok").click()
    assert target.is_file() and ui.c.inspect_pack(str(target))["count"] == 5


def test_zero_ticked_snippets_uses_the_snippet_word(ui, monkeypatch, tmp_path):
    ui.c.snippets.add("sig", "Best")
    dialog = open_share(ui, "snippets")
    ui.find("pack_name").setText("X")
    ui.find("pack_select_none").click()
    ui.find("pack_export_ok").click()
    assert dialog.error.text() == "Tick at least one snippet to put in the pack."


def test_a_long_list_scrolls_inside_a_dialog_that_stays_on_screen(ui, monkeypatch, tmp_path):
    names = many_terms(ui.c, 60)
    dialog = open_share(ui, "dictionary")
    listing = ui.find("pack_entries", dialog)
    assert listing.height() <= 150 and len(ui.find_all("pack_entry", dialog)) == 60
    bar = listing.verticalScrollBar()
    QApplication.processEvents()
    assert bar.maximum() > 0, "60 entries must scroll, not grow the dialog"
    assert dialog.height() <= ui.win.height()
    last = ui.find_all("pack_entry", dialog)[-1]
    listing.ensureWidgetVisible(last)
    QApplication.processEvents()
    assert bar.value() > 0
    last.click()
    assert ui.find("pack_selected_count").text() == "IN THE PACK  ·  59 OF 60"
    target = tmp_path / "long.voxpack"
    save_to(monkeypatch, target)
    ui.find("pack_name").setText("Long")
    ui.find("pack_export_ok").click()
    packed = {e["term"] for e in json.loads(target.read_text(encoding="utf-8"))["entries"]}
    assert len(packed) == 59 and last.text() not in packed and set(names) - packed == {last.text()}
    ui.render()


def test_a_short_list_is_not_given_an_empty_scroll_box(ui):
    ui.c.dictionary.add("Solo")
    dialog = open_share(ui, "dictionary")
    listing = ui.find("pack_entries", dialog)
    assert 30 <= listing.height() <= 40 and listing.verticalScrollBar().maximum() == 0


def test_an_entry_deleted_while_the_dialog_is_open_gives_a_sentence_not_a_crash(ui, monkeypatch, tmp_path):
    populate(ui.c)
    dialog = open_share(ui, "dictionary")
    boxes = {b.text(): b for b in ui.find_all("pack_entry", dialog)}
    boxes["VoxFlow"].click()                                      # only Kubernetes stays ticked ...
    gone = next(e for e in ui.c.dictionary.list() if e["term"] == "Kubernetes")
    ui.c.dictionary.remove(gone["id"])                            # ... and it disappears (e.g. through sync)
    target = tmp_path / "gone.voxpack"
    save_to(monkeypatch, target)
    ui.find("pack_name").setText("Gone")
    ui.find("pack_export_ok").click()
    assert not target.exists()
    shown = ui.find_all("pack_export")
    message = (shown[0].error.text() if shown and shown[0].isVisible() else "") + " ".join(ui.toasts())
    assert "Pick at least one entry to share." in message


# =========================================================================== 3. the CheckBox control
@pytest.fixture
def box(qtbot):
    host = QWidget()
    host.resize(320, 60)
    cb = CheckBox("History", True, "12 transcripts", host)
    cb.setGeometry(10, 10, 300, 28)
    host.show()
    qtbot.addWidget(host)
    QApplication.processEvents()
    yield cb
    T.apply_theme("dark", False)


def test_checkbox_starts_as_asked_and_describes_itself(box):
    assert box.isChecked() and box.isCheckable()
    assert box.text() == "History" and box.accessibleName() == "History"
    assert box.accessibleDescription() == "12 transcripts" and box.note == "12 transcripts"
    assert box.height() == 28 and box.sizeHint().height() == 28
    assert box.sizeHint().width() > CheckBox("History").sizeHint().width(), "the note counts towards the width"
    assert not CheckBox("x", False).isChecked()


def test_checkbox_toggles_by_mouse_and_emits_once_per_change(box, qtbot):
    seen = []
    box.toggled.connect(seen.append)
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton)
    assert seen == [False] and not box.isChecked()
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton, pos=QPoint(box.width() - 5, 14))   # anywhere on the row
    assert seen == [False, True] and box.isChecked()


@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_checkbox_toggles_by_keyboard(box, qtbot, key):
    seen = []
    box.toggled.connect(seen.append)
    box.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(box, key)
    assert seen == [False], f"one press of {key} is one change"
    qtbot.keyClick(box, key)
    assert seen == [False, True]


def test_checkbox_set_from_code_emits_only_when_the_value_changes(box):
    seen = []
    box.toggled.connect(seen.append)
    box.setChecked(True)
    assert seen == []
    box.setChecked(False)
    box.setChecked(False)
    assert seen == [False]


def test_disabled_checkbox_ignores_mouse_and_keyboard(box, qtbot):
    seen = []
    box.toggled.connect(seen.append)
    box.setEnabled(False)
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton)
    qtbot.keyClick(box, Qt.Key.Key_Space)
    assert seen == [] and box.isChecked()
    box.setChecked(False)                                         # code may still change it
    assert seen == [False]


def _pixels(widget) -> bytes:
    image = widget.grab().toImage()
    return bytes(image.constBits())


def test_checkbox_paints_each_state_differently_in_both_themes(box):
    shots = {}
    for theme in ("dark", "light"):
        T.apply_theme(theme, False)
        for checked in (True, False):
            for enabled in (True, False):
                box.setEnabled(True)
                box.setChecked(checked)
                box.setEnabled(enabled)
                QApplication.processEvents()
                shots[(theme, checked, enabled)] = _pixels(box)
    assert len(set(shots.values())) == 8, "theme, tick and disabled state must each change the picture"
    box.setEnabled(True)


def test_checkbox_shows_a_focus_ring_for_keyboard_focus_only(box, qtbot):
    box.clearFocus()
    plain = _pixels(box)
    box.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert box.hasFocus() and box._kbd and _pixels(box) != plain, "Tab focus must be visible"
    box.clearFocus()
    box.setFocus(Qt.FocusReason.MouseFocusReason)
    QApplication.processEvents()
    assert not box._kbd and _pixels(box) == plain, "a mouse click does not draw the keyboard ring"


def test_checkbox_elides_a_long_label_instead_of_painting_outside(qtbot):
    host = QWidget()
    cb = CheckBox("A very long dictionary term that cannot possibly fit in a narrow column", True, "note", host)
    cb.setGeometry(0, 0, 90, 28)
    host.resize(200, 40)
    host.show()
    qtbot.addWidget(host)
    assert cb.minimumSizeHint().width() <= 90
    image = host.grab().toImage()
    background = image.pixel(150, 20)
    assert all(image.pixel(x, y) == background for x in range(92, 200, 4) for y in range(2, 38, 4))


# =========================================================================== 4. "Basic clean-up" on history rows
def add_entries(rig):
    h = rig.c.history
    plain = h.add("Plain sentence cleaned by the model.", "plain sentence raw", 300, app_exe="slack.exe",
                  duration_s=2.0)
    basic = h.add("Basic sentence cleaned by rules only.", "basic sentence raw um", 4100, app_exe="slack.exe",
                  duration_s=2.0, rules_only=True)
    command = h.add("Rewritten by command mode.", "make it formal", 900, app_exe="slack.exe", duration_s=2.0,
                    mode="command")
    return plain, basic, command


def tag_rects(ui, lst) -> dict:
    ui.render(lst)
    return {lst.items[i]["id"]: rect for i, rect in lst._tags.items()}


def test_the_flag_is_stored_only_on_entries_that_have_it(ui):
    plain, basic, command = add_entries(ui.rig)
    assert basic.get("rules_only") is True
    assert "rules_only" not in plain and "rules_only" not in command
    assert ui.c.history.get(basic["id"]).get("rules_only") is True


def test_tag_is_painted_only_on_rows_cleaned_by_rules(make_ui):
    made = {}
    ui = make_ui(prepare=lambda rig: made.update(zip(("plain", "basic", "command"), add_entries(rig))))
    page = ui.page("history")
    lst = page.list
    assert len(lst.items) == 3
    tags = tag_rects(ui, lst)
    assert set(tags) == {made["basic"]["id"]}
    rect = tags[made["basic"]["id"]]
    card = lst._card_rect(next(i for i, it in enumerate(lst.items) if it["id"] == made["basic"]["id"]))
    assert card.contains(rect), "the tag sits inside its own card"
    need = QFontMetricsF(lst._meta_font).horizontalAdvance(HistoryList.RULES_ONLY_TAG)
    assert rect.width() >= need + 8, "the words 'Basic clean-up' fit their outline"
    assert HistoryList.RULES_ONLY_TAG == "Basic clean-up"


def test_an_old_history_file_without_the_key_shows_no_tag(make_ui, tmp_path):
    def old_file(rig):
        seed(rig, 2)
        path = rig.data_dir / "history.json"
        assert path.is_file() and "rules_only" not in path.read_text(encoding="utf-8")

    ui = make_ui(prepare=old_file)
    page = ui.page("history")
    assert len(page.list.items) == 2 and tag_rects(ui, page.list) == {}


@pytest.mark.parametrize("value", [False, None, 0, "", "true", 1])
def test_only_a_real_true_in_the_history_file_counts(tmp_path, value):
    """A hand-edited or damaged history file cannot switch the tag on with something that is not ``true``."""
    from voxflow import store
    first = store.History(tmp_path)
    entry = first.add("Tampered entry.", "tampered", 100, rules_only=True)
    first.flush()                                               # the history is written behind (QA D17)
    path = tmp_path / "history.json"
    text = path.read_text(encoding="utf-8")
    assert '"rules_only": true' in text
    path.write_text(text.replace('"rules_only": true', '"rules_only": ' + json.dumps(value)), encoding="utf-8")
    again = store.History(tmp_path)
    loaded = again.get(entry["id"])
    assert loaded is not None and not loaded.get("rules_only")


def test_tag_survives_search_and_the_raw_view(make_ui):
    made = {}
    ui = make_ui(prepare=lambda rig: made.update(zip(("plain", "basic", "command"), add_entries(rig))))
    page = ui.page("history")
    lst = page.list
    ui.find("history_search").setText("rules only")
    ui.wait(lambda: len(lst.items) == 1)
    assert set(tag_rects(ui, lst)) == {made["basic"]["id"]}
    page.view.set_value(True, emit=True)
    assert lst.show_raw and set(tag_rects(ui, lst)) == {made["basic"]["id"]}, "raw view keeps the tag"
    ui.find("history_search").setText("plain")
    ui.wait(lambda: len(lst.items) == 1 and lst.items[0]["id"] == made["plain"]["id"])
    assert tag_rects(ui, lst) == {}, "no stale tag from the previous result"
    ui.find("history_search").setText("")
    ui.wait(lambda: len(lst.items) == 3)
    assert set(tag_rects(ui, lst)) == {made["basic"]["id"]}


def test_tag_explains_itself_on_hover_and_to_screen_readers(make_ui):
    made = {}
    ui = make_ui(prepare=lambda rig: made.update(zip(("plain", "basic", "command"), add_entries(rig))))
    page = ui.page("history")
    lst = page.list
    rect = tag_rects(ui, lst)[made["basic"]["id"]]
    ui.qtbot.mouseMove(lst, rect.center())
    QApplication.processEvents()
    assert lst.toolTip() == HistoryList.RULES_ONLY_TIP
    assert "language model was too slow" in lst.toolTip() and "as you said them" in lst.toolTip()
    card = lst._card_rect(next(i for i, it in enumerate(lst.items) if it["id"] == made["plain"]["id"]))
    ui.qtbot.mouseMove(lst, QPoint(card.left() + 30, card.bottom() - 12))
    QApplication.processEvents()
    assert lst.toolTip() == "", "the explanation goes away with the pointer"

    lst.setFocus()
    index = next(i for i, it in enumerate(lst.items) if it["id"] == made["basic"]["id"])
    for _ in range(index + 1):
        ui.qtbot.keyClick(lst, Qt.Key.Key_Down)
        if lst.selected() and lst.selected()["id"] == made["basic"]["id"]:
            break
    assert lst.selected()["id"] == made["basic"]["id"]
    assert "basic clean-up only" in lst.accessibleDescription()
    ui.qtbot.keyClick(lst, Qt.Key.Key_Down if index + 1 < len(lst.items) else Qt.Key.Key_Up)
    assert "basic clean-up only" not in lst.accessibleDescription()


def test_tag_hover_does_not_break_the_row_actions(make_ui):
    made = {}
    ui = make_ui(prepare=lambda rig: made.update(zip(("plain", "basic", "command"), add_entries(rig))))
    lst = ui.page("history").list
    index = next(i for i, it in enumerate(lst.items) if it["id"] == made["basic"]["id"])
    copy_rect = lst._action_rects(lst._card_rect(index))[0]
    ui.qtbot.mouseMove(lst, copy_rect.center())
    QApplication.processEvents()
    assert lst.toolTip() and lst.toolTip() != HistoryList.RULES_ONLY_TIP
    ui.qtbot.mouseClick(lst, Qt.MouseButton.LeftButton, pos=copy_rect.center())
    assert ui.rig.injector.copied == [made["basic"]["text"]]


def test_tag_gives_way_on_a_narrow_window_instead_of_overlapping(make_ui):
    made = {}
    ui = make_ui(prepare=lambda rig: made.update(zip(("plain", "basic", "command"), add_entries(rig))))
    lst = ui.page("history").list
    ui.win.resize(ui.win.minimumWidth(), ui.win.height())
    QApplication.processEvents()
    tags = tag_rects(ui, lst)
    index = next(i for i, it in enumerate(lst.items) if it["id"] == made["basic"]["id"])
    card = lst._card_rect(index)
    for rect in tags.values():
        assert card.contains(rect)
        assert not any(rect.intersects(a) for a in lst._action_rects(card)), "the tag never covers the row buttons"


def test_a_real_dictation_cleaned_by_rules_gets_flag_hint_and_tag(ui):
    """The whole chain on the real Controller: cleaner reports rules-only -> entry flag -> bar message -> tag."""
    ui.rig.cleaner.last_info = {"rules_only": True}
    seen = []
    ui.c.state_changed.connect(lambda s, m, msg: seen.append((s, msg)))
    ui.rig.dictate("the model was too slow for this one")
    assert ("done", "Pasted" + ctl_mod.RULES_ONLY_SUFFIX) in seen
    entry = ui.rig.last_entry()
    assert entry.get("rules_only") is True and ui.c.rules_only_count == 1
    assert ui.shell.overlay.text == "Pasted · basic clean-up"
    ui.rig.cleaner.last_info = {"rules_only": False}
    ui.wait(lambda: ui.c.state == "idle")
    ui.rig.dictate("and this one was fine")
    assert seen[-1] == ("done", "Pasted") or ("done", "Pasted") in seen[-3:]
    lst = ui.page("history").list
    ui.wait(lambda: len(lst.items) == 2)
    assert set(tag_rects(ui, lst)) == {entry["id"]}


def test_command_mode_never_gets_the_flag_or_the_hint(ui):
    ui.rig.cleaner.last_info = {"rules_only": True}
    ui.rig.injector.selection = "hey bob send the report"
    seen = []
    ui.c.state_changed.connect(lambda s, m, msg: seen.append((s, m, msg)))
    ui.rig.dictate("make this more formal", binding="command")
    done = [x for x in seen if x[0] == "done"]
    assert done and ctl_mod.RULES_ONLY_SUFFIX not in done[-1][2]
    assert not ui.rig.last_entry().get("rules_only")
    lst = ui.page("history").list
    assert tag_rects(ui, lst) == {}


# =========================================================================== 5. Flow bar secondary hint
SUFFIX = ctl_mod.RULES_ONLY_SUFFIX
HINTED = ["Pasted" + SUFFIX, "Typed" + SUFFIX, "Copied. Press Ctrl+V" + SUFFIX]


def settle(overlay, n: int = 4) -> None:
    """Let the bar reach its final size at once (animations snap when motion is reduced; make_ui resets it)."""
    T.set_reduced_motion(True)
    for _ in range(max(2, min(n, 6))):
        overlay._tick()
    QApplication.processEvents()


class Recorder:
    """Stands in for the painter: records what ``_paint_text`` draws, with which pen, where."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, float]] = []
        self._pen = None

    def setFont(self, font):
        self.font = font

    def setPen(self, pen):
        self._pen = pen

    def drawText(self, area, flags, text):
        c = self._pen
        self.calls.append((text, (c.red(), c.green(), c.blue()), area.left()))


def drawn(overlay, text: str):
    rec = Recorder()
    overlay._paint_text(rec, overlay.pill_rect(), 1.0, text)
    return rec.calls


def test_the_separator_is_the_one_the_controller_uses():
    assert SUFFIX.startswith(overlay_mod.HINT_SEP) and SUFFIX == " · basic clean-up"
    assert max(len(m) for m in HINTED) <= 37


@pytest.mark.parametrize("message", HINTED)
def test_hint_is_painted_dimmer_after_the_main_outcome(ui, message):
    overlay = ui.shell.overlay
    overlay.set_state("done", message, "dictate")
    settle(overlay)
    assert overlay._display_text() == message
    main, hint = message.split(overlay_mod.HINT_SEP)
    calls = drawn(overlay, overlay._display_text())
    assert [c[0] for c in calls] == [main, overlay_mod.HINT_SEP + hint]
    assert calls[0][1] != calls[1][1], "the hint uses a dimmer colour than the outcome"
    bright = lambda rgb: sum(rgb)  # noqa: E731
    assert bright(calls[1][1]) < bright(calls[0][1])
    fm = QFontMetricsF(overlay._font)
    assert calls[1][2] == pytest.approx(calls[0][2] + fm.horizontalAdvance(main)), "the hint starts where the outcome ends"
    ui.render(overlay)


@pytest.mark.parametrize("message", HINTED)
def test_hinted_messages_fit_the_pill_without_clipping(ui, message):
    overlay = ui.shell.overlay
    overlay.set_state("done", message, "dictate")
    settle(overlay, 80)
    fm = QFontMetricsF(overlay._font)
    need = fm.horizontalAdvance(message)
    assert need <= overlay_mod.MAX_TEXT_W, f"{message!r} needs {need:.0f} px, the bar allows {overlay_mod.MAX_TEXT_W}"
    rect = overlay.pill_rect()
    assert rect.width() - 44 - 20 + 2 >= need - 0.5, "the pill grew wide enough for the whole message"
    calls = drawn(overlay, message)
    assert "".join(c[0] for c in calls) == message and "…" not in "".join(c[0] for c in calls)
    assert overlay.rect().contains(rect.toAlignedRect()), "the pill stays inside its window"


@pytest.mark.parametrize("state, message", [("done", "Pasted"), ("done", "Typed"), ("done", "Copied. Press Ctrl+V"),
                                            ("done", "Rewritten"), ("arrange", "Drag me where you want me")])
def test_messages_without_a_hint_are_painted_as_before(ui, state, message):
    overlay = ui.shell.overlay
    overlay.set_state(state, message, "dictate")
    settle(overlay)
    calls = drawn(overlay, message)
    assert [c[0] for c in calls] == [message], "one piece, one colour"
    overlay.set_state("idle")


def test_the_bar_shrinks_back_for_the_next_plain_message(ui):
    overlay = ui.shell.overlay
    overlay.set_state("done", HINTED[2], "dictate")
    settle(overlay, 80)
    wide = overlay.pill_rect().width()
    overlay.set_state("done", "Pasted", "dictate")
    settle(overlay, 80)
    assert overlay.pill_rect().width() < wide - 100


@pytest.mark.parametrize("message", ["Couldn't paste", "Didn't catch that", "Microphone unavailable",
                                     "Speech model still loading…"])
def test_error_states_never_carry_the_hint(ui, message):
    overlay = ui.shell.overlay
    overlay.set_state("error", message, "dictate")
    settle(overlay)
    calls = drawn(overlay, overlay._display_text())
    assert [c[0] for c in calls] == [message]


def test_a_failed_paste_of_a_rules_only_dictation_has_no_hint_but_keeps_the_flag(ui):
    ui.rig.cleaner.last_info = {"rules_only": True}
    ui.rig.injector.outcome = "failed"
    seen = []
    ui.c.state_changed.connect(lambda s, m, msg: seen.append((s, msg)))
    ui.rig.dictate("this one cannot be pasted")
    assert not any(SUFFIX in msg for _s, msg in seen), seen
    assert ui.rig.last_entry().get("rules_only") is True


def test_a_text_that_only_looks_like_a_hint_is_elided_as_a_whole_when_too_long(ui):
    overlay = ui.shell.overlay
    long = "Window changed. Result copied: press Ctrl+V" + SUFFIX + " and a lot more than fits"
    overlay.set_state("done", long, "dictate")
    settle(overlay, 80)
    calls = drawn(overlay, long)
    painted = "".join(c[0] for c in calls)
    fm = QFontMetricsF(overlay._font)
    assert painted.endswith("…") and fm.horizontalAdvance(painted) <= overlay.pill_rect().width() - 44 - 20 + 4
    assert calls[0][0] == "Window changed. Result copied: press Ctrl+V", "the main outcome is never cut first"
    overlay.set_state("idle")


def test_flow_bar_with_hint_renders_in_light_and_dark(ui):
    overlay = ui.shell.overlay
    overlay.set_state("done", HINTED[0], "dictate")
    settle(overlay)
    shots = []
    for theme in ("dark", "light"):
        ui.c.set("theme", theme)
        settle(overlay)
        image = ui.render(overlay).toImage()
        rect = overlay.pill_rect()
        opaque = sum(1 for x in range(int(rect.left()), int(rect.right()), 3)
                     for y in range(int(rect.top()), int(rect.bottom()), 2) if (image.pixel(x, y) >> 24) & 0xFF > 40)
        assert opaque > 10, f"the pill did not paint in the {theme} theme"
        shots.append(opaque)
