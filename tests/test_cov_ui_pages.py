"""``voxflow.ui.pages``: the six pages on the in-memory demo controller: what the user sees when a store or the
controller fails (an error state or a toast, never a crash), demo-mode pasting, pack and backup dialogs fed with
unusual files, the Style preview in a language that keeps its capitals, and hotkey recording through Qt keys.
File dialogs are replaced; every file lives in ``tmp_path``."""
from __future__ import annotations

import datetime as dt
import json

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import demo, flush, make_demo  # noqa: F401 - fixtures
from voxflow.ui import dialogs
from voxflow.ui import pages as P

LEFT, NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton


def boom(message="the file is locked", kind=RuntimeError):
    def raiser(*_a, **_k):
        raise kind(message)
    return raiser


def open_from(monkeypatch, path) -> None:
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: str(path) if path else "")


def save_to(monkeypatch, path) -> list:
    asked: list = []
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: asked.append(a[1:]) or (str(path) if path else ""))
    return asked


def clear_toasts(demo) -> None:
    for toast in list(demo.win.toasts._toasts):
        toast.dismiss()
    flush()
    assert demo.toasts() == []


def mouse(widget, kind, pos, button=NONE, buttons=NONE) -> None:
    pos = QPointF(pos)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, QPointF(widget.mapToGlobal(pos.toPoint())), button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


# =========================================================================== helpers
def test_ago_and_plural_wording():
    now = dt.datetime(2025, 1, 20, 15, 0).timestamp()         # winter: no clock change within the spans below
    at = lambda **delta: now - dt.timedelta(**delta).total_seconds()   # noqa: E731
    assert P.ago(now, now) == "just now" and P.ago(at(seconds=89), now) == "just now"
    assert P.ago(now + 500, now) == "just now", "a clock that is slightly ahead is not 'in the future'"
    assert P.ago(at(minutes=4), now) == "4 min ago" and P.ago(at(minutes=59), now) == "59 min ago"
    assert P.ago(at(hours=2), now) == "2 h ago" and P.ago(at(hours=14, minutes=30), now) == "14 h ago"
    assert P.ago(at(hours=20), now) == "yesterday at 19:00"
    assert P.ago(at(days=9), now) == "11 January at 15:00"
    assert P.ago(at(days=400), now) == "17 December at 15:00, 2023"
    assert P.ago("never", now) == "at an unknown time" and P.ago(None) == "at an unknown time"
    assert P.plural(1, "word") == "1 word" and P.plural(2, "word") == "2 words"
    assert P.plural(1500, "damaged entry", "damaged entries") == "1,500 damaged entries"
    assert P._split_aliases("cube are netties; kubernetties\n k8s ,, ") == ["cube are netties", "kubernetties", "k8s"]
    assert [P.greeting(dt.datetime(2025, 1, 1, h)) for h in (4, 5, 11, 12, 17, 18)] == \
        ["Good evening", "Good morning", "Good morning", "Good afternoon", "Good afternoon", "Good evening"]


def test_a_page_turns_any_failure_into_a_toast(demo):
    page = demo.page("home")
    assert page.run(lambda: 7) == 7 and page.run(lambda: None) is True and demo.toasts() == []
    assert page.run(lambda: None, success="Done.") is True and demo.toasts() == ["Done."]
    assert page.run(boom("That name is taken.", ValueError)) is None
    assert demo.toasts()[-1] == "That name is taken."
    assert page.run(boom("disk is full", OSError)) is None
    assert demo.toasts()[-1] == "That did not work: disk is full"
    assert page.setting("theme", "pink") is False and demo.toasts()[-1] == "Unknown theme."
    assert page.setting("theme", "light") is True
    with pytest.raises(NotImplementedError):
        P._ImportExport()._store()


# =========================================================================== home
def test_home_survives_failing_stats_and_history(demo):
    page, c = demo.page("home"), demo.c
    assert page.stats_box.isVisibleTo(page) and page.recent.isVisibleTo(page) and len(page.chart.series) == 14
    c.stats.daily = boom()
    page.range.changed.emit(page.range.value)
    assert page.chart.series == [], "the chart shows its empty state"
    c.stats.summary = boom()
    c.stats.top_apps = boom()
    page.refresh()
    assert page.s_words.value == "0" and page.s_words.caption == "0 dictations" and page.s_streak.caption == "days in a row"
    assert page.hero.isVisibleTo(page) and not page.stats_box.isVisibleTo(page), "nothing readable: the first-run look"
    assert not page.recent.isVisibleTo(page) and page.empty.isVisibleTo(page)
    assert demo.render()


def test_home_first_run_and_history_off(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("home"), demo.c
    assert page.hero.isVisibleTo(page) and page.empty.title.text() and not page.recent_title.isVisibleTo(page)
    c.set("hotkey", "ctrl+shift+d")
    assert page.caps.combo == "ctrl+shift+d" and "Ctrl + Shift + D" in page.empty.sub.text()
    c.set("save_history", False)
    assert page.empty.title.text() == "History is off" and page.recent_title.isVisibleTo(page)
    page.empty.button.click()
    assert c.get("save_history") is True
    c.demo_add_transcript("One two three.")
    assert not page.hero.isVisibleTo(page) and page.recent.isVisibleTo(page) and page.s_streak.caption == "days in a row"
    c.stats._daily.append({"date": dt.date.today().isoformat(), "words": 3, "sessions": 1, "seconds": 2.0})
    c.stats_changed.emit()
    page.refresh()
    assert page.s_streak.value == "1" and page.s_streak.caption == "day in a row" and page.s_words.caption == "3 today"


# =========================================================================== history
def test_paste_again_in_demo_mode_explains_instead_of_pasting(demo):
    page, c = demo.page("history"), demo.c
    entry = page.list.items[0]
    page.list.paste_requested.emit(entry)
    assert demo.win.isVisible(), "demo mode has nothing to paste into: the window stays"
    assert demo.toasts() == ["Demo mode: this would paste into the app you were using."]
    page.list.paste_requested.emit({"id": "gone"})
    assert demo.toasts() == ["Demo mode: this would paste into the app you were using.", "There is nothing to paste yet."]
    demo.page("home").recent.paste_requested.emit(entry)
    assert len(demo.toasts()) == 2, "the same message is not stacked"


def test_history_delete_copy_export_menu_and_load_failure(demo, monkeypatch, tmp_path):
    page, c = demo.page("history"), demo.c
    total = c.history.count()
    copied: list = []
    c.copy_text = lambda text: copied.append(text) or True
    entry = page.list.items[1]
    page.list.copy_requested.emit(entry)
    page.view.set_value(True, emit=True)                       # Raw: copy takes what is shown
    page.list.copy_requested.emit(entry)
    assert copied == [entry["text"], entry["raw"]]
    page.list.delete_requested.emit(entry)
    assert c.history.count() == total - 1 and demo.toasts() == ["Transcript deleted."]
    page.list.delete_requested.emit(entry)                     # already gone: nothing to announce
    assert demo.toasts() == ["Transcript deleted."]
    page.list._tick.stop()

    first = page._export_menu()
    assert [it["text"] for it in first.items()][1:] == ["Plain text (.txt)", "Spreadsheet (.csv)", "JSON (.json)"]
    first.close()
    assert page._export_menu() is first, "one menu object, reused"
    out = tmp_path / "history.json"
    asked = save_to(monkeypatch, out)
    first.items()[3]["on_click"]()
    assert len(json.loads(out.read_text(encoding="utf-8"))) == total - 1 and asked[0][2] == "JSON files (*.json)"
    assert demo.toasts()[-1] == f"Exported {total - 1} transcripts to history.json."
    first.close()

    c.history.count = boom("history.db is locked")
    page.refresh()
    assert page.empty.isVisibleTo(page) and page.empty.kind == "error"
    assert page.empty.title.text() == "Your history could not be loaded"
    assert page.empty.sub.text() == "history.db is locked. Nothing was changed on disk."
    assert not page.btn_export.isEnabled() and not page.btn_clear.isEnabled()
    del c.history.count
    page.empty.button.click()                                  # Try again
    assert page.list.isVisibleTo(page) and page.count.text() == f"{total - 1} transcripts"


def test_history_search_paging_and_states(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("history"), demo.c
    assert page.empty.title.text() == "No transcripts yet"
    c.set("save_history", False)
    assert page.empty.title.text() == "History is off"
    page.empty.button.click()
    assert c.get("save_history") is True
    for i in range(5):
        c.demo_add_transcript(f"Entry number {i}.")
    page.PAGE_SIZE = page._limit = 2
    page.refresh()
    assert len(page.list.items) == 2 and page.btn_more.isVisibleTo(page) and page.count.text() == "5 transcripts"
    page.btn_more.click()
    page.btn_more.click()
    assert len(page.list.items) == 5 and not page.btn_more.isVisibleTo(page)
    page.search.setText("number")
    page._debounce.stop()
    page._search_changed()
    assert page.count.text() == "2+ matches for “number”" and page._limit == 2
    page.search.setText("number 3")
    page._debounce.stop()
    page._search_changed()
    assert page.count.text() == "1 match for “number 3”"
    page.search.setText("zebra")
    page._debounce.stop()
    page._search_changed()
    assert page.empty.title.text() == "No matches" and "zebra" in page.empty.sub.text()
    page.empty.button.click()                                  # Clear search
    page._debounce.stop()
    page._search_changed()
    assert page.search.text() == "" and len(page.list.items) == 2
    page.btn_clear.click()
    confirm = demo.find("history_clear_confirm")
    assert "5 transcripts" in confirm.findChildren(P.Label)[1].text()
    confirm.btn_primary.click()
    assert c.history.count() == 0 and demo.toasts()[-1] == "History cleared."


# =========================================================================== dictionary + snippets
def test_dictionary_load_failure_and_a_failed_removal(demo):
    page, c = demo.page("dictionary"), demo.c
    entry = c.dictionary.list()[0]
    real_list = c.dictionary.list
    c.dictionary.list = boom("dictionary.json is locked")
    page.refresh()
    assert page.empty.isVisibleTo(page) and page.empty.kind == "error"
    assert page.empty.title.text() == "Your dictionary could not be loaded"
    assert page.empty.sub.text() == "dictionary.json is locked. Nothing was changed on disk."
    assert page.inbox.caption.text() == "SUGGESTIONS  ·  0" and not page.btn_export.isEnabled()
    c.dictionary.list = boom("", OSError)
    page.empty.button.click()                                  # Try again: still failing, named by its type
    assert page.empty.sub.text() == "OSError. Nothing was changed on disk."
    c.dictionary.list = real_list
    page.empty.button.click()
    assert page.entries.isVisibleTo(page) and not page.empty.isVisibleTo(page)

    c.dictionary.remove = boom("That entry is read-only.", ValueError)
    page._remove(entry)
    assert demo.toasts() == ["That entry is read-only."], "no 'Removed' and no Undo for something still there"
    del c.dictionary.remove
    page._remove(entry)
    assert demo.toasts()[-1] == f"Removed “{entry['term']}”." and entry["term"] not in c.dictionary.terms()
    demo.win.toasts._toasts[-1].btn.click()                    # Undo
    flush()
    assert entry["term"] in c.dictionary.terms()


def test_dictionary_search_and_empty_states(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("dictionary"), demo.c
    assert page.empty.title.text() == "Your dictionary is empty" and not page.search.isVisibleTo(page)
    page.empty.button.click()                                  # "Add your first word"
    editor = demo.find("dict_editor")
    demo.find("dict_term", editor).setText("Kubernetes")
    demo.find("dict_aliases", editor).setText("cube are netties; kubernetties")
    editor.btn_primary.click()
    assert c.dictionary.list()[0]["aliases"] == ["cube are netties", "kubernetties"]
    for i in range(7):
        c.dictionary.add(f"Term {i}")
    assert page.search.isVisibleTo(page) and page.entries.caption.text() == "YOUR WORDS  ·  8"
    page.search.setText("zebra")
    page._debounce.stop()
    page._search_changed()
    assert page.empty.title.text() == "No matches" and "zebra" in page.empty.sub.text()
    page.empty.button.click()
    page._debounce.stop()
    page._search_changed()
    assert page.search.text() == "" and page.entries.isVisibleTo(page)


def test_snippets_load_failure_and_a_failed_removal(demo):
    page, c = demo.page("snippets"), demo.c
    entry = c.snippets.list()[0]
    real_list = c.snippets.list
    c.snippets.list = boom("snippets.json is locked")
    page.refresh()
    assert page.empty.kind == "error" and page.empty.title.text() == "Your snippets could not be loaded"
    assert page.empty.sub.text() == "snippets.json is locked. Nothing was changed on disk."
    assert not page.rows.isVisibleTo(page) and not page.btn_share.isEnabled()
    c.snippets.list = real_list
    page.empty.button.click()
    assert page.rows.isVisibleTo(page) and page.rows.caption.text() == f"YOUR SNIPPETS  ·  {len(real_list())}"

    c.snippets.remove = boom("That snippet is read-only.", ValueError)
    page._remove(entry)
    assert demo.toasts() == ["That snippet is read-only."]
    del c.snippets.remove
    page._remove(entry)
    assert demo.toasts()[-1] == f"Deleted the snippet “{entry['trigger']}”."
    demo.win.toasts._toasts[-1].btn.click()                    # Undo
    flush()
    assert entry["trigger"] in [e["trigger"] for e in c.snippets.list()]


# =========================================================================== import / export / packs
def test_plain_import_and_export_report_what_happened(demo, monkeypatch, tmp_path):
    page, c = demo.page("snippets"), demo.c
    out = tmp_path / "snippets.json"
    save_to(monkeypatch, out)
    page.do_export()
    n = len(c.snippets.list())
    assert demo.toasts() == [f"Exported {n} snippets to snippets.json."]
    open_from(monkeypatch, out)
    page.do_import()
    assert demo.toasts()[-1] == "No new snippets found in snippets.json."
    c.snippets.remove(c.snippets.list()[0]["id"])
    page.do_import()
    assert demo.toasts()[-1] == "Imported 1 new snippets."
    clear_toasts(demo)
    c.snippets.import_ = lambda path: None                     # a store that reports nothing counts as "no news"
    page.do_import()
    assert demo.toasts() == ["No new snippets found in snippets.json."]
    c.snippets.export = boom("That folder is read-only.", ValueError)
    page.do_export()
    assert demo.toasts()[-1] == "That folder is read-only."
    before = list(demo.toasts())
    open_from(monkeypatch, "")
    save_to(monkeypatch, "")
    page.do_import()
    page.do_export()
    assert demo.toasts() == before, "cancelled dialogs do nothing"
    del c.snippets.import_
    open_from(monkeypatch, tmp_path / "missing.json")
    page.do_import()
    assert demo.toasts()[-1] == "Could not read snippets from missing.json."


def test_importing_a_pack_shows_what_is_in_it(demo, monkeypatch, tmp_path):
    page, c = demo.page("dictionary"), demo.c
    path = tmp_path / "team.voxpack"
    info = {"kind": "dictionary", "name": "", "description": "", "count": 3, "created": 0, "skipped": 2}
    c.inspect_pack = lambda p: dict(info)
    open_from(monkeypatch, path)
    page.do_import()
    dialog = demo.find("pack_import")
    facts = [lab.text() for lab in dialog.findChildren(P.Card)[0].findChildren(P.Label)]
    assert facts == ["team", "3 words", "2 damaged entries in the file will be left out"], \
        "no name: the file name; no description, date or new/existing split: left out"

    c.import_pack = boom("team.voxpack was changed while it was open.", ValueError)
    dialog.btn_primary.click()
    assert dialog.isVisible() and dialog.error.text() == "team.voxpack was changed while it was open."
    c.import_pack = lambda p, mode: {"added": 2, "updated": 1, "removed": 0, "skipped": 1}
    dialog.mode.set_value("replace", emit=True)
    assert dialog.btn_primary.text() == "Replace" and dialog.btn_primary.kind == "danger" and dialog.error.text() == ""
    dialog.btn_primary.click()
    assert not dialog.isVisible()
    assert demo.toasts() == ["Pack imported: 2 added, 1 updated.",
                             "1 damaged entry in the pack could not be read and was left out."]

    info.update(name="Team words", description="For the platform team", created=dt.datetime.now().timestamp() - 7200,
                new=1, existing=2, skipped=0)
    c.import_pack = lambda p, mode: {"added": 0, "updated": 0, "removed": 3}
    dialog = page.import_pack(str(path))
    facts = [lab.text() for lab in dialog.findChildren(P.Card)[0].findChildren(P.Label)]
    assert facts[:3] == ["Team words", "For the platform team", "3 words: 1 new, 2 already in your list"]
    assert facts[3].startswith("Made ")
    dialog.btn_primary.click()
    assert demo.toasts()[-1] == "Pack imported: 0 added, 3 removed."
    c.import_pack = lambda p, mode: {"added": 0}
    page.import_pack(str(path)).btn_primary.click()
    assert demo.toasts()[-1] == "Nothing to import: you already have every word in this pack."

    info["kind"] = "snippets"                                  # the wrong page for this pack
    assert page.import_pack(str(path)) is None
    assert demo.toasts()[-1] == "team.voxpack is a snippets pack. Import it on the Snippets page."
    demo.win.toasts._toasts[-1].btn.click()
    flush()
    assert demo.win.current == "snippets"
    info["kind"] = ""
    assert demo.page("snippets").import_pack(str(path)) is None
    assert demo.toasts()[-1] == "team.voxpack is a different pack. Import it on the Dictionary page."
    c.inspect_pack = boom("team.voxpack is not a VoxFlow pack.", ValueError)
    assert page.import_pack(str(path)) is None and demo.toasts()[-1] == "team.voxpack is not a VoxFlow pack."


def test_sharing_a_pack(demo, monkeypatch, tmp_path):
    page, c = demo.page("snippets"), demo.c
    dialog = page.share_pack()
    count = demo.find("pack_selected_count", dialog)
    n = len(c.snippets.list())
    assert len(dialog.entries) == n and count.text() == f"IN THE PACK  ·  {n} OF {n}"
    dialog.btn_primary.click()
    assert dialog.error.text().startswith("Give the pack a name")
    name = demo.find("pack_name", dialog)
    name.setText("Team  Snippets")
    assert dialog.error.text() == "", "typing clears the complaint"
    demo.find("pack_select_none", dialog).click()
    assert count.text() == f"IN THE PACK  ·  0 OF {n}"
    dialog.btn_primary.click()
    assert dialog.error.text() == "Tick at least one snippet to put in the pack."
    demo.find("pack_select_all", dialog).click()
    dialog.entries[0][1].setChecked(False)
    asked = save_to(monkeypatch, "")
    dialog.btn_primary.click()
    assert dialog.isVisible() and asked[0][1] == "team-snippets.voxpack", "cancelling the file dialog keeps the form"
    out = tmp_path / "team-snippets.voxpack"
    save_to(monkeypatch, out)
    c.export_pack = boom("That folder is read-only.", ValueError)
    dialog.btn_primary.click()
    assert dialog.isVisible() and dialog.error.text() == "That folder is read-only."
    del c.export_pack
    dialog.btn_primary.click()
    assert not dialog.isVisible() and demo.toasts() == [f"Pack saved: {n - 1} snippets in team-snippets.voxpack."]
    assert len(json.loads(out.read_text(encoding="utf-8"))["items"]) == n - 1

    c.snippets.list = boom()                                   # the list cannot be read: the form still opens
    bare = page.share_pack()
    assert bare.entries == [] and demo.find_all("pack_entries", bare) == []
    bare.close_modal()
    flush()


# =========================================================================== merge / replace dialog
def test_merge_or_replace_dialog_with_parts(demo):
    calls: list = []
    results = iter(["The backup is from a newer VoxFlow.", True, True])

    def on_import(mode, keys):
        calls.append((mode, keys))
        return next(results)

    parts = [{"key": "settings", "label": "Settings", "available": True, "merge": False},
             {"key": "dictionary", "label": "Dictionary", "note": "3 words", "available": True},
             {"key": "history", "label": "History", "available": False}]
    dialog = P.merge_or_replace(demo.win, "Restore this backup?", ["backup.voxbackup", "3 words"], "Adds what is missing.",
                                "Replaces the ticked parts.", on_import, name="restore", parts=parts)
    box = dialog.parts
    assert not box["settings"].isEnabled() and not box["settings"].isChecked(), "a merge never changes settings"
    assert not box["history"].isEnabled() and box["history"].toolTip() == "Not in this backup"
    dialog.mode.changed.emit("merge")                          # choosing Merge again changes nothing
    assert not box["settings"].isChecked() and box["dictionary"].isChecked()
    box["dictionary"].setChecked(False)
    dialog.btn_primary.click()
    assert calls == [] and dialog.error.text() == "Tick at least one part to import."
    box["dictionary"].setChecked(True)
    assert dialog.error.text() == ""
    dialog.btn_primary.click()
    assert calls == [("merge", None)] and dialog.error.text() == "The backup is from a newer VoxFlow."

    dialog.mode.set_value("replace", emit=True)
    explain = demo.find("restore_explain", dialog)
    assert explain.text() == "Replaces the ticked parts." and box["settings"].isEnabled() and box["settings"].isChecked()
    box["settings"].setChecked(False)
    dialog.mode.set_value("merge", emit=True)
    dialog.mode.set_value("replace", emit=True)
    assert not box["settings"].isChecked(), "the user's own tick is remembered across Merge"
    dialog.btn_primary.click()
    assert calls[-1] == ("replace", ["dictionary"]) and not dialog.isVisible()
    flush()

    simple = P.merge_or_replace(demo.win, "Import this pack?", ["Team"], "Adds.", "Replaces.",
                                lambda mode: calls.append(mode) or True, name="simple")
    assert simple.parts == {}
    simple.btn_primary.click()
    assert calls[-1] == "merge" and not simple.isVisible()
    flush()


# =========================================================================== style
def test_style_preview_follows_the_real_tone_pass_and_the_language(demo):
    page, c = demo.page("style"), demo.c
    card = page.cards["work"]
    card.tone.set_value("very_casual", emit=True)
    assert c.styles.tones()["work"] == "very_casual"
    english = "I don't think we can ship this on Friday. let's sync tomorrow morning and decide"
    assert card.preview.text() == english and card.hint.text() == P._TONE_HINTS["very_casual"]
    c.set("language", "de")                                    # German keeps its capitals: only the full stop goes
    sample = P._SAMPLES_BY_LANGUAGE["de"]["work"]
    assert card.preview.text() == sample[:-1]
    assert card.hint.text() == "Same words. No full stop at the very end; this language keeps its capitals."
    assert page.cards["email"].preview.text() == P._SAMPLES_BY_LANGUAGE["de"]["email"], "formal changes nothing"
    c.set("language", "pl")                                    # no samples of its own: the English ones
    assert card.preview.text() == english

    c.preview_tone = boom("tone rules missing")
    page.refresh()
    assert card.preview.text() == P._SAMPLES["work"], "the untouched sample rather than an empty box"
    del c.preview_tone
    c.styles.set_tone = boom("Unknown style.", ValueError)
    card.tone.set_value("formal", emit=True)
    assert demo.toasts() == ["Unknown style."] and card.tone.value == "very_casual", "the control snaps back"
    assert demo.render()


def test_style_apps_search_add_and_load_failure(demo):
    page, c = demo.page("style"), demo.c

    def app_rows():
        flush()
        return [r.title.text() for r in page.apps._rows][1:]

    everything = app_rows()
    assert "Slack" in everything and len(everything) > 5
    page.app_search.setText("sla")
    assert app_rows() == ["Slack"]
    page.app_search.setText("vs code")                         # found by its friendly name too
    assert app_rows() == ["VS Code"]
    page.app_search.setText("")

    demo.find("style_add_app").click()
    assert demo.toasts() == ["Type the app's file name first, for example signal.exe."]
    page.new_exe.setText("/")
    demo.find("style_add_app").click()
    assert demo.toasts()[-1] == "Enter an app name such as slack.exe." and len(demo.toasts()) == 2
    page.new_exe.setText("Signal")
    page.new_cat.set_value("personal")
    demo.find("style_add_app").click()
    assert demo.toasts()[-1] == "Signal added." and c.styles.category_for("signal.exe") == "personal"
    assert app_rows()[0] == "Signal", "your own choices come first"
    resets = demo.find_all("style_app_reset")
    assert len(resets) == 1 and resets[0].toolTip()
    demo.find_all("style_app_category")[0].chosen.emit("work")
    assert c.styles.category_for("signal.exe") == "work"
    demo.find("style_app_reset").click()
    assert "Signal" not in app_rows()

    c.styles.app_categories = boom("styles.json is locked")
    page.refresh()
    assert demo.toasts()[-1] == "Could not read your styles: styles.json is locked"
    assert app_rows() == [] and all(card.tone.value == "formal" for card in page.cards.values())
    page.resize(600, page.height())
    QApplication.processEvents()
    assert page._cols == 1, "the tone cards stack on a narrow window"


# =========================================================================== settings
def test_settings_finder_and_sections(demo):
    page = demo.page("settings")
    sections = page._sections()
    page.scroll.verticalScrollBar().setValue(0)
    page.show_section("no-such-section")
    QApplication.processEvents()
    assert page.scroll.verticalScrollBar().value() == 0
    page.find.setText("zzz")
    assert page.no_match.isVisibleTo(page)
    page.show_section("models")                                # an alias; also clears the search
    QApplication.processEvents()
    assert page.find.text() == "" and not page.no_match.isVisibleTo(page)
    target = next(s for s in sections if s.caption.text().lower() == "on-device models")
    bar = page.scroll.verticalScrollBar()
    at_models = min(bar.maximum(), max(0, target.y() - 4))
    assert bar.value() == at_models > 0
    page._jump_to(99)
    page._jump_to(-1)
    assert bar.value() == at_models, "an unknown section moves nothing"
    page.jump.chosen.emit(1)
    assert bar.value() == max(0, sections[1].y() - 4) < at_models and page.jump.value is None


def test_settings_language_hint_volume_and_microphones(demo):
    page, c = demo.page("settings"), demo.c
    hint = page.language_row.sub
    assert hint.text() == "Picks the rules from what you say. Full clean-up in 7 languages, basic in the other 18."
    c.set("language", "de")
    assert hint.text() == "German: full clean-up, including spoken punctuation, self-corrections and numbers."
    c.set("language", "pl")
    assert hint.text().startswith("Polish: basic clean-up. Spoken commands like “new line” work in English, ")
    assert hint.text().endswith("and Dutch.")
    real = c.language_support
    c.language_support = lambda code: real(code) if code != "xx" else boom("unknown language")()
    page._sync_language("xx")
    assert hint.text().startswith("This language: basic clean-up.")

    page.volume.setSliderDown(True)                            # the user is dragging: a stored value must not fight it
    c.set("sound_volume", 0.9)
    assert page.volume.value() == 50 and page.vol_label.text() == "50%"
    page.volume.setSliderDown(False)
    page._vol_timer.stop()
    c.set("sound_volume", 0.8)
    assert page.volume.value() == 80 and page.vol_label.text() == "80%"
    page.volume.setValue(30)
    assert page.vol_label.text() == "30%" and page._vol_timer.isActive()
    page._vol_timer.stop()
    page._vol_timer.timeout.emit()
    assert c.get("sound_volume") == 0.3

    c.played.clear()
    mouse(page.volume, QEvent.Type.MouseButtonPress, QPoint(100, 14), LEFT, LEFT)
    mouse(page.volume, QEvent.Type.MouseButtonRelease, QPoint(100, 14), LEFT)
    assert c.played == ["start"], "letting go of the volume slider plays the preview tone once"
    page._vol_timer.stop()

    c.microphones_changed.emit([])
    assert page.mic.options == [(None, "System default")]
    c.config.mic_device = "Ghost microphone"
    c.microphones_changed.emit(c.microphones())
    assert page.mic.options[-1] == ("Ghost microphone", "Ghost microphone (not connected)") and len(page.mic.options) == 4


def test_settings_models_and_a_controller_that_cannot_report_them(demo):
    page, c = demo.page("settings"), demo.c
    asr = page.models["asr"]
    c.demo_set_model("asr", "downloading", "42 MB of 600 MB", 0.07)
    assert asr.state.text() == "Downloading 7%" and asr.bar.isVisibleTo(asr) and asr.bar.value == 0.07
    c.demo_set_model("asr", "failed", "Out of memory")
    assert asr.state.text() == "Failed" and asr.retry.isVisibleTo(asr) and asr.sub.text() == "Out of memory"
    asr.retry.click()
    assert asr.state.text() == "Ready" and not asr.retry.isVisibleTo(asr)
    c.model_status_changed.emit("no-such-model", {"state": "failed"})
    c.model_status_changed.emit("asr", None)
    assert asr.state.text() == "Waiting"
    c.model_status = boom("models not initialised")
    page.refresh()
    assert asr.state.text() == page.models["cleanup"].state.text() == "Waiting"


def test_settings_hotkeys_recorded_through_qt_keys(demo):
    page, c = demo.page("settings"), demo.c
    dictation, command = page.recorders["hotkey"], page.recorders["command_hotkey"]
    press = lambda rec, *keys: [demo.qtbot.keyPress(rec.btn, k) for k in keys]       # noqa: E731
    release = lambda rec, *keys: [demo.qtbot.keyRelease(rec.btn, k) for k in keys]   # noqa: E731
    dictation.btn.click()
    assert page._capturing is dictation and not c._capturing, "demo mode: no global hook to arm"
    press(dictation, Qt.Key.Key_Control, Qt.Key.Key_J)
    release(dictation, Qt.Key.Key_J, Qt.Key.Key_Control)
    assert c.get("hotkey") == "ctrl+j" and page._capturing is None and dictation.msg.text() == "Saved"

    dictation.btn.click()                                      # recorded combo is refused by the validator
    press(dictation, Qt.Key.Key_Alt, Qt.Key.Key_Meta)
    release(dictation, Qt.Key.Key_Meta, Qt.Key.Key_Alt)
    assert dictation.msg.text() == "Already used for Command mode." and c.get("hotkey") == "ctrl+j"
    assert page._capturing is dictation and not dictation.capturing
    command.btn.click()                                        # starting another recorder takes over
    assert page._capturing is command and command.capturing
    dictation.btn.click()
    assert page._capturing is dictation and not command.capturing, "one recorder at a time"
    dictation.btn.click()                                      # Cancel
    assert page._capturing is None

    c.set = boom("That combination is reserved by Windows.", ValueError)
    command.changed.emit("ctrl+shift+k")
    assert command.combo == "alt+win" and command.msg.text() == "That combination is reserved by Windows."
    del c.set
    command.btn_clear.click()
    assert c.get("command_hotkey") == "" and command.msg.text() == "Turned off"
    command.btn.click()
    demo.win.show_page("home")
    assert not command.capturing and page._capturing is None, "leaving the page ends a recording"
    for rec in (dictation, command):
        rec._msg_timer.stop()


def test_settings_restore_dialog_for_unusual_backups(demo, monkeypatch, tmp_path):
    page, c = demo.page("settings"), demo.c
    path = tmp_path / "old.voxbackup"
    open_from(monkeypatch, path)
    c.inspect_backup = lambda p: {"counts": {"dictionary": 1, "snippets": 0}, "includes": ["dictionary", "snippets"],
                                  "device": "", "same_device": False, "created": 0, "skipped": 3}
    dialog = page.import_backup()
    facts = [lab.text() for lab in dialog.findChildren(P.Card)[0].findChildren(P.Label)]
    assert facts == ["old.voxbackup", "1 word, 0 snippets, no history", "3 damaged entries in the file will be left out"]
    assert not dialog.parts["settings"].isEnabled() and not dialog.parts["history"].isEnabled()

    c.import_backup = boom("old.voxbackup was changed while it was open.", ValueError)
    dialog.btn_primary.click()
    assert dialog.isVisible() and dialog.error.text() == "old.voxbackup was changed while it was open."
    c.import_backup = lambda p, mode, keys: {"counts": {"dictionary": 1}, "settings_skipped": ["mic_device", "asr_device"],
                                             "skipped": 1}
    dialog.btn_primary.click()
    assert not dialog.isVisible()
    assert demo.toasts() == ["Backup merged: 1 word added.",
                             "2 settings from the backup could not be used on this PC and kept its current value.",
                             "1 damaged entry in the backup could not be read and was left out."]

    c.inspect_backup = lambda p: {"counts": {"dictionary": 1, "snippets": 2, "history": 4, "styles": 1},
                                  "includes": ["settings", "dictionary", "snippets", "styles", "stats", "history"],
                                  "device": "Laptop", "same_device": False, "version": "0.1.0",
                                  "created": dt.datetime.now().timestamp() - 30}
    seen: list = []
    c.import_backup = lambda p, mode, keys: seen.append((mode, keys)) or {"counts": {}}
    dialog = page.import_backup()
    facts = [lab.text() for lab in dialog.findChildren(P.Card)[0].findChildren(P.Label)]
    assert facts == ["old.voxbackup", "Made just now on Laptop, VoxFlow 0.1.0", "1 word, 2 snippets, 4 transcripts",
                     "Also contains your settings, styles and stats"]
    dialog.btn_primary.click()
    assert seen == [("merge", None)] and demo.toasts()[-1] == "Backup merged: everything in it was already here."
    dialog = page.import_backup()
    dialog.mode.set_value("replace", emit=True)
    dialog.parts["snippets"].setChecked(False)
    dialog.btn_primary.click()
    assert seen[-1] == ("replace", ["settings", "dictionary", "styles", "history", "stats"]), \
        "statistics travel with the history"
    assert demo.toasts()[-1] == "The parts you ticked were replaced with the backup's."
    dialog = page.import_backup()
    dialog.mode.set_value("replace", emit=True)
    dialog.btn_primary.click()
    assert seen[-1] == ("replace", None) and demo.toasts()[-1] == "Your profile was replaced with the backup."

    c.inspect_backup = boom("old.voxbackup is not a VoxFlow backup.", ValueError)
    assert page.import_backup() is None and demo.toasts()[-1] == "old.voxbackup is not a VoxFlow backup."
    open_from(monkeypatch, "")
    assert page.import_backup() is None


def test_settings_backup_export_dialog(demo, monkeypatch, tmp_path):
    page, c = demo.page("settings"), demo.c
    dialog = page.export_backup()
    asked = save_to(monkeypatch, "")
    dialog.btn_primary.click()
    assert dialog.isVisible() and asked[0][1].endswith(".voxbackup"), "cancelling the file dialog keeps the form"
    save_to(monkeypatch, tmp_path / "missing-folder" / "mine.voxbackup")
    dialog.btn_primary.click()
    assert dialog.isVisible() and dialog.error.text().startswith("mine.voxbackup could not be written")
    save_to(monkeypatch, tmp_path / "mine.voxbackup")
    demo.find("backup_export_history", dialog).setChecked(False)
    dialog.btn_primary.click()
    assert not dialog.isVisible()
    words, snippets = len(c.dictionary.list()), len(c.snippets.list())
    assert demo.toasts() == [f"Backup saved to mine.voxbackup: your settings, {words} words, {snippets} snippets."]
    assert "history" not in json.loads((tmp_path / "mine.voxbackup").read_text(encoding="utf-8"))
    dialog = page.export_backup()
    dialog.btn_primary.click()
    assert demo.toasts()[-1].endswith(f"{snippets} snippets, {c.history.count()} transcripts.")


def test_settings_sync_row(demo, monkeypatch, tmp_path):
    page, c = demo.page("settings"), demo.c
    status = page.sync_row.sub
    assert status.text().startswith("Off. Pick a folder") and not page.btn_sync_now.isVisibleTo(page)
    assert page.btn_sync_choose.text() == "Choose folder…"
    monkeypatch.setattr(dialogs, "ask_folder", lambda *a, **k: "")
    page.choose_sync_folder()
    assert c.get("sync_folder") == ""
    monkeypatch.setattr(dialogs, "ask_folder", lambda *a, **k: str(tmp_path))
    page.btn_sync_choose.click()
    assert c.get("sync_folder") == str(tmp_path) and demo.toasts() == ["Sync folder set. VoxFlow keeps it up to date from now on."]
    assert status.text().startswith("Up to date. No other device has used this folder yet. Last synced just now.")
    assert page.btn_sync_choose.text() == "Change…" and page.sync_folder.text() == str(tmp_path)
    c.demo_set_sync("idle", devices=["Laptop", "Studio", "Tablet", "Phone"])
    assert page.sync_folder.text() == f"{tmp_path}   ·   shared with Laptop, Studio, Tablet and 1 more"
    c.demo_set_sync("syncing")
    assert status.text() == "Syncing…" and not page.btn_sync_now.isEnabled()
    c.demo_set_sync("error", "The folder is offline.")
    assert status.text().startswith("The folder is offline. Last synced ") and page.btn_sync_now.isEnabled()
    page._show_sync({"state": "idle", "folder": str(tmp_path), "last_sync": None, "newer": 1})
    assert status.text() == "Up to date. Not synced yet."
    c.sync_status = boom("sync state is unreadable")
    page._show_sync()
    assert status.text() == "sync state is unreadable. Not synced yet."
    del c.sync_status
    page.btn_sync_now.click()
    assert status.text().startswith("Up to date with 4 other devices.")
    page.btn_sync_off.click()
    assert c.get("sync_folder") == "" and status.text().startswith("Off.")


def test_settings_mic_test_flow_bar_arranging_and_reset(demo):
    page, c = demo.page("settings"), demo.c
    page.btn_test.click()
    assert c._monitor and page.btn_test.text() == "Stop" and page.meter.source == c.level
    page.btn_test.click()
    assert not c._monitor and page.btn_test.text() == "Test" and page.meter.source is None
    page.btn_test.click()
    overlay = demo.shell.overlay
    page.btn_move.click()
    assert overlay.arranging and page.btn_move.text() == "Done" and "Drag the Flow bar" in page.pos_row.sub.text()
    overlay.position_changed.emit([500, 400])
    assert c.get("overlay_position") == [500, 400] and page.btn_pos_reset.isVisibleTo(page)
    demo.win.show_page("home")                                 # leaving the page ends the test and the arranging
    assert not c._monitor and not overlay.arranging and page.btn_move.text() == "Move…"
    demo.page("settings")
    assert page.pos_row.sub.text() == "Custom position, remembered between launches."
    page.btn_pos_reset.click()
    assert c.get("overlay_position") is None and not page.btn_pos_reset.isVisibleTo(page)
    page.btn_mic_refresh.click()
    c.set("language", "fr")
    page.btn_reset.click()
    demo.find("settings_reset_confirm").btn_primary.click()
    assert c.get("language") == "auto" and demo.toasts()[-1] == "Settings reset to defaults."
    seen: list = []
    demo.win.onboarding_requested.connect(lambda: seen.append(1))
    page.btn_onboarding.click()
    page.btn_data.click()
    page.btn_logs.click()
    QApplication.processEvents()
    assert seen == [1] and demo.shell.onboarding is not None
    assert demo.toasts()[-2:] == [f"Your data lives in {c.data_dir}.", f"Logs are written to {c.log_path.parent}."]
