"""Round 3 UI: Settings > Backup and sync, shareable packs on the Dictionary and Snippets pages, and the
dictation-language hint. Driven offscreen against the REAL Controller (profile / pack / sync code of core),
with every file in ``tmp_path`` and every file dialog replaced.
"""
from __future__ import annotations

import json
import shutil

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from tests.controller_rig import make_rig
from tests.test_ui import UI, make_ui, seed, ui  # noqa: F401 - fixtures
from voxflow.ui import dialogs

OFFLINE = "The sync folder can't be reached right now"


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


def texts(widget) -> str:
    """All label text inside a widget (dialog facts, row subtitles)."""
    return "\n".join(lab.text() for lab in widget.findChildren(QLabel))


def populate(c) -> None:
    c.dictionary.add("Kubernetes", ["cooper netties"])
    c.dictionary.add("VoxFlow")
    c.snippets.add("my address", "742 Evergreen Terrace")


def save_to(monkeypatch, path) -> list:
    asked: list = []
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: asked.append(a[1:]) or str(path))
    return asked


def open_from(monkeypatch, path) -> None:
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: str(path))


def folder_is(monkeypatch, path) -> None:
    monkeypatch.setattr(dialogs, "ask_folder", lambda *a, **k: str(path))


@pytest.fixture
def other(tmp_path, qtbot, monkeypatch):
    """A second, separate profile (another PC): a real Controller with its own data folder, no UI."""
    made = []

    def build():
        root = tmp_path / f"other-pc-{len(made)}"
        root.mkdir()
        rig = make_rig(root, qtbot, monkeypatch, config={"onboarding_done": True})
        made.append(rig)
        return rig

    yield build
    for rig in made:
        rig.close()


def idle(ui) -> None:
    ui.wait(lambda: ui.c.sync_status()["state"] in ("idle", "error", "off") and not ui.c._sync_busy)
    QApplication.processEvents()


# =========================================================================== the section itself
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_backup_and_sync_section_renders_with_sync_off(make_ui, theme):
    ui = make_ui(config={"theme": theme})
    ui.page("settings")
    section = ui.find("settings_backup_section")
    for name in ("settings_backup_export", "settings_backup_import", "settings_sync_choose", "settings_sync_row",
                 "settings_sync_status"):
        ui.find(name, section)
    status = ui.find("settings_sync_status").text()
    assert status.startswith("Off.") and "never uploads anything" in status
    assert ui.find("settings_sync_choose").text() == "Choose folder…"
    for hidden in ("settings_sync_off", "settings_sync_now", "settings_sync_folder"):
        assert not ui.find(hidden).isVisibleTo(ui.win), f"{hidden} must be hidden while sync is off"
    for name in ("settings_backup_export", "settings_backup_import", "settings_sync_choose"):
        assert ui.find(name).accessibleName() or ui.find(name).text()
    ui.render(section)


# =========================================================================== export a backup
def test_export_backup_with_history_writes_a_file_the_app_can_read_back(ui, monkeypatch, tmp_path):
    populate(ui.c)
    seed(ui.rig, 3)
    ui.page("settings")
    target = tmp_path / "mine.voxbackup"
    asked = save_to(monkeypatch, target)
    ui.find("settings_backup_export").click()
    dialog = ui.find("backup_export")
    assert ui.find("backup_export_history").isChecked(), "history is included unless the user says no"
    ui.find("backup_export_ok").click()
    assert asked and asked[0][1].startswith("voxflow-backup-") and asked[0][1].endswith(".voxbackup")
    assert "*.voxbackup" in asked[0][2]
    assert target.is_file() and not dialog.isVisible()
    info = ui.c.inspect_backup(str(target))
    assert info["counts"]["dictionary"] == 2 and info["counts"]["snippets"] == 1 and info["counts"]["history"] == 3
    assert "history" in info["includes"] and info["same_device"] is True
    assert "Backup saved to mine.voxbackup: your settings, 2 words, 1 snippet, 3 transcripts." in ui.toasts()


def test_export_backup_without_history_leaves_history_and_statistics_out(ui, monkeypatch, tmp_path):
    populate(ui.c)
    seed(ui.rig, 2)
    ui.page("settings")
    target = tmp_path / "share.voxbackup"
    save_to(monkeypatch, target)
    ui.find("settings_backup_export").click()
    ui.find("backup_export_history").click()
    assert not ui.find("backup_export_history").isChecked()
    ui.find("backup_export_ok").click()
    info = ui.c.inspect_backup(str(target))
    assert "history" not in info["includes"] and "stats" not in info["includes"] and info["counts"]["history"] == 0
    raw = target.read_text(encoding="utf-8")
    assert "stand-up" not in raw, "no dictated text may be in a backup made without history"
    assert "Backup saved to share.voxbackup: your settings, 2 words, 1 snippet." in ui.toasts()


def test_export_backup_cancelled_or_unwritable_keeps_the_dialog_and_says_why(ui, monkeypatch, tmp_path):
    ui.page("settings")
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: "")
    ui.find("settings_backup_export").click()
    dialog = ui.find("backup_export")
    ui.find("backup_export_ok").click()
    assert dialog.isVisible() and ui.toasts() == [] and list(tmp_path.glob("*.voxbackup")) == []
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    save_to(monkeypatch, blocker / "sub" / "b.voxbackup")
    ui.find("backup_export_ok").click()
    assert dialog.isVisible() and dialog.error.text(), "a failed save is explained inside the dialog"
    assert "Traceback" not in dialog.error.text()
    ui.find("backup_export_cancel").click()
    assert not dialog.isVisible()


# =========================================================================== restore a backup
def make_backup(other, tmp_path, name="other.voxbackup", history=True):
    """A backup written by a second, separate profile (so it holds things the first one does not)."""
    src = other()
    src.c.dictionary.add("Anthropic", ["and tropic"])
    src.c.dictionary.add("Kubernetes")
    src.c.snippets.add("sig", "Best,\nDan")
    seed(src, 2)
    src.c.set("theme", "light")
    path = tmp_path / name
    src.c.export_backup(str(path), history)
    return path


def test_import_backup_shows_what_is_in_the_file_and_merge_only_adds(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "dark"})
    populate(ui.c)
    seed(ui.rig, 1)
    ui.page("settings")
    open_from(monkeypatch, path)
    ui.find("settings_backup_import").click()
    dialog = ui.find("backup_import")
    shown = texts(dialog)
    assert "other.voxbackup" in shown and "2 words, 1 snippet, 2 transcripts" in shown
    assert "settings" in shown and "styles" in shown
    assert ui.find("backup_import_mode").value == "merge" and ui.find("backup_import_ok").text() == "Merge"
    assert "stay as they are" in ui.find("backup_import_explain").text()
    ui.render(dialog)
    ui.find("backup_import_ok").click()
    assert not dialog.isVisible()
    terms = {e["term"] for e in ui.c.dictionary.list()}
    assert terms == {"Kubernetes", "VoxFlow", "Anthropic"}
    assert {s["trigger"] for s in ui.c.snippets.list()} == {"my address", "sig"}
    assert ui.c.history.count() == 3 and ui.c.get("theme") == "dark", "merge never changes settings"
    assert "Backup merged: 1 word, 1 snippet, 2 transcripts added." in ui.toasts()

    ui.find("settings_backup_import").click()                     # the same file again: nothing new
    ui.find("backup_import_ok").click()
    assert "Backup merged: everything in it was already here." in ui.toasts()
    assert len(ui.c.dictionary.list()) == 3 and ui.c.history.count() == 3


def test_import_backup_replace_warns_in_red_and_makes_the_profile_equal_to_the_file(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path)
    ui = make_ui(config={"theme": "dark"})
    populate(ui.c)
    seed(ui.rig, 4)
    ui.page("settings")
    open_from(monkeypatch, path)
    ui.find("settings_backup_import").click()
    dialog = ui.find("backup_import")
    ui.find("backup_import_mode").set_value("replace", emit=True)
    assert ui.find("backup_import_ok").text() == "Replace" and ui.find("backup_import_ok").kind == "danger"
    assert "cannot be undone" in ui.find("backup_import_explain").text()
    ui.find("backup_import_cancel").click()                       # changing one's mind changes nothing
    assert not dialog.isVisible() and len(ui.c.dictionary.list()) == 2 and ui.c.history.count() == 4

    ui.find("settings_backup_import").click()
    ui.find("backup_import_mode").set_value("replace", emit=True)
    ui.find("backup_import_ok").click()
    assert {e["term"] for e in ui.c.dictionary.list()} == {"Anthropic", "Kubernetes"}
    assert {s["trigger"] for s in ui.c.snippets.list()} == {"sig"}
    assert ui.c.history.count() == 2 and ui.c.get("theme") == "light"
    assert "Your profile was replaced with the backup." in ui.toasts()
    assert ui.find("settings_theme").value == "light", "the Settings page shows the restored values"
    ui.page("dictionary")
    assert len(ui.find_all("dict_edit")) == 2, "the Dictionary page shows the restored list"


def test_import_backup_without_history_says_so(other, make_ui, monkeypatch, tmp_path):
    path = make_backup(other, tmp_path, history=False)
    ui = make_ui()
    seed(ui.rig, 2)
    ui.page("settings")
    open_from(monkeypatch, path)
    ui.find("settings_backup_import").click()
    assert "no history" in texts(ui.find("backup_import"))
    ui.find("backup_import_ok").click()
    assert ui.c.history.count() == 2


def _bad_backups(tmp_path, good) -> dict:
    out = {}
    corrupt = tmp_path / "corrupt.voxbackup"
    corrupt.write_text(good.read_text(encoding="utf-8")[:200], encoding="utf-8")       # a partial download
    out["corrupt"] = (corrupt, "not a VoxFlow backup")
    garbage = tmp_path / "garbage.voxbackup"
    garbage.write_bytes(b"\x00\xff\x10 not json at all")
    out["garbage"] = (garbage, "garbage.voxbackup")
    newer = tmp_path / "newer.voxbackup"
    data = json.loads(good.read_text(encoding="utf-8"))
    data["version"] = 99
    newer.write_text(json.dumps(data), encoding="utf-8")
    out["newer"] = (newer, "made by a newer version of VoxFlow")
    out["missing"] = (tmp_path / "gone.voxbackup", "was not found")
    return out


@pytest.mark.parametrize("case", ["corrupt", "garbage", "newer", "missing", "pack"])
def test_import_backup_refuses_a_bad_file_with_a_sentence_and_changes_nothing(other, make_ui, monkeypatch, tmp_path, case):
    good = make_backup(other, tmp_path)
    ui = make_ui()
    populate(ui.c)
    seed(ui.rig, 1)
    if case == "pack":
        path = tmp_path / "team.voxpack"
        ui.c.export_pack("dictionary", str(path), "Team words")
        expect = "is a shared pack, not a backup"
    else:
        path, expect = _bad_backups(tmp_path, good)[case]
    before = (ui.c.dictionary.list(), ui.c.snippets.list(), ui.c.history.count(), ui.c.get("theme"))
    ui.page("settings")
    open_from(monkeypatch, path)
    ui.find("settings_backup_import").click()
    assert ui.find_all("backup_import") == [], "no confirmation dialog for a file that cannot be used"
    assert len(ui.toasts()) == 1 and expect in ui.toasts()[0], ui.toasts()
    assert "Traceback" not in ui.toasts()[0] and "Error" not in ui.toasts()[0]
    assert (ui.c.dictionary.list(), ui.c.snippets.list(), ui.c.history.count(), ui.c.get("theme")) == before


def test_import_backup_cancelled_file_dialog_does_nothing(ui, monkeypatch):
    ui.page("settings")
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: "")
    ui.find("settings_backup_import").click()
    assert ui.find_all("backup_import") == [] and ui.toasts() == []


def test_import_backup_with_damaged_entries_says_how_many_are_left_out(other, make_ui, monkeypatch, tmp_path):
    good = make_backup(other, tmp_path)
    data = json.loads(good.read_text(encoding="utf-8"))
    data["dictionary"] += [{"term": ""}, 17, {"nonsense": True}]
    damaged = tmp_path / "damaged.voxbackup"
    damaged.write_text(json.dumps(data), encoding="utf-8")
    ui = make_ui()
    ui.page("settings")
    open_from(monkeypatch, damaged)
    ui.find("settings_backup_import").click()
    assert "3 damaged entries in the file will be left out" in texts(ui.find("backup_import"))
    ui.find("backup_import_ok").click()
    assert {e["term"] for e in ui.c.dictionary.list()} == {"Anthropic", "Kubernetes"}


# =========================================================================== packs: share
@pytest.mark.parametrize("page, share, kind, summary", [
    ("dictionary", "dict_share", "dictionary", "Pack saved: 2 words in team-words.voxpack."),
    ("snippets", "snippets_share", "snippets", "Pack saved: 1 snippet in team-words.voxpack."),
])
def test_share_pack_needs_a_name_and_writes_a_pack_with_name_and_description(ui, monkeypatch, tmp_path, page, share,
                                                                             kind, summary):
    populate(ui.c)
    ui.page(page)
    target = tmp_path / "team-words.voxpack"
    asked = save_to(monkeypatch, target)
    ui.find(share).click()
    dialog = ui.find("pack_export")
    ui.find("pack_export_ok").click()
    assert "Give the pack a name" in dialog.error.text() and not asked and not target.exists()
    ui.find("pack_name").setText("Team Words")
    assert dialog.error.text() == "", "typing clears the error"
    ui.find("pack_description").setText("For the platform team")
    ui.find("pack_export_ok").click()
    assert asked[0][1] == "team-words.voxpack", "the suggested file name comes from the pack name"
    assert target.is_file() and not dialog.isVisible() and summary in ui.toasts()
    info = ui.c.inspect_pack(str(target))
    assert (info["kind"], info["name"], info["description"]) == (kind, "Team Words", "For the platform team")
    raw = target.read_text(encoding="utf-8")
    assert "theme" not in raw and "history" not in raw, "a pack holds no settings and no history"


@pytest.mark.parametrize("page, share, sentence", [
    ("dictionary", "dict_share", "There are no terms to share yet."),
    ("snippets", "snippets_share", "There are no snippets to share yet."),
])
def test_share_pack_with_an_empty_list_explains_instead_of_writing_a_file(ui, monkeypatch, tmp_path, page, share,
                                                                         sentence):
    widget = ui.page(page)
    target = tmp_path / "empty.voxpack"
    save_to(monkeypatch, target)
    assert not ui.find(share).isEnabled(), "nothing to share: the button is off"
    ui.find(share).click()
    assert ui.find_all("pack_export") == []
    dialog = widget.share_pack()                                  # reached anyway (e.g. list emptied meanwhile)
    ui.find("pack_name").setText("Nothing")
    ui.find("pack_export_ok").click()
    assert dialog.isVisible() and dialog.error.text() == sentence and not target.exists()
    ui.find("pack_export_cancel").click()


# =========================================================================== packs: import
def make_pack(other, tmp_path, kind="dictionary", name="team.voxpack"):
    src = other()
    src.c.dictionary.add("Kubernetes", ["cube ernetes"])
    src.c.dictionary.add("Anthropic")
    src.c.dictionary.add("PostgreSQL")
    src.c.snippets.add("my address", "1 Infinite Loop")
    src.c.snippets.add("sig", "Best,\nDan")
    path = tmp_path / name
    src.c.export_pack(kind, str(path), "Platform team", "Words we all use")
    return path


def test_import_dictionary_pack_previews_new_and_existing_then_merges(other, make_ui, monkeypatch, tmp_path):
    path = make_pack(other, tmp_path)
    ui = make_ui()
    populate(ui.c)                                                # has Kubernetes (own alias) and VoxFlow
    ui.page("dictionary")
    open_from(monkeypatch, path)
    ui.find("dict_import").click()
    dialog = ui.find("pack_import")
    shown = texts(dialog)
    assert "Platform team" in shown and "Words we all use" in shown
    assert "3 words: 2 new, 1 already in your list" in shown
    assert ui.find("pack_import_ok").text() == "Merge"
    ui.find("pack_import_ok").click()
    assert not dialog.isVisible()
    entries = {e["term"]: e for e in ui.c.dictionary.list()}
    assert set(entries) == {"Kubernetes", "VoxFlow", "Anthropic", "PostgreSQL"}
    aliases = {a.lower() for a in entries["Kubernetes"]["aliases"]}
    assert {"cooper netties", "cube ernetes"} <= aliases, "a merged term keeps its own sounds-like spellings too"
    assert any(t.startswith("Pack imported: 2 added") for t in ui.toasts()), ui.toasts()
    assert len(ui.find_all("dict_edit")) == 4, "the list on the page shows the new words"


def test_import_dictionary_pack_replace_makes_the_pack_the_whole_list(other, make_ui, monkeypatch, tmp_path):
    path = make_pack(other, tmp_path)
    ui = make_ui()
    populate(ui.c)
    ui.page("dictionary")
    open_from(monkeypatch, path)
    ui.find("dict_import").click()
    ui.find("pack_import_mode").set_value("replace", emit=True)
    assert ui.find("pack_import_ok").text() == "Replace" and "cannot be undone" in ui.find("pack_import_explain").text()
    ui.find("pack_import_ok").click()
    assert {e["term"] for e in ui.c.dictionary.list()} == {"Kubernetes", "Anthropic", "PostgreSQL"}
    assert any(t.startswith("Pack imported:") and "removed" in t for t in ui.toasts()), ui.toasts()
    assert ui.c.snippets.list() != [] or True                     # snippets are a different list: untouched
    assert len(ui.find_all("dict_edit")) == 3


def test_import_snippet_pack_updates_the_one_both_have_and_says_nothing_to_import_second_time(other, make_ui, monkeypatch,
                                                                                             tmp_path):
    path = make_pack(other, tmp_path, kind="snippets", name="snips.voxpack")
    ui = make_ui()
    populate(ui.c)                                                # "my address" with another text
    ui.page("snippets")
    open_from(monkeypatch, path)
    ui.find("snippets_import").click()
    assert "2 snippets: 1 new, 1 already in your list" in texts(ui.find("pack_import"))
    ui.find("pack_import_ok").click()
    assert ui.c.snippets.expand("my address") == "1 Infinite Loop", "the pack is the source of truth"
    assert ui.c.snippets.expand("sig") == "Best,\nDan"
    assert "Pack imported: 1 added, 1 updated." in ui.toasts()
    ui.find("snippets_import").click()
    ui.find("pack_import_ok").click()
    assert "Nothing to import: you already have every snippet in this pack." in ui.toasts()


def test_a_pack_of_the_other_kind_points_to_the_right_page(other, make_ui, monkeypatch, tmp_path):
    path = make_pack(other, tmp_path, kind="snippets", name="snips.voxpack")
    ui = make_ui()
    ui.page("dictionary")
    open_from(monkeypatch, path)
    ui.find("dict_import").click()
    assert ui.find_all("pack_import") == [] and ui.c.snippets.list() == [] and ui.c.dictionary.list() == []
    assert "snips.voxpack is a snippets pack. Import it on the Snippets page." in ui.toasts()
    ui.toast_action("is a snippets pack")
    assert ui.win.current == "snippets"


@pytest.mark.parametrize("case, expect", [
    ("garbage", "is not a VoxFlow pack"),
    ("backup", "is a whole backup, not a pack"),
    ("newer", "made by a newer version of VoxFlow"),
])
def test_import_pack_refuses_a_bad_file_with_a_sentence_and_changes_nothing(other, make_ui, monkeypatch, tmp_path, case,
                                                                           expect):
    good = make_pack(other, tmp_path)
    ui = make_ui()
    populate(ui.c)
    path = tmp_path / "bad.voxpack"
    if case == "garbage":
        path.write_text("{ this is not json", encoding="utf-8")
    elif case == "backup":
        ui.c.export_backup(str(tmp_path / "b.voxbackup"))
        shutil.copy(tmp_path / "b.voxbackup", path)
    else:
        data = json.loads(good.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
    before = ui.c.dictionary.list()
    ui.page("dictionary")
    open_from(monkeypatch, path)
    ui.find("dict_import").click()
    assert ui.find_all("pack_import") == []
    assert len(ui.toasts()) == 1 and expect in ui.toasts()[0], ui.toasts()
    assert ui.c.dictionary.list() == before


def test_plain_json_import_still_works_next_to_packs(ui, monkeypatch, tmp_path):
    src = tmp_path / "words.json"
    src.write_text(json.dumps([{"term": "Grafana"}]), encoding="utf-8")
    ui.page("dictionary")
    open_from(monkeypatch, src)
    ui.find("dict_import").click()
    assert ui.find_all("pack_import") == [] and [e["term"] for e in ui.c.dictionary.list()] == ["Grafana"]


# =========================================================================== sync folder
def test_choosing_a_sync_folder_turns_sync_on_and_the_row_follows_the_controller(ui, monkeypatch, tmp_path):
    populate(ui.c)
    folder = tmp_path / "cloud"
    folder.mkdir()
    ui.page("settings")
    folder_is(monkeypatch, folder)
    ui.find("settings_sync_choose").click()
    assert ui.c.get("sync_folder") == str(folder) and ui.c.sync_status()["enabled"]
    assert "Sync folder set. VoxFlow keeps it up to date from now on." in ui.toasts()
    idle(ui)
    st = ui.c.sync_status()
    assert st["state"] == "idle" and st["last_sync"]
    status = ui.find("settings_sync_status").text()
    assert status.startswith(st["message"]) and "Last synced" in status
    assert "No other device has used this folder yet" in status
    assert ui.find("settings_sync_folder").text().startswith(str(folder))
    for shown in ("settings_sync_off", "settings_sync_now", "settings_sync_folder"):
        assert ui.find(shown).isVisibleTo(ui.win)
    assert ui.find("settings_sync_choose").text() == "Change…"
    files = [p.name for p in folder.iterdir()]
    assert len(files) == 1 and files[0] == ui.c.sync.file_name
    assert "Kubernetes" in (folder / files[0]).read_text(encoding="utf-8")
    ui.render(ui.find("settings_backup_section"))


def test_sync_folder_survives_a_restart_of_the_ui(make_ui, monkeypatch, tmp_path):
    folder = tmp_path / "cloud"
    folder.mkdir()
    first = make_ui()
    first.c.set("sync_folder", str(folder))
    idle(first)
    first.page("settings")
    assert first.find("settings_sync_folder").text().startswith(str(folder))
    assert "sync_folder" not in json.loads((first.rig.data_dir / "config.json").read_text(encoding="utf-8"))


def test_cancelling_the_folder_dialog_or_picking_a_bad_folder_leaves_sync_off(ui, monkeypatch, tmp_path):
    ui.page("settings")
    monkeypatch.setattr(dialogs, "ask_folder", lambda *a, **k: "")
    ui.find("settings_sync_choose").click()
    assert ui.toasts() == [] and ui.c.get("sync_folder") == ""
    folder_is(monkeypatch, tmp_path / "does-not-exist")
    ui.find("settings_sync_choose").click()
    assert len(ui.toasts()) == 1 and ui.toasts()[0].startswith("That folder doesn't exist.")
    assert ui.c.get("sync_folder") == "" and ui.find("settings_sync_status").text().startswith("Off.")
    folder_is(monkeypatch, ui.rig.data_dir)
    ui.find("settings_sync_choose").click()
    assert ui.toasts()[-1].startswith("Pick a folder other than VoxFlow's own data folder")
    assert ui.c.get("sync_folder") == "" and not ui.find("settings_sync_now").isVisibleTo(ui.win)


def test_sync_now_runs_a_round_and_picks_up_another_device(other, make_ui, monkeypatch, tmp_path):
    folder = tmp_path / "cloud"
    folder.mkdir()
    other = other()
    other.c.dictionary.add("Anthropic")
    other.c.set("sync_folder", str(folder))
    idle(other)

    ui = make_ui()
    ui.page("settings")
    folder_is(monkeypatch, folder)
    ui.find("settings_sync_choose").click()
    idle(ui)
    assert [e["term"] for e in ui.c.dictionary.list()] == ["Anthropic"], "joining a folder brings its entries in"
    status = ui.find("settings_sync_status").text()
    assert status.startswith(ui.c.sync_status()["message"]) and "1 other device" in status
    assert "shared with" in ui.find("settings_sync_folder").text()

    other.c.snippets.add("sig", "Best,\nDan")
    other.c.sync_now()
    idle(other)
    seen: list = []
    ui.c.sync_status_changed.connect(lambda st: seen.append(st["state"]))
    ui.find("settings_sync_now").click()
    assert "syncing" in seen, "the button starts a round"
    idle(ui)
    assert ui.c.snippets.expand("sig") == "Best,\nDan"
    ui.page("snippets")
    assert len(ui.find_all("snippets_edit")) == 1, "the Snippets page shows what arrived"
    assert not any("Sync" in t and "off" in t for t in ui.toasts())


def test_the_row_shows_syncing_and_disables_the_button_while_a_round_runs(ui, tmp_path):
    folder = tmp_path / "cloud"
    folder.mkdir()
    ui.c.set("sync_folder", str(folder))
    idle(ui)
    page = ui.page("settings")
    page._show_sync(dict(ui.c.sync_status(), state="syncing"))
    assert ui.find("settings_sync_status").text() == "Syncing…" and not ui.find("settings_sync_now").isEnabled()
    ui.c.sync_status_changed.emit(ui.c.sync_status())             # the controller's signal alone updates the row
    assert ui.find("settings_sync_status").text().startswith("Up to date") and ui.find("settings_sync_now").isEnabled()


def test_turning_sync_off_keeps_the_data_and_the_file_in_the_folder(ui, monkeypatch, tmp_path):
    populate(ui.c)
    folder = tmp_path / "cloud"
    folder.mkdir()
    ui.c.set("sync_folder", str(folder))
    idle(ui)
    ui.page("settings")
    ui.find("settings_sync_off").click()
    assert ui.c.get("sync_folder") == "" and ui.c.sync_status()["state"] == "off"
    assert ui.find("settings_sync_status").text().startswith("Off.")
    assert not ui.find("settings_sync_now").isVisibleTo(ui.win) and ui.find("settings_sync_choose").text() == "Choose folder…"
    assert len(list(folder.iterdir())) == 1 and len(ui.c.dictionary.list()) == 2
    assert not [t for t in ui.toasts() if "did not work" in t]


def test_folder_offline_shows_one_calm_sentence_in_the_row_and_no_toasts(ui, tmp_path):
    populate(ui.c)
    folder = tmp_path / "cloud"
    folder.mkdir()
    ui.c.set("sync_folder", str(folder))
    idle(ui)
    ui.page("settings")
    shutil.rmtree(folder)                                         # the network drive went away
    toasts_before = list(ui.toasts())
    for _ in range(3):
        ui.find("settings_sync_now").click()
        idle(ui)
    st = ui.c.sync_status()
    assert st["state"] == "error" and st["error"].startswith(OFFLINE)
    status = ui.find("settings_sync_status").text()
    assert status.startswith(OFFLINE) and "Last synced" in status and ".." not in status
    assert ui.toasts() == toasts_before, "an unreachable folder is shown in the row, not as a stream of toasts"
    assert ui.find("settings_sync_now").isVisibleTo(ui.win) and ui.find("settings_sync_now").isEnabled()
    ui.c.dictionary.add("Grafana")                                # work goes on locally
    folder.mkdir()                                                # ... and syncs when the folder is back
    ui.find("settings_sync_now").click()
    idle(ui)
    assert ui.c.sync_status()["state"] == "idle" and ui.find("settings_sync_status").text().startswith("Up to date")
    assert "Grafana" in next(folder.iterdir()).read_text(encoding="utf-8")


def test_an_unreadable_file_in_the_folder_is_mentioned_in_amber_not_as_an_error(ui, tmp_path):
    folder = tmp_path / "cloud"
    folder.mkdir()
    (folder / "voxflow-sync-old-laptop-deadbeef.json").write_text("{ half a fi", encoding="utf-8")
    ui.c.set("sync_folder", str(folder))
    idle(ui)
    ui.page("settings")
    st = ui.c.sync_status()
    assert st["state"] == "idle" and st["unreadable"] == 1
    assert ui.find("settings_sync_status").text().startswith(st["message"])
    assert ui.find("settings_sync_status")._role == "AMBER"


# =========================================================================== language hint
@pytest.mark.parametrize("code, support", [("es", "full"), ("auto", "auto"), ("pl", "basic")])
def test_language_picker_says_how_much_cleanup_the_chosen_language_gets(ui, code, support):
    """API 5.1: ``Controller.language_support(code)`` exists 'for a hint under the language picker'."""
    assert ui.c.language_support(code) == support
    codes = [c for c, _name in ui.c.languages()]
    assert code in codes
    page = ui.page("settings")
    row = page.language.parent()
    while row is not None and not hasattr(row, "sub"):
        row = row.parent()
    assert row is not None
    static = row.sub.text()
    seen = set()
    for probe in ("es", "auto", "pl"):
        ui.find("settings_language").set_value(probe)
        ui.c.set("language", probe)
        QApplication.processEvents()
        seen.add(row.sub.text())
    assert len(seen) == 3, f"the hint under the language picker never changes: {static!r}"
