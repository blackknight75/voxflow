"""Data-layer findings of the 1.0 QA pass, sync side (``voxflow.sync``): records stamped by a clock that ran
ahead (DL-03), an import that is all or nothing on disk too (DL-04) and is undone exactly (DL-05), a lost sync
state file (DL-08), hostile JSON (DL-10) and deletions that outlast a long absence (DL-11)."""
from __future__ import annotations

import json
import os
import stat
import time

import pytest

from tests.test_sync import Clock, Device, everything, pair, rounds, sync_files
from voxflow import config as cfgmod
from voxflow import sync as S

DAY = 86400.0
DEEP = "[" * 100_000 + "]" * 100_000
HUGE = 10 ** 400


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def folder(tmp_path):
    shared = tmp_path / "cloud"
    shared.mkdir()
    return shared


def records(dev, kind="dictionary") -> dict:
    return dev.engine._state["records"][kind]


# =========================================================================== DL-03: a clock that ran ahead
def two_clocks(tmp_path, folder):
    a, b = Device(tmp_path, "desk", Clock()), Device(tmp_path, "laptop", Clock())
    a.engine.set_folder(str(folder))
    b.engine.set_folder(str(folder))
    return a, b


def test_an_entry_edited_while_the_clock_ran_ahead_syncs_again_once_the_clock_is_right(tmp_path, folder):
    a, b = two_clocks(tmp_path, folder)
    a.dictionary.add("Zanzibar")
    rounds(a, b)
    a.clock.t += 30 * DAY                                  # the PC's clock is a month ahead for one edit
    a.dictionary.update(a.term_id("Zanzibar"), aliases=["made while the clock was ahead"])
    a.engine.sync()
    assert records(a)["zanzibar"]["t"] == a.clock.t
    a.clock.t -= 30 * DAY                                  # ... and is set right again
    rounds(a, b)
    assert b.terms() == {"Zanzibar": ["made while the clock was ahead"]}, "no longer ignored for a month"
    assert records(a)["zanzibar"]["t"] <= a.clock.t and records(a)["zanzibar"]["d"] == a.engine.device_id
    a.dictionary.update(a.term_id("Zanzibar"), aliases=["a later edit"])
    rounds(a, b)
    assert b.terms() == {"Zanzibar": ["a later edit"]}
    b.dictionary.update(b.term_id("Zanzibar"), aliases=["edited on the laptop"])
    rounds(a, b)
    assert a.terms() == {"Zanzibar": ["edited on the laptop"]}, "and it no longer beats every edit made elsewhere"
    a.dictionary.remove(a.term_id("Zanzibar"))
    rounds(a, b)
    assert a.terms() == b.terms() == {}


def test_a_deletion_stamped_ahead_is_brought_back_too(tmp_path, folder):
    a, b = two_clocks(tmp_path, folder)
    a.snippets.add("sig", "Best")
    rounds(a, b)
    a.clock.t += 400 * DAY
    a.snippets.remove(a.snip_id("sig"))
    a.engine.stamp_local()
    a.clock.t -= 400 * DAY
    rounds(a, b)
    assert b.snips() == {} and records(a, "snippets")["sig"] == {"x": 1, "t": pytest.approx(a.clock.t, abs=60),
                                                                 "d": a.engine.device_id}


def test_when_most_of_this_devices_stamps_look_ahead_it_is_todays_clock_that_is_wrong(tmp_path, folder):
    a, b = two_clocks(tmp_path, folder)
    for term in ("Alpha", "Bravo", "Charlie", "Delta"):
        a.dictionary.add(term)
    rounds(a, b)
    stamps = {k: r["t"] for k, r in records(a).items()}
    a.clock.t -= 10 * DAY                                  # a dead battery: today's clock is in the past
    assert a.engine.stamp_local() is False
    assert {k: r["t"] for k, r in records(a).items()} == stamps, "an edit is never dated back"
    a.clock.t += 10 * DAY
    a.clock.t += 30 * DAY                                  # one of the four, and only that one, was made ahead
    a.dictionary.update(a.term_id("Delta"), aliases=["ahead"])
    a.engine.stamp_local()
    a.clock.t -= 30 * DAY
    assert a.engine.stamp_local() is True
    assert records(a)["delta"]["t"] == a.clock.t and {k: records(a)[k]["t"] for k in ("alpha", "bravo", "charlie")} == {
        k: stamps[k] for k in ("alpha", "bravo", "charlie")}


def test_only_this_devices_own_records_are_given_a_new_time(tmp_path, folder):
    a, b = two_clocks(tmp_path, folder)
    for term in ("Alpha", "Bravo", "Charlie"):
        a.dictionary.add(term)
    rounds(a, b)
    for dev in (a, b):
        dev.clock.t += 5 * DAY
    b.dictionary.add("FromTheLaptop")
    rounds(a, b)
    theirs = dict(records(a)["fromthelaptop"])
    assert theirs["d"] == b.engine.device_id
    a.clock.t -= 1.5 * DAY                                 # the laptop's record now lies beyond this PC's skew window
    assert theirs["t"] > a.clock.t + DAY
    assert a.engine.stamp_local() is False and records(a)["fromthelaptop"] == theirs
    assert all(r["t"] == S.BASELINE for r in records(a, "settings").values()), "what it joined with is not a stamp"


# =========================================================================== DL-04 / DL-05: imports
def full_backup(dev, tmp_path) -> str:
    return S.export_profile(dev.ds, str(tmp_path / "b"))["path"]


@pytest.fixture
def dev(tmp_path, clock):
    d = Device(tmp_path, "alpha", clock)
    d.dictionary.add("Kubernetes", ["cooper netties"])
    d.dictionary.add("VoxFlow")
    d.snippets.add("my email", "ada@example.com")
    d.styles.set_app_category("obsidian.exe", "other")
    d.history.add("first dictation")
    d.stats.record(12, 5.0, "notepad.exe")
    d.settings["sounds_enabled"] = False
    return d


def beta(tmp_path, clock) -> Device:
    """A second profile with data of its own, an old history file's backup copy and a copy of a damaged one."""
    home = tmp_path / "beta"
    home.mkdir()
    (home / "history.json").write_text(json.dumps([{"text": "my history", "ts": clock.t - 60}]), encoding="utf-8")
    (home / "history.corrupt-20260101-000000.json").write_text("{ damaged", encoding="utf-8")
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("Mine", ["mein"])
    other.dictionary.suggest("VoxFlow", ["vox flow"])      # the backup has this term: adding it drops the suggestion
    other.snippets.add("mine", "text")
    other.styles.set_app_category("mine.exe", "work")
    other.stats.record(3, 1.0, "mine.exe")
    for st in stores(other):
        st.flush()
    assert (home / "history.v1.json").exists()
    return other


def stores(d: Device):
    return d.dictionary, d.snippets, d.styles, d.stats, d.history


def exactly(d: Device) -> dict:
    """Everything, ids and the suggestion queue included, plus the history's kept copies."""
    return {"all": everything(d), "dictionary": d.dictionary.list(), "suggestions": d.dictionary.suggestions(),
            "snippets": d.snippets.list(), "history": d.history.snapshot(),
            "copies": sorted(p.name for p in d.dir.glob("history.*.json"))}


def test_an_import_whose_file_cannot_be_written_is_undone_in_memory_and_on_disk(dev, tmp_path, clock, monkeypatch):
    path = full_backup(dev, tmp_path)
    other = beta(tmp_path, clock)
    before = exactly(other)
    topics: list[str] = []
    other.snippets.subscribe(topics.append)
    real = cfgmod._atomic_write
    written: list[str] = []

    def write(target, text):
        if target == other.snippets.path:
            raise PermissionError("another program has the file open")
        written.append(target.name)
        return real(target, text)

    with monkeypatch.context() as m:
        m.setattr(cfgmod, "_atomic_write", write)
        with pytest.raises(ValueError) as err:
            S.import_profile(other.ds, path, "replace")
        assert str(err.value) == ("Your snippets could not be saved, so the import was undone and nothing was changed. "
                                  "Check that the disk has room and that VoxFlow's data folder can be written to.")
        assert written[0] == "dictionary.json" and "history.json" not in written[:1], "one file at a time"
        assert exactly(other) == before
    assert "save_failed:snippets" in topics
    assert exactly(other.reopen()) == before, "the files hold what they held before the import"


def test_a_history_file_that_is_read_only_fails_the_import_after_the_other_files_were_written(dev, tmp_path, clock):
    path = full_backup(dev, tmp_path)
    other = beta(tmp_path, clock)
    before = exactly(other)
    os.chmod(other.history.path, stat.S_IREAD)
    try:
        with pytest.raises(ValueError) as err:
            S.import_profile(other.ds, path, "replace")
    finally:
        os.chmod(other.history.path, stat.S_IWRITE | stat.S_IREAD)
    assert str(err.value) == ("Your history could not be saved (history.json is read-only), so the import was undone "
                              "and nothing was changed.")
    assert exactly(other) == before
    assert exactly(other.reopen()) == before, "the dictionary, snippets, styles and statistics were written back"


def test_a_pack_whose_file_cannot_be_written_is_not_imported(dev, tmp_path, clock, monkeypatch):
    path = S.export_pack(dev.ds, "snippets", str(tmp_path / "p"))["path"]
    other = beta(tmp_path, clock)
    before = exactly(other)
    real = cfgmod._atomic_write
    with monkeypatch.context() as m:
        m.setattr(cfgmod, "_atomic_write", lambda target, text: (_ for _ in ()).throw(OSError("disk full"))
                  if target == other.snippets.path else real(target, text))
        with pytest.raises(ValueError, match="Your snippets could not be saved, so the import was undone"):
            S.import_pack(other.ds, path, "merge")
    assert exactly(other) == before == exactly(other.reopen())


def test_an_import_that_works_is_on_disk_when_it_returns_history_included(dev, tmp_path, clock):
    path = full_backup(dev, tmp_path)
    other = beta(tmp_path, clock)
    out = S.import_profile(other.ds, path, "replace")
    assert out["counts"]["history"] == 1 and not any(st._dirty for st in stores(other))
    assert json.loads(other.history.path.read_text(encoding="utf-8"))["items"][0]["text"] == "first dictation"
    assert sorted(p.name for p in other.dir.glob("history.*.json")) == [], "replaced for good: no copy survives"
    assert everything(other.reopen()) == everything(other)


@pytest.mark.parametrize("step", ["settings", "history", "stats"])
def test_a_failed_import_changes_nothing_not_even_ids_suggestions_or_the_kept_copies(dev, tmp_path, clock, monkeypatch,
                                                                                    step):
    path = full_backup(dev, tmp_path)
    other = beta(tmp_path, clock)
    before = exactly(other)
    assert before["suggestions"] and before["copies"] == ["history.corrupt-20260101-000000.json", "history.v1.json"]

    def boom(*a, **k):
        raise RuntimeError("no")

    if step == "settings":
        monkeypatch.setattr(other.ds, "set_settings", boom)        # after the stores were changed
    elif step == "history":
        monkeypatch.setattr(other.history, "import_entries", boom)
    else:
        monkeypatch.setattr(other.stats, "import_days", boom)      # after the history was replaced
    with pytest.raises(ValueError, match="could not be completed, so nothing was changed"):
        S.import_profile(other.ds, path, "replace")
    monkeypatch.undo()
    assert exactly(other) == before
    assert exactly(other.reopen()) == before


def test_settings_changed_by_an_import_that_fails_later_are_put_back(dev, tmp_path, clock, monkeypatch):
    path = full_backup(dev, tmp_path)
    other = beta(tmp_path, clock)
    assert other.settings["sounds_enabled"] is True
    monkeypatch.setattr(other.history, "import_entries", lambda *a, **k: (_ for _ in ()).throw(OSError("locked")))
    with pytest.raises(ValueError, match="nothing was changed"):
        S.import_profile(other.ds, path, "replace")
    assert other.settings == cfgmod.defaults()


# =========================================================================== DL-08: the sync state file
def reopened(dev: Device, name: str = "desk") -> S.SyncEngine:
    return S.SyncEngine(dev.ds, dev.dir / "sync_state.json", clock=dev.clock, name=name)


@pytest.mark.parametrize("damage", ["truncated", "empty", "a list", "nested"])
def test_a_sync_state_file_that_cannot_be_read_is_said_kept_and_leaves_no_ghost_device(tmp_path, folder, clock, damage):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    state = a.dir / "sync_state.json"
    former, former_id = a.engine.file_name, a.engine.device_id
    content = {"truncated": state.read_text(encoding="utf-8")[:60], "empty": "", "a list": "[1, 2]",
               "nested": DEEP}[damage]
    state.write_text(content, encoding="utf-8")

    eng = reopened(a)
    [copy] = a.dir.glob("sync_state.corrupt-*.json")
    assert copy.read_text(encoding="utf-8") == content
    assert eng.load_problem == ("VoxFlow's sync information on this PC could not be read. If you use a sync folder, "
                                "choose it again in Settings: sync is off here until then, and nothing was deleted. "
                                f"The old file was kept as {copy.name} in VoxFlow's data folder.")
    assert eng.status()["state"] == "off" and eng.device_id != former_id
    assert json.loads(state.read_text(encoding="utf-8"))["lost"] is True, "a readable file again: said once"
    later = reopened(a)
    assert later.load_problem == "" and later.device_id == eng.device_id and later._state["lost"] is True
    assert len(list(a.dir.glob("sync_state.corrupt-*.json"))) == 1

    # files this device must never touch: another name, not JSON, a name that is not its own id's
    keep = {"voxflow-sync-desk-aaaaaaaa.json": {"name": "desk!", "device": "aaaaaaaa0000"},
            "voxflow-sync-desk-bbbbbbbb.json": None,
            "voxflow-sync-desk-cccccccc.json": {"name": "desk", "device": "dddddddd0000"},
            "voxflow-sync-desk-eeeeeeee.json": {"name": "desk", "device": 7}}
    for name, doc in keep.items():
        text = json.dumps({"format": S.SYNC_FORMAT, "version": 1, "records": {}, **(doc or {})})
        (folder / name).write_text(text if doc else "{ half a file", encoding="utf-8")
    assert former in sync_files(folder)
    later.set_folder(str(folder))                          # the user chooses the folder again
    assert later._state["lost"] is False
    assert sync_files(folder) == sorted([b.engine.file_name, *keep]), "this PC's former file is gone, nothing else"
    a.engine = later
    for name in keep:
        (folder / name).unlink()
    rounds(a, b)
    assert [d["name"] for d in b.engine.status()["devices"]] == ["desk"], "one desk, not two"
    assert a.terms() == b.terms() == {"Kubernetes": []}
    assert reopened(a).load_problem == ""


def test_a_sync_state_file_that_cannot_even_be_copied_is_left_alone(tmp_path, clock):
    d = Device(tmp_path, "alpha", clock)
    (d.dir / "sync_state.json").mkdir()                    # e.g. no permission to read it: nothing to copy
    eng = S.SyncEngine(d.ds, d.dir / "sync_state.json", clock=clock)
    assert eng.load_problem == ("VoxFlow's sync information on this PC could not be read. If you use a sync folder, "
                                "choose it again in Settings: sync is off here until then, and nothing was deleted.")
    assert (d.dir / "sync_state.json").is_dir() and not list(d.dir.glob("sync_state.corrupt*"))
    assert d.engine.load_problem == "", "no file yet is not a problem"


def test_the_former_file_is_only_removed_when_that_is_safe_and_possible(tmp_path, folder, clock, monkeypatch):
    a, b = pair(tmp_path, folder, clock)
    rounds(a, b)
    former = a.engine.file_name
    state = a.dir / "sync_state.json"

    def lose() -> S.SyncEngine:
        state.write_text("{ damaged " + str(clock.tick()), encoding="utf-8")
        return reopened(a)

    with monkeypatch.context() as m:                       # larger than any sync file: not read at all
        m.setattr(S, "_MAX_FILE_BYTES", 10)
        lose().set_folder(str(folder))
        assert former in sync_files(folder)
    real_unlink = type(folder).unlink
    with monkeypatch.context() as m:                       # the folder's sync tool has it open
        m.setattr(type(folder), "unlink", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("in use"))
                  if self.name == former else real_unlink(self, *a, **k))
        lose().set_folder(str(folder))
        assert former in sync_files(folder)
    real_listdir = os.listdir
    with monkeypatch.context() as m:                       # the folder went away between the check and the look
        m.setattr(S.os, "listdir", lambda path: (_ for _ in ()).throw(OSError("offline")))
        eng = lose()
        eng.set_folder(str(folder))
        assert eng.enabled and former in real_listdir(folder)
    eng = lose()
    eng.set_folder(str(folder))
    assert former not in sync_files(folder) and b.engine.file_name in sync_files(folder)
    eng.set_folder(str(folder))                            # nothing is lost any more: nothing is looked for
    assert sync_files(folder) == [b.engine.file_name]


# =========================================================================== DL-10: hostile JSON
def test_a_sync_file_nested_too_deeply_is_skipped_and_the_round_goes_on(tmp_path, folder, clock):
    a, b = pair(tmp_path, folder, clock)
    (folder / "voxflow-sync-evil-00000001.json").write_text(DEEP, encoding="utf-8")
    (folder / "voxflow-sync-evil-00000002.json").write_text('{"format": "voxflow-sync", "records": ' + DEEP + "}",
                                                            encoding="utf-8")
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    assert b.terms() == {"Kubernetes": []}
    assert a.engine.status()["unreadable"] == 2 and "could not be read yet" in a.engine.status()["message"]


def test_a_number_too_large_for_a_float_in_a_sync_file_costs_that_record_only(tmp_path, folder, clock):
    a, b = pair(tmp_path, folder, clock)
    t = clock.t
    doc = {"format": "voxflow-sync", "version": 1, "device": "odd00000", "name": "odd", "updated": HUGE,
           "records": {"dictionary": {"huge": {"v": {"term": "Huge"}, "t": HUGE, "d": "odd00000"},
                                      "good": {"v": {"term": "Good"}, "t": t, "d": "odd00000"},
                                      "true": {"v": {"term": "True"}, "t": True, "d": "odd00000"}}}}
    (folder / "voxflow-sync-odd-odd00000.json").write_text(json.dumps(doc), encoding="utf-8")
    rounds(a, b)
    assert a.terms() == b.terms() == {"Good": []}
    assert {"id": "odd00000", "name": "odd", "updated": 0.0} in a.engine.status()["devices"]


def test_a_sync_state_file_with_a_number_too_large_loads(tmp_path, folder, clock):
    d = Device(tmp_path, "alpha", clock)
    d.engine.set_folder(str(folder))
    d.dictionary.add("Kubernetes")
    d.engine.sync()
    state = d.dir / "sync_state.json"
    data = json.loads(state.read_text(encoding="utf-8"))
    data["last_sync"] = HUGE
    data["records"]["dictionary"]["kubernetes"]["t"] = HUGE
    state.write_text(json.dumps(data), encoding="utf-8")
    eng = S.SyncEngine(d.ds, state, clock=clock)
    assert eng.load_problem == "" and eng.enabled and eng.status()["last_sync"] is None
    assert "kubernetes" not in eng._state["records"]["dictionary"]


def test_backups_and_packs_with_hostile_json_are_refused_with_a_sentence_or_read_without_the_bad_value(tmp_path, clock):
    d = Device(tmp_path, "alpha", clock)
    for name, what in (("x.voxpack", "pack"), ("x.voxbackup", "backup")):
        path = tmp_path / name
        path.write_text(DEEP, encoding="utf-8")
        read = S.read_pack if what == "pack" else S.read_profile
        with pytest.raises(ValueError, match=f"{name} is not a VoxFlow {what} \\(it is not valid JSON\\)"):
            read(str(path))
    pack = tmp_path / "huge.voxpack"
    pack.write_text(json.dumps({"format": "voxflow-pack", "version": 1, "kind": "dictionary", "created": HUGE,
                                "entries": [{"term": "Kubernetes"}]}), encoding="utf-8")
    assert S.read_pack(str(pack))["created"] is None and S.describe_pack(d.ds, str(pack))["count"] == 1
    backup = tmp_path / "huge.voxbackup"
    backup.write_text(json.dumps({
        "format": "voxflow-profile", "version": 1, "created": HUGE, "dictionary": [{"term": "Kubernetes"}],
        "stats": {"2026-01-05": {"words": HUGE, "sessions": 3}},
        "history": [{"text": "kept", "ts": HUGE, "duration_s": HUGE, "latency_ms": HUGE}]}), encoding="utf-8")
    assert S.describe_profile(str(backup))["created"] is None
    out = S.import_profile(d.ds, str(backup), "replace")
    assert out["counts"]["history"] == 1 and out["counts"]["stats"] == 1
    assert abs(d.history.snapshot()[0]["ts"] - time.time()) < 60, "an unusable time becomes the time of the import"
    assert d.stats.export_days()["2026-01-05"] == {"words": 0, "sessions": 3, "seconds": 0.0, "timed_words": 0,
                                                   "apps": {}}
    assert S._finite(1) == 1.0 and S._finite(float("nan")) is None and S._finite("1") is None
    assert S._finite(False) is None


# =========================================================================== DL-11: a long absence
def trio(tmp_path, folder, clock):
    a, b = pair(tmp_path, folder, clock)
    c = Device(tmp_path, "tablet", clock)
    c.engine.set_folder(str(folder))
    a.dictionary.add("Zanzibar")
    a.dictionary.add("Kubernetes")
    rounds(a, b, c)
    assert c.terms() == {"Zanzibar": [], "Kubernetes": []}
    a.dictionary.remove(a.term_id("Zanzibar"))
    rounds(a, b)                                           # the tablet is switched off
    return a, b, c


def test_a_device_away_longer_than_the_tombstone_window_does_not_bring_a_deleted_entry_back(tmp_path, folder, clock):
    a, b, c = trio(tmp_path, folder, clock)
    for _ in range(19):                                    # 190 days: longer than deletions used to be remembered
        clock.tick(10 * DAY)
        rounds(a, b, n=1)
    assert records(a)["zanzibar"].get("x") and records(b)["zanzibar"].get("x"), "kept: the tablet has not seen it"
    rounds(a, b, c)
    assert a.terms() == b.terms() == c.terms() == {"Kubernetes": []}, "deleted on the tablet too, not brought back"
    clock.tick(DAY)
    rounds(a, b, c)
    assert all("zanzibar" not in records(d) for d in (a, b, c)), "every device has written since: forgotten now"


def test_a_device_that_never_comes_back_is_not_waited_for_for_ever(tmp_path, folder, clock):
    a, b, c = trio(tmp_path, folder, clock)
    clock.tick((S.ABSENT_DAYS - 5) * DAY)
    rounds(a, b, n=1)
    assert "zanzibar" in records(a)
    clock.tick(10 * DAY)
    rounds(a, b, n=1)
    assert "zanzibar" not in records(a) and "zanzibar" not in records(b)


def test_without_another_device_a_deletion_is_forgotten_after_the_tombstone_window(tmp_path, folder, clock):
    d = Device(tmp_path, "alone", clock)
    d.engine.set_folder(str(folder))
    d.dictionary.add("Zanzibar")
    d.engine.sync()
    d.dictionary.remove(d.term_id("Zanzibar"))
    d.engine.sync()
    clock.tick((S.TOMBSTONE_DAYS - 1) * DAY)
    d.engine.sync()
    assert "zanzibar" in records(d)
    clock.tick(2 * DAY)
    d.engine.sync()
    assert "zanzibar" not in records(d)
