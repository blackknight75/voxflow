"""voxflow.sync: remaining corners of entry validation, profile building, the all-or-nothing import and
the sync engine (oversized / shapeless files, old deletions, rounds while sync is off)."""
from __future__ import annotations

import json

import pytest

from tests.test_sync import Clock, Device
from voxflow import sync as S

DAY = 86400.0


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def dev(tmp_path, clock):
    d = Device(tmp_path, "alpha", clock)
    d.dictionary.add("Kubernetes", ["cooper netties"])
    d.snippets.add("my email", "ada@example.com")
    return d


@pytest.fixture
def folder(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    return shared


def other_file(folder, name="voxflow-sync-beta-12345678.json", **doc):
    base = {"format": S.SYNC_FORMAT, "version": S.VERSION, "device": "beta-device", "name": "beta",
            "updated": 1_800_000_000.0, "records": {}}
    base.update(doc)
    path = folder / name
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


# ---------------------------------------------------------------- validation
def test_norm_rejects_unknown_kinds_and_style_keys_that_are_not_text():
    assert S._norm("styles", 5, "formal") is None
    assert S._norm("styles", None, "work") is None
    assert S._norm("history", "some-id", {"text": "hello"}) is None
    assert S._norm("styles", "tone:nonsense", "formal") is None
    assert S._norm("styles", "tone:email", "formal") == ("tone:email", "formal")


# ---------------------------------------------------------------- profiles
def test_profile_of_a_data_set_without_stats_has_history_but_no_stats(dev, clock):
    dev.history.add("first dictation", raw="first dictation", ms=80.0)
    ds = S.DataSet(dev.dictionary, dev.snippets, dev.styles, dev.history, lambda: dict(dev.settings),
                   dev._set_settings)                                   # no stats store
    profile = S.build_profile(ds, include_history=True, clock=clock)
    assert [e["text"] for e in profile["history"]] == ["first dictation"]
    assert "stats" not in profile
    assert "history" not in S.build_profile(ds, include_history=False, clock=clock)


def test_import_that_fails_and_cannot_be_rolled_back_still_reports_a_human_error(dev, caplog):
    def broken(items, replace=False):
        raise RuntimeError("history file is gone")

    dev.history.import_entries = broken                # fails during the import ...
    dev.history.rollback = broken                      # ... and the history cannot be put back either
    incoming = {"dictionary": {"pytorch": {"term": "PyTorch", "aliases": []}}}
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="could not be completed, so nothing was changed"):
            S._import(dev.ds, incoming, [{"text": "hello"}], None, "merge")
    assert "import failed; restoring the previous state" in caplog.text
    assert "restoring after a failed import also failed" in caplog.text
    assert "PyTorch" not in dev.terms(), "the stores that could be put back were put back"


# ---------------------------------------------------------------- engine: local stamping
def test_a_setting_this_version_no_longer_has_is_dropped_from_the_records_not_tombstoned(dev, folder, clock):
    dev.engine.set_folder(str(folder))
    dev.engine.sync()
    settings = dev.engine._state["records"]["settings"]
    assert "sounds_enabled" in settings
    del dev.settings["sounds_enabled"]                 # e.g. a downgrade to a version without this setting
    clock.tick()
    assert dev.engine.stamp_local() is True
    assert "sounds_enabled" not in dev.engine._state["records"]["settings"], "removed outright, no deletion record"
    assert dev.engine.stamp_local() is False, "nothing further to stamp"


# ---------------------------------------------------------------- engine: reading the folder
def test_an_oversized_file_in_the_folder_is_skipped_as_unreadable(dev, folder, monkeypatch):
    dev.engine.set_folder(str(folder))
    other_file(folder, records={"dictionary": {"pytorch": {"v": {"term": "PyTorch", "aliases": []},
                                                           "t": 1_700_000_000.0, "d": "beta-device"}}})
    monkeypatch.setattr(S, "_MAX_FILE_BYTES", 16)
    remote = dev.engine.read_remote()
    assert remote["docs"] == [] and remote["unreadable"] == 1 and remote["newer"] == 0
    monkeypatch.setattr(S, "_MAX_FILE_BYTES", 50 * 1024 * 1024)
    assert len(dev.engine.read_remote()["docs"]) == 1


def test_a_sync_file_without_a_records_table_contributes_nothing(dev, folder):
    dev.engine.set_folder(str(folder))
    other_file(folder, records=["not", "a", "table"])
    remote = dev.engine.read_remote()
    assert remote == {"docs": [], "unreadable": 0, "newer": 0, "own_present": False}
    status = dev.engine.sync()
    assert status["state"] == "idle" and status["unreadable"] == 0 and dev.terms() == {"Kubernetes": ["cooper netties"]}


# ---------------------------------------------------------------- engine: merging
def test_merge_and_write_do_nothing_while_sync_is_off(dev, folder):
    assert not dev.engine.enabled
    remote = {"docs": [{"device": "beta-device", "records": {"dictionary": {
        "pytorch": {"v": {"term": "PyTorch", "aliases": []}, "t": 1_700_000_000.0, "d": "beta-device"}}}}],
        "unreadable": 0, "newer": 0, "own_present": False}
    assert dev.engine.merge(remote) is None
    assert "PyTorch" not in dev.terms()
    dev.engine.write_own(("{}", "digest"))
    assert list(folder.iterdir()) == [] and dev.engine._state["written"] == ""
    assert dev.engine.status()["state"] == "off"


def test_a_deletion_older_than_the_tombstone_window_for_an_unknown_entry_is_not_recorded(dev, folder, clock):
    dev.engine.set_folder(str(folder))
    dev.engine.sync()
    now = clock.tick()
    old, recent = now - (S.TOMBSTONE_DAYS + 20) * DAY, now - 5 * DAY
    remote = {"docs": [{"device": "beta-device", "name": "beta", "updated": now, "records": {"dictionary": {
        "ancient ghost": {"x": 1, "t": old, "d": "beta-device"},
        "recent ghost": {"x": 1, "t": recent, "d": "beta-device"}}}}],
        "unreadable": 0, "newer": 0, "own_present": True}
    dev.engine.merge(remote)
    records = dev.engine._state["records"]["dictionary"]
    assert "ancient ghost" not in records
    assert records["recent ghost"] == {"x": 1, "t": recent, "d": "beta-device"}, "a recent deletion is remembered"
    assert dev.terms() == {"Kubernetes": ["cooper netties"]}


# ---------------------------------------------------------------- engine: status notes
def test_a_postponed_round_goes_back_to_idle(dev, folder):
    dev.engine.set_folder(str(folder))
    dev.engine.sync()
    dev.engine.note_syncing()
    assert dev.engine.status()["state"] == "syncing"
    dev.engine.note_idle()
    status = dev.engine.status()
    assert status["state"] == "idle" and status["error"] == ""


def test_a_file_written_while_the_folder_was_changed_does_not_count_for_the_new_folder(dev, folder, tmp_path,
                                                                                       monkeypatch):
    """Only a timing-dependent test reached the "folder changed meanwhile" branch of ``write_own``; it was missed in
    one full run of the 1.0 gate."""
    other = tmp_path / "other"
    other.mkdir()
    dev.engine.set_folder(str(folder))
    real = S.cfgmod._atomic_write

    def write_then_switch(path, text):
        real(path, text)
        dev.engine._state["folder"] = str(other)                # the user picked another folder during the write
    monkeypatch.setattr(S.cfgmod, "_atomic_write", write_then_switch)
    dev.engine.write_own(("{}", "digest-of-the-old-folder"))
    monkeypatch.setattr(S.cfgmod, "_atomic_write", real)
    assert dev.engine._state["written"] == "" and dev.engine.folder == str(other)
