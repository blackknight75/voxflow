"""Profiles (backups), shareable packs and the sync folder (``voxflow.sync``), on real stores in temp folders.

A ``Device`` is one VoxFlow profile: its own data folder, real stores, a settings dict with the real validation,
and a ``SyncEngine`` driven by a clock the test controls.
"""
from __future__ import annotations

import json
import os
import random
import shutil

import pytest

from voxflow import config as cfgmod
from voxflow import store
from voxflow import sync as S


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float = 10.0) -> float:
        self.t += seconds
        return self.t


class Device:
    def __init__(self, root, name: str, clock: Clock | None = None):
        self.dir = root / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock or Clock()
        self.dictionary = store.Dictionary(self.dir)
        self.snippets = store.Snippets(self.dir)
        self.styles = store.Styles(self.dir)
        self.history = store.History(self.dir)
        self.stats = store.Stats(self.dir)
        self.settings = cfgmod.defaults()
        self.refuse: set[str] = set()                     # settings this device cannot apply
        self.ds = S.DataSet(self.dictionary, self.snippets, self.styles, self.history,
                            lambda: dict(self.settings), self._set_settings, self.stats)
        self.engine = S.SyncEngine(self.ds, self.dir / "sync_state.json", clock=self.clock, name=name)

    def _set_settings(self, values: dict) -> list[str]:
        failed = []
        for key, value in values.items():
            if key in self.refuse:
                failed.append(key)
                continue
            try:
                self.settings[key] = cfgmod.Config.validate(key, value)
            except ValueError:
                failed.append(key)
        return failed

    def reopen(self) -> "Device":
        """The same profile after a restart of the app."""
        twin = Device.__new__(Device)
        twin.__dict__.update(self.__dict__)
        twin.dictionary, twin.snippets = store.Dictionary(self.dir), store.Snippets(self.dir)
        twin.styles, twin.history, twin.stats = store.Styles(self.dir), store.History(self.dir), store.Stats(self.dir)
        twin.ds = S.DataSet(twin.dictionary, twin.snippets, twin.styles, twin.history,
                            lambda: dict(twin.settings), twin._set_settings, twin.stats)
        twin.engine = S.SyncEngine(twin.ds, self.dir / "sync_state.json", clock=self.clock)
        return twin

    # ---- shortcuts
    def terms(self) -> dict:
        return {e["term"]: sorted(e["aliases"]) for e in self.dictionary.list()}

    def snips(self) -> dict:
        return {e["trigger"]: e["text"] for e in self.snippets.list()}

    def portable(self) -> dict:
        return S.snapshot(self.ds)

    def term_id(self, term: str) -> str:
        return next(e["id"] for e in self.dictionary.list() if e["term"] == term)

    def snip_id(self, trigger: str) -> str:
        return next(e["id"] for e in self.snippets.list() if e["trigger"] == trigger)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def dev(tmp_path, clock):
    d = Device(tmp_path, "alpha", clock)
    d.dictionary.add("Kubernetes", ["cooper netties"])
    d.dictionary.add("VoxFlow")
    d.snippets.add("my email", "ada@example.com")
    d.snippets.add("sign off", "Best regards,\nAda")
    d.styles.set_tone("email", "very_casual")
    d.styles.set_app_category("obsidian.exe", "other")
    d.history.add("first dictation", raw="first dictation", ms=80.0)
    d.history.add("second dictation", raw="second dictation", ms=90.0)
    d.stats.record(12, 5.0, "notepad.exe")
    d.settings["sounds_enabled"] = False
    d.settings["mic_device"] = "Studio Mic"
    return d


def everything(d: Device) -> dict:
    return {"snap": S.snapshot(d.ds, every_setting=True), "history": [e["text"] for e in d.history.snapshot()],
            "stats": d.stats.export_days()}


# =========================================================================== backups: export / inspect
def test_export_writes_one_json_file_with_every_part_and_reports_counts(dev, tmp_path):
    out = S.export_profile(dev.ds, str(tmp_path / "mine"), device_id=dev.engine.device_id)
    assert out["path"].endswith(".voxbackup") and os.path.isfile(out["path"])
    assert out["includes"] == ["settings", "dictionary", "snippets", "styles", "stats", "history"]
    assert out["counts"]["dictionary"] == 2 and out["counts"]["snippets"] == 2 and out["counts"]["styles"] == 1
    assert out["counts"]["history"] == 2 and out["counts"]["stats"] == 1 and out["counts"]["settings"] > 20
    data = json.loads(open(out["path"], encoding="utf-8").read())
    assert data["format"] == "voxflow-profile" and data["version"] == 1
    assert data["settings"]["mic_device"] == "Studio Mic"
    for never in ("config_version", "ui_state", "overlay_position", "onboarding_done"):
        assert never not in data["settings"]


def test_export_without_history_leaves_out_history_and_statistics(dev, tmp_path):
    out = S.export_profile(dev.ds, str(tmp_path / "small.voxbackup"), include_history=False)
    assert out["includes"] == ["settings", "dictionary", "snippets", "styles"]
    assert out["counts"]["history"] == 0 and out["counts"]["stats"] == 0
    text = open(out["path"], encoding="utf-8").read()
    assert "first dictation" not in text


def test_export_keeps_an_explicit_extension_and_reports_an_unwritable_target(dev, tmp_path):
    out = S.export_profile(dev.ds, str(tmp_path / "backup.json"))
    assert out["path"].endswith("backup.json")
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="Couldn't write"):
        S.export_profile(dev.ds, str(blocker / "sub" / "x.voxbackup"))


def test_inspect_says_what_is_inside_and_whether_it_is_from_this_device(dev, tmp_path):
    path = S.export_profile(dev.ds, str(tmp_path / "b"), device_id=dev.engine.device_id)["path"]
    info = S.describe_profile(path, dev.engine.device_id)
    assert info["same_device"] is True and info["skipped"] == 0 and info["version"]
    assert info["counts"]["dictionary"] == 2 and info["counts"]["history"] == 2 and info["counts"]["styles"] == 1
    assert info["created"] and info["device"]
    assert S.describe_profile(path, "someone-else")["same_device"] is False
    assert S.describe_profile(path, "")["same_device"] is False
    before = everything(dev)
    S.describe_profile(path)
    assert everything(dev) == before, "inspecting changes nothing"


# =========================================================================== backups: import
def test_merge_into_an_empty_profile_brings_the_data_but_not_the_settings(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"), device_id=dev.engine.device_id)["path"]
    new = Device(tmp_path, "beta", clock)
    out = S.import_profile(new.ds, path, "merge", device_id=new.engine.device_id)
    assert new.terms() == dev.terms() and new.snips() == dev.snips()
    assert new.styles.user_apps() == {"obsidian.exe": "other"}
    assert [e["text"] for e in new.history.snapshot()] == [e["text"] for e in dev.history.snapshot()]
    assert new.stats.export_days() == dev.stats.export_days()
    assert new.settings == cfgmod.defaults(), "merge never changes settings"
    assert new.styles.tones()["email"] == "formal", "merge never changes a tone"
    assert out["mode"] == "merge" and out["counts"]["dictionary"] == 2 and out["counts"]["snippets"] == 2
    assert out["counts"]["history"] == 2 and out["counts"]["settings"] == 0 and out["skipped"] == 0
    assert out["details"]["dictionary"] == {"added": 2, "updated": 0, "removed": 0}


def test_merge_changes_nothing_that_exists_and_adds_only_what_is_missing(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("kubernetes", ["kube"])                 # same term, other spelling + other alias
    other.dictionary.add("Zephyr")
    other.snippets.add("my email", "grace@example.com")          # same trigger, other text
    other.styles.set_app_category("obsidian.exe", "work")
    S.import_profile(other.ds, path, "merge")
    assert other.terms() == {"kubernetes": ["cooper netties", "kube"], "VoxFlow": [], "Zephyr": []}
    assert other.snips() == {"my email": "grace@example.com", "sign off": "Best regards,\nAda"}
    assert other.styles.user_apps() == {"obsidian.exe": "work"}


def test_merging_the_same_backup_twice_adds_nothing_the_second_time(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    S.import_profile(other.ds, path, "merge")
    before = everything(other)
    again = S.import_profile(other.ds, path, "merge")
    assert everything(other) == before
    assert again["counts"]["dictionary"] == 0 and again["counts"]["snippets"] == 0 and again["counts"]["history"] == 0


def test_replace_makes_the_profile_equal_to_the_file(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"), device_id=dev.engine.device_id)["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("Zephyr")
    other.snippets.add("old one", "gone soon")
    other.styles.set_app_category("slack.exe", "work")
    other.history.add("their own dictation")
    out = S.import_profile(other.ds, path, "replace", device_id=other.engine.device_id)
    assert other.terms() == dev.terms() and other.snips() == dev.snips()
    assert other.styles.user_apps() == {"obsidian.exe": "other"} and other.styles.tones() == dev.styles.tones()
    assert [e["text"] for e in other.history.snapshot()] == [e["text"] for e in dev.history.snapshot()]
    assert other.settings["sounds_enabled"] is False, "portable settings come from the file"
    assert other.settings["mic_device"] == cfgmod.defaults()["mic_device"], \
        "a setting that describes the other computer is not taken over"
    assert out["details"]["dictionary"]["removed"] == 1 and out["details"]["snippets"]["removed"] == 1


def test_replace_on_the_same_device_restores_the_computer_settings_too(dev, tmp_path):
    path = S.export_profile(dev.ds, str(tmp_path / "b"), device_id=dev.engine.device_id)["path"]
    dev.settings["mic_device"] = ""
    dev.settings["sounds_enabled"] = True
    S.import_profile(dev.ds, path, "replace", device_id=dev.engine.device_id)
    assert dev.settings["mic_device"] == "Studio Mic" and dev.settings["sounds_enabled"] is False


def test_parts_limit_what_is_imported(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    out = S.import_profile(other.ds, path, "replace", parts=["dictionary"])
    assert other.terms() == dev.terms()
    assert other.snips() == {} and other.history.count() == 0 and other.settings == cfgmod.defaults()
    assert out["counts"]["snippets"] == 0
    with pytest.raises(ValueError, match="not a part of a backup"):
        S.import_profile(other.ds, path, "merge", parts=["dictionary", "passwords"])
    small = S.export_profile(dev.ds, str(tmp_path / "nohist"), include_history=False)["path"]
    with pytest.raises(ValueError, match="does not contain any of the parts"):
        S.import_profile(other.ds, small, "merge", parts=["history"])


def test_an_unknown_mode_is_refused_before_anything_changes(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    with pytest.raises(ValueError, match="merge.*replace"):
        S.import_profile(other.ds, path, "overwrite")
    assert other.terms() == {}


def test_a_setting_that_cannot_be_applied_is_reported_not_fatal(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.refuse = {"sounds_enabled"}
    out = S.import_profile(other.ds, path, "replace")
    assert out["settings_skipped"] == ["sounds_enabled"]
    assert other.settings["sounds_enabled"] is True and other.terms() == dev.terms()


# ---- files that are not (good) backups
def _write(path, payload) -> str:
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("payload, message", [
    ("{ this is not json", "not valid JSON"),
    ("", "not valid JSON"),
    ([1, 2, 3], "is not a VoxFlow backup"),
    ({"hello": "world"}, "is not a VoxFlow backup"),
    ({"format": "voxflow-pack", "version": 1, "kind": "dictionary", "entries": ["a"]}, "shared pack, not a backup"),
    ({"format": "voxflow-profile"}, "has no version"),
    ({"format": "voxflow-profile", "version": "1"}, "has no version"),
    ({"format": "voxflow-profile", "version": True}, "has no version"),
    ({"format": "voxflow-profile", "version": 0}, "has no version"),
    ({"format": "voxflow-profile", "version": 2, "dictionary": []}, "newer version of VoxFlow"),
    ({"format": "voxflow-profile", "version": 1}, "it is empty"),
    ({"format": "voxflow-profile", "version": 1, "dictionary": "nope", "settings": 7}, "it is empty"),
])
def test_bad_backup_files_are_refused_with_a_sentence_and_change_nothing(dev, tmp_path, payload, message):
    path = _write(tmp_path / "bad.voxbackup", payload)
    before = everything(dev)
    for call in (lambda: S.describe_profile(path), lambda: S.import_profile(dev.ds, path, "replace")):
        with pytest.raises(ValueError, match=message) as err:
            call()
        assert "Traceback" not in str(err.value) and "bad.voxbackup" in str(err.value)
    assert everything(dev) == before


def test_missing_unreadable_and_oversized_files(dev, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="was not found"):
        S.import_profile(dev.ds, str(tmp_path / "nope.voxbackup"))
    binary = tmp_path / "bin.voxbackup"
    binary.write_bytes(b"\xff\xfe\x00\x81" * 10)
    with pytest.raises(ValueError, match="Couldn't read|not valid JSON"):
        S.import_profile(dev.ds, str(binary))
    with pytest.raises(ValueError):
        S.import_profile(dev.ds, str(tmp_path))                 # a folder
    monkeypatch.setattr(S, "_MAX_FILE_BYTES", 10)
    big = S.export_profile.__wrapped__ if hasattr(S.export_profile, "__wrapped__") else None
    assert big is None
    path = _write(tmp_path / "big.voxbackup", {"format": "voxflow-profile", "version": 1, "dictionary": ["a" * 50]})
    with pytest.raises(ValueError, match="too large"):
        S.import_profile(dev.ds, path)


def test_a_backup_with_a_utf8_bom_is_read(dev, tmp_path, clock):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    raw = open(path, "rb").read()
    open(path, "wb").write(b"\xef\xbb\xbf" + raw)
    other = Device(tmp_path, "beta", clock)
    S.import_profile(other.ds, path, "merge")
    assert other.terms() == dev.terms()


def test_unusable_entries_are_dropped_and_counted_the_rest_is_imported(tmp_path, clock):
    path = _write(tmp_path / "mixed.voxbackup", {
        "format": "voxflow-profile", "version": 1,
        "settings": {"sounds_enabled": False, "sound_volume": "loud", "no_such_setting": 1, "config_version": 99,
                     "history_limit": float("inf") if False else 10 ** 12},
        "dictionary": [{"term": "Good"}, {"term": ""}, {"term": 5}, 17, None, {"term": "Alias", "aliases": "x"},
                       "PlainString", {"term": "good"}],
        "snippets": [{"trigger": "ok", "text": "fine"}, {"trigger": "", "text": "x"}, {"trigger": "t"},
                     {"trigger": "empty", "text": "   "}, "nope"],
        "styles": {"tones": {"email": "casual", "email2": "formal", "work": "shouting"},
                   "apps": {"slack.exe": "work", "": "work", "x.exe": "nonsense"}},
        "history": [{"text": "kept"}, {"text": 5}, "nope", {"raw": "no text"}],
        "stats": {"2026-11-01": {"words": 5, "seconds": 2.0}, "garbage": 4},
    })
    info = S.describe_profile(path)
    assert info["counts"]["dictionary"] == 2 and info["counts"]["snippets"] == 1 and info["counts"]["history"] == 1
    assert info["skipped"] >= 12
    d = Device(tmp_path, "gamma", clock)
    out = S.import_profile(d.ds, path, "replace")
    assert set(d.terms()) == {"Good", "PlainString"}
    assert d.snips() == {"ok": "fine"}
    assert d.styles.tones()["email"] == "casual" and d.styles.user_apps() == {"slack.exe": "work"}
    assert d.settings["sounds_enabled"] is False
    assert d.settings["sound_volume"] == cfgmod.defaults()["sound_volume"]
    assert [e["text"] for e in d.history.snapshot()] == ["kept"]
    assert out["skipped"] == info["skipped"]


def test_a_backup_from_an_older_build_with_fewer_sections_and_settings_imports(tmp_path, clock):
    """Version 1 files written before stats / styles / newer settings existed must still open."""
    path = _write(tmp_path / "old.voxbackup", {
        "format": "voxflow-profile", "version": 1, "app_version": "0.1.0", "created": 1_700_000_000,
        "settings": {"hotkey": "ctrl+win", "cleanup_enabled": False, "llm_model": "removed-in-0.2", "paste_delay": 0.3},
        "dictionary": ["Parakeet", {"term": "Qwen"}],
    })
    info = S.describe_profile(path)
    assert info["version"] == "0.1.0" and info["includes"] == ["settings", "dictionary"] and info["skipped"] == 2
    assert info["counts"]["history"] == 0 and info["counts"]["stats"] == 0
    d = Device(tmp_path, "gamma", clock)
    d.snippets.add("keep", "me")
    d.history.add("still here")
    S.import_profile(d.ds, path, "replace")
    assert set(d.terms()) == {"Parakeet", "Qwen"} and d.settings["cleanup_enabled"] is False
    assert d.snips() == {"keep": "me"} and d.history.count() == 1, "sections the file lacks are left alone"


def test_created_that_is_not_a_finite_number_is_ignored(tmp_path):
    for created in ("yesterday", True, None, 1e400, float("nan")):
        text = json.dumps({"format": "voxflow-profile", "version": 1, "created": 0, "dictionary": ["A"]})
        text = text.replace('"created": 0', '"created": ' + json.dumps(created))
        path = _write(tmp_path / "c.voxbackup", text)
        assert S.describe_profile(path)["created"] is None


def test_a_failure_half_way_puts_everything_back(dev, tmp_path, clock, monkeypatch):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("Mine")
    other.snippets.add("mine", "text")
    other.history.add("my history")
    before = everything(other)
    real = other.styles.apply_changes
    calls = {"n": 0}

    def explode(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(other.styles, "apply_changes", explode)
    with pytest.raises(ValueError, match="could not be completed, so nothing was changed"):
        S.import_profile(other.ds, path, "replace")
    assert everything(other) == before


def test_a_failure_while_importing_history_puts_the_stores_back(dev, tmp_path, clock, monkeypatch):
    path = S.export_profile(dev.ds, str(tmp_path / "b"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("Mine")
    other.history.add("my history")
    before = everything(other)
    real = other.history.import_entries
    calls = {"n": 0}

    def explode(items, replace=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("history is locked")
        return real(items, replace=replace)

    monkeypatch.setattr(other.history, "import_entries", explode)
    with pytest.raises(ValueError, match="nothing was changed"):
        S.import_profile(other.ds, path, "replace")
    assert everything(other) == before


def test_round_trip_export_wipe_replace_restores_the_profile_exactly(dev, tmp_path):
    before = everything(dev)
    path = S.export_profile(dev.ds, str(tmp_path / "b"), device_id=dev.engine.device_id)["path"]
    for e in dev.dictionary.list():
        dev.dictionary.remove(e["id"])
    for e in dev.snippets.list():
        dev.snippets.remove(e["id"])
    dev.history.clear()
    dev.stats.reset()
    dev.styles.remove_app_category("obsidian.exe")
    dev.styles.set_tone("email", "casual")
    dev.settings.update(cfgmod.defaults())
    S.import_profile(dev.ds, path, "replace", device_id=dev.engine.device_id)
    assert everything(dev) == before


# =========================================================================== packs
def test_export_a_dictionary_pack_and_a_snippet_pack(dev, tmp_path):
    out = S.export_pack(dev.ds, "dictionary", str(tmp_path / "team terms"), name="  Team   terms ",
                        description="Product names")
    assert out["path"].endswith(".voxpack") and out["count"] == 2 and out["name"] == "Team terms"
    data = json.loads(open(out["path"], encoding="utf-8").read())
    assert data["format"] == "voxflow-pack" and data["kind"] == "dictionary"
    assert all("id" not in e for e in data["entries"]), "local ids do not travel"
    assert "first dictation" not in json.dumps(data) and "settings" not in data
    snip = S.export_pack(dev.ds, "snippets", str(tmp_path / "s.voxpack"))
    assert snip["count"] == 2 and snip["name"] == "s", "the file name is the default pack name"


def test_export_a_selection_and_the_empty_cases(dev, tmp_path, clock):
    one = S.export_pack(dev.ds, "dictionary", str(tmp_path / "one"), ids=[dev.term_id("VoxFlow")])
    assert one["count"] == 1
    with pytest.raises(ValueError, match="Pick at least one entry to share"):
        S.export_pack(dev.ds, "dictionary", str(tmp_path / "none"), ids=[])
    with pytest.raises(ValueError, match="Pick at least one entry to share"):
        S.export_pack(dev.ds, "snippets", str(tmp_path / "none"), ids=["no-such-id"])
    empty = Device(tmp_path, "empty", clock)
    with pytest.raises(ValueError, match="There are no terms to share yet"):
        S.export_pack(empty.ds, "dictionary", str(tmp_path / "e"))
    with pytest.raises(ValueError, match="There are no snippets to share yet"):
        S.export_pack(empty.ds, "snippets", str(tmp_path / "e"))
    with pytest.raises(ValueError, match="either dictionary terms or snippets"):
        S.export_pack(dev.ds, "history", str(tmp_path / "h"))
    assert not (tmp_path / "none.voxpack").exists() and not (tmp_path / "e.voxpack").exists()


def test_inspect_a_pack_counts_what_is_new_here(dev, tmp_path, clock):
    path = S.export_pack(dev.ds, "dictionary", str(tmp_path / "p"), name="Terms", description="d")["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("VOXFLOW")
    info = S.describe_pack(other.ds, path)
    assert info == {"kind": "dictionary", "name": "Terms", "description": "d", "count": 2, "created": info["created"],
                    "new": 1, "existing": 1, "skipped": 0}
    assert other.terms() == {"VOXFLOW": []}


def test_import_pack_merge_updates_from_the_pack_and_keeps_own_aliases(dev, tmp_path, clock):
    path = S.export_pack(dev.ds, "dictionary", str(tmp_path / "p"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("kubernetes", ["kube"])
    other.dictionary.add("Zephyr")
    out = S.import_pack(other.ds, path, "merge")
    assert other.terms() == {"Kubernetes": ["cooper netties", "kube"], "VoxFlow": [], "Zephyr": []}
    assert out["kind"] == "dictionary" and out["added"] == 1 and out["updated"] == 1 and out["removed"] == 0
    spath = S.export_pack(dev.ds, "snippets", str(tmp_path / "s"))["path"]
    other.snippets.add("my email", "old@example.com")
    out = S.import_pack(other.ds, spath, "merge")
    assert other.snips() == dev.snips() and out["updated"] == 1 and out["added"] == 1


def test_import_pack_replace_makes_the_pack_the_whole_list_and_touches_nothing_else(dev, tmp_path, clock):
    path = S.export_pack(dev.ds, "dictionary", str(tmp_path / "p"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.dictionary.add("Zephyr")
    other.snippets.add("keep", "me")
    out = S.import_pack(other.ds, path, "replace")
    assert other.terms() == dev.terms() and out["removed"] == 1
    assert other.snips() == {"keep": "me"} and other.settings == cfgmod.defaults()
    with pytest.raises(ValueError, match="merge.*replace"):
        S.import_pack(other.ds, path, "append")


def test_plain_exports_and_bare_lists_are_packs_too(dev, tmp_path, clock):
    dict_json, snip_json = tmp_path / "dict.json", tmp_path / "snips.json"
    dev.dictionary.export(str(dict_json))
    dev.snippets.export(str(snip_json))
    other = Device(tmp_path, "beta", clock)
    assert S.describe_pack(other.ds, str(dict_json))["kind"] == "dictionary"
    assert S.import_pack(other.ds, str(dict_json))["added"] == 2
    assert S.import_pack(other.ds, str(snip_json))["kind"] == "snippets"
    assert other.terms() == dev.terms() and other.snips() == dev.snips()
    bare = _write(tmp_path / "bare.json", ["Alpha", "Beta", {"term": "Gamma", "aliases": ["gama"]}])
    out = S.import_pack(other.ds, bare)
    assert out["added"] == 3 and out["name"] == "bare" and other.terms()["Gamma"] == ["gama"]
    bare_snips = _write(tmp_path / "bs.json", [{"trigger": "hi", "text": "Hello there"}])
    assert S.import_pack(other.ds, bare_snips)["kind"] == "snippets"


@pytest.mark.parametrize("payload, message", [
    ("not json at all", "not valid JSON"),
    ({"format": "voxflow-profile", "version": 1, "dictionary": ["a"]}, "whole backup, not a pack"),
    ({"format": "voxflow-pack", "version": 5, "kind": "dictionary", "entries": ["a"]}, "newer version of VoxFlow"),
    ({"format": "voxflow-pack", "kind": "dictionary", "entries": ["a"]}, "has no version"),
    ({"format": "voxflow-pack", "version": 1, "kind": "dictionary"}, "is not a VoxFlow pack"),
    ({"format": "voxflow-pack", "version": 1, "kind": "dictionary", "entries": {"a": 1}}, "is not a VoxFlow pack"),
    ({"something": "else"}, "is not a VoxFlow pack"),
    (42, "is not a VoxFlow pack"),
    ({"format": "voxflow-pack", "version": 1, "kind": "dictionary", "entries": []}, "does not contain any usable terms"),
    ({"format": "voxflow-pack", "version": 1, "kind": "snippets", "entries": [{"trigger": "", "text": ""}, 5]},
     "does not contain any usable snippets"),
])
def test_bad_pack_files_are_refused_with_a_sentence_and_change_nothing(dev, tmp_path, payload, message):
    path = _write(tmp_path / "bad.voxpack", payload)
    before = everything(dev)
    for call in (lambda: S.describe_pack(dev.ds, path), lambda: S.import_pack(dev.ds, path, "replace")):
        with pytest.raises(ValueError, match=message):
            call()
    assert everything(dev) == before


def test_a_pack_with_some_bad_rows_imports_the_good_ones_and_counts_the_rest(tmp_path, clock):
    path = _write(tmp_path / "p.voxpack", {"format": "voxflow-pack", "version": 1, "kind": "snippets", "name": "N",
                                          "entries": [{"trigger": "a", "text": "A"}, {"trigger": "A", "text": "dup"},
                                                      {"trigger": "b"}, None, {"trigger": "c", "text": "C"}]})
    d = Device(tmp_path, "gamma", clock)
    info = S.describe_pack(d.ds, path)
    assert info["count"] == 2 and info["skipped"] == 2
    assert S.import_pack(d.ds, path)["skipped"] == 2 and d.snips() == {"a": "A", "c": "C"}


def test_a_failed_pack_import_puts_the_list_back(dev, tmp_path, clock, monkeypatch):
    path = S.export_pack(dev.ds, "snippets", str(tmp_path / "p"))["path"]
    other = Device(tmp_path, "beta", clock)
    other.snippets.add("mine", "text")
    monkeypatch.setattr(other.snippets, "apply_changes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    with pytest.raises(ValueError, match="nothing was changed"):
        S.import_pack(other.ds, path, "replace")
    assert other.snips() == {"mine": "text"}


# =========================================================================== sync folder: basics
@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "cloud"
    f.mkdir()
    return f


def pair(tmp_path, folder, clock):
    a, b = Device(tmp_path, "desk", clock), Device(tmp_path, "laptop", clock)
    a.engine.set_folder(str(folder))
    b.engine.set_folder(str(folder))
    return a, b


def rounds(*devices, n: int = 3):
    for _ in range(n):
        for d in devices:
            d.clock.tick(1)
            st = d.engine.sync()
            assert st["state"] == "idle", st
    return devices


def sync_files(folder) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith("voxflow-sync-"))


def test_sync_is_off_until_a_folder_is_chosen(tmp_path, clock):
    d = Device(tmp_path, "solo", clock)
    st = d.engine.status()
    assert st["state"] == "off" and st["message"] == "Sync is off." and st["enabled"] is False
    assert st["folder"] == "" and st["last_sync"] is None and st["devices"] == [] and st["error"] == ""
    assert d.engine.sync()["state"] == "off" and d.engine.stamp_local() is False
    with pytest.raises(S.SyncUnavailable):
        d.engine.read_remote()


def test_choosing_a_folder_validates_it(tmp_path, clock, folder):
    d = Device(tmp_path, "solo", clock)
    with pytest.raises(ValueError, match="That folder doesn't exist"):
        d.engine.set_folder(str(tmp_path / "missing"))
    afile = tmp_path / "file.txt"
    afile.write_text("x")
    with pytest.raises(ValueError, match="That folder doesn't exist"):
        d.engine.set_folder(str(afile))
    with pytest.raises(ValueError, match="other than VoxFlow's own data folder"):
        d.engine.set_folder(str(d.dir))
    with pytest.raises(ValueError, match="must be a folder path"):
        d.engine.set_folder(42)
    assert d.engine.enabled is False
    d.engine.set_folder(f"  {folder}  ")
    assert d.engine.enabled and os.path.samefile(d.engine.folder, folder)
    assert not [p for p in folder.iterdir()], "the write test leaves nothing behind"
    assert d.engine.status()["message"] == "Waiting for the first sync."


def test_a_folder_that_cannot_be_written_is_refused(tmp_path, clock, folder, monkeypatch):
    from pathlib import Path
    d = Device(tmp_path, "solo", clock)
    real = Path.write_text

    def deny(self, *a, **k):
        if ".voxflow-write-test-" in self.name:
            raise PermissionError("read-only share")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", deny)
    with pytest.raises(ValueError, match="can't save files in that folder"):
        d.engine.set_folder(str(folder))
    assert d.engine.enabled is False


def test_first_round_writes_exactly_one_file_named_after_the_device(tmp_path, clock, folder):
    d = Device(tmp_path, "My Desk PC!", clock)
    d.dictionary.add("Kubernetes")
    d.engine.set_folder(str(folder))
    st = d.engine.sync()
    assert st["state"] == "idle" and st["message"] == "Up to date. No other device has used this folder yet."
    assert st["last_sync"] == clock.t
    files = sync_files(folder)
    assert files == [d.engine.file_name] and files[0].startswith("voxflow-sync-My-Desk-PC-")
    assert [p.name for p in folder.iterdir()] == files, "no temp files left in the folder"
    doc = json.loads((folder / files[0]).read_text(encoding="utf-8"))
    assert doc["format"] == "voxflow-sync" and doc["device"] == d.engine.device_id
    assert set(doc["records"]) == {"settings", "dictionary", "snippets", "styles"}
    assert "kubernetes" in doc["records"]["dictionary"]


def test_nothing_private_or_per_computer_is_written_to_the_folder(tmp_path, clock, folder):
    d = Device(tmp_path, "desk", clock)
    d.history.add("a very private dictation")
    d.stats.record(5, 2.0, "secretapp.exe")
    d.dictionary.suggest("Suggested", source="cleanup")
    d.settings["mic_device"] = "Studio Mic"
    d.engine.set_folder(str(folder))
    d.engine.sync()
    text = (folder / d.engine.file_name).read_text(encoding="utf-8")
    for secret in ("a very private dictation", "secretapp", "Suggested", "Studio Mic"):
        assert secret not in text
    doc = json.loads(text)
    assert not set(doc["records"]["settings"]) & set(S.LOCAL_SETTINGS)
    assert set(doc["records"]["settings"]) == set(S.portable_settings())


def test_an_idle_round_does_not_rewrite_the_file(tmp_path, clock, folder):
    d = Device(tmp_path, "desk", clock)
    d.dictionary.add("Kubernetes")
    d.engine.set_folder(str(folder))
    d.engine.sync()
    target = folder / d.engine.file_name
    stamp = target.stat().st_mtime_ns
    content = target.read_bytes()
    for _ in range(3):
        clock.tick(30)
        assert d.engine.sync()["last_sync"] == clock.t
    assert target.read_bytes() == content and target.stat().st_mtime_ns == stamp
    target.unlink()                                             # someone tidied the folder: the file comes back
    clock.tick(30)
    d.engine.sync()
    assert target.exists()


def test_two_profiles_exchange_everything_portable(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes", ["cooper netties"])
    a.snippets.add("my email", "ada@example.com")
    a.styles.set_tone("email", "very_casual")
    a.styles.set_app_category("obsidian.exe", "other")
    a.settings["sounds_enabled"] = False
    a.settings["mic_device"] = "Desk Mic"
    a.history.add("desk only")
    rounds(a, b)
    assert b.terms() == {"Kubernetes": ["cooper netties"]} and b.snips() == {"my email": "ada@example.com"}
    assert b.styles.tones()["email"] == "very_casual" and b.styles.user_apps() == {"obsidian.exe": "other"}
    assert b.settings["sounds_enabled"] is False
    assert b.settings["mic_device"] == cfgmod.defaults()["mic_device"] and b.history.count() == 0
    b.dictionary.add("Parakeet")
    b.snippets.update(b.snip_id("my email"), text="grace@example.com")
    b.settings["language"] = "de"
    rounds(a, b)
    assert a.terms() == b.terms() == {"Kubernetes": ["cooper netties"], "Parakeet": []}
    assert a.snips() == {"my email": "grace@example.com"} and a.settings["language"] == "de"
    assert a.portable() == b.portable()
    assert len(sync_files(folder)) == 2
    st = a.engine.status()
    assert st["message"] == "Up to date with 1 other device." and st["devices"][0]["name"] == "laptop"
    assert st["devices"][0]["id"] == b.engine.device_id and st["devices"][0]["updated"] > 0


def test_three_devices_and_the_plural_message(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    c = Device(tmp_path, "tablet", clock)
    c.engine.set_folder(str(folder))
    a.dictionary.add("One")
    b.dictionary.add("Two")
    c.dictionary.add("Three")
    rounds(a, b, c)
    assert a.terms() == b.terms() == c.terms() == {"One": [], "Two": [], "Three": []}
    assert a.engine.status()["message"] == "Up to date with 2 other devices."


# =========================================================================== sync folder: conflicts
def test_the_newest_change_wins_on_both_sides(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.snippets.add("sig", "v0")
    rounds(a, b)
    a.snippets.update(a.snip_id("sig"), text="from the desk")
    clock.tick(5)
    a.engine.stamp_local()
    clock.tick(60)
    b.snippets.update(b.snip_id("sig"), text="from the laptop, later")
    b.engine.stamp_local()
    rounds(a, b)
    assert a.snips() == b.snips() == {"sig": "from the laptop, later"}


def test_the_change_made_first_loses_even_if_that_device_syncs_last(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    b.dictionary.update(b.term_id("Kubernetes"), aliases=["early edit"])
    b.engine.stamp_local()                                      # noticed at t0 (e.g. while the folder was offline)
    clock.tick(3600)
    a.dictionary.update(a.term_id("Kubernetes"), aliases=["late edit"])
    rounds(a)                                                   # a syncs first ...
    rounds(b, a, b)                                             # ... b afterwards, with its older change
    assert a.terms() == b.terms() == {"Kubernetes": ["late edit"]}


def test_a_tie_is_broken_the_same_way_on_every_device(tmp_path, folder):
    ca, cb = Clock(), Clock()
    a, b = Device(tmp_path, "desk", ca), Device(tmp_path, "laptop", cb)
    a.engine.set_folder(str(folder))
    b.engine.set_folder(str(folder))
    a.engine.sync()
    b.engine.sync()
    ca.tick(100)
    cb.tick(100)                                                # same instant on both
    a.snippets.add("sig", "desk text")
    b.snippets.add("sig", "laptop text")
    a.engine.stamp_local()
    b.engine.stamp_local()
    for _ in range(3):
        a.engine.sync()
        b.engine.sync()
    winner = "desk text" if a.engine.device_id > b.engine.device_id else "laptop text"
    assert a.snips() == b.snips() == {"sig": winner}


def test_different_entries_edited_at_the_same_time_both_survive(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("One")
    a.dictionary.add("Two")
    rounds(a, b)
    a.dictionary.update(a.term_id("One"), aliases=["wun"])
    b.dictionary.update(b.term_id("Two"), aliases=["too"])
    b.snippets.add("new", "snippet")
    rounds(a, b)
    assert a.terms() == b.terms() == {"One": ["wun"], "Two": ["too"]} and a.snips() == {"new": "snippet"}


def test_a_term_respelled_with_other_capitals_is_one_entry_everywhere(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("github")
    rounds(a, b)
    clock.tick(10)
    b.dictionary.update(b.term_id("github"), term="GitHub")
    rounds(a, b)
    assert a.terms() == b.terms() == {"GitHub": []}


# =========================================================================== sync folder: deletions
def test_a_deletion_travels_as_a_tombstone(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    a.snippets.add("sig", "x")
    a.styles.set_app_category("obsidian.exe", "other")
    rounds(a, b)
    clock.tick(10)
    a.dictionary.remove(a.term_id("Kubernetes"))
    a.snippets.remove(a.snip_id("sig"))
    a.styles.remove_app_category("obsidian.exe")
    rounds(a, b)
    assert b.terms() == {} and b.snips() == {} and b.styles.user_apps() == {}
    doc = json.loads((folder / a.engine.file_name).read_text(encoding="utf-8"))
    assert doc["records"]["dictionary"]["kubernetes"].get("x") == 1
    rounds(a, b)
    assert a.terms() == {} and b.terms() == {}, "a deleted entry does not come back by itself"


def test_an_edit_made_after_a_deletion_brings_the_entry_back(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.snippets.add("sig", "v1")
    rounds(a, b)
    clock.tick(10)
    a.snippets.remove(a.snip_id("sig"))
    a.engine.stamp_local()
    clock.tick(100)
    b.snippets.update(b.snip_id("sig"), text="v2, edited later")
    rounds(b, a)
    assert a.snips() == b.snips() == {"sig": "v2, edited later"}


def test_a_deletion_made_after_an_edit_wins(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.snippets.add("sig", "v1")
    rounds(a, b)
    b.snippets.update(b.snip_id("sig"), text="v2")
    b.engine.stamp_local()
    clock.tick(100)
    a.snippets.remove(a.snip_id("sig"))
    rounds(a, b)
    assert a.snips() == b.snips() == {}


def test_adding_the_same_entry_again_after_deleting_it_works(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    clock.tick(5)
    a.dictionary.remove(a.term_id("Kubernetes"))
    rounds(a, b)
    clock.tick(5)
    b.dictionary.add("Kubernetes", ["k8s"])
    rounds(a, b)
    assert a.terms() == b.terms() == {"Kubernetes": ["k8s"]}


def test_tombstones_are_forgotten_after_180_days(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    clock.tick(5)
    a.dictionary.remove(a.term_id("Kubernetes"))
    rounds(a, b)
    clock.tick(181 * 86400)
    rounds(a, b)
    for d in (a, b):
        doc = json.loads((folder / d.engine.file_name).read_text(encoding="utf-8"))
        assert "kubernetes" not in doc["records"]["dictionary"]
    assert a.terms() == b.terms() == {}


def test_tones_and_settings_are_never_deleted_only_changed(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    rounds(a, b)
    evil = {"format": "voxflow-sync", "version": 1, "device": "evil0000", "name": "evil", "updated": clock.t,
            "records": {"settings": {"sounds_enabled": {"x": 1, "t": clock.t + 5, "d": "evil0000"}},
                        "styles": {"tone:email": {"x": 1, "t": clock.t + 5, "d": "evil0000"}}}}
    (folder / "voxflow-sync-evil-evil0000.json").write_text(json.dumps(evil), encoding="utf-8")
    clock.tick(10)
    rounds(a, b)
    assert "sounds_enabled" in a.settings and "email" in a.styles.tones()
    assert a.portable()["settings"]["sounds_enabled"] is True


# =========================================================================== sync folder: joining
def test_joining_what_the_folder_holds_wins_and_the_rest_is_added(tmp_path, clock, folder):
    a = Device(tmp_path, "desk", clock)
    a.dictionary.add("Kubernetes", ["folder alias"])
    a.snippets.add("sig", "folder text")
    a.settings["sounds_enabled"] = False
    a.engine.set_folder(str(folder))
    rounds(a)
    clock.tick(86400)
    b = Device(tmp_path, "laptop", clock)                       # has used VoxFlow on its own for a while
    b.dictionary.add("kubernetes", ["laptop alias"])
    b.dictionary.add("OnlyOnLaptop")
    b.snippets.add("sig", "laptop text")
    b.snippets.add("laptop only", "x")
    b.settings["sounds_enabled"] = True
    b.settings["language"] = "fr"
    b.engine.set_folder(str(folder))
    rounds(b, a, b)
    assert b.terms() == a.terms() == {"Kubernetes": ["folder alias"], "OnlyOnLaptop": []}
    assert b.snips() == a.snips() == {"sig": "folder text", "laptop only": "x"}
    assert b.settings["sounds_enabled"] is False and a.settings["sounds_enabled"] is False


def test_joining_does_not_delete_entries_because_of_an_old_tombstone(tmp_path, clock, folder):
    a = Device(tmp_path, "desk", clock)
    a.dictionary.add("Kubernetes")
    a.engine.set_folder(str(folder))
    rounds(a)
    clock.tick(5)
    a.dictionary.remove(a.term_id("Kubernetes"))
    rounds(a)
    clock.tick(86400)
    b = Device(tmp_path, "laptop", clock)
    b.dictionary.add("Kubernetes", ["mine"])                    # the newcomer has the very entry that was deleted
    b.engine.set_folder(str(folder))
    rounds(b, a, b)
    assert b.terms() == a.terms() == {"Kubernetes": ["mine"]}


def test_pointing_at_another_folder_joins_it_afresh(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Shared")
    rounds(a, b)
    other = tmp_path / "other-cloud"
    other.mkdir()
    c = Device(tmp_path, "tablet", clock)
    c.dictionary.add("FromTablet")
    c.engine.set_folder(str(other))
    rounds(c)
    b.engine.set_folder(str(other))
    assert b.engine.status()["message"] == "Waiting for the first sync." and b.engine.status()["devices"] == []
    rounds(b, c, b)
    assert b.terms() == c.terms() == {"Shared": [], "FromTablet": []}
    assert len(sync_files(folder)) == 2, "the file in the old folder stays for the other devices"
    b.engine.set_folder(str(other))                             # same folder again: nothing is reset
    assert b.engine.status()["last_sync"] is not None


def test_turning_sync_off_leaves_the_file_and_the_local_data(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    b.engine.set_folder("")
    assert b.engine.status()["state"] == "off" and b.terms() == {"Kubernetes": []}
    assert len(sync_files(folder)) == 2
    b.dictionary.add("WhileOff")
    clock.tick(10)
    assert b.engine.sync()["state"] == "off"
    rounds(a)
    assert a.terms() == {"Kubernetes": []}
    b.engine.set_folder(None)                                   # None also means off
    b.engine.set_folder(str(folder))                            # back on: joins again, nothing lost
    rounds(b, a, b)
    assert a.terms() == b.terms() == {"Kubernetes": [], "WhileOff": []}


# =========================================================================== sync folder: offline, partial writes
OFFLINE = "The sync folder can't be reached right now. Your changes are kept and will sync when it is back."


def test_folder_offline_is_an_error_state_and_changes_sync_when_it_is_back(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.snippets.add("sig", "v1")
    rounds(a, b)
    last = b.engine.status()["last_sync"]
    away = tmp_path / "unplugged"
    folder.rename(away)                                         # the drive is gone
    clock.tick(60)
    b.snippets.update(b.snip_id("sig"), text="edited offline at t1")
    b.dictionary.add("AddedOffline")
    b.engine.stamp_local()                                      # what the app does at the start of every round
    st = b.engine.sync()
    assert st["state"] == "error" and st["error"] == OFFLINE and st["message"] == OFFLINE
    assert st["last_sync"] == last and st["enabled"] is True
    assert b.snips() == {"sig": "edited offline at t1"} and "AddedOffline" in b.terms()
    clock.tick(3600)
    assert b.engine.sync()["state"] == "error"
    a.snippets.update(a.snip_id("sig"), text="edited on the desk at t2, later")
    a.engine.stamp_local()
    away.rename(folder)                                         # back
    clock.tick(60)
    rounds(b, a, b)
    assert b.engine.status()["state"] == "idle" and b.engine.status()["error"] == ""
    assert a.snips() == b.snips() == {"sig": "edited on the desk at t2, later"}, \
        "the offline edit kept the time it was made, so the later desk edit wins"
    assert "AddedOffline" in a.terms()


def test_the_one_call_round_also_notes_when_an_offline_edit_was_made(tmp_path, clock, folder):
    """``SyncEngine.sync()`` is documented as one whole round. With the folder offline it fails before it looks at
    the local changes, so an offline edit is stamped only when the folder is back and then beats a later edit."""
    a, b = pair(tmp_path, folder, clock)
    a.snippets.add("sig", "v1")
    rounds(a, b)
    away = tmp_path / "unplugged"
    folder.rename(away)
    clock.tick(60)
    b.snippets.update(b.snip_id("sig"), text="edited offline, first")
    assert b.engine.sync()["state"] == "error"
    clock.tick(3600)
    a.snippets.update(a.snip_id("sig"), text="edited on the desk, an hour later")
    a.engine.stamp_local()
    away.rename(folder)
    clock.tick(60)
    rounds(b, a, b)
    assert a.snips() == b.snips() == {"sig": "edited on the desk, an hour later"}


def test_a_write_that_fails_is_reported_and_retried(tmp_path, clock, folder, monkeypatch):
    a, b = pair(tmp_path, folder, clock)
    rounds(a, b)
    a.dictionary.add("Kubernetes")
    real = cfgmod._atomic_write

    def fail_in_folder(path, text):
        if str(path).startswith(str(folder)):
            raise OSError("disk quota exceeded")
        return real(path, text)

    monkeypatch.setattr(cfgmod, "_atomic_write", fail_in_folder)
    clock.tick(5)
    st = a.engine.sync()
    assert st["state"] == "error" and st["error"] == OFFLINE
    monkeypatch.setattr(cfgmod, "_atomic_write", real)
    rounds(a, b)
    assert b.terms() == {"Kubernetes": []}


def test_a_half_written_file_from_another_device_is_skipped_until_it_is_complete(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    clock.tick(5)
    a.dictionary.add("Parakeet")
    rounds(a)
    target = folder / a.engine.file_name
    whole = target.read_bytes()
    target.write_bytes(whole[: len(whole) // 2])                # the cloud tool has copied only half so far
    clock.tick(5)
    st = b.engine.sync()
    assert st["state"] == "idle" and st["unreadable"] == 1
    assert st["message"].endswith("A file in the folder could not be read yet and was skipped.")
    assert b.terms() == {"Kubernetes": []}, "nothing is lost or deleted because a file was unreadable"
    target.write_bytes(whole)
    clock.tick(5)
    st = b.engine.sync()
    assert st["unreadable"] == 0 and b.terms() == {"Kubernetes": [], "Parakeet": []}


@pytest.mark.parametrize("content", [b"", b"\x00" * 64, b"[]", b'{"format": "something-else"}',
                                     b'{"format": "voxflow-sync"}', b'{"format": "voxflow-sync", "version": "1"}',
                                     "{\"format\": \"voxflow-sync\", \"version\": 1, \"records\": ".encode()])
def test_garbage_files_with_a_sync_name_never_break_a_round(tmp_path, clock, folder, content):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    (folder / "voxflow-sync-broken-12345678.json").write_bytes(content)
    rounds(a, b)
    assert b.terms() == {"Kubernetes": []} and b.engine.status()["unreadable"] == 1


def test_records_of_the_wrong_shape_inside_a_valid_file_are_ignored(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    t = clock.t
    doc = {"format": "voxflow-sync", "version": 1, "device": "odd00000", "name": "odd", "updated": "yesterday",
           "records": {
               "dictionary": {"good": {"v": {"term": "Good", "aliases": []}, "t": t, "d": "odd00000"},
                              "no-time": {"v": {"term": "NoTime"}, "d": "odd00000"},
                              "bool-time": {"v": {"term": "BoolTime"}, "t": True, "d": "odd00000"},
                              "negative": {"v": {"term": "Negative"}, "t": -5, "d": "odd00000"},
                              "no-device": {"v": {"term": "NoDevice"}, "t": t},
                              "bad-value": {"v": "just a string", "t": t, "d": "odd00000"},
                              "list": [1, 2, 3]},
               "snippets": "not a dict", "settings": {"no_such_setting": {"v": 1, "t": t, "d": "odd00000"},
                                                     "mic_device": {"v": "Their Mic", "t": t, "d": "odd00000"},
                                                     "sound_volume": {"v": "loud", "t": t, "d": "odd00000"}},
               "styles": {"tone:email": {"v": "shouting", "t": t, "d": "odd00000"},
                          "app:": {"v": "work", "t": t, "d": "odd00000"}},
               "plugins": {"x": 1}}}
    (folder / "voxflow-sync-odd-odd00000.json").write_text(json.dumps(doc), encoding="utf-8")
    clock.tick(5)
    rounds(a, b)
    assert a.terms() == b.terms() == {"Good": []}
    assert a.settings == cfgmod.defaults() and a.styles.user_apps() == {}


def test_a_record_stamped_far_in_the_future_is_ignored(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.snippets.add("sig", "real")
    rounds(a, b)
    doc = {"format": "voxflow-sync", "version": 1, "device": "future00", "name": "time traveller",
           "updated": clock.t, "records": {"snippets": {
               "sig": {"v": {"trigger": "sig", "text": "from 2099"}, "t": clock.t + 30 * 86400, "d": "future00"},
               "soon": {"v": {"trigger": "soon", "text": "clock a bit fast"}, "t": clock.t + 3600, "d": "future00"}}}}
    (folder / "voxflow-sync-future-future00.json").write_text(json.dumps(doc), encoding="utf-8")
    clock.tick(5)
    rounds(a, b)
    assert a.snips() == b.snips() == {"sig": "real", "soon": "clock a bit fast"}


def test_a_file_from_a_newer_voxflow_is_skipped_and_said_so(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    doc = {"format": "voxflow-sync", "version": 2, "device": "new00000", "name": "new", "updated": clock.t,
           "records": {"dictionary": {"x": {"v": {"term": "FromTheFuture"}, "t": clock.t, "d": "new00000"}}}}
    (folder / "voxflow-sync-new-new00000.json").write_text(json.dumps(doc), encoding="utf-8")
    rounds(a, b)
    st = b.engine.status()
    assert st["newer"] == 1 and "newer VoxFlow was skipped" in st["message"] and st["state"] == "idle"
    assert b.terms() == {"Kubernetes": []}


def test_conflicted_copies_made_by_the_cloud_tool_are_merged_too(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    clock.tick(5)
    a.dictionary.add("InTheConflictCopy")
    rounds(a)
    mine = folder / a.engine.file_name
    copy = folder / mine.name.replace(".json", " (conflicted copy 2026-11-13).json")
    shutil.copy(mine, copy)
    clock.tick(5)
    a.dictionary.remove(a.term_id("InTheConflictCopy"))         # the main file moves on, the copy is stale
    a.dictionary.add("Newer")
    rounds(a, b)
    assert b.terms() == {"Kubernetes": [], "Newer": []}, "the stale copy cannot undo a later deletion"
    assert len(b.engine.status()["devices"]) == 1, "a copy is not counted as another device"


def test_other_files_in_the_folder_are_left_alone(tmp_path, clock, folder):
    (folder / "holiday.jpg").write_bytes(b"\xff\xd8\xff")
    (folder / "notes.json").write_text("{}", encoding="utf-8")
    (folder / "voxflow-sync-readme.txt").write_text("not ours", encoding="utf-8")
    (folder / "sub").mkdir()
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    assert b.engine.status()["unreadable"] == 0
    assert (folder / "holiday.jpg").read_bytes() == b"\xff\xd8\xff" and (folder / "notes.json").read_text() == "{}"
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["holiday.jpg", "notes.json", "voxflow-sync-readme.txt", "sub", a.engine.file_name, b.engine.file_name])


# =========================================================================== sync folder: state on disk
def test_the_device_keeps_its_identity_and_records_across_restarts(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    again = b.reopen()
    assert again.engine.device_id == b.engine.device_id and again.engine.file_name == b.engine.file_name
    # P10: the state file is only rewritten when something changed, so a restart may show the last CHANGE as
    # "last synced" until the first round of the new run (which starts at once) refreshes it
    assert again.engine.enabled and 0 < again.engine.status()["last_sync"] <= b.engine.status()["last_sync"]
    clock.tick(5)
    a.dictionary.remove(a.term_id("Kubernetes"))
    rounds(a, again)
    assert again.terms() == {}


def test_changes_made_while_the_app_was_closed_are_noticed_at_the_next_round(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    clock.tick(5)
    b.dictionary.remove(b.term_id("Kubernetes"))                # e.g. edited, then the app was closed at once
    b.dictionary.add("Parakeet")
    again = b.reopen()
    rounds(again, a)
    assert a.terms() == {"Parakeet": []}


@pytest.mark.parametrize("content", ["", "{not json", "[]", "null", '{"records": "x", "device": 5, "last_sync": "y"}',
                                     '{"records": {"dictionary": {"a": 1, "b": {"v": 2, "t": "x", "d": "d"}}}}'])
def test_a_damaged_state_file_means_joining_again_not_a_crash(tmp_path, clock, folder, content):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    rounds(a, b)
    (b.dir / "sync_state.json").write_text(content, encoding="utf-8")
    again = b.reopen()
    assert again.engine.enabled is False and again.engine.status()["state"] == "off"
    again.engine.set_folder(str(folder))
    again.dictionary.add("AfterTheDamage")
    rounds(again, a, again)
    assert a.terms() == again.terms() == {"Kubernetes": [], "AfterTheDamage": []}


def test_the_state_file_cannot_be_saved_is_not_fatal(tmp_path, clock, folder, monkeypatch):
    a, b = pair(tmp_path, folder, clock)
    a.dictionary.add("Kubernetes")
    real = cfgmod._atomic_write

    def fail_state(path, text):
        if str(path).endswith("sync_state.json"):
            raise OSError("profile folder is read-only")
        return real(path, text)

    monkeypatch.setattr(cfgmod, "_atomic_write", fail_state)
    rounds(a, b)
    assert b.terms() == {"Kubernetes": []}


def test_a_setting_that_cannot_be_applied_here_keeps_its_local_value_for_everyone(tmp_path, clock, folder):
    a, b = pair(tmp_path, folder, clock)
    rounds(a, b)
    b.refuse = {"hotkey"}                                       # e.g. clashes with something on this computer
    clock.tick(5)
    a.settings["hotkey"] = "ctrl+alt+f9"
    rounds(a, b, a, b)
    assert b.settings["hotkey"] == cfgmod.defaults()["hotkey"]
    assert a.settings["hotkey"] == cfgmod.defaults()["hotkey"], "the value that works everywhere wins"


# =========================================================================== sync folder: convergence
@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_random_edits_on_three_devices_always_converge(tmp_path, folder, seed):
    rng = random.Random(seed)
    clock = Clock()
    devices = [Device(tmp_path, name, clock) for name in ("desk", "laptop", "tablet")]
    for d in devices:
        d.engine.set_folder(str(folder))
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for step in range(120):
        d = rng.choice(devices)
        clock.tick(rng.choice([0.5, 1, 5, 60]))
        op = rng.random()
        word = rng.choice(words)
        try:
            if op < 0.25:
                d.dictionary.add(word.capitalize(), [f"{word} {step}"])
            elif op < 0.4 and d.dictionary.list():
                e = rng.choice(d.dictionary.list())
                d.dictionary.update(e["id"], aliases=[f"alias {step}"])
            elif op < 0.5 and d.dictionary.list():
                d.dictionary.remove(rng.choice(d.dictionary.list())["id"])
            elif op < 0.7:
                d.snippets.add(word, f"text {step}")
            elif op < 0.8 and d.snippets.list():
                e = rng.choice(d.snippets.list())
                d.snippets.update(e["id"], text=f"edited {step}")
            elif op < 0.88 and d.snippets.list():
                d.snippets.remove(rng.choice(d.snippets.list())["id"])
            elif op < 0.94:
                d.styles.set_tone(rng.choice(list(store.CATEGORIES)), rng.choice(list(store.TONES)))
            else:
                d.settings["sounds_enabled"] = rng.random() < 0.5
        except ValueError:
            pass                                                # e.g. adding a term that already exists
        if rng.random() < 0.4:
            rng.choice(devices).engine.sync()
    rounds(*devices, n=4)
    first = devices[0].portable()
    assert all(d.portable() == first for d in devices)
    files = [json.loads((folder / d.engine.file_name).read_text(encoding="utf-8"))["records"] for d in devices]
    assert files[0] == files[1] == files[2], "every device ends with the same records, tombstones included"
