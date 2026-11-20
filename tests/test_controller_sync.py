"""Backups, packs and the sync folder through the Controller (what the UI calls): signals, threading, timers,
recording in progress, shutdown. The engine itself is tested in ``test_sync.py``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxflow import controller as ctl_mod

from controller_rig import make_rig

OFFLINE = "The sync folder can't be reached right now. Your changes are kept and will sync when it is back."



def peek(path, encoding="utf-8") -> str:
    """Read a file the application replaces atomically. On Windows a reader that opens it during the swap gets
    PermissionError (or the file is briefly missing): that is the reader's problem, not a product state, so a poll
    treats it as "nothing yet". (Review: this was the intermittent failure of three sync tests under CPU load.)"""
    try:
        return Path(path).read_text(encoding=encoding)
    except (PermissionError, FileNotFoundError):
        return ""


def read_settled(path, encoding="utf-8") -> str:
    """Like ``peek`` for a single assertion: retry the sharing violation, never return a made-up empty string."""
    import time
    for _ in range(400):
        try:
            return Path(path).read_text(encoding=encoding)
        except PermissionError:
            time.sleep(0.005)                                    # an OS-level sharing violation, not a product wait
    return Path(path).read_text(encoding=encoding)

def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


@pytest.fixture
def fast_sync(monkeypatch):
    monkeypatch.setattr(ctl_mod, "SYNC_DEBOUNCE_MS", 30)
    monkeypatch.setattr(ctl_mod, "SYNC_INTERVAL_MS", 150)


@pytest.fixture
def rigs(tmp_path, qtbot, monkeypatch, fast_sync):
    made = []

    def make(name: str, **kw):
        base = tmp_path / name
        base.mkdir(exist_ok=True)
        rig = make_rig(base, qtbot, monkeypatch, **kw)
        rig.sync_events = []
        rig.c.sync_status_changed.connect(rig.sync_events.append)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


@pytest.fixture
def rig(rigs):
    r = rigs("alpha")
    r.c.dictionary.add("Kubernetes", ["cooper netties"])
    r.c.snippets.add("my email", "ada@example.com")
    r.c.history.add("a dictation", raw="a dictation", ms=50.0)
    return r


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "cloud"
    f.mkdir()
    return f


def idle(rig) -> bool:
    return not rig.c._sync_busy and rig.c.sync_status()["state"] in ("idle", "error")


def synced(rig) -> bool:
    return idle(rig) and rig.c.sync_status()["last_sync"] is not None


def terms(rig) -> dict:
    return {e["term"]: sorted(e["aliases"]) for e in rig.c.dictionary.list()}


def snips(rig) -> dict:
    return {e["trigger"]: e["text"] for e in rig.c.snippets.list()}


# =========================================================================== backups
def test_suffixes_and_parts_are_published_for_the_dialogs(rig):
    assert rig.c.BACKUP_SUFFIX == ".voxbackup" and rig.c.PACK_SUFFIX == ".voxpack"
    assert [key for key, _label in rig.c.backup_parts()] == ["settings", "dictionary", "snippets", "styles", "stats",
                                                            "history"]
    assert all(label and label[0].isupper() for _key, label in rig.c.backup_parts())


def test_backup_round_trip_between_two_profiles_fires_the_usual_signals(rig, rigs, tmp_path):
    rig.c.set("sounds_enabled", False)
    out = rig.c.export_backup(str(tmp_path / "mine"))
    assert out["path"].endswith(".voxbackup") and out["counts"]["dictionary"] == 1 and out["counts"]["history"] == 1
    info = rig.c.inspect_backup(out["path"])
    assert info["same_device"] is True and info["counts"]["snippets"] == 1

    other = rigs("beta")
    assert other.c.inspect_backup(out["path"])["same_device"] is False
    other.spy.clear()
    summary = other.c.import_backup(out["path"], "replace")
    assert terms(other) == {"Kubernetes": ["cooper netties"]} and snips(other) == {"my email": "ada@example.com"}
    assert other.c.history.count() == 1 and other.c.get("sounds_enabled") is False
    assert summary["mode"] == "replace" and summary["settings_skipped"] == []
    assert other.spy.of("dictionary_changed") and other.spy.of("snippets_changed") and other.spy.of("history_changed")
    assert ("sounds_enabled", False) in other.spy.of("config_changed")
    saved = json.loads(read_settled(other.data_dir / "config.json"))
    assert saved["sounds_enabled"] is False, "imported settings are persisted"
    assert other.c.sounds.enabled is False, "imported settings take effect, not only get stored"


def test_import_merge_keeps_settings_and_reports_what_was_added(rig, rigs, tmp_path):
    rig.c.set("sounds_enabled", False)
    path = rig.c.export_backup(str(tmp_path / "mine"))["path"]
    other = rigs("beta")
    other.c.dictionary.add("Zephyr")
    summary = other.c.import_backup(path)
    assert summary["mode"] == "merge" and summary["counts"]["dictionary"] == 1
    assert terms(other) == {"Kubernetes": ["cooper netties"], "Zephyr": []}
    assert other.c.get("sounds_enabled") is True


def test_import_with_parts(rig, rigs, tmp_path):
    path = rig.c.export_backup(str(tmp_path / "mine"))["path"]
    other = rigs("beta")
    other.c.import_backup(path, "replace", parts=["snippets"])
    assert snips(other) == {"my email": "ada@example.com"} and terms(other) == {} and other.c.history.count() == 0


@pytest.mark.parametrize("content, message", [
    ("garbage", "not valid JSON"),
    (json.dumps({"format": "voxflow-pack", "version": 1, "kind": "dictionary", "entries": ["a"]}), "not a backup"),
    (json.dumps({"format": "voxflow-profile", "version": 9, "dictionary": ["a"]}), "newer version"),
])
def test_bad_backups_raise_a_sentence_and_emit_nothing(rig, tmp_path, content, message):
    bad = tmp_path / "bad.voxbackup"
    bad.write_text(content, encoding="utf-8")
    rig.spy.clear()
    with pytest.raises(ValueError, match=message):
        rig.c.inspect_backup(str(bad))
    with pytest.raises(ValueError, match=message):
        rig.c.import_backup(str(bad), "replace")
    assert terms(rig) == {"Kubernetes": ["cooper netties"]} and rig.c.history.count() == 1
    assert not rig.spy.of("dictionary_changed") and not rig.spy.of("config_changed") and not rig.spy.notices()


def test_importing_a_backup_while_recording_ends_the_recording_first(rig, rigs, tmp_path):
    path = rig.c.export_backup(str(tmp_path / "mine"))["path"]
    other = rigs("beta")
    other.c._on_hotkey("dictate", "start")
    assert other.c.state == "listening"
    other.c.import_backup(path, "replace")
    assert other.c.state == "idle" and terms(other) == {"Kubernetes": ["cooper netties"]}


def test_replacing_shortcuts_from_a_backup_rebinds_the_live_hotkeys(rig, rigs, tmp_path):
    rig.c.set("hotkey", "ctrl+alt+f9")
    path = rig.c.export_backup(str(tmp_path / "mine"))["path"]
    other = rigs("beta")
    other.c.import_backup(path, "replace")
    assert other.c.get("hotkey") == "ctrl+alt+f9"
    assert other.c.listener.machine.combo == "ctrl+alt+f9" or "f9" in str(other.c.listener.machine.__dict__)


# =========================================================================== packs
def test_pack_round_trip_and_messages(rig, rigs, tmp_path):
    n = rig.c.export_pack("dictionary", str(tmp_path / "team"), name="Team", description="Our words")
    assert n == 1 and (tmp_path / "team.voxpack").is_file()
    other = rigs("beta")
    other.c.dictionary.add("kubernetes", ["kube"])
    info = other.c.inspect_pack(str(tmp_path / "team.voxpack"))
    assert (info["kind"], info["name"], info["description"], info["count"], info["new"], info["existing"]) == (
        "dictionary", "Team", "Our words", 1, 0, 1)
    other.spy.clear()
    out = other.c.import_pack(str(tmp_path / "team.voxpack"))
    assert out == {"kind": "dictionary", "name": "Team", "added": 0, "updated": 1, "removed": 0, "skipped": 0,
                   "aliases_skipped": []}
    assert terms(other) == {"Kubernetes": ["cooper netties", "kube"]}
    assert other.spy.of("dictionary_changed")

    assert rig.c.export_pack("snippets", str(tmp_path / "s.voxpack"), ids=[rig.c.snippets.list()[0]["id"]]) == 1
    with pytest.raises(ValueError, match="Pick at least one entry to share"):
        rig.c.export_pack("snippets", str(tmp_path / "none"), ids=[])
    with pytest.raises(ValueError, match="There are no snippets to share yet"):
        other.c.export_pack("snippets", str(tmp_path / "none"))
    backup = rig.c.export_backup(str(tmp_path / "b"))["path"]
    with pytest.raises(ValueError, match="whole backup, not a pack"):
        other.c.import_pack(backup)
    assert snips(other) == {}


# =========================================================================== sync folder
def test_sync_is_off_by_default_and_not_a_config_key(rig):
    st = rig.c.sync_status()
    assert st["state"] == "off" and st["message"] == "Sync is off." and rig.c.sync_folder == ""
    assert rig.c.get("sync_folder") == ""
    assert "sync_folder" not in json.loads(read_settled(rig.data_dir / "config.json"))
    rig.spy.clear()
    rig.c.sync_now()
    assert rig.spy.notices() == [("info", "Sync is off. Choose a sync folder first.")]


def test_choosing_a_folder_starts_syncing_and_announces_it(rig, folder):
    rig.spy.clear()
    rig.c.set("sync_folder", str(folder))
    assert ("sync_folder", rig.c.sync_folder) in rig.spy.of("config_changed") and rig.c.sync_folder
    assert rig.c.get("sync_folder") == rig.c.sync_folder
    rig.wait(lambda: synced(rig))
    states = [e["state"] for e in rig.sync_events]
    assert "syncing" in states and states[-1] == "idle"
    assert rig.c.sync_status()["message"] == "Up to date. No other device has used this folder yet."
    files = [p.name for p in folder.iterdir()]
    assert len(files) == 1 and files[0].startswith("voxflow-sync-")
    assert "sync_folder" not in json.loads(read_settled(rig.data_dir / "config.json"))
    assert (rig.data_dir / "sync_state.json").is_file()
    assert "a dictation" not in read_settled(folder / files[0])


@pytest.mark.parametrize("make_bad, message", [
    (lambda tmp, rig: str(tmp / "missing"), "That folder doesn't exist"),
    (lambda tmp, rig: str(rig.data_dir), "other than VoxFlow's own data folder"),
    (lambda tmp, rig: 5, "must be a folder path"),
])
def test_a_bad_folder_is_refused_and_sync_stays_as_it_was(rig, tmp_path, make_bad, message):
    rig.spy.clear()
    with pytest.raises(ValueError, match=message):
        rig.c.set("sync_folder", make_bad(tmp_path, rig))
    assert rig.c.sync_status()["state"] == "off" and not rig.spy.of("config_changed")


def test_two_running_apps_sync_through_one_folder(rig, rigs, folder):
    other = rigs("beta")
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: synced(rig))
    other.c.set_sync_folder(str(folder))
    other.wait(lambda: terms(other) == {"Kubernetes": ["cooper netties"]})
    assert snips(other) == {"my email": "ada@example.com"} and other.c.history.count() == 0

    other.spy.clear()
    other.c.snippets.add("sign off", "Best regards")           # a local change syncs by itself, debounced
    other.c.set("language", "de")
    rig.wait(lambda: snips(rig).get("sign off") == "Best regards" and rig.c.get("language") == "de", timeout=8000)
    assert rig.spy.of("snippets_changed") and ("language", "de") in rig.spy.of("config_changed")
    rig.wait(lambda: rig.c.sync_status()["message"] == "Up to date with 1 other device.", timeout=8000)

    entry = rig.c.dictionary.list()[0]
    rig.c.dictionary.remove(entry["id"])                       # a deletion travels too
    other.wait(lambda: terms(other) == {}, timeout=8000)
    rig.wait(lambda: idle(rig))
    assert terms(rig) == {}, "applying the other side's data does not bounce back"


def test_per_computer_settings_and_history_stay_home(rig, rigs, folder):
    other = rigs("beta")
    rig.c.set_sync_folder(str(folder))
    other.c.set_sync_folder(str(folder))
    rig.wait(lambda: synced(rig) and synced(other))
    rig.c.set("restore_delay_s", 1.5)
    rig.c.set("start_minimized", True)
    rig.c.set("sound_volume", 0.2)
    other.wait(lambda: other.c.get("sound_volume") == 0.2, timeout=8000)
    assert other.c.get("restore_delay_s") != 1.5 and other.c.get("start_minimized") is False
    assert other.c.history.count() == 0


def test_offline_folder_is_a_status_not_a_notice_and_recovers(rig, rigs, folder, tmp_path):
    other = rigs("beta")
    rig.c.set_sync_folder(str(folder))
    other.c.set_sync_folder(str(folder))
    rig.wait(lambda: synced(rig) and synced(other))
    away = tmp_path / "unplugged"
    rig.wait(lambda: idle(rig) and idle(other))
    folder.rename(away)
    rig.spy.clear()
    rig.c.snippets.add("offline", "made while the drive was gone")
    rig.wait(lambda: rig.c.sync_status()["state"] == "error", timeout=8000)
    st = rig.c.sync_status()
    assert st["error"] == OFFLINE and st["message"] == OFFLINE
    rig.settle(500)                                            # several more rounds while offline
    assert rig.spy.notices() == [], "an unreachable folder is shown in the status row, never as a notice"
    assert snips(rig)["offline"] == "made while the drive was gone"
    away.rename(folder)
    other.wait(lambda: snips(other).get("offline") == "made while the drive was gone", timeout=8000)
    rig.wait(lambda: rig.c.sync_status()["state"] == "idle" and rig.c.sync_status()["error"] == "", timeout=8000)


def test_no_sync_round_runs_under_a_recording(rig, rigs, folder):
    other = rigs("beta")
    rig.c.set_sync_folder(str(folder))
    other.c.set_sync_folder(str(folder))
    rig.wait(lambda: synced(rig) and synced(other))
    other.wait(lambda: idle(other))
    other.c._on_hotkey("dictate", "start")
    assert other.c.state == "listening"
    rig.c.snippets.add("during", "added while the other one records")
    rig.wait(lambda: "during" in peek(folder / rig.c.sync.file_name), timeout=8000)
    other.settle(600)                                          # several intervals pass
    assert "during" not in snips(other), "the stores are not changed while a recording is active"
    other.c.cancel()
    other.wait(lambda: snips(other).get("during") == "added while the other one records", timeout=8000)


def test_sync_now_runs_a_round_at_once_and_coalesces(rig, rigs, folder, monkeypatch):
    monkeypatch.setattr(ctl_mod, "SYNC_INTERVAL_MS", 3_600_000)
    monkeypatch.setattr(ctl_mod, "SYNC_DEBOUNCE_MS", 3_600_000)
    other = rigs("gamma")                                      # built with the long timers
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: synced(rig), timeout=8000)
    other.c.set_sync_folder(str(folder))
    other.wait(lambda: synced(other))
    before = other.c.sync_status()["last_sync"]
    other.c.dictionary.add("OnlyViaSyncNow")
    for _ in range(5):
        other.c.sync_now()                                     # five clicks, no pile-up
    other.wait(lambda: idle(other) and "onlyviasyncnow" in peek(folder / other.c.sync.file_name).lower())
    assert other.c.sync_status()["last_sync"] >= before


def test_turning_sync_off_stops_the_rounds_and_keeps_everything(rig, rigs, folder):
    other = rigs("beta")
    rig.c.set_sync_folder(str(folder))
    other.c.set_sync_folder(str(folder))
    other.wait(lambda: terms(other) == {"Kubernetes": ["cooper netties"]})
    other.spy.clear()
    other.c.set("sync_folder", "")
    assert other.c.sync_status()["state"] == "off" and ("sync_folder", "") in other.spy.of("config_changed")
    assert other.sync_events[-1]["state"] == "off"
    assert terms(other) == {"Kubernetes": ["cooper netties"]} and len([f for f in folder.iterdir() if f.suffix != ".tmp"]) == 2   # a peer's write may be in flight
    rig.c.dictionary.add("AfterOff")
    rig.wait(lambda: "afteroff" in peek(folder / rig.c.sync.file_name).lower(), timeout=8000)
    other.settle(500)
    assert "AfterOff" not in terms(other)


def test_the_folder_and_identity_survive_a_restart(rigs, folder, tmp_path, qtbot, monkeypatch):
    first = rigs("alpha")
    first.c.dictionary.add("Kubernetes")
    first.c.set_sync_folder(str(folder))
    first.wait(lambda: synced(first))
    device, name = first.c.sync.device_id, first.c.sync.file_name
    first.c.dictionary.add("JustBeforeQuit")                   # not yet synced: must not be lost
    first.close()
    again = rigs("alpha")                                      # same data folder
    assert again.c.sync_folder and again.c.sync.device_id == device and again.c.sync.file_name == name
    again.wait(lambda: "justbeforequit" in peek(folder / name).lower(), timeout=8000)
    assert len(list(folder.iterdir())) == 1


def test_shutdown_during_a_round_is_clean(rig, folder):
    rig.c.set_sync_folder(str(folder))
    rig.c.sync_now()
    rig.c.shutdown()                                           # a worker thread may still be reading the folder
    rig.settle(300)                                            # its result arrives after shutdown: must be ignored
    assert rig.c.sync_status()["state"] in ("idle", "syncing", "error")


def test_language_support_levels():
    C = ctl_mod.Controller
    for code in ("es", "fr", "de", "it", "pt", "nl"):
        assert C.language_support(code) == "full"
    assert C.language_support("auto") == "auto"
    for code in ("pl", "sv", "uk", "fi"):
        assert C.language_support(code) == "basic"
    assert C.language_support("en") in ("full", "auto", "basic")


# =========================================================================== a damaged file must not sync as a deletion
def test_a_damaged_dictionary_or_settings_file_on_one_device_does_not_empty_the_others(rigs, folder, caplog):
    """QA DL-01: a store that could not be read starts empty. Synced as it is, every entry would count as deleted by
    the user (and the default settings as a fresh choice), and the other devices would apply that."""
    a, b = rigs("alpha"), rigs("beta")
    for term in ("Kubernetes", "Grafana", "Parakeet"):
        a.c.dictionary.add(term)
    a.c.snippets.add("my email", "ada@example.com")
    a.c.set_sync_folder(str(folder))
    b.c.set_sync_folder(str(folder))
    everything = {"Grafana", "Kubernetes", "Parakeet"}
    b.wait(lambda: set(terms(b)) == everything and synced(a) and synced(b), timeout=8000)
    b.c.set("language", "de")
    a.wait(lambda: a.c.get("language") == "de" and synced(a) and synced(b), timeout=8000)

    a.close()                                                   # alpha is closed and two of its files get damaged
    for name in ("dictionary.json", "config.json"):
        path = a.data_dir / name
        path.write_bytes(path.read_bytes()[:40])
    with caplog.at_level("WARNING", logger="voxflow.controller"):
        a2 = rigs("alpha")
    assert "could not be read" in a2.c.dictionary.load_problem and "joining the sync folder again" in caplog.text
    a2.wait(lambda: set(terms(a2)) == everything and synced(a2) and synced(b), timeout=8000)
    assert set(terms(b)) == everything, "beta keeps its words, and alpha has them back from the folder"
    assert set(snips(a2)) == set(snips(b)) and len(snips(b)) == 1
    assert b.c.get("language") == "de" and a2.c.get("language") == "de", "the folder's setting wins"


def test_rejoining_without_a_sync_folder_does_nothing(rig):
    assert rig.c.sync.rejoin() is False and not rig.c.sync.enabled
