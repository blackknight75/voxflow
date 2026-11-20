"""Findings of the 1.0 security review (S-01 ... S-17), each with the fix it guards."""
from __future__ import annotations

from tests.controller_rig import make_rig
from voxflow import inject
from voxflow.command import tidy_output

ESC, BEL, RLO, LRI, PDI, RLM = chr(0x1B), chr(7), chr(0x202E), chr(0x2066), chr(0x2069), chr(0x200F)


# ============================================================================ S-06: what command mode pastes
def test_command_output_loses_control_characters_and_direction_overrides():
    """The selection may be someone else's text (a web page); the model can be talked into repeating an escape
    sequence or an invisible override. Tabs and line breaks are text and stay."""
    out = tidy_output(f"first line{ESC}[2J{BEL}\r\nsecond{RLO}\tline{LRI}x{PDI}{chr(0x85)}{chr(0)}", "some selection")
    assert out == "first line[2J\nsecond\tlinex"


def test_direction_marks_that_the_selection_itself_uses_are_kept():
    selection = f"{RLM}שלום"
    assert tidy_output(f"{RLM}שלום עולם{RLO}", selection) == f"{RLM}שלום עולם"


# ============================================================================ S-05: fallback copies stay private
class _Clip:
    def __init__(self):
        self.calls = []

    def snapshot(self):
        return "before"

    def set_text(self, text, private=True):
        self.calls.append((text, private))
        return len(self.calls)

    def sequence(self):
        return len(self.calls)

    def restore(self, snap):
        return True


class _Keys:
    def modifiers_down(self):
        return False


def test_a_copy_made_instead_of_a_paste_is_kept_out_of_the_clipboard_history():
    clip = _Clip()
    inj = inject.Injector(clip, _Keys(), restore=True)
    assert inj.copy_private("the rewritten confidential paragraph") is True
    assert inj.copy_text("copied with the Copy button") is True
    assert clip.calls == [("the rewritten confidential paragraph", True), ("copied with the Copy button", False)]


def test_a_clipboard_without_the_private_switch_still_copies():
    class Old:
        def set_text(self, text):
            self.text = text
            return 1
    old = Old()
    inj = inject.Injector(old, _Keys(), restore=False)
    assert inj.copy_private("x") is True and old.text == "x"


def test_the_window_changed_and_own_window_fallbacks_use_the_private_copy(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        private = []
        rig.injector.copy_private = lambda text: (private.append(text), True)[1]
        rig.injector.selection = "some selected text"
        hwnd = [rig.context.hwnd]
        rig.c._hwnd_fn = lambda: hwnd[0]
        rig.asr.text = "make it shorter"
        rig.pause()
        rig.press("command")
        hwnd[0] = 0x999                                         # the user switched windows while the model worked
        rig.release("command")
        rig.wait(lambda: rig.c.state in ("error", "idle") and private)
        assert private == ["REWRITTEN"] and rig.injector.copied == [], "not through the history-visible copy"
    finally:
        rig.close()


# ============================================================================ S-11: no telemetry events
def test_onnxruntime_telemetry_events_are_switched_off_before_it_is_used(monkeypatch):
    import sys
    import types
    from voxflow import asr
    calls = []
    fake = types.ModuleType("onnxruntime")
    fake.disable_telemetry_events = lambda: calls.append("off")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    assert asr._onnxruntime() is fake and calls == ["off"]
    fake.disable_telemetry_events = lambda: (_ for _ in ()).throw(RuntimeError("not in this build"))
    assert asr._onnxruntime() is fake                           # a build that cannot: nothing to switch off
    del fake.disable_telemetry_events
    assert asr._onnxruntime() is fake


# ============================================================================ S-17: the current directory is no DLL root
def test_the_current_directory_and_relative_entries_are_not_import_roots(monkeypatch, tmp_path):
    import os
    import sys
    from voxflow import system
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", ["", ".", "relative", str(tmp_path), str(tmp_path).upper(), str(site)])
    assert system.import_roots() == [str(site)]
    assert os.getcwd() == str(tmp_path)


# ============================================================================ S-13: a download is checked before use
def test_a_downloaded_model_file_with_the_right_size_and_wrong_content_is_refused_and_deleted(monkeypatch, tmp_path):
    import hashlib
    import pytest
    import huggingface_hub
    from voxflow import modelhub
    repo, rev, good = "org/tiny-model", "a" * 40, b"the reviewed model file"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setitem(modelhub.PINNED, repo, (rev, {"model.gguf": (len(good), hashlib.sha256(good).hexdigest())}))
    monkeypatch.setattr(modelhub.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 ** 12})())
    snap = modelhub.snapshot_dir(repo)
    snap.mkdir(parents=True)
    served = [b"X" * len(good)]                                  # same size, other bytes

    def hf_hub_download(_repo, filename, revision=None, **kw):
        (snap / filename).write_bytes(served[0])
        return str(snap / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    with pytest.raises(modelhub.ChecksumMismatch, match="did not arrive as published"):
        modelhub.model_file(repo, "model.gguf")
    assert not (snap / "model.gguf").exists(), "never left in the cache for the next start to load"
    served[0] = good
    assert modelhub.model_file(repo, "model.gguf") == str(snap / "model.gguf")


# ============================================================================ S-14: the log does not name the app
def test_the_log_line_of_a_dictation_has_timings_but_not_the_program_dictated_into(tmp_path, qtbot, monkeypatch, caplog):
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        with caplog.at_level("INFO", logger="voxflow.controller"):
            rig.dictate("um hello world")
        line = next(r.getMessage() for r in caplog.records if "release->paste" in r.getMessage())
        assert "words" in line and "notepad" not in caplog.text.lower() and "hello" not in caplog.text.lower()
    finally:
        rig.close()


# ============================================================================ S-12: the Releases page link
def test_a_release_read_back_from_the_state_file_cannot_point_the_link_anywhere_else():
    from voxflow import updater as U
    base = {"version": "1.2.3", "tag": "v1.2.3", "title": "VoxFlow 1.2.3", "notes": "", "published": ""}
    own = f"https://github.com/{U.REPO}/releases/tag/v1.2.3"
    assert U.Release.from_dict(dict(base, page=own)).page == own
    for evil in ("file:///C:/Windows/System32/calc.exe", "https://github.com.evil.example/x", "javascript:alert(1)",
                 f"https://github.com/{U.REPO}/releases/tag/v1/../../../other", 7, ""):
        assert U.Release.from_dict(dict(base, page=evil)).page == U.RELEASES_PAGE, evil


# ============================================================================ S-08: the dictionary CSV runs nothing
def test_the_dictionary_csv_export_marks_formula_cells_and_the_import_takes_the_mark_off(tmp_path):
    from voxflow import store
    d = store.Dictionary(tmp_path / "d.json")
    d.add("=HYPERLINK(1)", ["@SUM(1+1)", "plain alias"])
    d.add("-dash", ["+plus"])
    d.add("O'Brien")
    out = tmp_path / "words.csv"
    assert d.export(str(out)) == 3
    cells = [c for line in out.read_text(encoding="utf-8-sig").splitlines()[1:] for c in line.split(",")]
    assert not [c for c in cells if c[:1] in "=+-@"], cells
    again = store.Dictionary(tmp_path / "again.json")
    again.import_(str(out))
    assert {e["term"]: sorted(e["aliases"]) for e in again.list()} == {e["term"]: sorted(e["aliases"]) for e in d.list()}


# ============================================================================ S-01: what a sync folder may do
def _pair(tmp_path):
    from tests.test_sync import Clock, Device
    clock = Clock()
    folder = tmp_path / "shared"
    folder.mkdir()
    a, b = Device(tmp_path, "desk", clock), Device(tmp_path, "laptop", clock)
    for d in (a, b):
        d.engine.set_folder(str(folder))
    return a, b, clock, folder


def _rounds(*devices):
    for _ in range(3):
        for d in devices:
            d.clock.tick(1)
            assert d.engine.sync()["state"] == "idle"


def test_history_settings_and_shortcuts_never_travel_through_the_sync_folder(tmp_path):
    import json
    from voxflow import sync
    a, b, clock, folder = _pair(tmp_path)
    _rounds(a, b)
    before = dict(b.settings)
    clock.tick(5)
    a.settings.update(save_history=not before["save_history"], history_limit=100, history_retention_days=1,
                      hotkey="ctrl+shift+f9", command_hotkey="ctrl+shift+f10", language="de")
    _rounds(a, b, a, b)
    assert b.settings["language"] == "de", "ordinary settings still sync"
    for key in sync.NOT_SYNCED:
        assert b.settings[key] == before[key], key
    # ... and a file that carries them anyway (written by something else) is not obeyed
    doc = json.loads((folder / a.engine.file_name).read_text(encoding="utf-8"))
    doc["records"]["settings"]["history_retention_days"] = {"v": 1, "t": clock() + 60, "d": doc["device"]}
    doc["records"]["settings"]["hotkey"] = {"v": "ctrl+shift+f9", "t": clock() + 60, "d": doc["device"]}
    (folder / a.engine.file_name).write_text(json.dumps(doc), encoding="utf-8")
    clock.tick(120)
    _rounds(b)
    assert b.settings["history_retention_days"] == before["history_retention_days"]
    assert b.settings["hotkey"] == before["hotkey"]


def test_text_from_a_sync_folder_is_cleaned_and_bounded_before_it_can_be_typed(tmp_path):
    import json
    from voxflow import store
    a, b, clock, folder = _pair(tmp_path)
    a.snippets.add("my address", "1 Main Street")
    _rounds(a, b)
    doc = json.loads((folder / a.engine.file_name).read_text(encoding="utf-8"))
    stamp = clock() + 30
    esc, rlo, c1 = chr(0x1B), chr(0x202E), chr(0x85)
    doc["records"]["snippets"]["yes"] = {"v": {"trigger": "yes", "text": f"{esc}[2Jdone{rlo}{c1}"}, "t": stamp,
                                         "d": doc["device"]}
    doc["records"]["snippets"]["huge"] = {"v": {"trigger": "huge", "text": "x" * (store.MAX_SNIPPET_LEN + 1)},
                                          "t": stamp, "d": doc["device"]}
    doc["records"]["dictionary"]["evil"] = {"v": {"term": f"Evil{rlo}Term", "aliases": []}, "t": stamp,
                                            "d": doc["device"]}
    (folder / a.engine.file_name).write_text(json.dumps(doc), encoding="utf-8")
    clock.tick(60)
    assert b.engine.sync()["state"] == "idle"
    assert b.engine.last_applied.get("snippets") == 1, "what the controller's notice counts"
    _rounds(b)
    got = {e["trigger"]: e["text"] for e in b.snippets.list()}
    assert got["yes"] == "[2Jdone", "no escape character, no direction override, no C1 control"
    assert "huge" not in got and got["my address"] == "1 Main Street"
    assert "EvilTerm" in [e["term"] for e in b.dictionary.list()]
    assert store.foreign_text("ok") == "ok" and store.foreign_text("x" * 5, limit=4) is None


def test_a_snippet_changed_through_the_sync_folder_is_announced(tmp_path, qtbot, monkeypatch):
    from voxflow import controller as ctl_mod
    monkeypatch.setattr(ctl_mod, "SYNC_DEBOUNCE_MS", 30)
    monkeypatch.setattr(ctl_mod, "SYNC_INTERVAL_MS", 150)
    folder = tmp_path / "cloud"
    folder.mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    a, b = make_rig(tmp_path / "alpha", qtbot, monkeypatch), make_rig(tmp_path / "beta", qtbot, monkeypatch)
    try:
        a.c.set_sync_folder(str(folder))
        b.c.set_sync_folder(str(folder))
        a.c.snippets.add("my email", "ada@example.com")
        b.wait(lambda: [e["trigger"] for e in b.c.snippets.list()] == ["my email"], timeout=8000)
        b.wait(lambda: any("Sync changed 1 snippet from another PC" in m for _lvl, m in b.spy.notices()))
        assert not any("Sync changed" in m for _lvl, m in a.spy.notices()), "not on the PC where it was typed"
    finally:
        a.close()
        b.close()


# ============================================================================ S-10: a backup cannot choose the models
def test_a_backup_never_sets_model_names_or_update_settings_even_with_this_pcs_id(tmp_path):
    from tests.test_sync import Clock, Device
    from voxflow import sync
    clock = Clock()
    d = Device(tmp_path, "desk", clock)
    me = d.engine._state["device"]
    backup = tmp_path / "backup.voxflow"
    d.settings.update(cleanup_model="evil/repo:model.gguf", asr_model="evil-asr", update_mode="off", language="fr",
                      start_minimized=True)
    sync.export_profile(d.engine._ds, str(backup), device_id=me)
    fresh = Device(tmp_path, "desk2", clock)
    before = dict(fresh.settings)
    sync.import_profile(fresh.engine._ds, str(backup), mode="replace", device_id=me)
    for key in sync.NEVER_FROM_A_FILE:
        assert fresh.settings[key] == before[key], key
    assert fresh.settings["language"] == "fr" and fresh.settings["start_minimized"] is True, "the rest is restored"
