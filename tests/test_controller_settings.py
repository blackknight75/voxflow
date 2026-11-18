"""Controller: settings (validate / apply live / persist / announce), hotkey capture, status, microphones and
hot-plug health checks, model download progress, lifecycle and shutdown hygiene, demo mode."""
from __future__ import annotations

import json
import threading
import time

import pytest

from tests.controller_rig import CTRL, WIN, Rig, TRecorder, make_rig
from voxflow import config as cfgmod
from voxflow import controller as ctl_mod
from voxflow.controller import Controller


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


def saved(rig) -> dict:
    return json.loads((rig.data_dir / "config.json").read_text(encoding="utf-8"))


# =========================================================================== get / set
def test_get_and_set_roundtrip_persist_and_announce_once(rig):
    assert rig.c.get("language") == "auto" and rig.c.config.language == "auto"
    rig.c.set("language", " DE ")
    assert rig.c.get("language") == "de" and saved(rig)["language"] == "de"
    assert rig.spy.of("config_changed") == [("language", "de")]
    assert len(rig.spy.of("status_changed")) == 1
    rig.spy.clear()
    rig.c.set("language", "de")                                   # same value: nothing happens
    assert rig.spy.events == []
    with pytest.raises(ValueError, match="Unknown setting"):
        rig.c.get("nope")


@pytest.mark.parametrize("key,value", [
    ("language", "klingon"), ("hotkey", ""), ("hotkey", "a"), ("hotkey", "ctrl+c"), ("sound_volume", "loud"),
    ("double_tap_lock", "yes"), ("nope", 1), ("config_version", 3), ("theme", "neon"), ("mic_device", 2.5),
])
def test_invalid_values_raise_a_human_message_and_change_nothing(rig, key, value):
    before = rig.c.config.to_dict()
    disk = saved(rig)
    with pytest.raises(ValueError) as exc:
        rig.c.set(key, value)
    assert len(str(exc.value)) > 8 and str(exc.value).rstrip()[-1] in ".!?"
    assert rig.c.config.to_dict() == before and saved(rig) == disk
    assert rig.spy.of("config_changed") == []


def test_every_documented_setting_can_be_set_through_the_controller(rig):
    samples = {
        "hotkey": "ctrl+alt+space", "command_hotkey": "ctrl+alt+c", "paste_last_hotkey": "ctrl+alt+v",
        "undo_hotkey": "ctrl+alt+z", "double_tap_lock": False, "cleanup_enabled": False, "smart_formatting": False,
        "auto_learn": False, "language": "fr", "whisper_mode": True, "mic_device": "USB Mic", "sounds_enabled": False,
        "sound_volume": 0.8, "mute_media": True, "launch_at_login": True, "start_minimized": True,
        "save_history": False, "history_retention_days": 30, "history_limit": 100, "restore_clipboard": False,
        "restore_delay_s": 1.5, "show_idle_pill": True, "live_preview": False, "theme": "light",
        "reduced_motion": True,
        "overlay_position": [10, 20], "ui_state": {"page": "history"}, "onboarding_done": True,
        "asr_device": "cpu", "cleanup_device": "vulkan", "cleanup_timeout_s": 6.0, "command_timeout_s": 30.0,
        "min_record_s": 0.5, "max_record_s": 60.0, "update_mode": "off", "update_install_when": "time",
        "update_time": "04:30",
    }
    untested = set(cfgmod.defaults()) - set(samples) - {"config_version", "asr_model", "cleanup_model"}
    assert not untested, f"new settings without a controller test: {untested}"
    for key, value in samples.items():
        rig.c.set(key, value)
        assert rig.c.get(key) == value, key
    assert [k for k, _v in rig.spy.of("config_changed")] == list(samples)
    reloaded = cfgmod.load_config(rig.data_dir / "config.json").to_dict()
    for key, value in samples.items():
        assert reloaded[key] == value, key


def test_settings_take_effect_immediately(rig):
    c = rig.c
    c.set("restore_clipboard", False)
    c.set("restore_delay_s", 1.5)
    assert (rig.injector.restore, rig.injector.restore_delay) == (False, 1.5)
    c.set("sounds_enabled", False)
    c.set("sound_volume", 0.25)
    assert (c.sounds.enabled, c.sounds.volume) == (False, 0.25)
    c.set("cleanup_timeout_s", 6.0)
    c.set("command_timeout_s", 30.0)
    assert rig.cleaner.timeout_s == 6.0 and c.command.timeout_s == 30.0
    c.set("double_tap_lock", False)
    assert c.listener.machine.double_tap is False
    c.set("hotkey", "ctrl+alt+space")
    c.set("paste_last_hotkey", "")
    c.set("undo_hotkey", "ctrl+alt+z")
    assert c.listener.machine.combo == "ctrl+alt+space"
    assert c.listener.binding("paste_last") is None and c.listener.binding("undo").combo == "ctrl+alt+z"
    for i in range(30):
        c.history.add(f"entry {i}")
    c.set("history_limit", 10)
    assert c.history.count() == 10
    c.set("history_retention_days", 7)
    assert c.history.retention_days == 7


def test_hotkey_conflicts_are_refused_with_the_other_actions_name(rig):
    with pytest.raises(ValueError, match="already used for Command mode"):
        rig.c.set("hotkey", "win+alt")
    with pytest.raises(ValueError, match="overlaps with the Dictation shortcut"):
        rig.c.set("command_hotkey", "ctrl+shift+win")
    assert rig.c.validate_hotkey("paste_last_hotkey", "ctrl+win") == "Ctrl + Win is already used for Dictation."
    assert rig.c.validate_hotkey("paste_last_hotkey", "ctrl+alt+v") == ""
    assert rig.c.validate_hotkey("hotkey", "ctrl+win") == "", "an action never conflicts with its own shortcut"
    assert rig.c.validate_hotkey("hotkey", "") == "The dictation shortcut cannot be empty."
    assert rig.c.validate_hotkey("undo_hotkey", "") == "" and rig.c.validate_hotkey("undo_hotkey", None) == ""
    assert rig.c.validate_hotkey("language", "ctrl+win") == "That is not a shortcut setting."
    assert "typing" in rig.c.validate_hotkey("command_hotkey", "a")
    rig.c.set("command_hotkey", "")                               # switching the other action off frees its combo
    rig.c.set("hotkey", "alt+win")
    assert rig.c.get("hotkey") == "alt+win"


def test_pretty_hotkey_and_choice_lists(rig):
    assert rig.c.pretty_hotkey("ctrl+win") == "Ctrl + Win" and rig.c.pretty_hotkey("") == "Not set"
    assert rig.c.languages()[0] == ("auto", "Auto-detect") and len(rig.c.languages()) == 26
    assert rig.c.tones() == [("formal", "Formal"), ("casual", "Casual"), ("very_casual", "Very casual")]
    assert [c for c, _label in rig.c.categories()] == ["personal", "work", "email", "other"]
    assert rig.c.preview_tone("Hello there. See you.", "very_casual") == "hello there. see you"
    rig.c.languages().clear()
    assert len(rig.c.languages()) == 26


def test_launch_at_login_goes_through_the_autostart_backend(rig):
    rig.c.set("launch_at_login", True)
    assert rig.autostart.calls == [True] and rig.c.get("launch_at_login") is True
    rig.autostart.refuse = True
    with pytest.raises(ValueError, match="Windows did not let VoxFlow"):
        rig.c.set("launch_at_login", False)
    assert rig.c.get("launch_at_login") is True and saved(rig)["launch_at_login"] is True, "nothing changes on refusal"


def test_real_mode_mirrors_the_registry_state_at_start(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False, start=False, config={"launch_at_login": False})
    try:
        assert rig.c.get("launch_at_login") is False
    finally:
        rig.close()
    rig2 = None
    try:
        TRecorder.reset()
        from tests.controller_rig import TAsr, TAutostart, TCleaner, TInjector, TMixer
        rig2 = Controller(fake=False, data_dir=tmp_path / "data", install_hook=False,
                          modules=(TRecorder, TAsr(), TCleaner()), injector=TInjector(), sound_backend=lambda w: None,
                          mute_backend=TMixer(), context_fn=lambda: None, autostart=TAutostart(enabled=True))
        assert rig2.get("launch_at_login") is True, "the Run key, not the config file, is the source of truth"
    finally:
        if rig2 is not None:
            rig2.shutdown()


def test_a_setting_that_cannot_be_saved_stays_active_and_the_user_is_told(rig, monkeypatch):
    monkeypatch.setattr(cfgmod, "save_config", lambda *a, **k: (_ for _ in ()).throw(PermissionError("read-only")))
    rig.c.set("language", "fr")
    assert rig.c.get("language") == "fr"
    assert rig.spy.notices() == [("warning", "Your change is active, but it could not be saved to disk and will be "
                                             "lost when VoxFlow closes.")]
    assert rig.spy.of("config_changed") == [("language", "fr")]


def test_reset_settings_restores_defaults_but_keeps_data_and_onboarding(rig):
    c = rig.c
    c.dictionary.add("Kubernetes")
    c.history.add("keep me")
    for key, value in {"hotkey": "ctrl+alt+space", "command_hotkey": "ctrl+win", "language": "fr", "theme": "light",
                       "sound_volume": 0.9, "launch_at_login": True, "onboarding_done": True,
                       "ui_state": {"page": "history"}, "overlay_position": [5, 6], "mic_device": "USB Mic",
                       "whisper_mode": True}.items():
        c.set(key, value)
    rig.spy.clear()
    c.reset_settings()
    defaults = cfgmod.defaults()
    kept = {"onboarding_done": True, "ui_state": {"page": "history"}, "overlay_position": [5, 6]}
    for key, value in defaults.items():
        assert c.get(key) == kept.get(key, value), key
    assert c.listener.machine.combo == "ctrl+win" and c.listener.binding("command").combo == "alt+win"
    assert rig.autostart.value is False and rig.rec.device is None and rig.rec.level_boost == 1.0
    assert c.dictionary.terms() == ["Kubernetes"] and c.history.count() == 1
    changed = [k for k, _v in rig.spy.of("config_changed")]
    assert set(changed) >= {"hotkey", "language", "theme", "sound_volume", "launch_at_login", "mic_device"}
    final = dict(rig.spy.of("config_changed"))
    assert final["command_hotkey"] == "alt+win" and final["hotkey"] == "ctrl+win"
    assert cfgmod.load_config(rig.data_dir / "config.json").hotkey == "ctrl+win"


def test_c60_reset_announces_each_setting_once_and_only_if_it_changed(rig):
    rig.c.set("language", "fr")
    rig.spy.clear()
    rig.c.reset_settings()
    changed = [k for k, _v in rig.spy.of("config_changed")]
    assert changed == ["language"], f"settings that did not change were announced (and rewritten): {changed}"


def test_reset_survives_a_refused_setting(rig, caplog):
    rig.c.set("launch_at_login", True)
    rig.autostart.refuse = True
    rig.c.set("language", "fr")
    with caplog.at_level("WARNING"):
        rig.c.reset_settings()
    assert rig.c.get("language") == "auto" and rig.c.get("launch_at_login") is True
    assert "launch_at_login not restored" in caplog.text


def test_changing_a_model_setting_reloads_the_models(rig):
    loads = (rig.asr.loads, rig.cleaner.loads)
    rig.c.set("asr_device", "cpu")
    rig.wait(lambda: (rig.asr.loads, rig.cleaner.loads) == (loads[0] + 1, loads[1] + 1))
    rig.wait(lambda: [n for n, _s in rig.spy.of("model_status_changed")].count("asr") >= 2)
    assert [s["state"] for n, s in rig.spy.of("model_status_changed") if n == "asr"] == ["loading", "ready"]
    with pytest.raises(ValueError, match="Unknown model"):
        rig.c.model_status("llm")
    status = rig.c.model_status("asr")
    status["state"] = "hacked"
    assert rig.c.model_status("asr")["state"] == "ready", "model_status() returns a copy"


def test_corrupt_config_file_is_reported_at_startup(tmp_path, qtbot, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text("{ broken", encoding="utf-8")
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        notices = [args for name, args in rig.startup_events if name == "notice"]
        assert len(notices) == 1 and notices[0][0] == "warning" and "config.json.bad" in notices[0][1]
        assert (data / "config.json.bad").read_text(encoding="utf-8") == "{ broken"
        assert rig.c.get("hotkey") == "ctrl+win"
    finally:
        rig.close()


def test_stores_live_in_the_data_dir_and_stats_are_seeded_from_existing_history(tmp_path, qtbot, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "history.json").write_text(json.dumps([
        {"ts": time.time() - 60, "text": "an old version one transcript", "raw": "", "ms": 500}]), encoding="utf-8")
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        c = rig.c
        assert c.data_dir == data and c.log_path == data / "logs" / "voxflow.log"
        assert c.history.count() == 1 and c.last_transcript["text"] == "an old version one transcript"
        assert c.stats.summary()["total_words"] == 5, "first run after the upgrade derives stats from the history"
        for store, name in ((c.dictionary, "dictionary"), (c.snippets, "snippets"), (c.history, "history"),
                            (c.stats, "stats"), (c.styles, "styles")):
            assert store.path == data / f"{name}.json"
        assert c.version == __import__("voxflow").__version__ and c.is_demo is False
    finally:
        rig.close()


# =========================================================================== status
def test_status_priorities(rig):
    c = rig.c
    assert c.status() == {"level": "ok", "title": "Ready", "detail": "Hold Ctrl + Win to dictate."}
    c._warnings["mic"] = "“Yeti” is not connected, so the system default microphone is being used."
    assert c.status()["level"] == "warning" and "Yeti" in c.status()["detail"]
    c._models["cleanup"]["state"] = "loading"
    c._warnings.clear()
    assert c.status()["detail"] == "Hold Ctrl + Win to dictate. The cleanup model is still loading."
    c.set("cleanup_enabled", False)
    assert c.status()["detail"] == "Hold Ctrl + Win to dictate."
    c._models["asr"] = {"state": "downloading", "detail": "Downloading… 12 MB of 2,550 MB", "progress": 0.01, "backend": ""}
    assert c.status() == {"level": "loading", "title": "Downloading speech model…",
                          "detail": "Downloading… 12 MB of 2,550 MB"}
    c.set_paused(True)
    assert c.status()["level"] == "paused"
    c._problems["mic"] = "No microphone found."
    assert c.status() == {"level": "error", "title": "Microphone problem", "detail": "No microphone found."}
    c._problems["asr"] = "The speech model could not be loaded."
    assert c.status()["title"] == "Speech model unavailable"
    c._problems["hook"] = "The global shortcuts could not be registered."
    assert c.status()["title"] == "Shortcuts unavailable", "the most fundamental problem is shown first"


def test_hook_failure_is_reported_as_a_problem(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, install_hook=True)     # the harness refuses the real hook
    try:
        status = rig.c.status()
        assert status["level"] == "error" and status["title"] == "Shortcuts unavailable"
        assert "could not be registered" in status["detail"] and "Restart VoxFlow" in status["detail"]
        rig.c.begin_hotkey_capture()
        assert rig.spy.of("hotkey_captured") == [("",)] and rig.spy.notices()[0][0] == "info"
    finally:
        rig.close()


# =========================================================================== hotkey capture
@pytest.fixture
def capture_rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    r.c._install_hook = True                                    # pretend the hook is running; keys come from the rig
    yield r
    r.close()


def test_capture_reports_a_normalised_combo(capture_rig):
    rig = capture_rig
    rig.c.begin_hotkey_capture()
    rig.keys((0xA4, True), (0xA2, True), (0x20, True), (0x20, False), (0xA2, False), (0xA4, False))
    assert rig.spy.of("hotkey_captured") == [("ctrl+alt+space",)]
    assert rig.c.get("hotkey") == "ctrl+win", "capturing only reports; the UI decides what to do with it"
    rig.pause()
    rig.dictate("dictation still works after a capture")
    assert rig.injector.pasted == ["Dictation still works after a capture."]


def test_capture_of_an_unusable_combo_explains_why_and_reports_nothing(capture_rig):
    rig = capture_rig
    rig.c.begin_hotkey_capture()
    rig.keys((0x41, True), (0x41, False))                       # a lone "A"
    assert rig.spy.of("hotkey_captured") == [("",)]
    assert rig.spy.notices() == [("warning", "A would get in the way of typing. Add Ctrl, Alt or Win.")]


def test_capture_cancel_escape_and_timeout_all_report_an_empty_combo(capture_rig):
    rig = capture_rig
    rig.c.begin_hotkey_capture()
    rig.c.cancel_hotkey_capture()
    rig.c.cancel_hotkey_capture()                               # second cancel: nothing more
    rig.c.begin_hotkey_capture()
    rig.keys((0x1B, True), (0x1B, False))                       # Esc
    rig.c.begin_hotkey_capture()
    rig.keys()                                                  # installs the rig clock
    rig.now += 60
    rig.tick()                                                  # the hook thread's watchdog
    assert rig.spy.of("hotkey_captured") == [("",), ("",), ("",)]
    rig.keys((CTRL, True), (WIN, True))
    assert rig.c.state == "listening", "after a cancelled capture the shortcut works again"
    rig.c.cancel()


def test_capture_aborts_a_running_recording_and_stray_results_are_ignored(capture_rig):
    rig = capture_rig
    rig.press()
    rig.c.begin_hotkey_capture()
    assert rig.c.state == "idle" and not rig.rec.is_recording
    rig.c.cancel_hotkey_capture()
    rig.spy.clear()
    rig.c._on_captured("ctrl+alt+x")                            # a capture result nobody asked for
    assert rig.spy.of("hotkey_captured") == []


# =========================================================================== microphones
def test_microphone_list_in_fake_mode_and_a_disconnected_choice(rig):
    from voxflow import fakes
    mics = rig.c.microphones()
    assert mics[0] == {"id": None, "name": f"System default ({fakes.FAKE_MICROPHONES[0]})", "is_default": True}
    assert [m["id"] for m in mics[1:]] == list(fakes.FAKE_MICROPHONES)
    rig.c.set("mic_device", "Blue Snowball")
    assert rig.c.microphones()[-1] == {"id": "Blue Snowball", "name": "Blue Snowball (not connected)",
                                       "is_default": False}
    rig.c.refresh_microphones()
    assert rig.spy.of("microphones_changed")[-1] == (rig.c.microphones(),)


def test_level_and_waveform_for_the_overlay(rig):
    assert rig.c.level() == 0.42 and rig.c.recent_samples(256).shape == (256,)
    rig.rec.level = lambda: 1 / 0
    rig.rec.recent_samples = lambda n: 1 / 0
    assert rig.c.level() == 0.0 and not rig.c.recent_samples(64).any()


def test_level_monitor_emits_only_while_the_mic_test_runs(rig):
    rig.c.set_level_monitor(True)
    rig.wait(lambda: len(rig.spy.of("level_changed")) >= 3)
    assert rig.spy.of("level_changed")[0] == (0.42,)
    rig.c.set_level_monitor(False)
    assert rig.spy.of("level_changed")[-1] == (0.0,), "the meter drops to zero when monitoring stops"
    count = len(rig.spy.of("level_changed"))
    rig.settle(100)
    assert len(rig.spy.of("level_changed")) == count
    rig.press()
    rig.settle(150)
    assert len(rig.spy.of("level_changed")) == count, "P7: the Flow bar polls level() itself; no 30 Hz signal while dictating"
    rig.c.cancel()


# ---- real-mode microphone handling (fake=False, every back-end still injected) ---------------------------
@pytest.fixture
def real_rig(tmp_path, qtbot, monkeypatch):
    """Not in fake mode: microphone health checks and model-cache logic are live, their OS calls are patched."""
    state = {"count": 2, "devices": [(1, "Yeti Nano"), (2, "Jabra Evolve2")], "default": "Yeti Nano", "rescans": 0}
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: state["count"])
    monkeypatch.setattr(ctl_mod.audiomod, "list_input_devices", lambda: list(state["devices"]))
    monkeypatch.setattr(ctl_mod.audiomod, "default_input_name", lambda: state["default"])
    monkeypatch.setattr(ctl_mod.audiomod, "rescan_devices", lambda: state.__setitem__("rescans", state["rescans"] + 1))
    monkeypatch.setattr(ctl_mod.modelhub, "is_cached", lambda repo, filename=None: True)
    r = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    r.os = state
    yield r
    r.close()


def test_real_microphone_list_uses_names_as_ids(real_rig):
    mics = real_rig.c.microphones()
    assert mics == [{"id": None, "name": "System default (Yeti Nano)", "is_default": True},
                    {"id": "Yeti Nano", "name": "Yeti Nano", "is_default": False},
                    {"id": "Jabra Evolve2", "name": "Jabra Evolve2", "is_default": False}]
    real_rig.os["default"] = ""
    assert real_rig.c.microphones()[0]["name"] == "System default"


def test_plugging_a_microphone_in_refreshes_the_list(real_rig):
    rig = real_rig
    rig.os["devices"].append((3, "USB Mic"))
    rig.os["count"] = 3
    rig.c._health_tick()
    assert rig.os["rescans"] == 1 and rig.rec.closed is False and rig.rec.reopened == 1
    (mics,), = rig.spy.of("microphones_changed")
    assert [m["id"] for m in mics] == [None, "Yeti Nano", "Jabra Evolve2", "USB Mic"]
    rig.spy.clear()
    rig.c._health_tick()                                        # nothing changed: nothing announced
    assert rig.spy.of("microphones_changed") == []


def test_device_change_during_a_recording_does_not_disturb_it(real_rig):
    rig = real_rig
    rig.press()
    rig.os["count"] = 3
    rig.c._health_tick()
    assert rig.c.state == "listening" and rig.os["rescans"] == 0 and rig.rec.is_recording
    rig.release()
    rig.wait(lambda: rig.c.state == "done")


def test_microphone_unplugged_mid_sentence_uses_what_was_said(real_rig):
    rig = real_rig
    rig.press()
    rig.rec.alive = False
    rig.c._t_start -= 2.0
    rig.c._health_tick()
    assert ("warning", "The microphone was disconnected. What you said so far is being used.") in rig.spy.notices()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["Hello world."]
    rig.keys((WIN, False), (CTRL, False))                       # the late key release is harmless
    rig.settle()
    assert len(rig.injector.pasted) == 1


def test_idle_microphone_loss_and_recovery(real_rig):
    rig = real_rig
    rig.rec.alive = False
    rig.rec.error = OSError("Invalid device [PaErrorCode -9996]")
    rig.c._health_tick()
    assert rig.rec.reopened == 1
    status = rig.c.status()
    assert status["level"] == "error" and status["title"] == "Microphone problem" and "PaErrorCode" not in status["detail"]
    rig.rec.error = None                                        # plugged back in
    rig.spy.clear()
    rig.c._health_tick()
    assert rig.rec.reopened == 2 and rig.rec.alive
    assert ("success", "Microphone connected again.") in rig.spy.notices()
    assert rig.c.status()["level"] == "ok"


def test_missing_chosen_microphone_is_a_warning_while_the_default_is_used(real_rig):
    rig = real_rig
    rig.c.set("mic_device", "Blue Snowball")
    rig.rec.using_fallback = True
    rig.c._check_mic()
    status = rig.c.status()
    assert status["level"] == "warning"
    assert status["detail"] == "“Blue Snowball” is not connected, so the system default microphone is being used."
    rig.rec.using_fallback = False
    rig.c._check_mic()
    assert rig.c.status()["level"] == "ok"


def test_recorder_that_cannot_be_created_is_a_problem_not_a_crash(real_rig):
    rig = real_rig
    TRecorder.fail_init = OSError("No default input device available")
    rig.c.set("mic_device", "Jabra Evolve2")
    assert rig.c.status()["title"] == "Microphone problem" and "No microphone found." in rig.c.status()["detail"]
    TRecorder.fail_init = None
    rig.c.set("mic_device", None)
    assert rig.c.status()["level"] == "ok"


def test_refresh_survives_a_failing_reopen(real_rig, caplog):
    rig = real_rig
    rig.rec.error = OSError("device busy")
    with caplog.at_level("WARNING"):
        rig.c.refresh_microphones()
    assert "microphone refresh" in caplog.text and rig.spy.of("microphones_changed")
    assert rig.c.status()["title"] == "Microphone problem"


def test_history_retention_is_enforced_hourly(real_rig):
    rig = real_rig
    rig.c.history.retention_days = 1
    rig.c.history._items.append(dict(rig.c.history.add("old"), ts=time.time() - 3 * 86400, id="old-one"))
    rig.c._health_ticks = 1799
    rig.c._health_tick()
    assert rig.c.history.get("old-one") is None


# =========================================================================== model download progress
def test_download_progress_is_reported_while_a_model_is_fetched(tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 1)
    monkeypatch.setattr(ctl_mod.modelhub, "is_cached", lambda repo, filename=None: not repo)
    sizes = iter([100_000_000, 400_000_000, 900_000_000])
    monkeypatch.setattr(ctl_mod.modelhub, "downloaded_bytes", lambda repo: next(sizes, 2_000_000_000))
    real_watcher = ctl_mod.modelhub.DownloadWatcher
    monkeypatch.setattr(ctl_mod.modelhub, "DownloadWatcher",
                        lambda repo, cb, interval_s=0.5: real_watcher(repo, cb, interval_s=0.01))
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False, start=False)
    try:
        rig.asr.load_gate = threading.Event()
        rig.c.start()
        rig.wait(lambda: any(s["progress"] for n, s in rig.spy.of("model_status_changed") if n == "asr"))
        states = [s for n, s in rig.spy.of("model_status_changed") if n == "asr"]
        assert states[0] == {"state": "downloading", "detail": "Downloading the speech model (one time)…",
                             "progress": None, "backend": ""}
        with_progress = [s for s in states if s["progress"]]
        assert with_progress[0]["detail"] == "Downloading… 100 MB of 2,550 MB"
        assert 0 < with_progress[0]["progress"] < 0.05
        assert rig.c.status()["title"] == "Downloading speech model…"
        rig.asr.load_gate.set()
        rig.wait(lambda: rig.c.model_status("asr")["state"] == "ready")
        assert all(s["progress"] <= 0.99 for s in with_progress), "100 % is only shown when the model is loaded"
    finally:
        rig.asr.load_gate.set()
        rig.close()


def test_a_superseded_load_cannot_overwrite_the_newer_status(rig):
    gate = threading.Event()
    rig.asr.load_gate = gate
    rig.c.reload_models()                                       # load #2 hangs
    rig.wait(lambda: rig.c.model_status("asr")["state"] == "loading")
    rig.asr.load_gate = None
    rig.c.reload_models()                                       # load #3 finishes at once
    rig.wait(lambda: rig.c.model_status("asr")["state"] == "ready")
    rig.asr.fail_load = RuntimeError("stale failure")
    gate.set()                                                  # load #2 now fails, late
    rig.settle(100)
    assert rig.c.model_status("asr")["state"] == "ready", "a stale loader thread reported over the current state"


# =========================================================================== lifecycle
def test_start_is_idempotent_and_announces_the_initial_state(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert ("state_changed", ("idle", "dictate", "")) in rig.startup_events
        assert any(name == "status_changed" for name, _a in rig.startup_events)
        loads = rig.asr.loads
        rig.c.start()
        rig.settle()
        assert rig.asr.loads == loads
    finally:
        rig.close()


def test_shutdown_releases_everything_and_is_idempotent(tmp_path, qtbot, monkeypatch):
    before = {t.name for t in threading.enumerate()}
    rig = make_rig(tmp_path, qtbot, monkeypatch, config={"mute_media": True})
    rig.dictate("one sentence before quitting")
    rig.pause()
    rig.press()                                                 # quit in the middle of a recording
    assert rig.mixer.muted
    rig.c.shutdown()
    assert not rig.rec.is_recording and rig.rec.closed
    assert not rig.mixer.muted and rig.injector.closed == 1
    assert not rig.c._worker.is_alive()
    assert not any(t.isActive() for t in (rig.c._idle_timer, rig.c._level_timer, rig.c._cap_timer, rig.c._health_timer))
    rig.c.shutdown()                                            # second call: nothing
    assert rig.injector.closed == 1
    rig.c._on_hotkey("dictate", "start")                        # a late hotkey event after shutdown is ignored
    assert rig.rec.starts == 2
    deadline = time.time() + 3
    def leftovers():
        return {t.name for t in threading.enumerate() if t.name.startswith("voxflow")} - before
    while leftovers() and time.time() < deadline:
        time.sleep(0.02)
    assert leftovers() == set(), "threads left running after shutdown"


def test_shutdown_before_start_and_emergency_restore(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False, config={"mute_media": True})
    rig.c.shutdown()                                            # never started: must not hang or raise
    rig2 = make_rig(tmp_path / "two", qtbot, monkeypatch) if (tmp_path / "two").mkdir() is None else None
    try:
        rig2.c.set("mute_media", True)
        rig2.press()
        rig2.c.emergency_restore()                              # what the crash handler calls
        assert not rig2.mixer.muted and rig2.injector.closed == 1
        rig2.mixer.unmute = lambda ids: 1 / 0
        rig2.injector.close = lambda: 1 / 0
        rig2.c.emergency_restore()                              # must never raise from a crash handler
    finally:
        rig2.mixer.unmute = lambda ids: None
        rig2.injector.close = lambda: None
        rig2.close()


def test_second_launch_asks_the_running_instance_to_show_itself(rig):
    rig.c.request_activate()
    assert rig.spy.of("activate_requested") == [()]


def test_copy_text_and_folder_actions(rig, monkeypatch):
    assert rig.c.copy_text("for the clipboard") is True and rig.injector.copied == ["for the clipboard"]
    assert rig.c.copy_text("") is False
    rig.injector.copy_ok = False
    assert rig.c.copy_text("x") is False
    rig.injector.copy_text = lambda text: 1 / 0
    assert rig.c.copy_text("x") is False
    assert [level for level, _m in rig.spy.notices()] == ["error"] * 3
    assert "clipboard is in use" in rig.spy.notices()[0][1]

    opened = []
    monkeypatch.setattr(ctl_mod.system, "open_folder", lambda path: opened.append(path) or True)
    rig.c.open_data_folder()
    rig.c.open_log_folder()
    assert opened == [rig.data_dir, rig.data_dir / "logs"]
    rig.spy.clear()
    monkeypatch.setattr(ctl_mod.system, "open_folder", lambda path: False)
    rig.c.open_data_folder()
    rig.c.open_log_folder()
    assert [level for level, _m in rig.spy.notices()] == ["error", "error"]


# =========================================================================== hermetic defaults + demo
def test_fake_controller_with_default_backends_is_fully_hermetic(tmp_path, qtbot):
    """`Controller(fake=True, data_dir=tmp, install_hook=False)` exactly as documented for tests:
    the harness raises if it reaches the real clipboard, keyboard, microphone, registry or profile."""
    from voxflow import fakes
    c = Controller(fake=True, data_dir=tmp_path, install_hook=False)
    try:
        assert isinstance(c.injector, fakes.FakeInjector) and isinstance(c._recorder, fakes.FakeRecorder)
        c.start()
        qtbot.waitUntil(lambda: c.model_status("asr")["state"] == "ready" and c.model_status("cleanup")["state"] == "ready",
                        timeout=5000)
        c.set("launch_at_login", True)
        c.set("mute_media", True)
        c._on_hotkey("dictate", "start")
        assert c.state == "listening"
        c._t_start -= 1.0
        qtbot.wait(350)                                         # the fake recorder measures real time
        c._on_hotkey("dictate", "stop")
        qtbot.waitUntil(lambda: c.state == "done", timeout=5000)
        assert c.injector.pasted == ["So this is a demo transcript you know with no real model."]
        qtbot.waitUntil(lambda: c.history.count() == 1 and c.stats.summary()["sessions"] == 1, timeout=5000)
        qtbot.waitUntil(lambda: (tmp_path / "stats.json").exists(), timeout=5000)   # P8: filed after "done"
        qtbot.waitUntil(lambda: (tmp_path / "history.json").exists(), timeout=5000)  # written behind (QA D17)
        assert c.history.list()[0]["app_exe"] == "notepad.exe"
        assert c.paste_last() and c.copy_text("x")
        assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".json") == [
            "config.json", "history.json", "stats.json"]
    finally:
        c.shutdown()


def test_demo_controller_is_self_contained_and_seeded(qtbot, _isolated_profile):
    c = Controller(demo=True)
    data = c.data_dir
    try:
        assert c.is_demo and data.name.startswith("voxflow-demo-") and data.is_dir()
        assert not (_isolated_profile / "VoxFlow").exists(), "demo mode must not create or touch the user's profile"
        assert c.history.count() > 20 and len(c.dictionary.list()) == 5 and len(c.dictionary.suggestions()) == 2
        assert len(c.snippets.list()) == 3 and c.stats.summary()["total_words"] > 500
        assert c.stats.summary()["streak_days"] >= 1 and len(c.stats.top_apps()) >= 4
        assert sum(1 for d in c.stats.daily(14) if d["words"] == 0) == 2, "the chart shows a couple of quiet days"
        assert c.last_transcript is not None
        c.start()
        qtbot.waitUntil(lambda: c.model_status("asr")["state"] == "ready", timeout=5000)
        assert c.status()["level"] == "demo"
        assert c.listener._thread is None, "demo mode never installs the keyboard hook"

        seen = []
        c.state_changed.connect(lambda *a: seen.append(a))
        for state, mode, message in (("listening", "dictate", ""), ("locked", "dictate", ""),
                                     ("processing", "command", ""), ("done", "command", "Rewritten"),
                                     ("error", "dictate", "Didn't catch that"), ("idle", "dictate", "")):
            c.demo_set_state(state, mode, message)
            assert (c.state, c.mode, c.message) == (state, mode, message)
        assert len(seen) == 6
        c.demo_set_state("done", "dictate", "Pasted")
        qtbot.wait(ctl_mod.DONE_MS + 150)
        assert c.state == "done", "screenshots decide when a demo state ends"
        with pytest.raises(ValueError):
            c.demo_set_state("exploding")
        with pytest.raises(ValueError):
            c.demo_set_state("idle", "telepathy")

        n = c.history.count()
        entry = c.demo_add_transcript()
        assert c.history.count() == n + 1 and c.last_transcript == entry
        custom = c.demo_add_transcript("A custom line.")
        assert custom["text"] == "A custom line." and custom["raw"] == "a custom line"
        c.set("save_history", False)
        c.demo_add_transcript()
        assert c.history.count() == n + 2

        c.demo_set_model("asr", "downloading", "Downloading… 312 MB of 2,550 MB", 0.12)
        assert c.model_status("asr")["progress"] == 0.12 and c.status()["level"] == "loading"
        c.demo_set_model("asr", "failed", "The speech model could not be loaded.")
        assert c.status()["title"] == "Speech model unavailable"
        c.demo_set_model("asr", "ready", "Simulated (demo).", None, "demo")
        assert c.status()["level"] == "demo"
        with pytest.raises(ValueError):
            c.demo_set_model("llm", "ready")
    finally:
        c.shutdown()
    assert not data.exists(), "the throw-away demo profile is removed on exit"


def test_demo_helpers_do_nothing_outside_demo_mode(rig):
    rig.c.demo_set_state("listening")
    rig.c.demo_set_model("asr", "failed")
    assert rig.c.demo_add_transcript() is None
    assert rig.c.state == "idle" and rig.c.model_status("asr")["state"] == "ready" and rig.c.history.count() == 0


def test_helper_texts():
    assert ctl_mod._short(RuntimeError("boom")) == "RuntimeError: boom"
    assert ctl_mod._short(ValueError()) == "ValueError"
    long = ctl_mod._short(RuntimeError("x" * 500))
    assert len(long) == 118 and long.endswith("…")
    assert ctl_mod._backend_text("cuda") == "Running on the GPU (CUDA)."
    assert ctl_mod._backend_text("cpu") == "Running on the CPU."
    assert ctl_mod._backend_text("") == "Ready." and ctl_mod._backend_text("rules") == "Basic rules only."
    assert ctl_mod.STATES == ("idle", "listening", "locked", "processing", "done", "error")
    assert ctl_mod.MODES == ("dictate", "command")


def test_ui01_a_timed_pause_ends_by_itself_and_any_resume_cancels_it(rig, qtbot):
    c = rig.c
    seen = []
    c.status_changed.connect(seen.append)
    c.set_paused(True, for_s=0.05)
    assert c.paused and c.paused_until is not None and not c.listener.enabled
    at = time.strftime("%H:%M", time.localtime(c.paused_until))
    assert c.status()["title"] == f"Paused until {at}" and at in c.status()["detail"]
    qtbot.waitUntil(lambda: not c.paused, timeout=2000)
    assert c.paused_until is None and c.listener.enabled and c.status()["title"] == "Ready"
    # QA UI-01: pause for a while, resume another way, pause again: the old deadline must not un-pause it
    c.set_paused(True, for_s=0.05)
    c.set_paused(False)
    c.set_paused(True)
    qtbot.wait(200)
    assert c.paused and c.paused_until is None and c.status()["title"] == "Paused"
    n = len(seen)
    c.set_paused(True, for_s=3600)                       # already paused: becoming a timed pause still tells the UI
    assert len(seen) == n + 1 and c.status()["title"].startswith("Paused until ")
    c.set_paused(True)                                   # ... and so does going back to an open-ended pause
    assert len(seen) == n + 2 and c.paused_until is None and not c._unpause_timer.isActive()
    c.set_paused(True)
    assert len(seen) == n + 2, "nothing changed, nothing sent"
    c.set_paused(False)
