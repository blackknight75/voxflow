"""Config v2: defaults, per-key validation, clamping, migration, load/save robustness."""
from __future__ import annotations

import json
import math

import pytest

from voxflow import config as C
from voxflow.config import Config, load_config, save_config


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


# ---------------------------------------------------------------- defaults / documentation parity
DOCUMENTED_DEFAULTS = {            # the documented config defaults
    "hotkey": "ctrl+win", "command_hotkey": "alt+win", "paste_last_hotkey": "alt+shift+z", "undo_hotkey": "",
    "double_tap_lock": True, "cleanup_enabled": True, "smart_formatting": True, "auto_learn": True,
    "language": "auto", "whisper_mode": False, "mic_device": None, "sounds_enabled": True, "sound_volume": 0.5,
    "mute_media": False, "launch_at_login": False, "start_minimized": False, "save_history": True,
    "history_retention_days": 0, "history_limit": 5000, "restore_clipboard": True, "restore_delay_s": 0.5,
    "show_idle_pill": False, "live_preview": True, "theme": "system", "reduced_motion": False,
    "overlay_position": None, "ui_state": {},
    "onboarding_done": False, "update_mode": "notify", "update_install_when": "quit", "update_time": "03:00",
    "asr_model": "nemo-parakeet-tdt-0.6b-v3", "asr_device": "auto",
    "cleanup_model": None, "cleanup_device": "auto", "cleanup_timeout_s": 4.0, "command_timeout_s": 20.0,
    "min_record_s": 0.3, "max_record_s": 120.0, "config_version": 2,
}


def test_defaults_match_the_documented_contract():
    assert Config().to_dict() == DOCUMENTED_DEFAULTS
    assert C.defaults() == DOCUMENTED_DEFAULTS
    C.defaults()["ui_state"]["x"] = 1
    assert C.defaults()["ui_state"] == {}, "defaults() must hand out a fresh copy"
    assert Config().ui_state is not Config().ui_state


def test_every_default_passes_its_own_validation():
    for key, value in DOCUMENTED_DEFAULTS.items():
        assert Config.validate(key, value) == value, key


def test_default_hotkeys_do_not_clash_with_each_other():
    from voxflow.hotkey import combos_clash, hotkey_problem
    combos = [v for k, v in DOCUMENTED_DEFAULTS.items() if k in C.HOTKEY_KEYS and v]
    for i, a in enumerate(combos):
        assert hotkey_problem(a) == ""
        for b in combos[i + 1:]:
            assert not combos_clash(a, b), (a, b)
    assert set(C.HOTKEY_LABELS) == set(C.HOTKEY_KEYS)


def test_languages_are_auto_plus_the_25_parakeet_v3_languages():
    codes = [code for code, _ in C.LANGUAGES]
    assert codes[0] == "auto" and len(codes) == 26 and len(set(codes)) == 26
    assert {"en", "de", "fr", "es", "it", "pl", "uk", "ru", "nl", "sv"} <= set(codes)
    assert all(label and label[0].isupper() for _, label in C.LANGUAGES)
    assert C.LANGUAGE_CODES == tuple(codes)


# ---------------------------------------------------------------- validate()
@pytest.mark.parametrize("key,value,expected", [
    ("hotkey", "Win + Control", "ctrl+win"), ("hotkey", "F9", "f9"), ("command_hotkey", "", ""),
    ("command_hotkey", "  ", ""), ("paste_last_hotkey", "shift+alt+Z", "alt+shift+z"),
    ("mic_device", None, None), ("mic_device", "  Yeti Nano ", "Yeti Nano"), ("mic_device", "   ", None),
    ("mic_device", 3, 3),
    ("cleanup_model", None, None), ("cleanup_model", " repo:file.gguf ", "repo:file.gguf"),
    ("overlay_position", None, None), ("overlay_position", [10.7, -20], [10, -20]),
    ("overlay_position", (5, 6), [5, 6]),
    ("ui_state", {"page": "home", "geo": [1, 2]}, {"page": "home", "geo": [1, 2]}),
    ("language", " DE ", "de"), ("theme", "Dark", "dark"), ("asr_device", "CPU", "cpu"), ("asr_device", "WebGPU", "webgpu"),
    ("cleanup_device", " Vulkan ", "vulkan"), ("cleanup_device", "auto", "auto"),
    ("double_tap_lock", False, False),
    ("sound_volume", 0.25, 0.25), ("sound_volume", 7, 1.0), ("sound_volume", -1, 0.0),
    ("sound_volume", float("inf"), 1.0), ("max_record_s", float("-inf"), 5.0),
    ("history_retention_days", 30.9, 30), ("history_retention_days", -5, 0), ("history_retention_days", 99999, 3650),
    ("history_limit", 3, 10), ("history_limit", 10 ** 9, 100000),
    ("restore_delay_s", 0, 0.05), ("restore_delay_s", 60, 5.0), ("cleanup_timeout_s", 0.1, 0.5),
    ("command_timeout_s", 1, 2.0), ("command_timeout_s", 500, 120.0),
    ("min_record_s", 0, 0.05), ("max_record_s", 1, 5.0), ("max_record_s", 10 ** 6, 3600.0),
    ("asr_model", "  my-model ", "my-model"),
])
def test_validate_normalises_and_clamps(key, value, expected):
    got = Config.validate(key, value)
    assert got == expected and type(got) is type(expected)


@pytest.mark.parametrize("key,value", [
    ("hotkey", ""), ("hotkey", None), ("hotkey", 5), ("hotkey", "a"), ("hotkey", "shift"), ("hotkey", "foo+bar"),
    ("hotkey", "ctrl+c"), ("hotkey", "ctrl+a+b"), ("hotkey", "ctrl+esc"), ("command_hotkey", "space"),
    ("mic_device", True), ("mic_device", 2.5), ("mic_device", ["x"]),
    ("cleanup_model", ""), ("cleanup_model", 5),
    ("overlay_position", [1]), ("overlay_position", [1, 2, 3]), ("overlay_position", ["a", "b"]),
    ("overlay_position", [True, False]), ("overlay_position", "1,2"),
    ("ui_state", []), ("ui_state", "x"), ("ui_state", {"k": object()}), ("ui_state", {1.5: {2, 3}}),
    ("language", "xx"), ("language", 5), ("theme", "neon"), ("asr_device", "gpu"),
    ("cleanup_device", "webgpu"), ("cleanup_device", None),
    ("double_tap_lock", 1), ("double_tap_lock", "true"), ("cleanup_enabled", None),
    ("sound_volume", "loud"), ("sound_volume", True), ("sound_volume", float("nan")), ("history_limit", None),
    ("asr_model", ""), ("asr_model", "   "), ("asr_model", 7),
    ("no_such_key", 1),
])
def test_validate_rejects_with_a_message_fit_for_a_toast(key, value):
    with pytest.raises(ValueError) as exc:
        Config.validate(key, value)
    msg = str(exc.value)
    assert len(msg) > 8 and msg.rstrip()[-1] in ".!?", f"not a sentence: {msg!r}"
    assert not msg[0].islower(), f"not a sentence: {msg!r}"
    assert "Traceback" not in msg and "object at 0x" not in msg


@pytest.mark.parametrize("key,value", [
    ("history_limit", float("inf")), ("history_retention_days", float("-inf")),
    ("overlay_position", [float("inf"), 0]),
])
def test_c43_infinity_is_rejected_or_clamped_never_an_overflow(key, value):
    try:
        got = Config.validate(key, value)
    except ValueError:
        return
    flat = got if isinstance(got, list) else [got]
    assert all(math.isfinite(v) for v in flat)


def test_c43_config_file_with_infinity_still_loads(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"history_limit": Infinity, "history_retention_days": -Infinity, '
                 '"overlay_position": [Infinity, 1], "hotkey": "f9"}', encoding="utf-8")
    cfg = load_config(p)                       # must not raise: the app starts from this call
    assert cfg.hotkey == "f9" and 10 <= cfg.history_limit <= 100000


# ---------------------------------------------------------------- from_dict / migration
def test_from_dict_keeps_valid_values_and_defaults_the_rest():
    cfg = Config.from_dict({
        "hotkey": "ctrl+alt+space", "language": "fr", "theme": "pink", "sound_volume": "x", "save_history": "yes",
        "history_limit": 250, "unknown_future_key": 1, "mic_device": "USB Mic", "ui_state": {"a": 1},
        "config_version": 99})
    assert (cfg.hotkey, cfg.language, cfg.history_limit, cfg.mic_device, cfg.ui_state) == (
        "ctrl+alt+space", "fr", 250, "USB Mic", {"a": 1})
    assert (cfg.theme, cfg.sound_volume, cfg.save_history) == ("system", 0.5, True)
    # a newer VoxFlow wrote this file: its version is never lowered and its keys are kept for the next save (QA D13)
    assert cfg.config_version == 99 and cfg._extras == {"unknown_future_key": 1}
    assert not hasattr(cfg, "unknown_future_key") and "unknown_future_key" not in cfg.to_dict()


@pytest.mark.parametrize("junk", [None, [], "x", 5, {}])
def test_from_dict_with_junk_is_all_defaults(junk):
    assert Config.from_dict(junk).to_dict() == DOCUMENTED_DEFAULTS


def test_v1_config_is_migrated():
    v1 = {"hotkey": "ctrl+win", "double_tap_lock": False, "cleanup_enabled": False, "mic_device": 2,
          "asr_model": "nemo-parakeet-tdt-0.6b-v3", "asr_device": "auto", "cleanup_model": None,
          "cleanup_timeout_s": 4.0, "min_record_s": 0.3, "max_record_s": 120.0, "restore_delay_s": 0.5,
          "history_limit": 200, "save_history": True, "restore_clipboard": False, "show_idle_pill": False}
    cfg = Config.from_dict(v1)
    assert cfg.history_limit == 5000, "the old 200-entry cap was a v1 UI limitation"
    assert (cfg.double_tap_lock, cfg.cleanup_enabled, cfg.restore_clipboard, cfg.show_idle_pill) == (False,) * 4
    assert cfg.mic_device == 2, "a legacy device index is still accepted"
    assert cfg.command_hotkey == "alt+win" and cfg.config_version == 2
    assert v1["history_limit"] == 200, "migrate() must not mutate its input"


@pytest.mark.parametrize("stored,expected", [
    ({"history_limit": 200}, 5000),                                  # no version -> v1
    ({"history_limit": 200, "config_version": 1}, 5000),
    ({"history_limit": 200, "config_version": 2}, 200),              # a deliberate v2 choice is kept
    ({"history_limit": 200, "config_version": True}, 5000),
    ({"history_limit": 200, "config_version": "2"}, 5000),
    ({"history_limit": 300, "config_version": 0}, 300),
])
def test_migrate_history_limit(stored, expected):
    assert C.migrate(stored)["history_limit"] == expected
    assert C.migrate(stored)["config_version"] == 2
    assert Config.from_dict(stored).history_limit == expected


@pytest.mark.parametrize("stored,expected", [
    ({"hotkey": "alt+win"}, ("alt+win", "", "alt+shift+z")),                      # command default is taken
    ({"hotkey": "ctrl+win", "command_hotkey": "ctrl+win"}, ("ctrl+win", "alt+win", "alt+shift+z")),
    ({"command_hotkey": "ctrl+alt+win"}, ("ctrl+win", "alt+win", "alt+shift+z")),  # superset of dictation
    ({"paste_last_hotkey": "alt+win"}, ("ctrl+win", "alt+win", "alt+shift+z")),
    ({"hotkey": "alt+shift+z", "command_hotkey": "alt+win", "paste_last_hotkey": "alt+shift+z"},
     ("alt+shift+z", "alt+win", "")),
])
def test_two_actions_never_share_a_shortcut_after_loading(stored, expected):
    cfg = Config.from_dict(stored)
    assert (cfg.hotkey, cfg.command_hotkey, cfg.paste_last_hotkey) == expected


# ---------------------------------------------------------------- load / save
def test_first_run_creates_the_file_in_the_isolated_profile(data_dir):
    assert C.app_dir() == data_dir and C.config_path() == data_dir / "config.json"
    assert C.history_path() == data_dir / "history.json"
    cfg = load_config()
    assert cfg.to_dict() == DOCUMENTED_DEFAULTS
    assert json.loads((data_dir / "config.json").read_text(encoding="utf-8")) == DOCUMENTED_DEFAULTS
    assert C.last_load_problem == ""


def test_app_dir_without_appdata_falls_back_to_the_home_config(monkeypatch, tmp_path):
    monkeypatch.delenv("APPDATA")
    monkeypatch.setattr(C.Path, "home", classmethod(lambda cls: tmp_path))
    assert C.app_dir() == tmp_path / ".config" / "VoxFlow" and C.app_dir().is_dir()


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "sub" / "config.json"                             # parent is created
    cfg = Config(hotkey="f9", language="de", ui_state={"geo": [1, 2, 3]}, overlay_position=[4, 5],
                 mic_device="Yeti", sound_volume=0.8)
    save_config(cfg, p)
    assert load_config(p) == cfg
    assert not list(p.parent.glob("*.tmp")), "atomic write must not leave temp files behind"


@pytest.mark.parametrize("content", ["{ broken", "", "[1, 2, 3]", '"text"', "null", "\x00\x00"])
def test_unreadable_config_gives_defaults_keeps_a_copy_and_explains(tmp_path, content):
    p = tmp_path / "config.json"
    p.write_text(content, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.to_dict() == DOCUMENTED_DEFAULTS
    assert (tmp_path / "config.json.bad").read_text(encoding="utf-8") == content
    assert p.read_text(encoding="utf-8") == content, "loading must not overwrite the damaged file"
    assert "config.json.bad" in C.last_load_problem and C.last_load_problem.endswith(".")
    p.write_text("{}", encoding="utf-8")
    load_config(p)
    assert C.last_load_problem == "", "the message must reset after a clean load"


def test_config_in_invalid_encoding_is_handled_like_a_corrupt_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_bytes(b'{"hotkey": "f9\xff\xfe"}')
    assert load_config(p).hotkey == "ctrl+win" and (tmp_path / "config.json.bad").exists()


def test_missing_file_in_an_unwritable_place_still_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_atomic_write", lambda *a, **k: (_ for _ in ()).throw(PermissionError("read-only")))
    assert load_config(tmp_path / "nope" / "config.json").to_dict() == DOCUMENTED_DEFAULTS


def test_atomic_write_failure_cleans_up_and_keeps_the_old_file(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    save_config(Config(hotkey="f9"), p)
    monkeypatch.setattr(C.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError("locked by AV")))
    with pytest.raises(PermissionError):
        save_config(Config(hotkey="f10"), p)
    assert load_config(p).hotkey == "f9"
    assert [f.name for f in tmp_path.iterdir() if f.is_file()] == ["config.json"], "the temp file must be removed"


def test_history_is_still_importable_from_config_for_old_callers():
    from voxflow.store import History
    assert C.History is History
    with pytest.raises(AttributeError):
        C.NoSuchThing
