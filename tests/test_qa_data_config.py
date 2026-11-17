"""QA "data" fixes in the settings file (``voxflow.config``).

D12  an invalid value is reset *and reported*; "false" / "off" / "100" written as text are read as meant
D13  an older VoxFlow keeps the settings and the version a newer one wrote
"""
from __future__ import annotations

import json

from voxflow import config as C


def _load(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return C.load_config(path), path


# =========================================================================== D12
def test_text_values_are_read_as_the_setting_means_them(tmp_path):
    cfg, _ = _load(tmp_path, {"save_history": "false", "history_limit": "100", "sound_volume": " 0.8 ",
                              "mute_media": "ON", "reduced_motion": "no", "sounds_enabled": "1"})
    assert cfg.save_history is False, "a privacy setting written as text is honoured, not ignored"
    assert (cfg.history_limit, cfg.sound_volume) == (100, 0.8)
    assert (cfg.mute_media, cfg.reduced_motion, cfg.sounds_enabled) == (True, False, True)
    assert C.last_load_problem == ""


def test_invalid_values_are_reset_and_the_user_is_told(tmp_path):
    cfg, _ = _load(tmp_path, {"save_history": "maybe", "theme": "pink", "history_limit": "lots",
                              "language": "en", "hotkey": "ctrl+alt+space"})
    assert (cfg.save_history, cfg.theme, cfg.history_limit) == (True, "system", 5000)
    assert (cfg.language, cfg.hotkey) == ("en", "ctrl+alt+space")
    assert C.last_load_problem == ("Some settings in your settings file were not valid and are back to their "
                                   "defaults: save history, history limit, theme.")
    _load(tmp_path, {"theme": "dark"})
    assert C.last_load_problem == "", "the next good load clears it"
    assert C.Config.from_dict({"theme": "pink"}).theme == "system", "callers that do not ask are not told"


def test_a_bad_setting_is_announced_at_start(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch, config={"save_history": "nope", "sound_volume": "0.25"})
    try:
        notices = [args for name, args in rig.startup_events if name == "notice"]
        assert ("warning", "Some settings in your settings file were not valid and are back to their defaults: "
                           "save history.") in notices
        assert rig.c.get("sound_volume") == 0.25
    finally:
        rig.close()


# =========================================================================== D13
def test_settings_from_a_newer_version_survive_a_save(tmp_path):
    cfg, path = _load(tmp_path, {"config_version": 3, "hotkey": "ctrl+alt+space", "future_setting": {"x": 1}})
    assert cfg.config_version == 3 and C.last_load_problem == ""
    cfg.theme = "dark"
    C.save_config(cfg, path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["config_version"] == 3, "never lowered: the newer version does not migrate again"
    assert on_disk["future_setting"] == {"x": 1} and on_disk["theme"] == "dark"
    assert "future_setting" not in cfg.to_dict(), "the extras are never treated as settings of this version"

    old, path2 = _load(tmp_path, {"config_version": 1, "history_limit": 200})
    assert old.config_version == C.CONFIG_VERSION and old.history_limit == 5000, "older files still migrate"
    C.save_config(C.Config(), path2)                              # a Config built in code has no extras
    assert json.loads(path2.read_text(encoding="utf-8"))["config_version"] == C.CONFIG_VERSION
