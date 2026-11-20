"""Data-layer findings of the 1.0 QA pass, settings side (``voxflow.config``): the copy of an unreadable settings
file is never overwritten by the next one (D-09), and JSON nested too deeply is an unreadable file (DL-10)."""
from __future__ import annotations

from voxflow import config as C
from voxflow.config import Config, load_config


def bad_copies(folder) -> dict:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("config.json.bad*"))}


def test_the_first_copy_of_an_unreadable_settings_file_is_kept_and_later_ones_are_numbered(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{ the user's own settings, cut off", encoding="utf-8")
    assert load_config(p) == Config()
    assert C.last_load_problem == ("Your settings file could not be read, so defaults are in use. "
                                   "The old file was kept as config.json.bad.")
    p.write_text("{ damaged again", encoding="utf-8")
    load_config(p)
    assert C.last_load_problem.endswith("The old file was kept as config.json.bad.2.")
    assert bad_copies(tmp_path) == {"config.json.bad": "{ the user's own settings, cut off",
                                    "config.json.bad.2": "{ damaged again"}, "the first copy is not overwritten"
    load_config(p)                                         # the same damaged file at the next start: no new copy
    assert C.last_load_problem.endswith("The old file was kept as config.json.bad.2.")
    assert len(bad_copies(tmp_path)) == 2
    for n in (3, 4, 5):
        p.write_text(f"{{ damage {n}", encoding="utf-8")
        load_config(p)
        assert C.last_load_problem.endswith(f"The old file was kept as config.json.bad.{n}.")
    p.write_text("{ damage 6", encoding="utf-8")           # every name is taken: the newest copy is replaced
    load_config(p)
    assert C.last_load_problem.endswith("The old file was kept as config.json.bad.5.")
    assert bad_copies(tmp_path) == {
        "config.json.bad": "{ the user's own settings, cut off", "config.json.bad.2": "{ damaged again",
        "config.json.bad.3": "{ damage 3", "config.json.bad.4": "{ damage 4", "config.json.bad.5": "{ damage 6"}
    p.write_text("{ the user's own settings, cut off", encoding="utf-8")
    load_config(p)
    assert C.last_load_problem.endswith("The old file was kept as config.json.bad."), "an identical copy is named"
    assert len(bad_copies(tmp_path)) == 5


def test_a_settings_file_nested_too_deeply_gives_defaults_and_a_copy(tmp_path):
    p = tmp_path / "config.json"
    deep = '{"ui_state": ' + "[" * 100_000 + "]" * 100_000 + "}"
    p.write_text(deep, encoding="utf-8")
    assert load_config(p) == Config()
    assert (tmp_path / "config.json.bad").read_text(encoding="utf-8") == deep
    assert "could not be read" in C.last_load_problem and p.read_text(encoding="utf-8") == deep


def test_a_number_too_large_for_a_float_costs_that_setting_only(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"hotkey": "f9", "sound_volume": 1' + "0" * 400 + ', "history_limit": 1' + "0" * 400 + "}",
                 encoding="utf-8")
    cfg = load_config(p)
    assert cfg.hotkey == "f9" and not list(tmp_path.glob("config.json.bad*"))
    assert (cfg.sound_volume, cfg.history_limit) == (0.5, 5000)
    assert C.last_load_problem == ("Some settings in your settings file were not valid and are back to their defaults: "
                                   "sound volume, history limit.")
