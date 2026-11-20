"""voxflow.config: validation fall-throughs and the failure paths of the atomic save / tolerant load."""
from __future__ import annotations

import json
import math
import os

import pytest

from tests.helpers import fake_time
from voxflow import config as cfgmod
from voxflow.config import Config, load_config, save_config


# ---------------------------------------------------------------- validation
def test_a_number_without_a_range_must_be_finite(monkeypatch):
    # every numeric setting shipped today has a range; a future one without must still refuse inf
    monkeypatch.setitem(cfgmod._DEFAULTS, "future_gain", 1.5)
    assert Config.validate("future_gain", 2) == 2.0
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="Future gain must be a number"):
            Config.validate("future_gain", bad)


def test_infinity_is_clamped_into_the_range_of_a_ranged_number():
    assert Config.validate("sound_volume", math.inf) == 1.0
    assert Config.validate("sound_volume", -math.inf) == 0.0


def test_a_text_setting_cannot_be_blank():
    assert Config.validate("asr_model", "  my-model  ") == "my-model"
    for bad in ("", "   ", None, 3):
        with pytest.raises(ValueError, match="Asr model cannot be empty"):
            Config.validate("asr_model", bad)


def test_a_setting_of_an_unconstrained_type_is_passed_through(monkeypatch):
    monkeypatch.setitem(cfgmod._DEFAULTS, "future_list", [])
    value = ["a", 1]
    assert Config.validate("future_list", value) is value


# ---------------------------------------------------------------- _replace / _atomic_write
def test_replace_gives_up_after_about_two_seconds_of_locked_attempts(tmp_path, monkeypatch):
    attempts, sleeps = [], []

    def locked(src, dst):
        attempts.append((src, dst))
        raise PermissionError("in use")

    monkeypatch.setattr(cfgmod.os, "replace", locked)
    monkeypatch.setattr(cfgmod, "time", fake_time(sleep=sleeps.append))
    with pytest.raises(PermissionError):
        cfgmod._replace("x.tmp", tmp_path / "config.json")
    assert len(attempts) == 24 and len(sleeps) == 23
    assert 1.9 < sum(sleeps) < 2.2, "about 2 s, as documented (it was 3.6 s: QA R-01)"
    assert sleeps[0] == 0.005 and max(sleeps) == 0.1, "back-off doubles and is capped at 100 ms"


def test_replace_succeeds_once_the_reader_lets_go(tmp_path, monkeypatch):
    real_replace, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use")
        real_replace(src, dst)

    monkeypatch.setattr(cfgmod.os, "replace", flaky)
    monkeypatch.setattr(cfgmod, "time", fake_time(sleep=lambda s: None))
    target = tmp_path / "config.json"
    cfgmod._atomic_write(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello" and len(calls) == 3
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.json"], "no temp file left behind"


def test_save_works_on_a_file_system_without_fsync(tmp_path, monkeypatch):
    def no_fsync(fd):
        raise OSError("fsync not supported")

    monkeypatch.setattr(cfgmod.os, "fsync", no_fsync)
    target = tmp_path / "config.json"
    save_config(Config(theme="dark"), target)
    assert json.loads(target.read_text(encoding="utf-8"))["theme"] == "dark"


def test_failed_save_removes_its_temp_file_and_keeps_the_old_config(tmp_path, monkeypatch):
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")

    def refuse(tmp, path):
        raise OSError("disk full")

    monkeypatch.setattr(cfgmod, "_replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        cfgmod._atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "{}"
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.json"]


def test_failed_save_reports_the_original_error_even_if_the_temp_file_cannot_be_removed(tmp_path, monkeypatch):
    target = tmp_path / "config.json"

    def refuse(tmp, path):
        raise OSError("disk full")

    def locked(path):
        raise PermissionError("temp file locked")

    monkeypatch.setattr(cfgmod, "_replace", refuse)
    monkeypatch.setattr(cfgmod.os, "unlink", locked)
    with pytest.raises(OSError, match="disk full"):
        cfgmod._atomic_write(target, "new")
    assert not target.exists()


# ---------------------------------------------------------------- load_config
def test_unreadable_config_falls_back_to_defaults_even_when_the_backup_copy_fails(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text("{ not json", encoding="utf-8")
    real_write_bytes = type(path).write_bytes

    def write_bytes(self, data):
        if self.name.endswith(".bad"):
            raise PermissionError("read-only folder")
        return real_write_bytes(self, data)

    monkeypatch.setattr(type(path), "write_bytes", write_bytes)
    cfg = load_config(path)
    assert cfg == Config()
    assert cfgmod.last_load_problem == "Your settings file could not be read, so defaults are in use.", \
        "no copy could be kept, so none is promised"
    assert not (tmp_path / "config.json.bad").exists()
    assert path.read_text(encoding="utf-8") == "{ not json", "the damaged file itself is left alone"


def test_missing_config_in_a_read_only_place_still_loads_defaults(tmp_path, monkeypatch):
    def refuse(cfg, path=None):
        raise PermissionError("read-only")

    monkeypatch.setattr(cfgmod, "save_config", refuse)
    cfg = load_config(tmp_path / "nowhere" / "config.json")
    assert cfg == Config() and cfgmod.last_load_problem == ""
