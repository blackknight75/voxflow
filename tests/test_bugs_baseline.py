"""Regression tests for the baseline bugs (C-xx ids).

Each test asserts the EXPECTED behaviour. While a bug is open its test carries ``@bug("C-xx")`` (a non-strict
xfail); when the owner fixes it the test XPASSes, qa re-verifies, removes the mark and it becomes a normal
regression guard. Only public, stable entry points are used (``rule_clean``, ``Config``, ``HotkeyMachine``,
``Injector``) so the tests survive internal refactors.
"""
from __future__ import annotations

import json
import math

import pytest

from voxflow.cleanup import rule_clean
from voxflow.config import Config, load_config
from voxflow.hotkey import HotkeyMachine, parse_hotkey
from voxflow.inject import Injector

VK_LCTRL, VK_LWIN, VK_ESC, VK_LEFT, VK_A, VK_NUMPAD0 = 0xA2, 0x5B, 0x1B, 0x25, 0x41, 0x60


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


# ============================================================================ config
def test_c01_unparsable_hotkey_in_config_falls_back_to_a_usable_one():
    cfg = Config.from_dict({"hotkey": "foo+bar"})
    assert parse_hotkey(cfg.hotkey)            # must not raise: the app builds its hook from this value
    HotkeyMachine(cfg.hotkey)


@pytest.mark.parametrize("key,value", [
    ("restore_delay_s", -1), ("max_record_s", 0), ("min_record_s", -5), ("cleanup_timeout_s", 0),
    ("history_limit", -3), ("restore_delay_s", float("nan")), ("max_record_s", float("inf")),
])
def test_c09_numeric_settings_are_clamped_to_sane_ranges(key, value):
    got = getattr(Config.from_dict({key: value}), key)
    assert math.isfinite(got), f"{key}={value!r} was accepted as {got!r}"
    if key == "min_record_s":
        assert got >= 0
    else:
        assert got > 0, f"{key}={value!r} was accepted as {got!r}"


def test_c10_config_with_utf8_bom_is_still_read(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("﻿" + json.dumps({"hotkey": "f9", "cleanup_enabled": False}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.hotkey == "f9" and cfg.cleanup_enabled is False


def test_c11_bool_is_not_a_microphone_index():
    assert Config.from_dict({"mic_device": True}).mic_device is None


def test_c12_history_entries_without_a_timestamp_are_normalised_or_dropped(tmp_path):
    from voxflow import config as cfgmod
    History = cfgmod.History
    p = tmp_path / "history.json"
    p.write_text(json.dumps([{"text": "no ts"}, {"text": "bad ts", "ts": "yesterday"},
                             {"text": "fine", "ts": 1700000000.0, "raw": "fine", "ms": 300}]), encoding="utf-8")
    try:
        h = History(path=p)
    except TypeError:                      # new store API: History(dir)
        h = History(tmp_path)
    items = h.snapshot() if hasattr(h, "snapshot") else h.list(limit=None)
    assert any(it["text"] == "fine" for it in items)
    for it in items:
        assert isinstance(it.get("ts"), (int, float)) and not isinstance(it.get("ts"), bool), it


# ============================================================================ hotkey
def _capture(vks) -> tuple[HotkeyMachine, str | None]:
    m = HotkeyMachine()
    m.start_capture()
    res = None
    for vk in vks:
        m.feed(vk, True)
    for vk in reversed(vks):
        res = m.feed(vk, False)
    return m, res.captured


def _combo_is_refused(combo: str | None) -> bool:
    """True if the app has a way to refuse ``combo`` as a hold-to-talk hotkey."""
    if not combo:
        return True
    import voxflow.hotkey as hk
    checker = getattr(hk, "hotkey_problem", None)
    if checker is not None:
        try:
            return bool(checker(combo, {}))
        except TypeError:
            return bool(checker(combo))
    return False


@pytest.mark.parametrize("vks,what", [([VK_A], "a lone letter"), ([0xA0], "a lone Shift"), ([0x20], "a lone Space")])
def test_c03_capture_never_yields_a_hotkey_that_breaks_typing(vks, what):
    _m, combo = _capture(vks)
    assert _combo_is_refused(combo), f"{what} was captured as hotkey {combo!r} and nothing rejects it"


def test_c03_escape_cancels_capture_instead_of_becoming_the_hotkey():
    m, combo = _capture([VK_ESC])
    assert combo != "esc"
    assert not m.capturing


def test_c04_capturing_an_unnamed_key_is_reported_not_silently_dropped():
    m, combo = _capture([VK_NUMPAD0])
    assert combo or m.capturing, "capture ended without a result: the UI would wait for a combo forever"
    m, combo = _capture([0xFF])                      # a key nobody has a name for
    assert combo is None and m.capturing, "an unnamed key is ignored; the capture keeps waiting (then times out)"


def test_c19_the_escape_that_cancels_a_recording_is_swallowed():
    m = HotkeyMachine()
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["start"]
    res = m.feed(VK_ESC, True)
    assert res.events == ["cancel"]
    assert res.swallow, "Esc leaks to the focused application"


def test_c20_a_different_shortcut_containing_the_hotkey_cancels_the_recording():
    m = HotkeyMachine()
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["start"]
    res = m.feed(VK_LEFT, True)                      # Ctrl+Win+Left = switch virtual desktop
    assert "cancel" in res.events
    assert not res.swallow, "the foreign shortcut itself must still reach Windows"


# ============================================================================ inject
class _Clip:
    def __init__(self, text="old"):
        self.text, self.seq, self.restored = text, 1, 0

    def snapshot(self):
        return [(13, self.text)]

    def set_text(self, t):
        self.text = t
        self.seq += 1
        return self.seq

    def sequence(self):
        return self.seq

    def restore(self, snap):
        self.text = snap[0][1]
        self.restored += 1
        return True

    def get_text(self):
        return self.text


class _Keys:
    def __init__(self, mods=False, paste_ok=True):
        self.mods, self.paste_ok, self.pastes, self.typed = mods, paste_ok, 0, []

    def paste(self):
        self.pastes += 1
        return self.paste_ok

    def type_text(self, t):
        self.typed.append(t)
        return True

    def modifiers_down(self):
        return self.mods


def _injector(clip, keys, **kw):
    now = [0.0]
    return Injector(clip, keys, sleep=lambda s: now.__setitem__(0, now[0] + max(0.0, s)), clock=lambda: now[0], **kw)


def test_c06_no_paste_is_sent_while_win_alt_or_shift_is_still_held():
    clip, keys = _Clip(), _Keys(mods=True)
    result = _injector(clip, keys).inject("dictated")
    assert keys.pastes == 0, "Ctrl+V was sent with Win/Alt/Shift down (Win+V opens Clipboard History)"
    assert not result
    assert clip.text == "dictated", "the text must stay on the clipboard so the user can paste it"


def test_c07_failed_paste_leaves_the_text_on_the_clipboard():
    clip, keys = _Clip(), _Keys(paste_ok=False)
    assert not _injector(clip, keys).inject("dictated")
    assert clip.text == "dictated", "app reports 'Copied' but the old clipboard was put back"


def test_c09_negative_restore_delay_cannot_break_the_paste():
    clip, keys = _Clip(), _Keys()
    assert Injector(clip, keys, restore_delay=-1, sleep=lambda s: None).inject("x") is True
    import time
    inj = Injector(clip, keys, restore_delay=-1, modifier_timeout=0.0)       # real time.sleep
    t0 = time.perf_counter()
    assert inj.inject("y") is True
    assert time.perf_counter() - t0 < 1.0
    assert clip.text == "old"


# ============================================================================ cleanup rules
@pytest.mark.parametrize("spoken,must_contain", [
    ("the gap is 5 mm wide", "5 mm"),
    ("is it 5 mm or 6 mm", "6 mm"),
    ("took him to the er last night", "the er "),
    ("to err is human they say", "err"),
])
def test_c25_filler_removal_keeps_real_words(spoken, must_contain):
    assert must_contain in rule_clean(spoken).lower()


@pytest.mark.parametrize("spoken,must_contain", [
    ("open the .gitignore file please", " .gitignore"),
    ("we target .net and use std::vector here", " .net"),
    ("we target .net and use std::vector here", "std::vector"),
])
def test_c26_tidy_spaces_keeps_dotted_tokens_and_scope_operators(spoken, must_contain):
    assert must_contain in rule_clean(spoken).lower()


@pytest.mark.parametrize("spoken", [
    "have a great weekend everyone",
    "do not forget the milk",
    "when we met last week you mentioned the budget",
    "what a great day it was",
    "will smith is an actor",
])
def test_c27_statements_are_not_turned_into_questions(spoken):
    assert not rule_clean(spoken).endswith("?")


@pytest.mark.parametrize("spoken", ["what is the capital of france", "can you send me the file",
                                    "are we still meeting tomorrow", "how do i reset my password"])
def test_c27_real_questions_still_get_a_question_mark(spoken):
    assert rule_clean(spoken).endswith("?")


def test_c28_every_sentence_start_is_capitalised():
    assert rule_clean("hello. how are you") .startswith("Hello. How are you")


def test_c29_ie_is_not_capitalised():
    assert " i.e. " in rule_clean("use the second one i.e. the blue one")


@pytest.mark.parametrize("spoken", ["we signed a new line of credit", "my main bullet point is cost"])
def test_c30_formatting_words_inside_prose_are_not_commands(spoken):
    out = rule_clean(spoken)
    assert "\n" not in out and "- " not in out


def test_c30_real_formatting_commands_still_work():
    assert rule_clean("first item new line second item") == "First item\nSecond item."
    assert rule_clean("shopping list bullet point milk bullet point eggs").split("\n")[1:] == ["- Milk", "- Eggs"]


def test_c31_a_leading_line_break_command_is_kept():
    assert rule_clean("new line hello there").startswith("\n")


def test_c32_wait_between_numbers_is_not_always_a_correction():
    assert "one" in rule_clean("go to page one wait 5 minutes").lower()


# ============================================================================ ui fakes
def test_u10_fake_recorder_stop_without_start_returns_no_audio(monkeypatch):
    from tests.helpers import fake_time
    from voxflow.ui import fakes
    # keep the allocation tiny if the bug is present; only this module's clock is replaced, never the global one
    monkeypatch.setattr(fakes, "time", fake_time(perf_counter=lambda: 10.0))
    assert fakes.FakeRecorder().stop().size == 0
