"""Guards for the few spots excluded from coverage as unreachable, plus the one that turned out reachable.

Each ``# pragma: no cover`` / ``# pragma: no branch`` in ``voxflow`` rests on a premise; where that premise is
data (not control flow), it is asserted here so the exclusion cannot silently go stale.
"""
from __future__ import annotations

import ctypes
import importlib.util
import itertools
import sys

from voxflow import cleanup, hotkey, inject


def test_proper_and_common_word_sets_are_disjoint():
    """cleanup: the ``elif low in _COMMON`` fallback only runs for a word in both sets."""
    assert cleanup._PROPER & cleanup._COMMON == set()


def test_spoken_underscore_and_slash_chains_settle_well_inside_the_pass_limit():
    """cleanup._smart_line: the ``range(6)`` bound is a safety net, real chains reach a fixed point first."""
    words = ["a", "underscore", "slash", "forward", "b.", "-x"]
    for n in range(1, 7):
        for combo in itertools.product(words, repeat=n):
            t, passes = " ".join(combo), 0
            while True:
                new = cleanup._UNDERSCORE_RE.sub(lambda m: m.group(1) + "_", t)
                new = cleanup._SLASH_RE.sub(lambda m: m.group(1) + "/", new)
                passes += 1
                if new == t:
                    break
                t = new
            assert passes <= 2, combo
    assert cleanup._smart_line("user underscore id underscore two") == "user_id_two"


def test_every_key_the_capture_accepts_has_a_name_in_the_combo():
    """hotkey: a captured key set is never rendered as an empty combo."""
    named = [vk for vk in range(256) if hotkey.vk_group_name(vk) is not None]
    assert named
    assert all(hotkey.combo_from_vks({vk}) for vk in named)


def test_inject_imports_without_touching_win32_on_another_platform(monkeypatch):
    """inject: off Windows the module must import (constants, Injector class) without loading user32/kernel32."""
    def no_windll(*a, **k):
        raise AssertionError("WinDLL must not be loaded off Windows")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "WinDLL", no_windll, raising=False)
    spec = importlib.util.spec_from_file_location("voxflow_inject_offwin", inject.__file__)
    mod = importlib.util.module_from_spec(spec)          # a private copy: the real module is left untouched
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "user32") and not hasattr(mod, "kernel32")
    assert mod.EXTRA_MARKER == inject.EXTRA_MARKER and hasattr(mod, "Injector")
