"""Hotkey validation helpers, press-mode machines, capture safety and the multi-binding listener routing.

The listener is driven through its public ``feed`` / ``tick`` (what the hook thread calls); the real
``WH_KEYBOARD_LL`` hook is never installed.
"""
from __future__ import annotations

import pytest

from voxflow import hotkey as hk
from voxflow.hotkey import (HotkeyListener, HotkeyMachine, combos_clash, hotkey_problem, normalize_hotkey,
                            parse_hotkey, pretty_hotkey)

def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


CTRL, WIN, ALT, SHIFT, ESC = 0xA2, 0x5B, 0xA4, 0xA0, 0x1B
Z, A, LEFT, F9, SPACE = 0x5A, 0x41, 0x25, 0x78, 0x20


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def down_up(target, *vks):
    """Press the keys in order, release in reverse; returns every Result."""
    out = [target.feed(vk, True) for vk in vks]
    out += [target.feed(vk, False) for vk in reversed(vks)]
    return out


# ---------------------------------------------------------------- naming helpers
@pytest.mark.parametrize("raw,expected", [
    ("Win + Ctrl", "ctrl+win"), ("CONTROL+windows", "ctrl+win"), ("shift+alt+Z", "alt+shift+z"),
    ("cmd+option+space", "alt+win+space"), ("ctrl+ctrl+win", "ctrl+win"), ("f9", "f9"), ("", ""),
    ("escape+ctrl", "ctrl+esc"), ("super+return", "win+enter"),
])
def test_normalize_hotkey(raw, expected):
    assert normalize_hotkey(raw) == expected


@pytest.mark.parametrize("combo,expected", [("ctrl+win", "Ctrl + Win"), ("alt+shift+z", "Alt + Shift + Z"),
                                            ("f9", "F9"), ("ctrl+space", "Ctrl + Space"),
                                            ("ctrl+pagedown", "Ctrl + Pagedown")])
def test_pretty_hotkey(combo, expected):
    assert pretty_hotkey(combo) == expected


def test_new_key_names_parse_and_capture_roundtrip():
    for name, vk in [("num0", 0x60), ("num9", 0x69), ("numadd", 0x6B), ("pause", 0x13), ("menu", 0x5D),
                     ("scrolllock", 0x91), ("f24", 0x87)]:
        assert parse_hotkey(f"ctrl+{name}") == [hk._MOD_GROUPS["ctrl"], frozenset({vk})]
        assert hk.combo_from_vks({CTRL, vk}) == f"ctrl+{name}"
    assert len(set(hk._NAMED.values())) == len(hk._NAMED), "two names for one key would make captures ambiguous"


@pytest.mark.parametrize("combo", ["ctrl+win", "alt+win", "alt+shift+z", "ctrl+alt+space", "f9", "shift+f5",
                                   "ctrl+shift+d", "ctrl+alt+shift+win", "win+f1", "ctrl+num0", "pause"[:0] or "f24"])
def test_usable_shortcuts_have_no_problem(combo):
    assert hotkey_problem(combo) == ""


@pytest.mark.parametrize("combo,fragment", [
    ("", "Press a key combination"), ("+", "Press a key combination"),
    ("foo", "can't use that key"), ("ctrl+nope", "can't use that key"),
    ("a", "get in the way of typing"), ("shift+a", "get in the way of typing"), ("space", "get in the way of typing"),
    ("enter", "get in the way of typing"), ("shift", "single modifier"), ("win", "single modifier"),
    ("esc", "Esc cancels"), ("ctrl+esc", "Esc cancels"), ("ctrl+a+b", "at most one other key"),
    ("ctrl+c", "already used by Windows"), ("V+CTRL", "already used by Windows"), ("alt+tab", "already used"),
    ("win+l", "already used"), ("alt+f4", "already used"), ("ctrl+shift", "already used"), ("alt+shift", "already used"),
])
def test_unusable_shortcuts_are_explained(combo, fragment):
    msg = hotkey_problem(combo)
    assert fragment in msg, msg
    assert msg.rstrip().endswith("."), msg


def test_hotkey_problem_reports_conflicts_with_other_actions():
    others = {"Command mode": "alt+win", "Paste last transcript": "alt+shift+z", "Undo last paste": ""}
    assert hotkey_problem("ctrl+win", others) == ""
    assert hotkey_problem("win+alt", others) == "Alt + Win is already used for Command mode."
    msg = hotkey_problem("ctrl+alt+win", others)
    assert "overlaps with the Command mode shortcut (Alt + Win)" in msg
    assert "overlaps with the Paste last transcript shortcut" in hotkey_problem("ctrl+alt+shift+z", others)
    assert hotkey_problem("ctrl+alt+d", {"Broken": "not+a+key"}) == "", "an unparsable other binding cannot clash"


@pytest.mark.parametrize("a,b,clash", [
    ("ctrl+win", "win+ctrl", True), ("ctrl+win", "ctrl+alt+win", True), ("ctrl+alt+win", "ctrl+win", True),
    ("ctrl+win", "alt+win", False), ("alt+shift+z", "alt+shift+x", False), ("f9", "ctrl+f9", True),
    ("ctrl+win", "garbage+key", False), ("", "ctrl+win", False),
])
def test_combos_clash(a, b, clash):
    assert combos_clash(a, b) is clash


# ---------------------------------------------------------------- press-mode machine
def test_press_machine_fires_once_per_press_and_swallows_the_letter():
    m = HotkeyMachine("alt+shift+z", double_tap=False, mode="press")
    assert m.feed(ALT, True).events == [] and m.feed(SHIFT, True).events == []
    res = m.feed(Z, True)
    assert res.events == ["press"] and res.swallow, "the letter must not reach the focused app"
    assert not m.active
    assert m.feed(Z, True).events == [] and m.feed(Z, True).swallow, "auto-repeat: no second press, still swallowed"
    up = m.feed(Z, False)
    assert up.swallow and up.events == []
    again = m.feed(Z, True)
    assert again.events == ["press"] and again.swallow, "Alt+Shift held, Z tapped again: a second paste, like Ctrl+V V"
    assert m.feed(Z, False).swallow
    alt_up = m.feed(ALT, False)
    assert alt_up.mask, "a lone Alt release would focus the menu bar: it must be masked"
    m.feed(SHIFT, False)
    assert m.state == "idle"
    assert [r.events for r in down_up(m, ALT, SHIFT, Z)][2] == ["press"]


def test_press_machine_never_locks_or_emits_hold_events():
    clock = Clock()
    m = HotkeyMachine("f9", double_tap=True, clock=clock, mode="press")
    events = []
    for _ in range(3):
        for r in down_up(m, F9):
            events += r.events
        clock.t += 0.05
    assert events == ["press", "press", "press"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        HotkeyMachine("f9", mode="toggle")


def test_blocked_machine_does_not_activate():
    m = HotkeyMachine("ctrl+win")
    m.blocked = True
    assert [r.events for r in down_up(m, CTRL, WIN)] == [[], [], [], []]
    m.blocked = False
    assert down_up(m, CTRL, WIN)[1].events == ["start"]


# ---------------------------------------------------------------- hold machine: cancel paths (C-19, C-20)
def test_escape_cancel_swallows_key_down_and_up_then_recovers():
    m = HotkeyMachine("ctrl+win")
    m.feed(CTRL, True)
    m.feed(WIN, True)
    esc = m.feed(ESC, True)
    assert esc.events == ["cancel"] and esc.swallow
    assert m.feed(ESC, False).swallow
    assert m.feed(WIN, False).events == [] and m.feed(CTRL, False).events == [], "no stop after a cancel"
    assert m.state == "idle"
    assert m.feed(ESC, True).swallow is False, "Esc outside a recording belongs to the user"
    m.feed(ESC, False)
    assert down_up(m, CTRL, WIN)[1].events == ["start"]


def test_escape_cancels_hands_free_lock_too():
    clock = Clock()
    m = HotkeyMachine("ctrl+win", clock=clock)
    for _ in range(2):
        down_up(m, CTRL, WIN)
        clock.t += 0.1
    assert m.state == "locked"
    assert m.feed(ESC, True).events == ["cancel"]
    m.feed(ESC, False)
    assert m.state == "idle"


def test_foreign_key_does_not_cancel_hands_free_mode():
    clock = Clock()
    m = HotkeyMachine("ctrl+win", clock=clock)
    for _ in range(2):
        down_up(m, CTRL, WIN)
        clock.t += 0.1
    assert m.state == "locked"
    assert [r.events for r in down_up(m, A)] == [[], []], "typing while hands-free is locked must not cancel"
    assert m.state == "locked"


def test_cancelled_tap_does_not_count_towards_a_double_tap():
    clock = Clock()
    m = HotkeyMachine("ctrl+win", clock=clock)
    m.feed(CTRL, True)
    m.feed(WIN, True)
    m.feed(LEFT, True)                       # Ctrl+Win+Left: cancel
    for vk in (LEFT, WIN, CTRL):
        m.feed(vk, False)
    clock.t += 0.1
    events = [e for r in down_up(m, CTRL, WIN) for e in r.events]
    assert events == ["start", "stop"], "must be a plain tap, not a hands-free lock"


# ---------------------------------------------------------------- capture safety (C-03, C-04, C-05)
def test_capture_reports_combo_and_swallows_non_modifiers():
    m = HotkeyMachine()
    m.start_capture()
    assert m.capturing
    assert not m.feed(CTRL, True).swallow and not m.feed(ALT, True).swallow
    assert m.feed(SPACE, True).swallow
    assert m.feed(SPACE, False).swallow
    alt_up = m.feed(ALT, False)
    assert alt_up.mask and alt_up.captured is None
    done = m.feed(CTRL, False)
    assert done.captured == "ctrl+alt+space" and not m.capturing


def test_capture_expires_only_while_nothing_is_pressed():
    clock = Clock()
    m = HotkeyMachine(clock=clock)
    assert m.expire_capture() is False                      # not capturing
    m.start_capture()
    clock.t += hk.CAPTURE_TIMEOUT_S - 0.1
    assert m.expire_capture() is False and m.capturing
    clock.t += 0.2
    assert m.expire_capture() is True and not m.capturing
    m.start_capture()
    m.feed(CTRL, True)                                      # user is in the middle of pressing a combo
    clock.t += 60
    assert m.expire_capture() is False and m.capturing
    m.cancel_capture()
    assert not m.capturing


# ---------------------------------------------------------------- listener routing
@pytest.fixture
def listener():
    log = []
    lst = HotkeyListener("ctrl+win", lambda e: log.append(("dictate", e)), lambda c: log.append(("captured", c)))
    lst.add_binding("command", "alt+win", lambda e: log.append(("command", e)))
    lst.add_binding("paste_last", "alt+shift+z", lambda e: log.append(("paste_last", e)), mode="press")
    lst.add_binding("undo", "", lambda e: log.append(("undo", e)), mode="press")
    lst.log = log
    return lst


def test_each_binding_fires_its_own_callback(listener):
    down_up(listener, CTRL, WIN)
    down_up(listener, ALT, WIN)
    down_up(listener, ALT, SHIFT, Z)
    assert listener.log == [("dictate", "start"), ("dictate", "stop"), ("command", "start"), ("command", "stop"),
                            ("paste_last", "press")]
    assert listener.binding("undo") is None and listener.binding("command").combo == "alt+win"


def test_listener_merges_swallow_and_mask_across_bindings(listener):
    listener.feed(ALT, True)
    listener.feed(SHIFT, True)
    assert listener.feed(Z, True).swallow
    assert listener.feed(Z, False).swallow
    listener.feed(SHIFT, False)
    assert listener.feed(ALT, False).mask
    listener.feed(CTRL, True)
    listener.feed(WIN, True)
    assert listener.feed(WIN, False).mask, "Win released after a dictation must not open the Start menu"


def test_c49_a_cancelled_dictation_does_not_morph_into_command_mode(listener):
    listener.feed(CTRL, True)
    listener.feed(WIN, True)                           # dictation is recording
    listener.feed(ALT, True)                           # Ctrl+Win+Alt: some other shortcut -> dictation cancels
    assert listener.log == [("dictate", "start"), ("dictate", "cancel")],         "Alt+Win is now 'held' too, but the user never asked for command mode"


def test_only_one_binding_can_be_active(listener):
    listener.feed(ALT, True)
    listener.feed(WIN, True)                           # command mode is recording
    assert listener.binding("command").active and not listener.machine.active
    listener.feed(CTRL, True)                          # Ctrl+Win is now physically held as well
    assert listener.log == [("command", "start"), ("command", "cancel")], \
        "dictation must not start while command mode owns the keyboard"
    for vk in (CTRL, WIN, ALT):
        listener.feed(vk, False)
    assert listener.log == [("command", "start"), ("command", "cancel")]
    assert listener.machine.state == "idle" and listener.binding("command").state == "idle"


def test_press_bindings_are_blocked_while_hands_free_is_locked(listener):
    for _ in range(2):
        down_up(listener, CTRL, WIN)
    assert listener.machine.state == "locked"
    listener.log.clear()
    down_up(listener, ALT, SHIFT, Z)
    assert listener.log == [], "paste-last must not fire in the middle of a hands-free dictation"
    down_up(listener, CTRL, WIN)
    assert listener.log == [("dictate", "stop")]


def test_set_binding_changes_enables_and_disables(listener):
    listener.set_binding("undo", "ctrl+alt+z")
    assert listener.binding("undo").mode == "press"
    down_up(listener, CTRL, ALT, Z)
    assert listener.log == [("undo", "press")]
    listener.set_binding("undo", "ctrl+alt+u")
    listener.log.clear()
    down_up(listener, CTRL, ALT, Z)
    assert listener.log == []
    down_up(listener, CTRL, ALT, 0x55)
    assert listener.log == [("undo", "press")]
    listener.set_binding("command", "")
    listener.log.clear()
    down_up(listener, ALT, WIN)
    assert listener.log == [] and listener.binding("command") is None
    with pytest.raises(ValueError):
        listener.set_binding("paste_last", "not+a+key")
    with pytest.raises(KeyError):
        listener.set_binding("nope", "f9")


def test_primary_combo_change_and_late_callback_lookup(listener):
    listener.set_combo("f9")
    seen = []
    listener.on_event = seen.append                    # the controller may swap the callback after construction
    down_up(listener, F9)
    assert seen == ["start", "stop"]
    down_up(listener, CTRL, WIN)
    assert seen == ["start", "stop"]


def test_disabled_listener_lets_everything_through(listener):
    listener.enabled = False
    results = down_up(listener, CTRL, WIN) + down_up(listener, ALT, SHIFT, Z)
    assert listener.log == []
    assert not any(r.swallow or r.mask or r.events for r in results)
    listener.enabled = True
    down_up(listener, CTRL, WIN)
    assert listener.log == [("dictate", "start"), ("dictate", "stop")]


def test_capture_takes_priority_over_bindings_and_works_while_paused(listener):
    listener.enabled = False                            # paused dictation must still allow changing the shortcut
    listener.begin_capture()
    down_up(listener, CTRL, ALT, SPACE)
    assert listener.log == [("captured", "ctrl+alt+space")]
    listener.log.clear()
    listener.enabled = True
    listener.begin_capture()
    down_up(listener, CTRL, WIN)                        # the current dictation shortcut is just a combo here
    assert listener.log == [("captured", "ctrl+win")]


def test_capture_cancel_paths_report_an_empty_combo(listener):
    listener.begin_capture()
    esc = listener.feed(ESC, True)
    assert esc.swallow and listener.log == [("captured", "")]
    listener.feed(ESC, False)
    listener.log.clear()

    clock = Clock()
    listener.machine.clock = clock
    listener.begin_capture()
    clock.t += hk.CAPTURE_TIMEOUT_S + 1
    listener.tick(lambda vk: False)
    assert listener.log == [("captured", "")] and not listener.machine.capturing
    listener.log.clear()
    listener.tick(lambda vk: False)
    assert listener.log == [], "the timeout is reported exactly once"

    listener.begin_capture()
    listener.cancel_capture()
    down_up(listener, CTRL, WIN)
    assert listener.log == [("dictate", "start"), ("dictate", "stop")]


def test_tick_runs_the_watchdog_of_every_binding(listener):
    listener.feed(ALT, True)
    listener.feed(WIN, True)
    assert listener.log == [("command", "start")]
    listener.tick(lambda vk: False)                     # the OS says nothing is held: the key-up was lost
    assert listener.log == [("command", "start"), ("command", "stop")]


def test_force_idle_releases_every_binding(listener):
    listener.feed(ALT, True)
    listener.feed(WIN, True)
    listener.force_idle()
    assert not listener.binding("command").active
    listener.feed(WIN, False)
    listener.feed(ALT, False)
    assert listener.log == [("command", "start")], "the release after an external stop must stay silent"


def test_listener_start_is_blocked_by_the_harness(listener):
    """The real hook must never be installed from a test."""
    listener.start()
    assert listener.error, "the guarded hook thread dies immediately, so start() must report failure"
    listener.stop()


def test_listener_start_off_windows(monkeypatch, listener):
    monkeypatch.setattr(hk.sys, "platform", "linux")
    assert listener.start() is False and "Windows" in listener.error
