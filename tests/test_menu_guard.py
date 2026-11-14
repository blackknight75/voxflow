"""The focus repair after a shortcut that contains Alt (bug C-69: Windows 11 Notepad moves its focus to the menu
bar, and the paste that follows a dictation is lost). ``hotkey.MenuGuard`` against a scripted window; the wiring
in ``HotkeyListener.feed`` and in the paste path; ``context.focus_state``. The live proof is
``scripts/live_e2e.py --alt-shortcuts``."""
from __future__ import annotations

import threading

import pytest

from voxflow import context, hotkey
from voxflow.hotkey import HotkeyListener, MenuGuard
from voxflow.inject import Injector

WINDOW, TEXT, MENU = 0x100, 0x111, 0x122
VK_LCTRL, VK_LALT, VK_LSHIFT, VK_F13, VK_LWIN, VK_Z = 0xA2, 0xA4, 0xA0, 0x7C, 0x5B, 0x5A


class Desk:
    """A foreground window whose focus can sit in its text or in its menu; an Alt tap toggles (like Notepad)."""

    def __init__(self, toggles: bool = True):
        self.window, self.focus, self.menu_mode = WINDOW, TEXT, False
        self.toggles = toggles
        self.taps = 0
        self.held = False                    # the user still holds a modifier
        self.now = 100.0

    def state(self):
        return self.window, self.focus, self.menu_mode

    def tap(self):
        self.taps += 1
        if self.toggles:
            self.focus = TEXT if self.focus == MENU else MENU

    def guard(self) -> MenuGuard:
        return MenuGuard(self.state, self.tap, lambda: not self.held, clock=lambda: self.now)

    def later(self, seconds: float = 0.2):
        self.now += seconds


def shortcut(guard: MenuGuard):
    guard.chord_began()
    guard.fired()
    guard.alt_released()


def test_focus_that_moved_into_the_menu_is_brought_back_with_one_alt_tap():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    assert guard.pending and guard.wait(0) is False, "the paste must wait while the check is outstanding"
    desk.focus = MENU                                   # Notepad reacted to the Alt key
    guard.step()
    assert desk.taps == 0, "the window gets time to react before the guard looks"
    desk.later()
    guard.step()
    assert desk.taps == 1 and desk.focus == TEXT and guard.pending
    desk.later()
    guard.step()
    assert desk.taps == 1 and guard.taps == 1 and not guard.pending and guard.wait(0) is True


def test_menu_mode_of_a_classic_window_counts_as_moved_even_with_the_same_focus_handle():
    desk = Desk(toggles=False)

    def leave_menu():
        desk.taps += 1
        desk.menu_mode = False

    guard = MenuGuard(desk.state, leave_menu, lambda: True, clock=lambda: desk.now)
    shortcut(guard)
    desk.menu_mode = True
    desk.later()
    guard.step()
    assert desk.taps == 1 and desk.menu_mode is False
    desk.later()
    guard.step()
    assert desk.taps == 1 and not guard.pending


def test_a_window_that_did_not_react_is_left_alone():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    desk.later()
    guard.step()
    assert desk.taps == 0 and not guard.pending and guard.wait(0) is True


def test_a_tap_that_did_not_restore_the_focus_is_undone_by_a_second_tap():
    """The focus moved for another reason (the user clicked another control): the window is left as it was found."""
    desk = Desk(toggles=False)
    guard = desk.guard()
    shortcut(guard)
    desk.focus = 0x133                                  # some other control, not a menu
    desk.later()
    guard.step()
    assert desk.taps == 1
    desk.later()
    guard.step()
    assert desk.taps == 2 and guard.taps == 2 and not guard.pending


def test_no_tap_goes_to_another_window_that_came_to_the_front():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    desk.window, desk.focus = 0x900, 0x901              # the user switched windows meanwhile
    desk.later()
    guard.step()
    assert desk.taps == 0 and not guard.pending


def test_no_second_tap_into_another_window_either():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    desk.focus = MENU
    desk.later()
    guard.step()
    assert desk.taps == 1
    desk.window, desk.focus = 0x900, 0x901
    desk.later()
    guard.step()
    assert desk.taps == 1 and not guard.pending


def test_no_tap_while_a_modifier_is_still_held_and_it_gives_up_after_a_while():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    desk.focus, desk.held = MENU, True
    for _ in range(3):
        desk.later()
        guard.step()
    assert desk.taps == 0 and guard.pending, "an Alt tap now would combine with the key the user holds"
    desk.held = False
    desk.later()
    guard.step()
    assert desk.taps == 1

    desk2 = Desk()
    guard2 = desk2.guard()
    shortcut(guard2)
    desk2.focus, desk2.held = MENU, True
    for _ in range(hotkey.MENU_RETRIES + 2):
        desk2.later()
        guard2.step()
    assert desk2.taps == 0 and not guard2.pending and guard2.wait(0) is True, "never blocks the paste for ever"


def test_nothing_happens_without_a_recognised_shortcut_or_without_a_known_focus():
    desk = Desk()
    guard = desk.guard()
    guard.chord_began()
    guard.alt_released()                                # Alt was typed, but no shortcut fired
    desk.focus = MENU
    desk.later()
    guard.step()
    assert desk.taps == 0 and not guard.pending and guard.wait(0) is True

    guard.fired()                                       # fired without a chord start: nothing to compare with
    guard2 = desk.guard()
    guard2.fired()
    guard2.alt_released()
    assert not guard2.pending

    blind = Desk()
    blind.focus = 0                                     # the focused control is unknown (elevated window)
    g = blind.guard()
    shortcut(g)
    blind.focus = MENU
    blind.later()
    g.step()
    assert blind.taps == 0 and not g.pending


def test_a_second_shortcut_during_a_check_does_not_restart_it():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    desk.focus = MENU
    guard.chord_began()                                 # would otherwise remember the MENU as the place to return to
    guard.fired()
    guard.alt_released()
    desk.later()
    guard.step()
    desk.later()
    guard.step()
    assert desk.focus == TEXT and desk.taps == 1


def test_wait_blocks_the_paste_thread_until_the_check_is_over():
    desk = Desk()
    guard = desk.guard()
    shortcut(guard)
    done = []
    t = threading.Thread(target=lambda: done.append(guard.wait(2.0)))
    t.start()
    t.join(0.05)
    assert t.is_alive() and not done
    desk.later()
    guard.step()
    t.join(2)
    assert done == [True]
    assert guard.wait(0.01) is True


# ---------------------------------------------------------------- wiring in the listener
class Probe:
    def __init__(self):
        self.calls: list[str] = []
        self.pending = False

    def chord_began(self):
        self.calls.append("began")

    def fired(self):
        self.calls.append("fired")

    def alt_released(self):
        self.calls.append("alt_up")

    def wait(self, timeout=0.6):
        self.calls.append(f"wait {timeout}")
        return True


def listener(combo: str) -> tuple[HotkeyListener, Probe, list]:
    events: list = []
    hl = HotkeyListener(combo, events.append)
    hl.guard = Probe()
    return hl, hl.guard, events


def press(hl, *vks):
    held = set()
    for vk in vks:
        held.add(vk)
        hl.feed(vk, True, lambda key: key in held)
    for vk in reversed(vks):
        held.discard(vk)
        hl.feed(vk, False, lambda key: key in held)


def test_a_shortcut_with_alt_arms_the_guard_and_reports_the_alt_release():
    hl, probe, events = listener("ctrl+alt+f13")
    press(hl, VK_LCTRL, VK_LALT, VK_F13)
    assert events[:1] == ["start"]
    assert probe.calls[0] == "began" and "fired" in probe.calls and probe.calls[-1] == "alt_up"
    assert probe.calls.index("fired") < probe.calls.index("alt_up")


def test_a_shortcut_without_alt_never_touches_the_guard():
    hl, probe, events = listener("ctrl+shift+f13")
    press(hl, VK_LCTRL, VK_LSHIFT, VK_F13)
    assert events[:1] == ["start"] and probe.calls == []


def test_alt_typed_on_its_own_only_reports_the_release():
    hl, probe, events = listener("ctrl+win")
    press(hl, VK_LALT)
    assert events == [] and probe.calls == ["alt_up"]


def test_the_default_paste_last_shortcut_is_watched_too():
    hl, probe, _events = listener("ctrl+win")
    fired = []
    hl.add_binding("paste_last", "alt+shift+z", fired.append, mode="press")
    press(hl, VK_LALT, VK_LSHIFT, VK_Z)
    assert fired and "began" in probe.calls and "fired" in probe.calls and probe.calls[-1] == "alt_up"


def test_settle_asks_the_guard_and_is_instant_without_one():
    hl, probe, _ = listener("ctrl+alt+f13")
    assert hl.settle(0.25) is True and probe.calls == ["wait 0.25"]
    hl.guard = None
    assert hl.settle() is True


def test_a_paused_listener_leaves_the_guard_alone():
    hl, probe, events = listener("ctrl+alt+f13")
    hl.enabled = False
    press(hl, VK_LCTRL, VK_LALT, VK_F13)
    assert events == [] and probe.calls == []


# ---------------------------------------------------------------- the paste path waits for the guard
def test_the_injector_settles_before_it_sends_the_paste():
    from test_inject_flow import Clip, Keys
    clip = Clip()
    keys = Keys(clip)
    order: list[str] = []
    real_paste = keys.paste
    keys.paste = lambda: (order.append("paste"), real_paste())[1]
    inj = Injector(clip, keys, sleep=lambda s: None, clock=lambda: 0.0, restore_delay=0.0, modifier_timeout=0.0)
    inj.settle = lambda: order.append("settle")
    assert inj.inject("hello")
    assert order[:2] == ["settle", "paste"]
    inj.settle = None
    assert inj.inject("again")


def test_the_controller_connects_the_paste_path_to_the_listener(tmp_path, qtbot, monkeypatch):
    from controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert rig.c.injector.settle == rig.c.listener.settle
    finally:
        rig.close()


# ---------------------------------------------------------------- reading the focus
def test_focus_state_is_read_only_and_never_raises(monkeypatch):
    state = context.focus_state()
    assert isinstance(state, tuple) and len(state) == 3
    assert isinstance(state[0], int) and isinstance(state[1], int) and isinstance(state[2], bool)

    class Broken:
        def __getattr__(self, name):
            raise OSError("user32 is gone")

    monkeypatch.setattr(context, "_focus_api", (Broken(), None))
    assert context.focus_state() == (0, 0, False)
    monkeypatch.setattr(context.sys, "platform", "linux")
    assert context.focus_state() == (0, 0, False)


@pytest.mark.parametrize("foreground, thread_info_ok, expect", [
    (0, True, (0, 0, False)),
    (0x500, False, (0x500, 0, False)),
])
def test_focus_state_with_no_foreground_window_or_no_thread_info(monkeypatch, foreground, thread_info_ok, expect):
    import ctypes
    from ctypes import wintypes

    class Info(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndFocus", wintypes.HWND)]

    class User32:
        def GetForegroundWindow(self):
            return foreground

        def GetWindowThreadProcessId(self, hwnd, pid):
            return 7

        def GetGUIThreadInfo(self, tid, info):
            return thread_info_ok

    monkeypatch.setattr(context, "_focus_api", (User32(), Info))
    assert context.focus_state() == expect
