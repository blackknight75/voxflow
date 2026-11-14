import pytest

from voxflow.hotkey import (HotkeyMachine, VK_ESC, VK_LALT, VK_LCTRL, VK_LWIN, VK_RCTRL, VK_RWIN, combo_from_vks,
                            parse_hotkey, pretty_hotkey)


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def mk(combo="ctrl+win", **kw):
    c = Clock()
    return HotkeyMachine(combo, clock=c, **kw), c


def press(m, *vks):
    return [m.feed(v, True) for v in vks]


def test_parse():
    g = parse_hotkey("Ctrl + Win")
    assert VK_LCTRL in g[0] and VK_RWIN in g[1]
    assert parse_hotkey("ctrl+alt+space")[2] == frozenset({0x20})
    assert parse_hotkey("f9") == [frozenset({0x78})]
    for bad in ("", "+", "ctrl+banana"):
        with pytest.raises(ValueError):
            parse_hotkey(bad)
    assert pretty_hotkey("ctrl+win") == "Ctrl + Win"
    assert combo_from_vks({VK_RWIN, VK_LCTRL}) == "ctrl+win"


def test_hold_to_talk_ctrl_then_win():
    m, c = mk()
    r1 = m.feed(VK_LCTRL, True)
    assert r1.events == [] and not r1.swallow
    r2 = m.feed(VK_LWIN, True)
    assert r2.events == ["start"]
    c.t += 1.5
    r3 = m.feed(VK_LWIN, False)           # releasing Win ends it, and masks the Start menu
    assert r3.events == ["stop"] and r3.mask
    assert m.feed(VK_LCTRL, False).events == []


def test_win_first_then_ctrl_masks_win_on_release():
    m, c = mk()
    m.feed(VK_LWIN, True)
    assert m.feed(VK_RCTRL, True).events == ["start"]
    c.t += 1
    r = m.feed(VK_RCTRL, False)
    assert r.events == ["stop"] and not r.mask       # ctrl up doesn't need a mask
    assert m.feed(VK_LWIN, False).mask               # win up afterwards still masked (no Start menu)


def test_other_keys_do_not_trigger():
    m, _ = mk()
    assert all(not r.events for r in press(m, VK_LCTRL, 0x43))   # ctrl+c
    m2, _ = mk()
    assert all(not r.events for r in press(m2, VK_LWIN, 0x45))   # win+e


def test_nonmodifier_key_is_swallowed():
    m, c = mk("ctrl+space")
    m.feed(VK_LCTRL, True)
    r = m.feed(0x20, True)
    assert r.events == ["start"] and r.swallow
    assert m.feed(0x20, True).swallow                 # auto-repeat swallowed
    c.t += 1
    r = m.feed(0x20, False)
    assert r.swallow and r.events == ["stop"]


def test_autorepeat_does_not_restart():
    m, _ = mk()
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["start"]
    assert m.feed(VK_LWIN, True).events == []
    assert m.feed(VK_LCTRL, True).events == []


def test_double_tap_lock_and_stop():
    m, c = mk()
    m.feed(VK_LCTRL, True)
    m.feed(VK_LWIN, True)
    c.t += 0.1
    assert m.feed(VK_LWIN, False).events == ["stop"]   # first tap: short -> stop (ignored by app)
    m.feed(VK_LCTRL, False)
    c.t += 0.2
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["start"]
    c.t += 0.1
    assert m.feed(VK_LWIN, False).events == ["lock"]
    m.feed(VK_LCTRL, False)
    c.t += 3                                           # hands-free for a while
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["stop"]    # next press stops
    assert m.feed(VK_LWIN, False).events == []
    m.feed(VK_LCTRL, False)
    assert m.state == "idle"


def test_slow_taps_do_not_lock_and_disabled():
    m, c = mk()
    m.feed(VK_LCTRL, True)
    m.feed(VK_LWIN, True)
    c.t += 0.1
    m.feed(VK_LWIN, False)
    m.feed(VK_LCTRL, False)
    c.t += 2.0                                         # too slow
    m.feed(VK_LCTRL, True)
    m.feed(VK_LWIN, True)
    c.t += 0.1
    assert m.feed(VK_LWIN, False).events == ["stop"]
    m2, c2 = mk(double_tap=False)
    for _ in range(2):
        m2.feed(VK_LCTRL, True)
        m2.feed(VK_LWIN, True)
        c2.t += 0.1
        assert m2.feed(VK_LWIN, False).events == ["stop"]
        m2.feed(VK_LCTRL, False)
        c2.t += 0.1


def test_escape_cancels():
    m, c = mk()
    m.feed(VK_LCTRL, True)
    m.feed(VK_LWIN, True)
    assert m.feed(VK_ESC, True).events == ["cancel"]
    m.feed(VK_ESC, False)
    c.t += 1
    assert m.feed(VK_LWIN, False).events == []         # no stop after cancel
    m.feed(VK_LCTRL, False)
    assert m.state == "idle"


def test_capture_mode():
    m, _ = mk()
    m.start_capture()
    m.feed(VK_LCTRL, True)
    m.feed(VK_LALT, True)
    r = m.feed(0x20, True)
    assert r.swallow
    assert m.feed(0x20, False).swallow
    m.feed(VK_LALT, False)
    r = m.feed(VK_LCTRL, False)
    assert r.captured == "ctrl+alt+space" and not m.capturing
    # modifier-only combos work too, and Win release is masked
    m.start_capture()
    m.feed(VK_LCTRL, True)
    m.feed(VK_RWIN, True)
    assert m.feed(VK_RWIN, False).mask
    assert m.feed(VK_LCTRL, False).captured == "ctrl+win"


def test_set_combo_resets():
    m, _ = mk()
    m.set_combo("ctrl+alt")
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LALT, True).events == ["start"]
