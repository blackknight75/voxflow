"""Issue 4: missed key-ups, stuck keys and the max-hold safety net."""
from voxflow.hotkey import HotkeyMachine, VK_LCTRL, VK_LWIN


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def mk(**kw):
    c = Clock()
    return HotkeyMachine("ctrl+win", clock=c, **kw), c


def test_stale_win_does_not_let_ctrl_alone_start():
    m, _ = mk()
    m.feed(VK_LWIN, True)                    # Win+L style: the key-up is never delivered
    physical = set()                         # ...and the user really has nothing held
    r = m.feed(VK_LCTRL, True, probe=lambda vk: vk in physical | {VK_LCTRL})
    assert r.events == []                    # without reconciliation this would start recording
    assert VK_LWIN not in m.down


def test_without_stale_key_combo_still_starts_with_probe():
    m, _ = mk()
    held = {VK_LWIN}
    m.feed(VK_LWIN, True, probe=lambda vk: vk in held)
    held.add(VK_LCTRL)
    assert m.feed(VK_LCTRL, True, probe=lambda vk: vk in held).events == ["start"]


def test_tick_stops_when_keyup_was_lost():
    m, c = mk()
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["start"]
    c.t += 2
    assert m.tick(lambda vk: vk in {VK_LCTRL, VK_LWIN}) == []          # still really held
    assert m.tick(lambda vk: False) == ["stop"]                        # Win-up was swallowed by the OS
    assert m.state == "idle" and not m.down
    assert m.feed(VK_LCTRL, True).events == []                         # and no phantom auto-repeat state


def test_max_hold_forces_stop_even_if_keys_stay_down_or_locked():
    m, c = mk(max_hold_s=30)
    m.feed(VK_LCTRL, True)
    m.feed(VK_LWIN, True)
    c.t += 29
    assert m.tick(lambda vk: True) == []
    c.t += 2
    assert m.tick(lambda vk: True) == ["stop"]
    assert m.tick(lambda vk: True) == []                               # only once
    # hands-free lock is capped as well
    m2, c2 = mk(max_hold_s=30)
    for _ in range(2):
        m2.feed(VK_LCTRL, True)
        m2.feed(VK_LWIN, True)
        c2.t += 0.1
        ev = m2.feed(VK_LWIN, False).events
        m2.feed(VK_LCTRL, False)
        c2.t += 0.1
    assert ev == ["lock"]
    c2.t += 31
    assert m2.tick(lambda vk: False) == ["stop"]


def test_swallowed_keys_are_not_reconciled_away():
    m = HotkeyMachine("ctrl+space")
    m.feed(VK_LCTRL, True)
    assert m.feed(0x20, True).events == ["start"]
    # blocked events never reach GetAsyncKeyState, so Space looks "up" there - must not stop us
    assert m.tick(lambda vk: vk == VK_LCTRL) == []
    assert m.state == "hold"
