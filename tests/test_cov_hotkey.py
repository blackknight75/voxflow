"""voxflow.hotkey: remaining corners of the state machine, the listener lifecycle and the hook thread's
menu-guard timer. The hook thread body runs synchronously against fake user32 / kernel32 DLLs; the real
keyboard is never hooked and no real key is sent."""
from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

import pytest

from voxflow import hotkey as hk
from voxflow.hotkey import HOLD, IDLE, LOCKED, WAIT_RELEASE, HotkeyListener, HotkeyMachine, combo_from_vks

CTRL, RCTRL, WIN, ALT, SHIFT, Z, A = 0xA2, 0xA3, 0x5B, 0xA4, 0xA0, 0x5A, 0x41
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_TIMER = 0x100, 0x101, 0x104, 0x113


# ---------------------------------------------------------------- combo_from_vks
def test_combo_from_vks_ignores_unnamed_keys_and_duplicates():
    assert combo_from_vks({0xFF, CTRL, WIN}) == "ctrl+win", "a key without a name is skipped"
    assert combo_from_vks({CTRL, RCTRL, WIN}) == "ctrl+win", "left and right Ctrl are one modifier"
    assert combo_from_vks([A, A, ALT, CTRL]) == "ctrl+alt+a", "a repeated key is listed once"
    assert combo_from_vks({0xFF}) == ""


# ---------------------------------------------------------------- state machine
def test_watchdog_during_a_capture_only_forgets_keys_that_are_no_longer_down():
    m = HotkeyMachine("ctrl+win")
    m.start_capture()
    m.feed(CTRL, True)
    m.feed(WIN, True)
    assert m.down == {CTRL, WIN}
    assert m.tick(lambda vk: vk == CTRL) == [], "no dictation event can come out of a capture"
    assert m.down == {CTRL} and m.capturing and m.state == IDLE
    res = m.feed(CTRL, False)
    assert res.captured == "ctrl+win", "the combination that was held is still what gets captured"


def test_a_key_press_reveals_that_the_held_shortcut_was_released_unseen():
    now = [10.0]
    m = HotkeyMachine("ctrl+win", clock=lambda: now[0])
    assert m.feed(CTRL, True).events == [] and m.feed(WIN, True).events == ["start"]
    assert m.state == HOLD
    now[0] += 2.0
    # Ctrl went up without a hook event (secure desktop); the next key press comes with a probe of the real state
    res = m.feed(A, True, lambda vk: vk in (WIN, A))
    assert res.events == ["stop"] and m.state == IDLE and CTRL not in m.down
    assert not res.swallow


def test_a_stale_key_does_not_end_the_hold_when_the_new_key_completes_the_shortcut_again():
    now = [10.0]
    m = HotkeyMachine("ctrl+win", clock=lambda: now[0])
    m.feed(CTRL, True)
    m.feed(WIN, True)
    # left Ctrl was released unseen, but the key going down now is the right Ctrl: still Ctrl+Win
    res = m.feed(RCTRL, True, lambda vk: vk in (WIN, RCTRL))
    assert "stop" not in res.events and m.state == HOLD and m.down == {WIN, RCTRL}


def _locked_machine(now):
    """An ``alt+z`` machine locked hands-free by a double tap; every key is up again."""
    m = HotkeyMachine("alt+z", clock=lambda: now[0])
    seen = []
    for _tap in range(2):
        for vk, down in ((ALT, True), (Z, True), (Z, False), (ALT, False)):
            seen += m.feed(vk, down).events
            now[0] += 0.02
    assert seen == ["start", "stop", "start", "lock"] and m.state == LOCKED and not m.down
    return m


def test_the_letter_of_the_shortcut_is_not_typed_while_a_hands_free_recording_runs():
    now = [10.0]
    m = _locked_machine(now)
    letter = m.feed(Z, True)                     # the letter first, the modifier after it
    assert letter.events == [] and letter.swallow and m.state == LOCKED
    assert m.feed(ALT, True).events == ["stop"], "completing the shortcut ends the locked recording"
    assert m.state == WAIT_RELEASE
    assert m.feed(Z, False).swallow, "the matching key-up is eaten as well"
    m.feed(ALT, False)
    assert m.state == IDLE


def test_the_shortcut_pressed_right_after_an_externally_ended_recording_is_swallowed_not_restarted():
    now = [10.0]
    m = _locked_machine(now)
    m.force_idle()                               # e.g. the max-length cap ended the recording
    assert m.state == WAIT_RELEASE
    assert m.feed(ALT, True).events == []
    letter = m.feed(Z, True)
    assert letter.events == [] and letter.swallow, "no new recording, and no stray letter in the document"
    assert m.feed(Z, False).swallow
    m.feed(ALT, False)
    assert m.state == IDLE
    assert m.feed(ALT, True).events == [] and m.feed(Z, True).events == ["start"], "the next press works again"


# ---------------------------------------------------------------- listener lifecycle
@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_start_is_a_no_op_while_the_hook_thread_is_running(monkeypatch):
    release, runs = threading.Event(), []

    def run(self):
        runs.append(threading.current_thread().name)
        self._ready.set()
        release.wait(10)

    monkeypatch.setattr(HotkeyListener, "_run", run)
    lst = HotkeyListener("ctrl+win", lambda e: None)
    try:
        assert lst.start() is True
        thread = lst._thread
        assert thread.is_alive() and runs == ["voxflow-hook"]
        assert lst.start() is True, "already running: reports the current state"
        assert lst._thread is thread and len(runs) == 1, "no second hook thread"
        lst.error = "hook lost"
        assert lst.start() is False
    finally:
        release.set()
        lst.stop()
    assert lst._thread is None and not thread.is_alive()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_a_hook_thread_that_never_reports_back_is_an_error(monkeypatch):
    release = threading.Event()

    class NeverReady:
        def clear(self):
            pass

        def set(self):
            pass

        def wait(self, timeout):
            assert timeout == 3.0
            return False

    monkeypatch.setattr(HotkeyListener, "_run", lambda self: release.wait(10))
    lst = HotkeyListener("ctrl+win", lambda e: None)
    lst._ready = NeverReady()
    try:
        assert lst.start() is False
        assert lst.error == "the keyboard hook did not start in time"
    finally:
        release.set()
        lst.stop()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_stop_posts_wm_quit_to_the_hook_thread_exactly_once(monkeypatch):
    posted = []

    class User32:
        @staticmethod
        def PostThreadMessageW(tid, msg, w, l):
            posted.append((tid, msg, w, l))
            return 1

    class Windll:
        user32 = User32()

    monkeypatch.setattr(hk.ctypes, "windll", Windll())
    lst = HotkeyListener("ctrl+win", lambda e: None)
    lst._tid = 4242
    lst.stop()
    lst.stop()
    assert posted == [(4242, 0x0012, 0, 0)] and lst._tid == 0


# ---------------------------------------------------------------- hook thread: menu guard timer
class KBD(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


WATCHDOG_TIMER, GUARD_TIMER = 1, 7


class GuardUser32:
    """Scripted message loop: ("key", vk, wm) calls the hook procedure, ("timer", id) returns WM_TIMER,
    ("focus", state) changes what the fake focus query reports. Exhausted script = WM_QUIT."""

    def __init__(self, script, focus):
        self.script, self.focus = list(script), focus
        self.proc = None
        self.down: set[int] = set()
        self.calls: list[str] = []

        def set_hook(kind, proc, module, tid):
            self.proc = proc
            return 0xBEEF

        # plain functions accept the argtypes / restype attributes that _run assigns
        self.CallNextHookEx = lambda hook, n, w, l: 77
        self.SetWindowsHookExW = set_hook
        self.GetAsyncKeyState = lambda vk: -32768 if vk in self.down else 0

    def SetTimer(self, hwnd, ident, ms, callback):
        self.calls.append(f"SetTimer({ms})")
        return GUARD_TIMER if ms == 40 else WATCHDOG_TIMER

    def KillTimer(self, hwnd, ident):
        self.calls.append(f"KillTimer({ident})")
        return 1

    def GetMessageW(self, msg_ref, hwnd, lo, hi):
        msg = msg_ref._obj
        while self.script:
            step = self.script.pop(0)
            if step[0] == "key":
                _kind, vk, wm = step
                (self.down.add if wm in (WM_KEYDOWN, WM_SYSKEYDOWN) else self.down.discard)(vk)
                k = KBD(vk, 0, 0, 0, 0)
                self.proc(0, wm, ctypes.addressof(k))
            elif step[0] == "focus":
                self.focus[0] = step[1]
            elif step[0] == "hold":
                self.down.add(step[1])
            elif step[0] == "let-go":
                self.down.discard(step[1])
            else:
                msg.message, msg.wParam = WM_TIMER, step[1]
                return 1
        return 0

    def UnhookWindowsHookEx(self, hook):
        self.calls.append("Unhook")
        return 1


class _Kernel32:
    def GetCurrentThreadId(self):
        return 4242


EDITOR, MENU = (0x100, 0x200, False), (0x100, 0x300, False)


def _run_guarded_hook(monkeypatch, script, events):
    """Run the original hook thread body with fake DLLs; returns (fake user32, fake desktop, listener)."""
    from tests.win32_fakes import install
    desktop = install(monkeypatch)[0]                 # Alt taps / mask keys go to the fake SendInput
    focus = [EDITOR]
    user32, real = GuardUser32(script, focus), hk.ctypes.WinDLL
    monkeypatch.setattr(hk.ctypes, "WinDLL",
                        lambda name, **kw: {"user32": user32, "kernel32": _Kernel32()}.get(name) or real(name, **kw))
    monkeypatch.setattr(hk, "focus_state", lambda: focus[0])
    monkeypatch.setattr(hk, "MENU_SETTLE_S", 0.0)
    monkeypatch.setattr(hk, "MENU_VERIFY_S", 0.0)
    lst = HotkeyListener("alt+win", events.append)
    ticks = []
    real_tick = lst.tick
    lst.tick = lambda probe: (ticks.append(1), real_tick(probe))[1]
    hk.HotkeyListener._run_unguarded(lst)
    return user32, desktop, lst, ticks


SHORTCUT = [("key", ALT, WM_SYSKEYDOWN), ("key", WIN, WM_SYSKEYDOWN), ("key", WIN, WM_KEYUP)]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_hook_thread_runs_the_menu_guard_on_its_own_timer_and_stops_it_when_done(monkeypatch):
    events = []
    script = SHORTCUT + [
        ("focus", MENU),                              # the Alt key-up moves the focus into the window's menu
        ("key", ALT, WM_KEYUP),
        ("timer", WATCHDOG_TIMER),                    # the watchdog still runs while a focus check is due
        ("timer", GUARD_TIMER),                       # check: focus moved -> one Alt tap, then verify
        ("focus", EDITOR),                            # the tap closed the menu
        ("timer", GUARD_TIMER),                       # verify: focus is back -> done, timer killed
        ("timer", GUARD_TIMER),                       # a late WM_TIMER of the killed timer is a watchdog pass
    ]
    user32, desktop, lst, ticks = _run_guarded_hook(monkeypatch, script, events)
    assert events == ["start", "stop"]
    assert user32.calls == ["SetTimer(250)", "SetTimer(40)", "KillTimer(7)", "Unhook"]
    assert lst.guard.taps == 1 and not lst.guard.pending and lst.guard.wait(0)
    alt_taps = [(vk, bool(flags & 2)) for vk, _s, flags, extra in desktop.sent
                if vk == hk.VK_LALT and extra == hk.EXTRA_MARKER]
    assert alt_taps == [(hk.VK_LALT, False), (hk.VK_LALT, True)], "one tagged Alt press + release"
    assert len(ticks) == 2, "the guard's timer messages are not mistaken for the watchdog"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_hook_thread_guard_waits_while_a_modifier_is_physically_held(monkeypatch):
    events = []
    script = SHORTCUT + [
        ("focus", MENU),
        ("key", ALT, WM_KEYUP),
        ("hold", hk.VK_SHIFT),                           # the user holds Shift: an Alt tap now would be Alt+Shift
        ("timer", GUARD_TIMER),
        ("let-go", hk.VK_SHIFT),
        ("timer", GUARD_TIMER),                       # now idle: tap
        ("focus", EDITOR),
        ("timer", GUARD_TIMER),
    ]
    user32, desktop, lst, _ticks = _run_guarded_hook(monkeypatch, script, events)
    assert lst.guard.taps == 1 and not lst.guard.pending
    assert user32.calls.count("SetTimer(40)") == 1 and user32.calls.count("KillTimer(7)") == 1
    assert sum(1 for vk, *_r in desktop.sent if vk == hk.VK_LALT) == 2
