"""The real hook thread body (``HotkeyListener._run``) against fake user32/kernel32 DLLs.

The fake ``GetMessageW`` plays a script: it calls the installed low-level keyboard procedure exactly like Windows
would (through the ctypes callback thunk, with a pointer to a KBDLLHOOKSTRUCT) and delivers WM_TIMER messages.
Nothing here touches the real keyboard: the harness blocks the real hook, and these tests run the original
thread body (``_run_unguarded``) synchronously with ``ctypes.WinDLL`` replaced.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

from tests.win32_fakes import install  # noqa: E402
from voxflow import hotkey as hk  # noqa: E402
from voxflow.inject import EXTRA_MARKER  # noqa: E402

WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP, WM_TIMER = 0x100, 0x101, 0x104, 0x105, 0x113
CTRL, WIN, ALT, Z, SHIFT = 0xA2, 0x5B, 0xA4, 0x5A, 0xA0
NEXT_HOOK = 77          # what the fake CallNextHookEx returns ("event passed on")


class KBD(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HookUser32:
    """Scripted message loop.

    Script steps: ("key", vk, wm, extra)  deliver a key event to the hook procedure
                  ("ncode",)              deliver an event with nCode < 0
                  ("lose-keys",)          every key goes up WITHOUT a hook event (Win+L, UAC prompt)
                  ("timer",)              GetMessageW returns WM_TIMER
                  ("other",)              GetMessageW returns an unrelated message
    When the script is exhausted GetMessageW returns 0 (WM_QUIT).
    """

    def __init__(self, script, hook_handle=0xBEEF):
        self.script = list(script)
        self.proc = None
        self.returns: list[int] = []         # what the hook procedure returned per delivered event
        self.down: set[int] = set()
        self.calls: list[str] = []

        def call_next(hook, n_code, w_param, l_param):
            return NEXT_HOOK

        def set_hook(kind, proc, module, thread_id):
            assert kind == 13, "must be WH_KEYBOARD_LL"
            self.proc = proc
            self.calls.append("SetWindowsHookExW")
            return hook_handle

        def get_async(vk):
            return -32768 if vk in self.down else 0

        # plain functions accept the argtypes / restype attributes that _run assigns
        self.CallNextHookEx, self.SetWindowsHookExW, self.GetAsyncKeyState = call_next, set_hook, get_async

    def SetTimer(self, hwnd, ident, ms, callback):
        self.calls.append(f"SetTimer({ms})")
        return 1

    def GetMessageW(self, msg_ref, hwnd, lo, hi):
        msg = msg_ref._obj
        while self.script:
            step = self.script.pop(0)
            if step[0] == "key":
                _kind, vk, wm, extra = step
                (self.down.add if wm in (WM_KEYDOWN, WM_SYSKEYDOWN) else self.down.discard)(vk)
                k = KBD(vk, 0, 0, 0, extra)
                self.returns.append(int(self.proc(0, wm, ctypes.addressof(k))))
            elif step[0] == "ncode":
                k = KBD(Z, 0, 0, 0, 0)
                self.returns.append(int(self.proc(-1, WM_KEYDOWN, ctypes.addressof(k))))
            elif step[0] == "lose-keys":
                self.down.clear()
            elif step[0] == "timer":
                msg.message = WM_TIMER
                return 1
            else:
                msg.message = 0x0400
                return 1
        return 0

    def TranslateMessage(self, msg_ref):
        self.calls.append("TranslateMessage")

    def DispatchMessageW(self, msg_ref):
        self.calls.append("DispatchMessageW")

    def UnhookWindowsHookEx(self, hook):
        self.calls.append(f"Unhook({hook:#x})")
        return 1


class HookKernel32:
    def GetCurrentThreadId(self):
        return 4242


def run_hook(monkeypatch, listener, script, hook_handle=0xBEEF) -> HookUser32:
    """Run the original thread body synchronously with fake DLLs."""
    user32, real = HookUser32(script, hook_handle), hk.ctypes.WinDLL
    monkeypatch.setattr(hk.ctypes, "WinDLL", lambda name, **kw: {"user32": user32, "kernel32": HookKernel32()}.get(
        name) or real(name, **kw))
    hk.HotkeyListener._run_unguarded(listener)
    return user32


def key(vk, wm, extra=0):
    return ("key", vk, wm, extra)


@pytest.fixture
def events():
    return []


@pytest.fixture
def listener(events):
    lst = hk.HotkeyListener("ctrl+win", lambda e: events.append(("dictate", e)),
                            lambda c: events.append(("captured", c)))
    lst.add_binding("paste_last", "alt+shift+z", lambda e: events.append(("paste_last", e)), mode="press")
    return lst


def test_hook_installs_runs_the_loop_and_unhooks(monkeypatch, listener, events):
    u = run_hook(monkeypatch, listener, [key(CTRL, WM_KEYDOWN), key(WIN, WM_KEYDOWN), key(CTRL, WM_KEYUP),
                                         key(WIN, WM_KEYUP), ("other",)])
    assert events == [("dictate", "start"), ("dictate", "stop")]
    assert u.calls[:2] == ["SetWindowsHookExW", "SetTimer(250)"]
    assert u.calls[-3:] == ["TranslateMessage", "DispatchMessageW", "Unhook(0xbeef)"]
    assert listener._hook is None and listener.error is None and listener._ready.is_set() and listener._tid == 4242
    assert u.returns == [NEXT_HOOK] * 4, "modifier keys are never swallowed"


def test_swallowed_keys_return_1_everything_else_is_passed_on_and_alt_is_masked(monkeypatch, listener, events):
    desktop = install(monkeypatch)[0]        # the mask key goes through inject.send_keys -> fake SendInput
    u = run_hook(monkeypatch, listener, [
        key(ALT, WM_SYSKEYDOWN), key(SHIFT, WM_SYSKEYDOWN), key(Z, WM_SYSKEYDOWN),      # paste-last fires, Z eaten
        key(Z, WM_SYSKEYUP), key(SHIFT, WM_KEYUP), key(ALT, WM_KEYUP),
        key(Z, WM_KEYDOWN), key(Z, WM_KEYUP), ("ncode",)])
    assert events == [("paste_last", "press")]
    assert u.returns == [NEXT_HOOK, NEXT_HOOK, 1, 1, NEXT_HOOK, NEXT_HOOK, NEXT_HOOK, NEXT_HOOK, NEXT_HOOK]
    assert [(vk, extra) for vk, _s, _f, extra in desktop.sent] == [(hk.VK_MASK, EXTRA_MARKER)] * 2, \
        "the lone Alt release is masked with a tagged dummy key so the menu bar does not take focus"


def test_our_own_synthetic_events_are_ignored_by_the_hook(monkeypatch, listener, events):
    u = run_hook(monkeypatch, listener, [key(CTRL, WM_KEYDOWN, EXTRA_MARKER), key(WIN, WM_KEYDOWN, EXTRA_MARKER),
                                         key(WIN, WM_KEYUP, EXTRA_MARKER), key(CTRL, WM_KEYUP, EXTRA_MARKER)])
    assert events == [], "keys VoxFlow injected itself must never be read back as the user pressing them"
    assert u.returns == [NEXT_HOOK] * 4


def test_timer_watchdog_ends_a_recording_whose_key_up_was_lost(monkeypatch, listener, events):
    install(monkeypatch)
    run_hook(monkeypatch, listener, [key(CTRL, WM_KEYDOWN), key(WIN, WM_KEYDOWN), ("timer",)])
    assert events == [("dictate", "start")], "keys still physically down: the watchdog must not stop the recording"
    listener.force_idle()
    events.clear()
    lst = hk.HotkeyListener("ctrl+win", lambda e: events.append(e))
    run_hook(monkeypatch, lst, [key(CTRL, WM_KEYDOWN), key(WIN, WM_KEYDOWN), ("lose-keys",), ("timer",)])
    assert events == ["start", "stop"]


def test_paused_listener_skips_the_watchdog_but_still_times_out_a_capture(monkeypatch, events):
    lst = hk.HotkeyListener("ctrl+win", events.append, lambda c: events.append(("captured", c)))
    lst.enabled = False
    ticks = []
    lst.tick = lambda probe: ticks.append(lst.machine.capturing)
    run_hook(monkeypatch, lst, [("timer",)])
    assert ticks == []
    lst.begin_capture()
    run_hook(monkeypatch, lst, [("timer",), ("timer",)])
    assert ticks == [True, True]


def test_exceptions_in_callbacks_never_escape_into_windows(monkeypatch):
    def boom(_event):
        raise RuntimeError("callback bug")

    lst = hk.HotkeyListener("ctrl+win", boom)
    lst.tick = lambda probe: boom(None)
    u = run_hook(monkeypatch, lst, [key(CTRL, WM_KEYDOWN), key(WIN, WM_KEYDOWN), key(WIN, WM_KEYUP),
                                    key(CTRL, WM_KEYUP), ("timer",)])
    assert u.returns == [NEXT_HOOK] * 4, "a failing callback must still pass the key on (never freeze the keyboard)"
    assert u.calls[-1].startswith("Unhook")


def test_failed_hook_installation_is_reported(monkeypatch, listener):
    u = run_hook(monkeypatch, listener, [key(CTRL, WM_KEYDOWN)], hook_handle=0)
    assert listener.error and "SetWindowsHookEx failed" in listener.error
    assert listener._ready.is_set()
    assert u.calls == ["SetWindowsHookExW"] and u.returns == []


def test_start_and_stop_lifecycle_with_the_harness_guard(listener):
    assert listener.start() is False, "the harness refuses the real hook, exactly like a failed SetWindowsHookEx"
    assert "blocked in tests" in listener.error
    listener.stop()
    listener.stop()                                          # idempotent
    assert listener._thread is None and listener._tid == 0
