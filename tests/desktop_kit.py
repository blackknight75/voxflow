"""Safety kit for tests and scripts that use the REAL desktop (opt-in: VOXFLOW_DESKTOP_TESTS=1).

Rules these helpers enforce:

* A synthetic key-down or paste is sent only while a window the test itself created is the foreground window,
  checked with GetForegroundWindow immediately before every send. Otherwise ``ForegroundLost`` is raised and
  nothing is sent. Key-ups for keys we pressed are always sent (never leave a modifier stuck).
* Taking the foreground never uses keystrokes (no "tap Alt" trick): only window-management calls.
* The user's clipboard is snapshotted before and restored after (``saved_clipboard``).
"""
from __future__ import annotations

import contextlib
import ctypes
import sys
import time
from ctypes import wintypes

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.SetFocus.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetAsyncKeyState.restype = ctypes.c_short

KEYEVENTF_KEYUP, KEYEVENTF_EXTENDEDKEY = 0x2, 0x1
VK = {"ctrl": 0xA2, "win": 0x5B, "alt": 0xA4, "shift": 0xA0, "esc": 0x1B, "z": 0x5A, "s": 0x53, "w": 0x57,
      "a": 0x41, "c": 0x43, "v": 0x56, "end": 0x23, "home": 0x24, "enter": 0x0D, "f4": 0x73,
      "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78,
      "f13": 0x7C, "f14": 0x7D, "f15": 0x7E, "f16": 0x7F}


class ForegroundLost(RuntimeError):
    """The window we own is not the foreground window: nothing was sent."""


def foreground() -> int:
    return int(user32.GetForegroundWindow() or 0)


def window_text(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 2)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def find_windows(predicate) -> list[int]:
    """Top-level visible windows for which ``predicate(hwnd, title, cls)`` is true."""
    found: list[int] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def each(hwnd, _lparam):
        try:
            if user32.IsWindowVisible(hwnd) and predicate(int(hwnd), window_text(hwnd), window_class(hwnd)):
                found.append(int(hwnd))
        except Exception:  # noqa: BLE001
            pass
        return True

    user32.EnumWindows(proto(each), 0)
    return found


def take_foreground(hwnd: int, pump=None, tries: int = 10) -> bool:
    """Make ``hwnd`` (a window WE created) the foreground window, without sending any keystroke."""
    pump = pump or (lambda s: time.sleep(s))
    my_tid = kernel32.GetCurrentThreadId()
    for attempt in range(tries):
        if foreground() == hwnd:
            return True
        fg = foreground()
        fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        target_tid = user32.GetWindowThreadProcessId(hwnd, None)
        attached = []
        for a, b in ((my_tid, fg_tid), (target_tid, fg_tid)):
            if a and b and a != b and user32.AttachThreadInput(a, b, True):
                attached.append((a, b))
        if attempt % 3 == 2:                     # last resort: minimise + restore our own window
            user32.ShowWindow(hwnd, 6)
            pump(0.05)
            user32.ShowWindow(hwnd, 9)
        user32.ShowWindow(hwnd, 5)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        for a, b in attached:
            user32.AttachThreadInput(a, b, False)
        pump(0.25)
    return foreground() == hwnd


def require_foreground(allowed) -> int:
    """``allowed``: hwnd, iterable of hwnds, or a callable returning one of those. Raises ForegroundLost."""
    value = allowed() if callable(allowed) else allowed
    hwnds = {int(value)} if isinstance(value, int) else {int(h) for h in value}
    fg = foreground()
    if not fg or fg not in hwnds:
        raise ForegroundLost(f"foreground window {fg:#x} ({window_text(fg)!r}) is not ours {sorted(hwnds)}")
    return fg


def _send(vk: int, down: bool, extra: int = 0) -> None:
    from voxflow import inject as inj
    ext = KEYEVENTF_EXTENDEDKEY if vk in (0x5B, 0x5C, 0xA3, 0xA5, 0x23, 0x24, 0x2E, 0x25, 0x26, 0x27, 0x28) else 0
    arr = (inj.INPUT * 1)()
    arr[0].type = 1
    arr[0].ki = inj.KEYBDINPUT(vk, user32.MapVirtualKeyW(vk, 0), (0 if down else KEYEVENTF_KEYUP) | ext, 0, extra)
    user32.SendInput(1, arr, ctypes.sizeof(inj.INPUT))


class Keyboard:
    """Synthetic keys that look physical to VoxFlow's hook (not tagged), sent only into our own foreground window."""

    def __init__(self, allowed):
        self.allowed = allowed
        self.held: list[int] = []

    def down(self, name: str) -> None:
        vk = VK[name]
        try:
            require_foreground(self.allowed)
        except ForegroundLost:
            self.release_all()
            raise
        self.held.append(vk)
        _send(vk, True)

    def up(self, name: str) -> None:
        vk = VK[name]
        if vk in self.held:
            self.held.remove(vk)
        _send(vk, False)                          # releasing a key is always safe

    def release_all(self) -> None:
        for vk in reversed(self.held):
            _send(vk, False)
        self.held.clear()

    def modifiers_stuck(self) -> list[str]:
        """Names of modifier keys Windows still reports as held (must be empty after a test)."""
        return [n for n, vk in (("ctrl", 0x11), ("alt", 0x12), ("shift", 0x10), ("win", 0x5B), ("rwin", 0x5C))
                if user32.GetAsyncKeyState(vk) & 0x8000]

    def chord(self, *names: str, hold_s: float = 0.05, pump=None) -> None:
        pump = pump or time.sleep
        try:
            for n in names:
                self.down(n)
                pump(0.03)
            pump(hold_s)
        finally:
            for n in reversed(names):
                self.up(n)
                pump(0.02)


def guarded_keys(allowed):
    """A ``voxflow.inject.Win32Keys`` that refuses to send unless one of OUR windows is the foreground window."""
    from voxflow import inject as inj

    class GuardedKeys(inj.Win32Keys):
        blocked = 0

        def _check(self) -> bool:
            try:
                require_foreground(allowed)
                return True
            except ForegroundLost:
                GuardedKeys.blocked += 1
                return False

        def paste(self):
            return self._check() and super().paste()

        def copy(self):
            return self._check() and super().copy()

        def undo(self):
            return self._check() and super().undo()

        def type_text(self, text):
            return self._check() and super().type_text(text)

    return GuardedKeys()


@contextlib.contextmanager
def saved_clipboard():
    """Snapshot the user's clipboard, give it back afterwards. Yields the Win32Clipboard, or None if the
    clipboard could not be saved (then the caller must not touch it)."""
    from voxflow.inject import Win32Clipboard
    cb = Win32Clipboard()
    snap = cb.snapshot()
    if snap is None:
        yield None
        return
    try:
        yield cb
    finally:
        for _ in range(5):
            if cb.restore(snap):
                break
            time.sleep(0.05)
