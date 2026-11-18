"""Which application is the user dictating into? (foreground window -> exe name + window title)."""
from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from dataclasses import dataclass

log = logging.getLogger("voxflow.context")

_MAX_TITLE = 200


@dataclass(frozen=True)
class AppContext:
    exe: str = ""        # lower-case file name, e.g. "slack.exe" ("" when unknown)
    title: str = ""      # window title, trimmed
    hwnd: int = 0        # native window handle (0 when unknown)


def exe_name(path: str | None) -> str:
    """'C:\\Program Files\\Slack\\Slack.exe' -> 'slack.exe' (pure helper)."""
    name = re.split(r"[\\/]", (path or "").strip())[-1]
    return name.lower()


def clean_title(title: str | None) -> str:
    """Collapse whitespace, drop control characters, cap the length (pure helper)."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", title or "")
    return " ".join(text.split())[:_MAX_TITLE]


def make_context(path: str | None, title: str | None, hwnd: int = 0) -> AppContext:
    return AppContext(exe_name(path), clean_title(title), int(hwnd or 0))


def _window_info(hwnd: int) -> tuple[str, str]:
    """(process image path, window title) for a window handle, through Win32."""
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                    ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 2)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value

    path = ""
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value:
        handle = kernel32.OpenProcess(0x1000, False, pid.value)     # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            try:
                size = wintypes.DWORD(1024)
                name = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
                    path = name.value
            finally:
                kernel32.CloseHandle(handle)
    return path, title


def foreground_hwnd() -> int:
    if sys.platform != "win32":
        return 0
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        return int(user32.GetForegroundWindow() or 0)
    except Exception:  # noqa: BLE001
        return 0


def foreground_is_own() -> bool:
    """True when the window that has the focus belongs to this very process (VoxFlow's own window). Never raises;
    False off Windows and whenever it cannot tell."""
    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return bool(pid.value) and int(pid.value) == os.getpid()
    except Exception:  # noqa: BLE001
        return False


_focus_api = None


def focus_state() -> tuple[int, int, bool]:
    """``(foreground window, the control in it that has the keyboard focus, window is in menu mode)``.

    Read-only and quick (no message is sent to the window). Never raises; unknown parts are 0 / False."""
    global _focus_api
    if sys.platform != "win32":
        return 0, 0, False
    try:
        from ctypes import wintypes

        if _focus_api is None:
            class GUITHREADINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                            ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND),
                            ("hwndMenuOwner", wintypes.HWND), ("hwndMoveSize", wintypes.HWND),
                            ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            user32.GetWindowThreadProcessId.restype = wintypes.DWORD
            user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
            _focus_api = (user32, GUITHREADINFO)
        user32, GUITHREADINFO = _focus_api
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return 0, 0, False
        info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        if not user32.GetGUIThreadInfo(user32.GetWindowThreadProcessId(hwnd, None), ctypes.byref(info)):
            return int(hwnd), 0, False
        return int(hwnd), int(info.hwndFocus or 0), bool(info.flags & 0x4)          # GUI_INMENUMODE
    except Exception:  # noqa: BLE001
        return 0, 0, False


def foreground_app() -> AppContext:
    """The application that currently has focus. Never raises; unknown parts are empty."""
    hwnd = foreground_hwnd()
    if not hwnd:
        return AppContext()
    try:
        path, title = _window_info(hwnd)
    except Exception as exc:  # noqa: BLE001
        log.debug("foreground window lookup failed: %s", exc)
        return AppContext(hwnd=hwnd)
    return make_context(path, title, hwnd)
