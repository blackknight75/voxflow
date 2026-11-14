"""voxflow.context: foreground application lookup (pure helpers + the Win32 path against fake DLLs)."""
from __future__ import annotations

import ctypes
import sys

import pytest

from voxflow import context as ctx
from voxflow.context import AppContext, clean_title, exe_name, foreground_app, make_context


@pytest.mark.parametrize("path,expected", [
    (r"C:\Program Files\Slack\Slack.exe", "slack.exe"), ("C:/Tools/Code.EXE", "code.exe"), ("notepad.exe", "notepad.exe"),
    ("  C:\\x\\Teams.exe  ", "teams.exe"), (r"\\server\share\app\Tool.exe", "tool.exe"), ("", ""), (None, ""),
])
def test_exe_name(path, expected):
    assert exe_name(path) == expected


@pytest.mark.parametrize("title,expected", [
    ("  Inbox (3) -   Gmail  ", "Inbox (3) - Gmail"), ("line one\r\nline two\ttabbed", "line one line two tabbed"),
    ("bell\x07 and\x00 null\x7f", "bell and null"), ("", ""), (None, ""), ("x" * 500, "x" * 200),
    ("Ünïcödé – 文档 – Edge", "Ünïcödé – 文档 – Edge"),
])
def test_clean_title(title, expected):
    assert clean_title(title) == expected


def test_make_context_and_defaults():
    c = make_context(r"C:\Apps\Slack.exe", " #general  | Acme ", 0x1234)
    assert c == AppContext("slack.exe", "#general | Acme", 0x1234)
    assert make_context(None, None, None) == AppContext() == AppContext("", "", 0)
    with pytest.raises(Exception):
        c.exe = "other.exe"                                     # frozen: safe to share between threads


class FakeUser32:
    def __init__(self, hwnd=0xABC, title="Quarterly report - Word", pid=4321):
        self.hwnd, self.title, self.pid = hwnd, title, pid

        def get_foreground():
            return self.hwnd

        def text_length(hwnd):
            return len(self.title)

        def get_text(hwnd, buf, n):
            buf.value = self.title[:n - 1]
            return len(buf.value)

        def thread_pid(hwnd, pid_ref):
            pid_ref._obj.value = self.pid
            return 99

        self.GetForegroundWindow, self.GetWindowTextLengthW = get_foreground, text_length
        self.GetWindowTextW, self.GetWindowThreadProcessId = get_text, thread_pid


class FakeKernel32:
    def __init__(self, image=r"C:\Program Files\Microsoft Office\WINWORD.EXE", can_open=True, can_query=True):
        self.image, self.opened, self.closed = image, [], []

        def open_process(access, inherit, pid):
            self.opened.append((access, pid))
            return 0x77 if can_open else 0

        def query_image(handle, flags, buf, size_ref):
            if not can_query:
                return 0
            buf.value = self.image
            size_ref._obj.value = len(self.image)
            return 1

        def close_handle(handle):
            self.closed.append(handle)
            return 1

        self.OpenProcess, self.QueryFullProcessImageNameW, self.CloseHandle = open_process, query_image, close_handle


@pytest.fixture
def desktop(monkeypatch):
    def build(user32=None, kernel32=None):
        u, k = user32 or FakeUser32(), kernel32 or FakeKernel32()
        monkeypatch.setattr(ctx.sys, "platform", "win32")
        monkeypatch.setattr(ctx.ctypes, "WinDLL", lambda name, **kw: {"user32": u, "kernel32": k}[name])
        return u, k

    return build


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.wintypes layout")
def test_foreground_app_reads_exe_title_and_handle(desktop):
    u, k = desktop()
    assert foreground_app() == AppContext("winword.exe", "Quarterly report - Word", 0xABC)
    assert k.opened == [(0x1000, 4321)], "PROCESS_QUERY_LIMITED_INFORMATION is enough and works for elevated apps"
    assert k.closed == [0x77], "the process handle must be closed"


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.wintypes layout")
def test_unknown_parts_are_empty_never_an_exception(desktop):
    desktop(FakeUser32(hwnd=0))
    assert foreground_app() == AppContext()                     # no foreground window (lock screen)
    desktop(kernel32=FakeKernel32(can_open=False))              # protected / elevated process
    assert foreground_app() == AppContext("", "Quarterly report - Word", 0xABC)
    u, k = desktop(kernel32=FakeKernel32(can_query=False))
    assert foreground_app() == AppContext("", "Quarterly report - Word", 0xABC) and k.closed == [0x77]
    desktop(FakeUser32(pid=0))
    assert foreground_app().exe == ""
    desktop(FakeUser32(title=""))
    assert foreground_app() == AppContext("winword.exe", "", 0xABC)


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.wintypes layout")
def test_win32_failures_are_contained(desktop, monkeypatch):
    u, k = desktop()
    u.GetWindowTextLengthW = lambda hwnd: (_ for _ in ()).throw(OSError("access violation"))
    assert foreground_app() == AppContext(hwnd=0xABC)
    monkeypatch.setattr(ctx.ctypes, "WinDLL", lambda name, **kw: (_ for _ in ()).throw(OSError("no user32")))
    assert ctx.foreground_hwnd() == 0 and foreground_app() == AppContext()


def test_off_windows_there_is_no_context(monkeypatch):
    monkeypatch.setattr(ctx.sys, "platform", "linux")
    assert ctx.foreground_hwnd() == 0 and foreground_app() == AppContext()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_real_lookup_is_read_only_and_well_formed():
    """The real call only reads window state, so it is allowed in the hermetic suite."""
    c = foreground_app()
    assert isinstance(c, AppContext) and c.exe == c.exe.lower() and len(c.title) <= 200
    assert isinstance(ctypes.sizeof(ctypes.c_void_p), int)
