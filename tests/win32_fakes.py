"""In-memory stand-ins for user32 / kernel32 so the Win32 clipboard + SendInput layer can be tested hermetically.

Global memory handles are real ctypes buffers (handle == address), so ``ctypes.memmove`` / ``string_at`` /
``wstring_at`` in the code under test work unchanged.
"""
from __future__ import annotations

import ctypes

CF_TEXT, CF_BITMAP, CF_DIB, CF_UNICODETEXT, CF_HDROP = 1, 2, 8, 13, 15
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x1, 0x2, 0x4


class FakeKernel32:
    def __init__(self):
        self.mem: dict[int, ctypes.Array] = {}
        self.fail_alloc = False
        self.locks = 0

    def GlobalAlloc(self, _flags, size):
        if self.fail_alloc:
            return 0
        buf = ctypes.create_string_buffer(int(size))
        handle = ctypes.addressof(buf)
        self.mem[handle] = buf
        return handle

    def GlobalLock(self, handle):
        if handle in self.mem:
            self.locks += 1
            return handle
        return 0

    def GlobalUnlock(self, handle):
        self.locks -= 1
        return 1

    def GlobalSize(self, handle):
        return len(self.mem[handle]) if handle in self.mem else 0

    def GlobalFree(self, handle):
        self.mem.pop(handle, None)
        return 0

    def GetCurrentThreadId(self):
        return 4242


class FakeUser32:
    """Clipboard + keyboard state. ``data`` maps format -> handle in insertion order (like the real clipboard)."""

    def __init__(self, kernel32: FakeKernel32):
        self.k = kernel32
        self.data: dict[int, int] = {}
        self.seq = 1
        self.is_open = False
        self.busy_opens = 0             # the next N OpenClipboard calls fail (another app holds the clipboard)
        self.fail_set_formats: set[int] = set()
        self.fail_next_sets = 0         # the next N SetClipboardData calls fail (transient error)
        self.registered: dict[str, int] = {}
        self.keys_down: set[int] = set()
        self.sent: list[tuple[int, int, int, int]] = []   # (vk, scan, flags, extra)
        self.accept_inputs: int | None = None             # None = accept all
        self.open_calls = 0
        self.close_calls = 0

    # ---- helpers for tests
    def put(self, fmt: int, payload: bytes) -> None:
        handle = self.k.GlobalAlloc(0, max(1, len(payload)))
        ctypes.memmove(handle, payload, len(payload))
        self.data[fmt] = handle
        self.seq += 1

    def put_text(self, text: str) -> None:
        self.put(CF_UNICODETEXT, (text + "\0").encode("utf-16-le"))

    def bytes_of(self, fmt: int) -> bytes | None:
        handle = self.data.get(fmt)
        return None if not handle else ctypes.string_at(handle, self.k.GlobalSize(handle))

    def text(self) -> str | None:
        raw = self.bytes_of(CF_UNICODETEXT)
        return None if raw is None else raw.decode("utf-16-le").split("\0")[0]

    def typed_text(self) -> str:
        units = [scan for vk, scan, flags, _x in self.sent if flags & KEYEVENTF_UNICODE and not flags & KEYEVENTF_KEYUP]
        return b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le")

    # ---- clipboard API
    def OpenClipboard(self, _hwnd):
        self.open_calls += 1
        if self.busy_opens > 0:
            self.busy_opens -= 1
            return 0
        assert not self.is_open, "OpenClipboard while already open (missing CloseClipboard)"
        self.is_open = True
        return 1

    def CloseClipboard(self):
        self.close_calls += 1
        self.is_open = False
        return 1

    def EmptyClipboard(self):
        assert self.is_open
        self.data.clear()
        self.seq += 1
        return 1

    def SetClipboardData(self, fmt, handle):
        assert self.is_open
        if fmt in self.fail_set_formats:
            return 0
        if self.fail_next_sets > 0:
            self.fail_next_sets -= 1
            return 0
        self.data[fmt] = handle
        self.seq += 1
        return handle

    def GetClipboardData(self, fmt):
        assert self.is_open
        return self.data.get(fmt, 0)

    def EnumClipboardFormats(self, fmt):
        assert self.is_open
        keys = list(self.data)
        if fmt == 0:
            return keys[0] if keys else 0
        i = keys.index(fmt)
        return keys[i + 1] if i + 1 < len(keys) else 0

    def GetClipboardSequenceNumber(self):
        return self.seq

    def RegisterClipboardFormatW(self, name):
        return self.registered.setdefault(name, 0xC000 + len(self.registered) + 1)

    # ---- keyboard API
    def GetAsyncKeyState(self, vk):
        return -32768 if vk in self.keys_down else 0      # c_short with the high bit set

    def MapVirtualKeyW(self, vk, _kind):
        return vk & 0xFF

    def SendInput(self, n, arr, _size):
        accepted = n if self.accept_inputs is None else min(n, self.accept_inputs)
        for i in range(accepted):
            ki = arr[i].ki
            self.sent.append((int(ki.wVk), int(ki.wScan), int(ki.dwFlags), int(ki.dwExtraInfo)))
        return accepted


def install(monkeypatch):
    """Replace the DLL objects used by voxflow.inject. Returns (user32, kernel32)."""
    import voxflow.inject as inj
    k = FakeKernel32()
    u = FakeUser32(k)
    monkeypatch.setattr(inj, "user32", u)
    monkeypatch.setattr(inj, "kernel32", k)
    return u, k
