"""Text injection: clipboard + Ctrl+V (SendInput) with full clipboard restore.

``Injector`` holds the (testable) logic; ``Win32Clipboard`` / ``Win32Keys`` are the real backends.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from ctypes import wintypes

log = logging.getLogger("voxflow.inject")

EXTRA_MARKER = 0x564F5846  # "VOXF" - tags our synthetic key events so the hook ignores them

VK_CONTROL, VK_V, VK_C, VK_Z, VK_RETURN, VK_SHIFT = 0x11, 0x56, 0x43, 0x5A, 0x0D, 0x10
_WAIT_RELEASED = (0x5B, 0x5C, 0xA4, 0xA5, 0xA0, 0xA1)  # Win, Alt, Shift must be up before pasting

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.EnumClipboardFormats.argtypes = [wintypes.UINT]
    user32.EnumClipboardFormats.restype = wintypes.UINT
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]

KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, KEYEVENTF_EXTENDEDKEY = 0x2, 0x4, 0x1
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x2
# Only formats whose clipboard handle is a plain HGLOBAL are snapshotted: the standard memory formats
# (text, DIB/DIBV5, TIFF, locale, HDROP, ...) and registered formats (>=0xC000: "HTML Format", RTF, ...).
# GDI handle formats (CF_BITMAP, palettes, metafiles), private ranges (0x200-0x3FF) and others are skipped;
# Windows re-synthesises CF_BITMAP etc. from the DIB we do restore.
_SAFE_STD_FORMATS = {1, 4, 5, 6, 7, 8, 11, 13, 15, 16, 17}
_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


def _is_safe_format(fmt: int) -> bool:
    return fmt in _SAFE_STD_FORMATS or 0xC000 <= fmt <= 0xFFFF
_EXT_VKS = {0x5B, 0x5C, 0xA3, 0xA5, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22, 0x25, 0x26, 0x27, 0x28}


def send_keys(events) -> int:
    """events: iterable of (vk, is_down). Returns number of events accepted by SendInput."""
    events = list(events)
    arr = (INPUT * len(events))()
    for i, (vk, down) in enumerate(events):
        flags = (0 if down else KEYEVENTF_KEYUP) | (KEYEVENTF_EXTENDEDKEY if vk in _EXT_VKS else 0)
        arr[i].type = 1
        arr[i].ki = KEYBDINPUT(vk, user32.MapVirtualKeyW(vk, 0), flags, 0, EXTRA_MARKER)
    return int(user32.SendInput(len(events), arr, ctypes.sizeof(INPUT)))


def send_unicode(text: str) -> bool:
    """Type ``text`` with synthetic Unicode key events. True only if every event was accepted.

    Line breaks are sent as Shift+Enter (a soft break: it never submits a chat message or a form) because
    most applications ignore a Unicode packet carrying a newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    ok = True
    for i, line in enumerate(text.split("\n")):
        if i:
            ok &= send_keys([(VK_SHIFT, True), (VK_RETURN, True), (VK_RETURN, False), (VK_SHIFT, False)]) == 4
        evs = []
        for ch in line:
            for unit in _utf16_units(ch):
                evs.append((unit, 0))
                evs.append((unit, KEYEVENTF_KEYUP))
        if not evs:
            continue
        arr = (INPUT * len(evs))()
        for j, (unit, fl) in enumerate(evs):
            arr[j].type = 1
            arr[j].ki = KEYBDINPUT(0, unit, KEYEVENTF_UNICODE | fl, 0, EXTRA_MARKER)
        ok &= int(user32.SendInput(len(evs), arr, ctypes.sizeof(INPUT))) == len(evs)
    return bool(ok)


def _utf16_units(ch: str):
    b = ch.encode("utf-16-le")
    return [int.from_bytes(b[i:i + 2], "little") for i in range(0, len(b), 2)]


class Win32Keys:
    @staticmethod
    def _ctrl_chord(vk: int) -> bool:
        if user32.GetAsyncKeyState(VK_CONTROL) & 0x8000:
            # the user is still physically holding Ctrl: don't inject a Ctrl-up (it would un-hold it)
            return send_keys([(vk, True), (vk, False)]) == 2
        return send_keys([(VK_CONTROL, True), (vk, True), (vk, False), (VK_CONTROL, False)]) == 4

    def paste(self) -> bool:
        return self._ctrl_chord(VK_V)

    def copy(self) -> bool:
        return self._ctrl_chord(VK_C)

    def undo(self) -> bool:
        return self._ctrl_chord(VK_Z)

    def type_text(self, text: str) -> bool:
        return send_unicode(text)

    def modifiers_down(self) -> bool:
        return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in _WAIT_RELEASED)


class Win32Clipboard:
    """Clipboard access with multi-format snapshot / restore."""

    def __init__(self):
        self._fmt_exclude = user32.RegisterClipboardFormatW("ExcludeClipboardContentFromMonitorHistory")
        self._fmt_history = user32.RegisterClipboardFormatW("CanIncludeInClipboardHistory")
        self._fmt_cloud = user32.RegisterClipboardFormatW("CanUploadToCloudClipboard")

    def _open(self, tries: int = 20) -> bool:
        for _ in range(tries):
            if user32.OpenClipboard(None):
                return True
            time.sleep(0.01)
        return False

    @staticmethod
    def _alloc(data: bytes):
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
        if not h:
            return None
        p = kernel32.GlobalLock(h)
        ctypes.memmove(p, data, len(data))
        kernel32.GlobalUnlock(h)
        return h

    def sequence(self) -> int:
        return int(user32.GetClipboardSequenceNumber())

    def snapshot(self):
        """Return list[(format, bytes)] or None if the clipboard could not be read."""
        if not self._open():
            return None
        items = []
        total = 0
        present = 0
        try:
            fmt = 0
            while True:
                fmt = user32.EnumClipboardFormats(fmt)
                if not fmt:
                    break
                present += 1
                if not _is_safe_format(fmt):
                    continue
                h = user32.GetClipboardData(fmt)
                if not h:
                    continue
                size = kernel32.GlobalSize(h)
                if not size or total + size > _MAX_SNAPSHOT_BYTES:
                    continue
                p = kernel32.GlobalLock(h)
                if not p:
                    continue
                try:
                    items.append((fmt, ctypes.string_at(p, size)))
                    total += size
                finally:
                    kernel32.GlobalUnlock(h)
        finally:
            user32.CloseClipboard()
        if present and not items:
            return None          # content we cannot give back (GDI-only / huge): callers must not clobber it
        return items

    def text_snapshot(self):
        """Minimal fallback snapshot (just the text) used when the full snapshot failed."""
        t = self.get_text()
        return [(CF_UNICODETEXT, (t + "\0").encode("utf-16-le"))] if t else None

    @staticmethod
    def _put(fmt: int, data: bytes) -> bool:
        """SetClipboardData with ownership handling: a handle the clipboard did not take is freed."""
        h = Win32Clipboard._alloc(data)
        if not h:
            return False
        if not user32.SetClipboardData(fmt, h):
            kernel32.GlobalFree(h)
            return False
        return True

    def set_text(self, text: str, private: bool = True) -> int | None:
        """Place text on the clipboard. Returns the new sequence number, or None if that failed.

        ``private`` hides it from the clipboard history / cloud clipboard (used for transient pastes)."""
        if not self._open():
            return None
        try:
            user32.EmptyClipboard()
            if not self._put(CF_UNICODETEXT, (text + "\0").encode("utf-16-le")):
                return None
            if private:
                for fmt, val in ((self._fmt_exclude, b"\x01\x00\x00\x00"),
                                 (self._fmt_history, b"\x00\x00\x00\x00"),
                                 (self._fmt_cloud, b"\x00\x00\x00\x00")):
                    if fmt:
                        self._put(fmt, val)
        finally:
            user32.CloseClipboard()
        return self.sequence()

    def restore(self, snap) -> bool:
        if not self._open():
            return False
        try:
            user32.EmptyClipboard()
            for fmt, data in snap or []:
                self._put(fmt, data)
        finally:
            user32.CloseClipboard()
        return True

    def get_text(self) -> str:
        if not self._open():
            return ""
        try:
            h = user32.GetClipboardData(CF_UNICODETEXT)
            if not h:
                return ""
            p = kernel32.GlobalLock(h)
            try:
                return ctypes.wstring_at(p)
            finally:
                kernel32.GlobalUnlock(h)
        finally:
            user32.CloseClipboard()


PASTED, TYPED, COPIED, FAILED = "pasted", "typed", "copied", "failed"


class Injector:
    """Paste ``text`` into the focused window, then put the previous clipboard back.

    ``clipboard`` needs: snapshot(), set_text(text)->token|None, sequence(), restore(snap)
                         (optional: text_snapshot() fallback; get_text() for ``copy_selection``)
    ``keys``      needs: paste(), type_text(text), modifiers_down()  (optional: copy(), undo())

    Never destroys the user's clipboard: if neither a full nor a text-only snapshot can be taken, the
    text is typed with synthetic Unicode keystrokes instead of touching the clipboard.
    ``restore_delay`` is how long we wait after Ctrl+V before restoring (slow targets such as Electron,
    Office or remote desktop read the clipboard late).

    ``last_outcome`` tells what the last call did: ``pasted``, ``typed``, ``copied`` (the paste could not
    be sent, e.g. Win/Alt/Shift still held or input blocked by an elevated window; the text was left on the
    clipboard for a manual Ctrl+V) or ``failed``.

    ``inject`` is the blocking form. ``begin`` + ``finish_later`` return as soon as Ctrl+V is sent and
    restore the clipboard from a timer, so callers do not pay the restore delay.
    """

    def __init__(self, clipboard=None, keys=None, restore: bool = True, restore_delay: float = 0.5,
                 modifier_timeout: float = 1.5, sleep=time.sleep, clock=time.monotonic, paste_delay=None):
        self.clipboard = clipboard if clipboard is not None else Win32Clipboard()
        self.keys = keys if keys is not None else Win32Keys()
        self.restore = restore
        self.restore_delay = restore_delay if paste_delay is None else paste_delay
        self.modifier_timeout = modifier_timeout
        self.last_outcome = FAILED
        self._sleep, self._clock = sleep, clock
        self._lock = threading.RLock()
        self._pending = None                 # (snapshot, sequence) awaiting restore
        self.settle = None                   # optional callable: wait until the target window is ready for keys
        self._timer: threading.Timer | None = None

    def _delay(self) -> float:
        d = self.restore_delay
        return d if isinstance(d, (int, float)) and d == d and d > 0 else 0.0

    def modifiers_held(self) -> bool:
        """True when Win/Alt/Shift is down right now, i.e. a paste would have to wait for it (and after
        ``modifier_timeout`` would be left on the clipboard instead). Never raises."""
        try:
            return bool(self.keys.modifiers_down())
        except Exception:  # noqa: BLE001 - an unreadable keyboard state is "not held": the paste decides
            return False

    def _wait_modifiers(self) -> bool:
        """Wait for Win/Alt/Shift to be released. False if they are still down at the timeout."""
        end = self._clock() + self.modifier_timeout
        while self.keys.modifiers_down():
            if self._clock() >= end:
                return False
            self._sleep(0.01)
        if self.settle is not None:
            self.settle()                    # a shortcut with Alt may have opened the target's menu (hotkey.MenuGuard)
        return True

    def _snapshot(self):
        snap = self.clipboard.snapshot()
        if snap is None and hasattr(self.clipboard, "text_snapshot"):
            snap = self.clipboard.text_snapshot()
        return snap

    def _type(self, text: str) -> bool:
        ok = bool(self.keys.type_text(text))
        self.last_outcome = TYPED if ok else FAILED
        return ok

    # ---- pasting
    def inject(self, text: str) -> bool:
        """Paste and (after ``restore_delay``) restore the clipboard. True if the text reached the window."""
        if not text:
            return False
        ok = self.begin(text)
        if self._pending is not None:
            self._sleep(self._delay())
            self.finish()
        return ok

    def begin(self, text: str) -> bool:
        """Send the paste now; the clipboard restore stays pending until ``finish`` / ``finish_later``."""
        if not text:
            return False
        with self._lock:
            self.finish()                    # a previous paste's clipboard goes back first
            if not self._wait_modifiers():
                # Ctrl+V now would arrive as Win+V / Alt+Ctrl+V: leave the text ready for a manual paste
                self.last_outcome = COPIED if self.clipboard.set_text(text) is not None else FAILED
                return False
            snap = None
            if self.restore:
                snap = self._snapshot()
                if snap is None:
                    # clipboard unreadable: do not overwrite what we cannot give back
                    return self._type(text)
            seq = self.clipboard.set_text(text)
            if seq is None:                  # clipboard unavailable -> type it directly
                if snap is not None:
                    self.clipboard.restore(snap)     # set_text may already have emptied it
                return self._type(text)
            if not self.keys.paste():
                self.last_outcome = COPIED   # keep our text on the clipboard so "Copied" is true
                return False
            self.last_outcome = PASTED
            if self.restore and snap is not None:
                self._pending = (snap, seq)
            return True

    def finish(self) -> None:
        """Put the pre-paste clipboard back, unless someone else changed the clipboard meanwhile."""
        with self._lock:
            timer, self._timer = self._timer, None
            pending, self._pending = self._pending, None
        if timer is not None:
            timer.cancel()
        if pending is not None:
            snap, seq = pending
            try:
                if self.clipboard.sequence() == seq:
                    self.clipboard.restore(snap)
            except Exception:  # noqa: BLE001 - a locked clipboard must not break dictation
                log.warning("clipboard restore failed", exc_info=True)

    def finish_later(self) -> None:
        """Schedule ``finish`` after ``restore_delay`` without blocking the caller."""
        with self._lock:
            if self._pending is None:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay(), self.finish)
            self._timer.daemon = True
            self._timer.name = "voxflow-clipboard-restore"
            self._timer.start()

    def close(self) -> None:
        """Restore anything still pending right now (shutdown)."""
        self.finish()

    # ---- selection + undo (command mode, undo-last-paste)
    def copy_selection(self, timeout: float = 0.6):
        """Text selected in the focused window, read through Ctrl+C with the clipboard put back afterwards.

        Returns ``""`` when nothing is selected and ``None`` when the clipboard or keyboard cannot be used
        safely (so callers can tell the user instead of guessing)."""
        if not (hasattr(self.keys, "copy") and hasattr(self.clipboard, "get_text")):
            return None
        with self._lock:
            self.finish()
            if not self._wait_modifiers():
                return None
            snap = self._snapshot()
            if snap is None:
                return None
            seq0 = self.clipboard.sequence()
            if not self.keys.copy():
                return None
            end = self._clock() + timeout
            while self.clipboard.sequence() == seq0 and self._clock() < end:
                self._sleep(0.01)
            if self.clipboard.sequence() == seq0:
                return ""                    # the app copied nothing: no selection
            self._sleep(0.03)                # let the app finish writing every format
            text = self.clipboard.get_text() or ""
            self.clipboard.restore(snap)
            return text

    def undo(self) -> bool:
        """Send Ctrl+Z to the focused window."""
        if not hasattr(self.keys, "undo"):
            return False
        with self._lock:
            if not self._wait_modifiers():
                return False
            return bool(self.keys.undo())

    def copy_text(self, text: str) -> bool:
        """Put ``text`` on the clipboard for the user (visible in the clipboard history)."""
        with self._lock:
            self.finish()
            try:
                return self.clipboard.set_text(text, private=False) is not None
            except TypeError:                # clipboard backend without the ``private`` switch
                return self.clipboard.set_text(text) is not None
