"""voxflow.inject: remaining failure paths of the Win32 clipboard layer (against the fake DLLs) and the
typed fallback of the injector. The real clipboard and keyboard are never touched."""
from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

from tests.helpers import fake_time  # noqa: E402
from tests.win32_fakes import CF_TEXT, CF_UNICODETEXT, KEYEVENTF_KEYUP, install  # noqa: E402
from voxflow import inject as inj  # noqa: E402
from voxflow.inject import EXTRA_MARKER, Injector  # noqa: E402

VK_CONTROL, VK_C, VK_Z = 0x11, 0x43, 0x5A


@pytest.fixture
def win(monkeypatch):
    monkeypatch.setattr(inj, "time", fake_time(sleep=lambda s: None))
    return install(monkeypatch)


# ---------------------------------------------------------------- keys
@pytest.mark.parametrize("method,vk", [("copy", VK_C), ("undo", VK_Z)])
def test_copy_and_undo_send_a_tagged_ctrl_chord(win, method, vk):
    u, _k = win
    assert getattr(inj.Win32Keys(), method)() is True
    assert [(v, bool(flags & KEYEVENTF_KEYUP)) for v, _s, flags, _x in u.sent] == [
        (VK_CONTROL, False), (vk, False), (vk, True), (VK_CONTROL, True)]
    assert all(extra == EXTRA_MARKER for *_rest, extra in u.sent)


@pytest.mark.parametrize("method", ["copy", "undo"])
def test_copy_and_undo_report_input_blocked_by_an_elevated_window(win, method):
    u, _k = win
    u.accept_inputs = 2                                     # UIPI let only part of the chord through
    assert getattr(inj.Win32Keys(), method)() is False


# ---------------------------------------------------------------- clipboard
def test_out_of_memory_fails_set_text_without_leaving_the_clipboard_open(win):
    u, k = win
    clip = inj.Win32Clipboard()
    k.fail_alloc = True
    assert clip.set_text("hello") is None
    assert not u.is_open and u.open_calls == u.close_calls == 1
    assert u.text() is None and k.mem == {}, "nothing was allocated or handed to the clipboard"


def test_restore_skips_formats_it_cannot_allocate(win):
    u, k = win
    clip = inj.Win32Clipboard()
    k.fail_alloc = True
    assert clip.restore([(CF_UNICODETEXT, "old\0".encode("utf-16-le"))]) is True
    assert u.data == {} and not u.is_open


def test_snapshot_skips_a_format_without_data_and_one_that_cannot_be_locked(win, monkeypatch):
    u, k = win
    clip = inj.Win32Clipboard()
    u.put_text("keep me")
    u.data[CF_TEXT] = 0                                     # announced (delayed rendering) but never rendered
    u.put(0xC123, b"locked-format")
    locked = u.data[0xC123]
    real_lock = k.GlobalLock
    monkeypatch.setattr(k, "GlobalLock", lambda h: 0 if h == locked else real_lock(h))
    snap = clip.snapshot()
    assert snap == [(CF_UNICODETEXT, "keep me\0".encode("utf-16-le"))]
    assert not u.is_open and k.locks == 0, "every successful lock was released"


def test_public_set_text_does_not_add_the_privacy_formats(win):
    u, _k = win
    clip = inj.Win32Clipboard()
    seq = clip.set_text("for the user", private=False)
    assert seq == u.seq and u.text() == "for the user"
    assert list(u.data) == [CF_UNICODETEXT], "visible in the clipboard history like any normal copy"


def test_privacy_formats_windows_could_not_register_are_left_out(win, monkeypatch):
    u, _k = win
    real_register = u.RegisterClipboardFormatW
    monkeypatch.setattr(u, "RegisterClipboardFormatW",
                        lambda name: 0 if name == "CanUploadToCloudClipboard" else real_register(name))
    clip = inj.Win32Clipboard()
    assert clip._fmt_cloud == 0 and clip._fmt_exclude and clip._fmt_history
    assert clip.set_text("secret") is not None
    assert set(u.data) == {CF_UNICODETEXT, clip._fmt_exclude, clip._fmt_history}
    assert 0 not in u.data


# ---------------------------------------------------------------- injector
class _Clip:
    def __init__(self):
        self.log = []

    def snapshot(self):
        self.log.append("snapshot")
        return ["old"]

    def set_text(self, text):
        self.log.append("set_text")
        return None                                          # clipboard held by another program

    def sequence(self):
        return 1

    def restore(self, snap):
        self.log.append("restore")


class _Keys:
    def __init__(self):
        self.typed, self.pasted = [], 0

    def paste(self):
        self.pasted += 1
        return True

    def type_text(self, text):
        self.typed.append(text)
        return True

    def modifiers_down(self):
        return False


def test_without_clipboard_restore_an_unavailable_clipboard_falls_back_to_typing():
    clip, keys = _Clip(), _Keys()
    injector = Injector(clip, keys, restore=False, sleep=lambda s: None, clock=lambda: 0.0)
    assert injector.begin("hello") is True
    assert keys.typed == ["hello"] and keys.pasted == 0 and injector.last_outcome == "typed"
    assert clip.log == ["set_text"], "no snapshot was taken, so there is nothing to restore"
    assert injector._pending is None


def test_with_clipboard_restore_the_snapshot_is_put_back_before_typing():
    clip, keys = _Clip(), _Keys()
    injector = Injector(clip, keys, restore=True, sleep=lambda s: None, clock=lambda: 0.0)
    assert injector.begin("hello") is True
    assert clip.log == ["snapshot", "set_text", "restore"] and keys.typed == ["hello"]
