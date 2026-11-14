"""The real Win32Clipboard / Win32Keys / send_* code, driven against in-memory user32 + kernel32 fakes."""
from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

from tests.win32_fakes import (CF_BITMAP, CF_DIB, CF_UNICODETEXT, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP,  # noqa: E402
                               KEYEVENTF_UNICODE, install)
from voxflow import inject as inj  # noqa: E402

VK_CONTROL, VK_V, VK_LWIN, VK_RETURN = 0x11, 0x56, 0x5B, 0x0D


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


@pytest.fixture
def win(monkeypatch):
    from tests.helpers import fake_time
    monkeypatch.setattr(inj, "time", fake_time(sleep=lambda s: None))
    return install(monkeypatch)


# ---------------------------------------------------------------- clipboard
def test_set_and_get_text_roundtrip_including_astral_characters(win):
    u, _k = win
    cb = inj.Win32Clipboard()
    seq = cb.set_text("héllo 🎤 wörld\nline two")
    assert seq == u.seq == cb.sequence()
    assert cb.get_text() == "héllo 🎤 wörld\nline two"
    assert not u.is_open and u.open_calls == u.close_calls


def test_dictated_text_is_hidden_from_clipboard_history_and_cloud_sync(win):
    u, _k = win
    inj.Win32Clipboard().set_text("secret")
    by_name = {name: u.bytes_of(fmt) for name, fmt in u.registered.items()}
    assert by_name["ExcludeClipboardContentFromMonitorHistory"] is not None
    assert by_name["CanIncludeInClipboardHistory"] == b"\x00\x00\x00\x00"
    assert by_name["CanUploadToCloudClipboard"] == b"\x00\x00\x00\x00"


def test_get_text_on_empty_or_non_text_clipboard_is_empty_string(win):
    u, _k = win
    cb = inj.Win32Clipboard()
    assert cb.get_text() == ""
    u.put(CF_DIB, b"\x28" + b"\0" * 39)
    assert cb.get_text() == ""
    assert not u.is_open


def test_snapshot_and_restore_preserve_every_safe_format_bytes(win):
    u, _k = win
    html = u.RegisterClipboardFormatW("HTML Format")
    u.put_text("old text")
    u.put(CF_DIB, bytes(range(64)))
    u.put(html, b"<b>old</b>")
    cb = inj.Win32Clipboard()
    snap = cb.snapshot()
    assert dict(snap) == {CF_UNICODETEXT: "old text\0".encode("utf-16-le"), CF_DIB: bytes(range(64)),
                          html: b"<b>old</b>"}
    cb.set_text("dictation")
    assert u.text() == "dictation" and CF_DIB not in u.data
    assert cb.restore(snap) is True
    assert u.text() == "old text" and u.bytes_of(CF_DIB) == bytes(range(64)) and u.bytes_of(html) == b"<b>old</b>"
    assert not u.is_open


def test_snapshot_skips_gdi_handle_formats_and_is_empty_list_for_an_empty_clipboard(win):
    u, _k = win
    cb = inj.Win32Clipboard()
    assert cb.snapshot() == []
    u.put(CF_BITMAP, b"not-a-global-handle")
    u.put_text("t")
    assert [f for f, _ in cb.snapshot()] == [CF_UNICODETEXT]


def test_snapshot_respects_the_size_budget(win, monkeypatch):
    u, _k = win
    monkeypatch.setattr(inj, "_MAX_SNAPSHOT_BYTES", 100)
    u.put(CF_DIB, b"x" * 80)
    u.put_text("y" * 40)                      # 82 bytes: would exceed the budget
    assert [f for f, _ in inj.Win32Clipboard().snapshot()] == [CF_DIB]


def test_busy_clipboard_is_retried_then_reported_as_unavailable(win):
    u, _k = win
    cb = inj.Win32Clipboard()
    u.put_text("keep")
    u.busy_opens = 3
    assert cb.get_text() == "keep"            # succeeded on the 4th attempt
    u.busy_opens = 10 ** 6
    assert cb.snapshot() is None
    assert cb.set_text("x") is None
    assert cb.restore([(CF_UNICODETEXT, b"a\0\0\0")]) is False
    assert cb.get_text() == ""
    assert cb.text_snapshot() is None
    assert u.text() == "keep"


def test_text_snapshot_is_a_restorable_text_only_snapshot(win):
    u, _k = win
    cb = inj.Win32Clipboard()
    u.put_text("only text")
    snap = cb.text_snapshot()
    cb.set_text("other")
    cb.restore(snap)
    assert u.text() == "only text"


def test_clipboard_is_always_closed_even_when_an_api_call_raises(win, monkeypatch):
    u, _k = win
    cb = inj.Win32Clipboard()
    monkeypatch.setattr(u, "EmptyClipboard", lambda: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        cb.set_text("x")
    assert not u.is_open
    with pytest.raises(OSError):
        cb.restore([])
    assert not u.is_open


def test_c22_failed_set_text_does_not_destroy_the_clipboard(win):
    u, _k = win
    u.put_text("precious")
    u.fail_next_sets = 1                                  # the dictation cannot be placed
    cb = inj.Win32Clipboard()
    injector = inj.Injector(cb, inj.Win32Keys(), sleep=lambda s: None)
    assert injector.inject("dictation") is True          # typed instead
    assert u.typed_text() == "dictation" and injector.last_outcome == "typed"
    assert cb.get_text() == "precious", "set_text emptied the clipboard, failed, and nothing put the old content back"


def test_c22_handles_the_clipboard_did_not_take_are_freed(win):
    u, k = win
    u.fail_set_formats = {CF_UNICODETEXT}
    inj.Win32Clipboard().set_text("x")
    assert not k.mem, f"{len(k.mem)} HGLOBAL(s) leaked after a failed SetClipboardData"


def test_c24_clipboard_with_only_unsaveable_formats_is_not_wiped(win):
    u, _k = win
    u.put(CF_BITMAP, b"gdi-bitmap-handle")
    injector = inj.Injector(inj.Win32Clipboard(), inj.Win32Keys(), sleep=lambda s: None)
    assert injector.inject("dictation")
    assert CF_BITMAP in u.data, "the user's (unsaveable) clipboard content was replaced by nothing"


# ---------------------------------------------------------------- keyboard
def test_send_keys_tags_events_and_sets_keyup_and_extended_flags(win):
    u, _k = win
    assert inj.send_keys([(VK_CONTROL, True), (VK_LWIN, True), (VK_LWIN, False), (VK_CONTROL, False)]) == 4
    assert [(vk, flags) for vk, _s, flags, _x in u.sent] == [
        (VK_CONTROL, 0), (VK_LWIN, KEYEVENTF_EXTENDEDKEY), (VK_LWIN, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
        (VK_CONTROL, KEYEVENTF_KEYUP)]
    assert all(extra == inj.EXTRA_MARKER for *_r, extra in u.sent), "untagged events would re-enter our own hook"


def test_send_unicode_types_text_as_tagged_utf16_units(win):
    u, _k = win
    assert inj.send_unicode("aé🎤") is True
    assert len(u.sent) == 8                                          # a, é, and a surrogate pair; down + up each
    assert u.typed_text() == "aé🎤"
    assert all(vk == 0 and flags & KEYEVENTF_UNICODE and extra == inj.EXTRA_MARKER for vk, _s, flags, extra in u.sent)
    u.sent.clear()
    assert inj.send_unicode("") is True and u.sent == []


def test_c23_typed_fallback_keeps_line_breaks(win):
    u, _k = win
    assert inj.Win32Keys().type_text("one\r\ntwo\nthree") is True
    assert u.typed_text() == "onetwothree"
    downs = [i for i, (vk, _s, flags, _x) in enumerate(u.sent) if vk == VK_RETURN and not flags & KEYEVENTF_KEYUP]
    assert len(downs) == 2
    for i in downs:
        assert u.sent[i - 1][0] == 0x10, "soft break (Shift+Enter): never submits a chat message mid-dictation"
    assert all(extra == inj.EXTRA_MARKER for *_r, extra in u.sent)
    assert any(vk == VK_RETURN for vk, *_r in u.sent), "'\\n' was sent as a Unicode packet, which most apps ignore"


def test_c23_partially_accepted_typing_is_reported_as_failure(win):
    u, _k = win
    u.accept_inputs = 3
    assert inj.Win32Keys().type_text("hello world") is False


def test_paste_sends_ctrl_v_and_reports_partial_injection(win):
    u, _k = win
    keys = inj.Win32Keys()
    assert keys.paste() is True
    assert [(vk, bool(flags & KEYEVENTF_KEYUP)) for vk, _s, flags, _x in u.sent] == [
        (VK_CONTROL, False), (VK_V, False), (VK_V, True), (VK_CONTROL, True)]
    u.sent.clear()
    u.accept_inputs = 2                                    # UIPI blocked part of the batch
    assert keys.paste() is False


def test_paste_does_not_release_a_ctrl_key_the_user_is_still_holding(win):
    u, _k = win
    u.keys_down = {VK_CONTROL}
    assert inj.Win32Keys().paste() is True
    assert [vk for vk, *_r in u.sent] == [VK_V, VK_V]


@pytest.mark.parametrize("vk,expected", [(0x5B, True), (0x5C, True), (0xA4, True), (0xA5, True), (0xA0, True),
                                         (0xA1, True), (0x11, False), (0x41, False)])
def test_modifiers_down_reports_win_alt_shift_only(win, vk, expected):
    u, _k = win
    u.keys_down = {vk}
    assert inj.Win32Keys().modifiers_down() is expected


# ---------------------------------------------------------------- whole injector on the fake desktop
def test_injector_end_to_end_pastes_then_restores_the_previous_clipboard(win):
    u, _k = win
    u.put_text("what the user copied")
    slept = []
    injector = inj.Injector(inj.Win32Clipboard(), inj.Win32Keys(), restore_delay=0.5, sleep=slept.append)
    seen_at_paste = []
    real_send = u.SendInput
    u.SendInput = lambda n, arr, size: (seen_at_paste.append(u.text()), real_send(n, arr, size))[1]
    assert injector.inject("Hello world.") is True
    assert seen_at_paste == ["Hello world."], "the dictated text must be on the clipboard when Ctrl+V is sent"
    assert 0.5 in slept
    assert u.text() == "what the user copied"


def test_injector_end_to_end_keeps_a_newer_copy_made_during_the_restore_delay(win):
    u, _k = win
    u.put_text("old")
    injector = inj.Injector(inj.Win32Clipboard(), inj.Win32Keys(), sleep=lambda s: u.put_text("copied meanwhile"))
    assert injector.inject("dictation") is True
    assert u.text() == "copied meanwhile"


def test_injector_end_to_end_types_when_the_clipboard_is_locked_by_another_app(win):
    u, _k = win
    u.put_text("untouchable")
    u.busy_opens = 10 ** 6
    injector = inj.Injector(inj.Win32Clipboard(), inj.Win32Keys(), sleep=lambda s: None)
    assert injector.inject("typed instead") is True
    assert u.typed_text() == "typed instead"
    assert u.text() == "untouchable"
    assert not any(vk == VK_V for vk, *_r in u.sent)


def test_harness_blocks_the_real_desktop():
    """Without the fake DLLs the guard must stop real clipboard / SendInput access."""
    with pytest.raises(AssertionError, match="blocked in tests"):
        inj.send_keys([(VK_V, True)])
    with pytest.raises(AssertionError, match="blocked in tests"):
        inj.Win32Clipboard().set_text("must never reach the real clipboard")
