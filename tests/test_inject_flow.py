"""Injector outcomes, the non-blocking begin / finish_later flow, selection grab, undo and copy_text (all fakes)."""
from __future__ import annotations

import threading

import pytest

from voxflow import inject as inj
from voxflow.inject import COPIED, FAILED, PASTED, TYPED, Injector


class Clip:
    """Text-only clipboard with a sequence number, like the real one."""

    def __init__(self, text="old"):
        self.text, self.seq, self.log = text, 1, []
        self.readable = True
        self.writable = True
        self.private_flags: list[bool] = []

    def snapshot(self):
        self.log.append("snapshot")
        return [("text", self.text)] if self.readable else None

    def set_text(self, text, private=True):
        if not self.writable:
            return None
        self.text, self.seq = text, self.seq + 1
        self.private_flags.append(private)
        self.log.append(f"set:{text}")
        return self.seq

    def sequence(self):
        return self.seq

    def restore(self, snap):
        self.text, self.seq = snap[0][1], self.seq + 1
        self.log.append(f"restore:{self.text}")
        return True

    def get_text(self):
        return self.text

    def user_copies(self, text):
        self.text, self.seq = text, self.seq + 1


class Keys:
    def __init__(self, clip):
        self.clip = clip
        self.mods_for = 0                 # modifiers_down() stays True for this many polls
        self.paste_ok = self.copy_ok = self.undo_ok = self.type_ok = True
        self.pasted, self.typed, self.undone, self.copies = [], [], 0, 0
        self.selection: str | None = "selected words"      # what the focused app copies on Ctrl+C

    def modifiers_down(self):
        if self.mods_for > 0:
            self.mods_for -= 1
            return True
        return False

    def paste(self):
        if self.paste_ok:
            self.pasted.append(self.clip.text)
        return self.paste_ok

    def type_text(self, text):
        if self.type_ok:
            self.typed.append(text)
        return self.type_ok

    def copy(self):
        self.copies += 1
        if self.copy_ok and self.selection is not None:
            self.clip.user_copies(self.selection)
        return self.copy_ok

    def undo(self):
        self.undone += 1
        return self.undo_ok


@pytest.fixture
def rig():
    clip = Clip()
    keys = Keys(clip)
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    injector = Injector(clip, keys, sleep=sleep, clock=lambda: now[0], restore_delay=0.5, modifier_timeout=0.3)
    injector.slept = slept
    return injector, clip, keys


# ---------------------------------------------------------------- outcomes
def test_default_backends_are_the_win32_ones_and_never_used_for_empty_text(monkeypatch):
    built = []
    monkeypatch.setattr(inj, "Win32Clipboard", lambda: built.append("clip") or object())
    monkeypatch.setattr(inj, "Win32Keys", lambda: built.append("keys") or object())
    injector = Injector()
    assert built == ["clip", "keys"] and injector.last_outcome == FAILED
    assert injector.inject("") is False and injector.begin("") is False


def test_paste_outcome_and_blocking_restore(rig):
    injector, clip, keys = rig
    assert injector.inject("hello") is True
    assert injector.last_outcome == PASTED and keys.pasted == ["hello"]
    assert clip.text == "old" and 0.5 in injector.slept
    assert clip.private_flags == [True], "a transient paste must be hidden from the clipboard history"


def test_typed_outcome_when_the_clipboard_cannot_be_saved(rig):
    injector, clip, keys = rig
    clip.readable = False
    assert injector.inject("hello") is True
    assert injector.last_outcome == TYPED and keys.typed == ["hello"] and keys.pasted == []
    assert clip.text == "old" and not any(e.startswith("set") for e in clip.log)


def test_typed_outcome_when_the_clipboard_cannot_be_written(rig):
    injector, clip, keys = rig
    clip.writable = False
    assert injector.inject("hello") is True
    assert injector.last_outcome == TYPED and keys.typed == ["hello"]
    assert "restore:old" in clip.log, "set_text may have emptied the clipboard before failing: put it back"


def test_failed_outcome_when_typing_fails_too(rig):
    injector, clip, keys = rig
    clip.readable = False
    keys.type_ok = False
    assert injector.inject("hello") is False and injector.last_outcome == FAILED


def test_copied_outcome_when_modifiers_stay_down(rig):
    injector, clip, keys = rig
    keys.mods_for = 10 ** 6
    assert injector.inject("hello") is False
    assert injector.last_outcome == COPIED and clip.text == "hello" and keys.pasted == [] and keys.typed == []
    clip.writable = False
    keys.mods_for = 10 ** 6
    assert injector.inject("again") is False and injector.last_outcome == FAILED


def test_copied_outcome_when_the_paste_keystroke_is_blocked(rig):
    injector, clip, keys = rig
    keys.paste_ok = False
    assert injector.inject("hello") is False
    assert injector.last_outcome == COPIED and clip.text == "hello"
    assert "restore:old" not in clip.log, "restoring would make 'Copied' a lie"


def test_restore_disabled_never_snapshots_or_restores(rig):
    injector, clip, keys = rig
    injector.restore = False
    assert injector.inject("hello") is True
    assert clip.text == "hello" and "snapshot" not in clip.log and injector.slept == []
    clip.readable = False                               # unreadable clipboard is irrelevant without restore
    assert injector.inject("more") is True and keys.pasted == ["hello", "more"]


@pytest.mark.parametrize("delay", [-1, 0, float("nan"), None, "soon"])
def test_unusable_restore_delays_mean_no_wait(rig, delay):
    injector, clip, keys = rig
    injector.restore_delay = delay
    assert injector.inject("hello") is True
    assert clip.text == "old" and all(s == 0.0 for s in injector.slept)


def test_legacy_paste_delay_argument_still_sets_the_restore_delay():
    assert Injector(Clip(), Keys(Clip()), paste_delay=0.9).restore_delay == 0.9


# ---------------------------------------------------------------- begin / finish (non-blocking)
def test_begin_returns_before_the_restore_and_finish_completes_it(rig):
    injector, clip, keys = rig
    assert injector.begin("hello") is True
    assert injector.slept == [], "begin() must not pay the restore delay (bug C-08)"
    assert clip.text == "hello" and keys.pasted == ["hello"]
    injector.finish()
    assert clip.text == "old"
    injector.finish()                                   # idempotent
    assert clip.log.count("restore:old") == 1


def test_finish_keeps_a_newer_copy_made_by_the_user(rig):
    injector, clip, keys = rig
    injector.begin("hello")
    clip.user_copies("user copied this")
    injector.finish()
    assert clip.text == "user copied this"


def test_a_second_paste_first_restores_the_clipboard_of_the_previous_one(rig):
    injector, clip, keys = rig
    injector.begin("first")
    injector.begin("second")                            # previous restore still pending
    assert keys.pasted == ["first", "second"]
    injector.finish()
    assert clip.text == "old", "the ORIGINAL clipboard must come back, not the first dictation"


def test_finish_later_restores_from_a_timer_and_can_be_superseded(rig):
    injector, clip, keys = rig
    injector.restore_delay = 0.05
    injector.finish_later()                             # nothing pending: no timer
    assert injector._timer is None
    injector.begin("hello")
    injector.finish_later()
    first = injector._timer
    assert first.daemon and first.name == "voxflow-clipboard-restore"
    injector.finish_later()                             # rescheduling replaces the timer
    assert injector._timer is not first
    injector._timer.join(2)
    first.join(2)
    assert clip.text == "old" and clip.log.count("restore:old") == 1
    assert injector._pending is None


def test_close_restores_immediately_and_cancels_the_timer(rig):
    injector, clip, keys = rig
    injector.restore_delay = 30
    injector.begin("hello")
    injector.finish_later()
    timer = injector._timer
    injector.close()
    assert clip.text == "old" and injector._timer is None
    timer.join(1)
    assert not timer.is_alive(), "a 30 s restore timer must not outlive shutdown"


def test_finish_survives_a_locked_clipboard(rig, caplog):
    injector, clip, keys = rig
    injector.begin("hello")
    clip.restore = lambda snap: (_ for _ in ()).throw(OSError("clipboard locked"))
    with caplog.at_level("WARNING"):
        injector.finish()                               # must not raise into the pipeline
    assert "clipboard restore failed" in caplog.text


def test_pastes_from_two_threads_do_not_interleave(rig):
    injector, clip, keys = rig
    barrier = threading.Barrier(2)

    def worker(text):
        barrier.wait()
        for _ in range(20):
            injector.inject(text)

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("aaa", "bbb")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(set(keys.pasted)) == ["aaa", "bbb"] and len(keys.pasted) == 40
    assert clip.text == "old"


# ---------------------------------------------------------------- selection grab (command mode)
def test_copy_selection_returns_the_selection_and_restores_the_clipboard(rig):
    injector, clip, keys = rig
    assert injector.copy_selection() == "selected words"
    assert keys.copies == 1 and clip.text == "old"


def test_copy_selection_with_nothing_selected_is_an_empty_string(rig):
    injector, clip, keys = rig
    keys.selection = None                               # the app does not touch the clipboard
    assert injector.copy_selection(timeout=0.2) == ""
    assert clip.text == "old" and sum(injector.slept) >= 0.2


@pytest.mark.parametrize("breakage", ["mods", "unreadable", "copy_blocked", "no_copy_key", "no_get_text"])
def test_copy_selection_returns_none_when_it_cannot_be_done_safely(rig, breakage):
    injector, clip, keys = rig
    if breakage == "mods":
        keys.mods_for = 10 ** 6
    elif breakage == "unreadable":
        clip.readable = False
    elif breakage == "copy_blocked":
        keys.copy_ok = False
    elif breakage == "no_copy_key":
        injector.keys = type("K", (), {"modifiers_down": lambda s: False, "paste": lambda s: True,
                                       "type_text": lambda s, t: True})()
    else:
        injector.clipboard = type("C", (), {"snapshot": lambda s: [], "set_text": lambda s, t: 1,
                                            "sequence": lambda s: 1, "restore": lambda s, x: True})()
    assert injector.copy_selection() is None
    assert clip.text == "old"
    if breakage in ("mods", "unreadable"):
        assert keys.copies == 0, "Ctrl+C must not be sent when the clipboard could not be given back"


def test_copy_selection_completes_a_pending_restore_first(rig):
    injector, clip, keys = rig
    injector.begin("dictation")
    assert injector.copy_selection() == "selected words"
    assert clip.text == "old", "must restore the user's clipboard, not the previous dictation"


# ---------------------------------------------------------------- undo / copy_text
def test_undo_sends_ctrl_z_only_when_safe(rig):
    injector, clip, keys = rig
    assert injector.undo() is True and keys.undone == 1
    keys.undo_ok = False
    assert injector.undo() is False
    keys.mods_for = 10 ** 6
    assert injector.undo() is False and keys.undone == 2, "no Ctrl+Z while Win/Alt/Shift is held"
    injector.keys = type("K", (), {"modifiers_down": lambda s: False})()
    assert injector.undo() is False


def test_copy_text_is_visible_in_the_clipboard_history_and_supersedes_a_pending_restore(rig):
    injector, clip, keys = rig
    injector.begin("dictation")
    assert injector.copy_text("for the user") is True
    assert clip.text == "for the user" and clip.private_flags[-1] is False
    injector.finish()
    assert clip.text == "for the user"
    clip.writable = False
    assert injector.copy_text("nope") is False


def test_copy_text_works_with_a_clipboard_backend_without_the_private_switch(rig):
    injector, clip, keys = rig

    class OldClip(Clip):
        def set_text(self, text):               # no ``private`` parameter
            return super().set_text(text)

    injector.clipboard = OldClip()
    assert injector.copy_text("x") is True and injector.clipboard.text == "x"
