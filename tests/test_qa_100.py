"""Findings of the 1.0 QA pass outside the UI pages, each with the fix it guards: Esc (U-01, C-04), "paused"
feedback (U-04), work moved off the paste path (D-03, U-09), History keyboard (U-03), the double-tap cue (C-05),
leftover temp files (D-11), a paste under held keys (F-01) and undo while busy (F-02)."""
from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import Qt

from tests.controller_rig import ALT, CTRL, ESC, SHIFT, WIN, make_rig
from tests.test_ui import make_ui, seed  # noqa: F401  (fixture re-exported)
from voxflow import controller as ctl_mod
from voxflow.hotkey import HotkeyListener

A = 0x41


# ============================================================================ U-01
def test_escape_in_a_search_box_clears_the_search_and_only_then_hides_the_window(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 5))
    page = ui.page("history")
    search = ui.find("history_search")
    search.setFocus()
    search.setText("dentist")
    ui.wait(lambda: len(page.list.items) == 1)
    ui.qtbot.keyClick(search, Qt.Key.Key_Escape)
    assert search.text() == "" and ui.win.isVisible(), "the first Esc clears the search and keeps the window"
    ui.wait(lambda: len(page.list.items) == 5)
    ui.qtbot.keyClick(search, Qt.Key.Key_Escape)
    assert not ui.win.isVisible(), "Esc in an empty search box hides the window, as everywhere else"


def test_escape_clears_the_settings_search_too(make_ui):
    ui = make_ui()
    ui.page("settings")
    find = ui.find("settings_search")
    find.setFocus()
    find.setText("microphone")
    ui.qtbot.keyClick(find, Qt.Key.Key_Escape)
    assert find.text() == "" and ui.win.isVisible()


# ============================================================================ C-04: the hook
def _listener():
    events = []
    lst = HotkeyListener("ctrl+win", events.append)
    return lst, events


def test_the_hook_leaves_escape_alone_unless_a_dictation_is_being_worked_on():
    lst, events = _listener()
    for down in (True, False):
        assert not lst.feed(ESC, down).swallow
    assert events == []


def test_an_armed_escape_cancels_once_and_is_swallowed_down_and_up():
    lst, events = _listener()
    lst.cancel_armed = True
    assert lst.feed(ESC, True).swallow and events == ["cancel"] and not lst.cancel_armed
    assert lst.feed(ESC, True).swallow and events == ["cancel"], "auto-repeat of the same press: eaten, no new event"
    assert lst.feed(ESC, False).swallow
    for down in (True, False):                                 # the next Esc is the app's again
        assert not lst.feed(ESC, down).swallow
    assert events == ["cancel"]


def test_an_armed_escape_also_works_while_paused_and_other_keys_are_never_touched():
    lst, events = _listener()
    lst.enabled, lst.cancel_armed = False, True
    assert not lst.feed(A, True).swallow and not lst.feed(A, False).swallow and lst.cancel_armed
    assert lst.feed(ESC, True).swallow and lst.feed(ESC, False).swallow
    assert events == ["cancel"]


def test_escape_during_a_new_hold_cancels_that_recording_not_the_armed_one():
    lst, events = _listener()
    lst.cancel_armed = True                                    # the UI thread has not caught up with the new press yet
    lst.feed(CTRL, True)
    lst.feed(WIN, True)
    assert events == ["start"]
    lst.feed(ESC, True)
    assert events == ["start", "cancel"] and lst.cancel_armed, "the machine's own cancel; the armed one is untouched"


# ============================================================================ C-04: the controller
@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


def _gate_asr(rig):
    """The recogniser blocks until the returned event is set: the dictation stays in "processing"."""
    gate, entered = threading.Event(), threading.Event()
    text = rig.asr.text

    def slow(_audio):
        entered.set()
        gate.wait(5)
        return text
    rig.asr.text = slow
    return gate, entered


def _drain(rig):
    """Wait until the worker has finished everything queued so far."""
    rig.c._work.put({"kind": "paste", "text": "", "context": rig.context})
    rig.wait(lambda: ("begin", "") in rig.injector.events)


def test_escape_after_the_release_throws_the_dictation_away(rig):
    gate, entered = _gate_asr(rig)
    rig.dictate(wait=False)
    assert rig.c.state == "processing" and rig.c.listener.cancel_armed
    rig.wait(entered.is_set)
    rig.keys((ESC, True), (ESC, False))
    rig.wait(lambda: rig.c.state == "idle")
    assert not rig.c.listener.cancel_armed
    gate.set()
    _drain(rig)
    rig.settle(80)
    assert rig.injector.pasted == [""], "only the drain marker: the cancelled text was not pasted"
    assert rig.c.history.count() == 0 and rig.c.last_transcript is None
    assert rig.c.state == "idle" and rig.spy.of("transcript_ready") == []
    assert "error" not in rig.spy.states()


def test_a_cancelled_dictation_that_then_fails_stays_silent(rig):
    rig.asr.delay_s, rig.asr.fail_transcribe = 0.25, RuntimeError("boom")
    rig.dictate(wait=False)
    rig.c.cancel()
    assert rig.c.state == "idle"
    _drain(rig)
    rig.settle(80)
    assert rig.c.state == "idle" and "error" not in rig.spy.states() and rig.spy.notices() == []


def test_escape_is_not_armed_outside_processing_and_the_next_dictation_is_unaffected(rig):
    assert not rig.c.listener.cancel_armed
    gate, entered = _gate_asr(rig)
    rig.dictate(wait=False)
    rig.wait(entered.is_set)
    rig.c.cancel()
    gate.set()
    _drain(rig)
    rig.asr.text = "second one"
    rig.dictate()
    assert rig.c.state == "done" and not rig.c.listener.cancel_armed
    assert rig.injector.pasted[-1].lower().startswith("second one") and rig.c.history.count() == 1
    rig.c.cancel()                                              # "done" is not "processing": nothing to cancel
    assert rig.c.state == "done"


def test_once_the_paste_has_begun_escape_no_longer_cancels(rig):
    """The worker claims the dictation just before it pastes; a cancel that arrives after that changes nothing."""
    claimed, go = threading.Event(), threading.Event()
    real = rig.c._claim_paste

    def claim(gen):
        ok = real(gen)
        claimed.set()
        go.wait(5)
        return ok
    rig.c._claim_paste = claim
    rig.dictate(wait=False)
    rig.wait(claimed.is_set)
    assert rig.c.state == "processing" and not rig.c.listener.cancel_armed, "disarmed with the claim"
    rig.c.cancel()
    assert rig.c.state == "processing"
    go.set()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.injector.pasted) == 1 and rig.c.history.count() == 1


def test_command_mode_is_cancelled_the_same_way(rig):
    rig.injector.selection = "some selected text"
    rig.asr.text = "make it shorter"
    gate, entered = _gate_asr(rig)
    rig.pause()
    rig.press("command")
    rig.release("command")
    assert rig.c.state == "processing"
    rig.wait(entered.is_set)
    rig.keys((ESC, True), (ESC, False))
    gate.set()
    _drain(rig)
    rig.settle(80)
    assert rig.injector.pasted == [""] and rig.c.state == "idle" and rig.c.history.count() == 0


# ============================================================================ U-04: the hook
def test_a_shortcut_pressed_while_not_enabled_is_reported_once_per_press_and_passes_through():
    lst, events = _listener()
    lst.add_binding("command", "alt+win", events.append)
    blocked = []
    lst.on_blocked = lambda: blocked.append(1)
    lst.enabled = False
    held: set[int] = set()

    def key(vk, down):
        (held.add if down else held.discard)(vk)
        res = lst.feed(vk, down, lambda k: k in held)
        assert not res.swallow and not res.mask
    key(CTRL, True)
    assert blocked == []
    key(WIN, True)
    assert blocked == [1]
    key(WIN, True)                                              # auto-repeat
    assert blocked == [1]
    key(WIN, False), key(CTRL, False)
    key(ALT, True), key(WIN, True)                              # the command shortcut counts too
    assert blocked == [1, 1]
    key(WIN, False), key(ALT, False)
    key(SHIFT, True), key(CTRL, True), key(WIN, True)           # a different chord that contains ours: not ours
    assert blocked == [1, 1]
    key(WIN, False), key(CTRL, False), key(SHIFT, False)
    key(A, True), key(A, False)
    assert blocked == [1, 1] and events == []


def test_a_key_whose_release_was_missed_does_not_block_the_report():
    lst, _events = _listener()
    blocked = []
    lst.on_blocked = lambda: blocked.append(1)
    lst.enabled = False
    lst.feed(SHIFT, True, lambda k: True)                       # its key-up never arrives (Win+L)
    held = {CTRL}
    lst.feed(CTRL, True, lambda k: k in held)
    held.add(WIN)
    lst.feed(WIN, True, lambda k: k in held)
    assert blocked == [1]


def test_without_a_callback_or_once_enabled_again_nothing_is_reported():
    lst, events = _listener()
    lst.enabled = False
    lst.feed(CTRL, True), lst.feed(WIN, True)                   # no callback set: nothing to call
    blocked = []
    lst.on_blocked = lambda: blocked.append(1)
    lst.enabled = True
    lst.feed(WIN, False), lst.feed(CTRL, False)
    lst.feed(CTRL, True), lst.feed(WIN, True)
    assert blocked == [] and events == ["start"] and not lst._idle_down


# ============================================================================ U-04: the controller
def test_pressing_the_hotkey_while_paused_says_so_on_the_flow_bar(rig):
    rig.c.set_paused(True)
    rig.spy.clear()
    rig.press()
    rig.wait(lambda: rig.c.state == "error")
    assert rig.c.message == ctl_mod.PAUSED_HINT
    assert rig.c.error_info()["message"] == "", "a hint, not a failure with a 'fix it' action"
    rig.keys((WIN, False), (CTRL, False))
    rig.wait(lambda: rig.c.state == "idle")
    rig.settle(80)                                              # cues play on their own thread
    assert rig.rec.starts == 0 and rig.c.paused and rig.sounds == [], "and no error sound"


def test_a_late_blocked_report_after_resuming_is_ignored(rig):
    rig.c._on_hotkey("dictate", "blocked")
    assert rig.c.state == "idle"


# ============================================================================ off the paste path (QA D-03, U-09)
def test_a_dictionary_edit_rebuilds_the_matchers_before_the_next_dictation(rig, monkeypatch):
    monkeypatch.setattr(ctl_mod, "DICT_PREP_MS", 1)
    rig.c._dict_prep.setInterval(1)
    rig.c.dictionary.add("Kubernetes")
    assert rig.c.dictionary._regex is None, "the edit threw the matchers away"
    rig.wait(lambda: rig.c.dictionary._regex is not None)       # rebuilt on its own thread, no dictation needed


def test_a_failing_rebuild_is_logged_and_left_for_the_first_use(rig, monkeypatch, caplog):
    monkeypatch.setattr(rig.c.dictionary, "prepare", lambda: (_ for _ in ()).throw(RuntimeError("no")))
    rig.c._prepare_dictionary()
    assert "dictionary matchers could not be prepared" in caplog.text


def test_a_suggestion_is_learned_after_the_text_was_pasted_and_filed(rig, monkeypatch):
    order = []
    real_begin, real_add = rig.injector.begin, rig.c.history.add_entry
    monkeypatch.setattr(rig.injector, "begin", lambda text: (order.append("paste"), real_begin(text))[1])
    monkeypatch.setattr(rig.c.history, "add_entry", lambda e: (order.append("filed"), real_add(e))[1])
    monkeypatch.setattr(rig.c.dictionary, "learn", lambda raw, cleaned: (order.append("learn"), [])[1])
    rig.dictate("um hello world")
    rig.wait(lambda: "learn" in order)
    assert order == ["paste", "filed", "learn"]


def test_the_flow_bar_draws_no_frames_for_a_status_change_that_changes_nothing(make_ui):
    ui = make_ui()
    ov = ui.shell.overlay
    ui.wait(lambda: not ov._timer.isActive(), timeout=6000)     # the idle pill has settled
    ui.c.status_changed.emit(ui.c.status())
    assert not ov._timer.isActive(), "the same paused state again: the animation stays off"
    ui.c.set_paused(True)
    assert ov._timer.isActive() and ov.paused


def test_the_level_meter_stops_repainting_once_a_silent_microphone_has_settled(qtbot):
    from voxflow.ui.widgets import LevelMeter
    level = [0.4]
    meter = LevelMeter(lambda: level[0])
    qtbot.addWidget(meter)
    paints = []
    meter.update = lambda: paints.append(1)
    meter._tick()
    assert paints and meter._level > 0
    level[0] = 0.0
    for _ in range(400):                                        # 400 frames of at most 0.1 s: long settled
        meter._last -= 0.1
        meter._tick()
    assert (meter._level, meter._peak) == (0.0, 0.0)
    paints.clear()
    for _ in range(5):
        meter._tick()
    assert paints == [], "nothing moves: nothing is drawn"
    level[0] = 0.3
    meter._tick()
    assert paints == [1]


# ============================================================================ History keyboard (QA U-03)
def test_deleting_a_transcript_with_the_keyboard_keeps_the_place_in_the_list(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 4))
    lst = ui.page("history").list
    lst.setFocus()
    ui.qtbot.keyClick(lst, Qt.Key.Key_Home)
    ui.qtbot.keyClick(lst, Qt.Key.Key_Down)
    chosen, following = lst.items[1], lst.items[2]
    assert lst.selected() is chosen
    ui.qtbot.keyClick(lst, Qt.Key.Key_Delete)
    ui.wait(lambda: ui.c.history.count() == 3)
    assert lst.selected()["id"] == following["id"], "the next transcript is chosen, not nothing"
    ui.qtbot.keyClick(lst, Qt.Key.Key_End)
    last = lst.selected()
    ui.qtbot.keyClick(lst, Qt.Key.Key_Delete)
    ui.wait(lambda: ui.c.history.count() == 2)
    assert lst.selected() is lst.items[-1] and lst.selected()["id"] != last["id"], "the last one: the one before it"
    ui.find("history_search").setText("zzz-not-there")          # a different list is no deletion: nothing is chosen
    ui.wait(lambda: not lst.items)
    assert lst.selected() is None


# ============================================================================ double-tap cue (QA C-05)
def test_a_double_tap_plays_one_start_cue_and_a_later_press_plays_its_own(rig):
    now = [100.0]
    rig.c._mono = lambda: now[0]
    rig.tap()
    rig.wait(lambda: rig.sounds == ["start"])                   # cues play on their own thread
    assert rig.c.state == "idle"
    now[0] += 0.2
    rig.tap()
    assert rig.c.state == "locked"
    rig.settle(80)
    assert rig.sounds == ["start"], "the second tap is the same gesture: no second cue"
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done" and rig.sounds == ["start", "stop"])
    rig.pause()
    rig.tap()                                                   # a lone tap ...
    now[0] += 2.0
    rig.pause()
    rig.press()                                                 # ... and a new dictation long after it
    rig.wait(lambda: rig.sounds == ["start", "stop", "start", "start"])
    rig.c.cancel()


def test_with_double_tap_off_or_in_command_mode_every_press_has_its_cue(rig):
    rig.c.set("double_tap_lock", False)
    rig.tap()
    rig.pause(0.1)
    rig.tap()
    rig.wait(lambda: rig.sounds == ["start", "start"])
    rig.c.set("double_tap_lock", True)
    rig.pause()
    rig.tap("command")
    rig.tap("command")
    rig.wait(lambda: rig.sounds == ["start", "start", "command", "command"])


# ============================================================================ leftovers of a killed save (QA D-11)
def test_start_removes_old_temporary_files_of_an_interrupted_save_and_nothing_else(tmp_path, qtbot, monkeypatch, caplog):
    import os
    import time
    from voxflow import config as cfgmod
    data = tmp_path / "data"
    data.mkdir()
    old, fresh, other = data / "tmpab12cd34.tmp", data / "tmpzz99yy88.tmp", data / "notes.tmp"
    for f in (old, fresh, other):
        f.write_text("{}", encoding="utf-8")
    (data / "tmpfolder.tmp").mkdir()
    long_ago = time.time() - 7200
    os.utime(old, (long_ago, long_ago))
    os.utime(other, (long_ago, long_ago))
    with caplog.at_level("INFO", logger="voxflow.controller"):
        r = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert not old.exists(), "the leftover of a save that was killed an hour or more ago is gone"
        assert fresh.exists() and other.exists() and (data / "tmpfolder.tmp").is_dir()
        assert (data / "config.json").exists() and "interrupted save" in caplog.text
    finally:
        r.close()
    assert cfgmod.sweep_tmp(tmp_path / "missing") == 0
    monkeypatch.setattr(cfgmod.Path, "glob", lambda self, pattern: (_ for _ in ()).throw(OSError("offline")))
    assert cfgmod.sweep_tmp(data) == 0


def test_a_leftover_that_cannot_be_removed_is_skipped(tmp_path, monkeypatch):
    import os
    import time
    from voxflow import config as cfgmod
    stuck = tmp_path / "tmpstuck000.tmp"
    stuck.write_text("{}", encoding="utf-8")
    long_ago = time.time() - 7200
    os.utime(stuck, (long_ago, long_ago))
    monkeypatch.setattr(cfgmod.Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("in use")))
    assert cfgmod.sweep_tmp(tmp_path) == 0 and stuck.exists()


# ============================================================================ the next press is already held (QA F-01)
def test_a_dictation_that_finishes_while_the_next_one_is_held_is_pasted_at_the_release(rig):
    rig.injector.modifiers_held = lambda: WIN in rig.held       # what the keyboard would say
    rig.asr.delay_s = 0.1
    texts = iter(["first sentence", "second sentence"])
    rig.asr.text = lambda _audio: next(texts)
    rig.dictate(wait=False)                                     # A: released, being worked on
    rig.pause()
    rig.press()                                                 # B: the user talks on at once
    assert rig.c.state == "listening"
    rig.settle(350)                                             # A is finished long before B is released
    assert rig.injector.pasted == [] and rig.injector.copied == [], "A waits: no paste into held keys, no clipboard"
    rig.release()
    rig.wait(lambda: rig.c.history.count() == 2 and rig.c.state == "done")
    assert rig.injector.pasted == ["First sentence.", "Second sentence."] and rig.spy.notices() == []


def test_the_wait_for_the_release_has_an_end_and_needs_a_keyboard_state(rig, monkeypatch):
    monkeypatch.setattr(ctl_mod, "HELD_PASTE_WAIT_S", 0.05)
    rig.injector.modifiers_held = lambda: True                  # a key that never comes up
    rig.press()
    assert rig.c._active
    assert rig.c._paste("after the wait")[0] and rig.injector.pasted == ["after the wait"]
    rig.c.cancel()
    del rig.injector.modifiers_held
    rig.c.injector = type("Bare", (), {"begin": lambda self, text: True, "finish_later": lambda self: None,
                                       "close": lambda self: None})()
    assert rig.c._paste("no keyboard state to ask") == (True, "Pasted")


def test_a_paste_left_on_the_clipboard_under_a_newer_recording_is_announced(rig):
    entry = dict(rig.c.history.add("An earlier dictation.", "an earlier dictation", 100), rules_only=False)
    rig.press()                                                 # a newer recording owns the bar
    rig.spy.clear()
    rig.c._on_done({"entry": entry, "gen": rig.c._gen - 1, "message": "Copied. Press Ctrl+V", "pasted": True})
    assert rig.c.state == "listening"
    assert [m for _lvl, m in rig.spy.notices() if "press Ctrl+V" in m], "the bar cannot say it: a notice does"
    rig.spy.clear()
    rig.c._on_done({"entry": entry, "gen": rig.c._gen - 1, "message": "Pasted", "pasted": True})
    assert rig.spy.notices() == []
    rig.c.cancel()


# ============================================================================ undo while busy (QA F-02)
def test_undo_waits_for_a_dictation_that_is_still_being_worked_on(rig):
    rig.dictate("the first one")
    gate, entered = _gate_asr(rig)
    rig.dictate(wait=False)                                     # a second one, still with the recogniser
    rig.wait(entered.is_set)
    assert rig.c.undo_last_paste() is False and rig.injector.undone == 0
    assert ("info", "Wait for the dictation to be pasted, then undo.") in rig.spy.notices()
    gate.set()
    rig.wait(lambda: rig.c.state == "done" and len(rig.injector.pasted) == 2)
    assert rig.c.undo_last_paste() is True                      # now it removes the paste that is on screen
    rig.wait(lambda: rig.injector.undone == 1)
