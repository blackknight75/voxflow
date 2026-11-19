"""The live-preview caption of the Flow bar (``OverlayPill.set_preview``): what it shows, when, how it elides long
text, that it never outlives its states, never keeps the bar's timer running, never speaks or logs the words, and how
the controller's ``partial_text`` signal and the Settings toggle reach it."""
from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QObject

from tests.cov_ui_kit import demo, flush, grab_bytes, make_demo  # noqa: F401 - fixtures
from tests.helpers import fake_time
from voxflow.config import Config
from voxflow.ui import overlay as O
from voxflow.ui import theme as T
from voxflow.ui.overlay import OverlayPill

LONG = ("The quick brown fox jumps over the lazy dog. Please schedule the meeting for Thursday at three thirty in "
        "the afternoon and send the report to the engineering team before the end of the day.")


class Voice:
    def level(self):
        return 0.3

    def recent_samples(self, n):
        import numpy as np
        return np.full(n, 0.3, dtype=np.float32)


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 900.0}
    monkeypatch.setattr(O, "time", fake_time(perf_counter=lambda: state["t"]))
    return state


@pytest.fixture
def pill(qapp, qt_messages, clock):
    T.apply_theme("dark", True)                      # reduced motion: every weight snaps, one frame is the final look
    p = OverlayPill(Voice(), show_idle=False)
    yield p
    p._timer.stop()
    p._follow.stop()
    p.hide()
    p.deleteLater()
    flush()
    T.apply_theme("dark", False)
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


def frames(p, clock, n=1, dt=0.02):
    """Advance the scripted clock and run the animation by hand (the real 16 ms timer is stopped)."""
    p._timer.stop()
    for _ in range(n):
        clock["t"] += dt
        p._tick()


def shot(p, clock):
    frames(p, clock)
    return grab_bytes(p)


# =========================================================================== when it shows
@pytest.mark.parametrize("state", ["listening", "locked", "processing"])
def test_the_caption_is_painted_while_listening_hands_free_and_processing(pill, clock, state):
    pill.set_state("listening")
    pill.set_state(state)
    bare = shot(pill, clock)
    pill.set_preview("hello there", "my friend")
    assert pill._pv_head == "hello there " and pill._pv_tail == "my friend"
    assert shot(pill, clock) != bare


@pytest.mark.parametrize("state", ["idle", "done", "error", "loading", "arrange"])
def test_text_that_arrives_in_any_other_state_is_dropped(pill, clock, state):
    pill.set_state(state, "x")
    before = shot(pill, clock)
    pill.set_preview("hello there", "my friend")
    assert (pill._pv_committed, pill._pv_tentative, pill._pv_head, pill._pv_tail) == ("", "", "", "")
    assert shot(pill, clock) == before


def test_the_caption_goes_when_the_bar_leaves_its_states_or_a_new_utterance_starts(pill, clock):
    pill.set_state("listening")
    pill.set_preview("one two", "three")
    pill.set_state("locked")
    pill.set_state("processing")
    assert pill._pv_head, "hands-free and processing keep what was shown"
    pill.set_state("done", "Pasted")
    assert (pill._pv_head, pill._pv_tail, pill._pv_w) == ("", "", 0.0)
    pill.set_state("listening")
    pill.set_preview("four", "")
    pill.set_state("processing")
    pill.set_state("listening")                              # the next utterance starts while the old one finishes
    assert pill._pv_head == "" and pill._pv_committed == ""
    pill.set_preview("five", "")
    pill.set_state("idle")
    assert pill._pv_head == ""
    pill.set_state("listening")
    pill.set_preview("six", "")
    pill.set_preview("", "")
    assert (pill._pv_head, pill._pv_tail, pill._pv_head_w) == ("", "", 0.0)
    frames(pill, clock)
    assert pill._pv_w == 0.0


def test_a_caption_that_has_not_changed_does_nothing(pill, clock, monkeypatch):
    pill.set_state("listening")
    pill.set_preview("hello", "wor")
    kicks = []
    monkeypatch.setattr(pill, "_kick", lambda: kicks.append(1))
    pill.set_preview("hello", "wor")
    pill.set_preview("  hello ", " wor ")                    # the same words
    assert kicks == []
    pill.set_preview("hello", "world")
    pill.set_preview(None, None)
    assert kicks == [1, 1]
    assert pill._pv_head == ""


# =========================================================================== long text
def test_long_text_keeps_its_end_and_loses_its_start(pill, clock):
    pill.set_state("listening")
    committed, tentative = LONG[:-20], LONG[-20:].strip()
    pill.set_preview(committed, tentative)
    shown = pill._pv_head + pill._pv_tail
    assert shown.startswith("…") and shown.endswith(tentative) and len(shown) < len(LONG)
    assert pill._pv_tail == tentative, "the unstable words stay the dimmed part"
    assert pill._pv_head_w + pill._pv_tail_w <= O.PREVIEW_MAX_W - 2 * O.PREVIEW_PAD_X + 1
    pill.set_preview(committed, "")
    assert pill._pv_tail == "" and pill._pv_head.endswith(committed[-20:])
    bubble_w = pill._pv_head_w + 2 * O.PREVIEW_PAD_X
    assert bubble_w <= O.PREVIEW_MAX_W + 1 < O.CANVAS_W, "the bubble stays inside the Flow bar's window"


def test_the_split_lands_on_a_word_boundary_with_one_space_when_the_start_is_elided(pill):
    """What the engine sends: whole words, committed + tentative joined by one space."""
    words = (LONG + " " + LONG).split()
    for n_tentative in (1, 2, 3, 5):
        committed, tentative = " ".join(words[:-n_tentative]), " ".join(words[-n_tentative:])
        pill.set_state("idle")
        pill.set_state("listening")
        pill.set_preview(committed, tentative)
        head, tail = pill._pv_head, pill._pv_tail
        assert head.startswith("…") and head.endswith(" ") and not head.endswith("  "), (head, tail)
        assert tail == tentative and not tail.startswith(" ")
        assert (head + tail).split(" ")[1:] == words[-len((head + tail).split(" ")[1:]):], "whole words only"
        assert pill._pv_head_w > 0 and pill._pv_tail_w > 0


def test_the_elided_start_is_a_whole_word_never_punctuation_or_half_a_word(pill):
    """Qt elides at a character; seen in the 0.3.0 demo as "…. uh. move the the meeting" (the cut landed just after
    "should", before its full stop). Every cut position over a sentence full of punctuation must start the caption
    on a whole word."""
    spoken = ("Um, so I think we should. uh. move the the meeting to Thursday. no wait, Friday at 3.30. and uh, "
              "send the report to the engineering team before the end of the day, you know, so they can review it.")
    for drop in range(0, 110):                # the caption keeps the END, so shortening the end moves the cut
        full = spoken[:len(spoken) - drop].strip()
        pill.set_state("idle")
        pill.set_state("listening")
        pill.set_preview(full, "")
        shown = pill._pv_head + pill._pv_tail
        assert shown.startswith("…"), full
        rest = shown[1:]
        assert full.endswith(rest) and rest[0] not in O._CUT_LEAD, (full, shown)
        assert full[len(full) - len(rest) - 1] in O._CUT_LEAD, f"half a word after the ellipsis: {shown!r}"


@pytest.mark.parametrize("text, expect", [
    ("x" * 400, "x"),                       # one word longer than the bubble: the plain cut is kept
    ("a" + "." * 400, "."),                 # only punctuation is left: the plain cut is kept
    ("y" * 400 + " !!!", "y"),              # the word after the cut is punctuation only: no move to it
])
def test_an_elided_start_with_no_word_to_move_to_keeps_the_plain_cut(pill, text, expect):
    pill.set_state("listening")
    pill.set_preview(text, "")
    assert pill._pv_head.startswith("…" + expect) and text.endswith(pill._pv_head[1:])
    assert pill._pv_head_w <= O.PREVIEW_MAX_W - 2 * O.PREVIEW_PAD_X + 1


def test_a_tentative_part_longer_than_the_bubble_is_cut_but_still_dimmed(pill, clock):
    pill.set_state("listening")
    pill.set_preview("ok", LONG)
    assert pill._pv_head == "…" and pill._pv_tail.endswith("end of the day.")
    pill.set_preview("", "just this")
    assert (pill._pv_head, pill._pv_tail) == ("", "just this")


def test_short_text_is_shown_whole(pill):
    pill.set_state("listening")
    pill.set_preview("Hello", "")
    assert (pill._pv_head, pill._pv_tail) == ("Hello", "")
    pill.set_preview("", "wor")
    assert (pill._pv_head, pill._pv_tail) == ("", "wor")
    pill.set_preview("Hello", "wor")
    assert (pill._pv_head, pill._pv_tail) == ("Hello ", "wor")


# =========================================================================== look and feel
def test_the_caption_wins_the_spot_over_the_hands_free_hint(pill, clock, monkeypatch):
    pill.set_state("listening")
    pill.set_state("locked")
    pill.set_preview("hello", "there")
    frames(pill, clock)
    with_both = grab_bytes(pill)                             # the hint would be showing right now
    monkeypatch.setattr(O, "LOCK_HINT_S", 0.0)
    assert grab_bytes(pill) == with_both, "the hint adds nothing while a caption is shown"
    pill.set_preview("", "")
    monkeypatch.setattr(O, "LOCK_HINT_S", 2.6)
    pill._lock_t = clock["t"]
    assert grab_bytes(pill) != with_both


def test_the_caption_fades_in_unless_motion_is_reduced(pill, clock):
    T.apply_theme("dark", False)
    pill.set_state("listening")
    pill.set_preview("hello", "there")
    assert pill._pv_w == 0.0
    frames(pill, clock, 1, dt=0.016)
    first = pill._pv_w
    frames(pill, clock, 40, dt=0.016)
    assert 0.0 < first < pill._pv_w <= 1.0 and pill._pv_w > 0.95
    T.apply_theme("dark", True)
    pill.set_preview("hello", "")
    pill._pv_w = 0.0
    frames(pill, clock)
    assert pill._pv_w == 1.0


def test_the_bubble_is_drawn_once_per_change_not_once_per_frame(pill, clock):
    pill.set_state("listening")
    pill.set_preview("hello", "there")
    shot(pill, clock)
    first = pill._pv_pm
    assert first is not None
    shot(pill, clock)
    assert pill._pv_pm is first, "the same pixmap is blitted every frame"
    pill._pv_dpr = 7.0                                       # the bar moved to a screen with another scale
    shot(pill, clock)
    assert pill._pv_pm is not first and pill._pv_dpr == pill.devicePixelRatioF()
    pill.set_preview("hello", "there again")
    assert pill._pv_pm is None, "new words are drawn on the next paint"
    pill.set_preview("", "")
    assert pill._pv_pm is None


def test_the_caption_does_not_keep_the_bar_awake(pill, clock):
    pill.set_state("listening")
    pill.set_preview("hello", "there")
    pill.set_state("idle")
    for _ in range(100):                                     # the bar fades out and stops its own timer
        if not pill._timer.isActive():
            break
        clock["t"] += 0.05
        pill._tick()
    assert not pill._timer.isActive() and not pill.isVisible()
    pill.set_preview("late words", "")
    assert not pill.isVisible() and not pill._timer.isActive(), "a stale caption cannot wake the bar"


# =========================================================================== privacy
def test_the_words_are_neither_announced_nor_logged(pill, clock, monkeypatch, caplog):
    said = []
    monkeypatch.setattr(O.a11y, "announce", lambda *a, **k: said.append(a[1]))
    caplog.set_level(logging.DEBUG)
    pill.set_state("listening")
    pill.set_preview("my secret", "password")
    pill.set_state("processing")
    pill.set_state("done", "Pasted")
    assert said == ["Pasted"]
    assert "secret" not in caplog.text and "password" not in caplog.text


# =========================================================================== wiring
def test_the_controllers_partial_text_reaches_the_flow_bar(demo):
    overlay, c = demo.shell.overlay, demo.c
    c.demo_set_state("listening")
    c.partial_text.emit("good morning", "every")
    assert (overlay._pv_committed, overlay._pv_tentative) == ("good morning", "every")
    c.partial_text.emit("", "")
    assert overlay._pv_head == "" and overlay._pv_tail == ""
    c.partial_text.emit("late", "")
    c.demo_set_state("done", "dictate", "Pasted")
    c.partial_text.emit("later still", "")
    assert overlay._pv_head == ""


def test_the_setting_is_on_by_default_and_has_a_toggle(demo):
    assert Config().live_preview is True and demo.c.get("live_preview") is True
    toggle = demo.shell.window.pages["settings"].findChild(QObject, "settings_live_preview")
    assert toggle is not None and toggle.accessibleName()
    toggle.click()
    assert demo.c.get("live_preview") is False
