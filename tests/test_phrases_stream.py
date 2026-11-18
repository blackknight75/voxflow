"""``Previewer(on_words=...)`` and ``PhraseStream`` (voxflow.phrases) with the scripted doubles of test_stream.py.
No model, no microphone (phrase-streaming design notes 4.2, 4.3, 10.2)."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.test_stream import SR, Rig, loud, words
from voxflow import phrases as ph
from voxflow import stream


def rd(ws):
    return [(w, round(t, 3)) for w, t in ws]


def rig(hyps=(), **kw):
    got: list[tuple] = []
    gaps: list[bool] = []

    def on_words(*a, gap=False):
        got.append(a)
        gaps.append(gap)

    r = Rig(hyps, on_words=on_words, **kw)
    r.got, r.gaps = got, gaps
    return r


# ======================================================================== on_words from the Previewer
def test_on_words_follows_every_pass_with_only_the_new_committed_words_and_the_audio_time():
    r = rig([words("hello wor"), words("hello world how"), words("hello world how are you"),
             words("hello world how are you")])
    r.begin(7)
    r.tick()
    assert r.got[-1][:3] == (7, [], [("hello", 0.0), ("wor", 0.3)])
    assert r.got[-1][3] == pytest.approx(0.5)                                       # recorded samples / 16000
    r.tick()
    gen, new, tent, now = r.got[-1]
    assert new == [("hello", 0.0)] and tent == [("world", 0.3), ("how", 0.6)] and now == pytest.approx(1.0)
    r.tick()
    assert rd(r.got[-1][1]) == [("world", 0.3), ("how", 0.6)] and rd(r.got[-1][2]) == [("are", 0.9), ("you", 1.2)]
    r.tick()
    assert rd(r.got[-1][1]) == [("are", 0.9), ("you", 1.2)] and r.got[-1][2] == []
    n_got, n_out = len(r.got), len(r.out)
    r.tick()                                                                        # same words: the caption is silent ...
    assert len(r.out) == n_out and len(r.got) == n_got + 1                          # ... on_words is not
    assert r.got[-1][1] == [] and r.got[-1][2] == []


def test_the_committed_words_handed_over_are_a_copy_with_absolute_times():
    r = rig([words("one two", start=0.2), words("one two three", start=0.2)])
    r.begin()
    r.tick()
    r.tick()
    assert r.got[-1][1] == [("one", 0.2), ("two", 0.5)]
    r.got[-1][1].append(("x", 9.0))
    assert r.p._la.committed == [("one", 0.2), ("two", 0.5)]


def test_a_tick_without_a_pass_is_reported_as_time_only():
    # too little new audio: nobody looked at it, time does not move
    r = rig([words("hi")])
    r.begin()
    r.tick(grow=0.1)
    assert r.got == [(1, [], None, 0.0)]
    # before the first 0.7 s: the same
    r = rig([words("hi")])
    r.begin(gate=True)
    r.tick(grow=0.5)
    assert r.got == [(1, [], None, 0.0)] and r.asr.calls == []
    # silence, after the first look: the peak was measured, so the time moves
    r = rig([[], words("back")])
    r.rec.audio = np.concatenate([np.zeros(SR * 3, np.float32), loud(5)])
    r.begin()
    r.tick(1.0)
    assert r.got[-1][2] == []                                                       # a pass (no words): tentative []
    r.tick(1.0)
    assert r.got[-1] == (1, [], None, pytest.approx(2.0))                           # skipped silence: time only
    # backing off: the audio was only peak-checked, not transcribed, so time does not move
    r = rig([words("hi")])
    r.begin()
    r.p._hold_until, r.p._ref_peak = r.clock.t + 100, 0.1
    r.tick(1.0)
    assert r.got == [(1, [], None, 0.0)] and r.asr.calls == []


def test_time_only_ticks_never_report_audio_that_was_not_examined():
    # a pass looks at 0.5 s; a short tick and a busy model then leave the reported time at that pass's end
    r = rig([words("hi there you")])
    r.begin()
    r.tick(0.5)
    assert r.got[-1][3] == pytest.approx(0.5) and r.got[-1][2] is not None
    r.tick(grow=0.1)                                                                # too little new audio
    assert r.got[-1] == (1, [], None, pytest.approx(0.5))
    r.asr.busy = True
    r.tick(1.0)                                                                     # the model is busy: 1.2 s exist
    assert r.got[-1] == (1, [], None, pytest.approx(0.5)) and r.rec.n == int(1.6 * SR)
    r.asr.busy = False
    r.tick(0.0)
    assert r.got[-1][3] == pytest.approx(1.6) and r.got[-1][2] is not None          # looked at now: time moves


def test_nothing_is_reported_after_stop_or_for_a_replaced_session():
    r = rig([words("never shown")])
    r.begin()
    r.asr.hook = r.p.stop
    r.tick()
    assert r.got == []
    r.p._tick()                                                                     # not active: nothing
    assert r.got == []
    # a busy tick whose session ended during the call
    r = rig([words("x")])
    r.begin()
    r.asr.busy, r.asr.hook = True, r.p.stop
    r.tick()
    assert r.got == []
    # replaced during the call
    r = rig([words("old words")])
    r.begin(1)

    def replace():
        r.asr.hook = None
        r.begin(2)

    r.asr.hook = replace
    r.tick()
    assert r.got == []


def test_on_words_is_called_after_the_lock_is_released():
    r = rig([words("a b c")])
    seen = []

    def on_words(*a, **kw):
        got = r.p._cv.acquire(blocking=False)
        seen.append(got)
        if got:
            r.p._cv.release()

    r.p._on_words = on_words
    r.begin()
    r.tick()
    r.tick(grow=0.0)
    assert seen and all(seen)


def test_a_failing_words_receiver_is_swallowed_and_never_logs_the_words(caplog):
    r = rig([words("secret words")])
    r.p._on_words = lambda *a, **kw: 1 / 0
    r.begin()
    with caplog.at_level(logging.DEBUG, logger="voxflow.stream"):
        r.tick()
        r.tick(grow=0.1)
    assert "words receiver failed (ZeroDivisionError)" in caplog.text and "secret" not in caplog.text
    assert r.out == [(1, "", "secret words")]                                       # the caption still came


def test_without_on_words_the_caption_is_the_same_as_before():
    plain = Rig([words("hello wor"), words("hello world how"), words("hello world how are you")])
    wired = rig([words("hello wor"), words("hello world how"), words("hello world how are you")])
    for r in (plain, wired):
        r.begin(7)
        for _ in range(3):
            r.tick()
        r.tick(grow=0.1)
        r.asr.busy = True
        r.tick()
    assert plain.out == wired.out and plain.asr.calls[0]["size"] == wired.asr.calls[0]["size"]


# ======================================================================== PhraseStream
class Snap:
    """A recorder whose ``snapshot(start)`` returns the audio from ``start``."""

    def __init__(self, audio):
        self.audio, self.calls, self.fail = audio, [], False

    def snapshot(self, start=0):
        self.calls.append(start)
        if self.fail:
            raise RuntimeError("recorder is gone")
        return self.audio[start:].copy()


def stream_rig(audio=None, **kw):
    out: list[tuple] = []
    audio = np.full(SR * 20, 0.003, np.float32) if audio is None else audio     # room noise: no speech
    rec = Snap(audio)
    ps = ph.PhraseStream(rec, lambda g, p: out.append((g, p)), **kw)
    return ps, rec, out


SENTENCE = [("we", 1.0), ("met", 1.5), ("today.", 2.0), ("Then", 3.0)]


def test_a_phrase_is_emitted_with_the_gen_and_the_cut_at_the_quietest_point():
    audio = np.full(SR * 20, 0.003, np.float32)
    audio[int(2.71 * SR):int(2.75 * SR)] = 0.0                                       # a quiet frame inside the window
    ps, rec, out = stream_rig(audio)
    ps.begin(5)
    ps.feed(5, SENTENCE, [("Then", 3.0)], 3.4)
    (gen, p), = out
    assert gen == 5 and p.text == "we met today." and p.words == tuple(SENTENCE[:3])
    assert p.cut_lo < p.cut_s < p.cut_hi and p.cut_s == pytest.approx(2.73, abs=0.005)
    assert rec.calls == [int(p.cut_lo * SR)]


def test_feed_without_tentative_words_and_time_only_ticks():
    ps, rec, out = stream_rig()
    ps.begin(1)
    ps.feed(1, SENTENCE[:3], [], 2.2)                                               # no tentative: [] -> start None
    assert out == []
    ps.feed(1, [], None, 3.0)                                                       # 0.45 s: a "." not yet confirmed
    assert out == []
    ps.feed(1, [], None, 4.5)                                                       # time only: 1.95 s of silence
    assert [p.text for _, p in out] == ["we met today."]
    ps.feed(1, [], None, 9.0)
    assert len(out) == 1


def test_gen_mismatch_and_calls_after_end_or_before_begin_are_ignored():
    ps, rec, out = stream_rig()
    ps.feed(1, SENTENCE[:3], [], 9.0)                                               # no session yet
    ps.begin(2)
    ps.feed(1, SENTENCE[:3], [], 9.0)                                               # another gen
    assert out == [] and ps.splitter.pending == ()
    ps.feed(2, SENTENCE[:3], [], 9.0)
    assert len(out) == 1
    ps.end()
    ps.feed(2, [("more", 10.0)], [], 20.0)
    assert len(out) == 1
    ps.begin(3)                                                                     # a new session starts clean
    assert ps.splitter.pending == ()
    ps.feed(3, SENTENCE[:3], [], 9.0)
    assert [g for g, _ in out] == [2, 3] and out[1][1].seq == 1


def test_the_cut_falls_back_to_the_midpoint_when_the_recorder_fails_or_is_empty():
    ps, rec, out = stream_rig()
    rec.fail = True
    ps.begin(1)
    ps.feed(1, SENTENCE, [], 9.0)
    p = out[-1][1]
    assert p.cut_s == pytest.approx((p.cut_lo + p.cut_hi) / 2)
    ps2, rec2, out2 = stream_rig(np.zeros(0, np.float32))
    ps2.begin(1)
    ps2.feed(1, SENTENCE, [], 9.0)
    p = out2[-1][1]
    assert p.cut_s == pytest.approx((p.cut_lo + p.cut_hi) / 2)


def test_cuts_never_go_backwards_and_a_custom_splitter_and_frame_are_used():
    sp = ph.PhraseSplitter(min_words=1)
    ps, rec, out = stream_rig(splitter=sp, frame_s=0.02)
    assert ps.splitter is sp
    ps.begin(1)
    ws = [("one,", 1.0), ("two", 2.5), ("three,", 4.0), ("four", 5.5)]
    ps.feed(1, ws, [], 5.7)
    cuts = [p.cut_s for _, p in out]
    assert len(cuts) >= 2 and cuts == sorted(cuts) and all(p.cut_lo <= p.cut_s <= p.cut_hi for _, p in out)
    # a refined cut that lies before the previous one is raised to it
    ps._cut = 99.0
    ps.feed(1, [("five", 7.0), ("six", 8.5)], [], 8.7)
    assert out[-1][1].cut_s == 99.0


def test_the_real_chain_previewer_to_phrase_stream():
    """Previewer.on_words -> PhraseStream.feed -> phrases, with committed words coming from scripted passes."""
    hyps = [[("we", 1.0), ("met", 1.5), ("today.", 2.0), ("Then", 3.2)]] * 6
    r = Rig(hyps, seconds=30.0)
    snap = Snap(np.zeros(SR * 30, np.float32))
    out = []
    ps = ph.PhraseStream(snap, lambda g, p: out.append((g, p)))
    r.p._on_words = ps.feed
    ps.begin(1)
    r.begin(1)
    for _ in range(6):
        r.tick(0.7)
    assert out and out[0][1].text == "we met today." and out[0][0] == 1
    assert stream.norm("x") == "x"


def test_no_pause_phrase_is_emitted_across_audio_nobody_examined():
    """Splitter level: only the examined time counts as silence. The words end at 2.0; 0.5 s were examined, and the
    recording went on while the model was busy (3 s more that may hold speech): no pause, no cut window there."""
    sp = ph.PhraseSplitter()
    ws = [("we", 1.0), ("met", 1.5), ("today", 2.0)]
    assert sp.feed(ws, None, 2.6) == []
    for _ in range(5):
        assert sp.tick(2.6) == []                                                   # time does not move: nothing
    out = sp.tick(3.3)                                                              # examined silence: the pause
    assert [p.text for p in out] == ["we met today"] and out[0].cut_hi <= 3.3


def test_a_busy_model_never_lets_the_chain_cut_into_unseen_speech():
    hyps = [[("we", 1.0), ("met", 1.5), ("today", 2.0)]] * 4
    r = Rig(hyps, seconds=30.0)
    out = []
    ps = ph.PhraseStream(Snap(np.zeros(SR * 30, np.float32)), lambda g, p: out.append(p))
    r.p._on_words = ps.feed
    ps.begin(1)
    r.begin(1)
    r.tick(2.6)                                                                     # examined up to 2.6 s
    r.asr.busy = True
    for _ in range(3):
        r.tick(1.0)                                                                 # 3 s more recorded, never examined
    assert out == []
    r.asr.busy = False
    r.tick(0.0)                                                                     # looked at: it agrees with pass 1
    assert [p.text for p in out] == ["we met today"] and all(p.cut_hi <= r.rec.n / SR for p in out)


def test_heard_gives_every_word_in_time_order_and_the_covered_time_and_outlives_end():
    ps, rec, out = stream_rig()
    assert ps.heard() == ((), 0.0)
    ps.begin(1)
    assert ps.heard() == ((), 0.0)
    ps.feed(1, [("we", 1.0), ("met", 1.5)], [("today", 2.0)], 2.6)
    assert ps.heard() == ((("we", 1.0), ("met", 1.5), ("today", 2.0)), 2.6)
    ps.feed(1, [("today", 2.0)], [("then", 3.0)], 3.4)                        # the tentative words are replaced
    assert ps.heard() == ((("we", 1.0), ("met", 1.5), ("today", 2.0), ("then", 3.0)), 3.4)
    ps.feed(1, [], None, 9.0)                                                  # time only: nothing was looked at
    assert ps.heard()[1] == 3.4
    ps.feed(2, [("x", 9.0)], [], 9.5)                                          # another gen: ignored
    assert len(ps.heard()[0]) == 4
    ps.end()
    assert ps.heard()[1] == 3.4 and len(ps.heard()[0]) == 4                    # still valid after end()
    ps.begin(2)
    assert ps.heard() == ((), 0.0)


def test_tail_hint_is_the_first_word_at_or_after_the_cut():
    heard = (("we", 1.0), ("met", 1.5), ("today", 2.0), ("then", 3.0))
    assert ph.tail_hint(heard, 2.5) == "then"
    assert ph.tail_hint(heard, 1.96) == "today"                                # 50 ms of slack before the cut
    assert ph.tail_hint(heard, 3.5) is None and ph.tail_hint((), 0.0) is None


def test_begin_end_feed_heard_exclude_each_other():
    ps, rec, out = stream_rig()
    ps.begin(1)
    seen = []

    def on_phrase(g, p):
        got = ps._lock.acquire(blocking=False)                                  # the lock is held while phrases go out
        seen.append(got)

    ps._on_phrase = on_phrase
    ps.feed(1, SENTENCE[:3], [], 9.0)
    assert seen == [False]


# ======================================================================== audio the preview skipped (gap)
def test_a_normal_session_never_reports_a_gap():
    r = rig([words("one two three four five six"), words("one two three four five six seven eight")])
    r.begin()
    for _ in range(10):
        r.tick(0.5)
    r.asr.busy = True
    r.tick(0.1)
    r.tick(0.5)                                                                     # busy: a time-only tick, no gap
    assert r.gaps and not any(r.gaps)


def test_spaced_out_calls_that_skip_tentative_words_are_a_gap_reported_once():
    """Jump 1 with a real loss: the tentative words of the last pass are cut off by the forced window start."""
    r = rig([words("aa bb cc")], trim_s=3.0, seconds=60.0)
    r.begin()
    r.tick(0.5)
    r.tick(0.5)
    assert r.gaps == [False, False]
    r.tick(8.0)                                                                     # 9 s in the window > 3 s + interval
    assert r.gaps == [False, False, True] and r.p._win_start == int(6.0 * SR)
    r.p._trim_s = 100.0                                                             # no window floor from here on
    r.tick(0.5)
    r.tick(0.5)
    assert r.gaps == [False, False, True, False, False]                             # once


def test_a_jump_over_audio_nobody_examined_is_a_gap_even_without_tentative_words():
    """Jump 1 with a real loss: the model was not called for a long time, the new window start lies after the audio
    the last pass examined (0.5 s), so what lies between was never transcribed."""
    r = rig([[]], trim_s=3.0, seconds=60.0)
    r.begin()
    r.tick(0.5)
    r.p._hold_until = 0.0
    r.tick(8.0)
    assert r.gaps == [False, True]


def test_a_jump_over_audio_that_was_examined_and_held_no_words_is_not_a_gap():
    """Jump 1 without a loss: a long pause, passes with no words at all; the window start moves over audio that the
    previous passes did examine."""
    r = rig([[]], trim_s=3.0, seconds=60.0)
    r.begin()
    for _ in range(8):
        r.p._hold_until = 0.0
        r.tick(2.0)
    assert len(r.gaps) == 8 and not any(r.gaps) and r.p._win_start > 0              # the jump did happen


def test_a_forced_jump_on_a_tick_the_model_was_busy_for_is_still_reported_once():
    r = rig([words("aa bb cc")], trim_s=3.0, seconds=60.0)
    r.begin()
    r.tick(0.5)
    r.asr.busy = True
    r.tick(8.0)                                                                     # the window start moved, no pass
    assert r.got[-1][2] is None and r.gaps[-1] is True
    r.p._trim_s = 100.0                                                             # no window floor from here on
    r.asr.busy = False
    r.tick(0.5)
    r.tick(0.5)
    assert r.gaps[-2:] == [False, False]


def test_the_window_floor_that_cuts_off_tentative_words_is_a_gap_reported_once():
    """Jump 2 with a real loss: the floor lies after the start of the first tentative word."""
    r = rig([words("aa bb"), words("cc dd")], trim_s=1.0)
    r.begin()
    r.tick(0.6)
    assert r.gaps == [False]
    r.tick(0.6)                                                                     # 1.2 s > 1 s, the words never agree
    assert r.gaps == [False, True] and r.p._win_start == int(0.2 * SR)
    r.tick(0.1)                                                                     # too little audio: time only
    assert r.gaps[-1] is False


def test_the_window_floor_over_wordless_or_uncut_audio_is_not_a_gap():
    """Jump 2 without a loss: no tentative words (the audio up to ``end`` was examined and wordless), or the floor
    lies before the first tentative word (nothing was cut off)."""
    r = rig([[]], trim_s=1.0)
    r.begin()
    r.tick(0.6)
    r.p._hold_until = 0.0
    r.tick(0.6)
    assert r.p._win_start == int(0.2 * SR) and r.gaps == [False, False]            # forced, nobody lost
    r = rig([words("aa bb", start=0.35), words("cc dd", start=0.35)], trim_s=1.0)
    r.begin()
    r.tick(0.6)
    r.tick(0.6)
    assert r.p._win_start == int(0.2 * SR) and r.gaps == [False, False]            # floor 0.2 s < first word 0.35 s


def test_trim_to_a_word_is_not_a_gap():
    """Starting the window at the first uncommitted word skips only audio whose words are committed."""
    hyps = [words("aa bb"), words("aa bb cc dd"), words("aa bb cc dd ee ff")]
    r = rig(hyps, trim_s=1.0, bucket_s=0, interval_s=2.0)
    r.begin()
    for _ in range(3):
        r.tick(0.5)
    assert r.p._win_start == pytest.approx(0.8 * SR, abs=2) and r.gaps == [False, False, False]


def test_a_long_noisy_pause_does_not_stop_the_stream_end_to_end():
    """A speaker stops talking for longer than the window in a room that is not silent: passes with no words, the
    window floor and the spaced-out jump both fire on audio that WAS transcribed and was wordless. No gap; the
    phrases go on afterwards."""
    first = words("we met today", start=0.0, step=0.5)
    again = words("and then more words", start=1.5, step=0.4)
    r = Rig([first, first] + [[]] * 14 + [again, again, []], trim_s=3.0, seconds=90.0)
    out = []
    gaps = []
    ps = ph.PhraseStream(r.rec, lambda g, p: out.append(p))

    def on_words(*a, gap=False):
        gaps.append(gap)
        ps.feed(*a, gap=gap)

    r.p._on_words = on_words
    ps.begin(1)
    r.begin(1)
    for _ in range(50):
        r.p._hold_until = 0.0
        r.tick(0.4)                                                                 # one preview interval of audio per tick
    assert not any(gaps) and not ps.stopped                                         # the pause was not a gap
    texts = [p.text for p in out]
    assert texts[0] == "we met today" and len(texts) >= 2 and texts[1].endswith("words")
    assert [w for w, _ in ps.heard()[0]][-4:] == ["and", "then", "more", "words"]


def test_phrase_stream_stops_for_good_on_a_gap_and_heard_keeps_working():
    ps, rec, out = stream_rig()
    ps.begin(1)
    assert not ps.stopped
    ps.feed(1, SENTENCE[:3], [], 9.0)                                               # a phrase before the gap stays sent
    assert len(out) == 1 and not ps.stopped
    ps.feed(1, [("next", 10.0), ("one", 10.4), ("here.", 10.8), ("Then", 12.0)], [], 12.5, gap=True)
    assert ps.stopped and len(out) == 1                                             # nothing from the call with the gap
    ps.feed(1, [("more", 13.0), ("words", 13.4), ("again.", 13.8), ("Ok", 15.0)], [], 16.0)
    assert len(out) == 1                                                            # nor ever after
    words_, covered = ps.heard()
    assert ("Then", 12.0) in words_ and words_[-1] == ("Ok", 15.0) and covered == 16.0
    ps.begin(2)
    assert not ps.stopped
    ps.feed(2, SENTENCE[:3], [], 9.0)
    assert len(out) == 2


def test_words_inside_skipped_audio_are_never_claimed_by_a_phrase_cut():
    """The bench's real loss (similarity 0.535): the preview stalled and its window jumped over 17 s of speech, so the
    committed words ended at "at" and resumed at "And". A "pause" phrase "... at" then got a cut window over all of
    the skipped audio and its cut fell in the middle of words nobody had heard."""
    ws = [("Please", 1.0), ("schedule", 1.4), ("the", 1.8), ("meeting", 2.0), ("at", 2.6)]
    resumed = [("And", 22.0), ("then", 22.3)]
    ps, rec, out = stream_rig()
    ps.begin(1)
    ps.feed(1, ws, [], 3.0)                                                         # no pause yet
    ps.feed(1, resumed, [], 23.0, gap=True)                                         # the window jumped past 3 s .. 22 s
    assert out == [] and ps.stopped
    # the splitter alone, which cannot know about the jump, did claim a cut inside the skipped audio
    sp = ph.PhraseSplitter()
    sp.feed(ws, None, 3.0)
    claimed = sp.feed(resumed, None, 23.0)
    assert claimed and claimed[0].cut_lo < 6.0 < claimed[0].cut_s < 22.0
