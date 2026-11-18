"""``Previewer(on_words=...)`` and ``PhraseStream`` (voxflow.phrases) with the scripted doubles of test_stream.py.
No model, no microphone (plan/PHRASE_STREAMING_DESIGN.md 4.2, 4.3, 10.2)."""
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
    r = Rig(hyps, on_words=lambda *a: got.append(a), **kw)
    r.got = got
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
    # too little new audio
    r = rig([words("hi")])
    r.begin()
    r.tick(grow=0.1)
    assert r.got == [(1, [], None, pytest.approx(0.1))]
    # before the first 0.7 s
    r = rig([words("hi")])
    r.begin(gate=True)
    r.tick(grow=0.5)
    assert r.got == [(1, [], None, pytest.approx(0.5))] and r.asr.calls == []
    # silence, after the first look
    r = rig([[], words("back")])
    r.rec.audio = np.concatenate([np.zeros(SR * 3, np.float32), loud(5)])
    r.begin()
    r.tick(1.0)
    assert r.got[-1][2] == []                                                       # a pass (no words): tentative []
    r.tick(1.0)
    assert r.got[-1] == (1, [], None, pytest.approx(2.0))                           # skipped silence: time only
    # backing off
    r = rig([words("hi")])
    r.begin()
    r.p._hold_until, r.p._ref_peak = r.clock.t + 100, 0.1
    r.tick(1.0)
    assert r.got == [(1, [], None, pytest.approx(1.0))] and r.asr.calls == []
    # a busy model
    r = rig([words("hi")])
    r.begin()
    r.asr.busy = True
    r.tick(1.0)
    assert r.got == [(1, [], None, pytest.approx(1.0))]


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

    def on_words(*a):
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
    r.p._on_words = lambda *a: 1 / 0
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
    audio = np.full(SR * 20, 0.1, np.float32) if audio is None else audio
    rec = Snap(audio)
    ps = ph.PhraseStream(rec, lambda g, p: out.append((g, p)), **kw)
    return ps, rec, out


SENTENCE = [("we", 1.0), ("met", 1.5), ("today.", 2.0), ("Then", 3.0)]


def test_a_phrase_is_emitted_with_the_gen_and_the_cut_at_the_quietest_point():
    audio = np.full(SR * 20, 0.1, np.float32)
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
    ps.feed(1, [], None, 3.0)                                                       # time only: the pause is heard
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
    r.rec.audio[:] = 0.1
    snap = Snap(r.rec.audio)
    out = []
    ps = ph.PhraseStream(snap, lambda g, p: out.append((g, p)))
    r.p._on_words = ps.feed
    ps.begin(1)
    r.begin(1)
    for _ in range(6):
        r.tick(0.7)
    assert out and out[0][1].text == "we met today." and out[0][0] == 1
    assert stream.norm("x") == "x"
