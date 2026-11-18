"""The live-preview engine (voxflow.stream) with a fake clock, a scripted recorder and a scripted transcriber, and its
parts in audio.py / asr.py / fakes.py. No model, no microphone, no sleeping longer than a few milliseconds."""
from __future__ import annotations

import logging
import threading
import time
import types

import numpy as np
import pytest

from voxflow import asr, stream
from voxflow.asr import MIN_SAMPLES, PreviewUnavailable, Transcriber
from voxflow.fakes import FakeRecorder, FakeTranscriber
from voxflow.stream import LocalAgreement, Previewer, norm, words_with_times

SR = 16000


# ======================================================================== doubles
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Rec:
    """A recorder whose recording the test grows: ``audio`` is the whole utterance, ``n`` how much exists."""

    def __init__(self, audio=None):
        self.audio = np.zeros(0, np.float32) if audio is None else audio
        self.n = 0
        self.snaps: list[int] = []

    @property
    def recorded_samples(self):
        return self.n

    def snapshot(self, start=0):
        self.snaps.append(start)
        return self.audio[start:self.n].copy()

    def grow(self, seconds):
        self.n = min(self.audio.size, self.n + int(seconds * SR))


class Asr:
    """Returns the scripted hypotheses one per call (the last one repeats); costs ``cost`` seconds of fake time."""

    backend = "cpu"

    def __init__(self, clock, hyps=(), cost=0.02):
        self.clock, self.hyps, self.cost = clock, list(hyps), cost
        self.calls: list[dict] = []
        self.hook = None
        self.result = None                   # when set: returned instead of the script (None-able via ``busy``)
        self.busy = False

    def transcribe_timed(self, audio, *, bucket_s=0.0, blocking=False):
        self.calls.append({"size": audio.size, "bucket_s": bucket_s, "blocking": blocking, "audio": audio})
        if self.hook:
            self.hook()
        if self.busy:
            return None
        self.clock.t += self.cost
        i = min(len(self.calls), len(self.hyps)) - 1
        return list(self.hyps[i]) if self.hyps else []


def words(text, start=0.0, step=0.3):
    return [(w, start + i * step) for i, w in enumerate(text.split())]


def loud(seconds):
    return np.full(int(seconds * SR), 0.1, np.float32)


class Rig:
    def __init__(self, hyps=(), seconds=30.0, cost=0.02, **kw):
        self.clock = Clock()
        self.rec = Rec(loud(seconds))
        self.asr = Asr(self.clock, hyps, cost)
        self.out: list[tuple] = []
        self.p = Previewer(self.rec, self.asr, lambda *a: self.out.append(a), clock=self.clock, **kw)

    def begin(self, gen=1):
        with self.p._cv:
            assert self.p._begin(gen)

    def tick(self, grow=0.5):
        self.rec.grow(grow)
        self.clock.t += 0.5
        self.p._tick()


# ======================================================================== pure helpers
def test_norm_and_words_with_times():
    assert norm("Dog.") == norm("dog") == "dog" and norm("don't") == "don't"
    assert words_with_times([" The", " qu", "ick", " bro", "wn", "."], [0.0, 0.24, 0.4, 0.56, 0.72, 0.8]) == [
        ("The", 0.0), ("quick", 0.24), ("brown.", 0.56)]
    assert words_with_times([], []) == []


def test_local_agreement_commits_only_what_two_hypotheses_share():
    la = LocalAgreement()
    assert la.update(words("the quick")) == words("the quick") and la.committed == []
    tent = la.update(words("The quick brown fox"))
    assert [w for w, _ in la.committed] == ["The", "quick"] and [w for w, _ in tent] == ["brown", "fox"]


def test_local_agreement_new_part_drops_the_boundary_word_and_falls_back_to_the_time_cut():
    la = LocalAgreement()
    assert la.new_part(words("a b")) == words("a b")                  # nothing committed: everything is new
    la.committed = words("send the report")
    hyp = [("send", 0.0), ("the", 0.31), ("report", 0.58), ("to", 0.9), ("the", 1.1)]
    assert [w for w, _ in la.new_part(hyp)] == ["to", "the"]
    la.committed = [("alpha", 1.0)]
    assert [w for w, _ in la.new_part([("beta", 0.9), ("gamma", 1.2)])] == ["gamma"]


# ======================================================================== the tick
def test_partials_are_committed_by_two_agreeing_hypotheses_and_only_emitted_when_they_change():
    r = Rig([words("hello wor"), words("hello world how"), words("hello world how are you"),
             words("hello world how are you")])
    r.begin(7)
    r.tick()
    assert r.out == [(7, "", "hello wor")]
    r.tick()
    assert r.out[-1] == (7, "hello", "world how")
    r.tick()
    assert r.out[-1] == (7, "hello world how", "are you")
    n = len(r.out)
    r.tick()                                                          # same words again: nothing to say
    assert len(r.out) == n + 1 and r.out[-1] == (7, "hello world how are you", "")
    r.tick()
    assert len(r.out) == n + 1
    assert all(c["blocking"] is False for c in r.asr.calls)


def test_a_tick_waits_for_a_quarter_second_of_new_audio_and_the_preroll_is_not_new():
    r = Rig([words("hi")])
    r.rec.n = int(0.3 * SR)                                           # the pre-roll exists when the session starts
    r.begin()
    r.tick(grow=0.1)
    assert r.asr.calls == []
    r.tick(grow=0.2)
    assert len(r.asr.calls) == 1
    assert r.asr.calls[0]["size"] == int(0.6 * SR)                    # the window includes the pre-roll


def test_silence_is_looked_at_once_then_skipped_until_there_is_sound_again():
    r = Rig([[], words("back")])
    r.rec.audio = np.concatenate([np.zeros(SR * 3, np.float32), loud(5)])
    r.begin()
    r.tick(1.0)                                                       # silence, first time: one call
    r.tick(1.0)                                                       # silence again: skipped
    r.tick(1.0)
    assert len(r.asr.calls) == 1
    r.tick(1.5)                                                       # sound: called again
    assert len(r.asr.calls) == 2
    assert r.rec.snaps[-1] == 0


def test_a_busy_model_skips_the_tick_and_the_audio_is_still_new_next_time():
    r = Rig([words("one two")])
    r.begin()
    r.asr.busy = True
    r.tick()
    assert r.out == [] and r.p._last_n == 0
    r.asr.busy = False
    r.tick(0.0)
    assert len(r.asr.calls) == 2 and r.out == [(1, "", "one two")]


def test_stop_during_a_call_drops_its_result():
    r = Rig([words("never shown")])
    r.begin()
    r.asr.hook = r.p.stop
    r.tick()
    assert r.out == [] and r.p._last_n == 0
    r.p._tick()                                                       # not active: nothing happens, no new call
    assert len(r.asr.calls) == 1


def test_a_new_session_during_a_call_drops_the_old_result_and_keeps_the_new_state():
    r = Rig([words("old words")])
    r.begin(1)

    def replace():
        r.asr.hook = None
        r.begin(2)

    r.asr.hook = replace
    r.tick()
    assert r.out == [] and r.p._gen == 2 and r.p._la.committed == [] and r.p._last_n == r.rec.n


def test_gain_is_applied_to_the_window_before_it_is_transcribed():
    r = Rig([words("x")], gain=lambda a: a * 2)
    r.begin()
    r.tick()
    assert float(r.asr.calls[0]["audio"][0]) == pytest.approx(0.2)


def test_emit_survives_a_receiver_that_raises_and_never_logs_the_words(caplog):
    r = Rig([words("secret words")])
    r.p._on_partial = lambda *a: 1 / 0
    r.begin()
    with caplog.at_level(logging.DEBUG, logger="voxflow.stream"):
        r.tick()
    assert "receiver failed (ZeroDivisionError)" in caplog.text and "secret" not in caplog.text


# ======================================================================== window trimming
def test_the_window_stays_short_by_starting_at_the_first_uncommitted_word():
    hyps = [words("aa bb"), words("aa bb cc dd"), words("aa bb cc dd ee ff")]
    r = Rig(hyps, trim_s=1.0, bucket_s=0)
    r.begin()
    r.tick(0.5)
    assert r.p._win_start == 0                                        # window 0.5 s: not longer than trim_s
    r.tick(0.5)                                                       # 1.0 s: still not longer
    assert r.p._win_start == 0
    r.tick(0.5)                                                       # 1.5 s > 1: "aa".."dd" agree; start at "ee" (1.2 s) - 0.2 s
    assert r.p._win_start == int(1.0 * SR)
    r.tick(0.5)
    assert r.rec.snaps[-1] == int(1.0 * SR)


def test_the_window_starts_after_the_last_committed_word_when_nothing_is_tentative():
    r = Rig([words("aa bb"), words("aa bb"), words("aa bb")], trim_s=1.0)
    r.begin()
    for _ in range(3):
        r.tick(0.6)
    assert r.p._la.prev == [] and r.p._win_start == int((0.3 + 0.4) * SR)


def test_the_window_is_left_alone_when_there_is_nothing_to_anchor_it():
    r = Rig([[]], trim_s=1.0)
    r.begin()
    r.tick(2.0)
    assert r.p._win_start == 0


def test_defaults_depend_on_the_backend_and_arguments_override_them():
    r = Rig([words("a")])
    r.begin()
    assert (r.p._trim_s, r.p._bucket()) == (8.0, 0.0)
    r.tick()
    assert r.asr.calls[0]["bucket_s"] == 0.0
    g = Rig([words("a")])
    g.asr.backend = "cuda"
    g.begin()
    assert (g.p._trim_s, g.p._bucket()) == (15.0, 1.0)
    o = Rig([words("a")], trim_s=5, bucket_s=2.0)
    o.asr.backend = "cuda"
    o.begin()
    assert (o.p._trim_s, o.p._bucket()) == (5.0, 2.0)


# ======================================================================== governor
def test_the_next_call_is_at_least_three_costs_away():
    r = Rig([words("a")], cost=0.05, interval_s=0.4)
    r.begin()
    t0 = r.clock.t + 0.5
    r.tick()
    assert r.p._next_at == pytest.approx(t0 + 0.4)                    # 3 x 0.05 < interval
    r2 = Rig([words("a")], cost=0.2, interval_s=0.4)
    r2.begin()
    t1 = r2.clock.t + 0.5
    r2.tick()
    assert r2.p._next_at == pytest.approx(t1 + 0.6)                   # 3 x 0.2


def test_a_slow_call_shrinks_the_window_down_to_a_floor():
    r = Rig([words("a")], cost=0.3, trim_s=8)
    r.begin()
    seen = []
    for _ in range(5):
        r.tick()
        seen.append(round(r.p._trim_s, 2))
    assert seen == [6.0, 4.5, 4.0, 4.0, 4.0] and r.p.available       # 0.3 s is slow, but not hopeless


def test_two_hopeless_calls_in_a_row_at_the_floor_switch_the_preview_off_for_good(caplog):
    r = Rig([words("a")], cost=0.5, trim_s=4)
    r.begin()
    with caplog.at_level(logging.INFO, logger="voxflow.stream"):
        r.tick()
        assert r.p.available and r.p._slow == 1
        r.tick()
    assert not r.p.available and not r.p._active
    assert caplog.text.count("live preview is off") == 1 and "too slow" in caplog.text
    r.p.start(2)                                                       # a later utterance: nothing starts
    assert not r.p._active and r.p._thread is None
    r.p._disable("again")                                              # already off: silent
    assert caplog.text.count("live preview is off") == 1


def test_a_fast_call_resets_the_count_and_a_slow_one_above_the_floor_does_not_count():
    r = Rig([words("a")], cost=0.5, trim_s=4)
    r.begin()
    r.tick()
    assert r.p._slow == 1
    r.asr.cost = 0.05
    r.tick()
    assert r.p._slow == 0 and r.p.available
    r2 = Rig([words("a")], cost=0.5, trim_s=8)
    r2.begin()
    r2.tick()
    assert r2.p._slow == 0 and r2.p._trim_s == 6.0


def test_preview_unavailable_switches_the_preview_off_and_says_why_without_the_audio(caplog):
    r = Rig()

    def boom(*a, **k):
        raise PreviewUnavailable("RuntimeError")

    r.asr.transcribe_timed = boom
    r.begin()
    with caplog.at_level(logging.INFO, logger="voxflow.stream"):
        r.tick()
    assert not r.p.available and "RuntimeError" in caplog.text
    r2 = Rig()
    r2.asr.transcribe_timed = lambda *a, **k: (_ for _ in ()).throw(PreviewUnavailable())
    r2.begin()
    r2.tick()
    assert not r2.p.available


# ======================================================================== the thread
def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.002)


def test_the_thread_previews_while_the_session_is_open_and_goes_quiet_after_stop():
    rec, asr_ = FakeRecorder(), FakeTranscriber(text="one two three four five six seven eight", delay_s=0)
    out = []
    p = Previewer(rec, asr_, lambda *a: out.append(a), interval_s=0.02)
    rec.start()
    p.start(5)
    p.start(5)                                                         # a second start reuses the thread
    wait_for(lambda: out and out[-1][1])
    assert all(g == 5 for g, _, _ in out) and out[-1][1].split()[0] == "one"
    p.stop()
    rec.stop()
    time.sleep(0.1)
    n = len(out)
    time.sleep(0.1)
    assert len(out) == n
    assert [t.name for t in threading.enumerate()].count("voxflow-preview") == 1
    p.close()
    p.start(6)                                                         # closed: no more sessions
    assert not p._active
    assert "voxflow-preview" not in [t.name for t in threading.enumerate()]


def test_close_works_before_any_session_and_twice():
    p = Previewer(Rec(), Asr(Clock()), lambda *a: None)
    p.close()
    p.close()


def test_an_unexpected_error_in_the_thread_switches_the_preview_off(caplog):
    class Broken(Rec):
        @property
        def recorded_samples(self):
            if self.n:
                raise OSError("device gone")
            self.n = 1
            return 0

    rec = Broken()
    p = Previewer(rec, Asr(Clock()), lambda *a: None, interval_s=0.01)
    with caplog.at_level(logging.INFO, logger="voxflow.stream"):
        p.start(1)
        wait_for(lambda: not p.available)
    assert "preview failed (OSError)" in caplog.text
    p.close()


# ======================================================================== Recorder.snapshot / recorded_samples
sd = pytest.importorskip("sounddevice")
from voxflow import audio as audiomod  # noqa: E402


class _Stream:
    active = True

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(sd, "InputStream", lambda **kw: _Stream())
    return audiomod.Recorder()


def feed(rec, x):
    rec._callback(np.asarray(x, np.float32).reshape(-1, 1), len(x), None, None)


def test_snapshot_is_empty_and_the_count_zero_when_not_recording(recorder):
    feed(recorder, np.full(2000, 0.1))
    assert recorder.recorded_samples == 0 and recorder.snapshot().size == 0
    recorder.start()
    recorder.stop()
    assert recorder.recorded_samples == 0 and recorder.snapshot(5).size == 0


def test_snapshot_grows_with_the_recording_and_starts_where_it_is_told_to(recorder):
    feed(recorder, np.full(8000, 0.5))
    recorder.start()
    pre = int(audiomod.PREROLL_S * SR)
    assert recorder.recorded_samples == pre
    a, b = np.arange(512, dtype=np.float32), np.arange(512, 1024, dtype=np.float32)
    feed(recorder, a)
    feed(recorder, b)
    assert recorder.recorded_samples == pre + 1024
    full = recorder.snapshot()
    assert full.dtype == np.float32 and full.size == pre + 1024
    assert np.allclose(full[:pre], 0.5) and np.array_equal(full[pre:], np.arange(1024))
    assert np.array_equal(recorder.snapshot(pre + 700), np.arange(700, 1024))      # inside the second chunk
    assert np.array_equal(recorder.snapshot(pre + 512), b)                         # on a chunk boundary
    assert recorder.snapshot(pre + 1024).size == 0 and recorder.snapshot(10 ** 6).size == 0
    out = recorder.stop()
    assert np.array_equal(out, np.clip(full, -1, 1))                                               # the same samples stop() returns


# ======================================================================== Transcriber.transcribe_timed
class TimedModel:
    def __init__(self, tokens=(" Hello", " wor", "ld"), stamps=(0.15, 0.55, 0.7), fail=None):
        self.tokens, self.stamps, self.fail, self.calls = list(tokens), list(stamps), fail, []

    def with_timestamps(self):
        return self

    def recognize(self, audio, sample_rate=None, **kw):
        self.calls.append((audio.size, sample_rate))
        if self.fail:
            raise self.fail
        return types.SimpleNamespace(tokens=self.tokens, timestamps=self.stamps)


def speech(seconds):
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def loaded(model):
    tr = Transcriber(device="cpu")
    tr._model, tr._backend = model, "cpu"
    return tr


def test_transcribe_timed_returns_words_with_times_from_the_start_of_the_audio():
    m = TimedModel()
    tr = loaded(m)
    assert tr.transcribe_timed(speech(1.0)) == [("Hello", pytest.approx(0.0)), ("world", pytest.approx(0.4))]
    assert m.calls == [(int(1.0 * SR) + 2 * int(0.15 * SR), SR)]       # padded like transcribe, no bucket
    m.tokens = m.stamps = None                                         # a result without tokens: no words
    assert tr.transcribe_timed(speech(1.0)) == []


def test_transcribe_timed_guards_and_buckets():
    m = TimedModel()
    tr = loaded(m)
    assert tr.transcribe_timed(np.zeros(MIN_SAMPLES - 1, np.float32)) == []
    assert tr.transcribe_timed(np.zeros(SR, np.float32)) == []
    assert m.calls == []
    tr.transcribe_timed(speech(1.3), bucket_s=1.0)
    assert m.calls[-1][0] % SR == 0 and m.calls[-1][0] >= int(1.3 * SR) + 2 * int(0.15 * SR)
    tr.transcribe_timed(speech(1.7), bucket_s=1.0)                     # the same bucket again: the same length
    assert m.calls[-1][0] == m.calls[-2][0]
    with pytest.raises(PreviewUnavailable):
        Transcriber(device="cpu").transcribe_timed(speech(1.0))


def test_transcribe_timed_never_waits_unless_asked_and_releases_the_lock():
    tr = loaded(TimedModel())
    with tr._lock:
        assert tr.transcribe_timed(speech(1.0)) is None                # a final pass is running
    assert tr.transcribe_timed(speech(1.0), blocking=True)
    assert not tr._lock.locked()


def test_a_failing_preview_is_reported_without_moving_the_model_to_the_cpu(monkeypatch):
    tr = loaded(TimedModel(fail=RuntimeError("CUDA error")))
    tr._backend = "cuda"
    monkeypatch.setattr(tr, "_build", lambda b: pytest.fail("the model must not be rebuilt for a preview"))
    with pytest.raises(PreviewUnavailable, match="RuntimeError"):
        tr.transcribe_timed(speech(1.0))
    assert tr.backend == "cuda" and not tr._lock.locked()


# ======================================================================== the stand-ins
def test_fake_recorder_snapshot_grows_with_the_clock():
    rec = FakeRecorder(max_seconds=1.0)
    assert rec.recorded_samples == 0 and rec.snapshot().size == 0
    rec.start()
    time.sleep(0.05)
    n = rec.recorded_samples
    assert n > 0 and rec.snapshot().size >= n and rec.snapshot(100).size >= n - 100
    rec._t0 -= 5.0                                                     # past the cap
    assert rec.recorded_samples == SR
    rec.stop()


def test_fake_transcriber_timed_shows_three_words_a_second_and_honours_blocking():
    tr = FakeTranscriber(text="a b c d e f g", delay_s=0)
    assert tr.transcribe_timed(np.zeros(SR, np.float32)) == [("a", 0.0), ("b", pytest.approx(1 / 3)),
                                                              ("c", pytest.approx(2 / 3))]
    assert len(tr.transcribe_timed(np.zeros(10 * SR, np.float32), blocking=True)) == 7
    with tr._lock:
        assert tr.transcribe_timed(np.zeros(SR, np.float32)) is None
    assert not tr._lock.locked()


def test_module_constants_describe_the_governor():
    assert stream.COST_BUDGET_S < stream.GIVE_UP_S and stream.TRIM_FLOOR_S == 4.0 and asr.PAD_S == 0.15


# ======================================================================== a caption is cleared when the preview gives up
def test_giving_up_in_the_middle_clears_the_caption_once():
    r = Rig([words("a b"), words("a b")], cost=0.5, trim_s=4)
    r.begin(3)
    r.tick()
    r.tick()
    assert not r.p.available
    assert r.out == [(3, "", "a b"), (3, "a b", ""), (3, "", "")]
    assert r.out.count((3, "", "")) == 1
    n = len(r.out)
    r.p._disable("again")
    assert len(r.out) == n


def test_nothing_is_cleared_when_no_text_was_ever_shown_or_the_session_was_already_over():
    r = Rig([[]], cost=0.5, trim_s=4)
    r.begin()
    r.tick()
    r.tick()
    assert not r.p.available and r.out == []
    q = Rig([words("a b")])
    q.begin()
    q.tick()
    q.p.stop()                                                         # released: the controller clears the caption
    n = len(q.out)
    q.p._disable("late")
    assert len(q.out) == n


def test_an_error_clears_the_caption_too_and_a_failing_receiver_does_not_matter(caplog):
    r = Rig([words("hello there")])
    r.begin(4)
    r.tick()
    r.asr.transcribe_timed = lambda *a, **k: (_ for _ in ()).throw(PreviewUnavailable("boom"))
    r.tick()
    assert r.out[-1] == (4, "", "") and not r.p.available
    q = Rig([words("hello")])
    q.begin()
    q.tick()
    q.p._on_partial = lambda *a: 1 / 0
    with caplog.at_level(logging.WARNING, logger="voxflow.stream"):
        q.p._disable("x")
    assert "receiver failed" in caplog.text


def test_an_unexpected_error_in_the_thread_clears_the_caption():
    class Dies(Rec):
        reads = 0

        @property
        def recorded_samples(self):
            Dies.reads += 1
            if Dies.reads > 2:
                raise OSError("device gone")
            return self.n

    rec = Dies(loud(5))
    rec.n = 8000
    out = []
    asr_ = Asr(Clock(), [words("x y")])
    p = Previewer(rec, asr_, lambda *a: out.append(a), interval_s=0.01)
    rec.n = 0
    p.start(9)
    rec.n = 8000                                                       # one tick of new audio, then the device dies
    wait_for(lambda: not p.available)
    assert out[0] == (9, "", "x y") and out[-1] == (9, "", "")
    p.close()
