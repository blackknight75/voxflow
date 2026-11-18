"""Phrase streaming through the real Controller (design: plan/PHRASE_STREAMING_DESIGN.md sections 5, 8, 9, 11).

The engine (``voxflow.phrases``) is replaced by a scripted ``FakePhraseStream`` that hands over ready-made phrases, and
the previewer by a ``FakePreviewer``; what is checked here is the controller's own part: when a session exists, what is
sent and in which order, what the release does with the rest, history / stats / undo, cancel, and privacy. Most tests use a
plain stand-in for ``phrases.finish_chunk`` (joining with a space) so they do not depend on the text rules, which have their
own tests; the tests marked ``real_text`` use the real ones.

The two "identical to today" tests compare a streamed-setting run against a run of the unchanged path."""
from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from tests.controller_rig import ESC, TRecorder, make_rig, speech
from voxflow import audio as audiomod
from voxflow import controller as ctl_mod
from voxflow import phrases
from voxflow.context import AppContext

SECRET = "zyzzyva quokka"                  # a phrase that must never reach a log


class FakePreviewer:
    instances: list = []

    def __init__(self, recorder, transcriber, on_partial, *, gain=None, on_words=None, **kw):
        self.recorder, self.on_partial, self.on_words = recorder, on_partial, on_words
        self.available = True
        self.log: list = []
        FakePreviewer.instances.append(self)

    def start(self, gen):
        self.log.append(("start", gen))

    def stop(self):
        self.log.append(("stop",))

    def close(self):
        self.log.append(("close",))

    def emit(self, gen, committed, tentative=""):
        self.on_partial(gen, committed, tentative)

    def starts(self):
        return [e[1] for e in self.log if e[0] == "start"]


class FakePhraseStream:
    instances: list = []

    def __init__(self, recorder, on_phrase, **kw):
        self.recorder, self.on_phrase = recorder, on_phrase
        self.log: list = []
        self.heard_value: tuple = ((), 0.0)             # what ``heard()`` says: (words with start times, covered_s)
        FakePhraseStream.instances.append(self)

    def heard(self):
        return self.heard_value

    def begin(self, gen):
        self.log.append(("begin", gen))

    def end(self):
        self.log.append(("end",))

    def feed(self, gen, new_committed, tentative, now_s):
        self.log.append(("feed", gen))

    def emit(self, gen, phrase):
        self.on_phrase(gen, phrase)


def ph(text, kind="sentence", cut=1.0, ends=True, seq=1):
    words = tuple((w, 0.3 * i) for i, w in enumerate(text.split()))
    return phrases.Phrase(seq, words, text, kind, ends, cut - 0.1, cut + 0.1, cut)


def join_chunk(prev, cleaned, raw, kind, ends):
    return (" " + cleaned if prev else cleaned) if cleaned else ""


@pytest.fixture
def plain_text(monkeypatch):
    """The text rules replaced by a stand-in: the wiring is under test, not ``finish_chunk``."""
    calls = []

    def finish(prev, cleaned, raw, kind, ends, hint=None):
        calls.append((prev, cleaned, raw, kind, ends, hint))
        return join_chunk(prev, cleaned, raw, kind, ends)

    monkeypatch.setattr(ctl_mod.phrases, "finish_chunk", finish)
    return calls


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch, plain_text):
    FakePreviewer.instances, FakePhraseStream.instances = [], []
    monkeypatch.setattr(ctl_mod.stream, "Previewer", FakePreviewer)
    monkeypatch.setattr(ctl_mod.phrases, "PhraseStream", FakePhraseStream)
    r = make_rig(tmp_path, qtbot, monkeypatch)
    r.sleeps = []
    r.c._sleep = r.sleeps.append                     # pacing is recorded, not waited for
    r.c.set("phrase_streaming", True)
    yield r
    r.close()


_REAL_TEXT_RULES = (phrases.finish_chunk,)


@pytest.fixture
def real_text(monkeypatch, rig):
    """The real ``phrases.finish_chunk`` for the tests that name it."""
    monkeypatch.setattr(ctl_mod.phrases, "finish_chunk", _REAL_TEXT_RULES[0])


def pstream(rig) -> FakePhraseStream:
    return rig.c._phrase_stream


def previewer(rig) -> FakePreviewer:
    return rig.c._preview


def locked(rig, seconds=8.0, text="um the tail words"):
    TRecorder.audio = speech(seconds)
    rig.asr.text = text
    rig.lock()
    return rig.c._gen


def send(rig, gen, *items):
    """Emit phrases (as the preview thread would) and wait until each one has been typed."""
    for item in items:
        n = len(rig.injector.pasted)
        pstream(rig).emit(gen, item)
        rig.wait(lambda n=n: len(rig.injector.pasted) > n)


def finished(rig, entries=1):
    rig.wait(lambda: rig.c.state in ("done", "error", "idle") and rig.c.history.count() >= entries)


def snapshot(rig):
    """Everything that makes a dictation observable: what was pasted, what the clean-up saw, what the model heard."""
    last = rig.last_entry()
    return (list(rig.injector.pasted), [c[0] for c in rig.cleaner.clean_calls],
            [a.tobytes() for a in rig.asr.heard], last["text"] if last else None, last["raw"] if last else None)


# =========================================================================== sending, in order
def test_phrases_are_pasted_in_order_one_at_a_time(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0), ph("then we ship it", cut=2.0))
    assert rig.injector.pasted == ["So the plan is simple.", " Then we ship it."]
    assert rig.injector.events == [("begin", "So the plan is simple."), ("finish_later",),
                                   ("begin", " Then we ship it."), ("finish_later",)]
    assert rig.c.state == "locked"
    stream = rig.c._stream
    assert stream.sent == rig.injector.pasted and stream.sent_cut_s == 2.0


def test_words_before_the_lock_are_sent_after_it(rig):
    rig.pause()
    rig.press()
    gen = rig.c._gen
    pstream(rig).emit(gen, ph("held until the lock"))
    rig.settle(50)
    assert rig.injector.pasted == [] and rig.c._stream.phrases_seen == 1 and len(rig.c._stream.queue) == 1
    rig.c._on_hotkey("dictate", "lock")
    rig.wait(lambda: rig.injector.pasted)
    assert rig.injector.pasted == ["Held until the lock."] and rig.c._stream.queue == []
    rig.c.cancel()


def test_a_hold_session_never_sends_and_runs_todays_pipeline(rig):
    rig.asr.text = "um hello world"
    rig.c.set("phrase_streaming", False)
    rig.dictate()
    finished(rig)
    rig.wait(lambda: rig.c.state == "idle")
    unchanged = snapshot(rig)
    rig.c.set("phrase_streaming", True)
    rig.injector.pasted.clear()
    rig.cleaner.clean_calls.clear()
    rig.asr.heard.clear()
    rig.pause()
    rig.press()
    pstream(rig).emit(rig.c._gen, ph("hello world here"))
    pstream(rig).emit(rig.c._gen, ph("and some more words", kind="pause"))
    rig.release()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.c._stream is None
    assert rig.injector.pasted == unchanged[0] and [c[0] for c in rig.cleaner.clean_calls] == unchanged[1]
    assert [a.tobytes() for a in rig.asr.heard] == unchanged[2]


# =========================================================================== identical to today
def test_setting_off_is_identical_to_today(rig):
    rig.c.set("phrase_streaming", False)
    rig.c.set("live_preview", False)
    rig.asr.text = "um hello world"
    rig.lock()
    assert rig.c._stream is None and previewer(rig).starts() == [], "no session, and the previewer is not started"
    pstream(rig).emit(rig.c._gen, ph("hello world again"))
    rig.settle(50)
    assert rig.injector.pasted == [] and not [e for e in pstream(rig).log if e[0] in ("begin", "feed")]
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.c.set("phrase_streaming", True)
    rig.c.set("live_preview", True)
    assert snapshot(rig)[0] == ["Hello world."] and snapshot(rig)[1] == ["um hello world"]
    assert snapshot(rig)[2] == [np.asarray(TRecorder.audio).tobytes()]


def test_zero_phrases_is_identical_to_today(rig):
    outcomes = []
    rig.c.snippets.add("my address", "221B Baker Street")
    for on in (False, True):
        rig.c.set("phrase_streaming", on)
        rig.asr.text = "um insert my address please"
        rig.injector.pasted.clear()
        rig.cleaner.clean_calls.clear()
        rig.asr.heard.clear()
        rig.lock()
        assert (rig.c._stream is not None) is on
        rig.stop_locked()
        rig.wait(lambda: rig.c.state == "done")
        rig.wait(lambda: rig.c.history.count() == len(outcomes) + 1)
        rig.wait(lambda: rig.c.state == "idle")
        outcomes.append(snapshot(rig))
        rig.pause()
    assert outcomes[0] == outcomes[1] and outcomes[0][0], "same pasted text, clean-up calls, audio and history entry"


def test_command_mode_never_streams(rig):
    rig.cleaner.reply = lambda messages: "REWRITTEN"
    rig.pause()
    rig.press("command")
    assert rig.c._stream is None and pstream(rig).log == [] and previewer(rig).starts() == [rig.c._gen]
    pstream(rig).emit(rig.c._gen, ph("some words said aloud"))
    rig.release("command")
    rig.wait(lambda: rig.c.state in ("done", "error"))
    assert rig.injector.pasted == ["REWRITTEN"]


# =========================================================================== focus: VoxFlow's own window
def test_own_window_holds_the_phrase_and_merges_it_into_the_next(rig):
    gen = locked(rig)
    rig.own_focus = True
    pstream(rig).emit(gen, ph("first held phrase", cut=1.0))
    rig.wait(lambda: len(rig.c._stream.held) == 1)
    assert rig.injector.pasted == []
    rig.own_focus = False
    send(rig, gen, ph("second phrase here", cut=2.0))
    assert rig.injector.pasted == ["First held phrase. Second phrase here."], "one paste for the whole burst"
    stream = rig.c._stream
    assert stream.held == [] and stream.sent_cut_s == 2.0 and len(stream.raws) == 2


def test_own_window_at_release_copies_the_tail(rig):
    gen = locked(rig)
    send(rig, gen, ph("something typed before", cut=1.0))
    rig.own_focus = True
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["Something typed before."]
    assert rig.injector.copied == [" The tail words."], "never typed into VoxFlow's own window"
    assert rig.c.last_transcript["text"] == "Something typed before. The tail words."
    assert rig.c.state == "done" and rig.c.message == "Copied. Press Ctrl+V"


def test_own_window_at_release_with_a_failed_copy_is_a_paste_failure(rig):
    gen = locked(rig)
    send(rig, gen, ph("something typed before", cut=1.0))
    rig.own_focus, rig.injector.copy_ok = True, False
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "error")
    assert rig.c.history.count() == 1 and "Couldn't paste" in rig.c.message


# =========================================================================== failures while streaming
def test_a_failed_paste_stops_streaming_and_the_tail_covers_the_rest(rig):
    gen = locked(rig)
    rig.injector.outcomes = ["pasted", "copied"]                # the second chunk only reaches the clipboard
    send(rig, gen, ph("the first phrase", cut=1.0))
    pstream(rig).emit(gen, ph("the second phrase", cut=2.0))
    told = ("warning", "Couldn't type into the app, so the rest will be pasted when you stop.")
    rig.wait(lambda: rig.c._stream.stopped and told in rig.spy.notices())
    pstream(rig).emit(gen, ph("the third phrase", cut=3.0))
    rig.settle(50)
    assert rig.injector.pasted == ["The first phrase."] and rig.injector.copied == [" The second phrase."]
    assert rig.c._stream.sent_cut_s == 1.0, "only a chunk that reached the field moves the cut"
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard[-1]) == len(TRecorder.audio) - 16000
    assert rig.injector.pasted == ["The first phrase.", " The tail words."]


def test_a_failure_with_nothing_sent_before_it_runs_todays_pipeline(rig):
    gen = locked(rig)
    rig.injector.outcomes = ["failed"]
    pstream(rig).emit(gen, ph("the first phrase", cut=1.0))
    rig.wait(lambda: rig.c._stream.stopped)
    rig.injector.outcome = "pasted"
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard[-1]) == len(TRecorder.audio) and rig.injector.pasted == ["The tail words."]


def test_a_crash_inside_a_phrase_job_stops_streaming_without_failing_the_bar(rig, monkeypatch):
    gen = locked(rig)
    monkeypatch.setattr(rig.c, "_clean_phrase", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    pstream(rig).emit(gen, ph("the first phrase", cut=1.0))
    rig.wait(lambda: rig.c._stream.stopped and rig.spy.notices())
    assert rig.c.state == "locked"
    assert rig.spy.notices()[-1] == ("error", "Typing while you speak stopped. The rest will be pasted when you stop.")
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["The tail words."], "nothing was sent: today's pipeline"


# =========================================================================== the release
def test_release_pastes_only_the_tail_from_the_last_sent_cut(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0), ph("then we ship it", cut=2.5))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple.", " Then we ship it.", " The tail words."]
    assert rig.asr.heard and len(rig.asr.heard) == 1


def test_the_tail_never_repeats_a_phrase(rig):
    gen = locked(rig, 8.0)
    send(rig, gen, ph("so the plan is simple", cut=1.234), ph("then we ship it", cut=3.3333))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    start = round(3.3333 * 16000)
    assert np.array_equal(rig.asr.heard[-1], TRecorder.audio[start:]), "exactly the audio after the last sent cut"


def test_a_phrase_that_was_not_sent_is_in_the_tail_audio(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.own_focus = True
    pstream(rig).emit(gen, ph("held and never sent", cut=2.0))
    rig.wait(lambda: len(rig.c._stream.held) == 1)
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard[-1]) == len(TRecorder.audio) - 16000, "the held phrase's audio is re-read, not re-sent"


def test_history_has_one_entry_with_phrases_and_tail(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0), ph("then we ship it", cut=2.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.wait(lambda: rig.c.history.count() == 1)
    entry = rig.last_entry()
    assert entry["text"] == "So the plan is simple. Then we ship it. The tail words."
    assert entry["raw"] == "so the plan is simple then we ship it um the tail words"
    assert entry["words"] == 13 and entry["app_exe"] == "notepad.exe" and entry["mode"] == "dictate"
    assert rig.c.history.count() == 1 and len(rig.spy.of("transcript_ready")) == 1
    assert rig.c.last_transcript["text"] == entry["text"]


def test_stats_counted_once(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.wait(lambda: rig.c.stats.summary()["sessions"] == 1)
    rig.settle(60)
    assert rig.c.stats.summary()["sessions"] == 1 and rig.c.stats.summary()["total_words"] == 9


def test_empty_tail_is_not_an_error(rig):
    gen = locked(rig, text="")
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple."] and rig.c.message == "Pasted"
    assert rig.c.history.count() == 1 and rig.last_entry()["text"] == "So the plan is simple."


def test_a_silent_tail_is_not_transcribed_and_is_not_an_error(rig):
    gen = locked(rig)
    TRecorder.audio = np.concatenate([speech(2.0), np.zeros(16000 * 3, dtype=np.float32)])
    send(rig, gen, ph("so the plan is simple", cut=2.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.asr.heard == [] and rig.injector.pasted == ["So the plan is simple."]


def test_a_cut_at_the_very_end_leaves_no_audio_for_the_tail(rig):
    gen = locked(rig, 2.0)
    send(rig, gen, ph("so the plan is simple", cut=9.0))      # past the end of the recording
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.asr.heard == [] and rig.injector.pasted == ["So the plan is simple."]


def test_a_filler_only_phrase_sends_nothing_and_moves_no_cut(rig):
    gen = locked(rig)
    real_clean = rig.cleaner.clean
    rig.cleaner.clean = lambda text, **kw: "" if text == "um uh um" else real_clean(text, **kw)   # all of it is filler
    pstream(rig).emit(gen, ph("um uh um", cut=1.0))
    rig.settle(60)
    assert rig.injector.pasted == [] and rig.c._stream.sent_cut_s == 0.0
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard[-1]) == len(TRecorder.audio)


def test_tail_asr_failure_files_what_was_sent(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.asr.fail_transcribe = RuntimeError("model fell over")
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "error")
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.last_entry()["text"] == "So the plan is simple." and rig.c.message.startswith("The speech model")


def test_scratch_that_never_reaches_back_into_sent_text(rig):
    gen = locked(rig, text="scratch that")
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple."] and rig.injector.undone == 0
    assert rig.last_entry()["text"] == "So the plan is simple."


def test_cap_stop_ends_the_session_like_a_release(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.c._t_start -= 7.0
    rig.rec.hit_cap = True
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple.", " The tail words."]
    assert pstream(rig).log[-1] == ("end",) and rig.c._stream is None


def test_whisper_gain_applies_to_the_tail(rig):
    rig.c.set("whisper_mode", True)
    gen = locked(rig, 6.0)
    TRecorder.audio = speech(6.0, amp=0.004)
    send(rig, gen, ph("so the plan is simple", cut=2.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    tail = TRecorder.audio[round(2.0 * 16000):]
    assert np.array_equal(rig.asr.heard[-1], audiomod.normalize_gain(tail))
    assert float(np.max(np.abs(rig.asr.heard[-1]))) > float(np.max(np.abs(tail)))


# =========================================================================== stale phrases, sessions
def test_a_stale_gen_phrase_is_dropped(rig):
    gen = locked(rig)
    pstream(rig).emit(gen + 7, ph("from another recording"))
    rig.settle(60)
    assert rig.injector.pasted == [] and rig.c._stream.phrases_seen == 0


def test_a_phrase_after_the_release_is_dropped_and_its_words_stay_in_the_tail(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.asr.delay_s = 0.15
    rig.stop_locked()
    assert rig.c.state == "processing"
    pstream(rig).emit(gen, ph("arrived too late", cut=2.0))
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple.", " The tail words."]
    assert len(rig.asr.heard[-1]) == len(TRecorder.audio) - 16000


def test_a_phrase_with_no_session_is_ignored(rig):
    rig.c.set("phrase_streaming", False)
    rig.lock()
    rig.c.set("phrase_streaming", True)
    pstream(rig).emit(rig.c._gen, ph("nobody asked for this"))
    rig.settle(50)
    assert rig.injector.pasted == []
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")


def test_previewer_runs_for_streaming_with_live_preview_off(rig):
    rig.c.set("live_preview", False)
    rig.lock()
    gen = rig.c._gen
    assert previewer(rig).starts()[-1] == gen and rig.c._stream is not None
    previewer(rig).emit(gen, "so far", "")
    assert rig.spy.of("partial_text") == [], "the caption is shown only when live_preview is on"
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")


def test_live_preview_off_mid_session_keeps_streaming(rig):
    gen = locked(rig)
    n = len(previewer(rig).log)
    rig.c.set("live_preview", False)
    assert ("stop",) not in previewer(rig).log[n:], "the words are still needed for the phrases"
    assert rig.spy.of("partial_text")[-1] == ("", "")
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted[0] == "So the plan is simple."


@pytest.mark.parametrize("caption", [True, False])
def test_streaming_off_mid_session_stops_sending(rig, caption):
    rig.c.set("live_preview", caption)
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    n = len(previewer(rig).log)
    rig.c.set("phrase_streaming", False)
    assert (("stop",) in previewer(rig).log[n:]) is (not caption)
    pstream(rig).emit(gen, ph("not typed any more", cut=2.0))
    rig.settle(60)
    assert rig.injector.pasted == ["So the plan is simple."]
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard[-1]) == len(TRecorder.audio) - 16000 and rig.injector.pasted[-1] == " The tail words."


def test_streaming_switched_on_mid_recording_applies_from_the_next_one(rig):
    rig.c.set("phrase_streaming", False)
    rig.lock()
    rig.c.set("phrase_streaming", True)
    assert rig.c._stream is None
    rig.c.set("phrase_streaming", False)                           # off again, with no session to stop
    rig.c.set("phrase_streaming", True)
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.pause()
    rig.lock()
    assert rig.c._stream is not None
    rig.c.cancel()


def test_streaming_off_mid_session_without_a_session_changes_nothing(rig):
    rig.c.set("phrase_streaming", False)
    rig.c.set("phrase_streaming", True)
    rig.c.set("phrase_streaming", False)


def test_previewer_gave_up_midsession_tail_covers_the_rest(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    previewer(rig).available = False                              # no more phrases will come
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple.", " The tail words."]
    assert rig.c.status()["live_preview_available"] is False
    rig.dictate("still works")
    assert rig.injector.pasted[-1] == "Still works." and rig.c._stream is None


def test_the_previewer_is_not_started_for_streaming_when_it_is_unavailable(rig):
    rig.c.set("live_preview", False)
    previewer(rig).available = False
    rig.lock()
    assert rig.c._stream is None and previewer(rig).starts() == []
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")


# =========================================================================== cancel, pause, shutdown, swap
def test_cancel_keeps_sent_text_drops_unsent_and_files_one_history_entry(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.own_focus = True
    pstream(rig).emit(gen, ph("held and then dropped", cut=2.0))
    rig.wait(lambda: len(rig.c._stream.held) == 1)
    rig.keys((ESC, True), (ESC, False))
    assert rig.c.state == "idle"
    rig.wait(lambda: rig.c.history.count() == 1)
    rig.wait(lambda: rig.spy.of("transcript_ready"))
    rig.settle(60)
    assert rig.injector.pasted == ["So the plan is simple."] and rig.last_entry()["text"] == "So the plan is simple."
    assert rig.c.state == "idle" and "done" not in rig.spy.states(), "the bar is already idle after Esc"
    assert len(rig.spy.of("transcript_ready")) == 1 and pstream(rig).log[-1] == ("end",)
    assert rig.c.undo_last_paste() is True, "the last typed phrase can still be undone"
    pstream(rig).emit(gen, ph("too late", cut=3.0))
    rig.settle(50)
    assert rig.injector.pasted == ["So the plan is simple."]


def test_cancel_with_nothing_sent_files_nothing(rig):
    gen = locked(rig)
    rig.own_focus = True
    pstream(rig).emit(gen, ph("held and then dropped", cut=2.0))
    rig.wait(lambda: len(rig.c._stream.held) == 1)
    rig.keys((ESC, True), (ESC, False))
    rig.settle(80)
    assert rig.c.history.count() == 0 and rig.spy.of("transcript_ready") == []


def test_cancel_before_any_phrase_queues_nothing(rig):
    locked(rig)
    rig.keys((ESC, True), (ESC, False))
    rig.settle(50)
    assert rig.c.history.count() == 0 and rig.c._stream is None


def test_pause_cancels_the_session_like_escape(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.c.set_paused(True)
    rig.wait(lambda: rig.c.history.count() == 1)
    rig.c.set_paused(False)
    assert rig.c.state == "idle" and rig.injector.pasted == ["So the plan is simple."]


def test_shutdown_mid_session_skips_queued_phrases(rig, monkeypatch):
    gen = locked(rig)
    send(rig, gen, ph("the one already sent", cut=1.0))
    gate = threading.Event()
    real_clean = rig.c._clean_phrase

    def slow(session, raw, ctx):
        gate.wait(5)
        return real_clean(session, raw, ctx)

    monkeypatch.setattr(rig.c, "_clean_phrase", slow)
    pstream(rig).emit(gen, ph("the one already started", cut=1.0))
    rig.wait(lambda: rig.c._work.qsize() == 0)                      # picked up by the worker, held at the gate
    pstream(rig).emit(gen, ph("queued behind it", cut=2.0))
    threading.Timer(0.2, gate.set).start()
    rig.c.shutdown()
    assert rig.injector.pasted == ["The one already sent."], "the one being cleaned is dropped at the re-check, the queued one skipped"
    assert rig.c.history.count() == 1
    rig.c.shutdown()


def test_swapping_the_microphone_under_a_streamed_session_cancels_it(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    old = pstream(rig)
    rig.c._apply("mic_device", "Other Mic")
    assert pstream(rig) is not old and old.log[-1] == ("end",) and rig.c._stream is None
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.last_entry()["text"] == "So the plan is simple."
    old.emit(gen, ph("from the old recorder", cut=2.0))
    rig.settle(50)
    assert rig.injector.pasted == ["So the plan is simple."]


def test_recorder_stop_failure_files_what_was_sent(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    TRecorder.fail_stop = RuntimeError("device lost")
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "error")
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.last_entry()["text"] == "So the plan is simple." and rig.c.state == "error"


def test_too_short_after_phrases_files_what_was_sent(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    TRecorder.speech_seconds = 0.0
    TRecorder.audio = np.zeros(10, dtype=np.float32)
    rig.stop_locked()
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.c.state == "idle" and rig.last_entry()["text"] == "So the plan is simple."


def test_second_dictation_queues_behind_the_first_sessions_jobs(rig):
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.asr.delay_s = 0.2
    rig.stop_locked()
    assert rig.c.state == "processing"
    rig.pause()
    rig.press()
    rig.release()
    rig.wait(lambda: len(rig.injector.pasted) == 3 and rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple.", " The tail words.", "The tail words."], \
        "the first dictation's tail is typed before the second dictation's text"


# =========================================================================== clean-up per phrase
def test_cpu_cleaner_uses_rules_only_per_phrase_gpu_uses_the_model(rig):
    gen = locked(rig)
    send(rig, gen, ph("the gpu phrase here", cut=1.0))
    assert [c[0] for c in rig.cleaner.clean_calls] == ["the gpu phrase here"]
    rig.keys((ESC, True), (ESC, False))
    rig.settle(40)
    rig.cleaner.backend = "cpu"
    rig.cleaner.clean_calls.clear()
    gen = locked(rig)
    send(rig, gen, ph("the cpu phrase here", cut=1.0), ph("and the next one", cut=2.0))
    assert rig.cleaner.clean_calls == [], "a CPU model is not used per phrase"
    assert rig.injector.pasted[-2:] == ["The cpu phrase here.", " And the next one."]
    assert rig.c._rule_cleaner is not None, "one rules-only cleaner, built once"


def test_cleanup_off_pastes_the_raw_phrase(rig):
    rig.c.set("cleanup_enabled", False)
    gen = locked(rig)
    send(rig, gen, ph("the raw phrase here", cut=1.0))
    assert rig.injector.pasted == ["the raw phrase here"] and rig.cleaner.clean_calls == []


def test_a_cleanup_failure_pastes_the_raw_phrase_and_logs_no_text(rig, caplog):
    caplog.set_level(logging.DEBUG)
    rig.cleaner.fail_clean = RuntimeError(f"cannot clean {SECRET}")
    gen = locked(rig)
    send(rig, gen, ph(f"{SECRET} is here", cut=1.0))
    assert rig.injector.pasted == [f"{SECRET} is here"]
    assert SECRET not in caplog.text


@pytest.mark.parametrize("how", ["slow", "rules_only"])
def test_a_slow_model_call_switches_the_rest_of_the_session_to_rules(rig, monkeypatch, how):
    gen = locked(rig)
    real_clean = rig.cleaner.clean
    if how == "slow":
        monkeypatch.setattr(ctl_mod, "PHRASE_LLM_SLOW_S", 0.02)

        def clean(text, **kw):
            threading.Event().wait(0.06)
            return real_clean(text, **kw)
    else:
        def clean(text, **kw):
            rig.cleaner.last_info = {"rules_only": True}
            return real_clean(text, **kw)

    monkeypatch.setattr(rig.cleaner, "clean", clean)
    send(rig, gen, ph("the first phrase", cut=1.0), ph("the second phrase", cut=2.0))
    assert [c[0] for c in rig.cleaner.clean_calls] == ["the first phrase"], "the model is used no more after the slow call"
    assert rig.injector.pasted[1] == " The second phrase." and rig.c._stream.llm is False


def test_a_fast_model_call_keeps_the_model(rig):
    gen = locked(rig)
    rig.cleaner.last_info = {"rules_only": False}
    send(rig, gen, ph("the first phrase", cut=1.0), ph("the second phrase", cut=2.0))
    assert len(rig.cleaner.clean_calls) == 2 and rig.c._stream.llm is True


def test_dictionary_applies_per_chunk_snippets_do_not(rig):
    rig.c.dictionary.add("Kubernetes", ["cube are netties"])
    rig.c.snippets.add("my address", "221B Baker Street")
    rig.asr.text = "my address"
    gen = locked(rig, text="my address")
    send(rig, gen, ph("we deploy on cube are netties", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted[0] == "We deploy on Kubernetes."
    assert "221B" not in "".join(rig.injector.pasted), "no snippet expansion once a chunk was sent"
    assert rig.injector.pasted[-1] == " My address."


def test_learning_from_a_phrase_and_from_the_tail_suggests_dictionary_words(rig):
    rig.cleaner.rewrites["i pushed it to get hub"] = "I pushed it to GitHub."
    rig.cleaner.rewrites["we use pie torch here"] = "We use PyTorch here."
    gen = locked(rig, text="we use pie torch here")
    send(rig, gen, ph("i pushed it to get hub", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.wait(lambda: len(rig.spy.of("dictionary_suggestion")) == 2)
    assert {a[0]["term"] for a in rig.spy.of("dictionary_suggestion")} == {"GitHub", "PyTorch"}
    rig.c.set("auto_learn", False)
    rig.spy.clear()
    rig.pause()
    gen = locked(rig, text="we use pie torch here")
    send(rig, gen, ph("i pushed it to get hub", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.spy.of("dictionary_suggestion") == []


def test_a_tail_the_model_was_too_slow_for_is_marked_like_a_normal_dictation(rig):
    rig.cleaner.last_info = {"rules_only": True}
    gen = locked(rig)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.c.message.endswith(ctl_mod.RULES_ONLY_SUFFIX)
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.last_entry()["rules_only"] is True


# =========================================================================== the text rules (the real ones)
def test_spacing_and_case_follow_the_previous_chunk(real_text, rig):
    gen = locked(rig, text="")
    send(rig, gen, ph("so the plan is simple", kind="pause", cut=1.0, ends=False),
         ph("then we ship it", kind="sentence", cut=2.0))
    assert rig.injector.pasted == ["So the plan is simple", " then we ship it."]


# =========================================================================== pacing, undo, privacy
@pytest.mark.parametrize("delay", [0.5, 1.5])
def test_pasting_is_paced_by_the_whole_restore_delay(rig, delay):
    rig.injector.restore_delay = delay
    rig.c._mono = lambda: 100.0                                    # no time passes between the two pastes
    gen = locked(rig)
    send(rig, gen, ph("the first phrase", cut=1.0), ph("the second phrase", cut=2.0))
    assert rig.sleeps == [delay], "nothing before the first paste, then exactly the delay (no cap)"
    rig.injector.restore = False
    send(rig, gen, ph("the third phrase", cut=3.0))
    assert rig.sleeps == [delay], "without a clipboard restore there is nothing to wait for"


def test_a_slow_tail_then_the_next_sessions_first_phrase_is_paced(rig):
    rig.c._mono = lambda: 100.0
    gen = locked(rig)
    send(rig, gen, ph("the first phrase", cut=1.0))
    rig.asr.delay_s = 0.2                                          # a slow final pass
    rig.stop_locked()
    gen2 = locked(rig)                                             # the next session starts while it runs
    pstream(rig).emit(gen2, ph("next session phrase", cut=1.0))
    rig.wait(lambda: len(rig.injector.pasted) == 3)
    assert rig.injector.pasted == ["The first phrase.", " The tail words.", "Next session phrase."]
    assert rig.sleeps == [0.5, 0.5], "the tail waited for phrase one, the next session's phrase for the tail"


def test_todays_paths_are_not_delayed_but_record_their_paste_time(rig):
    rig.dictate("one")
    rig.wait(lambda: rig.c.state == "idle")
    assert rig.c._last_paste_t is not None and rig.sleeps == []
    rig.dictate("two")
    rig.wait(lambda: rig.c.state == "idle")
    assert rig.sleeps == [], "a normal dictation is never paced"


def test_pacing_waits_nothing_when_enough_time_has_passed(rig):
    gen = locked(rig)
    clock = iter([100.0, 105.0, 105.0])
    rig.c._mono = lambda: next(clock, 105.0)
    send(rig, gen, ph("the first phrase", cut=1.0), ph("the second phrase", cut=2.0))
    assert rig.sleeps == []


def test_undo_after_a_streamed_session_is_one_ctrl_z_in_the_last_chunks_window(rig):
    gen = locked(rig)
    rig.context = AppContext("notepad.exe", "A", 0x100)
    send(rig, gen, ph("the first phrase", cut=1.0))
    rig.context = AppContext("winword.exe", "B", 0x200)             # the user moved on while talking
    send(rig, gen, ph("the second phrase", cut=2.0))
    assert rig.c._last_paste["hwnd"] == 0x200 and rig.c._last_paste["text"] == " The second phrase."
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.c._last_paste["hwnd"] == 0x200, "the tail was typed where the focus was"
    rig.context = AppContext("notepad.exe", "A", 0x100)
    assert rig.c.undo_last_paste() is False and rig.injector.undone == 0, "only in the window that got the last text"
    rig.context = AppContext("winword.exe", "B", 0x200)
    assert rig.c.undo_last_paste() is True
    rig.wait(lambda: rig.injector.undone == 1)
    assert rig.c.undo_last_paste() is False, "one paste, one undo"


def test_hwnd_lookup_failure_is_treated_as_unknown(rig):
    gen = locked(rig)

    def boom():
        raise OSError("no window")

    rig.c._hwnd_fn = boom
    send(rig, gen, ph("the first phrase", cut=1.0))
    assert rig.c._last_paste["hwnd"] == 0


def test_phrase_text_is_never_logged(rig, caplog):
    caplog.set_level(logging.DEBUG)
    gen = locked(rig, text=f"{SECRET} tail")
    rig.injector.outcomes = ["pasted", "failed"]
    send(rig, gen, ph(f"{SECRET} one", cut=1.0))
    pstream(rig).emit(gen, ph(f"{SECRET} two", cut=2.0))
    rig.wait(lambda: rig.c._stream.stopped)
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.asr.fail_transcribe = RuntimeError(f"cannot hear {SECRET}")
    gen = locked(rig, text=f"{SECRET} again")
    rig.cleaner.fail_clean = RuntimeError(f"cannot clean {SECRET}")
    send(rig, gen, ph(f"{SECRET} three", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "error")
    assert SECRET not in caplog.text


# =========================================================================== wiring of the engine
def test_the_previewer_feeds_the_phrase_stream(rig):
    assert previewer(rig).on_words == pstream(rig).feed
    assert pstream(rig).recorder is rig.rec


def test_a_session_begins_and_ends_the_phrase_stream(rig):
    gen = locked(rig)
    assert pstream(rig).log[-1] == ("begin", gen)
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert pstream(rig).log[-2:] == [("begin", gen), ("end",)]


def test_the_signal_carries_a_phrase_from_the_preview_thread_to_the_ui_thread(rig):
    gen = locked(rig)
    seen = []
    rig.c._sig_phrase.connect(lambda g, p: seen.append((g, p.text)))
    done = threading.Event()
    threading.Thread(target=lambda: (pstream(rig).on_phrase(gen, ph("from another thread")), done.set())).start()
    done.wait(2)
    rig.wait(lambda: seen)
    rig.wait(lambda: rig.injector.pasted)
    assert seen == [(gen, "from another thread")] and rig.injector.pasted == ["From another thread."]


def test_the_real_signal_path_from_the_stream_to_the_controller(tmp_path, qtbot, monkeypatch, plain_text):
    """The real ``PhraseStream`` with a splitter that cuts after every committed word group."""
    monkeypatch.setattr(ctl_mod.stream, "Previewer", FakePreviewer)
    r = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        r.c.set("phrase_streaming", True)
        assert isinstance(r.c._phrase_stream, phrases.PhraseStream)
        r.lock()
        assert r.c._stream is not None
        r.c._sig_phrase.emit(r.c._gen, ph("straight to the controller"))
        r.wait(lambda: r.injector.pasted)
        assert r.injector.pasted == ["Straight to the controller."]
        r.c.cancel()
    finally:
        r.close()


# =========================================================================== round 2 (plan/RULINGS_REVIEW.md, section C)
def test_m1_the_tail_gets_the_first_heard_word_after_the_cut_as_a_hint(rig, plain_text):
    gen = locked(rig)
    pstream(rig).heard_value = ((("so", 0.2), ("and", 1.0), ("send", 1.4)), 6.0)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert plain_text[-1][3] == "tail" and plain_text[-1][5] == "and"


def test_m1_a_tail_that_starts_mid_sentence_is_not_capitalised(real_text, rig):
    gen = locked(rig, text="and send the report")
    pstream(rig).heard_value = ((("so", 0.2), ("and", 1.0)), 6.0)
    send(rig, gen, ph("so the plan is simple", kind="pause", cut=1.0, ends=False))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["So the plan is simple", " and send the report."]
    assert rig.last_entry()["text"] == "So the plan is simple and send the report."


def test_m1_the_session_keeps_what_the_stream_had_heard_at_the_release(rig):
    gen = locked(rig)
    pstream(rig).heard_value = ((("a", 0.1),), 3.0)
    send(rig, gen, ph("so the plan is simple", cut=1.0))
    session = rig.c._stream
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert session.heard == ((("a", 0.1),), 3.0)


def test_m3_a_tail_of_noise_is_not_sent_to_the_model(rig):
    gen = locked(rig)
    noise = (np.random.default_rng(1).uniform(-0.002, 0.002, 16000 * 3)).astype(np.float32)
    TRecorder.audio = np.concatenate([speech(2.0), noise])
    pstream(rig).heard_value = ((("so", 0.2),), 2.0)
    rig.asr.text = "what is it"                                    # what a model invents from noise
    send(rig, gen, ph("so the plan is simple", cut=2.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.asr.heard == [] and rig.injector.pasted == ["So the plan is simple."]
    assert rig.c.history.count() == 1 and rig.last_entry()["text"] == "So the plan is simple."


def test_m3_a_word_in_the_last_uncovered_half_second_is_transcribed(rig):
    gen = locked(rig)
    TRecorder.audio = np.concatenate([speech(2.0), np.zeros(int(16000 * 2.5), dtype=np.float32), speech(0.5)])
    pstream(rig).heard_value = ((("so", 0.2),), 4.0)               # the preview looked up to 4.0 s only
    send(rig, gen, ph("so the plan is simple", cut=2.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard) == 1 and rig.injector.pasted[-1] == " The tail words."


def test_m3_a_heard_word_after_the_cut_is_enough_to_transcribe(rig):
    gen = locked(rig)
    TRecorder.audio = np.concatenate([speech(2.0), np.full(16000 * 2, 0.001, dtype=np.float32)])
    pstream(rig).heard_value = ((("so", 0.2), ("then", 2.3)), 4.0)
    send(rig, gen, ph("so the plan is simple", cut=2.0))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.asr.heard) == 1


def test_m4_a_held_modifier_holds_the_phrase_like_the_own_window_rule(rig):
    gen = locked(rig)
    rig.injector.modifiers = True
    pstream(rig).emit(gen, ph("first held phrase", cut=1.0))
    rig.wait(lambda: len(rig.c._stream.held) == 1)
    assert rig.injector.pasted == [] and rig.injector.copied == [], "nothing left on the clipboard"
    rig.injector.modifiers = False
    send(rig, gen, ph("second phrase here", cut=2.0))
    assert rig.injector.pasted == ["First held phrase. Second phrase here."]


def test_m4_an_injector_without_the_modifier_check_still_pastes(rig, monkeypatch):
    from tests.controller_rig import TInjector
    monkeypatch.delattr(TInjector, "modifiers_held")
    gen = locked(rig)
    send(rig, gen, ph("the first phrase", cut=1.0))
    assert rig.injector.pasted == ["The first phrase."]


def test_m5_an_empty_tail_after_a_pause_phrase_pastes_nothing_more(rig):
    gen = locked(rig, text="")
    send(rig, gen, ph("so the plan is simple today", kind="pause", cut=1.0, ends=False))
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert len(rig.injector.pasted) == 1, "no closing full stop"
    assert "closing_period" not in open(ctl_mod.__file__, encoding="utf-8").read()


def test_m1_escape_during_a_phrase_clean_does_not_type_it(rig, monkeypatch):
    gen = locked(rig)
    gate = threading.Event()
    real_clean = rig.c._clean_phrase

    def slow(session, raw, ctx):
        gate.wait(5)
        return real_clean(session, raw, ctx)

    monkeypatch.setattr(rig.c, "_clean_phrase", slow)
    pstream(rig).emit(gen, ph("the cancelled phrase", cut=1.0))
    rig.wait(lambda: rig.c._work.qsize() == 0)
    rig.keys((ESC, True), (ESC, False))
    gate.set()
    rig.settle(80)
    assert rig.c.state == "idle" and rig.injector.pasted == [] and rig.c.history.count() == 0


def test_m2_the_own_window_check_comes_after_the_pacing_wait(rig):
    gen = locked(rig)
    send(rig, gen, ph("the first phrase", cut=1.0))
    rig.c._sleep = lambda s: setattr(rig, "own_focus", True)       # VoxFlow gets the focus while the wait runs
    pstream(rig).emit(gen, ph("the second phrase", cut=2.0))
    rig.wait(lambda: len(rig.c._stream.held) == 1)
    assert rig.injector.pasted == ["The first phrase."], "not typed into VoxFlow's own window"


def test_m2_the_tail_checks_the_window_after_the_pacing_wait(rig):
    gen = locked(rig)
    send(rig, gen, ph("the first phrase", cut=1.0))
    rig.c._sleep = lambda s: setattr(rig, "own_focus", True)
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["The first phrase."] and rig.injector.copied == [" The tail words."]


def test_m3_undo_during_a_streamed_session_does_nothing_but_say_so(rig):
    gen = locked(rig)
    send(rig, gen, ph("the first phrase", cut=1.0))
    assert rig.c.undo_last_paste() is False
    assert rig.spy.notices()[-1] == ("info", "Stop dictating first, then undo.")
    rig.settle(50)
    assert rig.injector.undone == 0
    rig.stop_locked()
    rig.wait(lambda: rig.c.state == "done")
    rig.wait(lambda: rig.c.history.count() == 1)
    assert rig.c.undo_last_paste() is True, "after the release the undo works as before"
