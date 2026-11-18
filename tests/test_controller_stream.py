"""Live preview through the real Controller (design: plan/STREAMING_DESIGN.md sections 2.4, 3, 4, 6-B).

Two kinds of tests. Most use ``FakePreviewer`` (a scripted stand-in for ``voxflow.stream.Previewer``) so the
controller's own rules are checked deterministically: when a session starts and stops, that stale or late results
are dropped, what happens on cancel / pause / setting change / shutdown. A few use the real ``Previewer`` with the
scripted ``TAsr.transcribe_timed`` to check the whole path, and that the pasted text does not depend on the preview."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from tests.controller_rig import ESC, TRecorder, make_rig
from voxflow import controller as ctl_mod
from voxflow import stream
from voxflow.fakes import FakeTranscriber


class FakePreviewer:
    """Records start / stop / close; ``emit`` plays a result as the preview thread would deliver it."""

    instances: list = []

    def __init__(self, recorder, transcriber, on_partial, *, gain=None, **kw):
        self.recorder, self.transcriber, self.on_partial, self.gain = recorder, transcriber, on_partial, gain
        self.available = True
        self.log: list = []
        self.recording_at_stop: list[bool] = []
        FakePreviewer.instances.append(self)

    def start(self, gen):
        self.log.append(("start", gen))

    def stop(self):
        self.log.append(("stop",))
        self.recording_at_stop.append(bool(self.recorder.is_recording))

    def close(self):
        self.log.append(("close",))

    def emit(self, gen, committed, tentative=""):
        self.on_partial(gen, committed, tentative)

    def starts(self):
        return [e[1] for e in self.log if e[0] == "start"]


@pytest.fixture
def fake_preview(monkeypatch):
    FakePreviewer.instances = []
    monkeypatch.setattr(ctl_mod.stream, "Previewer", FakePreviewer)
    return FakePreviewer


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch, fake_preview):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


def preview(rig) -> FakePreviewer:
    return rig.c._preview


def partials(rig) -> list[tuple]:
    return rig.spy.of("partial_text")


# =========================================================================== session lifecycle
def test_a_hold_starts_one_session_for_its_generation_and_the_release_stops_it_before_the_recorder(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    assert p.starts() == [rig.c._gen] and rig.c.state == "listening"
    rig.release()
    rig.wait(lambda: rig.c.state == "done")
    assert p.log == [("start", rig.c._gen), ("stop",)]
    assert p.recording_at_stop == [True], "stop() must come before recorder.stop(): no new call after the release"
    assert p.transcriber is rig.c._asr and p.recorder is rig.rec


def test_command_mode_is_previewed_too(rig):
    rig.cleaner.reply = lambda messages: "REWRITTEN"
    rig.pause()
    rig.press("command")
    assert preview(rig).starts() == [rig.c._gen] and rig.c.mode == "command"
    rig.release("command")
    rig.wait(lambda: rig.c.state in ("done", "error"))


def test_double_tap_lock_keeps_one_session_per_generation(rig):
    p = preview(rig)
    rig.tap()
    first = rig.c._gen
    rig.tap()
    assert rig.c.state == "locked"
    assert p.starts() == [first, first + 1], "each press starts its own recording; the lock adds no session"
    gen = rig.c._gen
    rig.pause(5.0)
    rig.c._t_start -= 5.0
    rig.tap()
    rig.wait(lambda: rig.c.state == "done")
    assert p.starts() == [first, first + 1] and gen == first + 1


def test_a_quick_tap_stops_the_session_it_started(rig):
    p = preview(rig)
    rig.tap()
    assert p.log == [("start", rig.c._gen), ("stop",)] and rig.c.state == "idle"


def test_escape_cancel_stops_the_session_and_nothing_more_arrives(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    gen = rig.c._gen
    p.emit(gen, "hello")
    rig.keys((ESC, True), (ESC, False))
    assert rig.c.state == "idle" and p.log[-1] == ("stop",)
    assert p.recording_at_stop[-1] is True
    before = len(partials(rig))
    p.emit(gen, "hello world")                                  # a result that was already on its way
    rig.settle()
    assert len(partials(rig)) == before


def test_the_cap_ends_the_session_like_a_release(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    rig.c._t_start -= 7.0
    rig.rec.hit_cap = True
    rig.wait(lambda: rig.c.state == "done")
    assert p.log.count(("stop",)) == 1


def test_pause_stops_the_session(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    rig.c.set_paused(True)
    assert p.log[-1] == ("stop",) and rig.c.state == "idle"
    rig.c.set_paused(False)


def test_shutdown_closes_the_previewer(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    rig.c.shutdown()
    assert p.log[-1] == ("close",)
    rig.c.shutdown()                                            # idempotent


def test_a_new_recorder_or_new_models_get_a_new_previewer(rig):
    old = preview(rig)
    rig.c.set("mic_device", "Other Mic")
    new = preview(rig)
    assert new is not old and new.recorder is rig.c._recorder and old.log[-1] == ("close",)
    again = rig.c._preview
    rig.c.reload_models()                                       # injected modules: nothing is rebuilt, so neither is it
    assert rig.c._preview is again
    rig.c._modules_override = None
    rig.c._Rec = TRecorder
    rig.c._build_modules = lambda: setattr(rig.c, "_asr", rig.asr)
    rig.c.reload_models(("cleanup",))
    assert rig.c._preview is not again and again.log[-1] == ("close",)


# =========================================================================== when there is no preview
def test_not_started_while_the_speech_model_is_not_ready(rig):
    p = preview(rig)
    rig.c._models["asr"]["state"] = "loading"
    rig.pause()
    rig.press()
    assert rig.c.state == "error" and p.starts() == []
    rig.release()
    rig.c._models["asr"]["state"] = "ready"


def test_the_setting_off_means_no_session_and_on_means_a_session(rig):
    p = preview(rig)
    rig.c.set("live_preview", False)
    rig.dictate("quiet please")
    assert p.starts() == [] and rig.injector.pasted == ["Quiet please."]
    rig.c.set("live_preview", True)
    rig.dictate("loud please")
    assert len(p.starts()) == 1


def test_switching_it_off_mid_utterance_stops_and_clears_the_caption(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    p.emit(rig.c._gen, "so far")
    assert partials(rig) == [("so far", "")]
    rig.c.set("live_preview", False)
    assert p.log[-1] == ("stop",) and partials(rig)[-1] == ("", "")
    rig.release()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["Hello world."]
    rig.c.set("live_preview", True)
    rig.c.set("live_preview", False)                            # switching it off while idle changes nothing else
    assert partials(rig)[-1] == ("", "")


def test_a_previewer_that_gave_up_is_not_started_and_the_status_says_so(rig):
    p = preview(rig)
    assert rig.c.status()["live_preview_available"] is True
    p.available = False
    assert rig.c.status()["live_preview_available"] is False and rig.c.status()["level"] == "ok"
    rig.dictate("still works")
    assert p.starts() == [] and rig.injector.pasted == ["Still works."]


# =========================================================================== what reaches the caption
def test_partials_reach_the_caption_in_order_while_the_key_is_held(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    gen = rig.c._gen
    for committed, tentative in (("", "hello"), ("hello", "wor"), ("hello world", "")):
        p.emit(gen, committed, tentative)
    assert partials(rig) == [("", "hello"), ("hello", "wor"), ("hello world", "")]
    rig.release()
    rig.wait(lambda: rig.c.state == "done")


def test_nothing_reaches_the_caption_after_the_release(rig):
    p = preview(rig)
    rig.asr.delay_s = 0.15                                      # the final pass is still running
    rig.pause()
    rig.press()
    gen = rig.c._gen
    p.emit(gen, "hello")
    rig.release()
    assert rig.c.state == "processing"
    p.emit(gen, "hello world")                                  # the preview thread's last result lands late
    rig.settle()
    assert partials(rig) == [("hello", "")], "the caption freezes at the release"
    rig.wait(lambda: rig.c.state == "done")
    p.emit(gen, "after done")
    assert partials(rig) == [("hello", "")]


def test_a_result_of_an_older_utterance_never_shows_under_the_next_one(rig):
    p = preview(rig)
    rig.asr.delay_s = 0.2
    rig.pause()
    rig.press()
    first = rig.c._gen
    p.emit(first, "first")
    rig.release()
    rig.pause()
    rig.press()                                                 # the second dictation starts while the first is processing
    second = rig.c._gen
    assert second == first + 1 and rig.c.state == "listening"
    p.emit(first, "first and stale")
    p.emit(second, "second")
    assert partials(rig) == [("first", ""), ("second", "")]
    rig.release()
    rig.wait(lambda: rig.c.state == "done" and len(rig.injector.pasted) == 2)


def test_a_result_cannot_arrive_through_the_queued_signal_after_the_stop(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    gen = rig.c._gen
    p.emit(gen, "hello")
    rig.keys((ESC, True), (ESC, False))
    rig.c._sig_partial.emit(gen, "late", "")                    # what the queued cross-thread signal would deliver
    rig.settle()
    assert partials(rig) == [("hello", "")]


# =========================================================================== the whole path, real Previewer
@pytest.fixture
def real(tmp_path, qtbot, monkeypatch):
    """A rig with the real ``Previewer`` ticking every 20 ms."""
    monkeypatch.setattr(ctl_mod.stream, "Previewer", functools.partial(stream.Previewer, interval_s=0.02))
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


def grow(rig, seconds):
    rig.rec.recorded = min(len(TRecorder.audio), rig.rec.recorded + int(seconds * 16000))


def test_real_previewer_shows_growing_words_and_stops_at_the_release(real):
    real.asr.partials = [["hello"], ["hello", "world"], ["hello", "world", "again"]]
    TRecorder.audio = np.full(16000 * 6, 0.2, dtype=np.float32)
    real.pause()
    real.press()
    for _ in range(3):
        grow(real, 0.8)
        n = len(real.spy.of("partial_text"))
        real.wait(lambda n=n: len(real.spy.of("partial_text")) > n)
    texts = [(c + " " + t).strip() for c, t in real.spy.of("partial_text")]
    assert texts[0] == "hello" and texts[-1] == "hello world again"
    assert all(real.asr.timed_calls[i]["blocking"] is False for i in range(len(real.asr.timed_calls)))
    real.release()
    real.wait(lambda: real.c.state == "done")
    seen = len(real.spy.of("partial_text"))
    real.settle(150)
    assert len(real.spy.of("partial_text")) == seen, "no caption update after the release"
    assert real.asr.heard and len(real.asr.heard[-1]) == len(TRecorder.audio)


def test_the_pasted_text_is_the_same_with_the_preview_on_and_off(real):
    real.asr.partials = lambda n, audio: ["totally", "different", "preview", "words"]
    texts = {}
    for on in (True, False):
        real.c.set("live_preview", on)
        real.pause()
        real.press()
        if on:
            grow(real, 1.0)
            real.wait(lambda: real.spy.of("partial_text"))
        real.release()
        real.wait(lambda: real.c.state == "done")
        real.wait(lambda: real.c.history.count() == (1 if on else 2))
        real.wait(lambda: real.c.state == "idle")
        texts[on] = (real.injector.pasted[-1], real.cleaner.clean_calls[-1][0], real.asr.heard[-1].tobytes())
    assert texts[True] == texts[False] == ("Hello world.", "um hello world", np.asarray(TRecorder.audio).tobytes())
    assert real.c.history.list(limit=2)[0]["text"] == real.c.history.list(limit=2)[1]["text"]


def test_a_model_that_cannot_preview_disables_it_and_dictation_goes_on(real):
    from voxflow import asr
    real.asr.fail_timed = asr.PreviewUnavailable("no")
    real.pause()
    real.press()
    grow(real, 1.0)
    real.wait(lambda: not real.c.status()["live_preview_available"])
    real.release()
    real.wait(lambda: real.c.state == "done")
    assert real.injector.pasted == ["Hello world."] and real.spy.of("partial_text") == []
    real.dictate("again")                                       # no session is started for it any more
    assert real.injector.pasted[-1] == "Again"


def test_a_busy_model_skips_the_preview_not_the_dictation(real):
    real.asr.timed_busy = True
    real.pause()
    real.press()
    grow(real, 1.0)
    real.wait(lambda: real.asr.timed_calls)
    real.release()
    real.wait(lambda: real.c.state == "done")
    assert real.spy.of("partial_text") == [] and real.injector.pasted == ["Hello world."]


def test_silence_says_nothing_and_does_not_hammer_the_model(real):
    TRecorder.audio = np.zeros(16000 * 4, dtype=np.float32)
    real.pause()
    real.press()
    for _ in range(4):
        grow(real, 0.5)
        real.settle(60)
    assert len(real.asr.timed_calls) <= 2, "one look at the silence, one more so the tail can settle, then none"
    real.release()
    real.wait(lambda: real.c.state in ("done", "error", "idle"))


def test_a_two_minute_utterance_keeps_previewing_and_the_final_pass_gets_all_of_it(real):
    TRecorder.audio = np.full(16000 * 120, 0.2, dtype=np.float32)
    real.asr.partials = lambda n, audio: [f"w{i}" for i in range(min(40, n + 1))]
    real.pause()
    real.press()
    for _ in range(6):
        grow(real, 20.0)
        real.settle(60)
    assert len(real.asr.timed_calls) >= 3 and real.spy.of("partial_text")
    real.release()
    real.wait(lambda: real.c.state == "done")
    assert len(real.asr.heard[-1]) == 16000 * 120 and len(real.injector.pasted) == 1


# =========================================================================== whisper mode and the fake models
def test_whisper_mode_raises_the_preview_audio_like_the_final_pass(rig):
    quiet = (np.sin(np.arange(16000 * 2) / 7.0) * 0.003 * (1 + np.sin(np.arange(16000 * 2) / 4000.0))).astype(np.float32)
    gain = rig.c._preview.gain
    assert gain(quiet) is quiet
    rig.c.set("whisper_mode", True)
    loud = gain(quiet)
    assert loud is not quiet and float(np.max(np.abs(loud))) > float(np.max(np.abs(quiet)))


def test_the_fake_models_run_with_the_real_previewer(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch, asr=FakeTranscriber(text="hello there world", delay_s=0.0))
    try:
        r.pause()
        r.press()
        assert r.c._preview.available
        r.release()
        r.wait(lambda: r.c.state in ("done", "error", "idle"))
        assert r.injector.pasted == ["Hello there world."]
    finally:
        r.close()


# =========================================================================== the previewer gave up (round 2)
def test_status_is_announced_once_when_the_previewer_is_seen_to_have_given_up(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    rig.spy.clear()
    p.available = False
    p.emit(rig.c._gen, "", "")                                   # its clear when it disables itself
    rig.release()
    rig.wait(lambda: rig.c.state == "done")
    statuses = [a[0] for a in rig.spy.of("status_changed") if a[0].get("live_preview_available") is False]
    assert len(statuses) == 1, "once, however many times it is observed afterwards"
    rig.pause()
    rig.press()
    rig.keys((ESC, True), (ESC, False))                          # observed again on the cancel path
    assert len([a for a in rig.spy.of("status_changed") if a[0].get("live_preview_available") is False]) == 1


def test_status_is_announced_from_the_release_and_cancel_paths(rig):
    p = preview(rig)
    rig.pause()
    rig.press()
    p.available = False
    rig.spy.clear()
    rig.keys((ESC, True), (ESC, False))
    assert [a[0]["live_preview_available"] for a in rig.spy.of("status_changed")] == [False]
    rig.release()
    rig.c._build_preview()                                       # a rebuilt previewer is told afresh
    p2 = preview(rig)
    rig.pause()
    rig.press()
    p2.available = False
    rig.spy.clear()
    rig.release()
    assert [a[0]["live_preview_available"] for a in rig.spy.of("status_changed")] == [False]
    rig.wait(lambda: rig.c.state == "done")


def test_the_disable_clear_of_the_real_previewer_reaches_the_caption(real):
    from voxflow import asr
    real.asr.partials = [["hello"], ["hello", "world"]]
    real.pause()
    real.press()
    grow(real, 1.0)
    real.wait(lambda: real.spy.of("partial_text"))
    real.asr.fail_timed = asr.PreviewUnavailable("gone")
    grow(real, 1.0)
    real.wait(lambda: not real.c._preview.available)
    real.wait(lambda: real.spy.of("partial_text")[-1] == ("", ""))
    real.wait(lambda: [a for a in real.spy.of("status_changed") if a[0].get("live_preview_available") is False])
    real.release()
    real.wait(lambda: real.c.state == "done")
    assert real.injector.pasted == ["Hello world."]


# =========================================================================== round 3
def test_a_result_queued_before_the_setting_was_switched_off_does_not_bring_the_caption_back(rig):
    rig.pause()
    rig.press()
    gen = rig.c._gen
    rig.c.set("live_preview", False)
    rig.c._on_partial(gen, "queued earlier", "")                # already in the Qt queue when it was switched off
    assert partials(rig) == [("", "")]


def test_swapping_the_microphone_under_a_session_clears_the_caption(rig):
    rig.pause()
    rig.press()
    preview(rig).emit(rig.c._gen, "so far")
    rig.c._apply("mic_device", "Other Mic")                     # rebuilds the previewer; the new one has no session
    assert partials(rig)[-1] == ("", "")
    rig.release()
    rig.wait(lambda: rig.c.state == "done")
    assert len(preview(rig).starts()) == 0
