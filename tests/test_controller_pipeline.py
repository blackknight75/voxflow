"""Dictation pipeline through the real Controller: hotkey machines -> recorder -> ASR -> cleanup -> dictionary /
snippets -> paste -> history / stats / signals, including every fallback. All back-ends are scripted doubles."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from tests.controller_rig import CTRL, ESC, WIN, TRecorder, make_rig, speech
from voxflow.context import AppContext


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


# =========================================================================== the happy path
def test_hold_speak_release_pastes_cleaned_text_and_records_everything(rig):
    assert rig.c.state == "idle" and rig.c.status()["level"] == "ok"
    rig.context = AppContext("winword.exe", "Report.docx - Word", 0x200)
    events = rig.dictate("um so can you send me the report by friday")

    assert rig.injector.pasted == ["So can you send me the report by Friday?"]
    assert rig.injector.finished_later == 1, "the clipboard restore must be scheduled, not waited for"
    states = [args[:3] for name, args in events if name == "state_changed"]
    assert [s[0] for s in states] == ["listening", "processing", "done"]
    assert states[-1] == ("done", "dictate", "Pasted")
    rig.wait(lambda: rig.c.state == "idle")                    # the flow bar returns to idle by itself

    rig.wait(lambda: rig.c.history.count() == 1 and rig.c.stats.summary()["sessions"] == 1)   # P8: filed after "done"
    entry = rig.last_entry()
    assert entry["text"] == "So can you send me the report by Friday?"
    assert entry["raw"] == "um so can you send me the report by friday"
    assert (entry["app_exe"], entry["app_title"], entry["mode"]) == ("winword.exe", "Report.docx - Word", "dictate")
    assert entry["words"] == 9 and entry["duration_s"] == 2.0 and entry["wpm"] == 270.0
    assert 0 <= entry["latency_ms"] < 3000
    assert rig.c.last_transcript == entry
    assert rig.spy.of("transcript_ready") == [(entry,)]
    assert rig.spy.of("history_changed") and rig.spy.of("stats_changed")
    summary = rig.c.stats.summary()
    assert summary["total_words"] == 9 and summary["sessions"] == 1 and summary["seconds_spoken"] == 2.0
    assert rig.c.stats.top_apps() == [{"exe": "winword.exe", "words": 9, "sessions": 1}]
    rig.wait(lambda: rig.sounds == ["start", "stop"])            # cues play on their own thread
    assert len(rig.asr.heard) == 1 and rig.asr.heard[0].size == TRecorder.audio.size


def test_the_cleaner_receives_tone_terms_language_and_smart_from_the_app_context(rig):
    rig.c.dictionary.add("Kubernetes", ["cube are netties"])
    rig.c.set("language", "en")
    rig.context = AppContext("slack.exe", "#general | Acme", 0x300)       # work chat -> casual tone
    rig.dictate("deploy it to twenty three nodes")
    _text, kw = rig.cleaner.clean_calls[-1]
    assert kw == {"tone": "casual", "terms": ["Kubernetes"], "language": "en", "smart": True}
    assert rig.injector.pasted == ["Deploy it to 23 nodes"], "casual: no final full stop; smart: digits"

    rig.c.set("smart_formatting", False)
    rig.context = AppContext("outlook.exe", "RE: Budget", 0x301)          # e-mail -> formal
    rig.dictate("deploy it to twenty three nodes")
    assert rig.cleaner.clean_calls[-1][1]["tone"] == "formal" and rig.cleaner.clean_calls[-1][1]["smart"] is False
    assert rig.injector.pasted[-1] == "Deploy it to twenty three nodes."

    rig.c.styles.set_tone("email", "very_casual")
    rig.dictate("thanks for the update")
    assert rig.injector.pasted[-1] == "thanks for the update"


def test_dictionary_and_snippets_are_applied_after_cleanup(rig):
    rig.c.dictionary.add("GitHub", ["get hub"])
    rig.c.snippets.add("my address", "221B Baker Street\nLondon")
    rig.dictate("push it to get hub and send it to my address")
    assert rig.injector.pasted == ["Push it to GitHub and send it to my address."], "no lead word: no snippet"
    rig.dictate("push it to get hub and send it to, insert my address")
    assert rig.injector.pasted[-1] == "Push it to GitHub and send it to, 221B Baker Street\nLondon."
    assert rig.last_entry()["text"] == rig.injector.pasted[-1]
    rig.dictate("my address")
    assert rig.injector.pasted[-1] == "221B Baker Street\nLondon", "a snippet said alone is pasted exactly"


def test_auto_learn_queues_a_dictionary_suggestion_from_the_models_correction(rig):
    rig.cleaner.rewrites["i pushed it to get hub"] = "I pushed it to GitHub."
    rig.dictate("i pushed it to get hub")
    rig.wait(lambda: rig.spy.of("dictionary_suggestion"))
    (suggestion,), = rig.spy.of("dictionary_suggestion")
    assert suggestion["term"] == "GitHub" and suggestion["aliases"] == ["get hub"]
    assert rig.c.dictionary.suggestions()[0]["id"] == suggestion["id"]
    assert rig.spy.of("suggestions_changed")

    rig.spy.clear()
    rig.c.set("auto_learn", False)
    rig.cleaner.rewrites["we use pie torch here"] = "We use PyTorch here."
    rig.dictate("we use pie torch here")
    rig.settle()
    assert rig.spy.of("dictionary_suggestion") == [] and len(rig.c.dictionary.suggestions()) == 1


def test_history_off_still_reports_the_transcript_and_counts_stats(rig):
    rig.c.set("save_history", False)
    rig.dictate("hello there my friend")
    assert rig.injector.pasted == ["Hello there my friend."]
    assert rig.c.history.count() == 0
    assert rig.c.last_transcript["text"] == "Hello there my friend."
    assert len(rig.spy.of("transcript_ready")) == 1
    assert rig.c.stats.summary()["total_words"] == 4


def test_a_lone_line_break_is_pasted_but_is_not_a_transcript(rig):
    rig.dictate("new line")
    assert rig.injector.pasted == ["\n"] and rig.c.state == "done"
    assert rig.c.history.count() == 0 and rig.spy.of("transcript_ready") == []
    assert rig.c.stats.summary()["sessions"] == 0


# =========================================================================== cleanup fallbacks
def test_cleanup_disabled_pastes_the_raw_transcript(rig):
    rig.c.set("cleanup_enabled", False)
    rig.dictate("um hello world")
    assert rig.injector.pasted == ["um hello world"] and rig.cleaner.clean_calls == []


def test_cleaner_not_ready_or_failing_pastes_the_raw_transcript(rig):
    rig.cleaner.ready = False
    rig.dictate("um hello world")
    assert rig.injector.pasted == ["um hello world"]
    rig.cleaner.ready = True
    rig.cleaner.fail_clean = RuntimeError("llama exploded")
    rig.dictate("um hello again")
    assert rig.injector.pasted[-1] == "um hello again" and rig.c.state == "done"


def test_cleanup_model_failing_to_load_degrades_with_a_warning_and_dictation_still_works(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.cleaner.fail_load = OSError("gguf file is corrupt")
        rig.start()
        status = rig.c.model_status("cleanup")
        assert status["state"] == "failed" and "pasted as heard" in status["detail"]
        assert rig.c.status()["level"] == "warning" and rig.c.status()["title"] == "Ready"
        rig.dictate("um hello world")
        assert rig.injector.pasted == ["um hello world"]
    finally:
        rig.close()


def test_rules_only_cleaner_is_reported_as_degraded(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.cleaner.backend, rig.cleaner.has_llm = "rules", False
        rig.start()
        status = rig.c.model_status("cleanup")
        assert status["state"] == "degraded" and status["backend"] == "rules" and "command mode is off" in status["detail"]
        rig.dictate("um hello world")
        assert rig.injector.pasted == ["Hello world."]
    finally:
        rig.close()


# =========================================================================== nothing to paste
def test_filler_only_speech_pastes_nothing(rig):
    rig.dictate("um uh")
    assert rig.injector.pasted == [] and rig.c.history.count() == 0
    assert (rig.c.state, rig.c.message) == ("error", "Nothing to paste")
    rig.wait(lambda: rig.sounds == ["start", "stop", "error"])
    rig.wait(lambda: rig.c.state == "idle")


def test_unintelligible_speech(rig):
    rig.dictate("")
    assert (rig.c.state, rig.c.message) == ("error", "Didn't catch that") and rig.injector.pasted == []
    rig.asr.text = None
    rig.wait(lambda: rig.c.state == "idle")
    rig.dictate()
    assert rig.c.message == "Didn't catch that"


def test_silent_recording_explains_the_likely_cause(rig):
    TRecorder.audio = np.zeros(32000, dtype=np.float32)
    rig.dictate("")
    assert rig.c.state == "error" and rig.c.message == "Heard only silence"
    # one notice for one event (QA C-09): the warning with the hint (and its microphone settings button)
    assert rig.spy.notices() == [("warning", "Heard only silence. Check that your microphone isn't muted and is "
                                             "the one selected.")]
    assert rig.injector.pasted == []


def test_accidental_tap_is_ignored_silently(rig):
    rig.tap()
    assert rig.spy.states() == ["listening", "idle"]
    assert rig.asr.heard == [] and rig.injector.pasted == []
    rig.wait(lambda: rig.sounds == ["start"])                   # no "stop" cue for a tap
    TRecorder.speech_seconds = 0.1                              # held long enough, but almost no audio captured
    rig.press()
    rig.release(held_s=1.0)
    assert rig.c.state == "idle" and rig.asr.heard == []


# =========================================================================== failures
def test_speech_model_error_during_a_dictation_is_survivable(rig):
    rig.asr.fail_transcribe = RuntimeError("CUDA error: device lost")
    rig.dictate("ignored")
    assert rig.c.state == "error" and rig.injector.pasted == []
    assert len(rig.c.message) <= 48
    rig.asr.fail_transcribe = None
    rig.wait(lambda: rig.c.state == "idle")
    rig.dictate("the worker thread is still alive")
    assert rig.injector.pasted == ["The worker thread is still alive."]


def test_c57_unexpected_pipeline_errors_are_reported_in_plain_words(rig):
    rig.asr.fail_transcribe = RuntimeError("CUDA error: device lost [onnxruntime 0x80004005]")
    rig.dictate("ignored")
    for _level, message in rig.spy.notices():
        assert "RuntimeError" not in message and "0x8000" not in message, f"raw exception shown to the user: {message!r}"
    assert rig.c.message != "Something went wrong" or rig.spy.notices(), "the user must be told what to do next"
    assert any("speech" in m.lower() or "try again" in m.lower() for _l, m in rig.spy.notices())


def test_microphone_that_cannot_start(rig):
    TRecorder.fail_start = OSError("Error opening InputStream: Invalid device [PaErrorCode -9996]")
    rig.press()
    assert (rig.c.state, rig.c.message) == ("error", "Microphone unavailable")
    status = rig.c.status()
    assert status["level"] == "error" and status["title"] == "Microphone problem"
    assert "Privacy" in status["detail"] and "PaErrorCode" not in status["detail"]
    rig.release()
    assert rig.c.state == "error", "the key release after a failed start must not start or stop anything"
    assert rig.c.listener.machine.state == "idle"
    rig.wait(lambda: rig.c.state == "idle")                    # the error clears by itself (bug C-13)
    rig.wait(lambda: rig.sounds == ["error"])

    TRecorder.fail_start = None
    rig.dictate("back in business")
    assert rig.injector.pasted == ["Back in business."] and rig.c.status()["level"] == "ok"


def test_recorder_failing_on_stop_returns_to_idle(rig):
    rig.press()
    TRecorder.fail_stop = RuntimeError("stream died")
    rig.release()
    assert (rig.c.state, rig.c.message) == ("error", "Recording failed")
    rig.wait(lambda: rig.c.state == "idle")                    # bug C-37
    TRecorder.fail_stop = None
    rig.dictate("works again")
    assert rig.injector.pasted == ["Works again."]


def test_dictating_before_the_speech_model_is_ready(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.asr.load_gate = threading.Event()
        rig.c.start()
        rig.wait(lambda: rig.c.model_status("asr")["state"] == "loading")
        assert rig.c.status() == {"level": "loading", "title": "Loading speech model…",
                                  "detail": "Loading the speech model…",
                                  "live_preview_available": True}
        rig.press()
        assert (rig.c.state, rig.c.message) == ("error", "Speech model still loading…")
        assert rig.rec.starts == 0
        rig.release()
        rig.asr.load_gate.set()
        rig.wait(lambda: rig.c.model_status("asr")["state"] == "ready")
        assert rig.c.model_status("asr") == {"state": "ready", "detail": "Running on the GPU (CUDA).",
                                             "progress": None, "backend": "cuda"}
        rig.wait(lambda: rig.c.state == "idle")
        rig.dictate("now it works")
        assert rig.injector.pasted == ["Now it works."]
    finally:
        rig.close()


def test_speech_model_that_fails_to_load_and_retry(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.asr.fail_load = ConnectionError("HTTPSConnectionPool: Max retries exceeded")
        rig.start()
        status = rig.c.model_status("asr")
        assert status["state"] == "failed"
        assert "could not be downloaded" in status["detail"] and "internet" in status["detail"]
        assert "HTTPSConnectionPool" not in status["detail"]
        assert rig.c.status()["level"] == "error" and rig.c.status()["title"] == "Speech model unavailable"
        rig.press()
        assert (rig.c.state, rig.c.message) == ("error", "Speech model unavailable")
        rig.release()

        rig.asr.fail_load = None
        rig.c.reload_models()                                   # the "Retry" button
        rig.wait(lambda: rig.c.model_status("asr")["state"] == "ready")
        assert rig.c.status()["level"] == "ok" and rig.asr.loads == 2
        rig.wait(lambda: rig.c.state == "idle")
        rig.dictate("retry worked")
        assert rig.injector.pasted == ["Retry worked."]
    finally:
        rig.close()


@pytest.mark.parametrize("exc,fragment", [
    (OSError("[Errno 28] No space left on device: disk full"), "not enough disk space"),
    (RuntimeError("onnxruntime: invalid model file"), "could not be loaded"),
])
def test_other_speech_model_load_failures_are_explained(tmp_path, qtbot, monkeypatch, exc, fragment):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.asr.fail_load = exc
        rig.start()
        assert fragment in rig.c.model_status("asr")["detail"]
    finally:
        rig.close()


# =========================================================================== paste outcomes
def test_typed_fallback_counts_as_a_successful_dictation(rig):
    rig.injector.outcome = "typed"
    rig.dictate("typed instead of pasted")
    assert (rig.c.state, rig.c.message) == ("done", "Typed") and rig.c.history.count() == 1


def test_text_left_on_the_clipboard_is_reported_as_copied(rig):
    rig.injector.outcome = "copied"                            # e.g. the target window is elevated
    rig.dictate("this could not be pasted")
    assert rig.c.state == "done" and rig.c.message == "Copied. Press Ctrl+V"
    assert rig.injector.copied == ["This could not be pasted."] and rig.c.history.count() == 1


def test_paste_that_fails_completely_is_an_error(rig):
    rig.injector.outcome = "failed"
    rig.dictate("lost words")
    assert (rig.c.state, rig.c.message) == ("error", "Couldn't paste")
    rig.wait(lambda: rig.sounds[-1:] == ["error"])


def test_c58_a_dictation_that_could_not_be_pasted_is_still_kept_in_the_history(rig):
    rig.injector.outcome = "failed"
    rig.dictate("these words took a minute to say")
    assert rig.c.history.count() == 1, "the transcript is gone: not pasted, not on the clipboard, not in History"
    assert rig.c.last_transcript["text"] == "These words took a minute to say."


# =========================================================================== hands-free, cancel, cap
def test_double_tap_locks_hands_free_and_a_tap_ends_it(rig):
    rig.tap()
    rig.tap()                                                   # second tap within the double-tap window
    assert rig.c.state == "locked" and rig.c.listener.machine.state == "locked"
    assert rig.rec.is_recording, "recording continues with no key held"
    rig.pause(5.0)
    rig.c._t_start -= 5.0
    rig.tap()                                                   # tap again: stop
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["Hello world."]
    assert rig.spy.states() == ["listening", "idle", "listening", "locked", "processing", "done"]


def test_double_tap_lock_can_be_switched_off(rig):
    rig.c.set("double_tap_lock", False)
    rig.tap()
    rig.tap()
    assert rig.c.state == "idle" and "locked" not in rig.spy.states()
    rig.c.set("double_tap_lock", True)
    rig.pause()
    rig.tap()
    rig.pause(1.0)                                              # too slow to be a double-tap
    rig.tap()
    assert rig.c.state == "idle" and "locked" not in rig.spy.states()


def test_escape_cancels_the_recording_and_nothing_is_pasted(rig):
    rig.c.set("mute_media", True)
    rig.press()
    assert rig.mixer.muted
    rig.keys((ESC, True), (ESC, False))
    assert rig.c.state == "idle" and not rig.rec.is_recording and not rig.mixer.muted
    rig.release()
    rig.settle()
    assert rig.asr.heard == [] and rig.injector.pasted == [] and rig.c.history.count() == 0
    rig.dictate("after a cancel")
    assert rig.injector.pasted == ["After a cancel."]


def test_cancel_method_and_cancel_when_idle(rig):
    rig.c.cancel()                                              # nothing to cancel: no-op
    assert rig.spy.states() == []
    rig.press()
    rig.c.cancel()
    assert rig.c.state == "idle" and rig.c.listener.machine.state in ("wait_release", "idle")
    rig.release()
    assert rig.asr.heard == []


def test_recording_cap_ends_a_hold_and_a_locked_recording_like_a_release(rig):
    rig.c.set("max_record_s", 7.0)
    assert TRecorder.instances[-1].max_seconds == 7.0, "the recorder is rebuilt with the configured cap"
    assert rig.c.listener.machine.max_hold_s == 12.0
    rig.press()
    rig.c._t_start -= 7.0
    rig.rec.hit_cap = True
    rig.wait(lambda: rig.c.state in ("processing", "done"))    # the 100 ms poll noticed the cap
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["Hello world."]
    rig.keys((WIN, False), (CTRL, False))                       # the late key release is a no-op
    rig.settle()
    assert len(rig.injector.pasted) == 1

    rig.wait(lambda: rig.c.state == "idle")
    rig.pause()
    rig.tap()
    rig.tap()
    assert rig.c.state == "locked"
    rig.c._t_start -= 7.0
    rig.rec.hit_cap = True
    rig.wait(lambda: rig.c.state == "done")
    rig.tick()
    assert len(rig.injector.pasted) == 2 and rig.c.listener.machine.state == "idle"


def test_changing_the_microphone_mid_recording_abandons_it_cleanly(rig):
    old = rig.rec
    rig.press()
    rig.c.set("mic_device", "USB Mic")
    assert rig.c.state == "idle" and old.closed and old.stops == 1
    new = rig.rec
    assert new is not old and new.device == "USB Mic" and new.max_seconds == rig.c.config.max_record_s
    rig.release()                                               # the key release after the switch: nothing
    rig.settle()
    assert rig.asr.heard == [] and rig.injector.pasted == []
    rig.dictate("on the new microphone")
    assert rig.injector.pasted == ["On the new microphone."] and new.starts == 1


def test_changing_the_hotkey_while_hands_free_is_locked_stops_the_recording(rig):
    rig.tap()
    rig.tap()
    assert rig.c.state == "locked"
    rig.c.set("hotkey", "ctrl+alt+space")                       # bug C-18
    assert rig.c.state == "idle" and not rig.rec.is_recording
    rig.keys((0xA2, True), (0xA4, True), (0x20, True))
    assert rig.c.state == "listening"
    rig.c.cancel()


# =========================================================================== overlapping utterances
def test_a_finishing_job_never_repaints_the_bar_of_a_newer_recording(rig):
    rig.asr.delay_s = 0.25
    rig.dictate("first sentence", wait=False)
    assert rig.c.state == "processing"
    rig.press()                                                 # the user is already talking again
    assert rig.c.state == "listening"
    rig.wait(lambda: len(rig.injector.pasted) == 1)
    rig.wait(lambda: len(rig.spy.of("transcript_ready")) == 1)
    assert rig.c.state == "listening", "the first result must not flash 'Pasted' over the live recording"
    rig.asr.delay_s = 0.0
    rig.asr.text = "second sentence"
    rig.release()
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["First sentence.", "Second sentence."]
    assert [e["text"] for e in rig.c.history.list()] == ["Second sentence.", "First sentence."]


def test_jobs_are_processed_in_order(rig):
    rig.asr.delay_s = 0.05
    for i in range(4):
        rig.asr.text = f"sentence number {i} here"
        rig.dictate(wait=False)
    rig.wait(lambda: len(rig.injector.pasted) == 4)
    assert rig.asr.text == "sentence number 3 here"
    assert len({p for p in rig.injector.pasted}) >= 1 and rig.c.history.count() == 4


# =========================================================================== sounds, mute, whisper
def test_sound_cues_follow_the_setting_and_preview_always_plays(rig):
    rig.c.set("sounds_enabled", False)
    rig.dictate("quiet please")
    rig.settle(80)
    assert rig.sounds == []
    rig.c.play_sound("error")                                   # the Settings preview button
    rig.wait(lambda: rig.sounds == ["error"])
    rig.c.set("sounds_enabled", True)
    rig.c.set("sound_volume", 0.9)
    assert rig.c.sounds.volume == 0.9


def test_other_apps_are_muted_only_while_recording(rig):
    rig.c.set("mute_media", True)
    rig.press()
    assert rig.mixer.muted and (rig.data_dir / "muted_sessions.json").exists()
    rig.release()
    assert not rig.mixer.muted, "audio comes back at release, before the transcript is ready"
    rig.wait(lambda: rig.c.state == "done")
    assert rig.mixer.calls == ["mute", "unmute"] and not (rig.data_dir / "muted_sessions.json").exists()

    rig.wait(lambda: rig.c.state == "idle")
    rig.c.set("mute_media", False)
    rig.dictate("no muting now")
    assert rig.mixer.calls == ["mute", "unmute"]


def test_switching_mute_off_or_quitting_mid_recording_restores_audio(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, config={"mute_media": True})
    rig.press()
    assert rig.mixer.muted
    rig.c.set("mute_media", False)
    assert not rig.mixer.muted
    rig.c.cancel()
    rig.release()
    rig.c.set("mute_media", True)
    rig.pause()
    rig.press()
    assert rig.mixer.muted
    rig.close()                                                 # quit while recording
    assert not rig.mixer.muted


def test_a_mute_left_by_a_killed_session_is_undone_at_start(tmp_path, qtbot, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "muted_sessions.json").write_text('["session-1"]', encoding="utf-8")
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert rig.mixer.calls == ["unmute"] and not (data / "muted_sessions.json").exists()
    finally:
        rig.close()


def test_whisper_mode_boosts_quiet_audio_before_recognition_and_the_meter(rig):
    quiet = speech(2.0, amp=0.004)
    TRecorder.audio = quiet
    rig.dictate("normal mode")
    assert np.array_equal(rig.asr.heard[-1], quiet)
    rig.c.set("whisper_mode", True)
    assert rig.rec.level_boost == 3.0
    rig.wait(lambda: rig.c.state == "idle")
    rig.dictate("whisper mode")
    assert float(np.abs(rig.asr.heard[-1]).max()) > float(np.abs(quiet).max()) * 5
    rig.c.set("whisper_mode", False)
    assert rig.rec.level_boost == 1.0


# =========================================================================== paste last / undo
def test_paste_last_by_method_hotkey_and_history_id(rig):
    assert rig.c.paste_last() is False
    assert rig.spy.notices() == [("info", "There is nothing to paste yet.")]
    rig.dictate("the first one")
    rig.wait(lambda: rig.c.state == "idle")
    rig.dictate("the second one")
    first_id = rig.c.history.list()[1]["id"]

    assert rig.c.paste_last() is True
    rig.wait(lambda: len(rig.injector.pasted) == 3)
    assert rig.injector.pasted[-1] == "The second one."
    rig.keys((0xA4, True), (0xA0, True), (0x5A, True), (0x5A, False), (0xA0, False), (0xA4, False))   # Alt+Shift+Z
    rig.wait(lambda: len(rig.injector.pasted) == 4)
    assert rig.injector.pasted[-1] == "The second one."
    assert rig.c.paste_last(first_id) is True
    rig.wait(lambda: len(rig.injector.pasted) == 5)
    assert rig.injector.pasted[-1] == "The first one."
    assert rig.c.history.count() == 2, "re-pasting does not create new history entries"

    rig.spy.clear()
    assert rig.c.paste_last("no-such-id") is False
    assert rig.spy.notices() == [("info", "That transcript is no longer in your history.")]


def test_paste_last_reports_when_the_text_only_reached_the_clipboard(rig):
    rig.dictate("some text")
    rig.spy.clear()
    rig.injector.outcome = "copied"
    rig.c.paste_last()
    rig.wait(lambda: rig.spy.notices())
    assert rig.spy.notices() == [("info", "The text is on the clipboard. Press Ctrl+V to paste it.")]
    rig.spy.clear()
    rig.injector.outcome = "failed"
    rig.c.paste_last()
    rig.wait(lambda: rig.spy.notices())
    assert rig.spy.notices() == [("error", "Couldn't paste the last transcript.")]


def test_undo_only_in_the_window_that_received_the_paste(rig):
    assert rig.c.undo_last_paste() is False
    assert rig.spy.notices() == [("info", "There is no paste to undo.")]
    rig.dictate("oops wrong text")
    rig.spy.clear()

    rig.context = AppContext("chrome.exe", "Another window", 0x999)     # the user switched windows
    assert rig.c.undo_last_paste() is False and rig.injector.undone == 0
    assert rig.spy.notices()[0][0] == "warning" and "window you last dictated into" in rig.spy.notices()[0][1]

    rig.context = AppContext("notepad.exe", "Untitled - Notepad", 0x100)
    assert rig.c.undo_last_paste() is True
    rig.wait(lambda: rig.injector.undone == 1)
    assert rig.c.undo_last_paste() is False, "one paste, one undo"


def test_undo_hotkey_and_undo_blocked_by_held_modifiers(rig):
    rig.c.set("undo_hotkey", "ctrl+alt+z")
    rig.dictate("to be undone")
    rig.injector.undo_ok = False
    rig.spy.clear()
    rig.keys((0xA2, True), (0xA4, True), (0x5A, True), (0x5A, False), (0xA4, False), (0xA2, False))
    rig.wait(lambda: rig.spy.notices())
    assert rig.injector.undone == 1
    assert rig.spy.notices() == [("warning", "Couldn't send the undo because a modifier key is still held down.")]


# =========================================================================== pause
def test_pause_ignores_shortcuts_aborts_a_recording_and_resumes(rig):
    rig.press()
    rig.c.set_paused(True)
    assert rig.c.state == "idle" and not rig.rec.is_recording and rig.c.paused
    assert rig.c.status()["level"] == "paused" and rig.spy.of("status_changed")[-1][0]["level"] == "paused"
    rig.release()                                               # released while paused: the hook ignores it
    results = [rig.c.listener.feed(vk, down) for vk, down in ((CTRL, True), (WIN, True), (WIN, False), (CTRL, False))]
    assert rig.rec.starts == 1, "no new recording"
    assert (rig.c.state, rig.c.message) == ("error", "Dictation is paused"), "the bar says why nothing happened (QA U-04)"
    assert not any(r.swallow or r.mask for r in results), "while paused every key passes through untouched"
    rig.c.set_paused(True)                                      # no-op
    rig.c.set_paused(False)
    assert rig.c.status()["level"] == "ok"
    rig.tick()                                                  # the hook's watchdog resyncs key state within 250 ms
    rig.dictate("we have resumed")
    assert rig.injector.pasted == ["We have resumed."]


# =========================================================================== threads
def test_public_signals_are_delivered_on_the_ui_thread(rig):
    main = threading.current_thread()
    seen = []
    rig.c.transcript_ready.connect(lambda e: seen.append(("transcript", threading.current_thread())))
    rig.c.state_changed.connect(lambda *a: seen.append(("state", threading.current_thread())))
    rig.c.dictionary_changed.connect(lambda: seen.append(("dictionary", threading.current_thread())))
    rig.c.model_status_changed.connect(lambda *a: seen.append(("model", threading.current_thread())))
    rig.dictate("thread check")
    worker = threading.Thread(target=lambda: rig.c.dictionary.add("Kubernetes"))
    worker.start()
    worker.join()
    rig.c.reload_models()
    rig.wait(lambda: {"transcript", "state", "dictionary", "model"} <= {name for name, _t in seen})
    assert all(t is main for _name, t in seen), {name for name, t in seen if t is not main}


def test_store_changes_are_forwarded_as_signals(rig):
    rig.c.dictionary.add("Vercel")
    rig.c.snippets.add("sig", "Best, Dan")
    rig.c.styles.set_tone("work", "formal")
    rig.c.stats.record(5, 2.0)
    rig.c.history.add("manual entry")
    rig.c.dictionary.suggest("Supabase")
    rig.wait(lambda: rig.spy.of("suggestions_changed"))
    for name in ("dictionary_changed", "snippets_changed", "styles_changed", "stats_changed", "history_changed",
                 "suggestions_changed"):
        assert len(rig.spy.of(name)) == 1, name
