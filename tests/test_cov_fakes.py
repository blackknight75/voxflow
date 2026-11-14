"""voxflow.fakes: the demo stand-ins must honour the same contracts as the real classes."""
from __future__ import annotations

import numpy as np

from tests.helpers import fake_time
from voxflow import fakes
from voxflow.fakes import (FAKE_MICROPHONES, FakeAutostart, FakeCleaner, FakeInjector, FakeMuteBackend,
                           FakeRecorder, FakeTranscriber)


def test_fake_recorder_lifecycle(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(fakes, "time", fake_time(perf_counter=lambda: now[0]))
    rec = FakeRecorder(device="Studio", max_seconds=2.0)
    assert rec.device == "Studio" and rec.alive and not rec.using_fallback and rec.error is None

    silent = rec.stop()                                       # stop without start: nothing recorded
    assert silent.size == 0 and silent.dtype == np.float32 and rec.last_speech_seconds == 0.0

    rec.start()
    assert rec.is_recording and not rec.hit_cap
    now[0] += 0.5
    audio = rec.stop()
    assert not rec.is_recording
    assert audio.dtype == np.float32 and audio.size == int(0.5 * fakes.SAMPLE_RATE)
    assert rec.last_speech_seconds == 0.5 and float(np.abs(audio).max()) <= 1.0

    rec.start()
    now[0] += 5.0                                             # past the cap
    assert rec.hit_cap
    assert rec.stop().size == int(2.0 * fakes.SAMPLE_RATE), "a recording is cut at max_seconds"
    assert rec.last_speech_seconds == 2.0 and not rec.hit_cap


def test_fake_recorder_meter_follows_the_voice_switch_and_the_boost(monkeypatch):
    monkeypatch.setattr(fakes, "time", fake_time(perf_counter=lambda: 12.345))
    rec = FakeRecorder()
    samples = rec.recent_samples(256)
    assert samples.shape == (256,) and samples.dtype == np.float32 and np.abs(samples).max() > 0
    level = rec.level()
    assert 0.0 < level <= 1.0
    rec.level_boost = 1000.0
    assert rec.level() == 1.0, "the meter is clamped"
    rec.active_voice = False
    assert rec.level() == 0.0
    quiet = rec.recent_samples(64)
    assert quiet.shape == (64,) and not quiet.any()


def test_fake_recorder_device_and_close():
    rec = FakeRecorder()
    rec.set_device("Webcam Microphone")
    assert rec.device == "Webcam Microphone"
    rec.start()
    rec.close()
    assert rec.closed and not rec.is_recording
    rec.reopen()
    assert not rec.closed
    rec.close()
    rec.reopen(rescan=False)
    assert not rec.closed


def test_fake_transcriber_returns_its_script():
    asr = FakeTranscriber("any-model", text="hello there", delay_s=0.0, device="cpu")
    asr.load()
    assert asr.backend == "demo"
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "hello there"


def test_fake_cleaner_runs_the_real_rules_and_a_canned_command_mode():
    cleaner = FakeCleaner(delay_s=0.0)
    assert not cleaner.ready and cleaner.has_llm
    cleaner.load()
    assert cleaner.ready
    assert cleaner.clean("um hello world") == "Hello world."
    assert cleaner.last_info, "reports like the real Cleaner"
    rules = cleaner._rules
    cleaner.clean("so yeah")
    assert cleaner._rules is rules, "the rule pipeline is built once"
    assert cleaner.count_tokens("") == 1 and cleaner.count_tokens("abcdefghi") == 3

    rewrite = cleaner.generate([{"role": "system", "content": "s"},
                                {"role": "user", "content": "Make it loud\n\nTEXT:\n  some words \n"}])
    assert rewrite == "SOME WORDS"
    fresh = cleaner.generate([{"role": "user", "content": "write a poem"}])
    assert fresh == "Here is a short demo text written by command mode."


def test_fake_injector_records_instead_of_typing():
    inj = FakeInjector(selection="picked")
    assert inj.last_outcome == "failed"
    assert inj.begin("") is False and inj.pasted == [] and inj.last_outcome == "failed"
    assert inj.begin("one") is True and inj.inject("two") is True
    assert inj.pasted == ["one", "two"] and inj.last_outcome == "pasted"
    assert inj.finish() is None and inj.finish_later() is None and inj.close() is None
    assert inj.copy_selection() == "picked" and inj.copy_selection(timeout=0.1) == "picked"
    assert inj.undo() is True and inj.undo() is True and inj.undone == 2
    assert inj.copy_text("clip") is True and inj.copied == ["clip"]


def test_fake_mute_backend_and_autostart():
    mixer = FakeMuteBackend()
    ids = mixer.mute_others()
    assert ids == ["demo-session"] and mixer.muted == ids
    ids.append("x")
    assert mixer.muted == ["demo-session"], "callers get a copy"
    mixer.unmute(ids)
    assert mixer.muted == [] and mixer.calls == ["mute", "unmute"]

    auto = FakeAutostart()
    assert auto.enabled() is False
    auto.set(1)
    assert auto.enabled() is True
    auto.set(False)
    assert auto.enabled() is False
    assert FakeAutostart(True).enabled() is True
    assert len(FAKE_MICROPHONES) == 3 and all(isinstance(n, str) and n for n in FAKE_MICROPHONES)
