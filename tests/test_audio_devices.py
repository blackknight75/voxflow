"""voxflow.audio beyond the basic recorder: device discovery by name, hot-unplug recovery, error messages and
whisper-mode gain. PortAudio is replaced by a fake; no real microphone is opened."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voxflow import audio
from voxflow.audio import (MIC_HELP, SAMPLE_RATE, Recorder, default_input_name, describe_mic_error,
                           list_input_devices, match_device, normalize_gain)


class FakeStream:
    def __init__(self, sd, **kw):
        self.sd, self.kw = sd, kw
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class FakeSD:
    """A minimal sounddevice: a device table, a default input and stream construction that can fail."""

    MME, WASAPI = 0, 1

    def __init__(self):
        self.devices = [
            dict(name="Microsoft Sound Mapper - Input", hostapi=0, max_input_channels=2),
            dict(name="Microphone (Yeti Nano)", hostapi=0, max_input_channels=2),
            dict(name="Headset Microphone (Jabra Evolve", hostapi=0, max_input_channels=1),      # MME: 31 chars
            dict(name="Speakers (Realtek Audio)", hostapi=0, max_input_channels=0),
            dict(name="Microphone (Yeti Nano)", hostapi=1, max_input_channels=2),
            dict(name="Headset Microphone (Jabra Evolve2 65 Mono)", hostapi=1, max_input_channels=1),
            dict(name="Headset Microphone (Jabra Evolve2 65 Mono) Hands-Free", hostapi=1, max_input_channels=1),
        ]
        self.default = types.SimpleNamespace(device=[1, 3])
        self.streams: list[FakeStream] = []
        self.fail_devices: dict = {}                 # device -> exception
        self.rescans = 0

    def query_devices(self, index=None):
        if index is None:
            return list(self.devices)
        return self.devices[index]

    def InputStream(self, **kw):
        if kw["device"] in self.fail_devices:
            raise self.fail_devices[kw["device"]]
        s = FakeStream(self, **kw)
        self.streams.append(s)
        return s

    def _terminate(self):
        self.rescans += 1

    def _initialize(self):
        pass


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setattr(audio, "_sd", lambda: fake)
    return fake


def feed(rec, samples):
    rec._callback(np.asarray(samples, dtype=np.float32).reshape(-1, 1), len(samples), None, None)


# ---------------------------------------------------------------- device discovery
def test_devices_are_listed_once_with_their_full_names(sd):
    assert list_input_devices() == [(1, "Microphone (Yeti Nano)"), (2, "Headset Microphone (Jabra Evolve2 65 Mono)")]


def test_only_the_default_host_api_is_listed_and_outputs_are_skipped(sd):
    sd.default.device = [4, 3]                        # default input is on the second host API
    assert [name for _i, name in list_input_devices()] == [
        "Microphone (Yeti Nano)", "Headset Microphone (Jabra Evolve2 65 Mono)",
        "Headset Microphone (Jabra Evolve2 65 Mono) Hands-Free"]
    sd.default.device = [-1, -1]                      # no default device: list every input once
    names = [name for _i, name in list_input_devices()]
    assert "Speakers (Realtek Audio)" not in names and len(names) == len(set(names)) and len(names) == 4


def test_no_audio_subsystem_is_an_empty_list_not_a_crash(sd, caplog):
    sd.query_devices = lambda index=None: (_ for _ in ()).throw(RuntimeError("PortAudio not initialised"))
    with caplog.at_level("WARNING"):
        assert list_input_devices() == []
    assert "could not list microphones" in caplog.text
    assert default_input_name() == ""


def test_default_input_name(sd):
    assert default_input_name() == "Microphone (Yeti Nano)"
    sd.default.device = [2, 3]
    assert default_input_name() == "Headset Microphone (Jabra Evolve2 65 Mono)", "full name, not the 31-char MME one"
    sd.default.device = [-1, 3]
    assert default_input_name() == ""
    sd.default.device = [None, None]
    assert default_input_name() == ""


DEVICES = [(1, "Microphone (Yeti Nano)"), (2, "Headset Microphone (Jabra Evolve2 65 Mono)"), (7, "USB Mic")]


@pytest.mark.parametrize("spec,expected", [
    (None, None), ("", None), ("   ", None),
    ("Microphone (Yeti Nano)", 1), ("microphone (yeti nano)", 1), ("  USB Mic ", 7),
    ("Headset Microphone (Jabra Evolve", 2),             # stored when only the truncated name was known
    ("Headset Microphone (Jabra Evolve2 65 Mono) Extra", 2),
    ("yeti", 1), ("Jabra", 2),
    ("Blue Snowball", None),                             # unplugged
    (5, 5), (0, 0),                                      # legacy index: used as-is
    (True, None),
])
def test_match_device(spec, expected):
    got = match_device(spec, DEVICES)
    if spec is True:
        assert got is None or got == 1, "a bool is never a meaningful device"
    else:
        assert got == expected


@pytest.mark.parametrize("exc,start", [
    (RuntimeError("Error querying device -1"), "No microphone found."),
    (ValueError("No default input device available"), "No microphone found."),
    (ValueError("No input device matching 'Yeti'"), "The selected microphone is not available."),
    (RuntimeError("Invalid device [PaErrorCode -9996]"), "The selected microphone is not available."),
    (OSError("Unanticipated host error [PaErrorCode -9999]: 'Access is denied'"), "The microphone could not be opened."),
    (None, "The microphone could not be opened."),
])
def test_mic_errors_are_explained_in_plain_words(exc, start):
    msg = describe_mic_error(exc)
    assert msg.startswith(start) and msg.endswith(MIC_HELP)
    assert "PaErrorCode" not in msg and "Privacy" in msg


def test_rescan_reinitialises_portaudio_and_survives_failure(sd, caplog):
    audio.rescan_devices()
    assert sd.rescans == 1
    sd._initialize = lambda: (_ for _ in ()).throw(RuntimeError("busy"))
    with caplog.at_level("WARNING"):
        audio.rescan_devices()
    assert "rescan failed" in caplog.text


def test_input_device_count(monkeypatch):
    if sys.platform == "win32":
        assert audio.input_device_count() >= 0          # read-only query of the wave-in device count
    monkeypatch.setattr(audio.sys, "platform", "linux")
    assert audio.input_device_count() == -1


# ---------------------------------------------------------------- recorder: device handling
def test_recorder_opens_the_named_device_at_16k_mono(sd):
    rec = Recorder("Headset Microphone (Jabra Evolve2 65 Mono)")
    kw = sd.streams[0].kw
    assert kw["device"] == 2 and kw["samplerate"] == SAMPLE_RATE and kw["channels"] == 1 and kw["dtype"] == "float32"
    assert rec.device == "Headset Microphone (Jabra Evolve2 65 Mono)"
    assert rec.error is None and not rec.using_fallback and rec.alive
    assert sd.streams[0].active, "the stream is kept open so the first syllable is never clipped"


def test_missing_named_device_falls_back_to_the_system_default(sd):
    rec = Recorder("Blue Snowball")
    assert sd.streams[0].kw["device"] is None and rec.using_fallback and rec.error is None
    assert rec.device == "Blue Snowball", "the user's choice is remembered for when it is plugged back in"


def test_device_that_fails_to_open_falls_back_to_the_default(sd, caplog):
    sd.fail_devices = {1: OSError("device busy")}
    with caplog.at_level("WARNING"):
        rec = Recorder("Microphone (Yeti Nano)")
    assert [s.kw["device"] for s in sd.streams] == [None] and rec.using_fallback
    assert "trying the system default" in caplog.text


def test_no_working_microphone_is_remembered_and_raised_on_start(sd):
    sd.fail_devices = {None: OSError("no default input"), 1: OSError("device busy")}
    rec = Recorder()                                       # constructor never raises
    assert isinstance(rec.error, OSError) and not rec.alive and rec.level() == 0.0
    with pytest.raises(OSError, match="no default input"):
        rec.start()
    assert not rec.is_recording
    named = Recorder("Microphone (Yeti Nano)")
    assert str(named.error) == "no default input"
    sd.fail_devices = {}
    rec.start()                                            # the microphone came back: works without a restart
    assert rec.is_recording and rec.error is None


def test_unplugged_device_is_detected_and_the_dead_stream_is_released(sd, monkeypatch):
    now = [100.0]
    from tests.helpers import fake_time
    monkeypatch.setattr(audio, "time", fake_time(monotonic=lambda: now[0]))
    rec = Recorder("Microphone (Yeti Nano)")
    feed(rec, np.full(512, 0.5))
    assert rec.alive
    now[0] += audio._STALL_S + 0.1                         # callbacks stopped arriving
    assert not rec.alive
    first = sd.streams[0]
    first.active = False                                   # PortAudio marks the stream as stopped
    assert not rec.alive
    rec.start()                                            # re-opens transparently
    assert first.closed, "the dead stream must be closed, not leaked (bug C-16)"
    assert len(sd.streams) == 2 and rec.alive
    assert not rec.stop()[:int(audio.PREROLL_S * SAMPLE_RATE)].any(), "no stale pre-roll from before the outage"


def test_alive_is_false_without_a_stream_or_when_the_stream_object_is_broken(sd):
    rec = Recorder()
    rec.close()
    assert not rec.alive
    rec2 = Recorder()

    class Broken:
        @property
        def active(self):
            raise RuntimeError("stream handle invalid")

    rec2._stream = Broken()
    assert not rec2.alive


def test_reopen_rescans_and_opens_again(sd):
    rec = Recorder()
    rec.reopen()
    assert sd.rescans == 1 and sd.streams[0].closed and len(sd.streams) == 2 and rec.alive
    rec.reopen(rescan=False)
    assert sd.rescans == 1 and len(sd.streams) == 3
    sd.fail_devices = {None: OSError("gone")}
    with pytest.raises(OSError):
        rec.reopen()
    assert not rec.alive and isinstance(rec.error, OSError)


def test_switching_microphone_mid_recording_keeps_what_was_recorded(sd):
    rec = Recorder("Microphone (Yeti Nano)")
    rec.start()
    feed(rec, np.full(1600, 0.25))
    rec.set_device("Headset Microphone (Jabra Evolve2 65 Mono)")
    assert rec.is_recording and sd.streams[0].closed and sd.streams[1].kw["device"] == 2
    feed(rec, np.full(1600, 0.5))
    out = rec.stop()
    pre = int(audio.PREROLL_S * SAMPLE_RATE)
    assert out.size == pre + 3200
    assert np.allclose(out[pre:pre + 1600], 0.25) and np.allclose(out[pre + 1600:], 0.5), "bug C-17"


def test_close_survives_a_stream_that_raises(sd):
    rec = Recorder()
    sd.streams[0].stop = lambda: (_ for _ in ()).throw(RuntimeError("already stopped"))
    sd.streams[0].close = lambda: (_ for _ in ()).throw(RuntimeError("already closed"))
    rec.start()
    rec.close()
    rec.close()
    assert not rec.is_recording and not rec.alive


def test_level_boost_scales_the_meter_only(sd):
    rec = Recorder()
    quiet = np.full(512, 0.01)
    feed(rec, quiet)
    normal = rec.level()
    rec2 = Recorder()
    rec2.level_boost = 5.0
    feed(rec2, quiet)
    assert rec2.level() == pytest.approx(normal * 5, rel=1e-3)
    rec2.level_boost = 10_000
    feed(rec2, quiet)
    assert rec2.level() == 1.0, "the meter is clamped"
    rec2.start()
    feed(rec2, quiet)
    assert np.allclose(rec2.stop()[-512:], 0.01), "the recorded audio itself is not boosted"


def test_empty_callback_is_ignored(sd):
    rec = Recorder()
    rec._callback(np.zeros((0, 1), dtype=np.float32), 0, None, None)
    assert rec.level() == 0.0


# ---------------------------------------------------------------- whisper-mode gain
def speech(level, seconds=1.0, pause_ratio=0.4, seed=0):
    """Noise bursts with pauses: speech-like dynamics at a given RMS level."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    x = rng.normal(0, level, n).astype(np.float32)
    frame = 320
    for i in range(0, n // frame):
        if (i // 8) % 5 < int(5 * pause_ratio):
            x[i * frame:(i + 1) * frame] *= 0.02
    return x


def rms_of_loud_frames(x):
    frames = x[:x.size // 320 * 320].reshape(-1, 320)
    return float(np.percentile(np.sqrt((frames ** 2).mean(axis=1)), 90))


def test_whispered_speech_is_brought_up_to_the_target_level():
    quiet = speech(0.004)
    out = normalize_gain(quiet)
    assert out.dtype == np.float32 and out.shape == quiet.shape
    assert rms_of_loud_frames(out) == pytest.approx(0.07, rel=0.15)
    assert np.max(np.abs(out)) <= 1.0
    assert np.corrcoef(quiet, out)[0, 1] > 0.99, "pure gain: the waveform shape is preserved"


def test_normal_and_loud_speech_is_never_attenuated_or_changed():
    for level in (0.07, 0.2, 0.5):
        x = speech(level)
        assert np.array_equal(normalize_gain(x), x)


def test_gain_is_capped_and_peaks_are_soft_limited_not_clipped():
    barely = speech(0.0001)
    out = normalize_gain(barely)
    assert rms_of_loud_frames(out) == pytest.approx(rms_of_loud_frames(barely) * 40.0, rel=0.05), "max_gain = 40"
    spiky = speech(0.004)
    spiky[5000] = 0.5                                      # a click: 0.5 * ~17 would be far beyond full scale
    limited = normalize_gain(spiky)
    assert 0.8 < abs(float(limited[5000])) <= 1.0
    assert np.all(np.abs(limited) <= 1.0)
    custom = normalize_gain(speech(0.004), target_rms=0.02, max_gain=2.0)
    assert rms_of_loud_frames(custom) == pytest.approx(0.008, rel=0.1)


@pytest.mark.parametrize("make", [
    lambda: np.zeros(SAMPLE_RATE, dtype=np.float32),                                   # digital silence
    lambda: np.random.default_rng(1).normal(0, 0.003, SAMPLE_RATE).astype(np.float32),  # steady hiss, no speech
    lambda: np.zeros(0, dtype=np.float32),
    lambda: speech(0.004)[:1000],                                                      # too short to judge
])
def test_silence_hiss_and_tiny_clips_are_left_alone(make):
    x = make()
    assert np.array_equal(normalize_gain(x), x)


def test_normalize_gain_accepts_lists_and_2d_input_without_modifying_it():
    x = speech(0.004)
    original = x.copy()
    out = normalize_gain(x.reshape(-1, 1).astype(np.float64))
    assert out.ndim == 1 and out.dtype == np.float32
    assert np.array_equal(x, original), "the caller's buffer must not be modified in place"
    assert normalize_gain([0.0] * 10).size == 10
