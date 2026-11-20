import numpy as np
import pytest

sd = pytest.importorskip("sounddevice")

from voxflow.audio import SAMPLE_RATE, Recorder  # noqa: E402


class FakeStream:
    active = True

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def rec(monkeypatch):
    monkeypatch.setattr(sd, "InputStream", lambda **kw: FakeStream())
    return Recorder()


def feed(rec, x):
    rec._callback(x.reshape(-1, 1).astype(np.float32), len(x), None, None)


def test_contract(rec):
    assert SAMPLE_RATE == 16000
    assert not rec.is_recording
    assert rec.level() == 0.0
    assert rec.recent_samples(100).shape == (100,)


def test_record_returns_preroll_and_samples(rec):
    feed(rec, np.full(8000, 0.1, np.float32))  # before start -> pre-roll
    rec.start()
    assert rec.is_recording
    feed(rec, np.full(1600, 0.2, np.float32))
    out = rec.stop()
    assert not rec.is_recording
    assert out.dtype == np.float32 and out.ndim == 1
    assert out.size == int(0.3 * SAMPLE_RATE) + 1600
    assert np.isclose(out[0], 0.1) and np.isclose(out[-1], 0.2)


def test_stop_without_start_is_empty(rec):
    assert rec.stop().size == 0


def test_level_and_recent(rec):
    feed(rec, np.full(512, 0.05, np.float32))
    assert 0.0 < rec.level() <= 1.0
    feed(rec, np.full(512, 2.0, np.float32))
    assert rec.level() == 1.0
    r = rec.recent_samples(512)
    assert r.size == 512 and np.allclose(r, 2.0)
    assert rec.recent_samples(10**7).size == rec._ring.size


def test_clipping(rec):
    rec.start()
    feed(rec, np.full(100, 3.0, np.float32))
    assert rec.stop().max() <= 1.0


@pytest.mark.desktop
def test_real_microphone_capture():
    try:
        devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception:
        devs = []
    if not devs:
        pytest.skip("no input device")
    import time
    r = Recorder()
    try:
        r.start()
        time.sleep(0.6)
        audio = r.stop()
    finally:
        r.close()
    assert audio.dtype == np.float32
    assert audio.size > 0.5 * SAMPLE_RATE


def test_last_speech_seconds_excludes_preroll(rec):
    from voxflow.audio import PREROLL_S
    assert PREROLL_S == 0.3
    rec.start()
    assert rec.stop().size == int(PREROLL_S * SAMPLE_RATE)
    assert rec.last_speech_seconds == 0.0  # instant tap
    rec.start()
    feed(rec, np.full(8000, 0.1, np.float32))
    rec.stop()
    assert abs(rec.last_speech_seconds - 0.5) < 1e-6


def test_max_seconds_cap(monkeypatch):
    monkeypatch.setattr(sd, "InputStream", lambda **kw: FakeStream())
    r = Recorder(max_seconds=1.0)
    r.start()
    for _ in range(10):
        feed(r, np.full(4000, 0.1, np.float32))
    out = r.stop()
    assert out.size == SAMPLE_RATE
    assert r.is_recording is False


def test_hit_cap_flag(monkeypatch):
    monkeypatch.setattr(sd, "InputStream", lambda **kw: FakeStream())
    r = Recorder(max_seconds=1.0)
    assert r.hit_cap is False
    r.start()
    feed(r, np.full(4000, 0.1, np.float32))
    assert r.hit_cap is False
    for _ in range(4):
        feed(r, np.full(4000, 0.1, np.float32))
    assert r.hit_cap is True
    r.stop()
    assert r.hit_cap is True  # stays readable after stop
    r.start()
    assert r.hit_cap is False


def test_close_twice_and_set_device(monkeypatch):
    made = []

    class S(FakeStream):
        closed = 0

        def close(self):
            S.closed += 1
            self.active = False

    def mk(**kw):
        made.append(S())
        return made[-1]

    monkeypatch.setattr(sd, "InputStream", mk)
    r = Recorder()
    r.close()
    r.close()
    assert S.closed == 1
    r.set_device(3)
    r.set_device(4)
    assert len(made) == 3 and S.closed == 2 and made[-1].active


# ---------------------------------------------------------------- release tail (QA C-01: "it cuts off my last word")
def test_stop_keeps_the_audio_that_arrives_just_after_the_release(rec):
    """What was said right before the key came up reaches the callback a few tens of ms after it."""
    import threading
    import time
    from voxflow import audio
    assert rec.release_tail_s == audio.RELEASE_TAIL_S == 0.08
    rec.start()
    feed(rec, np.full(512, 0.2, np.float32))
    late = threading.Timer(0.03, lambda: feed(rec, np.full(512, 0.5, np.float32)))     # the block still on its way
    later = threading.Timer(0.06, lambda: feed(rec, np.full(2048, 0.9, np.float32)))   # reaches past the tail
    late.start(), later.start()
    t0 = time.perf_counter()
    out = rec.stop()
    waited = time.perf_counter() - t0
    late.join(), later.join()
    pre, tail = int(0.3 * SAMPLE_RATE), int(0.08 * SAMPLE_RATE)
    assert out.size == pre + 512 + tail, "exactly the tail, cut inside the block that reached past it"
    assert np.isclose(out[pre + 512], 0.5) and np.isclose(out[-1], 0.9)
    assert waited < 0.2 and not rec.is_recording
    assert np.isclose(rec.last_speech_seconds, (512 + tail) / SAMPLE_RATE)


def test_stop_does_not_wait_for_a_microphone_that_delivers_nothing(rec):
    import time
    rec.start()
    feed(rec, np.full(512, 0.2, np.float32))
    rec._last_cb -= 1.0                                         # the last block came a second ago: the stream is dead
    t0 = time.perf_counter()
    out = rec.stop()
    assert time.perf_counter() - t0 < 0.03 and out.size == int(0.3 * SAMPLE_RATE) + 512


def test_stop_gives_up_when_the_tail_never_fills_and_at_the_cap(rec, monkeypatch):
    import time
    rec.start()
    feed(rec, np.full(512, 0.2, np.float32))
    t0 = time.perf_counter()
    assert rec.stop().size == int(0.3 * SAMPLE_RATE) + 512     # nothing more arrives: the wait has an end
    assert 0.08 <= time.perf_counter() - t0 < 0.3
    rec._max_samples = int(0.3 * SAMPLE_RATE) + 512             # the recording is full: no tail to wait for
    rec.start()
    feed(rec, np.full(512, 0.2, np.float32))
    t0 = time.perf_counter()
    assert rec.stop().size == rec._max_samples and time.perf_counter() - t0 < 0.03


def test_discard_ends_a_recording_at_once_and_keeps_nothing(rec):
    import time
    rec.start()
    feed(rec, np.full(512, 0.2, np.float32))
    t0 = time.perf_counter()
    rec.discard()
    assert time.perf_counter() - t0 < 0.03 and not rec.is_recording
    assert rec.stop().size == 0 and rec.last_speech_seconds == 0.0


# ---------------------------------------------------------------- a start that came late (QA F-03)
def test_backdate_puts_the_audio_since_the_key_went_down_in_front(rec):
    feed(rec, np.full(8000, 0.1, np.float32))                   # long before the key
    feed(rec, np.full(4800, 0.3, np.float32))                   # the 0.3 s right before the key went down
    feed(rec, np.full(8000, 0.6, np.float32))                   # said while the start was held up (0.5 s)
    rec.release_tail_s = 0.0
    rec.start()                                                 # only now: the pre-roll is the END of that speech
    feed(rec, np.full(512, 0.9, np.float32))
    assert rec.backdate(0.5) == 8000
    out = rec.stop()
    assert out.size == 4800 + 8000 + 512
    assert np.allclose(out[:4800], 0.3) and np.allclose(out[4800:12800], 0.6) and np.isclose(out[-1], 0.9)
    assert np.isclose(rec.last_speech_seconds, (8000 + 512) / SAMPLE_RATE), "what was said counts as speech"


def test_backdate_is_limited_to_what_the_ring_holds_and_does_nothing_when_idle(rec):
    assert rec.backdate(1.0) == 0                               # not recording
    rec.release_tail_s = 0.0
    rec.start()
    room = rec._ring.size - int(0.3 * SAMPLE_RATE)
    assert rec.backdate(60.0) == room and rec.backdate(1.0) == 0 and rec.backdate(-1.0) == 0
    assert rec.stop().size == rec._ring.size
