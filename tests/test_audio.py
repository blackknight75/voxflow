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
