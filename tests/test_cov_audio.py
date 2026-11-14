"""voxflow.audio: remaining edge cases of the ring buffer and the wave-in device probe."""
from __future__ import annotations

import sys

import numpy as np
import pytest

sd = pytest.importorskip("sounddevice")

from voxflow import audio  # noqa: E402
from voxflow.audio import Recorder  # noqa: E402


class _Stream:
    active = True

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def test_a_block_longer_than_the_ring_keeps_only_its_newest_samples(monkeypatch):
    monkeypatch.setattr(sd, "InputStream", lambda **kw: _Stream())
    rec = Recorder()
    size = rec._ring.size
    block = np.linspace(-0.5, 0.5, size + 1000, dtype=np.float32)
    rec._callback(block.reshape(-1, 1), block.size, None, None)
    assert np.array_equal(rec._ring, block[-size:])
    assert np.array_equal(rec.recent_samples(16), block[-16:])
    # a block of exactly the ring size replaces it completely
    exact = np.full(size, 0.25, dtype=np.float32)
    rec._callback(exact.reshape(-1, 1), size, None, None)
    assert np.array_equal(rec._ring, exact)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_input_device_count_is_minus_one_when_winmm_cannot_be_asked(monkeypatch):
    import ctypes

    class _NoWinmm:
        @property
        def winmm(self):
            raise OSError("winmm.dll not available")

    monkeypatch.setattr(ctypes, "windll", _NoWinmm())
    assert audio.input_device_count() == -1
