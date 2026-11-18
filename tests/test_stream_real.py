"""The live preview with the real speech model on the CPU, in simulated real time (scripts/bench_stream.py).
Thresholds are those of plan/STREAMING_DESIGN.md section 6-D. A few seconds of speech: about 15 s of wall time."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx_asr")
pytestmark = pytest.mark.model

from voxflow.asr import SAMPLE_RATE, Transcriber  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_stream", Path(__file__).resolve().parent.parent / "scripts"
                                               / "bench_stream.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)
WAV = str(Path(__file__).parent / "tts_sample.wav")


@pytest.fixture(scope="module")
def tr():
    t = Transcriber(device="cpu")
    t.load()
    return t


def test_the_preview_is_quick_cheap_stable_and_does_not_slow_the_final_pass(tr):
    audio = bs.make_audio(WAV, 7.0)
    r = bs.bench(tr, audio, runs=1, quiet=True)
    assert r["first_words_s"] <= 1.5
    assert r["lag_median_s"] <= 1.5
    assert r["duty"] <= 0.35
    assert r["regression_ms"] <= 150
    assert r["similarity"] >= 0.8 and r["committed_match"] >= 0.8       # a 7 s clip ends mid-sentence
    assert r["retracted"] == 0
    assert r["available_at_end"]


def test_a_cpu_four_times_too_slow_gets_the_preview_switched_off_and_the_final_pass_is_unharmed(tr):
    audio = bs.make_audio(WAV, 4.0)
    r = bs.bench(tr, audio, runs=1, slowdown=8.0, trim_s=4.0, quiet=True)
    assert not r["available_at_end"]                                     # two calls over 400 ms at the window floor
    assert r["regression_ms"] <= 150
    assert r["trim_s_at_end"] == 4.0


def test_sim_recorder_plays_a_clip_at_the_speed_of_the_clock():
    rec = bs.SimRecorder(np.ones(SAMPLE_RATE * 10, np.float32))
    assert rec.recorded_samples < SAMPLE_RATE and not rec.done()
    assert rec.last_start == 0
    rec.t0 -= 60
    assert rec.done() and rec.snapshot(SAMPLE_RATE).size == 9 * SAMPLE_RATE and rec.last_start == SAMPLE_RATE
