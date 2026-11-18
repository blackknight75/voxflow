"""Phrase streaming with the real speech model, in simulated real time (scripts/bench_phrases.py).
Thresholds: plan/PHRASE_STREAMING_DESIGN.md 10.5, checked against the numbers in plan/report-pE.md. A 20 s clip is
spoken at the speed of the clock: about 30 s of wall time per device."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("onnx_asr")
pytestmark = pytest.mark.model

from voxflow.asr import Transcriber  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_phrases", Path(__file__).resolve().parent.parent / "scripts"
                                               / "bench_phrases.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)
WAV = str(Path(__file__).parent / "tts_sample.wav")


def _devices():
    out = ["cpu"]
    try:
        import onnxruntime
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            out.insert(0, "cuda")
    except Exception:  # noqa: BLE001 - no onnxruntime: CPU only
        pass
    return out


@pytest.fixture(scope="module", params=_devices())
def tr(request):
    t = Transcriber(device=request.param)
    t.load()
    return t


# (gap between repetitions, noise peak in the gaps, thinking pauses, seconds): the configs of plan/report-pE.md
CLIPS = [pytest.param(0.4, 0.0, False, 20.0, id="fluent"), pytest.param(0.9, 0.0, False, 20.0, id="paused"),
         pytest.param(0.4, 0.004, False, 20.0, id="noisy-gaps"), pytest.param(0.4, 0.0, True, 30.0, id="thinking")]


@pytest.mark.parametrize("gap_s, noise, thinking, seconds", CLIPS)
def test_phrases_lose_and_repeat_nothing_and_arrive_in_time(tr, gap_s, noise, thinking, seconds):
    gpu = tr.backend in ("cuda", "webgpu")
    audio = bp.make_clip(WAV, seconds, gap_s, noise, thinking)
    r = bp.bench(tr, audio, runs=1, quiet=True)
    print("MODEL", tr.backend, gap_s, noise, thinking, {k: r[k] for k in (
        "phrases_median", "phrase_lag_median_s", "phrase_lag_p95_s", "stream_similarity_median", "lost_at_cuts",
        "extra_at_cuts", "runs_stopped", "lost_elsewhere", "extra_elsewhere", "replaced_elsewhere", "release_to_tail_median_ms",
        "release_to_final_median_ms")})
    if r["stream_similarity_median"] < 0.88:
        print("TEXTS", *r["example_texts"], "FULL", r["example_full_text"], sep=chr(10))
    # Thresholds (plan/report-pE.md, Round 3). Correctness is asserted as measured everywhere: nothing lost at a cut,
    # no retraction, similarity. Timing and counts depend on the PC. Measured 4 times under
    # `scripts/coverage.py -k test_phrases_model` (every test traced): GPU lag median 1.97-2.21 s, p95 2.1-2.8 s, phrases
    # 3 (6 for "thinking"); CPU lag median 2.12-2.49 s, p95 2.15-2.76 s, phrases 2-5, tail 102-483 ms vs full pass
    # 435-693 ms. The lead's full-suite gate measured on the CPU: lag median 3.65 s and phrases_median 1 (the rest of
    # the suite slows the previewer, which then falls behind, or stops after a gap, and the tail covers the rest).
    assert r["phrases_median"] >= (2 if gpu else 0)
    assert r["cuts_monotone"] and r["retracted"] == 0 and r["available_at_end"]
    # similarity measured 0.896-1.0 (the 20 s cut ends mid-word, "3.30" / "3 30" differ between passes)
    assert r["stream_similarity_median"] >= 0.88, (r["example_texts"], r["example_full_text"])
    assert r["lost_at_cuts"] == 0 and r["extra_at_cuts"] <= max(1, int(0.03 * r["total_words_median"])), r["example_texts"]
    if r["phrase_lag_median_s"] is not None:          # a stopped or slow session may have sent nothing
        assert r["phrase_lag_median_s"] <= (3.0 if gpu else 5.0)    # measured GPU 1.98-2.19 s, CPU 2.16-3.65 s
        assert r["phrase_lag_p95_s"] <= (4.5 if gpu else 8.0)       # measured GPU 2.1-2.8 s, CPU 2.2-4.7 s
    # the tail is shorter than the whole recording: measured GPU 49-130 ms vs 65-76 ms, CPU 97-415 ms vs 446-685 ms
    assert r["release_to_tail_median_ms"] <= max(2 * r["release_to_final_median_ms"], 500)


def test_a_cpu_four_times_too_slow_degrades_to_no_phrases_and_the_tail_covers_everything():
    t = Transcriber(device="cpu")
    t.load()
    audio = bp.make_clip(WAV, 4.0)
    r = bp.bench(t, audio, runs=1, slowdown=8.0, trim_s=4.0, quiet=True)
    assert not r["available_at_end"]
    assert r["stream_similarity_median"] >= 0.85
    assert r["lost_at_cuts"] + r["extra_at_cuts"] <= 1


def test_align_counts_lost_extra_and_replaced_words_near_the_cuts():
    full = "one two three four five six seven eight nine ten".split()
    streamed = "one two three four six six seven eight nine ten".split()
    assert bp.align(full, streamed, [4])["replaced_at_cuts"] == 1
    assert bp.align(full, "one two three five six seven eight nine ten".split(), [3])["lost_at_cuts"] == 1
    assert bp.align(full, "one two three four four five six seven eight nine ten".split(), [4])["extra_at_cuts"] == 1
    far = bp.align(full, "one two three four five six seven eight nine".split(), [0])
    assert far["lost_elsewhere"] == 1 and far["lost_at_cuts"] == 0
