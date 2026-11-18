"""Benchmark of the live preview (voxflow.stream.Previewer) with the real speech model, in simulated real time.

A WAV is "spoken" at the speed of the clock: the preview thread reads the growing recording through the same
``recorded_samples`` / ``snapshot`` interface as ``voxflow.audio.Recorder`` and re-transcribes it through the real
``voxflow.asr.Transcriber``. At "release" the preview is stopped and today's final pass runs on the whole recording.
Prints, and returns, the metrics the design asks for (plan/STREAMING_DESIGN.md section 6-D):

    first words shown after the speech starts; median caption lag (wall clock minus the audio time of the newest
    shown word); duty (model time of the preview / speaking time); release-to-final with the preview running and
    without it (median of --runs, each on a length the model has not seen); committed-vs-final similarity; whether a
    committed word was ever taken back; what the governor did.

    python scripts/bench_stream.py [--device auto|cpu|cuda|webgpu] [--seconds 20] [--runs 5] [--slowdown 1]
                                   [--wav tests/tts_sample.wav] [--trim-s N] [--lead-noise-s N] [--json out.json]

``--slowdown N`` makes every model call (preview and final) take N times longer, to imitate a weak CPU.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxflow.app import read_wav_16k  # noqa: E402
from voxflow.asr import SAMPLE_RATE, Transcriber  # noqa: E402
from voxflow.stream import Previewer, norm  # noqa: E402


class SimRecorder:
    """The part of ``Recorder`` the preview uses, for a WAV that is played at the speed of the clock."""

    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.last_start = 0
        self.t0 = time.perf_counter()

    @property
    def recorded_samples(self) -> int:
        return min(self.audio.size, int((time.perf_counter() - self.t0) * SAMPLE_RATE))

    def snapshot(self, start: int = 0) -> np.ndarray:
        self.last_start = start                      # where the window the preview is about to transcribe begins
        return self.audio[start:self.recorded_samples].copy()

    def done(self) -> bool:
        return self.recorded_samples >= self.audio.size


class Probe:
    """Wraps a Transcriber: optional slow-down, and a record of every preview call for the lag measurement."""

    def __init__(self, tr: Transcriber, slowdown: float = 1.0):
        self.tr, self.slowdown = tr, slowdown
        self.calls: list[dict] = []
        self.rec: SimRecorder | None = None

    @property
    def backend(self) -> str:
        return self.tr.backend

    def _slow(self, started: float) -> None:
        if self.slowdown > 1.0:
            time.sleep((time.perf_counter() - started) * (self.slowdown - 1.0))

    def transcribe_timed(self, audio, *, bucket_s=0.0, blocking=False):
        t = time.perf_counter()
        words = self.tr.transcribe_timed(audio, bucket_s=bucket_s, blocking=blocking)
        if words is not None:
            self._slow(t)
            self.calls.append({"end": time.perf_counter(), "audio_s": audio.size / SAMPLE_RATE,
                               "cost": time.perf_counter() - t,
                               "last_word_s": self.rec.last_start / SAMPLE_RATE + words[-1][1] if words else None})
        return words

    def transcribe(self, audio):
        t = time.perf_counter()
        text = self.tr.transcribe(audio)
        self._slow(t)
        return text


def speech_start_s(audio: np.ndarray) -> float:
    loud = np.flatnonzero(np.abs(audio) > 0.02)
    return float(loud[0]) / SAMPLE_RATE if loud.size else 0.0


def similarity(a: list[str], b: list[str]) -> float:
    return difflib.SequenceMatcher(None, [norm(w) for w in a], [norm(w) for w in b]).ratio()


def play(probe: Probe, audio: np.ndarray, *, interval_s: float, trim_s, preview: bool) -> dict:
    """One utterance: speak ``audio`` in real time, release, run the final pass. Returns the raw measurements."""
    rec = SimRecorder(audio)
    probe.rec, probe.calls = rec, []
    shown: list[dict] = []                        # every caption update: wall time, committed, tentative

    def on_partial(gen, committed, tentative):
        shown.append({"t": time.perf_counter(), "committed": committed, "tentative": tentative,
                      "n_calls": len(probe.calls)})

    p = Previewer(rec, probe, on_partial, interval_s=interval_s, trim_s=trim_s)
    if preview:
        p.start(1)
    while not rec.done():
        time.sleep(0.005)
    t_release = time.perf_counter()
    if preview:
        p.stop()                                  # as the controller does, before the recorder stops
    final = probe.transcribe(audio)
    release_to_final = time.perf_counter() - t_release
    p.close()
    return {"rec": rec, "shown": shown, "calls": list(probe.calls), "final": final, "release_to_final_s": release_to_final,
            "available": p.available, "trim_s": p._trim_s}


def metrics(r: dict, audio: np.ndarray) -> dict:
    rec, shown, calls = r["rec"], r["shown"], r["calls"]
    start = speech_start_s(audio)
    speaking = audio.size / SAMPLE_RATE
    first = next((s for s in shown if s["committed"] or s["tentative"]), None)
    # lag: when a call ended, how far the speaker had got beyond the start of the newest word that call returned
    lags = [(c["end"] - rec.t0) - c["last_word_s"] for c in calls if c["last_word_s"] is not None]
    # the caption cleared when the governor gives up is not a retraction and not the text that was last shown
    texts = [s for s in shown if s["committed"] or s["tentative"]]
    committed_words = [s["committed"].split() for s in texts]
    retracted = sum(1 for a, b in zip(committed_words, committed_words[1:]) if b[:len(a)] != a)
    last = texts[-1] if texts else {"committed": "", "tentative": ""}
    final_words = r["final"].split()
    shown_words = (last["committed"] + " " + last["tentative"]).split()
    com = last["committed"].split()
    return {
        "first_words_s": None if first is None else first["t"] - rec.t0 - start,
        "lag_median_s": statistics.median(lags) if lags else None,
        "lag_max_s": max(lags) if lags else None,
        "duty": sum(c["cost"] for c in calls) / speaking,
        "calls": len(calls), "max_window_s": max([c["audio_s"] for c in calls], default=0.0),
        "cost_median_ms": 1000 * statistics.median([c["cost"] for c in calls]) if calls else 0.0,
        "cost_max_ms": 1000 * max([c["cost"] for c in calls], default=0.0),
        "similarity": similarity(shown_words, final_words),
        "committed_match": similarity(com, final_words[:len(com)]) if com else 1.0,
        "retracted": retracted, "updates": len(shown),
        "release_to_final_ms": r["release_to_final_s"] * 1000,
        "available_at_end": r["available"], "trim_s_at_end": r["trim_s"],
    }


def make_audio(wav: str, seconds: float, lead_noise_s: float = 0.0) -> np.ndarray:
    """``seconds`` of the clip looped, after ``lead_noise_s`` of steady low noise (peak 0.01: above the preview's
    silence gate, below speech) that nobody speaks over: the case that used to make the window grow."""
    clip = read_wav_16k(wav)
    gap = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    parts, n = [], 0
    while n < seconds * SAMPLE_RATE:
        parts += [clip, gap]
        n += clip.size + gap.size
    speech = np.concatenate(parts)[: int(seconds * SAMPLE_RATE)]
    noise = np.random.default_rng(0).uniform(-0.01, 0.01, int(lead_noise_s * SAMPLE_RATE)).astype(np.float32)
    return np.concatenate([noise, speech]).astype(np.float32)


def bench(tr: Transcriber, audio: np.ndarray, *, runs: int = 5, slowdown: float = 1.0, interval_s: float = 0.4,
          trim_s=None, quiet: bool = False) -> dict:
    """Median over ``runs`` utterances, each of a length the model has not seen, with the preview and without it."""
    probe = Probe(tr, slowdown)
    for _ in range(2):
        tr.transcribe(audio)                                       # warm-up
    base, with_p, per_run = [], [], []
    for i in range(runs):
        a = audio[: audio.size - 797 * (i + 1)]                    # a new length each time, like a real dictation
        t = time.perf_counter()
        probe.transcribe(a)
        base.append((time.perf_counter() - t) * 1000)
        r = play(probe, a, interval_s=interval_s, trim_s=trim_s, preview=True)
        m = metrics(r, a)
        with_p.append(m["release_to_final_ms"])
        per_run.append(m)
        if not quiet:
            print(f"  run {i + 1}: first words {m['first_words_s']}, lag {m['lag_median_s']}, duty {m['duty']:.1%}, "
                  f"calls {m['calls']} (max {m['cost_max_ms']:.0f} ms), final {m['release_to_final_ms']:.0f} ms vs "
                  f"{base[-1]:.0f} ms without, similarity {m['similarity']:.3f}", flush=True)
    med = lambda k: statistics.median([m[k] for m in per_run if m[k] is not None]) if any(  # noqa: E731
        m[k] is not None for m in per_run) else None
    return {"backend": tr.backend, "slowdown": slowdown, "seconds": audio.size / SAMPLE_RATE, "runs": runs,
            "first_words_s": med("first_words_s"), "lag_median_s": med("lag_median_s"), "duty": med("duty"),
            "calls": med("calls"), "max_window_s": max(m["max_window_s"] for m in per_run), "cost_max_ms": max(m["cost_max_ms"] for m in per_run),
            "similarity": med("similarity"), "committed_match": med("committed_match"),
            "retracted": sum(m["retracted"] for m in per_run),
            "release_to_final_with_ms": statistics.median(with_p), "release_to_final_without_ms": statistics.median(base),
            "regression_ms": statistics.median(with_p) - statistics.median(base),
            "available_at_end": all(m["available_at_end"] for m in per_run),
            "trim_s_at_end": min(m["trim_s_at_end"] for m in per_run)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", default=os.path.join(os.path.dirname(__file__), "..", "tests", "tts_sample.wav"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--lead-noise-s", type=float, default=0.0, help="seconds of low noise (no speech) before the speech")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--interval-s", type=float, default=0.4)
    ap.add_argument("--trim-s", type=float, default=None)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    audio = make_audio(a.wav, a.seconds, a.lead_noise_s)
    tr = Transcriber(device=a.device)
    t = time.perf_counter()
    tr.load()
    print(f"backend {tr.backend}, loaded in {time.perf_counter() - t:.1f} s; audio {audio.size / SAMPLE_RATE:.1f} s, "
          f"slowdown x{a.slowdown:g}, {a.runs} runs (real time: about {a.runs * audio.size / SAMPLE_RATE:.0f} s)")
    res = bench(tr, audio, runs=a.runs, slowdown=a.slowdown, interval_s=a.interval_s, trim_s=a.trim_s)
    print("\n" + json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
