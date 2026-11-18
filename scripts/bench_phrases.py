"""Benchmark of phrase streaming (voxflow.phrases) with the real speech model, in simulated real time.

A WAV is "spoken" at the speed of the clock into a real ``Previewer`` whose ``on_words`` feeds a real ``PhraseStream``.
Every phrase is recorded with the wall time it was emitted. At "release" both are stopped, the audio after the last
phrase's cut goes through the model (the tail) and the whole audio goes through it (today's final pass).
Metrics (plan/PHRASE_STREAMING_DESIGN.md 10.5):

    phrases / minute, phrase lag (emission wall time minus the start of the phrase's last word), boundaries rejected
    for "no room" (hi - lo < MIN_GAP_S), word-level similarity of phrases + tail with the full pass, words lost /
    extra / replaced within +-3 words of a cut (an alignment, difflib opcodes), release-to-tail vs release-to-full-pass
    (median of --runs), and the unchanged preview numbers of scripts/bench_stream.py (duty, caption lag, retractions).

    python scripts/bench_phrases.py [--device auto|cpu|cuda|webgpu] [--seconds 20] [--runs 5] [--gap-s 0.4]
                                    [--slowdown 1] [--wav tests/tts_sample.wav] [--json out.json]
                                    [--pause-s N] [--sent-pause-s N] [--min-words N] [--end-base-s N]
                                    [--end-per-char-s N] [--min-gap-s N] [--cut-margin-s N] [--quiet]
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from voxflow import phrases  # noqa: E402
from voxflow.asr import SAMPLE_RATE, Transcriber  # noqa: E402
from voxflow.stream import Previewer, norm  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_stream", os.path.join(HERE, "bench_stream.py"))
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

NEAR_CUT = 3          # words either side of a cut that count as "at the cut"


def align(full: list[str], streamed: list[str], boundaries: list[int]) -> dict:
    """Words of the full pass missing from the streamed text (lost), words only in the streamed text (extra: a word
    repeated or invented) and words that differ (replaced), counted near a cut and elsewhere. ``boundaries``:
    positions in ``streamed`` where a cut lies."""
    a, b = [norm(w) for w in full], [norm(w) for w in streamed]
    out = {"lost_at_cuts": 0, "extra_at_cuts": 0, "replaced_at_cuts": 0,
           "lost_elsewhere": 0, "extra_elsewhere": 0, "replaced_elsewhere": 0}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        near = any(j1 - NEAR_CUT <= c <= j2 + NEAR_CUT for c in boundaries)
        where = "at_cuts" if near else "elsewhere"
        if tag == "delete":
            out[f"lost_{where}"] += i2 - i1
        elif tag == "insert":
            out[f"extra_{where}"] += j2 - j1
        else:
            out[f"replaced_{where}"] += max(i2 - i1, j2 - j1)
    return out


def play(tr: Transcriber, audio: np.ndarray, *, slowdown: float, interval_s: float, trim_s, splitter) -> dict:
    """One utterance with phrase streaming: speak in real time, release, run the tail and the full pass."""
    probe = bs.Probe(tr, slowdown)
    rec = bs.SimRecorder(audio)
    probe.rec = rec
    shown: list[dict] = []
    emitted: list[dict] = []

    def on_partial(gen, committed, tentative):
        shown.append({"t": time.perf_counter(), "committed": committed, "tentative": tentative, "n_calls": len(probe.calls)})

    def on_phrase(gen, p):
        emitted.append({"t": time.perf_counter(), "phrase": p})

    ps = phrases.PhraseStream(rec, on_phrase, splitter=splitter)
    p = Previewer(rec, probe, on_partial, interval_s=interval_s, trim_s=trim_s, on_words=ps.feed)
    ps.begin(1)
    p.start(1)
    while not rec.done():
        time.sleep(0.005)
    t_release = time.perf_counter()
    p.stop()
    ps.end()
    cut = emitted[-1]["phrase"].cut_s if emitted else 0.0
    tail = probe.transcribe(audio[min(audio.size, int(cut * SAMPLE_RATE)):]) if cut < audio.size / SAMPLE_RATE else ""
    release_to_tail = time.perf_counter() - t_release
    t = time.perf_counter()
    final = probe.transcribe(audio)
    full_s = time.perf_counter() - t
    p.close()
    return {"rec": rec, "shown": shown, "calls": list(probe.calls), "final": final, "release_to_final_s": full_s,
            "available": p.available, "trim_s": p._trim_s, "emitted": emitted, "tail": tail,
            "release_to_tail_s": release_to_tail, "cut": cut,
            "rejected": splitter.rejected_final + len(splitter.rejected_at)}


def metrics(r: dict, audio: np.ndarray) -> dict:
    rec, emitted = r["rec"], r["emitted"]
    m = bs.metrics(r, audio)                         # the preview numbers, unchanged
    seconds = audio.size / SAMPLE_RATE
    streamed: list[str] = []
    boundaries: list[int] = []
    for e in emitted:
        streamed += e["phrase"].text.split()
        boundaries.append(len(streamed))
    streamed += r["tail"].split()
    full = r["final"].split()
    lags = [e["t"] - rec.t0 - e["phrase"].words[-1][1] for e in emitted]
    m.update(align(full, streamed, boundaries))
    m.update({
        "phrases": len(emitted), "phrases_per_min": len(emitted) / seconds * 60.0,
        "phrase_lag_median_s": statistics.median(lags) if lags else None,
        "phrase_lag_max_s": max(lags) if lags else None, "phrase_lags": lags,
        "kinds": {k: sum(1 for e in emitted if e["phrase"].kind == k) for k in ("sentence", "pause", "cap")},
        "rejected_no_room": r["rejected"],
        "stream_similarity": bs.similarity(streamed, full),
        "release_to_tail_ms": r["release_to_tail_s"] * 1000, "last_cut_s": r["cut"],
        "tail_words": len(r["tail"].split()), "total_words": len(full),
        "cuts_monotone": all(a["phrase"].cut_s <= b["phrase"].cut_s for a, b in zip(emitted, emitted[1:])),
        "texts": [e["phrase"].text for e in emitted] + ["| tail: " + r["tail"]],
    })
    return m


def bench(tr: Transcriber, audio: np.ndarray, *, runs: int = 5, slowdown: float = 1.0, interval_s: float = 0.4,
          trim_s=None, splitter_args=None, quiet: bool = False) -> dict:
    """Median over ``runs`` utterances, each of a length the model has not seen."""
    for _ in range(2):
        tr.transcribe(audio)                                       # warm-up
    per_run = []
    for i in range(runs):
        a = audio[: audio.size - 797 * (i + 1)]
        sp = phrases.PhraseSplitter(**(splitter_args or {}))
        m = metrics(play(tr, a, slowdown=slowdown, interval_s=interval_s, trim_s=trim_s, splitter=sp), a)
        per_run.append(m)
        if not quiet:
            print(f"  run {i + 1}: {m['phrases']} phrases {m['kinds']}, rejected {m['rejected_no_room']}, lag "
                  f"{m['phrase_lag_median_s']}, similarity {m['stream_similarity']:.3f}, at cuts lost "
                  f"{m['lost_at_cuts']} extra {m['extra_at_cuts']} replaced {m['replaced_at_cuts']} (elsewhere "
                  f"{m['lost_elsewhere']}/{m['extra_elsewhere']}/{m['replaced_elsewhere']}), tail "
                  f"{m['release_to_tail_ms']:.0f} ms vs full {m['release_to_final_ms']:.0f} ms", flush=True)
    med = lambda k: statistics.median([m[k] for m in per_run if m[k] is not None]) if any(  # noqa: E731
        m[k] is not None for m in per_run) else None
    lags = sorted(x for m in per_run for x in m["phrase_lags"])
    return {
        "backend": tr.backend, "slowdown": slowdown, "seconds": audio.size / SAMPLE_RATE, "runs": runs,
        "phrases_median": med("phrases"), "phrases_per_min": med("phrases_per_min"),
        "kinds": {k: sum(m["kinds"][k] for m in per_run) for k in ("sentence", "pause", "cap")},
        "rejected_no_room_total": sum(m["rejected_no_room"] for m in per_run),
        "phrase_lag_median_s": statistics.median(lags) if lags else None,
        "phrase_lag_p95_s": lags[min(len(lags) - 1, int(0.95 * len(lags)))] if lags else None,
        "phrase_lag_max_s": max(lags) if lags else None,
        "stream_similarity_median": med("stream_similarity"),
        "stream_similarity_min": min(m["stream_similarity"] for m in per_run),
        "lost_at_cuts": sum(m["lost_at_cuts"] for m in per_run), "extra_at_cuts": sum(m["extra_at_cuts"] for m in per_run),
        "replaced_at_cuts": sum(m["replaced_at_cuts"] for m in per_run),
        "lost_elsewhere": sum(m["lost_elsewhere"] for m in per_run), "extra_elsewhere": sum(m["extra_elsewhere"] for m in per_run),
        "replaced_elsewhere": sum(m["replaced_elsewhere"] for m in per_run),
        "worst_at_cuts_per_run": max(m["lost_at_cuts"] + m["extra_at_cuts"] for m in per_run),
        "total_words_median": med("total_words"),
        "release_to_tail_median_ms": med("release_to_tail_ms"), "release_to_final_median_ms": med("release_to_final_ms"),
        "tail_words_median": med("tail_words"),
        "preview_duty": med("duty"), "preview_lag_median_s": med("lag_median_s"), "retracted": sum(m["retracted"] for m in per_run),
        "available_at_end": all(m["available_at_end"] for m in per_run),
        "cuts_monotone": all(m["cuts_monotone"] for m in per_run),
        "example_texts": per_run[0]["texts"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", default=os.path.join(HERE, "..", "tests", "tts_sample.wav"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--gap-s", type=float, default=0.4, help="silence between two repetitions of the clip")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--interval-s", type=float, default=0.4)
    ap.add_argument("--trim-s", type=float, default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--quiet", action="store_true")
    for name, mod in (("pause-s", "PAUSE_S"), ("sent-pause-s", "SENT_PAUSE_S"), ("min-words", "MIN_WORDS"),
                      ("end-base-s", "END_BASE_S"), ("end-per-char-s", "END_PER_CHAR_S"), ("min-gap-s", "MIN_GAP_S"),
                      ("cut-margin-s", "CUT_MARGIN_S")):
        ap.add_argument("--" + name, type=float, default=None, help=f"override phrases.{mod}")
    a = ap.parse_args()
    splitter_args = {}
    for name, key in (("pause_s", "pause_s"), ("sent_pause_s", "sent_pause_s")):
        if getattr(a, name) is not None:
            splitter_args[key] = getattr(a, name)
    if a.min_words is not None:
        splitter_args["min_words"] = int(a.min_words)
    for name, mod in (("end_base_s", "END_BASE_S"), ("end_per_char_s", "END_PER_CHAR_S"), ("min_gap_s", "MIN_GAP_S"),
                      ("cut_margin_s", "CUT_MARGIN_S")):
        if getattr(a, name) is not None:
            setattr(phrases, mod, getattr(a, name))
    audio = bs.make_audio(a.wav, a.seconds, 0.0, a.gap_s)
    tr = Transcriber(device=a.device)
    t = time.perf_counter()
    tr.load()
    print(f"backend {tr.backend}, loaded in {time.perf_counter() - t:.1f} s; audio {audio.size / SAMPLE_RATE:.1f} s, gap "
          f"{a.gap_s:g} s, slowdown x{a.slowdown:g}, {a.runs} runs, splitter {splitter_args or 'defaults'} "
          f"(real time: about {a.runs * audio.size / SAMPLE_RATE:.0f} s)")
    res = bench(tr, audio, runs=a.runs, slowdown=a.slowdown, interval_s=a.interval_s, trim_s=a.trim_s,
                splitter_args=splitter_args, quiet=a.quiet)
    res["gap_s"] = a.gap_s
    res["splitter_args"] = splitter_args
    print("\n" + json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
