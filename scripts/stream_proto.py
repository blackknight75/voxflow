"""Throwaway prototype: live partial transcripts by re-running the existing Parakeet model on a growing buffer.

Not used by the app. It answers one question with numbers: how often could VoxFlow update a live preview while the
user speaks, what does that cost, and does the final result (the one that is pasted) arrive any later than today?

How it works (LocalAgreement-2, as in whisper_streaming by Machacek et al., https://github.com/ufal/whisper_streaming):
a WAV file is "spoken" in simulated real time. Every ``--interval-ms`` a background thread re-transcribes the audio
heard so far (or, with ``--trim-s``, only the part after the last committed word) through the same
``voxflow.asr.Transcriber`` and lock the app uses. Words on which two consecutive hypotheses agree are committed
(they will not change in the preview); the rest is shown as tentative. At "release" the normal final pass runs on
the whole recording, exactly as today, and has to wait for any preview update that is still running.

    python scripts/stream_proto.py [--wav tests/tts_sample.wav] [--device cuda|webgpu|cpu|auto]
                                   [--interval-ms 500] [--repeat 1] [--trim-s 0] [--quiet]

``--repeat N`` concatenates the clip N times (0.4 s gaps) to simulate a long dictation, where the growing buffer
gets expensive. Output: one line per update, then a summary (update cost, duty cycle, release-to-final latency with
and without the preview loop, how close the committed preview was to the final text).
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxflow.app import read_wav_16k  # noqa: E402
from voxflow.asr import SAMPLE_RATE, Transcriber  # noqa: E402

PAD_S = 0.15                       # Transcriber.transcribe pads both ends by this much; the preview does the same


def norm(word: str) -> str:
    """Compare words without case or punctuation: Parakeet re-punctuates as more context arrives."""
    return re.sub(r"[^\w']+", "", word.lower())


def words_with_times(tokens, stamps) -> list[tuple[str, float]]:
    """Join sub-word tokens (a leading space starts a word) into (word, start time in s)."""
    out: list[list] = []
    for tok, ts in zip(tokens, stamps):
        if tok.startswith(" ") or not out:
            out.append([tok.strip(), float(ts)])
        else:
            out[-1][0] += tok
    return [(w, t) for w, t in out if w]


class LocalAgreement:
    """Commit the longest prefix on which the last two hypotheses agree (after what is already committed)."""

    def __init__(self) -> None:
        self.committed: list[tuple[str, float]] = []     # (word, absolute start time)
        self.prev: list[tuple[str, float]] = []

    def new_part(self, hyp: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """The words of a full hypothesis that come after the committed text. Timestamps move by a frame or two
        between passes, so a time cut alone repeats the boundary word: like whisper_streaming, words near the cut
        whose n-gram (n <= 5) equals the end of the committed text are dropped too."""
        if not self.committed:
            return hyp
        last_t = self.committed[-1][1]
        hyp = [(w, t) for w, t in hyp if t > last_t - 1.0]          # keep a margin around the cut
        tail = [norm(w) for w, _ in self.committed[-5:]]
        for n in range(min(5, len(hyp), len(tail)), 0, -1):
            for skip in range(0, min(3, len(hyp) - n) + 1):           # the n-gram may start a few words in
                if [norm(w) for w, _ in hyp[skip:skip + n]] == tail[-n:]:
                    return hyp[skip + n:]
        return [(w, t) for w, t in hyp if t > last_t + 0.08]

    def update(self, hyp: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """``hyp``: words of the newest hypothesis that start after the committed ones. Returns the tentative tail."""
        n = 0
        while n < len(hyp) and n < len(self.prev) and norm(hyp[n][0]) == norm(self.prev[n][0]):
            n += 1
        self.committed += hyp[:n]
        self.prev = hyp[n:]
        return self.prev


def recognize_timed(tr: Transcriber, audio: np.ndarray, bucket_s: float = 0.0):
    """Same call the app makes, but with token timestamps; holds the Transcriber's lock like transcribe() does.

    ``bucket_s``: pad the end with silence up to a multiple of this, so the model sees a few input lengths again
    and again. A length the CUDA session has not seen yet costs ~2x (measured: ~60 ms instead of ~25 ms)."""
    pad = np.zeros(int(PAD_S * SAMPLE_RATE), dtype=np.float32)
    tail = pad
    if bucket_s:
        step = int(bucket_s * SAMPLE_RATE)
        n = audio.size + 2 * pad.size
        tail = np.zeros(pad.size + (-n) % step, dtype=np.float32)
    with tr._lock:
        res = tr._model.with_timestamps().recognize(np.concatenate([pad, audio, tail]), sample_rate=SAMPLE_RATE)
    return words_with_times(res.tokens or [], [t - PAD_S for t in (res.timestamps or [])])


def run(tr: Transcriber, audio: np.ndarray, interval_s: float, trim_s: float, quiet: bool,
        bucket_s: float = 0.0) -> dict:
    total_s = audio.size / SAMPLE_RATE
    la = LocalAgreement()
    updates: list[dict] = []
    stop = threading.Event()
    t0 = time.perf_counter()

    def heard() -> int:
        return min(audio.size, int((time.perf_counter() - t0) * SAMPLE_RATE))

    def loop() -> None:
        start = 0                                      # buffer start (samples): moves forward when trimming
        next_tick = t0 + interval_s
        while not stop.is_set():
            wait = next_tick - time.perf_counter()
            if wait > 0 and stop.wait(wait):
                return
            next_tick = max(next_tick + interval_s, time.perf_counter())   # never queue up: skip missed ticks
            end = heard()
            if end - start < int(0.5 * SAMPLE_RATE):
                continue
            c0 = time.perf_counter()
            hyp = recognize_timed(tr, audio[start:end], bucket_s)
            cost = time.perf_counter() - c0
            off = start / SAMPLE_RATE
            hyp = la.new_part([(w, t + off) for w, t in hyp])
            tentative = la.update(hyp)
            if trim_s and (end - start) / SAMPLE_RATE > trim_s and tentative:
                # restart the window at the first uncommitted word (a little before it), as whisper_streaming does
                start = max(start, int((tentative[0][1] - 0.2) * SAMPLE_RATE))
            elif trim_s and (end - start) / SAMPLE_RATE > trim_s and not tentative and la.committed:
                start = max(start, int(min(end / SAMPLE_RATE, la.committed[-1][1] + 0.4) * SAMPLE_RATE))
            u = {"at_s": end / SAMPLE_RATE, "buf_s": (end - start) / SAMPLE_RATE, "cost_ms": cost * 1000,
                 "lag_s": time.perf_counter() - t0 - end / SAMPLE_RATE, "n_commit": len(la.committed)}
            updates.append(u)
            if not quiet:
                shown = " ".join(w for w, _ in la.committed[-12:])
                tent = " ".join(w for w, _ in tentative)
                print(f"[{u['at_s']:6.2f}s audio | {u['cost_ms']:6.1f} ms | win {u['buf_s']:5.1f}s] "
                      f"...{shown[-70:]} \x1b[2m{tent[:60]}\x1b[0m", flush=True)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    while heard() < audio.size:                        # the user is "speaking"
        time.sleep(0.005)
    t_release = time.perf_counter()
    stop.set()
    tr._lock.acquire()                                 # an update still running holds the model: wait for it
    lock_wait = time.perf_counter() - t_release
    tr._lock.release()
    final = tr.transcribe(audio)                       # today's final pass, on the same model and lock
    release_to_final = time.perf_counter() - t_release
    th.join()
    preview = " ".join(w for w, _ in la.committed + la.prev)
    sim = difflib.SequenceMatcher(None, [norm(w) for w in preview.split()],
                                  [norm(w) for w in final.split()]).ratio()
    costs = [u["cost_ms"] for u in updates]
    return {"audio_s": total_s, "updates": len(updates), "final": final, "preview": preview,
            "release_to_final_ms": release_to_final * 1000, "lock_wait_ms": lock_wait * 1000, "similarity": sim,
            "cost_med_ms": float(np.median(costs)) if costs else 0.0, "cost_max_ms": max(costs, default=0.0),
            "duty": sum(costs) / 1000 / total_s, "lag_max_s": max((u["lag_s"] for u in updates), default=0.0),
            "first_commit_s": next((u["at_s"] for u in updates if u["n_commit"]), None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", default=os.path.join(os.path.dirname(__file__), "..", "tests", "tts_sample.wav"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--interval-ms", type=float, default=500)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--trim-s", type=float, default=0.0, help="re-transcribe only after the committed text once the "
                                                              "window is longer than this (0: always the whole buffer)")
    ap.add_argument("--bucket-s", type=float, default=0.0, help="pad preview inputs to a multiple of this length")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    clip = read_wav_16k(a.wav)
    gap = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([x for _ in range(max(1, a.repeat)) for x in (clip, gap)][:-1])
    tr = Transcriber(device=a.device)
    t = time.perf_counter()
    tr.load()
    print(f"backend {tr.backend}, loaded in {time.perf_counter() - t:.1f} s; audio {audio.size / SAMPLE_RATE:.1f} s")

    warm, cold = [], []
    for i in range(5):                                 # today: one pass after release, nothing running before it
        t = time.perf_counter()
        tr.transcribe(audio)                           # the same length again (how the README number is measured)
        warm.append((time.perf_counter() - t) * 1000)
        fresh = np.concatenate([audio, np.zeros(480 * (i + 1) + 7919, dtype=np.float32)])
        t = time.perf_counter()
        tr.transcribe(fresh)                           # a length this session has not seen: every real dictation
        cold.append((time.perf_counter() - t) * 1000)
    r = run(tr, audio, a.interval_ms / 1000, a.trim_s, a.quiet, a.bucket_s)
    print(f"\ninterval {a.interval_ms:.0f} ms, trim {a.trim_s or 'off'}, {r['updates']} updates over "
          f"{r['audio_s']:.1f} s")
    print(f"update cost: median {r['cost_med_ms']:.1f} ms, max {r['cost_max_ms']:.1f} ms; "
          f"model busy {r['duty'] * 100:.1f}% of the speaking time; worst display lag {r['lag_max_s'] * 1000:.0f} ms")
    print(f"first committed word after {r['first_commit_s']} s of audio")
    print(f"release -> final text: {r['release_to_final_ms']:.1f} ms with preview loop (of which "
          f"{r['lock_wait_ms']:.1f} ms waiting for a running update); today, no preview: "
          f"{float(np.median(cold)):.1f} ms for a new length, {float(np.median(warm)):.1f} ms for a repeated one "
          f"(medians of 5)")
    print(f"preview vs final word similarity: {r['similarity']:.3f}")
    print(f"final:   {r['final'][:160]}")
    print(f"preview: {r['preview'][:160]}")


if __name__ == "__main__":
    main()
