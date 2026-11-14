"""Benchmark ASR: prints backend, load time and transcribe time for a ~10 s clip.

Usage: python scripts/bench_asr.py [--device auto|cuda|cpu] [--wav path.wav]
Without --wav, a clip is synthesised with Windows System.Speech TTS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxflow.asr import Transcriber  # noqa: E402

TEXT = ("The quick brown fox jumps over the lazy dog. Please schedule the meeting for Thursday "
        "at three thirty in the afternoon, and send the report to the engineering team before the end of the day.")


def make_tts(path: str) -> None:
    ps = ("Add-Type -AssemblyName System.Speech;"
          "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
          f"$s.SetOutputToWaveFile('{path}',$f);$s.Speak('{TEXT}');$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wav")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    wav = a.wav
    if not wav:
        wav = os.path.join(tempfile.gettempdir(), "voxflow_bench.wav")
        make_tts(wav)
    audio = read_wav(wav)
    t = Transcriber(device=a.device)
    t0 = time.perf_counter()
    t.load()
    load = time.perf_counter() - t0
    times, text = [], ""
    for _ in range(a.runs):
        t0 = time.perf_counter()
        text = t.transcribe(audio)
        times.append(time.perf_counter() - t0)
    print(f"backend:    {t.backend}")
    print(f"load time:  {load:.2f} s")
    print(f"clip:       {len(audio) / 16000:.1f} s")
    print(f"transcribe: first {times[0] * 1000:.0f} ms, best {min(times) * 1000:.0f} ms, "
          f"mean {np.mean(times) * 1000:.0f} ms over {a.runs} runs")
    print(f"text:       {text}")


if __name__ == "__main__":
    main()
