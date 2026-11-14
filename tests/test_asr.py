import os
import re
import subprocess
import tempfile
import wave

import numpy as np
import pytest

pytest.importorskip("onnx_asr")
pytestmark = pytest.mark.model

from voxflow.asr import Transcriber  # noqa: E402


@pytest.fixture(scope="module")
def tr():
    t = Transcriber()
    t.load()
    return t


def test_backend(tr):
    assert tr.backend in ("cuda", "webgpu", "cpu")


def test_silence_and_short(tr):
    assert tr.transcribe(np.zeros(16000, np.float32)) == ""
    assert tr.transcribe(np.zeros(0, np.float32)) == ""
    assert tr.transcribe(np.random.randn(500).astype(np.float32) * 0.1) == ""


def test_tts_accuracy(tr):
    text = "Please send the report to the engineering team before the end of the day."
    path = os.path.join(tempfile.gettempdir(), "voxflow_test_tts.wav")
    ps = ("Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
          f"$s.SetOutputToWaveFile('{path}',$f);$s.Speak('{text}');$s.Dispose()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60)
    except Exception:
        pytest.skip("System.Speech TTS unavailable")
    with wave.open(path) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    got = tr.transcribe(audio)
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).split()  # noqa: E731
    want, have = norm(text), norm(got)
    hits = sum(1 for a, b in zip(want, have) if a == b)
    assert hits / len(want) > 0.9, got
