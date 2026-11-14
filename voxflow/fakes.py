"""Stand-ins for the microphone, the models and the desktop.

Used by ``--demo`` (the whole app without models or hardware), by ``--fake`` (real hotkey and paste, stub
audio and models) and by tests. They mirror the real classes' interfaces exactly.
"""
from __future__ import annotations

import math
import time

import numpy as np

SAMPLE_RATE = 16000


class FakeRecorder:
    """Synthesises a speech-like signal (syllable bursts) so the UI can be shown without a microphone."""

    alive = True
    using_fallback = False
    error = None

    def __init__(self, device=None, max_seconds: float = 300.0):
        self.device = device
        self.max_seconds = max_seconds
        self.level_boost = 1.0
        self._rec = False
        self._t0 = 0.0
        self._last_speech_s = 0.0
        self.active_voice = True
        self.closed = False

    def _env(self, t: np.ndarray) -> np.ndarray:
        syl = 0.5 + 0.5 * np.sin(2 * math.pi * 3.1 * t + 1.3 * np.sin(2 * math.pi * 0.7 * t))
        phr = np.clip(np.sin(2 * math.pi * 0.45 * t) * 1.6, 0, 1) ** 0.7
        return (syl ** 1.4) * (0.15 + 0.85 * phr) * 0.32

    def start(self) -> None:
        self._rec, self._t0 = True, time.perf_counter()

    def stop(self) -> np.ndarray:
        if not self._rec:
            self._last_speech_s = 0.0
            return np.zeros(0, dtype=np.float32)
        self._rec = False
        seconds = min(time.perf_counter() - self._t0, self.max_seconds)
        self._last_speech_s = seconds
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        return (self._env(t) * np.sin(2 * math.pi * 160 * t)).astype(np.float32)

    @property
    def is_recording(self) -> bool:
        return self._rec

    @property
    def hit_cap(self) -> bool:
        return self._rec and time.perf_counter() - self._t0 >= self.max_seconds

    @property
    def last_speech_seconds(self) -> float:
        return self._last_speech_s

    def recent_samples(self, n: int = 1024) -> np.ndarray:
        if not self.active_voice:
            return np.zeros(n, dtype=np.float32)
        t = time.perf_counter() - np.arange(n)[::-1] / SAMPLE_RATE
        f0 = 140 + 40 * np.sin(t * 2.1)
        sig = np.sin(2 * math.pi * f0 * t) + 0.5 * np.sin(2 * math.pi * 2 * f0 * t)
        return (self._env(t) * sig * 0.8).astype(np.float32)

    def level(self) -> float:
        if not self.active_voice:
            return 0.0
        s = self.recent_samples(800)
        return float(min(1.0, np.sqrt((s ** 2).mean()) * 3.2 * self.level_boost))

    def reopen(self, rescan: bool = True) -> None:
        self.closed = False

    def set_device(self, device) -> None:
        self.device = device

    def close(self) -> None:
        self._rec = False
        self.closed = True


class FakeTranscriber:
    backend = "demo"

    def __init__(self, *a, text: str = "um so this is a demo transcript you know with no real model",
                 delay_s: float = 0.4, **k):
        self.text = text
        self.delay_s = delay_s

    def load(self) -> None:
        time.sleep(min(self.delay_s, 0.2))

    def transcribe(self, audio) -> str:
        time.sleep(self.delay_s)
        return self.text


class FakeCleaner:
    """Rule-based clean-up (the real, model-free pass) plus a canned "LLM" for command mode."""

    backend = "demo"
    n_ctx = 4096

    def __init__(self, *a, delay_s: float = 0.15, **k):
        self.delay_s = delay_s
        self.timeout_s = 4.0
        self._ready = False
        self._rules = None              # the real rule pipeline (every language), built on first use
        self.has_llm = True
        self.last_info: dict = {}

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        self._ready = True

    def clean(self, text: str, *, tone=None, terms=None, language=None, smart=False) -> str:
        from .cleanup import Cleaner

        time.sleep(self.delay_s)
        if self._rules is None:
            self._rules = Cleaner(use_llm=False)
            self._rules.load()
        out = self._rules.clean(text, tone=tone, terms=terms, language=language, smart=smart)
        self.last_info = dict(self._rules.last_info)      # same report as the real Cleaner (never ``rules_only``)
        return out

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def generate(self, messages, max_tokens: int = 512, timeout_s: float = 20.0, temperature: float = 0.2) -> str:
        time.sleep(self.delay_s)
        user = messages[-1]["content"]
        if "\n\nTEXT:\n" in user:
            return user.split("\n\nTEXT:\n", 1)[1].strip().upper()      # a visible, deterministic "rewrite"
        return "Here is a short demo text written by command mode."


class FakeInjector:
    """Records what would have been pasted instead of touching the clipboard or keyboard."""

    def __init__(self, selection: str = ""):
        self.restore = True
        self.restore_delay = 0.5
        self.last_outcome = "failed"
        self.selection = selection
        self.pasted: list[str] = []
        self.copied: list[str] = []
        self.undone = 0

    def begin(self, text: str) -> bool:
        if not text:
            return False
        self.pasted.append(text)
        self.last_outcome = "pasted"
        return True

    def inject(self, text: str) -> bool:
        return self.begin(text)

    def finish(self) -> None:
        pass

    def finish_later(self) -> None:
        pass

    def close(self) -> None:
        pass

    def copy_selection(self, timeout: float = 0.6):
        return self.selection

    def undo(self) -> bool:
        self.undone += 1
        return True

    def copy_text(self, text: str) -> bool:
        self.copied.append(text)
        return True


class FakeMuteBackend:
    def __init__(self):
        self.muted: list[str] = []
        self.calls: list[str] = []

    def mute_others(self) -> list[str]:
        self.calls.append("mute")
        self.muted = ["demo-session"]
        return list(self.muted)

    def unmute(self, session_ids) -> None:
        self.calls.append("unmute")
        self.muted = []


class FakeAutostart:
    """In-memory replacement for the Run registry key."""

    def __init__(self, enabled: bool = False):
        self.value = enabled

    def enabled(self) -> bool:
        return self.value

    def set(self, enabled: bool) -> None:
        self.value = bool(enabled)


FAKE_MICROPHONES = ["Studio Microphone (USB Audio)", "Headset (Arctis Nova Pro Wireless)", "Webcam Microphone"]
