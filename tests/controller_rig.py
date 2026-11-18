"""Test rig for ``voxflow.controller.Controller``: every desktop / hardware / model dependency is a scripted
double that is passed in through the controller's constructor, so nothing real is touched.

    rig = make_rig(tmp_path, qtbot)          # controller started, models "loaded"
    rig.dictate("um hello world")            # press hotkey, "speak", release, wait for the result
    rig.injector.pasted                      # -> ["Hello world."]
"""
from __future__ import annotations

import time

import numpy as np

from voxflow import controller as ctl_mod
from voxflow.context import AppContext
from voxflow.controller import Controller

CTRL, WIN, ALT, SHIFT, ESC, Z = 0xA2, 0x5B, 0xA4, 0xA0, 0x1B, 0x5A
SAMPLE_RATE = 16000


def speech(seconds: float = 2.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


class TRecorder:
    """Scripted microphone. Class attributes are the knobs shared by every instance a controller builds."""

    instances: list = []
    audio = speech()
    speech_seconds = 2.0
    fail_start: Exception | None = None
    fail_stop: Exception | None = None
    fail_init: Exception | None = None

    def __init__(self, device=None, max_seconds: float = 300.0):
        if TRecorder.fail_init is not None:
            raise TRecorder.fail_init
        self.device, self.max_seconds = device, max_seconds
        self.level_boost = 1.0
        self.is_recording = False
        self.hit_cap = False
        self.alive = True
        self.error = None
        self.using_fallback = False
        self.closed = False
        self.reopened = 0
        self.starts = self.stops = 0
        self.last_speech_seconds = 0.0
        self.recorded = 0                     # samples "recorded so far" in the current utterance (the preview reads it)
        self.snapshots: list[int] = []        # the ``start`` offset of every snapshot() call
        TRecorder.instances.append(self)

    def start(self):
        if TRecorder.fail_start is not None:
            raise TRecorder.fail_start
        self.starts += 1
        self.is_recording, self.hit_cap = True, False
        self.recorded = min(len(TRecorder.audio), 4800)      # the pre-roll; tests grow it with ``recorded += n``

    def stop(self):
        if TRecorder.fail_stop is not None:
            self.is_recording = False
            raise TRecorder.fail_stop
        self.stops += 1
        self.is_recording = False
        self.last_speech_seconds = TRecorder.speech_seconds
        return TRecorder.audio

    @property
    def recorded_samples(self):
        return self.recorded if self.is_recording else 0

    def snapshot(self, start=0):
        self.snapshots.append(start)
        if not self.is_recording:
            return np.zeros(0, dtype=np.float32)
        return np.array(TRecorder.audio[start:self.recorded], copy=True)

    def level(self):
        return 0.42

    def recent_samples(self, n=1024):
        return np.full(n, 0.1, dtype=np.float32)

    def reopen(self, rescan=True):
        self.reopened += 1
        if self.error is not None:
            raise self.error
        self.alive, self.closed = True, False

    def close(self):
        self.closed = True
        self.is_recording = False

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.audio, cls.speech_seconds = speech(), 2.0
        cls.fail_start = cls.fail_stop = cls.fail_init = None


class TAsr:
    backend = "cuda"

    def __init__(self, text="um hello world"):
        self.text = text
        self.heard: list[np.ndarray] = []
        self.fail_load: Exception | None = None
        self.fail_transcribe: Exception | None = None
        self.load_gate = None                 # threading.Event: load() blocks until it is set
        self.delay_s = 0.0
        self.loads = 0
        self.timed_calls: list[dict] = []     # every transcribe_timed() call: samples, bucket_s, blocking
        self.partials = None                  # scripted preview: a list of word lists (one per call), or a callable
        self.timed_busy = False               # True: the model is "busy", transcribe_timed returns None
        self.fail_timed: Exception | None = None

    def transcribe_timed(self, audio, *, bucket_s=0.0, blocking=False):
        """The preview call: scripted hypotheses (``partials``), else a growing prefix of ``text``."""
        self.timed_calls.append({"samples": len(audio), "bucket_s": bucket_s, "blocking": blocking})
        if self.fail_timed is not None:
            raise self.fail_timed
        if self.timed_busy:
            return None
        if float(np.max(np.abs(audio), initial=0.0)) < 1e-4:
            return []                         # like the real model: silence says nothing
        n = len(self.timed_calls) - 1
        if callable(self.partials):
            words = self.partials(n, audio)
        elif self.partials is not None:
            words = self.partials[min(n, len(self.partials) - 1)]
        else:
            text = self.text(audio) if callable(self.text) else self.text
            every = text.split()
            words = every[:max(1, int(len(every) * min(1.0, len(audio) / (2.0 * SAMPLE_RATE))))]
        return [(w, 0.3 * i) for i, w in enumerate(words)]

    def load(self):
        self.loads += 1
        if self.load_gate is not None:
            self.load_gate.wait(10)
        if self.fail_load is not None:
            raise self.fail_load

    def transcribe(self, audio):
        self.heard.append(np.array(audio, copy=True))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_transcribe is not None:
            raise self.fail_transcribe
        return self.text(audio) if callable(self.text) else self.text


class TCleaner:
    """The real rule pass (no model) plus scripted overrides and a scripted "LLM" for command mode."""

    n_ctx = 4096

    def __init__(self):
        self.backend = "cuda"
        self.ready = False
        self.has_llm = True
        self.timeout_s = 4.0
        self.fail_load: Exception | None = None
        self.fail_clean: Exception | None = None
        self.fail_generate: Exception | None = None
        self.rewrites: dict[str, str] = {}    # raw transcript -> "model" output
        self.clean_calls: list[tuple[str, dict]] = []
        self.generate_calls: list[dict] = []
        self.reply = lambda messages: "REWRITTEN"
        self.loads = 0

    def load(self):
        self.loads += 1
        if self.fail_load is not None:
            raise self.fail_load
        self.ready = True

    def clean(self, text, *, tone=None, terms=None, language=None, smart=False):
        from voxflow.cleanup import apply_tone, clean_foreign, is_english, rule_clean, smart_format
        self.clean_calls.append((text, dict(tone=tone, terms=terms, language=language, smart=smart)))
        if self.fail_clean is not None:
            raise self.fail_clean
        if not is_english(text, language):
            out = clean_foreign(text)
        else:
            out = self.rewrites.get(text) or rule_clean(text)
            if smart and out.strip():
                out = smart_format(out)
        return apply_tone(out, tone) if tone else out

    def count_tokens(self, text):
        return max(1, len(text) // 3)

    def generate(self, messages, max_tokens=512, timeout_s=20.0, temperature=0.2):
        self.generate_calls.append(dict(messages=messages, max_tokens=max_tokens, timeout_s=timeout_s))
        if self.fail_generate is not None:
            raise self.fail_generate
        return self.reply(messages)


class TInjector:
    def __init__(self):
        self.restore, self.restore_delay = True, 0.5
        self.outcome = "pasted"               # what the next paste does: pasted | typed | copied | failed
        self.last_outcome = "failed"
        self.selection: str | None = ""
        self.pasted: list[str] = []
        self.copied: list[str] = []
        self.undone = 0
        self.undo_ok = True
        self.copy_ok = True
        self.finished_later = 0
        self.closed = 0
        self.selection_grabs = 0

    def begin(self, text):
        self.last_outcome = self.outcome
        if self.outcome in ("pasted", "typed"):
            self.pasted.append(text)
            return True
        if self.outcome == "copied":
            self.copied.append(text)
        return False

    def finish_later(self):
        self.finished_later += 1

    def finish(self):
        pass

    def close(self):
        self.closed += 1

    def copy_selection(self, timeout=0.6):
        self.selection_grabs += 1
        if isinstance(self.selection, Exception):
            raise self.selection
        return self.selection

    def undo(self):
        self.undone += 1
        return self.undo_ok

    def copy_text(self, text):
        if self.copy_ok:
            self.copied.append(text)
        return self.copy_ok


class TMixer:
    def __init__(self):
        self.muted = False
        self.calls: list[str] = []

    def mute_others(self):
        self.calls.append("mute")
        self.muted = True
        return ["session-1"]

    def unmute(self, ids):
        self.calls.append("unmute")
        self.muted = False


class TAutostart:
    def __init__(self, enabled=False):
        self.value = enabled
        self.refuse = False
        self.calls: list[bool] = []

    def enabled(self):
        return self.value

    def set(self, enabled):
        if self.refuse:
            raise ValueError("Windows did not let VoxFlow change the launch-at-login setting.")
        self.calls.append(enabled)
        self.value = enabled


class Spy:
    """Records every emission of the controller's public signals as (name, args)."""

    NAMES = ("state_changed", "level_changed", "transcript_ready", "history_changed", "stats_changed",
             "dictionary_changed", "suggestions_changed", "dictionary_suggestion", "snippets_changed",
             "styles_changed", "config_changed", "status_changed", "model_status_changed", "microphones_changed",
             "hotkey_captured", "notice", "activate_requested", "update_changed", "update_available", "partial_text",
             "restart_requested")

    def __init__(self, controller):
        self.events: list[tuple[str, tuple]] = []
        self._slots = []
        for name in self.NAMES:
            slot = (lambda n: lambda *args: self.events.append((n, args)))(name)
            self._slots.append(slot)
            getattr(controller, name).connect(slot)

    def of(self, name) -> list[tuple]:
        return [args for n, args in self.events if n == name]

    def states(self) -> list[str]:
        return [args[0] for args in self.of("state_changed")]

    def notices(self) -> list[tuple[str, str]]:
        return [(a[0], a[1]) for a in self.of("notice")]

    def clear(self):
        self.events.clear()


class Rig:
    def __init__(self, tmp_path, qtbot, *, start=True, fake=True, config=None, context=None, **kw):
        TRecorder.reset()
        self.qtbot = qtbot
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir(exist_ok=True)
        if config is not None:
            import json
            (self.data_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        # asr= / cleaner= swap in other doubles, e.g. voxflow.fakes.FakeTranscriber / FakeCleaner (the real rules)
        self.asr, self.cleaner = kw.pop("asr", None) or TAsr(), kw.pop("cleaner", None) or TCleaner()
        self.injector, self.mixer, self.autostart = TInjector(), TMixer(), TAutostart()
        self.sounds: list[str] = []
        self.context = context or AppContext("notepad.exe", "Untitled - Notepad", 0x100)
        self.c = Controller(fake=fake, data_dir=self.data_dir, install_hook=kw.pop("install_hook", False),
                            modules=(TRecorder, self.asr, self.cleaner), injector=self.injector,
                            sound_backend=self._play, mute_backend=self.mixer,
                            context_fn=lambda: self.context, autostart=self.autostart, **kw)
        self.c.muter._threaded = False                      # deterministic: mute / unmute run inline
        self.now = 1000.0                                   # the hotkey machines' clock (seconds), under test control
        self.held: set[int] = set()                         # keys physically down (what GetAsyncKeyState would say)
        self.startup_events: list = []
        self.spy = Spy(self.c)
        if start:
            self.start()

    def _play(self, wav: bytes):
        # identify the cue by its samples at the player's current volume
        from voxflow import sounds
        known = {sounds.to_wav(sounds.synth(n), self.c.sounds.volume): n for n in sounds.SOUND_NAMES}
        self.sounds.append(known.get(wav, "?"))

    # ---- lifecycle
    def start(self):
        self.c.start()
        self.wait(lambda: self.c.model_status("asr")["state"] in ("ready", "failed")
                  and self.c.model_status("cleanup")["state"] in ("ready", "degraded", "failed"))
        self.startup_events = list(self.spy.events)         # tests start from a clean slate
        self.spy.clear()

    def close(self):
        self.c.shutdown()

    def wait(self, condition, timeout=5000):
        self.qtbot.waitUntil(lambda: bool(condition()), timeout=timeout)

    def settle(self, ms=30):
        self.qtbot.wait(ms)

    @property
    def rec(self) -> TRecorder:
        return self.c._recorder

    # ---- keyboard
    def keys(self, *events):
        for machine, _callback in self.c.listener._machines():
            machine.clock = lambda: self.now                # tap / double-tap timing is decided by this clock
        for vk, down in events:
            (self.held.add if down else self.held.discard)(vk)
            # like the real hook: every event comes with a probe of the physical key state
            self.c.listener.feed(vk, down, lambda key: key in self.held)

    def tick(self):
        """One pass of the hook thread's 250 ms watchdog."""
        self.c.listener.tick(lambda key: key in self.held)

    def press(self, binding="dictate"):
        first = ALT if binding == "command" else CTRL
        self.keys((first, True), (WIN, True))

    def release(self, binding="dictate", held_s=1.0):
        """Release after holding for ``held_s`` seconds (long enough not to be a tap)."""
        self.now += held_s
        self.c._t_start -= held_s
        first = ALT if binding == "command" else CTRL
        self.keys((WIN, False), (first, False))

    def tap(self, binding="dictate"):
        """A quick press and release (50 ms), followed by a 100 ms pause."""
        first = ALT if binding == "command" else CTRL
        self.keys((first, True), (WIN, True))
        self.now += 0.05
        self.keys((WIN, False), (first, False))
        self.now += 0.10

    def pause(self, seconds=2.0):
        """Let time pass between utterances so they are not taken for a double-tap."""
        self.now += seconds

    # ---- whole utterances
    def dictate(self, spoken=None, binding="dictate", wait=True):
        if spoken is not None:
            self.asr.text = spoken
        before = len(self.spy.events)
        self.pause()
        self.press(binding)
        assert self.c.state == "listening", f"did not start listening (state={self.c.state}, {self.c.message!r})"
        self.release(binding)
        if wait:
            self.wait(lambda: self.c.state in ("done", "error", "idle"))
        return self.spy.events[before:]

    def last_entry(self):
        items = self.c.history.list(limit=1)
        return items[0] if items else None


def make_rig(tmp_path, qtbot, monkeypatch=None, **kw) -> Rig:
    if monkeypatch is not None:
        monkeypatch.setattr(ctl_mod, "DONE_MS", 40)
        monkeypatch.setattr(ctl_mod, "ERROR_MS", 40)
    return Rig(tmp_path, qtbot, **kw)
