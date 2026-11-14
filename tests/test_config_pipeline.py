"""Issues 1, 2, 3, 8a: pipeline empty-vs-failure semantics, minimum-length filter, recorder cleanup."""
import numpy as np

from voxflow import app
from voxflow.config import Config

SR = 16000


class Asr:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return self.text


class Cleaner:
    def __init__(self, out=None, ready=True, exc=None):
        self.out, self.ready, self.exc = out, ready, exc

    def clean(self, text):
        if self.exc:
            raise self.exc
        return text if self.out is None else self.out


def test_filler_only_is_dropped_not_pasted_raw():
    res = app.run_pipeline(np.zeros(10), Asr("um"), Cleaner(out=""), True)
    assert res["text"] == "" and res["dropped"] is True and res["raw"] == "um"


def test_cleaned_text_used_and_unchanged_ok():
    res = app.run_pipeline(np.zeros(10), Asr("um hello there"), Cleaner(out="Hello there."), True)
    assert res["text"] == "Hello there." and not res["dropped"] and res["cleaned"]
    res = app.run_pipeline(np.zeros(10), Asr("hello"), Cleaner(), True)
    assert res["text"] == "hello" and not res["dropped"]


def test_failure_or_unavailable_cleaner_pastes_raw():
    assert app.run_pipeline(np.zeros(10), Asr("hello"), Cleaner(exc=RuntimeError("x")), True)["text"] == "hello"
    assert app.run_pipeline(np.zeros(10), Asr("hello"), Cleaner(ready=False, out=""), True)["text"] == "hello"
    assert app.run_pipeline(np.zeros(10), Asr("hello"), Cleaner(out=""), False)["text"] == "hello"   # cleanup off


def test_empty_asr_is_not_marked_dropped():
    res = app.run_pipeline(np.zeros(10), Asr("  "), Cleaner(out=""), True)
    assert res["text"] == "" and res["dropped"] is False        # -> "Didn't catch that"


class Rec:
    def __init__(self, speech=None):
        if speech is not None:
            self.last_speech_seconds = speech


def test_too_short_uses_hold_time_first():
    audio = np.zeros(int(2 * SR), dtype=np.float32)
    assert app.too_short(audio, Rec(2.0), held_s=0.1, min_s=0.3) is True      # accidental tap


def test_too_short_prefers_recorder_speech_seconds_property_or_method():
    audio = np.zeros(int(2 * SR), dtype=np.float32)
    assert app.too_short(audio, Rec(0.1), 1.0, 0.3) is True
    assert app.too_short(audio, Rec(1.2), 1.0, 0.3) is False
    r = Rec()
    r.last_speech_seconds = lambda: 0.05
    assert app.too_short(audio, r, 1.0, 0.3) is True


def test_too_short_subtracts_preroll_fallback(monkeypatch):
    import types
    import sys
    fake = types.ModuleType("voxflow.audio")
    fake.PREROLL_S = 0.3
    monkeypatch.setitem(sys.modules, "voxflow.audio", fake)
    tap = np.zeros(int(0.3 * SR), dtype=np.float32)            # exactly the pre-roll: no real audio
    assert app.too_short(tap, Rec(), held_s=1.0, min_s=0.3) is True
    ok = np.zeros(int(0.9 * SR), dtype=np.float32)
    assert app.too_short(ok, Rec(), held_s=1.0, min_s=0.3) is False


def test_close_recorder_is_safe():
    class R:
        closed = 0

        def close(self):
            R.closed += 1
            raise RuntimeError("boom")

    app.close_recorder(R())          # swallows errors
    app.close_recorder(object())     # no close() at all
    assert R.closed == 1


def test_new_config_fields_and_no_dead_state():
    c = Config()
    assert c.max_record_s == 120.0 and c.restore_delay_s == 0.5
    from pathlib import Path
    assert "busy" not in (Path(app.__file__)).read_text(encoding="utf-8")


def test_recorder_hit_cap_polling_helper():
    class R:
        hit_cap = True

    class NoAttr:
        pass

    class Raises:
        @property
        def hit_cap(self):
            raise RuntimeError("x")

    assert app.recorder_hit_cap(R()) is True
    assert app.recorder_hit_cap(NoAttr()) is False      # older Recorder without the property
    assert app.recorder_hit_cap(Raises()) is False


def test_cap_ends_hold_and_locked_state_like_a_release():
    from voxflow.hotkey import HotkeyMachine, VK_LCTRL, VK_LWIN
    m = HotkeyMachine("ctrl+win")
    m.feed(VK_LCTRL, True)
    assert m.feed(VK_LWIN, True).events == ["start"]
    m.force_idle()                                       # cap hit while still holding
    assert m.feed(VK_LWIN, False).events == []           # the later release must not emit a second stop
    m.feed(VK_LCTRL, False)
    assert m.state == "idle"
    m.feed(VK_LCTRL, True)                               # ...and the next press starts normally
    assert m.feed(VK_LWIN, True).events == ["start"]


# The two "running app" scenarios that used to live here (recording cap ends a hands-free recording; changing the
# microphone mid-recording) now run against the real Controller in tests/test_controller_pipeline.py:
# test_recording_cap_ends_a_hold_and_a_locked_recording_like_a_release and
# test_changing_the_microphone_mid_recording_abandons_it_cleanly. App start-up is covered by tests/test_app.py.


def test_make_recorder_passes_configured_cap_and_tolerates_stubs():
    seen = []

    class Real:
        def __init__(self, device=None, max_seconds=300.0):
            seen.append((device, max_seconds))

    class Stub:
        def __init__(self, device=None):
            seen.append((device, None))

    cfg = Config(max_record_s=42.0)
    app.make_recorder(Real, 3, cfg)
    app.make_recorder(Stub, None, cfg)
    assert seen == [(3, 42.0), (None, None)]


def test_whitespace_only_cleaner_result_is_pasted_empty_is_not():
    for nl in (chr(10), chr(10) * 2, " "):          # real line break characters, as worker2's Cleaner returns
        res = app.run_pipeline(np.zeros(10), Asr("new line"), Cleaner(out=nl), True)
        assert res["text"] == nl and res["dropped"] is False
    res = app.run_pipeline(np.zeros(10), Asr("new line"), Cleaner(out=""), True)
    assert res["text"] == "" and res["dropped"] is True


def test_hotkey_machine_is_thread_safe_under_force_idle():
    import threading
    from voxflow.hotkey import HotkeyMachine, VK_LCTRL, VK_LWIN
    m = HotkeyMachine("ctrl+win")
    errors, stop = [], threading.Event()

    def hook_thread():
        try:
            for _ in range(20000):
                for vk, down in ((VK_LCTRL, True), (VK_LWIN, True), (VK_LWIN, False), (VK_LCTRL, False)):
                    m.feed(vk, down, probe=lambda k: True)
                m.tick(lambda k: True)
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()

    def ui_thread():
        try:
            while not stop.is_set():
                m.force_idle()
        except Exception as exc:
            errors.append(exc)

    ts = [threading.Thread(target=hook_thread), threading.Thread(target=ui_thread)]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert not errors
    assert m.state in ("idle", "hold", "wait_release", "locked")
    assert hasattr(m, "_lock")
