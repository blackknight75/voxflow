"""voxflow.app paths the main suite leaves out: the keyword-probe cache's edge cases, ``too_short`` fallbacks,
the optional ``--selftest`` steps driven by environment variables, and the less common start-up branches of
``run_app`` (no answer on the instance channel, channel unavailable, the demo cycle filing transcripts)."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tests.test_app import (_private_instance_channel, _stop_quit_timers, clean_process_state,  # noqa: F401
                            desktop, quit_after)
from voxflow import app, fakes


# =========================================================================== keyword probing
def test_accepts_keywords_works_for_callables_that_cannot_be_cached():
    class Unhashable:
        __hash__ = None

        def __call__(self, text, *, tone=None):
            return text

    fn = Unhashable()
    before = dict(app._KW_CACHE)
    assert app._accepts_keywords(fn, ("tone",)) is True
    assert app._accepts_keywords(fn, ("smart",)) is False
    assert app._KW_CACHE == before, "nothing is stored for a key that cannot be hashed"


def test_accepts_keywords_is_false_when_the_signature_cannot_be_read():
    class Opaque:
        __signature__ = "not a signature"                         # inspect.signature() raises TypeError

        def __call__(self, *args, **kwargs):
            return None

    assert app._accepts_keywords(Opaque(), ("tone",)) is False


def test_make_recorder_passes_the_cap_only_to_classes_that_take_it():
    class Capped:
        def __init__(self, device, max_seconds=300.0):
            self.device, self.max_seconds = device, max_seconds

    class Plain:
        def __init__(self, device):
            self.device = device

    cfg = types.SimpleNamespace(max_record_s=42.0)
    capped, plain = app.make_recorder(Capped, "Yeti", cfg), app.make_recorder(Plain, "Yeti", cfg)
    assert (capped.device, capped.max_seconds) == ("Yeti", 42.0)
    assert plain.device == "Yeti" and not hasattr(plain, "max_seconds")


# =========================================================================== too_short
def test_too_short_falls_back_to_the_audio_length_when_the_recorder_cannot_say():
    from voxflow.audio import PREROLL_S

    class Rec:
        def last_speech_seconds(self):
            raise RuntimeError("stream closed")

    long_audio = np.zeros(int((PREROLL_S + 1.0) * app.SAMPLE_RATE), dtype=np.float32)
    short_audio = np.zeros(int((PREROLL_S + 0.1) * app.SAMPLE_RATE), dtype=np.float32)
    assert app.too_short(long_audio, Rec(), held_s=2.0, min_s=0.3) is False
    assert app.too_short(short_audio, Rec(), held_s=2.0, min_s=0.3) is True, "the pre-roll is not speech"

    class Calls:
        def last_speech_seconds(self):
            return 0.1

    assert app.too_short(long_audio, Calls(), held_s=2.0, min_s=0.3) is True, "a callable answer is used"


def test_too_short_without_an_audio_module_counts_the_whole_recording(monkeypatch):
    def broken(name):
        raise ImportError("sounddevice is missing")

    monkeypatch.setattr(app, "importlib", types.SimpleNamespace(import_module=broken))
    audio = np.zeros(int(0.4 * app.SAMPLE_RATE), dtype=np.float32)
    assert app.too_short(audio, object(), held_s=2.0, min_s=0.3) is False      # no pre-roll is subtracted
    assert app.too_short(audio, object(), held_s=2.0, min_s=0.5) is True


# =========================================================================== selftest options
@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_hub_step_downloads_one_small_file(desktop, capsys, monkeypatch, tmp_path):
    import huggingface_hub

    from voxflow import modelhub
    from voxflow.config import Config
    asked = []
    fetched = tmp_path / "config.json"
    fetched.write_bytes(b"x" * 12)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, name: asked.append((repo, name)) or str(fetched))
    monkeypatch.setenv("VOXFLOW_SELFTEST_HUB", "1")
    assert app.selftest(None, fake=True) == 0
    out = capsys.readouterr().out
    assert asked == [(modelhub.ASR_REPOS[Config().asr_model], "config.json")]
    assert "[ok] model hub download" in out and f"downloaded 12 bytes into {modelhub.cache_root()}" in out

    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, name: (_ for _ in ()).throw(ConnectionError("offline")))
    assert app.selftest(None, fake=True) == 1
    out = capsys.readouterr().out
    assert "[FAIL] model hub download: ConnectionError: offline" in out and out.rstrip().endswith("SELFTEST FAILED")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_without_the_hub_variable_never_downloads(desktop, capsys, monkeypatch):
    import huggingface_hub
    monkeypatch.delenv("VOXFLOW_SELFTEST_HUB", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("network used"))
    assert app.selftest(None, fake=True) == 0
    assert "model hub download" not in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
@pytest.mark.parametrize("forced,speech,language", [("cpu", "cpu", "cpu"), ("CUDA ", "cuda", "cuda"),
                                                    ("webgpu", "webgpu", "vulkan"), ("Vulkan", "webgpu", "vulkan")])
def test_selftest_forces_the_device_of_both_models(desktop, capsys, monkeypatch, forced, speech, language):
    seen = {}

    class Asr(fakes.FakeTranscriber):
        def __init__(self, model, device):
            super().__init__()
            seen["asr"] = device

    class Cln(fakes.FakeCleaner):
        def __init__(self, model, device, timeout_s):
            super().__init__()
            seen["cleanup"] = device

    monkeypatch.setattr(app, "_real", lambda mod, attr: {"audio": fakes.FakeRecorder, "asr": Asr, "cleanup": Cln}[mod])
    monkeypatch.setenv("VOXFLOW_DEVICE", forced)
    want = forced.strip().lower()
    assert app.selftest(None, fake=False) == 0
    out = capsys.readouterr().out
    assert seen == {"asr": speech, "cleanup": language}
    assert f"[note] VOXFLOW_DEVICE={want}" in out
    assert "using stand-ins" not in out, "no stand-in note when every real module was built"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_selftest_ignores_an_unknown_forced_device(desktop, capsys, monkeypatch):
    from voxflow.config import Config
    seen = {}
    real = app.make_modules

    def make_modules(cfg, fake):
        seen["asr_device"] = cfg.asr_device
        return real(cfg, fake)

    monkeypatch.setattr(app, "make_modules", make_modules)
    monkeypatch.setenv("VOXFLOW_DEVICE", "quantum")
    assert app.selftest(None, fake=True) == 0
    assert seen["asr_device"] == Config().asr_device and "VOXFLOW_DEVICE" not in capsys.readouterr().out


# =========================================================================== the Qt application
def test_second_launch_exits_even_when_the_first_instance_does_not_answer(qapp, clean_process_state, data_dir,
                                                                          capsys):
    from PySide6.QtCore import QLockFile
    lock = QLockFile(str(data_dir / "voxflow.lock"))               # held, but nobody listens on the channel
    assert lock.tryLock(100)
    try:
        assert app.run_app(fake=True, no_hook=True) == 0
        assert "already running" in capsys.readouterr().out
        assert not (data_dir / "logs").exists(), "the second launch must not start a controller"
    finally:
        lock.unlock()


def test_app_runs_without_the_instance_channel_and_with_a_ui_that_has_no_close(qapp, clean_process_state, data_dir,
                                                                               monkeypatch):
    from PySide6.QtNetwork import QLocalServer

    from voxflow.ui import shell
    built = []
    monkeypatch.setattr(QLocalServer, "listen", lambda self, name: False)
    monkeypatch.setattr(shell, "build_ui", lambda controller, show_window=False: built.append(controller) or object())
    quit_after(200)
    assert app.run_app(fake=True, no_hook=True) == 0
    log_text = (data_dir / "logs" / "voxflow.log").read_text(encoding="utf-8")
    assert "single-instance channel unavailable" in log_text and "VoxFlow stopped (exit code 0)" in log_text
    assert len(built) == 1 and built[0]._closed, "the controller is shut down on exit"


def test_a_knock_whose_connection_is_already_gone_still_shows_the_window(qapp, clean_process_state, data_dir,
                                                                        monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtNetwork import QLocalServer, QLocalSocket

    from voxflow import controller as ctl
    activations = []
    real_request = ctl.Controller.request_activate
    monkeypatch.setattr(ctl.Controller, "request_activate",
                        lambda self: activations.append(True) or real_request(self))
    monkeypatch.setattr(QLocalServer, "nextPendingConnection", lambda self: None)

    def knock():
        sock = QLocalSocket()
        sock.connectToServer(app._instance_name())
        assert sock.waitForConnected(1000)
        sock.disconnectFromServer()

    quit_after(900)
    QTimer.singleShot(300, knock)
    assert app.run_app(fake=True, no_hook=True) == 0
    assert activations == [True]


def test_demo_cycle_files_sample_transcripts_for_its_first_three_rounds_only(qapp, clean_process_state, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from voxflow import controller as ctl
    cycle = [("done", "dictate", "Pasted", 0.005), ("idle", "dictate", "", 0.005)]
    monkeypatch.setattr(app, "_DEMO_CYCLE", cycle)
    steps, added = [], []
    real_state, real_add = ctl.Controller.demo_set_state, ctl.Controller.demo_add_transcript

    def set_state(self, state, mode="dictate", message=""):
        steps.append(state)
        real_state(self, state, mode, message)
        if len(steps) >= 5 * len(cycle):
            QApplication.instance().quit()

    monkeypatch.setattr(ctl.Controller, "demo_set_state", set_state)
    monkeypatch.setattr(ctl.Controller, "demo_add_transcript",
                        lambda self, text=None: added.append(True) or real_add(self, text))
    quit_after(15000)                                             # safety net only
    assert app.run_app(demo=True) == 0
    assert len(steps) >= 5 * len(cycle) and steps[:4] == ["done", "idle", "done", "idle"]
    assert len(added) == 3, "one sample per 'done' during the first three rounds, none afterwards"


# =========================================================================== recorder helpers
def test_recorder_helpers_tolerate_foreign_recorder_objects():
    class Odd:
        closes = 0

        @property
        def hit_cap(self):
            raise RuntimeError("stream closed")

        def close(self):
            self.closes += 1
            raise OSError("already closed")

    odd = Odd()
    assert app.recorder_hit_cap(odd) is False
    assert app.recorder_hit_cap(types.SimpleNamespace(hit_cap=True)) is True
    assert app.recorder_hit_cap(types.SimpleNamespace(hit_cap=1)) is False, "only a real True counts"
    app.close_recorder(odd)                                       # a failing close is swallowed
    assert odd.closes == 1
    app.close_recorder(object())                                  # nothing to close: not an error
