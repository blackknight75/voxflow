"""Regression tests for the robustness QA findings (R1-R19, see CHANGELOG): a damaged install, an unwritable
data folder, log redaction, the crash handler, a PC without a microphone, a read-only settings file, damaged
model files, a full disk, the CUDA cache location, crash reports and launch at login."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from tests.controller_rig import TAutostart, TCleaner, TInjector, TMixer, TRecorder
from tests.test_cov_controller import patch_os
from voxflow import app, controller as ctl_mod
from voxflow.config import Config
from voxflow.context import AppContext
from voxflow.controller import Controller


def _real_controller(tmp_path, **kw):
    kw.setdefault("injector", TInjector())
    return Controller(fake=False, data_dir=tmp_path, install_hook=False, sound_backend=lambda wav: None,
                      mute_backend=TMixer(), context_fn=lambda: AppContext(), autostart=TAutostart(),
                      speed_check=False, updates=False, **kw)


# =========================================================================== R1: no demo stand-ins in the real app
def test_r1_a_broken_speech_module_fails_to_load_instead_of_pasting_the_demo_sentence(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch)
    TRecorder.reset()
    real = app._real
    monkeypatch.setitem(sys.modules, "voxflow.asr", None)          # "import voxflow.asr" raises ImportError
    # only the speech module is really imported (and fails); never the real cleanup model
    monkeypatch.setattr(app, "_real", lambda mod, attr: {"audio": TRecorder, "cleanup": lambda *a: TCleaner()}.get(mod)
                        or real(mod, attr))
    c = _real_controller(tmp_path)
    try:
        c.start()
        qtbot.waitUntil(lambda: c.model_status("asr")["state"] == "failed", timeout=5000)
        detail = c.model_status("asr")["detail"]
        assert detail == "VoxFlow's installation is damaged (ModuleNotFoundError in asr.py). Reinstall VoxFlow."
        assert c.status()["level"] == "error" and c.status()["title"] == "Speech model unavailable"
        assert c.model_status("asr")["backend"] != "demo"
        with pytest.raises(app.DamagedInstall):
            c._asr.transcribe(np.zeros(16000, dtype=np.float32))
    finally:
        c.shutdown()


def test_r1_a_broken_cleanup_module_pastes_as_heard_and_says_to_reinstall(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch)
    monkeypatch.setitem(sys.modules, "voxflow.cleanup", None)
    cfg = Config()
    Rec, asr, cleaner, notes = app.make_modules(cfg)
    assert isinstance(cleaner, app.BrokenModel) and notes == ["cleanup broken"]
    assert not cleaner.ready and not cleaner.has_llm and cleaner.clean("text") == "text"
    with pytest.raises(app.DamagedInstall, match="Error in cleanup.py"):
        cleaner.generate([], max_tokens=5, timeout_s=1)
    # the pipeline skips a cleaner that is not ready: the words are pasted as heard
    out = app.run_pipeline(np.zeros(16000, dtype=np.float32), type("A", (), {"transcribe": lambda s, a: "hi"})(),
                           cleaner)
    assert out["text"] == "hi"

    TRecorder.reset()
    monkeypatch.setattr(app, "make_modules", lambda c, fake, measured=None: (TRecorder, _ReadyAsr(), cleaner, []))
    c = _real_controller(tmp_path)
    try:
        c.start()
        qtbot.waitUntil(lambda: c.model_status("cleanup")["state"] == "failed", timeout=5000)
        assert "Reinstall VoxFlow" in c.model_status("cleanup")["detail"]
        assert "DamagedInstall" not in c.model_status("cleanup")["detail"]
    finally:
        c.shutdown()


class _ReadyAsr:
    backend = "cpu"

    def load(self):
        pass

    def transcribe(self, audio):
        return "hello"


def test_r1_a_broken_audio_module_gives_a_recorder_that_never_records(monkeypatch):
    monkeypatch.setattr(app, "_IMPORT_ERRORS", {})
    real = app._real
    monkeypatch.setattr(app, "_real", lambda mod, attr: real(mod, "NoSuchRecorder") if mod == "audio" else None)
    Rec, *_rest, notes = app.make_modules(Config())
    assert Rec is app.BrokenRecorder and notes[0] == "audio broken"
    rec = Rec("mic", max_seconds=3)
    assert "AttributeError in audio.py" in str(rec.error) and rec.alive is False
    for call in (rec.start, rec.reopen):
        with pytest.raises(app.DamagedInstall):
            call()
    assert rec.stop().size == 0 and rec.level() == 0.0 and rec.recent_samples(4).tolist() == [0, 0, 0, 0]
    rec.set_device("other")
    rec.close()
    assert rec.device == "other" and not rec.is_recording and not rec.hit_cap and rec.last_speech_seconds == 0
    assert str(app.damaged_install("nothing")) == "VoxFlow's installation is damaged (nothing.py is missing). " \
                                                  "Reinstall VoxFlow."


def test_r1_fake_mode_still_uses_the_stand_ins(monkeypatch):
    Rec, asr, cleaner, notes = app.make_modules(Config(), fake=True)
    assert notes == ["audio stub", "asr stub", "cleanup stub"]
