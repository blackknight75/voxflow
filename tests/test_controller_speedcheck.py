"""The speed check inside the controller: when it runs, what it stores, how a result is applied and shown.

The child processes are replaced by a scripted ``run_child`` (no subprocess, model or GPU); the hardware
fingerprint by a fixed ``machine()``. Delays are shortened to milliseconds.
"""
from __future__ import annotations

import json
import logging
import queue
import threading

import pytest

from tests.controller_rig import make_rig
from voxflow import app as appmod, config as cfgmod, controller as ctl_mod, speedcheck, sync as syncmod
from voxflow.context import AppContext
from voxflow.controller import Controller


def ok(ms, auto=True):
    return {"available": True, "auto": auto, "ok": True, "median_ms": ms, "runs_ms": [ms] * 3}


NVIDIA = {"asr": {"cuda": ok(28.0), "webgpu": ok(63.0), "cpu": ok(143.0)},
          "cleanup": {"cuda": ok(76.0), "vulkan": ok(78.0), "cpu": ok(949.0)}}
IGPU = {"asr": {"cuda": {"available": False, "reason": "no NVIDIA driver (nvcuda.dll)"},
                "webgpu": ok(118.0, auto=False), "cpu": ok(402.0)},
        "cleanup": {"cuda": {"available": False, "reason": "no NVIDIA driver (nvcuda.dll)"},
                    "vulkan": {"available": True, "ok": False, "error": "TimeoutError: slow"}, "cpu": ok(2650.0)}}


class Children:
    """The scripted child processes: ``results`` answers, ``calls`` records, ``gate`` holds a child running."""

    def __init__(self, results=NVIDIA):
        self.results = results
        self.calls: list = []
        self.gate: threading.Event | None = None
        self.started = threading.Event()

    def __call__(self, model, backend, model_name, budget, cancel):
        self.calls.append((model, backend, model_name, budget))
        self.started.set()
        if self.gate is not None and not self.gate.wait(0) and backend == "cuda" and model == "asr":
            while not cancel.wait(0.005):
                if self.gate.is_set():
                    break
            if cancel.is_set():
                raise speedcheck.Cancelled()
        return speedcheck.clean_result(self.results[model][backend])


@pytest.fixture
def speed(monkeypatch):
    monkeypatch.setattr(speedcheck, "machine", lambda: {"cpu": "Test CPU", "gpus": [["Test GPU", "1.0"]]})
    monkeypatch.setattr(ctl_mod, "SPEED_DELAY_MS", 5)
    monkeypatch.setattr(ctl_mod, "SPEED_RETRY_MS", 5)
    children = Children()
    monkeypatch.setattr(speedcheck, "run_child", children)
    return children


def rig_with(tmp_path, qtbot, monkeypatch, **kw):
    kw.setdefault("speed_check", True)
    return make_rig(tmp_path, qtbot, monkeypatch, **kw)


def wait_done(rig, timeout=5000):
    rig.wait(lambda: rig.c._speed_state["state"] == "done", timeout)


# =========================================================================== when it runs
def test_the_first_start_measures_once_in_the_background_and_stores_the_result(tmp_path, qtbot, monkeypatch,
                                                                                 speed, caplog):
    with caplog.at_level(logging.INFO, logger="voxflow.controller"):
        rig = rig_with(tmp_path, qtbot, monkeypatch)
        try:
            wait_done(rig)
            assert [c[:2] for c in speed.calls] == [(m, b) for m in speedcheck.MODELS
                                                    for b in speedcheck.backends(m)]
            assert speed.calls[0][2:] == ("nemo-parakeet-tdt-0.6b-v3", 4.0)
            stored = speedcheck.load(rig.data_dir)
            assert stored["fingerprint"] == rig.c._speed_key()
            assert stored["static"] == {"asr": "cuda", "cleanup": "cuda"}
            assert stored["choice"] == {"asr": None, "cleanup": None}, "cuda was fastest: nothing changes"
            assert rig.asr.loads == 1 and rig.cleaner.loads == 1, "no reload when nothing changes"
        finally:
            rig.close()
    assert "speed check: speech: cuda 28 ms, webgpu 63 ms, cpu 143 ms -> keep the automatic choice (cuda)" \
        in caplog.text
    assert "speed check: language model: cuda 76 ms, vulkan 78 ms, cpu 949 ms" in caplog.text

    speed.calls.clear()
    rig = rig_with(tmp_path, qtbot, monkeypatch)                  # second start, same PC
    try:
        rig.settle(80)
        assert speed.calls == [], "a PC is measured once, not at every start"
        assert rig.c.device_info()["check"]["measured_at"] == stored["measured_at"]
        assert rig.c._speed_state["state"] == "idle"
    finally:
        rig.close()


def test_a_changed_fingerprint_or_a_broken_file_means_measuring_again(tmp_path, qtbot, monkeypatch, speed):
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    wait_done(rig)
    rig.close()
    for change in ("driver", "corrupt", "model"):
        speed.calls.clear()
        if change == "driver":
            monkeypatch.setattr(speedcheck, "machine", lambda: {"cpu": "Test CPU", "gpus": [["Test GPU", "2.0"]]})
        elif change == "corrupt":
            speedcheck.cache_path(rig.data_dir).write_text("{broken", encoding="utf-8")
        cfg = {"asr_model": "other-model"} if change == "model" else None
        rig = rig_with(tmp_path, qtbot, monkeypatch, config=cfg)
        try:
            wait_done(rig)
            assert len(speed.calls) == 6, change
        finally:
            rig.close()


def test_it_never_starts_while_a_dictation_is_under_way(tmp_path, qtbot, monkeypatch, speed):
    monkeypatch.setattr(ctl_mod, "SPEED_DELAY_MS", 60_000)
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        assert rig.c._speed_timer.isActive() and rig.c._speed_state["state"] == "waiting"
        rig.press()                                               # listening
        rig.c._speed_tick()
        assert speed.calls == [] and rig.c._speed_state["state"] == "waiting"
        rig.c._speed_timer.stop()
        rig.c.state = "processing"                                # the worker has the dictation
        rig.c._active = False
        rig.c._speed_tick()
        assert speed.calls == []
        rig.c.state = "idle"
        worker_queue, rig.c._work = rig.c._work, queue.Queue()  # a job still queued (no worker reads this one)
        rig.c._work.put({"kind": "dictate"})
        rig.c._speed_tick()
        assert speed.calls == []
        rig.c._work = worker_queue
        rig.c._speed_tick()
        wait_done(rig)
        assert len(speed.calls) == 6
    finally:
        rig.close()


def test_a_dictation_stops_a_running_check_which_resumes_when_idle(tmp_path, qtbot, monkeypatch, speed, caplog):
    speed.gate = threading.Event()                                # the first child keeps running
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        assert speed.started.wait(5)
        rig.wait(lambda: rig.c._speed_state["state"] == "running")
        with caplog.at_level(logging.INFO, logger="voxflow.controller"):
            rig.dictate("hello there")                            # the child is killed when recording starts
            rig.wait(lambda: "speed check postponed" in caplog.text)
        assert rig.injector.pasted == ["Hello there."]
        speed.gate.set()
        wait_done(rig)
        assert [c[:2] for c in speed.calls].count(("asr", "cuda")) == 2, "started over from the beginning"
        assert speedcheck.load(rig.data_dir) is not None
    finally:
        rig.close()


def test_the_override_wins_and_switches_the_automatic_check_off(tmp_path, qtbot, monkeypatch, speed):
    monkeypatch.setenv("VOXFLOW_DEVICE", "cpu")
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        rig.settle(60)
        assert speed.calls == [] and not rig.c._speed_timer.isActive()
        info = rig.c.device_info()
        assert info["forced"] == "cpu" and info["asr"]["next"] == "cpu" and info["cleanup"]["next"] == "cpu"
        rig.c.measure_devices()                                    # asked for: it still measures (and stores)
        wait_done(rig)
        assert len(speed.calls) == 6
    finally:
        rig.close()


def test_without_the_real_models_there_is_nothing_to_measure(tmp_path, qtbot, monkeypatch, speed):
    rig = make_rig(tmp_path, qtbot, monkeypatch)                  # the rig's stand-in models: no speed check
    try:
        rig.settle(40)
        assert speed.calls == [] and rig.c.device_info()["check"]["available"] is False
        with pytest.raises(ValueError, match="real models"):
            rig.c.measure_devices()
    finally:
        rig.close()


def test_the_default_follows_the_real_models_and_the_environment(tmp_path, monkeypatch, speed):
    kw = dict(data_dir=tmp_path, install_hook=False, injector=None, sound_backend=lambda w: None,
              context_fn=lambda: AppContext(), autostart=None)
    monkeypatch.setattr(appmod, "make_modules", lambda cfg, fake, measured=None: (object, object(), object(), []))
    monkeypatch.setenv("VOXFLOW_SPEED_CHECK", "1")
    from tests.controller_rig import TAutostart, TInjector, TMixer
    kw.update(injector=TInjector(), mute_backend=TMixer(), autostart=TAutostart())
    monkeypatch.setattr(ctl_mod.Controller, "_make_recorder", lambda self, device: None)
    real = Controller(**kw)
    fake = Controller(fake=True, **kw)
    monkeypatch.setenv("VOXFLOW_SPEED_CHECK", "0")
    off = Controller(**kw)
    try:
        assert (real._speed_auto, fake._speed_auto, off._speed_auto) == (True, False, False)
    finally:
        for c in (real, fake, off):
            c._closed = True                                      # never started: nothing to shut down


# =========================================================================== applying a result
def test_a_faster_speech_device_is_used_at_once_and_the_language_model_at_the_next_start(tmp_path, qtbot,
                                                                                          monkeypatch, speed):
    speed.results = IGPU
    rig = rig_with(tmp_path, qtbot, monkeypatch, start=False)
    rig.asr.device = "auto"                                       # what the speech model was built with
    rig.start()
    try:
        wait_done(rig)
        rec = speedcheck.load(rig.data_dir)
        assert rec["static"] == {"asr": "cpu", "cleanup": "cpu"} and rec["choice"] == {"asr": "webgpu",
                                                                                       "cleanup": None}
        rig.wait(lambda: rig.asr.loads == 2)
        assert rig.cleaner.loads == 1, "only the speech model is reloaded"
        assert rig.c._measured_devices() == {"asr": "webgpu", "cleanup": None}
        info = rig.c.device_info()
        assert info["asr"]["auto"] == "webgpu" and info["asr"]["fastest"] == "webgpu"
        cuda = info["asr"]["options"][0]
        assert cuda == {"id": "cuda", "measured": True, "available": False, "reason": "No NVIDIA driver",
                        "ms": None, "failed": False}
        vulkan = info["cleanup"]["options"][1]
        assert vulkan["failed"] and vulkan["available"] and vulkan["ms"] is None
    finally:
        rig.close()


def test_the_measured_choice_reaches_the_models_of_the_next_start(tmp_path, monkeypatch, speed):
    rec = speedcheck.make_record("", {"cpu": "Test CPU", "gpus": [["Test GPU", "1.0"]]},
                                 {"asr": "nemo-parakeet-tdt-0.6b-v3", "cleanup": ""}, IGPU, 1.0)
    rec["choice"]["cleanup"] = "vulkan"
    rec["fingerprint"] = speedcheck.fingerprint(rec["machine"], rec["models"])
    speedcheck.save(tmp_path, rec)
    seen = []
    monkeypatch.setattr(appmod, "make_modules",
                        lambda cfg, fake, measured=None: seen.append(measured) or (object, object(), object(), []))
    monkeypatch.setattr(ctl_mod.Controller, "_make_recorder", lambda self, device: None)
    from tests.controller_rig import TAutostart, TInjector, TMixer
    c = Controller(data_dir=tmp_path, install_hook=False, injector=TInjector(), sound_backend=lambda w: None,
                   mute_backend=TMixer(), context_fn=lambda: AppContext(), autostart=TAutostart(), speed_check=True)
    c._closed = True
    assert seen == [{"asr": "webgpu", "cleanup": "vulkan"}]


def test_a_switch_waits_for_the_dictation_and_ends_when_nothing_differs(tmp_path, qtbot, monkeypatch, speed):
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        wait_done(rig)
        rig.c._apply_measured()                                   # the rig's TAsr says nothing about its device
        rig.asr.device = "auto"
        rig.c._apply_measured()                                   # same device: nothing to do
        assert rig.asr.loads == 1
        rig.c._speed_record["choice"]["asr"] = "webgpu"
        rig.press()
        rig.c._apply_measured()                                   # busy: later
        assert rig.asr.loads == 1
        rig.release()
        rig.wait(lambda: rig.asr.loads == 2)
        rig.c._closed = True
        rig.c._apply_measured()
    finally:
        rig.c._closed = False
        rig.close()


def test_choosing_a_language_model_device_is_saved_and_applies_at_the_next_start(tmp_path, qtbot, monkeypatch,
                                                                                  speed):
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        wait_done(rig)
        changes = []
        rig.c.devices_changed.connect(changes.append)
        rig.c.set("cleanup_device", "vulkan")
        assert rig.cleaner.loads == 1, "llama.cpp cannot change its build in a running process"
        assert cfgmod.load_config(rig.data_dir / "config.json").cleanup_device == "vulkan"
        info = rig.c.device_info()["cleanup"]
        assert (info["setting"], info["current"], info["next"], info["restart"]) == ("vulkan", "cuda", "vulkan", True)
        rig.c.set("cleanup_device", "auto")
        assert rig.c.device_info()["cleanup"]["restart"] is False
    finally:
        rig.close()


def test_a_result_that_cannot_be_saved_is_still_used_until_the_app_closes(tmp_path, qtbot, monkeypatch, speed,
                                                                         caplog):
    def refuse(data_dir, record):
        raise OSError("disk full")
    monkeypatch.setattr(speedcheck, "save", refuse)
    with caplog.at_level(logging.WARNING, logger="voxflow.controller"):
        rig = rig_with(tmp_path, qtbot, monkeypatch)
        try:
            wait_done(rig)
            assert rig.c._speed_valid() is not None and "could not save the speed check result" in caplog.text
        finally:
            rig.close()


def test_reloading_one_model_keeps_the_other_one_loaded(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import TAsr, TAutostart, TCleaner, TInjector, TMixer, TRecorder
    TRecorder.reset()
    built = []

    def make_modules(cfg, fake, measured=None):
        built.append((TAsr(), TCleaner()))
        return TRecorder, built[-1][0], built[-1][1], []

    monkeypatch.setattr(appmod, "make_modules", make_modules)
    c = Controller(fake=True, data_dir=tmp_path, install_hook=False, injector=TInjector(),
                   sound_backend=lambda w: None, mute_backend=TMixer(), context_fn=lambda: AppContext(),
                   autostart=TAutostart())
    try:
        c.start()
        qtbot.waitUntil(c._models_settled, timeout=5000)
        asr0, cleaner0 = built[0]
        c.reload_models(("asr",))
        assert c._asr is built[1][0] and c._cleaner is cleaner0 and c.command.cleaner is cleaner0
        qtbot.waitUntil(lambda: built[1][0].loads == 1 and c._models_settled(), timeout=5000)
        c.reload_models(["cleanup"])
        assert c._asr is built[1][0] and c._cleaner is built[2][1] and c.command.cleaner is built[2][1]
        qtbot.waitUntil(lambda: built[2][1].loads == 1 and c._models_settled(), timeout=5000)
        assert (asr0.loads, cleaner0.loads, built[1][0].loads, built[2][0].loads) == (1, 1, 1, 0)
    finally:
        c.shutdown()


# =========================================================================== asking for it, and the odd moments
def test_measure_again_runs_now_and_a_reload_postpones_it(tmp_path, qtbot, monkeypatch, speed):
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        wait_done(rig)
        speed.calls.clear()
        speed.gate, speed.started = threading.Event(), threading.Event()
        rig.c.measure_devices()
        rig.wait(lambda: speed.started.is_set() and rig.c._speed.running)
        rig.c.measure_devices()                                   # already running: a no-op
        rig.c.reload_models()                                     # a reload cancels it ...
        rig.wait(lambda: rig.c._speed_state["state"] in ("waiting", "running") and len(speed.calls) >= 2, 5000)
        speed.gate.set()
        wait_done(rig)                                            # ... and it resumes once the models are back
        assert speed.calls.count(speed.calls[0]) >= 2
    finally:
        rig.close()


def test_a_request_while_the_models_load_waits_for_them(tmp_path, qtbot, monkeypatch, speed):
    monkeypatch.setattr(ctl_mod, "SPEED_DELAY_MS", 60_000)
    rig = rig_with(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.c.measure_devices()
        assert rig.c._speed_state["state"] == "waiting" and speed.calls == []
        rig.c._speed_tick()                                       # models not ready yet: nothing happens
        assert speed.calls == []
        rig.start()
        wait_done(rig)
        assert len(speed.calls) == 6
    finally:
        rig.close()


def test_a_failed_speech_model_is_not_measured_and_events_after_closing_are_ignored(tmp_path, qtbot,
                                                                                     monkeypatch, speed):
    rig = rig_with(tmp_path, qtbot, monkeypatch, start=False)
    rig.asr.fail_load = RuntimeError("no model")
    rig.start()
    try:
        rig.settle(40)
        assert speed.calls == [] and not rig.c._speed_timer.isActive()
    finally:
        rig.close()
    rig.c._on_speed({"event": "cancelled"})                       # arrives after shutdown: ignored
    rig.c._speed_tick()
    rig.c._speed_consider()
    assert not rig.c._speed_timer.isActive()


def test_shutdown_stops_a_running_check_and_waits_for_its_child(tmp_path, qtbot, monkeypatch, speed):
    speed.gate = threading.Event()
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    assert speed.started.wait(5)
    rig.close()
    assert not rig.c._speed.running


def test_demo_mode_shows_a_result_and_measures_with_simulated_children(tmp_path, qtbot, monkeypatch):
    import functools
    monkeypatch.setattr(speedcheck, "demo_child", functools.partial(speedcheck.demo_child, delay_s=0.0))
    c = Controller(demo=True)
    try:
        c.start()
        qtbot.waitUntil(lambda: c._models_settled(), timeout=5000)
        info = c.device_info()
        assert info["asr"]["auto"] == "cuda" and info["asr"]["options"][0]["ms"] == 29.0
        assert info["check"]["measured_at"] is not None and not c._speed_timer.isActive(), "demo: never automatic"
        c.measure_devices()
        qtbot.waitUntil(lambda: c._speed_state["state"] == "done", timeout=5000)
        assert not speedcheck.cache_path(c.data_dir).exists(), "the demo stores nothing"
    finally:
        c.shutdown()


# =========================================================================== stays on this PC
def test_the_result_never_travels_in_a_backup_a_pack_or_the_sync_folder(tmp_path, qtbot, monkeypatch, speed):
    assert "cleanup_device" in syncmod.LOCAL_SETTINGS and "cleanup_device" not in syncmod.portable_settings()
    rig = rig_with(tmp_path, qtbot, monkeypatch)
    try:
        wait_done(rig)
        assert speedcheck.cache_path(rig.data_dir).exists()
        backup = tmp_path / "all.voxbackup"
        rig.c.export_backup(str(backup))
        rig.c.dictionary.add("Kubernetes", [])
        pack = tmp_path / "words.voxpack"
        rig.c.export_pack("dictionary", str(pack))
        folder = tmp_path / "sync"
        folder.mkdir()
        rig.c.set_sync_folder(str(folder))
        rig.c.sync_now()
        rig.wait(lambda: any(folder.iterdir()))
        rig.settle(100)
        fingerprint = rig.c._speed_key()
        for path in [backup, pack, *folder.iterdir()]:
            text = path.read_text(encoding="utf-8")
            assert fingerprint not in text and "median_ms" not in text and "speedcheck" not in text, path.name
        snap = json.loads(backup.read_text(encoding="utf-8"))
        assert "speed" not in json.dumps(snap.get("settings", {}))
    finally:
        rig.close()
