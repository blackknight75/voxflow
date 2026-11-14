"""Controller paths the other suites leave out: defensive branches around foreign recorders / cleaners, the
sync round's failure and postponement paths, start-up and shutdown robustness, save retries, and the pipeline
worker's handling of jobs that blow up. Everything runs on the scripted rig (tests/controller_rig.py)."""
from __future__ import annotations

import logging
import threading

import pytest

from tests.controller_rig import (ALT, WIN, TAsr, TAutostart, TCleaner, TInjector, TMixer, TRecorder, make_rig)
from voxflow import controller as ctl_mod
from voxflow import sync as syncmod
from voxflow.context import AppContext
from voxflow.controller import Controller


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


@pytest.fixture
def slow_sync(monkeypatch):
    """Sync timers that never fire on their own: every round in these tests is started explicitly."""
    monkeypatch.setattr(ctl_mod, "SYNC_INTERVAL_MS", 3_600_000)
    monkeypatch.setattr(ctl_mod, "SYNC_DEBOUNCE_MS", 3_600_000)


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "cloud"
    f.mkdir()
    return f


def actions(rig) -> list:
    seen: list = []
    rig.c.notice_action.connect(lambda message, action, label: seen.append((message, action, label)))
    return seen


def sync_idle(rig) -> bool:
    return not rig.c._sync_busy and rig.c.sync_status()["state"] in ("idle", "error")


def patch_os(monkeypatch, cached=True):
    """Real (non-fake) mode with every OS / hub lookup answered from here."""
    monkeypatch.setattr(ctl_mod.audiomod, "input_device_count", lambda: 1)
    monkeypatch.setattr(ctl_mod.audiomod, "list_input_devices", lambda: [(1, "Yeti Nano")])
    monkeypatch.setattr(ctl_mod.audiomod, "default_input_name", lambda: "Yeti Nano")
    monkeypatch.setattr(ctl_mod.audiomod, "rescan_devices", lambda: None)
    monkeypatch.setattr(ctl_mod.modelhub, "is_cached", lambda repo, filename=None: cached)


class _ReadOnlyAfterInit:
    """Mixin: ``locked`` attributes can be set once (by __init__) and never again."""
    locked: tuple = ()

    def __setattr__(self, name, value):
        if name in self.locked and name in self.__dict__:
            raise AttributeError(f"{name} is read-only")
        super().__setattr__(name, value)


class FrozenBoostRecorder(_ReadOnlyAfterInit, TRecorder):
    locked = ("level_boost",)


class FrozenTimeoutCleaner(_ReadOnlyAfterInit, TCleaner):
    locked = ("timeout_s",)


# =========================================================================== desktop back-end defaults
def test_real_autostart_delegates_to_the_system_module(monkeypatch):
    calls = []
    monkeypatch.setattr(ctl_mod.system, "autostart_enabled", lambda: True)
    monkeypatch.setattr(ctl_mod.system, "set_autostart", calls.append)
    real = ctl_mod._RealAutostart()
    assert real.enabled() is True
    real.set(False)
    assert calls == [False]


def test_real_mode_without_an_injector_builds_the_win32_one_from_the_settings(tmp_path, qapp):
    import json

    from voxflow.inject import Injector
    TRecorder.reset()
    (tmp_path / "config.json").write_text(json.dumps({"restore_clipboard": False, "restore_delay_s": 1.25}),
                                          encoding="utf-8")
    c = Controller(fake=False, data_dir=tmp_path, install_hook=False, modules=(TRecorder, TAsr(), TCleaner()),
                   sound_backend=lambda wav: None, mute_backend=TMixer(), context_fn=lambda: AppContext(),
                   autostart=TAutostart())
    try:
        assert isinstance(c.injector, Injector)
        assert c.injector.restore is False and c.injector.restore_delay == 1.25
        assert c.injector.settle == c.listener.settle, "pastes wait for the hook's menu-focus check"
    finally:
        c.shutdown()


def test_stand_in_modules_in_real_mode_are_logged_and_a_reload_rebuilds_the_models(tmp_path, qtbot, monkeypatch,
                                                                                    caplog):
    from voxflow import app as appmod
    patch_os(monkeypatch)
    TRecorder.reset()
    built = []

    class OtherRecorder(TRecorder):
        pass

    def make_modules(cfg, fake, measured=None):
        built.append((TAsr(), TCleaner()))
        return (TRecorder if len(built) == 1 else OtherRecorder), built[-1][0], built[-1][1], ["asr stub"]

    monkeypatch.setattr(appmod, "make_modules", make_modules)
    c = Controller(fake=False, data_dir=tmp_path, install_hook=False, injector=TInjector(),
                   sound_backend=lambda wav: None, mute_backend=TMixer(), context_fn=lambda: AppContext(),
                   autostart=TAutostart())
    try:
        with caplog.at_level(logging.WARNING, logger="voxflow.controller"):
            c.start()
        assert "running with stand-ins: asr stub" in caplog.text
        qtbot.waitUntil(lambda: c.model_status("asr")["state"] == "ready"
                        and c.model_status("cleanup")["state"] == "ready", timeout=5000)
        first_asr, first_cleaner = built[0]
        assert c._asr is first_asr and c.command.cleaner is first_cleaner

        recorder = c._recorder
        c.reload_models()
        second_asr, second_cleaner = built[1]
        assert c._asr is second_asr and c._cleaner is second_cleaner
        assert c.command.cleaner is second_cleaner, "command mode must talk to the new language model"
        assert c._Rec is TRecorder and c._recorder is recorder, "a model reload leaves the microphone alone"
        qtbot.waitUntil(lambda: second_asr.loads == 1 and second_cleaner.loads == 1
                        and c.model_status("cleanup")["state"] == "ready", timeout=5000)
        assert first_asr.loads == 1, "the replaced model is not loaded again"
    finally:
        c.shutdown()


def test_cleanup_model_download_reports_progress(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch, cached=False)

    class Watcher:
        def __init__(self, repo, callback, interval_s=0.5):
            self.repo, self.callback = repo, callback

        def __enter__(self):
            self.callback(50_000_000, 100_000_000)
            self.callback(75_000_000, None)                     # size unknown: no fraction
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ctl_mod.modelhub, "DownloadWatcher", Watcher)
    rig = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    try:
        states = [s for name, (n, s) in ((e[0], e[1]) for e in rig.startup_events
                                         if e[0] == "model_status_changed") if n == "cleanup"]
        assert states[0] == {"state": "downloading", "detail": "Downloading the cleanup model (one time)…",
                             "progress": None, "backend": ""}
        assert {"state": "downloading", "detail": "Downloading… 50 MB of 100 MB", "progress": 0.5,
                "backend": ""} in states
        assert {"state": "downloading", "detail": "Downloading… 75 MB", "progress": None, "backend": ""} in states
        assert states[-1]["state"] == "ready" and rig.cleaner.loads == 1
    finally:
        rig.close()


def test_a_cleaner_that_loads_but_is_not_ready_is_reported_as_failed(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.cleaner.load = lambda: None                         # returns without becoming ready
        rig.start()
        status = rig.c.model_status("cleanup")
        assert status["state"] == "failed"
        assert status["detail"] == "The cleanup model could not be loaded, so dictation is pasted as heard."
        assert rig.c.status()["level"] == "warning"
    finally:
        rig.close()


def test_a_hook_that_installs_is_not_a_problem(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False, install_hook=True)
    try:
        started = []
        monkeypatch.setattr(rig.c.listener, "start", lambda: started.append(True) or True)
        rig.start()
        assert started == [True] and "hook" not in rig.c._problems
        assert rig.c.status()["level"] == "ok"
    finally:
        rig.close()


# =========================================================================== foreign recorders / cleaners
def test_a_recorder_without_a_settable_level_boost_still_works(rig):
    rig.c._Rec = FrozenBoostRecorder
    rig.c.set("mic_device", "Yeti Nano")                        # builds a new recorder of that class
    assert isinstance(rig.rec, FrozenBoostRecorder) and rig.rec.device == "Yeti Nano"
    rig.spy.clear()
    rig.c.set("whisper_mode", True)                             # the boost cannot be applied: not an error
    assert rig.c.get("whisper_mode") is True and rig.rec.level_boost == 1.0
    assert ("whisper_mode", True) in rig.spy.of("config_changed") and rig.spy.notices() == []
    rig.dictate("still dictating")
    assert rig.injector.pasted == ["Still dictating."]


def test_a_cleaner_without_a_settable_timeout_keeps_the_setting(rig):
    locked = FrozenTimeoutCleaner()
    rig.c._cleaner = locked
    rig.c.set("cleanup_timeout_s", 9.0)
    assert rig.c.get("cleanup_timeout_s") == 9.0 and locked.timeout_s == 4.0


def test_changing_the_recording_cap_mid_recording_keeps_the_live_recorder(rig):
    rig.press()
    live = rig.rec
    rig.c.set("max_record_s", 60.0)
    assert rig.rec is live and live.is_recording and rig.c.state == "listening"
    assert rig.c.listener.machine.max_hold_s == 65.0 and rig.c.listener.binding("command").max_hold_s == 65.0
    rig.c.cancel()
    rig.c.set("max_record_s", 90.0)                             # idle: the recorder is rebuilt with the new cap
    assert rig.rec is not live and rig.rec.max_seconds == 90.0


# =========================================================================== shutdown / crash handler
class _BrokenRecorder:
    closed = False

    @property
    def is_recording(self):
        raise OSError("device handle is gone")

    def close(self):
        self.closed = True


def test_shutdown_completes_when_sync_a_store_and_the_recorder_all_fail(tmp_path, qtbot, monkeypatch, caplog):
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    flushed = []
    monkeypatch.setattr(rig.c.sync, "stamp_local", lambda: (_ for _ in ()).throw(OSError("state file locked")))
    monkeypatch.setattr(rig.c.dictionary, "flush", lambda: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(rig.c.snippets, "flush", lambda: flushed.append("snippets"))
    broken = rig.c._recorder = _BrokenRecorder()
    with caplog.at_level(logging.ERROR, logger="voxflow.controller"):
        rig.c.shutdown()
    assert "could not record pending sync changes" in caplog.text
    assert "could not flush dictionary.json" in caplog.text
    assert flushed == ["snippets"], "one failing store does not stop the others from being written"
    assert rig.injector.closed == 1 and broken.closed and not rig.c._worker.is_alive()


def test_emergency_restore_gives_the_clipboard_back_even_if_unmuting_fails(rig, monkeypatch):
    monkeypatch.setattr(rig.c.muter, "restore_now", lambda: (_ for _ in ()).throw(OSError("audio service down")))
    rig.c.emergency_restore()
    assert rig.injector.closed == 1


# =========================================================================== settings batches
def test_a_settings_batch_reports_unusable_values_and_applies_the_rest(rig):
    skipped = rig.c._apply_settings({"sound_volume": "loud", "sounds_enabled": False, "no_such_setting": 1})
    assert "sound_volume" in skipped and "sounds_enabled" not in skipped
    assert rig.c.get("sounds_enabled") is False and rig.c.sounds.enabled is False
    assert rig.c.get("sound_volume") == ctl_mod.cfgmod.defaults()["sound_volume"]


def test_a_settings_batch_never_leaves_two_actions_on_one_shortcut(rig):
    dictate = rig.c.get("hotkey")
    default_command = ctl_mod.cfgmod.defaults()["command_hotkey"]
    rig.c.set("command_hotkey", "ctrl+alt+k")
    skipped = rig.c._apply_settings({"command_hotkey": dictate})      # clashes with the dictation shortcut
    assert skipped == ["command_hotkey"]
    assert rig.c.get("hotkey") == dictate and rig.c.get("command_hotkey") == default_command
    rig.keys((ALT, True), (WIN, True))                          # the live binding follows the corrected value
    assert rig.c.state == "listening" and rig.c.mode == "command"
    rig.c.cancel()
    rig.keys((WIN, False), (ALT, False))


def test_a_refused_shortcut_that_also_ends_up_corrected_is_reported_once(rig):
    command = rig.c.get("command_hotkey")
    skipped = rig.c._apply_settings({"command_hotkey": 123, "hotkey": command})
    assert skipped == ["command_hotkey"], "refused as given, then moved off the dictation shortcut: one entry"
    assert rig.c.get("hotkey") == command and rig.c.get("command_hotkey") != command


# =========================================================================== sync rounds
def test_choosing_the_same_sync_folder_again_is_not_announced_twice(rig, slow_sync, folder):
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: sync_idle(rig))
    rig.spy.clear()
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: sync_idle(rig))
    assert [a for a in rig.spy.of("config_changed") if a[0] == "sync_folder"] == []


def test_sync_folder_chosen_before_start_syncs_once_without_starting_the_timers(tmp_path, qtbot, monkeypatch,
                                                                               slow_sync, folder):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.c.set_sync_folder(str(folder))
        assert not rig.c._sync_timer.isActive(), "the periodic round belongs to a started controller"
        rig.wait(lambda: sync_idle(rig) and rig.c.sync_status()["last_sync"] is not None)
        assert (folder / rig.c.sync.file_name).exists()
        rig.c.snippets.add("early", "added before start")       # a change before start() schedules nothing
        assert not rig.c._sync_soon.isActive() and not rig.c._sync_busy
    finally:
        rig.close()


def test_no_round_starts_when_sync_is_off_or_after_shutdown(tmp_path, qtbot, monkeypatch, slow_sync, folder):
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    events = []
    rig.c.sync_status_changed.connect(events.append)
    rig.c._sync_round()                                         # sync is off
    assert events == [] and not rig.c._sync_busy
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: sync_idle(rig))
    rig.close()
    events.clear()
    rig.c._sync_round()                                         # a late timer after shutdown
    assert events == [] and not rig.c._sync_busy


def test_a_sync_thread_that_outlives_the_controller_ends_quietly(rig, monkeypatch):
    class GoneSignal:
        def emit(self, *args):
            raise RuntimeError("Internal C++ object (Controller) already deleted.")

    crashes, ran = [], threading.Event()
    monkeypatch.setattr(threading, "excepthook", lambda args: crashes.append(args))
    rig.c._sig_sync = GoneSignal()
    try:
        rig.c._sync_thread("read", ran.set)
        assert ran.wait(5)
        for t in [t for t in threading.enumerate() if t.name == "voxflow-sync-read"]:
            t.join(5)
    finally:
        del rig.c._sig_sync                                     # back to the class-level signal
    assert crashes == []


def test_an_unexpected_sync_error_is_logged_and_shown_as_a_status(rig, slow_sync, folder, caplog):
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: sync_idle(rig))
    events = []
    rig.c.sync_status_changed.connect(events.append)
    rig.c._sync_busy = True
    with caplog.at_level(logging.ERROR, logger="voxflow.controller"):
        rig.c._on_sync_step("read", ValueError("bad json in a device file"))
    assert "sync read failed" in caplog.text
    assert not rig.c._sync_busy and events[-1]["state"] == "error" and rig.spy.notices() == []

    caplog.clear()
    rig.c._sync_busy = True
    with caplog.at_level(logging.ERROR, logger="voxflow.controller"):
        rig.c._on_sync_step("written", syncmod.SyncUnavailable("the folder is offline"))
    assert "failed" not in caplog.text, "an unreachable folder is expected, not an error in the log"
    assert events[-1]["state"] == "error" and not rig.c._sync_busy


def test_remote_data_read_during_a_recording_is_merged_on_the_next_round(rig, slow_sync, folder, monkeypatch):
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: sync_idle(rig))
    rig.c._sync_soon.stop()
    monkeypatch.setattr(rig.c.sync, "merge", lambda remote: pytest.fail("merged under a live recording"))
    rig.press()
    rig.c._sync_busy = True
    rig.c._on_sync_step("read", {})
    assert not rig.c._sync_busy and rig.c.sync_status()["state"] == "idle"
    assert rig.c._sync_soon.isActive() and not rig.c._sync_again, "the postponed round is rescheduled"
    rig.c.cancel()


def test_a_failing_merge_becomes_a_sync_error_and_the_next_round_can_run(rig, slow_sync, folder, monkeypatch, caplog):
    rig.c.set_sync_folder(str(folder))
    rig.wait(lambda: sync_idle(rig))
    monkeypatch.setattr(rig.c.sync, "merge", lambda remote: (_ for _ in ()).throw(KeyError("entries")))
    rig.c._sync_busy = True
    with caplog.at_level(logging.ERROR, logger="voxflow.controller"):
        rig.c._on_sync_step("read", {})
    assert "sync merge failed" in caplog.text
    assert rig.c.sync_status()["state"] == "error" and not rig.c._sync_busy and not rig.c._sync_applying
    monkeypatch.undo()
    rig.c.sync_now()
    rig.wait(lambda: sync_idle(rig) and rig.c.sync_status()["state"] == "idle")


# =========================================================================== microphone health (real mode)
@pytest.fixture
def real_rig(tmp_path, qtbot, monkeypatch):
    patch_os(monkeypatch)
    r = make_rig(tmp_path, qtbot, monkeypatch, fake=False)
    yield r
    r.close()


def test_a_stale_microphone_problem_is_cleared_once_the_recorder_is_healthy(real_rig):
    rig = real_rig
    rig.c._problems["mic"] = "The microphone could not be opened."
    assert rig.c.status()["title"] == "Microphone problem"
    rig.c._health_tick()
    assert "mic" not in rig.c._problems and rig.c.status()["level"] == "ok"
    assert rig.spy.of("status_changed")[-1][0]["level"] == "ok" and rig.rec.reopened == 0


def test_a_stream_that_died_quietly_is_reopened_without_a_notice(real_rig):
    rig = real_rig
    rig.rec.alive = False
    rig.c._health_tick()
    assert rig.rec.reopened == 1 and rig.rec.alive is True
    assert rig.spy.notices() == [] and rig.c.status()["level"] == "ok"


# =========================================================================== flow-bar state
def test_idle_timer_firing_late_does_not_touch_another_state(rig):
    rig.press()
    rig.spy.clear()
    rig.c._back_to_idle()                                       # a done/error timer that fires after a new start
    assert rig.c.state == "listening" and rig.spy.states() == []
    rig.c.cancel()


def test_fail_without_sound_shows_the_error_silently(rig):
    rig.c._fail("Didn't catch that", sound=False)
    assert rig.c.state == "error" and rig.c.message == "Didn't catch that" and rig.sounds == []
    assert rig.c.error_info()["kind"] == "no_speech"


def test_retry_saves_when_nothing_failed_says_nothing(rig):
    seen = actions(rig)
    out = rig.c.retry_saves()
    assert set(out) == {"history", "dictionary", "snippets", "stats", "styles", "settings"}
    assert all(v == {"ok": True, "error": ""} for v in out.values())
    assert rig.spy.notices() == [] and seen == []


def test_settings_that_could_not_be_saved_are_retried(rig, monkeypatch):
    import json
    seen = actions(rig)
    real_save = ctl_mod.cfgmod.save_config
    monkeypatch.setattr(ctl_mod.cfgmod, "save_config", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    rig.c.set("sound_volume", 0.25)
    assert rig.c._settings_unsaved and seen[-1][1:] == ("retry_save", "Try again")

    out = rig.c.retry_saves()                                   # still failing
    assert out["settings"] == {"ok": False, "error": "disk full"} and out["history"]["ok"]
    level, message = rig.spy.notices()[-1]
    assert level == "error" and message.startswith("Still couldn't save settings.")
    assert seen[-1] == (message, "retry_save", "Try again")

    monkeypatch.setattr(ctl_mod.cfgmod, "save_config",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert rig.c.retry_saves()["settings"] == {"ok": False, "error": "OSError"}

    monkeypatch.setattr(ctl_mod.cfgmod, "save_config", real_save)
    out = rig.c.retry_saves()
    assert out["settings"] == {"ok": True, "error": ""} and not rig.c._settings_unsaved
    assert rig.spy.notices()[-1] == ("success", "Everything is saved now.")
    assert json.loads((rig.data_dir / "config.json").read_text(encoding="utf-8"))["sound_volume"] == 0.25


# =========================================================================== hotkey events
def test_stray_hotkey_events_are_ignored(rig):
    for event in ("lock", "stop", "cancel", "no-such-event"):
        rig.c._on_hotkey("dictate", event)
    assert rig.c.state == "idle" and rig.spy.states() == [] and rig.rec.starts == 0 and rig.rec.stops == 0


def test_a_second_start_while_recording_or_while_paused_does_nothing(rig):
    rig.press()
    rig.c._on_hotkey("command", "start")
    assert rig.rec.starts == 1 and rig.c.mode == "dictate" and rig.c.state == "listening"
    rig.c.cancel()
    rig.c.set_paused(True)
    rig.c._on_hotkey("dictate", "start")
    assert rig.rec.starts == 1 and rig.c.state == "idle"


def test_context_capture_that_fails_or_returns_junk_is_an_unknown_app(rig):
    rig.c._context_fn = lambda: (_ for _ in ()).throw(OSError("access denied"))
    rig.dictate("typed into an elevated window")
    assert rig.injector.pasted == ["Typed into an elevated window."]
    assert rig.last_entry()["app_exe"] == "" and rig.c._last_paste["hwnd"] == 0
    rig.wait(lambda: rig.c.state == "idle")
    rig.c._context_fn = lambda: "notepad.exe"                   # not an AppContext
    rig.dictate("second one")
    assert rig.last_entry()["app_exe"] == ""


def test_cancelling_when_the_recorder_already_stopped_or_cannot_stop(rig, caplog):
    rig.press()
    rig.rec.is_recording = False                                # e.g. the cap ended it a moment ago
    rig.c.cancel()
    assert rig.rec.stops == 0 and rig.c.state == "idle" and not rig.mixer.muted

    rig.pause()
    rig.keys((WIN, False), (0xA2, False))
    rig.press()
    TRecorder.fail_stop = OSError("stream closed")
    with caplog.at_level(logging.WARNING, logger="voxflow.controller"):
        rig.c.cancel()
    assert "recorder stop failed while cancelling" in caplog.text
    assert rig.c.state == "idle" and not rig.c._active and rig.injector.pasted == []


def test_a_recorder_that_does_not_report_speech_time_falls_back_to_the_audio_length(rig):
    TRecorder.speech_seconds = None
    rig.dictate("two seconds of audio")
    assert rig.injector.pasted == ["Two seconds of audio."]
    assert rig.last_entry()["duration_s"] == pytest.approx(len(TRecorder.audio) / 16000, abs=0.01)


def test_undo_when_the_foreground_window_cannot_be_read(rig):
    rig.dictate("oops")
    rig.spy.clear()
    rig.c._hwnd_fn = lambda: (_ for _ in ()).throw(OSError("no desktop"))
    assert rig.c.undo_last_paste() is False and rig.injector.undone == 0
    assert "window you last dictated into" in rig.spy.notices()[0][1]
    assert rig.c._last_paste is not None, "the paste can still be undone once the window is back"


# =========================================================================== pipeline worker
def test_the_worker_skips_jobs_it_does_not_know(rig):
    rig.c._work.put({"kind": "mystery"})
    rig.dictate("still working")
    assert rig.injector.pasted == ["Still working."] and rig.c.state == "done"
    assert rig.spy.notices() == []


def test_a_crashing_dictation_job_is_reported_and_the_worker_survives(rig, monkeypatch, caplog):
    monkeypatch.setattr(rig.c.snippets, "expand", lambda text: (_ for _ in ()).throw(KeyError("corrupt snippet")))
    with caplog.at_level(logging.ERROR, logger="voxflow.controller"):
        rig.dictate("this one blows up")
    assert "pipeline job 'dictate' failed" in caplog.text
    assert rig.c.state == "error" and rig.c.message == "Something went wrong with that dictation"
    assert ("error", ctl_mod._JOB_FAILURES["dictate"]) in rig.spy.notices()
    assert all("KeyError" not in m for _l, m in rig.spy.notices())
    monkeypatch.undo()
    rig.wait(lambda: rig.c.state == "idle")
    rig.dictate("and this one works")
    assert rig.injector.pasted == ["And this one works."]


def test_a_crashing_command_job_says_nothing_was_changed(rig, monkeypatch):
    rig.injector.selection = "some text"
    monkeypatch.setattr(rig.c.dictionary, "apply", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    rig.dictate("make it formal", binding="command")
    assert rig.c.state == "error" and rig.injector.pasted == []
    assert ("error", ctl_mod._JOB_FAILURES["command"]) in rig.spy.notices()


def test_a_crashing_paste_or_undo_job_tells_the_user(rig, monkeypatch):
    """Regression: these jobs have no flow-bar turn (no ``gen``), so their failure used to be dropped silently."""
    rig.dictate("first")
    rig.wait(lambda: rig.c.state == "idle")
    rig.spy.clear()
    monkeypatch.setattr(rig.injector, "begin", lambda text: (_ for _ in ()).throw(OSError("clipboard exploded")))
    assert rig.c.paste_last() is True
    rig.wait(lambda: rig.spy.notices())
    assert rig.spy.notices() == [("error", ctl_mod._JOB_FAILURES["paste"])]
    assert rig.c.state == "idle", "the flow bar belongs to dictations"

    rig.spy.clear()
    monkeypatch.setattr(rig.injector, "undo", lambda: (_ for _ in ()).throw(OSError("SendInput failed")))
    assert rig.c.undo_last_paste() is True
    rig.wait(lambda: rig.spy.notices())
    assert rig.spy.notices() == [("error", ctl_mod._JOB_FAILURES["undo"])]


def test_speech_model_error_in_command_mode_leaves_the_selection_alone(rig):
    rig.injector.selection = "keep me"
    rig.asr.fail_transcribe = RuntimeError("CUDA error: device lost")
    rig.dictate("ignored", binding="command")
    assert rig.c.state == "error" and rig.injector.pasted == [] and rig.cleaner.generate_calls == []
    assert ("error", ctl_mod._SPEECH_FAILURE) in rig.spy.notices()
    assert rig.c.error_info()["kind"] == "model"


def test_command_without_a_known_window_skips_the_window_check(rig):
    rig.context = AppContext("notepad.exe", "Untitled - Notepad", 0)
    rig.injector.selection = "some text"
    rig.pause()
    rig.press("command")
    rig.c._hwnd_fn = lambda: pytest.fail("no window was captured, so there is nothing to compare with")
    rig.release("command")
    rig.wait(lambda: rig.c.state in ("done", "error"))
    assert rig.injector.pasted == ["REWRITTEN"] and rig.c.message == "Rewritten"


def test_command_pastes_when_the_foreground_window_cannot_be_read(rig):
    rig.injector.selection = "some text"
    rig.pause()
    rig.press("command")
    rig.c._hwnd_fn = lambda: (_ for _ in ()).throw(OSError("no desktop"))
    rig.release("command")
    rig.wait(lambda: rig.c.state in ("done", "error"))
    assert rig.injector.pasted == ["REWRITTEN"] and rig.injector.copied == []


def test_command_result_that_had_to_be_typed_says_so(rig):
    rig.injector.selection = "some text"
    rig.injector.outcome = "typed"
    rig.dictate("make it formal", binding="command")
    assert (rig.c.state, rig.c.message) == ("done", "Typed") and rig.injector.pasted == ["REWRITTEN"]


def test_a_line_break_that_could_not_be_pasted_is_a_plain_paste_error(rig):
    rig.injector.outcome = "failed"
    rig.dictate("new line")
    assert (rig.c.state, rig.c.message) == ("error", "Couldn't paste")
    assert rig.c.history.count() == 0 and rig.spy.of("transcript_ready") == []
    assert all("History" not in m for _l, m in rig.spy.notices()), "nothing was saved, so nothing is promised"


def test_a_failure_from_an_older_job_does_not_repaint_the_bar(rig):
    rig.press()
    rig.spy.clear()
    rig.c._on_failed("Didn't catch that", rig.c._gen - 1)
    assert rig.c.state == "listening" and rig.spy.states() == [] and rig.spy.notices() == []
    rig.c.cancel()


def test_unknown_store_topics_are_ignored(rig):
    rig.c._on_store_event("something-new")
    assert rig.spy.events == []

