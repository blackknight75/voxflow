"""Regression tests for the findings of the review round (tester, performance and design reviews).

Each test asserts the EXPECTED behaviour; verified findings are plain guards, a finding still open carries
``@open_("T-xx")`` (a non-strict xfail). Only public entry points are used.
"""
from __future__ import annotations

import csv
import json
import os
import time

import pytest

from controller_rig import make_rig
from voxflow import config as cfgmod
from voxflow import store
from voxflow.cleanup import rule_clean, smart_format


def open_(fid: str):
    return pytest.mark.xfail(reason=f"open review finding {fid}", strict=False)


def clean(text: str) -> str:
    return smart_format(rule_clean(text))


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


def _refuse(path, text):
    raise OSError(28, "No space left on device")


# ============================================================================ T-01 silent save failures
def test_t01_a_failed_store_save_tells_the_user(rig, monkeypatch):
    monkeypatch.setattr(cfgmod, "_atomic_write", _refuse)
    rig.spy.clear()
    rig.c.dictionary.add("Kubernetes")
    rig.dictate("hello there general")
    rig.settle(100)
    notes = [(lvl, msg) for lvl, msg in rig.spy.notices() if lvl in ("warning", "error")]
    assert any("sav" in msg.lower() for _lvl, msg in notes), rig.spy.notices()


def test_t01_save_failure_notice_is_rate_limited(rig, monkeypatch):
    monkeypatch.setattr(cfgmod, "_atomic_write", _refuse)
    rig.spy.clear()
    for i in range(10):
        rig.c.dictionary.add(f"term{i}")
    rig.settle(100)
    saves = [m for _lvl, m in rig.spy.notices() if "sav" in m.lower()]
    assert 1 <= len(saves) <= 2, saves


# ============================================================================ T-02 lone surrogates
def test_t02_history_survives_a_lone_surrogate(tmp_path):
    h = store.History(tmp_path / "h.json")
    h.add("good one")
    h.add("bad \ud800 text")          # must not raise
    h.add("good two")
    assert store.History(tmp_path / "h.json").count() == 3, "every entry must reach the disk"


def test_t02_dictionary_survives_a_lone_surrogate(tmp_path):
    d = store.Dictionary(tmp_path / "d.json")
    try:
        d.add("\ud800")
    except ValueError:
        pass                            # refusing it is fine too
    d.add("Kubernetes")
    assert "Kubernetes" in [e["term"] for e in store.Dictionary(tmp_path / "d.json").list()]


def test_t02_a_surrogate_in_the_text_does_not_turn_a_paste_into_an_error(rig):
    rig.dictate("hello \ud800 world")
    assert rig.injector.pasted
    assert "error" not in rig.spy.states()
    rig.dictate("second one is fine")
    assert rig.c.history.count() >= 2
    assert rig.c.stats.summary()["sessions"] >= 1


def test_t02_a_store_failure_after_the_paste_never_reports_an_error(rig, monkeypatch):
    def boom(entry):
        raise RuntimeError("boom")
    monkeypatch.setattr(rig.c.history, "add_entry", boom)
    rig.dictate("hello there general")
    assert rig.injector.pasted == ["Hello there general."]
    assert "error" not in rig.spy.states()


# ============================================================================ T-03 .. T-05, T-15 rule fixes
@pytest.mark.parametrize("spoken,want", [
    ("version three point two point one", "Version 3.2.1."),
    ("version two point five point one", "Version 2.5.1."),
])
def test_t03_version_numbers_are_whole(spoken, want):
    out = clean(spoken)
    assert "point" not in out, out
    assert out == want


def test_t03_plain_decimals_still_work():
    assert clean("pi is three point one four") == "Pi is 3.14."
    assert clean("twelve point five percent").startswith("12.5%")


@pytest.mark.parametrize("spoken,want", [
    ("see you on monday", "See you on Monday."),
    ("let's meet next friday at noon", "Let's meet next Friday at noon."),
    ("it happens in january", "It happens in January."),
    ("on tuesday and wednesday", "On Tuesday and Wednesday."),
])
def test_t04_weekdays_and_months_are_capitalised(spoken, want):
    assert clean(spoken) == want


def test_t04_the_modal_verb_may_is_left_alone():
    assert clean("i may go home") == "I may go home."


@pytest.mark.parametrize("spoken,want", [
    ("five dollars and fifty cents", "$5.50"),
    ("three dollars fifty", "$3.50"),
])
def test_t05_dollars_and_cents_are_combined(spoken, want):
    assert want in clean(spoken)


def test_t06_rule_clean_is_linear_on_long_unpunctuated_text():
    text = "hello world um " * 2000                  # 30 000 characters, no sentence end
    t0 = time.perf_counter()
    rule_clean(text)
    assert time.perf_counter() - t0 < 0.5


def test_t06_scratch_that_still_works_after_the_guard():
    assert clean("scratch that I mean hello") == "I mean hello."
    assert clean("we leave at noon scratch that we leave at one") == "We leave at one."


def test_t07_zero_width_only_text_is_empty():
    assert rule_clean("​‍﻿") == ""


def test_t07_a_zero_width_only_transcript_is_not_pasted(rig):
    rig.dictate("​")
    assert rig.injector.pasted == []
    assert rig.c.state in ("error", "idle")


def test_t08_nul_does_not_become_a_bullet():
    out = rule_clean("a\x00b um c")
    assert "- " not in out and "\n" not in out, repr(out)


def test_t15_p_m_and_a_m_are_treated_alike():
    assert clean("I will see you at 5 p m").startswith("I will see you at 5 PM")
    assert clean("seven a m").startswith("7 AM")


def test_t15_a_stray_er_is_a_filler():
    assert not clean("er hmm well okay then").lower().startswith("er ")


# ============================================================================ T-09 .. T-14 stores
@pytest.mark.parametrize("lead", ["=", "+", "-", "@"])
def test_t09_csv_export_neutralises_formulas(tmp_path, lead):
    h = store.History(tmp_path / "h.json")
    h.add(f'{lead}HYPERLINK("http://x")')
    out = tmp_path / "e.csv"
    h.export(str(out), "csv")
    rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert not rows[1][7].startswith(lead), rows[1][7]


def test_t09_json_export_keeps_the_text_verbatim(tmp_path):
    h = store.History(tmp_path / "h.json")
    h.add("=1+1")
    h.export(str(tmp_path / "e.json"), "json")
    assert json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))[0]["text"] == "=1+1"


@pytest.mark.parametrize("ts", [-1e10, 1e13, float("nan"), float("inf")])
def test_t10_stats_record_survives_an_absurd_clock(tmp_path, ts):
    s = store.Stats(tmp_path / "s.json")
    s.record(3, 1.0, "a.exe", ts)          # must not raise
    s.record(4, 1.0, "a.exe")
    assert s.summary()["total_words"] >= 4


def test_t11_future_entries_do_not_outlive_the_retention(tmp_path):
    now = [1_800_000_000.0]
    h = store.History(tmp_path / "h.json")
    h._clock = lambda: now[0]
    h.add_entry(store.make_entry("future", ts=now[0] + 86400 * 365 * 5))
    h.add_entry(store.make_entry("today", ts=now[0]))
    h.configure(retention_days=1)
    h.prune()
    now[0] += 3 * 86400
    h.prune()
    assert h.count() == 0, [e["text"] for e in h.snapshot()]


def test_t12_control_characters_never_reach_the_dictionary(tmp_path):
    d = store.Dictionary(tmp_path / "d.json")
    try:
        d.add("x\x00y")
    except ValueError:
        return
    assert all("\x00" not in e["term"] for e in d.list())


def test_t13_bulk_paths_save_once_and_a_single_add_stays_cheap(tmp_path):
    """No UI path adds words one by one in a loop: imports, packs and sync use ``apply_changes`` (one save).
    A single add rewrites the file (and fsyncs), which is fine at a few ms; write-behind was rejected on purpose
    because the file must be current right after every change."""
    d = store.Dictionary(tmp_path / "d.json")
    t0 = time.perf_counter()
    d.apply_changes(upserts=[{"term": f"term{i}", "aliases": [f"alias {i} x"]} for i in range(2000)])
    assert time.perf_counter() - t0 < 1.0
    assert len(store.Dictionary(tmp_path / "d.json").list()) == 2000
    t0 = time.perf_counter()
    for i in range(10):
        d.add(f"single{i}")
    assert (time.perf_counter() - t0) / 10 < 0.05


def test_t13_history_add_stays_cheap_with_a_full_history(tmp_path):
    h = store.History(tmp_path / "h.json")
    h.configure(limit=5000)
    h.import_entries([{"text": "w " * 20, "raw": "r", "ts": 1_700_000_000 + i} for i in range(5000)])
    t0 = time.perf_counter()
    for _ in range(10):
        h.add("a new dictation of some length")
    assert (time.perf_counter() - t0) / 10 < 0.012          # was about 27 ms each


def test_t14_atomic_write_flushes_to_disk_before_the_swap(tmp_path, monkeypatch):
    calls = []
    real = os.fsync

    def spy(fd):
        calls.append(fd)
        real(fd)
    monkeypatch.setattr(os, "fsync", spy)
    cfgmod._atomic_write(tmp_path / "x.json", "{}")
    assert calls, "os.fsync was never called"
    assert (tmp_path / "x.json").read_text() == "{}"


# ============================================================================ T-16 instance channel
def test_t16_instance_pipe_name_depends_on_the_profile_folder(tmp_path, monkeypatch):
    from voxflow import app
    monkeypatch.setenv("APPDATA", str(tmp_path / "a"))
    first = app._instance_name()
    monkeypatch.setenv("APPDATA", str(tmp_path / "b"))
    assert app._instance_name() != first


# ============================================================================ P1 / P4 / P7 / P8 / P11 / P12 (perf round)
def test_p1_the_private_cuda_cache_is_configured_before_cuda_starts(tmp_path):
    import subprocess
    import sys
    code = "import os, voxflow.modelhub; print(os.environ['CUDA_CACHE_PATH']); print(os.environ['CUDA_CACHE_MAXSIZE'])"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA_CACHE")}
    env["APPDATA"], env["LOCALAPPDATA"] = str(tmp_path / "roaming"), str(tmp_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env={**env, "PYTHONPATH": root},
                         capture_output=True, text=True, timeout=60)
    path, size = out.stdout.split()
    assert path == str(tmp_path / "VoxFlow" / "cuda-cache"), out          # Local AppData, never Roaming (QA R16)
    assert int(size) >= 1 << 30
    env["CUDA_CACHE_PATH"] = str(tmp_path / "mine")                       # an explicit setting always wins
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env={**env, "PYTHONPATH": root},
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.split()[0] == str(tmp_path / "mine")


def test_p4_the_gpu_arena_allocates_what_is_asked(monkeypatch):
    import sys
    import types

    from voxflow import asr
    seen = {}
    ort = types.ModuleType("onnxruntime")
    ort.set_default_logger_severity = lambda level: None
    ort.SessionOptions = object
    onnx_asr = types.ModuleType("onnx_asr")
    onnx_asr.load_model = lambda name, **kw: seen.update(kw) or "model"
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)
    asr._build_cuda("model-name")
    cuda = seen["providers"][0]
    assert cuda[0] == "CUDAExecutionProvider" and cuda[1]["arena_extend_strategy"] == "kSameAsRequested"


def test_p12_the_model_is_not_memory_mapped_when_it_lives_in_vram(monkeypatch):
    import sys
    import types

    from voxflow.cleanup import Cleaner
    calls = []
    llama = types.ModuleType("llama_cpp")
    llama.Llama = lambda **kw: calls.append(kw) or object()
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)
    c = Cleaner()
    c._build("x.gguf", True)
    c._build("x.gguf", False)
    assert calls[0]["use_mmap"] is False and calls[1]["use_mmap"] is True


def test_p7_the_level_clock_only_runs_while_the_mic_test_does(rig):
    c = rig.c
    heard = []
    c.level_changed.connect(heard.append)
    rig.press()
    assert c.state == "listening" and not c._level_timer.isActive(), "no 30 Hz wake-ups just for dictating"
    rig.release()
    rig.wait(lambda: c.state == "idle")
    c.set_level_monitor(True)
    assert c._level_timer.isActive()
    rig.wait(lambda: len(heard) > 2)
    c.set_level_monitor(False)
    assert not c._level_timer.isActive()
    assert c.level() == pytest.approx(0.42), "level() stays available for the Flow bar"


def test_p8_the_flow_bar_hears_done_before_the_history_is_written(rig):
    order = []
    rig.c.state_changed.connect(lambda state, *_a: order.append(state) if state == "done" else None)
    rig.c.history_changed.connect(lambda: order.append("history"))
    rig.dictate("hello there general")
    rig.wait(lambda: "history" in order)
    assert order.index("done") < order.index("history"), order
    assert rig.c.history.count() == 1 and rig.c.stats.summary()["sessions"] == 1


def test_p11_keyword_probe_is_computed_once_per_function(monkeypatch):
    import inspect

    from voxflow import app

    def clean(text, *, tone=None, terms=None):
        return text
    calls = []
    real = inspect.signature
    monkeypatch.setattr(app.inspect, "signature", lambda fn: calls.append(fn) or real(fn))
    assert app._accepts_keywords(clean, ("tone", "terms"))
    assert app._accepts_keywords(clean, ("tone", "terms"))
    assert app._accepts_keywords(clean, ("tone", "terms"))
    assert len(calls) == 1


# ============================================================================ T-13 / P3 coalesced writes: never lose a change
def test_t13_the_first_change_is_on_disk_when_the_call_returns(tmp_path):
    d = store.Dictionary(tmp_path / "d.json")
    d.add("Kubernetes")
    assert [e["term"] for e in store.Dictionary(tmp_path / "d.json").list()] == ["Kubernetes"]


def test_t13_a_burst_of_changes_is_complete_after_flush_and_after_the_quiet_time(tmp_path):
    d = store.Dictionary(tmp_path / "d.json")
    for i in range(50):
        d.add(f"term{i}")
    d.flush()
    assert len(store.Dictionary(tmp_path / "d.json").list()) == 50
    for i in range(50, 60):
        d.add(f"term{i}")
    deadline = time.time() + 3
    while time.time() < deadline and len(store.Dictionary(tmp_path / "d.json").list()) < 60:
        time.sleep(0.05)
    assert len(store.Dictionary(tmp_path / "d.json").list()) == 60, "the trailing write must fire by itself"


def test_t13_batch_writes_once_at_the_end(tmp_path, monkeypatch):
    writes = []
    real = cfgmod._atomic_write
    monkeypatch.setattr(cfgmod, "_atomic_write", lambda p, t: (writes.append(1), real(p, t))[1])
    d = store.Dictionary(tmp_path / "d.json")
    with d.batch():
        for i in range(30):
            d.add(f"term{i}")
        assert not writes
    assert len(writes) == 1
    assert len(store.Dictionary(tmp_path / "d.json").list()) == 30


def test_t13_shutdown_flushes_pending_store_writes(rig):
    c = rig.c
    for i in range(40):
        c.dictionary.add(f"term{i}")
    c.shutdown()
    assert len(store.Dictionary(c.data_dir / "dictionary.json").list()) == 40


def test_t13_a_failed_trailing_write_is_reported_and_retried(rig, monkeypatch):
    c = rig.c
    c.dictionary.add("first")
    real = cfgmod._atomic_write
    monkeypatch.setattr(cfgmod, "_atomic_write", _refuse)
    rig.spy.clear()
    c.dictionary.add("second")
    c.dictionary.add("third")
    c.dictionary.flush()
    rig.settle(100)
    assert any("sav" in m.lower() for _l, m in rig.spy.notices())
    monkeypatch.setattr(cfgmod, "_atomic_write", real)
    assert c.dictionary.retry_save() is True
    assert {e["term"] for e in store.Dictionary(c.data_dir / "dictionary.json").list()} == {"first", "second", "third"}


# ============================================================================ T-01 / D-15 retry and notice actions
def test_t01_a_save_failure_offers_a_retry_and_retry_saves_recovers(rig, monkeypatch):
    c = rig.c
    actions = []
    c.notice_action.connect(lambda msg, action_id, label: actions.append((msg, action_id, label)))
    real = cfgmod._atomic_write
    monkeypatch.setattr(cfgmod, "_atomic_write", _refuse)
    c.dictionary.add("Kubernetes")
    rig.settle(100)
    assert any(a[1] == "retry_save" and a[2] for a in actions), actions
    monkeypatch.setattr(cfgmod, "_atomic_write", real)
    result = c.retry_saves()
    assert result["dictionary"]["ok"] is True
    assert "Kubernetes" in [e["term"] for e in store.Dictionary(c.data_dir / "dictionary.json").list()]
    assert any(lvl == "success" for lvl, _m in rig.spy.notices()), rig.spy.notices()


def test_t01_retry_saves_reports_a_store_that_still_fails(rig, monkeypatch):
    c = rig.c
    monkeypatch.setattr(cfgmod, "_atomic_write", _refuse)
    c.history.add("something dictated")
    result = c.retry_saves()
    assert result["history"]["ok"] is False and result["history"]["error"]


def test_d05_a_microphone_failure_offers_to_open_the_microphone_settings(rig):
    from controller_rig import TRecorder
    actions = []
    rig.c.notice_action.connect(lambda msg, action_id, label: actions.append(action_id))
    TRecorder.fail_start = OSError("no device")
    rig.pause()
    rig.press()
    rig.wait(lambda: rig.c.state in ("error", "idle"))
    TRecorder.fail_start = None
    rig.release()
    assert "open_settings:microphone" in actions, actions


# ============================================================================ T-15 ranges
@pytest.mark.parametrize("spoken,want", [
    ("between five and ten", "Between 5 and 10."),
    ("from five to twenty", "From 5 to 20."),
    ("between five and six", "Between five and six."),
    ("one hundred and five", "105."),
    ("nine one one", "Nine one one."),
])
def test_t15_ranges_use_one_style(spoken, want):
    assert clean(spoken) == want


def test_d05_the_same_failure_reason_is_announced_once_per_30_seconds(rig):
    from controller_rig import TRecorder
    rig.spy.clear()
    for _ in range(3):
        TRecorder.fail_start = OSError("no device")
        rig.pause()
        rig.press()
        rig.settle(80)
        TRecorder.fail_start = None
        rig.release()
        rig.wait(lambda: rig.c.state == "idle")
    warnings = [m for lvl, m in rig.spy.notices() if lvl == "warning" and "icrophone" in m]
    assert len(warnings) == 1, warnings


def test_p6_diag_mode_logs_the_process_cpu_once_a_minute(rig, monkeypatch, caplog):
    from voxflow import controller as ctl_mod
    monkeypatch.setattr(ctl_mod, "_DIAG", True)
    caplog.set_level("INFO", logger="voxflow.controller")
    rig.c._health_ticks = 0
    for _ in range(61):
        rig.c._health_tick()
    assert [r for r in caplog.records if r.getMessage().startswith("diag:")], "no diag line after 61 health ticks"


def test_p6_diag_is_silent_by_default(rig, caplog):
    caplog.set_level("INFO", logger="voxflow.controller")
    for _ in range(61):
        rig.c._health_tick()
    assert not [r for r in caplog.records if r.getMessage().startswith("diag:")]
