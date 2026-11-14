"""``voxflow.ui.fakes``: the in-memory controller twin behind the demo / screenshot script. The pages are written
against the controller facade, so the twin has to behave like it: same validation sentences, same signals, same
file formats. Everything here is in memory or in ``tmp_path``.
"""
from __future__ import annotations

import csv
import datetime as dt
import json

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication

from tests.cov_ui_kit import Signals
from tests.helpers import fake_time
from voxflow.ui import fakes
from voxflow.ui.fakes import FakeController, FakeRecorder


@pytest.fixture
def ctl(qapp):
    c = FakeController()
    yield c
    c.shutdown()
    c.deleteLater()


@pytest.fixture
def empty(qapp):
    c = FakeController(empty=True)
    yield c
    c.shutdown()
    c.deleteLater()


# =========================================================================== recorder
def test_recorder_returns_as_much_audio_as_was_recorded_capped_at_the_maximum(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(fakes, "time", fake_time(perf_counter=lambda: clock["t"]))
    rec = FakeRecorder(max_seconds=2.0)
    assert rec.stop().size == 0 and rec.last_speech_seconds() == 0.0, "nothing recorded before start()"
    rec.start()
    assert rec.is_recording
    clock["t"] += 0.5
    assert rec.last_speech_seconds() == pytest.approx(0.5)
    audio = rec.stop()
    assert audio.dtype == np.float32 and audio.size == 8000 and not rec.is_recording
    assert float(np.abs(audio).max()) <= 0.33
    rec.start()
    clock["t"] += 60
    assert rec.stop().size == 2 * fakes.SAMPLE_RATE, "max_seconds caps the clip"
    rec.start()
    rec.close()
    assert not rec.is_recording and rec.stop().size == 0


def test_recorder_level_and_samples_are_silent_when_the_voice_is_off():
    rec = FakeRecorder()
    samples = rec.recent_samples(512)
    assert samples.shape == (512,) and samples.dtype == np.float32
    assert 0.0 <= rec.level() <= 1.0
    rec.active_voice = False
    assert rec.level() == 0.0 and not rec.recent_samples(64).any()


# =========================================================================== dictionary
def test_dictionary_add_update_remove_validate(ctl):
    d = ctl.dictionary
    seen = Signals(changed=ctl.dictionary_changed)
    before = len(d.list())
    entry = d.add("  Zeta  ", ["zeta", "Zeta", " zita ", ""])
    assert entry["term"] == "Zeta" and entry["aliases"] == ["zeta", "zita"], "aliases are trimmed and de-duplicated"
    assert len(seen.changed) == 1 and "Zeta" in d.terms() and d.terms() == sorted(d.terms(), key=str.lower)
    with pytest.raises(ValueError, match="already in your dictionary"):
        d.add("zeta")
    with pytest.raises(ValueError, match="Enter the word"):
        d.add("   ")
    renamed = d.update(entry["id"], term="Zeta Prime")
    assert renamed["term"] == "Zeta Prime" and renamed["aliases"] == ["zeta", "zita"]
    assert d.update(entry["id"], aliases=["zed"])["aliases"] == ["zed"]
    assert d.update(entry["id"], term="zeta prime")["term"] == "zeta prime", "an entry may be renamed to itself"
    with pytest.raises(ValueError, match="no longer exists"):
        d.update("nope", term="x")
    assert d.apply("leave me alone") == "leave me alone"
    n = len(seen.changed)
    assert d.remove(entry["id"]) is True and len(seen.changed) == n + 1
    assert d.remove(entry["id"]) is False and len(seen.changed) == n + 1, "removing nothing announces nothing"
    assert len(d.list()) == before


def test_dictionary_list_is_a_copy(ctl):
    listed = ctl.dictionary.list()
    listed[0]["term"] = "tampered"
    listed[0]["aliases"].append("tampered")
    assert "tampered" not in ctl.dictionary.terms()
    assert all("tampered" not in e["aliases"] for e in ctl.dictionary.list())


def test_dictionary_import_reads_json_text_and_csv_and_skips_what_is_already_there(empty, tmp_path):
    d = empty.dictionary
    as_json = tmp_path / "words.json"
    as_json.write_text(json.dumps([{"term": "Alpha", "aliases": ["alfa"]}, {"term": "Beta"}, {"aliases": ["x"]},
                                   "junk", {"term": "alpha"}]), encoding="utf-8")
    assert d.import_(str(as_json)) == 2
    assert [(e["term"], e["aliases"]) for e in d.list()] == [("Alpha", ["alfa"]), ("Beta", [])]
    as_text = tmp_path / "words.txt"
    as_text.write_text("Gamma: gama, gamma ray\n\nDelta\nbeta: again\n", encoding="utf-8")
    assert d.import_(str(as_text)) == 2
    assert next(e for e in d.list() if e["term"] == "Gamma")["aliases"] == ["gama", "gamma ray"]
    as_csv = tmp_path / "words.csv"
    as_csv.write_text("Epsilon,epsylon,ep\n", encoding="utf-8")
    assert d.import_(str(as_csv)) == 1
    assert next(e for e in d.list() if e["term"] == "Epsilon")["aliases"] == ["epsylon", "ep"]
    not_a_list = tmp_path / "object.json"
    not_a_list.write_text('{"term": "Nope"}', encoding="utf-8")
    assert d.import_(str(not_a_list)) == 0


def test_dictionary_import_reports_unreadable_files_in_plain_words(empty, tmp_path):
    with pytest.raises(ValueError, match="Could not read missing.json"):
        empty.dictionary.import_(str(tmp_path / "missing.json"))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json is not valid JSON"):
        empty.dictionary.import_(str(broken))


@pytest.mark.parametrize("suffix, expected", [
    (".json", None),
    (".txt", "Alpha: alfa, alpha one\nBeta\n"),
    (".csv", "Alpha,alfa, alpha one\nBeta\n"),
])
def test_dictionary_export_round_trips_each_format(empty, tmp_path, suffix, expected):
    d = empty.dictionary
    d.add("Beta")
    d.add("Alpha", ["alfa", "alpha one"])
    path = tmp_path / f"out{suffix}"
    assert d.export(str(path)) == 2
    if expected is None:
        assert json.loads(path.read_text(encoding="utf-8")) == [{"term": "Alpha", "aliases": ["alfa", "alpha one"]},
                                                                {"term": "Beta", "aliases": []}]
    else:
        assert path.read_text(encoding="utf-8") == expected
    other = FakeController(empty=True)
    try:
        assert other.dictionary.import_(str(path)) == 2
        assert other.dictionary.terms() == ["Alpha", "Beta"]
    finally:
        other.deleteLater()


def test_dictionary_suggestions_can_be_accepted_merged_or_dismissed(ctl):
    d = ctl.dictionary
    seen = Signals(sugg=ctl.suggestions_changed, changed=ctl.dictionary_changed)
    first, second = d.suggestions()
    added = d.accept_suggestion(first["id"])
    assert added["term"] == first["term"] and added["aliases"] == first["aliases"]
    assert [s["id"] for s in d.suggestions()] == [second["id"]] and len(seen.sugg) == 1
    with pytest.raises(ValueError, match="no longer available"):
        d.accept_suggestion(first["id"])
    existing = d.add(second["term"].upper(), ["old alias"])        # the user already has that word
    merged = d.accept_suggestion(second["id"])
    assert merged["id"] == existing["id"] and merged["aliases"] == ["old alias"] + second["aliases"]
    assert d.suggestions() == []
    d._sugg.append({"id": "s1", "term": "Qt", "aliases": ["cute"], "ts": 0.0, "source": "cleanup"})
    n = len(seen.sugg)
    assert d.dismiss_suggestion("s1") is True and len(seen.sugg) == n + 1
    assert d.dismiss_suggestion("s1") is False and len(seen.sugg) == n + 1


# =========================================================================== snippets
def test_snippets_add_update_remove_validate(ctl):
    s = ctl.snippets
    seen = Signals(changed=ctl.snippets_changed)
    entry = s.add("  door code ", "1234#")
    assert entry["trigger"] == "door code" and entry["text"] == "1234#" and len(seen.changed) == 1
    for trigger, text, message in (("", "x", "Enter the phrase"), ("Door Code", "x", "already have a snippet"),
                                   ("new one", "   ", "Enter the text")):
        with pytest.raises(ValueError, match=message):
            s.add(trigger, text)
    assert s.update(entry["id"], text="9999#")["text"] == "9999#"
    assert s.update(entry["id"], trigger="DOOR CODE")["trigger"] == "DOOR CODE"
    with pytest.raises(ValueError, match="no longer exists"):
        s.update("nope", text="x")
    assert s.expand("say door code") == "say door code"
    assert [e["trigger"] for e in s.list()] == sorted((e["trigger"] for e in s.list()), key=str.lower)
    assert s.remove(entry["id"]) is True and s.remove(entry["id"]) is False
    assert len(seen.changed) == 4, "add, two updates, one real removal"


def test_snippets_export_and_import_round_trip(ctl, empty, tmp_path):
    path = tmp_path / "snippets.json"
    n = ctl.snippets.export(str(path))
    assert n == len(ctl.snippets.list()) > 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert all(set(d) == {"trigger", "text"} for d in data)
    data += ["junk", {"trigger": "", "text": "no trigger"}, {"trigger": data[0]["trigger"], "text": "duplicate"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert empty.snippets.import_(str(path)) == n, "damaged and duplicate rows are skipped"
    assert [e["text"] for e in empty.snippets.list()] == [e["text"] for e in ctl.snippets.list()]
    obj = tmp_path / "object.json"
    obj.write_text("{}", encoding="utf-8")
    assert empty.snippets.import_(str(obj)) == 0
    for bad in (tmp_path / "missing.json", tmp_path / "broken.json"):
        if bad.name == "broken.json":
            bad.write_text("[", encoding="utf-8")
        with pytest.raises(ValueError, match=f"Could not read snippets from {bad.name}"):
            empty.snippets.import_(str(bad))


# =========================================================================== history
def test_history_list_search_get_delete_clear(ctl):
    h = ctl.history
    seen = Signals(changed=ctl.history_changed)
    total = h.count()
    assert total == len(h.list(None)) > 5
    assert len(h.list(2)) == 2 and h.list(2, offset=1)[0]["id"] == h.list(None)[1]["id"]
    newest = h.list(1)[0]
    assert ctl.last_transcript["id"] == newest["id"] and h.get(newest["id"])["text"] == newest["text"]
    assert h.get("nope") is None
    word = newest["text"].split()[0]
    assert newest["id"] in [e["id"] for e in h.search(word.upper())]
    assert len(h.search("")) == total and len(h.search("", limit=3)) == 3
    assert [e["app_exe"] for e in h.search("outlook")] == ["outlook.exe"], "the app name is searched too"
    assert h.delete(newest["id"]) is True and h.count() == total - 1 and len(seen.changed) == 1
    assert h.delete(newest["id"]) is False and len(seen.changed) == 1
    h.clear()
    assert h.count() == 0 and h.list() == [] and ctl.last_transcript is None and len(seen.changed) == 2


def test_history_export_writes_text_csv_and_json(ctl, tmp_path):
    h = ctl.history
    n = h.count()
    as_json = tmp_path / "h.json"
    assert h.export(str(as_json)) == n
    assert [e["id"] for e in json.loads(as_json.read_text(encoding="utf-8"))] == [e["id"] for e in h.list(None)]
    as_csv = tmp_path / "h.csv"
    assert h.export(str(as_csv)) == n
    with as_csv.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["time", "app", "words", "text", "raw"] and len(rows) == n + 1
    assert rows[1][3] == h.list(1)[0]["text"] and dt.datetime.fromisoformat(rows[1][0])
    as_txt = tmp_path / "h.txt"
    assert h.export(str(as_txt)) == n
    text = as_txt.read_text(encoding="utf-8")
    assert text.count("\n\n[") + 1 >= n and h.list(1)[0]["text"] in text
    bare = tmp_path / "no_suffix"
    assert h.export(str(bare)) == n and bare.read_text(encoding="utf-8") == text, "no suffix means plain text"
    forced = tmp_path / "forced.dat"
    assert h.export(str(forced), "JSON") == n and json.loads(forced.read_text(encoding="utf-8"))


# =========================================================================== stats
def test_stats_summary_daily_top_apps_and_reset(ctl):
    st = ctl.stats
    s = st.summary()
    assert s["total_words"] == sum(d["words"] for d in st._daily) and s["sessions"] > 0
    assert s["today_words"] == st._daily[-1]["words"]
    assert s["streak_days"] == 7, "the seeded fortnight ends with seven active days in a row"
    assert s["avg_wpm"] == pytest.approx(s["total_words"] / s["seconds_spoken"] * 60)
    assert s["time_saved_s"] == pytest.approx(s["total_words"] / 40 * 60 - s["seconds_spoken"])
    week = st.daily(7)
    today = dt.date.today()
    assert [d["date"] for d in week] == [(today - dt.timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    assert week[-1]["words"] == s["today_words"]
    assert st.daily(30)[0] == {"date": (today - dt.timedelta(days=29)).isoformat(), "words": 0, "sessions": 0,
                               "seconds": 0.0}, "days without data are zero-filled"
    top = st.top_apps(3)
    assert len(top) == 3 and [a["words"] for a in top] == sorted((a["words"] for a in top), reverse=True)
    seen = Signals(changed=ctl.stats_changed)
    st.reset()
    assert len(seen.changed) == 1 and st.top_apps() == []
    assert st.summary() == {"total_words": 0, "sessions": 0, "seconds_spoken": 0, "avg_wpm": 0.0, "time_saved_s": 0.0,
                            "streak_days": 0, "today_words": 0}


# =========================================================================== styles
def test_styles_tones_and_app_categories(ctl):
    st = ctl.styles
    seen = Signals(changed=ctl.styles_changed)
    assert st.tones() == {"personal": "casual", "work": "casual", "email": "formal", "other": "formal"}
    st.set_tone("work", "formal")
    assert st.tones()["work"] == "formal" and st.tone_for("slack.exe") == "formal" and len(seen.changed) == 1
    for category, tone in (("nope", "formal"), ("work", "shouty")):
        with pytest.raises(ValueError, match="Unknown style"):
            st.set_tone(category, tone)
    assert st.category_for("C:\\Apps\\Slack.EXE") == "work" and st.category_for("unknown-app") == "other"
    with pytest.raises(ValueError, match="Enter an app name"):
        st.category_for("  ")
    with pytest.raises(ValueError, match="Unknown category"):
        st.set_app_category("signal", "nope")
    st.set_app_category("Signal", "personal")                 # an app the user added
    st.set_app_category("slack.exe", "personal")              # a built-in app the user re-filed
    rows = {r["exe"]: r for r in st.app_categories()}
    assert rows["signal.exe"] == {"exe": "signal.exe", "category": "personal", "builtin": False, "overridden": False}
    assert rows["slack.exe"] == {"exe": "slack.exe", "category": "personal", "builtin": True, "overridden": True}
    assert rows["teams.exe"]["overridden"] is False and list(rows) == sorted(rows)
    n = len(seen.changed)
    assert st.remove_app_category("slack") is True and st.category_for("slack.exe") == "work"
    assert st.remove_app_category("slack") is False and len(seen.changed) == n + 1


# =========================================================================== controller: settings
def test_controller_starts_seeded_or_empty(ctl, empty):
    assert ctl.is_demo and ctl.state == "idle" and not ctl.paused
    assert ctl.history.count() and ctl.dictionary.list() and ctl.snippets.list() and ctl.dictionary.suggestions()
    assert sum(1 for e in ctl.history.list(None) if e.get("rules_only")) == 1
    assert (empty.history.count(), empty.dictionary.list(), empty.snippets.list(), empty.stats.top_apps()) == (0, [], [], [])
    assert empty.last_transcript is None


def test_set_validates_and_announces_only_real_changes(ctl, tmp_path):
    seen = Signals(config=ctl.config_changed, status=ctl.status_changed, sync=ctl.sync_status_changed)
    ctl.set("language", "de")
    assert ctl.get("language") == "de" and seen.config == [("language", "de")]
    ctl.set("language", "de")
    assert len(seen.config) == 1, "setting the same value again is silent"
    for key, value, message in (("nope", 1, "Unknown setting"), ("sound_volume", 1.5, "Volume must be"),
                                ("history_retention_days", 4000, "at most 10 years"), ("theme", "pink", "Unknown theme"),
                                ("sync_folder", str(tmp_path / "gone"), "does not exist any more"),
                                ("hotkey", "a", "Use at least one modifier"),
                                ("command_hotkey", "ctrl+win", "Already used for Dictation")):
        with pytest.raises(ValueError, match=message):
            ctl.set(key, value)
    assert len(seen.config) == 1 and seen.status == []
    ctl.set("hotkey", "ctrl+shift+d")
    assert seen.status[-1][0]["detail"] == "Hold Ctrl + Shift + D to dictate"
    ctl.set("cleanup_enabled", False)
    assert len(seen.status) == 2
    ctl.set("undo_hotkey", "")                                # an optional hotkey may be cleared
    ctl.set("sync_folder", str(tmp_path))
    assert seen.sync[-1][0]["enabled"] and seen.sync[-1][0]["last_sync"] is not None
    ctl.set("sync_folder", "")
    assert seen.sync[-1][0]["state"] == "off" and seen.sync[-1][0]["last_sync"] is None


def test_reset_settings_restores_defaults_but_keeps_first_run_state(ctl):
    ctl.set("language", "fr")
    ctl.set("sound_volume", 0.9)
    ctl.set("onboarding_done", True)
    ctl.set("ui_state", {"page": "history"})
    seen = Signals(config=ctl.config_changed, status=ctl.status_changed)
    ctl.reset_settings()
    assert sorted(seen.config) == [("language", "auto"), ("sound_volume", 0.5)] and len(seen.status) == 1
    assert ctl.get("onboarding_done") is True and ctl.get("ui_state") == {"page": "history"}


@pytest.mark.parametrize("key, combo, problem", [
    ("hotkey", "", "The dictation hotkey cannot be empty."),
    ("undo_hotkey", "", ""),
    ("hotkey", "ctrl+win", ""),
    ("hotkey", "ctrl+alt", ""),
    ("hotkey", "f9", ""),
    ("hotkey", "ctrl+f9", ""),
    ("hotkey", "q", "Use at least one modifier (Ctrl, Alt, Shift, Win) plus a key, or two modifiers."),
    ("hotkey", "ctrl", "Use at least one modifier (Ctrl, Alt, Shift, Win) plus a key, or two modifiers."),
    ("hotkey", "ctrl+a+b", "Use only one non-modifier key."),
    ("hotkey", "win+alt", "Already used for Command mode."),
    ("command_hotkey", "alt+win", ""),
    ("undo_hotkey", "ctrl+z", "Ctrl + Z is a system shortcut. Pick something else."),
    ("undo_hotkey", "alt+f4", "Alt + F4 is a system shortcut. Pick something else."),
])
def test_validate_hotkey(ctl, key, combo, problem):
    assert ctl.validate_hotkey(key, combo) == problem


def test_pretty_hotkey_and_capture(ctl):
    assert ctl.pretty_hotkey("ctrl+win") == "Ctrl + Win" and ctl.pretty_hotkey("alt+shift+z") == "Alt + Shift + Z"
    assert ctl.pretty_hotkey("ctrl+space") == "Ctrl + Space" and ctl.pretty_hotkey("ctrl+pageup") == "Ctrl + Pageup"
    assert ctl.pretty_hotkey("") == "Not set" and ctl.pretty_hotkey(None) == "Not set"
    seen = Signals(captured=ctl.hotkey_captured)
    ctl.cancel_hotkey_capture()
    assert seen.captured == [], "nothing to cancel"
    ctl.begin_hotkey_capture()
    ctl.cancel_hotkey_capture()
    ctl.cancel_hotkey_capture()
    assert seen.captured == [("",)]


def test_languages_tones_categories_and_tone_preview(ctl):
    codes = [c for c, _name in ctl.languages()]
    assert codes[0] == "auto" and len(codes) == len(set(codes)) == 26
    assert [k for k, _v in ctl.tones()] == list(ctl.styles.TONES)
    assert [k for k, _v in ctl.categories()] == list(ctl.styles.CATEGORIES)
    assert ctl.language_support("auto") == "auto" and ctl.language_support("de") == "full"
    assert ctl.language_support("pl") == "basic"
    from voxflow.cleanup import apply_tone
    sample = "Hello there. I will send it tomorrow."
    for tone in ctl.styles.TONES:
        assert ctl.preview_tone(sample, tone) == apply_tone(sample, tone, None)
        assert ctl.preview_tone(sample, tone, "auto") == apply_tone(sample, tone, None)
    assert ctl.preview_tone("Guten Tag. Ich komme morgen.", "very_casual", "de") == \
        apply_tone("Guten Tag. Ich komme morgen.", "very_casual", "de")


# =========================================================================== controller: status, audio, actions
def test_status_follows_pause_and_the_models(ctl):
    seen = Signals(status=ctl.status_changed, model=ctl.model_status_changed)
    assert ctl.status() == {"level": "demo", "title": "Ready", "detail": "Hold Ctrl + Win to dictate"}
    ctl.demo_set_model("asr", "downloading", "42 MB of 600 MB", 0.07)
    assert ctl.status() == {"level": "loading", "title": "Loading speech model…", "detail": "42 MB of 600 MB"}
    assert ctl.model_status("asr") == {"state": "downloading", "detail": "42 MB of 600 MB", "progress": 0.07, "backend": ""}
    assert seen.model[-1] == ("asr", ctl.model_status("asr")) and seen.status[-1] == (ctl.status(),)
    ctl.demo_set_model("asr", "failed", "Out of memory")
    assert ctl.status() == {"level": "error", "title": "Speech model failed", "detail": "Out of memory"}
    ctl.demo_set_model("asr", "ready")
    ctl.demo_set_model("cleanup", "loading", "Warming up")
    assert ctl.status()["title"] == "Loading cleanup model…"
    ctl.demo_set_model("cleanup", "failed", "No GPU")
    assert ctl.status()["level"] == "demo", "dictation still works without the cleanup model"
    ctl.reload_models()
    assert ctl.model_status("cleanup")["state"] == "ready" and ctl.model_status("asr")["backend"] == "cuda"
    n = len(seen.status)
    ctl.set_paused(True)
    ctl.set_paused(True)
    assert ctl.paused and len(seen.status) == n + 1 and ctl.status()["level"] == "paused"
    ctl.set_paused(False)
    assert not ctl.paused and ctl.status()["level"] == "demo"
    ctl.start()
    assert seen.status[-1] == (ctl.status(),)


def test_microphones_level_monitor_and_sounds(ctl):
    seen = Signals(mics=ctl.microphones_changed)
    mics = ctl.microphones()
    assert mics[0] == {"id": None, "name": "System default", "is_default": True} and len(mics) == 3
    mics[0]["name"] = "tampered"
    assert ctl.microphones()[0]["name"] == "System default"
    ctl.refresh_microphones()
    assert seen.mics == [(ctl.microphones(),)]
    assert ctl.level() == 0.0 and not ctl.recent_samples(128).any(), "silent while idle and not monitoring"
    ctl._recorder.level = lambda: 0.5
    ctl.set_level_monitor(True)
    assert ctl.level() == 0.5 and ctl.recent_samples(128).shape == (128,)
    ctl.set_level_monitor(False)
    ctl.demo_set_state("listening")
    assert ctl.level() == 0.5 and ctl.recent_samples(256).shape == (256,)
    ctl.cancel()
    assert ctl.state == "idle" and ctl.level() == 0.0
    ctl.play_sound("start")
    ctl.play_sound("done")
    assert ctl.played == ["start", "done"]


def test_demo_state_changes_are_announced_and_the_idle_timer_brings_it_back(ctl, qtbot):
    seen = Signals(state=ctl.state_changed)
    ctl.demo_set_state("processing", "command", "Thinking")
    assert (ctl.state, ctl.mode, ctl.message) == ("processing", "command", "Thinking")
    assert seen.state == [("processing", "command", "Thinking")]
    ctl._idle_timer.start(0)
    qtbot.waitUntil(lambda: ctl.state == "idle", timeout=2000)
    assert seen.state[-1] == ("idle", "dictate", "") and not ctl._idle_timer.isActive()
    ctl._idle_timer.start(60_000)
    ctl.demo_set_state("listening")
    assert not ctl._idle_timer.isActive(), "a new state cancels the pending return to idle"
    ctl._idle_timer.start(60_000)
    ctl.shutdown()
    assert not ctl._idle_timer.isActive()


@pytest.mark.parametrize("kind, action, label", [
    ("silence", "microphone", "Open microphone settings"),
    ("mic", "microphone", "Open microphone settings"),
    ("model", "models", "Open model settings"),
    ("mystery", "", ""),
])
def test_demo_error_carries_a_reason_and_a_next_step(ctl, kind, action, label):
    seen = Signals(info=ctl.error_info_changed, notice=ctl.notice, action=ctl.notice_action, state=ctl.state_changed)
    assert ctl.error_info() == {"kind": "", "action": "", "hint": "", "message": ""}
    ctl.demo_error(kind)
    info = ctl.error_info()
    assert info["kind"] == kind and info["action"] == action and seen.info == [(info,)]
    assert ctl.state == "error" and seen.state[-1][2] == info["hint"].split(".")[0]
    if action:
        assert seen.notice == [("warning", info["hint"])]
        assert seen.action == [(info["hint"], f"open_settings:{action}", label)]
    else:
        assert seen.notice == [] and seen.action == [] and info["hint"] == ""


def test_paste_undo_copy_retry_and_folder_actions(ctl, empty, monkeypatch):
    seen = Signals(notice=ctl.notice)
    assert ctl.paste_last() is True and ctl.paste_last(ctl.history.list(1)[0]["id"]) is True
    assert ctl.paste_last("no-such-entry") is False
    assert seen.notice == [("warning", "There is nothing to paste yet.")]
    assert ctl.undo_last_paste() is False and seen.notice[-1] == ("info", "Nothing to undo in the preview.")
    assert ctl.retry_saves() == {} and ctl.retried == 1 and seen.notice[-1] == ("success", "Saved.")
    ctl.open_data_folder()
    assert seen.notice[-1] == ("info", f"Your data lives in {ctl.data_dir}.")
    ctl.open_log_folder()
    assert seen.notice[-1] == ("info", f"Logs are written to {ctl.log_path.parent}.")
    empty_seen = Signals(notice=empty.notice)
    assert empty.paste_last() is False and empty_seen.notice == [("warning", "There is nothing to paste yet.")]

    class Clipboard:                                           # never the real clipboard
        text = None

        def setText(self, text):
            Clipboard.text = text

    monkeypatch.setattr(QGuiApplication, "clipboard", staticmethod(lambda: Clipboard()))
    assert ctl.copy_text("copied words") is True and Clipboard.text == "copied words"


def test_demo_add_transcript_feeds_history_stats_and_the_signal(ctl, empty):
    seen = Signals(ready=ctl.transcript_ready, history=ctl.history_changed, stats=ctl.stats_changed)
    words_before, total = ctl.stats._daily[-1]["words"], ctl.history.count()
    entry = ctl.demo_add_transcript("One two three.")
    assert entry["words"] == 3 and "rules_only" not in entry and entry["raw"] == "one two three"
    assert ctl.history.count() == total + 1 and ctl.last_transcript["id"] == entry["id"]
    assert ctl.stats._daily[-1]["words"] == words_before + 3
    assert seen.ready == [(entry,)] and len(seen.history) == 1 and len(seen.stats) == 1
    ctl.set("save_history", False)
    quiet = ctl.demo_add_transcript("Not kept.", rules_only=True)
    assert quiet["rules_only"] is True and ctl.history.count() == total + 1 and len(seen.history) == 1
    assert len(seen.ready) == 2 and len(seen.stats) == 2, "stats and the result still arrive with history off"
    first = empty.demo_add_transcript()
    assert empty.history.count() == 1 and empty.stats.summary()["total_words"] == 0 and first["words"] == 7


# =========================================================================== backup, packs, sync
def test_backup_export_inspect_and_merge(ctl, empty, tmp_path):
    out = ctl.export_backup(str(tmp_path / "mine"))
    assert out["path"].endswith(".voxbackup") and out["includes"][-2:] == ["stats", "history"]
    assert out["counts"]["history"] == ctl.history.count() and out["counts"]["dictionary"] == len(ctl.dictionary.list())
    info = empty.inspect_backup(out["path"])
    assert info["same_device"] and info["version"] == ctl.version and info["skipped"] == 0
    assert info["counts"] == {**out["counts"], "settings": info["counts"]["settings"], "styles": 0}
    assert info["includes"] == ["settings", "dictionary", "snippets", "styles", "stats", "history"]
    seen = Signals(d=empty.dictionary_changed, s=empty.snippets_changed, h=empty.history_changed, st=empty.stats_changed)
    res = empty.import_backup(out["path"])
    assert res["mode"] == "merge" and res["counts"]["dictionary"] == len(ctl.dictionary.list())
    assert res["counts"]["snippets"] == len(ctl.snippets.list()) and res["counts"]["history"] == ctl.history.count()
    assert [len(x) for x in (seen.d, seen.s, seen.h, seen.st)] == [1, 1, 1, 1]
    assert empty.dictionary.terms() == ctl.dictionary.terms()
    assert [e["id"] for e in empty.history.list(None)] == [e["id"] for e in ctl.history.list(None)], "newest first"
    again = empty.import_backup(out["path"], "merge")
    assert again["counts"] == {"settings": 0, "dictionary": 0, "snippets": 0, "styles": 0, "stats": 0, "history": 0}
    with pytest.raises(ValueError, match="merge the backup or replace"):
        empty.import_backup(out["path"], "shred")


def test_backup_without_history_and_replace_of_chosen_parts(ctl, empty, tmp_path):
    ctl.set("language", "fr")
    ctl.styles.set_tone("email", "casual")
    ctl.styles.set_app_category("signal.exe", "personal")
    path = str(tmp_path / "slim.voxbackup")
    out = ctl.export_backup(path, include_history=False)
    assert out["path"] == path and out["counts"]["history"] == out["counts"]["stats"] == 0
    assert out["includes"] == ["settings", "dictionary", "snippets", "styles"]
    info = empty.inspect_backup(path)
    assert "history" not in info["includes"] and info["counts"]["history"] == 0 and info["counts"]["styles"] == 1

    mine = empty.dictionary.add("Only mine")
    empty.snippets.add("keep me", "still here")
    empty.demo_add_transcript("An old dictation.")
    res = empty.import_backup(path, "replace", parts=["dictionary"])
    assert res["counts"]["dictionary"] == len(ctl.dictionary.list()) and res["counts"]["snippets"] == 0
    assert mine["term"] not in empty.dictionary.terms(), "the ticked part is replaced"
    assert [e["trigger"] for e in empty.snippets.list()] == ["keep me"], "the unticked parts are left alone"
    assert empty.history.count() == 1 and empty.get("language") == "auto" and empty.styles.tones()["email"] == "formal"

    seen = Signals(config=empty.config_changed, styles=empty.styles_changed)
    empty.set("onboarding_done", True)
    empty.import_backup(path, "replace")
    assert ("language", "fr") in seen.config and empty.get("onboarding_done") is True
    assert empty.styles.tones()["email"] == "casual" and empty.styles.category_for("signal.exe") == "personal"
    assert len(seen.styles) == 1 and [e["trigger"] for e in empty.snippets.list()] == \
        [e["trigger"] for e in ctl.snippets.list()]
    assert empty.history.count() == 1, "a backup without history does not wipe the history"


def test_backup_files_that_cannot_be_used_are_refused_with_a_reason(ctl, tmp_path):
    with pytest.raises(ValueError, match="gone.voxbackup could not be read"):
        ctl.inspect_backup(str(tmp_path / "gone.voxbackup"))
    junk = tmp_path / "junk.voxbackup"
    junk.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ValueError, match="junk.voxbackup is not a VoxFlow backup"):
        ctl.inspect_backup(str(junk))
    junk.write_text(json.dumps({"voxflow": "pack"}), encoding="utf-8")
    with pytest.raises(ValueError, match="junk.voxbackup is not a VoxFlow backup"):
        ctl.import_backup(str(junk))
    junk.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a VoxFlow pack"):
        ctl.inspect_pack(str(junk))
    with pytest.raises(ValueError, match="x.voxbackup could not be written"):
        ctl.export_backup(str(tmp_path / "no-such-folder" / "x.voxbackup"))


@pytest.mark.parametrize("kind, key", [("dictionary", "term"), ("snippets", "trigger")])
def test_pack_export_inspect_and_import(ctl, empty, tmp_path, kind, key):
    store, target = getattr(ctl, kind), getattr(empty, kind)
    everything = store.list()
    base = str(tmp_path / f"team-{kind}")
    assert ctl.export_pack(kind, base, "  Team pack ", " for the team ") == len(everything)
    path = base + ".voxpack"
    info = empty.inspect_pack(path)
    assert info["kind"] == kind and info["name"] == "Team pack" and info["description"] == "for the team"
    assert (info["count"], info["new"], info["existing"], info["skipped"]) == (len(everything), len(everything), 0, 0)
    assert ctl.inspect_pack(path)["new"] == 0 and ctl.inspect_pack(path)["existing"] == len(everything)

    seen = Signals(d=empty.dictionary_changed, s=empty.snippets_changed)
    res = empty.import_pack(path)
    assert res == {"kind": kind, "name": "Team pack", "added": len(everything), "updated": 0, "removed": 0, "skipped": 0}
    assert (len(seen.d), len(seen.s)) == ((1, 0) if kind == "dictionary" else (0, 1))
    assert [e[key] for e in target.list()] == [e[key] for e in everything]
    assert empty.import_pack(path, "merge")["added"] == 0, "nothing new the second time"

    two = [e["id"] for e in everything[:2]]
    small = str(tmp_path / "small.voxpack")
    assert ctl.export_pack(kind, small, "Small", ids=two) == 2
    res = empty.import_pack(small, "replace")
    assert res["removed"] == len(everything) - 2 and res["added"] == 0
    assert [e[key] for e in target.list()] == [e[key] for e in everything[:2]]
    with pytest.raises(ValueError, match="merge the pack or replace"):
        empty.import_pack(small, "shred")


def test_pack_refusals(ctl, empty, tmp_path):
    with pytest.raises(ValueError, match="Only the dictionary and snippets"):
        ctl.export_pack("history", str(tmp_path / "h.voxpack"))
    with pytest.raises(ValueError, match="nothing to share yet"):
        empty.export_pack("snippets", str(tmp_path / "s.voxpack"))
    with pytest.raises(ValueError, match="nothing to share yet"):
        ctl.export_pack("dictionary", str(tmp_path / "d.voxpack"), ids=[])
    odd = tmp_path / "odd.voxpack"
    odd.write_text(json.dumps({"voxflow": "pack", "kind": "styles", "items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="odd.voxpack is not a dictionary or snippet pack"):
        ctl.import_pack(str(odd))
    partly = tmp_path / "partly.voxpack"
    partly.write_text(json.dumps({"voxflow": "pack", "kind": "dictionary",
                                  "items": [{"term": "Brand new"}, {"aliases": ["no term"]}]}), encoding="utf-8")
    assert empty.import_pack(str(partly))["added"] == 1 and empty.dictionary.terms() == ["Brand new"]


def test_sync_status_messages(ctl, tmp_path):
    seen = Signals(sync=ctl.sync_status_changed, notice=ctl.notice)
    off = ctl.sync_status()
    assert off["state"] == "off" and not off["enabled"] and off["message"] == "Sync is off." and off["devices"] == []
    ctl.sync_now()
    assert seen.notice == [("info", "Choose a sync folder first.")] and seen.sync == []
    ctl.set("sync_folder", str(tmp_path))
    status = ctl.sync_status()
    assert status["state"] == "idle" and status["folder"] == str(tmp_path) and status["device"] == "this-pc"
    assert status["message"] == "Up to date. No other device has used this folder yet."
    ctl.demo_set_sync("idle", devices=["Laptop"])
    assert ctl.sync_status()["message"] == "Up to date with 1 other device."
    assert ctl.sync_status()["devices"] == [{"name": "Laptop"}]
    ctl.demo_set_sync("idle", devices=["Laptop", "Studio"], last_sync=1234.5)
    assert ctl.sync_status()["message"] == "Up to date with 2 other devices." and ctl.sync_status()["last_sync"] == 1234.5
    ctl.demo_set_sync("syncing")
    assert ctl.sync_status()["message"] == "Syncing…" and ctl.sync_status()["last_sync"] == 1234.5
    ctl.demo_set_sync("error", "The folder is offline.")
    assert ctl.sync_status()["message"] == ctl.sync_status()["error"] == "The folder is offline."
    n = len(seen.sync)
    ctl.sync_now()
    assert len(seen.sync) == n + 1 and seen.sync[-1][0]["state"] == "idle" and seen.sync[-1][0]["error"] == ""
    assert seen.sync[-1][0]["last_sync"] > 1234.5
