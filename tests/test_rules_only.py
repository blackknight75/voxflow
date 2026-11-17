"""The quiet "basic clean-up" signal (core ff7a04c): when the language model was wanted for a dictation but timed
out or had no time left, the text gets the rule clean-up only, and the user can see that: a flag on the entry, a
hint on the Flow bar, and one notice when it keeps happening."""
from __future__ import annotations

import json
import time

import pytest

from voxflow import app as appmod
from voxflow import controller as ctl_mod
from voxflow import fakes, store
from voxflow import sync as S
from voxflow.cleanup import Cleaner

from controller_rig import make_rig
from test_cleanup_features import FakeLLM, with_llm

MESSY = "i pushed the the branch to get hub"
MESSY_ES = "eh creo que que deberiamos salir manana no perdon pasado manana"


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


def settle(c: Cleaner, llm: FakeLLM):
    llm.release.set()
    end = time.time() + 3
    while c._abandoned.is_set() and time.time() < end:
        time.sleep(0.01)


# =========================================================================== Cleaner.last_info["rules_only"]
def test_a_timeout_is_reported_as_rules_only_and_the_text_is_the_rule_text():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()
    assert c.clean(MESSY) == "I pushed the the branch to get hub."
    assert c.last_info["rules_only"] is True and c.last_info["timeout"] is True
    settle(c, llm)


def test_a_dictation_that_arrives_while_a_stuck_generation_unwinds_is_flagged_too():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()
    c.clean(MESSY)
    assert c.clean("we we should deploy it") == "We we should deploy it."
    assert c.last_info["rules_only"] is True, "the model was wanted for this one as well and could not be used"
    settle(c, llm)
    c.timeout_s = 5.0
    assert c.clean(MESSY) == "I pushed the branch to GitHub."
    assert c.last_info["rules_only"] is False and c.last_info["llm"] is True, "back to normal, flag cleared"


@pytest.mark.parametrize("spoken, reply", [
    ("Please send the report to the team.", "unused"),                       # nothing for the model to do
    (MESSY, "I pushed the branch to GitHub."),                               # the model answered in time
    (MESSY, "The capital of France is Paris, a beautiful city."),            # answered, rejected by the guardrail
    ("um uh", "unused"),                                                     # only fillers: nothing left
])
def test_not_rules_only_when_the_model_was_not_needed_or_did_answer(spoken, reply):
    c, _llm = with_llm(reply)
    c.clean(spoken)
    assert c.last_info["rules_only"] is False


def test_not_rules_only_when_there_is_no_model_at_all():
    """No language model (switched off, failed to load): that is the 'Limited' state shown in Settings, not this."""
    c = Cleaner(use_llm=False)
    c.clean(MESSY)
    assert c.last_info["rules_only"] is False
    broken, _ = with_llm(RuntimeError("llama decode failed"))
    broken.clean(MESSY)
    assert broken.last_info["rules_only"] is False, "a crash of the model is a different failure, reported elsewhere"


def test_a_call_with_nothing_to_clean_does_not_keep_the_report_of_the_previous_call():
    """``clean("")`` returns early and leaves ``last_info`` of the call before it, so a caller that reads the report
    sees 'rules only' for a call that did nothing. The app never calls it with empty text; direct callers can."""
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()
    c.clean(MESSY)
    assert c.last_info["rules_only"] is True
    settle(c, llm)
    assert c.clean("") == ""
    assert c.last_info.get("rules_only") is False
    assert c.clean("   ") == "   "
    assert c.last_info.get("rules_only") is False


def test_the_flag_is_reset_by_every_call():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()
    c.clean(MESSY)
    settle(c, llm)
    c.clean("Please send the report to the team.")
    assert c.last_info["rules_only"] is False


def test_a_later_line_that_found_no_time_left_is_flagged():
    """Several lines share one time budget: when the first line used it up, the others get rules only."""
    c, llm = with_llm(lambda user, messages: "I pushed the branch to GitHub.", timeout_s=0.25)
    llm.delay = 0.06                                           # slow, but the first line still finishes
    out = c.clean(f"{MESSY} new line we we should deploy it new line they they said yes")
    assert out.count("\n") == 2
    assert c.last_info["rules_only"] is True
    settle(c, llm)


def test_the_non_english_pipeline_reports_it_the_same_way():
    c, llm = with_llm("Creo que deberíamos salir pasado mañana.", timeout_s=0.15)
    llm.release.clear()
    out = c.clean(MESSY_ES, language="es")
    assert c.last_info["rules_only"] is True and c.last_info["language"] == "es" and out
    settle(c, llm)
    c.timeout_s = 5.0
    c.clean(MESSY_ES, language="es")
    assert c.last_info["rules_only"] is False
    c.clean("Hola, ¿qué tal estás hoy?", language="es")
    assert c.last_info["rules_only"] is False


def test_the_stand_in_cleaner_reports_the_same_fields_and_never_the_flag():
    f = fakes.FakeCleaner()
    f.clean(MESSY)
    assert f.last_info["rules_only"] is False and {"llm", "rejected", "ms"} <= set(f.last_info)
    f.clean(MESSY_ES, language="es")
    assert f.last_info["rules_only"] is False


# =========================================================================== app.run_pipeline
class Asr:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return self.text


class ScriptedCleaner:
    ready = True

    def __init__(self, info):
        self.last_info = info

    def clean(self, text, **kw):
        return text.capitalize() + "."


def run(cleaner, text=MESSY, **kw):
    import numpy as np
    return appmod.run_pipeline(np.zeros(16000, dtype=np.float32), Asr(text), cleaner, **kw)


@pytest.mark.parametrize("info, expect", [
    ({"rules_only": True}, True), ({"rules_only": False}, False), ({}, False), (None, False), ("yes", False),
    ({"rules_only": 1}, True), ({"timeout": True}, False),
])
def test_run_pipeline_passes_the_flag_on(info, expect):
    assert run(ScriptedCleaner(info))["rules_only"] is expect


def test_run_pipeline_without_cleanup_or_without_text_never_sets_it():
    assert run(None)["rules_only"] is False
    assert run(ScriptedCleaner({"rules_only": True}), text="")["rules_only"] is False

    class Old:                                               # a cleaner from before the field existed
        ready = True

        def clean(self, text):
            return text

    assert run(Old())["rules_only"] is False

    class Raising(ScriptedCleaner):
        def clean(self, text, **kw):
            raise RuntimeError("boom")

    out = run(Raising({"rules_only": True}))
    assert out["rules_only"] is False and out["cleaned"] is False, "a failed clean-up is not a rules-only clean-up"


# =========================================================================== store
def test_make_entry_has_the_key_only_when_it_is_true():
    assert store.make_entry("a", "a", rules_only=True)["rules_only"] is True
    assert "rules_only" not in store.make_entry("a", "a")
    for odd in (False, None, 1, "yes", "true", [1]):
        assert "rules_only" not in store.make_entry("a", "a", rules_only=odd)


def test_history_keeps_the_flag_on_disk_and_ignores_garbage(tmp_path):
    h = store.History(tmp_path)
    flagged = h.add_entry(store.make_entry("basic", "basic raw", rules_only=True))
    plain = h.add_entry(store.make_entry("full", "full raw"))
    again = store.History(tmp_path)
    assert again.get(flagged["id"])["rules_only"] is True and "rules_only" not in again.get(plain["id"])
    assert [e.get("rules_only") for e in again.search("basic")] == [True]

    path = next(p for p in tmp_path.iterdir() if p.name.startswith("history"))
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    for item, value in zip(items, ("yes", 1)):
        item["rules_only"] = value                             # hand-edited file
    path.write_text(json.dumps(data), encoding="utf-8")
    assert all("rules_only" not in e for e in store.History(tmp_path).snapshot())


def test_the_flag_travels_in_exports_imports_and_backups(tmp_path):
    h = store.History(tmp_path / "a")
    h.add_entry(store.make_entry("basic", "raw", rules_only=True))
    h.add_entry(store.make_entry("full", "raw"))
    other = store.History(tmp_path / "b")
    assert other.import_entries(h.snapshot()) == 2
    assert sorted((e["text"], e.get("rules_only")) for e in other.snapshot()) == [("basic", True), ("full", None)]
    out = tmp_path / "h.json"
    h.export(str(out), "json")
    assert '"rules_only": true' in out.read_text(encoding="utf-8").lower().replace(" ", " ")

    def dataset(folder):
        return S.DataSet(store.Dictionary(folder), store.Snippets(folder), store.Styles(folder),
                         store.History(folder), lambda: {}, lambda values: [], store.Stats(folder))

    backup = S.export_profile(dataset(tmp_path / "a"), str(tmp_path / "p"))["path"]
    fresh = dataset(tmp_path / "c")
    S.import_profile(fresh, backup, "replace", parts=["history"])
    assert sorted((e["text"], e.get("rules_only")) for e in fresh.history.snapshot()) == [("basic", True),
                                                                                         ("full", None)]


# =========================================================================== Controller
@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    r.cleaner.last_info = {"rules_only": False}
    r.clock = [1000.0]
    r.c._mono = lambda: r.clock[0]
    yield r
    r.close()


def slow(rig, on=True):
    rig.cleaner.last_info = {"rules_only": bool(on)}


def done_message(events) -> str:
    return [args[2] for name, args in events if name == "state_changed" and args[0] == "done"][-1]


def test_constants_are_what_the_ui_was_told():
    assert ctl_mod.RULES_ONLY_SUFFIX == " · basic clean-up"
    assert (ctl_mod.RULES_ONLY_BURST, ctl_mod.RULES_ONLY_WINDOW_S, ctl_mod.RULES_ONLY_QUIET_S) == (3, 300.0, 1800.0)
    assert "basic" in ctl_mod.RULES_ONLY_NOTICE and "History" in ctl_mod.RULES_ONLY_NOTICE
    assert "Traceback" not in ctl_mod.RULES_ONLY_NOTICE and len(ctl_mod.RULES_ONLY_NOTICE) < 240


def test_a_normal_dictation_has_no_flag_no_hint_no_count(rig):
    events = rig.dictate("please send the report")
    assert done_message(events) == "Pasted"
    ready = rig.spy.of("transcript_ready")[-1][0]
    assert "rules_only" not in ready and "rules_only" not in rig.last_entry()
    assert rig.c.rules_only_count == 0 and rig.spy.notices() == []


def test_a_slow_model_marks_the_entry_and_the_bar_but_raises_no_notice(rig):
    slow(rig)
    events = rig.dictate("please send the report")
    assert done_message(events) == "Pasted · basic clean-up"
    assert rig.spy.of("transcript_ready")[-1][0]["rules_only"] is True
    assert rig.last_entry()["rules_only"] is True and rig.c.last_transcript["rules_only"] is True
    assert rig.c.rules_only_count == 1 and rig.spy.notices() == [], "one slow answer never raises a notice"
    assert rig.injector.pasted[-1] == "Please send the report.", "the text itself is pasted as usual"
    rig.c.history.flush()                                        # the history is written behind (QA D17)
    saved = json.loads(next(p for p in rig.data_dir.iterdir() if p.name.startswith("history")).read_text("utf-8"))
    assert "rules_only" in json.dumps(saved)


@pytest.mark.parametrize("outcome, message", [
    ("pasted", "Pasted · basic clean-up"),
    ("typed", "Typed · basic clean-up"),
    ("copied", "Copied. Press Ctrl+V · basic clean-up"),
])
def test_the_hint_follows_every_successful_outcome_and_stays_short(rig, outcome, message):
    slow(rig)
    rig.injector.outcome = outcome
    assert done_message(rig.dictate("please send the report")) == message
    assert len(message) <= 37


def test_no_hint_when_the_paste_failed_but_the_entry_is_still_marked(rig):
    slow(rig)
    rig.injector.outcome = "failed"
    rig.dictate("please send the report")
    messages = [args[2] for name, args in rig.spy.events if name == "state_changed"]
    assert not any(ctl_mod.RULES_ONLY_SUFFIX in m for m in messages)
    assert rig.last_entry()["rules_only"] is True


def test_command_mode_is_never_marked(rig):
    slow(rig)
    rig.injector.selection = "hey bob can u send me the report"
    events = rig.dictate("make this more formal", binding="command")
    assert not any(ctl_mod.RULES_ONLY_SUFFIX in args[2] for name, args in events if name == "state_changed")
    assert "rules_only" not in (rig.last_entry() or {}) and rig.c.rules_only_count == 0


def test_cleanup_switched_off_is_never_marked(rig):
    slow(rig)
    rig.c.set("cleanup_enabled", False)
    events = rig.dictate("please send the report")
    assert ctl_mod.RULES_ONLY_SUFFIX not in done_message(events) and "rules_only" not in rig.last_entry()


def test_the_flag_is_kept_when_history_is_off(rig):
    slow(rig)
    rig.c.set("save_history", False)
    rig.dictate("please send the report")
    assert rig.spy.of("transcript_ready")[-1][0]["rules_only"] is True and rig.c.rules_only_count == 1


def test_three_in_five_minutes_raise_one_notice_then_quiet_for_half_an_hour(rig):
    slow(rig)
    for _ in range(2):
        rig.dictate("please send the report")
        rig.clock[0] += 60
    assert rig.spy.notices() == []
    rig.dictate("please send the report")
    assert rig.spy.notices() == [("info", ctl_mod.RULES_ONLY_NOTICE)]
    for _ in range(6):                                         # it keeps happening for the next 25 minutes
        rig.clock[0] += 250
        rig.dictate("please send the report")
    assert len(rig.spy.notices()) == 1, "told once, then left alone"
    assert rig.c.rules_only_count == 9
    rig.clock[0] += 400                                        # more than 30 minutes after the notice
    for _ in range(3):
        rig.dictate("please send the report")
        rig.clock[0] += 10
    assert len(rig.spy.notices()) == 2


def test_slow_answers_far_apart_never_raise_a_notice(rig):
    slow(rig)
    for _ in range(6):
        rig.dictate("please send the report")
        rig.clock[0] += 200                                    # never three inside five minutes
    assert rig.spy.notices() == [] and rig.c.rules_only_count == 6


def test_normal_dictations_in_between_do_not_count_and_do_not_reset(rig):
    slow(rig)
    rig.dictate("please send the report")
    slow(rig, False)
    for _ in range(5):
        rig.dictate("thanks a lot")
    slow(rig)
    rig.dictate("please send the report")
    assert rig.spy.notices() == []
    rig.dictate("please send the report")
    assert len(rig.spy.notices()) == 1 and rig.c.rules_only_count == 3


def test_paste_last_of_a_marked_entry_is_a_plain_paste(rig):
    slow(rig)
    rig.dictate("please send the report")
    rig.spy.clear()
    assert rig.c.paste_last()
    rig.wait(lambda: len(rig.injector.pasted) == 2)
    rig.settle(60)
    assert rig.c.rules_only_count == 1
    assert not any(ctl_mod.RULES_ONLY_SUFFIX in args[2] for name, args in rig.spy.events if name == "state_changed")


def test_a_cleaner_without_the_report_behaves_as_before(rig):
    del rig.cleaner.last_info
    events = rig.dictate("please send the report")
    assert done_message(events) == "Pasted" and "rules_only" not in rig.last_entry()


def test_the_real_cleaner_drives_the_whole_chain(tmp_path, qtbot, monkeypatch):
    """Real ``Cleaner`` with a stuck model inside the Controller: timeout -> entry flag -> bar hint."""
    r = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    c.load = lambda: None
    r.c._cleaner = c
    r.c.command._cleaner = c if hasattr(r.c.command, "_cleaner") else None
    r.start()
    try:
        llm.release.clear()
        events = r.dictate(MESSY)
        assert done_message(events) == "Pasted · basic clean-up"
        assert r.last_entry()["rules_only"] is True and r.last_entry()["text"] == "I pushed the the branch to get hub."
        settle(c, llm)
        c.timeout_s = 5.0
        events = r.dictate(MESSY)
        assert done_message(events) == "Pasted" and r.last_entry()["text"] == "I pushed the branch to GitHub."
    finally:
        llm.release.set()
        r.close()
