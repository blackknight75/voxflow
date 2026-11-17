"""Fixes for the core QA report (QA-core.md, C-01 .. C-12): the dictation pipeline with the real models.

Each test names its finding. They run on fakes (no model, no GPU); the real-model checks are in the report."""
from __future__ import annotations

import threading
import time

import pytest

from voxflow import cleanup
from voxflow.cleanup import Cleaner, LlmBusy
from voxflow.command import CommandError, CommandMode

from test_cleanup_features import with_llm

MESSY = "i pushed the the branch to get hub"


def _settle(c: Cleaner, llm) -> None:
    llm.release.set()
    assert c._idle.wait(3), "the abandoned generation never unwound"


# =========================================================================== C-02: one stuck call, the next ones wait
def test_c02_a_dictation_right_after_a_timeout_waits_for_the_stuck_call_and_gets_the_model():
    """The stuck generation finishes 0.2 s later: the next dictation must not give up after 0.1 ms."""
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.3)
    llm.release.clear()
    assert c.clean(MESSY) == "I pushed the the branch to get hub."          # this one timed out
    assert c.last_info["timeout"] is True
    threading.Timer(0.1, llm.release.set).start()                            # ... and unwinds shortly after
    c.timeout_s = 2.0
    t0 = time.perf_counter()
    assert c.clean(MESSY) == "I pushed the branch to GitHub."
    assert time.perf_counter() - t0 < 1.5
    assert c.last_info["rules_only"] is False and "skipped_busy" not in c.last_info


def test_c02_the_wait_is_deducted_from_the_callers_budget():
    c, llm = with_llm("x")
    c._idle.clear()
    c._abandoned.set()

    def finish_later():
        time.sleep(0.2)
        c._abandoned.clear()
        c._idle.set()

    threading.Thread(target=finish_later).start()
    with pytest.raises(TimeoutError, match="job timed out"):
        c._run_bounded(lambda abort: abort.wait(1) and "late", 0.3, "job")    # 0.3 s - 0.2 s waited = 0.1 s left
    assert 0.0 < c._waited_s < 0.3
    assert c._idle.wait(2)


def test_c02_a_call_that_finds_the_model_busy_for_its_whole_budget_raises_llmbusy_and_is_reported_apart():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()
    c.clean(MESSY)
    with pytest.raises(LlmBusy):
        c._run_bounded(lambda abort: "never", 0.05, "job")
    assert isinstance(LlmBusy("x"), TimeoutError), "callers that only know TimeoutError keep working"
    c.clean("we we should deploy it")
    assert c.last_info["skipped_busy"] is True and c.last_info["rules_only"] is True
    assert "timeout" not in c.last_info, "it did not time out: the model was never asked"
    _settle(c, llm)


def test_c02_a_busy_skip_teaches_the_speed_average_nothing():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    c._learn_speed = True
    llm.release.clear()
    c.clean(MESSY)                                     # timed out: a real (slow) call, it is learnt
    learnt = c._ms_per_word
    assert learnt is not None
    c.clean(MESSY)                                     # skipped: the model did not run
    assert c.last_info["skipped_busy"] is True and c._ms_per_word == learnt
    _settle(c, llm)


def test_c02_time_spent_waiting_for_an_earlier_call_is_not_counted_as_this_calls_speed():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()
    c.clean(MESSY)
    c._learn_speed, c._ms_per_word = True, None
    threading.Timer(0.3, llm.release.set).start()
    c.timeout_s = 3.0
    c.clean(MESSY)
    assert c.last_info["llm"] and not c.last_info["rules_only"]
    assert c._ms_per_word < 300 / 8, "the 0.3 s wait would make the model look far slower than it is"


def test_c02_other_languages_report_a_busy_skip_too():
    c, llm = with_llm("Creo que deberíamos salir pasado mañana.", timeout_s=0.15)
    llm.release.clear()
    c.clean("eh creo que que deberiamos salir manana no perdon pasado manana", language="es")
    c.clean("eh creo que que deberiamos salir manana no perdon pasado manana", language="es")
    assert c.last_info["skipped_busy"] is True and c.last_info["rules_only"] is True
    _settle(c, llm)


class _BusyCleaner:
    has_llm, n_ctx = True, 4096

    def __init__(self, exc):
        self.exc = exc

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages, max_tokens, timeout_s):
        raise self.exc


def test_c02_command_mode_says_the_model_is_busy_not_that_the_command_took_too_long():
    mode = CommandMode(_BusyCleaner(LlmBusy("previous LLM generation still running")), injector=None)
    with pytest.raises(CommandError) as busy:
        mode.run("make this more formal", "hey send me the file")
    assert "still finishing an earlier request" in str(busy.value) and "too long" not in str(busy.value)
    from voxflow.controller import classify_error
    assert classify_error(str(busy.value))["action"] == "", "nothing is broken: no 'open model settings' button"
    mode = CommandMode(_BusyCleaner(TimeoutError("the language model timed out")), injector=None)
    with pytest.raises(CommandError, match="took too long"):
        mode.run("make this more formal", "hey send me the file")
