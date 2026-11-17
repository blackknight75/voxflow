"""Grammar / punctuation improvements of the English cleanup: the rule fixes, the run-on gate for the language
model, the guardrail changes, and a score floor on the frozen eval set (tests/grammar_eval.json)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from voxflow import cleanup
from voxflow.cleanup import Cleaner, _accept, grammar_rules, looks_unpunctuated, rule_clean

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- grammar rules
@pytest.mark.parametrize("raw, want", [
    ("Yes that sounds good to me.", "Yes, that sounds good to me."),
    ("Well I guess we can try it.", "Well, I guess we can try it."),
    ("No I don't think so.", "No, I don't think so."),
    ("Okay so here is the plan.", "Okay, so here is the plan."),
    ("Done. Yeah we shipped it.", "Done. Yeah, we shipped it."),
    ("Their going to be late.", "They're going to be late."),
    ("I think their is a problem.", "I think there is a problem."),
    ("Its been a long week.", "It's been a long week."),
    ("Your welcome to join us.", "You're welcome to join us."),
    ("Lets go with the second option.", "Let's go with the second option."),
    ("So lets meet at noon.", "So let's meet at noon."),
    ("I would of called you.", "I would have called you."),
    ("It was smaller then expected.", "It was smaller than expected."),
    ("We need more then ten people.", "We need more than ten people."),
    ("Okay, lets go.", "Okay, let's go."),
    ("Hey you made it.", "Hey, you made it."),
])
def test_grammar_rules_fix(raw, want):
    assert grammar_rules(raw) == want


@pytest.mark.parametrize("text", [
    "No one came to the party.",            # "No one" is not an interjection
    "Well done, everyone.",
    "Thanks for the update.",
    "Their going away party is on Friday.",  # a possessive before a gerund stays
    "Its owner called.",                     # possessive "its"
    "Your welcome email is ready.",          # "welcome" as an adjective
    "She lets me borrow the car.",           # the verb "lets"
    "We could of course try again.",         # "could of course" is not "could have"
    "Finish it and then we can go.",         # "then" of time
    "The dog wagged its tail.",
    "Yesterday I left early.",
    "Right there in the corner.",
    "The company has its very own office.",
    "The market and its going rate.",
    # review regressions
    "Things were better then.",
    "It was easier then, back in school.",
    "The plan is better then we go home.",   # ambiguous: left alone
    "Hey there, how are you?",
    "Hey there.",
    "She lets the dog out and lets go.",
    "He finally lets go.",
    "Remove its not-null checks.",
    "Its a-side was the hit.",
])
def test_grammar_rules_leave_correct_text(text):
    assert grammar_rules(text) == text


def test_rule_clean_applies_grammar_rules():
    assert rule_clean("yes that works for me") == "Yes, that works for me."
    assert rule_clean("lets go with the second option") == "Let's go with the second option."


# ---------------------------------------------------------------- the run-on gate
@pytest.mark.parametrize("text", [
    "I finished the report last night I will send it to you after lunch.",
    "The server crashed again we need to look at the logs.",
    "If you have time today could you review my pull request?",
    "When the build finishes let me know.",
    "The plan looks good however we still need a budget.",
    "I saw your message what time works for you.",
])
def test_looks_unpunctuated(text):
    assert looks_unpunctuated(text)


@pytest.mark.parametrize("text", [
    "I'll review the pull request this afternoon and leave comments.",
    "Can you send me the latest numbers before Friday?",
    "When you get a chance, please review the draft.",
    "I think we should wait until the end of the month.",
    "Good morning everyone, and thank you for joining the weekly planning call.",
    "Please let me know if you need anything else from me.",
    # review regression: an opening adverbial is not a run-on, and "told you" is an object, not a new clause
    "Last week I told you that.",
    "Yesterday we shipped it.",
    "This morning I spoke with the client.",
    "Back then we had no tests at all.",
])
def test_looks_unpunctuated_leaves_finished_text(text):
    assert not looks_unpunctuated(text)


def _fake_cleaner(reply, punctuation_pass):
    c = Cleaner()
    c._ready, c._llm = True, object()
    c.punctuation_pass = punctuation_pass
    calls = []

    def fake(body, max_ms=None):
        calls.append(body)
        return reply(body)

    c._llm_clean = fake
    return c, calls


def test_punctuation_pass_sends_run_ons_to_the_model():
    text = "I finished the report last night I will send it to you after lunch."
    fixed = "I finished the report last night. I will send it to you after lunch."
    c, calls = _fake_cleaner(lambda body: fixed, punctuation_pass=True)
    assert c.clean(text) == fixed
    assert calls == [text]


def test_without_punctuation_pass_finished_looking_text_skips_the_model():
    """A slow (CPU) model keeps the old behaviour: text the rules leave alone is never sent to it."""
    text = "I finished the report last night I will send it to you after lunch."
    c, calls = _fake_cleaner(lambda body: body, punctuation_pass=False)
    assert c.clean(text) == text
    assert calls == []


def test_punctuation_pass_skips_text_that_reads_fine():
    c, calls = _fake_cleaner(lambda body: body, punctuation_pass=True)
    assert c.clean("Can you send me the latest numbers before Friday?") == \
        "Can you send me the latest numbers before Friday?"
    assert calls == []


def test_grammar_rule_fix_alone_does_not_wake_the_model():
    """The comma the rules add after "Yes" must not count as "the rules changed it": that would make a slow model
    run where it did not before."""
    c, calls = _fake_cleaner(lambda body: body, punctuation_pass=False)
    assert c.clean("Yes that sounds good to me.") == "Yes, that sounds good to me."
    assert calls == []


class _Timed:
    """Stands in for Llama: every chat completion takes ``seconds``."""
    def __init__(self, seconds):
        self.seconds = seconds

    def tokenize(self, *a, **k):
        return [1, 2, 3]

    def create_chat_completion(self, **k):
        import time

        time.sleep(self.seconds)
        return iter([{"choices": [{"delta": {"content": "ok"}}]}])


def _load(monkeypatch, libs, seconds, device="auto", count=None):
    monkeypatch.setattr(cleanup, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(cleanup, "_select_llama_libs", lambda device: libs)
    monkeypatch.setattr(cleanup, "_gpu_offload_supported", lambda: True)
    c = Cleaner(model=__file__, device=device, timeout_s=4.0)
    fake = _Timed(seconds)
    if count is not None:
        orig = fake.create_chat_completion

        def counted(**k):
            count.append(k["messages"][-1]["content"])
            return orig(**k)

        fake.create_chat_completion = counted
    monkeypatch.setattr(c, "_build", lambda path, gpu: fake)
    c.load()
    return c, fake


@pytest.mark.parametrize("seconds, expect", [(0.01, True), (0.6, False)])
def test_vulkan_speed_check_decides_punctuation_pass(monkeypatch, seconds, expect):
    """A non-default build (Vulkan) is timed at load anyway: that measurement is reused."""
    c, _ = _load(monkeypatch, "vulkan", seconds)
    assert c.backend == "vulkan"
    assert c.punctuation_pass is expect


def test_default_cuda_build_adds_no_speed_check_to_the_load(monkeypatch):
    """Review: the default CUDA build must not pay an extra timed sentence at every start."""
    calls = []
    c, _ = _load(monkeypatch, None, 0.0, count=calls)
    assert c.backend == "cuda"
    assert len(calls) == 1                      # the warm-up only, as before this branch
    assert cleanup._SPEED_CHECK not in calls
    assert c.punctuation_pass is True           # optimistic until real calls say otherwise


def test_real_calls_switch_the_punctuation_pass(monkeypatch):
    c, fake = _load(monkeypatch, None, 0.0)
    text = "I finished the report last night I will send it to you after lunch."
    fake.seconds = 1.0                          # a slow GPU: about 70 ms per word
    c.clean(text)
    assert c.punctuation_pass is False
    calls = []
    c._llm_clean = lambda body, max_ms=None: calls.append(body) or body
    assert c.clean(text) == text and calls == []   # no extra call while slow
    c._note_speed(text, 0.05)
    c._note_speed(text, 0.05)
    assert c.punctuation_pass is True           # fast again: back on


def test_timeouts_count_as_slow(monkeypatch):
    c, _ = _load(monkeypatch, None, 0.0)

    def stuck(body, max_ms=None):
        import time

        time.sleep(0.6)                         # about 46 ms per word, then the budget ran out
        raise TimeoutError("busy")

    c._llm_clean = stuck
    c.clean("i finished the report last night i will send it to you after lunch")
    assert c.punctuation_pass is False


def test_pass_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(cleanup, "PUNCTUATION_PASS_MAX_MS", 0)
    c, _ = _load(monkeypatch, None, 0.0)
    assert c.punctuation_pass is False
    c._note_speed("a b c", 0.001)
    assert c.punctuation_pass is False


def test_cpu_model_never_gets_punctuation_pass(monkeypatch):
    c, fake = _load(monkeypatch, "cpu", 0.0, device="cpu")
    assert c.backend == "cpu" and c.punctuation_pass is False
    c.clean("i finished the report last night i will send it to you after lunch")
    # the CPU learns its speed (QA C-10: a call it cannot finish in time is not started), never the punctuation pass
    assert c.punctuation_pass is False and c._ms_per_word is not None


def test_failed_gpu_attempt_leaves_no_pass(monkeypatch):
    """The GPU loads but is too slow: the CPU fallback must not inherit the GPU's punctuation pass."""
    c, _ = _load(monkeypatch, "vulkan", 2.1)
    assert c.backend == "cpu" and c.punctuation_pass is False


# ---------------------------------------------------------------- guardrails
@pytest.mark.parametrize("src, out", [
    ("Their going to be late because there car broke down.", "They're going to be late because their car broke down."),
    ("It was smaller then we expected.", "It was smaller than we expected."),   # the model may still fix it
    ("Its been a long week and the team needs it's rest.", "It's been a long week, and the team needs its rest."),
    ("Send it to mark no wait send it to lisa.", "Send it to Lisa."),
    ("I finished the report last night I will send it to you after lunch.",
     "I finished the report last night. I will send it to you after lunch."),
])
def test_accept_allows_grammar_fixes(src, out):
    assert _accept(src, out)


@pytest.mark.parametrize("src, out", [
    ("Your welcome to join us for dinner.", "Welcome to join us for dinner."),        # a word dropped
    ("Send it to mark no wait send it to lisa.", "Send it to Mark."),                 # the earlier value kept
    ("Please email me at anna dot lee at example dot com.", "Please email me at anna.dot.lee at example.dot.com."),
    ("We have three options rewrite it patch it or leave it alone.",
     "We have three options: rewrite, patch it, or leave it alone."),
])
def test_accept_still_rejects(src, out):
    assert not _accept(src, out)


# ---------------------------------------------------------------- the eval set
def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_grammar", ROOT / "scripts" / "eval_grammar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_set_is_well_formed():
    cases = _eval_module().load_cases()
    assert 60 <= len(cases) <= 80
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["raw"] and c["expect"] and all(isinstance(e, str) for e in c["expect"])


def test_eval_score():
    ev = _eval_module()
    assert ev.score("Hello there.", ["Hello there."]) == {"exact": True, "pter": 0.0, "wer": 0.0}
    s = ev.score("hello there", ["Hello there."])
    assert not s["exact"] and s["pter"] == 1.0 and s["wer"] == 0.0


def test_rules_only_score_floor():
    """The rule pass alone (a PC where the model could not load), heldout included."""
    ev = _eval_module()
    s = ev.evaluate(Cleaner(use_llm=False), ev.load_cases(), quiet=True)
    assert s["exact"] >= 58, s["exact"]          # measured 58/107 (the base: 50/107)
    assert s["pter"] <= 0.12, s["pter"]           # measured 0.118 (the base: 0.131)


def test_heldout_sentences_come_through_unchanged():
    """Already-correct sentences (cat "heldout", not used to tune the rules): the rules leave them alone and a fast
    GPU's punctuation pass skips all but one of them (a fronted "Over the weekend we", measured)."""
    cases = [c for c in _eval_module().load_cases() if c["cat"] == "heldout"]
    assert len(cases) == 15
    extra = []
    for case in cases:
        assert rule_clean(case["raw"]) == case["raw"]
        calls = {}
        for on in (False, True):
            c, calls[on] = _fake_cleaner(lambda body: body, punctuation_pass=on)
            assert c.clean(case["raw"]) == case["raw"]
        extra += calls[True][len(calls[False]):]
    assert extra == ["Over the weekend we cleaned out the garage."], extra


def test_command_mode_keeps_the_old_rule_set():
    """Spoken instructions are not prose: the new comma / homophone rules do not touch them."""
    from voxflow.command import clean_instruction

    assert clean_instruction("yes make it shorter") == "Yes make it shorter"
    assert clean_instruction("rewrite it so its not so formal") == "Rewrite it so its not so formal"
    assert rule_clean("yes make it shorter", grammar=False) == "Yes make it shorter."
