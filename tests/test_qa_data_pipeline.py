"""QA "data" fixes on the dictation path (the real Controller with the test rig's fake ASR and cleaner).

D8   auto-learn never suggests what smart formatting wrote by itself ("3 PM" -> "PM")
D9   a snippet does not inflate the speed and "time saved" statistics: the spoken words count
D10  the language model's hint keeps the terms that sound like the dictation, not the alphabetically first 60
"""
from __future__ import annotations

import time

import pytest

from tests.controller_rig import make_rig
from voxflow import cleanup, store


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


# =========================================================================== D8
def test_learn_ignores_what_smart_formatting_writes(tmp_path):
    d = store.Dictionary(tmp_path)
    assert d.learn("meet at three pm", "Meet at 3 PM.") == []
    assert d.learn("call me at ten thirty am", "Call me at 10:30 AM.") == []
    got = d.learn("i pushed it to get hub at three pm", "I pushed it to GitHub at 3 PM.")
    assert [s["term"] for s in got] == ["GitHub"] and [s["term"] for s in d.suggestions()] == ["GitHub"]


def test_a_dictation_with_a_time_queues_no_am_pm_suggestion(rig):
    rig.cleaner.rewrites["see you at three pm"] = "See you at 3 PM."
    rig.dictate("see you at three pm")
    assert rig.injector.pasted[-1] == "See you at 3 PM."
    assert rig.c.dictionary.suggestions() == []


# =========================================================================== D9
def test_a_snippet_counts_the_words_that_were_said(rig):
    rig.c.snippets.add("my address", "221B Baker Street, London NW1 6XE, United Kingdom, second floor")
    rig.dictate("my address")
    entry = rig.last_entry()
    assert entry["text"].startswith("221B Baker Street") and entry["words"] == 2
    assert entry["wpm"] == round(2 / (entry["duration_s"] / 60.0), 1)
    assert rig.c.stats.summary()["total_words"] == 2, "Home's totals, speed and time saved use the spoken words"
    again = store.History(rig.c.data_dir)
    assert again.get(entry["id"])["words"] == 2, "the count survives a restart"


def test_a_stored_word_count_is_never_more_than_was_heard_or_written():
    assert store.make_entry("two words", "", words=500)["words"] == 2
    assert store.make_entry("221B Baker Street London", "my address", words=2)["words"] == 2
    assert store.make_entry("hi", "insert my address", words=3)["words"] == 3
    assert store.make_entry("x", "", words=-4)["words"] == 0


# =========================================================================== D10
def test_terms_go_to_the_cleaner_newest_first(rig):
    for term in ("Zendesk", "Asana", "Monday"):
        rig.c.dictionary.add(term)
    assert rig.c.dictionary.terms(newest_first=True) == ["Monday", "Asana", "Zendesk"]
    rig.dictate("open zen desk")
    assert rig.cleaner.clean_calls[-1][1]["terms"] == ["Monday", "Asana", "Zendesk"]


def test_the_prompt_keeps_the_terms_that_sound_like_the_dictation():
    many = [f"Aardvark{i:03d}" for i in range(200)] + ["Zendesk", "TypeScript"]
    newest_first = list(reversed(many))
    hint = cleanup._terms_hint(cleanup.rank_terms(newest_first, "please open zen desk and the type script file"))
    assert "Zendesk" in hint and "TypeScript" in hint
    assert "Zendesk" not in cleanup._terms_hint(sorted(many)), "alphabetically, it never made it into the prompt"
    ranked = cleanup.rank_terms(newest_first, "the zen dusk ticket")
    assert ranked[0] == "Zendesk", "a close sound is ranked before the rest"
    few = ["Beta", "Alpha"]
    assert cleanup.rank_terms(few, "anything") == few, "everything fits: the order is left alone"
    ranked = cleanup.rank_terms([None, " "] + many + ["Ok"], "ok then")
    assert None not in ranked and " " not in ranked and len(ranked) == len(many) + 1
    assert ranked[-1] == "Ok", "two letters are too short to be matched by sound: it keeps its place"


def test_ranking_is_cheap_even_for_a_large_dictionary():
    many = [f"Term{i:04d} name" for i in range(1500)]
    text = " ".join(["we talked about the quarterly roadmap and the kubernetes migration"] * 6)
    cleanup.rank_terms(many, text)                               # warm the phonetic cache like a real session
    t0 = time.perf_counter()
    for _ in range(5):
        cleanup.rank_terms(many, text)
    took_ms = (time.perf_counter() - t0) / 5 * 1000
    assert took_ms < 50, f"ranking 1,500 terms took {took_ms:.1f} ms"


def test_a_one_letter_token_is_never_a_name():
    assert store._interesting_casing("X") is False and store._interesting_casing("'I") is False
    assert store._interesting_casing("GitHub") is True
