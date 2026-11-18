"""voxflow.phrases: splitter, cut window, quiet point, chunk text helpers (pure; plan/PHRASE_STREAMING_DESIGN.md 4, 10.1)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from voxflow import phrases as ph
from voxflow.stream import LocalAgreement

SR = 16000


def W(*pairs):
    return [(w, float(t)) for w, t in pairs]


def spaced(words, start=1.0, step=1.0):
    """Words one ``step`` apart: every gap has plenty of room."""
    return [(w, start + i * step) for i, w in enumerate(words)]


def feed_all(sp, words, tent=None, now=None):
    return sp.feed(words, tent, words[-1][1] + 30 if now is None else now)


# ---------------------------------------------------------------------- sentence ends
def test_sentence_end_waits_for_a_later_word_or_a_pause():
    sp = ph.PhraseSplitter()
    ws = spaced(["we", "met", "today."])
    # the sentence end is the newest word: the model may still change its punctuation, and no pause is heard yet
    assert sp.feed(ws, None, ws[-1][1] + 0.5) == []
    assert sp.tick(ws[-1][1] + 0.7) == []                       # 0.7 - est 0.5 = 0.2 s of silence < 0.35
    out = sp.tick(ws[-1][1] + 1.2)                              # silence 0.7 s >= SENT_PAUSE_S
    assert [p.text for p in out] == ["we met today."] and out[0].kind == "sentence" and out[0].ends_sentence
    # a later committed word confirms it at once
    sp2 = ph.PhraseSplitter()
    out = sp2.feed(spaced(["we", "met", "today.", "Then"]), None, 4.3)
    assert [p.text for p in out] == ["we met today."] and sp2.pending == (("Then", 4.0),)


def test_abbreviations_initials_decimals_are_not_sentence_ends():
    for word in ["Dr.", "Mr.", "e.g.", "i.e.", "J.", "vs.", "No.", "Inc.", "3.30", "wait...", "now,", "x:"]:
        assert not ph._terminal(word, "Then"), word
    for word in ["done.", "done!", "done?", "done…", 'done."', "done.)", "3.30."]:
        assert ph._terminal(word, None), word
    # a.m. / p.m. / etc. end a sentence only before a capitalised word
    assert ph._terminal("a.m.", "Then") and not ph._terminal("a.m.", "then") and not ph._terminal("a.m.", None)
    assert ph._terminal("etc.", "We") and not ph._terminal("etc.", "we")
    sp = ph.PhraseSplitter()
    out = sp.feed(spaced(["see", "Dr.", "Smith", "now"]), None, 5.0)
    assert out == []                                            # no boundary: nothing ends a sentence, no pause


def test_ends_sentence_flag_follows_the_last_word():
    sp = ph.PhraseSplitter()
    out = sp.feed(spaced(["one", "two", "three,", "four"], step=1.2), None, 5.0)       # a 0.65 s gap after "three,"
    assert out[0].kind == "pause" and not out[0].ends_sentence


# ---------------------------------------------------------------------- pauses
def test_a_pause_ends_a_phrase_a_short_gap_does_not():
    sp = ph.PhraseSplitter()
    fluent = [("one", 1.0), ("two", 1.4), ("three", 1.8), ("four", 2.5)]     # gaps well under 0.6 s after the estimate
    assert sp.feed(fluent, None, 3.4) == []
    sp = ph.PhraseSplitter()
    paused = [("one", 1.0), ("two", 1.4), ("three", 1.8), ("four", 3.0)]     # est end of "three" 2.19 -> gap 0.81
    out = sp.feed(paused, None, 3.4)
    assert [p.kind for p in out] == ["pause"] and out[0].words[-1][0] == "three"


def test_min_words_and_long_pause_waiver():
    sp = ph.PhraseSplitter()
    assert sp.feed(spaced(["hello", "there", "friend"], step=1.0)[:2] + [("friend", 3.0)], None, 3.2) == []
    # two words then a 2 s hole: LONG_PAUSE_S waives MIN_WORDS
    sp = ph.PhraseSplitter()
    out = sp.feed(W(("hello", 1.0), ("there", 1.4), ("friend", 4.0)), None, 4.2)
    assert [p.text for p in out] == ["hello there"] and out[0].kind == "pause"
    # three words are enough for an ordinary pause
    sp = ph.PhraseSplitter()
    out = sp.feed(W(("a", 1.0), ("b", 1.3), ("c", 1.6), ("d", 3.5)), None, 3.8)
    assert [p.text for p in out] == ["a b c"]


def test_a_tentative_word_blocks_a_trailing_pause():
    sp = ph.PhraseSplitter()
    ws = spaced(["we", "met", "today"])
    assert sp.feed(ws, 3.5, 10.0) == []                         # a tentative word starts at 3.5: the silence is not known
    out = sp.feed([], 10.0, 10.0)                               # the new pass says it starts now: the silence is real
    assert [p.text for p in out] == ["we met today"]


def test_time_only_tick_ends_a_phrase_on_a_pause():
    sp = ph.PhraseSplitter()
    ws = spaced(["we", "met", "today"])
    assert sp.feed(ws, None, 3.4) == []
    assert sp.tick(3.9) == []                                   # silence since the estimated end (3.0 + 0.5 + ...) is short
    out = sp.tick(5.0)
    assert [p.text for p in out] == ["we met today"] and out[0].cut_hi == pytest.approx(5.0 - 0.05)
    assert sp.pending == ()


# ---------------------------------------------------------------------- cap
def test_cap_cuts_at_the_last_comma_else_at_the_last_valid_gap():
    words = [(f"w{i}" + ("," if i == 9 else ""), 1.0 + 0.9 * i) for i in range(30)]
    # no gap anywhere is long enough for a pause, no word ends a sentence
    sp = ph.PhraseSplitter(max_words=12)
    out = sp.feed(words[:13], None, words[12][1] + 0.2)
    assert len(out) == 1 and out[0].kind == "cap" and out[0].words[-1][0] == "w9,"       # the comma word wins
    # without a comma: after the last word that has a following committed word
    plain = [(f"w{i}", 1.0 + 0.9 * i) for i in range(13)]
    sp = ph.PhraseSplitter(max_words=12)
    out = sp.feed(plain, None, plain[-1][1] + 0.2)
    assert out[0].kind == "cap" and out[0].words[-1][0] == "w11"
    assert sp.pending == (("w12", plain[12][1]),)
    # a phrase cannot be capped on its newest word: it needs one committed word after the cut
    sp = ph.PhraseSplitter(max_words=12)
    sp = ph.PhraseSplitter(max_words=12)
    out = sp.feed(plain[:12], 8.4, plain[11][1] + 0.2)                     # never on the newest word
    assert [p.words[-1][0] for p in out] == ["w10"] and sp.pending == (plain[11],)


def test_cap_by_span():
    sp = ph.PhraseSplitter(max_span_s=3.0)
    ws = [(f"w{i}", 1.0 + 0.6 * i) for i in range(8)]                   # spans 4.2 s with 8 words
    out = sp.feed(ws, None, 6.0)
    assert out and out[0].kind == "cap"


def test_cap_does_not_leave_fewer_than_min_words():
    sp = ph.PhraseSplitter(max_words=2, min_words=3)
    ws = [(f"w{i}", 1.0 + 0.6 * i) for i in range(3)]
    assert sp.feed(ws, None, ws[-1][1] + 0.2) == []          # the only place to cut would leave 2 words: wait


def test_cap_with_no_valid_window_waits():
    sp = ph.PhraseSplitter(max_words=3)
    ws = [(f"w{i}", 1.0 + 0.2 * i) for i in range(5)]                   # touching words: nowhere to cut
    assert sp.feed(ws, None, 2.0) == []
    assert len(sp.rejected_at) > 0


# ---------------------------------------------------------------------- cut windows
def test_cut_window_lies_between_the_words():
    sp = ph.PhraseSplitter()
    ws = [("we", 1.0), ("met", 1.5), ("today.", 2.0), ("Then", 3.0)]
    p = sp.feed(ws, None, 3.4)[0]
    end = ph.end_est("today.", 2.0)                                     # 5 letters -> 0.25 + 0.3
    assert p.cut_lo == pytest.approx(end) and p.cut_hi == pytest.approx(3.0 - ph.CUT_MARGIN_S)
    assert p.cut_s == pytest.approx((p.cut_lo + p.cut_hi) / 2)
    assert p.cut_lo < p.cut_s < p.cut_hi and p.seq == 1


def test_end_est_is_capped():
    assert ph.end_est("a", 0.0) == pytest.approx(0.31)
    assert ph.end_est("internationalization", 1.0) == pytest.approx(1.9)


def test_a_boundary_without_room_is_skipped():
    sp = ph.PhraseSplitter()
    ws = [("we", 1.0), ("met", 1.3), ("today.", 1.6), ("Then", 1.9), ("we", 2.2), ("left.", 2.5), ("Ok", 4.0)]
    out = sp.feed(ws, None, 4.2)
    assert [p.text for p in out] == ["we met today. Then we left."]      # the first sentence end had no room
    assert sp.rejected_at == set() and sp.rejected_final == 1 and sp.pending == (("Ok", 4.0),)
    sp.reset()
    assert sp.rejected_at == set()


def test_rejected_boundary_is_remembered_until_it_is_emitted():
    sp = ph.PhraseSplitter()
    ws = [("we", 1.0), ("met", 1.3), ("today.", 1.6), ("Then", 1.9)]
    assert sp.feed(ws, None, 2.0) == [] and sp.rejected_at == {1.6}


def test_reset():
    sp = ph.PhraseSplitter()
    sp.feed(spaced(["we", "met", "today.", "Then"]), 9.0, 9.9)
    sp.reset()
    assert sp.pending == () and sp.tick(50.0) == []
    out = sp.feed(spaced(["we", "met", "today.", "Then"]), None, 9.9)
    assert out[0].seq == 1


def test_empty_input_never_raises():
    sp = ph.PhraseSplitter()
    assert sp.feed([], None, 0.0) == [] and sp.tick(1.0) == [] and sp.feed([], 1.0, 1.0) == []


# ---------------------------------------------------------------------- punctuation flipping through the real agreement
def test_flipping_punctuation_between_passes_does_not_move_a_boundary():
    la, sp = LocalAgreement(), ph.PhraseSplitter()
    passes = [("we meet at 3:30 tomorrow.", 1.0), ("we meet at 3 30 tomorrow", 1.0), ("we meet at 3:30 tomorrow. Thanks", 1.0)]
    emitted, committed_all = [], []
    for k, (text, t0) in enumerate(passes):
        hyp = [(w, t0 + 0.7 * i) for i, w in enumerate(text.split())]
        k0 = len(la.committed)
        tent = la.update(la.new_part(hyp))
        new = la.committed[k0:]
        committed_all += new
        emitted += sp.feed(new, tent[0][1] if tent else None, 4.0 + k)
        if k == 1:
            assert emitted == []                                        # the sentence end is not committed yet
    got = [w for p in emitted for w in p.words] + list(sp.pending)
    assert got == committed_all and len(got) == len(set(got))
    assert [w for w, _ in committed_all][:5] == ["we", "meet", "at", "3:30", "tomorrow."][:len(committed_all[:5])]


# ---------------------------------------------------------------------- property
def test_property_no_word_is_lost_or_repeated():
    rnd = random.Random(7)
    for _ in range(2000):
        sp = ph.PhraseSplitter(max_words=rnd.choice([5, 25]), max_span_s=rnd.choice([4.0, 10.0]))
        t, now, committed, emitted, last_cut = 0.5, 0.0, [], [], 0.0
        for _step in range(rnd.randint(1, 12)):
            new = []
            for _ in range(rnd.randint(0, 4)):
                t += rnd.choice([0.05, 0.2, 0.4, 0.9, 1.8])
                w = "".join(rnd.choice("abcdefgh") for _ in range(rnd.randint(1, 9)))
                w = rnd.choice([w, w + ",", w + ".", w + "?", "Dr.", w + "..."])
                new.append((w, t))
            committed += new
            now = max(now, t + rnd.choice([0.0, 0.3, 1.0, 3.0]))
            tent = rnd.choice([None, now, t + 0.1])
            out = sp.tick(now) if rnd.random() < 0.3 and not new else sp.feed(new, tent, now)
            for p in out:
                assert p.cut_lo - 1e-9 <= p.cut_s <= p.cut_hi + 1e-9 and p.cut_s >= last_cut
                assert p.cut_lo >= max(ph.end_est(*p.words[-1]), 0.0) - 1e-9
                last_cut = p.cut_s
                emitted.append(p)
        got = [w for p in emitted for w in p.words] + list(sp.pending)
        assert got == committed
        assert [p.seq for p in emitted] == list(range(1, len(emitted) + 1))
        assert all(p.words and p.text == " ".join(w for w, _ in p.words) for p in emitted)


# ---------------------------------------------------------------------- quiet point
def tone(seconds, amp=0.3):
    return (amp * np.sin(np.arange(int(seconds * SR)) * 0.3)).astype(np.float32)


def test_quiet_point_picks_the_quietest_frame():
    s = np.concatenate([tone(0.2), np.zeros(int(0.04 * SR), np.float32), tone(0.2)])
    q = ph.quiet_point(s, 1.0, 1.0, 1.44, 0.04)
    assert q == pytest.approx(1.0 + 0.2 + 0.02)


def test_quiet_point_flat_signal_is_the_middle():
    s = np.zeros(SR, np.float32)
    assert ph.quiet_point(s, 2.0, 2.0, 2.4, 0.04) == pytest.approx(2.2, abs=0.021)
    assert ph.quiet_point(np.full(SR, 0.1, np.float32), 2.0, 2.0, 2.4) == pytest.approx(2.2, abs=0.021)


def test_quiet_point_empty_or_short_slice_is_the_middle():
    assert ph.quiet_point(np.zeros(0, np.float32), 1.0, 1.0, 2.0) == 1.5
    assert ph.quiet_point(np.zeros(100, np.float32), 1.0, 1.0, 1.003) == pytest.approx(1.0015)
    assert ph.quiet_point(np.zeros(SR, np.float32), 1.0, 1.0, 1.01) == pytest.approx(1.005)       # window < one frame


def test_quiet_point_is_clamped_to_the_window():
    s = np.concatenate([tone(0.1), np.zeros(int(0.5 * SR), np.float32)])
    q = ph.quiet_point(s, 0.0, 0.05, 0.5, 0.04)
    assert 0.05 <= q <= 0.5
    # samples that start before the window (t0 < lo) are cropped
    q = ph.quiet_point(np.concatenate([np.zeros(SR // 2, np.float32), tone(0.5)]), 0.0, 0.2, 0.8, 0.04)
    assert 0.2 <= q <= 0.5


# ---------------------------------------------------------------------- finish_chunk
def fc(prev, cleaned, raw=None, kind="sentence", ends=True):
    return ph.finish_chunk(prev, cleaned, cleaned if raw is None else raw, kind, ends)


def test_finish_chunk_spacing():
    assert fc("", "Hello there.") == "Hello there."
    assert fc("Hello there.", "How are you?") == " How are you?"
    assert fc("Hello ", "world.", "world") == "world."
    assert fc("Hello\n", "World.") == "World."
    assert fc("Hello\r", "World.") == "World."
    assert fc("Hello there", ", and more") == ", and more"
    assert fc("Hello there", "and more", "and more", "pause", False) == " and more"
    for lead in ",.;:!?)]}%":
        assert fc("Hello there", lead + "x") == lead + "x"
    assert fc("Hello", "\nNext") == "\nNext" and fc("Hello", " next") == " next"
    assert fc("Hello", "") == ""


def test_finish_chunk_strips_an_added_period_for_pause_and_cap_only():
    assert fc("", "We met today.", "we met today", "pause", False) == "We met today"
    assert fc("", "Did we meet?", "did we meet", "cap", False) == "Did we meet"
    assert fc("", "We met today.", "we met today", "sentence") == "We met today."
    assert fc("", "We met today.", "we met today", "tail") == "We met today."
    # the ASR itself ended the phrase: keep it
    assert fc("", "We met today.", "we met today.", "pause", True) == "We met today."
    assert fc("", "We met!", "we met", "pause", False) == "We met!"           # only "." and "?" were cleanup's
    assert fc("", "We met", "we met", "pause", False) == "We met"


def test_finish_chunk_lowercases_by_the_asr_evidence():
    prev = "so we went to"
    assert fc(prev, "The shop.", "the shop", "pause", False) == " the shop"             # the / The lowered
    assert fc(prev, "Berlin.", "Berlin", "tail") == " Berlin."                          # raw capitalised: kept
    assert fc(prev, "I think.", "I think", "tail") == " I think."                       # raw "I": kept
    assert fc(prev, "The shop", "um so the shop", "sentence") == " the shop"            # filler in front, raw word "the"
    assert fc(prev, "NASA said", "nasa said", "tail") == " NASA said"                   # all caps kept
    assert fc(prev, "Hello", "something else", "tail") == " Hello"                      # no match: kept
    assert fc(prev, "The, shop", "the shop", "tail") == " the, shop"
    assert fc(prev, "Der Hund", "der Hund", "tail") == " der Hund"
    assert fc(prev, "A shop", "a shop", "tail") == " a shop"
    assert fc(prev, "42 shops", "42 shops", "tail") == " 42 shops"                      # not a letter: untouched
    assert fc(prev, "the shop", "the shop", "tail") == " the shop"                      # already lower: untouched


def test_finish_chunk_keeps_the_capital_after_a_sentence_end_or_a_line_or_nothing():
    for prev in ["", "It is done.", "Done!", "Done?", "Done…", "Name:", "Done.  ", "line one\n", "line one\r\n"]:
        out = fc(prev, "The shop", "the shop", "tail")
        assert "The shop" in out and "the shop" not in out


def test_closing_period():
    cp = ph.closing_period
    assert cp("so we went to the shop", "pause") == "." and cp("so we went to the shop", "cap") == "."
    assert cp("so we went to the shop", "sentence") == "" and cp("so we went to the shop", "tail") == ""
    assert cp("Hello. we went home", "pause") == "." and cp("Hello. we went", "pause") == ""
    assert cp("Done. we went home", "cap") == "."
    assert cp("Done. Hello there", "pause") == ""                      # the last sentence has two words
    assert cp("Done.\nHello there", "pause") == ""
    assert cp("so we went to the shop,", "pause") == "" and cp("so we went to the shop ", "pause") == ""
    assert cp("", "pause") == "" and cp("so we went to the 5", "pause") == "."
