"""LocalAgreement.new_part matches re-heard text on joined normalised characters, so a pass that splits the same
characters differently ("3.30" / "3 30" / "3:30") does not add them as new words. The hypotheses of the first test are
the real ones captured from scripts/bench_phrases.py (CPU, `--thinking`, under coverage tracing) where a phrase was typed
as "... at 3.30 3 30 in the afternoon." (bench, round 4)."""
from __future__ import annotations

from voxflow.stream import LocalAgreement


def words(text, step=0.3, start=0.0):
    return [(w, round(start + i * step, 2)) for i, w in enumerate(text.split())]


def hyp_at(pairs, off):
    return [(w, round(t + off, 2)) for w, t in pairs]


def run(passes, off=0.0):
    """Feed hypotheses (word, time in window) the way Previewer does; returns (committed + tentative words)."""
    la = LocalAgreement()
    tent = []
    for pairs in passes:
        tent = la.update(la.new_part(hyp_at(pairs, off), off))
    return [w for w, _ in la.committed] + [w for w, _ in tent], la


# the real sequence (window offset 14.42 s), passes 41-47 of the captured run, trimmed to what matters
REAL = [
    [("For", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.69), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.73), ("for", 3.05), ("Thursday", 3.21), ("at", 3.61), ("3.30.", 3.69)],
    [("for", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.69), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.81), ("for", 3.05), ("Thursday", 3.21), ("at", 3.69), ("3.30", 3.77), ("in", 4.41), ("the", 4.49)],
    [("For", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.69), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.81), ("for", 3.13), ("Thursday", 3.29), ("at", 3.77), ("3", 3.85), ("30", 4.17), ("in", 4.65),
     ("the", 4.73), ("afternoon.", 4.81)],
    [("For", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.69), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.81), ("for", 3.13), ("Thursday", 3.29), ("at", 3.77), ("3", 3.85), ("30", 4.17), ("in", 4.73),
     ("the", 4.81), ("afternoon.", 4.97)],
    [("For", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.69), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.81), ("for", 3.13), ("Thursday", 3.29), ("at", 3.77), ("3.30", 3.93), ("in", 4.73), ("the", 4.89),
     ("afternoon.", 4.97)],
    [("For", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.61), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.81), ("for", 3.13), ("Thursday", 3.29), ("at", 3.77), ("3.30", 3.93), ("in", 4.73), ("the", 4.89),
     ("afternoon.", 5.05)],
    [("For", 0.01), ("the", 0.17), ("lazy", 0.33), ("dog.", 0.89), ("Please", 1.69), ("schedule", 2.17), ("the", 2.65),
     ("meeting", 2.81), ("for", 3.13), ("Thursday", 3.29), ("at", 3.77), ("3:30", 3.93), ("in", 4.73), ("the", 4.89),
     ("afternoon.", 5.05)],
]


def digits(words_):
    return "".join(c for c in " ".join(words_) if c.isdigit())


def test_the_real_captured_sequence_does_not_double_the_number():
    # the window offset was rounded in the capture; which of "3.30 30" / "3.30 3 30" the old code gave depended on a
    # float tie at the 80 ms time cut, so several offsets around the real one are tried
    for off in (14.415, 14.4175, 14.42, 14.4225, 14.4249):
        out, la = run(REAL, off=off)
        assert digits(out) == "330", (off, out)                        # "3.30 30" / "3.30 3 30" doubled it
        low = [w.lower().strip(".") for w in out]
        assert low.count("in") == 1 and low.count("afternoon") == 1 and low.count("thursday") == 1


def test_committed_three_thirty_then_heard_as_three_space_thirty():
    out, la = run([words("we meet at 3.30"), words("we meet at 3.30 in"), words("meet at 3 30 in the room", start=0.3)])
    assert digits(out) == "330"
    assert [w for w, _ in la.committed][:4] == ["we", "meet", "at", "3.30"]


def test_committed_three_space_thirty_then_heard_as_three_thirty():
    passes = [words("we meet at 3 30"), words("we meet at 3 30 in"), words("meet at 3.30 in the room", start=0.3)]
    out, la = run(passes)
    assert digits(out) == "330"
    assert [w for w, _ in la.committed][:5] == ["we", "meet", "at", "3", "30"]


def test_colon_and_dot_and_space_forms_are_the_same_characters():
    for committed, heard in (("3:30", "3 30"), ("3 30", "3:30"), ("3.30", "3:30"), ("o'clock", "o'clock")):
        la = LocalAgreement()
        la.committed = [(w, 1.0 + 0.3 * i) for i, w in enumerate(committed.split())]
        t_last = la.committed[-1][1]
        hyp = [(w, round(t_last + 0.12 * i - 0.1, 2)) for i, w in enumerate(heard.split())] + [("next", t_last + 0.9)]
        assert la.new_part(hyp) == [("next", t_last + 0.9)], (committed, heard)


def test_a_genuine_repetition_that_starts_clearly_later_is_still_new():
    for word in ("very", "no", "30"):
        la = LocalAgreement()
        la.committed = [("go", 0.5), (word, 1.0)]
        hyp = [(word, 1.0), (word, 1.6), ("on", 2.0)]                  # heard again, then said again 0.6 s later
        assert la.new_part(hyp, since=0.4) == [(word, 1.6), ("on", 2.0)], word
    la = LocalAgreement()                                              # "30 30" spoken twice while "30" is committed
    la.committed = [("at", 0.5), ("30", 1.0)]
    assert la.new_part([("30", 1.0), ("30", 1.5)]) == [("30", 1.5)]
    # when the window starts after the last committed word, nothing committed can be heard again
    la.committed = [("very", 1.0)]
    assert la.new_part([("very", 2.0), ("good", 2.3)], since=1.8) == [("very", 2.0), ("good", 2.3)]


def test_words_without_characters_are_never_matched_and_fall_back_to_the_time_cut():
    la = LocalAgreement()
    la.committed = [("-", 1.0)]
    assert la.new_part([("-", 1.0), ("x", 1.2)]) == [("x", 1.2)]


def test_committed_three_then_heard_as_three_dot_thirty_adds_only_the_rest():
    """The traced bench run that typed "... at 3 3.30 in the afternoon.": "3" was committed (the window ended there), the
    next pass heard the one word "3.30"."""
    passes = [words("we meet at 3"), words("we meet at 3"), words("we meet at 3.30 in the room")]
    out, la = run(passes)
    assert out[:5] == ["we", "meet", "at", "3", "30"] and digits(out) == "330" and out[5:] == ["in", "the", "room"]
    # the same with a colon, and when the committed text ends inside a longer run
    out, _ = run([words("see you at 3"), words("see you at 3"), words("see you at 3:30 sharp")])
    assert out == ["see", "you", "at", "3", "30", "sharp"]


def test_a_word_that_merely_starts_with_the_committed_one_is_not_split():
    # "the" committed, now heard as "there": a different word, not a re-tokenisation: nothing is invented
    la = LocalAgreement()
    la.committed = [("say", 0.5), ("the", 1.0)]
    assert la.new_part([("say", 0.5), ("there", 1.0), ("now", 1.4)]) == [("now", 1.4)]
    assert ("re", 1.0) not in la.new_part([("say", 0.5), ("there", 1.0), ("now", 1.4)])
    la.committed = [("at", 0.5), ("3", 1.0)]
    assert la.new_part([("at", 0.5), ("3.", 1.0)]) == []                   # nothing after the separator: nothing new


def test_the_second_captured_sequence_three_then_thirty_then_three_thirty_never_doubles():
    """The traced bench run that typed "... at 3 30 3 30 in the afternoon." (real new_part inputs, window offset 0):
    "3" committed from "3 30.", the next pass heard one word "3:30" (only "30" is new), and the following passes split it
    again as "3" "30", 0.4 s after "3"."""
    la = LocalAgreement()
    la.committed = [("for", 4.65), ("Thursday", 4.73)]
    la.prev = [("at", 5.13), ("3.", 5.29)]
    passes = [
        [("Please", 3.29), ("schedule", 3.77), ("the", 4.25), ("meeting", 4.41), ("for", 4.73), ("Thursday", 4.89),
         ("at", 5.29), ("3", 5.45), ("30.", 5.69)],
        [("the", 4.25), ("meeting", 4.41), ("for", 4.73), ("Thursday", 4.89), ("at", 5.29), ("3:30", 5.45), ("in", 6.09),
         ("the", 6.17), ("aft.", 6.25)],
        [("meeting", 4.41), ("for", 4.73), ("Thursday", 4.89), ("at", 5.37), ("3", 5.53), ("30", 5.85), ("in", 6.33),
         ("the", 6.49), ("afternoon.", 6.57)],
        [("the", 4.33), ("meeting", 4.41), ("for", 4.73), ("Thursday", 4.89), ("at", 5.37), ("3:30", 5.53), ("in", 6.33),
         ("the", 6.49), ("afternoon.", 6.65)],
        [("meeting", 4.41), ("for", 4.73), ("Thursday", 4.89), ("at", 5.37), ("3", 5.53), ("30", 5.85), ("in", 6.33),
         ("the", 6.49), ("afternoon.", 6.65)],
        [("at", 5.37), ("3:30", 5.53), ("in", 6.33), ("the", 6.49), ("afternoon.", 6.65), ("And", 8.2)],
    ]
    for hyp in passes:
        tent = la.update(la.new_part(hyp, 0.0))
        out = [w for w, _ in la.committed] + [w for w, _ in tent]
        assert digits(out[2:]) == "330", out                     # "3 30 3 30" / "3 30 30" doubled it
    assert out[-3:] == ["in", "the", "afternoon."] or out[-1] == "And"
