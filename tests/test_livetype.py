"""``voxflow.livetype``: the live-typing logic against a simulated text field. No keyboard, window or Qt is used."""
from __future__ import annotations

import random

import pytest

from voxflow import livetype as L

PRE = "Hi "


class SimField:
    """A text field with the cursor at its end that applies ``type_text`` / ``backspace`` the way an editor does:
    one Backspace removes one cluster. Text that was there before (``pre``) must never be touched."""

    def __init__(self, pre: str = PRE):
        self.pre = pre
        self.text = pre
        self.typed_clusters = 0          # clusters this test double was asked to type
        self.deleted_clusters = 0
        self.calls: list[tuple] = []

    def type_text(self, text: str) -> None:
        self.calls.append(("type", text))
        self.text += text
        self.typed_clusters += len(L.clusters(text))

    def backspace(self, count: int) -> None:
        self.calls.append(("back", count))
        have = L.clusters(self.text)
        assert count <= len(have) - len(L.clusters(self.pre)), "deleted text that was there before"
        self.text = "".join(have[:len(have) - count])
        self.deleted_clusters += count
        assert self.text.startswith(self.pre)


class BrokenKeys:
    """Keys that raise after ``ok`` successful calls."""

    def __init__(self, ok: int = 0):
        self.ok = ok

    def _call(self):
        if self.ok <= 0:
            raise OSError("SendInput failed")
        self.ok -= 1

    def type_text(self, text):
        self._call()

    def backspace(self, count):
        self._call()


# ====================================================================== clusters / plan_edit
@pytest.mark.parametrize("text, expected", [
    ("", []),
    ("abc", ["a", "b", "c"]),
    ("éx", ["é", "x"]),                                      # combining acute
    ("a\r\nb", ["a", "\r\n", "b"]),
    ("a\n\rb", ["a", "\n", "\r", "b"]),                                  # only CR LF is one key
    ("😀z", ["😀", "z"]),                            # a surrogate pair written as two chars
    ("\U0001F600", ["\U0001F600"]),
    ("\U0001F44D\U0001F3FD!", ["\U0001F44D\U0001F3FD", "!"]),            # skin tone
    ("\U0001F468‍\U0001F469‍\U0001F467", ["\U0001F468‍\U0001F469‍\U0001F467"]),
    ("❤️", ["❤️"]),                                  # variation selector
    ("\U0001F1E9\U0001F1EA\U0001F1EB\U0001F1F7", ["\U0001F1E9\U0001F1EA", "\U0001F1EB\U0001F1F7"]),   # two flags
    ("́a", ["́", "a"]),                                        # a lone mark starts a cluster
    ("\U000E0067x", ["\U000E0067", "x"]),                                # a tag character with nothing before it
])
def test_clusters(text, expected):
    assert L.clusters(text) == expected
    assert "".join(L.clusters(text)) == text


@pytest.mark.parametrize("typed, target, expected", [
    ("", "", (0, "")),
    ("", "hello", (0, "hello")),
    ("hello", "hello", (0, "")),
    ("hello", "hello world", (0, " world")),
    ("hello wrld", "hello world", (3, "orld")),       # keep "hello w", redo "rld"
    ("hello", "", (5, "")),
    ("cat", "dog", (3, "dog")),
    ("e", "é", (1, "é")),                                    # never leaves the base letter behind
    ("é", "e", (1, "e")),
    ("a\r\n", "a\r", (1, "\r")),                                         # CR LF is one Backspace
    ("x\U0001F44D", "x\U0001F44D\U0001F3FD", (1, "\U0001F44D\U0001F3FD")),
    ("\U0001F1E9\U0001F1EA", "\U0001F1E9\U0001F1EA\U0001F1EB\U0001F1F7", (0, "\U0001F1EB\U0001F1F7")),
])
def test_plan_edit_examples(typed, target, expected):
    assert L.plan_edit(typed, target) == expected


def test_plan_edit_applied_to_a_field():
    field = SimField()
    field.type_text("the quick brown")
    backspaces, text = L.plan_edit("the quick brown", "the quiet brown fox")
    field.backspace(backspaces)
    field.type_text(text)
    assert field.text == PRE + "the quiet brown fox"


# ====================================================================== LiveTyper
def test_types_only_new_committed_words():
    field = SimField()
    typer = L.LiveTyper(field)
    assert typer.state == "typing"
    assert typer.update("") is False                     # nothing yet
    assert typer.update("please") is True
    assert typer.update("please") is False               # unchanged: no keys
    assert typer.update("please schedule  ") is True     # trailing blanks are not typed
    assert typer.update("please schedule the meeting") is True
    assert field.calls == [("type", "please"), ("type", " schedule"), ("type", " the meeting")]
    assert typer.typed == "please schedule the meeting" and field.text == PRE + typer.typed


def test_finish_reconciles_to_the_final_text():
    field = SimField()
    typer = L.LiveTyper(field)
    typer.update("please schedule the meating")
    assert typer.finish("Please schedule the meeting.") == L.TYPED
    assert field.text == PRE + "Please schedule the meeting."
    assert field.calls[-2:] == [("back", 27), ("type", "Please schedule the meeting.")]
    assert typer.typed == "Please schedule the meeting." and typer.state == "done"


def test_finish_with_the_same_text_sends_no_keys():
    field = SimField()
    typer = L.LiveTyper(field)
    typer.update("hello there")
    assert typer.finish("hello there") == L.TYPED
    assert field.calls == [("type", "hello there")]


def test_finish_only_deletes_the_changed_tail():
    field = SimField()
    typer = L.LiveTyper(field)
    typer.update("send it to jon")
    assert typer.finish("send it to John.") == L.TYPED
    assert field.calls[1:] == [("back", 3), ("type", "John.")]


def test_finish_with_a_shorter_final_only_deletes():
    field = SimField()
    typer = L.LiveTyper(field)
    typer.update("hello there um")
    assert typer.finish("hello there") == L.TYPED
    assert field.calls[1:] == [("back", 3)] and field.text == PRE + "hello there"


def test_nothing_typed_means_fallback():
    field = SimField()
    typer = L.LiveTyper(field)
    assert typer.finish("Hello.") == L.FALLBACK
    assert field.calls == [] and typer.state == "done"


def test_command_mode_never_types():
    field = SimField()
    typer = L.LiveTyper(field, mode="command")
    assert typer.state == "off" and typer.reason == "command mode"
    assert typer.update("rewrite this") is False
    typer.window_changed()
    assert typer.cancel() is True
    assert typer.finish("Rewritten.") == L.FALLBACK
    assert field.calls == []


def test_window_change_stops_typing_and_the_final_falls_back():
    field = SimField()
    typer = L.LiveTyper(field)
    typer.update("hello")
    typer.window_changed()
    assert typer.state == "stopped" and typer.reason == "the target window changed"
    assert typer.update("hello world") is False          # nothing more is typed into the other window
    typer.window_changed()                               # a second report changes nothing
    assert typer.cancel() is False                       # and nothing is deleted there either
    assert typer.finish("Hello world.") == L.FALLBACK
    assert field.calls == [("type", "hello")] and typer.typed == "hello"


def test_rewritten_committed_text_stops_typing():
    field = SimField()
    typer = L.LiveTyper(field)
    typer.update("send it to jon")
    assert typer.update("send it to John") is False      # a committed word was taken back: never edit mid-way
    assert typer.state == "stopped" and typer.reason == "committed text was rewritten"
    assert typer.finish("Send it to John.") == L.FALLBACK
    assert field.calls == [("type", "send it to jon")]


def test_a_combining_mark_cannot_open_the_text():
    field = SimField()
    typer = L.LiveTyper(field)
    assert typer.update("́accent") is False         # it would attach to the character left of the cursor
    assert typer.state == "stopped" and typer.finish("Accent.") == L.FALLBACK
    assert field.calls == []


def test_too_much_to_delete_leaves_the_field_alone():
    field = SimField()
    typer = L.LiveTyper(field, max_backspaces=5)
    typer.update("this was heard wrongly")
    assert typer.finish("Something else entirely.") == L.LEFT
    assert typer.reason == "too much to delete" and typer.state == "done"
    assert field.text == PRE + "this was heard wrongly"
    assert typer.typed == "this was heard wrongly"


def test_the_cap_counts_the_whole_backspace_run():
    field = SimField()
    typer = L.LiveTyper(field, max_backspaces=3)
    typer.update("hello wrld")
    assert typer.finish("hello world") == L.TYPED        # 3 characters: exactly the cap
    assert field.text == PRE + "hello world"


def test_cancel_takes_back_what_was_typed():
    field = SimField()
    typer = L.LiveTyper(field)
    assert typer.cancel() is True                        # nothing typed yet
    typer.update("hello there")
    assert typer.cancel() is True
    assert field.text == PRE and typer.typed == ""
    assert typer.update("again") is True                 # typing may go on after a take-back (a new cycle)
    assert field.text == PRE + "again"


def test_cancel_gives_up_when_there_is_too_much():
    field = SimField()
    typer = L.LiveTyper(field, max_backspaces=4)
    typer.update("a long sentence")
    assert typer.cancel() is False
    assert typer.state == "stopped" and typer.reason == "too much to take back"
    assert field.text == PRE + "a long sentence"


@pytest.mark.parametrize("ok_calls", [0, 1])
def test_a_failing_keyboard_stops_everything(ok_calls):
    typer = L.LiveTyper(BrokenKeys(ok_calls))
    assert typer.update("hello") is bool(ok_calls)
    if ok_calls:
        assert typer.update("hello world") is False      # the second call raises
    assert typer.state == "failed" and typer.reason == "OSError"
    assert typer.update("hello world again") is False    # nothing more is attempted
    assert typer.cancel() is False
    assert typer.finish("Hello.") == L.LEFT


def test_a_failing_keyboard_during_finish_is_reported():
    keys = BrokenKeys(ok=1)
    typer = L.LiveTyper(keys)
    typer.update("hello wrld")                           # uses the one good call
    assert typer.finish("hello world") == L.LEFT
    assert typer.state == "done" and typer.reason == "OSError"


def test_cancel_with_a_failing_keyboard():
    keys = BrokenKeys(ok=1)
    typer = L.LiveTyper(keys)
    typer.update("hello")
    assert typer.cancel() is False and typer.state == "failed"


# ====================================================================== randomised property
_PIECES = ["a", "b", "xyz", "Zoë", "é", "x́̂", "ß", "\U0001F600", "😀", "\U0001F44D\U0001F3FD",
           "\U0001F468‍\U0001F469‍\U0001F467", "\U0001F1E9\U0001F1EA", "\U0001F1EB\U0001F1F7\U0001F1E9",
           "❤️", "\r\n", "\r", "\n", "́", "‍", "\ud83d", "\ude00", "\U000E0067", "\t", "-", ".", "9"]


def _word(rng: random.Random) -> str:
    return "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 4)))


def _mutate(rng: random.Random, text: str) -> str:
    """A final text that resembles ``text``: same, edited somewhere, extended, shortened or unrelated."""
    kind = rng.randrange(5)
    if kind == 0:
        return text
    if kind == 1 and text:
        i = rng.randrange(len(text))
        return text[:i] + _word(rng) + text[i + rng.randint(0, 3):]
    if kind == 2:
        return text + rng.choice(["", " ", "."]) + _word(rng)
    if kind == 3:
        return text[:rng.randint(0, len(text))]
    return " ".join(_word(rng) for _ in range(rng.randint(0, 4)))


def test_property_the_field_always_ends_equal_to_the_final_text():
    rng = random.Random(20260617)
    outcomes = {L.TYPED: 0, L.FALLBACK: 0, L.LEFT: 0}
    for _case in range(2000):
        field = SimField()
        typer = L.LiveTyper(field, max_backspaces=rng.choice([3, 10, 40, 1000]))
        words = [_word(rng) for _ in range(rng.randint(0, 6))]
        for n in range(1, len(words) + 1):
            for _ in range(rng.randint(1, 2)):                     # the same text may arrive twice
                typer.update(" ".join(words[:n]))
        if rng.random() < 0.1:
            typer.window_changed()
        before = field.text
        final = _mutate(rng, " ".join(words).strip())
        result = typer.finish(final)
        outcomes[result] += 1
        assert field.text.startswith(PRE), "touched text that was there before"
        assert field.deleted_clusters <= field.typed_clusters, "deleted more than it typed"
        if result == L.TYPED:
            assert field.text == PRE + final and typer.typed == final
        else:                                                      # fallback / left: finish left the field as it was
            assert field.text == before
            assert typer.typed == before[len(PRE):]
    assert all(outcomes.values()), outcomes                        # every kind of outcome was exercised


def test_property_plan_edit_is_minimal_and_exact():
    rng = random.Random(7)
    for _case in range(2000):
        typed = "".join(_word(rng) for _ in range(rng.randint(0, 3)))
        target = _mutate(rng, typed)
        backspaces, text = L.plan_edit(typed, target)
        have = L.clusters(typed)
        assert 0 <= backspaces <= len(have)
        assert "".join(have[:len(have) - backspaces]) + text == target
        if backspaces and text:                                    # minimal: the next kept cluster would differ
            assert have[len(have) - backspaces] != L.clusters(target)[len(have) - backspaces]
