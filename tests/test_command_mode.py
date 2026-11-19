"""Command mode: the pure helpers in voxflow.command, CommandMode with fakes, and the whole flow through the
Controller (hold the command shortcut, speak an instruction, the selection is rewritten in place)."""
from __future__ import annotations

import pytest

from tests.controller_rig import TCleaner, TInjector, make_rig
from voxflow import command as cmd
from voxflow.command import CommandError, CommandMode, build_messages, clean_instruction, tidy_output
from voxflow.context import AppContext


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


# =========================================================================== pure helpers
@pytest.mark.parametrize("spoken,expected", [
    ("um make this more formal", "Make this more formal"),
    ("translate to spanish", "Translate to spanish"),
    ("uh", "uh"),                                              # nothing but filler: keep what was said
    ("", ""), (None, ""), ("   ", ""),
    ("turn this into bullet points", "Turn this into bullet points"),      # not executed as a formatting command
])
def test_clean_instruction(spoken, expected):
    assert clean_instruction(spoken) == expected


def test_build_messages_with_and_without_a_selection():
    system, user = build_messages("Make it formal", "hey whats up")
    assert system["role"] == "system" and user["role"] == "user"
    assert user["content"] == "INSTRUCTION: Make it formal\n\nTEXT:\nhey whats up"
    assert "Output ONLY the resulting text" in system["content"]
    assert "Do not add facts" in system["content"]
    _system, user = build_messages("Write a haiku about rain", "   ")
    assert "TEXT:\n" not in user["content"] and "There is no TEXT" in user["content"]


def test_build_messages_tone_terms_and_language():
    base = build_messages("x", "y")[0]["content"]
    assert build_messages("x", "y", tone="formal")[0]["content"] != base
    assert "professional tone" in build_messages("x", "y", tone="formal")[0]["content"]
    assert "text friends" in build_messages("x", "y", tone="very_casual")[0]["content"]
    assert build_messages("x", "y", tone="shouty")[0]["content"] == base
    terms = build_messages("x", "y", terms=["GitHub", "  Py  Torch ", ""])[0]["content"]
    assert "GitHub, Py Torch." in terms
    many = build_messages("x", "y", terms=[f"T{i}" for i in range(200)])[0]["content"]
    assert "T39" in many and "T40" not in many, "the prompt must stay bounded"
    assert "ISO code 'de'" in build_messages("x", "", language="de")[0]["content"]
    assert "ISO code" not in build_messages("x", "selected text", language="de")[0]["content"], \
        "with a selection the text's own language is kept"
    for lang in (None, "auto", "en"):
        assert "ISO code" not in build_messages("x", "", language=lang)[0]["content"]


@pytest.mark.parametrize("raw,selection,expected", [
    ("Dear Bob,\nthank you.", "", "Dear Bob,\nthank you."),
    ("  padded  ", "", "padded"),
    ("Sure, here is the formal version:\n\nDear Bob, thank you.", "", "Dear Bob, thank you."),
    ("Here's the rewritten text:\nDear Bob", "", "Dear Bob"),
    ("```\nprint('hi')\n```", "", "print('hi')"),
    ("```python\nprint('hi')\n```", "", "print('hi')"),
    ("TEXT:\nDear Bob", "", "Dear Bob"),
    ("Rewritten text: Dear Bob", "", "Dear Bob"),
    ('"Dear Bob, thank you."', "hey bob thx", "Dear Bob, thank you."),
    ("“Dear Bob”", "hey bob", "Dear Bob"),
    ('"Dear Bob," she said, "thank you."', "x", '"Dear Bob," she said, "thank you."'),     # inner quotes: real content
    ('"quoted input"', '"quoted input"', '"quoted input"'),                                  # the selection was quoted
    ("line one  \nline two", "", "line one\nline two"),
    ("Sure thing is what he said.", "", "Sure thing is what he said."),                      # not a preamble
    ("", "", ""), (None, "", ""),
])
def test_tidy_output(raw, selection, expected):
    assert tidy_output(raw, selection) == expected


# =========================================================================== CommandMode with fakes
@pytest.fixture
def mode():
    cleaner, injector = TCleaner(), TInjector()
    cleaner.ready = True
    return CommandMode(cleaner, injector, timeout_s=7.0)


def test_available_follows_the_language_model(mode):
    assert mode.available
    mode.cleaner.has_llm = False
    assert not mode.available
    assert not CommandMode(object(), TInjector()).available


def test_grab_selection(mode):
    mode.injector.selection = "selected text"
    assert mode.grab_selection() == "selected text"
    mode.injector.selection = ""
    assert mode.grab_selection() == ""
    for broken in (None, OSError("clipboard locked")):
        mode.injector.selection = broken
        with pytest.raises(CommandError, match="clipboard is busy. Nothing was changed."):
            mode.grab_selection()
    mode.injector.modifiers = True                              # the real reason is named (QA C-07)
    with pytest.raises(CommandError, match="Win, Alt or Shift is still held down. Nothing was changed."):
        mode.grab_selection()
    mode.injector = type("NoKeyboardState", (), {"copy_selection": lambda self: None})()
    with pytest.raises(CommandError, match="clipboard is busy"):
        mode.grab_selection()


def test_run_rewrites_a_selection(mode):
    mode.cleaner.reply = lambda messages: "Dear Bob, thank you for your help."
    out = mode.run("um make this formal", "hey bob thx for the help", tone="formal", terms=["Bob"], language="en")
    assert out == "Dear Bob, thank you for your help."
    call = mode.cleaner.generate_calls[0]
    assert call["timeout_s"] == 7.0 and 256 <= call["max_tokens"] <= 4096
    assert call["messages"][-1]["content"] == "INSTRUCTION: Make this formal\n\nTEXT:\nhey bob thx for the help"
    assert "professional tone" in call["messages"][0]["content"] and "Bob." in call["messages"][0]["content"]


def test_run_generates_when_nothing_is_selected(mode):
    mode.cleaner.reply = lambda messages: "Rain taps the window"
    assert mode.run("write a haiku about rain") == "Rain taps the window"
    assert mode.cleaner.generate_calls[0]["max_tokens"] == 700


@pytest.mark.parametrize("setup,match", [
    (lambda m: None, "Didn't catch the instruction"),                                   # instruction "" (see call)
    (lambda m: setattr(m.cleaner, "has_llm", False), "needs the language model"),
    (lambda m: setattr(m.cleaner, "fail_generate", TimeoutError()), "took too long"),
    (lambda m: setattr(m.cleaner, "fail_generate", RuntimeError("llama_decode returned -1")), "model failed"),
    (lambda m: setattr(m.cleaner, "reply", lambda msgs: "   "), "returned nothing"),
    (lambda m: setattr(m.cleaner, "reply", lambda msgs: "I'm sorry, but I can't help with that."), "declined"),
    (lambda m: setattr(m.cleaner, "reply", lambda msgs: "As an AI language model I cannot"), "declined"),
    (lambda m: setattr(m.cleaner, "reply", lambda msgs: "blah " * 2000), "far longer than the selection"),
])
def test_every_failure_is_a_command_error_that_leaves_the_selection_alone(mode, setup, match):
    setup(mode)
    instruction = "" if match.startswith("Didn't") else "make it formal"
    with pytest.raises(CommandError, match=match) as exc:
        mode.run(instruction, "hey bob")
    message = str(exc.value)
    assert message.endswith(".") and "llama_decode" not in message
    if not match.startswith("Didn't"):
        assert "Nothing was changed" in message or "nothing was changed" in message


def test_refusal_guard_does_not_block_text_that_itself_starts_like_a_refusal(mode):
    mode.cleaner.reply = lambda messages: "I cannot attend the meeting on Friday."
    assert mode.run("fix the grammar", "i cannot attend meeting friday") == "I cannot attend the meeting on Friday."


def test_long_generation_without_a_selection_is_allowed(mode):
    mode.cleaner.reply = lambda messages: "word " * 400
    assert len(mode.run("write a long story")) > 1500


def test_token_budget_scales_with_the_selection_and_rejects_oversized_ones(mode):
    mode.cleaner.reply = lambda messages: "ok"
    mode.run("shorten", "x" * 3000)                              # ~1000 tokens
    assert mode.cleaner.generate_calls[-1]["max_tokens"] == int(1000 * 1.6) + 128
    with pytest.raises(CommandError, match=r"too long for command mode \(about [\d,]+ words at most\)"):
        mode.run("shorten", "x" * 12000)                         # ~4000 tokens: no room left for the answer
    assert len(mode.cleaner.generate_calls) == 1, "the model must not be called with a prompt that cannot fit"
    mode.cleaner.n_ctx = 16384
    mode.run("shorten", "x" * 12000)
    assert len(mode.cleaner.generate_calls) == 2


def test_budget_never_exceeds_the_room_left_in_the_context(mode):
    mode.cleaner.reply = lambda messages: "ok"
    mode.run("shorten", "x" * 5100)                              # 1700 tokens of selection in a 4096 context
    room = 4096 - cmd._PROMPT_OVERHEAD - 2 - 1700 - cmd._OUTPUT_RESERVE
    assert mode.cleaner.generate_calls[-1]["max_tokens"] == room == 2038
    with pytest.raises(CommandError, match="too long"):
        mode.run("shorten", "x" * 5400)                          # 1800 tokens: the answer could no longer fit


def test_c59_non_string_terms_are_not_put_into_the_prompt():
    assert "None" not in build_messages("x", "y", terms=["GitHub", None])[0]["content"]


# =========================================================================== through the Controller
@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    yield r
    r.close()


def test_command_rewrites_the_selection_in_place(rig):
    rig.context = AppContext("slack.exe", "#general | Acme", 0x500)
    rig.injector.selection = "hey bob can u send me the report"
    rig.cleaner.reply = lambda messages: "Hi Bob, could you please send me the report?"
    rig.c.dictionary.add("Bob")
    events = rig.dictate("um make this more polite", binding="command")

    assert rig.injector.selection_grabs == 1
    assert rig.injector.pasted == ["Hi Bob, could you please send me the report?"]
    states = [args for name, args in events if name == "state_changed"]
    assert [(s[0], s[1]) for s in states] == [("listening", "command"), ("processing", "command"), ("done", "command")]
    assert states[-1][2] == "Rewritten"
    rig.wait(lambda: rig.sounds == ["command", "stop"])
    prompt = rig.cleaner.generate_calls[0]["messages"]
    assert prompt[-1]["content"].startswith("INSTRUCTION: Make this more polite\n\nTEXT:\nhey bob can u")
    assert "relaxed, friendly tone" in prompt[0]["content"], "tone comes from the app (Slack -> casual)"
    assert "Bob." in prompt[0]["content"]

    entry = rig.last_entry()
    assert entry["mode"] == "command" and entry["text"] == "Hi Bob, could you please send me the report?"
    assert entry["raw"] == "um make this more polite", "the History's 'heard' side is the instruction as spoken"
    assert entry["app_exe"] == "slack.exe" and entry["wpm"] == 0.0
    assert rig.c.stats.summary()["sessions"] == 0, "command results are not counted as dictated words"
    assert rig.c.last_transcript == entry and len(rig.spy.of("transcript_ready")) == 1
    rig.wait(lambda: rig.c.state == "idle")
    assert rig.c.mode == "command"


def test_command_without_a_selection_writes_new_text(rig):
    rig.cleaner.reply = lambda messages: "Rain taps the window\nSoft drums on a sleeping roof\nMorning smells of earth"
    rig.dictate("write a haiku about rain", binding="command")
    assert (rig.c.state, rig.c.message) == ("done", "Written")
    assert rig.injector.pasted[0].count("\n") == 2
    assert "There is no TEXT" in rig.cleaner.generate_calls[0]["messages"][-1]["content"]


def test_dictionary_spellings_are_applied_to_the_result(rig):
    rig.c.dictionary.add("GitHub", ["get hub"])
    rig.injector.selection = "push to get hub"
    rig.cleaner.reply = lambda messages: "Please push to get hub."
    rig.dictate("make it polite", binding="command")
    assert rig.injector.pasted == ["Please push to GitHub."]


def test_dictation_after_command_mode_is_a_normal_dictation_again(rig):
    rig.injector.selection = "text"
    rig.dictate("uppercase it", binding="command")
    rig.wait(lambda: rig.c.state == "idle")
    rig.dictate("plain dictation again")
    assert rig.c.mode == "dictate" and rig.injector.pasted[-1] == "Plain dictation again."
    assert rig.last_entry()["mode"] == "dictate"


@pytest.mark.parametrize("arrange,bar,notice_fragment", [
    (lambda r: setattr(r.injector, "selection", None), "Couldn't read the selection because the clipboard is busy",
     "Nothing was changed"),
    (lambda r: setattr(r.asr, "text", ""), "Didn't catch the instruction", "Hold the command shortcut"),
    (lambda r: setattr(r.cleaner, "fail_generate", TimeoutError()), "Command mode took too long and was stopped",
     "Nothing was changed"),
    (lambda r: setattr(r.cleaner, "reply", lambda m: "I'm sorry, but I cannot do that."),
     "The language model declined that request", "Nothing was changed"),
])
def test_command_failures_leave_the_selection_untouched_and_explain(rig, arrange, bar, notice_fragment):
    rig.injector.selection = "precious text"
    rig.asr.text = "make it formal"
    arrange(rig)
    rig.dictate(binding="command")
    assert rig.c.state == "error" and rig.injector.pasted == [] and rig.c.history.count() == 0
    assert rig.c.message == (bar if len(bar) <= 48 else "Something went wrong")
    errors = [m for level, m in rig.spy.notices() if level == "error"]
    assert errors and notice_fragment in errors[0] and bar in errors[0]
    rig.wait(lambda: rig.sounds[-1:] == ["error"])


def test_command_mode_needs_the_language_model(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.cleaner.backend, rig.cleaner.has_llm = "rules", False
        rig.start()
        rig.press("command")
        assert (rig.c.state, rig.c.mode, rig.c.message) == ("error", "command", "Command mode needs the language model")
        assert rig.rec.starts == 0 and rig.injector.selection_grabs == 0
        rig.release("command")
        rig.wait(lambda: rig.c.state == "idle")
        rig.dictate("dictation still works")
        assert rig.injector.pasted == ["Dictation still works."]
    finally:
        rig.close()


def test_command_mode_while_the_language_model_is_still_loading(tmp_path, qtbot, monkeypatch):
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    try:
        rig.cleaner.has_llm = False
        rig.start()
        rig.c._models["cleanup"]["state"] = "loading"
        rig.press("command")
        assert rig.c.message == "Language model still loading…"
        rig.release("command")
    finally:
        rig.close()


def test_result_is_not_pasted_into_a_different_window(rig):
    rig.injector.selection = "original"
    rig.cleaner.reply = lambda messages: "REWRITTEN TEXT HERE"
    rig.press("command")
    rig.context = AppContext("chrome.exe", "Somewhere else", 0x999)       # the user alt-tabbed while the model worked
    rig.release("command")
    rig.wait(lambda: rig.c.state == "error")
    assert rig.injector.pasted == [], "pasting over the wrong window could destroy unrelated text"
    assert rig.injector.copied == ["REWRITTEN TEXT HERE"]
    assert ("error", "Window changed. Result copied: press Ctrl+V") in rig.spy.notices()
    rig.spy.clear()
    rig.wait(lambda: rig.c.state == "idle")
    rig.injector.copy_ok = False
    rig.context = AppContext("notepad.exe", "Untitled - Notepad", 0x100)
    rig.pause()
    rig.press("command")
    rig.context = AppContext("chrome.exe", "Somewhere else", 0x999)
    rig.release("command")
    rig.wait(lambda: rig.c.state == "error")
    assert rig.c.message == "Window changed, so nothing was pasted"


def test_command_shortcut_can_be_switched_off_and_changed(rig):
    rig.c.set("command_hotkey", "")
    rig.press("command")
    assert rig.c.state == "idle" and rig.rec.starts == 0
    rig.release("command")
    rig.c.set("command_hotkey", "ctrl+alt+c")
    rig.injector.selection = "text"
    rig.pause()
    rig.keys((0xA2, True), (0xA4, True), (0x43, True))
    assert (rig.c.state, rig.c.mode) == ("listening", "command")
    rig.now += 1.0
    rig.c._t_start -= 1.0
    rig.keys((0x43, False), (0xA4, False), (0xA2, False))
    rig.wait(lambda: rig.c.state == "done")
    assert rig.injector.pasted == ["REWRITTEN"]
    assert rig.c.listener.binding("command").max_hold_s == rig.c.config.max_record_s + 5


def test_command_timeout_setting_reaches_the_model_call(rig):
    rig.c.set("command_timeout_s", 9.5)
    rig.injector.selection = "text"
    rig.dictate("uppercase", binding="command")
    assert rig.cleaner.generate_calls[0]["timeout_s"] == 9.5


def test_whisper_mode_also_boosts_the_spoken_instruction(rig):
    import numpy as np
    from tests.controller_rig import TRecorder, speech
    quiet = speech(2.0, amp=0.004)
    TRecorder.audio = quiet
    rig.c.set("whisper_mode", True)
    rig.injector.selection = "text"
    rig.dictate("uppercase", binding="command")
    assert float(np.abs(rig.asr.heard[-1]).max()) > float(np.abs(quiet).max()) * 5


def test_fake_cleaner_supports_command_mode_for_the_demo():
    from voxflow import fakes
    cleaner = fakes.FakeCleaner(delay_s=0)
    cleaner.load()
    mode = CommandMode(cleaner, fakes.FakeInjector(selection="hello"))
    assert mode.available and mode.grab_selection() == "hello"
    assert mode.run("shout it", "hello") == "HELLO"
    assert "demo text" in mode.run("write something")
