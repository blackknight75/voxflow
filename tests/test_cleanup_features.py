"""Cleanup features: smart formatting, tone, language handling, dictionary terms,
free-form generation (command mode) and the bounded LLM call. A scripted fake model stands in for llama.cpp."""
from __future__ import annotations

import threading
import time

import pytest

from voxflow import cleanup as C
from voxflow.cleanup import Cleaner, apply_tone, clean_foreign, is_english, rule_clean, smart_format


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


# =========================================================================== smart formatting
@pytest.mark.parametrize("spoken,written", [
    # numbers
    ("i have twenty three apples", "I have 23 apples"),
    ("three hundred fifty", "350"),
    ("one hundred and twenty", "120"),
    ("ten thousand five hundred", "10,500"),
    ("one million two hundred thousand", "1,200,000"),
    ("a hundred people", "100 people"),
    ("ten out of ten", "10 out of 10"),
    ("twenty and thirty", "20 and 30"),
    ("version two point five", "Version 2.5"),
    ("three point one four", "3.14"),
    ("zero point five", "0.5"),
    ("call me at five five five one two three four", "Call me at 5551234"),
    ("five six seven eight", "5678"),
    # small numbers stay words in prose
    ("i ate one apple and two pears", "I ate one apple and two pears"),
    ("no one knows", "No one knows"),
    ("one of them", "One of them"),
    ("the two of us", "The two of us"),
    ("one two", "One two"),
    ("hundred hundred", "Hundred hundred"),
    # units
    ("it costs five dollars", "It costs $5"),
    ("ten bucks", "$10"),
    ("five euros", "€5"),
    ("fifty cents", "50 cents"),
    ("fifty percent of users", "50% of users"),
    ("a twenty percent raise", "A 20% raise"),
    ("ninety nine point nine percent", "99.9%"),
    # years, dates, times
    ("nineteen eighty four", "1984"),
    ("twenty twenty", "2020"),
    ("two thousand and five", "2005"),
    ("on january fifth twenty twenty six", "On January 5th 2026"),
    ("march fifth", "March 5th"),
    ("the first of march", "The 1st of March"),
    ("the twenty first of june", "The 21st of June"),
    ("march two people", "March two people"),
    ("may fifth", "May fifth"),                                   # "may" is too ambiguous to be treated as a month
    ("meet me at three thirty pm", "Meet me at 3:30 PM"),
    ("five oh five pm", "5:05 PM"),
    ("twelve fifteen a m", "12:15 AM"),
    ("at nine am", "At 9 AM"),
    ("i am here", "I am here"),
    # e-mail, URLs, files, code
    ("send it to john dot smith at gmail dot com", "Send it to john.smith@gmail.com"),
    ("my email is john underscore doe at example dot com", "My email is john_doe@example.com"),
    ("email sam at example dot com", "Email sam@example.com"),
    ("it is hosted at example dot com", "It is hosted at example.com"),
    ("he is at home", "He is at home"),
    ("go to www dot example dot com slash docs", "Go to www.example.com/docs"),
    ("example dot org slash a slash b", "example.org/a/b"),
    ("www dot google dot co dot uk", "www.google.co.uk"),
    ("open the file readme dot md", "Open the file readme.md"),
    ("src slash main dot py", "src/main.py"),
    ("user underscore id", "user_id"),
    ("dash dash verbose", "--verbose"),
    ("dot gitignore", ".gitignore"),
    ("connect the dots", "Connect the dots"),
    # symbols and punctuation words
    ("open paren hello close paren", "(hello)"),
    ("he said open quote hello close quote", "He said “hello”"),
    ("look at me dot dot dot", "Look at me…"),
    ("hashtag winning", "#winning"),
    ("at sign", "@"),
    ("hello comma world", "Hello, world"),
    ("wait colon here", "Wait: here"),
    ("end of story period", "End of story."),
    ("semicolon", ";"),
    # ... but not when they are ordinary words
    ("the period of time", "The period of time"),
    ("an oxford comma", "An oxford comma"),
    ("comma separated values", "Comma separated values"),
    ("the colon key", "The colon key"),
    # lists
    ("number one milk number two eggs number three bread", "1. Milk\n2. Eggs\n3. Bread"),
    ("the number one priority is speed number two is cost", "The number one priority is speed number two is cost"),
    ("- twenty items", "- 20 items"),
    ("line one\ntwenty two apples", "Line one\n22 apples"),
])
def test_smart_format(spoken, written):
    assert smart_format(spoken) == written


@pytest.mark.parametrize("text", ["", "   ", "\n", "Already fine.", "src/main.py", "https://example.com/a?b=1",
                                  "bob@example.com", "C:\\Users\\dan\\file.txt"])
def test_smart_format_leaves_empty_and_already_written_text_alone(text):
    assert smart_format(text) == text


def test_smart_format_is_idempotent_on_its_own_output():
    for spoken in ("it costs twenty five dollars", "meet me at three thirty pm on january fifth",
                   "email sam at example dot com", "number one milk number two eggs", "fifty percent of users"):
        once = smart_format(spoken)
        assert smart_format(once) == once, spoken


def test_smart_format_never_loses_the_dictation_when_a_rule_crashes(monkeypatch, caplog):
    monkeypatch.setattr(C, "_smart_line", lambda line: 1 / 0)
    with caplog.at_level("WARNING"):
        assert smart_format("twenty three apples") == "twenty three apples"
    assert "smart formatting failed" in caplog.text


@pytest.mark.parametrize("spoken,written", [
    ("the twenty first century", "The twenty first century"),          # currently "The 20 first century"
    ("the dot net runtime", "The .net runtime"),                       # currently "the.net runtime"
    ("at the dot com bubble", "At the dot com bubble"),                # currently "At the.com bubble"
    ("he said quote hello unquote", "He said “hello”"),                # currently only the closing quote
])
def test_c55_smart_format_edge_cases(spoken, written):
    assert smart_format(spoken) == written


def test_c55_terms_hint_skips_non_strings():
    assert "None" not in C._terms_hint(["GitHub", None])


# =========================================================================== tone
BASE = "Hello there. I will send the report to Bob tomorrow."


def test_tone_formal_none_and_unknown_change_nothing():
    for tone in ("formal", None, "", "shouty"):
        assert apply_tone(BASE, tone) == BASE
    assert C.TONES == ("formal", "casual", "very_casual")


def test_tone_casual_drops_only_the_final_full_stop():
    assert apply_tone(BASE, "casual") == "Hello there. I will send the report to Bob tomorrow"
    assert apply_tone("Are you coming?", "casual") == "Are you coming?"
    assert apply_tone("Great!", "casual") == "Great!"
    assert apply_tone("Wait...", "casual") == "Wait..."
    assert apply_tone("Apples, pears, etc.", "casual") == "Apples, pears, etc."
    assert apply_tone("First line.\nSecond line.", "casual") == "First line.\nSecond line"
    assert apply_tone("Done.\n", "casual") == "Done\n", "trailing line breaks (a spoken 'new line') are kept"
    assert apply_tone("- Milk\n- Eggs", "casual") == "- Milk\n- Eggs"


def test_tone_very_casual_lowercases_sentence_starts_but_keeps_names_i_and_acronyms():
    assert apply_tone(BASE, "very_casual") == "hello there. I will send the report to Bob tomorrow"
    assert apply_tone("Thanks! See you Monday.", "very_casual") == "thanks! see you Monday"
    assert apply_tone("I'll call NASA. I'm late.", "very_casual") == "I'll call NASA. I'm late"
    assert apply_tone("Python is fun. Paris too.", "very_casual") == "Python is fun. Paris too"
    assert apply_tone("First line.\nSecond line.", "very_casual") == "first line.\nsecond line"
    assert apply_tone("USA wins. OK.", "very_casual") == "USA wins. OK"


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_tone_on_empty_text(text):
    for tone in C.TONES:
        assert apply_tone(text, tone) == text


def test_tone_never_changes_words():
    import re
    text = "Hey Bob, I can't make it at 5. Let's move the Q3 review to Friday. Thanks!"
    for tone in C.TONES:
        assert [w.lower() for w in re.findall(r"[\w']+", apply_tone(text, tone))] == \
               [w.lower() for w in re.findall(r"[\w']+", text)]


@pytest.mark.parametrize("text,tone,expected", [
    ("See you at 5 p.m.", "casual", "See you at 5 p.m."),              # the abbreviation's own period is not a full stop
    ("- Milk\n- Eggs", "very_casual", "- milk\n- eggs"),                # currently "- Milk\n- eggs"
    ("Dr. Smith said no.", "very_casual", "Dr. Smith said no"),         # currently "dr. smith said no"
])
def test_c56_tone_edge_cases(text, tone, expected):
    assert apply_tone(text, tone) == expected


# =========================================================================== language
@pytest.mark.parametrize("text,language,expected", [
    ("the quick brown fox jumps over the lazy dog", None, True),
    ("the quick brown fox", "auto", True),
    ("hola como estas el dia es muy bueno para la playa", None, False),
    ("je ne sais pas ce que vous voulez", "auto", False),
    ("ich bin der mann und nicht die frau", None, False),
    ("Привет как дела сегодня", None, False),                     # non-Latin script
    ("Καλημέρα σας", None, False),
    ("ok", None, True), ("bonjour", None, True),                  # too short to tell: English rules are harmless
    ("123 456 789", None, True), ("", None, True), ("!!!", None, True),
    ("hello there my friend", "de", False),                       # the user's explicit choice wins
    ("bonjour tout le monde", "en", True),
    ("the meeting is in el paso at la guardia", None, True),      # English with a few foreign words
])
def test_is_english(text, language, expected):
    assert is_english(text, language) is expected


@pytest.mark.parametrize("text,expected", [
    ("  hola   como estas  ", "Hola como estas."),
    ("guten Morgen , wie geht es dir ?", "Guten Morgen, wie geht es dir?"),
    ("Ça va bien.", "Ça va bien."),
    ("bonjour", "Bonjour"),
    ("привет как дела", "Привет как дела."),
    ("", ""), ("   ", ""),
    ("um die ecke", "Um die ecke."),                              # "um" is a German word, not an English filler
])
def test_clean_foreign_only_tidies(text, expected):
    assert clean_foreign(text) == expected


# =========================================================================== Cleaner: rules-only paths
@pytest.fixture
def rules():
    c = Cleaner(use_llm=False)
    c.load()
    return c


SPOKEN = "um so i have twenty three issues on get hub new line send it to bob at example dot com"


def test_clean_keyword_options(rules):
    assert rules.backend == "rules" and rules.ready and not rules.has_llm
    plain = rules.clean(SPOKEN)
    assert plain == "So I have twenty three issues on get hub\nSend it to bob at example dot com."
    assert rules.clean(SPOKEN, smart=True) == "So I have 23 issues on get hub\nSend it to bob@example.com."
    assert rules.clean(SPOKEN, tone="casual") == plain[:-1]
    assert rules.clean(SPOKEN, smart=True, tone="very_casual") == \
        "so I have 23 issues on get hub\nsend it to bob@example.com"
    assert rules.clean(SPOKEN, terms=["GitHub"]) == plain, "without a model, terms are applied by the dictionary store"
    assert rules.clean(SPOKEN, tone="formal", terms=None, language="auto", smart=False) == plain


def test_clean_one_argument_form_is_unchanged(rules):
    assert rules.clean("what is the capital of france") == "What is the capital of France?" or \
        rules.clean("what is the capital of france") == "What is the capital of france?"
    assert rules.clean("") == "" and rules.clean("   ") == "   "
    assert rules.clean("um") == ""


def test_non_english_dictation_skips_english_rules(rules):
    german = "ich bin um acht uhr da und nicht früher"
    assert rules.clean(german) == "Ich bin um acht uhr da und nicht früher."        # "um" survives
    assert rules.clean("um so this is english", language="de") == "Um so this is english."
    assert rules.clean("hola como estas el dia es muy bueno", smart=True, tone="casual") == \
        "Hola como estas el dia es muy bueno"
    assert rules.clean("bonjour tout le monde new line merci", language="en") == "Bonjour tout le monde\nMerci"


def test_filler_only_dictation_stays_empty_with_every_option(rules):
    for kw in ({}, {"smart": True}, {"tone": "very_casual"}, {"smart": True, "tone": "casual", "terms": ["X"]}):
        assert rules.clean("um uh", **kw) == ""


def test_clean_returns_the_input_if_anything_unexpected_fails(rules, monkeypatch, caplog):
    monkeypatch.setattr(C, "rule_clean", lambda *a, **k: 1 / 0)
    with caplog.at_level("WARNING"):
        assert rules.clean("hello world") == "hello world"
    assert "cleanup failed" in caplog.text
    assert rules._terms is None, "per-call terms must not leak into the next call"


# =========================================================================== Cleaner with a scripted model
class FakeLLM:
    """Stands in for llama_cpp.Llama: replies come from ``script`` (callable or list), streamed word by word."""

    def __init__(self, script=None, delay=0.0):
        self.script = script
        self.delay = delay
        self.calls: list[dict] = []
        self.closed = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def tokenize(self, data: bytes, add_bos=True):
        return list(range(max(1, len(data.split()))))

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kw):
        self.calls.append(dict(messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream, **kw))
        user = messages[-1]["content"]
        reply = self.script(user, messages) if callable(self.script) else self.script
        if isinstance(reply, Exception):
            raise reply
        outer = self

        class Stream:
            def __iter__(self):
                outer.started.set()
                for word in reply.split(" "):
                    outer.release.wait(10)
                    if outer.delay:
                        time.sleep(outer.delay)
                    yield {"choices": [{"delta": {"content": word + " "}}]}
                yield {"choices": [{"delta": {}}]}

            def close(self):
                outer.closed += 1

        return Stream()


def with_llm(script=None, **kw) -> tuple[Cleaner, FakeLLM]:
    c = Cleaner(**kw)
    llm = FakeLLM(script)
    c._llm, c._ready, c.backend = llm, True, "cuda"
    return c, llm


def test_llm_result_is_used_when_it_is_a_faithful_rewrite():
    c, llm = with_llm("I pushed the branch to GitHub last night.")
    assert c.clean("i pushed the branch to get hub last night") == "I pushed the branch to GitHub last night."
    assert c.last_info["llm"] is True and c.last_info["rejected"] is False and c.last_info["ms"] >= 0
    call = llm.calls[0]
    assert call["temperature"] == 0.0 and call["top_k"] == 1 and call["stream"] is True
    assert call["messages"][0]["role"] == "system" and "never answer" in call["messages"][0]["content"].lower()
    assert call["messages"][-1] == {"role": "user", "content": "I pushed the branch to get hub last night."}
    assert llm.closed == 1, "the completion stream must be closed"
    assert c.has_llm


def test_llm_answering_instead_of_cleaning_is_rejected():
    c, _llm = with_llm("The capital of France is Paris, a beautiful city on the Seine.")
    out = c.clean("so um what is the capital of france")
    assert out == "So what is the capital of France?", "only the capitalisation may be salvaged from a rejected reply"
    assert "Paris" not in out and c.last_info["rejected"] is True


def test_dictionary_terms_reach_the_system_prompt_for_that_call_only():
    c, llm = with_llm(lambda user, msgs: user)
    c.clean("i pushed the the branch to get hub", terms=["GitHub", "PyTorch"])
    system = llm.calls[-1]["messages"][0]["content"]
    assert "Preferred spellings" in system and "GitHub, PyTorch." in system
    assert "never insert one that was not spoken" in system
    c.clean("i pushed the the branch to get hub")
    assert "Preferred spellings" not in llm.calls[-1]["messages"][0]["content"]
    assert c._terms is None


def test_terms_hint_is_bounded():
    assert C._terms_hint(None) == "" and C._terms_hint([]) == "" and C._terms_hint(["", "  "]) == ""
    many = C._terms_hint([f"Term{i}" for i in range(500)])
    assert many.count("Term") == C._MAX_HINT_TERMS and len(many) < C._MAX_HINT_CHARS + 300
    assert "A B" in C._terms_hint(["  A   B  "])
    assert "x" * 50 not in C._terms_hint(["x" * 5000, "Short"]) and "Short" in C._terms_hint(["x" * 5000, "Short"])


def test_smart_and_tone_are_applied_after_the_model():
    c, _llm = with_llm("So I have twenty three issues on GitHub.")
    out = c.clean("so i have twenty three issues on get hub", smart=True, tone="casual")
    assert out == "So I have 23 issues on GitHub"


def test_model_is_not_consulted_for_text_that_needs_nothing():
    c, llm = with_llm("SHOULD NOT BE USED")
    assert c.clean("Hello world.") == "Hello world."
    assert c.clean("ok thanks") == "Ok, thanks."          # the comma is the rule pass's (grammar_rules)
    assert llm.calls == [] and c.last_info["llm"] is False


def test_model_error_falls_back_to_the_rule_result():
    c, _llm = with_llm(RuntimeError("llama decode failed"))
    assert c.clean("so we should uh move the the meeting") == "so we should uh move the the meeting"
    # the documented contract: an unexpected failure returns the input verbatim; a timeout returns the rule result
    c2, _ = with_llm("x")
    c2._llm_clean = lambda body, max_ms=None: (_ for _ in ()).throw(TimeoutError())
    assert c2.clean("so we should uh move the the meeting") == "So we should move the the meeting."
    assert c2.last_info["timeout"] is True


def test_each_line_is_cleaned_separately_and_bullets_survive():
    c, llm = with_llm(lambda user, msgs: user.replace("get hub", "GitHub"))
    out = c.clean("shopping list bullet point push to to get hub bullet point eggs new line then we we deploy")
    assert out.split("\n")[1] == "- Push to to GitHub"
    assert [call["messages"][-1]["content"] for call in llm.calls] == ["Push to to get hub", "Then we we deploy."]


def test_a_model_that_invents_a_bullet_or_quotes_is_normalised_or_rejected():
    c, _ = with_llm('"I pushed the branch to GitHub."')
    assert c.clean("i pushed the the branch to get hub") == "I pushed the branch to GitHub."
    c, _ = with_llm("- I pushed the branch")
    assert c.clean("i pushed the the branch to get hub") == "I pushed the the branch to get hub."
    assert c.last_info["rejected"] is True


# ---- bounded generation ------------------------------------------------------------------------------
def test_timeout_abandons_the_generation_then_recovers():
    c, llm = with_llm("I pushed the branch to GitHub.", timeout_s=0.15)
    llm.release.clear()                                      # the model hangs mid-generation
    t0 = time.perf_counter()
    assert c.clean("i pushed the the branch to get hub") == "I pushed the the branch to get hub."
    assert time.perf_counter() - t0 < 2.0, "the timeout is a hard bound for the caller"
    assert c.last_info["timeout"] is True and c._abandoned.is_set()
    # while the abandoned generation is still unwinding, a new call waits at most its own budget, then falls back
    t0 = time.perf_counter()
    assert c.clean("we we should deploy it") == "We we should deploy it."
    assert time.perf_counter() - t0 < 0.15 + 0.2
    assert c.last_info["skipped_busy"] is True and "timeout" not in c.last_info
    llm.release.set()                                        # the old generation finishes
    deadline = time.time() + 3
    while c._abandoned.is_set() and time.time() < deadline:
        time.sleep(0.01)
    assert not c._abandoned.is_set(), "cleanup stays rules-only forever (bug C-33)"
    c.timeout_s = 5.0
    assert c.clean("i pushed the the branch to get hub") == "I pushed the branch to GitHub."


def test_c33_finish_and_timeout_racing_never_leaves_the_abandoned_flag_set():
    """Hammer the hand-off: generations that finish right at the deadline."""
    c, llm = with_llm("x")
    for i in range(150):
        budget = 0.0005 if i % 2 else 0.002
        try:
            c._run_bounded(lambda abort: time.sleep(0.001) or "ok", budget, "race")
        except TimeoutError:
            pass
        deadline = time.time() + 1
        while c._abandoned.is_set() and time.time() < deadline:
            time.sleep(0.001)
        assert not c._abandoned.is_set(), f"flag stuck after iteration {i}"


def test_run_bounded_propagates_errors_and_results():
    c, _ = with_llm("x")
    assert c._run_bounded(lambda abort: "done", 1.0, "job") == "done"
    with pytest.raises(KeyError):
        c._run_bounded(lambda abort: {}["missing"], 1.0, "job")
    with pytest.raises(TimeoutError, match="job timed out"):
        c._run_bounded(lambda abort: abort.wait(2) and "late", 0.05, "job")


# ---- generate / count_tokens (command mode) ----------------------------------------------------------
def test_generate_runs_a_free_form_completion():
    c, llm = with_llm("Bonjour le monde")
    msgs = [{"role": "system", "content": "Translate to French."}, {"role": "user", "content": "Hello world"}]
    assert c.generate(msgs, max_tokens=64, timeout_s=5, temperature=0.2) == "Bonjour le monde"
    call = llm.calls[0]
    assert call["messages"] == msgs and call["max_tokens"] == 64 and call["temperature"] == 0.2
    assert call["top_k"] == 40 and call["top_p"] == 0.9
    c.generate(msgs, temperature=0)
    assert llm.calls[-1]["top_k"] == 1 and "top_p" not in llm.calls[-1]


def test_generate_without_a_model_and_on_timeout():
    rules = Cleaner(use_llm=False)
    rules.load()
    with pytest.raises(RuntimeError, match="no language model"):
        rules.generate([{"role": "user", "content": "hi"}])
    c, llm = with_llm("slow answer")
    llm.release.clear()
    with pytest.raises(TimeoutError, match="language model timed out"):
        c.generate([{"role": "user", "content": "hi"}], timeout_s=0.1)
    llm.release.set()


def test_count_tokens():
    assert Cleaner(use_llm=False).count_tokens("x" * 30) == 10
    assert Cleaner(use_llm=False).count_tokens("") == 1
    c, _ = with_llm("x")
    assert c.count_tokens("one two three") == 3


# ---- loading -----------------------------------------------------------------------------------------
def test_load_tries_gpu_then_cpu_and_warms_up(monkeypatch):
    c = Cleaner(model="repo:file.gguf", device="auto")
    built = []
    monkeypatch.setattr(C, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(C, "_select_llama_libs", lambda device: None)       # llama_cpp's default folder
    monkeypatch.setattr(c, "_resolve_path", lambda: "C:/models/file.gguf")

    def build(path, gpu):
        built.append((path, gpu))
        if gpu:
            raise RuntimeError("CUDA out of memory")
        return FakeLLM("What is the capital of France?")

    monkeypatch.setattr(c, "_build", build)
    c.load()
    assert built == [("C:/models/file.gguf", True), ("C:/models/file.gguf", False)]
    assert c.backend == "cpu" and c.ready and c.has_llm
    assert len(c._llm.calls) == 1, "one warm-up generation primes the prompt cache"


@pytest.mark.parametrize("device,expected", [("cpu", [False]), ("cuda", [True, False]), ("gpu", [True, False]),
                                             ("vulkan", [True, False]), ("auto", [True, False]), ("quantum", [])])
def test_load_device_selection(monkeypatch, device, expected):
    c = Cleaner(model="repo:file.gguf", device=device)
    built = []
    monkeypatch.setattr(C, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(C, "_select_llama_libs", lambda device: None)
    monkeypatch.setattr(c, "_resolve_path", lambda: "model.gguf")
    monkeypatch.setattr(c, "_build", lambda path, gpu: built.append(gpu) or (_ for _ in ()).throw(OSError("nope")))
    c.load()
    assert built == expected
    assert c.backend == "rules" and c.ready and not c.has_llm, "any load failure degrades to the rule pass"
    assert c.clean("um hello there") == "Hello there."


def test_load_with_a_missing_model_degrades_to_rules(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(C, "_setup_dll_paths", lambda: None)
    c = Cleaner(model=str(tmp_path / "missing.gguf"))
    with caplog.at_level("WARNING"):
        c.load()
    assert c.backend == "rules" and c.ready and "rule-based pass only" in caplog.text


def test_resolve_path_prefers_local_files_then_the_offline_cache(monkeypatch, tmp_path):
    local = tmp_path / "model.gguf"
    local.write_bytes(b"gguf")
    assert Cleaner(model=str(local))._resolve_path() == str(local)
    with pytest.raises(FileNotFoundError, match="repo_id:filename.gguf"):
        Cleaner(model="C:\\nope\\model.gguf")._resolve_path()
    with pytest.raises(FileNotFoundError):
        Cleaner(model="no-colon-here")._resolve_path()

    import sys
    import types
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def download(repo, fname, local_files_only=False):
        calls.append((repo, fname, local_files_only))
        if local_files_only and hub.cached is False:
            raise OSError("not in cache")
        return f"/cache/{repo}/{fname}"

    hub.hf_hub_download, hub.cached = download, True
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert Cleaner(model="org/repo:file.gguf")._resolve_path() == "/cache/org/repo/file.gguf"
    assert calls == [("org/repo", "file.gguf", True)], "a cached model must load without touching the network"
    hub.cached = False
    calls.clear()
    Cleaner(model="org/repo:file.gguf")._resolve_path()
    assert calls == [("org/repo", "file.gguf", True), ("org/repo", "file.gguf", False)]


def test_default_model_and_public_names():
    assert Cleaner().model == C.DEFAULT_MODEL and ":" in C.DEFAULT_MODEL and C.DEFAULT_MODEL.endswith(".gguf")
    for name in C.__all__:
        assert hasattr(C, name)
    assert C.phonetic_key("pie torch".replace(" ", "")) == C.phonetic_key("pytorch")


# =========================================================================== rule follow-ups found while re-verifying
@pytest.mark.parametrize("spoken,expected", [
    ("the beam is 12 mm thick and 3 m long", "The beam is 12 mm thick and 3 m long."),
    ("i saw him in the ER", "I saw him in the ER."),
    ("the um report is uh done", "The report is done."),
    ("open the .env file", "Open the .env file."),
    ("use the .NET framework", "Use the .NET framework."),
    ("version 2.5.1 is out", "Version 2.5.1 is out."),
    ("go to example.com now", "Go to example.com now."),
    ("the file is main.py", "The file is main.py"),
    ("can you send me the file", "Can you send me the file?"),
    ("have you seen the report", "Have you seen the report?"),
    ("what time is it", "What time is it?"),
    ("have fun tonight", "Have fun tonight."),
    ("does not matter", "Does not matter."),
    ("when you are ready let me know", "When you are ready let me know."),
    ("where there is smoke there is fire", "Where there is smoke there is fire."),
    ("what a mess", "What a mess."),
    ("would love to join", "Would love to join."),
    ("hello. how are you. i am fine", "Hello. How are you. I am fine."),
    ("i think i.e. is latin", "I think i.e. is latin."),
    ("visit foo.bar for more", "Visit foo.bar for more."),
    ("new line", "\n"),
    ("dear bob new line how are you new paragraph best dan", "Dear bob\nHow are you\n\nBest dan."),
    ("the new line of products is great", "The new line of products is great."),
    ("start a new paragraph here", "Start a new paragraph here."),
    ("bullet point milk bullet point eggs", "- Milk\n- Eggs"),
    ("the first bullet point was wrong", "The first bullet point was wrong."),
    ("is it done question mark", "Is it done?"),
    ("wow exclamation mark", "Wow!"),
    ("that is all full stop", "That is all."),
    ("meet at 5 no wait 6", "Meet at 6."),
    ("i have two no three cats", "I have three cats."),
    ("send it to bob scratch that send it to alice", "Send it to alice."),
    ("there is no 5 in that list", "There is no 5 in that list."),
    ("say no 2 times", "Say no 2 times."),
    ("the answer is no 4 is wrong", "The answer is no 4 is wrong."),
    ("i waited one no i mean two hours", "I waited two hours."),
    ("hello world!", "Hello world!"),
    ("it's 5 o'clock", "It's 5 o'clock."),
    ("the cost is $5", "The cost is $5."),
    ("3:30 pm works", "3:30 pm works."),
    ("i mean it", "I mean it."),
])
def test_rule_clean_regressions(spoken, expected):
    assert rule_clean(spoken) == expected


@pytest.mark.parametrize("spoken,expected", [
    ("do it now", "Do it now."),                                        # imperative, currently "Do it now?"
    ("which is why i left", "Which is why I left."),                    # currently "...?"
    ("put a question mark at the end", "Put a question mark at the end."),   # currently "Put a? At the end."
    ("we came to a full stop", "We came to a full stop."),              # currently "We came to a."
    ("uh huh that works", "Uh huh that works."),                        # currently "Huh that works."
    ("mm hmm sounds right", "Mm hmm sounds right."),                    # currently "Mm sounds right."
    ("std::map and foo::bar", "std::map and foo::bar."),                # currently "Std::map ..."
    ("this is a test,", "This is a test."),                             # trailing comma left behind
])
def test_c54_rule_clean_follow_ups(spoken, expected):
    assert rule_clean(spoken) == expected
