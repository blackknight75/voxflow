"""Coverage of the remaining corners of ``voxflow.cleanup``: the CUDA DLL setup, the one-time CUDA 12 install,
guardrail rejections, smart-formatting refusals and the bounded LLM call. No model, no network, no GPU."""
from __future__ import annotations

import importlib.util
import os
import runpy
import sys
import threading
import types
import warnings
import zipfile

import pytest

from voxflow import cleanup as C
from voxflow.cleanup import Cleaner, apply_tone, rule_clean, smart_format

from test_cleanup_features import with_llm


# =========================================================================== CUDA DLL discovery
def _patch_find_spec(monkeypatch, llama_cpp):
    """Only the lookup of ``llama_cpp`` is scripted; every other import keeps working."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: llama_cpp() if name == "llama_cpp" else real(name, *a))


@pytest.fixture
def dll_env(monkeypatch, tmp_path):
    """A fake site-packages: llama_cpp/lib, a second llama_cpp location without lib, and two nvidia/*/bin."""
    site = tmp_path / "site"
    (site / "llama_cpp" / "lib").mkdir(parents=True)
    (tmp_path / "bare" / "llama_cpp").mkdir(parents=True)
    (site / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (site / "nvidia" / "refused" / "bin").mkdir(parents=True)
    added: list[str] = []

    def add_dll_directory(d):
        if "refused" in d:
            raise OSError("not a directory Windows accepts")
        added.append(d)

    monkeypatch.setattr(C, "_dll_dirs_added", False)
    monkeypatch.setattr(C.sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setenv("PATH", "ORIGINAL")
    monkeypatch.setattr(sys, "path", [str(site), str(tmp_path / "no-such-dir")])
    spec = types.SimpleNamespace(submodule_search_locations=[str(tmp_path / "bare" / "llama_cpp"),
                                                             str(site / "llama_cpp")])
    _patch_find_spec(monkeypatch, lambda: spec)
    return types.SimpleNamespace(site=site, added=added)


def test_setup_dll_paths_registers_llama_and_nvidia_dirs_once(dll_env, monkeypatch):
    C._setup_dll_paths()
    lib = str(dll_env.site / "llama_cpp" / "lib")
    cublas = str(dll_env.site / "nvidia" / "cublas" / "bin")
    refused = str(dll_env.site / "nvidia" / "refused" / "bin")
    assert dll_env.added == [lib, cublas], "the location without lib/ is skipped, a refused directory is survived"
    path = os.environ["PATH"].split(os.pathsep)
    assert path[-1] == "ORIGINAL" and set(path[:-1]) == {lib, cublas, refused}, "PATH is prepended, never replaced"

    # a second call is a no-op: nothing registered twice, PATH does not grow
    before = os.environ["PATH"]
    C._setup_dll_paths()
    assert dll_env.added == [lib, cublas] and os.environ["PATH"] == before


def test_setup_dll_paths_without_llama_cpp_still_adds_the_nvidia_dirs(dll_env, monkeypatch):
    _patch_find_spec(monkeypatch, lambda: None)
    C._setup_dll_paths()
    assert dll_env.added == [str(dll_env.site / "nvidia" / "cublas" / "bin")]


def test_setup_dll_paths_survives_a_broken_llama_cpp_lookup(dll_env, monkeypatch):
    def boom():
        raise ImportError("broken distribution metadata")

    _patch_find_spec(monkeypatch, boom)
    C._setup_dll_paths()                                      # must not raise
    assert dll_env.added == [str(dll_env.site / "nvidia" / "cublas" / "bin")]
    assert C._dll_dirs_added is True


# =========================================================================== one-time CUDA 12 install
def _fake_llama_spec(monkeypatch, location):
    spec = None if location is None else types.SimpleNamespace(submodule_search_locations=[str(location)])
    _patch_find_spec(monkeypatch, lambda: spec)


def test_install_cuda12_runtime_copies_only_the_cuda12_dlls(monkeypatch, tmp_path, capsys):
    import subprocess
    pkg = tmp_path / "llama_cpp"
    (pkg / "lib").mkdir(parents=True)
    _fake_llama_spec(monkeypatch, pkg)
    commands = []

    def fake_pip(cmd):
        commands.append(cmd)
        target = cmd[cmd.index("-d") + 1]
        with zipfile.ZipFile(os.path.join(target, "nvidia_cuda_runtime_cu12.whl"), "w") as z:
            z.writestr("nvidia/cuda_runtime/bin/cudart64_12.dll", b"CUDART")
            z.writestr("nvidia/cuda_runtime/bin/cudart64_13.dll", b"WRONG MAJOR")
            z.writestr("nvidia/cuda_runtime/METADATA", b"text")
        with zipfile.ZipFile(os.path.join(target, "nvidia_cublas_cu12.whl"), "w") as z:
            z.writestr("nvidia/cublas/bin/cublas64_12.dll", b"CUBLAS")
            z.writestr("nvidia/cublas/bin/cublasLt64_12.dll", b"CUBLASLT")
            z.writestr("nvidia/cublas/bin/nvblas64_12.dll", b"NOT NEEDED")

    monkeypatch.setattr(subprocess, "check_call", fake_pip)
    with pytest.raises(SystemExit) as refused:                  # not the reviewed wheels: nothing is copied
        C.install_cuda12_runtime()
    assert "is not the reviewed file" in str(refused.value) and list((pkg / "lib").iterdir()) == []

    def digests(cmd):
        import hashlib
        fake_pip(cmd)
        target = cmd[cmd.index("-d") + 1]
        monkeypatch.setattr(C, "_CUDA12_WHEELS", tuple(
            (n, v, hashlib.sha256(open(os.path.join(target, f), "rb").read()).hexdigest())
            for (n, v, _h), f in zip(C._CUDA12_WHEELS, sorted(os.listdir(target), reverse=True))))

    commands.clear()
    monkeypatch.setattr(subprocess, "check_call", digests)
    C.install_cuda12_runtime()

    (cmd,) = commands
    assert cmd[:4] == [sys.executable, "-m", "pip", "download"] and "--no-deps" in cmd
    assert "nvidia-cuda-runtime-cu12==12.9.79" in cmd and "nvidia-cublas-cu12==12.9.2.10" in cmd
    lib = pkg / "lib"
    assert sorted(p.name for p in lib.iterdir()) == ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]
    assert (lib / "cudart64_12.dll").read_bytes() == b"CUDART"
    assert (lib / "cublasLt64_12.dll").read_bytes() == b"CUBLASLT"
    out = capsys.readouterr().out
    assert "installed cudart64_12.dll" in out and "cudart64_13" not in out


def test_install_cuda12_runtime_needs_llama_cpp(monkeypatch):
    import subprocess
    _fake_llama_spec(monkeypatch, None)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd: pytest.fail("nothing may be downloaded"))
    with pytest.raises(SystemExit) as exc:
        C.install_cuda12_runtime()
    assert "llama-cpp-python is not installed" in str(exc.value)


def _run_as_script(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # "found in sys.modules": expected for python -m
        runpy.run_module("voxflow.cleanup", run_name="__main__")


def test_module_entry_point_prints_usage(monkeypatch, capsys):
    _run_as_script(monkeypatch, ["voxflow.cleanup"])
    assert capsys.readouterr().out.strip() == "usage: python -m voxflow.cleanup --install-cuda"


def test_module_entry_point_install_cuda_runs_the_installer(monkeypatch):
    _fake_llama_spec(monkeypatch, None)                       # the installer stops before any download
    with pytest.raises(SystemExit) as exc:
        _run_as_script(monkeypatch, ["voxflow.cleanup", "--install-cuda"])
    assert "re-run --install-cuda" in str(exc.value)


# =========================================================================== rule pass
def test_wait_before_a_duration_is_not_a_number_correction():
    assert rule_clean("give me five wait six minutes") == "Give me five wait six minutes."
    assert rule_clean("give me five wait six") == "Give me six.", "without the duration it is a correction"


# =========================================================================== guardrails
@pytest.mark.parametrize("word,kind", [("5", "num"), ("five", "num"), ("monday", "day"), ("may", "month"),
                                       ("grey", "colour"), ("cat", None)])
def test_kind_of_a_replaceable_value(word, kind):
    assert C._kind(word) == kind


@pytest.mark.parametrize("src,out", [
    ("meet on monday tuesday afternoon", "Meet on Tuesday afternoon"),       # day replaced by the later day
    ("ship it in march april maybe", "Ship it in April maybe"),              # month
    ("paint it red blue today", "Paint it blue today"),                      # colour
    ("we got five six items", "We got six items"),                           # number
])
def test_accept_allows_dropping_the_earlier_of_two_values_of_the_same_kind(src, out):
    assert C._accept(src, out) is True


@pytest.mark.parametrize("src,out,why", [
    ("meet on monday afternoon please", "Meet on afternoon please", "a dropped value with no replacement"),
    ("paint it red tuesday today", "Paint it Tuesday today", "the replacement is of a different kind"),
    ("hello there", "!!!", "no words at all"),
    ("hello there", "hello there ```", "a code fence"),
    ("hello there", "hello <b>there</b>", "markup that was not dictated"),
    ("hello there my friend", "hello there my pal", "a new short word"),
    ("alpha beta gamma delta", "beta alpha gamma delta", "words out of order"),
    ("version 3.5 has two bugs", "version 3 5 has two bugs", "a number that was not copied verbatim"),
])
def test_accept_rejects_unfaithful_output(src, out, why):
    assert C._accept(src, out) is False, why


def test_accept_limits_the_number_of_respelt_words():
    src = "i recieve the seperate calender"
    assert C._accept(src, "I receive the separate calendar") is True, "three respellings are fine"
    assert C._accept(src + " tommorow", "I receive the separate calendar tomorrow") is False,         "more than three fuzzy matches is a rewrite"


def test_close_span_ignores_short_words():
    assert C._close_span("hubs", ["get", "hub"]) is None
    assert list(C._close_span("github", ["get", "hub"])) == [0, 1]


# =========================================================================== tone
def test_very_casual_keeps_hyphenated_and_elided_names_in_other_languages():
    assert apply_tone("Jean-Pierre arrive demain.", "very_casual", "fr") == "Jean-Pierre arrive demain"
    assert apply_tone("D'Artagnan arrive demain.", "very_casual", "fr") == "D'Artagnan arrive demain"
    assert apply_tone("Diz-me tudo. Demain il pleut.", "very_casual", "pt") == "diz-me tudo. demain il pleut", \
        "a hyphen followed by lower case is ordinary grammar, not a name"


# =========================================================================== smart formatting
@pytest.mark.parametrize("tokens", [["five", "twenty"], ["twenty", "thirty"], ["one", "thousand", "two", "million"]])
def test_parse_number_rejects_runs_that_are_not_one_number(tokens):
    assert C._parse_number(tokens) is None


@pytest.mark.parametrize("spoken,written", [
    ("page five twenty is torn", "Page five twenty is torn"),                      # ones then tens: two numbers
    ("one thousand two million stars", "One thousand two million stars"),          # scales must get smaller
    ("it costs five dollars and two hundred cents", "It costs $5 and 200 cents"),  # 200 cents is not a price
    ("five dollars two people", "$5 two people"),                                  # no "cents", below ten
    ("version three point two point one percent", "Version three point two point one percent"),
    ("see you march eleventh", "See you March 11th"),
    ("see you march twelfth or thirteenth", "See you March 12th or thirteenth"),
    ("see you march twenty tenth", "See you march twenty tenth"),                  # not a day of the month
    ("the twenty tenth of march", "The twenty tenth of march"),
    ("the thirty fifth of march", "The thirty fifth of march"),                    # day 35
    ("the one am calling", "The one am calling"),                                  # "am" the verb, not a time
    ("call me at one am", "Call me at 1 AM"),
    ("it was 13.30 pm sharp", "It was 13.30 pm sharp"),                            # no such hour
    ("at 15 pm sharp", "At 15 pm sharp"),
    ("at 3 pm sharp", "At 3 PM sharp"),
])
def test_smart_format_leaves_what_is_not_a_number_date_or_time(spoken, written):
    assert smart_format(spoken) == written


def test_numbered_list_skips_an_item_that_was_never_spoken():
    assert smart_format("number one milk number two") == "1. Milk"
    assert smart_format("number one milk number two eggs") == "1. Milk\n2. Eggs"


def test_smart_format_keeps_blank_lines():
    assert smart_format("hello\n\nworld of twenty five things") == "Hello\n\nWorld of 25 things"


# =========================================================================== Cleaner with a scripted model
def test_run_bounded_keeps_a_result_that_arrives_exactly_at_the_deadline():
    """The generation finishes between the wait timing out and the hand-off lock being taken: the result counts,
    nothing is abandoned and the next call is not refused."""
    c = Cleaner(use_llm=False)
    main = threading.current_thread()
    finish, worker_done = threading.Event(), threading.Event()
    real = threading.Lock()

    class Handoff:
        def __enter__(self):
            if threading.current_thread() is main:          # the caller, after its wait timed out
                finish.set()                                # ... lets the generation complete first
                assert worker_done.wait(5)
            real.acquire()

        def __exit__(self, *exc):
            real.release()
            if threading.current_thread() is not main:
                worker_done.set()

    c._handoff = Handoff()

    def job(abort):
        assert finish.wait(5)
        return "just in time"

    assert c._run_bounded(job, 0.05, "cleanup LLM") == "just in time"
    assert not c._abandoned.is_set()
    assert c._run_bounded(lambda abort: "next", 1.0, "cleanup LLM") == "next"


def test_blank_lines_between_paragraphs_never_reach_the_model():
    c, llm = with_llm(lambda user, msgs: user)
    out = c.clean("hello there my um friend new paragraph how are you um today", language="en")
    assert out == "Hello there my friend\n\nHow are you today?"
    assert [call["messages"][-1]["content"] for call in llm.calls] == ["Hello there my friend", "How are you today?"]


def test_a_full_stop_the_model_adds_to_a_bullet_is_removed():
    c, llm = with_llm(lambda user, msgs: user + ".")
    out = c.clean("bullet point buy the um milk and eggs today bullet point call my um mother later tonight",
                  language="en")
    assert out == "- Buy the milk and eggs today\n- Call my mother later tonight"
    assert len(llm.calls) == 2 and c.last_info["llm"] is True and c.last_info["rejected"] is False


def test_other_language_lines_are_left_to_the_rules_when_no_time_is_left():
    c, llm = with_llm("NO DEBERÍA USARSE", timeout_s=0.01)
    text = "creo que deberíamos eh desplegarlo en en staging primero"
    out = c.clean(text, language="es")
    assert out == "Creo que deberíamos desplegarlo en en staging primero."
    assert llm.calls == [], "below 50 ms of budget the model is not even started"
    assert c.last_info["rules_only"] is True and c.last_info["llm"] is False and c.last_info["language"] == "es"
