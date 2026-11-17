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
    assert c.clean(MESSY) == "I pushed the branch to get hub."          # this one timed out
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


# =========================================================================== C-01: GPU code compiled on first use
from voxflow import gpuprobe  # noqa: E402
from voxflow.gpuprobe import CudaInfo  # noqa: E402

RTX50 = CudaInfo(13030, ((12, 0),))
RTX40 = CudaInfo(13030, ((8, 9),))


class _FakeCuda:
    """nvcuda.dll as ctypes sees it: every call writes through its pointer arguments and returns a CUresult."""

    def __init__(self, version=13030, caps=((12, 0),), fail=None):
        self.version, self.caps, self.fail = version, caps, fail

    def _ret(self, name):
        return 999 if name == self.fail else 0

    def cuInit(self, flags):
        return self._ret("cuInit")

    def cuDriverGetVersion(self, ref):
        ref._obj.value = self.version
        return self._ret("cuDriverGetVersion")

    def cuDeviceGetCount(self, ref):
        ref._obj.value = len(self.caps)
        return self._ret("cuDeviceGetCount")

    def cuDeviceGet(self, ref, ordinal):
        ref._obj.value = ordinal
        return self._ret("cuDeviceGet")

    def cuDeviceGetAttribute(self, ref, attribute, device):
        ref._obj.value = self.caps[device.value][0 if attribute == 75 else 1]
        return self._ret("cuDeviceGetAttribute")


def test_c01_cuda_info_reads_the_driver_version_and_every_devices_compute_capability(monkeypatch):
    loads = []
    monkeypatch.setattr(gpuprobe, "_cuda_cache", [])
    monkeypatch.setattr(gpuprobe, "_load_library",
                        lambda name: loads.append(name) or _FakeCuda(13030, ((12, 0), (8, 6))))
    info = gpuprobe.cuda_info()
    assert info == CudaInfo(13030, ((12, 0), (8, 6))) and str(info) == "CUDA 13.3 driver, sm_120, sm_86"
    assert gpuprobe.cuda_info() is info and loads == ["nvcuda.dll"], "asked once per process"
    assert str(CudaInfo(12080, ())) == "CUDA 12.8 driver, no devices"


@pytest.mark.parametrize("fail", ["cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet",
                                  "cuDeviceGetAttribute", "load"])
def test_c01_a_driver_that_does_not_answer_means_no_cuda_info(monkeypatch, caplog, fail):
    monkeypatch.setattr(gpuprobe, "_cuda_cache", [])

    def load(name):
        if fail == "load":
            raise OSError("Could not find module nvcuda.dll")
        return _FakeCuda(fail=fail)

    monkeypatch.setattr(gpuprobe, "_load_library", load)
    with caplog.at_level("INFO"):
        assert gpuprobe.cuda_info() is None and gpuprobe.cuda_info() is None
    assert "no CUDA devices" in caplog.text


def test_c01_the_native_code_tables_match_the_builds():
    """Read from the fatbins of the two ggml-cuda.dll files (see the report): an RTX 50xx has none in the CUDA 12
    build, an RTX 30xx / 40xx has its own in both."""
    assert (12, 0) not in cleanup._CUDA12_NATIVE and (12, 0) in cleanup._CUDA13_NATIVE
    assert {(8, 6), (8, 9)} <= set(cleanup._CUDA12_NATIVE) & set(cleanup._CUDA13_NATIVE)
    rows = {b.folder: b for b in cleanup.LLAMA_BACKENDS}
    assert rows["lib_cuda13"].native == cleanup._CUDA13_NATIVE and rows["lib"].native == cleanup._CUDA12_NATIVE
    assert str(rows["lib_cuda13"]) == "cuda 13" and str(rows["lib"]) == "cuda" and rows["lib_cuda13"].name == "cuda"


@pytest.mark.parametrize("driver, info, why", [
    (False, RTX50, "no NVIDIA driver (nvcuda.dll)"),
    (True, None, "the NVIDIA driver reports no CUDA device"),
    (True, CudaInfo(13030, ()), "the NVIDIA driver reports no CUDA device"),
    (True, RTX40, "not needed, the CUDA 12 build has native code for CUDA 13.3 driver, sm_89"),
    (True, CudaInfo(13030, ((12, 0), (7, 5))), "no native code for every device (CUDA 13.3 driver, sm_120, sm_75)"),
    (True, CudaInfo(12080, ((12, 0),)), "needs a driver for CUDA 13.0 or newer (CUDA 12.8 driver, sm_120)"),
    (True, RTX50, None),
    (True, CudaInfo(13000, ((12, 0), (8, 9))), None),          # an RTX 5080 next to an RTX 4090
])
def test_c01_the_cuda13_build_is_only_for_gpus_the_cuda12_build_has_no_code_for(monkeypatch, driver, info, why):
    monkeypatch.setattr(gpuprobe, "nvidia_driver", lambda: driver)
    monkeypatch.setattr(gpuprobe, "cuda_info", lambda: info)
    assert cleanup._cuda13_unusable(False) == why


@pytest.mark.parametrize("build, info, expect", [
    ("lib", RTX50, True),                    # the CUDA 12 build on an RTX 5080: compiled on first use
    ("lib", RTX40, False),
    ("lib_cuda13", RTX50, False),
    (None, RTX40, True),                     # a build VoxFlow did not choose: nothing is known to be native
    ("lib", None, False),                    # the driver does not answer: no CUDA GPU to compile for
])
def test_c01_jit_is_expected_only_where_the_build_lacks_code_for_a_gpu(monkeypatch, build, info, expect):
    row = next((b for b in cleanup.LLAMA_BACKENDS if b.folder == build), None)
    monkeypatch.setattr(cleanup, "_llama_build", row)
    monkeypatch.setattr(gpuprobe, "cuda_info", lambda: info)
    assert cleanup._jit_likely() is expect


def _gpu_load(monkeypatch, jit, script="What is the capital of France?", libs="cuda"):
    from test_cleanup_features import FakeLLM
    monkeypatch.setattr(cleanup, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(cleanup, "_select_llama_libs", lambda d: libs)
    monkeypatch.setattr(cleanup, "_gpu_offload_supported", lambda: True)
    monkeypatch.setattr(cleanup, "_jit_likely", lambda: jit)
    llm = FakeLLM(script)
    c = Cleaner(model=__file__, device="auto")
    monkeypatch.setattr(c, "_build", lambda path, gpu: llm)
    return c, llm


def _users(llm) -> list:
    return [call["messages"][-1]["content"] for call in llm.calls]


def test_c01_a_gpu_without_native_code_also_prefills_a_short_turn_during_the_load(monkeypatch):
    c, llm = _gpu_load(monkeypatch, jit=True)
    c.load()
    assert c.backend == "cuda" and c.has_llm
    assert _users(llm)[:2] == ["um so what is the capital of france", cleanup._WARM_SHORT]
    assert llm.calls[1]["max_tokens"] == 1, "a prefill and one token: the kernels, not an answer"
    assert len(cleanup._WARM_SHORT.split()) <= 3, "a few tokens: the small-batch kernels"


def test_c01_a_gpu_with_native_code_pays_nothing_extra(monkeypatch):
    c, llm = _gpu_load(monkeypatch, jit=False)
    c.load()
    assert cleanup._WARM_SHORT not in _users(llm)


def test_c01_no_short_warm_up_on_vulkan_or_the_cpu(monkeypatch):
    c, llm = _gpu_load(monkeypatch, jit=True, libs="vulkan")
    c.load()
    c2, llm2 = _gpu_load(monkeypatch, jit=True, libs="cpu")
    c2.load()
    assert (c.backend, c2.backend) == ("vulkan", "cpu")
    assert cleanup._WARM_SHORT not in _users(llm) + _users(llm2)


def test_c01_a_short_warm_up_that_times_out_keeps_the_gpu(monkeypatch, caplog):
    """Compiling slower than the budget is not a broken GPU: the load finishes on the GPU and later calls wait."""
    c, llm = _gpu_load(monkeypatch, jit=True)
    real = Cleaner._run_bounded

    def bounded(job, budget, what):
        if what == "the short warm-up":
            raise TimeoutError("the short warm-up timed out")
        return real(c, job, budget, what)

    monkeypatch.setattr(c, "_run_bounded", bounded)
    with caplog.at_level("WARNING"):
        c.load()
    assert c.backend == "cuda" and c.has_llm
    assert "the short warm-up timed out; the first dictations may wait for the GPU" in caplog.text


def test_c01_the_cuda13_cublas_folder_is_on_the_dll_search_path(monkeypatch, tmp_path):
    site = tmp_path / "site"
    (site / "nvidia" / "cu13" / "bin" / "x86_64").mkdir(parents=True)
    (site / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    added = []
    monkeypatch.setattr(cleanup, "_dll_dirs_added", False)
    monkeypatch.setattr(cleanup.sys, "platform", "win32")
    monkeypatch.setattr(cleanup.sys, "path", [str(site)])
    monkeypatch.setattr(cleanup.os, "add_dll_directory", lambda d: added.append(d), raising=False)
    monkeypatch.setenv("PATH", "")
    cleanup._setup_dll_paths()
    assert str(site / "nvidia" / "cu13" / "bin" / "x86_64") in added
    assert str(site / "nvidia" / "cublas" / "bin") in added


def test_c01_the_speed_check_measures_cuda_when_either_cuda_build_can_run(monkeypatch):
    from test_speedcheck import FakeCleaner
    from voxflow import speedcheck as sc
    from voxflow.cleanup import LlamaBackend
    state = {"cuda13": "not needed, the CUDA 12 build has native code", "cuda12": None}
    rows = (LlamaBackend("cuda", "lib_cuda13", True, lambda asked: state["cuda13"]),
            LlamaBackend("cuda", "lib", True, lambda asked: state["cuda12"]),
            LlamaBackend("cpu", "lib_cpu", False, lambda asked: None))
    monkeypatch.setattr(cleanup, "LLAMA_BACKENDS", rows)
    monkeypatch.setattr(cleanup, "Cleaner", FakeCleaner)
    monkeypatch.setattr(cleanup, "_folder_unusable", lambda folder: None)
    monkeypatch.setattr(cleanup, "_llama_pkg_dir", lambda: "C:/site/llama_cpp")
    r = sc._measure_cleanup("cuda", "", 4.0)
    assert r["available"] and r["auto"] and r["ok"], "the CUDA 12 build runs: CUDA is measured"
    state.update(cuda13=None, cuda12="no NVIDIA driver (nvcuda.dll)")
    assert sc._measure_cleanup("cuda", "", 4.0)["available"], "the CUDA 13 build alone is enough too"
    state.update(cuda13="no NVIDIA driver (nvcuda.dll)")
    assert sc._measure_cleanup("cuda", "", 4.0) == {"available": False, "auto": False,
                                                    "reason": "no NVIDIA driver (nvcuda.dll)"}


class _Asr:
    def transcribe(self, audio):
        return "please send the report"


class _Reporting:
    ready = True

    def __init__(self, info):
        self.last_info = info

    def clean(self, text, **kw):
        return "Please send the report."


@pytest.mark.parametrize("info, rules_only, skipped", [
    ({"rules_only": True, "skipped_busy": True}, True, True),
    ({"rules_only": True, "skipped_busy": True, "timeout": True}, True, False),   # another line really timed out
    ({"rules_only": True, "timeout": True}, True, False),
    ({"rules_only": False, "skipped_busy": True}, False, False),
    ("garbage", False, False),
])
def test_c01_run_pipeline_tells_a_skip_behind_a_stuck_generation_apart(info, rules_only, skipped):
    from voxflow import app as appmod
    import numpy as np
    out = appmod.run_pipeline(np.zeros(16000, dtype=np.float32), _Asr(), _Reporting(info))
    assert (out["rules_only"], out["model_skipped"]) == (rules_only, skipped)


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    from controller_rig import make_rig
    r = make_rig(tmp_path, qtbot, monkeypatch)
    r.clock = [1000.0]
    r.c._mono = lambda: r.clock[0]
    yield r
    r.close()


def test_c01_dictations_skipped_behind_one_stuck_generation_do_not_raise_the_too_slow_notice(rig):
    """QA C-01 run 1: one stuck generation, two dictations skipped behind it, and "the computer is busy" was told."""
    from voxflow import controller as ctl_mod
    rig.cleaner.last_info = {"rules_only": True, "timeout": True}            # the one that was really slow
    rig.dictate("please send the report")
    rig.cleaner.last_info = {"rules_only": True, "skipped_busy": True}       # the ones behind it
    for _ in range(3):
        events = rig.dictate("please send the report")
        rig.clock[0] += 10
    assert rig.spy.notices() == [] and rig.c.rules_only_count == 1
    done = [args[2] for name, args in events if name == "state_changed" and args[0] == "done"][-1]
    assert done == "Pasted" + ctl_mod.RULES_ONLY_SUFFIX, "the text did get only the basic clean-up: still said"
    assert rig.last_entry()["rules_only"] is True
    rig.cleaner.last_info = {"rules_only": True, "timeout": True}
    rig.dictate("please send the report")
    rig.dictate("please send the report")
    assert rig.spy.notices() == [("info", ctl_mod.RULES_ONLY_NOTICE)], "three real timeouts still do"


# =========================================================================== C-03: "scratch that" after a full stop
from voxflow import langs  # noqa: E402
from voxflow.cleanup import rule_clean  # noqa: E402


@pytest.mark.parametrize("spoken, expect", [
    # Parakeet punctuates: the marker is a sentence of its own, so the sentence before it is taken back
    ("I will call you tomorrow. Scratch that. I will email you tomorrow.", "I will email you tomorrow."),
    ("We met. I will call you tomorrow? Scratch that! I will email.", "We met. I will email."),
    ("Hello. Scratch that.", ""),
    # unchanged: the clause before the marker inside its sentence, or nothing before it at all
    ("i will call you tomorrow scratch that i will email you tomorrow", "I will email you tomorrow."),
    ("First point. Second point, scratch that. Third point.", "First point. Third point."),
    ("Scratch that. Send it now.", "Send it now."),
])
def test_c03_scratch_that_takes_back_the_sentence_the_recogniser_ended(spoken, expect):
    assert rule_clean(spoken) == expect


@pytest.mark.parametrize("lang, spoken, expect", [
    ("es", "Te llamo mañana. Borra eso. Te escribo mañana.", "Te escribo mañana."),
    ("de", "Wir sehen uns. Ich rufe dich morgen an. Streich das! Ich schreibe dir.", "Wir sehen uns. Ich schreibe dir."),
    ("fr", "je t'appelle demain efface ça je t'écris demain", "Je t'écris demain."),
])
def test_c03_the_other_languages_take_it_back_the_same_way(lang, spoken, expect):
    assert langs.rule_clean(spoken, langs.get(lang)) == expect


# =========================================================================== C-04: corrections written with digits
def test_c04_the_guard_accepts_the_model_resolving_a_correction_written_with_digits():
    src = "The deadline is the 10th I mean the 12th."
    assert cleanup._accept(src, "The deadline is the 12th.")
    assert cleanup._accept("The deadline is the tenth I mean the twelfth.", "The deadline is the twelfth.")
    assert not cleanup._accept(src, "The deadline is the 10th."), "keeping the EARLIER value is still refused"
    assert not cleanup._accept("Send 5 boxes by the 3rd.", "Send boxes by the 3rd."), "a number never just goes"


@pytest.mark.parametrize("spoken, expect", [
    ("the deadline is the 10th i mean the 12th", "The deadline is the 12th."),
    ("its the 3rd sorry the 4th floor", "It's the 4th floor."),
    ("meet at 10 i mean the 12th", "Meet at the 12th."),            # the article was not a repeat: it stays
    ("the red no the blue car", "The blue car."),
    ("call me at 5 no 6", "Call me at 6."),                        # unchanged
    ("page one wait 5 minutes", "Page one wait 5 minutes."),        # unchanged: not a correction
    ("wait the 5 minutes are up", "Wait the 5 minutes are up."),
])
def test_c04_the_rules_resolve_digit_corrections_with_a_repeated_article(spoken, expect):
    assert rule_clean(spoken) == expect


def test_c04_the_whole_pipeline_pastes_the_corrected_date():
    c, llm = with_llm("The deadline is the 12th.")
    assert c.clean("The deadline is the 10th I mean the 12th.") == "The deadline is the 12th."
    assert c.last_info["rejected"] is False


# =========================================================================== C-05: stutters
@pytest.mark.parametrize("spoken, expect", [
    ("I I think that the problem is in the the database.", "I think that the problem is in the database."),
    ("we need to to fix it", "We need to fix it."),
    ("i uh i think so", "I think so."),
    ("We, we should ship it", "We should ship it."),
    # real doubles stay
    ("he had had enough", "He had had enough."),
    ("I know that that is true", "I know that that is true."),
    ("what it is is fine", "What it is is fine."),
    ("come in in five minutes", "Come in in five minutes."),
    ("he he that is funny", "He he that is funny."),
])
def test_c05_the_rules_take_out_stutters_on_words_that_are_never_doubled(spoken, expect):
    assert rule_clean(spoken) == expect


def test_c05_a_refused_model_answer_no_longer_brings_the_stutters_back():
    """QA case "repeat": the model also dropped "that", the guard refused its answer, the stutters were pasted."""
    c, llm = with_llm("I think the problem is in the database.")
    assert c.clean("I I think that the problem is in the the database.") == \
        "I think that the problem is in the database."
    assert c.last_info["rejected"] is True


# =========================================================================== C-06: VRAM held after long dictations
import types  # noqa: E402

import numpy as np  # noqa: E402

from voxflow import asr  # noqa: E402
from test_asr_unit import fake_ort  # noqa: E402,F401  (fixture)


class _Encoder:
    def __init__(self):
        self.runs = []

    def run(self, output_names, input_feed, run_options=None):
        self.runs.append(run_options)
        return ["out"]

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def run_options(monkeypatch, fake_ort):  # noqa: F811
    class RunOptions:
        def __init__(self):
            self.entries = {}

        def add_run_config_entry(self, key, value):
            self.entries[key] = value

    fake_ort.RunOptions = RunOptions
    return fake_ort


def _features(seconds):
    return {"audio_signal": np.zeros((1, 128, int(seconds * 100)), dtype=np.float32), "length": np.array([1])}


@pytest.mark.parametrize("seconds, free, shrinks", [
    (80, 900, True),                 # a long clip on a GPU short of memory: its working memory goes back
    (80, 8000, False),               # room to spare: kept, so the next long clip is as fast as before
    (80, None, False),               # cannot tell: as before
    (20, 900, False),                # a short clip never needs much
])
def test_c06_a_long_clip_gives_its_gpu_memory_back_only_when_the_gpu_is_short_of_it(run_options, monkeypatch,
                                                                                   seconds, free, shrinks):
    asked = []
    monkeypatch.setattr(gpuprobe, "cuda_free_mb", lambda: asked.append(1) or free)
    encoder = _Encoder()
    session = asr._ShrinkingSession(encoder)
    assert session.run(["outputs"], _features(seconds)) == ["out"]
    (options,) = encoder.runs
    if shrinks:
        assert options.entries == {"memory.enable_memory_arena_shrinkage": "gpu:0"}
    else:
        assert options is None
    assert bool(asked) == (seconds > asr.SHRINK_AFTER_S), "short clips do not even ask"


def test_c06_the_wrapper_is_the_encoder_for_everything_else(run_options, monkeypatch):
    monkeypatch.setattr(gpuprobe, "cuda_free_mb", lambda: 100)
    encoder = _Encoder()
    session = asr._ShrinkingSession(encoder)
    mine = object()
    session.run(["outputs"], _features(80), mine)
    session.run(["outputs"], [np.zeros(3)])                     # not a feed it understands: passed on as is
    assert encoder.runs == [mine, None], "the caller's own options win"
    assert session.get_providers() == encoder.get_providers()
    model = types.SimpleNamespace(asr=types.SimpleNamespace(_encoder=session))
    assert asr._runs_on(model, "CUDAExecutionProvider"), "the check that the encoder runs on the GPU still works"


def test_c06_the_cuda_build_wraps_its_encoder_and_sets_the_memory_probe_up(fake_ort, monkeypatch, caplog):  # noqa: F811
    probes = []
    monkeypatch.setattr(gpuprobe, "cuda_free_mb", lambda: probes.append(1) or 5000)
    encoder = _Encoder()

    def load_model(name, **kw):
        decoder = types.SimpleNamespace(_model_path="decoder_joint-model.onnx")
        return types.SimpleNamespace(asr=types.SimpleNamespace(_decoder_joint=decoder, _encoder=encoder))

    monkeypatch.setattr(fake_ort.onnx_asr, "load_model", load_model)
    model = asr._build_cuda("my-model")
    assert isinstance(model.asr._encoder, asr._ShrinkingSession) and model.asr._encoder._session is encoder
    assert probes == [1], "the first probe (NVML start) belongs to the load, not to a dictation"
    bare = types.SimpleNamespace(asr=types.SimpleNamespace())
    with caplog.at_level("INFO"):
        assert asr._shrink_long_runs(bare) is bare and not hasattr(bare.asr, "_encoder")
    assert "not shrunk after long clips" in caplog.text


class _FakeNvml:
    def __init__(self, free_mb=1500, init=0, handle=0, info=0):
        self.free_mb, self.init, self.handle, self.info = free_mb, init, handle, info
        self.bus = None

    def nvmlInit_v2(self):
        return self.init

    def nvmlDeviceGetHandleByPciBusId_v2(self, bus, ref):
        self.bus = bus.value
        return self.handle

    def nvmlDeviceGetMemoryInfo(self, handle, ref):
        ref._obj.free = self.free_mb << 20
        return self.info


class _FakeCudaBus(_FakeCuda):
    def __init__(self, bus_status=0):
        super().__init__()
        self.bus_status = bus_status

    def cuDeviceGetPCIBusId(self, buf, size, dev):
        buf.value = b"0000:01:00.0"
        return self.bus_status


def _probe(monkeypatch, cu, nvml, info=RTX50):
    monkeypatch.setattr(gpuprobe, "_cuda_cache", [info])
    monkeypatch.setattr(gpuprobe, "_nvml_cache", [])
    monkeypatch.setattr(gpuprobe, "_load_library", lambda name: {"nvcuda.dll": cu, "nvml.dll": nvml}[name])


def test_c06_free_gpu_memory_comes_from_nvml_for_the_cuda_device_by_pci_bus(monkeypatch):
    nvml = _FakeNvml(free_mb=1500)
    _probe(monkeypatch, _FakeCudaBus(), nvml)
    assert gpuprobe.cuda_free_mb() == 1500 and nvml.bus == b"0000:01:00.0"
    nvml.free_mb = 9000
    assert gpuprobe.cuda_free_mb() == 9000, "asked fresh every time, set up once"
    nvml.info = 3
    assert gpuprobe.cuda_free_mb() is None


@pytest.mark.parametrize("cu, nvml, info", [
    (_FakeCudaBus(), _FakeNvml(), None),                   # no CUDA driver at all
    (_FakeCudaBus(bus_status=1), _FakeNvml(), RTX50),
    (_FakeCudaBus(), _FakeNvml(init=9), RTX50),
    (_FakeCudaBus(), _FakeNvml(handle=13), RTX50),
])
def test_c06_free_gpu_memory_unknown_when_the_driver_cannot_say(monkeypatch, cu, nvml, info):
    _probe(monkeypatch, cu, nvml, info)
    assert gpuprobe.cuda_free_mb() is None and gpuprobe.cuda_free_mb() is None


# =========================================================================== C-07: "actually let's make it ..."
@pytest.mark.parametrize("src, out, ok", [
    ("Let's meet on Monday actually let's make it Tuesday.", "Let's meet on Tuesday.", True),
    ("Set the color to red actually change it to blue.", "Set the color to blue.", True),
    ("Book it for 5 people actually make that 6 people.", "Book it for 6 people.", True),
    # still refused: the later value dropped, words that are not a rephrasing, values of another kind
    ("Let's meet on Monday actually let's make it Tuesday.", "Let's meet on Monday.", False),
    ("Meet on Monday actually Tuesday works.", "Meet on Monday.", False),
    ("Meet on Monday actually let us skip it Tuesday.", "Meet on Tuesday.", False),
    ("Call me Monday actually the meeting is Tuesday.", "Call me Tuesday.", False),
    ("Send it to Bob actually let us make it the report.", "Send it to the report.", False),
])
def test_c07_a_rephrased_lead_in_after_the_marker_is_an_allowed_correction(src, out, ok):
    assert cleanup._accept(src, out) is ok


def test_c07_the_corrected_day_is_pasted():
    c, llm = with_llm("Let's meet on Tuesday.")
    assert c.clean("let's meet on monday actually let's make it tuesday") == "Let's meet on Tuesday."
    assert c.last_info["rejected"] is False


# =========================================================================== C-08 (and QA-robust R11): ASR on the CPU
def _gpu_error_once(rig):
    """The transcriber's own behaviour (voxflow.asr): a GPU error mid-dictation, rebuilt on the CPU, text returned."""
    real = rig.asr.transcribe

    def transcribe(audio):
        rig.asr.backend = "cpu"
        return real(audio)

    rig.asr.transcribe = transcribe


def test_c08_a_dictation_that_moved_speech_to_the_cpu_is_reported_once_with_a_way_back(rig):
    from voxflow import controller as ctl_mod
    actions = []
    rig.c.notice_action.connect(lambda *a: actions.append(a))
    assert rig.c.model_status("asr")["backend"] == "cuda"
    _gpu_error_once(rig)
    events = rig.dictate("please send the report")
    rig.wait(lambda: rig.c.model_status("asr")["state"] == "degraded")
    assert rig.injector.pasted[-1] == "Please send the report.", "the dictation itself still went through"
    status = rig.c.model_status("asr")
    assert (status["backend"], status["detail"]) == ("cpu", ctl_mod.ASR_CPU_FALLBACK)
    assert rig.spy.notices() == [("warning", ctl_mod.ASR_CPU_FALLBACK)]
    assert actions == [(ctl_mod.ASR_CPU_FALLBACK, "open_settings:models", "Open model settings")]
    assert rig.c.status()["level"] == "warning" and rig.c.status()["detail"] == ctl_mod.ASR_CPU_FALLBACK
    assert "Retry" in ctl_mod.ASR_CPU_FALLBACK and "slower" in ctl_mod.ASR_CPU_FALLBACK
    events = rig.dictate("and another one")
    assert rig.injector.pasted[-1] == "And another one.", "dictation keeps working on the CPU"
    assert len(rig.spy.notices()) == 1, "told once, not after every dictation"
    del rig.asr.transcribe                              # the GPU works again ...
    rig.asr.backend = "cuda"
    rig.c.reload_models(["asr"])                       # ... and the card's Retry loads the model on it
    rig.wait(lambda: rig.c.model_status("asr")["state"] == "ready")
    assert rig.c.status()["level"] != "warning" and rig.c.model_status("asr")["backend"] == "cuda"


def test_c08_command_mode_reports_the_move_too(rig):
    _gpu_error_once(rig)
    rig.injector.selection = "hey bob can u send me the report"
    rig.dictate("make this more formal", binding="command")
    rig.wait(lambda: rig.c.model_status("asr")["state"] == "degraded")
    assert len(rig.spy.notices()) >= 1


def test_c08_a_transcriber_that_was_on_the_cpu_from_the_start_is_not_reported(rig):
    rig.c._models["asr"] = dict(rig.c._models["asr"], backend="cpu")
    rig.asr.backend = "cpu"
    rig.dictate("please send the report")
    assert rig.c.model_status("asr")["state"] == "ready" and rig.spy.notices() == []


def test_c08_a_failed_transcription_still_checks(rig):
    def broken(audio):
        rig.asr.backend = "cpu"
        raise RuntimeError("the CPU rebuild failed as well")

    rig.asr.transcribe = broken
    rig.dictate("please send the report")
    rig.wait(lambda: rig.c.model_status("asr")["state"] == "degraded")


# =========================================================================== C-09: silence raises one notice
def test_c09_a_silent_recording_raises_one_notice_with_the_microphone_button(rig):
    from controller_rig import TRecorder
    actions = []
    rig.c.notice_action.connect(lambda *a: actions.append(a))
    TRecorder.audio = np.zeros(32000, dtype=np.float32)
    rig.dictate("")
    assert rig.c.message == "Heard only silence" and rig.c.error_info()["kind"] == "silence"
    hint = "Heard only silence. Check that your microphone isn't muted and is the one selected."
    assert rig.spy.notices() == [("warning", hint)]
    assert actions == [(hint, "open_settings:microphone", "Open microphone settings")]
