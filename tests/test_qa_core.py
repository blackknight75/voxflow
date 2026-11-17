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
