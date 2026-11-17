"""Which llama.cpp build the language model loads (CUDA, Vulkan or CPU) and how a load falls back.

Everything is faked: the machine probe, the build folders, the DLL load and llama_cpp itself. No model, no
network, no GPU, no real DLL."""
from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import types
from importlib import metadata

import pytest

from voxflow import app, cleanup as C, fakes, gpuprobe
from voxflow.cleanup import Cleaner
from voxflow.config import Config
from voxflow.gpuprobe import Gpu

from test_cleanup_features import FakeLLM
from tests.test_app import _private_instance_channel, clean_process_state, desktop  # noqa: F401

RTX = Gpu("NVIDIA GeForce RTX 5080", 0x10DE, "discrete", 15977)
RX = Gpu("AMD Radeon RX 7600", 0x1002, "discrete", 8176)
IGPU = Gpu("AMD Radeon(TM) Graphics", 0x1002, "integrated", 10791)
SMALL = Gpu("Intel(R) Arc(TM) A310", 0x8086, "discrete", 3072)


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A PC to script: which drivers answer, which build folders exist, which llama.dll loads."""
    pkg = tmp_path / "site" / "llama_cpp"
    pkg.mkdir(parents=True)
    state = types.SimpleNamespace(pkg=pkg, nvidia=False, gpus=[], broken=set(), preloaded=[], version="0.3.36",
                                  cuda=None)

    def add(folder, version=None):
        (pkg / folder).mkdir()
        (pkg / folder / "llama.dll").write_bytes(b"MZ")
        if version is not None:
            (pkg / folder / "version.txt").write_text(version + "\n", "utf-8")
        return str(pkg / folder)

    def preload(folder):
        state.preloaded.append(os.path.basename(folder))
        if os.path.basename(folder) in state.broken:
            raise OSError("[WinError 126] The specified module could not be found")

    def version(dist):
        if state.version is None:
            raise metadata.PackageNotFoundError(dist)
        return state.version

    state.add = add
    monkeypatch.setattr(C, "_llama_choice", C._UNSET)
    monkeypatch.setattr(C, "_llama_build", None)
    monkeypatch.setattr(gpuprobe, "cuda_info", lambda: state.cuda)
    monkeypatch.setattr(C.sys, "platform", "win32")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", "")                 # restored (removed) after the test
    monkeypatch.setattr(C, "_llama_pkg_dir", lambda: str(pkg))
    monkeypatch.setattr(C, "_preload", preload)
    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(gpuprobe, "nvidia_driver", lambda: state.nvidia)
    monkeypatch.setattr(gpuprobe, "vulkan_gpus", lambda: list(state.gpus))
    monkeypatch.setattr(gpuprobe, "_vulkan_cache", None)
    return state


def all_builds(machine):
    return machine.add("lib"), machine.add("lib_vulkan", "0.3.36"), machine.add("lib_cpu", "0.3.36")


# =========================================================================== the ordered list
def test_the_backends_are_an_ordered_list_with_the_cpu_last():
    assert [(b.name, b.folder, b.gpu) for b in C.LLAMA_BACKENDS] == [
        ("cuda", "lib_cuda13", True), ("cuda", "lib", True), ("vulkan", "lib_vulkan", True), ("cpu", "lib_cpu", False)]
    assert C._GPU_DEVICES == ("auto", "gpu", "cuda", "cuda", "vulkan")
    assert C._cpu_unusable(False) is None and C._cpu_unusable(True) is None


def test_nvidia_pc_takes_the_cuda_build_and_asks_no_further(machine, caplog):
    cuda, _vulkan, _cpu = all_builds(machine)
    machine.nvidia, machine.gpus = True, [RTX]
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") == "cuda"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == cuda and machine.preloaded == ["lib"]
    assert f"llama.cpp libraries: cuda build, {cuda} (device auto; cuda 13: not installed)" in caplog.text
    assert C._llama_build.folder == "lib"


def test_qa_c01_an_rtx_50_takes_the_cuda13_build_with_native_code(machine, caplog):
    cuda13 = machine.add("lib_cuda13", "0.3.36")
    all_builds(machine)
    machine.nvidia, machine.gpus, machine.cuda = True, [RTX], gpuprobe.CudaInfo(13030, ((12, 0),))
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") == "cuda"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == cuda13 and machine.preloaded == ["lib_cuda13"]
    assert f"llama.cpp libraries: cuda 13 build, {cuda13} (device auto)" in caplog.text
    assert C._llama_build.folder == "lib_cuda13" and C._jit_likely() is False


def test_qa_c01_a_cuda13_build_that_does_not_load_falls_back_to_the_cuda12_one(machine, caplog):
    """No CUDA 13 cuBLAS, say: the CUDA 12 build still runs the model (compiling its code on first use)."""
    machine.add("lib_cuda13", "0.3.36")
    cuda, _vulkan, _cpu = all_builds(machine)
    machine.nvidia, machine.cuda, machine.broken = True, gpuprobe.CudaInfo(13030, ((12, 0),)), {"lib_cuda13"}
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("cuda") == "cuda"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == cuda and machine.preloaded == ["lib_cuda13", "lib"]
    assert "cuda 13: did not load ([WinError 126]" in caplog.text and C._jit_likely() is True


def test_qa_c01_an_rtx_40_keeps_the_cuda12_build_as_before(machine, caplog):
    machine.add("lib_cuda13", "0.3.36")
    cuda, _vulkan, _cpu = all_builds(machine)
    machine.nvidia, machine.cuda = True, gpuprobe.CudaInfo(13030, ((8, 9),))
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") == "cuda"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == cuda and machine.preloaded == ["lib"], "the CUDA 13 build is not loaded"
    assert "cuda 13: not needed, the CUDA 12 build has native code for CUDA 13.3 driver, sm_89" in caplog.text
    assert C._jit_likely() is False


def test_amd_or_intel_card_takes_the_vulkan_build(machine, caplog):
    _cuda, vulkan, _cpu = all_builds(machine)
    machine.gpus = [IGPU, RX]
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") == "vulkan"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == vulkan and machine.preloaded == ["lib_vulkan"]
    assert "(device auto; cuda 13: not installed; cuda: no NVIDIA driver (nvcuda.dll))" in caplog.text, \
        "the log says why CUDA was passed over"


def test_pc_without_a_gpu_driver_takes_the_cpu_build(machine, caplog):
    """The release blocker: the CUDA build cannot even be loaded here, and the model must still run."""
    _cuda, _vulkan, cpu = all_builds(machine)
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") == "cpu"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == cpu and machine.preloaded == ["lib_cpu"]
    assert ("cuda: no NVIDIA driver (nvcuda.dll); vulkan: no discrete Vulkan GPU with 3500 MiB of memory "
            "(devices: none)") in caplog.text


def test_a_build_that_does_not_load_is_passed_over_for_the_next(machine, caplog):
    """NVIDIA driver without the CUDA 12 runtime, then a Vulkan driver that fails: the CPU build is used."""
    _cuda, _vulkan, cpu = all_builds(machine)
    machine.nvidia, machine.gpus, machine.broken = True, [RTX], {"lib", "lib_vulkan"}
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") == "cpu"
    assert machine.preloaded == ["lib", "lib_vulkan", "lib_cpu"] and os.environ["LLAMA_CPP_LIB_PATH"] == cpu
    assert "cuda: did not load ([WinError 126]" in caplog.text and "vulkan: did not load" in caplog.text


def test_integrated_and_too_small_gpus_are_not_picked_automatically(machine, caplog):
    all_builds(machine)
    machine.gpus = [IGPU, SMALL]
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("gpu") == "cpu"
    assert "no discrete Vulkan GPU with 3500 MiB of memory (devices: AMD Radeon(TM) Graphics (AMD, integrated, " \
           "10791 MiB), Intel(R) Arc(TM) A310 (Intel, discrete, 3072 MiB))" in caplog.text


def test_asking_for_vulkan_skips_cuda_and_accepts_an_integrated_gpu(machine):
    _cuda, vulkan, _cpu = all_builds(machine)
    machine.nvidia, machine.gpus = True, [IGPU]
    assert C._select_llama_libs("vulkan") == "vulkan"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == vulkan and machine.preloaded == ["lib_vulkan"]


def test_asking_for_vulkan_without_any_gpu_ends_on_the_cpu_build(machine, caplog):
    all_builds(machine)
    machine.gpus = [SMALL]
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("vulkan") == "cpu"
    assert "no discrete or integrated Vulkan GPU with 3500 MiB" in caplog.text


def test_asking_for_cuda_never_tries_vulkan(machine):
    all_builds(machine)
    machine.gpus = [RX]
    assert C._select_llama_libs("cuda") == "cpu" and machine.preloaded == ["lib_cpu"]


def test_asking_for_the_cpu_takes_the_cpu_build_even_on_a_gpu_pc(machine):
    _cuda, _vulkan, cpu = all_builds(machine)
    machine.nvidia, machine.gpus = True, [RTX]
    assert C._select_llama_libs("cpu") == "cpu"
    assert os.environ["LLAMA_CPP_LIB_PATH"] == cpu and machine.preloaded == ["lib_cpu"]


def test_the_choice_is_made_once_per_process(machine):
    all_builds(machine)
    machine.nvidia = True
    assert C._select_llama_libs("auto") == "cuda"
    machine.nvidia = False
    assert C._select_llama_libs("cpu") == "cuda", "the DLLs are loaded: a later request cannot change them"
    assert machine.preloaded == ["lib"]


def test_nothing_usable_leaves_llama_cpp_to_its_default_and_says_how_to_fix_it(machine, caplog):
    machine.add("lib")                                           # a source install without the extra builds
    with caplog.at_level("WARNING"):
        assert C._select_llama_libs("auto") is None
    assert os.environ["LLAMA_CPP_LIB_PATH"] == "" and machine.preloaded == []
    assert "no llama.cpp build is usable here (cuda 13: not installed; cuda: no NVIDIA driver (nvcuda.dll); " \
           "vulkan: not installed; cpu: not installed)" in caplog.text
    assert "scripts/fetch_llama_backends.py" in caplog.text
    assert C._select_llama_libs("auto") is None and machine.preloaded == []      # remembered, not retried


def test_a_folder_the_user_set_is_left_alone(machine, monkeypatch, caplog):
    all_builds(machine)
    machine.nvidia = True
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", r"D:\my-llama")
    with caplog.at_level("INFO"):
        assert C._select_llama_libs("auto") is None
    assert os.environ["LLAMA_CPP_LIB_PATH"] == r"D:\my-llama" and machine.preloaded == []
    assert r"D:\my-llama (LLAMA_CPP_LIB_PATH was set outside VoxFlow)" in caplog.text


def test_nothing_is_chosen_off_windows_or_without_llama_cpp(machine, monkeypatch):
    all_builds(machine)
    monkeypatch.setattr(C.sys, "platform", "darwin")
    assert C._select_llama_libs("auto") is None
    monkeypatch.setattr(C, "_llama_choice", C._UNSET)
    monkeypatch.setattr(C.sys, "platform", "win32")
    monkeypatch.setattr(C, "_llama_pkg_dir", lambda: None)
    assert C._select_llama_libs("auto") is None and machine.preloaded == []


# =========================================================================== one build folder
def test_a_build_of_another_release_is_never_loaded(machine, caplog):
    """A stale folder (llama-cpp-python was upgraded, the extra builds were not) would crash in native code."""
    machine.add("lib")
    machine.add("lib_vulkan", "0.3.35")
    machine.add("lib_cpu", "0.3.35")
    machine.gpus = [RX]
    with caplog.at_level("WARNING"):
        assert C._select_llama_libs("auto") is None
    assert "vulkan: built for llama-cpp-python 0.3.35, installed is 0.3.36" in caplog.text and machine.preloaded == []


def test_folder_checks(machine):
    pkg = machine.pkg
    assert C._folder_unusable(str(pkg / "lib_cpu")) == "not installed"
    assert C._folder_unusable(machine.add("lib")) is None, "the wheel's own folder has no version file"
    cpu = machine.add("lib_cpu", "0.3.36")
    assert C._folder_unusable(cpu) is None
    machine.version = "0.4.0"
    assert C._folder_unusable(cpu) == "built for llama-cpp-python 0.3.36, installed is 0.4.0"
    machine.version = None                                       # no package metadata: nothing to compare with
    assert C._folder_unusable(cpu) is None


def test_the_package_folder_is_where_llama_cpp_is_installed(monkeypatch, tmp_path):
    real = importlib.util.find_spec
    spec = {"value": None}
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: spec["value"] if name == "llama_cpp" else real(name, *a))
    assert C._llama_pkg_dir() is None
    spec["value"] = types.SimpleNamespace(submodule_search_locations=None)
    assert C._llama_pkg_dir() is None
    spec["value"] = types.SimpleNamespace(submodule_search_locations=[str(tmp_path / "a"), str(tmp_path / "b")])
    assert C._llama_pkg_dir() == str(tmp_path / "a")


def test_preload_loads_the_folders_own_llama_dll(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda path: asked.append(path), raising=False)
    C._preload(str(tmp_path / "lib_cpu"))
    assert asked == [str(tmp_path / "lib_cpu" / "llama.dll")], "a full path: its dependencies come from that folder"


def test_machine_probes_of_the_backends(monkeypatch):
    monkeypatch.setattr(gpuprobe, "nvidia_driver", lambda: True)
    assert C._cuda_unusable(False) is None
    monkeypatch.setattr(gpuprobe, "nvidia_driver", lambda: False)
    assert C._cuda_unusable(True) == "no NVIDIA driver (nvcuda.dll)"
    monkeypatch.setattr(gpuprobe, "_vulkan_cache", [IGPU])
    assert C._vulkan_unusable(True) is None
    assert C._vulkan_unusable(False).startswith("no discrete Vulkan GPU with 3500 MiB of memory (devices: AMD Radeon")
    monkeypatch.setattr(gpuprobe, "_vulkan_cache", [RX])
    assert C._vulkan_unusable(False) is None


# =========================================================================== what the loaded library says
def test_gpu_offload_is_asked_from_the_loaded_library(monkeypatch):
    fake = types.ModuleType("llama_cpp")
    fake.llama_supports_gpu_offload = lambda: False
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    assert C._gpu_offload_supported() is False
    fake.llama_supports_gpu_offload = lambda: True
    assert C._gpu_offload_supported() is True
    monkeypatch.setitem(sys.modules, "llama_cpp", None)          # the import fails: the load attempt will tell why
    assert C._gpu_offload_supported() is True


# =========================================================================== Cleaner.load on each build
def loaded(monkeypatch, libs, device="auto", offload=True, fail_gpu=False, timeout_s=4.0):
    c = Cleaner(model="repo:file.gguf", device=device, timeout_s=timeout_s)
    built, asked = [], []
    monkeypatch.setattr(C, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(C, "_select_llama_libs", lambda device: asked.append(device) or libs)
    monkeypatch.setattr(C, "_gpu_offload_supported", lambda: offload)
    monkeypatch.setattr(c, "_resolve_path", lambda: "model.gguf")

    def build(path, gpu):
        built.append(gpu)
        if gpu and fail_gpu:
            raise RuntimeError("ErrorOutOfDeviceMemory")
        return FakeLLM("I think we should move the meeting to Friday at three thirty.")

    monkeypatch.setattr(c, "_build", build)
    c.load()
    return c, built, asked


def test_vulkan_build_loads_on_the_gpu_and_reports_vulkan(monkeypatch):
    c, built, asked = loaded(monkeypatch, "vulkan")
    assert asked == ["auto"] and built == [True]
    assert c.backend == "vulkan" and c.has_llm
    assert [call["messages"][-1]["content"] for call in c._llm.calls] == [
        "um so what is the capital of france", C._SPEED_CHECK], "warm-up, then one timed sentence"


def test_cuda_build_reports_cuda(monkeypatch):
    c, built, _ = loaded(monkeypatch, "cuda", device="cuda")
    assert built == [True] and c.backend == "cuda"


def test_a_failed_gpu_load_falls_back_to_the_cpu_not_to_the_rules(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        c, built, _ = loaded(monkeypatch, "vulkan", fail_gpu=True)
    assert built == [True, False]
    assert c.backend == "cpu" and c.has_llm, "the language model still runs"
    assert "cleanup load failed (gpu=True): ErrorOutOfDeviceMemory" in caplog.text
    assert len(c._llm.calls) == 1, "no speed check on the CPU: there is nothing left to fall back to"


def test_a_gpu_too_slow_for_dictation_gives_way_to_the_cpu(monkeypatch, caplog):
    """An integrated GPU can load the model and still be slower than the CPU: every dictation would time out."""
    with caplog.at_level("WARNING"):
        c, built, _ = loaded(monkeypatch, "vulkan", device="vulkan", timeout_s=0.0)   # any time is over the budget
    assert built == [True, False] and c.backend == "cpu" and c.has_llm
    assert "too slow for dictation" in caplog.text


def test_a_gpu_build_without_a_gpu_device_reports_the_cpu(monkeypatch):
    c, built, _ = loaded(monkeypatch, "vulkan", offload=False)
    assert built == [False] and c.backend == "cpu"


def test_cpu_build_never_attempts_the_gpu(monkeypatch):
    c, built, asked = loaded(monkeypatch, "cpu", offload=True)
    assert built == [False] and c.backend == "cpu" and c.has_llm
    c, built, asked = loaded(monkeypatch, "cpu", device="cpu")
    assert asked == ["cpu"] and built == [False] and c.backend == "cpu"


def test_no_model_at_all_still_ends_on_the_rule_pass(monkeypatch):
    c = Cleaner(model="repo:file.gguf")
    monkeypatch.setattr(C, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(C, "_select_llama_libs", lambda device: "cpu")
    monkeypatch.setattr(c, "_resolve_path", lambda: "model.gguf")
    monkeypatch.setattr(c, "_build", lambda path, gpu: (_ for _ in ()).throw(OSError("llama.dll is damaged")))
    c.load()
    assert c.backend == "rules" and c.ready and not c.has_llm, "what the app shows as basic clean-up"


# =========================================================================== the explicit override
@pytest.mark.parametrize("value,device", [("", ""), ("cpu", "cpu"), (" CUDA ", "cuda"), ("Vulkan", "vulkan"),
                                          ("WebGPU", "webgpu"), ("quantum", "")])
def test_voxflow_device_names_the_device(monkeypatch, value, device):
    monkeypatch.setenv("VOXFLOW_DEVICE", value)
    assert app.forced_device() == device


@pytest.mark.parametrize("value,speech,language", [("", "", ""), ("quantum", "", ""), ("cpu", "cpu", "cpu"),
                                                   ("cuda", "cuda", "cuda"), ("vulkan", "webgpu", "vulkan"),
                                                   ("webgpu", "webgpu", "vulkan")])
def test_voxflow_device_maps_to_one_backend_per_model(monkeypatch, value, speech, language):
    from voxflow import asr
    monkeypatch.setenv("VOXFLOW_DEVICE", value)
    assert app.forced_devices() == (speech, language)
    assert speech in ("",) + asr.DEVICES and language in ("",) + tuple(b.name for b in C.LLAMA_BACKENDS)


def test_the_app_passes_the_forced_device_to_both_models(monkeypatch):
    built = {}

    class Asr:
        def __init__(self, model, device):
            built["asr"] = device

    class Cln:
        def __init__(self, model, device, timeout_s):
            built["cleanup"] = device

    monkeypatch.setattr(app, "_real", lambda mod, attr: {"audio": object, "asr": Asr, "cleanup": Cln}[mod])
    monkeypatch.setenv("VOXFLOW_DEVICE", "vulkan")
    app.make_modules(Config(asr_device="cuda"))
    assert built == {"asr": "webgpu", "cleanup": "vulkan"}, "the override wins over the asr_device setting"
    monkeypatch.delenv("VOXFLOW_DEVICE")
    app.make_modules(Config(asr_device="cuda"))
    assert built == {"asr": "cuda", "cleanup": "auto"}


def test_the_status_text_names_the_vulkan_backend():
    from voxflow import controller
    assert controller._backend_text("vulkan") == "Running on the GPU (Vulkan)."
    assert controller._backend_text("webgpu") == "Running on the GPU (WebGPU)."
    assert controller._backend_text("cuda") == "Running on the GPU (CUDA)."
