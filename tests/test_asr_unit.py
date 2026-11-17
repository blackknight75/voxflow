"""Transcriber logic with a fake recognition model: no weights, no onnxruntime session, no GPU."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voxflow import asr
from voxflow.asr import MIN_SAMPLES, SAMPLE_RATE, Transcriber


class FakeModel:
    def __init__(self, result="hello world", fail_times=0, providers=("CPUExecutionProvider",)):
        self.result, self.fail_times, self.calls = result, fail_times, []
        self.session = types.SimpleNamespace(get_providers=lambda: list(providers))

    def recognize(self, audio, sample_rate=None, **kw):
        self.calls.append((np.array(audio, copy=True), sample_rate, kw))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("CUDA error: device lost")
        return self.result


def speech(seconds=1.0, amp=0.2):
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeEpDevice:
    """What onnxruntime.get_ep_devices() returns: an execution provider paired with a piece of hardware."""

    def __init__(self, ep_name="WebGpuExecutionProvider", **metadata):
        self.ep_name = ep_name
        self.device = types.SimpleNamespace(metadata=metadata)


def gpu_device(name="Radeon RX 7800 XT", discrete=True, memory="16000 MB", index="0"):
    return FakeEpDevice(Description=name, Discrete="1" if discrete else "0", DxgiVideoMemory=memory,
                        DxgiHighPerformanceIndex=index)


@pytest.fixture
def fake_ort(monkeypatch):
    """A stand-in onnxruntime (with the plugin API), WebGPU plugin package and onnx_asr: no GPU, no model."""
    ort = types.ModuleType("onnxruntime")
    ort.available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort.get_available_providers = lambda: list(ort.available)
    ort.set_default_logger_severity = lambda level: ort.severity.append(level)
    ort.severity, ort.registered, ort.ep_devices, ort.sessions = [], [], [], []
    ort.register_error = ort.session_error = None

    def register(name, path):
        if ort.register_error:
            raise ort.register_error
        ort.registered.append((name, path))

    class SessionOptions:
        def __init__(self):
            self.added = []

        def add_provider_for_devices(self, devices, options):
            self.added.append((list(devices), dict(options)))

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            if ort.session_error:
                raise ort.session_error
            self._model_path, self.sess_options, self.providers = path, sess_options, providers
            ort.sessions.append(self)

        def get_providers(self):
            return list(self.providers)

    ort.register_execution_provider_library = register
    ort.get_ep_devices = lambda: list(ort.ep_devices)
    ort.SessionOptions, ort.InferenceSession = SessionOptions, InferenceSession
    plugin = types.ModuleType("onnxruntime_ep_webgpu")
    plugin.get_library_path = lambda: "plugin/onnxruntime_providers_webgpu.dll"
    onnx_asr = types.ModuleType("onnx_asr")
    onnx_asr.loads = []

    def load_model(name, **kw):
        onnx_asr.loads.append((name, kw))
        decoder = types.SimpleNamespace(_model_path="decoder_joint-model.onnx", get_providers=lambda: ["GPU"])
        return types.SimpleNamespace(asr=types.SimpleNamespace(_decoder_joint=decoder))

    onnx_asr.load_model = load_model
    monkeypatch.setattr("voxflow.modelhub.model_folder", lambda name: None)     # never the real model cache
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", plugin)
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)
    monkeypatch.setattr(asr, "_add_nvidia_dll_dirs", lambda: None)
    monkeypatch.setattr(asr, "_webgpu_problem", asr._UNSET)
    monkeypatch.setattr(asr.gpuprobe, "nvidia_driver", lambda: True)
    ort.onnx_asr = onnx_asr
    return ort


PROVIDERS = {"cuda": "CUDAExecutionProvider", "webgpu": "WebGpuExecutionProvider", "cpu": "CPUExecutionProvider"}


@pytest.fixture
def machine(monkeypatch):
    """Replace the probes and builders of ASR_BACKENDS: ``pc.probe[name]`` is None (usable), a reason or an
    exception; ``pc.model[name]`` is the model its build returns, or an exception. Records what was asked."""
    import dataclasses

    pc = types.SimpleNamespace(probe={"cuda": None, "webgpu": None, "cpu": None}, asked=[], built=[],
                               model={n: FakeModel(providers=(p, "CPUExecutionProvider"))
                                      for n, p in PROVIDERS.items()})

    def probe(name):
        def unusable(asked_for):
            pc.asked.append((name, asked_for))
            if isinstance(pc.probe[name], Exception):
                raise pc.probe[name]
            return pc.probe[name]
        return unusable

    def builder(name):
        def build(model_name):
            pc.built.append(name)
            if isinstance(pc.model[name], Exception):
                raise pc.model[name]
            return pc.model[name]
        return build

    table = tuple(dataclasses.replace(b, unusable=probe(b.name), build=builder(b.name)) for b in asr.ASR_BACKENDS)
    monkeypatch.setattr(asr, "ASR_BACKENDS", table)
    monkeypatch.setattr(asr, "_CPU", table[-1])
    ort = types.ModuleType("onnxruntime")
    ort.set_default_logger_severity = lambda level: None
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return pc


def loaded(model, backend="cpu") -> Transcriber:
    tr = Transcriber()
    tr._model, tr._backend = model, backend
    return tr


# ---------------------------------------------------------------- transcribe
def test_transcribe_before_load_is_a_clear_error():
    tr = Transcriber()
    assert tr.backend == "unloaded"
    with pytest.raises(RuntimeError, match="load"):
        tr.transcribe(speech())


@pytest.mark.parametrize("audio", [
    np.zeros(0, dtype=np.float32),
    speech(0.1),                                           # shorter than MIN_SAMPLES
    np.zeros(SAMPLE_RATE, dtype=np.float32),               # digital silence
    np.full(SAMPLE_RATE, 5e-5, dtype=np.float32),          # below the noise gate
])
def test_silence_and_too_short_audio_never_reach_the_model(audio):
    m = FakeModel()
    assert loaded(m).transcribe(audio) == ""
    assert m.calls == []


def test_audio_is_flattened_cast_padded_and_passed_at_16k():
    m = FakeModel("  some text \n")
    audio = speech(1.0).astype(np.float64).reshape(-1, 1)   # wrong dtype and shape on purpose
    assert loaded(m).transcribe(audio) == "some text"
    sent, rate, _kw = m.calls[0]
    pad = int(0.15 * SAMPLE_RATE)
    assert rate == SAMPLE_RATE and sent.dtype == np.float32 and sent.ndim == 1
    assert sent.size == SAMPLE_RATE + 2 * pad
    assert not sent[:pad].any() and not sent[-pad:].any()
    assert np.allclose(sent[pad:-pad], speech(1.0), atol=1e-6)


def test_exactly_min_samples_is_accepted():
    m = FakeModel("ok")
    assert loaded(m).transcribe(speech(MIN_SAMPLES / SAMPLE_RATE)) == "ok"


def test_result_objects_with_a_text_attribute_are_supported():
    m = FakeModel(types.SimpleNamespace(text=" from object "))
    assert loaded(m).transcribe(speech()) == "from object"


def test_cpu_inference_errors_propagate_to_the_pipeline():
    m = FakeModel(fail_times=1)
    with pytest.raises(RuntimeError, match="device lost"):
        loaded(m, "cpu").transcribe(speech())


@pytest.mark.parametrize("backend", ["cuda", "webgpu"])
def test_gpu_inference_failure_switches_to_cpu_and_still_returns_text(monkeypatch, backend):
    gpu, cpu = FakeModel(fail_times=1), FakeModel("recovered on cpu")
    tr = loaded(gpu, backend)
    built = []
    monkeypatch.setattr(tr, "_build", lambda b: (built.append(b.name), cpu)[1])
    assert tr.transcribe(speech()) == "recovered on cpu"
    assert built == ["cpu"] and tr.backend == "cpu" and tr._model is cpu
    assert tr.transcribe(speech()) == "recovered on cpu"     # stays on CPU, no rebuild
    assert built == ["cpu"]


def test_transcribe_is_serialised_across_threads():
    import threading
    import time
    active, peak = [0], [0]

    class Slow(FakeModel):
        def recognize(self, audio, sample_rate=None, **kw):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            active[0] -= 1
            return "x"

    tr = loaded(Slow())
    threads = [threading.Thread(target=tr.transcribe, args=(speech(),)) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert peak[0] == 1


# ---------------------------------------------------------------- load / backend selection
def test_the_backends_are_tried_in_this_order():
    assert [(b.name, b.gpu, b.provider) for b in asr.ASR_BACKENDS] == [
        ("cuda", True, "CUDAExecutionProvider"), ("webgpu", True, "WebGpuExecutionProvider"),
        ("cpu", False, "CPUExecutionProvider")]
    assert asr.DEVICES == ("auto", "cuda", "webgpu", "cpu") and asr._GPU_NAMES == ("cuda", "webgpu")
    from voxflow.config import _CHOICES
    assert tuple(_CHOICES["asr_device"]) == asr.DEVICES, "the setting accepts exactly what the Transcriber knows"


def test_load_prefers_cuda_and_warms_up(machine):
    tr = Transcriber(device="auto")
    tr.load()
    assert machine.built == ["cuda"] and tr.backend == "cuda" and machine.asked == [("cuda", False)]
    assert len(machine.model["cuda"].calls) == 1, "load() must run one warm-up inference so the first dictation is fast"


def test_without_an_nvidia_card_a_discrete_gpu_is_used_through_webgpu(machine, caplog):
    machine.probe["cuda"] = "no NVIDIA driver (nvcuda.dll)"
    tr = Transcriber(device="auto")
    with caplog.at_level("INFO", logger="voxflow.asr"):
        tr.load()
    assert machine.built == ["webgpu"] and tr.backend == "webgpu"
    assert machine.asked == [("cuda", False), ("webgpu", False)]
    record = caplog.records[-1]
    assert record.levelname == "INFO", "a PC without an NVIDIA card is normal, not a warning"
    assert "runs on webgpu" in record.getMessage() and "cuda: no NVIDIA driver" in record.getMessage()


def test_without_any_usable_gpu_the_cpu_is_used(machine):
    machine.probe.update(cuda="no NVIDIA driver (nvcuda.dll)", webgpu="no discrete GPU")
    tr = Transcriber(device="auto")
    tr.load()
    assert machine.built == ["cpu"] and tr.backend == "cpu"


def test_a_gpu_backend_that_fails_to_build_or_warm_up_falls_through_to_the_next(machine, caplog):
    machine.model["cuda"] = OSError("cudnn64_9.dll not found")                       # the build raises
    machine.model["webgpu"] = FakeModel(fail_times=1, providers=("WebGpuExecutionProvider",))   # the warm-up raises
    tr = Transcriber(device="gpu")
    with caplog.at_level("INFO", logger="voxflow.asr"):
        tr.load()
    assert machine.built == ["cuda", "webgpu", "cpu"] and tr.backend == "cpu"
    assert tr.transcribe(speech()) == "hello world"
    record = caplog.records[-1]
    assert record.levelname == "WARNING" and "cudnn64_9.dll not found" in record.getMessage()
    assert "webgpu: failed (RuntimeError" in record.getMessage()


def test_a_probe_that_raises_counts_as_unusable(machine):
    machine.probe["cuda"] = RuntimeError("onnxruntime is broken")
    tr = Transcriber(device="auto")
    tr.load()
    assert machine.built == ["webgpu"] and tr.backend == "webgpu"


@pytest.mark.parametrize("name", ["cuda", "webgpu"])
def test_load_detects_a_session_that_silently_runs_on_cpu(machine, name):
    machine.model[name] = FakeModel()                                 # reports the CPU provider only
    tr = Transcriber(device=name)
    tr.load()
    assert machine.built == [name, "cpu"] and tr.backend == "cpu"


def test_a_named_gpu_backend_is_the_only_gpu_tried(machine):
    tr = Transcriber(device="webgpu")
    tr.load()
    assert machine.built == ["webgpu"] and tr.backend == "webgpu"
    assert machine.asked == [("webgpu", True)], "asked for by name: the probe may accept an integrated GPU"
    machine.built.clear()
    machine.probe["cuda"] = "no NVIDIA driver (nvcuda.dll)"
    tr = Transcriber(device="cuda")
    tr.load()
    assert machine.built == ["cpu"] and tr.backend == "cpu", "cuda was named: webgpu is not a substitute"


@pytest.mark.parametrize("device", ["cpu", "quantum"])
def test_forced_cpu_or_an_unknown_device_never_tries_the_gpu(machine, device):
    tr = Transcriber(device=device)
    tr.load()
    assert machine.built == ["cpu"] and tr.backend == "cpu" and machine.asked == []


def test_load_failure_on_cpu_propagates_so_the_app_can_report_it(machine):
    machine.model["cpu"] = OSError("model files missing")
    tr = Transcriber(device="cpu")
    with pytest.raises(OSError, match="model files missing"):
        tr.load()
    assert tr.backend == "unloaded"


def test_build_quietens_onnxruntime_and_calls_the_backend(fake_ort):
    built = []
    backend = asr.AsrBackend("x", False, "P", lambda asked: None, lambda name: built.append(name) or "model")
    assert Transcriber("my-model")._build(backend) == "model"
    assert built == ["my-model"] and fake_ort.severity == [3]


# ---------------------------------------------------------------- the probes
def test_cuda_needs_the_nvidia_driver_and_the_cuda_provider(fake_ort, monkeypatch):
    assert asr._cuda_unusable(False) is None
    fake_ort.available = ["CPUExecutionProvider"]
    assert asr._cuda_unusable(False) == "CUDAExecutionProvider not available"
    monkeypatch.setattr(asr.gpuprobe, "nvidia_driver", lambda: False)
    assert "no NVIDIA driver" in asr._cuda_unusable(True)


def test_the_cpu_is_always_usable():
    assert asr._cpu_unusable(False) is None


def test_the_webgpu_plugin_is_registered_once_per_process(fake_ort):
    assert asr._register_webgpu() is None and asr._register_webgpu() is None
    assert fake_ort.registered == [("voxflow_webgpu", "plugin/onnxruntime_providers_webgpu.dll")]


def test_a_plugin_that_does_not_register_is_remembered_as_the_reason(fake_ort):
    fake_ort.register_error = RuntimeError("ORT runtime version is below the minimum")
    why = asr._webgpu_unusable(False)
    assert "the WebGPU plugin did not load (RuntimeError: ORT runtime version" in why
    fake_ort.register_error = None
    assert asr._webgpu_unusable(True) == why and fake_ort.registered == [], "not retried in this process"


def test_adapters_are_read_from_the_plugins_devices_only(fake_ort):
    fake_ort.ep_devices = [FakeEpDevice("CPUExecutionProvider", Description="Ryzen"),
                           gpu_device("GeForce RTX 5080", True, "15977 MB", "0"),
                           gpu_device("Radeon(TM) Graphics", False, "485 MB", "1"),
                           FakeEpDevice()]                                   # an adapter that describes nothing
    rtx, radeon, bare = asr._webgpu_adapters()
    assert (rtx.name, rtx.discrete, rtx.vram_mb, rtx.index) == ("GeForce RTX 5080", True, 15977, "0")
    assert rtx.ep_device is fake_ort.ep_devices[1]
    assert str(rtx) == "GeForce RTX 5080 (discrete, 15977 MiB)"
    assert str(radeon) == "Radeon(TM) Graphics (integrated, 485 MiB)"
    assert (bare.name, bare.discrete, bare.vram_mb, bare.index) == ("GPU", False, 0, "")


@pytest.mark.parametrize("text,mb", [("15977 MB", 15977), ("485", 485), ("", 0), ("unknown", 0)])
def test_video_memory_is_parsed_leniently(text, mb):
    assert asr._megabytes(text) == mb


def test_an_integrated_or_too_small_gpu_is_not_picked_by_itself(fake_ort):
    fake_ort.ep_devices = [gpu_device("Radeon(TM) Graphics", False, "8000 MB", "0"),
                           gpu_device("GeForce GT 1030", True, "2000 MB", "1")]
    assert asr._webgpu_adapter(False) is None
    why = asr._webgpu_unusable(False)
    assert why == ("no discrete GPU with 3500 MiB of memory for WebGPU (adapters: Radeon(TM) Graphics (integrated, "
                   "8000 MiB), GeForce GT 1030 (discrete, 2000 MiB))")
    assert asr._webgpu_unusable(True) is None, "asked for by name: any adapter will do"
    assert asr._webgpu_adapter(True).name == "GeForce GT 1030", "a discrete card before an integrated GPU"


def test_the_discrete_card_with_the_most_memory_is_picked(fake_ort):
    fake_ort.ep_devices = [gpu_device("Arc A380", True, "6000 MB", "1"), gpu_device("Radeon RX 7800 XT", True),
                           gpu_device("Radeon(TM) Graphics", False, "32000 MB", "2")]
    assert asr._webgpu_unusable(False) is None
    assert asr._webgpu_adapter(False).name == asr._webgpu_adapter(True).name == "Radeon RX 7800 XT"


def test_a_pc_without_any_adapter_cannot_use_webgpu_even_when_asked(fake_ort):
    assert asr._webgpu_unusable(True) == "no GPU for WebGPU (adapters: none)"


# ---------------------------------------------------------------- the builders
def test_session_options_come_from_onnxruntime(fake_ort):
    assert isinstance(asr._session_options(), fake_ort.SessionOptions)


def test_the_cpu_build_names_the_cpu_provider_only(fake_ort):
    model = asr._build_cpu("my-model")
    (name, kw), = fake_ort.onnx_asr.loads
    assert name == "my-model" and kw["providers"] == ["CPUExecutionProvider"]
    assert isinstance(kw["sess_options"], fake_ort.SessionOptions)
    assert fake_ort.sessions == [], "nothing is replaced in a CPU model"
    assert model.asr._decoder_joint.get_providers() == ["GPU"]


def test_the_cuda_build_keeps_the_encoder_on_the_gpu_and_moves_the_decoder_to_the_cpu(fake_ort):
    model = asr._build_cuda("my-model")
    (name, kw), = fake_ort.onnx_asr.loads
    names = [p if isinstance(p, str) else p[0] for p in kw["providers"]]
    assert name == "my-model" and names == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    decoder, = fake_ort.sessions
    assert model.asr._decoder_joint is decoder
    assert decoder._model_path == "decoder_joint-model.onnx" and decoder.providers == ["CPUExecutionProvider"]


@pytest.mark.parametrize("index,options", [("0", {}), ("", {}), ("1", {"adapterIndex": "1"})])
def test_the_webgpu_build_attaches_the_plugin_through_session_options(fake_ort, index, options):
    fake_ort.ep_devices = [gpu_device(index=index)]
    model = asr._build_webgpu("my-model")
    (name, kw), = fake_ort.onnx_asr.loads
    assert name == "my-model"
    assert kw["providers"] == ["CPUExecutionProvider"], "naming the plugin here would silently give a CPU session"
    assert kw["asr_config"]["sess_options"].added == [([fake_ort.ep_devices[0]], options)]
    assert model.asr._decoder_joint.providers == ["CPUExecutionProvider"]


def test_the_webgpu_build_fails_cleanly_when_the_adapter_is_gone(fake_ort):
    with pytest.raises(RuntimeError, match="no adapter"):
        asr._build_webgpu("my-model")
    assert fake_ort.onnx_asr.loads == []


def test_the_decoder_is_left_alone_when_this_onnx_asr_has_no_such_session(fake_ort):
    no_attribute = types.SimpleNamespace(asr=types.SimpleNamespace())
    no_path = types.SimpleNamespace(asr=types.SimpleNamespace(_decoder_joint=types.SimpleNamespace()))
    no_recogniser = types.SimpleNamespace()
    for model in (no_attribute, no_path, no_recogniser):
        assert asr._decoder_on_cpu(model) is model
    assert fake_ort.sessions == [] and not hasattr(no_attribute.asr, "_decoder_joint")


def test_the_decoder_is_left_alone_when_the_cpu_session_cannot_be_made(fake_ort):
    original = types.SimpleNamespace(_model_path="decoder_joint-model.onnx")
    model = types.SimpleNamespace(asr=types.SimpleNamespace(_decoder_joint=original))
    fake_ort.session_error = OSError("file is gone")
    assert asr._decoder_on_cpu(model) is model and model.asr._decoder_joint is original


# ---------------------------------------------------------------- "does it really run there?"
@pytest.mark.parametrize("providers,expected", [(("CUDAExecutionProvider", "CPUExecutionProvider"), True),
                                                 (("CPUExecutionProvider",), False)])
def test_runs_on_asks_the_encoder_session(providers, expected):
    decoder = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])     # not what counts
    model = types.SimpleNamespace(asr=types.SimpleNamespace(
        _encoder=types.SimpleNamespace(get_providers=lambda: list(providers)), _decoder_joint=decoder))
    assert asr._runs_on(model, "CUDAExecutionProvider") is expected


def test_runs_on_finds_nested_sessions_when_the_encoder_has_another_name():
    def broken():
        raise RuntimeError("session closed")
    holder = types.SimpleNamespace(asr=types.SimpleNamespace(
        _encoder="not a session", other=types.SimpleNamespace(get_providers=broken),
        enc=types.SimpleNamespace(get_providers=lambda: ["WebGpuExecutionProvider", "CPUExecutionProvider"])))
    assert asr._runs_on(holder, "WebGpuExecutionProvider") is True
    assert asr._runs_on(holder, "CUDAExecutionProvider") is False
    assert len(asr._sessions(holder)) == 2


def test_runs_on_handles_cycles_and_objects_without_sessions():
    a = types.SimpleNamespace()
    a.self_ref = a
    assert asr._runs_on(a, "CUDAExecutionProvider") is False
    assert asr._runs_on(object(), "CUDAExecutionProvider") is False


# ---------------------------------------------------------------- CUDA wheel DLL folders (review: asr.py coverage hole)
def test_nvidia_dll_dirs_are_registered_once_and_put_on_path(tmp_path, monkeypatch):
    nv = tmp_path / "nvidia" / "cublas" / "bin"
    nv.mkdir(parents=True)
    (nv / "cublas64_13.dll").write_bytes(b"")
    (tmp_path / "nvidia" / "empty").mkdir()                       # a folder without DLLs is ignored
    added = []
    monkeypatch.setattr(asr, "_dll_dirs_added", False)
    monkeypatch.setattr(asr.sys, "platform", "win32")
    monkeypatch.setattr(asr.os, "add_dll_directory", lambda p: added.append(p), raising=False)
    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(asr.sys, "path", [str(tmp_path)])
    monkeypatch.setenv("PATH", "X")
    asr._add_nvidia_dll_dirs()
    assert added == [str(nv)]
    assert str(nv) in asr.os.environ["PATH"].split(asr.os.pathsep)
    asr._add_nvidia_dll_dirs()                                   # second call is a no-op
    assert added == [str(nv)]


def test_nvidia_dll_dirs_survive_a_refusing_add_dll_directory(tmp_path, monkeypatch):
    nv = tmp_path / "nvidia" / "x"
    nv.mkdir(parents=True)
    (nv / "a.dll").write_bytes(b"")

    def refuse(_p):
        raise OSError("no")
    monkeypatch.setattr(asr, "_dll_dirs_added", False)
    monkeypatch.setattr(asr.sys, "platform", "win32")
    monkeypatch.setattr(asr.os, "add_dll_directory", refuse, raising=False)
    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(asr.sys, "path", [])
    asr._add_nvidia_dll_dirs()                                   # must not raise


# ---------------------------------------------------------------- measuring instead of assuming
def test_the_timing_clip_is_audible_and_five_seconds_long():
    clip = asr._timing_clip()
    assert clip.dtype == np.float32 and clip.size == 5 * SAMPLE_RATE
    assert 0.05 < float(np.max(np.abs(clip))) <= 1.0, "loud enough to pass the silence gate, no clipping"
    assert asr._timing_clip(1.0).size == SAMPLE_RATE


def test_time_backends_measures_every_backend_as_if_asked_for_by_name(machine):
    results = asr.time_backends("my-model", runs=4)
    assert list(results) == ["cuda", "webgpu", "cpu"] and machine.built == ["cuda", "webgpu", "cpu"]
    assert machine.asked == [("cuda", True), ("webgpu", True), ("cpu", True)], "an integrated GPU is a candidate"
    for name, r in results.items():
        assert r["ok"] and len(r["runs_ms"]) == 4 and r["median_ms"] >= 0 and r["load_s"] >= 0
        calls = machine.model[name].calls
        assert len(calls) == 5, "one warm-up, then the timed runs"
        assert calls[-1][0].size == 5 * SAMPLE_RATE + 2 * int(0.15 * SAMPLE_RATE), "the built-in clip"


def test_time_backends_takes_a_list_of_candidates_and_a_clip_and_runs_at_least_once(machine):
    results = asr.time_backends(names=["cpu"], runs=0, audio=speech(1.0))
    assert list(results) == ["cpu"] and len(results["cpu"]["runs_ms"]) == 1
    assert machine.model["cpu"].calls[-1][0].size == SAMPLE_RATE + 2 * int(0.15 * SAMPLE_RATE)


def test_time_backends_reports_a_candidate_that_cannot_run_without_falling_through(machine):
    class DiesLater(FakeModel):
        def recognize(self, audio, sample_rate=None, **kw):
            if self.calls:                                   # the warm-up works, the first real run does not
                raise RuntimeError("device lost")
            return super().recognize(audio, sample_rate=sample_rate, **kw)

    machine.probe["cuda"] = "no NVIDIA driver (nvcuda.dll)"
    machine.model["webgpu"] = DiesLater(providers=("WebGpuExecutionProvider",))
    results = asr.time_backends(names=["cuda", "webgpu", "metal"])
    assert results["cuda"] == {"ok": False, "error": "RuntimeError: no NVIDIA driver (nvcuda.dll)"}
    assert results["webgpu"] == {"ok": False, "error": "RuntimeError: inference failed on webgpu"}
    assert results["metal"]["ok"] is False and "KeyError" in results["metal"]["error"]
    assert machine.built == ["webgpu", "cpu"], "cuda was never built; webgpu's failure is not timed as a CPU run"
