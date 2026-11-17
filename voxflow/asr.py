"""Parakeet TDT speech recognition via onnx-asr on onnxruntime: NVIDIA (CUDA), any discrete GPU (WebGPU), or the CPU.

    Transcriber(model, device="auto")     device: "auto" | "cuda" | "webgpu" | "cpu"   ("gpu" means "auto")
    .load()                -> picks a back-end (ASR_BACKENDS, in order), builds the model and warms it up
    .backend               -> "cuda" | "webgpu" | "cpu": where the encoder (nearly all of the work) really runs
    .transcribe(audio)     -> text; a GPU back-end that fails during a dictation is replaced by the CPU
    time_backends(...)     -> measured speed of each back-end on this PC (for choosing by measurement, not by rule)

How the names of the ``VOXFLOW_DEVICE`` override map to these back-ends: ``voxflow.app.DEVICE_NAMES``.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import gpuprobe
from . import modelhub  # noqa: F401  (import side effect: Hugging Face telemetry is switched off)

log = logging.getLogger("voxflow.asr")

SAMPLE_RATE = 16000
MIN_SAMPLES = int(0.2 * SAMPLE_RATE)
_dll_dirs_added = False


def _add_nvidia_dll_dirs() -> None:
    """Make the DLLs of the pip CUDA 13 wheels (nvidia-cublas, nvidia-cudnn-cu13, ...) loadable; no system CUDA needed."""
    global _dll_dirs_added
    if _dll_dirs_added or sys.platform != "win32":
        return
    _dll_dirs_added = True
    import site
    roots = list(site.getsitepackages()) + list(sys.path)
    seen = set()
    for root in roots:
        nv = os.path.join(root, "nvidia")
        if nv in seen or not os.path.isdir(nv):
            continue
        seen.add(nv)
        for dirpath, _dirs, files in os.walk(nv):
            if any(f.lower().endswith(".dll") for f in files):
                try:
                    os.add_dll_directory(dirpath)
                except OSError:
                    pass
                os.environ["PATH"] = dirpath + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------- where the model runs
# onnxruntime-gpu carries the CUDA execution provider (NVIDIA only). Every other vendor's card is reached through
# the WebGPU execution provider: a plugin (the onnxruntime-ep-webgpu package: Dawn on Direct3D 12) that is
# registered into the same onnxruntime at run time, so one install serves NVIDIA, AMD and Intel cards.
MIN_GPU_MEMORY_MB = 3500      # the model took about 2.6 GB on the GPU through WebGPU: a 4 GB card is the smallest
_WEBGPU_PROVIDER = "WebGpuExecutionProvider"
_UNSET = object()
_webgpu_problem: Any = _UNSET     # None once the plugin is registered in this process, else why it could not be


@dataclass(frozen=True)
class Adapter:
    """One graphics adapter as the WebGPU plugin lists it."""
    name: str
    discrete: bool
    vram_mb: int         # dedicated video memory (0: unknown)
    index: str           # its place in the "high performance first" order ("" if not reported)
    ep_device: Any       # the onnxruntime OrtEpDevice a session is created for

    def __str__(self) -> str:
        return f"{self.name} ({'discrete' if self.discrete else 'integrated'}, {self.vram_mb} MiB)"


def _register_webgpu() -> str | None:
    """Register the WebGPU plugin with onnxruntime, once per process. None, or why it is not there."""
    global _webgpu_problem
    if _webgpu_problem is _UNSET:
        try:
            import onnxruntime as ort
            import onnxruntime_ep_webgpu as plugin

            ort.register_execution_provider_library("voxflow_webgpu", plugin.get_library_path())
            _webgpu_problem = None
        except Exception as e:  # noqa: BLE001 - not installed, an onnxruntime that is too old, a DLL that does not load
            _webgpu_problem = f"the WebGPU plugin did not load ({type(e).__name__}: {e})"
    return _webgpu_problem


def _megabytes(text: str) -> int:
    digits = "".join(ch for ch in str(text).split(" ")[0] if ch.isdigit())
    return int(digits) if digits else 0


def _webgpu_adapters() -> list[Adapter]:
    """The GPUs the registered plugin can run on (call ``_register_webgpu`` first)."""
    import onnxruntime as ort

    found = []
    for ep_device in ort.get_ep_devices():
        if ep_device.ep_name != _WEBGPU_PROVIDER:
            continue
        meta = dict(ep_device.device.metadata)
        found.append(Adapter(meta.get("Description", "GPU"), meta.get("Discrete") == "1",
                             _megabytes(meta.get("DxgiVideoMemory", "")), meta.get("DxgiHighPerformanceIndex", ""),
                             ep_device))
    return found


def _webgpu_adapter(asked_for: bool) -> Adapter | None:
    """The adapter WebGPU should use. Without a measurement the instant choice is a discrete card with enough
    memory: the one integrated GPU measured so far (AMD Radeon) took 1.6-4.6 s for a clip the CPU of the same PC
    does in 0.25 s. That is a first guess, not a rule: asking for "webgpu" by name takes whatever adapter there
    is, and ``time_backends`` measures it against the others."""
    adapters = _webgpu_adapters()
    if not asked_for:
        adapters = [a for a in adapters if a.discrete and a.vram_mb >= MIN_GPU_MEMORY_MB]
    return max(adapters, key=lambda a: (a.discrete, a.vram_mb), default=None)


def _cuda_unusable(asked_for: bool) -> str | None:
    if not gpuprobe.nvidia_driver():
        return "no NVIDIA driver (nvcuda.dll)"
    _add_nvidia_dll_dirs()
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return "CUDAExecutionProvider not available"
    return None


def _webgpu_unusable(asked_for: bool) -> str | None:
    problem = _register_webgpu()
    if problem is not None:
        return problem
    if _webgpu_adapter(asked_for) is not None:
        return None
    seen = ", ".join(str(a) for a in _webgpu_adapters()) or "none"
    wanted = "GPU" if asked_for else f"discrete GPU with {MIN_GPU_MEMORY_MB} MiB of memory"
    return f"no {wanted} for WebGPU (adapters: {seen})"


def _cpu_unusable(asked_for: bool) -> None:
    return None


def _session_options():
    """The onnxruntime SessionOptions every model session starts from (one place, so a check can switch
    profiling on and see which provider ran each node)."""
    import onnxruntime as ort

    return ort.SessionOptions()


def _decoder_on_cpu(model):
    """Move the small decoder/joint network of a GPU model to the CPU; the encoder stays on the GPU.

    The TDT decoder is called once per output token with tiny tensors, so on a GPU every call pays a round trip
    that costs more than the arithmetic. Measured on the 11.6 s test clip: CUDA 54-61 ms -> 29 ms, WebGPU 200 ms ->
    65 ms. onnx-asr has no option for this, so its (pinned) private attribute is replaced; when that attribute or
    the session's file path is not there, or the CPU session cannot be made, the model is used as it was built."""
    recogniser = getattr(model, "asr", None)
    path = getattr(getattr(recogniser, "_decoder_joint", None), "_model_path", None)
    if not path:
        log.info("the decoder stays on the GPU (this onnx-asr has no _decoder_joint session to replace)")
        return model
    import onnxruntime as ort

    try:
        recogniser._decoder_joint = ort.InferenceSession(path, sess_options=_session_options(),
                                                         providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - slower, still correct
        log.info("the decoder stays on the GPU (%s: %s)", type(e).__name__, e)
    return model


# The CUDA memory arena grows to the largest clip it has seen and keeps that memory for the whole session: measured
# with Parakeet on an RTX 5080, +1.1 GB after an 80 s clip, +2.8 GB after 160 s, +5.2 GB after 240 s on top of the
# 3.3 GB the model takes (QA C-06). On an 8 GB card that ends in an out-of-memory error, and the session moves to
# the CPU for good. Giving the memory back costs time, though: the next long clip allocates it again (measured
# +15-25 ms for 80 s, +40-60 ms for 160-240 s of audio). So a run on audio longer than SHRINK_AFTER_S hands the
# arena's free chunks back when it ends only while the GPU has less than SHRINK_BELOW_FREE_MB free: a card with
# room keeps its speed, a full one (a small card, a game) stays bounded.
SHRINK_AFTER_S = 30.0
SHRINK_BELOW_FREE_MB = 2048
_FEATURE_FRAMES_PER_S = 100        # the encoder input: one feature frame per 10 ms of audio


class _ShrinkingSession:
    """An onnxruntime session (the encoder) whose long runs shrink the CUDA arena when they end, while the GPU is
    short of memory. Everything else is the session's own."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def run(self, output_names, input_feed, run_options=None):
        signal = input_feed.get("audio_signal") if isinstance(input_feed, dict) else None
        frames = int(signal.shape[-1]) if hasattr(signal, "shape") else 0
        if run_options is None and frames > SHRINK_AFTER_S * _FEATURE_FRAMES_PER_S:
            free = gpuprobe.cuda_free_mb()
            if free is not None and free < SHRINK_BELOW_FREE_MB:
                import onnxruntime as ort

                log.info("long clip with %d MiB of GPU memory free: its working memory is given back", free)
                run_options = ort.RunOptions()
                run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        return self._session.run(output_names, input_feed, run_options)


def _shrink_long_runs(model):
    """Let the encoder of a CUDA model give back the memory a long clip needed (onnx-asr's pinned private
    attribute, like ``_decoder_on_cpu``; without it the model is used as it was built)."""
    recogniser = getattr(model, "asr", None)
    encoder = getattr(recogniser, "_encoder", None)
    if encoder is None or not hasattr(encoder, "run"):
        log.info("the encoder's GPU memory is not shrunk after long clips (this onnx-asr has no _encoder session)")
        return model
    recogniser._encoder = _ShrinkingSession(encoder)
    gpuprobe.cuda_free_mb()            # set the memory probe up now, during the load, not in a long dictation
    return model


def _build_cuda(name: str):
    import onnx_asr

    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC",
                                            # allocate what is asked, not the next power of two: ~0.3-0.8 GB less VRAM
                                            "arena_extend_strategy": "kSameAsRequested"}),
                 "CPUExecutionProvider"]
    model = onnx_asr.load_model(name, sess_options=_session_options(), providers=providers)
    return _shrink_long_runs(_decoder_on_cpu(model))


def _build_webgpu(name: str):
    import onnx_asr

    adapter = _webgpu_adapter(True)       # the probe passed: after a discrete card (auto) this picks the same one
    if adapter is None:
        raise RuntimeError("no adapter for WebGPU")
    options = _session_options()
    # The plugin only attaches this way: naming it in ``providers`` gives a session that silently runs on the CPU.
    # Dawn takes the high-performance adapter by itself (index 0); another one has to be named. One process can
    # hold one WebGPU adapter.
    options.add_provider_for_devices([adapter.ep_device],
                                     {} if adapter.index in ("", "0") else {"adapterIndex": adapter.index})
    log.info("WebGPU adapter: %s", adapter)
    # onnx-asr always passes ``providers``: the CPU for its preprocessor, these options for the model itself
    return _decoder_on_cpu(onnx_asr.load_model(name, providers=["CPUExecutionProvider"],
                                               asr_config={"sess_options": options}))


def _build_cpu(name: str):
    import onnx_asr

    return onnx_asr.load_model(name, sess_options=_session_options(), providers=["CPUExecutionProvider"])


@dataclass(frozen=True)
class AsrBackend:
    """One way to run the model. ``unusable(asked_for)`` is the cheap machine probe: None, or why this PC cannot
    use it (``asked_for``: the user named this back-end, it was not just next in line). ``build(model name)``
    returns the loaded onnx-asr model; ``provider`` is what its encoder session must then report."""
    name: str                                  # also what ``Transcriber.backend`` reports
    gpu: bool
    provider: str
    unusable: Callable[[bool], str | None]
    build: Callable[[str], Any]


# In order of preference. A new back-end (Apple later: CoreML or MLX) is one more line with its probe and builder.
ASR_BACKENDS = (
    AsrBackend("cuda", True, "CUDAExecutionProvider", _cuda_unusable, _build_cuda),
    AsrBackend("webgpu", True, _WEBGPU_PROVIDER, _webgpu_unusable, _build_webgpu),
    AsrBackend("cpu", False, "CPUExecutionProvider", _cpu_unusable, _build_cpu),
)
_CPU = ASR_BACKENDS[-1]
_GPU_NAMES = tuple(b.name for b in ASR_BACKENDS if b.gpu)
DEVICES = ("auto",) + tuple(b.name for b in ASR_BACKENDS)       # what the ``asr_device`` setting accepts


def _sessions(model) -> list:
    """Every onnxruntime session reachable from a loaded model."""
    found, seen, stack = [], set(), [model]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if hasattr(o, "get_providers"):
            found.append(o)
            continue
        for v in getattr(o, "__dict__", {}).values():
            if hasattr(v, "__dict__") or hasattr(v, "get_providers"):
                stack.append(v)
    return found


def _runs_on(model, provider: str) -> bool:
    """True when the encoder session (any session, if this onnx-asr names it differently) lists ``provider``:
    onnxruntime leaves a provider out of that list when it could not be created."""
    encoder = getattr(getattr(model, "asr", None), "_encoder", None)
    for session in ([encoder] if hasattr(encoder, "get_providers") else _sessions(model)):
        try:
            if provider in session.get_providers():
                return True
        except Exception:  # noqa: BLE001 - a session that cannot say is not proof
            pass
    return False


def _timing_clip(seconds: float = 5.0) -> np.ndarray:
    """A built-in clip for timing: a hum with a wobbling pitch and syllable-like loudness. It is not speech, so the
    decoder emits few tokens; the encoder, which is nearly all of the work and the part that differs between
    back-ends, costs the same as for speech of that length."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    pitch = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)
    voice = sum(np.sin(2 * np.pi * k * np.cumsum(pitch) / SAMPLE_RATE) / k for k in (1, 2, 3, 5))
    return (0.1 * voice * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def time_backends(model: str = "nemo-parakeet-tdt-0.6b-v3", names=None, runs: int = 3, audio=None) -> dict:
    """Measure how fast each back-end really is on this PC, so a ranking by measurement can replace the fixed order
    of ASR_BACKENDS (which only knows "discrete before integrated").

    ``names``: back-end names, default every row of ASR_BACKENDS. Each is treated as asked for by name, so an
    integrated GPU is a candidate for "webgpu". ``audio``: 16 kHz float32 samples, default a built-in 5 s clip.
    Returns ``{name: {"ok": True, "load_s": build + warm-up, "median_ms": ..., "runs_ms": [...]}}`` or
    ``{name: {"ok": False, "error": why}}``, in the order asked; nothing falls through to another back-end.
    Each candidate is built in turn and released again: this takes seconds and gigabytes, so it belongs on a
    background thread, never in the start-up path. Nothing is stored and no setting is changed here."""
    import time

    known = {b.name: b for b in ASR_BACKENDS}
    clip = _timing_clip() if audio is None else audio
    results = {}
    for name in ([b.name for b in ASR_BACKENDS] if names is None else names):
        tr = Transcriber(model, name)
        try:
            backend = known[name]
            why = backend.unusable(True)
            if why is not None:
                raise RuntimeError(why)
            started = time.perf_counter()
            tr._start(backend)
            load_s = time.perf_counter() - started
            took = []
            for _ in range(max(1, runs)):
                started = time.perf_counter()
                tr.transcribe(clip)
                took.append((time.perf_counter() - started) * 1000.0)
            if tr.backend != name:
                raise RuntimeError(f"inference failed on {name}")
            results[name] = {"ok": True, "load_s": round(load_s, 3), "median_ms": round(float(np.median(took)), 1),
                             "runs_ms": [round(x, 1) for x in took]}
        except Exception as e:  # noqa: BLE001 - a candidate that cannot run is a result, not a failure of the survey
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        tr._model = None
    return results


class Transcriber:
    def __init__(self, model: str = "nemo-parakeet-tdt-0.6b-v3", device: str = "auto"):
        self.model_name = model
        self.device = device
        self._model = None
        self._backend = "unloaded"
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        return self._backend

    def _build(self, backend: AsrBackend):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        return backend.build(self.model_name)

    def _warmup(self) -> None:
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        tone = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        self._model.recognize(tone, sample_rate=SAMPLE_RATE)

    def _start(self, backend: AsrBackend) -> None:
        """Build the model for ``backend``, run it once and make sure it runs where it was meant to."""
        self._model = self._build(backend)
        self._warmup()
        if not _runs_on(self._model, backend.provider):
            raise RuntimeError(f"the session does not run on {backend.provider}")
        self._backend = backend.name

    def load(self) -> None:
        """"auto" / "gpu" try the GPU back-ends of ASR_BACKENDS in order; a back-end name ("cuda", "webgpu") tries
        that one; anything that does not work out ends on the CPU, whose own failure is the caller's to report."""
        skipped, report = [], log.info
        for backend in ASR_BACKENDS:
            if not backend.gpu or self.device not in ("auto", "gpu", backend.name):
                continue
            try:
                why = backend.unusable(self.device == backend.name)
                if why is None:
                    self._start(backend)
            except Exception as e:  # noqa: BLE001 - a driver, a DLL, the model on that device: the next one is tried
                why, report = f"failed ({type(e).__name__}: {e})", log.warning
            if why is None:
                break
            self._model = None
            skipped.append(f"{backend.name}: {why}")
        else:
            self._start(_CPU)
        report("speech recognition runs on %s (device %s%s)", self._backend, self.device,
               "".join(f"; {s}" for s in skipped))

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("Transcriber.load() has not been called")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < MIN_SAMPLES:
            return ""
        if float(np.max(np.abs(audio))) < 1e-4:
            return ""
        pad = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([pad, audio, pad])
        with self._lock:
            try:
                res = self._model.recognize(audio, sample_rate=SAMPLE_RATE)
            except Exception as e:  # noqa: BLE001
                if self._backend not in _GPU_NAMES:
                    raise
                log.warning("%s inference failed (%s); switching to CPU", self._backend, e)
                self._model = self._build(_CPU)
                self._backend = "cpu"
                res = self._model.recognize(audio, sample_rate=SAMPLE_RATE)
        text = res if isinstance(res, str) else getattr(res, "text", str(res))
        return text.strip()
