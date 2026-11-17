"""The speed check: measure where each model runs fastest on this PC, once, and let "Automatic" use that.

The fixed order of ``voxflow.asr.ASR_BACKENDS`` and ``voxflow.cleanup.LLAMA_BACKENDS`` is the instant choice
("static"): it needs no measurement, so the app always starts on it. Once both models are ready and VoxFlow is
idle, the controller runs this check in the background:

* every back-end of both models is tried in a CHILD process of its own (``python -m voxflow --speed-check ...``
  or the frozen exe with the same hidden flag), one after the other, at below-normal priority. A child keeps the
  app's own WebGPU adapter and llama.cpp build unpinned (each is chosen once per process), frees its video
  memory when it exits, and a driver crash in it cannot take the app down. Each child has a time limit, writes
  its result as JSON to a file it is given (and prints it), and anything it returns is checked before use;
* each child asks the back-end's probe first, as if the user had named it, so an integrated GPU is a candidate
  too; a back-end that passes is timed: speech on a built-in 5 s clip (``asr.time_backends``), the language
  model on short real clean-ups (its language-model pass);
* the fastest back-end by median replaces the static choice only when it is clearly faster: at least
  ``MIN_GAIN`` (15 %) AND ``MIN_SAVING_MS`` (10 ms) quicker than the static choice measured in the same run.
  When the static choice itself could not be measured, nothing changes. Noise can never make VoxFlow slower;
* the result is stored in ``speedcheck.json`` in the profile folder: a file of this PC, never part of a backup,
  a pack or the sync folder. It is keyed by a fingerprint of the hardware and software (``machine()``: GPU
  names and driver versions, CPU, VoxFlow and the pinned onnxruntime / onnx-asr / llama-cpp-python versions,
  plus the two model names) and only measured again when that changes or the user asks.

How the choice is applied, and what wins over it, is the controller's and ``voxflow.app.make_modules``' business:
``VOXFLOW_DEVICE`` beats a device named in Settings, which beats the measured choice, which beats the static one.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import __version__, gpuprobe

log = logging.getLogger("voxflow.speedcheck")

CACHE_NAME = "speedcheck.json"
FORMAT = 1
MODELS = ("asr", "cleanup")
MIN_GAIN = 0.15                 # the winner must be at least 15 % faster than the static choice ...
MIN_SAVING_MS = 10.0            # ... and at least 10 ms faster
RUNS = 3
CHILD_TIMEOUT_S = {"asr": 180.0, "cleanup": 300.0}     # model load + a first GPU start's kernel compile + 3 runs
_PACKAGES = ("onnxruntime-gpu", "onnxruntime", "onnxruntime-ep-webgpu", "onnx-asr", "llama-cpp-python")
# short real clean-ups, each different so no run reuses the previous one's prompt cache
SENTENCES = ("i think we should uh move the meeting to friday at three thirty",
             "can you send me the the report before lunch no wait before the end of the day",
             "so um the new build is ready and i will share the link with the team tomorrow")
_BELOW_NORMAL_PRIORITY, _NO_WINDOW = 0x00004000, 0x08000000


# the names of ASR_BACKENDS / LLAMA_BACKENDS in their order (a test keeps them equal), written out so the app
# can show and choose devices without importing the model modules
BACKENDS = {"asr": ("cuda", "webgpu", "cpu"), "cleanup": ("cuda", "vulkan", "cpu")}


def backends(model: str) -> tuple[str, ...]:
    """The back-end names of ``model`` ("asr" | "cleanup"), in the order of the static choice."""
    return BACKENDS[model]


# --------------------------------------------------------------------------- what the result is keyed by
def _dist_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - not installed (or no metadata in a frozen build)
        return ""


def machine() -> dict:
    """What a measurement depends on. Registry and package metadata only: nothing is loaded or imported."""
    return {"app": __version__, "cpu": gpuprobe.cpu_name(),
            "gpus": [list(a) for a in gpuprobe.display_adapters()],
            "packages": {name: _dist_version(name) for name in _PACKAGES}}


def fingerprint(info: dict, models: dict) -> str:
    """A short stable key for ``machine()`` plus the two model names."""
    text = json.dumps({"machine": info, "models": models}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------- choosing
def _median(result) -> float | None:
    if not isinstance(result, dict) or result.get("ok") is not True:
        return None
    ms = result.get("median_ms")
    return float(ms) if isinstance(ms, (int, float)) and not isinstance(ms, bool) and math.isfinite(ms) else None


def choose(results: dict, order, min_gain: float = MIN_GAIN, min_ms: float = MIN_SAVING_MS):
    """``(static, choice)``. ``static``: the first back-end in ``order`` that passes the automatic probe (None if
    none is known to). ``choice``: the measured winner when it is clearly faster than ``static``, else None."""
    order = list(order)
    static = next((n for n in order if isinstance(results.get(n), dict) and results[n].get("auto") is True), None)
    times = {n: _median(results.get(n)) for n in order}
    times = {n: ms for n, ms in times.items() if ms is not None}
    if static not in times:
        return static, None
    best = min(times, key=lambda n: (times[n], order.index(n)))
    base = times[static]
    if best != static and times[best] <= base * (1.0 - min_gain) and base - times[best] >= min_ms:
        return static, best
    return static, None


def fastest(results: dict, order) -> str | None:
    """The back-end with the lowest measured median (None if nothing was measured)."""
    order = list(order)
    times = {n: _median(results.get(n)) for n in order}
    times = {n: ms for n, ms in times.items() if ms is not None}
    return min(times, key=lambda n: (times[n], order.index(n))) if times else None


def make_record(key: str, info: dict, models: dict, results: dict, took_s: float) -> dict:
    record = {"format": FORMAT, "fingerprint": key, "machine": info, "models": models,
              "measured_at": time.time(), "took_s": round(float(took_s), 1),
              "results": {}, "static": {}, "choice": {}}
    for model in MODELS:
        got = results.get(model) if isinstance(results.get(model), dict) else {}
        record["results"][model] = {n: clean_result(got.get(n)) for n in backends(model) if n in got}
        record["static"][model], record["choice"][model] = choose(record["results"][model], backends(model))
    return record


def describe(record: dict) -> list[str]:
    """The measured table, one line per model (for the log)."""
    lines = []
    for model, label in (("asr", "speech"), ("cleanup", "language model")):
        cells = []
        for name, r in record["results"].get(model, {}).items():
            ms = _median(r)
            cells.append(f"{name} {ms:.0f} ms" if ms is not None else
                         f"{name} unavailable ({r.get('reason', '')})" if not r.get("available") else
                         f"{name} failed ({r.get('error', '')})")
        static, choice = record["static"].get(model), record["choice"].get(model)
        verdict = f"switch to {choice}" if choice else f"keep the automatic choice ({static or 'unknown'})"
        lines.append(f"{label}: {', '.join(cells) or 'nothing measured'} -> {verdict}")
    return lines


# --------------------------------------------------------------------------- the stored result
def cache_path(data_dir) -> Path:
    return Path(data_dir) / CACHE_NAME


def load(data_dir) -> dict | None:
    """The stored record, or None when there is none or it is unreadable / not one of ours."""
    path = cache_path(data_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("speed check result %s is unreadable (%s); it will be measured again", path, exc)
        return None
    ok = (isinstance(data, dict) and data.get("format") == FORMAT and isinstance(data.get("fingerprint"), str)
          and all(isinstance(data.get(k), dict) for k in ("results", "static", "choice")))
    if not ok:
        log.warning("speed check result %s is not in the expected format; it will be measured again", path)
        return None
    for model in MODELS:
        results = data["results"].get(model)
        results = results if isinstance(results, dict) else {}
        names = backends(model)
        data["results"][model] = {n: clean_result(results[n]) for n in names if n in results}
        choice, static = data["choice"].get(model), data["static"].get(model)
        data["choice"][model] = choice if choice in names else None
        data["static"][model] = static if static in names else None
    return data


def save(data_dir, record: dict) -> None:
    from .config import _atomic_write

    _atomic_write(cache_path(data_dir), json.dumps(record, indent=2))


def clean_result(raw) -> dict:
    """One child's answer in a known shape, whatever it sent: ``available`` / ``auto`` (bool), ``reason``,
    ``ok`` (bool), ``median_ms`` / ``load_s`` (float or None), ``runs_ms`` (list), ``error``."""
    raw = raw if isinstance(raw, dict) else {}

    def number(v):
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 \
            else None

    out = {"available": raw.get("available") is True, "auto": raw.get("auto") is True,
           "reason": str(raw.get("reason") or "")[:300], "error": str(raw.get("error") or "")[:300],
           "median_ms": number(raw.get("median_ms")), "load_s": number(raw.get("load_s")),
           "runs_ms": [number(v) for v in raw.get("runs_ms", []) if number(v) is not None]
           if isinstance(raw.get("runs_ms"), list) else []}
    out["ok"] = raw.get("ok") is True and out["available"] and out["median_ms"] is not None
    if raw.get("ok") is True and not out["ok"] and not out["error"]:
        out["error"] = "no usable timing"
    return out


def short_reason(text: str) -> str:
    """A few words for the device menu from a probe's technical reason."""
    low = (text or "").lower()
    if "nvidia driver" in low:
        return "No NVIDIA driver"
    if "not installed" in low or "did not load" in low:
        return "Not installed"
    if "gpu" in low or "adapter" in low or "device" in low:
        return "No suitable GPU"
    return "Not available here"


# --------------------------------------------------------------------------- the child process
def child_command(model: str, backend: str, out_path: str, model_name: str, budget_s: float) -> list[str]:
    """How to start a measuring child: the frozen exe takes the flag itself, a source install goes through -m.
    ``budget_s``: the clean-up time limit of a dictation (``cleanup_timeout_s``), which the child's cleaner uses."""
    args = ["--speed-check", model, backend, out_path, model_name or "default", f"{float(budget_s):g}"]
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "voxflow", *args]


def _child_env() -> dict:
    env = dict(os.environ)
    env.pop("VOXFLOW_DEVICE", None)             # the child is told exactly what to run
    env.pop("LLAMA_CPP_LIB_PATH", None)         # set by this app's own cleaner: the child chooses its own build
    env["HF_HUB_OFFLINE"] = "1"                 # the models are already cached: never reach the network
    return env


class Cancelled(Exception):
    """The measurement was stopped (a dictation started, or the app is closing)."""


def run_child(model: str, backend: str, model_name: str, budget_s: float, cancel: threading.Event,
              popen=subprocess.Popen, poll_s: float = 0.1, timeout_s: float | None = None) -> dict:
    """Run one measuring child and return its checked result. Raises ``Cancelled`` when ``cancel`` is set
    (the child is killed at once, which frees whatever it held). ``timeout_s``: default ``CHILD_TIMEOUT_S``."""
    timeout_s = CHILD_TIMEOUT_S[model] if timeout_s is None else timeout_s
    fd, out_path = tempfile.mkstemp(prefix="voxflow-speed-", suffix=".json")
    os.close(fd)
    started = time.monotonic()
    try:
        cmd = child_command(model, backend, out_path, model_name, budget_s)
        flags = (_BELOW_NORMAL_PRIORITY | _NO_WINDOW) if sys.platform == "win32" else 0
        try:
            proc = popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=_child_env(), cwd=str(Path(__file__).resolve().parent.parent), creationflags=flags)
        except OSError as exc:
            return clean_result({"error": f"could not start the measurement ({exc})"})
        while proc.poll() is None:
            if cancel.is_set():
                proc.kill()
                proc.wait()
                raise Cancelled()
            if time.monotonic() - started > timeout_s:
                proc.kill()
                proc.wait()
                return clean_result({"error": f"timed out after {timeout_s:.0f} s"})
            cancel.wait(poll_s)
        try:
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            return clean_result({"error": f"the measurement ended without a result (exit code {proc.returncode})"})
        return clean_result(data)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _measure_asr(backend: str, model_name: str) -> dict:
    from . import asr

    known = {b.name: b for b in asr.ASR_BACKENDS}
    b = known[backend]
    named, auto = b.unusable(True), b.unusable(False)
    if named is not None:
        return {"available": False, "auto": False, "reason": named}
    r = asr.time_backends(model_name or "nemo-parakeet-tdt-0.6b-v3", [backend], runs=RUNS)[backend]
    return dict(r, available=True, auto=auto is None, reason=auto or "")


def _measure_cleanup(backend: str, model_name: str, timeout_s: float) -> dict:
    from . import cleanup

    pkg = cleanup._llama_pkg_dir()

    def why(b, asked_for: bool):
        folder = cleanup._folder_unusable(os.path.join(pkg, b.folder)) if pkg else "llama-cpp-python is not installed"
        return folder or b.unusable(asked_for)

    # several builds can share a name ("cuda": CUDA 13 for RTX 50xx, else CUDA 12): usable when one of them is, and
    # the load below picks among them as the app does; the reason given is that of the last, most general one
    rows = [b for b in cleanup.LLAMA_BACKENDS if b.name == backend]
    named, auto = (next((None for b in rows if why(b, asked) is None), why(rows[-1], asked)) for asked in (True, False))
    if named is not None:
        return {"available": False, "auto": False, "reason": named}
    out = {"available": True, "auto": auto is None, "reason": auto or ""}
    cleaner = cleanup.Cleaner(model_name or None, backend, timeout_s)
    t0 = time.perf_counter()
    cleaner.load()
    out["load_s"] = round(time.perf_counter() - t0, 3)
    if cleaner.backend != backend:
        return dict(out, ok=False, error=f"the model ran on {cleaner.backend} instead")
    took = []
    for sentence in SENTENCES[:RUNS]:
        t0 = time.perf_counter()
        cleaner._llm_clean(sentence, max_ms=60000)
        took.append((time.perf_counter() - t0) * 1000.0)
    took_sorted = sorted(took)
    return dict(out, ok=True, median_ms=round(took_sorted[len(took) // 2], 1), runs_ms=[round(x, 1) for x in took])


def child_main(args) -> int:
    """``--speed-check MODEL BACKEND OUT [MODEL_NAME|default [BUDGET_S]]``: measure one back-end, write the JSON
    result to OUT and print it. Always exits 0 with a result; the parent treats anything else as a failure."""
    args = list(args)
    model, backend, out_path = (args + ["", "", ""])[:3]
    model_name = args[3] if len(args) > 3 and args[3] != "default" else ""
    try:
        budget_s = float(args[4]) if len(args) > 4 else 4.0
    except ValueError:
        budget_s = 4.0
    # if the app that started this child is gone, a hung driver call must not keep it alive for good
    watchdog = threading.Timer(CHILD_TIMEOUT_S.get(model, 300.0) + 30.0, os._exit, args=(3,))
    watchdog.daemon = True
    watchdog.start()
    try:
        if model == "asr":
            result = _measure_asr(backend, model_name)
        elif model == "cleanup":
            result = _measure_cleanup(backend, model_name, budget_s if budget_s > 0 else 4.0)
        else:
            result = {"error": f"unknown model {model!r}"}
    except Exception as exc:  # noqa: BLE001 - a back-end that cannot run is a result
        result = {"available": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        watchdog.cancel()
    text = json.dumps(result)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    print(text)
    return 0


# what ``demo_child`` reports: plausible numbers for --demo (an NVIDIA PC); nothing is measured
DEMO_MS = {"asr": {"cuda": 29.0, "webgpu": 65.0, "cpu": 257.0},
           "cleanup": {"cuda": 186.0, "vulkan": 241.0, "cpu": 1930.0}}


def _demo_result(model: str, backend: str) -> dict:
    ms = DEMO_MS[model][backend]
    return clean_result({"available": True, "auto": True, "ok": True, "median_ms": ms, "runs_ms": [ms] * RUNS,
                         "load_s": 1.0})


def demo_results() -> dict:
    """``{model: {backend: result}}`` from ``DEMO_MS``, as a finished run reports it."""
    return {m: {b: _demo_result(m, b) for b in DEMO_MS[m]} for m in MODELS}


def demo_child(model: str, backend: str, model_name: str, budget_s: float, cancel: threading.Event,
               delay_s: float = 0.35) -> dict:
    """A stand-in for ``run_child`` in demo mode: waits a moment and answers from ``DEMO_MS``."""
    if cancel.wait(delay_s):
        raise Cancelled()
    return _demo_result(model, backend)


# --------------------------------------------------------------------------- running it from the app
class SpeedCheck:
    """Runs the measurement on a background thread, one child after the other. ``on_event(dict)`` is called on
    that thread with ``{"event": "progress", "done", "total", "model", "backend"}``, then once with
    ``{"event": "finished", "results", "took_s"}`` or ``{"event": "cancelled"}``."""

    def __init__(self, on_event, run=run_child):
        self.on_event = on_event
        self._run = run
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, models: dict, budget_s: float) -> bool:
        """Start measuring ``models`` ({"asr": name, "cleanup": name}); False if a run is already going.
        ``budget_s``: the clean-up time limit of a dictation (a GPU too slow for it fails, as in the app)."""
        if self.running:
            return False
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._body, args=(dict(models), float(budget_s), self._cancel),
                                        name="voxflow-speed-check", daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _body(self, models: dict, budget_s: float, cancel: threading.Event) -> None:
        plan = [(m, b) for m in MODELS for b in backends(m)]
        results: dict = {m: {} for m in MODELS}
        started = time.monotonic()
        for i, (model, backend) in enumerate(plan):
            self.on_event({"event": "progress", "done": i, "total": len(plan), "model": model, "backend": backend})
            try:
                if cancel.is_set():
                    raise Cancelled()
                results[model][backend] = self._run(model, backend, models.get(model) or "", budget_s, cancel)
            except Cancelled:
                self.on_event({"event": "cancelled"})
                return
        self.on_event({"event": "finished", "results": results, "took_s": time.monotonic() - started})
