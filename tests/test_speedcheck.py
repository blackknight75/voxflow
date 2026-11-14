"""The speed check (voxflow.speedcheck): choosing by measurement, the stored result, the child process.

Hermetic: no child process is started, no model is loaded and no GPU is touched. The child is a scripted
``popen``; the registry is a fake ``winreg`` module. The one real measurement is in ``test_speedcheck_real.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import types

import pytest

from voxflow import gpuprobe, speedcheck as sc


def ok(ms, auto=True):
    return {"available": True, "auto": auto, "ok": True, "median_ms": ms, "runs_ms": [ms] * 3, "load_s": 1.0}


def unavailable(reason="no NVIDIA driver (nvcuda.dll)"):
    return {"available": False, "auto": False, "reason": reason}


def failed(error="RuntimeError: out of memory", auto=True):
    return {"available": True, "auto": auto, "ok": False, "error": error}


ORDER = ("cuda", "webgpu", "cpu")


# =========================================================================== choosing
def test_backend_names_match_the_model_tables():
    from voxflow.asr import ASR_BACKENDS
    from voxflow.cleanup import LLAMA_BACKENDS
    assert sc.backends("asr") == tuple(b.name for b in ASR_BACKENDS)
    assert sc.backends("cleanup") == tuple(b.name for b in LLAMA_BACKENDS)


def test_the_static_choice_is_the_first_back_end_the_automatic_probe_accepts():
    results = {"cuda": unavailable(), "webgpu": ok(50.0, auto=False), "cpu": ok(140.0)}
    assert sc.choose(results, ORDER) == ("cpu", "webgpu"), "an integrated GPU is a candidate, and it won clearly"
    assert sc.choose({}, ORDER) == (None, None)


@pytest.mark.parametrize("static_ms,best_ms,switch", [
    (100.0, 85.0, True),      # exactly 15 % and 15 ms faster
    (100.0, 86.0, False),     # 14 % faster: noise
    (50.0, 40.0, True),       # 20 % and exactly 10 ms
    (50.0, 41.0, False),      # 18 % but only 9 ms
    (1000.0, 849.0, True),
    (30.0, 30.0, False),
])
def test_a_winner_replaces_the_static_choice_only_when_clearly_faster(static_ms, best_ms, switch):
    results = {"cuda": ok(static_ms), "webgpu": ok(best_ms), "cpu": ok(999.0)}
    assert sc.choose(results, ORDER) == ("cuda", "webgpu" if switch else None)


def test_nothing_changes_when_the_static_choice_could_not_be_measured():
    """e.g. the GPU ran out of memory in the child: the CPU 'winning' must not move a GPU user to the CPU."""
    assert sc.choose({"cuda": failed(), "webgpu": failed(), "cpu": ok(140.0)}, ORDER) == ("cuda", None)


def test_the_static_choice_staying_fastest_changes_nothing_and_ties_go_by_order():
    assert sc.choose({"cuda": ok(20.0), "webgpu": ok(60.0), "cpu": ok(140.0)}, ORDER) == ("cuda", None)
    assert sc.fastest({"cuda": ok(20.0), "webgpu": ok(20.0)}, ORDER) == "cuda"
    assert sc.fastest({"cuda": failed()}, ORDER) is None


def test_timings_that_are_not_finite_numbers_are_ignored():
    for bad in (True, float("inf"), float("nan"), "12", None):
        results = {"cuda": ok(100.0), "webgpu": dict(ok(10.0), median_ms=bad), "cpu": ok(140.0)}
        assert sc.choose(results, ORDER) == ("cuda", None)
    assert sc._median("not a dict") is None
    assert sc.choose({"cuda": ok(100.0), "metal": ok(1.0)}, ORDER) == ("cuda", None), "unknown names are ignored"


def test_a_record_holds_the_cleaned_results_and_the_choice_per_model(monkeypatch):
    results = {"asr": {"cuda": ok(28.0), "webgpu": ok(63.0), "cpu": ok(143.0), "metal": ok(1.0)},
               "cleanup": "junk"}
    rec = sc.make_record("key", {"cpu": "X"}, {"asr": "m"}, results, 12.34)
    assert rec["format"] == sc.FORMAT and rec["fingerprint"] == "key" and rec["took_s"] == 12.3
    assert set(rec["results"]["asr"]) == {"cuda", "webgpu", "cpu"} and rec["results"]["cleanup"] == {}
    assert rec["static"] == {"asr": "cuda", "cleanup": None} and rec["choice"] == {"asr": None, "cleanup": None}


def test_the_log_table_names_every_back_end_and_the_verdict():
    rec = sc.make_record("k", {}, {}, {"asr": {"cuda": unavailable(), "webgpu": ok(40.0, auto=False),
                                               "cpu": ok(150.0)},
                                       "cleanup": {"cuda": ok(76.0), "vulkan": failed("TimeoutError: slow"),
                                                   "cpu": ok(949.0)}}, 1.0)
    lines = sc.describe(rec)
    assert lines[0] == ("speech: cuda unavailable (no NVIDIA driver (nvcuda.dll)), webgpu 40 ms, cpu 150 ms "
                        "-> switch to webgpu")
    assert lines[1] == ("language model: cuda 76 ms, vulkan failed (TimeoutError: slow), cpu 949 ms "
                        "-> keep the automatic choice (cuda)")
    empty = sc.make_record("k", {}, {}, {}, 0.0)
    assert sc.describe(empty)[0] == "speech: nothing measured -> keep the automatic choice (unknown)"


# =========================================================================== what a child sends back
def test_a_result_is_brought_into_a_known_shape_whatever_was_sent():
    assert sc.clean_result("garbage") == {"available": False, "auto": False, "reason": "", "error": "",
                                          "median_ms": None, "load_s": None, "runs_ms": [], "ok": False}
    r = sc.clean_result({"available": True, "auto": "yes", "ok": True, "median_ms": 22.0, "load_s": -1,
                         "runs_ms": [22.0, "x", True, 30.5, float("inf")], "reason": 5, "error": None})
    assert r == {"available": True, "auto": False, "reason": "5", "error": "", "median_ms": 22.0, "load_s": None,
                 "runs_ms": [22.0, 30.5], "ok": True}
    assert sc.clean_result({"available": True, "ok": True, "runs_ms": "nope"})["runs_ms"] == []
    no_time = sc.clean_result({"available": True, "ok": True})
    assert not no_time["ok"] and no_time["error"] == "no usable timing"
    said = sc.clean_result({"available": True, "ok": True, "error": "odd"})
    assert said["error"] == "odd", "an error the child gave is kept"
    assert not sc.clean_result({"available": False, "ok": True, "median_ms": 5})["ok"]
    assert len(sc.clean_result({"error": "x" * 1000})["error"]) == 300


@pytest.mark.parametrize("reason,short", [
    ("no NVIDIA driver (nvcuda.dll)", "No NVIDIA driver"), ("not installed", "Not installed"),
    ("did not load ([WinError 126])", "Not installed"),
    ("no GPU for WebGPU (adapters: none)", "No suitable GPU"),
    ("no discrete Vulkan GPU with 3500 MiB of memory (devices: none)", "No suitable GPU"),
    ("the WebGPU plugin did not load (ImportError: x)", "Not installed"), ("", "Not available here"),
    ("CUDAExecutionProvider not available", "Not available here"),
])
def test_probe_reasons_become_a_few_words_for_the_menu(reason, short):
    assert sc.short_reason(reason) == short


# =========================================================================== the stored result
def test_a_missing_or_unreadable_result_means_measure_again(tmp_path, caplog):
    assert sc.load(tmp_path) is None
    path = sc.cache_path(tmp_path)
    for text in ("{not json", "[1, 2]", json.dumps({"format": 99, "fingerprint": "k", "results": {}, "static": {},
                                                    "choice": {}}),
                 json.dumps({"format": sc.FORMAT, "fingerprint": 5, "results": {}, "static": {}, "choice": {}}),
                 json.dumps({"format": sc.FORMAT, "fingerprint": "k", "results": [], "static": {}, "choice": {}})):
        path.write_text(text, encoding="utf-8")
        assert sc.load(tmp_path) is None, text
    assert "measured again" in caplog.text
    path.write_bytes(b"\xff\xfe\x00bad")
    assert sc.load(tmp_path) is None


def test_a_result_survives_a_save_and_load_and_foreign_values_are_dropped(tmp_path):
    rec = sc.make_record("k", {"cpu": "X"}, {"asr": "m"}, {"asr": {"cuda": ok(80.0), "webgpu": ok(30.0),
                                                                   "cpu": ok(140.0)}}, 3.0)
    sc.save(tmp_path, rec)
    assert sc.load(tmp_path) == rec
    data = json.loads(sc.cache_path(tmp_path).read_text(encoding="utf-8"))
    data["choice"] = {"asr": "metal", "cleanup": 5}
    data["static"]["cleanup"] = "vulkan"
    data["results"]["asr"]["cuda"] = "garbage"
    data["results"]["asr"]["metal"] = ok(1.0)
    data["results"]["cleanup"] = "junk"
    sc.cache_path(tmp_path).write_text(json.dumps(data), encoding="utf-8")
    got = sc.load(tmp_path)
    assert got["choice"] == {"asr": None, "cleanup": None}
    assert got["static"] == {"asr": "cuda", "cleanup": "vulkan"}
    assert got["results"]["asr"]["cuda"]["ok"] is False and "metal" not in got["results"]["asr"]
    assert got["results"]["cleanup"] == {}


# =========================================================================== what the result is keyed by
def test_the_fingerprint_follows_hardware_drivers_versions_and_models(monkeypatch):
    monkeypatch.setattr(gpuprobe, "cpu_name", lambda: "AMD Ryzen 7 9800X3D")
    monkeypatch.setattr(gpuprobe, "display_adapters", lambda: [("NVIDIA GeForce RTX 5080", "32.0.16.1088")])
    monkeypatch.setattr(sc, "_dist_version", lambda name: {"llama-cpp-python": "0.3.36"}.get(name, ""))
    info = sc.machine()
    assert info["cpu"] == "AMD Ryzen 7 9800X3D" and info["gpus"] == [["NVIDIA GeForce RTX 5080", "32.0.16.1088"]]
    assert info["packages"]["llama-cpp-python"] == "0.3.36" and info["packages"]["onnxruntime"] == ""
    models = {"asr": "nemo-parakeet-tdt-0.6b-v3", "cleanup": ""}
    key = sc.fingerprint(info, models)
    assert key == sc.fingerprint(json.loads(json.dumps(info)), dict(models)), "stable"
    newer_driver = dict(info, gpus=[["NVIDIA GeForce RTX 5080", "32.0.16.1100"]])
    for other in (newer_driver, dict(info, app="9.9.9"), dict(info, cpu="Intel"),
                  dict(info, packages=dict(info["packages"], **{"onnxruntime-gpu": "2.0"}))):
        assert sc.fingerprint(other, models) != key
    assert sc.fingerprint(info, dict(models, cleanup="other.gguf")) != key


def test_package_versions_come_from_metadata_and_a_missing_one_is_empty():
    assert sc._dist_version("pytest")
    assert sc._dist_version("no-such-distribution-voxflow") == ""


class FakeKey:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_winreg(tree: dict):
    """A ``winreg`` stand-in over ``{path: {"values": {...}, "subkeys": [...]}}``."""
    mod = types.ModuleType("winreg")
    mod.HKEY_LOCAL_MACHINE = object()

    def open_key(root, path):
        if path not in tree:
            raise OSError(2, "not found")
        return FakeKey(path)

    def query(key, name):
        values = tree[key.path].get("values", {})
        if name not in values:
            raise OSError(2, "no value")
        return values[name], 1

    mod.OpenKey = open_key
    mod.QueryValueEx = query
    mod.QueryInfoKey = lambda key: (len(tree[key.path].get("subkeys", [])), 0, 0)
    mod.EnumKey = lambda key, i: tree[key.path]["subkeys"][i]
    return mod


def test_display_adapters_and_cpu_name_come_from_the_registry(monkeypatch):
    cls = gpuprobe._DISPLAY_CLASS
    tree = {cls: {"subkeys": ["0000", "0001", "0002", "0003", "Properties"]},
            f"{cls}\\0000": {"values": {"DriverDesc": "NVIDIA GeForce RTX 5080 ", "DriverVersion": "32.0.16.1088"}},
            f"{cls}\\0001": {"values": {"DriverDesc": "AMD Radeon(TM) Graphics"}},
            f"{cls}\\0002": {"values": {"DriverVersion": "1.0"}},                     # no name: not an adapter
            f"{cls}\\0003": {"values": {"DriverDesc": "NVIDIA GeForce RTX 5080 ", "DriverVersion": "32.0.16.1088"}},
            gpuprobe._CPU_KEY: {"values": {"ProcessorNameString": "AMD Ryzen 7 9800X3D 8-Core Processor   "}}}
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg(tree))
    monkeypatch.setattr(gpuprobe.sys, "platform", "win32")
    assert gpuprobe.display_adapters() == [("AMD Radeon(TM) Graphics", ""),
                                           ("NVIDIA GeForce RTX 5080", "32.0.16.1088")]
    assert gpuprobe.cpu_name() == "AMD Ryzen 7 9800X3D 8-Core Processor"
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg({}))
    assert gpuprobe.display_adapters() == [] and gpuprobe.cpu_name() == ""


def test_the_registry_probes_answer_nothing_off_windows(monkeypatch):
    monkeypatch.setattr(gpuprobe.sys, "platform", "linux")
    assert gpuprobe.display_adapters() == [] and gpuprobe.cpu_name() == ""


# =========================================================================== the child process
def test_the_child_is_started_through_the_frozen_exe_or_python_m(monkeypatch):
    monkeypatch.setattr(sc.sys, "executable", "C:/VoxFlow/VoxFlow.exe")
    monkeypatch.setattr(sc.sys, "frozen", True, raising=False)
    assert sc.child_command("asr", "cuda", "out.json", "", 4.0) == [
        "C:/VoxFlow/VoxFlow.exe", "--speed-check", "asr", "cuda", "out.json", "default", "4"]
    monkeypatch.setattr(sc.sys, "frozen", False)
    assert sc.child_command("cleanup", "cpu", "o", "repo:f.gguf", 6.5)[:3] == ["C:/VoxFlow/VoxFlow.exe", "-m",
                                                                               "voxflow"]
    assert sc.child_command("cleanup", "cpu", "o", "repo:f.gguf", 6.5)[-2:] == ["repo:f.gguf", "6.5"]


def test_the_child_gets_a_clean_offline_environment(monkeypatch):
    monkeypatch.setenv("VOXFLOW_DEVICE", "cpu")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", "C:/somewhere/lib")
    env = sc._child_env()
    assert "VOXFLOW_DEVICE" not in env and "LLAMA_CPP_LIB_PATH" not in env and env["HF_HUB_OFFLINE"] == "1"


class FakeProc:
    """A scripted child: ``script(out_path)`` runs at start; it 'exits' after ``polls`` polls."""

    def __init__(self, cmd, polls=1, code=0, script=None):
        self.cmd, self.returncode, self._polls, self.killed = cmd, None, polls, False
        if script is not None:
            script(cmd[cmd.index("--speed-check") + 3])
        self._code = code

    def poll(self):
        if self.killed:
            return self.returncode
        self._polls -= 1
        if self._polls <= 0:
            self.returncode = self._code
        return self.returncode

    def kill(self):
        self.killed, self.returncode = True, -9

    def wait(self):
        return self.returncode


def popen_of(seen: list, **kw):
    def popen(cmd, **options):
        seen.append(options)
        proc = FakeProc(cmd, **kw)
        seen.append(proc)
        return proc
    return popen


def write(text):
    def script(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return script


def test_a_child_result_is_read_from_its_file_and_checked(monkeypatch):
    seen: list = []
    r = sc.run_child("asr", "cuda", "", 4.0, threading.Event(),
                     popen=popen_of(seen, polls=2, script=write(json.dumps(ok(22.0)))), poll_s=0.001)
    assert r["ok"] and r["median_ms"] == 22.0
    options = seen[0]
    assert options["stdout"] is subprocess.DEVNULL and options["env"]["HF_HUB_OFFLINE"] == "1"
    if sys.platform == "win32":
        assert options["creationflags"] & 0x00004000, "below-normal priority"
    out_path = seen[1].cmd[seen[1].cmd.index("--speed-check") + 3]
    import os
    assert not os.path.exists(out_path), "the result file is removed"


@pytest.mark.parametrize("text,code", [("", 3221225477), ("not json at all", 0), ("[1, 2, 3]", 0)])
def test_a_child_that_crashes_or_sends_garbage_is_a_failed_candidate(text, code):
    r = sc.run_child("cleanup", "vulkan", "", 4.0, threading.Event(),
                     popen=popen_of([], code=code, script=write(text)), poll_s=0.001)
    assert r["ok"] is False and r["available"] is False
    assert r["error"] == f"the measurement ended without a result (exit code {code})"


def test_a_child_that_hangs_is_killed_at_its_time_limit():
    seen: list = []
    r = sc.run_child("asr", "webgpu", "", 4.0, threading.Event(), popen=popen_of(seen, polls=10 ** 9),
                     poll_s=0.001, timeout_s=0.05)
    assert r["error"] == "timed out after 0 s" and seen[1].killed


def test_a_dictation_cancels_the_child_at_once():
    seen: list = []
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(sc.Cancelled):
        sc.run_child("asr", "cuda", "", 4.0, cancel, popen=popen_of(seen, polls=10 ** 9), poll_s=0.001)
    assert seen[1].killed


def test_a_child_that_cannot_start_and_a_result_file_already_gone(monkeypatch):
    def refuse(cmd, **options):
        raise OSError("the exe is missing")
    r = sc.run_child("asr", "cuda", "", 4.0, threading.Event(), popen=refuse)
    assert r["error"] == "could not start the measurement (the exe is missing)"

    def vanish(path):
        import os
        os.unlink(path)
    r = sc.run_child("asr", "cuda", "", 4.0, threading.Event(), popen=popen_of([], script=vanish), poll_s=0.001)
    assert "without a result" in r["error"]


def test_no_priority_flags_off_windows(monkeypatch):
    seen: list = []
    monkeypatch.setattr(sc.sys, "platform", "linux")
    sc.run_child("asr", "cpu", "", 4.0, threading.Event(), popen=popen_of(seen, script=write("{}")), poll_s=0.001)
    assert seen[0]["creationflags"] == 0


# =========================================================================== inside the child
def test_child_main_measures_writes_and_prints(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(sc, "_measure_asr", lambda b, m: calls.append(("asr", b, m)) or ok(24.0))
    monkeypatch.setattr(sc, "_measure_cleanup", lambda b, m, t: calls.append(("cleanup", b, m, t)) or ok(76.0))
    out = tmp_path / "r.json"
    assert sc.child_main(["asr", "cuda", str(out), "default", "4"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["median_ms"] == 24.0
    assert json.loads(capsys.readouterr().out)["median_ms"] == 24.0
    sc.child_main(["cleanup", "vulkan", str(out), "repo:f.gguf", "x"])
    sc.child_main(["cleanup", "cpu", str(out), "default", "-1"])
    sc.child_main(["cleanup", "cuda", str(out), "default", "6"])
    sc.child_main(["cleanup", "cuda", str(out)])
    assert calls == [("asr", "cuda", ""), ("cleanup", "vulkan", "repo:f.gguf", 4.0), ("cleanup", "cpu", "", 4.0),
                     ("cleanup", "cuda", "", 6.0), ("cleanup", "cuda", "", 4.0)]
    sc.child_main(["metal", "x", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == {"error": "unknown model 'metal'"}
    capsys.readouterr()

    def broken(backend, model):
        raise RuntimeError("the driver crashed")
    monkeypatch.setattr(sc, "_measure_asr", broken)
    assert sc.child_main(["asr"]) == 0, "no output file: the result is only printed"
    assert json.loads(capsys.readouterr().out) == {"available": True, "ok": False,
                                                   "error": "RuntimeError: the driver crashed"}


def test_the_command_line_flag_runs_the_child(monkeypatch):
    from voxflow import __main__ as main
    got = []
    monkeypatch.setattr(sc, "child_main", lambda args: got.append(args) or 0)
    assert main.main(["--speed-check", "asr", "cuda", "out.json", "default", "4"]) == 0
    assert got == [["asr", "cuda", "out.json", "default", "4"]]


def backend_row(name, named=None, auto=None):
    return types.SimpleNamespace(name=name, folder=f"lib_{name}",
                                 unusable=lambda asked_for: named if asked_for else auto)


def test_speech_is_timed_only_where_the_named_probe_passes(monkeypatch):
    from voxflow import asr
    monkeypatch.setattr(asr, "ASR_BACKENDS", (backend_row("cuda", named="no NVIDIA driver (nvcuda.dll)",
                                                          auto="no NVIDIA driver (nvcuda.dll)"),
                                              backend_row("webgpu", auto="no discrete GPU"), backend_row("cpu")))
    timed = []
    monkeypatch.setattr(asr, "time_backends",
                        lambda model, names, runs: timed.append((model, names, runs)) or {names[0]: ok(55.0)})
    assert sc._measure_asr("cuda", "") == unavailable()
    r = sc._measure_asr("webgpu", "")
    assert r["ok"] and r["available"] and r["auto"] is False and r["reason"] == "no discrete GPU"
    assert sc._measure_asr("cpu", "my-model")["auto"] is True
    assert timed == [("nemo-parakeet-tdt-0.6b-v3", ["webgpu"], sc.RUNS), ("my-model", ["cpu"], sc.RUNS)]


class FakeCleaner:
    backend_after_load = None
    made: list = []

    def __init__(self, model, device, timeout_s):
        self.model, self.device, self.timeout_s = model, device, timeout_s
        self.backend = "none"
        self.cleaned: list = []
        FakeCleaner.made.append(self)

    def load(self):
        self.backend = FakeCleaner.backend_after_load or self.device

    def _llm_clean(self, text, max_ms=None):
        self.cleaned.append(text)
        return text


def test_the_language_model_is_timed_on_short_real_cleanups(monkeypatch):
    from voxflow import cleanup
    monkeypatch.setattr(cleanup, "LLAMA_BACKENDS", (backend_row("cuda"), backend_row("vulkan", auto="no discrete"),
                                                    backend_row("cpu")))
    monkeypatch.setattr(cleanup, "Cleaner", FakeCleaner)
    monkeypatch.setattr(cleanup, "_folder_unusable", lambda folder: "not installed" if "cpu" in folder else None)
    monkeypatch.setattr(cleanup, "_llama_pkg_dir", lambda: "C:/site/llama_cpp")
    r = sc._measure_cleanup("vulkan", "", 6.0)
    assert r["ok"] and r["auto"] is False and r["reason"] == "no discrete" and len(r["runs_ms"]) == sc.RUNS
    made = FakeCleaner.made[-1]
    assert (made.model, made.device, made.timeout_s) == (None, "vulkan", 6.0)
    assert made.cleaned == list(sc.SENTENCES[:sc.RUNS]), "a different sentence each run"
    assert sc._measure_cleanup("cpu", "", 4.0) == {"available": False, "auto": False, "reason": "not installed"}
    FakeCleaner.backend_after_load = "cpu"
    try:
        r = sc._measure_cleanup("cuda", "repo:f.gguf", 4.0)
    finally:
        FakeCleaner.backend_after_load = None
    assert r["ok"] is False and r["error"] == "the model ran on cpu instead" and r["auto"] is True
    monkeypatch.setattr(cleanup, "_llama_pkg_dir", lambda: None)
    assert sc._measure_cleanup("cuda", "", 4.0)["reason"] == "llama-cpp-python is not installed"


# =========================================================================== which device a model is built for
def test_override_beats_setting_beats_measured_choice_beats_automatic(monkeypatch):
    from voxflow import app
    from voxflow.config import Config
    measured = {"asr": "webgpu", "cleanup": "vulkan"}
    monkeypatch.delenv("VOXFLOW_DEVICE", raising=False)
    assert app.model_devices(Config()) == ("auto", "auto")
    assert app.model_devices(Config(), measured) == ("webgpu", "vulkan")
    assert app.model_devices(Config(), {"asr": None, "cleanup": None}) == ("auto", "auto")
    assert app.model_devices(Config(asr_device="cpu", cleanup_device="cuda"), measured) == ("cpu", "cuda")
    monkeypatch.setenv("VOXFLOW_DEVICE", "cuda")
    assert app.model_devices(Config(asr_device="cpu", cleanup_device="cpu"), measured) == ("cuda", "cuda")


def test_the_models_are_built_for_the_measured_devices(monkeypatch):
    from voxflow import app
    from voxflow.config import Config
    monkeypatch.delenv("VOXFLOW_DEVICE", raising=False)
    built = {}

    class Asr:
        def __init__(self, model, device):
            built["asr"] = device

    class Cln:
        def __init__(self, model, device, timeout_s):
            built["cleanup"] = device

    monkeypatch.setattr(app, "_real", lambda mod, attr: {"audio": object, "asr": Asr, "cleanup": Cln}[mod])
    app.make_modules(Config(), measured={"asr": "webgpu", "cleanup": None})
    assert built == {"asr": "webgpu", "cleanup": "auto"}


# =========================================================================== the background runner
def test_the_runner_measures_every_back_end_of_both_models_in_turn():
    events, calls = [], []
    done = threading.Event()

    def run(model, backend, name, budget, cancel):
        calls.append((model, backend, name, budget))
        return sc.clean_result(ok(10.0))

    def on_event(ev):
        events.append(ev)
        if ev["event"] != "progress":
            done.set()

    check = sc.SpeedCheck(on_event, run)
    check.join()                                      # nothing started yet: a no-op
    assert check.start({"asr": "m", "cleanup": ""}, 5.0)
    done.wait(5)
    check.join(5)
    assert not check.running
    assert [c[:2] for c in calls] == [(m, b) for m in sc.MODELS for b in sc.backends(m)]
    assert calls[0][2:] == ("m", 5.0) and calls[-1][2:] == ("", 5.0)
    assert [e["done"] for e in events if e["event"] == "progress"] == list(range(6))
    assert events[-1]["event"] == "finished" and events[-1]["results"]["cleanup"]["cpu"]["ok"]


def test_the_runner_stops_when_cancelled_and_only_one_run_goes_at_a_time():
    events = []
    gate, done = threading.Event(), threading.Event()

    def run(model, backend, name, budget, cancel):
        gate.set()
        cancel.wait(5)
        raise sc.Cancelled()

    def on_event(ev):
        events.append(ev)
        if ev["event"] == "cancelled":
            done.set()

    check = sc.SpeedCheck(on_event, run)
    assert check.start({}, 4.0)
    gate.wait(5)
    assert check.running and not check.start({}, 4.0)
    check.cancel()
    done.wait(5)
    check.join(5)
    assert events[-1] == {"event": "cancelled"} and len([e for e in events if e["event"] == "progress"]) == 1


def test_a_cancel_between_two_children_starts_no_further_child():
    events, calls = [], []
    done = threading.Event()
    check = None

    def run(model, backend, name, budget, cancel):
        calls.append(backend)
        check.cancel()
        return sc.clean_result(ok(10.0))

    check = sc.SpeedCheck(lambda ev: (events.append(ev), ev["event"] == "cancelled" and done.set()), run)
    check.start({}, 4.0)
    done.wait(5)
    check.join(5)
    assert calls == ["cuda"] and events[-1] == {"event": "cancelled"}


def test_demo_mode_answers_with_plausible_numbers():
    r = sc.demo_child("asr", "cuda", "", 4.0, threading.Event(), delay_s=0.0)
    assert r["ok"] and r["median_ms"] == sc.DEMO_MS["asr"]["cuda"]
    stop = threading.Event()
    stop.set()
    with pytest.raises(sc.Cancelled):
        sc.demo_child("asr", "cuda", "", 4.0, stop)
    results = sc.demo_results()
    assert set(results) == set(sc.MODELS) and all(r["ok"] for m in results.values() for r in m.values())
    assert set(results["cleanup"]) == set(sc.backends("cleanup"))
