"""A slow CPU must still load the language model (v0.2.1 release run: the warm-up of the 562-token few-shot prompt
took longer than the old 60 s limit on a 4-vCPU GitHub runner, and the whole session fell back to rules only)."""
from __future__ import annotations

import time

import pytest

from voxflow import cleanup
from voxflow.cleanup import Cleaner


class _Slow:
    """Stands in for Llama: every chat completion takes ``seconds`` before its first token (a long prefill)."""
    def __init__(self, seconds):
        self.seconds = seconds
        self.calls = 0

    def tokenize(self, *a, **k):
        return [1, 2, 3]

    def create_chat_completion(self, **k):
        self.calls += 1
        time.sleep(self.seconds)
        return iter([{"choices": [{"delta": {"content": "What is the capital of France?"}}]}])


def _cleaner(monkeypatch, libs, device, build):
    monkeypatch.setattr(cleanup, "_setup_dll_paths", lambda: None)
    monkeypatch.setattr(cleanup, "_select_llama_libs", lambda d: libs)
    monkeypatch.setattr(cleanup, "_gpu_offload_supported", lambda: True)
    c = Cleaner(model=__file__, device=device, timeout_s=4.0)
    monkeypatch.setattr(c, "_build", build)
    return c


def test_slow_cpu_warmup_still_loads(monkeypatch):
    """A warm-up slower than the GPU limit is fine on the CPU, the last resort."""
    monkeypatch.setattr(cleanup, "WARMUP_GPU_S", 0.1)
    monkeypatch.setattr(cleanup, "WARMUP_CPU_S", 5.0)
    c = _cleaner(monkeypatch, "cpu", "cpu", lambda path, gpu: _Slow(0.4))
    c.load()
    assert c.backend == "cpu" and c.has_llm


def test_cpu_budget_is_far_longer_than_the_gpu_one():
    """The runner needed ~50 s (v0.2.0) and more than 60 s (v0.2.1); a desktop ~7 s."""
    assert cleanup.WARMUP_GPU_S == 60.0
    assert cleanup.WARMUP_CPU_S >= 5 * cleanup.WARMUP_GPU_S


def test_gpu_warmup_timeout_falls_back_to_a_working_cpu(monkeypatch):
    """The GPU's timed-out warm-up is still unwinding when the CPU attempt starts: it waits instead of failing at
    once with "previous LLM generation still running" (which left the session on rules only)."""
    monkeypatch.setattr(cleanup, "WARMUP_GPU_S", 0.1)
    monkeypatch.setattr(cleanup, "WARMUP_CPU_S", 5.0)
    models = {True: _Slow(0.5), False: _Slow(0.0)}
    c = _cleaner(monkeypatch, None, "auto", lambda path, gpu: models[gpu])
    c.load()
    assert c.backend == "cpu" and c.has_llm
    assert models[False].calls == 1                     # the CPU warm-up really ran
    assert not c._abandoned.is_set()


def test_cpu_warmup_is_still_bounded(monkeypatch):
    """A CPU that cannot finish even the long warm-up degrades to rules, as before (load never hangs)."""
    monkeypatch.setattr(cleanup, "WARMUP_CPU_S", 0.1)
    c = _cleaner(monkeypatch, "cpu", "cpu", lambda path, gpu: _Slow(0.5))
    t0 = time.perf_counter()
    c.load()
    assert time.perf_counter() - t0 < 2.0
    assert c.backend == "rules" and c.ready


@pytest.mark.parametrize("flag_set", [False, True])
def test_wait_unwound_is_bounded(flag_set):
    c = Cleaner()
    if flag_set:
        c._abandoned.set()
    t0 = time.perf_counter()
    c._wait_unwound(0.2)
    assert time.perf_counter() - t0 < (1.0 if flag_set else 0.1)
