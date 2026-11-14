"""The speed check's real child measurement for speech recognition, on this PC (slow: loads the model per back-end).

Runs exactly what the app runs in the background: ``python -m voxflow --speed-check asr <backend> ...`` in a child
process per back-end, read back through ``speedcheck.run_child``. Skipped in fast mode (``-m model``).
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("onnx_asr")
pytestmark = pytest.mark.model

from voxflow import modelhub, speedcheck  # noqa: E402


def test_the_real_child_measures_speech_on_every_back_end_of_this_pc():
    if not modelhub.is_cached(modelhub.asr_repo("nemo-parakeet-tdt-0.6b-v3")):
        pytest.skip("the speech model is not cached")
    results = {b: speedcheck.run_child("asr", b, "", 4.0, threading.Event()) for b in speedcheck.backends("asr")}
    for name, r in results.items():
        assert set(r) == {"available", "auto", "reason", "error", "median_ms", "load_s", "runs_ms", "ok"}, name
        if r["available"]:
            assert r["ok"], f"{name}: {r['error']}"
            assert 0 < r["median_ms"] < 5000 and len(r["runs_ms"]) == speedcheck.RUNS and r["load_s"] > 0, name
        else:
            assert r["reason"] and not r["ok"], name
    assert results["cpu"]["ok"] and results["cpu"]["auto"], "the CPU always runs"
    static, choice = speedcheck.choose(results, speedcheck.backends("asr"))
    assert static in speedcheck.backends("asr") and choice in (None,) + speedcheck.backends("asr")
