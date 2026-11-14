"""Small shared test helpers."""
from __future__ import annotations

import time as _real_time


class _TimeProxy:
    """Stands in for the ``time`` module inside ONE module under test.

    Never ``monkeypatch.setattr(some_module.time, "perf_counter", ...)``: ``some_module.time`` is the global
    ``time`` module, so that changes the clock for the whole process (Qt timers of other objects, worker threads,
    pytest-qt) for as long as the test runs. Replace the module's own ``time`` name instead:

        monkeypatch.setattr(some_module, "time", fake_time(perf_counter=lambda: 10.0))
    """

    def __init__(self, overrides: dict):
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(_real_time, name)


def fake_time(**overrides) -> _TimeProxy:
    return _TimeProxy(overrides)
