"""voxflow.system: the Run-value length limit and re-installing the crash handler."""
from __future__ import annotations

import logging
import sys
import threading

import pytest

from voxflow import system as sysmod


class _Registry:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


def test_install_path_too_long_for_the_run_key_is_refused_with_a_human_message(monkeypatch):
    reg = _Registry()
    monkeypatch.setattr(sysmod, "registry", reg)
    monkeypatch.setattr(sysmod, "launch_command", lambda: "x" * (sysmod.RUN_VALUE_LIMIT + 1))
    with pytest.raises(ValueError, match="path is too long"):
        sysmod.set_autostart(True)
    assert reg.values == {}, "nothing is written"
    monkeypatch.setattr(sysmod, "launch_command", lambda: "x" * sysmod.RUN_VALUE_LIMIT)
    sysmod.set_autostart(True)
    assert reg.values == {sysmod.RUN_VALUE: "x" * sysmod.RUN_VALUE_LIMIT}, "exactly at the limit is fine"


def test_installing_the_crash_handler_twice_keeps_the_one_fault_log(tmp_path):
    root = logging.getLogger("voxflow")
    handlers, level = list(root.handlers), root.level
    hooks = (sys.excepthook, threading.excepthook)
    try:
        sysmod.shutdown_logging()                       # start without an open fault file
        sysmod.install_crash_handler(tmp_path / "a")
        first = sysmod._fault_file
        assert first is not None and not first.closed
        first_hook = sys.excepthook
        sysmod.install_crash_handler(tmp_path / "b")    # e.g. a second Controller in the same process
        assert sysmod._fault_file is first, "the open fault log is reused, not replaced"
        assert sys.excepthook is not first_hook, "the hooks are re-pointed at the new data folder"
        assert not (tmp_path / "b" / "logs" / "fault.log").exists()
    finally:
        sysmod.shutdown_logging()
        for h in list(root.handlers):
            if h not in handlers:
                root.removeHandler(h)
                h.close()
        root.setLevel(level)
        sys.excepthook, threading.excepthook = hooks
    assert sysmod._fault_file is None
