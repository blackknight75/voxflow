"""``python -m voxflow``: the module-as-script guard exits with ``main()``'s return value."""
from __future__ import annotations

import runpy
import sys

import pytest

from voxflow import app


@pytest.mark.parametrize("argv,code", [(["--demo"], 0), (["--selftest", "clip.wav"], 7)])
def test_running_the_package_as_a_script_exits_with_the_result_of_main(monkeypatch, argv, code):
    calls = []
    monkeypatch.setattr(app, "run_app", lambda **kw: calls.append(("run", kw["demo"])) or 0)
    monkeypatch.setattr(app, "selftest", lambda wav, fake=False: calls.append(("selftest", wav)) or 7)
    monkeypatch.setattr(sys, "argv", ["voxflow", *argv])
    monkeypatch.delitem(sys.modules, "voxflow.__main__", raising=False)     # run it fresh, as ``python -m`` does
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("voxflow", run_name="__main__", alter_sys=True)
    assert exc.value.code == code
    assert calls == ([("run", True)] if code == 0 else [("selftest", "clip.wav")])


def test_importing_the_entry_module_does_not_start_the_app(monkeypatch):
    import importlib
    monkeypatch.setattr(app, "run_app", lambda **kw: pytest.fail("the app was started by an import"))
    monkeypatch.delitem(sys.modules, "voxflow.__main__", raising=False)
    entry = importlib.import_module("voxflow.__main__")
    assert callable(entry.main) and entry.__name__ == "voxflow.__main__"
