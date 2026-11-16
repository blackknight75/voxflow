"""Fixtures of the live update end-to-end tests: a fake GitHub over real TLS, a real stub installer exe, and the
real Controller (stand-in models) with a real Updater. Run with ``VOXFLOW_UPDATE_E2E=1``; see harness.py."""
from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tests.controller_rig import Rig
from tests.e2e_update import harness as H
from voxflow import updater as U

E2E = os.environ.get("VOXFLOW_UPDATE_E2E") == "1"


@pytest.fixture(scope="session")
def pki(tmp_path_factory):
    if not E2E:
        pytest.skip("set VOXFLOW_UPDATE_E2E=1")
    try:
        return H.make_pki(tmp_path_factory.mktemp("pki"))
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"no openssl to make the test certificates: {exc}")


@pytest.fixture(scope="session")
def stub_exe():
    if not E2E:
        pytest.skip("set VOXFLOW_UPDATE_E2E=1")
    cache = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "voxflow-e2e-stub"
    return H.build_stub_installer(cache)


class FastRng:
    """The first automatic check comes 0.3 s after start instead of 90-600 s."""

    def uniform(self, _a, _b):
        return 0.3


class App:
    """One run of VoxFlow: the real Controller (stand-in models and desktop), its real Updater, optionally the UI."""

    def __init__(self, env: "Env", rig: Rig, ui: bool):
        self.env, self.rig, self.c = env, rig, rig.c
        self.u: U.Updater = rig.c.updates
        self.restarts: list[int] = []
        self.c.restart_requested.connect(lambda: self.restarts.append(1))
        self.shell = None
        if ui:
            from voxflow.ui.shell import build_ui
            # ui="tray": on the real desktop, start in the tray (as at login) and let the test show the window
            # without activating it (test_update_desktop.py), so another app keeps the foreground
            self.shell = build_ui(self.c, show_window=ui != "tray")
            QApplication.processEvents()
        self.closed = False

    def wait(self, cond, timeout=15000):
        self.rig.qtbot.waitUntil(lambda: bool(cond()), timeout=timeout)

    def idle(self, timeout=15000):
        self.wait(lambda: self.u.phase == "idle", timeout)

    def check(self, user=True):
        before = len(self.env.sc.requests)
        assert self.u.check(user=user)
        self.idle()
        return self.env.sc.requests[before:]

    def exit(self, user_quit: bool = True):
        """What the app does on Quit: (tray) prepare_quit, the window closes, Controller.shutdown()."""
        if self.closed:
            return
        self.closed = True
        if user_quit:
            self.c.prepare_quit()
        if self.shell is not None:
            self.shell.close()
        self.c.shutdown()

    def files(self) -> list[str]:
        f = self.u.folder
        return sorted(p.name for p in f.iterdir()) if f.is_dir() else []


class Env:
    def __init__(self, tmp_path: Path, qtbot, sc: H.Scenario, gh: H.FakeGitHub):
        self.tmp, self.qtbot, self.sc, self.gh = tmp_path, qtbot, sc, gh
        self.record = tmp_path / "installer-ran.json"
        self.clock = H.Clock()
        self.now_value = None            # datetime the schedule sees (None: the real local time)
        self.idle_value = None           # seconds without input (None: the real GetLastInputInfo)
        self.apps: list[App] = []

    def app(self, *, mode="notify", when="quit", at="03:00", version="0.2.0", kind="installer", ui=False,
            timeout=5.0, config=True, metered=None) -> App:
        cfg = {"onboarding_done": True, "theme": "dark", "update_mode": mode, "update_install_when": when,
               "update_time": at} if config else None

        def factory(data_dir, get, busy):
            return U.Updater(data_dir, get, busy, version=version, http=U.Http(timeout=timeout), kind=kind,
                             metered=metered if metered is not None else U.connection_metered,
                             clock=self.clock,
                             now=lambda: self.now_value or __import__("datetime").datetime.now().astimezone(),
                             idle=lambda: self.idle_value if self.idle_value is not None else U.idle_seconds(),
                             rng=FastRng())

        rig = Rig(self.tmp, self.qtbot, config=cfg if not (self.tmp / "data" / "config.json").exists() else None,
                  updates=factory)
        a = App(self, rig, ui)
        self.apps.append(a)
        return a

    def ran(self, timeout=30.0) -> dict:
        return H.wait_for_record(self.record, timeout)

    def note(self, test: str, evidence: str) -> None:
        out = os.environ.get("VOXFLOW_E2E_EVIDENCE")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"test": test, "evidence": evidence}) + "\n")


@pytest.fixture
def env(tmp_path, qtbot, monkeypatch, pki, stub_exe):
    sc = H.Scenario(installer=stub_exe.read_bytes())
    gh = H.FakeGitHub(pki, sc)
    monkeypatch.setattr(socket, "create_connection", gh.connector(socket.create_connection))
    monkeypatch.setenv("SSL_CERT_FILE", str(pki["ca"]))
    temp = tmp_path / "temp"                    # %TEMP%: the updater's real download_folder() lands in it
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    e = Env(tmp_path, qtbot, sc, gh)
    monkeypatch.setenv("VOXFLOW_E2E_RECORD", str(e.record))   # inherited by the installer the updater starts
    monkeypatch.setenv("_MEIPASS2", str(tmp_path))            # as if this were a frozen VoxFlow: must be stripped
    monkeypatch.setenv("VOXFLOW_E2E_CANARY", "1")
    monkeypatch.setattr(QApplication, "quit", lambda *a: None)   # the shell's quit: the test calls App.exit()
    yield e
    for a in e.apps:
        try:
            a.exit(user_quit=False)
        except Exception:  # noqa: BLE001
            pass
    gh.close()
