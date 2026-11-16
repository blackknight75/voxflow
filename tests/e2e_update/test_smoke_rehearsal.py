""".github/scripts/installer_smoke.ps1, with its update-path step (-UpdatePath), run for real against a FAKE Inno
Setup (fake_inno.py) in a temp folder and a throw-away HKCU key: VoxFlow is not installed on this PC. This proves
the PowerShell runs end to end (parsing, StrictMode, process handling, the checks' logic); the runner in
release.yml runs the same steps against the real installer."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import winreg
from pathlib import Path

import pytest

from tests.e2e_update import harness as H

pytestmark = pytest.mark.update_e2e

SCRIPT = H.HERE.parent.parent / ".github" / "scripts" / "installer_smoke.ps1"
REG = r"Software\VoxFlowSmokeRehearsal"


def _delete_tree(root, path):
    try:
        with winreg.OpenKey(root, path, 0, winreg.KEY_ALL_ACCESS) as k:
            while True:
                try:
                    sub = winreg.EnumKey(k, 0)
                except OSError:
                    break
                _delete_tree(root, rf"{path}\{sub}")
        winreg.DeleteKey(root, path)
    except OSError:
        pass


@pytest.fixture
def fake_setup(tmp_path):
    if os.environ.get("VOXFLOW_UPDATE_E2E") != "1":
        pytest.skip("set VOXFLOW_UPDATE_E2E=1")
    cache = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "voxflow-e2e-stub"
    exe = H.build_onefile(H.HERE / "fake_inno.py", cache)
    setup = tmp_path / "VoxFlow-Setup-9.9.9.exe"
    shutil.copy2(exe, setup)
    yield setup
    _delete_tree(winreg.HKEY_CURRENT_USER, REG)


def run_smoke(tmp_path, setup, *extra):
    install = tmp_path / "Programs" / "VoxFlow"
    env = {**os.environ, "FAKE_INNO_DIR": str(install), "FAKE_INNO_REG": REG}
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-Installer", str(setup),
           "-RunValue", "Foreign", "-SkipSelftest", "-InstallDir", str(install),
           "-RegistryBase", rf"HKCU:\{REG}", "-ProfileDir", str(tmp_path / "profile"),
           "-LogDir", str(tmp_path / "logs"), "-RelaunchWaitSec", "45", "-IUnderstandThisInstalls", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    return r, r.stdout + r.stderr


def test_the_smoke_script_with_the_update_path_passes_against_a_fake_installer(tmp_path, fake_setup):
    r, out = run_smoke(tmp_path, fake_setup, "-UpdatePath")
    (tmp_path / "smoke-output.txt").write_text(out, encoding="utf-8")
    assert r.returncode == 0, out
    for line in ("PASS  update path (relaunch=1): the installed VoxFlow is running before the update",
                 "PASS  update path (relaunch=1): Setup finished",
                 "PASS  update path (relaunch=1): the Setup log says the install succeeded",
                 "PASS  update path (relaunch=1): Setup closed the running VoxFlow",
                 "PASS  update path (relaunch=1): VoxFlow was started again",
                 "PASS  update path (relaunch=1): Setup started it with --after-update",
                 "PASS  update path (relaunch=1): the relaunched VoxFlow keeps running",
                 "PASS  update path (relaunch=0): Setup holds its mutex (SetupMutex=VoxFlowSetupRunning)",
                 "PASS  update path (relaunch=0): a VoxFlow started during Setup exits at once",
                 "PASS  update path (relaunch=0): it said 'Finishing the update' through a helper outside the install folder",
                 "PASS  update path (relaunch=0): Setup finished",
                 "PASS  update path (relaunch=0): Setup closed the running VoxFlow",
                 "PASS  update path (relaunch=0): VoxFlow was NOT started again",
                 "PASS  update path (relaunch=0): no VoxFlow process is left",
                 "PASS  program folder is gone", "RESULT: PASS"):
        assert line in out, out
    log = (tmp_path / "logs" / "update-relaunch1.log").read_text(encoding="utf-8")
    assert "/VERYSILENT" in log and "/RELAUNCH=1" in log and "/NORESTART" in log and "/SUPPRESSMSGBOXES" in log


def test_the_update_path_check_fails_when_setup_does_not_relaunch(tmp_path, fake_setup, monkeypatch):
    """The checks can fail: a Setup that ignores /RELAUNCH=1 is reported."""
    monkeypatch.setenv("FAKE_INNO_NO_RELAUNCH", "1")
    broken = fake_setup.with_name("VoxFlow-Setup-9.9.8.exe")
    shutil.copy2(fake_setup, broken)
    r, out = run_smoke(tmp_path, broken, "-UpdatePath")
    assert r.returncode == 1, out
    assert "FAIL  update path (relaunch=1): VoxFlow was started again" in out, out
    assert "PASS  update path (relaunch=0): VoxFlow was NOT started again" in out, out


def test_a_voxflow_that_stays_open_during_setup_is_reported(tmp_path, fake_setup, monkeypatch):
    """The other side of the second-start check: a VoxFlow that does not exit (as one holding its own message box
    did) fails it."""
    monkeypatch.setenv("FAKE_INNO_IGNORE_MUTEX", "1")
    r, out = run_smoke(tmp_path, fake_setup, "-UpdatePath")
    assert r.returncode == 1, out
    assert "FAIL  update path (relaunch=0): a VoxFlow started during Setup exits at once" in out, out
