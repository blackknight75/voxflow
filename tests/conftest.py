"""Hermetic test harness for VoxFlow.

Everything here runs BEFORE any ``voxflow`` module is imported, so no test can reach the user's real profile:

* ``APPDATA`` points at a throw-away directory (config, history, stores, logs, lock file all live under it).
* Qt runs on the ``offscreen`` platform: no window ever appears or takes focus.
* Synthetic keyboard input, the real Windows clipboard, the global keyboard hook, the real microphone and the
  ``HKCU\\...\\Run`` registry key are blocked. A test that reaches one of them fails with a clear message
  instead of silently typing into / copying over / registering something on the developer's desktop.
* Tests that genuinely need the desktop are marked ``@pytest.mark.desktop`` and only run with
  ``VOXFLOW_DESKTOP_TESTS=1``. Tests that need the real models are marked ``@pytest.mark.model`` and are skipped
  with ``VOXFLOW_SKIP_MODEL_TESTS=1`` (they skip themselves when the weights are not cached). The live update
  end-to-end tests (``tests/e2e_update``, a local fake GitHub, a real exe started) are marked
  ``@pytest.mark.update_e2e`` and only run with ``VOXFLOW_UPDATE_E2E=1``.
* ``VOXFLOW_SPEED_CHECK=0``: no controller starts the background speed check's child processes; the tests of
  the speed check fake the child or ask for it explicitly.
* At the end of the session the real ``%APPDATA%\\VoxFlow`` folder and the real autostart registry value are
  compared with what they were at the start; any difference is reported loudly and fails the run.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

DESKTOP = os.environ.get("VOXFLOW_DESKTOP_TESTS") == "1"
SKIP_MODELS = os.environ.get("VOXFLOW_SKIP_MODEL_TESTS") == "1"
UPDATE_E2E = os.environ.get("VOXFLOW_UPDATE_E2E") == "1"

REAL_APPDATA = os.environ.get("VOXFLOW_REAL_APPDATA") or os.environ.get("APPDATA") or ""
os.environ.setdefault("VOXFLOW_REAL_APPDATA", REAL_APPDATA)        # survives into subprocess drivers
_SESSION_DIR = tempfile.mkdtemp(prefix="voxflow-tests-")
# The NVIDIA JIT cache lives under %APPDATA%; keep using the real one (read-mostly, nothing of VoxFlow's goes in
# it) so the real-model fixture does not recompile the CUDA kernels (about 25 s) on every run. P2 of the review.
if REAL_APPDATA:
    os.environ.setdefault("CUDA_CACHE_PATH", str(Path(REAL_APPDATA) / "NVIDIA" / "ComputeCache"))
os.environ["APPDATA"] = _SESSION_DIR
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# no test starts the background speed check's child processes by accident; tests that want it say so
os.environ["VOXFLOW_SPEED_CHECK"] = "0"
if not DESKTOP:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")   # the offscreen platform has no font dir

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


class DesktopAccessError(AssertionError):
    """A test touched the real desktop (keyboard, clipboard, hook, mic, registry) without the desktop marker."""


def _blocked(what: str):
    def raiser(*_a, **_k):
        raise DesktopAccessError(
            f"{what} is blocked in tests. Inject a fake, or mark the test @pytest.mark.desktop and run with "
            f"VOXFLOW_DESKTOP_TESTS=1.")
    return raiser


_USER32_BLOCKED = {"SendInput", "OpenClipboard", "EmptyClipboard", "SetClipboardData", "CloseClipboard",
                   "GetClipboardData", "EnumClipboardFormats", "SetWindowsHookExW", "keybd_event", "mouse_event",
                   "PostMessageW", "SendMessageW", "SetForegroundWindow"}


class GuardedDll:
    """Proxy for a ctypes DLL: harmless reads pass through, calls that act on the desktop raise."""

    def __init__(self, real, name: str, blocked: set):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_blocked", blocked)

    def __getattr__(self, attr):
        if attr in self._blocked:
            return _blocked(f"{self._name}.{attr}")
        return getattr(self._real, attr)


# --------------------------------------------------------------------------- real-profile tripwire
def _profile_fingerprint() -> dict:
    out: dict = {}
    base = Path(REAL_APPDATA) / "VoxFlow" if REAL_APPDATA else None
    if base and base.is_dir():
        for p in sorted(base.rglob("*")):
            if (p.is_file() and p.suffix != ".lock" and "logs" not in p.parts and p.suffix != ".log"
                    and "cuda-cache" not in p.parts):      # the GPU driver's kernel cache, not user data
                try:
                    st = p.stat()
                    out[str(p.relative_to(base))] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    pass
    return out


def _autostart_values() -> dict:
    if sys.platform != "win32":
        return {}
    import winreg
    vals: dict = {}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            i = 0
            while True:
                try:
                    name, value, _typ = winreg.EnumValue(key, i)
                except OSError:
                    break
                if "voxflow" in name.lower() or "voxflow" in str(value).lower():
                    vals[name] = value
                i += 1
    except OSError:
        pass
    return vals


_BEFORE = {"profile": _profile_fingerprint(), "autostart": _autostart_values()}


def pytest_configure(config):
    config.addinivalue_line("markers", "desktop: needs the real desktop (clipboard, SendInput, hook, microphone); "
                                       "runs only with VOXFLOW_DESKTOP_TESTS=1")
    config.addinivalue_line("markers", "model: loads the real ASR / cleanup models (slow, needs cached weights)")
    config.addinivalue_line("markers", "bug(id): regression test for a tracked bug id")
    config.addinivalue_line("markers", "update_e2e: live updater end-to-end test (local fake GitHub over TLS, a real "
                                       "exe is started); runs only with VOXFLOW_UPDATE_E2E=1")


def pytest_collection_modifyitems(config, items):
    skip_desktop = pytest.mark.skip(reason="desktop test: set VOXFLOW_DESKTOP_TESTS=1 to run")
    skip_model = pytest.mark.skip(reason="model test skipped by VOXFLOW_SKIP_MODEL_TESTS=1")
    skip_update = pytest.mark.skip(reason="update end-to-end test: set VOXFLOW_UPDATE_E2E=1 to run")
    for item in items:
        if "update_e2e" in item.keywords and not UPDATE_E2E:
            item.add_marker(skip_update)
        if "desktop" in item.keywords and not DESKTOP:
            item.add_marker(skip_desktop)
        if "model" in item.keywords and SKIP_MODELS:
            item.add_marker(skip_model)


def pytest_sessionfinish(session, exitstatus):
    problems = []
    after_profile = _profile_fingerprint()
    if after_profile != _BEFORE["profile"]:
        changed = sorted(set(after_profile.items()) ^ set(_BEFORE["profile"].items()))
        problems.append(f"real %APPDATA%\\VoxFlow changed during the test run: {sorted({c[0] for c in changed})}")
    if _autostart_values() != _BEFORE["autostart"]:
        problems.append("the real HKCU Run (autostart) entry for VoxFlow changed during the test run")
    session.config._voxflow_tripwire = problems
    if problems and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    problems = getattr(config, "_voxflow_tripwire", [])
    if problems:
        terminalreporter.section("HERMETICITY VIOLATION", sep="!", red=True, bold=True)
        for p in problems:
            terminalreporter.write_line(p, red=True)
        terminalreporter.write_line("(if the real VoxFlow app was running during the tests this can be a false alarm)")


# --------------------------------------------------------------------------- per-test isolation
@pytest.fixture(autouse=True)
def _isolated_profile(tmp_path, monkeypatch):
    """Each test gets its own empty APPDATA, so config/history/stores never leak between tests."""
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    monkeypatch.setenv("APPDATA", str(appdata))
    yield appdata


@pytest.fixture(autouse=True)
def _no_real_cuda_probe(request, monkeypatch):
    """Outside the real-model tests the NVIDIA driver is never asked (``gpuprobe.cuda_info`` says "no driver"), so
    a test with a fake model behaves the same on a PC with an RTX card as on the CI runner. Tests that probe set
    their own answer."""
    if "model" in request.keywords:
        return
    from voxflow import gpuprobe
    monkeypatch.setattr(gpuprobe, "_cuda_cache", [None], raising=False)


@pytest.fixture
def data_dir(_isolated_profile) -> Path:
    """The VoxFlow data directory of this test (``<APPDATA>/VoxFlow``), created."""
    d = _isolated_profile / "VoxFlow"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _no_desktop_access(request, monkeypatch):
    """Block every path to the real keyboard / clipboard / hook / microphone / autostart key."""
    if "desktop" in request.keywords and DESKTOP:
        yield
        return
    if sys.platform == "win32":
        try:
            import voxflow.inject as inj
        except Exception:
            inj = None
        if inj is not None and hasattr(inj, "user32"):
            # reads (key state, sequence number, format registration) pass through; anything that types,
            # hooks or changes the clipboard raises. Tests of the Win32 layer install their own fake DLLs.
            monkeypatch.setattr(inj, "user32", GuardedDll(inj.user32, "user32", _USER32_BLOCKED))
        try:
            import voxflow.hotkey as hk
        except Exception:
            hk = None
        if hk is not None and hasattr(getattr(hk, "HotkeyListener", None), "_run"):
            def no_hook(self):
                # runs on the hook thread: report failure the way a refused SetWindowsHookEx would
                self.error = "the global keyboard hook is blocked in tests"
                self._ready.set()

            # fake-DLL tests of the thread body reach the original through this name
            monkeypatch.setattr(hk.HotkeyListener, "_run_unguarded", hk.HotkeyListener._run, raising=False)
            monkeypatch.setattr(hk.HotkeyListener, "_run", no_hook)
        import winreg
        real_open, real_create = winreg.OpenKey, winreg.CreateKeyEx

        def guarded(real):
            def opener(key, sub_key=None, *a, **k):
                access = k.get("access", a[1] if len(a) > 1 else winreg.KEY_READ)
                writes = access & (winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY | winreg.KEY_WRITE
                                   | winreg.KEY_ALL_ACCESS) & ~winreg.KEY_READ
                if key == winreg.HKEY_CURRENT_USER and isinstance(sub_key, str) \
                        and sub_key.lower().endswith(r"currentversion\run") and writes:
                    _blocked("writing the real HKCU Run (autostart) key")()
                return real(key, sub_key, *a, **k)
            return opener

        monkeypatch.setattr(winreg, "OpenKey", guarded(real_open))
        monkeypatch.setattr(winreg, "OpenKeyEx", guarded(winreg.OpenKeyEx))
        monkeypatch.setattr(winreg, "CreateKeyEx", guarded(real_create))
        monkeypatch.setattr(winreg, "CreateKey",
                            lambda key, sub_key: guarded(lambda k, s: winreg.CreateKeyEx(k, s))(
                                key, sub_key, 0, winreg.KEY_WRITE))
    try:
        import sounddevice as sd
    except Exception:
        sd = None
    if sd is not None:
        monkeypatch.setattr(sd, "InputStream", _blocked("the real microphone (sounddevice.InputStream)"))
        if hasattr(sd, "OutputStream"):
            monkeypatch.setattr(sd, "OutputStream", _blocked("the real speakers (sounddevice.OutputStream)"))
        if hasattr(sd, "play"):
            monkeypatch.setattr(sd, "play", _blocked("the real speakers (sounddevice.play)"))
    yield


@pytest.fixture
def qt_messages(qapp):
    """Collect Qt warnings/criticals emitted during a test; ``assert not qt_messages`` proves a clean run."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    seen: list[str] = []

    def handler(mode, _ctx, message):
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            # the offscreen platform has no system tray / native theme: not an application defect
            if "propagateSizeHints" in message or "This plugin does not support" in message:
                return
            seen.append(message)

    previous = qInstallMessageHandler(handler)
    try:
        yield seen
    finally:
        qInstallMessageHandler(previous)
