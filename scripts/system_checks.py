"""Real-system checks that need no foreground window (safe on a locked session).

    python scripts/system_checks.py mute        # real Core Audio: mute other apps, verify, restore, verify
    python scripts/system_checks.py autostart   # real HKCU Run key round trip under a THROWAWAY value name
    python scripts/system_checks.py download    # first run with an EMPTY temp Hugging Face cache: progress + load
    python scripts/system_checks.py soak --minutes 30   # repeated dictations, memory / handles / threads stay flat
    python scripts/system_checks.py screen      # real screen(s): scaling, native window geometry, Flow bar, tray

Each check uses a temp profile, restores what it changes and prints PASSED / FAILED. ``--json FILE`` saves details.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# =========================================================================== mute
TONE_CODE = (
    "import sys,time,numpy as np,sounddevice as sd;"
    "t=np.arange(int(48000*float(sys.argv[1])))/48000;"
    "import os;sd.play((0.02*np.sin(2*np.pi*440*t)).astype('float32'),48000);print(os.getpid(),flush=True);"
    "time.sleep(float(sys.argv[1]))")


def _sessions(backend) -> list[dict]:
    """Every audio session with its process id, state and mute flag (read-only)."""
    out: list[dict] = []

    def work():
        for volume, pid, session_id, state in backend._sessions():
            muted = ctypes.c_int()
            volume.call(6, ctypes.byref(muted), argtypes=(ctypes.POINTER(ctypes.c_int),))
            out.append({"pid": pid, "id": session_id, "state": state, "muted": bool(muted.value)})

    backend._with_com(work)
    return out


def check_mute(args) -> tuple[bool, dict]:
    from voxflow.sounds import CoreAudioBackend, MediaMuter
    state_dir = Path(tempfile.mkdtemp(prefix="voxflow-mute-"))
    backend = CoreAudioBackend()
    child = subprocess.Popen([sys.executable, "-c", TONE_CODE, "14"], stdout=subprocess.PIPE, text=True)
    info: dict = {}
    muter = MediaMuter(state_dir, backend=backend, threaded=False)
    ok = True
    try:
        # a venv python.exe is a launcher: the process that owns the audio session is its child
        tone_pid = int(child.stdout.readline().strip())
        info["tone_pid"] = tone_pid
        time.sleep(1.0)
        before = _sessions(backend)
        mine = [s for s in before if s["pid"] == tone_pid]
        info["sessions_before"] = [(s["pid"], s["state"], s["muted"]) for s in before]
        print(f"sessions before: {len(before)} (active {sum(s['state'] == 1 for s in before)}); "
              f"our tone process {tone_pid}: {mine}")
        if not mine or mine[0]["state"] != 1:
            print("the test tone did not show up as an active audio session")
            ok = False
        muter.mute()
        during = _sessions(backend)
        marker = state_dir / "muted_sessions.json"
        muted_ids = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else []
        info["muted_count"] = len(muted_ids)
        mine_during = [s for s in during if s["pid"] == tone_pid]
        own = [s for s in during if s["pid"] == os.getpid()]
        print(f"muted {len(muted_ids)} session(s); our tone muted: {[s['muted'] for s in mine_during]}; "
              f"marker file written: {marker.exists()}")
        ok &= bool(mine_during) and all(s["muted"] for s in mine_during) and marker.exists()
        ok &= not any(s["muted"] for s in own)
        # sessions the user had muted before must stay exactly as they were
        pre_muted = {s["id"] for s in before if s["muted"]}
        ok &= all(i not in pre_muted for i in muted_ids)

        # a killed VoxFlow leaves the marker behind: the next start must undo the mute
        survivor = MediaMuter(state_dir, backend=backend, threaded=False)
        survivor.recover()
        after_recover = _sessions(backend)
        mine_after = [s for s in after_recover if s["pid"] == tone_pid]
        print(f"after recover() from the marker file: our tone muted: {[s['muted'] for s in mine_after]}; "
              f"marker removed: {not marker.exists()}")
        ok &= bool(mine_after) and not any(s["muted"] for s in mine_after) and not marker.exists()

        muter._muted = []                       # the first muter's list was handled by recover()
        muter.mute()
        muter.unmute()                          # the normal path
        after = _sessions(backend)
        by_id_before = {(s["pid"], s["id"]): s["muted"] for s in before}
        changed = [(s["pid"], s["muted"]) for s in after
                   if (s["pid"], s["id"]) in by_id_before and by_id_before[(s["pid"], s["id"])] != s["muted"]]
        info["left_changed"] = changed
        print(f"after mute + unmute: sessions whose mute state differs from the start: {changed}")
        ok &= not changed
    finally:
        try:
            muter.restore_now()
            leftover = [s["id"] for s in _sessions(backend) if s["pid"] == info.get("tone_pid") and s["muted"]]
            if leftover:
                backend.unmute(leftover)
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True)
            shutil.rmtree(state_dir, ignore_errors=True)
    return bool(ok), info


# =========================================================================== autostart
def check_autostart(args) -> tuple[bool, dict]:
    import winreg

    from voxflow import system
    name = f"VoxFlow-QA-{uuid.uuid4().hex[:8]}"
    info: dict = {"value_name": name}

    def read(value_name):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, system.RUN_KEY, 0, winreg.KEY_READ) as key:
                return winreg.QueryValueEx(key, value_name)
        except FileNotFoundError:
            return None

    real_before = read("VoxFlow")
    original = system.RUN_VALUE
    system.RUN_VALUE = name                       # everything below goes through the app's own code path
    ok = True
    profile = Path(tempfile.mkdtemp(prefix="voxflow-autostart-"))
    proc = None
    try:
        ok &= system.autostart_enabled() is False
        system.set_autostart(True)
        stored = read(name)
        info["stored"] = stored[0] if stored else None
        print(f"Run\\{name} = {stored[0] if stored else None!r}")
        ok &= stored is not None and stored[1] == winreg.REG_SZ and stored[0] == system.launch_command()
        ok &= stored is not None and len(stored[0]) <= 260
        ok &= system.autostart_enabled() is True

        # run exactly the string Windows would run at login (temp profile, offscreen), for a few seconds
        env = dict(os.environ, APPDATA=str(profile), QT_QPA_PLATFORM="offscreen",
                   QT_QPA_FONTDIR=r"C:\Windows\Fonts", VOXFLOW_QA="1")
        proc = subprocess.Popen(stored[0], env=env, cwd=os.environ.get("SystemRoot", r"C:\Windows") + r"\System32")
        deadline = time.time() + 40
        log = profile / "VoxFlow" / "logs" / "voxflow.log"
        started = False
        while time.time() < deadline and proc.poll() is None:
            if log.exists() and "starting" in log.read_text(encoding="utf-8", errors="replace"):
                started = True
                break
            time.sleep(0.5)
        info["launched"] = started
        info["exit_code_while_waiting"] = proc.poll()
        print(f"the stored command started VoxFlow from System32 with a temp profile: {started} "
              f"(process alive: {proc.poll() is None})")
        ok &= started and proc.poll() is None
        if started:
            cfg = json.loads((profile / "VoxFlow" / "config.json").read_text(encoding="utf-8"))
            info["profile_files"] = sorted(p.name for p in (profile / "VoxFlow").iterdir())
            ok &= cfg["config_version"] == 2

        system.set_autostart(False)
        ok &= read(name) is None and system.autostart_enabled() is False
        system.set_autostart(False)               # removing twice is fine
        print(f"value removed: {read(name) is None}")
    finally:
        system.RUN_VALUE = original
        if proc is not None and proc.poll() is None:
            proc.kill()
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, system.RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass
        time.sleep(0.5)
        shutil.rmtree(profile, ignore_errors=True)
    untouched = read("VoxFlow") == real_before
    print(f"the user's own Run\\VoxFlow value is untouched: {untouched}")
    return bool(ok and untouched and read(name) is None), info


# =========================================================================== first-run download
def check_download(args) -> tuple[bool, dict]:
    if not args.child:
        cache = Path(tempfile.mkdtemp(prefix="voxflow-hfcache-"))
        profile = Path(tempfile.mkdtemp(prefix="voxflow-dl-"))
        env = dict(os.environ, HF_HUB_CACHE=str(cache), HF_HOME=str(cache.parent / (cache.name + "-home")),
                   APPDATA=str(profile), QT_QPA_PLATFORM="offscreen", QT_QPA_FONTDIR=r"C:\Windows\Fonts")
        for var in ("HF_HUB_OFFLINE", "HUGGINGFACE_HUB_CACHE"):
            env.pop(var, None)
        out_file = profile / "result.json"
        try:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "download", "--child",
                                  "--json", str(out_file)], env=env, cwd=str(ROOT))
            info = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {"error": f"exit {rc}"}
            size = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
            info["cache_bytes"] = size
            print(f"temp cache afterwards: {size / 1e9:.2f} GB")
        finally:
            shutil.rmtree(cache, ignore_errors=True)
            shutil.rmtree(profile, ignore_errors=True)
        return bool(info.get("ok")), info

    from PySide6.QtWidgets import QApplication

    from voxflow import fakes
    from voxflow.context import AppContext
    from voxflow.controller import Controller
    qapp = QApplication.instance() or QApplication([])
    t0 = time.time()
    ctl = Controller(data_dir=Path(os.environ["APPDATA"]) / "VoxFlow", install_hook=False,
                     injector=fakes.FakeInjector(), sound_backend=lambda wav: None,
                     mute_backend=fakes.FakeMuteBackend(), context_fn=lambda: AppContext("notepad.exe", "x", 1),
                     autostart=fakes.FakeAutostart(False))
    events: list = []
    ctl.model_status_changed.connect(lambda name, st: events.append((round(time.time() - t0, 1), name, dict(st))))
    statuses: list = []
    ctl.status_changed.connect(lambda st: statuses.append(st["title"]))
    ctl.start()
    deadline = time.time() + args.timeout
    done = lambda: all(ctl.model_status(n)["state"] in ("ready", "degraded", "failed") for n in ("asr", "cleanup"))  # noqa: E731
    while time.time() < deadline and not done():
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    info: dict = {"seconds": round(time.time() - t0, 1), "final": {n: ctl.model_status(n) for n in ("asr", "cleanup")}}
    ok = True
    for name in ("asr", "cleanup"):
        seq = [(t, st["state"], st["progress"], st["detail"]) for t, n, st in events if n == name]
        states = [s for _t, s, _p, _d in seq]
        prog = [p for _t, s, p, _d in seq if s == "downloading" and p is not None]
        info[name] = {"states": [s for i, s in enumerate(states) if i == 0 or states[i - 1] != s],
                      "progress_events": len(prog), "first_progress": prog[:1], "last_progress": prog[-1:],
                      "monotonic": all(b >= a for a, b in zip(prog, prog[1:])),
                      "max_progress": max(prog) if prog else None,
                      "sample_details": [d for _t, _s, _p, d in seq][:2] + [d for _t, _s, _p, d in seq][-2:],
                      "ready_at_s": next((t for t, s, _p, _d in seq if s in ("ready", "degraded", "failed")), None)}
        print(f"{name}: states {info[name]['states']}, {len(prog)} progress events "
              f"({prog[0] if prog else None} .. {prog[-1] if prog else None}), monotonic {info[name]['monotonic']}, "
              f"done after {info[name]['ready_at_s']} s -> {ctl.model_status(name)}")
        ok &= states[:1] == ["downloading"] and states[-1:] == ["ready"] and len(prog) >= 3
        ok &= info[name]["monotonic"] and (max(prog) if prog else 2) <= 0.99
    info["status_titles"] = [t for i, t in enumerate(statuses) if i == 0 or statuses[i - 1] != t]
    print("status line went through:", info["status_titles"])
    ok &= "Downloading speech model…" in statuses and ctl.status()["level"] == "ok"

    # the freshly downloaded models must actually work
    import numpy as np

    from voxflow import app as appmod
    clip = appmod.read_wav_16k(str(ROOT / "tests" / "tts_sample.wav"))

    class Clip(fakes.FakeRecorder):
        def stop(self):
            self._rec, self._last_speech_s = False, len(clip) / 16000
            return clip.copy()

        @property
        def hit_cap(self):
            return False

    ctl._Rec = Clip
    ctl._switch_recorder(None)
    got: list = []
    ctl.transcript_ready.connect(got.append)
    ctl._on_hotkey("dictate", "start")
    ctl._t_start -= 5
    ctl._on_hotkey("dictate", "stop")
    end = time.time() + 60
    while time.time() < end and not got and ctl.state != "error":
        qapp.processEvents()
        time.sleep(0.01)
    info["first_dictation"] = got[0]["text"] if got else f"(none, state {ctl.state}: {ctl.message})"
    print("first dictation with the downloaded models:", repr(info["first_dictation"]))
    ok &= bool(got) and "quick brown fox" in got[0]["text"].lower()
    ctl.shutdown()
    info["ok"] = bool(ok)
    if args.json:
        Path(args.json).write_text(json.dumps(info, indent=2), encoding="utf-8")
    _ = np
    return bool(ok), info


# =========================================================================== soak
def _process_counters() -> dict:
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    handles = wintypes.DWORD()
    k32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessHandleCount(k32.GetCurrentProcess(), ctypes.byref(handles))
    u32 = ctypes.WinDLL("user32")
    u32.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    import threading
    return {"rss_mb": round(pmc.WorkingSetSize / 1e6, 1), "private_mb": round(pmc.PrivateUsage / 1e6, 1),
            "handles": int(handles.value), "threads": threading.active_count(),
            "gdi": int(u32.GetGuiResources(k32.GetCurrentProcess(), 0)),
            "user": int(u32.GetGuiResources(k32.GetCurrentProcess(), 1))}


def _gpu_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            pid, _, mem = line.partition(",")
            if pid.strip() == str(os.getpid()) and mem.strip().isdigit():
                return float(mem.strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def check_soak(args) -> tuple[bool, dict]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    import statistics

    from PySide6.QtWidgets import QApplication

    from voxflow import app as appmod
    from voxflow import fakes
    from voxflow.context import AppContext
    from voxflow.controller import Controller
    from voxflow.ui.shell import build_ui
    qapp = QApplication.instance() or QApplication([])
    profile = Path(tempfile.mkdtemp(prefix="voxflow-soak-"))
    (profile / "config.json").write_text(json.dumps({"onboarding_done": True, "mute_media": False}), encoding="utf-8")
    clip = appmod.read_wav_16k(str(ROOT / "tests" / "tts_sample.wav"))
    messy = ("um so i think we should uh move the the meeting to thursday no wait friday at three thirty and uh send "
             "the report to the engineering team before the end of the day you know so they can review it")

    class Clip(fakes.FakeRecorder):
        def stop(self):
            was, self._rec = self._rec, False
            self._last_speech_s = len(clip) / 16000 if was else 0.0
            return clip.copy() if was else clip[:0]

        @property
        def hit_cap(self):
            return False

    from voxflow.asr import Transcriber
    from voxflow.cleanup import Cleaner
    asr, cleaner = Transcriber(), Cleaner()
    real_transcribe = asr.transcribe
    flip = {"n": 0}

    def transcribe(audio):                       # every second utterance needs the language model
        text = real_transcribe(audio)
        flip["n"] += 1
        return messy if flip["n"] % 2 == 0 else text

    asr.transcribe = transcribe
    injector = fakes.FakeInjector(selection="hey bob can u send me the report by friday thx")
    ctl = Controller(data_dir=profile, install_hook=False, modules=(Clip, asr, cleaner), injector=injector,
                     sound_backend=lambda wav: None, mute_backend=fakes.FakeMuteBackend(),
                     context_fn=lambda: AppContext("winword.exe", "Report.docx - Word", 1),
                     autostart=fakes.FakeAutostart(False))
    ui = build_ui(ctl, show_window=True)           # the whole UI is alive and repainting during the soak
    done: list = []
    ctl.transcript_ready.connect(done.append)
    errors: list = []
    ctl.state_changed.connect(lambda s, m, msg: errors.append(msg) if s == "error" else None)
    ctl.start()

    def pump(until, timeout_s):
        end = time.time() + timeout_s
        while time.time() < end:
            qapp.processEvents()
            if until():
                return True
            time.sleep(0.004)
        return False

    pump(lambda: ctl.model_status("asr")["state"] == "ready" and ctl.model_status("cleanup")["state"] == "ready", 180)
    pages = ["home", "history", "dictionary", "snippets", "style", "settings"]
    samples: list = []
    latencies: list = []
    t0 = time.time()
    next_sample = 0.0
    i = 0
    minutes = float(args.minutes)
    while time.time() - t0 < minutes * 60:
        i += 1
        mode = "command" if i % 10 == 0 else "dictate"
        n = len(done)
        ctl._on_hotkey(mode, "start")
        ctl._t_start -= len(clip) / 16000
        pump(lambda: False, 0.05)
        ctl._on_hotkey(mode, "stop")
        if pump(lambda: len(done) > n or ctl.state == "error", 60) and len(done) > n:
            latencies.append(done[-1]["latency_ms"])
        pump(lambda: ctl.state == "idle", 5)
        if i % 7 == 0:
            ui.window.show_page(pages[(i // 7) % len(pages)])
        if i % 25 == 0:
            ctl.set("theme", "light" if (i // 25) % 2 else "dark")
        if i % 40 == 0:
            ctl.paste_last()
        elapsed = time.time() - t0
        if elapsed >= next_sample:
            row = dict(_process_counters(), minute=round(elapsed / 60, 1), dictations=i, gpu_mb=_gpu_mb(),
                       history=ctl.history.count(),
                       latency_median_ms=round(statistics.median(latencies[-50:])) if latencies else None)
            samples.append(row)
            print(row, flush=True)
            next_sample += 60.0
    row = dict(_process_counters(), minute=round((time.time() - t0) / 60, 1), dictations=i, gpu_mb=_gpu_mb(),
               history=ctl.history.count(), latency_median_ms=round(statistics.median(latencies[-50:])))
    samples.append(row)
    print(row, flush=True)
    ui.close()
    ctl.shutdown()
    shutil.rmtree(profile, ignore_errors=True)

    # steady state = after the first 20 % of the run; compare the first and the last third of what remains
    steady = samples[max(1, len(samples) // 5):]
    third = max(1, len(steady) // 3)
    first, last = steady[:third], steady[-third:]

    def mean(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else 0.0

    growth = {k: round(mean(last, k) - mean(first, k), 1) for k in ("rss_mb", "private_mb", "handles", "threads",
                                                                    "gdi", "user", "gpu_mb")}
    info = {"minutes": round((time.time() - t0) / 60, 1), "dictations": i, "completed": len(done),
            "errors": errors[:10], "error_count": len(errors), "samples": samples, "growth_steady_state": growth,
            "latency_ms": {"median": round(statistics.median(latencies)), "p95": round(sorted(latencies)[int(len(latencies) * 0.95)]),
                           "max": max(latencies), "first_50_median": round(statistics.median(latencies[:50])),
                           "last_50_median": round(statistics.median(latencies[-50:]))},
            "history_entries": samples[-1]["history"]}
    print("growth between the first and last third of the steady state:", growth)
    print("latency:", info["latency_ms"], "| errors:", len(errors), "| dictations:", i, "completed:", len(done))
    ok = (len(errors) <= max(1, i // 200) and len(done) >= i * 0.99
          and growth["private_mb"] < 60 and growth["handles"] < 60 and growth["threads"] <= 2
          and growth["gdi"] < 30 and growth["user"] < 30
          and info["latency_ms"]["last_50_median"] <= info["latency_ms"]["first_50_median"] * 1.5 + 30)
    return bool(ok), info


# =========================================================================== real screen
def check_screen(args) -> tuple[bool, dict]:
    """The UI on the real Windows platform plugin (not offscreen): screens, scaling, native window geometry,
    Flow bar window styles, tray icon. Needs no foreground window and sends no input."""
    os.environ.pop("QT_QPA_PLATFORM", None)
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    from voxflow.controller import Controller
    from voxflow.ui.shell import build_ui
    qapp = QApplication.instance() or QApplication([])
    profile = Path(tempfile.mkdtemp(prefix="voxflow-screen-"))
    (profile / "config.json").write_text(json.dumps({"onboarding_done": True, "show_idle_pill": True}),
                                         encoding="utf-8")
    u32 = ctypes.WinDLL("user32")
    u32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    u32.GetForegroundWindow.restype = ctypes.c_void_p

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            qapp.processEvents()
            time.sleep(0.005)

    ok = True
    info: dict = {"platform": QGuiApplication.platformName()}
    ok &= info["platform"] == "windows"
    screens = QGuiApplication.screens()
    info["screens"] = [{"name": s.name(), "geometry": s.geometry().getRect(), "available": s.availableGeometry().getRect(),
                        "scale": s.devicePixelRatio(), "dpi": round(s.logicalDotsPerInch())} for s in screens]
    print(f"platform {info['platform']}, {len(screens)} screen(s):")
    for s in info["screens"]:
        print("  ", s)
    fg_before = u32.GetForegroundWindow()
    ctl = Controller(fake=True, data_dir=profile, install_hook=False)
    ui = build_ui(ctl, show_window=True)
    ctl.start()
    pump(1.5)
    try:
        win, bar = ui.window, ui.overlay

        def inside(rect: QRect, screen) -> bool:
            return screen.availableGeometry().adjusted(-12, -12, 12, 12).contains(rect)

        frame = win.frameGeometry()
        home = win.screen()
        info["window"] = {"frame": frame.getRect(), "screen": home.name(), "min": (win.minimumWidth(), win.minimumHeight())}
        fits = win.isVisible() and inside(frame, home)
        print(f"main window visible {win.isVisible()}, frame {frame.getRect()} fits its screen: {fits}")
        ok &= fits
        ok &= win.minimumWidth() <= home.availableGeometry().width() and win.minimumHeight() <= home.availableGeometry().height()

        per_screen = []
        for s in screens:                              # native move to every monitor, native resize, maximise, restore
            a = s.availableGeometry()
            win.showNormal()
            win.setGeometry(a.x() + 40, a.y() + 60, win.minimumWidth(), win.minimumHeight())
            pump(0.5)
            moved = win.screen().name() == s.name() and inside(win.frameGeometry(), s)
            shot = win.grab()
            scale_ok = abs(shot.devicePixelRatio() - s.devicePixelRatio()) < 0.01 and \
                abs(shot.width() - round(win.width() * s.devicePixelRatio())) <= 2
            img = shot.toImage()
            colours = {img.pixel(x, y) for x in range(0, img.width(), max(1, img.width() // 24))
                       for y in range(0, img.height(), max(1, img.height() // 24))}
            win.showMaximized()
            pump(0.6)
            g = win.geometry()
            maxed = win.isMaximized() and abs(g.width() - a.width()) <= 4 and abs(g.height() - a.height()) <= 40
            win.showNormal()
            pump(0.4)
            restored = not win.isMaximized() and inside(win.frameGeometry(), s)
            row = {"screen": s.name(), "scale": s.devicePixelRatio(), "moved": moved, "render_matches_scale": scale_ok,
                   "distinct_colours": len(colours), "maximised": maxed, "restored": restored}
            per_screen.append(row)
            print("  ", row)
            ok &= moved and scale_ok and len(colours) >= 8 and maxed and restored
        info["per_screen"] = per_screen

        # Flow bar: every state on the real compositor, never focusable, never outside the screen
        hwnd = int(bar.winId())
        ex = u32.GetWindowLongW(hwnd, -20) & 0xFFFFFFFF
        styles = {"topmost": bool(ex & 0x8), "transparent_for_input": bool(ex & 0x20), "tool_window": bool(ex & 0x80),
                  "no_activate": bool(ex & 0x08000000), "layered": bool(ex & 0x00080000)}
        info["flow_bar_styles"] = styles
        print("Flow bar window styles:", styles)
        ok &= styles["topmost"] and styles["transparent_for_input"] and styles["no_activate"] and styles["tool_window"]
        bar_rows = []
        for state, text, mode in (("idle", "", "dictate"), ("listening", "", "dictate"), ("locked", "", "dictate"),
                                  ("processing", "", "dictate"), ("done", "", "dictate"),
                                  ("error", "Microphone unavailable", "dictate"), ("listening", "", "command"),
                                  ("loading", "Loading speech model…", "dictate")):
            bar.set_state(state, text, mode)
            pump(0.45)
            g = bar.frameGeometry()
            # the bar's window carries a transparent margin for its glow, so compare with the whole screen and
            # check that what is actually painted (non-transparent pixels) sits above the taskbar
            img = bar.grab().toImage()
            rows = [y for y in range(img.height())
                    if any((img.pixel(x, y) >> 24) & 0xFF > 40 for x in range(0, img.width(), 3))]
            scale = img.devicePixelRatio() or 1.0
            painted_bottom = g.y() + (max(rows) / scale if rows else 0)
            scr = bar.screen()
            on_screen = (bar.isVisible() and scr.geometry().contains(g) and bool(rows)
                         and painted_bottom <= scr.availableGeometry().bottom() + 1)
            took_focus = u32.GetForegroundWindow() == hwnd
            bar_rows.append({"state": f"{state}/{mode}", "geometry": g.getRect(), "on_screen": on_screen,
                             "painted_bottom": round(painted_bottom), "available_bottom": scr.availableGeometry().bottom(),
                             "took_focus": took_focus})
            ok &= on_screen and not took_focus
        info["flow_bar"] = bar_rows
        print("Flow bar states (state, on screen, painted bottom edge, bottom of the usable screen):",
              [(r["state"], r["on_screen"], r["painted_bottom"], r["available_bottom"]) for r in bar_rows])

        tray_ok = QSystemTrayIcon.isSystemTrayAvailable() and ui.tray.icon.isVisible()
        info["tray"] = {"available": QSystemTrayIcon.isSystemTrayAvailable(), "visible": ui.tray.icon.isVisible()}
        print("tray icon:", info["tray"])
        ok &= tray_ok
        info["foreground_changed_to_flow_bar"] = u32.GetForegroundWindow() == hwnd
        info["foreground_before"] = fg_before
    finally:
        ui.close()
        ctl.shutdown()
        pump(0.3)
        shutil.rmtree(profile, ignore_errors=True)
    return bool(ok), info


# =========================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["mute", "autostart", "download", "soak", "screen"])
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=1500.0, help="download: seconds to wait for both models")
    ap.add_argument("--json", default="")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    ok, info = {"mute": check_mute, "autostart": check_autostart, "download": check_download,
                "soak": check_soak, "screen": check_screen}[args.check](args)
    if args.json and not (args.check == "download" and args.child):
        Path(args.json).write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"{args.check.upper()} CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
