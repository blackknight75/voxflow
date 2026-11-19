"""Record docs/assets/demo.gif: one real dictation on the REAL desktop, with the live-preview caption.

    python scripts/record_demo.py                 # writes docs/assets/demo.gif
    python scripts/record_demo.py --keep-video    # also keeps the lossless capture in build/demo/

What is real: the WH_KEYBOARD_LL hook and the dictation hotkey (held with SendInput, like a keyboard), the speech
model, the live preview, the language model, the clipboard, the Ctrl+V paste into a note page in a Microsoft Edge
app window, and the Flow bar. What is a stand-in: the microphone. A speech clip from Windows' offline speech engine
is fed, in real time and in 512-sample blocks, into the same audio callback a microphone would call, so the Flow
bar's level meter, the pre-roll, the live preview and the final transcript all see it as microphone audio.

The scene (a dark backdrop with the spoken sentence, the note window, the Flow bar under it) is placed in the middle
of the primary screen and captured with ffmpeg's gdigrab (ffmpeg must be on PATH). It takes over that part of the
screen for about 30 seconds after the models are loaded: do not touch the keyboard or mouse meanwhile.

Safety, as in scripts/live_e2e.py (``tests/desktop_kit.py``): keys and the paste are sent only while a window this
script created is the foreground window; the user's clipboard is saved and restored; everything runs in a temp
profile (never %APPDATA%/VoxFlow) and a throwaway browser profile; if another VoxFlow is running, private shortcuts
are used so the user's copy never fires.
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
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

SPOKEN = ("um, so I think we should, uh, move the the meeting to Thursday, no wait, Friday at three thirty, and uh, "
          "send the report to the engineering team before the end of the day, you know, so they can review it")
VOICE = "Microsoft Zira Desktop"               # the published take; David heard "move the the" as "move to the"
SCENE_W, SCENE_H = 1100, 680                  # logical pixels; the GIF is this size
NOTE = (67, 118, 966, 402)                     # note window inside the scene: x, y, w, h (logical)
PILL_BOTTOM = 650                              # Flow bar anchor (bottom centre of the pill), scene y
BACKDROP = "#13161c"
FPS = 15
SRATE, BLOCK = 16000, 512


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "demo.gif"))
    ap.add_argument("--voice", default="", help=f"Windows speech voice (default: {VOICE} if installed, else the "
                                                "first installed one)")
    ap.add_argument("--keep-video", action="store_true")
    args = ap.parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 1

    os.environ.pop("QT_QPA_PLATFORM", None)
    work = Path(tempfile.mkdtemp(prefix="voxflow-demo-"))
    os.environ["APPDATA"] = str(work / "appdata")
    (work / "appdata").mkdir()
    try:
        return _record(args, ffmpeg, work)
    finally:
        _kill_by_commandline(str(work))
        shutil.rmtree(work, ignore_errors=True)


def _kill_by_commandline(fragment: str) -> None:
    from scripts.live_e2e import kill_by_commandline
    kill_by_commandline(fragment)


def _record(args, ffmpeg: str, work: Path) -> int:
    import numpy as np
    import desktop_kit as kit
    from scripts.live_e2e import DEFAULT_KEYS, PRIVATE_KEYS, other_voxflow_running
    from scripts.speech_matrix import tts, voices
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QPainter
    from PySide6.QtWidgets import QApplication, QWidget

    from voxflow import app as appmod
    from voxflow import audio as audiomod
    from voxflow import fakes
    from voxflow.asr import Transcriber
    from voxflow.cleanup import Cleaner
    from voxflow.controller import Controller
    from voxflow.inject import Injector
    from voxflow.ui.shell import build_ui

    user32 = kit.user32
    qapp = QApplication.instance() or QApplication([])

    def pump(seconds: float, until=None) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            qapp.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.005)
        return until is None

    # ------------------------------------------------------------------ the speech clip
    wav = work / "spoken.wav"
    installed = voices()
    tts(SPOKEN, args.voice or (VOICE if VOICE in installed else installed[0]), wav)
    clip = appmod.read_wav_16k(str(wav))
    clip_s = len(clip) / SRATE
    print(f"speech clip: {clip_s:.1f} s")

    class ClipStream:
        """Stands where sounddevice's InputStream is: calls the recorder's audio callback every 32 ms, with the clip
        while a recording is running (from its first sample) and near-silence otherwise."""

        def __init__(self, rec):
            self.rec, self.active = rec, True
            self._rng = np.random.default_rng(7)
            threading.Thread(target=self._run, name="demo-clip-stream", daemon=True).start()

        def _run(self) -> None:
            pos, nxt = None, time.perf_counter()
            while self.active:
                if self.rec.is_recording:
                    pos = 0 if pos is None else pos
                else:
                    pos = None
                if pos is not None and pos < len(clip):
                    x = clip[pos:pos + BLOCK]
                    pos += BLOCK
                    if x.size < BLOCK:
                        x = np.concatenate([x, np.zeros(BLOCK - x.size, dtype=np.float32)])
                else:
                    x = (self._rng.standard_normal(BLOCK) * 2e-4).astype(np.float32)
                self.rec._callback(x.reshape(-1, 1), BLOCK, None, None)
                nxt += BLOCK / SRATE
                time.sleep(max(0.0, nxt - time.perf_counter()))

        def stop(self) -> None:
            self.active = False

        close = stop

    class ClipRecorder(audiomod.Recorder):
        def _open(self, device):
            return ClipStream(self)

    # ------------------------------------------------------------------ the scene
    screen = qapp.primaryScreen()
    dpr = screen.devicePixelRatio()
    avail = screen.availableGeometry()
    x0 = avail.x() + (avail.width() - SCENE_W) // 2
    y0 = avail.y() + (avail.height() - SCENE_H) // 2

    class Backdrop(QWidget):
        def paintEvent(self, _):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(BACKDROP))
            f = QFont("Segoe UI")
            f.setPixelSize(20)
            p.setFont(f)
            p.setPen(QColor("#cfe0ee"))
            p.drawText(QRect(NOTE[0] + 1, 10, NOTE[2], 100), int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                       f'Speaking: "{SPOKEN}"')
            p.end()

    backdrop = Backdrop()
    backdrop.setWindowFlags(Qt.WindowType.FramelessWindowHint)
    backdrop.setWindowTitle("VoxFlow demo backdrop")
    backdrop.setGeometry(x0, y0, SCENE_W, SCENE_H)
    backdrop.show()
    pump(0.5)

    token = uuid.uuid4().hex[:10]
    mark = f"vfdemo-{token}"
    page = work / "note.html"
    page.write_text(f"""<!doctype html><meta charset="utf-8"><title>{mark}</title>
<style>
html,body{{margin:0;height:100%;background:#f3f5f7;font-family:"Segoe UI",sans-serif}}
body{{box-sizing:border-box;padding:18px 26px 22px;display:flex;flex-direction:column}}
label{{font-size:15px;font-weight:600;color:#5b6b80;margin:0 0 10px 1px}}
textarea{{flex:1;resize:none;outline:none;border:1px solid #9cc7ee;border-radius:8px;background:#fff;
  padding:14px 18px;font:24px/1.45 "Segoe UI",sans-serif;color:#1d2733}}
textarea::placeholder{{color:#5f7489}}
</style>
<label for="t">New note</label>
<textarea id="t" autofocus spellcheck="false" placeholder="Hold Ctrl+Win and speak"></textarea>
<script>
const t = document.getElementById('t');
function up() {{ if (document.hasFocus() && document.activeElement === t) document.title = 'Notes'; }}
['focus'].forEach(e => t.addEventListener(e, up));
window.addEventListener('focus', () => {{ t.focus(); up(); }});
setInterval(up, 150);
</script>""", encoding="utf-8")
    edge = _find_browser()
    if edge is None:
        print("no Chromium browser found (Edge or Chrome)", file=sys.stderr)
        return 1
    subprocess.Popen([str(edge), f"--user-data-dir={work / 'browser-profile'}", "--no-first-run",
                      "--no-default-browser-check", "--disable-sync", "--disable-extensions",
                      "--disable-features=msEdgeSidebarV2,Translate", f"--app={page.as_uri()}"])
    hwnd = 0
    end = time.time() + 40
    while time.time() < end and not hwnd:
        found = kit.find_windows(lambda h, t, c: c.startswith("Chrome_WidgetWin") and t.startswith(mark))
        hwnd = found[0] if found else 0
        pump(0.2)
    if not hwnd:
        print("the note window did not appear within 40 s", file=sys.stderr)
        return 1
    _place_visible_frame(user32, hwnd, round((x0 + NOTE[0]) * dpr), round((y0 + NOTE[1]) * dpr),
                         round(NOTE[2] * dpr), round(NOTE[3] * dpr))
    own = {hwnd}
    allowed = lambda: [h for h in own if user32.IsWindow(h)]  # noqa: E731
    kb = kit.Keyboard(allowed)

    # ------------------------------------------------------------------ the real app
    profile = work / "profile"
    profile.mkdir()
    others = other_voxflow_running()
    keys = PRIVATE_KEYS if others else DEFAULT_KEYS
    if others:
        print(f"another VoxFlow is running (pid {others}): private shortcuts are used")
    (profile / "config.json").write_text(json.dumps(dict(
        keys, onboarding_done=True, mute_media=False, sounds_enabled=False, launch_at_login=False,
        start_minimized=True, live_preview=True, overlay_position=[x0 + SCENE_W // 2, y0 + PILL_BOTTOM])),
        encoding="utf-8")

    with kit.saved_clipboard() as clipboard:
        if clipboard is None:
            print("BLOCKED: the clipboard could not be saved, so nothing was run", file=sys.stderr)
            return 3
        injector = Injector(keys=kit.guarded_keys(allowed))
        ctl = Controller(data_dir=profile, install_hook=True, modules=(ClipRecorder, Transcriber(), Cleaner()),
                         injector=injector, sound_backend=lambda wav: None, mute_backend=fakes.FakeMuteBackend(),
                         autostart=fakes.FakeAutostart(False))
        ui = build_ui(ctl, show_window=False)
        done: list[dict] = []
        ctl.transcript_ready.connect(done.append)
        ctl.start()
        capture = None
        try:
            print("loading the models ...")
            ready = pump(300, lambda: ctl.model_status("asr")["state"] in ("ready", "failed")
                         and ctl.model_status("cleanup")["state"] in ("ready", "degraded", "failed"))
            asr, cln = ctl.model_status("asr"), ctl.model_status("cleanup")
            print(f"speech {asr['state']}/{asr['backend']}, cleanup {cln['state']}/{cln['backend']}")
            if not ready or asr["state"] != "ready" or cln["state"] != "ready" or "hook" in ctl._problems:
                print("the models or the keyboard hook are not ready; nothing was recorded", file=sys.stderr)
                return 1
            pump(3.0)                                # let a first-load notice on the Flow bar go away
            if not kit.take_foreground(hwnd, pump):
                print("could not bring the note window to the front; nothing was sent", file=sys.stderr)
                return 3
            if not pump(5, lambda: kit.window_text(hwnd) == "Notes"):
                print("the note's text field did not get the keyboard focus; nothing was sent", file=sys.stderr)
                return 3
            pump(1.0)

            video = work / "capture.mkv"
            capture = subprocess.Popen(
                [ffmpeg, "-v", "error", "-y", "-f", "gdigrab", "-framerate", str(FPS), "-draw_mouse", "0",
                 "-offset_x", str(round(x0 * dpr)), "-offset_y", str(round(y0 * dpr)),
                 "-video_size", f"{round(SCENE_W * dpr)}x{round(SCENE_H * dpr)}", "-i", "desktop",
                 "-c:v", "ffv1", str(video)], stdin=subprocess.PIPE)
            pump(1.8)

            names = keys["hotkey"].split("+")
            try:
                for n in names:
                    kb.down(n)
                    pump(0.03)
                end = time.time() + clip_s + 0.7      # hold until the sentence is over, like a person would
                while time.time() < end:
                    pump(0.05)
                    kit.require_foreground(allowed)    # the user clicked away: stop, release, send nothing more
            finally:
                for n in reversed(names):
                    kb.up(n)
                    pump(0.02)
            if not pump(60, lambda: bool(done) or ctl.state == "error"):
                print("no transcript arrived within 60 s", file=sys.stderr)
            pump(ctl.config.restore_delay_s + 3.2)    # the pasted text stays in view a moment
        except kit.ForegroundLost as exc:
            print(f"aborted, nothing more was sent: {exc}", file=sys.stderr)
            return 1
        finally:
            if capture is not None:
                try:
                    capture.communicate(b"q", timeout=20)
                except subprocess.TimeoutExpired:
                    capture.kill()
            kb.release_all()
            try:
                ui.close()
            finally:
                ctl.shutdown()
            user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            backdrop.close()
            pump(0.5)

    if not done:
        return 1
    entry = done[-1]
    print(f"pasted ({entry.get('latency_ms')} ms): {entry['text']!r}")
    if args.keep_video:
        keep = ROOT / "build" / "demo"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, keep / "capture.mkv")
    graph = (f"fps={FPS},scale={SCENE_W}:-1:flags=lanczos,split[a][b];"
             "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
    code = subprocess.call([ffmpeg, "-v", "error", "-y", "-i", str(video), "-vf", graph, "-loop", "0", args.out])
    if not code:
        print(f"wrote {args.out}")
    return code


def _find_browser() -> Path | None:
    pf, pf86, local = (os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"))
    candidates = [Path(pf86) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                  Path(pf) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                  Path(pf) / "Google" / "Chrome" / "Application" / "chrome.exe",
                  Path(local) / "Google" / "Chrome" / "Application" / "chrome.exe"]
    return next((c for c in candidates if c.is_file()), None)


def _place_visible_frame(user32, hwnd: int, x: int, y: int, w: int, h: int) -> None:
    """Move a window so its VISIBLE frame (without Windows 11's invisible resize borders) is at x, y, w, h (physical
    pixels)."""
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, wintypes.UINT]
    user32.SetWindowPos(hwnd, None, x, y, w, h, 0x0004 | 0x0010)     # NOZORDER | NOACTIVATE
    outer, seen = wintypes.RECT(), wintypes.RECT()
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect(hwnd, ctypes.byref(outer))
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 9, ctypes.byref(seen), ctypes.sizeof(seen)):
        return                                                        # DWMWA_EXTENDED_FRAME_BOUNDS unavailable
    left, top = seen.left - outer.left, seen.top - outer.top
    right, bottom = outer.right - seen.right, outer.bottom - seen.bottom
    user32.SetWindowPos(hwnd, None, x - left, y - top, w + left + right, h + top + bottom, 0x0004 | 0x0010)


if __name__ == "__main__":
    sys.exit(main())
