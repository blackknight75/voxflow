"""True end-to-end run on the REAL desktop: real keyboard hook, real models, real clipboard, real paste.

    python scripts/live_e2e.py [--json FILE] [--only qt,win32,...]

What is real: the WH_KEYBOARD_LL hook, the hotkeys (pressed with SendInput, exactly as a keyboard would), the
speech model, the language model, the clipboard, Ctrl+V / Ctrl+C / Ctrl+Z, the Flow bar and the tray icon.
What is not: the microphone. A speech clip is handed to the pipeline where the recorder would deliver its
audio (``scripts/mic_check.py`` covers the real microphone).

Safety rules, enforced in code (``tests/desktop_kit.py``):

* every key press and every paste is sent only while a window THIS SCRIPT created is the foreground window,
  checked with GetForegroundWindow immediately before each send; otherwise the step is aborted, nothing is sent;
* the user's clipboard is snapshotted first and restored at the end;
* everything runs in a temp profile (never %APPDATA%/VoxFlow); browser and editor use throwaway profiles;
* when another VoxFlow is already running, the test uses its own shortcuts so the user's copy never fires;
* Notepad is skipped when the user already has a Notepad window open (a new file would open as a tab there).

Exit code: 0 all run checks passed, 1 a check failed, 3 blocked (locked workstation / no foreground / clipboard).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

DEFAULT_KEYS = {"hotkey": "ctrl+win", "command_hotkey": "alt+win", "paste_last_hotkey": "alt+shift+z",
                "undo_hotkey": "ctrl+alt+f9"}
# F13..F16 exist on no ordinary keyboard and mean nothing to applications, and there is no Alt in them (holding
# Alt moves the focus into the menu of Windows 11 Notepad, bug C-69), so the target does not react to the shortcut.
PRIVATE_KEYS = {"hotkey": "ctrl+shift+f13", "command_hotkey": "ctrl+shift+f14", "paste_last_hotkey": "ctrl+shift+f15",
                "undo_hotkey": "ctrl+shift+f16"}
WM_CLOSE, EM_SETSEL = 0x0010, 0x00B1


def other_voxflow_running() -> list[int]:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe' OR Name='VoxFlow.exe'\" | "
          r"Where-Object { $_.CommandLine -match '-m\s+voxflow|run_module\(.voxflow|VoxFlow\.exe' } | "
          "ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit() and int(x) != os.getpid()]


def kill_by_commandline(fragment: str) -> None:
    """Stop the processes this script started (matched by the throwaway profile folder on their command line)."""
    frag = fragment.replace("'", "''")
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('" + frag +
          "') -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction "
          "SilentlyContinue }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=60)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="", help="comma list of: qt,win32,notepad,electron,browser,command,"
                                               "paste_last,undo,esc,lock,languages")
    ap.add_argument("--wav", default=str(ROOT / "tests" / "tts_sample.wav"))
    ap.add_argument("--alt-shortcuts", action="store_true",
                    help="use private shortcuts that contain Alt (Ctrl+Alt+F13..F16): the regression run for C-69")
    args = ap.parse_args(argv)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    want = lambda name: not only or name in only  # noqa: E731

    os.environ.pop("QT_QPA_PLATFORM", None)          # this run needs the real screen
    work = Path(tempfile.mkdtemp(prefix="voxflow-live-"))
    os.environ["APPDATA"] = str(work / "appdata")    # nothing may reach the real %APPDATA%/VoxFlow
    (work / "appdata").mkdir()

    import desktop_kit as kit
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    user32 = kit.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                       wintypes.HANDLE, wintypes.HANDLE, ctypes.c_void_p]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    results: list[dict] = []

    def record(name: str, outcome: str, detail: str = "") -> None:
        results.append({"check": name, "outcome": outcome, "detail": detail})
        print(f"  {outcome:<5} {name}" + (f": {detail}" if detail else ""), flush=True)

    def finish(code: int, note: str = "") -> int:
        if note:
            print(note)
        if code != 0 or note:
            shutil.rmtree(work, ignore_errors=True)
        if args.json:
            Path(args.json).write_text(json.dumps({"exit": code, "note": note, "results": results}, indent=2),
                                       encoding="utf-8")
        failed = [r for r in results if r["outcome"] == "FAIL"]
        label = {0: "PASSED", 1: "FAILED", 3: "BLOCKED"}[code]
        print(f"LIVE E2E {label} ({sum(r['outcome'] == 'PASS' for r in results)} passed, {len(failed)} failed, "
              f"{sum(r['outcome'] == 'SKIP' for r in results)} skipped)")
        return code

    qapp = QApplication.instance() or QApplication([])

    def pump(seconds: float, until=None) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            qapp.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.005)
        return until is None

    # ------------------------------------------------------------------ own windows + foreground probe
    token = uuid.uuid4().hex[:10]
    own: dict[int, object] = {}                      # hwnd -> "is this still our window?" check

    def allowed() -> list[int]:
        return [h for h, still_ours in own.items() if still_ours()]

    box = QPlainTextEdit()
    box.setWindowTitle(f"VoxFlow live test {token} (Qt text box)")
    box.setGeometry(160, 160, 620, 260)
    box.show()
    pump(0.3)
    qt_hwnd = int(box.winId())
    own[qt_hwnd] = lambda: True
    if not kit.take_foreground(qt_hwnd, pump):
        fg = kit.foreground()
        box.close()
        return finish(3, "BLOCKED: could not make the test window the foreground window (foreground is "
                         f"{kit.window_text(fg)!r}, class {kit.window_class(fg)!r}). The workstation is probably "
                         "locked. Nothing was typed, pasted or copied.")

    edit_hwnd = user32.CreateWindowExW(
        0, "EDIT", f"VoxFlow live test {token} (Win32 edit)",
        0x00CF0000 | 0x10000000 | 0x0004 | 0x0040 | 0x00200000,     # overlapped, visible, multiline, autovscroll
        200, 460, 620, 240, None, None, None, None)
    if edit_hwnd:
        own[int(edit_hwnd)] = lambda: True
    user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

    kb = kit.Keyboard(allowed)
    profile = work / "profile"
    profile.mkdir()
    others = other_voxflow_running()
    keys = PRIVATE_KEYS if others else DEFAULT_KEYS
    if args.alt_shortcuts:
        keys = {k: v.replace("ctrl+shift+", "ctrl+alt+") for k, v in PRIVATE_KEYS.items()}
    print(f"shortcuts used: {keys}" + (f" (another VoxFlow is running, pid {others}: its shortcuts are avoided)"
                                      if others else " (the defaults)"))
    (profile / "config.json").write_text(json.dumps(dict(
        keys, onboarding_done=True, mute_media=False, sounds_enabled=False, launch_at_login=False,
        start_minimized=True)), encoding="utf-8")

    with kit.saved_clipboard() as clipboard:
        if clipboard is None:
            box.close()
            return finish(3, "BLOCKED: the clipboard could not be saved (in use, or holds content that cannot be "
                             "restored), so it was not touched.")
        code = 1
        try:
            code = _run(args, want, kit, kb, own, allowed, box, qt_hwnd, int(edit_hwnd or 0), pump, record, keys,
                        profile, work, token, clipboard, qapp)
        except kit.ForegroundLost as exc:
            record("run", "FAIL", f"aborted, nothing more was sent: {exc}")
        finally:
            kb.release_all()
            stuck = kb.modifiers_stuck()
            record("no modifier key left held down", "FAIL" if stuck else "PASS", ", ".join(stuck))
            for hwnd in list(own):
                if hwnd not in (qt_hwnd, int(edit_hwnd or 0)) and user32.IsWindow(hwnd):
                    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            pump(0.5)
            kill_by_commandline(str(work))
            if edit_hwnd:
                user32.DestroyWindow(edit_hwnd)
            box.close()
            pump(0.3)
    pump(0.2)
    shutil.rmtree(work, ignore_errors=True)
    failed = any(r["outcome"] == "FAIL" for r in results)
    return finish(1 if failed or code else 0)


def _run(args, want, kit, kb, own, allowed, box, qt_hwnd, edit_hwnd, pump, record, keys, profile, work, token,
         clipboard, qapp) -> int:
    from voxflow import app as appmod
    from voxflow import fakes
    from voxflow.asr import Transcriber
    from voxflow.cleanup import Cleaner
    from voxflow.controller import Controller
    from voxflow.inject import Injector
    from voxflow.ui.shell import build_ui

    user32 = kit.user32
    sentinel = f"clipboard sentinel {token}"
    clipboard.set_text(sentinel)                     # every paste must put exactly this back

    dictation = appmod.read_wav_16k(args.wav)
    command_wav = work / "command.wav"
    from scripts.speech_matrix import tts, voices
    tts("Make this more formal.", voices()[0], command_wav)
    command_clip = appmod.read_wav_16k(str(command_wav))
    current = {"clip": dictation}

    # one human FLEURS clip per language for the typed-paste check: fetched first (network), the one with the most
    # accented / special characters in its reference text, so a lost character shows
    lang_clips: dict[str, tuple] = {}
    if want("languages"):
        import scripts.speech_matrix as sm
        for code in sm.FLEURS:
            try:
                got = sm.fetch_fleurs(code, work / f"fleurs-{code}", 6)
                best = max(got, key=lambda c: sum(1 for ch in c[1] if ord(ch) > 127)) if got else None
                if best:
                    lang_clips[code] = (sm.read_any_wav(best[0].read_bytes()), best[1])
            except Exception as exc:  # noqa: BLE001
                record(f"language {code}: clip download", "FAIL", f"{type(exc).__name__}: {exc}")

    class ClipRecorder(fakes.FakeRecorder):
        """Stands where the microphone is: 'recording' returns the speech clip."""

        def stop(self):
            was, self._rec = self._rec, False
            clip = current["clip"]
            self._last_speech_s = len(clip) / 16000 if was else 0.0
            return clip.copy() if was else clip[:0]

        @property
        def hit_cap(self):
            return False

    injector = Injector(keys=kit.guarded_keys(allowed))
    ctl = Controller(data_dir=profile, install_hook=True, modules=(ClipRecorder, Transcriber(), Cleaner()),
                     injector=injector, sound_backend=lambda wav: None, mute_backend=fakes.FakeMuteBackend(),
                     autostart=fakes.FakeAutostart(False))
    ui = build_ui(ctl, show_window=False)            # real Flow bar + tray icon
    done: list[dict] = []
    states: list[str] = []
    ctl.transcript_ready.connect(done.append)
    ctl.state_changed.connect(lambda s, m, msg: states.append(s))
    ctl.start()
    try:
        ready = pump(240, lambda: ctl.model_status("asr")["state"] in ("ready", "failed")
                     and ctl.model_status("cleanup")["state"] in ("ready", "degraded", "failed"))
        hook_ok = "hook" not in ctl._problems
        record("real models loaded, keyboard hook installed", "PASS" if ready and hook_ok else "FAIL",
               f"asr {ctl.model_status('asr')['state']}/{ctl.model_status('asr')['backend']}, cleanup "
               f"{ctl.model_status('cleanup')['state']}/{ctl.model_status('cleanup')['backend']}, "
               f"hook {'ok' if hook_ok else ctl._problems.get('hook')}")
        if not (ready and hook_ok):
            return 1

        def names(key: str) -> list[str]:
            return keys[key].split("+")

        def focus(hwnd: int, extra=None) -> None:
            if not kit.take_foreground(hwnd, pump):
                raise kit.ForegroundLost(f"could not bring our window {hwnd:#x} to the front")
            if extra:
                extra()
            pump(0.25)

        def hold(key: str, seconds: float, during=None) -> None:
            """Press and hold a shortcut like a person would, release it afterwards (always)."""
            try:
                for n in names(key):
                    kb.down(n)
                    pump(0.03)
                pump(seconds)
                if during:
                    during()
            finally:
                for n in reversed(names(key)):
                    kb.up(n)
                    pump(0.02)

        def tap(key: str) -> None:
            kb.chord(*names(key), hold_s=0.04, pump=pump)

        def dictate(clip=None, key: str = "hotkey", timeout: float = 45.0) -> dict | None:
            current["clip"] = dictation if clip is None else clip
            n = len(done)
            hold(key, 0.8)
            if not pump(timeout, lambda: len(done) > n or ctl.state == "error"):
                return None
            pump(ctl.config.restore_delay_s + 0.6)   # paste delivered + clipboard put back
            return done[-1] if len(done) > n else None

        expected = "quick brown fox"

        def check_text(name: str, entry, text: str) -> bool:
            ok = bool(entry) and expected in text.lower() and text.lower().count(expected) == 1
            record(name, "PASS" if ok else "FAIL",
                   f"{entry['latency_ms']} ms, window shows {text[:60]!r}" if ok
                   else f"transcript {entry and entry['text'][:50]!r}, window shows {text[:80]!r}, "
                        f"paste outcome {injector.last_outcome}, state {ctl.state}: {ctl.message}")
            return ok

        def edit_text() -> str:
            n = user32.GetWindowTextLengthW(edit_hwnd)
            buf = ctypes.create_unicode_buffer(n + 2)
            user32.GetWindowTextW(edit_hwnd, buf, n + 1)
            return buf.value

        # ---------------------------------------------------------------- 1. Qt text box
        if want("qt"):
            box.clear()
            focus(qt_hwnd, box.setFocus)
            entry = dictate()
            check_text("dictation -> Qt text box", entry, box.toPlainText())

        # ---------------------------------------------------------------- 2. classic Win32 edit control
        if want("win32") and edit_hwnd:
            user32.SetWindowTextW(edit_hwnd, "")
            focus(edit_hwnd, lambda: user32.SetFocus(edit_hwnd))
            entry = dictate()
            check_text("dictation -> classic Win32 EDIT control", entry, edit_text())

        # ---------------------------------------------------------------- 3. Notepad started by the test
        if want("notepad"):
            _notepad(kit, kb, own, pump, record, focus, dictate, expected, work, token)

        # ---------------------------------------------------------------- 4. Electron (VS Code, throwaway profile)
        if want("electron"):
            _vscode(kit, kb, own, pump, record, focus, dictate, expected, work, token)

        # ---------------------------------------------------------------- 5. browser text field (throwaway profile)
        if want("browser"):
            _browser(kit, own, pump, record, focus, dictate, expected, work, token)

        # ---------------------------------------------------------------- 6. command mode on a real selection
        if want("command"):
            original = "hey bob can u send me the report by friday thx"
            box.setPlainText(original)
            focus(qt_hwnd, box.setFocus)
            box.selectAll()
            pump(0.2)
            entry = dictate(command_clip, "command_hotkey", timeout=90.0)
            text = box.toPlainText().strip()
            ok = bool(entry) and text and text != original and "thx" not in text.lower() and "report" in text.lower()
            record("command mode rewrites a real selection (Ctrl+C, language model, Ctrl+V)", "PASS" if ok else "FAIL",
                   f"{original!r} -> {text[:90]!r}" + ("" if ok else f" | state {ctl.state}: {ctl.message}, "
                                                               f"outcome {injector.last_outcome}"))

        # ---------------------------------------------------------------- 7. paste last
        if want("paste_last") and edit_hwnd:
            if not done:
                box.clear()
                focus(qt_hwnd, box.setFocus)
                dictate()
            last = (ctl.last_transcript or {}).get("text", "")
            user32.SetWindowTextW(edit_hwnd, "")
            focus(edit_hwnd, lambda: user32.SetFocus(edit_hwnd))
            tap("paste_last_hotkey")
            pump(6, lambda: bool(last) and last[:25] in edit_text())
            pump(ctl.config.restore_delay_s + 0.6)
            got = edit_text()
            ok = bool(last) and got.strip() == last.strip()
            record("paste-last shortcut pastes the previous transcript", "PASS" if ok else "FAIL",
                   f"{got[:60]!r}" if ok else f"expected {last[:50]!r}, window shows {got[:60]!r}")

        # ---------------------------------------------------------------- 8. undo
        if want("undo"):
            box.setPlainText("")
            focus(qt_hwnd, box.setFocus)
            entry = dictate()
            pasted = expected in box.toPlainText().lower()
            tap("undo_hotkey")
            pump(4, lambda: expected not in box.toPlainText().lower())
            ok = bool(entry) and pasted and expected not in box.toPlainText().lower()
            record("undo shortcut takes the pasted text out again (Ctrl+Z in the same window)",
                   "PASS" if ok else "FAIL", "" if ok else f"pasted {pasted}, window shows {box.toPlainText()[:60]!r}")

        # ---------------------------------------------------------------- 9. Esc cancels
        if want("esc"):
            box.setPlainText("")
            focus(qt_hwnd, box.setFocus)
            n, seen = len(done), len(states)
            current["clip"] = dictation
            hold("hotkey", 0.6, during=lambda: (kb.chord("esc", pump=pump), pump(0.3)))
            pump(2.5)
            ok = (len(done) == n and box.toPlainText() == "" and ctl.state == "idle"
                  and "listening" in states[seen:] and "processing" not in states[seen:])
            record("Esc while holding the shortcut cancels: nothing is transcribed or pasted", "PASS" if ok else "FAIL",
                   "" if ok else f"states {states[seen:]}, window shows {box.toPlainText()[:40]!r}")

        # ---------------------------------------------------------------- 10. hands-free lock (double tap)
        if want("lock"):
            box.setPlainText("")
            focus(qt_hwnd, box.setFocus)
            n, seen = len(done), len(states)
            current["clip"] = dictation
            tap("hotkey")
            pump(0.12)
            tap("hotkey")
            locked = pump(2, lambda: ctl.state == "locked")
            pump(1.2)                                 # hands are off the keyboard, still recording
            still = ctl.state == "locked"
            tap("hotkey")
            pump(45, lambda: len(done) > n or ctl.state == "error")
            pump(ctl.config.restore_delay_s + 0.6)
            ok = locked and still and len(done) > n and expected in box.toPlainText().lower()
            record("hands-free: double tap locks, recording continues with no key held, next tap finishes",
                   "PASS" if ok else "FAIL", "" if ok else f"locked {locked}, still locked {still}, states "
                                                           f"{states[seen:]}, window shows {box.toPlainText()[:40]!r}")

        # ---------------------------------------------------------------- 11. non-English clips, typed paste
        if want("languages"):
            import scripts.speech_matrix as sm
            for code, (clip, reference) in lang_clips.items():
                ctl.set("language", code)
                special = sorted({ch for ch in reference if ord(ch) > 127})
                for label, target in (("Qt text box", "qt"), ("Win32 EDIT control", "win32")):
                    if target == "qt":
                        box.clear()
                        focus(qt_hwnd, box.setFocus)
                    elif edit_hwnd:
                        user32.SetWindowTextW(edit_hwnd, "")
                        focus(edit_hwnd, lambda: user32.SetFocus(edit_hwnd))
                    else:
                        continue
                    entry = dictate(clip, timeout=90.0)
                    shown = box.toPlainText() if target == "qt" else edit_text()
                    sent = (entry or {}).get("text", "")
                    norm = lambda t: t.replace(chr(13) + chr(10), chr(10)).strip()   # noqa: E731  (EDIT controls use CR LF)
                    same = bool(entry) and norm(shown) == norm(sent)
                    lost = [ch for ch in sent if ord(ch) > 127 and ch not in shown]
                    w = sm.wer(reference, shown) if shown else 1.0
                    ok = same and not lost and w < 0.35
                    record(f"language {code} typed paste -> {label}", "PASS" if ok else "FAIL",
                           f"WER {w:.0%}, special characters {''.join(special)[:12]!r} kept, window shows {shown[:60]!r}"
                           if ok else f"same as transcript {same}, lost {lost}, WER {w:.0%}, transcript "
                                      f"{sent[:50]!r}, window {shown[:50]!r}, outcome {injector.last_outcome}")
            ctl.set("language", "auto")

        # ---------------------------------------------------------------- afterwards
        pump(ctl.config.restore_delay_s + 1.0)
        back = clipboard.get_text()
        record("clipboard is back to what it held before every paste", "PASS" if back == sentinel else "FAIL",
               "" if back == sentinel else f"clipboard holds {back[:50]!r}")
        blocked = type(injector.keys).blocked
        record("no paste was refused by the foreground guard", "PASS" if blocked == 0 else "FAIL",
               "" if blocked == 0 else f"{blocked} send(s) refused because our window was not in front")
        record("every transcript is in the temp profile's history", "PASS" if ctl.history.count() == len(done)
               else "FAIL", f"{ctl.history.count()} entries for {len(done)} transcripts, in {profile}")
        return 0
    finally:
        try:
            ui.close()
        finally:
            ctl.shutdown()


# ======================================================================== external targets
def _wait_window(kit, pump, predicate, seconds: float) -> int:
    found: list[int] = []
    pump(seconds, lambda: bool(found.extend(kit.find_windows(predicate)) or found))
    return found[0] if found else 0


def _child_text(kit, hwnd: int, classes=("RichEditD2DPT", "Edit", "RICHEDIT50W")) -> str | None:
    """Text of the edit control inside another process's window, read with WM_GETTEXT (no keys, no clipboard)."""
    u = kit.user32
    found: list[int] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def each(h, _l):
        if kit.window_class(h) in classes:
            found.append(int(h))
        return True

    u.EnumChildWindows(hwnd, proto(each), 0)
    if not found:
        return None
    u.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_void_p, wintypes.UINT,
                                      wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
    out = ctypes.c_size_t(0)
    buf = ctypes.create_unicode_buffer(8192)
    ok = u.SendMessageTimeoutW(found[0], 0x000D, 8192, ctypes.cast(buf, ctypes.c_void_p), 0x0002, 2000,
                               ctypes.byref(out))
    return buf.value.replace("\r", "\n") if ok else None


def _notepad(kit, kb, own, pump, record, focus, dictate, expected, work, token) -> None:
    name = "dictation -> Notepad started by the test"
    is_notepad = lambda h, t, c: c == "Notepad"  # noqa: E731
    if kit.find_windows(is_notepad):
        record(name, "SKIP", "a Notepad window is already open; the new file would open as a tab in the user's "
                             "window, which the test must not type into")
        return
    path = work / f"vflive-{token}-notepad.txt"
    path.write_text("", encoding="utf-8")
    proc = subprocess.Popen(["notepad.exe", str(path)])
    hwnd = _wait_window(kit, pump, lambda h, t, c: c == "Notepad" and path.stem in t, 15)
    if not hwnd:
        record(name, "SKIP", "Notepad did not show a window for the test file within 15 s")
        proc.kill()
        return
    own[hwnd] = lambda: path.stem in kit.window_text(hwnd)       # only while OUR file is the active tab
    try:
        pump(2.5)                                                 # Windows 11 Notepad needs a moment to take focus
        focus(hwnd)
        entry = dictate()
        pump(1.0)
        shown = _child_text(kit, hwnd)                            # what the edit control holds right now
        focus(hwnd)
        kb.chord("ctrl", "s", pump=pump)
        pump(4, lambda: path.stat().st_size > 0)
        saved = path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
        text = shown if shown is not None else saved
        ok = bool(entry) and expected in text.lower() and text.lower().count(expected) == 1
        record(name, "PASS" if ok else "FAIL",
               f"{entry['latency_ms']} ms, edit control holds {text[:50]!r}; after Ctrl+S the file holds "
               f"{len(saved)} characters" if ok
               else f"transcript {entry and entry['text'][:40]!r}, edit control holds {str(shown)[:70]!r}, saved "
                    f"file holds {saved[:40]!r}, window title {kit.window_text(hwnd)!r}")
        if kit.foreground() == hwnd and path.stem in kit.window_text(hwnd):
            kb.chord("ctrl", "w", pump=pump)                      # close our tab (Windows 11 Notepad)
            pump(0.8)
    finally:
        if kit.user32.IsWindow(hwnd):
            kit.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        own.pop(hwnd, None)
        pump(0.5)
        if proc.poll() is None:
            proc.kill()


def _vscode(kit, kb, own, pump, record, focus, dictate, expected, work, token) -> None:
    # Electron editors that accept a throwaway profile folder (so the user's own windows and settings stay out of it)
    local, pf = Path(os.environ.get("LOCALAPPDATA", "")), Path(os.environ.get("ProgramFiles", ""))
    candidates = [local / "Programs" / "Microsoft VS Code" / "Code.exe", pf / "Microsoft VS Code" / "Code.exe",
                  local / "Programs" / "VSCodium" / "VSCodium.exe", pf / "VSCodium" / "VSCodium.exe",
                  local / "Programs" / "cursor" / "Cursor.exe", local / "Programs" / "Windsurf" / "Windsurf.exe"]
    exe = next((c for c in candidates if c.is_file()), None)
    name = f"dictation -> Electron app ({exe.stem if exe else 'none'}, throwaway profile)"
    if exe is None:
        record(name, "SKIP", "no Electron editor found (looked for VS Code, VSCodium, Cursor, Windsurf)")
        return
    path = work / f"vflive-{token}-code.txt"
    path.write_text("", encoding="utf-8")
    subprocess.Popen([str(exe), "--user-data-dir", str(work / "code-profile"), "--extensions-dir",
                      str(work / "code-ext"), "--disable-workspace-trust", "--skip-welcome", "--skip-release-notes",
                      "--disable-telemetry", "--disable-updates", "--new-window", str(path)])
    hwnd = _wait_window(kit, pump, lambda h, t, c: c == "Chrome_WidgetWin_1" and path.stem in t, 40)
    if not hwnd:
        record(name, "SKIP", f"{exe.stem} did not show a window for the test file within 40 s (a first-run "
                             "screen of a fresh profile can do that)")
        kill_by_commandline(str(work / "code-profile"))
        return
    own[hwnd] = lambda: path.stem in kit.window_text(hwnd)
    try:
        pump(5.0)                                                 # let the editor finish loading and take focus
        focus(hwnd)
        entry = dictate()
        focus(hwnd)
        kb.chord("ctrl", "s", pump=pump)
        pump(5, lambda: path.stat().st_size > 0)
        text = path.read_text(encoding="utf-8", errors="replace")
        ok = bool(entry) and expected in text.lower() and text.lower().count(expected) == 1
        record(name, "PASS" if ok else "FAIL", f"{entry['latency_ms']} ms, saved file holds {text[:60]!r}" if ok
               else f"transcript {entry and entry['text'][:40]!r}, saved file holds {text[:60]!r}")
    finally:
        if kit.user32.IsWindow(hwnd):
            kit.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        own.pop(hwnd, None)
        pump(1.0)
        kill_by_commandline(str(work / "code-profile"))


def _browser(kit, own, pump, record, focus, dictate, expected, work, token) -> None:
    name = "dictation -> browser text field (fresh window, throwaway profile)"
    pf, pf86, local = (os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"))
    candidates = [Path(pf86) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                  Path(pf) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                  Path(pf) / "Google" / "Chrome" / "Application" / "chrome.exe",
                  Path(local) / "Google" / "Chrome" / "Application" / "chrome.exe"]
    exe = next((c for c in candidates if c.is_file()), None)
    if exe is None:
        record(name, "SKIP", "no Chromium browser found (looked for Edge and Chrome)")
        return
    page = work / "page.html"
    mark = f"vflive-{token}-web"
    page.write_text(f"""<!doctype html><meta charset="utf-8"><title>{mark}|N|</title>
<textarea id="t" autofocus style="width:90%;height:60vh;font:16px sans-serif"></textarea>
<script>
const t = document.getElementById('t');
function up() {{
  document.title = '{mark}|' + ((document.hasFocus() && document.activeElement === t) ? 'F' : 'N') + '|'
                   + encodeURIComponent(t.value);
}}
['input', 'focus', 'blur'].forEach(e => t.addEventListener(e, up));
window.addEventListener('load', () => {{ t.focus(); up(); }});
setInterval(up, 150);
</script>""", encoding="utf-8")
    subprocess.Popen([str(exe), f"--user-data-dir={work / 'browser-profile'}", "--no-first-run",
                      "--no-default-browser-check", "--disable-sync", "--disable-extensions",
                      "--disable-features=msEdgeSidebarV2,Translate", "--new-window", page.as_uri()])
    hwnd = _wait_window(kit, pump, lambda h, t, c: c.startswith("Chrome_WidgetWin") and mark in t, 40)
    if not hwnd:
        record(name, "SKIP", f"{exe.stem} did not show the test page within 40 s")
        kill_by_commandline(str(work / "browser-profile"))
        return
    own[hwnd] = lambda: mark in kit.window_text(hwnd)

    def read() -> tuple[bool, str]:
        m = re.search(re.escape(mark) + r"\|([FN])\|(\S*)", kit.window_text(hwnd))
        return (m.group(1) == "F", urllib.parse.unquote(m.group(2))) if m else (False, "")

    try:
        pump(2.0)
        focus(hwnd)
        if not pump(5, lambda: read()[0]):
            record(name, "SKIP", "the page's text field did not get keyboard focus by itself; the test does not "
                                 "click or type to force it")
            return
        entry = dictate()
        pump(3, lambda: expected in read()[1].lower())
        text = read()[1]
        ok = bool(entry) and expected in text.lower() and text.lower().count(expected) == 1
        record(name, "PASS" if ok else "FAIL", f"{exe.stem}, {entry['latency_ms']} ms, field holds {text[:60]!r}"
               if ok else f"{exe.stem}, transcript {entry and entry['text'][:40]!r}, field holds {text[:60]!r}")
    finally:
        if kit.user32.IsWindow(hwnd):
            kit.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        own.pop(hwnd, None)
        pump(1.0)
        kill_by_commandline(str(work / "browser-profile"))


if __name__ == "__main__":
    sys.exit(main())
