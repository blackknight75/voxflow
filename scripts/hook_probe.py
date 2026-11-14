"""Live check of the WH_KEYBOARD_LL hook: hold the dictation shortcut for a second, verify the events and
that neither the Start menu nor another window comes to the front.

    python scripts/hook_probe.py [--combo ctrl+win]

Safe on a desktop in use: the keys are sent only while a window this script created is the foreground window
(checked immediately before every key press, see ``tests/desktop_kit.py``). When another VoxFlow is running the
probe uses Ctrl+Alt+F13 instead of Ctrl+Win, so the user's copy does not start recording.

Exit code: 0 passed, 1 failed, 3 blocked (the script's window could not become the foreground window, e.g. the
workstation is locked; nothing was sent).
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", default="")
    args = ap.parse_args(argv)
    os.environ.pop("QT_QPA_PLATFORM", None)

    import desktop_kit as kit
    from PySide6.QtWidgets import QApplication, QLabel

    from scripts.live_e2e import other_voxflow_running
    from voxflow.hotkey import HotkeyListener

    combo = args.combo
    if not combo:
        others = other_voxflow_running()
        combo = "ctrl+alt+f13" if others else "ctrl+win"
        if others:
            print(f"another VoxFlow is running (pid {others}): probing with {combo} so it does not start recording")

    app = QApplication.instance() or QApplication([])
    win = QLabel("VoxFlow hook probe: this window closes by itself.")
    win.setWindowTitle("VoxFlow hook probe")
    win.setGeometry(200, 200, 460, 120)
    win.show()

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.005)

    pump(0.3)
    hwnd = int(win.winId())
    if not kit.take_foreground(hwnd, pump):
        fg = kit.foreground()
        print(f"foreground window is {kit.window_text(fg)!r} ({kit.window_class(fg)!r}); the probe window could not "
              "be brought to the front. Nothing was sent.")
        print("HOOK PROBE BLOCKED")
        win.close()
        return 3

    events: list[str] = []
    listener = HotkeyListener(combo, events.append)
    if not listener.start():
        print("the keyboard hook could not be installed:", listener.error)
        print("HOOK PROBE FAILED")
        win.close()
        return 1
    kb = kit.Keyboard(hwnd)
    names = combo.split("+")
    try:
        for n in names:
            kb.down(n)
            pump(0.05)
        pump(1.0)
    except kit.ForegroundLost as exc:
        print("aborted, nothing more was sent:", exc)
    finally:
        for n in reversed(names):
            kb.up(n)
        kb.release_all()
    pump(0.8)
    after = kit.foreground()
    listener.stop()
    stuck = kb.modifiers_stuck()
    print("combination:", combo)
    print("events:", events)
    print("foreground afterwards:", repr(kit.window_text(after)), kit.window_class(after),
          "(still the probe window)" if after == hwnd else "(CHANGED)")
    print("modifier keys still down:", stuck or "none")
    ok = events == ["start", "stop"] and after == hwnd and not stuck
    print("HOOK PROBE", "PASSED" if ok else "FAILED")
    win.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
