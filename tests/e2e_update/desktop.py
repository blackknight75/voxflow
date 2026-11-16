"""Helpers of the desktop update-UI tests (test_update_desktop.py).

* ``uia``: Windows UI Automation through uia.ps1 (a separate PowerShell), while Qt keeps pumping events so the app
  answers UIA like a running app. Only UIA patterns (Invoke) and reads: no synthetic OS input of any kind.
* ``shot``: a PNG of one of VoxFlow's own windows, rendered by ``QWidget.grab()``: never the screen, so nothing
  else on the desktop can appear in it.
* ``clipped``: text that does not fit its widget at the current scale.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import QApplication, QLabel, QWidget

HERE = Path(__file__).resolve().parent
PS = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


def shots_dir() -> Path:
    d = Path(os.environ.get("VOXFLOW_DESKTOP_SHOTS") or (Path(tempfile.gettempdir()) / "voxflow-desktop-shots"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def uia(action: str, name: str = "", timeout_ms: int = 5000, wait: float = 90.0, invoke: bool = False,
        control: str = "") -> dict:
    """Run one uia.ps1 action against this process; Qt events are processed while PowerShell works."""
    fd, result = tempfile.mkstemp(suffix=".json", prefix="uia-")        # a pipe would fill up and block on a dump
    os.close(fd)
    cmd = [str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "uia.ps1"),
           "-ProcessId", str(os.getpid()), "-Action", action, "-TimeoutMs", str(timeout_ms), "-OutFile", result]
    if name:
        cmd += ["-Name", name]
    if invoke:
        cmd += ["-Invoke"]
    if control:
        cmd += ["-Type", f"ControlType.{control}"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         creationflags=subprocess.CREATE_NO_WINDOW)
    end = time.monotonic() + wait
    while p.poll() is None:
        QApplication.processEvents()
        time.sleep(0.01)
        if time.monotonic() > end:
            p.kill()
            raise TimeoutError(f"uia {action} {name!r}")
    _out, err = p.communicate()
    text = Path(result).read_text(encoding="utf-8-sig") if os.path.getsize(result) else ""
    os.remove(result)
    if p.returncode != 0 or not text.strip():
        raise RuntimeError(f"uia {action} {name!r} failed ({p.returncode}): {err}")
    return json.loads(text)


def pump(ms: int) -> None:
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def shot(name: str, widget: QWidget) -> Path:
    """VoxFlow's own window (or widget) as it renders, via QWidget.grab(): device pixels at the current scale."""
    pump(400)                                    # animations (Modal appear 170 ms, toasts) settle
    pix = widget.grab()
    path = shots_dir() / f"{name}.png"
    assert not pix.isNull() and pix.save(str(path), "PNG"), path
    return path


def clipped(root: QWidget) -> list[str]:
    """Visible text under ``root`` that does not fit: a one-line label wider than its widget, a wrapped label taller
    than the room it was given, a button narrower than its size hint, a dropdown whose choice is elided."""
    from voxflow.ui.widgets import Button, ChoiceButton
    bad = []
    for w in [root, *root.findChildren(QWidget)]:
        if not w.isVisible() or w.width() <= 0:
            continue
        what = f"{w.objectName() or type(w).__name__}"
        if isinstance(w, QLabel) and w.text() and not w.text().lstrip().startswith("<"):
            if w.wordWrap():
                need = w.heightForWidth(w.width())
                if need > w.height() + 1:
                    bad.append(f"{what} {w.text()[:60]!r}: needs {need} px high, has {w.height()}")
            else:
                m = w.contentsMargins()
                need = QFontMetricsF(w.font()).horizontalAdvance(w.text()) + m.left() + m.right()
                if need > w.width() + 1:
                    bad.append(f"{what} {w.text()[:60]!r}: needs {need:.0f} px wide, has {w.width()}")
        elif isinstance(w, ChoiceButton):
            avail = w.width() - 3 - 42                         # ChoiceButton.paintEvent: the text box
            need = QFontMetricsF(w._font).horizontalAdvance(w.label())
            if need > avail:
                bad.append(f"{what} {w.label()!r}: elided, needs {need:.0f} px, has {avail}")
        elif isinstance(w, Button) and w.text():
            if w.sizeHint().width() > w.width() + 1:
                bad.append(f"{what} {w.text()!r}: needs {w.sizeHint().width()} px wide, has {w.width()}")
    return bad
