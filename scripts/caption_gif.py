"""Render docs/assets/live-preview.gif: the real Flow bar and its live-preview caption through one scripted
dictation (words arriving one by one, processing, "Pasted").

Runs offscreen like scripts/demo_ui.py (no window appears, no model is loaded, nothing is typed anywhere): the
preview controller twin drives the Flow bar's states and the caption gets scripted words. Frames are saved to
build/caption-frames/ and turned into a GIF with ffmpeg (which must be on PATH).

    python scripts/caption_gif.py                  # writes docs/assets/live-preview.gif
    python scripts/caption_gif.py --out other.gif
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "build" / "caption-frames"
FPS = 15
WORDS = "Please move the meeting to Friday at three thirty and send the report to the team"


def render() -> int:
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    from voxflow.ui.fakes import FakeController
    from voxflow.ui.shell import build_ui

    ctl = FakeController()
    ctl.set("onboarding_done", True)
    ctl.set("theme", "dark")
    pill = build_ui(ctl, show_window=False).overlay
    crop = QRect(10, 10, 500, 130)           # the same crop as demo_ui.py's caption shots
    count = 0

    def frame() -> None:
        nonlocal count
        pm = pill.grab()
        top = pm.toImage()
        top.setDevicePixelRatio(1.0)
        img = QImage(top.width(), top.height(), QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        g = QLinearGradient(0, 0, img.width(), img.height())      # demo_ui.py's dark desktop backdrop
        g.setColorAt(0, QColor(30, 38, 70))
        g.setColorAt(1, QColor(70, 46, 96))
        p.fillRect(img.rect(), g)
        p.drawImage(0, 0, top)
        p.end()
        k = pm.devicePixelRatio()
        img = img.copy(int(crop.x() * k), int(crop.y() * k), int(crop.width() * k), int(crop.height() * k))
        img.save(str(FRAMES / f"frame_{count:04d}.png"))
        count += 1

    def run(seconds: float) -> None:
        """Let the widget animate in real time, saving FPS frames per second."""
        for _ in range(int(seconds * FPS)):
            end = time.perf_counter() + 1 / FPS
            while time.perf_counter() < end:
                app.processEvents()
            frame()

    ctl.demo_set_state("idle")
    run(0.2)
    ctl.demo_set_state("listening", "dictate", "")
    run(0.6)
    committed: list[str] = []
    for word in WORDS.split():
        pill.set_preview(" ".join(committed), word)        # the newest word is tentative until the next arrives
        run(0.32)
        committed.append(word)
    pill.set_preview(" ".join(committed), "")
    run(0.5)
    ctl.demo_set_state("processing", "dictate", "")
    run(0.8)
    ctl.demo_set_state("done", "dictate", "Pasted")
    run(1.6)
    ctl.demo_set_state("idle")
    run(0.6)
    print(f"{count} frames in {FRAMES}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "live-preview.gif"))
    ap.add_argument("--scale", type=int, default=150, help="UI scale in percent (default 150, as the screenshots)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return render()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 1
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=str(args.scale / 100),
               QT_ENABLE_HIGHDPI_SCALING="1")
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if fonts.is_dir():
        env.setdefault("QT_QPA_FONTDIR", str(fonts))           # the offscreen plugin ships no fonts of its own
    code = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, cwd=str(ROOT))
    if code:
        return code
    palette = "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a"
    code = subprocess.call([ffmpeg, "-v", "error", "-y", "-framerate", str(FPS), "-i",
                            str(FRAMES / "frame_%04d.png"), "-vf", palette, "-loop", "0", args.out])
    if not code:
        print(f"wrote {args.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
