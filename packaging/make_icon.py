"""Write packaging/voxflow.ico from the app's own logo painter (offscreen Qt, PNG-in-ICO)."""
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build(out: Path) -> None:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
    from voxflow.ui.widgets import paint_logo

    app = QGuiApplication.instance() or QGuiApplication([])
    images = []
    for size in (16, 24, 32, 48, 64, 128, 256):
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_logo(p, QRectF(size * 0.04, size * 0.04, size * 0.92, size * 0.92), None, None)
        p.end()
        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        images.append((size, bytes(data)))
    head = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, png in images:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(png), offset + len(blobs))
        blobs += png
    out.write_bytes(head + entries + blobs)
    del app


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "packaging" / "voxflow.ico"
    build(target)
    print("wrote", target, target.stat().st_size, "bytes")
