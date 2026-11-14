"""Real clipboard + real SendInput Ctrl+V into a Qt text field the test owns (desktop test, opt-in).

Rules (see ``desktop_kit``): the paste is sent only if the test's own window is the foreground window at that
instant, the foreground is taken without any keystroke, and the user's clipboard is saved and put back.
"""
import sys
import threading
import time

import pytest

pytestmark = [pytest.mark.desktop, pytest.mark.skipif(sys.platform != "win32", reason="Windows only")]


def _pump(app, secs):
    end = time.time() + secs
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def test_paste_into_qt_text_field_and_restore():
    import desktop_kit as kit
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    from voxflow.inject import Injector

    app = QApplication.instance() or QApplication([])
    box = QPlainTextEdit()
    box.setWindowTitle("VoxFlow paste test")
    box.setGeometry(200, 200, 400, 200)
    box.show()
    _pump(app, 0.4)
    hwnd = int(box.winId())
    try:
        if not kit.take_foreground(hwnd, lambda s: _pump(app, s)):
            pytest.skip("could not take the foreground (locked workstation?): nothing was sent")
        box.setFocus()
        _pump(app, 0.2)
        with kit.saved_clipboard() as clip:
            if clip is None:
                pytest.skip("the clipboard could not be saved, so it was not touched")
            clip.set_text("ORIGINAL CLIPBOARD")
            text = "Hello from VoxFlow - unicode ✓ ñ 你好\nsecond line"
            result = {}
            keys = kit.guarded_keys(hwnd)             # refuses to paste unless OUR window is in front
            injector = Injector(keys=keys)

            # Injector runs on a worker thread like in the app; the Qt loop keeps pumping so Ctrl+V is delivered
            th = threading.Thread(target=lambda: result.setdefault("ok", injector.inject(text)))
            th.start()
            while th.is_alive():
                app.processEvents()
                time.sleep(0.005)
            _pump(app, 0.3)
            if type(keys).blocked:
                pytest.skip("another window took the foreground during the test: the paste was not sent")
            assert result["ok"] is True
            # `in`, not `==`: the test window briefly owns the real foreground, so stray keystrokes typed by a
            # person at the machine can land in it; the pasted block itself must still arrive intact.
            assert text in box.toPlainText()
            assert box.toPlainText().count(text) == 1, "pasted text must occur exactly once (a double paste fails)"
            assert clip.get_text() == "ORIGINAL CLIPBOARD"        # clipboard restored by the Injector
    finally:
        box.close()
        box.deleteLater()
        _pump(app, 0.05)
