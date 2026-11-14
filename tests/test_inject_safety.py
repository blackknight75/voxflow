"""Issue 5 / 8b: clipboard safety and restore timing."""
import sys

import pytest

from voxflow.inject import Injector, _is_safe_format


class Clip:
    def __init__(self, snap="orig", text_snap="orig-text", seq_bump=0):
        self.content = "orig"
        self.snap, self.text_snap = snap, text_snap
        self.seq = 1
        self.log = []

    def snapshot(self):
        return self.snap

    def text_snapshot(self):
        return self.text_snap

    def set_text(self, t):
        self.content, self.seq = t, self.seq + 1
        self.log.append("set")
        return self.seq

    def sequence(self):
        return self.seq

    def restore(self, s):
        self.content = s
        self.log.append(("restore", s))


class Keys:
    def __init__(self):
        self.pasted, self.typed = 0, []

    def paste(self):
        self.pasted += 1
        return True

    def type_text(self, t):
        self.typed.append(t)
        return True

    def modifiers_down(self):
        return False


def mk(clip, **kw):
    sleeps = []
    return Injector(clip, Keys(), sleep=sleeps.append, **kw), sleeps


def test_failed_full_snapshot_falls_back_to_text_snapshot():
    clip = Clip(snap=None, text_snap="just text")
    inj, _ = mk(clip)
    assert inj.inject("hello")
    assert clip.log[-1] == ("restore", "just text")            # at least the text comes back


def test_unreadable_clipboard_is_never_clobbered():
    clip = Clip(snap=None, text_snap=None)
    inj, _ = mk(clip)
    assert inj.inject("hello")
    assert clip.log == [] and clip.content == "orig"           # clipboard untouched
    assert inj.keys.typed == ["hello"] and inj.keys.pasted == 0  # typed directly instead


def test_unreadable_clipboard_still_pastes_when_restore_disabled():
    clip = Clip(snap=None, text_snap=None)
    inj, _ = mk(clip, restore=False)
    inj.inject("hello")
    assert inj.keys.pasted == 1


def test_restore_delay_configurable_and_safer_default():
    inj, sleeps = mk(Clip())
    inj.inject("x")
    assert sleeps[-1] == 0.5                                   # default: 500 ms, was 180 ms
    inj, sleeps = mk(Clip(), restore_delay=1.25)
    inj.inject("x")
    assert sleeps[-1] == 1.25


def test_format_whitelist():
    for fmt in (1, 7, 8, 13, 15, 17, 0xC123, 0xC00A):          # text, DIB, HDROP, DIBV5, registered
        assert _is_safe_format(fmt)
    for fmt in (2, 3, 9, 14, 0x82, 0x200, 0x2FF, 0x300, 0x3FF):  # BITMAP, metafiles, palette, private, GDI
        assert not _is_safe_format(fmt)


@pytest.mark.desktop
@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_real_clipboard_roundtrip_text_image_and_empty():
    from PySide6.QtGui import QColor, QGuiApplication, QImage
    from PySide6.QtWidgets import QApplication
    from voxflow.inject import Win32Clipboard

    app = QApplication.instance() or QApplication([])
    cb = Win32Clipboard()
    qcb = QGuiApplication.clipboard()

    import desktop_kit as kit
    with kit.saved_clipboard() as saved:           # the user's clipboard is put back afterwards
        if saved is None:
            pytest.skip("the clipboard could not be saved (in use / locked workstation): not touched")

        img = QImage(8, 8, QImage.Format.Format_ARGB32)
        img.fill(QColor(10, 200, 30))
        qcb.setImage(img)
        app.processEvents()
        snap = cb.snapshot()
        assert snap, "image clipboard must produce a snapshot (via DIB)"
        cb.set_text("temporary dictation")
        assert cb.get_text() == "temporary dictation"
        assert cb.restore(snap)
        after = cb.snapshot()                      # compare raw formats (independent of Qt's clipboard cache)
        assert {f for f, _ in after} == {f for f, _ in snap}
        assert dict(after).get(8) == dict(snap).get(8) or dict(after).get(17) == dict(snap).get(17)
        assert any(f in (8, 17) for f, _ in after), "DIB must survive the round trip"

        # empty clipboard: snapshot is [] (not None) and restore leaves it empty
        cb.restore([])
        empty = cb.snapshot()
        assert empty == []
        cb.set_text("x")
        cb.restore(empty)
        assert cb.get_text() == ""

        # text-only fallback snapshot
        cb.set_text("keep me")
        ts = cb.text_snapshot()
        cb.set_text("other")
        cb.restore(ts)
        assert cb.get_text() == "keep me"
