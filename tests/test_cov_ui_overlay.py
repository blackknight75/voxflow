"""``voxflow.ui.overlay``: the Flow bar on its own, on a scripted clock: the hands-free and command looks, the
too-small-for-a-waveform moment, the glow cache, following the screens, the Win32 window flags (through a fake
``user32``) and the drag handlers outside arrange mode. Multi-monitor layouts are scripted screen objects."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import Signals, flush, grab_bytes
from tests.helpers import fake_time
from voxflow.ui import overlay as O
from voxflow.ui import theme as T
from voxflow.ui.overlay import OverlayPill

LEFT, RIGHT, NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton


class Voice:
    def __init__(self, level=0.4):
        self._level = level

    def level(self):
        return self._level

    def recent_samples(self, n):
        import numpy as np
        return np.full(n, self._level, dtype=np.float32)


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 500.0}
    monkeypatch.setattr(O, "time", fake_time(perf_counter=lambda: state["t"]))
    return state


@pytest.fixture
def pill(qapp, qt_messages, clock):
    made: list[OverlayPill] = []

    def build(**kw) -> OverlayPill:
        p = OverlayPill(Voice(), **kw)
        made.append(p)
        return p

    yield build
    for p in made:
        p._timer.stop()
        p._follow.stop()
        p.hide()
        p.deleteLater()
    flush()
    T.apply_theme("dark", False)
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


def frames(p: OverlayPill, clock, n=1, dt=0.02) -> None:
    """Advance the scripted clock and run the animation step by hand (the real 16 ms timer is stopped)."""
    p._timer.stop()
    for _ in range(n):
        clock["t"] += dt
        p._tick()


def picture(p: OverlayPill) -> bytes:
    return grab_bytes(p)


def mouse(widget, kind, pos, button=NONE, buttons=NONE) -> None:
    pos = QPointF(pos)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, QPointF(widget.mapToGlobal(pos.toPoint())), button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


# =========================================================================== looks
def test_hands_free_shows_a_lock_a_timer_and_a_short_hint(pill, clock):
    T.apply_theme("dark", True)                                # weights snap: one frame is the settled look
    p = pill()
    p.set_locked(True)
    assert p.state == "idle", "locking only means something while listening"
    p.set_state("listening")
    frames(p, clock)
    listening = picture(p)
    p.set_locked(True)
    assert p.state == "locked" and p._lock_t == clock["t"]
    frames(p, clock)
    with_hint = picture(p)
    assert with_hint != listening
    clock["t"] += 75                                           # the hint is gone, the timer reads 1:15
    frames(p, clock)
    later = picture(p)
    assert later not in (listening, with_hint)
    p.set_locked(True)
    assert p._lock_t < clock["t"] - 70, "locking twice does not restart the hint"
    p.set_locked(False)
    assert p.state == "listening"
    p.set_locked(False)
    assert p.state == "listening"


def test_command_mode_has_its_own_colours_and_glyph(pill, clock):
    T.apply_theme("dark", True)
    p = pill()
    p.set_state("listening", mode="dictate")
    frames(p, clock)
    dictate = picture(p)
    p.set_state("listening", mode="command")
    frames(p, clock)
    assert p.mode == "command" and p._cmd == 1.0 and picture(p) != dictate
    p.set_state("processing", mode="no-such-mode")
    assert p.mode == "command", "an unknown mode is ignored"


def test_every_state_has_its_own_picture_and_message(pill, clock):
    T.apply_theme("dark", True)
    p = pill()
    shots = {}
    for state, text, progress in (("idle", "", None), ("listening", "", None), ("processing", "", None),
                                  ("done", "", None), ("done", "Pasted · basic clean-up", None), ("error", "", None),
                                  ("loading", "", None), ("loading", "Downloading speech model", 0.42)):
        p.set_state(state, text, progress=progress)
        frames(p, clock)
        shots[(state, text)] = picture(p)
    assert len(set(shots.values())) == len(shots)
    p.set_state("done")
    assert p._display_text() == "Pasted"
    p.set_state("error")
    assert p._display_text() == "Something went wrong"
    p.set_state("loading")
    assert p._display_text() == "Getting ready…"
    p.set_state("loading", "Downloading speech model", progress=0.42)
    assert p._display_text() == "Downloading speech model  42%"
    p.set_state("no-such-state")
    assert p.state == "loading"
    p.set_paused(True)
    p.set_state("idle")
    frames(p, clock)
    assert p._target_size("idle") == O.PAUSED_SIZE and picture(p) != shots[("idle", "")], "paused says so on the bar"


def test_the_waveform_is_left_out_when_the_pill_has_no_room_for_it(pill, clock):
    p = pill()
    p.set_state("listening")
    frames(p, clock, 60)

    def bars(rect: QRectF, locked: float) -> int:
        img = QImage(300, 80, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        painter = QPainter(img)
        p._paint_wave(painter, rect, 1.0, 1.0, 0.0, T.DARK.CYAN, T.DARK.VIOLET, locked)
        painter.end()
        return sum(1 for x in range(0, 300, 2) for y in range(0, 80, 2) if img.pixelColor(x, y).alpha() > 0)

    assert bars(QRectF(10, 10, 250, 44), 0.0) > 200
    assert bars(QRectF(10, 10, 250, 44), 1.0) > 150, "hands-free leaves room for the timer but still has bars"
    assert bars(QRectF(10, 10, 110, 44), 1.0) == 0, "no room between the lock and the timer: no bars at all"
    assert bars(QRectF(10, 10, 60, 44), 0.0) == 0


def test_error_shakes_and_done_draws_its_tick(pill, clock):
    p = pill()
    p.set_state("error", "No speech heard")
    assert p._shake == 1.0
    frames(p, clock, 3)
    assert 0.0 < p._shake < 1.0 and picture(p)
    p.set_state("done")
    frames(p, clock, 2)
    early = picture(p)                                         # ring flash + the tick being drawn
    frames(p, clock, 40)
    assert picture(p) != early
    p.set_state("processing")
    frames(p, clock, 30)
    sweep = picture(p)
    frames(p, clock, 5)
    assert picture(p) != sweep, "the border sweep rotates while processing"
    p.set_state("loading", "Loading speech model")
    frames(p, clock, 30)
    p.set_state("done")                                        # the spinner fades out under the tick
    frames(p, clock, 2)
    assert p._weights["loading"] > 0.02 and picture(p)
    T.apply_theme("dark", True)
    p.set_state("error")
    assert p._shake == 0.0, "no shaking with reduced motion"


def test_the_glow_cache_is_bounded(pill, clock):
    T.apply_theme("dark", True)
    p = pill()
    p.set_state("processing")
    frames(p, clock)
    picture(p)
    assert len(p._glow_cache) == 1
    for i in range(30):
        p._glow_cache[("stale", i)] = (None, 0.0)
    p.set_state("done")                                        # other colours: a new entry is needed
    frames(p, clock)
    picture(p)
    assert len(p._glow_cache) == 1, "the cache is emptied instead of growing without limit"


def test_idle_settles_stops_the_clock_and_follows_or_hides(pill, clock):
    T.apply_theme("dark", True)
    shown = pill(show_idle=True)
    shown.set_state("done")
    shown.set_state("idle")
    assert shown._timer.isActive() and shown.isVisible()
    frames(shown, clock)
    assert not shown._timer.isActive() and shown._follow.isActive() and shown.isVisible(), "idle pill: no frames burnt"
    shown.set_show_idle(False)
    frames(shown, clock)
    assert not shown.isVisible() and not shown._follow.isActive()
    hidden = pill(show_idle=False)
    hidden.set_paused(True)
    assert not hidden.isVisible(), "nothing to show while idle with the idle pill off"
    hidden.set_state("processing")
    assert hidden.isVisible()
    frames(hidden, clock)
    assert not hidden._follow.isActive()


# =========================================================================== screens
class FakeScreen:
    def __init__(self, x, y, w, h, ratio=1.0):
        self._g, self._ratio = QRect(x, y, w, h), ratio

    def geometry(self):
        return self._g

    def availableGeometry(self):
        return self._g

    def devicePixelRatio(self):
        return self._ratio


class FakeUser32:
    def __init__(self, hwnd=1, rect=(0, 0, 0, 0), ok=1):
        self.hwnd, self.rect, self.ok = hwnd, rect, ok

    def GetForegroundWindow(self):
        return self.hwnd

    def GetWindowRect(self, _hwnd, ref):
        for i, v in enumerate(self.rect):
            ref._obj[i] = v
        return self.ok


def test_a_physical_point_is_mapped_to_the_screen_that_holds_it(monkeypatch, qapp):
    left, right = FakeScreen(0, 0, 1920, 1080), FakeScreen(1920, 0, 1280, 720, ratio=2.0)
    monkeypatch.setattr(O.QGuiApplication, "screens", staticmethod(lambda: [left, right]))
    assert O.screen_at_physical(10, 10) is left and O.screen_at_physical(1919, 1079) is left
    assert O.screen_at_physical(1920, 0) is right
    assert O.screen_at_physical(1920 + 2559, 1439) is right, "a scaled screen spans size x ratio physical pixels"
    assert O.screen_at_physical(1920 + 2560, 10) is None and O.screen_at_physical(-5, 10) is None


def test_the_active_screen_is_the_one_with_the_foreground_window(monkeypatch, qapp):
    real = QGuiApplication.primaryScreen()
    left, right = FakeScreen(0, 0, 1920, 1080), FakeScreen(1920, 0, 1920, 1080)
    monkeypatch.setattr(O.sys, "platform", "win32")
    monkeypatch.setattr(O.QGuiApplication, "screens", staticmethod(lambda: [left, right]))
    monkeypatch.setattr(O, "_user32", lambda: FakeUser32(rect=(2000, 100, 3000, 900)))
    assert O.active_screen() is right
    monkeypatch.setattr(O, "_user32", lambda: FakeUser32(rect=(100, 100, 900, 700)))
    assert O.active_screen() is left
    for u32 in (FakeUser32(hwnd=0), FakeUser32(ok=0), FakeUser32(rect=(-9000, -9000, -8000, -8000))):
        monkeypatch.setattr(O, "_user32", lambda u32=u32: u32)
        assert O.active_screen() is real, "no usable foreground window: the screen under the pointer"

    def broken():
        raise OSError("user32 is not available")

    monkeypatch.setattr(O, "_user32", broken)
    assert O.active_screen() is real
    monkeypatch.setattr(O.QGuiApplication, "screens", staticmethod(lambda: [left]))
    monkeypatch.setattr(O, "_user32", lambda: pytest.fail("one screen needs no Win32 call"))
    assert O.active_screen() is real


def test_the_real_user32_prototypes_are_declared(qapp):
    if O.sys.platform != "win32":
        pytest.skip("Windows only")
    u32 = O._user32()
    assert u32.GetForegroundWindow.argtypes == () and u32.GetWindowRect.restype is O.ctypes.c_int
    assert u32.SetWindowLongPtrW.restype is O.ctypes.c_ssize_t


def test_the_bar_is_placed_again_when_the_screens_change(pill, monkeypatch, clock):
    p = pill()
    placed: list = []
    monkeypatch.setattr(p, "_place", lambda: placed.append(1))
    watched: list = []
    monkeypatch.setattr(p, "_watch", watched.append)
    p._on_screens_changed()
    assert placed == [], "hidden: nothing to move"
    p.set_state("processing")
    placed.clear()
    screen = QGuiApplication.primaryScreen()
    p._on_screens_changed(screen)
    assert watched == [screen] and placed == [1], "a new screen is watched and the bar re-placed"
    p._on_screens_changed(QRect(0, 0, 10, 10))                 # a geometry change carries a rect, not a screen
    assert watched == [screen] and placed == [1, 1]

    def refuse(_screen):
        raise RuntimeError("screen already gone")

    monkeypatch.setattr(p, "_watch", refuse)
    p._on_screens_changed(screen)
    assert placed == [1, 1, 1]


def test_the_idle_pill_follows_the_active_screen_unless_the_user_placed_it(pill, monkeypatch, clock):
    T.apply_theme("dark", True)
    p = pill(show_idle=True)
    p.set_paused(False)                                        # brings the idle pill up
    assert p.isVisible()
    placed: list = []
    monkeypatch.setattr(p, "_place", lambda: placed.append(1))
    p._follow_screen()
    assert placed == [1]
    p._anchor = QPoint(400, 300)
    p._follow_screen()
    assert placed == [1], "a remembered position is not overridden"
    p._anchor = None
    p.state = "processing"
    p._follow_screen()
    p.state = "idle"
    p.hide()
    p._follow_screen()
    assert placed == [1]


# =========================================================================== native window flags
class StyleUser32:
    def __init__(self, style=0x100):
        self.style, self.calls = style, []

    def GetWindowLongPtrW(self, hwnd, index):
        self.calls.append(("get", index))
        return self.style

    def SetWindowLongPtrW(self, hwnd, index, value):
        self.calls.append(("set", index))
        self.style = value


def test_the_window_is_click_through_except_while_arranging(pill, monkeypatch, clock):
    u32 = StyleUser32()
    monkeypatch.setattr(O.sys, "platform", "win32")
    monkeypatch.setattr(O, "_user32", lambda: u32)
    p = pill()
    p.set_state("processing")
    assert p._native_done and u32.calls == [("get", -20), ("set", -20)]
    assert u32.style & 0x20 and u32.style & 0x80 and u32.style & 0x08000000 and u32.style & 0x80000 and u32.style & 0x100
    p.set_state("done")
    assert len(u32.calls) == 2, "applied once per native window"
    p.set_arranging(True)
    assert p.arranging and not u32.style & 0x20, "arranging: the bar takes the mouse"
    p.set_arranging(True)                                      # already arranging: nothing to do
    p.set_arranging(False)
    assert not p.arranging and u32.style & 0x20


def test_native_flags_are_skipped_off_windows_and_survive_a_failure(pill, monkeypatch, clock):
    p = pill()
    monkeypatch.setattr(O.sys, "platform", "linux")
    monkeypatch.setattr(O, "_user32", lambda: pytest.fail("no Win32 on other platforms"))
    p.set_state("processing")
    assert not p._native_done and p.isVisible()

    def broken():
        raise OSError("no user32")

    monkeypatch.setattr(O.sys, "platform", "win32")
    monkeypatch.setattr(O, "_user32", broken)
    p.set_state("done")
    assert not p._native_done and p.isVisible(), "the bar still shows; it is retried on the next change"


def test_only_the_pill_is_grabbable_while_arranging_on_a_real_desktop(pill, monkeypatch, clock):
    T.apply_theme("dark", True)
    p = pill()
    p.set_arranging(True)
    frames(p, clock)
    assert p._mask_rect.isNull() and p.mask().isEmpty(), "the offscreen platform has no input mask"
    monkeypatch.setattr(O.QGuiApplication, "platformName", staticmethod(lambda: "windows"))
    frames(p, clock)
    hit = p.pill_rect().adjusted(-10, -10, 10, 10).toAlignedRect()
    assert p._mask_rect == hit and p.mask().boundingRect() == hit
    masks: list = []
    monkeypatch.setattr(p, "setMask", masks.append)
    frames(p, clock)
    assert masks == [], "the mask is only set again when the pill changes size"
    p.set_arranging(False)
    assert p._mask_rect.isNull() and p.mask().isEmpty()


# =========================================================================== mouse outside arrange mode
def test_the_mouse_does_nothing_unless_arranging(pill, clock):
    p = pill()
    p.set_state("processing")
    moved = Signals(pos=p.position_changed)
    where = p.pos()
    centre = p.pill_rect().center().toPoint()
    mouse(p, QEvent.Type.MouseButtonPress, centre, LEFT, LEFT)
    mouse(p, QEvent.Type.MouseMove, centre + QPoint(40, -30), NONE, LEFT)
    mouse(p, QEvent.Type.MouseButtonRelease, centre + QPoint(40, -30), LEFT)
    assert p._drag is None and p.pos() == where and moved.pos == []
    p.set_arranging(True)
    mouse(p, QEvent.Type.MouseButtonPress, centre, RIGHT, RIGHT)
    assert p._drag is None, "only the left button drags"
    mouse(p, QEvent.Type.MouseButtonPress, centre, LEFT, LEFT)
    mouse(p, QEvent.Type.MouseMove, centre + QPoint(40, -30), NONE, LEFT)
    assert p.pos() == where + QPoint(40, -30)
    mouse(p, QEvent.Type.MouseButtonRelease, centre + QPoint(40, -30), LEFT)
    assert len(moved.pos) == 1 and p._drag is None
    anchor = moved.pos[0][0]
    assert anchor == [p._anchor.x(), p._anchor.y()]
    p.set_arranging(False)
    assert p.state == "processing", "the state from before arranging is back"


# =========================================================================== source + position
def test_a_failing_audio_source_reads_as_silence(pill, clock):
    p = pill()
    p.set_state("listening")
    frames(p, clock, 10)
    assert p._level > 0.5 and p._amp.max() > 0.3, "the scripted voice moves the bars"

    class Broken:
        def level(self):
            raise RuntimeError("stream died")

        def recent_samples(self, n):
            raise RuntimeError("stream died")

    p.set_source(Broken())
    frames(p, clock, 200)
    assert p._level < 0.01 and p._amp.max() < 0.01 and picture(p)
    p.set_source(None)
    frames(p, clock, 2)
    assert p._level < 0.01


def test_a_remembered_position_is_used_while_its_screen_exists(pill, clock):
    p = pill()
    area = QGuiApplication.primaryScreen().availableGeometry()
    default = p._target_anchor()
    assert default == QPoint(area.center().x(), area.bottom() - O.SCREEN_GAP)
    p.set_position([area.left() + 300, area.top() + 200])
    assert p._target_anchor() == QPoint(area.left() + 300, area.top() + 200)
    assert p.pos() == QPoint(area.left() + 300 - O.CANVAS_W // 2, area.top() + 200 - (O.CANVAS_H - O.BOTTOM_PAD))
    p.set_position([area.left() + 1, area.top() + 1])          # too close to the corner: pulled into the work area
    assert p._target_anchor() == QPoint(area.left() + 140, area.top() + 70)
    p.set_position([10 ** 6, 10 ** 6])                         # that monitor is gone
    assert p._target_anchor() == default
    for junk in ("x", [1], ["a", "b"], 5, None):
        p.set_position(junk)
        assert p._anchor is None and p._target_anchor() == default


def test_a_state_that_arrives_while_arranging_is_applied_afterwards(pill, clock):
    T.apply_theme("dark", True)
    p = pill()
    p.set_arranging(True)
    frames(p, clock)
    assert p._display_text() == "Drag me where you want me" and picture(p)
    p.set_state("done", "Pasted", "command")
    p.set_state("no-such-state")
    assert p.arranging
    p.set_arranging(False)
    assert (p.state, p.text, p.mode) == ("done", "Pasted", "command")


def test_going_idle_eases_out_over_several_frames_before_the_clock_stops(pill, clock):
    p = pill(show_idle=True)
    p.set_state("processing")
    frames(p, clock, 40)
    p.set_state("idle")
    p._timer.start()
    clock["t"] += 0.02
    p._tick()
    assert p._timer.isActive() and not p._follow.isActive() and p._w > O.SIZES["idle"][0] + 5, "still shrinking"
    for _ in range(400):
        clock["t"] += 0.02
        p._tick()
        if not p._timer.isActive():
            break
    assert not p._timer.isActive() and p._follow.isActive()
    assert (p._w, p._h) == O.SIZES["idle"] and p._master == pytest.approx(0.62)
