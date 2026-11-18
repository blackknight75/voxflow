"""The Flow bar: a floating dark-glass pill with the live waveform. Custom QPainter, 60 fps only while active.

States: idle, listening, locked (hands-free), processing, done, error, loading (model download / warm-up).
Modes: dictate (cyan -> violet) and command (magenta -> amber). The bar never takes focus and is click-through,
except in "arrange" mode where the user drags it to a new home that is then remembered.

It always uses the dark palette: it floats over arbitrary desktop content, not over the app's own surfaces.
"""
from __future__ import annotations

import ctypes
import math
import sys
import time

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QConicalGradient, QCursor, QFontMetricsF, QGuiApplication, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient, QRegion)
from PySide6.QtWidgets import QWidget

from . import a11y
from . import theme as T
from .icons import paint_icon

D = T.DARK
STATES = ("idle", "listening", "locked", "processing", "done", "error", "loading", "arrange")
MODES = ("dictate", "command")
# base pill size (w, h) per state; text states grow with their message
SIZES = {"idle": (64, 12), "listening": (236, 52), "locked": (276, 52), "processing": (172, 44), "done": (150, 44),
         "error": (236, 44), "loading": (250, 44), "arrange": (250, 52)}
CANVAS_W, CANVAS_H = 520, 150
BOTTOM_PAD = 40                 # canvas pixels below the pill (room for the glow)
SCREEN_GAP = 26                 # default distance from the bottom of the work area
PAUSED_SIZE = (92, 20)
LOCK_HINT_S = 2.6               # seconds the "Hands-free" hint stays above the bar after locking
MAX_TEXT_W = 350                # fits the longest controller message ("Window changed. Result copied: press Ctrl+V")
N_PTS = 36
_TEXT_STATES = ("done", "error", "loading", "arrange")
HINT_SEP = " · "                # the controller appends secondary hints to a message after this separator
_PREVIEW_STATES = ("listening", "locked", "processing")     # the live-preview caption exists only in these
PREVIEW_MAX_W = CANVAS_W - 40   # widest caption bubble (text plus padding); longer text loses its start
PREVIEW_PAD_X = 12              # bubble padding left and right of the text
PREVIEW_H = 26                  # bubble height
PREVIEW_GAP = 8                 # space between the bubble and the pill


def _user32():
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    hwnd_t = ctypes.c_void_p
    u32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    u32.GetWindowLongPtrW.argtypes = (hwnd_t, ctypes.c_int)
    u32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    u32.SetWindowLongPtrW.argtypes = (hwnd_t, ctypes.c_int, ctypes.c_ssize_t)
    u32.GetForegroundWindow.restype = hwnd_t
    u32.GetForegroundWindow.argtypes = ()
    u32.GetWindowRect.restype = ctypes.c_int
    u32.GetWindowRect.argtypes = (hwnd_t, ctypes.c_void_p)
    return u32


def screen_at_physical(x: int, y: int):
    """The screen containing a point given in physical pixels (what Win32 reports), or None.

    A QScreen's geometry has its origin in physical pixels and its size in logical ones, so the physical extent
    is ``size * devicePixelRatio`` from that origin."""
    for s in QGuiApplication.screens():
        g, k = s.geometry(), s.devicePixelRatio()
        if g.x() <= x < g.x() + g.width() * k and g.y() <= y < g.y() + g.height() * k:
            return s
    return None


def active_screen():
    """The screen the user is working on: the one holding the foreground window, else the one under the cursor."""
    if sys.platform == "win32" and len(QGuiApplication.screens()) > 1:
        try:
            u32 = _user32()
            hwnd = u32.GetForegroundWindow()
            rect = (ctypes.c_long * 4)()
            if hwnd and u32.GetWindowRect(hwnd, ctypes.byref(rect)):
                found = screen_at_physical((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
                if found is not None:
                    return found
        except Exception:
            pass
    return QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()


class OverlayPill(QWidget):
    position_changed = Signal(object)        # [x, y] (bottom-centre anchor, global logical px) after a drag

    def __init__(self, source=None, show_idle: bool = True):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                            | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("VoxFlow Flow bar")
        self.setAccessibleName("VoxFlow Flow bar")
        self.resize(CANVAS_W, CANVAS_H)

        self.source = source              # object with level() / recent_samples(n)
        self.show_idle = show_idle
        self.state = "idle"
        self.mode = "dictate"
        self.text = ""
        self.progress: float | None = None
        self.paused = False
        self._before_arrange = ("idle", "", "dictate")
        self._anchor: QPoint | None = None            # user-chosen home; None = bottom-centre of the active screen
        self._drag: QPoint | None = None
        self._mask_rect = QRect()
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._state_t = self._t0
        self._listen_t = self._t0
        self._weights = {s: (1.0 if s == "idle" else 0.0) for s in STATES}
        self._w, self._h = SIZES["idle"]
        self._text_w = 0.0
        self._cmd = 0.0                   # 0 = dictate colours, 1 = command colours
        self._master = 0.0                # overall opacity
        self._glow = 0.0
        self._amp = np.zeros(N_PTS)       # smoothed per-point amplitude 0..1
        self._hist = np.zeros(N_PTS // 2)
        self._acc = 0.0
        self._level = 0.0
        self._shake = 0.0
        self.fps = 0.0
        self._frames = 0
        self._fps_t = self._t0
        self._font = T.font(T.F_BODY, T.W_SEMI, 0.2)
        self._glow_cache: dict = {}       # (w, h, colours) -> pre-rendered glow at full strength (P5)
        self._lock_t = 0.0
        # live preview (set_preview): the words recognised so far, shown in a caption bubble above the pill. The
        # elided strings and their widths are worked out when the text changes, never while painting.
        self._pv_committed = ""
        self._pv_tentative = ""
        self._pv_head = ""                # the visible part that is final (may start with "…")
        self._pv_tail = ""                # the visible part that may still change
        self._pv_head_w = 0.0
        self._pv_tail_w = 0.0
        self._pv_w = 0.0                  # caption opacity weight 0..1
        self._pv_pm: QPixmap | None = None    # the finished bubble, drawn once per change and blitted every frame
        self._pv_dpr = 0.0
        self._pv_font = T.font(T.F_BODY, T.W_SEMI, 0.1)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._follow = QTimer(self)       # idle pill follows the active screen without any animation cost
        self._follow.setInterval(1500)
        self._follow.timeout.connect(self._follow_screen)
        self._native_done = False
        app = QGuiApplication.instance()
        app.screenAdded.connect(self._on_screens_changed)
        app.screenRemoved.connect(self._on_screens_changed)
        app.primaryScreenChanged.connect(self._on_screens_changed)
        for s in app.screens():
            self._watch(s)
        T.bus().changed.connect(self._kick)

    # ------------------------------------------------------------------ api
    def set_source(self, source) -> None:
        self.source = source

    def set_state(self, state: str, text: str = "", mode: str | None = None, progress: float | None = None) -> None:
        if state not in STATES or (self.state == "arrange" and state != "arrange"):
            if state in STATES:
                self._before_arrange = (state, text, mode or self.mode)   # applied when arranging ends
            return
        if mode in MODES:
            self.mode = mode
        now = time.perf_counter()
        if state == "locked" and self.state != "locked":
            self._lock_t = now
        was_listening = self.state in ("listening", "locked")
        prev = (self.state, self.text)
        if state != self.state:
            self._state_t = now
            if state == "error":
                self._shake = 0.0 if T.REDUCED_MOTION else 1.0
            if state in ("listening", "locked") and not was_listening:
                self._listen_t = now
                self._hist[:] = 0
                self._amp[:] = 0
                self._clear_preview()                  # a new utterance never starts under the last one's words
            if (self.state == "idle" and state != "idle") or (state == "listening" and not was_listening):
                self._place()                      # every new utterance appears on the screen in use at that moment
        self.state, self.text, self.progress = state, text, progress
        if state not in _PREVIEW_STATES:
            self._clear_preview()
        self._announce(prev)
        if state in _TEXT_STATES:
            shown = self._display_text()
            self._text_w = min(MAX_TEXT_W, QFontMetricsF(self._font).horizontalAdvance(shown))
        self._kick()

    _SPOKEN = {"done": "Pasted", "error": "Something went wrong", "loading": "Getting ready…"}

    def _announce(self, prev: tuple[str, str]) -> None:
        """The bar never has focus, so a screen reader hears its results only as announcements. Not while
        listening: speech from the speakers would end up in the dictation. Download progress is said once."""
        if self.state not in self._SPOKEN or prev == (self.state, self.text) or (
                self.state == "loading" and prev[0] == "loading"):
            return
        text = (self.text or self._SPOKEN[self.state]).replace(HINT_SEP, ". ")
        a11y.announce(self, text, assertive=self.state == "error")

    def set_preview(self, committed: str, tentative: str = "") -> None:
        """Live preview: the words recognised so far (``committed`` will not change, ``tentative`` may). Shown in a
        caption above the pill while listening, hands-free or processing; empty text removes it. Display only:
        nothing is stored, logged or announced. Text that arrives in any other state is dropped."""
        committed, tentative = str(committed or "").strip(), str(tentative or "").strip()
        if self.state not in _PREVIEW_STATES:
            committed = tentative = ""
        if (committed, tentative) == (self._pv_committed, self._pv_tentative):
            return
        self._pv_committed, self._pv_tentative = committed, tentative
        self._layout_preview()
        self._kick()

    def _clear_preview(self) -> None:
        self._pv_committed = self._pv_tentative = self._pv_head = self._pv_tail = ""
        self._pv_head_w = self._pv_tail_w = self._pv_w = 0.0
        self._pv_pm = None

    def _layout_preview(self) -> None:
        """Work out what the caption shows: the END of the text, with an ellipsis where the start was cut off, split
        into the final part and the part that may still change (drawn dimmer)."""
        head, tail = self._pv_committed, self._pv_tentative
        full = f"{head} {tail}" if head and tail else head or tail
        if not full:
            self._pv_head = self._pv_tail = ""
            self._pv_head_w = self._pv_tail_w = 0.0
            self._pv_pm = None
            return
        fm = QFontMetricsF(self._pv_font)
        shown = fm.elidedText(full, Qt.TextElideMode.ElideLeft, PREVIEW_MAX_W - 2 * PREVIEW_PAD_X)
        cut = max(1 if shown != full else 0, len(shown) - len(tail))     # where the tentative words begin
        self._pv_head, self._pv_tail = shown[:cut], shown[cut:]
        self._pv_head_w = fm.horizontalAdvance(self._pv_head)
        self._pv_tail_w = fm.horizontalAdvance(self._pv_tail)
        self._pv_pm = None                  # drawn again by the next paint (once, not every frame)

    def set_locked(self, locked: bool) -> None:
        """Hands-free lock indicator; only meaningful while listening."""
        if locked and self.state == "listening":
            self.set_state("locked", mode=self.mode)
        elif not locked and self.state == "locked":
            self.set_state("listening", mode=self.mode)

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        self._kick()

    def set_show_idle(self, show: bool) -> None:
        self.show_idle = bool(show)
        self._kick()

    def set_position(self, pos) -> None:
        """``[x, y]`` = remembered bottom-centre anchor, ``None`` = default (bottom-centre of the active screen)."""
        try:
            self._anchor = QPoint(int(pos[0]), int(pos[1])) if pos else None
        except (TypeError, ValueError, IndexError):
            self._anchor = None
        self._place()

    def set_arranging(self, on: bool) -> None:
        """Arrange mode: the bar becomes draggable and shows a hint; leaving it restores click-through."""
        if on == (self.state == "arrange"):
            return
        if on:
            self._before_arrange = (self.state, self.text, self.mode)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, not on)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not on)
        self.setCursor(Qt.CursorShape.SizeAllCursor if on else Qt.CursorShape.ArrowCursor)
        self._native_done = False
        if on:
            self.set_state("arrange", "Drag me where you want me")
        else:
            self.state = "idle"
            self._mask_rect = QRect()
            self.clearMask()
            self.set_state(*self._before_arrange)
        self._kick()

    @property
    def arranging(self) -> bool:
        return self.state == "arrange"

    def pill_rect(self) -> QRectF:
        """Current pill rectangle in widget coordinates."""
        return QRectF(CANVAS_W / 2 - self._w / 2, CANVAS_H - BOTTOM_PAD - self._h, self._w, self._h)

    # ------------------------------------------------------------------ window plumbing
    def _display_text(self) -> str:
        if self.state == "done":
            return self.text or "Pasted"
        if self.state == "error":
            return self.text or "Something went wrong"
        if self.state == "loading":
            base = self.text or "Getting ready…"
            return f"{base}  {int(self.progress * 100)}%" if self.progress is not None else base
        return self.text

    def _watch(self, screen) -> None:
        screen.geometryChanged.connect(self._on_screens_changed)
        screen.availableGeometryChanged.connect(self._on_screens_changed)

    def _on_screens_changed(self, *args) -> None:
        for a in args:
            if hasattr(a, "availableGeometryChanged"):       # a newly added QScreen
                try:
                    self._watch(a)
                except (RuntimeError, TypeError):
                    pass
        if self.isVisible():
            self._place()

    def _follow_screen(self) -> None:
        if self.state == "idle" and self.isVisible() and self._anchor is None:
            self._place()

    def _target_anchor(self) -> QPoint:
        if self._anchor is not None:
            for s in QGuiApplication.screens():
                g = s.availableGeometry()
                if g.adjusted(-8, -8, 8, 8).contains(self._anchor):
                    x = min(max(self._anchor.x(), g.left() + 140), g.right() - 140)
                    y = min(max(self._anchor.y(), g.top() + 70), g.bottom() - 6)
                    return QPoint(x, y)
        g = (active_screen() or QGuiApplication.primaryScreen()).availableGeometry()
        return QPoint(g.center().x(), g.bottom() - SCREEN_GAP)

    def _place(self) -> None:
        a = self._target_anchor()
        pos = QPoint(a.x() - CANVAS_W // 2, a.y() - (CANVAS_H - BOTTOM_PAD))
        if pos != self.pos():
            self.move(pos)

    def _apply_native_flags(self) -> None:
        if self._native_done or sys.platform != "win32":
            return
        try:
            u32 = _user32()
            hwnd = int(self.winId())
            gwl_exstyle = -20
            ex = u32.GetWindowLongPtrW(hwnd, gwl_exstyle)
            ex |= 0x80 | 0x08000000 | 0x80000            # TOOLWINDOW | NOACTIVATE | LAYERED
            ex = (ex & ~0x20) if self.state == "arrange" else (ex | 0x20)     # TRANSPARENT (click-through)
            u32.SetWindowLongPtrW(hwnd, gwl_exstyle, ex)
            self._native_done = True
        except Exception:
            pass

    def _kick(self) -> None:
        """(Re)start the animation clock and make sure the window is up when it should be."""
        if not self._timer.isActive():
            self._last = time.perf_counter()
            self._timer.start()
        if not self.isVisible() and (self.state != "idle" or self.show_idle):
            self._place()
            self.show()
        self._apply_native_flags()
        self.update()

    # ------------------------------------------------------------------ dragging (arrange mode only)
    def mousePressEvent(self, e):
        if self.state == "arrange" and e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        self._drag = None
        anchor = self.pos() + QPoint(CANVAS_W // 2, CANVAS_H - BOTTOM_PAD)
        self._anchor = anchor
        self._place()                       # clamp into the work area of whichever screen it was dropped on
        a = self._target_anchor()
        self._anchor = a
        self.position_changed.emit([a.x(), a.y()])

    # ------------------------------------------------------------------ audio
    def _read_audio(self, dt: float) -> None:
        samples = None
        lvl = 0.0
        src = self.source
        if src is not None and self.state in ("listening", "locked"):
            try:
                lvl = float(src.level())
                samples = src.recent_samples(1920)
            except Exception:
                samples = None
        self._level = T.smooth(self._level, min(1.0, lvl * 2.2), dt, 22 if lvl * 2.2 > self._level else 9)
        # push a new envelope value ~every 28 ms; newest sits in the centre and ripples outwards (mirrored)
        if samples is not None and len(samples) >= 256:
            s = np.asarray(samples[-640:], dtype=np.float32)
            env = float(np.sqrt((s ** 2).mean()))
        else:
            env = 0.0
        env = min(1.0, max(env * 9.0, self._level * 0.5))
        self._acc += dt
        while self._acc >= 0.028:
            self._acc -= 0.028
            self._hist = np.roll(self._hist, 1)
            self._hist[0] = env
        half = N_PTS // 2
        target = np.concatenate([self._hist[:half][::-1], self._hist[:half]])
        target = np.convolve(np.pad(target, 1, mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid")
        up = target > self._amp
        self._amp = np.where(up, self._amp + (target - self._amp) * min(1.0, dt * 28),
                             self._amp + (target - self._amp) * min(1.0, dt * 12))

    # ------------------------------------------------------------------ animation
    def _target_size(self, s: str) -> tuple[float, float]:
        w, h = SIZES[s]
        if s == "idle" and self.paused:
            w, h = PAUSED_SIZE                 # wide enough to say "Paused": the bar is the only place that answers "why nothing?"
        if s in _TEXT_STATES and s == self.state:
            w = max(120.0, 46 + self._text_w + 22 + (26 if s == "arrange" else 0))
        return w, h

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(0.05, now - self._last)
        self._last = now
        self._frames += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._frames / (now - self._fps_t)
            self._frames, self._fps_t = 0, now
        snap = T.REDUCED_MOTION

        for s in STATES:
            goal = 1.0 if s == self.state else 0.0
            self._weights[s] = goal if snap else T.smooth(self._weights[s], goal, dt, 11)
        tw, th = self._target_size(self.state)
        if snap:
            self._w, self._h = tw, th
        else:
            self._w = T.smooth(self._w, tw, dt, 15)
            self._h = T.smooth(self._h, th, dt, 15)
        idle = self.state == "idle"
        visible_target = (0.0 if not self.show_idle else (0.95 if self.paused else 0.62)) if idle else 1.0   # paused: its label must stay readable (>= 4.5:1)
        self._master = visible_target if snap else T.smooth(self._master, visible_target, dt, 10)
        cmd_target = 1.0 if self.mode == "command" else 0.0
        self._cmd = cmd_target if snap else T.smooth(self._cmd, cmd_target, dt, 12)
        glow_target = {"idle": 0.0, "listening": 0.5 + 0.5 * self._level, "locked": 0.5 + 0.5 * self._level,
                       "processing": 0.7, "done": 0.9, "error": 0.9, "loading": 0.45, "arrange": 0.8}[self.state]
        self._glow = glow_target if snap else T.smooth(self._glow, glow_target, dt, 8)
        pv_target = 1.0 if self._pv_head or self._pv_tail else 0.0
        self._pv_w = pv_target if snap else T.smooth(self._pv_w, pv_target, dt, 14)
        self._shake = max(0.0, self._shake - dt * 1.6)
        self._read_audio(dt)
        if self.state == "arrange" and QGuiApplication.platformName() != "offscreen":
            hit = self.pill_rect().adjusted(-10, -10, 10, 10).toAlignedRect()
            if hit != self._mask_rect:          # only the pill is grabbable; clicks around it pass through
                self._mask_rect = hit
                self.setMask(QRegion(hit))
        self.update()

        if idle:
            settled = abs(self._w - tw) < 0.3 and abs(self._h - th) < 0.3 and abs(self._master - visible_target) < 0.01 \
                and self._weights["idle"] > 0.995
            if settled:                    # nothing moves while idle: stop burning frames
                self._w, self._h, self._master = tw, th, visible_target
                self._timer.stop()
                self.update()
                if self.show_idle:
                    self._follow.start()
                else:
                    self._follow.stop()
                    self.hide()
                return
        self._follow.stop()

    # ------------------------------------------------------------------ painting
    def _accents(self) -> tuple[QColor, QColor]:
        a = T.mix(D.CYAN, D.MAGENTA, self._cmd)
        b = T.mix(D.VIOLET, D.AMBER, self._cmd)
        done_w, err_w = self._weights["done"], self._weights["error"]
        if done_w > 0.01 or err_w > 0.01:
            a = T.mix(T.mix(a, D.GREEN, done_w), D.RED, err_w)
            b = T.mix(T.mix(b, D.CYAN, done_w), D.MAGENTA, err_w)
        return a, b

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        now = time.perf_counter()
        t = 0.0 if T.REDUCED_MOTION else now - self._t0
        since = now - self._state_t
        wgt = self._weights

        shake_x = math.sin(since * 48) * 7 * self._shake * self._shake if self.state == "error" else 0.0
        rect = self.pill_rect().translated(shake_x, 0)
        radius = rect.height() / 2
        p.setOpacity(self._master)
        ca, cb = self._accents()

        self._paint_glow(p, rect, radius, ca, cb, t)
        self._paint_body(p, rect, radius, ca, cb, t)

        listen_w = wgt["listening"] + wgt["locked"]
        pw = wgt["processing"]
        if listen_w > 0.01 or pw > 0.01:
            p.save()
            p.setClipPath(T.rounded(rect, radius))
            self._paint_wave(p, rect, t, listen_w, pw, ca, cb, wgt["locked"])
            p.restore()
        if listen_w > 0.02:
            self._paint_listen_extras(p, rect, t, listen_w, wgt["locked"], ca, now)
        if self._pv_w > 0.01:
            self._paint_preview(p, rect, self._pv_w)
        elif self.state == "locked" and now - self._lock_t < LOCK_HINT_S:      # the caption wins the spot
            self._paint_lock_hint(p, rect, now - self._lock_t)
        if wgt["done"] > 0.02:
            self._paint_done(p, rect, since, wgt["done"])
        if wgt["error"] > 0.02:
            self._paint_error(p, rect, wgt["error"])
        if wgt["loading"] > 0.02:
            self._paint_loading(p, rect, t, wgt["loading"], ca, cb)
        if wgt["arrange"] > 0.02:
            self._paint_arrange(p, rect, wgt["arrange"], ca)
        if wgt["idle"] > 0.02 and self.state == "idle":
            self._paint_idle(p, rect, wgt["idle"], ca, cb)
        p.end()

    GLOW_LAYERS = 9

    def _glow_pixmap(self, rect, radius, ca, cb) -> tuple[QPixmap, float]:
        """The glow at full strength, rendered once per (pill size, accent colours) and blitted every frame.

        Painting nine anti-aliased gradient round-rects each frame was the bulk of the listening cost (P5); the
        glow only changes slowly, so its strength is applied as opacity when blitting."""
        reach = self.GLOW_LAYERS * 2.6
        dpr = self.devicePixelRatioF()
        key = (round(rect.width() / 2), round(rect.height() / 2), ca.rgb() >> 3, cb.rgb() >> 3, dpr)
        hit = self._glow_cache.get(key)
        if hit is None:
            if len(self._glow_cache) > 24:
                self._glow_cache.clear()
            w, h = int(rect.width() + 2 * reach) + 2, int(rect.height() + 2 * reach) + 2
            pm = QPixmap(int(w * dpr), int(h * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            base = QRectF(reach + 1, reach + 1, rect.width(), rect.height())
            q.setPen(Qt.PenStyle.NoPen)
            for i in range(self.GLOW_LAYERS, 0, -1):
                grow = i * 2.6
                r = base.adjusted(-grow, -grow, grow, grow)
                a = (1 - i / (self.GLOW_LAYERS + 1)) ** 2.2 * 0.10
                grad = QLinearGradient(r.left(), 0, r.right(), 0)
                grad.setColorAt(0, T.alpha(ca, a))
                grad.setColorAt(1, T.alpha(cb, a))
                q.setBrush(grad)
                q.drawRoundedRect(r, radius + grow, radius + grow)
            q.end()
            hit = self._glow_cache[key] = (pm, reach + 1)
        return hit

    def _paint_glow(self, p, rect, radius, ca, cb, t) -> None:
        g = self._glow * (0.85 + 0.15 * math.sin(t * 3.0))
        if g < 0.02:
            return
        pm, off = self._glow_pixmap(rect, radius, ca, cb)
        p.save()
        p.setOpacity(p.opacity() * min(1.0, g))
        p.drawPixmap(QPointF(rect.left() - off, rect.top() - off), pm)
        p.restore()

    def _paint_body(self, p, rect, radius, ca, cb, t) -> None:
        path = T.rounded(rect, radius)
        # contact shadow so the pill reads on light wallpapers too
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(5, 0, -1):
            p.setBrush(QColor(0, 0, 0, 10))
            p.drawRoundedRect(rect.adjusted(-i, -i + 2, i, i + 3), radius + i, radius + i)
        fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        fill.setColorAt(0, QColor(27, 31, 50, 242))
        fill.setColorAt(1, QColor(9, 11, 19, 247))
        p.fillPath(path, fill)
        listen_w = self._weights["listening"] + self._weights["locked"]
        wash = QRadialGradient(rect.center(), max(1.0, rect.width() * 0.6))
        wash.setColorAt(0, T.alpha(ca, 0.10 + 0.22 * self._level * listen_w))
        wash.setColorAt(1, T.alpha(cb, 0.0))
        p.fillPath(path, wash)
        if rect.height() > 20:               # top specular edge
            spec = QLinearGradient(rect.left(), 0, rect.right(), 0)
            spec.setColorAt(0, QColor(255, 255, 255, 0))
            spec.setColorAt(0.5, QColor(255, 255, 255, 60))
            spec.setColorAt(1, QColor(255, 255, 255, 0))
            p.setPen(QPen(spec, 1.0))
            p.drawLine(QPointF(rect.left() + radius, rect.top() + 0.8), QPointF(rect.right() - radius, rect.top() + 0.8))

        # border: linear accent gradient, becoming a rotating conical sweep while processing
        pw = self._weights["processing"]
        border_a = 0.38 + 0.5 * (1 - self._weights["idle"])
        lin = QLinearGradient(rect.left(), 0, rect.right(), 0)
        lin.setColorAt(0, T.alpha(ca, border_a))
        lin.setColorAt(1, T.alpha(cb, border_a))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(lin, 1.3))
        p.drawPath(path)
        if pw > 0.02 and not T.REDUCED_MOTION:
            con = QConicalGradient(rect.center(), -(t * 260) % 360)
            con.setColorAt(0.00, T.alpha(ca, 0.0))
            con.setColorAt(0.55, T.alpha(cb, 0.0))
            con.setColorAt(0.85, T.alpha(cb, 0.9 * pw))
            con.setColorAt(1.00, T.alpha(ca, 1.0 * pw))
            p.setPen(QPen(con, 2.2))
            p.drawPath(path)

    # ---- waveform ------------------------------------------------------
    def _wave_points(self, t, lw, pw):
        xs = np.linspace(0, 1, N_PTS)
        window = 0.25 + 0.75 * np.sin(np.pi * xs) ** 0.7
        # listening amplitude: voice envelope + faint ambient motion so it is alive in silence
        ambient = 0.07 + 0.04 * np.sin(xs * 9 + t * 2.4) + 0.025 * np.sin(xs * 17 - t * 3.7)
        voice = np.clip(self._amp * 1.5, 0, 1)
        listen = np.clip(np.maximum(ambient, ambient * 0.4 + voice), 0, 1) * window
        # processing: a bright bump travels across, otherwise a calm low ripple
        pos = (t * 0.9) % 1.4 - 0.2
        bump = np.exp(-((xs - pos) / 0.11) ** 2)
        proc = (0.06 + 0.04 * np.sin(xs * 12 - t * 6) + (0.62 * bump if not T.REDUCED_MOTION else 0.22)) * window
        return xs, (listen * lw + proc * pw) / (lw + pw + 1e-6)

    def _paint_wave(self, p, rect, t, lw, pw, ca, cb, locked_w) -> None:
        """Crisp mirrored bars: thin rounded capsules in the accent gradient with a soft neon glow."""
        xs, amp = self._wave_points(t, lw, pw)
        pad = rect.height() * 0.62
        x0 = rect.left() + pad + 7 * max(locked_w, self._cmd)     # the lock / sparkle glyph is wider than the dot
        x1 = rect.right() - pad * 0.7 - 40 * locked_w        # room for the hands-free timer
        if x1 - x0 < 20:
            return
        cy = rect.center().y()
        hmax = rect.height() * 0.40
        fade = min(1.0, (lw + pw))
        n = len(xs)
        step = (x1 - x0) / (n - 1)
        width = max(2.0, step * 0.52)
        flow = 0.5 + 0.5 * math.sin(t * 0.8)            # slow colour drift
        heights = np.maximum(1.3, np.clip(amp, 0, 1) ** 0.8 * hmax)
        cols = [T.mix(ca, cb, min(1.0, max(0.0, x * 0.85 + flow * 0.15))) for x in xs]
        p.setBrush(Qt.BrushStyle.NoBrush)
        for w_mul, a in ((3.4, 0.09), (2.1, 0.17)):     # glow passes (wide + faint), then the crisp core
            for i in range(n):
                px = x0 + step * i
                p.setPen(QPen(T.alpha(cols[i], a * fade), width * w_mul, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(px, cy - heights[i]), QPointF(px, cy + heights[i]))
        for i in range(n):
            px = x0 + step * i
            p.setPen(QPen(T.alpha(cols[i], 0.98 * fade), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(px, cy - heights[i]), QPointF(px, cy + heights[i]))
            if heights[i] > hmax * 0.35:                # bright specular core on taller bars
                p.setPen(QPen(T.alpha(QColor(255, 255, 255), 0.45 * fade), width * 0.34, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(px, cy - heights[i] * 0.55), QPointF(px, cy + heights[i] * 0.55))

    def _paint_listen_extras(self, p, rect, t, lw, locked_w, ca, now) -> None:
        r = rect.height() * 0.5
        c = QPointF(rect.left() + r * 0.95, rect.center().y())
        pulse = 0.5 + 0.5 * math.sin(t * 5.0)
        halo = QRadialGradient(c, 9 + 3 * pulse)
        halo.setColorAt(0, T.alpha(ca, 0.55 * lw))
        halo.setColorAt(1, T.alpha(ca, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(c, 12, 12)
        if locked_w > 0.5:                              # hands-free: a lock instead of the record dot
            paint_icon(p, "lock", QRectF(c.x() - 8, c.y() - 8.5, 16, 16), T.alpha(D.TEXT, lw), 1.8)
        elif self._cmd > 0.5:                           # command mode: sparkle
            paint_icon(p, "sparkle", QRectF(c.x() - 8, c.y() - 8, 16, 16), T.alpha(ca, lw), 1.8)
        else:
            p.setBrush(T.alpha(ca, lw))
            p.drawEllipse(c, 3.2 + 0.5 * pulse, 3.2 + 0.5 * pulse)
        if locked_w > 0.05:                             # elapsed time while hands-free
            secs = int(max(0.0, now - self._listen_t))
            p.setFont(T.font(T.F_SMALL, T.W_SEMI))
            p.setPen(T.alpha(D.TEXT_DIM, locked_w * lw))
            p.drawText(QRectF(rect.right() - 58, rect.top(), 42, rect.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{secs // 60}:{secs % 60:02d}")

    def _paint_lock_hint(self, p, rect, age) -> None:
        """A short caption above the bar right after the hands-free lock engages."""
        fade = min(1.0, age / 0.2, (LOCK_HINT_S - age) / 0.4)
        text = "Hands-free · tap the hotkey to finish"
        p.setFont(T.font(T.F_SMALL, T.W_SEMI))
        w = QFontMetricsF(p.font()).horizontalAdvance(text) + 24
        box = QRectF(rect.center().x() - w / 2, rect.top() - 30, w, 22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(12, 14, 24, int(235 * fade)))
        p.drawRoundedRect(box, 11, 11)
        p.setPen(T.alpha(D.TEXT, fade))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _preview_pixmap(self) -> QPixmap:
        """The caption bubble at full strength, rendered when the text (or the screen's scale) changes and blitted
        every frame with the fade as opacity: painting rounded rectangles and two text runs per frame costs more
        than the caption is worth."""
        dpr = self.devicePixelRatioF()
        if self._pv_pm is None or self._pv_dpr != dpr:
            width = self._pv_head_w + self._pv_tail_w + 2 * PREVIEW_PAD_X
            pm = QPixmap(int(math.ceil(width * dpr)), int(PREVIEW_H * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            q.setPen(QPen(QColor(255, 255, 255, 46), 1.0))
            q.setBrush(QColor(12, 14, 24, 238))
            q.drawRoundedRect(QRectF(0.5, 0.5, width - 1, PREVIEW_H - 1), PREVIEW_H / 2, PREVIEW_H / 2)
            q.setFont(self._pv_font)
            flags = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            q.setPen(D.TEXT)
            q.drawText(QRectF(PREVIEW_PAD_X, 0, self._pv_head_w + 4, PREVIEW_H), flags, self._pv_head)
            q.setPen(T.alpha(D.TEXT_DIM, 0.9))
            q.drawText(QRectF(PREVIEW_PAD_X + self._pv_head_w, 0, self._pv_tail_w + 4, PREVIEW_H), flags,
                       self._pv_tail)
            q.end()
            self._pv_pm, self._pv_dpr = pm, dpr
        return self._pv_pm

    def _paint_preview(self, p, rect, k) -> None:
        """The live-preview caption: a dark bubble above the pill, the final words in the normal text colour and the
        ones that may still change dimmer (see ``_layout_preview`` and ``_preview_pixmap``)."""
        pm = self._preview_pixmap()
        p.save()
        p.setOpacity(p.opacity() * k)
        p.drawPixmap(QPointF(rect.center().x() - pm.width() / pm.devicePixelRatio() / 2,
                             rect.top() - PREVIEW_GAP - PREVIEW_H), pm)
        p.restore()

    def _paint_text(self, p, rect, wgt, text, right_pad: float = 20) -> None:
        """Message text. Anything after " · " is a secondary hint (e.g. "Pasted · basic clean-up") and is
        painted dimmer, so the main outcome still reads first."""
        p.setFont(self._font)
        avail = rect.width() - 44 - right_pad
        fm = QFontMetricsF(self._font)
        shown = fm.elidedText(text, Qt.TextElideMode.ElideRight, avail + 2)
        main, sep, hint = shown.partition(HINT_SEP)
        area = QRectF(rect.left() + 44, rect.top(), avail + 4, rect.height())
        flags = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        p.setPen(T.alpha(D.TEXT, wgt))
        p.drawText(area, flags, main)
        if sep:
            p.setPen(T.alpha(D.TEXT_DIM, wgt))
            p.drawText(area.adjusted(fm.horizontalAdvance(main), 0, 0, 0), flags, sep + hint)

    def _paint_done(self, p, rect, since, wgt) -> None:
        k = T.ease_out(since / 0.55)
        if k < 1.0 and not T.REDUCED_MOTION:            # expanding ring flash
            ring = rect.adjusted(-k * 14, -k * 14, k * 14, k * 14)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(T.alpha(D.GREEN, (1 - k) * 0.6 * wgt), 2.0))
            p.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)
        c = QPointF(rect.left() + 26, rect.center().y())
        prog = 1.0 if T.REDUCED_MOTION else T.ease_out(since / 0.28)
        pts = [QPointF(c.x() - 6, c.y()), QPointF(c.x() - 2, c.y() + 4.5), QPointF(c.x() + 6.5, c.y() - 5)]
        path = QPainterPath(pts[0])
        if prog < 0.4:
            path.lineTo(pts[0] + (pts[1] - pts[0]) * (prog / 0.4))
        else:
            path.lineTo(pts[1])
            path.lineTo(pts[1] + (pts[2] - pts[1]) * ((prog - 0.4) / 0.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for width, a in ((6, 0.12), (3.6, 0.25), (2.2, 1.0)):
            p.setPen(QPen(T.alpha(D.GREEN, a * wgt), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.drawPath(path)
        if self.state == "done":
            self._paint_text(p, rect, wgt, self._display_text())

    def _paint_error(self, p, rect, wgt) -> None:
        c = QPointF(rect.left() + 26, rect.center().y())
        for width, a in ((6, 0.12), (3.6, 0.25), (2.2, 1.0)):
            p.setPen(QPen(T.alpha(D.RED, a * wgt), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(c.x() - 4.5, c.y() - 4.5), QPointF(c.x() + 4.5, c.y() + 4.5))
            p.drawLine(QPointF(c.x() - 4.5, c.y() + 4.5), QPointF(c.x() + 4.5, c.y() - 4.5))
        if self.state == "error":
            self._paint_text(p, rect, wgt, self._display_text())

    def _paint_loading(self, p, rect, t, wgt, ca, cb) -> None:
        c = QPointF(rect.left() + 26, rect.center().y())
        ring = QRectF(c.x() - 8, c.y() - 8, 16, 16)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(T.alpha(QColor(255, 255, 255), 0.14 * wgt), 2.2))
        p.drawEllipse(ring)
        p.setPen(QPen(T.alpha(ca, wgt), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if self.progress is not None:
            p.drawArc(ring, 90 * 16, int(-360 * 16 * max(0.02, min(1.0, self.progress))))
        else:
            p.drawArc(ring, int(-t * 300 * 16) + 90 * 16, -110 * 16)
        if self.state == "loading":
            self._paint_text(p, rect, wgt, self._display_text())

    def _paint_arrange(self, p, rect, wgt, ca) -> None:
        c = QPointF(rect.left() + 26, rect.center().y())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.alpha(ca, wgt))
        for dx in (-3.5, 3.5):                         # grip dots
            for dy in (-6, 0, 6):
                p.drawEllipse(QPointF(c.x() + dx, c.y() + dy), 1.7, 1.7)
        if self.state == "arrange":
            self._paint_text(p, rect, wgt, self.text)

    def _paint_idle(self, p, rect, wgt, ca, cb) -> None:
        cy = rect.center().y()
        if self.paused:                                 # "pause" glyph + a word: the bar must answer "why nothing?"
            p.setPen(QPen(T.alpha(D.TEXT, wgt), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            x = rect.left() + 12
            for dx in (0, 4.5):
                p.drawLine(QPointF(x + dx, cy - 3.2), QPointF(x + dx, cy + 3.2))
            p.setFont(T.font(T.F_CAPTION, T.W_SEMI, 0.3))
            p.setPen(T.alpha(D.TEXT, wgt))
            p.drawText(QRectF(x + 12, rect.top(), rect.width() - 26, rect.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "Paused")
            return
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, int(70 * wgt)), 1.0))       # hairline so it also reads on dark windows
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), rect.height() / 2, rect.height() / 2)
        g = QLinearGradient(rect.left() + 6, 0, rect.right() - 6, 0)
        g.setColorAt(0, T.alpha(ca, 0.0))
        g.setColorAt(0.5, T.alpha(ca, 0.75 * wgt))
        g.setColorAt(1, T.alpha(cb, 0.0))
        p.setPen(QPen(g, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(rect.left() + 9, cy), QPointF(rect.right() - 9, cy))
