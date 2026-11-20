"""Main window: frameless glass shell with sidebar navigation, six pages, toasts and in-window dialogs.

The window talks only to the controller facade (``voxflow.controller.Controller``). Closing it hides it to the tray.
Keyboard: Ctrl+1..6 switch pages, Ctrl+F goes to the search box of the page (History's when the page has none),
Esc closes a dialog or hides the window.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QLinearGradient, QPainter, QPen, QRadialGradient, QShortcut
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from . import theme as T
from .components import Modal, NavButton, StatusChip, ToastHost, WindowButton
from .pages import DictionaryPage, HistoryPage, HomePage, SettingsPage, SnippetsPage, StylePage
from .widgets import Button, bring_to_front, paint_logo, painter

PAGES = (("home", "Home", "home", HomePage), ("history", "History", "history", HistoryPage),
         ("dictionary", "Dictionary", "book", DictionaryPage), ("snippets", "Snippets", "snippet", SnippetsPage),
         ("style", "Style", "style", StylePage), ("settings", "Settings", "settings", SettingsPage))
PAGE_KEYS = tuple(p[0] for p in PAGES)


class _Panel(QWidget):
    """The visible rounded body of the window (everything inside the shadow margin)."""

    SIDEBAR_W = 212
    TITLE_H = 46

    def __init__(self, parent=None):
        super().__init__(parent)
        self.corner_radius = T.R_XL
        T.bus().changed.connect(self.update)

    def paintEvent(self, _):
        p = painter(self)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        rad = self.corner_radius
        path = T.rounded(r, rad)
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, T.BG1)
        g.setColorAt(1, T.BG0)
        p.fillPath(path, g)
        p.save()
        p.setClipPath(path)
        for col, centre, radius in ((T.CYAN, QPointF(r.left() + 70, r.top() + 10), 340.0),
                                    (T.VIOLET, QPointF(r.right() - 40, r.bottom() - 10), 420.0)):
            rg = QRadialGradient(centre, radius)        # ambient colour blooms
            rg.setColorAt(0, T.alpha(col, T.BLOOM))
            rg.setColorAt(1, T.alpha(col, 0))
            p.fillRect(r, rg)
        side = QRectF(r.left(), r.top(), self.SIDEBAR_W, r.height())
        p.fillRect(side, T.SIDEBAR)
        p.setPen(QPen(T.STROKE, 1))
        p.drawLine(QPointF(side.right(), r.top()), QPointF(side.right(), r.bottom()))
        p.restore()
        paint_logo(p, QRectF(r.left() + 20, r.top() + 18, 28, 28))
        p.setPen(T.TEXT)
        p.setFont(T.font(T.F_TITLE + 0.5, T.W_BOLD, 0.2))
        p.drawText(QRectF(r.left() + 58, r.top() + 18, 140, 28), Qt.AlignmentFlag.AlignVCenter, "VoxFlow")
        border = QLinearGradient(r.topLeft(), r.bottomRight())
        border.setColorAt(0, T.alpha(T.CYAN, 0.50))
        border.setColorAt(0.5, T.STROKE_STRONG)
        border.setColorAt(1, T.alpha(T.VIOLET, 0.50))
        p.setPen(QPen(border, 1.1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)


class MainWindow(QWidget):
    onboarding_requested = Signal()
    arrange_overlay_requested = Signal(bool)     # Settings -> shell: put the Flow bar into drag mode
    quit_requested = Signal()

    M = 14          # shadow margin around the panel
    GRIP = 7        # resize band (inside the margin, hugging the panel edge)
    MIN_W, MIN_H = 800, 580       # smallest panel on a roomy screen
    FLOOR_W, FLOOR_H = 640, 460   # smallest panel on a small one (1920x1080 at 200 %): sidebar + a scrolling page

    def __init__(self, controller):
        super().__init__(None)
        self.ctl = controller
        self.setObjectName("main_window")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("VoxFlow")
        self.setMouseTracking(True)
        self.setMinimumSize(self.MIN_W + 2 * self.M, self.MIN_H + 2 * self.M)
        self.resize(1080 + 2 * self.M, 720 + 2 * self.M)
        self._placed = False
        self._modal: Modal | None = None
        self._modal_slot = (None, None)         # (page, its focus slot) when the last dialog was opened

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(self.M, self.M, self.M, self.M)
        self.panel = _Panel(self)
        self.panel.setObjectName("panel")
        self.panel.setMouseTracking(True)
        self.panel.installEventFilter(self)
        self._root.addWidget(self.panel)

        body = QHBoxLayout(self.panel)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ---- sidebar
        side = QWidget()
        side.setFixedWidth(_Panel.SIDEBAR_W)
        side.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 66, 12, 14)
        sl.setSpacing(3)
        self.nav: dict[str, NavButton] = {}
        for i, (key, text, icon, _cls) in enumerate(PAGES):
            b = NavButton(icon, text)
            b.setObjectName(f"nav_{key}")
            b.setToolTip(f"{text}  (Ctrl+{i + 1})")
            b.clicked.connect(lambda _c=False, k=key: self.show_page(k))
            self.nav[key] = b
            sl.addWidget(b)
        sl.addStretch(1)
        self.btn_pause = Button("Pause dictation", "ghost", icon="pause", height=32)
        self.btn_pause.setObjectName("pause_button")
        self.btn_pause.clicked.connect(lambda: self.ctl.set_paused(not self.ctl.paused))
        sl.addWidget(self.btn_pause)
        sl.addSpacing(4)
        self.status = StatusChip()
        self.status.setObjectName("status_chip")
        self.status.activated.connect(self._show_status)
        sl.addWidget(self.status)
        body.addWidget(side)

        # ---- content
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 8, 8, 0)
        bar.setSpacing(2)
        bar.addStretch(1)
        self.btn_min = WindowButton("min")
        self.btn_max = WindowButton("max")
        self.btn_close = WindowButton("close")
        self.btn_min.setObjectName("window_min")
        self.btn_max.setObjectName("window_max")
        self.btn_close.setObjectName("window_close")
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_max.clicked.connect(self.toggle_maximized)
        self.btn_close.clicked.connect(self.hide)
        for b in (self.btn_min, self.btn_max, self.btn_close):
            bar.addWidget(b)
        col.addLayout(bar)
        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        for key, _text, _icon, cls in PAGES:
            page = cls(self.ctl, self)
            page.setObjectName(f"page_{key}")
            self.pages[key] = page
            self.stack.addWidget(page)
        col.addWidget(self.stack, 1)
        body.addLayout(col, 1)

        self.toasts = ToastHost(self.panel)
        self.toasts.setObjectName("toasts")
        self.toasts.left_inset = _Panel.SIDEBAR_W        # toasts live in the content area, never over the nav / status card

        # ---- shortcuts
        for i, key in enumerate(PAGE_KEYS):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda k=key: self.show_page(k))
        QShortcut(QKeySequence.StandardKey.Find, self).activated.connect(self._focus_search)

        # ---- controller wiring
        c = self.ctl
        c.status_changed.connect(self._on_status)
        c.suggestions_changed.connect(self._update_badges)
        c.notice.connect(self._on_notice)
        c.dictionary_suggestion.connect(self._on_suggestion)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save_state)

        self._on_status(c.status())
        self._update_badges()
        state = c.get("ui_state") or {}
        self.current = ""
        self.show_page(state.get("page") if state.get("page") in PAGE_KEYS else "home", remember=False)

    # ------------------------------------------------------------------ navigation
    def show_page(self, key: str, remember: bool = True) -> None:
        if key not in self.pages:
            return
        if self._modal is not None:
            self._modal.close_modal()
        old = self.pages.get(self.current)
        if old is not None and key != self.current:
            old.on_leave()
        self.current = key
        for k, b in self.nav.items():
            b.setChecked(k == key)
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        page.on_enter()
        if remember and old is not None and key != old.objectName().removeprefix("page_") and self.isVisible():
            self._fade_in(page)
        if remember:
            self._save_timer.start()

    def open_settings(self, section: str = "") -> None:
        """Show Settings, scrolled to a section (``"microphone"``, ``"models"``, ...)."""
        self.present("settings")
        self.pages["settings"].show_section(section)

    def _fade_in(self, page: QWidget) -> None:
        """A 120 ms cross-fade instead of a hard cut (skipped with reduced motion). The effect is removed again
        when it ends so the page does not stay rasterised through an offscreen buffer."""
        if T.REDUCED_MOTION:
            return
        fx = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(fx)
        anim = QPropertyAnimation(fx, b"opacity", page)
        anim.setDuration(120)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.finished.connect(lambda: page.setGraphicsEffect(None))
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _focus_search(self) -> None:
        """Ctrl+F: the search box of the page that is open, with its text selected. A page without one (or whose
        box is not shown, as on a short Dictionary) leads to History's. Nothing happens under an open dialog."""
        if self._modal is not None:
            return
        page = self.pages[self.current]
        box = page.search_box()
        if box is None:
            self.show_page("history")
            page = self.pages["history"]
            box = page.search
        if page.content.isAncestorOf(box):               # the box on Style is half-way down the page
            centre = box.mapTo(page.content, box.rect().center())
            page.scroll.ensureVisible(centre.x(), centre.y(), 0, box.height())
        box.setFocus(Qt.FocusReason.ShortcutFocusReason)
        box.selectAll()

    def toast(self, level: str, message: str, action: str = "", on_action=None) -> None:
        self.toasts.show_toast(level, message, action, on_action)

    def open_modal(self, modal: Modal, focus: QWidget | None = None) -> Modal:
        if self._modal is not None:
            self._modal.close_modal()
        self._modal = modal
        page = self.pages[self.current]
        self._modal_slot = (page, page.focus_slot())
        modal.closed.connect(self._modal_closed)
        modal.open(focus or getattr(modal, "btn_cancel", None))
        return modal

    def _modal_closed(self) -> None:
        self._modal = None
        page, slot = self._modal_slot       # the row the dialog was opened from may have been built anew meanwhile
        page.restore_focus(slot)

    # ------------------------------------------------------------------ controller events
    def _on_status(self, status: dict) -> None:
        self.status.set_status(status or {})
        paused = bool(self.ctl.paused)
        self.btn_pause.setText("Resume dictation" if paused else "Pause dictation")
        self.btn_pause.icon_name = "play" if paused else "pause"
        self.btn_pause.setAccessibleName(self.btn_pause.text())
        self.btn_pause.update()

    def _show_status(self) -> None:
        """The status chip was pressed: its whole text, for when the sidebar had to cut it short."""
        chip = self.status
        level = chip.level if chip.level in ("error", "warning") else "info"
        self.toast(level, f"{chip.title}. {chip.detail}" if chip.detail else chip.title)

    def _update_badges(self) -> None:
        try:
            n = len(self.ctl.dictionary.suggestions())
        except Exception:
            n = 0
        self.nav["dictionary"].set_badge(n)

    def _on_notice(self, level: str, message: str) -> None:
        if self.isVisible() and not self.isMinimized():
            self.toast(level, message)

    def _on_suggestion(self, sugg: dict) -> None:
        if self.isVisible() and not self.isMinimized() and self.current != "dictionary":
            self.toast("info", f"New dictionary suggestion: “{sugg.get('term', '')}”.", "Review",
                       lambda: self.show_page("dictionary"))

    # ------------------------------------------------------------------ show / hide / geometry
    def present(self, page: str | None = None) -> None:
        """Show, un-minimise, raise and focus (used by the tray, second launches and onboarding)."""
        if page:
            self.show_page(page)
        if self.isMinimized():
            self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        bring_to_front(self)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._placed:
            self._placed = True
            self._restore_geometry()
            self.windowHandle().screenChanged.connect(self._fit_minimum)
        self.pages[self.current].on_enter()

    def hideEvent(self, e):
        self.pages[self.current].on_leave()
        self._save_state()
        super().hideEvent(e)

    def closeEvent(self, e):
        e.ignore()          # the app lives in the tray; quitting is explicit (tray menu)
        self.hide()

    def _fit_minimum(self, screen) -> None:
        """The minimum size never exceeds the work area, or the bottom of the window (Pause, the status, the end
        of each page) would sit off-screen and could not be brought back by resizing."""
        ag = screen.availableGeometry()
        self.setMinimumSize(max(self.FLOOR_W, min(self.MIN_W, ag.width())) + 2 * self.M,
                            max(self.FLOOR_H, min(self.MIN_H, ag.height())) + 2 * self.M)

    def _restore_geometry(self) -> None:
        state = self.ctl.get("ui_state") or {}
        geo = state.get("window")
        screens = QGuiApplication.screens()
        rect = None
        if isinstance(geo, (list, tuple)) and len(geo) == 4 and all(isinstance(v, (int, float)) for v in geo):
            cand = QRect(int(geo[0]), int(geo[1]), int(geo[2]), int(geo[3]))
            for s in screens:
                vis = s.availableGeometry().intersected(cand)
                if vis.width() >= 200 and vis.height() >= 120:      # still reachable on some monitor
                    self._fit_minimum(s)
                    ag = s.availableGeometry()
                    cand.setWidth(min(cand.width(), ag.width() + 2 * self.M))
                    cand.setHeight(min(cand.height(), ag.height() + 2 * self.M))
                    rect = cand
                    break
        if rect is None:
            from .overlay import active_screen
            screen = active_screen() or QGuiApplication.primaryScreen()
            self._fit_minimum(screen)
            ag = screen.availableGeometry()
            w = min(self.width(), ag.width() + 2 * self.M)
            h = min(self.height(), ag.height() + 2 * self.M)
            rect = QRect(ag.x() + (ag.width() - w) // 2, ag.y() + (ag.height() - h) // 2, w, h)
        self.setGeometry(rect)
        if state.get("maximized"):
            self.showMaximized()

    def _save_state(self) -> None:
        self._save_timer.stop()
        try:
            state = dict(self.ctl.get("ui_state") or {})
            state["page"] = self.current
            state["maximized"] = bool(self.isMaximized())
            if not self.isMaximized() and not self.isMinimized() and self._placed:
                g = self.geometry()
                state["window"] = [g.x(), g.y(), g.width(), g.height()]
            if state != (self.ctl.get("ui_state") or {}):
                self.ctl.set("ui_state", state)
        except Exception:
            pass            # remembering the layout is a nicety; never let it break closing the window

    def toggle_maximized(self) -> None:
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def changeEvent(self, e):
        if e.type() == QEvent.Type.WindowStateChange:
            maxed = self.isMaximized()
            m = 0 if maxed else self.M
            self._root.setContentsMargins(m, m, m, m)
            self.panel.corner_radius = 0 if maxed else T.R_XL
            self.btn_max.set_kind("restore" if maxed else "max")
            self.update()
            if self.isMinimized():                  # nothing is visible: stop page timers (P9); restoring re-enters
                self.pages[self.current].on_leave()
            elif self.isVisible():
                self.pages[self.current].on_enter()
        super().changeEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._placed:
            self._save_timer.start()

    def moveEvent(self, e):
        super().moveEvent(e)
        if self._placed:
            self._save_timer.start()

    # ------------------------------------------------------------------ frameless move / resize
    def _edges_at(self, pos: QPoint) -> Qt.Edge:
        edges = Qt.Edge(0)
        if self.isMaximized():
            return edges
        r = self.panel.geometry()
        g = self.GRIP
        if not r.adjusted(-g, -g, g, g).contains(pos) or r.adjusted(3, 3, -3, -3).contains(pos):
            return edges
        if pos.x() <= r.left() + 3:
            edges |= Qt.Edge.LeftEdge
        if pos.x() >= r.right() - 3:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= r.top() + 3:
            edges |= Qt.Edge.TopEdge
        if pos.y() >= r.bottom() - 3:
            edges |= Qt.Edge.BottomEdge
        return edges

    def _cursor_for(self, edges: Qt.Edge) -> Qt.CursorShape:
        horiz = bool(edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge))
        vert = bool(edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge))
        if horiz and vert:
            diag = bool(edges & Qt.Edge.LeftEdge) == bool(edges & Qt.Edge.TopEdge)
            return Qt.CursorShape.SizeFDiagCursor if diag else Qt.CursorShape.SizeBDiagCursor
        if horiz:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor if vert else Qt.CursorShape.ArrowCursor

    def _press(self, pos: QPoint, button) -> bool:
        if button != Qt.MouseButton.LeftButton or self.windowHandle() is None:
            return False
        edges = self._edges_at(pos)
        if edges:
            self.windowHandle().startSystemResize(edges)
            return True
        r = self.panel.geometry()
        if r.contains(pos) and pos.y() < r.top() + _Panel.TITLE_H:
            self.windowHandle().startSystemMove()      # native move: Aero Snap, multi-monitor DPI changes
            return True
        return False

    def mousePressEvent(self, e):
        self._press(e.position().toPoint(), e.button())

    def mouseMoveEvent(self, e):
        self.setCursor(self._cursor_for(self._edges_at(e.position().toPoint())))

    def leaveEvent(self, e):
        self.unsetCursor()

    def eventFilter(self, obj, ev):
        if obj is self.panel:
            t = ev.type()
            if t == QEvent.Type.MouseButtonPress:
                return self._press(self.panel.mapTo(self, ev.position().toPoint()), ev.button())
            if t == QEvent.Type.MouseMove:
                self.setCursor(self._cursor_for(self._edges_at(self.panel.mapTo(self, ev.position().toPoint()))))
            elif t == QEvent.Type.MouseButtonDblClick and ev.position().y() < _Panel.TITLE_H:
                self.toggle_maximized()
                return True
        return False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(e)

    def paintEvent(self, _):
        if self.isMaximized():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        T.paint_shadow(p, QRectF(self.panel.geometry()), T.R_XL, spread=self.M - 1, strength=1.0, dy=3)
