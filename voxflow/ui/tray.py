"""System tray: state-aware icon, custom popup menu with status, quick toggles and the microphone picker."""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QSystemTrayIcon

from . import theme as T
from .widgets import PopupMenu, paint_logo

D = T.DARK
# what the icon looks like: (badge shape, badge colour, tile colours); None = plain logo
_LOOKS = {
    "ready": (None, None, (D.CYAN, D.VIOLET)),
    "listening": ("dot", D.RED, (D.CYAN, D.VIOLET)),
    "command": ("dot", D.RED, (D.MAGENTA, D.AMBER)),
    "processing": ("dot", D.CYAN, (D.CYAN, D.VIOLET)),
    "loading": ("dot", D.AMBER, (D.CYAN, D.VIOLET)),
    "warning": ("alert", D.AMBER, (D.CYAN, D.VIOLET)),
    "error": ("alert", D.RED, (D.CYAN, D.VIOLET)),
    "paused": ("pause", QColor(225, 230, 245), (QColor(120, 128, 150), QColor(84, 90, 112))),
}
ICON_STATES = tuple(_LOOKS)
_cache: dict[str, QIcon] = {}


def make_icon(look: str = "ready") -> QIcon:
    """Tray icon for a look from ICON_STATES (cached; rendered at several sizes for crisp scaling)."""
    look = look if look in _LOOKS else "ready"
    if look in _cache:
        return _cache[look]
    shape, colour, (c0, c1) = _LOOKS[look]
    icon = QIcon()
    for size in (16, 20, 24, 32, 48, 64):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_logo(p, QRectF(1, 1, size - 2, size - 2), c0, c1)
        if shape is not None:
            d = size * 0.50
            b = QRectF(size - d, size - d, d, d)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(8, 10, 20))
            if shape == "alert":
                tri = QPainterPath(QPointF(b.center().x(), b.top() - d * 0.06))
                tri.lineTo(b.right() + d * 0.04, b.bottom())
                tri.lineTo(b.left() - d * 0.04, b.bottom())
                tri.closeSubpath()
                p.drawPath(tri)
                inner = QPainterPath(QPointF(b.center().x(), b.top() + d * 0.20))
                inner.lineTo(b.right() - d * 0.14, b.bottom() - d * 0.12)
                inner.lineTo(b.left() + d * 0.14, b.bottom() - d * 0.12)
                inner.closeSubpath()
                p.setBrush(colour)
                p.drawPath(inner)
            else:
                p.drawEllipse(b)
                if shape == "pause":
                    p.setBrush(colour)
                    bw = d * 0.16
                    for dx in (-0.17, 0.17):
                        p.drawRoundedRect(QRectF(b.center().x() + d * dx - bw / 2, b.top() + d * 0.27, bw, d * 0.46),
                                          bw / 2, bw / 2)
                else:
                    p.setBrush(colour)
                    p.drawEllipse(b.adjusted(d * 0.17, d * 0.17, -d * 0.17, -d * 0.17))
        p.end()
        icon.addPixmap(pm)
    _cache[look] = icon
    return icon


def _short(text: str, n: int = 20) -> str:
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


class Tray(QObject):
    """Owns the tray icon. Everything it shows comes from the controller; clicks become signals or controller calls."""

    show_window = Signal(str)            # page key, or "" for "wherever it was"
    action_requested = Signal(str)       # a notification with a next step was clicked (action id, see Shell.ACTIONS)
    quit_requested = Signal()

    LEVEL_COLOURS = {"ok": "GREEN", "demo": "GREEN", "loading": "AMBER", "warning": "AMBER", "error": "RED",
                     "paused": "TEXT_DIM"}

    def __init__(self, controller):
        super().__init__()
        self.ctl = controller
        self.look = "loading"
        self.icon = QSystemTrayIcon(make_icon(self.look))
        self.icon.setToolTip("VoxFlow")
        self.icon.activated.connect(self._activated)
        self._pending_action = ""
        self.icon.messageClicked.connect(self._message_clicked)
        self.menu = PopupMenu(296)
        self.menu.setObjectName("tray_menu")
        self._page = "main"
        c = controller
        c.state_changed.connect(self.refresh)
        c.status_changed.connect(self.refresh)
        c.config_changed.connect(self._on_config)
        c.microphones_changed.connect(self._refill_if_open)
        self.refresh()

    def show(self) -> None:
        self.icon.show()

    def hide(self) -> None:
        self.menu.close()
        self.icon.hide()

    def set_pending_action(self, action: str) -> None:
        """The notification just shown gets a next step: clicking it runs ``action``."""
        self._pending_action = action

    def _message_clicked(self) -> None:
        action, self._pending_action = self._pending_action, ""
        if action:
            self.action_requested.emit(action)

    def notify(self, level: str, message: str, action: str = "") -> None:
        """Native notification for problems that happen while the main window is not showing. ``action`` is the id
        of the next step; clicking the notification asks the shell to run it (the OS shows no real button)."""
        self._pending_action = action
        if not self.icon.isVisible() or not QSystemTrayIcon.supportsMessages():
            return
        kind = {"error": QSystemTrayIcon.MessageIcon.Critical, "warning": QSystemTrayIcon.MessageIcon.Warning}.get(
            level, QSystemTrayIcon.MessageIcon.Information)
        self.icon.showMessage("VoxFlow", message, kind, 6000)

    # ---- state -> icon
    def current_look(self) -> str:
        c = self.ctl
        if c.state in ("listening", "locked"):
            return "command" if c.mode == "command" else "listening"
        if c.state == "processing":
            return "processing"
        level = (c.status() or {}).get("level", "ok")
        if c.paused or level == "paused":
            return "paused"
        return {"loading": "loading", "warning": "warning", "error": "error"}.get(level, "ready")

    def refresh(self, *_a) -> None:
        look = self.current_look()
        if look != self.look:
            self.look = look
            self.icon.setIcon(make_icon(look))
        st = self.ctl.status() or {}
        tip = f"VoxFlow: {st.get('title', '')}"
        if st.get("detail"):
            tip += f"\n{st['detail']}"
        self.icon.setToolTip(tip)
        self._refill_if_open()

    def _on_config(self, key: str, _value) -> None:
        if key in ("cleanup_enabled", "sounds_enabled", "mic_device", "hotkey", "paste_last_hotkey"):
            self._refill_if_open()

    def _refill_if_open(self, *_a) -> None:
        if self.menu.isVisible():
            self.menu.set_items(self.build_items())
            self.menu.update()

    def _activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_window.emit("")
        elif reason in (QSystemTrayIcon.ActivationReason.Context, QSystemTrayIcon.ActivationReason.MiddleClick):
            self.show_menu(QCursor.pos())

    # ---- menu
    def show_menu(self, pos, page: str = "main") -> PopupMenu:
        self._page = page
        if page == "mics":
            try:
                self.ctl.refresh_microphones()      # hot-plugged devices show up without a restart
            except Exception:
                pass
        self.menu.set_items(self.build_items())
        self.menu.popup_at(pos)
        return self.menu

    def _goto(self, page: str) -> None:
        self._page = page
        if page == "mics":
            try:
                self.ctl.refresh_microphones()
            except Exception:
                pass
        self.menu.set_items(self.build_items())
        self.menu.update()

    def _set(self, key: str, value) -> None:
        try:
            self.ctl.set(key, value)
        except ValueError as exc:
            self.notify("error", str(exc))
        self._refill_if_open()

    PAUSE_FOR_S = 3600

    def pause_for_hour(self) -> None:
        """The controller owns the deadline, so resuming by any route (sidebar, Flow bar, this menu) cancels it."""
        self.ctl.set_paused(True, for_s=self.PAUSE_FOR_S)
        self._refill_if_open()

    def _toggle_pause(self, on: bool) -> None:
        self.ctl.set_paused(bool(on))

    def _mic_name(self) -> str:
        cur = self.ctl.get("mic_device")
        for m in self.ctl.microphones():
            if m.get("id") == cur:
                return str(m.get("name", ""))
        return "System default" if cur is None else str(cur)

    def build_items(self) -> list[dict]:
        c = self.ctl
        if self._page == "mics":
            cur = c.get("mic_device")
            items = [{"kind": "action", "text": "Back", "icon": "chevron-left", "keep_open": True,
                      "on_click": lambda: self._goto("main")},
                     {"kind": "separator"}]
            mics = c.microphones()
            for m in mics:
                items.append({"kind": "choice", "text": _short(str(m.get("name", "Microphone")), 40),
                              "tip": str(m.get("name", "")), "checked": m.get("id") == cur,
                              "on_click": (lambda d=m.get("id"): self._set("mic_device", d))})
            if not mics:
                items.append({"kind": "info", "text": "No microphones found"})
            return items
        st = c.status() or {}
        dot = T.token(self.LEVEL_COLOURS.get(st.get("level", "ok"), "GREEN"))
        has_last = c.last_transcript is not None
        paste_key = c.get("paste_last_hotkey")
        return [
            {"kind": "header", "text": "VoxFlow", "sub": f"Local dictation  ·  v{c.version}"},
            {"kind": "status", "text": st.get("title", ""), "sub": st.get("detail", ""), "dot": dot},
            {"kind": "separator"},
            {"kind": "toggle", "text": "Pause dictation", "icon": "pause", "checked": bool(c.paused),
             "on_click": self._toggle_pause},
            {"kind": "action", "text": "Pause for 1 hour", "icon": "pause", "enabled": not c.paused,
             "on_click": self.pause_for_hour},
            {"kind": "toggle", "text": "Smart cleanup", "icon": "sparkle", "checked": bool(c.get("cleanup_enabled")),
             "on_click": lambda v: self._set("cleanup_enabled", bool(v))},
            {"kind": "toggle", "text": "Sound cues", "icon": "sound", "checked": bool(c.get("sounds_enabled")),
             "on_click": lambda v: self._set("sounds_enabled", bool(v))},
            {"kind": "action", "text": "Microphone", "icon": "mic", "sub": _short(self._mic_name()) + "  ›",
             "keep_open": True, "on_click": lambda: self._goto("mics")},
            {"kind": "action", "text": "Paste last transcript", "icon": "paste", "enabled": has_last,
             "sub": c.pretty_hotkey(paste_key) if paste_key else "", "on_click": lambda: c.paste_last()},
            {"kind": "separator"},
            {"kind": "action", "text": "Open VoxFlow", "icon": "window", "on_click": lambda: self.show_window.emit("home")},
            {"kind": "action", "text": "Settings", "icon": "settings", "on_click": lambda: self.show_window.emit("settings")},
            {"kind": "separator"},
            {"kind": "action", "text": "Quit VoxFlow", "icon": "power", "on_click": lambda: self.quit_requested.emit()},
        ]
