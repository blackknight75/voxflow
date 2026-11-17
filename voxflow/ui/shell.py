"""UI entry point: ``build_ui(controller)`` creates the Flow bar, tray, main window and (first run) the setup
guide, wires them to the controller's signals and returns the object that keeps them alive.

Nothing here reaches past the controller facade (``voxflow.controller.Controller``).
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from . import theme as T
from .onboarding import Onboarding
from .overlay import OverlayPill
from .tray import Tray
from .widgets import bring_to_front, style_tooltips
from .window import MainWindow

_LOADING_TEXT = {"downloading": "Downloading speech model", "loading": "Loading speech model"}


class Shell(QObject):
    """Owns every top-level UI object. ``overlay``, ``tray``, ``window`` and ``onboarding`` are public for tests."""

    def __init__(self, controller, show_window: bool = False):
        super().__init__()
        self.ctl = c = controller
        self.onboarding: Onboarding | None = None
        app = QApplication.instance()
        app.setFont(T.font(T.F_BODY))
        T.apply_theme(str(c.get("theme")), bool(c.get("reduced_motion")))
        style_tooltips()
        T.bus().changed.connect(style_tooltips)

        self.overlay = OverlayPill(c, show_idle=bool(c.get("show_idle_pill")))
        self.overlay.set_position(c.get("overlay_position"))
        self.overlay.set_paused(bool(c.paused))
        self.overlay.position_changed.connect(self._overlay_moved)
        self.window = MainWindow(c)
        self.window.onboarding_requested.connect(self.open_onboarding)
        self.window.arrange_overlay_requested.connect(self.overlay.set_arranging)
        self.tray = Tray(c)
        self.tray.show_window.connect(lambda page: self.window.present(page or None))
        self.tray.quit_requested.connect(self.quit)
        c.restart_requested.connect(lambda: QApplication.instance().quit())   # the update installs after shutdown
        c.update_available.connect(self._on_update_available)

        c.state_changed.connect(self._on_state)
        c.status_changed.connect(self._on_status)
        c.model_status_changed.connect(self._on_model)
        c.config_changed.connect(self._on_config)
        c.notice.connect(self._on_notice)
        if hasattr(c, "notice_action"):
            c.notice_action.connect(self._on_notice_action)       # a notice that carries an action id (D-15)
        self.tray.action_requested.connect(self.run_action)
        c.activate_requested.connect(lambda: self.window.present())
        app.aboutToQuit.connect(self.close)

        self.tray.show()
        self._on_state(c.state, c.mode, c.message)
        if not c.get("onboarding_done"):
            QTimer.singleShot(0, self.open_onboarding)
        elif show_window:
            self.window.present()

    # ---- controller -> UI
    def _on_state(self, state: str, mode: str, message: str) -> None:
        if state == "idle" and self._show_loading():
            return
        self.overlay.set_state(state, message, mode)

    def _show_loading(self) -> bool:
        """While idle, the Flow bar reports the speech model coming up (download progress on first run)."""
        try:
            st = self.ctl.model_status("asr")
        except Exception:
            return False
        text = _LOADING_TEXT.get(str(st.get("state")))
        if text is None:
            return False
        prog = st.get("progress")
        self.overlay.set_state("loading", text, "dictate", prog if isinstance(prog, (int, float)) else None)
        return True

    def _on_model(self, name: str, _status: dict) -> None:
        if name == "asr" and self.ctl.state == "idle" and not self._show_loading() and self.overlay.state == "loading":
            self.overlay.set_state("idle")

    def _on_status(self, _status: dict) -> None:
        self.overlay.set_paused(bool(self.ctl.paused))

    def _on_config(self, key: str, value) -> None:
        if key == "theme":
            T.apply_theme(str(value))
        elif key == "reduced_motion":
            T.set_reduced_motion(bool(value))
        elif key == "show_idle_pill":
            self.overlay.set_show_idle(bool(value))
        elif key == "overlay_position" and not self.overlay.arranging:
            self.overlay.set_position(value)

    # Next step ids a notice can carry (the ``notice_action`` signal): the controller sends ``(message, action_id,
    # label)`` right after the ``notice`` with that message, and the UI adds a button to that toast.
    def run_action(self, action: str) -> None:
        """Do what a notification button promises. Unknown ids are ignored."""
        action = str(action or "")
        if action.startswith("open_settings:"):
            self.window.open_settings(action.split(":", 1)[1])
        elif action == "update":
            self.window.open_settings("updates")
            self.window.pages["settings"].open_update()
        elif action in ("open_log_folder", "open_data_folder"):
            opener = getattr(self.ctl, action, None)
            if callable(opener):
                opener()
        elif action == "retry_save":
            retry = getattr(self.ctl, "retry_saves", None)
            if callable(retry):
                try:
                    retry()                     # reports its own result through notice / notice_action
                except Exception as exc:
                    self.window.toast("error", f"That did not work: {exc}")

    def _on_notice_action(self, message: str, action: str, label: str) -> None:
        if self.window.isVisible() and not self.window.isMinimized():
            self.window.toasts.attach_action(message, label, lambda a=action: self.run_action(a))
        else:                                   # the notice went out as a tray notification: a click runs the action
            self.tray.set_pending_action(action)

    def _on_notice(self, level: str, message: str) -> None:
        # The window shows its own toast when it is up. When it is hidden (e.g. right after "Paste again", which
        # hides it on purpose) anything the user needs to know goes to a native tray notification instead.
        showing = self.window.isVisible() and not self.window.isMinimized()
        if not showing and level != "success":
            self.tray.notify(level, message)

    def _on_update_available(self, info: dict) -> None:
        """A newer version (once per version, after "Remind me later" only when it is due): a toast with "See
        update" while the window is up, else a tray notification that opens the dialog when clicked."""
        rel = info.get("available") or {}
        if not rel:
            return
        message = f"VoxFlow {rel.get('version')} is available."
        if self.window.isVisible() and not self.window.isMinimized():
            self.window.toast("info", message, "See update", lambda: self.run_action("update"))
        else:
            self.tray.notify("info", message + " Click to see what is new and choose when to install it.",
                             action="update")

    def quit(self) -> None:
        """Quit from the tray: the user's own quit, so an update chosen for "when I quit" may install."""
        prepare = getattr(self.ctl, "prepare_quit", None)
        if callable(prepare):
            prepare()
        QApplication.instance().quit()

    def _overlay_moved(self, pos) -> None:
        try:
            self.ctl.set("overlay_position", pos)
        except ValueError as exc:
            self.window.toast("error", str(exc))

    # ---- onboarding
    def open_onboarding(self) -> Onboarding:
        if self.onboarding is not None:
            bring_to_front(self.onboarding)
            return self.onboarding
        ob = Onboarding(self.ctl)
        ob.finished.connect(self._onboarding_finished)
        ob.settings_requested.connect(self.window.open_settings)
        self.onboarding = ob
        bring_to_front(ob)
        return ob

    def _onboarding_finished(self, completed: bool) -> None:
        ob, self.onboarding = self.onboarding, None
        if ob is not None:
            ob.deleteLater()
        if completed:
            self.window.present("home")

    def close(self) -> None:
        """Hide everything (app shutdown). Safe to call more than once."""
        if self.onboarding is not None:
            self.onboarding.hide()
        self.overlay.set_arranging(False)
        self.overlay.hide()
        self.window.hide()
        self.tray.hide()


def build_ui(controller, *, show_window: bool = False) -> Shell:
    """Create overlay + tray + main window (+ onboarding on first run), wired to ``controller``. Never blocks."""
    return Shell(controller, show_window=show_window)
