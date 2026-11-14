"""``voxflow.ui.shell``: the wiring between controller, Flow bar, tray, window and setup guide, on the in-memory
demo controller: model loading on the Flow bar, notices with a next step, the actions behind them, and the
failure paths that must never take the UI down."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import demo, flush, make_demo  # noqa: F401 - fixtures
from voxflow.ui import theme as T
from voxflow.ui.fakes import FakeController
from voxflow.ui.shell import build_ui


def test_the_flow_bar_reports_the_speech_model_coming_up(demo):
    overlay, c = demo.shell.overlay, demo.c
    c.demo_set_model("asr", "downloading", "42 MB of 600 MB", 0.07)
    assert (overlay.state, overlay.text, overlay.progress) == ("loading", "Downloading speech model", 0.07)
    c.demo_set_state("idle")                                   # e.g. a cancelled recording: loading stays on the bar
    assert overlay.state == "loading"
    c.demo_set_model("asr", "loading", "Warming up", None)
    assert (overlay.state, overlay.text, overlay.progress) == ("loading", "Loading speech model", None)
    c.demo_set_model("asr", "downloading", "?", "soon")        # a progress that is not a number is left out
    assert overlay.progress is None
    c.demo_set_model("cleanup", "loading", "Warming up")       # the cleanup model does not show on the bar
    assert overlay.text == "Downloading speech model"
    c.demo_set_state("listening")
    assert overlay.state == "listening", "a dictation always wins"
    c.demo_set_model("asr", "ready")
    assert overlay.state == "listening"
    c.demo_set_state("idle")
    c.demo_set_model("asr", "loading")
    c.demo_set_model("asr", "ready")
    assert overlay.state == "idle"


def test_a_controller_that_cannot_report_its_models_still_gets_a_flow_bar(demo):
    def broken(_name):
        raise RuntimeError("models not initialised")

    demo.c.model_status = broken
    demo.c.demo_set_state("done", "dictate", "Pasted")
    demo.c.demo_set_state("idle")
    assert demo.shell.overlay.state == "idle"
    demo.shell._on_model("asr", {})
    assert demo.shell.overlay.state == "idle"


def test_config_changes_reach_the_theme_and_the_flow_bar(demo):
    c, overlay = demo.c, demo.shell.overlay
    c.set("theme", "light")
    assert T.RESOLVED == "light"
    c.set("reduced_motion", True)
    assert T.REDUCED_MOTION
    c.set("show_idle_pill", True)
    assert overlay.show_idle is True
    c.set("overlay_position", [640, 400])
    assert overlay._anchor is not None and (overlay._anchor.x(), overlay._anchor.y()) == (640, 400)
    overlay.set_arranging(True)
    c.set("overlay_position", None)
    assert overlay._anchor is not None, "while the user drags the bar a stored position is not applied over it"
    overlay.set_arranging(False)
    c.set_paused(True)
    assert overlay.paused
    c.set("language", "de")                                    # not the shell's business


def test_a_drag_of_the_flow_bar_is_stored_and_a_refusal_is_shown(demo):
    c = demo.c
    demo.shell.overlay.position_changed.emit([700, 500])
    assert c.get("overlay_position") == [700, 500]

    def refuse(key, value):
        raise ValueError("That position cannot be saved.")

    c.set = refuse
    demo.shell.overlay.position_changed.emit([10, 10])
    assert demo.toasts() == ["That position cannot be saved."]


def test_notice_actions_become_a_toast_button_or_a_tray_click(demo):
    c, shell = demo.c, demo.shell
    c.demo_error("silence")
    hint = c.error_info()["hint"]
    assert demo.toasts() == [hint]
    toast = demo.win.toasts._toasts[0]
    assert toast.btn is not None and toast.btn.text() == "Open microphone settings"
    toast.btn.click()
    flush()
    assert demo.win.current == "settings", "the button does what it says"

    demo.win.hide()                                            # now the same problem arrives while the window is away
    told: list = []
    shell.tray.notify = lambda level, message, action="": told.append((level, message))
    c.demo_error("model")
    assert told == [("warning", c.error_info()["hint"])] and shell.tray._pending_action == "open_settings:models"
    c.notice.emit("success", "Saved.")
    assert len(told) == 1, "good news is not worth a system notification"
    demo.win.show_page("home")
    shell.tray.icon.messageClicked.emit()                      # the user clicks the notification
    QApplication.processEvents()
    assert demo.win.isVisible() and demo.win.current == "settings"


def test_actions_behind_notifications(demo):
    c, shell = demo.c, demo.shell
    shell.run_action("retry_save")
    assert c.retried == 1 and demo.toasts() == ["Saved."]
    shell.run_action("no-such-action")
    shell.run_action(None)
    assert c.retried == 1 and demo.win.current == "home"

    def failing():
        raise OSError("disk is full")

    c.retry_saves = failing
    shell.run_action("retry_save")
    assert demo.toasts()[-1] == "That did not work: disk is full"
    c.retry_saves = None                                       # a controller without that ability
    shell.run_action("retry_save")
    assert len(demo.toasts()) == 2
    shell.run_action("open_settings:models")
    assert demo.win.current == "settings"


def test_the_setup_guide_opens_once_and_finishing_it_shows_the_window(make_demo):
    demo = make_demo(show_window=False, onboarding_done=False)
    QApplication.processEvents()
    guide = demo.shell.onboarding
    assert guide is not None and guide.isVisible() and not demo.win.isVisible()
    assert demo.shell.open_onboarding() is guide, "asking again brings the same guide forward"
    guide.go(len(guide.stack.children()))
    guide.btn_next.click()
    assert demo.shell.onboarding is None and demo.c.get("onboarding_done") is True
    assert demo.win.isVisible() and demo.win.current == "home"
    demo.win.hide()
    demo.shell._onboarding_finished(False)                     # a late signal after the guide is gone
    assert not demo.win.isVisible()
    again = demo.shell.open_onboarding()                       # "Run setup again" from Settings
    again.skip()
    assert demo.shell.onboarding is None and not demo.win.isVisible(), "skipping does not force the window open"
    demo.shell.close()
    demo.shell.close()                                         # safe to call twice
    assert not demo.shell.overlay.isVisible()


class _Legacy:
    """A controller from before notices carried a next step: no ``notice_action`` signal."""

    def __init__(self, real: FakeController):
        self._real = real

    def __getattr__(self, name):
        if name == "notice_action":
            raise AttributeError(name)
        return getattr(self._real, name)


def test_a_controller_without_notice_actions_is_accepted(qtbot, qt_messages):
    real = FakeController(empty=True)
    real.config.onboarding_done = True
    real.config.theme = "dark"
    shell = build_ui(_Legacy(real), show_window=True)
    try:
        QApplication.processEvents()
        real.notice.emit("warning", "No speech heard.")
        assert shell.window.toasts.messages() == ["No speech heard."]
        assert shell.window.toasts._toasts[0].btn is None
    finally:
        shell.close()
        real.shutdown()
        shell.overlay._timer.stop()
        shell.overlay._follow.stop()
        for widget in (shell.window, shell.overlay, shell.tray.menu):
            widget.deleteLater()
        shell.tray.icon.deleteLater()
        shell.tray.deleteLater()
        shell.deleteLater()
        flush()
        T.apply_theme("dark", False)
    assert qt_messages == []
