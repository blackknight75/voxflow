"""``voxflow.ui.onboarding``: the first-run guide on the demo controller: every step, the microphone check, the
practice dictation, model progress, and the failure paths (a controller call that raises must never trap the user
in the guide). The 450 ms "show the words anyway" timer is captured and run by hand."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWindow
from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import Signals, flush, grab_bytes
from voxflow.ui import onboarding as ob_mod
from voxflow.ui import theme as T
from voxflow.ui.fakes import FakeController
from voxflow.ui.onboarding import STEPS, Onboarding, _Feature, _ModelLine

LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton


@pytest.fixture
def later(monkeypatch):
    """``QTimer.singleShot`` calls made by the guide, captured instead of scheduled."""
    calls: list = []
    monkeypatch.setattr(ob_mod.QTimer, "singleShot", staticmethod(lambda ms, fn: calls.append((ms, fn))))
    return calls


@pytest.fixture
def make_guide(qapp, qt_messages, later):
    made: list = []

    def build(demo: bool = True, show: bool = True, **config):
        c = FakeController()
        c.is_demo = demo
        for key, value in config.items():
            setattr(c.config, key, value)
        guide = Onboarding(c)
        if show:
            guide.move(260, 190)
            guide.show()
            QApplication.processEvents()
        made.append((guide, c))
        return guide, c

    yield build
    for guide, c in made:
        guide._poll.stop()
        guide.hide()
        c.shutdown()
        guide.deleteLater()
        c.deleteLater()
    flush()
    T.apply_theme("dark", False)
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


def step(guide: Onboarding, name: str) -> None:
    guide.go(STEPS.index(name))
    QApplication.processEvents()


# =========================================================================== navigation
def test_the_model_step_comes_before_the_hotkey_practice():
    assert STEPS.index("models") < STEPS.index("hotkey"), "the practice needs a loaded speech model"


def test_walking_through_every_step(make_guide):
    guide, c = make_guide()
    c.demo_set_model("asr", "loading", "Loading the speech model…")
    seen = Signals(finished=guide.finished)
    assert STEPS[guide.index] == "welcome" and guide.btn_next.text() == "Get started"
    assert not guide.btn_back.isVisible() and guide.btn_skip.isVisible() and guide.dots.accessibleName() == "Step 1 of 5"
    pictures = []
    for name in STEPS[1:]:
        guide.btn_next.click()
        QApplication.processEvents()
        assert STEPS[guide.index] == name and guide.btn_back.isVisible()
        pictures.append(grab_bytes(guide))
        if name == "models":
            assert not guide.btn_next.isEnabled(), "Continue waits for the speech model"
            c.demo_set_model("asr", "ready", "Running on GPU (CUDA)", None, "cuda")
            assert guide.btn_next.isEnabled()
    assert len(set(pictures)) == 4
    assert guide.btn_next.text() == guide.btn_next.accessibleName() == "Start dictating" and not guide.btn_skip.isVisible()
    guide.btn_back.click()
    assert STEPS[guide.index] == "hotkey" and guide.btn_next.text() == "Continue" and guide.btn_skip.isVisible()
    guide.btn_back.click()
    assert STEPS[guide.index] == "mic", "the model is ready: Back passes over the model step"
    guide.go(-5)
    assert guide.index == 0
    guide.go(99)
    assert guide.index == len(STEPS) - 1
    guide.btn_next.click()
    assert seen.finished == [(True,)] and not guide.isVisible() and c.get("onboarding_done") is True
    guide.skip()
    guide.close()
    assert seen.finished == [(True,)], "finishing happens once"


@pytest.mark.parametrize("how", ["skip_button", "close_button", "escape", "window_close"])
def test_every_way_out_marks_setup_as_done(make_guide, qtbot, how):
    guide, c = make_guide()
    seen = Signals(finished=guide.finished)
    step(guide, "mic")
    assert guide._poll.isActive() and c._monitor
    qtbot.keyClick(guide, Qt.Key.Key_A)                        # any other key is not a way out
    assert guide.isVisible()
    {"skip_button": guide.btn_skip.click, "close_button": guide.btn_close.click,
     "escape": lambda: qtbot.keyClick(guide, Qt.Key.Key_Escape), "window_close": guide.close}[how]()
    assert seen.finished == [(False,)] and not guide.isVisible() and c.get("onboarding_done") is True
    assert not guide._poll.isActive() and not c._monitor, "the microphone is released on the way out"


def test_a_controller_that_cannot_save_does_not_trap_the_user(make_guide):
    guide, c = make_guide()
    seen = Signals(finished=guide.finished)

    def refuse(key, value):
        raise OSError("config file is read-only")

    c.set = refuse
    guide.skip()
    assert seen.finished == [(False,)] and not guide.isVisible(), "worst case the guide shows once more next launch"


def test_the_guide_opens_centred_and_can_be_dragged_by_its_top_band(make_guide, monkeypatch):
    guide, _c = make_guide()
    area = guide.screen().availableGeometry()
    assert abs(guide.geometry().center().x() - area.center().x()) <= 1
    assert abs(guide.geometry().center().y() - area.center().y()) <= 1
    moves: list = []
    monkeypatch.setattr(QWindow, "startSystemMove", lambda self: moves.append(1) or True)

    def press(pos, button):
        pos = QPointF(pos)
        guide.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, QPointF(guide.mapToGlobal(pos.toPoint())),
                                          button, button, Qt.KeyboardModifier.NoModifier))

    press(QPoint(300, 30), LEFT)
    assert moves == [1]
    press(QPoint(300, 30), RIGHT)
    press(QPoint(300, 200), LEFT)
    assert moves == [1], "only the left button on the top band drags"
    unshown, _ = make_guide(show=False)
    pos = QPointF(300, 30)
    unshown.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos, LEFT, LEFT,
                                        Qt.KeyboardModifier.NoModifier))
    assert moves == [1], "no native window yet: nothing to drag"


# =========================================================================== microphone step
def test_microphone_check_hears_the_user_after_a_few_loud_polls(make_guide):
    guide, c = make_guide()
    step(guide, "mic")
    guide._poll.stop()                                         # polled by hand below
    assert guide.mic_status.text() == "Listening… say something." and guide.meter.source == c.level
    levels = iter([0.01, 0.2, 0.3, 0.02, 0.5, 0.4, 0.9])
    c.level = lambda: next(levels)
    for expected in (0, 1, 2, 2, 3):
        guide._poll_level()
        assert guide._heard == expected and guide.mic_status.text() == "Listening… say something."
    guide._poll_level()
    assert guide.mic_status.text() == "Loud and clear. Your microphone works."
    guide._poll_level()
    assert guide._heard == 5 and guide.mic_status.text() == "Loud and clear. Your microphone works."

    def dead():
        raise RuntimeError("stream died")

    c.level = dead
    guide._poll_level()
    assert guide._heard == 5, "a failing level reads as silence"
    step(guide, "hotkey")
    assert guide.meter.source is None and not c._monitor and not guide._poll.isActive()


def test_microphone_list_choice_and_refusal(make_guide):
    guide, c = make_guide()
    step(guide, "mic")
    assert [t for _v, t in guide.mic.options] == ["System default", "Microphone (Razer Seiren V2 X)",
                                                 "Headset (WH-1000XM5)"]
    guide._heard = 3
    guide.mic.chosen.emit("Headset (WH-1000XM5)")
    assert c.get("mic_device") == "Headset (WH-1000XM5)" and guide._heard == 0, "a new microphone starts the check again"
    assert guide.mic.value == "Headset (WH-1000XM5)"

    c._mics = c._mics[:2]                                      # the headset is unplugged
    c.refresh_microphones()
    assert guide.mic.options[-1] == ("Headset (WH-1000XM5)", "Headset (WH-1000XM5) (not connected)")
    c.microphones_changed.emit([])
    assert guide.mic.options == [(None, "System default"), ("Headset (WH-1000XM5)", "Headset (WH-1000XM5) (not connected)")]
    c.config.mic_device = None
    c.microphones_changed.emit(None)
    assert guide.mic.options == [(None, "System default")]

    real_set = c.set

    def refuse(key, value):
        raise ValueError("That microphone cannot be opened.")

    c.set = refuse
    guide.mic.chosen.emit("Microphone (Razer Seiren V2 X)")
    assert guide.mic_status.text() == "That microphone cannot be opened." and guide.mic.value is None
    c.set = real_set

    def no_monitor(on):
        raise RuntimeError("no input device")

    c.set_level_monitor = no_monitor                           # must not keep the user from moving on
    step(guide, "welcome")
    step(guide, "mic")
    assert STEPS[guide.index] == "mic"
    guide._poll.stop()


# =========================================================================== hotkey step
def test_practice_dictation_reports_each_state_and_shows_the_words(make_guide, later):
    guide, c = make_guide()
    c.demo_set_state("listening")
    assert guide.try_status.text() == "Waiting for you to hold the hotkey.", "other steps ignore dictation states"
    c.demo_add_transcript("Not on this step.")
    assert later == [] and not guide._practised
    step(guide, "hotkey")
    assert guide.practice.hasFocus()
    for state, message, text in (("listening", "", "Listening… keep holding and speak."),
                                 ("locked", "", "Hands-free: tap the hotkey once more when you are done."),
                                 ("processing", "", "Transcribing…"),
                                 ("error", "No speech heard", "No speech heard"),
                                 ("error", "", "That did not work. Try once more."),
                                 ("idle", "", "Waiting for you to hold the hotkey.")):
        c.demo_set_state(state, "dictate", message)
        assert guide.try_status.text() == text
    c.demo_set_state("done", "dictate", "Pasted")              # not a state the guide comments on
    assert guide.try_status.text() == "Waiting for you to hold the hotkey."

    entry = c.demo_add_transcript("Let's ship the new build on Friday.")
    assert guide._practised and guide.try_status.text().startswith("That is all there is to it.")
    assert [ms for ms, _fn in later] == [450]
    c.demo_set_state("idle")
    assert guide.try_status.text().startswith("That is all there is to it."), "the praise is not overwritten by idle"
    later.pop()[1]()                                           # focus was elsewhere: the words are shown anyway
    assert guide.practice.toPlainText() == entry["text"]
    c.demo_add_transcript(entry["text"])
    later.pop()[1]()
    assert guide.practice.toPlainText() == entry["text"], "words that already landed are not added twice"
    c.transcript_ready.emit({"text": "   "})
    later.pop()[1]()
    assert guide.practice.toPlainText() == entry["text"]


def test_hotkey_can_be_changed_with_qt_keys_in_demo_mode(make_guide, qtbot):
    guide, c = make_guide()
    step(guide, "hotkey")
    rec = guide.recorder
    assert rec.use_qt_keys and not rec.caps.isVisible()
    rec.btn.click()
    assert rec.capturing and not c._capturing, "no global hook in demo mode"
    qtbot.keyPress(rec.btn, Qt.Key.Key_Control)
    qtbot.keyPress(rec.btn, Qt.Key.Key_J)
    qtbot.keyRelease(rec.btn, Qt.Key.Key_J)
    qtbot.keyRelease(rec.btn, Qt.Key.Key_Control)
    assert c.get("hotkey") == "ctrl+j" and guide.caps.combo == "ctrl+j" and rec.combo == "ctrl+j"

    real_set = c.set

    def refuse(key, value):
        raise ValueError("That combination is reserved by Windows.")

    c.set = refuse
    rec.changed.emit("ctrl+k")
    assert rec.combo == "ctrl+j" and rec.msg.text() == "That combination is reserved by Windows."
    c.set = real_set
    rec._msg_timer.stop()
    rec.btn.click()
    step(guide, "done")
    assert not rec.capturing, "leaving the step ends a recording in progress"
    c.set("language", "de")                                    # unrelated settings do not touch the guide
    assert guide.caps.combo == "ctrl+j"


def test_hotkey_capture_goes_through_the_global_hook_outside_demo_mode(make_guide):
    guide, c = make_guide(demo=False)
    step(guide, "hotkey")
    rec = guide.recorder
    assert not rec.use_qt_keys
    c.hotkey_captured.emit("ctrl+k")
    assert c.get("hotkey") == "ctrl+win", "a capture nobody asked for is ignored"
    rec.btn.click()
    assert c._capturing and rec.capturing
    c.hotkey_captured.emit("ctrl+k")
    assert c.get("hotkey") == "ctrl+k" and guide.caps.combo == "ctrl+k" and not rec.capturing
    rec.btn.click()
    rec.btn.click()                                            # Cancel
    assert not c._capturing and c.get("hotkey") == "ctrl+k"
    rec.btn.click()
    c.hotkey_captured.emit("")                                 # the hook reports "nothing captured"
    assert not rec.capturing and c.get("hotkey") == "ctrl+k"


# =========================================================================== models step
def test_a_model_that_is_already_loaded_costs_no_extra_screen(make_guide):
    guide, c = make_guide()                                    # the demo models are ready (cached and loaded)
    step(guide, "mic")
    guide.btn_next.click()
    assert STEPS[guide.index] == "hotkey", "straight from the microphone check to the practice"


def test_model_progress_and_the_note_below_it(make_guide):
    guide, c = make_guide()
    step(guide, "models")
    asr, cleanup = guide.model_lines["asr"], guide.model_lines["cleanup"]
    assert asr.state.text() == cleanup.state.text() == "Ready" and asr.bar.value == 1.0
    assert guide.models_note.text() == "Everything is downloaded and ready."
    assert guide.btn_next.isEnabled() and not guide.btn_retry.isVisibleTo(guide)
    c.demo_set_model("asr", "downloading", "42 MB of 600 MB", 0.07)
    assert asr.state.text() == "Downloading 7%" and asr.bar.value == 0.07 and asr.detail.text() == "42 MB of 600 MB"
    assert guide.models_note.text().startswith("The hotkey practice opens as soon as")
    assert not guide.btn_next.isEnabled() and guide.btn_next.toolTip() == "Waiting for speech recognition…"
    c.demo_set_model("asr", "downloading")
    assert asr.state.text() == "Downloading" and asr.bar.value is None and not asr.detail.isVisibleTo(asr)
    c.demo_set_model("asr", "loading")
    assert asr.state.text() == "Warming up" and asr.bar.value is None
    c.demo_set_model("cleanup", "degraded", "Running on CPU")
    assert cleanup.state.text() == "Basic mode" and cleanup.bar.value == 1.0
    c.demo_set_model("asr", "failed", "Out of memory")
    assert asr.state.text() == "Failed" and asr.bar.value == 0.0
    assert guide.models_note.text().startswith("Speech recognition could not be loaded")
    assert guide.btn_next.text() == "Skip practice" and guide.btn_next.isEnabled() and guide.btn_retry.isVisibleTo(guide)
    c.demo_set_model("asr", "ready")
    assert guide.models_note.text() == "Everything is downloaded and ready."
    assert guide.btn_next.text() == "Continue" and not guide.btn_retry.isVisibleTo(guide)
    c.demo_set_model("cleanup", "loading")
    assert guide.models_note.text().endswith("Smart cleanup keeps loading in the background.")
    c.demo_set_model("cleanup", "failed", "No GPU memory")
    assert "Smart cleanup could not be loaded" in guide.models_note.text() and guide.btn_retry.isVisibleTo(guide)
    assert guide.btn_next.isEnabled(), "a missing cleanup model never blocks the practice"
    guide.btn_retry.click()                                    # the demo reloads only the failed model
    assert cleanup.state.text() == "Ready" and guide.models_note.text() == "Everything is downloaded and ready."
    c.model_status_changed.emit("no-such-model", {"state": "ready"})
    guide._on_model("asr", None)
    assert asr.state.text() == "Waiting"
    line = _ModelLine("Extra")
    line.set_status({"state": "verifying"})
    assert line.state.text() == "Verifying", "an unknown state is shown as it is"

    def broken(_name):
        raise RuntimeError("models not initialised")

    c.model_status = broken
    step(guide, "welcome")
    step(guide, "models")
    assert asr.state.text() == cleanup.state.text() == "Waiting"
    assert guide.models_note.text().startswith("The hotkey practice opens as soon as")
    step(guide, "done")
    assert guide.btn_next.isEnabled(), "leaving the model step lifts the wait"


def test_a_slow_load_is_explained_and_a_failing_retry_does_not_trap_the_user(make_guide):
    guide, c = make_guide()
    c.demo_set_model("asr", "loading", "Loading the speech model…")
    step(guide, "models")
    assert guide._slow_timer.isActive() and guide._slow_timer.interval() == ob_mod.SLOW_LOAD_MS
    guide._slow_timer.timeout.emit()
    assert guide.models_note.text().startswith("This is taking a while.") and guide.models_note._role == "AMBER"
    step(guide, "welcome")
    assert not guide._slow_timer.isActive(), "only the model step waits"
    guide._on_slow()                                           # a late timeout elsewhere changes nothing
    step(guide, "models")
    assert guide.models_note.text().startswith("The hotkey practice opens as soon as"), "a new visit starts afresh"

    c.demo_set_model("asr", "failed", "No internet.")
    seen: list = []

    def refuse(which=None):
        seen.append(which)
        raise RuntimeError("busy")

    c.reload_models = refuse
    guide.btn_retry.click()
    assert seen == [["asr"]] and guide.models_note.text() == "That did not work: busy"
    assert guide.btn_next.isEnabled() and guide.btn_next.text() == "Skip practice"
    c.reload_models = lambda which=None: c.demo_set_model("asr", "downloading", "", 0.0)
    guide.btn_retry.click()
    assert guide._slow_timer.isActive() and not guide.btn_next.isEnabled(), "retrying waits again"
    guide.skip()
    assert not guide._slow_timer.isActive()


# =========================================================================== last step
def test_the_last_step_lists_only_the_hotkeys_that_are_set(make_guide):
    guide, c = make_guide()
    step(guide, "done")

    def tips():
        flush()
        return [w.findChildren(ob_mod.Label)[0].text() for w in guide.stack.currentWidget().findChildren(_Feature)]

    assert tips() == ["Dictate", "Command mode", "Paste it again", "Teach it your words"]
    step(guide, "models")
    c.set("command_hotkey", "")
    c.set("paste_last_hotkey", "")
    guide._tips_box.addSpacing(6)                              # anything that is not a tip is dropped with the old ones
    step(guide, "done")
    assert guide._tips_box.count() == 2
    assert tips() == ["Dictate", "Teach it your words"], "the old tips are replaced, not added to"
    assert guide.grab()
