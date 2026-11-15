"""First-run setup: welcome -> microphone check -> model download -> hotkey practice -> done.

The models start loading the moment the app starts, so they get the welcome and microphone steps as a head start.
The hotkey practice needs the speech model, so it only opens once that model is ready: the model step waits for it
(with progress), and is passed over when the model is already loaded (a cached model costs no extra screen). If the
speech model fails, the model step offers Try again or skipping the practice. Closing the guide never stops a
download: the models keep loading in the background (Settings > On-device models shows them).

Skippable at every step (Skip, Esc or the close button) and re-openable from Settings > Support.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from . import theme as T
from .components import WindowButton
from .icons import paint_icon
from .widgets import (Button, ChoiceButton, HotkeyRecorder, KeyCaps, Label, LevelMeter, ProgressBar, TextArea,
                      paint_logo, painter)

STEPS = ("welcome", "mic", "models", "hotkey", "done")
SLOW_LOAD_MS = 20_000          # after this long on the model step without a ready speech model, say why
BUSY_STATES = ("pending", "downloading", "loading")


class _Feature(QWidget):
    def __init__(self, icon: str, title: str, sub: str, parent=None):
        super().__init__(parent)
        self.icon_name = icon
        lay = QVBoxLayout(self)
        lay.setContentsMargins(52, 2, 0, 2)
        lay.setSpacing(1)
        lay.addWidget(Label(title, T.F_BODY_LG, "TEXT", T.W_SEMI))
        lay.addWidget(Label(sub, T.F_BODY, "TEXT_DIM"))
        T.bus().changed.connect(self.update)

    def paintEvent(self, _):
        p = painter(self)
        tile = QRectF(0.5, max(0.5, self.height() / 2 - 19), 38, 38)
        p.fillPath(T.rounded(tile, 11), T.neon_gradient(tile.left(), tile.right(), 0.16))
        p.setPen(QPen(T.alpha(T.CYAN, 0.30), 1))
        p.drawRoundedRect(tile, 11, 11)
        paint_icon(p, self.icon_name, tile.adjusted(9.5, 9.5, -9.5, -9.5), T.CYAN, 1.8)


class _Dots(QWidget):
    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.n, self.index = n, 0
        self.setFixedSize(n * 16 + 10, 12)
        T.bus().changed.connect(self.update)

    def set_index(self, i: int) -> None:
        self.index = i
        self.setAccessibleName(f"Step {i + 1} of {self.n}")
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        p.setPen(Qt.PenStyle.NoPen)
        x = 2.0
        for i in range(self.n):
            w = 20.0 if i == self.index else 6.0
            r = QRectF(x, 3, w, 6)
            if i == self.index:
                p.setBrush(T.neon_gradient(r.left(), r.right()))
            else:
                p.setBrush(T.alpha(T.CYAN, 0.55) if i < self.index else T.CONTROL_HI)
            p.drawRoundedRect(r, 3, 3)
            x += w + 6


class _ModelLine(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(Label(title, T.F_BODY_LG, "TEXT", T.W_SEMI, wrap=False))
        top.addStretch(1)
        self.state = Label("", T.F_SMALL, "TEXT_DIM", T.W_SEMI, wrap=False)
        top.addWidget(self.state)
        lay.addLayout(top)
        self.bar = ProgressBar()
        lay.addWidget(self.bar)
        self.detail = Label("", T.F_SMALL, "TEXT_DIM")
        lay.addWidget(self.detail)

    def set_status(self, st: dict) -> None:
        state = str(st.get("state", "pending"))
        prog = st.get("progress")
        labels = {"pending": ("Waiting", "TEXT_DIM"), "downloading": ("Downloading", "AMBER"),
                  "loading": ("Warming up", "AMBER"), "ready": ("Ready", "GREEN"),
                  "degraded": ("Basic mode", "AMBER"), "failed": ("Failed", "RED")}
        text, role = labels.get(state, (state.title(), "TEXT_DIM"))
        if state == "downloading" and isinstance(prog, (int, float)):
            text = f"Downloading {int(prog * 100)}%"
        self.state.set_role(role)
        self.state.setText(text)
        self.detail.set_role("RED" if state == "failed" else "TEXT_DIM")
        self.detail.setText(str(st.get("detail", "")))
        self.detail.setVisible(bool(st.get("detail")))
        if state in ("ready", "degraded"):
            self.bar.set_value(1.0)
        elif state == "failed":
            self.bar.set_value(0.0)
        elif state == "downloading" and isinstance(prog, (int, float)):
            self.bar.set_value(prog)
        else:
            self.bar.set_value(None)


class Onboarding(QWidget):
    finished = Signal(bool)        # True = went through to the end, False = skipped

    M = 14
    W, H = 640, 490

    def __init__(self, controller):
        super().__init__(None)
        self.ctl = controller
        self.setObjectName("onboarding")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Welcome to VoxFlow")
        self.setFixedSize(self.W + 2 * self.M, self.H + 2 * self.M)
        self._done = False
        self._heard = 0
        self._practised = False
        self._slow = False                 # the speech model has kept the user waiting on the model step

        root = QVBoxLayout(self)
        root.setContentsMargins(self.M + 36, self.M + 10, self.M + 36, self.M + 22)
        root.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, -26, 0)
        bar.addStretch(1)
        self.btn_close = WindowButton("close")
        self.btn_close.setObjectName("onboarding_close")
        self.btn_close.setToolTip("Skip setup")
        self.btn_close.setAccessibleName("Skip setup")
        self.btn_close.clicked.connect(self.skip)
        bar.addWidget(self.btn_close)
        root.addLayout(bar)

        self.stack = QStackedWidget()
        for build in (self._welcome, self._mic, self._models, self._hotkey, self._finish_page):
            self.stack.addWidget(build())
        root.addWidget(self.stack, 1)

        foot = QHBoxLayout()
        foot.setSpacing(T.S2)
        self.btn_skip = Button("Skip setup", "ghost")
        self.btn_skip.setObjectName("onboarding_skip")
        self.btn_skip.clicked.connect(self.skip)
        self.dots = _Dots(len(STEPS))
        self.btn_back = Button("Back", "secondary")
        self.btn_back.setObjectName("onboarding_back")
        self.btn_back.clicked.connect(self._back)
        self.btn_next = Button("Get started", "primary")
        self.btn_next.setObjectName("onboarding_next")
        self.btn_next.setMinimumWidth(128)
        self.btn_next.clicked.connect(self._next)
        for b in (self.btn_skip, self.btn_back):          # hidden buttons keep their slot: the dots stay centred
            policy = b.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            b.setSizePolicy(policy)
        foot.addWidget(self.btn_skip)
        foot.addStretch(1)
        foot.addWidget(self.dots)
        foot.addStretch(1)
        foot.addWidget(self.btn_back)
        foot.addWidget(self.btn_next)
        root.addLayout(foot)

        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._poll_level)
        self._slow_timer = QTimer(self)
        self._slow_timer.setSingleShot(True)
        self._slow_timer.setInterval(SLOW_LOAD_MS)
        self._slow_timer.timeout.connect(self._on_slow)
        c = self.ctl
        c.state_changed.connect(self._on_state)
        c.transcript_ready.connect(self._on_transcript)
        c.model_status_changed.connect(self._on_model)
        c.microphones_changed.connect(self._fill_mics)
        c.hotkey_captured.connect(self._on_captured)
        c.config_changed.connect(self._on_config)
        T.bus().changed.connect(self.update)
        self.index = 0
        self.go(0)

    # ------------------------------------------------------------------ pages
    def _page(self, title: str, sub: str) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 22, 0, 12)
        lay.setSpacing(T.S3)
        lay.addWidget(Label(title, T.F_DISPLAY - 3, "TEXT", T.W_BOLD, -0.3))
        lay.addWidget(Label(sub, T.F_BODY_LG, "TEXT_DIM"))
        lay.addSpacing(T.S2)
        return w, lay

    def _welcome(self) -> QWidget:
        w, lay = self._page("Welcome to VoxFlow", "Talk instead of typing, in any app. Setup takes about a minute.")
        lay.setContentsMargins(0, 70, 0, 12)
        lay.addSpacing(T.S2)
        for icon, title, sub in (
                ("lock", "Private by design", "Your voice is transcribed on this PC. Nothing is uploaded, ever."),
                ("keyboard", "Works everywhere", "Hold a hotkey, speak, let go. The text lands where your cursor is."),
                ("sparkle", "Cleans up as you talk", "Punctuation, filler words and “no wait, I meant…” are fixed for you.")):
            lay.addWidget(_Feature(icon, title, sub))
            lay.addSpacing(4)
        lay.addStretch(1)
        return w

    def _mic(self) -> QWidget:
        w, lay = self._page("Check your microphone", "Pick the microphone you want to use and say a few words.")
        self.mic = ChoiceButton(name="Microphone")
        self.mic.setObjectName("onboarding_mic")
        self.mic.chosen.connect(self._pick_mic)
        lay.addWidget(self.mic)
        lay.addSpacing(T.S3)
        self.meter = LevelMeter(None, height=16)
        self.meter.setObjectName("onboarding_meter")
        lay.addWidget(self.meter)
        self.mic_status = Label("Listening… say something.", T.F_BODY_LG, "TEXT_DIM")
        self.mic_status.setObjectName("onboarding_mic_status")
        lay.addWidget(self.mic_status)
        self.mic_help = Label("No movement? Choose another microphone above, and check that microphone access is on "
                              "in Windows Settings > Privacy & security > Microphone.", T.F_SMALL, "TEXT_DIM")
        lay.addWidget(self.mic_help)
        lay.addStretch(1)
        self._fill_mics(self.ctl.microphones())
        return w

    def _hotkey(self) -> QWidget:
        w, lay = self._page("Try your hotkey", "Click into the box, hold the keys, say a sentence, then let go.")
        row = QHBoxLayout()
        row.setSpacing(T.S3)
        self.caps = KeyCaps(self.ctl.get("hotkey"))
        row.addWidget(Label("Hold", T.F_BODY_LG, "TEXT", wrap=False))
        row.addWidget(self.caps)
        row.addStretch(1)
        self.recorder = HotkeyRecorder(self.ctl.get("hotkey"), name="Dictation hotkey",
                                       validator=lambda combo: self.ctl.validate_hotkey("hotkey", combo) or None)
        self.recorder.setObjectName("onboarding_hotkey_recorder")
        self.recorder.caps.hide()
        self.recorder.use_qt_keys = bool(self.ctl.is_demo)
        if not self.recorder.use_qt_keys:               # demo mode has no global hook: Qt key events only
            self.recorder.capture_requested.connect(self.ctl.begin_hotkey_capture)
            self.recorder.capture_cancelled.connect(self.ctl.cancel_hotkey_capture)
        self.recorder.changed.connect(self._hotkey_changed)
        row.addWidget(self.recorder)
        lay.addLayout(row)
        self.practice = TextArea("", "Your words will appear here…", name="Practice area")
        self.practice.setObjectName("onboarding_practice")
        self.practice.setFixedHeight(128)
        lay.addWidget(self.practice)
        self.try_status = Label("Waiting for you to hold the hotkey.", T.F_BODY_LG, "TEXT_DIM")
        self.try_status.setObjectName("onboarding_try_status")
        lay.addWidget(self.try_status)
        lay.addWidget(Label("Tip: double-tap the hotkey to keep listening hands-free, and tap once more to finish. "
                            "Press Esc to cancel a recording.", T.F_SMALL, "TEXT_DIM"))
        lay.addStretch(1)
        return w

    def _models(self) -> QWidget:
        w, lay = self._page("Getting the models ready",
                            "VoxFlow downloads its speech and cleanup models once (about 5 GB). After that it works "
                            "completely offline. You can try your hotkey as soon as speech recognition is ready.")
        self.model_lines = {"asr": _ModelLine("Speech recognition"), "cleanup": _ModelLine("Smart cleanup")}
        for name, line in self.model_lines.items():
            line.setObjectName(f"onboarding_model_{name}")
            lay.addWidget(line)
            lay.addSpacing(T.S2)
        self.models_note = Label("", T.F_SMALL, "TEXT_DIM")
        self.models_note.setObjectName("onboarding_models_note")
        lay.addWidget(self.models_note)
        self.btn_retry = Button("Try again", "secondary", height=30)
        self.btn_retry.setObjectName("onboarding_model_retry")
        self.btn_retry.clicked.connect(self._retry)
        self.btn_retry.hide()
        lay.addWidget(self.btn_retry, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)
        return w

    def _finish_page(self) -> QWidget:
        w, lay = self._page("You are all set", "VoxFlow now lives in the system tray, next to the clock.")
        self.tips: list[_Feature] = []
        lay.addSpacing(T.S1)
        self._tips_box = QVBoxLayout()
        self._tips_box.setSpacing(10)
        lay.addLayout(self._tips_box)
        lay.addStretch(1)
        return w

    def _fill_tips(self) -> None:
        while self._tips_box.count():
            item = self._tips_box.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        pretty = self.ctl.pretty_hotkey
        tips = [("mic", "Dictate", f"Hold {pretty(self.ctl.get('hotkey'))} and speak. Double-tap to go hands-free.")]
        if self.ctl.get("command_hotkey"):
            tips.append(("sparkle", "Command mode", f"Select text, hold {pretty(self.ctl.get('command_hotkey'))} and say "
                                                    "“make this friendlier” or “turn into bullets”."))
        if self.ctl.get("paste_last_hotkey"):
            tips.append(("paste", "Paste it again", f"{pretty(self.ctl.get('paste_last_hotkey'))} pastes your last "
                                                    "dictation wherever you are."))
        tips.append(("book", "Teach it your words", "Add names and jargon in Dictionary; build shortcuts in Snippets."))
        for icon, title, sub in tips:
            self._tips_box.addWidget(_Feature(icon, title, sub))

    # ------------------------------------------------------------------ navigation
    def go(self, index: int) -> None:
        index = max(0, min(len(STEPS) - 1, index))
        self._leave_step()
        self._slow_timer.stop()
        self.index = index
        self.stack.setCurrentIndex(index)
        self.dots.set_index(index)
        self.btn_back.setVisible(index > 0)
        self.btn_skip.setVisible(index < len(STEPS) - 1)
        self.btn_next.setText({0: "Get started", len(STEPS) - 1: "Start dictating"}.get(index, "Continue"))
        self.btn_next.setAccessibleName(self.btn_next.text())
        self.btn_next.setEnabled(True)
        self.btn_next.setToolTip("")
        step = STEPS[index]
        if step == "mic":
            self._heard = 0
            self.mic_status.set_role("TEXT_DIM")
            self.mic_status.setText("Listening… say something.")
            self.meter.set_source(self.ctl.level)
            self._monitor(True)
            self._poll.start()
        elif step == "hotkey":
            self.practice.setFocus(Qt.FocusReason.OtherFocusReason)
            if not self._practised and self._asr_state() != "ready":     # opened directly while it is not ready
                self.try_status.set_role("AMBER")
                self.try_status.setText("The speech model is not ready yet, so the hotkey has nothing to type with.")
        elif step == "models":
            self._slow = False
            for name in self.model_lines:      # starts the slow-load timer while the speech model is busy
                self._on_model(name, self._model_status(name))
        elif step == "done":
            self._fill_tips()
        self.btn_next.setFocus(Qt.FocusReason.OtherFocusReason) if step != "hotkey" else None

    def _passed_over(self, step: str) -> bool:
        """Steps that Continue and Back step over: the model step once the speech model is ready (nothing to wait
        for), and the hotkey practice while it is not (nothing would be typed)."""
        if step == "models":
            return self._asr_state() == "ready"
        if step == "hotkey":
            return self._asr_state() != "ready"
        return False

    def _step(self, direction: int) -> None:
        i = self.index + direction
        while 0 < i < len(STEPS) - 1 and self._passed_over(STEPS[i]):
            i += direction
        self.go(i)

    def _back(self) -> None:
        self._step(-1)

    def _leave_step(self) -> None:
        self._poll.stop()
        self.meter.set_source(None)
        self._monitor(False)
        if self.recorder.capturing:
            self.recorder.toggle_capture()

    def _next(self) -> None:
        if self.index >= len(STEPS) - 1:
            self._close(True)
        elif STEPS[self.index] == "models" and self._asr_state() in BUSY_STATES:
            return                    # Continue waits for the speech model (also when Enter is pressed)
        else:
            self._step(1)

    def skip(self) -> None:
        self._close(False)

    def _close(self, completed: bool) -> None:
        if self._done:
            return
        self._done = True
        self._leave_step()
        self._slow_timer.stop()          # only the guide goes away: the models keep loading in the controller
        try:
            self.ctl.set("onboarding_done", True)
        except Exception:
            pass                     # worst case the guide shows once more next launch
        self.hide()
        self.finished.emit(completed)

    def closeEvent(self, e):
        self._close(False)
        e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.skip()
        else:
            super().keyPressEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        screen = QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()
        g = screen.availableGeometry()
        self.move(g.center().x() - self.width() // 2, g.center().y() - self.height() // 2)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and e.position().y() < self.M + 60 and self.windowHandle():
            self.windowHandle().startSystemMove()

    # ------------------------------------------------------------------ controller events
    def _monitor(self, on: bool) -> None:
        try:
            self.ctl.set_level_monitor(on)
        except Exception:
            pass

    def _model_status(self, name: str) -> dict:
        try:
            return self.ctl.model_status(name) or {"state": "pending", "detail": ""}
        except Exception:
            return {"state": "pending", "detail": ""}

    def _asr_state(self) -> str:
        return str(self._model_status("asr").get("state", "pending"))

    def _fill_mics(self, mics) -> None:
        cur = self.ctl.get("mic_device")
        opts = [(m.get("id"), str(m.get("name", "Microphone"))) for m in (mics or [])] or [(None, "System default")]
        if cur is not None and cur not in [o[0] for o in opts]:
            opts.append((cur, f"{cur} (not connected)"))
        self.mic.set_options(opts, cur)

    def _pick_mic(self, device) -> None:
        try:
            self.ctl.set("mic_device", device)
        except ValueError as exc:
            self.mic_status.set_role("RED")
            self.mic_status.setText(str(exc))
            self._fill_mics(self.ctl.microphones())
            return
        self._heard = 0
        self.mic_status.set_role("TEXT_DIM")
        self.mic_status.setText("Listening… say something.")

    def _poll_level(self) -> None:
        try:
            lvl = float(self.ctl.level())
        except Exception:
            lvl = 0.0
        if lvl > 0.06:
            self._heard += 1
        if self._heard == 4:
            self.mic_status.set_role("GREEN")
            self.mic_status.setText("Loud and clear. Your microphone works.")

    def _on_state(self, state: str, mode: str, message: str) -> None:
        if STEPS[self.index] != "hotkey" or self._practised and state == "idle":
            return
        texts = {"listening": ("Listening… keep holding and speak.", "CYAN"),
                 "locked": ("Hands-free: tap the hotkey once more when you are done.", "CYAN"),
                 "processing": ("Transcribing…", "AMBER"),
                 "error": (message or "That did not work. Try once more.", "RED"),
                 "idle": ("Waiting for you to hold the hotkey.", "TEXT_DIM")}
        if state in texts:
            self.try_status.set_role(texts[state][1])
            self.try_status.setText(texts[state][0])

    def _on_transcript(self, entry: dict) -> None:
        if STEPS[self.index] != "hotkey":
            return
        self._practised = True
        self.try_status.set_role("GREEN")
        self.try_status.setText("That is all there is to it. Try another sentence, or continue.")
        text = str(entry.get("text", ""))
        QTimer.singleShot(450, lambda: self._ensure_practice_text(text))

    def _ensure_practice_text(self, text: str) -> None:
        """The paste normally lands in the practice box; if focus was elsewhere, show the words anyway."""
        if text.strip() and text.strip() not in self.practice.toPlainText():
            self.practice.appendPlainText(text)

    def _on_model(self, name: str, status: dict) -> None:
        line = self.model_lines.get(name)
        if line is not None:
            line.set_status(status or {})
        states = {n: str(self._model_status(n).get("state")) for n in self.model_lines}
        asr = states.get("asr", "pending")
        failed = [n for n, s in states.items() if s == "failed"]
        if asr == "failed":
            note, role = ("Speech recognition could not be loaded, so the hotkey practice has to wait. Try again, or "
                          "skip the practice and retry later from Settings > On-device models."), "RED"
        elif all(s in ("ready", "degraded") for s in states.values()):
            note, role = "Everything is downloaded and ready.", "TEXT_DIM"
        elif asr == "ready":
            note, role = ("Speech recognition is ready, so you can try your hotkey now. "
                          + ("Smart cleanup could not be loaded; you can retry it from Settings > On-device models."
                             if failed else "Smart cleanup keeps loading in the background.")), "TEXT_DIM"
        elif self._slow:
            note, role = ("This is taking a while. A first download can take several minutes, and a first start on "
                          "a graphics card up to 30 seconds. You can also close this guide: VoxFlow keeps loading in "
                          "the background, and Settings > On-device models shows the progress."), "AMBER"
        else:
            note, role = ("The hotkey practice opens as soon as speech recognition is ready. You can also close this "
                          "guide: VoxFlow keeps loading in the background."), "TEXT_DIM"
        self.models_note.set_role(role)
        self.models_note.setText(note)
        self.btn_retry.setVisible(bool(failed))
        if asr not in BUSY_STATES:
            self._slow_timer.stop()
        elif STEPS[self.index] == "models" and not self._slow and not self._slow_timer.isActive():
            self._slow_timer.start()     # also when a retry's "loading" arrives after the click (queued signal)
        if STEPS[self.index] == "models":
            self._gate_models_step(asr)

    def _gate_models_step(self, asr: str) -> None:
        """On the model step Continue waits for the speech model; after a failure it skips the practice instead."""
        was_waiting = not self.btn_next.isEnabled()
        busy = asr in BUSY_STATES
        text = "Skip practice" if asr == "failed" else "Continue"
        self.btn_next.setEnabled(not busy)
        self.btn_next.setText(text)
        self.btn_next.setAccessibleName(text)
        self.btn_next.setToolTip("Waiting for speech recognition…" if busy else "")
        if was_waiting and not busy and self.isVisible():
            self.btn_next.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_slow(self) -> None:
        if STEPS[self.index] == "models" and self._asr_state() in BUSY_STATES:
            self._slow = True
            self._on_model("asr", self._model_status("asr"))

    def _retry(self) -> None:
        failed = [n for n in self.model_lines if self._model_status(n).get("state") == "failed"]
        try:
            self.ctl.reload_models(failed or None)
        except Exception as exc:  # noqa: BLE001 - the guide must never trap the user
            self.models_note.set_role("RED")
            self.models_note.setText(f"That did not work: {exc}")
            return
        self._slow = False
        self._slow_timer.stop()
        for name in self.model_lines:          # the real controller reports "loading" later; _on_model starts the timer
            self._on_model(name, self._model_status(name))

    def _on_captured(self, combo: str) -> None:
        if self.recorder.capturing:
            self.recorder.captured(combo or None)

    def _hotkey_changed(self, combo: str) -> None:
        try:
            self.ctl.set("hotkey", combo)
        except ValueError as exc:
            self.recorder.set_combo(self.ctl.get("hotkey"))
            self.recorder.show_message(str(exc))

    def _on_config(self, key: str, value) -> None:
        if key == "hotkey":
            self.caps.set_combo(value)
            self.recorder.set_combo(value)
        elif key == "mic_device":
            self._fill_mics(self.ctl.microphones())

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.M
        r = QRectF(m + 0.5, m + 0.5, self.width() - 2 * m - 1, self.height() - 2 * m - 1)
        T.paint_shadow(p, r, T.R_XL, spread=m - 1, strength=1.0, dy=3)
        path = T.rounded(r, T.R_XL)
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, T.BG1)
        g.setColorAt(1, T.BG0)
        p.fillPath(path, g)
        p.save()
        p.setClipPath(path)
        for col, centre, radius in ((T.CYAN, QPointF(r.left() + 40, r.top()), 330.0),
                                    (T.VIOLET, QPointF(r.right() - 30, r.bottom()), 380.0)):
            rg = QRadialGradient(centre, radius)
            rg.setColorAt(0, T.alpha(col, T.BLOOM + 0.03))
            rg.setColorAt(1, T.alpha(col, 0))
            p.fillRect(r, rg)
        p.restore()
        border = QLinearGradient(r.topLeft(), r.bottomRight())
        border.setColorAt(0, T.alpha(T.CYAN, 0.5))
        border.setColorAt(0.5, T.STROKE_STRONG)
        border.setColorAt(1, T.alpha(T.VIOLET, 0.5))
        p.setPen(QPen(border, 1.1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        if self.index == 0:
            paint_logo(p, QRectF(r.left() + 36, r.top() + 44, 56, 56))
        else:
            paint_logo(p, QRectF(r.left() + 36, r.top() + 22, 24, 24))
