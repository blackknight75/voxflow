"""Settings > Updates and the "new version" dialog (Controller.update_info / check_updates / install_update).

Release notes are shown as plain text (``updater.plain_notes`` already removed images, HTML and link targets), so
nothing in them is rendered or fetched.
"""
from __future__ import annotations

import datetime as _dt

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QWidget

from . import theme as T
from .components import Modal, Row, Section, clock_label, day_label
from .widgets import Button, ChoiceButton, Label, ProgressBar, TextArea

MODE_OPTIONS = [("auto", "Install automatically"), ("notify", "Notify me, I choose"), ("off", "Off (never check)")]
WHEN_OPTIONS = [("quit", "When I quit VoxFlow"), ("time", "At a set time")]


def time_options(current: str) -> list[tuple[str, str]]:
    """Every hour of the day, plus the stored time when it is not on the hour."""
    hours = [f"{h:02d}:00" for h in range(24)]
    if current and current not in hours:
        hours = sorted(hours + [current])
    return [(t, t) for t in hours]


def checked_text(ts) -> str:
    if not ts:
        return "Not checked yet."
    return f"Last checked {day_label(ts).lower() if day_label(ts) in ('Today', 'Yesterday') else day_label(ts)} " \
           f"at {clock_label(ts)}."


def status_text(info: dict) -> tuple[str, str]:
    """(sentence, colour role) for the status row of Settings > Updates."""
    rel = info.get("available")
    phase = info.get("phase")
    version = rel["version"] if rel else ""
    if phase == "checking":
        return "Checking for updates…", "TEXT_DIM"
    if phase == "downloading":
        prog = info.get("progress")
        pct = f" {int(prog * 100)}%" if isinstance(prog, (int, float)) else ""
        return f"Downloading VoxFlow {version}…{pct}", "TEXT_DIM"
    if phase == "waiting":
        return f"VoxFlow will restart to install {version} as soon as you finish dictating.", "AMBER"
    if phase == "restarting":
        return f"Restarting to install VoxFlow {version}…", "AMBER"
    parts, role = [], "TEXT_DIM"
    if info.get("mode") == "off":
        parts.append("Automatic checks are off: VoxFlow asks GitHub only when you press Check now.")
    if rel:
        line = f"VoxFlow {version} is available."
        if info.get("skipped"):
            line += " You skipped this version."
        elif info.get("install_on_quit"):
            line += " It will be installed when you quit VoxFlow."
        elif info.get("mode") == "auto" and info.get("when") == "time" and info.get("downloaded"):
            line += f" It will be installed at {info.get('time')} when VoxFlow and the PC are idle."
        parts.append(line)
    elif info.get("last_result"):
        parts.append(str(info["last_result"]))
        if not info.get("last_ok"):
            role = "AMBER"
    if info.get("mode") != "off" or info.get("last_check"):
        parts.append(checked_text(info.get("last_check")))
    if info.get("managed"):
        parts.append(str(info["managed"]))
    if info.get("error"):
        parts.append(str(info["error"]))
        role = "AMBER"
    return " ".join(parts), role


class UpdatesSection(Section):
    """Mode, when to install automatically, the current version, the last check and "Check now"."""

    def __init__(self, page):
        super().__init__("Updates")
        self.page = page
        ctl = page.ctl
        self.mode = ChoiceButton(name="Updates")
        self.mode.setObjectName("settings_update_mode")
        self.mode.setFixedWidth(220)
        self.mode.set_options(MODE_OPTIONS, ctl.get("update_mode"))
        self.mode.chosen.connect(lambda v: page._set_or_resync("update_mode", v))
        self.mode_row = self.add(Row(
            "Updates", "VoxFlow asks GitHub once a day whether a new version is out. The request is anonymous: "
                       "nothing about you or your PC is sent. Off means no automatic request: VoxFlow asks only "
                       "when you press Check now.", self.mode,
            icon="refresh"))
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.when = ChoiceButton(name="Install automatically")
        self.when.setObjectName("settings_update_when")
        self.when.setFixedWidth(190)
        self.when.set_options(WHEN_OPTIONS, ctl.get("update_install_when"))
        self.when.chosen.connect(lambda v: page._set_or_resync("update_install_when", v))
        self.at = ChoiceButton(name="Install time")
        self.at.setObjectName("settings_update_time")
        self.at.setFixedWidth(96)
        self.at.setMinimumWidth(96)
        self.at.set_options(time_options(ctl.get("update_time")), ctl.get("update_time"))
        self.at.chosen.connect(lambda v: page._set_or_resync("update_time", v))
        lay.addWidget(self.when)
        lay.addWidget(self.at)
        self.when_row = self.add(Row(
            "Install", "Never while you dictate. At a set time VoxFlow waits until the PC has not been used for 10 "
                       "minutes, installs, and starts again.", box))
        self.when_row.setObjectName("settings_update_when_row")
        actions = QWidget()
        al = QHBoxLayout(actions)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)
        self.btn_see = Button("See update", "primary", height=30)
        self.btn_see.setObjectName("settings_update_see")
        self.btn_see.clicked.connect(lambda: page.open_update())
        self.btn_check = Button("Check now", "secondary", icon="refresh", height=30)
        self.btn_check.setObjectName("settings_update_check")
        self.btn_check.clicked.connect(lambda: page.run(ctl.check_updates))
        al.addWidget(self.btn_see)
        al.addWidget(self.btn_check)
        self.bar = ProgressBar()
        self.bar.hide()
        self.status_row = self.add(Row(f"VoxFlow {ctl.version}", "", actions, icon="info", below=self.bar))
        self.status_row.setObjectName("settings_update_status")
        for key in ("update_mode", "update_install_when", "update_time"):
            page._sync[key] = lambda _v: self.refresh()
        ctl.update_changed.connect(self.refresh)
        self.refresh()

    def refresh(self, info: dict | None = None) -> None:
        ctl = self.page.ctl
        info = info if isinstance(info, dict) else ctl.update_info()
        self.mode.set_value(info["mode"])
        self.when.set_value(info["when"])
        self.at.set_options(time_options(info["time"]), info["time"])
        self.at.setVisible(info["when"] == "time")
        finder = getattr(self.page, "find", None)              # the search box (QWidget.find before it exists)
        if not (hasattr(finder, "text") and finder.text().strip()):    # a search shows every matching row
            self.when_row.setVisible(info["mode"] == "auto" and bool(info["can_install"]))
        text, role = status_text(info)
        self.status_row.set_sub(text, role)
        idle = info["phase"] == "idle"
        self.btn_check.setEnabled(idle)
        self.btn_check.setToolTip("Asks GitHub once, now. Automatic checks stay off." if info["mode"] == "off"
                                  else "Asks GitHub now instead of waiting for the daily check.")
        self.btn_see.setVisible(bool(info.get("available")))
        downloading = info["phase"] == "downloading"
        self.bar.setVisible(downloading)
        if downloading:
            prog = info.get("progress")
            self.bar.set_value(prog if isinstance(prog, (int, float)) else None)


def _published(rel: dict) -> str:
    try:
        when = _dt.datetime.fromisoformat(str(rel.get("published", "")).replace("Z", "+00:00"))
    except ValueError:
        return ""
    return f" Released {when.strftime('%B')} {when.day}, {when.year}."


def update_dialog(win, ctl, info: dict) -> Modal:
    """"VoxFlow <version> is available": the release notes and the four choices (or, for a copy that cannot
    update itself, why not)."""
    rel = info["available"]
    version = rel["version"]
    m = Modal(win.panel, f"VoxFlow {version} is available",
              f"You have VoxFlow {info['version']}.{_published(rel)}", width=540, name="update_dialog")
    notes = TextArea(rel.get("notes") or "This release has no notes.", name="Release notes")
    notes.setObjectName("update_notes")
    notes.setReadOnly(True)
    notes.setFixedHeight(230)
    m.body.addWidget(notes)
    installable = bool(info.get("installable"))
    if installable:
        size = rel.get("size") or 0
        hint = "Install now restarts VoxFlow, after any dictation in progress." + (
            f" The download is about {size / 1e9:.1f} GB." if size >= 1e8 else
            f" The download is about {max(1, round(size / 1e6))} MB." if size else "")
    else:
        hint = info.get("managed") or "This release has no Windows installer. Get it from the releases page."
    lead = Label(hint, T.F_SMALL, "TEXT_DIM")
    lead.setObjectName("update_hint")
    m.body.addWidget(lead)
    m.body.addWidget(m.error)

    def done(fn, message: str):
        def go():
            try:
                fn()
            except ValueError as exc:
                m.show_error(str(exc))
                return
            m.close_modal()
            win.toast("info", message)                 # every choice says what happens next
        return go

    row = QHBoxLayout()
    row.setSpacing(T.S2)
    skip = Button("Skip this version", "ghost")
    skip.setObjectName("update_dialog_skip")
    skip.clicked.connect(done(ctl.skip_update, f"VoxFlow {version} will not be offered again. Check now shows it."))
    later = Button("Remind me later", "ghost")
    later.setObjectName("update_dialog_later")
    later.clicked.connect(done(ctl.remind_update_later, "VoxFlow will remind you tomorrow."))
    row.addWidget(skip)
    row.addStretch(1)
    row.addWidget(later)
    if installable:
        on_quit = Button("Install when I quit", "secondary")
        on_quit.setObjectName("update_dialog_quit")
        on_quit.clicked.connect(done(lambda: ctl.install_update("quit"),
                                     f"VoxFlow {version} will be installed when you quit VoxFlow."))
        now = Button("Install now", "primary")
        now.setObjectName("update_dialog_now")
        now.clicked.connect(done(lambda: ctl.install_update("now"),
                                 f"Downloading VoxFlow {version}. VoxFlow restarts to install it when it is ready."))
        row.addWidget(on_quit)
        row.addWidget(now)
        m.btn_primary = now
    else:
        page = Button("Download" if info.get("kind") in ("allusers", "portable") else "Releases page", "primary")
        page.setObjectName("update_dialog_page")
        page.clicked.connect(lambda: (QDesktopServices.openUrl(QUrl(info.get("releases_page"))), m.close_modal()))
        row.addWidget(page)
        m.btn_primary = page
    m.btn_cancel = later
    m.body.addSpacing(T.S1)
    m.body.addLayout(row)
    m.card.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return m
