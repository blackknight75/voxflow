"""The six pages of the main window. Each page reads and writes only through the controller facade."""
from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import QBoxLayout, QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from . import dialogs
from . import theme as T
from .components import (BarChart, Card, EmptyState, Modal, Row, Section, StatCard, TranscriptList, app_label,
                         confirm, fmt_count, fmt_duration)
from .updates import UpdatesSection, update_dialog
from .widgets import (Button, Caption, CheckBox, ChoiceButton, HotkeyRecorder, IconButton, KeyCaps, Label, LevelMeter, LineEdit,
                      PopupMenu, ProgressBar, ScrollArea, SearchField, Segmented, Slider, TextArea, Toggle, painter)


def field(modal: Modal, caption: str, widget: QWidget, hint: str = "") -> None:
    """Caption + input (+ hint) as one tight group inside a dialog."""
    box = QVBoxLayout()
    box.setSpacing(6)
    box.addWidget(Caption(caption))
    box.addWidget(widget)
    if hint:
        box.addWidget(Label(hint, T.F_SMALL, "TEXT_DIM"))
    modal.body.addLayout(box)


class Page(QWidget):
    """Header (title, subtitle, actions) above a scrolling body. Refreshes lazily when it becomes visible."""

    def __init__(self, ctl, win, title: str, subtitle: str = ""):
        super().__init__(win)
        self.ctl, self.win = ctl, win
        self._dirty = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 0, 8, 0)
        outer.setSpacing(0)
        col = QVBoxLayout()                     # title + actions on one line; subtitle and toolbar below
        col.setContentsMargins(0, 0, 20, 14)
        col.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(T.S2)
        self.title = Label(title, T.F_H1, "TEXT", T.W_BOLD, -0.2, wrap=False)
        head.addWidget(self.title)
        head.addStretch(1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(T.S2)
        head.addLayout(self.actions)
        col.addLayout(head)
        self.subtitle = Label(subtitle, T.F_BODY, "TEXT_DIM")
        self.subtitle.setVisible(bool(subtitle))
        col.addWidget(self.subtitle)
        self.head_col = col
        outer.addLayout(col)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 2, 8, 30)
        self.body.setSpacing(T.S4)
        self.scroll = ScrollArea(self.content)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)   # stable gutter, no reflow
        outer.addWidget(self.scroll, 1)

    # ---- lifecycle
    def on_enter(self) -> None:
        if self._dirty:
            self._dirty = False
            self.refresh()

    def on_leave(self) -> None:
        """Called when another page is shown or the window hides."""

    def invalidate(self, *_a) -> None:
        if self.isVisible() and self.win.current == self.objectName().removeprefix("page_"):
            self.refresh()
        else:
            self._dirty = True

    def refresh(self) -> None:
        """Rebuild from the controller's current data."""

    # ---- helpers
    def run(self, fn, *args, success: str = "", **kwargs):
        """Call a store/controller method; show a toast instead of raising. Returns None on failure."""
        try:
            res = fn(*args, **kwargs)
        except ValueError as exc:
            self.win.toast("error", str(exc))
            return None
        except Exception as exc:                 # never let a store problem take the window down
            self.win.toast("error", f"That did not work: {exc}")
            return None
        if success:
            self.win.toast("success", success)
        return True if res is None else res

    def setting(self, key: str, value) -> bool:
        return self.run(self.ctl.set, key, value) is not None


def greeting(now: _dt.datetime | None = None) -> str:
    h = (now or _dt.datetime.now()).hour
    return "Good morning" if 5 <= h < 12 else "Good afternoon" if 12 <= h < 18 else "Good evening"


# =========================================================================== Home
class _TopApps(QWidget):
    """Ranked list of the apps you dictate into most, with proportional bars."""

    ROW_H = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        self.setMinimumHeight(5 * self.ROW_H)
        self.setAccessibleName("Most used apps")
        T.bus().changed.connect(self.update)

    def set_rows(self, rows: list[dict]) -> None:
        self.rows = list(rows)[:5]
        self.setAccessibleDescription(", ".join(f"{app_label(r.get('exe', ''))} {int(r.get('words', 0))} words"
                                                for r in self.rows))
        self.update()

    def paintEvent(self, _):
        p = painter(self)
        if not self.rows:
            p.setPen(T.TEXT_FAINT)
            p.setFont(T.font(T.F_BODY))
            p.drawText(QRectF(self.rect()), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                       "The apps you dictate into\nwill be ranked here.")
            return
        peak = max(1, max(int(r.get("words", 0)) for r in self.rows))
        name_f, num_f = T.font(T.F_BODY, T.W_MEDIUM), T.font(T.F_SMALL)
        for i, r in enumerate(self.rows):
            y = i * self.ROW_H
            words = int(r.get("words", 0))
            num = f"{fmt_count(words)} words"
            nw = QFontMetricsF(num_f).horizontalAdvance(num)
            p.setFont(name_f)
            p.setPen(T.TEXT)
            avail = self.width() - nw - 12
            p.drawText(QRectF(0, y, avail, 20), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(app_label(str(r.get("exe", ""))) or "Unknown app",
                                                  Qt.TextElideMode.ElideRight, int(avail)))
            p.setFont(num_f)
            p.setPen(T.TEXT_DIM)
            p.drawText(QRectF(0, y, self.width(), 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, num)
            track = QRectF(0, y + 23, self.width(), 4)
            p.fillPath(T.rounded(track, 2), T.CONTROL)
            fill = QRectF(0, y + 23, max(4.0, track.width() * words / peak), 4)
            p.fillPath(T.rounded(fill, 2), T.neon_gradient(0, self.width()))


class HomePage(Page):
    def __init__(self, ctl, win):
        super().__init__(ctl, win, greeting())
        hint = QHBoxLayout()
        hint.setSpacing(8)
        hint.addWidget(Label("Hold", T.F_BODY, "TEXT_DIM", wrap=False))
        self.caps = KeyCaps(ctl.get("hotkey"))
        hint.addWidget(self.caps)
        hint.addWidget(Label("in any app, speak, and release. VoxFlow types it for you.", T.F_BODY, "TEXT_DIM",
                             wrap=False))
        hint.addStretch(1)
        self.head_col.addLayout(hint)

        stats = QGridLayout()
        stats.setSpacing(T.S3)
        self._stat_grid = stats
        self._stat_cols = 4
        self.s_words = StatCard("text", "Words dictated", name="stat_words")
        self.s_wpm = StatCard("gauge", "Average speed", name="stat_wpm")
        self.s_saved = StatCard("clock-fast", "Time saved", name="stat_saved")
        self.s_streak = StatCard("flame", "Day streak", name="stat_streak")
        self._stat_cards = (self.s_words, self.s_wpm, self.s_saved, self.s_streak)
        self._flow_stats(4)
        self.stats_box = QWidget()                 # hidden on a fresh install (D-09): no dashboard of zeros
        stats.setContentsMargins(0, 0, 0, 0)
        self.stats_box.setLayout(stats)
        self.body.addWidget(self.stats_box)

        self.hero = Card()                          # first run: one clear next step instead of empty charts
        self.hero.setObjectName("home_hero")
        self.hero.accent = 0.6
        self.hero.body.addWidget(Label("Try your first dictation", T.F_TITLE, "TEXT", T.W_SEMI, wrap=False))
        self.hero_hint = Label("", T.F_BODY, "TEXT_DIM")
        self.hero.body.addWidget(self.hero_hint)
        self.practice = TextArea("", "Click here, then hold the shortcut and say something…")
        self.practice.setObjectName("home_practice")
        self.practice.setAccessibleName("Practice box")
        self.practice.setFixedHeight(92)
        self.hero.body.addWidget(self.practice)
        self.hero.hide()
        self.body.addWidget(self.hero)

        self.mid_box = QWidget()
        mid = QHBoxLayout(self.mid_box)
        mid.setContentsMargins(0, 0, 0, 0)
        self._mid = mid
        mid.setSpacing(T.S3)
        act = Card()
        top = QHBoxLayout()
        top.addWidget(Label("Activity", T.F_TITLE, "TEXT", T.W_SEMI, wrap=False))
        top.addStretch(1)
        self.range = Segmented([(7, "7 days"), (14, "14 days"), (30, "30 days")], 14, name="Activity range")
        self.range.setObjectName("home_range")
        self.range.changed.connect(lambda _v: self._load_chart(True))
        top.addWidget(self.range)
        act.body.addLayout(top)
        self.chart = BarChart()
        self.chart.setObjectName("home_chart")
        act.body.addWidget(self.chart)
        mid.addWidget(act, 3)
        apps = Card()
        apps.body.addWidget(Label("Where you dictate", T.F_TITLE, "TEXT", T.W_SEMI, wrap=False))
        self.top_apps = _TopApps()
        self.top_apps.setObjectName("home_top_apps")
        apps.body.addWidget(self.top_apps, 1)
        mid.addWidget(apps, 2)
        self.body.addWidget(self.mid_box)

        rec = QHBoxLayout()
        self.recent_title = Label("Recent", T.F_TITLE, "TEXT", T.W_SEMI, wrap=False)
        rec.addWidget(self.recent_title)
        rec.addStretch(1)
        self.btn_all = Button("View all", "ghost", icon="chevron-right", height=30)
        self.btn_all.setObjectName("home_view_all")
        self.btn_all.clicked.connect(lambda: win.show_page("history"))
        rec.addWidget(self.btn_all)
        self.body.addLayout(rec)
        self.recent = TranscriptList(group_by_day=False, show_delete=False)
        self.recent.setObjectName("home_recent")
        self.recent.copy_requested.connect(lambda it: ctl.copy_text(self.recent.body_text(it)))
        self.recent.paste_requested.connect(lambda it: paste_again(self, it))
        self.body.addWidget(self.recent)
        self.empty = EmptyState("mic", "Nothing dictated yet")
        self.empty.setObjectName("home_empty")
        self.empty.button.setObjectName("home_empty_action")
        self.body.addWidget(self.empty)
        self.body.addStretch(1)

        ctl.stats_changed.connect(self.invalidate)
        ctl.history_changed.connect(self.invalidate)
        ctl.config_changed.connect(self._on_config)

    def _flow_stats(self, cols: int) -> None:
        """Four tiles in a row, or two by two when the window is narrow (D-13)."""
        self._stat_cols = cols
        for i, card in enumerate(self._stat_cards):
            self._stat_grid.addWidget(card, i // cols, i % cols)
        for c in range(4):
            self._stat_grid.setColumnStretch(c, 1 if c < cols else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        narrow = self.width() < 720
        cols = 2 if narrow else 4
        if cols != self._stat_cols:
            self._flow_stats(cols)
        self._mid.setDirection(QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight)

    def _on_config(self, key: str, value) -> None:
        if key == "hotkey":
            self.caps.set_combo(value)
            self.invalidate()
        elif key == "save_history":
            self.invalidate()

    def _load_chart(self, animate: bool = False) -> None:
        try:
            self.chart.set_series(self.ctl.stats.daily(int(self.range.value)), animate)
        except Exception:
            self.chart.set_series([])

    def refresh(self) -> None:
        self.title.setText(greeting())
        try:
            s = self.ctl.stats.summary()
        except Exception:
            s = {}
        today = int(s.get("today_words", 0))
        sessions = int(s.get("sessions", 0))
        self.s_words.set_value(fmt_count(s.get("total_words", 0)),
                               f"{fmt_count(today)} today" if today else f"{sessions:,} dictations")
        self.s_wpm.set_value(f"{float(s.get('avg_wpm', 0)):.0f}", "words per minute")
        self.s_saved.set_value(fmt_duration(s.get("time_saved_s", 0)), "vs typing at 40 wpm")
        streak = int(s.get("streak_days", 0))
        self.s_streak.set_value(str(streak), ("day" if streak == 1 else "days") + " in a row")
        self._load_chart()
        try:
            self.top_apps.set_rows(self.ctl.stats.top_apps(5))
            items = self.ctl.history.list(5)
        except Exception:
            items = []
        first_run = not items and sessions == 0
        for w in (self.stats_box, self.mid_box):
            w.setVisible(not first_run)
        self.hero.setVisible(first_run)
        self.recent_title.setVisible(not first_run or not self.ctl.get("save_history"))
        if first_run:
            self.hero_hint.setText("Say a sentence and let go. The text appears right here.")   # the keys are in the line above
        self.recent.set_items(items)
        self.recent.setVisible(bool(items))
        self.btn_all.setVisible(bool(items))
        self.empty.setVisible(not items)
        if not items:
            if self.ctl.get("save_history"):
                self.empty.show_state("empty", "mic", "Nothing dictated yet",
                                      f"Hold {self.ctl.pretty_hotkey(self.ctl.get('hotkey'))} in any app and start "
                                      "talking. Your latest transcripts will show up here.")
            else:
                self.empty.show_state("empty", "history", "History is off",
                                      "Transcripts are not being saved, but your stats are still counted.",
                                      "Turn history on", lambda: self.setting("save_history", True))


def paste_again(page: Page, entry: dict) -> None:
    """Paste a transcript into the app the user was in before VoxFlow: get out of the way first, then paste."""
    ctl, win = page.ctl, page.win
    entry_id = entry.get("id")
    if ctl.is_demo:
        if ctl.paste_last(entry_id):
            win.toast("info", "Demo mode: this would paste into the app you were using.")
        return
    win.hide()
    QTimer.singleShot(350, lambda: ctl.paste_last(entry_id))


# =========================================================================== History
class HistoryPage(Page):
    PAGE_SIZE = 200

    def __init__(self, ctl, win):
        super().__init__(ctl, win, "History", "Everything you have dictated, stored only on this PC.")
        self._limit = self.PAGE_SIZE
        self.search = SearchField("Search transcripts")
        self.search.setObjectName("history_search")
        self.search.setMinimumWidth(180)
        self.search.setMaximumWidth(340)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(160)
        self._debounce.timeout.connect(self._search_changed)
        self.search.textChanged.connect(lambda _t: self._debounce.start())
        self.view = Segmented([(False, "Cleaned"), (True, "Raw")], False, name="Show cleaned or raw transcript")
        self.view.setObjectName("history_raw_toggle")
        self.view.setToolTip("Cleaned = what was pasted. Raw = exactly what the speech model heard.")
        self.view.changed.connect(lambda v: self.list.set_show_raw(bool(v)))
        self.btn_export = Button("Export", "secondary", icon="export")
        self.btn_export.setObjectName("history_export")
        self.btn_export.clicked.connect(self._export_menu)
        self.btn_clear = Button("Clear all", "danger", icon="trash")
        self.btn_clear.setObjectName("history_clear")
        self.btn_clear.clicked.connect(self._confirm_clear)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 8, 0, 0)
        bar.setSpacing(T.S2)
        bar.addWidget(self.search, 1)
        bar.addStretch(1)
        for w in (self.view, self.btn_export, self.btn_clear):
            bar.addWidget(w)
        self.head_col.addLayout(bar)

        # history is off but older transcripts are still on disk: say so above them, so nobody thinks new ones are kept
        self.btn_off_on = Button("Turn history on", "secondary", height=30)
        self.btn_off_on.setObjectName("history_off_turn_on")
        self.btn_off_on.clicked.connect(lambda: self.setting("save_history", True))
        self.off_note = Card(padding=0, spacing=0)
        self.off_note.setObjectName("history_off_note")
        off_row = Row("History is off", "New dictations are not saved. The transcripts below were kept from before; "
                                        "Clear all deletes them.", self.btn_off_on, icon="history")
        off_row.divider = False
        self.off_note.body.addWidget(off_row)
        self.off_note.hide()
        self.body.addWidget(self.off_note)
        self.count = Label("", T.F_SMALL, "TEXT_DIM", wrap=False)
        self.count.setObjectName("history_count")
        self.body.addWidget(self.count)
        self.list = TranscriptList()
        self.list.setObjectName("history_list")
        self.list.copy_requested.connect(lambda it: ctl.copy_text(self.list.body_text(it)))
        self.list.paste_requested.connect(lambda it: paste_again(self, it))
        self.list.delete_requested.connect(self._delete)
        self.body.addWidget(self.list)
        self.btn_more = Button("Show older transcripts", "secondary")
        self.btn_more.setObjectName("history_more")
        self.btn_more.clicked.connect(self._more)
        self.body.addWidget(self.btn_more, 0, Qt.AlignmentFlag.AlignHCenter)
        self.empty = EmptyState("history", "No transcripts yet")
        self.empty.setObjectName("history_empty")
        self.empty.button.setObjectName("history_empty_action")
        self.body.addWidget(self.empty)
        self.body.addStretch(1)
        self._menu: PopupMenu | None = None

        ctl.history_changed.connect(self.invalidate)
        ctl.config_changed.connect(lambda k, _v: k == "save_history" and self.invalidate())

    def _search_changed(self) -> None:
        self._limit = self.PAGE_SIZE
        self.refresh()

    def _more(self) -> None:
        self._limit += self.PAGE_SIZE
        self.refresh()

    def refresh(self) -> None:
        q = self.search.text().strip()
        failed = ""
        try:
            total = self.ctl.history.count()
            items = self.ctl.history.search(q, self._limit + 1) if q else self.ctl.history.list(self._limit + 1)
        except Exception as exc:
            failed = str(exc) or type(exc).__name__
            total, items = 0, []
        more = len(items) > self._limit
        items = items[:self._limit]
        self.off_note.setVisible(total > 0 and not failed and not self.ctl.get("save_history"))
        self.list.set_items(items)
        self.list.setVisible(bool(items))
        self.btn_more.setVisible(more)
        self.btn_export.setEnabled(total > 0)
        self.btn_clear.setEnabled(total > 0)
        self.view.setEnabled(bool(items))
        if q:
            n = f"{len(items)}{'+' if more else ''}"
            self.count.setText(f"{n} match{'es' if len(items) != 1 or more else ''} for “{q}”")
        else:
            self.count.setText(f"{total:,} transcript{'s' if total != 1 else ''}")
        self.count.setVisible(bool(items))
        self.empty.setVisible(not items)
        if items:
            return
        if failed:
            self.empty.show_state("error", "warning", "Your history could not be loaded",
                                  f"{failed}. Nothing was changed on disk.", "Try again", self.refresh)
        elif q:
            self.empty.show_state("empty", "search", "No matches",
                                  f"Nothing in your history contains “{q}”. Try a different word.",
                                  "Clear search", self.search.clear)
        elif self.ctl.get("save_history"):
            self.empty.show_state("empty", "history", "No transcripts yet",
                                  f"Hold {self.ctl.pretty_hotkey(self.ctl.get('hotkey'))} and say something. Every "
                                  "dictation is kept here so you can copy or paste it again.")
        else:
            self.empty.show_state("empty", "history", "History is off",
                                  "VoxFlow is not saving your transcripts. Turn history on to keep them here, on "
                                  "this PC only.", "Turn history on", lambda: self.setting("save_history", True))

    def _delete(self, entry: dict) -> None:
        if self.run(self.ctl.history.delete, entry.get("id")):
            self.win.toast("success", "Transcript deleted.", "Undo",
                           lambda: self.run(self.ctl.history.restore, entry))

    def _confirm_clear(self) -> None:
        n = self.ctl.history.count()
        self.win.open_modal(confirm(
            self.win.panel, "Clear all history?",
            f"This permanently deletes {n:,} transcript{'s' if n != 1 else ''} from this PC. Your stats are kept. "
            "This cannot be undone.", "Delete everything",
            lambda: self.run(self.ctl.history.clear, success="History cleared."), name="history_clear_confirm"))

    def _export_menu(self) -> PopupMenu:
        if self._menu is None:
            self._menu = PopupMenu(230)
        m = self._menu
        m.set_items([{"kind": "info", "text": "Export all transcripts as"},
                     {"kind": "action", "text": "Plain text (.txt)", "icon": "text", "on_click": lambda: self.export("txt")},
                     {"kind": "action", "text": "Spreadsheet (.csv)", "icon": "export", "on_click": lambda: self.export("csv")},
                     {"kind": "action", "text": "JSON (.json)", "icon": "export", "on_click": lambda: self.export("json")}])
        m.popup_below(self.btn_export)
        return m

    def export(self, fmt: str) -> None:
        names = {"txt": "Text files (*.txt)", "csv": "CSV files (*.csv)", "json": "JSON files (*.json)"}
        stamp = _dt.date.today().isoformat()
        path = dialogs.ask_save_path(self.win, "Export history", f"voxflow-history-{stamp}.{fmt}", names[fmt])
        if not path:
            return
        n = self.run(self.ctl.history.export, path, fmt)
        if n is not None:
            n = 0 if n is True else int(n)
            self.win.toast("success", f"Exported {n:,} transcript{'s' if n != 1 else ''} to {Path(path).name}.")


# =========================================================================== Dictionary
def _split_aliases(text: str) -> list[str]:
    return [a.strip() for a in text.replace("\n", ",").replace(";", ",").split(",") if a.strip()]


def ago(ts, now: float | None = None) -> str:
    """'just now' / '4 min ago' / '2 h ago' / 'yesterday at 14:05' / '3 March at 14:05' for an epoch time."""
    try:
        then = _dt.datetime.fromtimestamp(float(ts))
    except (TypeError, ValueError, OSError, OverflowError):
        return "at an unknown time"
    cur = _dt.datetime.fromtimestamp(now if now is not None else time.time())   # same clock the timestamps use
    secs = max(0, round((cur - then).total_seconds()))
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if then.date() == cur.date():
        return f"{int(secs // 3600)} h ago"
    if (cur.date() - then.date()).days == 1:
        return f"yesterday at {then:%H:%M}"
    return f"{then.day} {then:%B} at {then:%H:%M}" + (f", {then.year}" if then.year != cur.year else "")


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def merge_or_replace(win, title: str, facts: list[str], merge_text: str, replace_text: str, on_import,
                     name: str, parts: list[dict] | None = None) -> Modal:
    """Dialog shown before anything is imported: what is in the file, and how to bring it in.

    ``on_import(mode)`` gets "merge" or "replace" and returns True when the import worked (the dialog then
    closes) or a human error sentence to show inside the dialog.

    ``parts`` (backups): ``[{key, label, note, available, merge}]`` adds one tick box per part of the file;
    ``on_import(mode, keys)`` then also gets the ticked keys, or None when every available part is ticked.
    A part with ``merge=False`` is something a merge never touches (settings): its box is off while Merge is
    chosen."""
    m = Modal(win.panel, title, "", width=480, name=name)
    box = Card(padding=12, spacing=4)
    for i, line in enumerate(facts):
        box.body.addWidget(Label(line, T.F_BODY, "TEXT" if i == 0 else "TEXT_DIM", T.W_SEMI if i == 0 else T.W_NORMAL))
    m.body.addWidget(box)
    boxes: dict[str, CheckBox] = {}
    if parts:
        m.body.addWidget(Caption("What to import"))
        grid = QVBoxLayout()
        grid.setSpacing(2)
        for i in range(0, len(parts), 2):
            row = QHBoxLayout()
            row.setSpacing(T.S3)
            for part in parts[i:i + 2]:
                cb = CheckBox(part["label"], bool(part["available"]), part.get("note", ""))
                cb.setObjectName(f"{name}_part_{part['key']}")
                cb.setEnabled(bool(part["available"]))
                if not part["available"]:
                    cb.setToolTip("Not in this backup")
                cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                boxes[part["key"]] = cb
                row.addWidget(cb, 1)
            if len(parts[i:i + 2]) == 1:
                row.addStretch(1)
            grid.addLayout(row)
        m.body.addLayout(grid)
    m.parts = boxes
    m.body.addWidget(Caption("How should it be imported?"))
    mode = Segmented([("merge", "Merge"), ("replace", "Replace")], "merge", name="Merge or replace")
    mode.setObjectName(f"{name}_mode")
    mode.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    m.body.addWidget(mode)
    explain = Label(merge_text, T.F_SMALL, "TEXT_DIM")
    explain.setObjectName(f"{name}_explain")
    explain.setMinimumHeight(52)
    explain.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    m.body.addWidget(explain)

    wanted: dict[str, bool] = {}          # ticks of merge-exempt parts, remembered while Merge has them off

    def usable(part: dict) -> bool:
        return bool(part["available"]) and (mode.value == "replace" or part.get("merge", True))

    def chosen() -> list[str] | None:
        """Ticked parts; None = everything this mode can import from this file (the plain, whole import)."""
        can = [pt["key"] for pt in parts if usable(pt)]
        ticked = [k for k in can if boxes[k].isChecked()]
        return None if ticked == can else ticked

    def go():
        if parts:
            keys = chosen()
            if keys == []:
                m.show_error("Tick at least one part to import.")
                return
            result = on_import(mode.value, keys)
        else:
            result = on_import(mode.value)
        if result is True:
            m.close_modal()
        else:
            m.show_error(str(result))

    ok = m.add_buttons("Merge", go)

    def show_mode(value):
        replace = value == "replace"
        explain.set_role("RED" if replace else "TEXT_DIM")
        explain.setText(replace_text if replace else merge_text)
        ok.kind = "danger" if replace else "primary"
        ok.setText("Replace" if replace else "Merge")
        ok.setAccessibleName(ok.text())
        for part in parts or []:
            if part["available"] and not part.get("merge", True):     # e.g. settings: a merge never changes them
                cb = boxes[part["key"]]
                if replace:
                    cb.setChecked(wanted.get(part["key"], True))      # back to what the user had for Replace
                elif cb.isEnabled():
                    wanted[part["key"]] = cb.isChecked()
                if not replace:
                    cb.setChecked(False)                              # shown unticked: it will not be imported
                cb.setEnabled(replace)
                cb.setToolTip("" if replace else "Merge never changes this; choose Replace to take it from the backup")
        m.show_error("")

    mode.changed.connect(show_mode)
    for cb in boxes.values():
        cb.toggled.connect(lambda _on: m.show_error(""))
    m.mode = mode
    show_mode("merge")
    return win.open_modal(m)


def aliases_left_out(items: list[dict]) -> str:
    """The sounds-like spellings an import left out because another word already uses them (one word must own a
    spelling, or it would never fire): '“cube” was left out of Kubernetes: Cuba already uses it.'"""
    if len(items) == 1:
        it = items[0]
        return f"“{it['alias']}” was left out of {it['term']}: {it['owner']} already uses it."
    shown = "; ".join(f"“{it['alias']}” ({it['owner']})" for it in items[:3]) + ("; …" if len(items) > 3 else "")
    return f"{len(items)} sounds-like spellings were left out because another word already uses them: {shown}."


class _ImportExport:
    """Shared import/export plumbing for the Dictionary and Snippets pages: plain files and shareable packs."""

    noun = "entries"
    file_stub = "voxflow"
    pack_kind = ""
    import_filters = "JSON (*.json)"
    import_patterns = "*.json"
    export_filters = "JSON (*.json)"

    def _store(self):
        raise NotImplementedError

    def _pack_suffix(self) -> str:
        return str(self.ctl.PACK_SUFFIX)

    def do_import(self) -> None:
        suffix = self._pack_suffix()
        filters = f"Packs and files (*{suffix} {self.import_patterns});;VoxFlow pack (*{suffix});;" + self.import_filters
        path = dialogs.ask_open_path(self.win, f"Import {self.noun}", filters)
        if not path:
            return
        if path.lower().endswith(suffix):
            self.import_pack(path)
            return
        res = self.run(self._store().import_, path)
        if res is None:
            return
        if isinstance(res, dict):                      # {"added", "updated"}: say both, so nothing changes unseen
            added, updated = int(res.get("added", 0)), int(res.get("updated", 0))
        else:
            added, updated = (0 if res is True else int(res)), 0
        name = Path(path).name
        if added and updated:
            text = f"Imported {added} new {self.noun} and updated {updated} you already had."
        elif added:
            text = f"Imported {added} new {self.noun}."
        elif updated:
            text = f"Updated {updated} of your {self.noun} from {name}."
        else:
            text = f"No new {self.noun} found in {name}."
        self.win.toast("success" if added or updated else "info", text)

    def do_export(self) -> None:
        path = dialogs.ask_save_path(self.win, f"Export {self.noun}", f"{self.file_stub}.json", self.export_filters)
        if not path:
            return
        n = self.run(self._store().export, path)
        if n is not None:
            n = 0 if n is True else int(n)
            self.win.toast("success", f"Exported {n} {self.noun} to {Path(path).name}.")

    # ---- packs
    def import_pack(self, path: str) -> Modal | None:
        info = self.run(self.ctl.inspect_pack, path)
        if info is None:
            return None
        kind = str(info.get("kind", ""))
        if kind != self.pack_kind:
            other = "Snippets" if kind == "snippets" else "Dictionary"
            self.win.toast("warning", f"{Path(path).name} is a {kind or 'different'} pack. Import it on the {other} "
                                      "page.", f"Open {other}", lambda: self.win.show_page(other.lower()))
            return None
        count = plural(int(info.get("count", 0)), self.pack_item)
        if "new" in info:
            count += f": {int(info['new'])} new, {int(info.get('existing', 0))} already in your list"
        facts = [str(info.get("name") or Path(path).stem), count]
        if info.get("description"):
            facts.insert(1, str(info["description"]))
        if info.get("created"):
            facts.append(f"Made {ago(info['created'])}")
        if int(info.get("skipped") or 0):
            facts.append(f"{plural(int(info['skipped']), 'damaged entry', 'damaged entries')} in the file will be left out")

        def go(mode: str):
            try:
                res = self.ctl.import_pack(path, mode)
            except ValueError as exc:
                return str(exc)
            added, updated, removed = (int(res.get(k, 0)) for k in ("added", "updated", "removed"))
            bits = [f"{added} added"] + ([f"{updated} updated"] if updated else []) + \
                   ([f"{removed} removed"] if removed else [])
            self.win.toast("success" if added or updated or removed else "info",
                           f"Pack imported: {', '.join(bits)}." if added or updated or removed else
                           f"Nothing to import: you already have every {self.pack_item} in this pack.")
            if int(res.get("skipped") or 0):
                self.win.toast("warning", f"{plural(int(res['skipped']), 'damaged entry', 'damaged entries')} in the "
                                          "pack could not be read and was left out.")
            if res.get("aliases_skipped"):
                self.win.toast("warning", aliases_left_out(res["aliases_skipped"]))
            return True

        return merge_or_replace(
            self.win, "Import this pack?", facts,
            "New entries are added; entries you already have are updated from the pack. Everything else in your "
            "list stays as it is.",
            f"Deletes all of your current {self.noun} and uses only what is in this pack. This cannot be undone.",
            go, name="pack_import")

    def share_pack(self) -> Modal:
        m = Modal(self.win.panel, f"Share your {self.pack_kind} as a pack",
                  f"A pack is one file with the {self.noun} you tick below, and nothing else: no history, no "
                  "settings. Send it to a colleague and they import it on this page.", width=480, name="pack_export")
        name = LineEdit("", f"e.g. Team {self.pack_kind}", name="Pack name")
        name.setObjectName("pack_name")
        field(m, "Pack name", name)
        desc = LineEdit("", "What is in it, who it is for (optional)", name="Pack description")
        desc.setObjectName("pack_description")
        field(m, "Description (optional)", desc)

        # which entries go in (all by default)
        try:
            entries = self._store().list()
        except Exception:
            entries = []
        key = "term" if self.pack_kind == "dictionary" else "trigger"
        head = QHBoxLayout()
        head.setSpacing(4)
        count = Caption("")
        count.setObjectName("pack_selected_count")
        head.addWidget(count)
        head.addStretch(1)
        btn_all = Button("All", "ghost", height=26)
        btn_all.setObjectName("pack_select_all")
        btn_all.setAccessibleName("Select all entries")
        btn_none = Button("None", "ghost", height=26)
        btn_none.setObjectName("pack_select_none")
        btn_none.setAccessibleName("Select no entries")
        head.addWidget(btn_all)
        head.addWidget(btn_none)
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(0)
        ticks: list[tuple[str, CheckBox]] = []
        for e in entries:
            note = ", ".join(e.get("aliases") or []) if self.pack_kind == "dictionary" else \
                " ".join(str(e.get("text", "")).split())
            cb = CheckBox(str(e.get(key, "")), True, note)
            cb.setObjectName("pack_entry")
            cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            ticks.append((str(e.get("id")), cb))
            col.addWidget(cb)
        col.addStretch(1)
        listing = ScrollArea(holder)
        listing.setObjectName("pack_entries")
        listing.setFixedHeight(max(34, min(150, 4 + 28 * len(ticks))))
        m.entries = ticks

        def recount(*_a) -> None:
            n = sum(1 for _i, cb in ticks if cb.isChecked())
            count.setText(f"IN THE PACK  ·  {n} OF {len(ticks)}")
            m.show_error("")

        def tick_all(on: bool) -> None:
            for _i, cb in ticks:
                cb.setChecked(on)

        btn_all.clicked.connect(lambda: tick_all(True))
        btn_none.clicked.connect(lambda: tick_all(False))
        for _i, cb in ticks:
            cb.toggled.connect(recount)
        if ticks:
            m.body.addLayout(head)
            m.body.addWidget(listing)
        recount()

        def save():
            if not name.text().strip():
                name.set_invalid(True)
                m.show_error("Give the pack a name, so the person who gets it knows what it is.")
                return
            picked = [i for i, cb in ticks if cb.isChecked()]
            if ticks and not picked:
                m.show_error(f"Tick at least one {self.pack_item} to put in the pack.")
                return
            ids = None if len(picked) == len(ticks) else picked       # None = the whole list
            stub = "-".join(name.text().strip().lower().split()) or self.file_stub
            suffix = self._pack_suffix()
            path = dialogs.ask_save_path(self.win, "Save pack", stub + suffix, f"VoxFlow pack (*{suffix})")
            if not path:
                return
            try:
                n = self.ctl.export_pack(self.pack_kind, path, name.text().strip(), desc.text().strip(), ids)
            except ValueError as exc:
                m.show_error(str(exc))
                return
            m.close_modal()
            self.win.toast("success", f"Pack saved: {plural(int(n), self.pack_item)} in {Path(path).name}.")

        name.textChanged.connect(lambda _t: (name.set_invalid(False), m.show_error("")))
        m.add_buttons("Save pack…", save)
        return self.win.open_modal(m, name)

    def add_share_button(self, before: QWidget) -> None:
        self.btn_share = IconButton("share", f"Share a pack: save your {self.noun}, or some of them, as one file")
        self.btn_share.setObjectName(f"{'dict' if self.pack_kind == 'dictionary' else 'snippets'}_share")
        self.btn_share.clicked.connect(self.share_pack)
        self.actions.insertWidget(self.actions.indexOf(before), self.btn_share)


class DictionaryPage(Page, _ImportExport):
    noun = "dictionary entries"
    file_stub = "voxflow-dictionary"
    pack_kind = "dictionary"
    pack_item = "word"
    import_filters = "Dictionary files (*.json *.csv *.txt)"
    import_patterns = "*.json *.csv *.txt"
    export_filters = "JSON (*.json);;CSV (*.csv);;Text (*.txt)"

    def __init__(self, ctl, win):
        super().__init__(ctl, win, "Dictionary",
                         "Names, jargon and brands VoxFlow should always spell your way.")
        self.btn_import = IconButton("import", "Import words from a file")
        self.btn_export = IconButton("export", "Export your dictionary to a file")
        self.btn_add = Button("Add word", "primary", icon="plus")
        self.btn_import.setObjectName("dict_import")
        self.btn_export.setObjectName("dict_export")
        self.btn_add.setObjectName("dict_add")
        self.btn_import.clicked.connect(self.do_import)
        self.btn_export.clicked.connect(self.do_export)
        self.btn_add.clicked.connect(lambda: self.edit(None))
        for b in (self.btn_import, self.btn_export, self.btn_add):
            self.actions.addWidget(b)
        self.add_share_button(self.btn_add)

        self.inbox = Section("Suggestions")
        self.inbox.setObjectName("dict_suggestions")
        self.body.addWidget(self.inbox)
        self.search = SearchField("Search dictionary")
        self.search.setObjectName("dict_search")
        self._debounce = QTimer(self)                 # one rebuild per pause in typing, not one per keystroke
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(160)
        self._debounce.timeout.connect(self._search_changed)
        self.search.textChanged.connect(lambda _t: self._debounce.start())
        self.body.addWidget(self.search)
        self.entries = Section("Your words")
        self.entries.setObjectName("dict_entries")
        self.body.addWidget(self.entries)
        self._limit = self.PAGE_SIZE
        self.btn_more = Button("Show more words", "secondary")
        self.btn_more.setObjectName("dict_more")
        self.btn_more.clicked.connect(self._more)
        self.body.addWidget(self.btn_more, 0, Qt.AlignmentFlag.AlignHCenter)
        self.empty = EmptyState("book", "Your dictionary is empty")
        self.empty.setObjectName("dict_empty")
        self.empty.button.setObjectName("dict_empty_action")
        self.body.addWidget(self.empty)
        self.body.addStretch(1)
        self._items: list[dict] = []
        self._failed = ""

        ctl.dictionary_changed.connect(self.invalidate)
        ctl.suggestions_changed.connect(self.invalidate)

    PAGE_SIZE = 100                                   # rows built at a time; "Show more words" adds the next ones

    def _store(self):
        return self.ctl.dictionary

    def _search_changed(self) -> None:
        self._limit = self.PAGE_SIZE
        self._fill_entries()

    def _more(self) -> None:
        self._limit += self.PAGE_SIZE
        self._fill_entries()

    def refresh(self) -> None:
        self._failed = ""
        try:
            self._items = self.ctl.dictionary.list()
            sugg = self.ctl.dictionary.suggestions()
        except Exception as exc:
            self._failed = str(exc) or type(exc).__name__
            self._items, sugg = [], []
        self.inbox.clear()
        self.inbox.caption.setText(f"SUGGESTIONS  ·  {len(sugg)}")
        rows = []
        for s in sugg:
            heard = ", ".join(f"“{a}”" for a in s.get("aliases", []))
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            add = Button("Add", "secondary", icon="check", height=30)
            add.setObjectName("dict_suggestion_accept")
            add.setAccessibleName(f"Add {s.get('term', '')} to the dictionary")
            add.clicked.connect(lambda _c=False, sid=s["id"], term=s.get("term", ""): self.run(
                self.ctl.dictionary.accept_suggestion, sid, success=f"Added “{term}” to your dictionary."))
            no = IconButton("close", f"Dismiss the suggestion {s.get('term', '')}", size=30)
            no.setObjectName("dict_suggestion_dismiss")
            no.clicked.connect(lambda _c=False, sid=s["id"]: self.run(self.ctl.dictionary.dismiss_suggestion, sid))
            lay.addWidget(add)
            lay.addWidget(no)
            term = str(s.get("term", ""))
            rows.append(Row(term, f"VoxFlow heard {heard} and wrote {term}. Add it so it is always spelled this way."
                            if heard else f"VoxFlow wrote {term} in a recent dictation. Add it so it is always "
                                          "spelled this way.", box, icon="sparkle"))
        self.inbox.add_many(rows)
        self.inbox.setVisible(bool(sugg))
        self._fill_entries()

    def _fill_entries(self) -> None:
        q = self.search.text().strip().lower()
        matching = [e for e in self._items if not q or q in e["term"].lower()
                    or any(q in a.lower() for a in e.get("aliases", []))]
        shown = matching[:self._limit]
        self.entries.clear()
        rows = []
        for e in shown:
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            ed = IconButton("edit", f"Edit {e['term']}")
            ed.setObjectName("dict_edit")
            ed.clicked.connect(lambda _c=False, entry=e: self.edit(entry))
            rm = IconButton("trash", f"Remove {e['term']}", danger=True)
            rm.setObjectName("dict_remove")
            rm.clicked.connect(lambda _c=False, entry=e: self._remove(entry))
            lay.addWidget(ed)
            lay.addWidget(rm)
            aliases = e.get("aliases") or []
            sub = "Heard as: " + ", ".join(aliases) if aliases else "Spelling and capitalisation only"
            rows.append(Row(e["term"], sub, box))
        self.entries.add_many(rows)
        hidden = len(matching) - len(shown)
        self.btn_more.setText(f"Show more words ({hidden:,} not shown)")
        self.btn_more.setVisible(hidden > 0)
        has_any = bool(self._items)
        self.search.setVisible(len(self._items) > 6 or bool(q))
        self.entries.setVisible(bool(shown))
        self.entries.caption.setText(f"YOUR WORDS  ·  {len(self._items)}")
        self.btn_export.setEnabled(has_any)
        self.btn_share.setEnabled(has_any)
        self.empty.setVisible(not shown)
        if shown:
            return
        if self._failed:
            self.empty.show_state("error", "warning", "Your dictionary could not be loaded",
                                  f"{self._failed}. Nothing was changed on disk.", "Try again", self.refresh)
        elif has_any:
            self.empty.show_state("empty", "search", "No matches",
                                  f"No dictionary entry contains “{self.search.text().strip()}”.",
                                  "Clear search", self.search.clear)
        else:
            self.empty.show_state("empty", "book", "Your dictionary is empty",
                                  "Add names, product terms or acronyms that get misheard. VoxFlow will write them "
                                  "exactly as you spell them here.", "Add your first word", lambda: self.edit(None))

    def _remove(self, entry: dict) -> None:
        if not self.run(self.ctl.dictionary.remove, entry["id"]):
            return
        term, aliases = entry["term"], list(entry.get("aliases") or [])
        self.win.toast("success", f"Removed “{term}”.", "Undo",
                       lambda: self.run(self.ctl.dictionary.add, term, aliases))

    def edit(self, entry: dict | None) -> Modal:
        m = Modal(self.win.panel, "Edit word" if entry else "Add a word",
                  "VoxFlow will always write this exactly as you type it here.", name="dict_editor")
        term = LineEdit(entry["term"] if entry else "", "e.g. Kubernetes", name="Word or phrase")
        term.setObjectName("dict_term")
        field(m, "Word or phrase", term)
        aliases = LineEdit(", ".join(entry.get("aliases", [])) if entry else "", "e.g. cube are netties, kubernetties",
                           name="Sounds like")
        aliases.setObjectName("dict_aliases")
        field(m, "Sounds like (optional)", aliases, "Separate several mishearings with commas. Whenever VoxFlow hears "
                                                    "one of them it writes your word instead.")

        def save():
            try:
                if entry:
                    self.ctl.dictionary.update(entry["id"], term.text(), _split_aliases(aliases.text()))
                else:
                    self.ctl.dictionary.add(term.text(), _split_aliases(aliases.text()))
            except ValueError as exc:
                term.set_invalid(True)
                m.show_error(str(exc))
                return
            m.close_modal()
            self.win.toast("success", f"Saved “{term.text().strip()}”.")

        term.textChanged.connect(lambda _t: (term.set_invalid(False), m.show_error("")))
        m.add_buttons("Save", save)
        return self.win.open_modal(m, term)


# =========================================================================== Snippets
class SnippetsPage(Page, _ImportExport):
    noun = "snippets"
    file_stub = "voxflow-snippets"
    pack_kind = "snippets"
    pack_item = "snippet"

    def __init__(self, ctl, win):
        super().__init__(ctl, win, "Snippets", "Say a short phrase, get a longer text: addresses, links, sign-offs.")
        self.btn_import = IconButton("import", "Import snippets from a file")
        self.btn_export = IconButton("export", "Export your snippets to a file")
        self.btn_add = Button("New snippet", "primary", icon="plus")
        self.btn_import.setObjectName("snippets_import")
        self.btn_export.setObjectName("snippets_export")
        self.btn_add.setObjectName("snippets_add")
        self.btn_import.clicked.connect(self.do_import)
        self.btn_export.clicked.connect(self.do_export)
        self.btn_add.clicked.connect(lambda: self.edit(None))
        for b in (self.btn_import, self.btn_export, self.btn_add):
            self.actions.addWidget(b)
        self.add_share_button(self.btn_add)
        self.rows = Section("Your snippets")
        self.rows.setObjectName("snippets_list")
        self.body.addWidget(self.rows)
        self._limit = self.PAGE_SIZE
        self.btn_more = Button("Show more snippets", "secondary")
        self.btn_more.setObjectName("snippets_more")
        self.btn_more.clicked.connect(self._more)
        self.body.addWidget(self.btn_more, 0, Qt.AlignmentFlag.AlignHCenter)
        self.empty = EmptyState("snippet", "No snippets yet")
        self.empty.setObjectName("snippets_empty")
        self.empty.button.setObjectName("snippets_empty_action")
        self.body.addWidget(self.empty)
        self.body.addStretch(1)
        ctl.snippets_changed.connect(self.invalidate)

    PAGE_SIZE = 100                                   # rows built at a time; "Show more snippets" adds the next ones

    def _store(self):
        return self.ctl.snippets

    def _more(self) -> None:
        self._limit += self.PAGE_SIZE
        self.refresh()

    def refresh(self) -> None:
        failed = ""
        try:
            items = self.ctl.snippets.list()
        except Exception as exc:
            failed = str(exc) or type(exc).__name__
            items = []
        self.rows.clear()
        self.rows.caption.setText(f"YOUR SNIPPETS  ·  {len(items)}")
        hidden = max(0, len(items) - self._limit)
        self.btn_more.setText(f"Show more snippets ({hidden:,} not shown)")
        self.btn_more.setVisible(hidden > 0)
        rows = []
        for s in items[:self._limit]:
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            ed = IconButton("edit", f"Edit the snippet {s['trigger']}")
            ed.setObjectName("snippets_edit")
            ed.clicked.connect(lambda _c=False, entry=s: self.edit(entry))
            rm = IconButton("trash", f"Delete the snippet {s['trigger']}", danger=True)
            rm.setObjectName("snippets_remove")
            rm.clicked.connect(lambda _c=False, entry=s: self._remove(entry))
            lay.addWidget(ed)
            lay.addWidget(rm)
            flat = " ⏎ ".join(line.strip() for line in str(s.get("text", "")).splitlines() if line.strip())
            rows.append(Row(f"“{s['trigger']}”", flat if len(flat) <= 110 else flat[:107] + "…", box,
                            icon="snippet"))
        self.rows.add_many(rows)
        self.rows.setVisible(bool(items))
        self.empty.setVisible(not items)
        self.btn_export.setEnabled(bool(items))
        self.btn_share.setEnabled(bool(items))
        if failed:
            self.empty.show_state("error", "warning", "Your snippets could not be loaded",
                                  f"{failed}. Nothing was changed on disk.", "Try again", self.refresh)
        elif not items:
            self.empty.show_state("empty", "snippet", "No snippets yet",
                                  "Create a snippet, then dictate just its phrase, for example “my address”, "
                                  "or say “insert my address” in a sentence, and VoxFlow types the full text.",
                                  "Create a snippet",
                                  lambda: self.edit(None))

    def _remove(self, entry: dict) -> None:
        if not self.run(self.ctl.snippets.remove, entry["id"]):
            return
        trig, text = entry["trigger"], entry.get("text", "")
        self.win.toast("success", f"Deleted the snippet “{trig}”.", "Undo",
                       lambda: self.run(self.ctl.snippets.add, trig, text))

    def edit(self, entry: dict | None) -> Modal:
        m = Modal(self.win.panel, "Edit snippet" if entry else "New snippet",
                  "Dictate just the phrase, or say “insert” (or “paste”, “type”) before it inside a sentence, and "
                  "VoxFlow types the text. Said in an ordinary sentence without “insert”, the phrase stays as it is.",
                  width=500, name="snippet_editor")
        trig = LineEdit(entry["trigger"] if entry else "", "e.g. my address", name="Spoken phrase")
        trig.setObjectName("snippet_trigger")
        field(m, "When I say", trig)
        text = TextArea(entry.get("text", "") if entry else "", "The text to type, line breaks included",
                        name="Snippet text")
        text.setObjectName("snippet_text")
        text.setFixedHeight(150)
        field(m, "Insert this text", text)

        def save():
            try:
                if entry:
                    self.ctl.snippets.update(entry["id"], trig.text(), text.toPlainText())
                else:
                    self.ctl.snippets.add(trig.text(), text.toPlainText())
            except ValueError as exc:
                m.show_error(str(exc))
                return
            m.close_modal()
            self.win.toast("success", f"Saved the snippet “{trig.text().strip()}”.")

        trig.textChanged.connect(lambda _t: m.show_error(""))
        m.add_buttons("Save", save)
        return self.win.open_modal(m, trig)


# =========================================================================== Style
# Example dictations for the live preview. The preview text itself always comes from
# ``controller.preview_tone`` (the real tone pass), so it can only ever show what dictation really does.
_SAMPLES = {
    "personal": "Hey, are we still on for dinner tonight? I'm running about ten minutes late.",
    "work": "I don't think we can ship this on Friday. Let's sync tomorrow morning and decide.",
    "email": "Hi Sam, I am following up on the proposal. Please let me know if you have any questions.",
    "other": "Remember to pick up the dry cleaning and call the dentist about Thursday.",
}
# The same four situations in the languages that have their own clean-up rules; shown when that language is
# the dictation language, so the preview demonstrates the tone pass on text like the user's own.
_SAMPLES_BY_LANGUAGE = {
    "es": {"personal": "Oye, ¿seguimos quedando para cenar esta noche? Llego unos diez minutos tarde.",
           "work": "No creo que podamos publicarlo el viernes. Hablamos mañana por la mañana y decidimos.",
           "email": "Hola Sam, te escribo por la propuesta. Dime si tienes alguna pregunta.",
           "other": "Acuérdate de recoger la ropa de la tintorería y de llamar al dentista por lo del jueves."},
    "fr": {"personal": "Salut, on se retrouve toujours pour dîner ce soir ? J'aurai environ dix minutes de retard.",
           "work": "Je ne pense pas qu'on puisse livrer ça vendredi. On en reparle demain matin et on décide.",
           "email": "Bonjour Sam, je reviens vers vous au sujet de la proposition. N'hésitez pas si vous avez des questions.",
           "other": "Pense à récupérer les vêtements au pressing et à appeler le dentiste pour jeudi."},
    "de": {"personal": "Hey, bleibt es bei heute Abend zum Essen? Ich komme ungefähr zehn Minuten später.",
           "work": "Ich glaube nicht, dass wir das am Freitag ausliefern können. Lass uns morgen früh sprechen und entscheiden.",
           "email": "Hallo Sam, ich melde mich wegen des Angebots. Sag mir bitte Bescheid, wenn du Fragen hast.",
           "other": "Denk daran, die Sachen aus der Reinigung zu holen und den Zahnarzt wegen Donnerstag anzurufen."},
    "it": {"personal": "Ehi, ci vediamo ancora a cena stasera? Arrivo con una decina di minuti di ritardo.",
           "work": "Non credo che riusciremo a rilasciarlo venerdì. Sentiamoci domani mattina e decidiamo.",
           "email": "Ciao Sam, ti scrivo per la proposta. Fammi sapere se hai domande.",
           "other": "Ricordati di ritirare i vestiti in lavanderia e di chiamare il dentista per giovedì."},
    "pt": {"personal": "Olá, ainda vamos jantar hoje à noite? Chego uns dez minutos atrasado.",
           "work": "Acho que não conseguimos lançar isto na sexta-feira. Falamos amanhã de manhã e decidimos.",
           "email": "Olá Sam, escrevo por causa da proposta. Diz-me se tiveres alguma dúvida.",
           "other": "Lembra-te de ir buscar a roupa à lavandaria e de ligar ao dentista por causa de quinta-feira."},
    "nl": {"personal": "Hé, gaan we vanavond nog steeds samen eten? Ik ben ongeveer tien minuten later.",
           "work": "Ik denk niet dat we dit vrijdag kunnen opleveren. Laten we morgenochtend overleggen en beslissen.",
           "email": "Hallo Sam, ik kom nog even terug op het voorstel. Laat het me weten als je vragen hebt.",
           "other": "Vergeet niet de kleren bij de stomerij op te halen en de tandarts te bellen over donderdag."},
}
_CATEGORY_HINTS = {
    "personal": "WhatsApp, Discord, Telegram, iMessage-style chats",
    "work": "Slack, Teams, Notion, your editor",
    "email": "Outlook, Gmail, Thunderbird",
    "other": "Everything else: documents, notes, browsers",
}
# What the dictation tone pass does (voxflow.cleanup.apply_tone): it never rewords anything.
_TONE_HINTS = {"formal": "Nothing is changed: capitals and full punctuation.",
               "casual": "Same words and capitals. Only the full stop at the very end is left off.",
               "very_casual": "Same words. No full stop at the very end, and sentences start in lower case; "
                              "names and “I” keep their capital."}


class _ToneCard(Card):
    def __init__(self, page: "StylePage", key: str, label: str):
        super().__init__(padding=T.S4, spacing=10)
        self.page, self.key = page, key
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addWidget(Label(label, T.F_BODY_LG, "TEXT", T.W_SEMI, wrap=False))
        col.addWidget(Label(_CATEGORY_HINTS.get(key, ""), T.F_SMALL - 0.2, "TEXT_DIM"))
        self.body.addLayout(col)
        self.tone = Segmented(page.ctl.tones(), None, name=f"Tone for {label}")
        self.tone.setObjectName(f"style_tone_{key}")
        self.tone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tone.changed.connect(self._changed)
        self.body.addWidget(self.tone)
        self.preview = Label("", T.F_BODY_LG, "TEXT")
        self.preview.setObjectName(f"style_preview_{key}")
        self.preview.setContentsMargins(14, 10, 12, 10)
        self.preview.setMinimumHeight(64)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.body.addWidget(self.preview)
        self.hint = Label("", T.F_CAPTION, "TEXT_DIM")
        self.body.addWidget(self.hint)

    def _changed(self, tone) -> None:
        if self.page.run(self.page.ctl.styles.set_tone, self.key, tone) is None:
            self.set_tone(self.page.ctl.styles.tones().get(self.key))
        else:
            self._show(tone)

    def set_tone(self, tone) -> None:
        self.tone.set_value(tone)
        self._show(tone)

    def _show(self, tone) -> None:
        language = str(self.page.ctl.get("language"))
        samples = _SAMPLES_BY_LANGUAGE.get(language)
        sample = (samples or _SAMPLES).get(self.key, _SAMPLES["other"])
        try:        # always the real tone pass, so the preview can only show what dictation does
            text = self.page.ctl.preview_tone(sample, tone, language) if samples else \
                self.page.ctl.preview_tone(sample, tone)
        except Exception:
            text = sample
        self.preview.setText(text)
        hint = _TONE_HINTS.get(tone, "")
        if tone != "formal" and text == (sample[:-1] if sample.endswith(".") else sample) and tone == "very_casual":
            hint = "Same words. No full stop at the very end; this language keeps its capitals."
        self.hint.setText(hint)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = painter(self)
        g = QRectF(self.preview.geometry()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillPath(T.rounded(g, T.R_SM + 1), T.SUNK)
        p.setPen(T.STROKE)
        p.drawRoundedRect(g, T.R_SM + 1, T.R_SM + 1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.neon_gradient(g.left(), g.left() + 3))
        p.drawRoundedRect(QRectF(g.left() + 1, g.top() + 8, 2.5, g.height() - 16), 1.2, 1.2)


class StylePage(Page):
    def __init__(self, ctl, win):
        super().__init__(ctl, win, "Style", "VoxFlow adjusts capitals and the final full stop to suit the app you "
                                            "are dictating into. Your words are never changed.")
        self.cards: dict[str, _ToneCard] = {}
        self._grid = QGridLayout()
        self._grid.setSpacing(T.S3)
        self._cols = 2
        for key, label in list(ctl.categories()):
            self.cards[key] = _ToneCard(self, key, label)
        self._flow_cards(2)
        self.body.addLayout(self._grid)
        self.note = Label("In Command mode the same setting also guides how formal the text VoxFlow writes for you "
                          "sounds.", T.F_SMALL, "TEXT_DIM")
        self.note.setObjectName("style_command_note")
        self.note.setContentsMargins(4, 0, 0, 0)
        self.body.addWidget(self.note)

        self.app_search = SearchField("Search apps")
        self.app_search.setObjectName("style_app_search")
        self.app_search.setMaximumWidth(300)
        self.app_search.textChanged.connect(lambda _t: self.refresh())
        self.body.addWidget(self.app_search)
        self.apps = Section("Apps")
        self.apps.setObjectName("style_apps")
        self.body.addWidget(self.apps)
        self.body.addStretch(1)
        ctl.styles_changed.connect(self.invalidate)
        ctl.config_changed.connect(lambda key, _v: key == "language" and self.invalidate())

    def _flow_cards(self, cols: int) -> None:
        """Two tone cards per row, one per row in a narrow window (D-13)."""
        self._cols = cols
        for i, card in enumerate(self.cards.values()):
            self._grid.addWidget(card, i // cols, i % cols)
        for c in range(2):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cols = 1 if self.width() < 700 else 2
        if cols != self._cols:
            self._flow_cards(cols)

    def refresh(self) -> None:
        try:
            tones = self.ctl.styles.tones()
            rows = self.ctl.styles.app_categories()
        except Exception as exc:
            self.win.toast("error", f"Could not read your styles: {exc}")
            tones, rows = {}, []
        for key, card in self.cards.items():
            card.set_tone(tones.get(key, "formal"))
        cats = list(self.ctl.categories())
        short = {k: v.replace(" messages", "") for k, v in cats}
        self.apps.clear()

        add_box = QWidget()
        lay = QHBoxLayout(add_box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.new_exe = LineEdit("", "app.exe", name="App executable name")
        self.new_exe.setObjectName("style_new_exe")
        self.new_exe.setMaximumWidth(260)
        self.new_cat = ChoiceButton(name="Category for the new app")
        self.new_cat.setObjectName("style_new_category")
        self.new_cat.set_options([(k, short[k]) for k, _v in cats], cats[1][0] if len(cats) > 1 else None)
        self.new_cat.setFixedWidth(150)
        add = Button("Add app", "secondary", icon="plus")
        add.setObjectName("style_add_app")
        add.clicked.connect(self._add_app)
        self.new_exe.returnPressed.connect(self._add_app)
        lay.addWidget(self.new_exe, 1)
        lay.addWidget(self.new_cat)
        lay.addWidget(add)
        lay.addStretch(1)
        self.apps.add(Row("Add an app", "Type the program's file name, for example signal.exe, and choose which "
                                        "style it should use.", None, below=add_box))
        def own(row: dict) -> bool:
            """The user's own choice: an app they added, or a built-in app whose category they changed.
            (The store sets ``overridden`` only for built-in apps; an added app is simply not ``builtin``.)"""
            return bool(row.get("overridden")) or not row.get("builtin")

        query = self.app_search.text().strip().lower()
        for r in sorted(rows, key=lambda r: not own(r)):       # your own choices first (stable sort)
            exe = str(r.get("exe", ""))
            if query and query not in exe.lower() and query not in app_label(exe).lower():
                continue
            box = QWidget()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(4)
            pick = ChoiceButton(name=f"Category for {app_label(exe)}")
            pick.setObjectName("style_app_category")
            pick.set_options([(k, short[k]) for k, _v in cats], r.get("category"))
            pick.setFixedWidth(150)
            pick.chosen.connect(lambda cat, e=exe: self.run(self.ctl.styles.set_app_category, e, cat))
            if own(r):
                builtin = bool(r.get("builtin"))
                undo = IconButton("undo" if builtin else "trash",
                                  f"Use the built-in category for {app_label(exe)}" if builtin else
                                  f"Remove {app_label(exe)}", danger=not builtin)
                undo.setObjectName("style_app_reset")
                undo.clicked.connect(lambda _c=False, e=exe: self.run(self.ctl.styles.remove_app_category, e))
                bl.addWidget(undo)
            bl.addWidget(pick)
            # only the exceptions are labelled: "built-in" on every row was noise (D-18)
            self.apps.add(Row(app_label(exe) or exe, f"{exe}  ·  your choice" if own(r) else exe, box))

    def _add_app(self) -> None:
        exe = self.new_exe.text().strip()
        if not exe:
            self.new_exe.set_invalid(True)
            self.win.toast("warning", "Type the app's file name first, for example signal.exe.")
            return
        if self.run(self.ctl.styles.set_app_category, exe, self.new_cat.value) is not None:
            self.win.toast("success", f"{app_label(exe) or exe} added.")


# =========================================================================== Settings
_HOTKEYS = (("hotkey", "Dictation", "Hold to talk, release to paste.", False),
            ("command_hotkey", "Command mode", "Hold and say what to do with the selected text, like “make this "
                                               "more formal”. With nothing selected it writes for you.", True),
            ("paste_last_hotkey", "Paste last transcript", "Pastes your most recent dictation again.", True),
            ("undo_hotkey", "Undo last paste", "Removes the text VoxFlow just pasted.", True))
_HOTKEY_WORDS = "hotkey shortcut keyboard keys"           # extra words the settings search matches (UI-04)
_DEVICE_WORDS = "gpu cuda cpu graphics card nvidia amd intel device speed"
_RETENTION = ((0, "Forever"), (1, "1 day"), (7, "7 days"), (30, "30 days"), (90, "90 days"), (365, "1 year"))
_MODEL_LABELS = {"pending": ("Waiting", "TEXT_DIM"), "downloading": ("Downloading", "AMBER"),
                 "loading": ("Loading", "AMBER"), "ready": ("Ready", "GREEN"), "degraded": ("Limited", "AMBER"),
                 "failed": ("Failed", "RED")}


class _ModelRow(Row):
    def __init__(self, page: "SettingsPage", name: str, title: str):
        self.page, self.name = page, name
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.state = Label("", T.F_SMALL, "TEXT_DIM", T.W_SEMI, wrap=False)
        self.state.setObjectName(f"settings_model_state_{name}")
        self.retry = Button("Try again", "secondary", icon="refresh", height=30)
        self.retry.setObjectName(f"settings_model_retry_{name}")
        self.retry.clicked.connect(lambda: page.run(page.ctl.reload_models))
        lay.addWidget(self.state)
        lay.addWidget(self.retry)
        self.bar = ProgressBar()
        super().__init__(title, "", box, icon="sparkle" if name == "cleanup" else "mic", below=self.bar)
        self.setObjectName(f"settings_model_{name}")

    def set_status(self, st: dict) -> None:
        state = str(st.get("state", "pending"))
        text, role = _MODEL_LABELS.get(state, (state.title(), "TEXT_DIM"))
        prog = st.get("progress")
        if state == "downloading" and isinstance(prog, (int, float)):
            text = f"Downloading {int(prog * 100)}%"
        self.state.set_role(role)
        self.state.setText(text)
        self.set_sub(str(st.get("detail", "")), "RED" if state == "failed" else "TEXT_DIM")
        self.retry.setVisible(state in ("failed", "degraded"))
        busy = state in ("downloading", "loading")
        self.bar.setVisible(busy)
        if busy:
            self.bar.set_value(prog if (state == "downloading" and isinstance(prog, (int, float))) else None)


_DEVICE_LABELS = {"cuda": "NVIDIA CUDA", "webgpu": "WebGPU", "vulkan": "Vulkan", "cpu": "CPU"}
_DEVICE_MODELS = (("asr", "asr_device", "Speech recognition device"),
                  ("cleanup", "cleanup_device", "Language model device"))


def _device_name(backend: str, ms=None) -> str:
    """ "NVIDIA CUDA · 29 ms", or just the name when it was not measured."""
    name = _DEVICE_LABELS.get(backend, backend)
    return f"{name} · {ms:,.0f} ms" if isinstance(ms, (int, float)) else name


def _device_options(view: dict) -> list[tuple]:
    """The device menu of one model: Automatic (with what it picks here), then every back-end."""
    times = {o.get("id"): o.get("ms") for o in view.get("options", [])}
    auto = view.get("auto")
    ms = times.get(auto)
    if auto and isinstance(ms, (int, float)):
        label = f"{_DEVICE_LABELS.get(auto, auto)}, {ms:,.0f} ms"
        head = f"Automatic (fastest here: {label})" if auto == view.get("fastest") else f"Automatic ({label})"
    elif auto:
        head = f"Automatic ({_DEVICE_LABELS.get(auto, auto)})"
    else:
        head = "Automatic"
    out = [("auto", head, "", True)]
    for o in view.get("options", []):
        available = bool(o.get("available", True))
        note = (o.get("reason") or "Not available here") if not available else (
            "Failed in the speed check" if o.get("failed") else "")
        out.append((o.get("id"), _device_name(o.get("id"), o.get("ms")), note, available))
    return out


def _device_sub(name: str, view: dict, forced: str) -> tuple[str, str]:
    current = view.get("current", "")
    now = f"Now running on {_DEVICE_LABELS[current]}." if current in _DEVICE_LABELS else ""
    if forced:
        return f"Set by VOXFLOW_DEVICE={forced}. {now}".strip(), "TEXT_DIM"
    if name == "cleanup":
        if view.get("restart") and view.get("next") in _DEVICE_LABELS:
            return f"{now} Moves to {_DEVICE_LABELS[view['next']]} when VoxFlow restarts.".strip(), "AMBER"
        return f"{now} A change takes effect when VoxFlow restarts.".strip(), "TEXT_DIM"
    return f"{now} Automatic uses the fastest device measured on this PC.".strip(), "TEXT_DIM"


def _speed_sub(check: dict) -> str:
    state = check.get("state", "idle")
    if not check.get("available", True):
        return "Needs the real models, which are not in use."
    if state == "running":
        total, done = int(check.get("total") or 0), int(check.get("done") or 0)
        if not total:
            return "Measuring…"
        what = "speech recognition" if check.get("model") == "asr" else "the language model"
        return (f"Measuring {what} on {_DEVICE_LABELS.get(check.get('backend'), check.get('backend'))} "
                f"({min(done + 1, total)} of {total})…")
    if state == "waiting":
        return "Starts when VoxFlow is idle and stops whenever you dictate."
    when = check.get("measured_at")
    if isinstance(when, (int, float)):
        day = _dt.datetime.fromtimestamp(when).strftime("%d %b %Y").lstrip("0")
        return f"Measured on this PC on {day}. Automatic picks the fastest device."
    return "Times each device once in the background, so Automatic can pick the fastest one on this PC."


class SettingsPage(Page):
    def __init__(self, ctl, win):
        super().__init__(ctl, win, "Settings")
        self._sync: dict[str, object] = {}        # config key -> function(value) that updates its control
        self.recorders: dict[str, HotkeyRecorder] = {}
        self._capturing: HotkeyRecorder | None = None
        self._testing = False
        self._arranging = False
        self._tab_fixed = False

        # ---- hotkeys
        sec = Section("Hotkeys")
        for key, title, sub, clearable in _HOTKEYS:
            rec = HotkeyRecorder(ctl.get(key), name=f"{title} hotkey", allow_clear=clearable,
                                 validator=lambda combo, k=key: ctl.validate_hotkey(k, combo) or None)
            rec.setObjectName(f"settings_{key}_recorder" if key != "hotkey" else "settings_hotkey_recorder")
            rec.use_qt_keys = bool(ctl.is_demo)          # the real app records through its global hook
            rec.capture_requested.connect(lambda r=rec: self._begin_capture(r))
            rec.capture_cancelled.connect(lambda r=rec: self._end_capture(r))
            rec.changed.connect(lambda combo, k=key, r=rec: self._hotkey_changed(k, combo, r))
            self.recorders[key] = rec
            self._sync[key] = rec.set_combo
            sec.add(Row(title, sub, rec, icon="keyboard" if key == "hotkey" else None)).keywords = _HOTKEY_WORDS
        sec.add(self._toggle("double_tap_lock", "Hands-free lock",
                             "Double-tap the dictation hotkey to keep listening; tap once more to finish."))
        sec._rows[-1].keywords = _HOTKEY_WORDS
        self.body.addWidget(sec)

        # ---- microphone
        sec = Section("Microphone")
        mic_box = QWidget()
        ml = QHBoxLayout(mic_box)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)
        self.mic = ChoiceButton(name="Microphone")
        self.mic.setObjectName("settings_mic")
        self.mic.setFixedWidth(280)
        self.mic.chosen.connect(lambda v: self._set_or_resync("mic_device", v))
        self.btn_mic_refresh = IconButton("refresh", "Look for microphones again")
        self.btn_mic_refresh.setObjectName("settings_mic_refresh")
        self.btn_mic_refresh.clicked.connect(lambda: self.run(ctl.refresh_microphones))
        ml.addWidget(self.mic)
        ml.addWidget(self.btn_mic_refresh)
        self._sync["mic_device"] = lambda _v: self._fill_mics(ctl.microphones())
        sec.add(Row("Input device", "The microphone VoxFlow listens to.", mic_box, icon="mic"))
        test_box = QWidget()
        tl = QHBoxLayout(test_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)
        self.meter = LevelMeter(None)                # fed only while the user is running the mic test
        self.meter.setObjectName("settings_mic_meter")
        self.meter.setFixedWidth(200)
        self.btn_test = Button("Test", "secondary", height=30)
        self.btn_test.setObjectName("settings_mic_test")
        self.btn_test.clicked.connect(lambda: self._set_testing(not self._testing))
        tl.addWidget(self.meter)
        tl.addWidget(self.btn_test)
        self.test_row = sec.add(Row("Input level", "Press Test and speak: the bar should move.", test_box))
        sec.add(self._toggle("whisper_mode", "Whisper mode",
                             "Boosts quiet speech so you can dictate under your breath."))
        self.body.addWidget(sec)

        # ---- dictation
        sec = Section("Dictation")
        self.language = ChoiceButton(name="Language", fallback="Auto-detect")
        self.language.setObjectName("settings_language")
        self.language.setFixedWidth(200)
        self.language.set_options(list(ctl.languages()), ctl.get("language"))
        self.language.chosen.connect(lambda v: self._set_or_resync("language", v))
        self._sync["language"] = self._sync_language
        self.language_row = sec.add(Row("Language", "", self.language, icon="globe"))
        self.language_row.setObjectName("settings_language_row")
        self.language_row.sub.setObjectName("settings_language_hint")
        sec.add(self._toggle("cleanup_enabled", "Smart cleanup",
                             "A local AI fixes punctuation and fillers. Off = paste exactly what was heard."))
        sec.add(self._toggle("smart_formatting", "Smart formatting",
                             "Writes spoken numbers, dates, emails, links and lists the way you would type them."))
        sec.add(self._toggle("auto_learn", "Learn new words",
                             "Suggests dictionary entries for names VoxFlow wrote differently from what it heard "
                             "(“get hub” written as GitHub), so they are always spelled that way. Paused while "
                             "Keep history is off."))
        self.body.addWidget(sec)

        # ---- sound
        sec = Section("Sound")
        snd_box = QWidget()
        sl = QHBoxLayout(snd_box)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        self.btn_preview = Button("Preview", "ghost", icon="play", height=30)
        self.btn_preview.setObjectName("settings_sound_preview")
        self.btn_preview.clicked.connect(lambda: self.run(ctl.play_sound, "start"))
        self.t_sounds = Toggle(bool(ctl.get("sounds_enabled")), name="Sound cues")
        self.t_sounds.setObjectName("settings_sounds_enabled")
        self.t_sounds.toggled.connect(lambda v: self._set_or_resync("sounds_enabled", bool(v)))
        self._sync["sounds_enabled"] = self._sync_sounds
        sl.addWidget(self.btn_preview)
        sl.addWidget(self.t_sounds)
        sec.add(Row("Sound cues", "Soft tones when listening starts, stops or something goes wrong.", snd_box,
                    icon="sound"))
        vol_box = QWidget()
        vl = QHBoxLayout(vol_box)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)
        self.volume = Slider(0, 100, int(round(float(ctl.get("sound_volume")) * 100)), name="Sound volume")
        self.volume.setObjectName("settings_sound_volume")
        self.volume.setFixedWidth(200)
        self.vol_label = Label("", T.F_SMALL, "TEXT_DIM", wrap=False)
        self.vol_label.setFixedWidth(40)
        self.vol_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.volume.valueChanged.connect(self._volume_moved)
        self.volume.sliderReleased.connect(lambda: self.run(ctl.play_sound, "start"))
        self._vol_timer = QTimer(self)
        self._vol_timer.setSingleShot(True)
        self._vol_timer.setInterval(180)
        self._vol_timer.timeout.connect(lambda: self._set_or_resync("sound_volume", self.volume.value() / 100.0))
        self._sync["sound_volume"] = self._sync_volume
        vl.addWidget(self.volume)
        vl.addWidget(self.vol_label)
        self.vol_row = sec.add(Row("Volume", "", vol_box))
        sec.add(self._toggle("mute_media", "Mute other audio while dictating",
                             "Silences music and videos while you speak, then brings them back."))
        self.body.addWidget(sec)

        # ---- appearance
        sec = Section("Appearance")
        self.theme = Segmented([("system", "System"), ("light", "Light"), ("dark", "Dark")], ctl.get("theme"),
                               name="Theme")
        self.theme.setObjectName("settings_theme")
        self.theme.changed.connect(lambda v: self._set_or_resync("theme", v))
        self._sync["theme"] = self.theme.set_value
        sec.add(Row("Theme", "System follows your Windows light or dark mode.", self.theme, icon="moon"))
        sec.add(self._toggle("reduced_motion", "Reduce motion", "Turns off decorative animation everywhere. Always "
                                                                "on while Windows' Animation effects are off."))
        sec.add(self._toggle("show_idle_pill", "Show the Flow bar when idle",
                             "Keeps a small pill on screen so you know VoxFlow is ready."))
        sec.add(self._toggle("live_preview", "Live preview", "Show the words in the Flow bar while you speak. The "
                                                             "pasted text is still tidied up after you release the key."))
        pos_box = QWidget()
        pl = QHBoxLayout(pos_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)
        self.btn_move = Button("Move…", "secondary", height=30)
        self.btn_move.setObjectName("settings_overlay_move")
        self.btn_move.clicked.connect(lambda: self._set_arranging(not self._arranging))
        self.btn_pos_reset = Button("Reset", "ghost", height=30)
        self.btn_pos_reset.setObjectName("settings_overlay_reset")
        self.btn_pos_reset.clicked.connect(lambda: (self._set_arranging(False), self.setting("overlay_position", None)))
        pl.addWidget(self.btn_pos_reset)
        pl.addWidget(self.btn_move)
        self.pos_row = sec.add(Row("Flow bar position", "", pos_box))
        self._sync["overlay_position"] = self._sync_position
        self.body.addWidget(sec)

        # ---- history & privacy
        sec = Section("History and privacy")
        row = sec.add(self._toggle("save_history", "Keep history",
                                   "Saves your transcripts on this PC so you can search, copy and paste them again. "
                                   "When it is off, no transcript is written to disk and no words are learned "
                                   "from your dictations.", icon="history"))
        row.findChild(Toggle, "settings_save_history").toggled.connect(self._history_toggled)   # after the setting
        self.retention = ChoiceButton(name="Keep transcripts for")
        self.retention.setObjectName("settings_retention")
        self.retention.setFixedWidth(160)
        self.retention.chosen.connect(self._retention_chosen)
        self._sync["history_retention_days"] = self._sync_retention
        self.retention_row = sec.add(Row("Keep transcripts for", "", self.retention))
        self._sync["history_limit"] = self._sync_history_limit
        self._sync_history_limit(self.ctl.get("history_limit"))
        sec.add(self._toggle("restore_clipboard", "Restore my clipboard",
                             "Puts back whatever you had copied before VoxFlow pasted."))
        self.body.addWidget(sec)

        # ---- backup and sync
        sec = Section("Backup and sync")
        sec.setObjectName("settings_backup_section")
        self.btn_backup_export = Button("Export…", "secondary", icon="export", height=30)
        self.btn_backup_export.setObjectName("settings_backup_export")
        self.btn_backup_export.clicked.connect(self.export_backup)
        sec.add(Row("Back up everything", "One file with your settings, dictionary, snippets and styles, and "
                                          "your history if you tick it. Keep it somewhere safe.",
                    self.btn_backup_export, icon="archive"))
        self.btn_backup_import = Button("Import…", "secondary", icon="import", height=30)
        self.btn_backup_import.setObjectName("settings_backup_import")
        self.btn_backup_import.clicked.connect(self.import_backup)
        sec.add(Row("Restore from a backup", "You choose whether to merge it into this profile or replace the "
                                             "profile with it.", self.btn_backup_import))
        sync_box = QWidget()
        sl2 = QHBoxLayout(sync_box)
        sl2.setContentsMargins(0, 0, 0, 0)
        sl2.setSpacing(6)
        self.btn_sync_off = Button("Turn off", "ghost", height=30)
        self.btn_sync_off.setObjectName("settings_sync_off")
        self.btn_sync_off.clicked.connect(lambda: self._set_or_resync("sync_folder", ""))
        self.btn_sync_now = Button("Sync now", "secondary", icon="sync", height=30)
        self.btn_sync_now.setObjectName("settings_sync_now")
        self.btn_sync_now.clicked.connect(lambda: self.run(ctl.sync_now))
        self.btn_sync_choose = Button("Choose folder…", "secondary", icon="folder", height=30)
        self.btn_sync_choose.setObjectName("settings_sync_choose")
        self.btn_sync_choose.clicked.connect(self.choose_sync_folder)
        for b in (self.btn_sync_off, self.btn_sync_now, self.btn_sync_choose):
            sl2.addWidget(b)
        self.sync_folder = Label("", T.F_SMALL, "TEXT", wrap=False, selectable=True)
        self.sync_folder.setObjectName("settings_sync_folder")
        self.sync_row = sec.add(Row("Sync folder", "", sync_box, icon="sync", below=self.sync_folder))
        self.sync_row.setObjectName("settings_sync_row")
        self.sync_row.sub.setObjectName("settings_sync_status")
        self._sync_timer = QTimer(self)                 # keeps "synced 4 min ago" honest while the page is open
        self._sync_timer.setInterval(30000)
        self._sync_timer.timeout.connect(self._show_sync)
        self.body.addWidget(sec)
        self._sync["sync_folder"] = lambda _v: self._show_sync()
        ctl.sync_status_changed.connect(self._show_sync)

        # ---- system
        sec = Section("System")
        sec.add(self._toggle("launch_at_login", "Launch at login", "Start VoxFlow when you sign in to Windows.",
                             icon="power")).keywords = "startup autostart boot windows sign in"
        sec.add(self._toggle("start_minimized", "Start in the tray", "Open quietly without showing this window.")
                ).keywords = "startup minimised minimized hidden"
        self.body.addWidget(sec)

        # ---- updates
        self.updates = UpdatesSection(self)
        self.body.addWidget(self.updates)

        # ---- models
        sec = Section("On-device models")
        self.models = {"asr": _ModelRow(self, "asr", "Speech recognition"),
                       "cleanup": _ModelRow(self, "cleanup", "Smart cleanup and Command mode")}
        self.devices: dict[str, ChoiceButton] = {}
        self.device_rows: dict[str, Row] = {}
        for name, key, title in _DEVICE_MODELS:
            sec.add(self.models[name]).keywords = "model download ai"
            pick = ChoiceButton(name=title, fallback="Automatic")
            pick.setObjectName(f"settings_device_{name}")
            pick.setFixedWidth(350)
            pick.chosen.connect(lambda v, k=key: self._set_or_resync(k, v))
            self._sync[key] = lambda _v: self._show_devices()
            self.devices[name] = pick
            self.device_rows[name] = sec.add(Row(title, "", pick))
            self.device_rows[name].keywords = _DEVICE_WORDS
            self.device_rows[name].setObjectName(f"settings_device_{name}_row")
        self.btn_measure = Button("Measure now", "secondary", icon="refresh", height=30)
        self.btn_measure.setObjectName("settings_speed_check")
        self.btn_measure.clicked.connect(lambda: self.run(ctl.measure_devices))
        self.speed_bar = ProgressBar()
        self.speed_bar.hide()
        self.speed_row = sec.add(Row("Speed check", "", self.btn_measure, below=self.speed_bar))
        self.speed_row.keywords = _DEVICE_WORDS + " benchmark performance"
        self.speed_row.setObjectName("settings_speed_row")
        self.body.addWidget(sec)
        ctl.devices_changed.connect(self._show_devices)

        # ---- support
        sec = Section("Support")
        self.btn_onboarding = Button("Open", "secondary", height=30)
        self.btn_onboarding.setObjectName("settings_onboarding")
        self.btn_onboarding.clicked.connect(win.onboarding_requested.emit)
        sec.add(Row("Setup guide", "Walk through the microphone check and hotkey practice again.", self.btn_onboarding,
                    icon="sparkle"))
        box = QWidget()
        fl = QHBoxLayout(box)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        self.btn_data = Button("Data folder", "secondary", icon="folder", height=30)
        self.btn_logs = Button("Logs", "secondary", icon="folder", height=30)
        self.btn_data.setObjectName("settings_open_data")
        self.btn_logs.setObjectName("settings_open_logs")
        self.btn_data.clicked.connect(lambda: self.run(ctl.open_data_folder))
        self.btn_logs.clicked.connect(lambda: self.run(ctl.open_log_folder))
        fl.addWidget(self.btn_data)
        fl.addWidget(self.btn_logs)
        sec.add(Row("Files", "Settings, history and logs live in one folder on this PC.", box, icon="folder"))
        self.btn_reset = Button("Reset…", "danger", height=30)
        self.btn_reset.setObjectName("settings_reset")
        self.btn_reset.clicked.connect(self._confirm_reset)
        sec.add(Row("Reset settings", "Back to the defaults. Your history, dictionary and snippets are kept.",
                    self.btn_reset, icon="refresh"))
        self.body.addWidget(sec)

        # ---- about
        sec = Section("About")
        self.about = Row("About VoxFlow", "Fully local dictation. No cloud, no account, no telemetry: your "
                                                 "voice never leaves this PC.", icon="info")
        self.about.setObjectName("settings_about")
        sec.add(self.about)
        self.body.addWidget(sec)
        self.body.addStretch(1)
        self._build_finder()

        ctl.config_changed.connect(self._on_config)
        ctl.hotkey_captured.connect(self._on_captured)
        ctl.microphones_changed.connect(self._fill_mics)
        ctl.model_status_changed.connect(self._on_model)

    # ---- find / jump (D-07)
    def _sections(self) -> list[Section]:
        return [self.body.itemAt(i).widget() for i in range(self.body.count())
                if isinstance(self.body.itemAt(i).widget(), Section)]

    def _build_finder(self) -> None:
        """Search box (filters rows by title and description) and a "Jump to" menu above the long page."""
        self.find = SearchField("Search settings")
        self.find.setObjectName("settings_search")
        self.find.setMaximumWidth(300)
        self.find.textChanged.connect(self._filter)
        self.jump = ChoiceButton(name="Jump to section", fallback="Jump to section")
        self.jump.setObjectName("settings_jump")
        self.jump.setFixedWidth(190)
        self.jump.set_options([(i, sec.caption.text().capitalize()) for i, sec in enumerate(self._sections())], None)
        self.jump.chosen.connect(self._jump_to)
        row = QHBoxLayout()
        row.setSpacing(T.S2)
        row.setContentsMargins(0, 8, 0, 0)
        row.addWidget(self.find, 1)
        row.addStretch(1)
        row.addWidget(self.jump)
        self.head_col.addLayout(row)
        self.no_match = Label("No setting matches that search.", T.F_BODY, "TEXT_DIM")
        self.no_match.hide()
        self.body.insertWidget(0, self.no_match)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._tab_fixed:            # once the page sits in the window: moving it there rebuilds its focus chain
            self._tab_fixed = True
            before = self.recorders["hotkey"].btn.previousInFocusChain()
            while before.focusPolicy() == Qt.FocusPolicy.NoFocus:   # setTabOrder ignores widgets without focus
                before = before.previousInFocusChain()
            QWidget.setTabOrder(before, self.find)          # built last but first on screen: Tab reaches the
            QWidget.setTabOrder(self.find, self.jump)       # search and the jump menu before the rows

    _SECTION_ALIASES = {"microphone": "Microphone", "models": "On-device models", "hotkeys": "Hotkeys",
                        "privacy": "History and privacy", "sync": "Backup and sync", "updates": "Updates"}

    def open_update(self) -> Modal | None:
        """The "new version" dialog, when there is a newer version (else a toast saying VoxFlow is up to date)."""
        info = self.ctl.update_info()
        if not info.get("available"):
            self.win.toast("info", "VoxFlow is up to date.")
            return None
        m = update_dialog(self.win, self.ctl, info)
        return self.win.open_modal(m, focus=m.btn_default)

    def show_section(self, name: str) -> None:
        """Scroll to the section called ``name`` (an alias such as ``"models"`` or its caption); no-op if unknown."""
        want = self._SECTION_ALIASES.get(str(name).lower(), str(name)).lower()
        for i, sec in enumerate(self._sections()):
            if sec.caption.text().lower() == want:
                self.find.setText("")
                QTimer.singleShot(0, lambda i=i: self._jump_to(i))    # after the page has been laid out
                return

    def _jump_to(self, index) -> None:
        secs = self._sections()
        if 0 <= int(index) < len(secs):
            self.scroll.verticalScrollBar().setValue(max(0, secs[int(index)].y() - 4))
            self.jump.set_value(None)

    def _filter(self, text: str) -> None:
        q = text.strip().lower()
        any_shown = False
        for sec in self._sections():
            shown = 0
            whole = bool(q) and q in sec.caption.text().lower()      # "privacy": the whole section
            for row in sec._rows:
                hit = not q or whole or any(q in t.lower() for t in (row.title.text(), row.sub.text(),
                                                                     getattr(row, "keywords", "")))
                row.setVisible(hit)
                shown += hit
            sec.setVisible(bool(shown))
            any_shown = any_shown or bool(shown)
        self.no_match.setVisible(bool(q) and not any_shown)
        if not q:
            self.updates.refresh()                   # rows that only apply to one update mode hide again

    # ---- builders
    def _toggle(self, key: str, title: str, sub: str, icon: str | None = None) -> Row:
        t = Toggle(bool(self.ctl.get(key)), name=title)
        t.setObjectName(f"settings_{key}")
        t.toggled.connect(lambda v, k=key: self._set_or_resync(k, bool(v)))
        self._sync[key] = t.set_checked_silent
        return Row(title, sub, t, icon=icon)

    def _set_or_resync(self, key: str, value) -> None:
        """Apply a setting; if the controller rejects it, snap the control back to the real value."""
        if not self.setting(key, value):
            self._on_config(key, self.ctl.get(key))

    def _history_toggled(self, on: bool) -> None:
        """History was just switched off: offer to delete what is already saved (keeping it is the default)."""
        if on or self.ctl.get("save_history"):
            return                                       # switched on, or the controller refused to switch it off
        try:
            n = int(self.ctl.history.count())
        except Exception:  # noqa: BLE001 - the offer is a convenience; History > Clear all still works
            return
        if n <= 0:
            return
        self.win.open_modal(confirm(
            self.win.panel, "Delete your saved history too?",
            f"New dictations will not be saved from now on. {n:,} transcript{'s are' if n != 1 else ' is'} still "
            "stored on this PC from before. Delete them now? Your stats are kept. This cannot be undone.",
            f"Delete {n:,} transcript{'s' if n != 1 else ''}",
            lambda: self.run(self.ctl.history.clear, success="Saved history deleted."),
            name="history_off_confirm", cancel="Keep them"))

    # ---- sync from controller
    def _on_config(self, key: str, value) -> None:
        fn = self._sync.get(key)
        if fn is not None:
            fn(value)

    def _sync_language(self, code) -> None:
        """Select the language and say, in one line, how much clean-up dictation in it gets."""
        self.language.set_value(code)
        try:
            level = self.ctl.language_support(code)
        except Exception:
            level = "basic"
        names = dict(self.ctl.languages())
        full = [name for c, name in self.ctl.languages() if c != "auto" and self.ctl.language_support(c) == "full"]
        full_list = ", ".join(full[:-1]) + " and " + full[-1] if len(full) > 1 else "".join(full)
        name = names.get(code, "This language")
        if level == "auto":
            text = (f"Picks the rules from what you say. Full clean-up in {len(full)} languages, "
                    f"basic in the other {len(names) - 1 - len(full)}.")
        elif level == "full":
            text = f"{name}: full clean-up, including spoken punctuation, self-corrections and numbers."
        else:
            text = f"{name}: basic clean-up. Spoken commands like “new line” work in {full_list}."
        self.language_row.set_sub(text, "TEXT_DIM")

    def _sync_sounds(self, on) -> None:
        self.t_sounds.set_checked_silent(bool(on))
        self.btn_preview.setEnabled(bool(on))
        self.volume.setEnabled(bool(on))

    def _sync_volume(self, v) -> None:
        if not self.volume.isSliderDown() and not self._vol_timer.isActive():
            self.volume.blockSignals(True)
            self.volume.setValue(int(round(float(v) * 100)))
            self.volume.blockSignals(False)
        self.vol_label.setText(f"{self.volume.value()}%")

    def _volume_moved(self, v: int) -> None:
        self.vol_label.setText(f"{v}%")
        self._vol_timer.start()

    def _retention_chosen(self, value) -> None:
        """A shorter "Keep transcripts for" deletes the older transcripts at once: ask first, like Clear all."""
        days = int(value)
        try:
            n = int(self.ctl.history.count_older_than(days))
        except Exception:  # noqa: BLE001 - not knowing how many is no reason to skip the question
            n = -1
        if n == 0:
            self._set_or_resync("history_retention_days", days)
            return
        self._sync_retention(self.ctl.get("history_retention_days"))    # the old choice stays until confirmed
        label = dict(_RETENTION).get(days, f"{days} days")
        what = f"{n:,} transcript{'s' if n != 1 else ''}" if n > 0 else "the transcripts"
        self.win.open_modal(confirm(
            self.win.panel, f"Delete {what} older than {label}?",
            f"Keeping transcripts for {label} deletes {what} saved before then from this PC now. Your stats are "
            "kept. This cannot be undone.", f"Delete {what}",
            lambda: self._set_or_resync("history_retention_days", days), name="retention_confirm",
            cancel="Keep them"))

    def _sync_history_limit(self, limit) -> None:
        self.retention_row.set_sub(f"Older transcripts are deleted automatically. Even with Forever, at most "
                                   f"{int(limit):,} are kept: the oldest go first.")

    def _sync_retention(self, days) -> None:
        opts = list(_RETENTION)
        if int(days) not in [d for d, _t in opts]:
            opts.append((int(days), f"{int(days)} days"))
        self.retention.set_options(opts, int(days))

    def _sync_position(self, pos) -> None:
        custom = bool(pos)
        self.btn_pos_reset.setVisible(custom)
        if self._arranging:
            self.pos_row.set_sub("Drag the Flow bar to where you want it, then press Done.", "CYAN")
        else:
            self.pos_row.set_sub("Custom position, remembered between launches." if custom else
                                 "Bottom centre of the screen you are working on.")

    def _fill_mics(self, mics) -> None:
        cur = self.ctl.get("mic_device")
        opts = [(m.get("id"), str(m.get("name", "Microphone"))) for m in (mics or [])]
        if not opts:
            opts = [(None, "System default")]
        if cur is not None and cur not in [o[0] for o in opts]:
            opts.append((cur, f"{cur} (not connected)"))
        self.mic.set_options(opts, cur)

    def _on_model(self, name: str, status: dict) -> None:
        if name in self.models:
            self.models[name].set_status(status or {})

    def _show_devices(self, info=None) -> None:
        """The device menus, where each model runs now and the speed check's state (``ctl.device_info()``)."""
        info = info if isinstance(info, dict) else self.ctl.device_info()
        forced = info.get("forced", "")
        for name, pick in self.devices.items():
            view = info.get(name, {})
            pick.set_options(_device_options(view), view.get("setting", "auto"))
            pick.setEnabled(not forced)
            self.device_rows[name].set_sub(*_device_sub(name, view, forced))
        check = info.get("check", {})
        running = check.get("state") == "running"
        self.speed_row.set_sub(_speed_sub(check))
        self.btn_measure.setText("Measuring…" if running else
                                 "Measure again" if check.get("measured_at") else "Measure now")
        self.btn_measure.setEnabled(bool(check.get("available", True)) and not running)
        self.speed_bar.setVisible(running)
        if running:
            total = int(check.get("total") or 0)
            self.speed_bar.set_value(int(check.get("done") or 0) / total if total else None)

    def refresh(self) -> None:
        for key, fn in self._sync.items():
            fn(self.ctl.get(key))
        for name, row in self.models.items():
            try:
                row.set_status(self.ctl.model_status(name))
            except Exception:
                row.set_status({"state": "pending", "detail": ""})

    def on_enter(self) -> None:
        self._dirty = True           # cheap, and always right even after external changes
        super().on_enter()
        self._sync_timer.start()

    # ---- backup, restore, sync folder
    def export_backup(self) -> Modal:
        """Ask what goes into the backup (history is the one private, bulky part), then where to save it."""
        m = Modal(self.win.panel, "Back up VoxFlow",
                  "The backup always contains your settings, dictionary, snippets and styles.", width=460,
                  name="backup_export")
        with_history = Toggle(True, name="Include my dictation history")
        with_history.setObjectName("backup_export_history")
        choice = Row("Include my dictation history",
                     "Everything you have dictated, and your statistics. Leave it out if the file is going to "
                     "someone else.", with_history)
        choice.divider = False
        m.body.addWidget(choice)

        def save():
            suffix = self.ctl.BACKUP_SUFFIX
            path = dialogs.ask_save_path(self.win, "Back up VoxFlow",
                                         f"voxflow-backup-{_dt.date.today().isoformat()}{suffix}",
                                         f"VoxFlow backup (*{suffix})")
            if not path:
                return
            try:
                res = self.ctl.export_backup(path, with_history.isChecked())
            except ValueError as exc:
                m.show_error(str(exc))
                return
            m.close_modal()
            c = res.get("counts") or {}
            saved = Path(str(res.get("path") or path)).name
            parts = [plural(int(c.get("dictionary", 0)), "word"), plural(int(c.get("snippets", 0)), "snippet")]
            if "history" in (res.get("includes") or []):
                parts.append(plural(int(c.get("history", 0)), "transcript"))
            self.win.toast("success", f"Backup saved to {saved}: your settings, {', '.join(parts)}.")

        m.add_buttons("Save backup…", save)
        return self.win.open_modal(m, m.btn_primary)

    def import_backup(self) -> Modal | None:
        suffix = self.ctl.BACKUP_SUFFIX
        path = dialogs.ask_open_path(self.win, "Restore from a backup", f"VoxFlow backup (*{suffix})")
        if not path:
            return None
        info = self.run(self.ctl.inspect_backup, path)
        if info is None:
            return None
        c = info.get("counts") or {}
        includes = list(info.get("includes") or [])
        facts = [Path(path).name]
        origin = "this PC" if info.get("same_device") else str(info.get("device") or "another PC")
        if info.get("created"):
            facts.append(f"Made {ago(info['created'])} on {origin}"
                         + (f", VoxFlow {info['version']}" if info.get("version") else ""))
        counts = [plural(int(c.get("dictionary", 0)), "word"), plural(int(c.get("snippets", 0)), "snippet")]
        counts.append(plural(int(c.get("history", 0)), "transcript") if "history" in includes else "no history")
        facts.append(", ".join(counts))
        extra = [x for x in includes if x in ("settings", "styles", "stats")]
        if extra:
            facts.append("Also contains your " + (", ".join(extra[:-1]) + " and " + extra[-1] if len(extra) > 1
                                                    else extra[0]))
        if int(info.get("skipped") or 0):
            facts.append(f"{plural(int(info['skipped']), 'damaged entry', 'damaged entries')} in the file will be left out")

        def go(mode: str, keys: list[str] | None):
            if keys is not None and "history" in keys and "stats" in includes:
                keys = keys + ["stats"]                 # statistics travel with the history they describe
            try:
                res = self.ctl.import_backup(path, mode, keys)
            except ValueError as exc:
                return str(exc)
            got = res.get("counts") or {}
            dropped = list(res.get("settings_skipped") or [])
            unusable = int(res.get("skipped") or 0)
            if mode == "replace":
                text = "Your profile was replaced with the backup." if keys is None else \
                    "The parts you ticked were replaced with the backup's."
            else:
                added = [plural(int(got.get(k, 0)), one) for k, one in (("dictionary", "word"), ("snippets", "snippet"),
                                                                       ("history", "transcript")) if got.get(k)]
                text = f"Backup merged: {', '.join(added)} added." if added else \
                    "Backup merged: everything in it was already here."
            self.win.toast("success", text)
            if dropped:
                self.win.toast("warning", f"{plural(len(dropped), 'setting')} from the backup could not be used on "
                                          "this PC and kept its current value.")
            if unusable:
                self.win.toast("warning", f"{plural(unusable, 'damaged entry', 'damaged entries')} in the backup "
                                          "could not be read and was left out.")
            if res.get("aliases_skipped"):
                self.win.toast("warning", aliases_left_out(res["aliases_skipped"]))
            self.refresh()
            return True

        return merge_or_replace(
            self.win, "Restore this backup?", facts,
            "Adds what is missing: the words, snippets and transcripts you do not have yet. Nothing you already "
            "have is changed; your settings and tones stay as they are.",
            "Makes the ticked parts of this profile equal to the backup: what you have there now is replaced, "
            "anything not ticked is left alone. This cannot be undone, so export a backup of this profile first if "
            "you might want it." + ("" if info.get("same_device") else " Microphone, model and start-up settings "
                                    "are not taken from a backup made on another computer."),
            go, name="backup_import", parts=[
                {"key": "settings", "label": "Settings", "note": "and tones", "available": "settings" in includes,
                 "merge": False},
                {"key": "dictionary", "label": "Dictionary", "note": plural(int(c.get("dictionary", 0)), "word"),
                 "available": "dictionary" in includes},
                {"key": "snippets", "label": "Snippets", "note": plural(int(c.get("snippets", 0)), "snippet"),
                 "available": "snippets" in includes},
                {"key": "styles", "label": "App styles", "note": plural(int(c.get("styles", 0)), "app rule"),
                 "available": "styles" in includes},
                {"key": "history", "label": "History", "note": plural(int(c.get("history", 0)), "transcript")
                 + " and statistics", "available": "history" in includes}])

    def choose_sync_folder(self) -> None:
        current = str(self.ctl.get("sync_folder") or "")
        path = dialogs.ask_folder(self.win, "Choose the sync folder", current)
        if path and self.setting("sync_folder", path):
            self.win.toast("success", "Sync folder set. VoxFlow keeps it up to date from now on.")

    def _show_sync(self, status: dict | None = None) -> None:
        if not isinstance(status, dict):
            try:
                status = self.ctl.sync_status()
            except Exception as exc:
                status = {"state": "error", "folder": str(self.ctl.get("sync_folder") or ""), "last_sync": None,
                          "error": str(exc)}
        state = str(status.get("state", "off"))
        folder = str(status.get("folder") or "")
        last = status.get("last_sync")
        when = f"Last synced {ago(last)}." if last else "Not synced yet."
        on = state != "off" and bool(folder)
        if not on:
            text, role = ("Off. Pick a folder (OneDrive, a network drive) and your dictionary, snippets, styles and "
                          "settings stay in step across PCs. VoxFlow never uploads anything."), "TEXT_DIM"
        elif state == "syncing":
            text, role = "Syncing…", "CYAN"
        elif state == "error":
            text, role = f"{str(status.get('error') or 'The last sync did not work').rstrip('.')}. {when}", "RED"
        else:
            # the controller's own one-line summary (who else uses the folder, skipped files), then when
            summary = str(status.get("message") or "Up to date.").strip()
            warn = bool(status.get("newer") or status.get("unreadable"))
            text, role = f"{summary} {when}", ("AMBER" if warn else "GREEN")
        self.sync_row.set_sub(text, role)
        names = [str(d.get("name", "")) for d in (status.get("devices") or []) if isinstance(d, dict) and d.get("name")]
        shown = folder + (f"   ·   shared with {', '.join(names[:3])}" + (f" and {len(names) - 3} more" if len(names) > 3
                                                                       else "") if names else "")
        self.sync_folder.setText(shown)
        self.sync_folder.setToolTip(folder)
        self.sync_folder.setVisible(on)
        self.btn_sync_off.setVisible(on)
        self.btn_sync_now.setVisible(on)
        self.btn_sync_now.setEnabled(state != "syncing")
        self.btn_sync_choose.setText("Change…" if on else "Choose folder…")
        self.btn_sync_choose.setAccessibleName("Change the sync folder" if on else "Choose a sync folder")

    def on_leave(self) -> None:
        self.cancel_capture()
        self._set_testing(False)
        self._set_arranging(False)
        self._sync_timer.stop()

    # ---- hotkey capture (one recorder at a time; always released when the page or window goes away)
    def _begin_capture(self, rec: HotkeyRecorder) -> None:
        if self._capturing is not None and self._capturing is not rec:
            other, self._capturing = self._capturing, None
            if other.capturing:
                other.toggle_capture()
        self._capturing = rec
        if not rec.use_qt_keys:
            self.run(self.ctl.begin_hotkey_capture)

    def _end_capture(self, rec: HotkeyRecorder) -> None:
        if self._capturing is rec:
            self._capturing = None
            if not rec.use_qt_keys:
                self.run(self.ctl.cancel_hotkey_capture)

    def cancel_capture(self) -> None:
        rec = self._capturing
        if rec is not None and rec.capturing:
            rec.toggle_capture()            # emits capture_cancelled -> _end_capture

    def _on_captured(self, combo: str) -> None:
        rec, self._capturing = self._capturing, None
        if rec is not None and rec.capturing:
            rec.captured(combo or None)

    def _hotkey_changed(self, key: str, combo: str, rec: HotkeyRecorder) -> None:
        if self._capturing is rec:           # recorded through Qt keys (demo mode): nothing global to release
            self._capturing = None
        try:
            self.ctl.set(key, combo)
        except ValueError as exc:
            rec.set_combo(self.ctl.get(key))
            rec.show_message(str(exc))
            return
        rec.show_message("Saved" if combo else "Turned off", "GREEN", 1800)

    # ---- mic test / flow bar arranging
    def _set_testing(self, on: bool) -> None:
        if on == self._testing:
            return
        self._testing = on
        self.btn_test.setText("Stop" if on else "Test")
        self.test_row.set_sub("Listening… say something." if on else "Press Test and speak: the bar should move.",
                              "CYAN" if on else "TEXT_DIM")
        if not on:
            self.meter.set_level(0.0)
        self.meter.set_source(self.ctl.level if on else None)
        self.run(self.ctl.set_level_monitor, on)

    def _set_arranging(self, on: bool) -> None:
        if on == self._arranging:
            return
        self._arranging = on
        self.btn_move.setText("Done" if on else "Move…")
        self.btn_move.kind = "primary" if on else "secondary"
        self.btn_move.update()
        self._sync_position(self.ctl.get("overlay_position"))
        self.win.arrange_overlay_requested.emit(on)

    def _confirm_reset(self) -> None:
        self.win.open_modal(confirm(
            self.win.panel, "Reset all settings?",
            "Hotkeys, microphone, sounds and appearance go back to their defaults. Your history, dictionary, "
            "snippets and styles are not touched.", "Reset settings",
            lambda: self.run(self.ctl.reset_settings, success="Settings reset to defaults."),
            name="settings_reset_confirm"))
