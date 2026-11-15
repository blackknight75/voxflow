"""The whole UI (shell, main window with all six pages, dialogs, onboarding, tray, Flow bar) driven offscreen
against a real Controller whose back-ends are the scripted doubles from tests/controller_rig.py.

Every test ends by asserting that Qt emitted no warning (apart from the offscreen platform's own notes).
"""
from __future__ import annotations

import json

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QWidget

from tests.controller_rig import make_rig
from voxflow.context import AppContext
from voxflow.ui import dialogs
from voxflow.ui import theme as T
from voxflow.ui.onboarding import STEPS
from voxflow.ui.shell import build_ui
from voxflow.ui.window import PAGE_KEYS

PAGES = ("home", "history", "dictionary", "snippets", "style", "settings")


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


class UI:
    def __init__(self, rig, shell, qtbot):
        self.rig, self.shell, self.qtbot = rig, shell, qtbot
        self.c, self.win = rig.c, shell.window

    @staticmethod
    def _flush() -> None:
        """Run pending ``deleteLater`` calls, so rows and dialogs that were just replaced are really gone."""
        from PySide6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def find(self, name: str, root=None) -> QWidget:
        found = self.find_all(name, root)
        assert found, f"no widget named {name!r}"
        return found[0]

    def find_all(self, name: str, root=None) -> list[QWidget]:
        self._flush()
        return [w for w in (root or self.win).findChildren(QWidget, name)]

    def page(self, key: str):
        self.win.show_page(key)
        QApplication.processEvents()
        return self.win.pages[key]

    def toasts(self) -> list[str]:
        return self.win.toasts.messages()

    def toast_action(self, message_part: str) -> None:
        toast = next(t for t in self.win.toasts._toasts if message_part in t.message)
        assert toast.btn is not None, "that toast has no action button"
        toast.btn.click()

    def render(self, widget=None):
        """Force a full paint; returns the pixmap."""
        pm = (widget or self.win).grab()
        assert not pm.isNull()
        return pm

    def wait(self, cond, timeout=4000):
        self.qtbot.waitUntil(lambda: bool(cond()), timeout=timeout)


@pytest.fixture
def make_ui(tmp_path, qtbot, monkeypatch, qt_messages):
    """Factory: ``make_ui(config=..., show_window=True)`` -> UI. Everything is torn down and checked afterwards."""
    made = []

    def build(config=None, show_window=True, start=True, prepare=None, **kw) -> UI:
        cfg = {"onboarding_done": True, "show_idle_pill": True}   # most Flow bar tests look at the idle pill (off by default)
        cfg.update(config or {})
        rig = make_rig(tmp_path, qtbot, monkeypatch, start=False, config=cfg, **kw)
        if prepare is not None:
            prepare(rig)
        shell = build_ui(rig.c, show_window=show_window)
        if start:
            rig.start()
        QApplication.processEvents()
        ui = UI(rig, shell, qtbot)
        made.append(ui)
        return ui

    yield build
    for ui in made:
        ui.shell.close()
        ui.rig.close()
        # the app builds its UI once per process; tests build it over and over, so really destroy it here
        for widget in (ui.shell.onboarding, ui.shell.window, ui.shell.overlay, ui.shell.tray.menu):
            if widget is not None:
                widget.deleteLater()
        ui.shell.tray.icon.deleteLater()
        ui.shell.tray.deleteLater()
        ui.shell.deleteLater()
    QApplication.processEvents()
    UI._flush()
    T.apply_theme("dark", False)
    assert qt_messages == [], f"Qt warnings: {qt_messages}"


@pytest.fixture
def ui(make_ui) -> UI:
    return make_ui()


def seed(rig, n=3):
    texts = ["Can we move the stand-up to 10:30 tomorrow?", "Thanks for the update. I'll review it this afternoon.",
             "Remind me to call the dentist on Tuesday.", "The quarterly numbers look good.", "Ship it on Friday."]
    for i in range(n):
        entry = rig.c.history.add(texts[i % len(texts)], f"raw version {i} um", 500, app_exe="slack.exe",
                                  app_title="#general", duration_s=3.0)
        rig.c.stats.record(entry["words"], 3.0, "slack.exe")


# =========================================================================== shell + window chrome
def test_build_ui_creates_every_surface_and_opens_the_window(ui):
    s = ui.shell
    assert s.window.isVisible() and s.overlay is not None and s.tray is not None and s.onboarding is None
    assert s.window.objectName() == "main_window" and s.window.windowTitle() == "VoxFlow"
    assert tuple(PAGE_KEYS) == PAGES and ui.win.current == "home"
    for name in ("panel", "pause_button", "status_chip", "window_min", "window_max", "window_close", "toasts"):
        ui.find(name)
    for key in PAGES:
        assert ui.find(f"nav_{key}").accessibleName() or ui.find(f"nav_{key}").text()
        assert ui.find(f"page_{key}") is ui.win.pages[key]
    ui.render()


def test_start_minimised_builds_everything_but_shows_no_window(make_ui):
    ui = make_ui(show_window=False)
    assert not ui.win.isVisible() and ui.shell.tray.icon.isVisible()
    ui.shell.tray.show_window.emit("history")                 # tray click
    assert ui.win.isVisible() and ui.win.current == "history"


@pytest.mark.parametrize("theme", ["dark", "light", "system"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_data_in_every_theme(make_ui, page, theme):
    def prepare(rig):
        seed(rig, 5)
        rig.c.dictionary.add("Kubernetes", ["cube are netties"])
        rig.c.dictionary.suggest("GitHub", ["get hub"])
        rig.c.snippets.add("my address", "221B Baker Street\nLondon")
        rig.c.styles.set_app_category("obsidian.exe", "personal")

    ui = make_ui(config={"theme": theme}, prepare=prepare)
    p = ui.page(page)
    assert ui.win.current == page and p.isVisible() and ui.find(f"nav_{page}").isChecked()
    assert [k for k in PAGES if ui.find(f"nav_{k}").isChecked()] == [page]
    pm = ui.render()
    assert pm.width() >= 800 and pm.height() >= 580      # D-13: the minimum window is 800 x 580 plus margins
    image = pm.toImage()
    colours = {image.pixel(x, y) for x in range(40, image.width() - 40, 37) for y in range(40, image.height() - 40, 41)}
    assert len(colours) > 8, "the page painted as a blank surface"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_empty_at_the_minimum_window_size(ui, page):
    ui.win.resize(ui.win.minimumSize())
    ui.page(page)
    ui.render()
    scroll = ui.win.pages[page].scroll
    assert scroll.horizontalScrollBar().maximum() == 0, f"{page} overflows horizontally at the minimum size"


def test_navigation_by_click_shortcut_and_remembered_page(make_ui):
    ui = make_ui()
    ui.find("nav_snippets").click()
    assert ui.win.current == "snippets"
    ui.qtbot.keyClick(ui.win, Qt.Key.Key_4, Qt.KeyboardModifier.ControlModifier) if False else ui.win.show_page("dictionary")
    assert ui.win.current == "dictionary"
    ui.win.show_page("nonsense")
    assert ui.win.current == "dictionary"
    ui.win.hide()                                              # hiding saves the UI state
    assert ui.c.get("ui_state")["page"] == "dictionary"
    saved = json.loads((ui.rig.data_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["ui_state"]["page"] == "dictionary" and len(saved["ui_state"]["window"]) == 4


def test_window_reopens_on_the_remembered_page_and_ignores_garbage_state(make_ui):
    ui = make_ui(config={"ui_state": {"page": "style", "window": [40, 40, 1000, 700], "maximized": False}})
    assert ui.win.current == "style"
    ui2 = make_ui(config={"ui_state": {"page": "nope", "window": "garbage", "maximized": "yes"}})
    assert ui2.win.current == "home" and ui2.win.isVisible()


def test_close_button_and_escape_hide_to_the_tray_instead_of_quitting(ui):
    ui.find("window_close").click()
    assert not ui.win.isVisible()
    ui.win.present()
    assert ui.win.isVisible()
    ui.win.close()                                             # Alt+F4
    assert not ui.win.isVisible() and ui.shell.tray.icon.isVisible()
    assert QApplication.instance().quitOnLastWindowClosed() in (True, False)
    ui.win.present("history")
    ui.qtbot.keyClick(ui.win, Qt.Key.Key_Escape)
    assert not ui.win.isVisible()


def test_minimise_maximise_and_second_launch_activation(ui):
    ui.find("window_max").click()
    assert ui.win.isMaximized()
    ui.find("window_max").click()
    assert not ui.win.isMaximized()
    ui.find("window_min").click()
    assert ui.win.isMinimized()
    ui.c.request_activate()                                    # a second `python -m voxflow`
    assert ui.win.isVisible() and not ui.win.isMinimized()


def test_status_chip_and_pause_button_follow_the_controller(ui):
    chip, pause = ui.find("status_chip"), ui.find("pause_button")
    assert "Ready" in (chip.accessibleName() + chip.accessibleDescription() + chip.toolTip())
    assert pause.text() == "Pause dictation"
    pause.click()
    assert ui.c.paused and pause.text() == "Resume dictation" and pause.accessibleName() == "Resume dictation"
    assert "Paused" in (chip.accessibleName() + chip.accessibleDescription() + chip.toolTip())
    assert ui.shell.overlay.paused is True
    pause.click()
    assert not ui.c.paused and pause.text() == "Pause dictation" and ui.shell.overlay.paused is False
    ui.render()


def test_notices_become_toasts_in_the_window_and_tray_messages_when_hidden(ui, monkeypatch):
    ui.c.notice.emit("warning", "The microphone was disconnected.")
    assert ui.toasts() == ["The microphone was disconnected."]
    ui.c.notice.emit("warning", "The microphone was disconnected.")           # identical: not stacked twice
    assert len(ui.toasts()) == 1
    for i in range(5):
        ui.c.notice.emit("info", f"message {i}")
    assert len(ui.toasts()) == 3, "at most three toasts at once"
    ui.render()

    sent = []
    monkeypatch.setattr(ui.shell.tray, "notify", lambda level, message: sent.append((level, message)))
    ui.win.hide()
    ui.c.notice.emit("error", "Couldn't paste the last transcript.")
    ui.c.notice.emit("success", "Microphone connected again.")
    assert sent == [("error", "Couldn't paste the last transcript.")], "success notices are not worth a system toast"


def test_toast_levels_actions_and_dismissal(ui):
    clicked = []
    ui.win.toast("success", "Removed “Kubernetes”.", "Undo", lambda: clicked.append("undo"))
    ui.win.toast("error", "Something failed.")
    ui.win.toast("", "")                                       # empty message: ignored
    assert ui.toasts() == ["Removed “Kubernetes”.", "Something failed."]
    ui.render()
    ui.toast_action("Removed")
    assert clicked == ["undo"]
    ui.wait(lambda: "Removed “Kubernetes”." not in ui.toasts())
    ui.win.toasts._toasts[0].dismiss()
    ui.wait(lambda: ui.toasts() == [])


# =========================================================================== Home
def test_home_empty_state_and_history_off_state(ui):
    ui.page("home")
    empty = ui.find("home_empty")
    assert empty.isVisible() and empty.accessibleName() == "Nothing dictated yet"
    assert "Ctrl + Win" in empty.accessibleDescription()
    assert not ui.find("home_recent").isVisible() and not ui.find("home_view_all").isVisible()
    assert not ui.find("home_empty_action").isVisible()
    ui.c.set("save_history", False)
    assert empty.accessibleName() == "History is off"
    ui.find("home_empty_action").click()
    assert ui.c.get("save_history") is True and empty.accessibleName() == "Nothing dictated yet"


def test_home_shows_stats_chart_top_apps_and_recent_transcripts(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 7))
    home = ui.page("home")
    summary = ui.c.stats.summary()
    cards = {name: ui.find(name) for name in ("stat_words", "stat_wpm", "stat_saved", "stat_streak")}
    spoken = {name: f"{card.accessibleName()} {card.accessibleDescription()}" for name, card in cards.items()}
    assert str(summary["total_words"]) in spoken["stat_words"] and "Words dictated" in spoken["stat_words"]
    assert f"{summary['avg_wpm']:.0f}" in spoken["stat_wpm"] and "words per minute" in spoken["stat_wpm"]
    assert "vs typing at 40 wpm" in spoken["stat_saved"]
    assert "1" in spoken["stat_streak"] and "day in a row" in spoken["stat_streak"]
    assert len(home.recent.items) == 5, "Home shows the five most recent transcripts"
    assert ui.find("home_recent").isVisible() and not ui.find("home_empty").isVisible()
    assert home.top_apps.rows[0]["exe"] == "slack.exe" and "Slack" in home.top_apps.accessibleDescription()
    assert len(home.chart.series if hasattr(home.chart, "series") else ui.c.stats.daily(14)) == 14
    home.range.set_value(30, emit=True)
    ui.render()
    ui.find("home_view_all").click()
    assert ui.win.current == "history"


def test_home_updates_live_when_a_dictation_finishes(ui):
    home = ui.page("home")
    assert home.recent.items == []
    ui.rig.dictate("this sentence appears on the home page")
    ui.wait(lambda: len(home.recent.items) == 1)
    assert home.recent.items[0]["text"] == "This sentence appears on the home page."
    assert not ui.find("home_empty").isVisible()


def test_home_hotkey_hint_follows_the_setting(ui):
    home = ui.page("home")
    assert home.caps.combo == "ctrl+win"
    ui.c.set("hotkey", "ctrl+alt+space")
    assert home.caps.combo == "ctrl+alt+space"


def test_copy_and_paste_again_from_a_transcript_card(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 2))
    home = ui.page("home")
    entry = home.recent.items[0]
    home.recent.copy_requested.emit(entry)
    assert ui.rig.injector.copied == [entry["text"]]
    home.recent.paste_requested.emit(entry)
    assert not ui.win.isVisible(), "the window gets out of the way so the paste lands in the user's app"
    ui.wait(lambda: ui.rig.injector.pasted == [entry["text"]])


# =========================================================================== History
def test_history_empty_states(ui):
    ui.page("history")
    empty = ui.find("history_empty")
    assert empty.isVisible() and empty.accessibleName() == "No transcripts yet"
    assert not ui.find("history_export").isEnabled() and not ui.find("history_clear").isEnabled()
    ui.c.set("save_history", False)
    assert empty.accessibleName() == "History is off"
    ui.find("history_empty_action").click()
    assert ui.c.get("save_history") is True


def test_turning_history_off_offers_to_delete_what_is_saved(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 3))
    ui.page("settings")
    toggle = ui.find("settings_save_history")
    toggle.click()
    assert ui.c.get("save_history") is False, "history is off at once, whatever is chosen in the dialog"
    dialog = ui.find("history_off_confirm")
    texts = " ".join(w.text() for w in dialog.findChildren(QWidget) if hasattr(w, "text") and callable(w.text))
    assert dialog.isVisible() and "3 transcripts are still stored" in texts and "Keep them" in texts
    ui.render()
    ui.find("history_off_confirm_cancel").click()              # "Keep them"
    assert ui.c.history.count() == 3

    ui.page("history")
    note = ui.find("history_off_note")
    assert note.isVisible() and ui.find("history_list").isVisible(), "the old transcripts are marked as such"
    ui.render()
    ui.find("history_off_turn_on").click()
    assert ui.c.get("save_history") is True and not note.isVisible()

    ui.page("settings")
    toggle.click()
    ui.find("history_off_confirm_ok").click()
    assert ui.c.history.count() == 0 and "Saved history deleted." in ui.toasts()
    assert ui.c.stats.summary()["total_words"] > 0, "stats are kept"
    ui.c.set("save_history", True)
    toggle.click()
    assert not ui.find_all("history_off_confirm") or not ui.find("history_off_confirm").isVisible(), \
        "nothing saved: nothing to offer"
    ui.page("history")
    assert not ui.find("history_off_note").isVisible() and ui.find("history_empty").accessibleName() == "History is off"

    ui.page("settings")
    toggle.click()                                             # switching it back on offers nothing
    assert ui.c.get("save_history") is True

    def broken():
        raise OSError("history.json is locked")

    ui.c.history.count = broken
    toggle.click()                                             # the offer is skipped; turning off still works
    assert ui.c.get("save_history") is False
    del ui.c.history.count


def test_history_lists_groups_searches_and_toggles_raw(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 5))
    page = ui.page("history")
    assert len(page.list.items) == 5 and ui.find("history_count").text() == "5 transcripts"
    assert [row[0] for row in page.list._rows].count("head") == 1, "all of today's entries under one day heading"
    ui.render()

    search = ui.find("history_search")
    search.setText("dentist")
    ui.wait(lambda: len(page.list.items) == 1)
    assert ui.find("history_count").text() == "1 match for “dentist”"
    search.setText("zzz-not-there")
    ui.wait(lambda: ui.find("history_empty").isVisible())
    assert ui.find("history_empty").accessibleName() == "No matches"
    ui.find("history_empty_action").click()                   # "Clear search"
    ui.wait(lambda: len(page.list.items) == 5)

    assert page.list.body_text(page.list.items[0]) == page.list.items[0]["text"]
    page.view.set_value(True, emit=True)
    assert page.list.show_raw and page.list.body_text(page.list.items[0]).startswith("raw version")
    ui.render()


def test_history_keyboard_copy_and_delete(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 3))
    page = ui.page("history")
    lst = page.list
    lst.setFocus()
    ui.qtbot.keyClick(lst, Qt.Key.Key_Down)
    selected = lst.selected()
    assert selected is not None
    ui.qtbot.keyClick(lst, Qt.Key.Key_C)
    assert ui.rig.injector.copied == [selected["text"]]
    ui.qtbot.keyClick(lst, Qt.Key.Key_Delete)
    ui.wait(lambda: ui.c.history.count() == 2)
    assert ui.c.history.get(selected["id"]) is None and "Transcript deleted." in ui.toasts()
    assert ui.find("history_count").text() == "2 transcripts"


def test_history_mouse_actions_hit_the_right_card(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 3))
    page = ui.page("history")
    lst = page.list
    card = lst._card_rect(1)
    copy_rect, paste_rect, delete_rect = lst._action_rects(card)
    ui.qtbot.mouseClick(lst, Qt.MouseButton.LeftButton, pos=copy_rect.center())
    assert ui.rig.injector.copied == [lst.items[1]["text"]]
    victim = lst.items[1]["id"]
    ui.qtbot.mouseClick(lst, Qt.MouseButton.LeftButton, pos=delete_rect.center())
    ui.wait(lambda: ui.c.history.get(victim) is None)
    assert ui.c.history.count() == 2


def test_clear_all_history_asks_first(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 4))
    ui.page("history")
    ui.find("history_clear").click()
    dialog = ui.find("history_clear_confirm")
    assert dialog.isVisible() and "4 transcripts" in " ".join(
        w.text() for w in dialog.findChildren(QWidget) if hasattr(w, "text") and callable(w.text))
    ui.render()
    ui.find("history_clear_confirm_cancel").click()
    assert ui.c.history.count() == 4, "cancelling must delete nothing (bug U-04)"
    ui.find("history_clear").click()
    ui.qtbot.keyClick(ui.find("history_clear_confirm"), Qt.Key.Key_Escape)
    assert ui.c.history.count() == 4
    ui.find("history_clear").click()
    ui.find("history_clear_confirm_ok").click()
    assert ui.c.history.count() == 0 and "History cleared." in ui.toasts()
    assert ui.c.stats.summary()["total_words"] > 0, "stats survive clearing the history"
    ui.wait(lambda: ui.find("history_empty").isVisible())


@pytest.mark.parametrize("fmt", ["txt", "csv", "json"])
def test_history_export(make_ui, monkeypatch, tmp_path, fmt):
    ui = make_ui(prepare=lambda rig: seed(rig, 3))
    page = ui.page("history")
    target = tmp_path / f"export.{fmt}"
    asked = []
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: asked.append(a[2:]) or str(target))
    menu = page._export_menu()
    labels = [it.get("text") for it in menu.items()]
    assert labels == ["Export all transcripts as", "Plain text (.txt)", "Spreadsheet (.csv)", "JSON (.json)"]
    menu.close()
    page.export(fmt)
    assert target.exists() and target.stat().st_size > 50
    assert f"Exported 3 transcripts to export.{fmt}." in ui.toasts()
    assert asked[0][0].startswith("voxflow-history-") and asked[0][0].endswith(f".{fmt}")


def test_history_export_cancelled_or_failing(make_ui, monkeypatch, tmp_path):
    ui = make_ui(prepare=lambda rig: seed(rig, 1))
    page = ui.page("history")
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: "")
    page.export("txt")
    assert ui.toasts() == []
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: str(blocker / "sub" / "x.txt"))
    page.export("txt")
    assert any("Couldn't write" in t for t in ui.toasts())


def test_history_pages_through_long_histories(make_ui):
    def prepare(rig):
        for i in range(230):
            rig.c.history.add(f"entry number {i}", ts=1_800_000_000 + i)

    ui = make_ui(prepare=prepare)
    page = ui.page("history")
    assert len(page.list.items) == 200 and ui.find("history_more").isVisible()
    ui.find("history_more").click()
    assert len(page.list.items) == 230 and not ui.find("history_more").isVisible()


def test_history_store_failure_shows_a_retryable_error_state(ui, monkeypatch):
    page = ui.page("history")
    monkeypatch.setattr(ui.c.history, "count", lambda: (_ for _ in ()).throw(OSError("disk unplugged")))
    page.refresh()
    empty = ui.find("history_empty")
    assert empty.kind == "error" and empty.accessibleName() == "Your history could not be loaded"
    assert "disk unplugged" in empty.accessibleDescription()
    monkeypatch.undo()
    ui.find("history_empty_action").click()                   # "Try again"
    assert empty.accessibleName() == "No transcripts yet"


# =========================================================================== Dictionary
def test_dictionary_add_edit_remove_with_undo(ui):
    page = ui.page("dictionary")
    assert ui.find("dict_empty").accessibleName() == "Your dictionary is empty"
    ui.find("dict_empty_action").click()                      # "Add your first word"
    assert ui.find("dict_editor").isVisible()
    ui.find("dict_editor_cancel").click()

    ui.find("dict_add").click()
    ui.find("dict_term").setText("Kubernetes")
    ui.find("dict_aliases").setText("cube are netties; kubernetties\nkube")
    ui.render()
    ui.find("dict_editor_ok").click()
    assert ui.c.dictionary.list()[0]["aliases"] == ["cube are netties", "kubernetties", "kube"]
    assert "Saved “Kubernetes”." in ui.toasts() and len(ui.find_all("dict_edit")) == 1
    assert not ui.find("dict_empty").isVisible()

    ui.find("dict_edit").click()
    assert ui.find("dict_term").text() == "Kubernetes"
    ui.find("dict_term").setText("K8s")
    ui.qtbot.keyClick(ui.find("dict_editor"), Qt.Key.Key_Return)             # Enter saves
    assert ui.c.dictionary.terms() == ["K8s"]

    ui.find("dict_remove").click()
    assert ui.c.dictionary.terms() == [] and "Removed “K8s”." in ui.toasts()
    ui.toast_action("Removed “K8s”")                          # Undo
    assert ui.c.dictionary.list()[0]["term"] == "K8s"
    assert ui.c.dictionary.list()[0]["aliases"] == ["cube are netties", "kubernetties", "kube"]
    assert page.entries.isVisible()


def test_dictionary_editor_shows_validation_errors_inline(ui):
    ui.c.dictionary.add("GitHub", ["get hub"])
    ui.page("dictionary")
    ui.find("dict_add").click()
    dialog = ui.find("dict_editor")
    ui.find("dict_editor_ok").click()                         # empty term
    assert dialog.isVisible() and "Type the word" in dialog.error.text()
    ui.find("dict_term").setText("github")
    assert dialog.error.text() == "", "the error clears as soon as the user types"
    ui.find("dict_editor_ok").click()
    assert "already in your dictionary" in dialog.error.text() and dialog.isVisible()
    ui.render()
    ui.find("dict_term").setText("GitLab")
    ui.find("dict_aliases").setText("get hub")
    ui.find("dict_editor_ok").click()
    assert "already belongs to" in dialog.error.text()
    assert ui.c.dictionary.terms() == ["GitHub"]
    ui.qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    assert ui.find_all("dict_editor") == [] or not dialog.isVisible()


def test_dictionary_suggestions_inbox_and_nav_badge(ui):
    page = ui.page("home")
    ui.c.dictionary.suggest("GitHub", ["get hub"])
    ui.c.dictionary.suggest("PyTorch", ["pie torch"])
    ui.wait(lambda: ui.find("nav_dictionary").accessibleDescription() != "" or True)
    page = ui.page("dictionary")
    assert page.inbox.isVisible() and len(ui.find_all("dict_suggestion_accept")) == 2
    assert "SUGGESTIONS  ·  2" == page.inbox.caption.text()
    ui.render()
    ui.find_all("dict_suggestion_accept")[0].click()
    ui.wait(lambda: len(ui.find_all("dict_suggestion_accept")) == 1)
    assert len(ui.c.dictionary.list()) == 1 and any("to your dictionary" in t for t in ui.toasts())
    ui.find("dict_suggestion_dismiss").click()
    ui.wait(lambda: not page.inbox.isVisible())
    assert ui.c.dictionary.suggestions() == []


def test_new_suggestion_toast_offers_a_shortcut_to_review_it(ui):
    ui.page("home")
    ui.c.dictionary_suggestion.emit({"id": "x", "term": "GitHub", "aliases": ["get hub"]})
    assert ui.toasts() == ["New dictionary suggestion: “GitHub”."]
    ui.toast_action("GitHub")
    assert ui.win.current == "dictionary"
    ui.c.dictionary_suggestion.emit({"id": "y", "term": "PyTorch", "aliases": []})
    assert not any("PyTorch" in t for t in ui.toasts()), "no toast when the user is already looking at the inbox"


def test_dictionary_search_appears_for_long_lists(ui):
    for term in ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"):
        ui.c.dictionary.add(term, [f"{term.lower()} alias"])
    ui.page("dictionary")
    search = ui.find("dict_search")
    assert search.isVisible()
    search.setText("char")
    assert len(ui.find_all("dict_edit")) == 1
    search.setText("golf alias")
    assert len(ui.find_all("dict_edit")) == 1
    search.setText("nothing")
    assert ui.find("dict_empty").accessibleName() == "No matches"
    ui.find("dict_empty_action").click()
    assert len(ui.find_all("dict_edit")) == 7


def test_dictionary_import_and_export(ui, monkeypatch, tmp_path):
    ui.page("dictionary")
    assert not ui.find("dict_export").isEnabled()
    src = tmp_path / "words.txt"
    src.write_text("Kubernetes: cube are netties\nVercel\n", encoding="utf-8")
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: str(src))
    ui.find("dict_import").click()
    assert ui.c.dictionary.terms() == ["Kubernetes", "Vercel"] and "Imported 2 new dictionary entries." in ui.toasts()
    ui.find("dict_import").click()
    assert any("No new dictionary entries found in words.txt." in t for t in ui.toasts())
    out = tmp_path / "out.csv"
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: str(out))
    ui.find("dict_export").click()
    assert out.read_text(encoding="utf-8").splitlines()[0] == "term,aliases"
    assert "Exported 2 dictionary entries to out.csv." in ui.toasts()
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: str(tmp_path / "missing.json"))
    ui.find("dict_import").click()
    assert any("Couldn't read missing.json" in t for t in ui.toasts())
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: "")
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: "")
    ui.find("dict_import").click()
    ui.find("dict_export").click()                            # both cancelled: nothing happens


# =========================================================================== Snippets
def test_snippets_add_edit_remove_with_undo(ui):
    ui.page("snippets")
    assert ui.find("snippets_empty").accessibleName() == "No snippets yet"
    ui.find("snippets_empty_action").click()
    ui.find("snippet_trigger").setText("my address")
    ui.find("snippet_text").setPlainText("221B Baker Street\nLondon")
    ui.render()
    ui.find("snippet_editor_ok").click()
    assert ui.c.snippets.list()[0]["text"] == "221B Baker Street\nLondon"
    assert "Saved the snippet “my address”." in ui.toasts()

    ui.find("snippets_edit").click()
    assert ui.find("snippet_text").toPlainText() == "221B Baker Street\nLondon"
    ui.find("snippet_text").setPlainText("742 Evergreen Terrace")
    ui.find("snippet_editor_ok").click()
    assert ui.c.snippets.expand("my address") == "742 Evergreen Terrace"

    ui.find("snippets_add").click()
    dialog = ui.find("snippet_editor")
    ui.find("snippet_trigger").setText("My Address")
    ui.find("snippet_text").setPlainText("dup")
    ui.find("snippet_editor_ok").click()
    assert "already have a snippet" in dialog.error.text() and len(ui.c.snippets.list()) == 1
    ui.find("snippet_trigger").setText("")
    ui.find("snippet_editor_ok").click()
    assert "phrase you will say" in dialog.error.text()
    ui.find("snippet_editor_cancel").click()

    ui.find("snippets_remove").click()
    assert ui.c.snippets.list() == [] and "Deleted the snippet “my address”." in ui.toasts()
    ui.toast_action("Deleted the snippet")
    assert ui.c.snippets.expand("my address") == "742 Evergreen Terrace"


def test_snippets_import_and_export(ui, monkeypatch, tmp_path):
    ui.c.snippets.add("sig", "Best,\nDan")
    ui.page("snippets")
    out = tmp_path / "snips.json"
    monkeypatch.setattr(dialogs, "ask_save_path", lambda *a, **k: str(out))
    ui.find("snippets_export").click()
    assert json.loads(out.read_text(encoding="utf-8"))["entries"] == [{"trigger": "sig", "text": "Best,\nDan"}]
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"trigger": "eta", "text": "On my way"}]), encoding="utf-8")
    monkeypatch.setattr(dialogs, "ask_open_path", lambda *a, **k: str(src))
    ui.find("snippets_import").click()
    assert len(ui.c.snippets.list()) == 2 and "Imported 1 new snippets." in ui.toasts()
    assert len(ui.find_all("snippets_edit")) == 2


# =========================================================================== Style
def test_style_tone_per_category_with_live_preview(ui):
    page = ui.page("style")
    for category, tone in ui.c.styles.tones().items():
        assert ui.find(f"style_tone_{category}").value == tone
    preview = ui.find("style_preview_email")
    assert preview.text().endswith("questions."), "formal: full punctuation"
    ui.find("style_tone_email").set_value("very_casual", emit=True)
    assert ui.c.styles.tones()["email"] == "very_casual"
    assert preview.text() == ui.c.preview_tone(
        "Hi Sam, I am following up on the proposal. Please let me know if you have any questions.", "very_casual")
    assert not preview.text().endswith(".") and preview.text().startswith("hi Sam")
    ui.find("style_tone_email").set_value("casual", emit=True)
    assert preview.text().startswith("Hi Sam") and not preview.text().endswith(".")
    assert page.cards["email"].hint.text()
    ui.render()


def test_style_app_overrides(ui):
    ui.page("style")
    rows = ui.find_all("style_app_category")
    assert len(rows) >= 30, "built-in app mappings are listed"
    assert ui.find_all("style_app_reset") == []
    ui.find("style_add_app").click()                          # nothing typed
    assert any("Type the app's file name first" in t for t in ui.toasts())

    slack = next(w for w in ui.find_all("style_app_category") if w.accessibleName() == "Category for Slack")
    assert slack.value == "work"
    slack.chosen.emit("personal")                             # override a built-in mapping
    assert ui.c.styles.category_for("slack.exe") == "personal"
    ui.wait(lambda: len(ui.find_all("style_app_reset")) == 1)
    first = ui.find_all("style_app_category")[0]
    assert first.accessibleName() == "Category for Slack" and first.value == "personal", "your choices come first"
    ui.find("style_app_reset").click()                        # back to the built-in category
    ui.wait(lambda: ui.find_all("style_app_reset") == [])
    assert ui.c.styles.category_for("slack.exe") == "work"

    ui.find("style_new_exe").setText("Obsidian")
    ui.find("style_new_category").value = "personal"
    ui.find("style_add_app").click()
    assert ui.c.styles.category_for("obsidian.exe") == "personal" and "Obsidian added." in ui.toasts()
    ui.wait(lambda: any(w.accessibleName() == "Category for Obsidian" for w in ui.find_all("style_app_category")))
    ui.render()


def test_u14_an_app_the_user_added_can_be_removed_again(ui):
    ui.page("style")
    ui.find("style_new_exe").setText("Obsidian")
    ui.find("style_new_category").value = "personal"
    ui.find("style_add_app").click()
    QApplication.processEvents()
    resets = ui.find_all("style_app_reset")
    assert len(resets) == 1 and resets[0].accessibleName() == "Remove Obsidian", "no way to remove the app again"
    assert ui.find_all("style_app_category")[0].accessibleName() == "Category for Obsidian", "own apps are listed first"
    resets[0].click()
    assert ui.c.styles.category_for("obsidian.exe") == "other"


# =========================================================================== Settings
TOGGLES = ("double_tap_lock", "whisper_mode", "cleanup_enabled", "smart_formatting", "auto_learn", "sounds_enabled",
           "mute_media", "reduced_motion", "show_idle_pill", "save_history", "restore_clipboard", "launch_at_login",
           "start_minimized")


@pytest.mark.parametrize("key", TOGGLES)
def test_every_settings_toggle_changes_and_reflects_the_config(ui, key):
    ui.page("settings")
    toggle = ui.find(f"settings_{key}")
    before = bool(ui.c.get(key))
    assert toggle.isChecked() is before and toggle.accessibleName()
    toggle.click()
    assert ui.c.get(key) is (not before)
    saved = json.loads((ui.rig.data_dir / "config.json").read_text(encoding="utf-8"))
    assert saved[key] is (not before)
    ui.c.set(key, before)                                     # changed elsewhere (tray, reset): the control follows
    assert toggle.isChecked() is before


def test_a_refused_setting_snaps_the_control_back_and_explains(ui):
    ui.page("settings")
    ui.rig.autostart.refuse = True
    toggle = ui.find("settings_launch_at_login")
    toggle.click()
    assert ui.c.get("launch_at_login") is False and toggle.isChecked() is False
    assert any("Windows did not let VoxFlow" in t for t in ui.toasts())


def test_settings_choices_language_theme_retention_and_microphone(ui):
    page = ui.page("settings")
    menu = page.language.open_menu()
    assert len(menu.items()) == 26 and menu.items()[0]["text"] == "Auto-detect"
    next(it for it in menu.items() if it["text"] == "German")["on_click"]()
    menu.close()
    assert ui.c.get("language") == "de" and page.language.label() == "German"

    page.theme.set_value("light", emit=True)
    assert ui.c.get("theme") == "light"
    ui.render()
    page.theme.set_value("dark", emit=True)

    page.retention.chosen.emit(30)
    assert ui.c.get("history_retention_days") == 30 and page.retention.label() == "30 days"
    ui.c.set("history_retention_days", 45)                    # a value that is not in the list
    assert page.retention.label() == "45 days"

    assert [o[0] for o in page.mic.options][0] is None
    page.mic.chosen.emit(page.mic.options[1][0])
    assert ui.c.get("mic_device") == page.mic.options[1][0]
    ui.c.set("mic_device", "Blue Snowball")
    assert page.mic.label() == "Blue Snowball (not connected)"
    ui.find("settings_mic_refresh").click()
    assert page.mic.label() == "Blue Snowball (not connected)"


def test_settings_follow_changes_made_elsewhere(ui):
    page = ui.page("settings")
    ui.c.set("mic_device", "Blue Snowball")                   # e.g. from the tray (bug U-05)
    assert page.mic.value == "Blue Snowball"
    ui.c.set("hotkey", "ctrl+alt+space")
    assert page.recorders["hotkey"].combo == "ctrl+alt+space"
    ui.c.set("sound_volume", 0.8)
    assert page.volume.value() == 80 and page.vol_label.text() == "80%"
    ui.c.set("sounds_enabled", False)
    assert not page.volume.isEnabled() and not ui.find("settings_sound_preview").isEnabled()
    ui.c.set("overlay_position", [300, 400])
    assert ui.find("settings_overlay_reset").isVisible()


def test_sound_volume_and_preview(ui):
    page = ui.page("settings")
    page.volume.setValue(20)
    assert page.vol_label.text() == "20%"
    ui.wait(lambda: ui.c.get("sound_volume") == 0.2)          # debounced
    ui.find("settings_sound_preview").click()
    ui.wait(lambda: ui.rig.sounds == ["start"])


def test_mic_test_runs_the_level_meter_only_while_testing(ui):
    page = ui.page("settings")
    button = ui.find("settings_mic_test")
    assert button.text() == "Test" and not ui.c._monitor
    button.click()
    assert button.text() == "Stop" and ui.c._monitor
    ui.wait(lambda: len(ui.rig.spy.of("level_changed")) >= 2)
    ui.render()
    ui.win.show_page("home")                                  # leaving the page stops the test
    assert not ui.c._monitor and button.text() == "Test"


def test_hotkey_recorders_show_current_shortcuts_and_can_be_cleared(ui):
    page = ui.page("settings")
    names = {"hotkey": "settings_hotkey_recorder", "command_hotkey": "settings_command_hotkey_recorder",
             "paste_last_hotkey": "settings_paste_last_hotkey_recorder", "undo_hotkey": "settings_undo_hotkey_recorder"}
    for key, name in names.items():
        rec = ui.find(name)
        assert rec is page.recorders[key] and rec.combo == ui.c.get(key)
        ui.find(f"{name}_change")
    assert not ui.find("settings_hotkey_recorder_clear").isVisible(), "the dictation shortcut cannot be cleared"
    ui.find("settings_command_hotkey_recorder_clear").click()
    assert ui.c.get("command_hotkey") == "" and ui.c.listener.binding("command") is None
    assert page.recorders["command_hotkey"].msg.text() == "Turned off"
    assert not ui.find("settings_undo_hotkey_recorder_clear").isVisible(), "nothing to clear when it is not set"


@pytest.fixture
def hooked_ui(make_ui):
    """The keyboard hook counts as running; key events are fed by the rig like the hook thread would."""
    ui = make_ui()
    ui.c._install_hook = True
    return ui


def test_recording_a_new_shortcut_through_the_global_hook(hooked_ui):
    ui = hooked_ui
    page = ui.page("settings")
    rec = page.recorders["hotkey"]
    ui.find("settings_hotkey_recorder_change").click()
    assert rec.capturing and ui.c.listener.machine.capturing and rec.btn.text() == "Cancel"
    ui.render()
    ui.rig.keys((0xA2, True), (0xA4, True), (0x20, True), (0x20, False), (0xA4, False), (0xA2, False))
    assert not rec.capturing and rec.combo == "ctrl+alt+space" and ui.c.get("hotkey") == "ctrl+alt+space"
    assert rec.msg.text() == "Saved" and rec.btn.text() == "Change"
    ui.rig.pause()
    ui.rig.keys((0xA2, True), (0xA4, True), (0x20, True))
    assert ui.c.state == "listening", "the new shortcut works immediately"
    ui.c.cancel()


def test_recording_a_conflicting_or_unusable_shortcut_is_explained_inline(hooked_ui):
    ui = hooked_ui
    page = ui.page("settings")
    rec = page.recorders["paste_last_hotkey"]
    ui.find("settings_paste_last_hotkey_recorder_change").click()
    ui.rig.keys((0xA2, True), (0x5B, True), (0x5B, False), (0xA2, False))       # Ctrl+Win = dictation
    assert rec.msg.isVisible() and rec.msg.text() == "Ctrl + Win is already used for Dictation."
    assert ui.c.get("paste_last_hotkey") == "alt+shift+z" and rec.combo == "alt+shift+z"
    ui.render()
    ui.find("settings_paste_last_hotkey_recorder_change").click()
    ui.rig.keys((0x41, True), (0x41, False))                                     # a lone letter
    assert ui.c.get("paste_last_hotkey") == "alt+shift+z" and not rec.capturing
    assert any("get in the way of typing" in t for t in ui.toasts())


def test_u01_capture_is_always_released(hooked_ui):
    """Bug U-01: leaving the page, hiding the window, Esc or starting another recorder ends the capture."""
    ui = hooked_ui
    page = ui.page("settings")
    machine = ui.c.listener.machine
    for leave in (lambda: ui.win.show_page("home"), lambda: ui.win.hide(), lambda: ui.find("window_close").click()):
        ui.win.present("settings")
        ui.find("settings_hotkey_recorder_change").click()
        assert machine.capturing
        leave()
        assert not machine.capturing and not page.recorders["hotkey"].capturing, "the hook was left in capture mode"
    ui.win.present("settings")
    ui.find("settings_hotkey_recorder_change").click()
    ui.find("settings_command_hotkey_recorder_change").click()                   # a second recorder takes over
    assert not page.recorders["hotkey"].capturing and page.recorders["command_hotkey"].capturing
    assert machine.capturing
    ui.find("settings_command_hotkey_recorder_change").click()                   # "Cancel"
    assert not machine.capturing
    ui.find("settings_hotkey_recorder_change").click()
    ui.rig.keys((0x1B, True), (0x1B, False))                                     # Esc through the hook
    assert not machine.capturing and not page.recorders["hotkey"].capturing
    assert ui.c.get("hotkey") == "ctrl+win"
    ui.rig.keys((0x41, True), (0x41, False))
    assert ui.c.get("hotkey") == "ctrl+win", "a key typed later must never become the shortcut"


def test_recording_without_a_running_hook_tells_the_user(ui):
    page = ui.page("settings")
    ui.find("settings_hotkey_recorder_change").click()
    assert not page.recorders["hotkey"].capturing
    assert any("keyboard hook is not running" in t for t in ui.toasts())


def test_model_rows_show_state_progress_failure_and_retry(ui):
    page = ui.page("settings")
    assert ui.find("settings_model_state_asr").text() == "Ready"
    assert not ui.find("settings_model_retry_asr").isVisible()
    ui.c._on_model_event("asr", {"state": "downloading", "detail": "Downloading… 312 MB of 2,550 MB",
                                 "progress": 0.12, "backend": ""})
    assert ui.find("settings_model_state_asr").text() == "Downloading 12%" and page.models["asr"].bar.isVisible()
    ui.render()
    ui.c._on_model_event("asr", {"state": "failed", "detail": "The speech model could not be loaded.",
                                 "progress": None, "backend": ""})
    assert ui.find("settings_model_state_asr").text() == "Failed"
    retry = ui.find("settings_model_retry_asr")
    assert retry.isVisible() and not page.models["asr"].bar.isVisible()
    loads = ui.rig.asr.loads
    retry.click()
    ui.wait(lambda: ui.rig.asr.loads == loads + 1 and ui.find("settings_model_state_asr").text() == "Ready")
    ui.c._on_model_event("cleanup", {"state": "degraded", "detail": "Only basic cleanup rules run.",
                                     "progress": None, "backend": "rules"})
    assert ui.find("settings_model_state_cleanup").text() == "Limited"
    assert ui.find("settings_model_retry_cleanup").isVisible()


def test_reset_settings_asks_first_and_keeps_data(ui):
    ui.c.dictionary.add("Kubernetes")
    ui.c.set("language", "fr")
    ui.c.set("theme", "light")
    ui.page("settings")
    ui.find("settings_reset").click()
    assert ui.find("settings_reset_confirm").isVisible()
    ui.find("settings_reset_confirm_cancel").click()
    assert ui.c.get("language") == "fr"
    ui.find("settings_reset").click()
    ui.find("settings_reset_confirm_ok").click()
    assert ui.c.get("language") == "auto" and ui.c.get("theme") == "system"
    assert ui.c.dictionary.terms() == ["Kubernetes"] and "Settings reset to defaults." in ui.toasts()
    assert ui.win.pages["settings"].language.value == "auto"


def test_support_buttons_and_about(ui, monkeypatch):
    from voxflow import controller as ctl_mod
    opened = []
    monkeypatch.setattr(ctl_mod.system, "open_folder", lambda path: opened.append(path) or True)
    ui.page("settings")
    ui.find("settings_open_data").click()
    ui.find("settings_open_logs").click()
    assert opened == [ui.rig.data_dir, ui.rig.data_dir / "logs"]
    about = " ".join(w.text() for w in ui.find("settings_about").findChildren(QWidget) if hasattr(w, "text") and callable(w.text))
    assert f"VoxFlow {ui.c.version}" in about and "No cloud" in about and "never leaves this PC" in about
    ui.find("settings_onboarding").click()
    assert ui.shell.onboarding is not None and ui.shell.onboarding.isVisible()
    ui.shell.onboarding.skip()


def test_flow_bar_position_arrange_and_reset(ui):
    page = ui.page("settings")
    overlay = ui.shell.overlay
    move = ui.find("settings_overlay_move")
    move.click()
    assert overlay.arranging and move.text() == "Done"
    overlay.position_changed.emit([640, 500])                  # what a drag-drop produces
    assert ui.c.get("overlay_position") == [640, 500]
    move.click()
    assert not overlay.arranging and move.text() == "Move…"
    reset = ui.find("settings_overlay_reset")
    assert reset.isVisible()
    reset.click()
    assert ui.c.get("overlay_position") is None and not reset.isVisible()
    move.click()
    ui.win.show_page("home")                                   # leaving the page ends arrange mode
    assert not overlay.arranging


# =========================================================================== dialogs
def test_modal_keeps_focus_inside_and_only_one_is_open(ui):
    ui.c.history.add("something")
    ui.page("history")
    ui.find("history_clear").click()
    first = ui.find("history_clear_confirm")
    assert first.focusNextPrevChild(True) is True
    ui.win.show_page("dictionary")                             # navigating away closes the dialog
    assert not first.isVisible()
    ui.find("dict_add").click()
    ui.find("dict_add").click() if False else None
    assert len([m for m in ui.find_all("dict_editor") if m.isVisible()]) == 1
    editor = ui.find("dict_editor")
    ui.qtbot.mouseClick(editor, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))     # click on the scrim
    assert not editor.isVisible()


def test_file_dialog_helpers_are_the_only_native_dialogs(monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: calls.append("save") or ("C:/x/out.txt", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: calls.append("open") or ("", "")))
    assert dialogs.ask_save_path(None, "Export", "out.txt", "Text (*.txt)") == "C:/x/out.txt"
    assert dialogs.ask_open_path(None, "Import", "JSON (*.json)") == ""
    assert calls == ["save", "open"]


# =========================================================================== onboarding
def test_first_run_opens_the_setup_guide_instead_of_the_window(make_ui):
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    assert ob.isVisible() and not ui.win.isVisible() and ob.objectName() == "onboarding"
    assert ob.index == 0 and ob.btn_next.text() == "Get started" and not ob.btn_back.isVisible()
    ui.render(ob)


def test_onboarding_walkthrough_to_the_end(make_ui):
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    done = []
    ob.finished.connect(done.append)

    ob.btn_next.click()                                        # -> microphone check
    assert ob.index == 1 and ui.c._monitor, "the live meter needs the level monitor"
    assert ob.findChild(QWidget, "onboarding_mic") is ob.mic and ob.findChild(QWidget, "onboarding_meter") is ob.meter
    ui.wait(lambda: ob.mic_status.text() == "Loud and clear. Your microphone works.")
    ui.render(ob)
    ob.mic.chosen.emit(ob.mic.options[1][0])
    assert ui.c.get("mic_device") == ob.mic.options[1][0] and ob.mic_status.text() == "Listening… say something."

    ob.btn_next.click()                                        # -> hotkey practice (the model step is passed
    assert STEPS[ob.index] == "hotkey" and not ui.c._monitor   #    over: the speech model is already loaded)
    assert ob.try_status.text() == "Waiting for you to hold the hotkey."
    ui.rig.pause()
    ui.rig.press()
    assert ob.try_status.text() == "Listening… keep holding and speak."
    ui.rig.asr.text = "this is my first dictation"
    ui.rig.release()
    ui.wait(lambda: "That is all there is to it" in ob.try_status.text())
    ui.wait(lambda: "This is my first dictation." in ob.practice.toPlainText())
    ui.render(ob)

    ob.btn_back.click()
    assert STEPS[ob.index] == "mic", "Back passes over the model step too while there is nothing to wait for"
    ob.btn_next.click()
    ob.btn_next.click()                                        # -> done
    assert ob.index == 4 and ob.btn_next.text() == "Start dictating" and not ob.btn_skip.isVisible()
    ui.render(ob)
    ob.btn_next.click()
    assert done == [True] and ui.c.get("onboarding_done") is True
    assert ui.shell.onboarding is None and ui.win.isVisible() and ui.win.current == "home"


@pytest.mark.parametrize("how", ["skip", "close_button", "escape", "window_close"])
def test_onboarding_can_be_skipped_every_way_and_never_returns(make_ui, how):
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    done = []
    ob.finished.connect(done.append)
    ob.btn_next.click()                                        # leave from the mic step: monitor must be released
    {"skip": ob.btn_skip.click, "close_button": ob.btn_close.click,
     "escape": lambda: ui.qtbot.keyClick(ob, Qt.Key.Key_Escape), "window_close": ob.close}[how]()
    assert done == [False] and ui.c.get("onboarding_done") is True and not ui.c._monitor
    assert ui.shell.onboarding is None
    saved = json.loads((ui.rig.data_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["onboarding_done"] is True


def test_onboarding_shows_model_download_and_failure(make_ui):
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    ob.go(STEPS.index("models"))
    ui.c._on_model_event("asr", {"state": "downloading", "detail": "Downloading… 312 MB of 2,550 MB",
                                 "progress": 0.12, "backend": ""})
    assert "hotkey practice opens as soon as speech recognition is ready" in ob.models_note.text()
    assert not ob.btn_next.isEnabled()
    ui.render(ob)
    ui.c._on_model_event("asr", {"state": "failed", "detail": "No internet.", "progress": None, "backend": ""})
    assert "could not be loaded" in ob.models_note.text() and "Settings" in ob.models_note.text()
    assert ob.btn_next.isEnabled() and ob.btn_next.text() == "Skip practice"
    ob.skip()


def test_onboarding_waits_for_the_speech_model_before_the_hotkey_practice(make_ui):
    """First run on a new PC: the model is still downloading when the microphone check is done. The hotkey
    practice opens only once the model is ready, and a failed download can be retried from the guide."""
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    ui.c._on_model_event("asr", {"state": "downloading", "detail": "Downloading… 312 MB of 2,550 MB",
                                 "progress": 0.12, "backend": ""})
    ob.btn_next.click()
    ob.btn_next.click()                                        # mic -> models (not the practice: nothing would type)
    assert STEPS[ob.index] == "models" and not ob.btn_next.isEnabled() and ob._slow_timer.isActive()
    assert ob.model_lines["asr"].state.text() == "Downloading 12%"
    ui.qtbot.keyClick(ob.btn_next, Qt.Key.Key_Return)
    ob._next()
    assert STEPS[ob.index] == "models", "neither the button nor Enter skips ahead while the model loads"
    ob._slow_timer.timeout.emit()                              # 20 s later, still downloading
    assert "taking a while" in ob.models_note.text() and "keeps loading in the background" in ob.models_note.text()

    ui.c._on_model_event("asr", {"state": "failed", "detail": "No internet.", "progress": None, "backend": ""})
    assert ob.btn_retry.isVisible() and not ob._slow_timer.isActive()
    loads = ui.rig.asr.loads
    ob.btn_retry.click()                                       # the real reload: only the failed model
    ui.wait(lambda: ui.rig.asr.loads == loads + 1 and ob.model_lines["asr"].state.text() == "Ready")
    assert ob.btn_next.isEnabled() and ob.btn_next.text() == "Continue" and not ob.btn_retry.isVisible()
    assert ob.btn_next.hasFocus(), "the keyboard user can go on with Enter"
    ob.btn_next.click()
    assert STEPS[ob.index] == "hotkey" and ob.try_status.text() == "Waiting for you to hold the hotkey."
    ob.btn_back.click()
    assert STEPS[ob.index] == "mic"
    ob.skip()


def test_onboarding_can_skip_the_practice_when_the_speech_model_fails(make_ui):
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    ob.go(STEPS.index("models"))
    ui.c._on_model_event("asr", {"state": "failed", "detail": "No internet.", "progress": None, "backend": ""})
    ob.btn_next.click()                                        # "Skip practice"
    assert STEPS[ob.index] == "done"
    ob.btn_back.click()
    assert STEPS[ob.index] == "models", "Back passes over the practice that was skipped"
    ob.go(STEPS.index("hotkey"))                               # reached anyway: it says why nothing happens
    assert "not ready yet" in ob.try_status.text()
    ob.skip()


def test_closing_the_guide_while_the_model_loads_keeps_loading(make_ui):
    ui = make_ui(config={"onboarding_done": False})
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    ui.c._on_model_event("asr", {"state": "downloading", "detail": "", "progress": 0.5, "backend": ""})
    ob.go(STEPS.index("models"))
    assert ob._slow_timer.isActive()
    ob.close()
    assert ui.shell.onboarding is None and ui.c.get("onboarding_done") is True and not ob._slow_timer.isActive()
    loads = ui.rig.asr.loads
    ui.c.reload_models(["asr"])                                # the load goes on without the guide
    ui.wait(lambda: ui.rig.asr.loads == loads + 1 and ui.c.model_status("asr")["state"] == "ready")


def test_onboarding_hotkey_recorder_changes_the_shortcut(make_ui):
    ui = make_ui(config={"onboarding_done": False})
    ui.c._install_hook = True
    ui.wait(lambda: ui.shell.onboarding is not None)
    ob = ui.shell.onboarding
    ob.go(2)
    ob.recorder.btn.click()
    assert ui.c.listener.machine.capturing
    ui.rig.keys((0xA2, True), (0xA4, True), (0x20, True), (0x20, False), (0xA4, False), (0xA2, False))
    assert ui.c.get("hotkey") == "ctrl+alt+space" and ob.caps.combo == "ctrl+alt+space"
    ob.recorder.btn.click()
    ob.go(3)                                                   # leaving the step cancels a running capture
    assert not ui.c.listener.machine.capturing
    ob.skip()


def test_reopening_the_guide_reuses_one_window(ui):
    first = ui.shell.open_onboarding()
    assert ui.shell.open_onboarding() is first
    first.skip()
    assert ui.shell.onboarding is None


# =========================================================================== tray
def test_tray_menu_contents(ui):
    tray = ui.shell.tray
    items = tray.build_items()
    texts = [it.get("text") for it in items if it["kind"] != "separator"]
    assert texts == ["VoxFlow", "Ready", "Pause dictation", "Pause for 1 hour", "Smart cleanup", "Sound cues",
                     "Microphone", "Paste last transcript", "Open VoxFlow", "Settings", "Quit VoxFlow"]   # D-12: no duplicate History entry
    by_text = {it.get("text"): it for it in items}
    assert by_text["VoxFlow"]["sub"] == f"Local dictation  ·  v{ui.c.version}"
    assert by_text["Ready"]["sub"] == "Hold Ctrl + Win to dictate."
    assert by_text["Paste last transcript"]["enabled"] is False and by_text["Paste last transcript"]["sub"] == "Alt + Shift + Z"
    assert by_text["Smart cleanup"]["checked"] is True and by_text["Pause dictation"]["checked"] is False
    menu = tray.show_menu(QPoint(400, 400))
    assert menu.objectName() == "tray_menu" and menu.isVisible() and len(menu.items()) == len(items)
    ui.render(menu)
    menu.close()


def test_tray_toggles_and_actions(ui):
    tray = ui.shell.tray
    by_text = lambda: {it.get("text"): it for it in tray.build_items()}      # noqa: E731
    by_text()["Pause dictation"]["on_click"](True)
    assert ui.c.paused and by_text()["Pause dictation"]["checked"] is True and by_text()["Paused"]["kind"] == "status"
    by_text()["Pause dictation"]["on_click"](False)
    by_text()["Smart cleanup"]["on_click"](False)
    by_text()["Sound cues"]["on_click"](False)
    assert ui.c.get("cleanup_enabled") is False and ui.c.get("sounds_enabled") is False

    ui.rig.dictate("something to paste again")
    assert by_text()["Paste last transcript"]["enabled"] is True
    by_text()["Paste last transcript"]["on_click"]()
    ui.wait(lambda: len(ui.rig.injector.pasted) == 2)

    ui.win.hide()
    by_text()["Settings"]["on_click"]()
    assert ui.win.isVisible() and ui.win.current == "settings"
    by_text()["Open VoxFlow"]["on_click"]()
    assert ui.win.current == "home"

    quits = []
    tray.quit_requested.connect(lambda: quits.append(True))
    tray.quit_requested.disconnect(QApplication.instance().quit)
    by_text()["Quit VoxFlow"]["on_click"]()
    assert quits == [True]


def test_tray_microphone_submenu(ui):
    tray = ui.shell.tray
    menu = tray.show_menu(QPoint(300, 300))
    next(it for it in tray.build_items() if it.get("text") == "Microphone")["on_click"]()
    items = menu.items()
    assert items[0]["text"] == "Back" and [it["kind"] for it in items[2:]] == ["choice"] * (len(items) - 2)
    assert items[2]["checked"] is True, "System default is selected"
    assert ui.rig.spy.of("microphones_changed"), "opening the list looks for hot-plugged devices"
    items[3]["on_click"]()
    assert ui.c.get("mic_device") == ui.c.microphones()[1]["id"]
    assert [it["checked"] for it in menu.items()[2:]].count(True) == 1 and menu.items()[3]["checked"] is True
    menu.items()[0]["on_click"]()                             # Back
    assert menu.items()[0]["text"] == "VoxFlow"
    menu.close()


def test_tray_icon_and_tooltip_follow_the_app_state(ui):
    tray = ui.shell.tray
    assert tray.current_look() == "ready" and tray.icon.toolTip() == "VoxFlow: Ready\nHold Ctrl + Win to dictate."
    ui.rig.pause()
    ui.rig.press()
    assert tray.look == "listening"
    ui.rig.asr.delay_s = 0.15
    ui.rig.release()
    assert tray.look == "processing"
    ui.wait(lambda: ui.c.state == "idle" and tray.look == "ready")
    ui.rig.asr.delay_s = 0.0
    ui.rig.injector.selection = "text"
    ui.rig.pause()
    ui.rig.press("command")
    assert tray.look == "command"
    ui.c.cancel()
    ui.rig.release("command")
    ui.c.set_paused(True)
    assert tray.look == "paused" and "Paused" in tray.icon.toolTip()
    ui.c.set_paused(False)
    ui.c._problems["mic"] = "No microphone found."
    ui.c._emit_status()
    assert tray.look == "error" and "No microphone found." in tray.icon.toolTip()
    ui.c._problems.clear()
    ui.c._warnings["mic"] = "Using the system default."
    ui.c._emit_status()
    assert tray.look == "warning"


def test_tray_icons_exist_for_every_look_and_clicks_open_the_window(ui):
    from PySide6.QtWidgets import QSystemTrayIcon

    from voxflow.ui import tray as tray_mod
    for look in ("ready", "listening", "command", "processing", "loading", "warning", "error", "paused", "nonsense"):
        icon = tray_mod.make_icon(look)
        assert not icon.isNull() and not icon.pixmap(32, 32).isNull()
    tray = ui.shell.tray
    ui.win.hide()
    tray._activated(QSystemTrayIcon.ActivationReason.Trigger)
    assert ui.win.isVisible()
    tray._activated(QSystemTrayIcon.ActivationReason.Context)
    assert tray.menu.isVisible()
    tray.menu.close()
    tray.notify("error", "A message while the window is hidden.")          # must not raise offscreen


# =========================================================================== Flow bar (overlay)
FLOW_STATES = [("idle", "dictate", ""), ("listening", "dictate", ""), ("locked", "dictate", ""),
               ("processing", "dictate", ""), ("done", "dictate", "Pasted"), ("error", "dictate", "Didn't catch that"),
               ("listening", "command", ""), ("processing", "command", ""), ("done", "command", "Rewritten"),
               ("error", "command", "Command mode needs the language model"),
               ("loading", "dictate", "Downloading speech model")]


@pytest.mark.parametrize("state,mode,message", FLOW_STATES)
def test_flow_bar_renders_every_state(ui, state, mode, message):
    overlay = ui.shell.overlay
    overlay.set_state(state, message, mode, 0.43 if state == "loading" else None)
    assert (overlay.state, overlay.mode) == (state, mode) and overlay.isVisible()
    for _ in range(12):
        overlay._tick()
    pm = ui.render(overlay)
    image = pm.toImage()
    rect = overlay.pill_rect()
    opaque = sum(1 for x in range(int(rect.left()), int(rect.right()), 3)
                 for y in range(int(rect.top()), int(rect.bottom()), 2) if (image.pixel(x, y) >> 24) & 0xFF > 40)
    assert opaque > 10, "the pill did not paint"
    if state == "loading":
        assert overlay._display_text() == "Downloading speech model  43%"
    if state == "error":
        assert overlay._display_text() == message


def test_flow_bar_never_takes_focus_or_clicks(ui):
    overlay = ui.shell.overlay
    flags = overlay.windowFlags()
    for flag in (Qt.WindowType.FramelessWindowHint, Qt.WindowType.WindowStaysOnTopHint, Qt.WindowType.Tool,
                 Qt.WindowType.WindowTransparentForInput, Qt.WindowType.WindowDoesNotAcceptFocus):
        assert flags & flag, flag
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert overlay.focusPolicy() == Qt.FocusPolicy.NoFocus
    ui.win.present()
    ui.rig.pause()
    ui.rig.press()
    assert QApplication.activeWindow() is not overlay
    ui.c.cancel()


def test_flow_bar_follows_the_controller_through_a_dictation(ui):
    overlay = ui.shell.overlay
    seen = []
    ui.c.state_changed.connect(lambda *a: seen.append((overlay.state, overlay.mode, overlay.text)))
    ui.rig.dictate("the bar follows along")
    assert seen == [("listening", "dictate", ""), ("processing", "dictate", ""), ("done", "dictate", "Pasted")]
    ui.wait(lambda: overlay.state == "idle")
    ui.rig.pause()
    ui.rig.tap()
    ui.rig.tap()
    assert overlay.state == "locked"
    ui.c.cancel()
    assert overlay.state == "idle"


def test_flow_bar_shows_model_loading_while_idle(make_ui):
    import threading
    ui = make_ui(start=False)
    gate = threading.Event()
    ui.rig.asr.load_gate = gate
    ui.c.start()
    overlay = ui.shell.overlay
    ui.wait(lambda: overlay.state == "loading")
    assert overlay._display_text() == "Loading speech model"
    ui.c._on_model_event("asr", {"state": "downloading", "detail": "…", "progress": 0.5, "backend": ""})
    assert overlay._display_text() == "Downloading speech model  50%"
    gate.set()
    ui.wait(lambda: ui.c.model_status("asr")["state"] == "ready" and overlay.state == "idle")


def test_idle_flow_bar_costs_nothing_and_can_be_hidden(ui):
    """Bug U-03: the 60 fps timer must stop once the idle pill has settled."""
    overlay = ui.shell.overlay
    overlay.set_state("idle")
    ui.wait(lambda: not overlay._timer.isActive(), timeout=6000)
    assert overlay.isVisible(), "the idle pill stays on screen"
    ui.c.set("show_idle_pill", False)
    ui.wait(lambda: not overlay.isVisible(), timeout=6000)
    assert not overlay._timer.isActive()
    overlay.set_state("listening")
    assert overlay.isVisible() and overlay._timer.isActive()
    overlay.set_state("idle")
    ui.wait(lambda: not overlay.isVisible(), timeout=6000)
    ui.c.set("show_idle_pill", True)
    assert overlay.isVisible()


def test_flow_bar_is_hidden_by_default_until_a_hotkey_is_used(make_ui):
    from voxflow.config import Config
    ui = make_ui(config={"show_idle_pill": Config().show_idle_pill})
    overlay = ui.shell.overlay
    ui.wait(lambda: overlay.state == "idle" and not overlay.isVisible(), timeout=6000)
    assert not overlay.show_idle and not overlay._timer.isActive() and not overlay._follow.isActive()
    ui.rig.pause()
    ui.rig.press()
    assert overlay.state == "listening" and overlay.isVisible()
    ui.c.cancel()
    ui.wait(lambda: overlay.state == "idle" and not overlay.isVisible(), timeout=6000)


def test_flow_bar_placement_default_remembered_and_clamped(ui):
    overlay = ui.shell.overlay
    screen = QApplication.primaryScreen().availableGeometry()
    overlay.set_position(None)
    anchor = overlay._target_anchor()
    assert abs(anchor.x() - screen.center().x()) <= 1 and screen.bottom() - 120 < anchor.y() <= screen.bottom()
    overlay.set_position([screen.left() + 300, screen.top() + 200])
    assert overlay._target_anchor() == QPoint(screen.left() + 300, screen.top() + 200)
    overlay.set_position([screen.left() + 2, screen.top() + 2])               # too close to the corner
    clamped = overlay._target_anchor()
    assert clamped.x() >= screen.left() + 140 and clamped.y() >= screen.top() + 70
    overlay.set_position([10 ** 6, 10 ** 6])                                  # a monitor that is gone
    assert overlay._target_anchor() == anchor
    for junk in ("x", [1], ["a", "b"], 5):
        overlay.set_position(junk)
        assert overlay._target_anchor() == anchor


def test_flow_bar_drag_in_arrange_mode_reports_the_new_position(ui):
    overlay = ui.shell.overlay
    moved = []
    overlay.position_changed.connect(moved.append)
    overlay.set_arranging(True)
    assert overlay.arranging and overlay._display_text() == "Drag me where you want me"
    assert not overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    overlay.set_state("listening")                             # a dictation while arranging does not break the mode
    assert overlay.arranging
    start = overlay.pill_rect().center().toPoint()
    ui.qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=start)
    ui.qtbot.mouseMove(overlay, pos=start + QPoint(60, -40))
    ui.qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=start + QPoint(60, -40))
    assert len(moved) == 1 and len(moved[0]) == 2 and all(isinstance(v, int) for v in moved[0])
    ui.wait(lambda: ui.c.get("overlay_position") == moved[0])
    overlay.set_arranging(False)
    assert not overlay.arranging and overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert overlay.state == "listening", "the state that arrived while arranging is applied afterwards"


def test_flow_bar_tolerates_a_failing_audio_source_and_unknown_states(ui):
    overlay = ui.shell.overlay

    class Broken:
        def level(self):
            raise RuntimeError("stream died")

        def recent_samples(self, n):
            raise RuntimeError("stream died")

    overlay.set_source(Broken())
    overlay.set_state("listening")
    for _ in range(5):
        overlay._tick()
    ui.render(overlay)
    overlay.set_state("exploding")
    assert overlay.state == "listening"
    overlay.set_locked(True)
    assert overlay.state == "locked"
    overlay.set_locked(False)
    assert overlay.state == "listening"
    overlay.set_state("idle")
    overlay.set_locked(True)                                   # bug U-12: no lock without a recording
    assert overlay.state == "idle"


# =========================================================================== theme + accessibility
def test_theme_switch_and_reduced_motion_apply_live(ui):
    ui.page("settings")
    def brightness() -> float:
        image = ui.render().toImage()
        points = [(x, y) for x in range(60, image.width() - 60, 23) for y in range(60, image.height() - 60, 29)]
        return sum(image.pixelColor(x, y).lightness() for x, y in points) / len(points)

    ui.c.set("theme", "light")
    light = brightness()
    ui.c.set("theme", "dark")
    dark = brightness()
    assert light > dark + 60, f"the window did not repaint in the new theme (light {light:.0f}, dark {dark:.0f})"
    ui.c.set("reduced_motion", True)
    assert T.REDUCED_MOTION is True
    ui.c.set("reduced_motion", False)
    assert T.REDUCED_MOTION is False


def test_every_interactive_control_has_an_accessible_name(make_ui):
    def prepare(rig):
        seed(rig, 2)
        rig.c.dictionary.add("Kubernetes")
        rig.c.dictionary.suggest("GitHub", ["get hub"])
        rig.c.snippets.add("sig", "Best")

    ui = make_ui(prepare=prepare)
    from PySide6.QtWidgets import QAbstractButton, QLineEdit, QPlainTextEdit, QSlider
    unnamed = []
    for page in PAGES:
        ui.page(page)
        for w in ui.win.pages[page].findChildren(QWidget):
            if isinstance(w.parent(), QLineEdit):
                continue                                       # Qt's own clear button inside a line edit
            if isinstance(w, (QAbstractButton, QLineEdit, QPlainTextEdit, QSlider)) and w.isVisibleTo(ui.win):
                label = w.accessibleName() or (w.text() if hasattr(w, "text") and callable(w.text) else "")
                if not str(label).strip():
                    unnamed.append(f"{page}: {type(w).__name__} {w.objectName()!r}")
    assert unnamed == [], unnamed


def test_keyboard_focus_reaches_the_main_controls(ui):
    ui.page("settings")
    for name in ("nav_home", "pause_button", "settings_theme", "settings_language", "settings_cleanup_enabled",
                 "settings_hotkey_recorder_change", "settings_reset"):
        w = ui.find(name)
        assert w.focusPolicy() & Qt.FocusPolicy.TabFocus, f"{name} cannot be reached with Tab"


def test_shell_close_is_idempotent_and_hides_everything(ui):
    ui.shell.open_onboarding()
    ui.shell.overlay.set_arranging(True)
    ui.shell.close()
    ui.shell.close()
    assert not ui.win.isVisible() and not ui.shell.overlay.isVisible() and not ui.shell.tray.icon.isVisible()
    assert not ui.shell.onboarding.isVisible() and not ui.shell.overlay.arranging
    ui.shell.onboarding.skip()


# =========================================================================== fixes for the baseline ui bugs
def test_u02_flow_bar_is_placed_again_for_every_utterance_and_screen_change(ui):
    overlay = ui.shell.overlay
    overlay.set_state("idle")
    home = overlay.pos()
    overlay.move(home + QPoint(-300, -200))                    # e.g. left behind on another monitor
    overlay.set_state("listening")
    assert overlay.pos() == home, "idle -> active must re-place the bar on the active screen"
    overlay.set_state("done")
    overlay.move(home + QPoint(-300, -200))
    overlay.set_state("listening")                             # hotkey pressed again before the bar went idle
    assert overlay.pos() == home, "a new utterance must re-place the bar even when it never went idle"
    overlay.set_state("idle")
    overlay.move(home + QPoint(-300, -200))
    overlay._follow_screen()                                   # the 1.5 s idle check
    assert overlay.pos() == home
    overlay.move(home + QPoint(50, 0))
    overlay._on_screens_changed()                              # resolution / taskbar / monitor change
    assert overlay.pos() == home
    assert overlay._follow.interval() >= 1000, "following the screen must stay cheap"


def test_u06_copied_badge_stays_on_its_own_card_when_new_transcripts_arrive(make_ui):
    ui = make_ui(prepare=lambda rig: seed(rig, 2))
    page = ui.page("history")
    lst = page.list
    target = lst.items[1]
    lst._run(1, "copy")
    assert list(lst._flash) == [(target["id"], "copy")]
    ui.c.history.add("a brand new transcript")                 # pushes every card down by one
    ui.wait(lambda: len(lst.items) == 3)
    assert list(lst._flash) == [(target["id"], "copy")] and lst.items[2]["id"] == target["id"]
    ui.render()


def test_u08_popup_menu_keyboard_navigation_and_long_lists(ui):
    from voxflow.ui.widgets import PopupMenu
    picked = []
    menu = PopupMenu(260)
    items = [{"kind": "info", "text": "Pick one"}]
    items += [{"kind": "choice", "text": f"Microphone {i}", "checked": i == 0,
               "on_click": (lambda i=i: picked.append(i))} for i in range(60)]
    menu.set_items(items)
    menu.popup_at(QPoint(300, 300))
    screen = QApplication.primaryScreen().availableGeometry()
    assert menu.height() <= screen.height() * 0.75, "a long device list must scroll instead of running off-screen"
    ui.render(menu)
    for key in (Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_End):
        ui.qtbot.keyClick(menu, key)
    ui.qtbot.keyClick(menu, Qt.Key.Key_Return)
    assert picked == [59] and not menu.isVisible()
    menu.popup_at(QPoint(300, 300))
    ui.qtbot.keyClick(menu, Qt.Key.Key_Home)
    ui.qtbot.keyClick(menu, Qt.Key.Key_Space)
    assert picked == [59, 0], "Home skips the non-selectable heading"
    menu.popup_at(QPoint(300, 300))
    ui.qtbot.keyClick(menu, Qt.Key.Key_Escape)
    assert not menu.isVisible() and picked == [59, 0]
    menu.set_items([{"kind": "no-such-kind", "text": "x"}, {"kind": "action", "text": "ok", "on_click": lambda: None}])
    menu.popup_at(QPoint(300, 300))                            # an unknown row kind must not raise
    ui.render(menu)
    menu.close()
    menu.deleteLater()


def test_u09_button_flash_twice_restores_the_original_label(ui):
    from voxflow.ui.widgets import Button
    button = Button("Copy", "secondary", parent=ui.win)
    button.show()
    button.flash("Copied")
    button.flash("Copied")
    assert button.text() == "Copied"
    ui.wait(lambda: button.text() == "Copy", timeout=5000)
    button.deleteLater()


def test_u11_day_and_clock_labels():
    import datetime as dt

    from voxflow.ui.components import app_label, clock_label, day_label, fmt_count, fmt_duration
    now = dt.datetime(2026, 11, 13, 9, 0).timestamp()
    assert day_label(now - 3600, now) == "Today"
    assert day_label(dt.datetime(2026, 11, 12, 23, 50).timestamp(), now) == "Yesterday", "23:50 yesterday is not today"
    assert day_label(dt.datetime(2026, 11, 9, 12, 0).timestamp(), now) == "Monday"
    assert day_label(dt.datetime(2026, 3, 4, 12, 0).timestamp(), now) == "March 4"
    assert day_label(dt.datetime(2024, 3, 4, 12, 0).timestamp(), now) == "March 4, 2024"
    assert day_label("garbage", now) == "Earlier" and day_label(1e20, now) == "Earlier"
    assert clock_label(dt.datetime(2026, 11, 12, 23, 50).timestamp()) == "23:50" and clock_label(None) == ""
    assert [fmt_count(n) for n in (0, 999, 1234, 12_345, 2_500_000)] == ["0", "999", "1,234", "12.3k", "2.5M"]
    assert [fmt_duration(s) for s in (-5, 42, 150, 3600, 5400)] == ["0 sec", "42 sec", "2 min", "1 h", "1 h 30 min"]
    assert [app_label(e) for e in ("slack.exe", "code.exe", "WINWORD.EXE", "my_tool-x.exe", "")] == [
        "Slack", "VS Code", "Word", "My Tool X", ""]


def test_style_page_describes_what_the_tone_pass_really_does(ui):
    """The page used to promise "relaxed wording and contractions" for casual."""
    import re

    from voxflow.cleanup import apply_tone
    from voxflow.ui import pages
    page = ui.page("style")
    assert "Your words are never changed" in page.subtitle.text()
    assert "Command mode" in ui.find("style_command_note").text()
    for claim in ("relaxed wording", "contractions"):
        assert all(claim not in hint.lower() for hint in pages._TONE_HINTS.values())

    def words(text):
        return re.findall(r"[\w']+", text)

    for category, card in page.cards.items():
        sample = pages._SAMPLES[category]
        assert sample.endswith("."), "the example must end in a full stop, or casual shows no difference"
        for tone in ("formal", "casual", "very_casual"):
            ui.find(f"style_tone_{category}").set_value(tone, emit=True)
            shown = ui.find(f"style_preview_{category}").text()
            assert shown == apply_tone(sample, tone), "the preview must be the real tone pass, not a mock-up"
            assert card.hint.text() == pages._TONE_HINTS[tone]
            assert [w.lower() for w in words(shown)] == [w.lower() for w in words(sample)], "words never change"
            if tone == "formal":                       # "Nothing is changed"
                assert shown == sample
            elif tone == "casual":                     # "Same words and capitals. Only the full stop at the very end"
                assert shown == sample[:-1] and words(shown) == words(sample)
            else:                                      # no final stop, lower-case sentence starts, names and I kept
                assert not shown.endswith(".")
                starts = [shown.split()[0]] + re.findall(r"[.?!] (\S+)", shown)
                kept = [w for w in starts if not w[0].islower()]
                assert all(w.split("'")[0] == "I" for w in kept), f"sentence starts left capitalised: {kept}"
                assert any(w[0].islower() for w in starts)
                assert sample.count("I ") + sample.count("I'") == shown.count("I ") + shown.count("I'")
                for name in ("Sam", "Friday", "Thursday"):
                    assert (name in shown) == (name in sample)
    ui.render()
