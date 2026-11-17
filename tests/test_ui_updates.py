"""Settings > Updates, the "new version" dialog, the notification and the setup guide's update line, on the in-memory
FakeController (the twin of Controller.update_info / check_updates / install_update)."""
from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import Signals, demo, flush, make_demo  # noqa: F401 - fixtures
from voxflow.ui import updates as UP
from voxflow.ui.fakes import FakeController
from voxflow.ui.onboarding import STEPS, Onboarding


def section(demo):
    page = demo.page("settings")
    return page, page.updates


def info(**kw):
    base = {"version": "0.2.0", "mode": "notify", "when": "quit", "time": "03:00", "kind": "installer",
            "can_install": True, "managed": "", "phase": "idle", "progress": None, "error": "", "last_check": 0,
            "last_ok": True, "last_result": "", "available": None, "installable": False, "skipped": False,
            "install_on_quit": False, "downloaded": False}
    base.update(kw)
    return base


REL = {"version": "0.3.0", "notes": "x", "page": "https://github.com/blackknight75/voxflow/releases/tag/v0.3.0"}


# =========================================================================== wording
def test_the_status_line_says_what_is_going_on():
    now = time.time()
    assert UP.status_text(info(phase="checking")) == ("Checking for updates…", "TEXT_DIM")
    assert UP.status_text(info(phase="downloading", progress=0.42, available=REL))[0] == \
        "Downloading VoxFlow 0.3.0… 42%"
    assert UP.status_text(info(phase="downloading", available=REL))[0] == "Downloading VoxFlow 0.3.0…"
    assert UP.status_text(info(phase="waiting", available=REL)) == (
        "VoxFlow will restart to install 0.3.0 as soon as you finish dictating.", "AMBER")
    assert UP.status_text(info(phase="restarting", available=REL))[1] == "AMBER"
    text, role = UP.status_text(info(last_check=now, last_result="VoxFlow is up to date."))
    assert text.startswith("VoxFlow is up to date. Last checked today at ") and role == "TEXT_DIM"
    assert UP.status_text(info(last_check=now, last_ok=False, last_result="GitHub could not be reached (x)."))[1] \
        == "AMBER"
    assert UP.status_text(info(mode="off")) == (
        "Automatic checks are off: VoxFlow asks GitHub only when you press Check now.", "TEXT_DIM")
    assert UP.status_text(info(mode="off", last_check=now, last_result="VoxFlow is up to date."))[0].startswith(
        "Automatic checks are off: VoxFlow asks GitHub only when you press Check now. VoxFlow is up to date. "
        "Last checked today")
    assert UP.status_text(info())[0] == "Not checked yet."
    assert "You skipped this version." in UP.status_text(info(available=REL, skipped=True))[0]
    assert "when you quit VoxFlow" in UP.status_text(info(available=REL, install_on_quit=True))[0]
    assert "installed at 03:00" in UP.status_text(info(available=REL, mode="auto", when="time", downloaded=True))[0]
    managed = UP.status_text(info(managed="Updates are managed by Scoop: run scoop update voxflow."))[0]
    assert managed.endswith("Updates are managed by Scoop: run scoop update voxflow.")
    assert UP.status_text(info(error="Waiting for a connection that is not metered."))[1] == "AMBER"
    assert UP.checked_text(now - 3 * 86400).startswith("Last checked ") and UP.checked_text(0) == "Not checked yet."


def test_the_time_menu_lists_every_hour_and_a_stored_odd_time():
    assert len(UP.time_options("03:00")) == 24 and UP.time_options("03:00")[3] == ("03:00", "03:00")
    odd = [v for v, _ in UP.time_options("04:30")]
    assert len(odd) == 25 and odd.index("04:30") == 5


# =========================================================================== Settings > Updates
def test_the_section_shows_the_version_the_mode_and_the_last_check(demo):
    page, sec = section(demo)
    assert sec.mode.label() == "Notify me, I choose"
    assert [o[1] for o in sec.mode.options] == ["Install automatically", "Notify me, I choose", "Off (never check)"]
    assert sec.status_row.title.text() == "Version 0.2.0"
    assert sec.status_row.sub.text().startswith("VoxFlow is up to date. Last checked today at")
    assert not sec.when_row.isVisible(), "the install timing only applies to automatic mode"
    assert not sec.btn_see.isVisible() and sec.btn_check.isEnabled()
    assert "anonymous" in sec.mode_row.sub.text()
    assert "Off means no automatic request: VoxFlow asks only when you press Check now." in sec.mode_row.sub.text()
    page.show_section("updates")
    demo.render()


def test_automatic_mode_shows_when_to_install_and_a_time_menu(demo):
    _page, sec = section(demo)
    sec.mode._pick("auto")
    flush()
    assert demo.c.config.update_mode == "auto" and sec.when_row.isVisible() and not sec.at.isVisible()
    assert sec.when.label() == "When I quit VoxFlow"
    sec.when._pick("time")
    flush()
    assert demo.c.config.update_install_when == "time" and sec.at.isVisible() and sec.at.label() == "03:00"
    sec.at._pick("05:00")
    flush()
    assert demo.c.config.update_time == "05:00" and sec.at.label() == "05:00"
    demo.c.demo_set_update(available=False, kind="scoop")
    flush()
    assert not sec.when_row.isVisible(), "a Scoop copy cannot install itself"
    assert sec.status_row.sub.text().endswith("Updates are managed by Scoop: run scoop update voxflow.")


def test_off_keeps_check_now_and_says_it_is_the_only_way_to_ask(demo):
    _page, sec = section(demo)
    sec.mode._pick("off")
    flush()
    assert demo.c.config.update_mode == "off" and sec.btn_check.isEnabled()
    assert sec.btn_check.toolTip() == "Asks GitHub once, now. Automatic checks stay off."
    assert sec.status_row.sub.text().startswith(
        "Automatic checks are off: VoxFlow asks GitHub only when you press Check now.")
    sec.btn_check.click()
    flush()
    assert demo.c.config.update_mode == "off" and sec.status_row.sub.text().startswith(
        "Automatic checks are off: VoxFlow asks GitHub only when you press Check now. VoxFlow 0.3.0 is available.")
    sec.mode._pick("notify")
    flush()
    assert sec.btn_check.toolTip() == "Asks GitHub now instead of waiting for the daily check."


def test_check_now_finds_the_update_and_offers_the_dialog(demo):
    page, sec = section(demo)
    seen = Signals(available=demo.c.update_available)
    sec.btn_check.click()
    flush()
    assert seen.available and sec.btn_see.isVisible()
    assert sec.status_row.sub.text().startswith("VoxFlow 0.3.0 is available.")
    assert "VoxFlow 0.3.0 is available." in demo.toasts(), "the shell's toast, with See update"
    sec.btn_see.click()
    flush()
    dlg = demo.find("update_dialog")
    assert dlg.title.text() == "VoxFlow 0.3.0 is available"
    notes = demo.find("update_notes")
    assert notes.isReadOnly() and notes.toPlainText().startswith("Added\n• App updates")
    assert "1.8 GB" in demo.find("update_hint").text()
    demo.render()


def test_the_dialog_choices(demo):
    page, _sec = section(demo)
    demo.c.check_updates()
    flush()
    page.open_update()
    demo.find("update_dialog_quit").click()
    flush()
    assert demo.c.update_info()["install_on_quit"] and not demo.find_all("update_dialog")
    assert "VoxFlow 0.3.0 will be installed when you quit VoxFlow." in demo.toasts()
    page.open_update()
    demo.find("update_dialog_now").click()
    flush()
    assert demo.c.update_info()["phase"] == "waiting"
    page.open_update()
    demo.find("update_dialog_later").click()
    flush()
    assert "VoxFlow will remind you tomorrow." in demo.toasts()
    page.open_update()
    demo.find("update_dialog_skip").click()
    flush()
    assert demo.c.update_info()["skipped"]


def test_the_dialog_shows_a_refusal_instead_of_closing(demo):
    page, _sec = section(demo)
    demo.c.check_updates()
    flush()
    dlg = page.open_update()
    demo.c.demo_set_update(kind="winget")                   # changed under the open dialog
    demo.find("update_dialog_now").click()
    flush()
    assert dlg.error.isVisible() and "winget" in dlg.error.text()


def test_a_managed_copy_offers_the_releases_page_instead_of_installing(demo, monkeypatch):
    opened = []
    monkeypatch.setattr(UP.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    page, _sec = section(demo)
    demo.c.demo_set_update(kind="scoop")
    flush()
    page.open_update()
    assert not demo.find_all("update_dialog_now") and not demo.find_all("update_dialog_quit")
    assert demo.find("update_hint").text() == "Updates are managed by Scoop: run scoop update voxflow."
    demo.find("update_dialog_page").click()
    flush()
    assert opened == ["https://github.com/blackknight75/voxflow/releases/tag/v0.3.0"]


def test_without_an_update_the_dialog_says_up_to_date(demo):
    page, _sec = section(demo)
    assert page.open_update() is None and "VoxFlow is up to date." in demo.toasts()


def test_a_search_does_not_bring_back_rows_of_another_mode(demo):
    page, sec = section(demo)
    page.find.setText("install")
    flush()
    assert sec.when_row.isVisible(), "a search shows every matching row"
    page.find.setText("")
    flush()
    assert not sec.when_row.isVisible()


def test_download_progress_shows_a_bar(demo):
    _page, sec = section(demo)
    demo.c.demo_set_update(phase="downloading", progress=0.5)
    flush()
    assert sec.bar.isVisible() and sec.bar.value == 0.5 and not sec.btn_check.isEnabled()
    demo.c.demo_set_update(phase="downloading", progress=None)
    flush()
    assert sec.bar.value is None
    demo.c.demo_set_update(phase="idle")
    flush()
    assert not sec.bar.isVisible()


# =========================================================================== notification and restart
def test_a_hidden_window_gets_a_tray_notification_that_opens_the_dialog(make_demo, monkeypatch):
    d = make_demo(show_window=False)
    shown = []
    monkeypatch.setattr(d.shell.tray, "notify", lambda level, msg, action="": shown.append((level, msg, action)))
    d.c.check_updates()
    flush()
    assert shown == [("info", "VoxFlow 0.3.0 is available. Click to see what is new and choose when to install it.",
                      "update")]
    d.shell.run_action("update")
    flush()
    assert d.win.isVisible() and d.find("update_dialog").isVisible()


def test_update_available_without_a_release_is_ignored(demo, monkeypatch):
    shown = []
    monkeypatch.setattr(demo.shell.tray, "notify", lambda *a, **k: shown.append(a))
    demo.shell._on_update_available({"available": None})
    assert shown == [] and "VoxFlow 0.3.0 is available." not in demo.toasts()


def test_quit_from_the_tray_is_the_users_own_and_a_restart_quits(demo, monkeypatch):
    quits = []
    monkeypatch.setattr(QApplication.instance(), "quit", lambda: quits.append(1))
    demo.shell.tray.quit_requested.emit()
    assert demo.c.quit_prepared and quits == [1]
    demo.c.restart_requested.emit()
    assert len(quits) == 2


# =========================================================================== the setup guide
def test_the_setup_guide_mentions_the_update_check_and_links_to_the_setting(qtbot):
    c = FakeController()
    ob = Onboarding(c)
    qtbot.addWidget(ob)
    asked = Signals(settings=ob.settings_requested, finished=ob.finished)
    ob.go(STEPS.index("done"))
    assert ob.update_note.text() == ("Updates: once a day VoxFlow asks GitHub, anonymously, whether a new version is "
                                     "out, and asks you first.")
    c.config.update_mode = "auto"
    ob._fill_tips()
    assert ob.update_note.text().endswith("and installs it for you.")
    c.config.update_mode = "off"
    ob._fill_tips()
    assert "only when you press Check now" in ob.update_note.text()
    ob.btn_updates.click()
    assert asked.settings == [("updates",)] and asked.finished == [(True,)] and c.config.onboarding_done


def test_the_guides_link_opens_settings_at_updates(make_demo):
    d = make_demo(show_window=False)
    ob = d.shell.open_onboarding()
    ob.go(STEPS.index("done"))
    ob.btn_updates.click()
    flush()
    assert d.win.isVisible() and d.win.current == "settings"


# =========================================================================== the twin itself
def test_the_fake_controller_refuses_like_the_real_one():
    c = FakeController()
    with pytest.raises(ValueError, match="no newer version"):
        c.install_update("now")
    c.check_updates()
    with pytest.raises(ValueError, match="install now or when you quit"):
        c.install_update("later")
    with pytest.raises(ValueError, match="03:00"):
        c.set("update_time", "25:00")
    with pytest.raises(ValueError, match="not a valid choice"):
        c.set("update_mode", "sometimes")
    c.set("update_time", "4:30")
    assert c.config.update_time == "04:30"
    c.skip_update()
    assert c.update_info()["skipped"]
    c.demo_set_update(available=False)
    c.skip_update()                                        # nothing to skip: no error
    c.demo_set_update(available={**REL, "installer_url": ""})
    assert not c.update_info()["installable"]


# =========================================================================== keyboard, screen readers, fit
def _dialog(demo, monkeypatch):
    page, _sec = section(demo)
    demo.c.check_updates()
    flush()
    calls = []
    monkeypatch.setattr(demo.c, "remind_update_later", lambda: calls.append("later"))
    dlg = page.open_update()
    flush()
    return page, dlg, calls


def test_the_dialog_opens_on_remind_me_later_and_a_stray_enter_does_not_install(demo, monkeypatch):
    """Found on the real desktop: it opened focused on Install now, so one Enter restarted VoxFlow."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    page, dlg, calls = _dialog(demo, monkeypatch)
    win = dlg.window()
    assert win.focusWidget() is demo.find("update_dialog_later")
    QTest.keyClick(demo.find("update_notes"), Qt.Key.Key_Return)          # Enter where no button has the focus
    flush()
    assert calls == ["later"] and demo.c.update_info()["phase"] == "idle" and not demo.find_all("update_dialog")


def test_esc_and_a_click_outside_mean_remind_me_later(demo, monkeypatch):
    """Esc / a click outside used to just close: the version was never announced again (notified stayed set)."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    page, dlg, calls = _dialog(demo, monkeypatch)
    QTest.keyClick(dlg.window().focusWidget(), Qt.Key.Key_Escape)
    flush()
    assert calls == ["later"] and not demo.find_all("update_dialog")
    dlg = page.open_update()
    flush()
    QTest.mouseClick(dlg, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(3, 3))
    flush()
    assert calls == ["later", "later"] and not demo.find_all("update_dialog")


def test_enter_on_a_focused_button_presses_that_button(demo, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    page, dlg, calls = _dialog(demo, monkeypatch)
    QTest.keyClick(demo.find("update_dialog_skip"), Qt.Key.Key_Return)
    flush()
    assert demo.c.update_info()["skipped"] and calls == []


def test_a_plain_confirm_dialog_still_closes_on_esc(demo):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from voxflow.ui.components import confirm
    done = []
    win = demo.page("settings").win
    m = win.open_modal(confirm(win.panel, "Delete?", "Sure?", "Delete", lambda: done.append(1)))
    flush()
    QTest.keyClick(m, Qt.Key.Key_Escape)
    flush()
    assert done == [] and not demo.find_all("confirm")


def test_the_dialog_widens_rather_than_squeeze_a_button(demo, monkeypatch):
    """Found at 150 % scale: the four buttons needed more than 540 px and Install when I quit was squeezed."""
    from PySide6.QtCore import QSize

    from voxflow.ui import widgets as W
    real = W.Button.sizeHint
    monkeypatch.setattr(W.Button, "sizeHint", lambda self: real(self) + QSize(40, 0))   # a wider font
    page, dlg, _calls = _dialog(demo, monkeypatch)
    assert dlg.card.width() > 540
    for b in dlg.findChildren(W.Button):
        assert b.width() >= b.sizeHint().width(), b.text()


def test_dropdowns_tell_screen_readers_the_choice(demo):
    """The choice and the menu rows are painted: UI Automation reads them from the description / the menu's name."""
    _page, sec = section(demo)
    assert sec.mode.accessibleDescription() == "Notify me, I choose"
    sec.mode.set_value("off")
    assert sec.mode.accessibleDescription() == "Off (never check)"
    menu = sec.mode.open_menu()
    flush()
    assert menu.accessibleName() == "Off (never check), selected"
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(menu, Qt.Key.Key_Up)
    assert menu.accessibleName() == "Notify me, I choose"
    QTest.keyClick(menu, Qt.Key.Key_Return)
    flush()
    assert sec.mode.accessibleDescription() == "Notify me, I choose"
    menu.set_items([{"kind": "choice", "text": "x"}])
    assert menu.accessibleName() == "Menu"
