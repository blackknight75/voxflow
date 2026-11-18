"""The "Type while you speak (hands-free)" setting: the toggle, its plumbing and the fact that it never syncs."""
from __future__ import annotations

import json

from PySide6.QtWidgets import QApplication

from tests.test_ui import make_ui, ui  # noqa: F401  (fixtures)
from voxflow import config as cfgmod
from voxflow import sync as syncmod

KEY = "phrase_streaming"


def test_phrase_streaming_toggle_exists_defaults_off_and_has_an_accessible_name(ui):
    ui.page("settings")
    toggle = ui.find(f"settings_{KEY}")
    assert toggle.isChecked() is False and ui.c.get(KEY) is False
    assert toggle.accessibleName() == "Type while you speak (hands-free)"


def test_toggling_reaches_the_controller_and_is_saved(ui):
    ui.page("settings")
    toggle = ui.find(f"settings_{KEY}")
    toggle.click()
    assert ui.c.get(KEY) is True
    saved = json.loads((ui.rig.data_dir / "config.json").read_text(encoding="utf-8"))
    assert saved[KEY] is True


def test_the_key_is_local_not_synced():
    assert KEY in syncmod.LOCAL_SETTINGS and KEY not in syncmod.portable_settings()


def test_default_is_off():
    assert cfgmod.Config().phrase_streaming is False and cfgmod.defaults()[KEY] is False


def test_reset_to_defaults_turns_it_off(ui):
    ui.page("settings")
    toggle = ui.find(f"settings_{KEY}")
    toggle.click()
    assert toggle.isChecked() is True
    ui.c.reset_settings()
    QApplication.processEvents()
    assert ui.c.get(KEY) is False and toggle.isChecked() is False


def test_the_settings_text_mentions_hands_free_and_undo(ui):
    from PySide6.QtWidgets import QLabel
    page = ui.page("settings")
    texts = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "hands-free" in texts and "Undo removes only the last phrase" in texts
