"""``voxflow.ui.dialogs``: the three native pickers. The static ``QFileDialog`` functions are replaced, so no
dialog opens; the tests check what the user would be shown (title, start folder, filters) and what comes back."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog, QWidget

from voxflow.ui import dialogs


@pytest.fixture
def parent(qtbot):
    w = QWidget()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def docs(monkeypatch, tmp_path):
    monkeypatch.setattr(QStandardPaths, "writableLocation", staticmethod(lambda _loc: str(tmp_path)))
    return tmp_path


def _answer(monkeypatch, name: str, result):
    calls: list[tuple] = []

    def fake(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(QFileDialog, name, staticmethod(fake))
    return calls


def test_save_dialog_starts_in_documents_with_the_suggested_name(monkeypatch, parent, docs):
    calls = _answer(monkeypatch, "getSaveFileName", (str(docs / "out.txt"), "Text (*.txt)"))
    assert dialogs.ask_save_path(parent, "Export", "voxflow.txt", "Text (*.txt);;JSON (*.json)") == str(docs / "out.txt")
    assert calls == [(parent, "Export", str(docs / "voxflow.txt"), "Text (*.txt);;JSON (*.json)")]
    _answer(monkeypatch, "getSaveFileName", ("", ""))
    assert dialogs.ask_save_path(parent, "Export", "voxflow.txt", "Text (*.txt)") == "", "cancelled"


def test_open_dialog_starts_in_documents(monkeypatch, parent, docs):
    calls = _answer(monkeypatch, "getOpenFileName", (str(docs / "in.json"), "JSON (*.json)"))
    assert dialogs.ask_open_path(parent, "Import", "JSON (*.json)") == str(docs / "in.json")
    assert calls == [(parent, "Import", str(docs), "JSON (*.json)")]
    _answer(monkeypatch, "getOpenFileName", (None, None))
    assert dialogs.ask_open_path(parent, "Import", "JSON (*.json)") == ""


def test_folder_dialog_starts_where_asked_or_in_documents(monkeypatch, parent, docs):
    calls = _answer(monkeypatch, "getExistingDirectory", str(docs / "sync"))
    assert dialogs.ask_folder(parent, "Choose the sync folder", "D:/Shared") == str(docs / "sync")
    assert dialogs.ask_folder(parent, "Choose the sync folder") == str(docs / "sync")
    assert calls == [(parent, "Choose the sync folder", "D:/Shared"), (parent, "Choose the sync folder", str(docs))]
    _answer(monkeypatch, "getExistingDirectory", "")
    assert dialogs.ask_folder(parent, "Choose the sync folder") == ""


def test_documents_falls_back_to_the_home_folder(monkeypatch):
    monkeypatch.setattr(QStandardPaths, "writableLocation", staticmethod(lambda _loc: ""))
    assert dialogs._documents() == Path.home()
