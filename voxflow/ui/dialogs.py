"""Native file pickers behind three functions, so tests can replace them and never block on a real dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog, QWidget


def _documents() -> Path:
    loc = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(loc) if loc else Path.home()


def ask_save_path(parent: QWidget, title: str, default_name: str, filters: str) -> str:
    """Path chosen by the user, or "" when cancelled. ``filters`` e.g. "Text (*.txt);;JSON (*.json)"."""
    path, _selected = QFileDialog.getSaveFileName(parent, title, str(_documents() / default_name), filters)
    return path or ""


def ask_open_path(parent: QWidget, title: str, filters: str) -> str:
    path, _selected = QFileDialog.getOpenFileName(parent, title, str(_documents()), filters)
    return path or ""


def ask_folder(parent: QWidget, title: str, start: str = "") -> str:
    """Existing folder chosen by the user, or "" when cancelled."""
    return QFileDialog.getExistingDirectory(parent, title, start or str(_documents())) or ""
