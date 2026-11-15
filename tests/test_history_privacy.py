"""History off means no dictated text on disk, and the log never holds dictated text at all.

With "Keep history" off a dictation is pasted and counted (stats), but no file under the profile may contain its
words: not ``history.json``, not ``voxflow.log``, not any other store. The log is checked with history on too:
the failure paths that could quote the words (an exception message from the clean-up, the pipeline, the store)
log the type and place of the error only (``system.redacted_exc``)."""
from __future__ import annotations

import logging
import sys
import threading

import pytest

from tests.controller_rig import make_rig
from voxflow import cleanup as cleanup_mod, system as sysmod

SECRET = "my bank pin is four four nine two"


@pytest.fixture
def clean_logging():
    """Leave the process-wide logging state as we found it."""
    root = logging.getLogger("voxflow")
    handlers, level = list(root.handlers), root.level
    hooks = (sys.excepthook, threading.excepthook)
    yield
    sysmod.shutdown_logging()
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    sys.excepthook, threading.excepthook = hooks


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch, clean_logging):
    r = make_rig(tmp_path, qtbot, monkeypatch)
    sysmod.setup_logging(r.data_dir, logging.DEBUG)          # the real rotating log file, at its chattiest
    yield r
    r.close()


def files_with(root, words: str) -> list[str]:
    """Every file below ``root`` whose bytes contain any 3-word run of ``words`` (case-insensitive)."""
    parts = words.lower().split()
    needles = [" ".join(parts[i:i + 3]).encode() for i in range(len(parts) - 2)]
    for h in logging.getLogger("voxflow").handlers:
        h.flush()
    hits = []
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes().lower()
            if any(n in data for n in needles):
                hits.append(path.name)
    return hits


def log_text(rig) -> str:
    for h in logging.getLogger("voxflow").handlers:
        h.flush()
    return (sysmod.log_dir(rig.data_dir) / sysmod.LOG_NAME).read_text(encoding="utf-8")


# =========================================================================== redaction helpers
def _raise(exc):
    raise exc


def test_redacted_exc_names_the_error_and_place_but_never_the_message():
    try:
        _raise(KeyError(SECRET))
    except KeyError as exc:
        text = sysmod.redacted_exc(exc)
    assert text.startswith("KeyError at test_history_privacy.py:") and text.endswith(" in _raise")
    assert "bank" not in text
    assert sysmod.redacted_exc(ValueError(SECRET)) == "ValueError", "never raised: no place to name"


def test_redacted_exc_keeps_the_reason_of_a_disk_error():
    text = sysmod.redacted_exc(PermissionError(13, "Access is denied", "C:/x/history.json"))
    assert text == "PermissionError: Access is denied C:/x/history.json"


def test_redacted_traceback_keeps_the_stack_and_drops_the_message():
    try:
        _raise(RuntimeError(SECRET))
    except RuntimeError as exc:
        text = sysmod.redacted_traceback(exc)
    assert text.startswith("Traceback (most recent call last):\n") and "in _raise" in text
    assert text.splitlines()[-1].startswith("RuntimeError at test_history_privacy.py:")
    assert "bank" not in text
    assert sysmod.redacted_traceback(ValueError("x")).endswith("ValueError")


# =========================================================================== history off: nothing on disk
def test_with_history_off_a_dictation_leaves_no_words_on_disk(rig):
    rig.c.set("save_history", False)
    rig.dictate(SECRET)
    rig.wait(lambda: rig.c.stats.summary()["sessions"] == 1)  # filed (stats) after the paste
    assert rig.injector.pasted and "bank" in rig.injector.pasted[0].lower()
    assert rig.c.history.count() == 0
    assert files_with(rig.data_dir, SECRET) == []


def test_history_off_survives_a_restart_and_keeps_saving_nothing(tmp_path, qtbot, monkeypatch, clean_logging):
    first = make_rig(tmp_path, qtbot, monkeypatch)
    first.c.set("save_history", False)
    first.close()
    again = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert again.c.get("save_history") is False, "the setting is stored with the other settings"
        again.dictate(SECRET)
        again.wait(lambda: again.c.stats.summary()["sessions"] == 1)
        assert again.c.history.count() == 0 and files_with(again.data_dir, SECRET) == []
    finally:
        again.close()


# =========================================================================== the log never quotes dictation
@pytest.mark.parametrize("save_history", [False, True])
def test_a_failing_cleanup_is_logged_without_the_words(rig, save_history):
    rig.c.set("save_history", save_history)
    rig.cleaner.fail_clean = KeyError(SECRET)                # e.g. a lookup that names the word it missed
    rig.dictate(SECRET)
    rig.wait(lambda: rig.c.stats.summary()["sessions"] == 1)
    log = log_text(rig)
    assert "cleanup raised, pasting the raw transcript: KeyError at " in log
    assert "bank" not in log.lower()
    if not save_history:
        assert files_with(rig.data_dir, SECRET) == []


@pytest.mark.parametrize("save_history", [False, True])
def test_a_failing_pipeline_job_is_logged_without_the_words(rig, monkeypatch, save_history):
    rig.c.set("save_history", save_history)

    def explode(text):
        raise ValueError(f"cannot expand {text!r}")

    monkeypatch.setattr(rig.c.snippets, "expand", explode)
    rig.dictate(SECRET)
    rig.wait(lambda: rig.c.state in ("error", "idle"))
    log = log_text(rig)
    assert "pipeline job 'dictate' failed" in log and "ValueError at " in log and "in explode" in log
    assert "bank" not in log.lower()
    assert files_with(rig.data_dir, SECRET) == []          # nothing was filed: the job failed before


def test_a_transcript_that_cannot_be_filed_is_logged_without_the_words(rig, monkeypatch):
    def refuse(entry):
        raise ValueError(f"bad entry {entry['text']!r}")

    monkeypatch.setattr(rig.c.history, "add_entry", refuse)
    rig.dictate(SECRET)
    rig.wait(lambda: "could not file the transcript" in log_text(rig))
    log = log_text(rig)
    assert "ValueError at " in log and "bank" not in log.lower()


def test_language_cleanup_failures_are_logged_without_the_words(caplog, monkeypatch):
    def explode(*_a, **_k):
        raise KeyError(SECRET)

    monkeypatch.setattr(cleanup_mod, "smart_format", explode, raising=False)
    caplog.set_level(logging.DEBUG, logger="voxflow")
    c = cleanup_mod.Cleaner.__new__(cleanup_mod.Cleaner)
    monkeypatch.setattr(cleanup_mod.langs, "get", explode)
    c.last_info = {}
    out = c._clean_other(SECRET, "de", False)
    assert out and all("bank" not in r.getMessage().lower() for r in caplog.records)
    assert any("KeyError at " in r.getMessage() for r in caplog.records)
