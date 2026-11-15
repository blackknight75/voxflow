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


def test_with_history_off_nothing_is_learned_from_dictation(rig):
    """Auto-learn would store a fragment of what was said ("get hub") in dictionary.json: history off pauses it,
    while the dictionary itself (manual entries, its use in clean-up) keeps working."""
    rig.c.set("save_history", False)
    assert rig.c.get("auto_learn") is True
    rig.cleaner.rewrites["i pushed it to get hub"] = "I pushed it to GitHub."
    rig.dictate("i pushed it to get hub")
    rig.wait(lambda: rig.c.stats.summary()["sessions"] == 1)
    rig.settle()
    assert rig.spy.of("dictionary_suggestion") == [] and rig.c.dictionary.suggestions() == []
    rig.c.dictionary.flush()
    assert files_with(rig.data_dir, "i pushed it to get hub") == []
    assert not any(b"get hub" in p.read_bytes().lower() for p in rig.data_dir.rglob("*") if p.is_file())
    rig.c.dictionary.add("VoxFlow", ["vox flow"])                 # manual edits are not affected
    assert [e["term"] for e in rig.c.dictionary.list()] == ["VoxFlow"]

    rig.c.set("save_history", True)                                 # history back on: learning resumes
    rig.dictate("i pushed it to get hub")
    rig.wait(lambda: rig.spy.of("dictionary_suggestion"))
    assert rig.c.dictionary.suggestions()[0]["term"] == "GitHub"


def test_clearing_history_also_deletes_its_backup_copies(tmp_path):
    tmp_path = tmp_path / "profile"
    tmp_path.mkdir()
    """``history.v1.json`` (kept when an old file is migrated) and ``history.corrupt.json`` (kept when the file
    could not be read) hold full transcripts too, so the delete offer and Clear all must remove them."""
    from voxflow import store
    (tmp_path / "history.json").write_text('[{"text": "%s", "ts": 1700000000}]' % SECRET, encoding="utf-8")
    h = store.History(tmp_path)                                   # a version 1 file: migrated, a copy is kept
    assert (tmp_path / "history.v1.json").exists() and h.count() == 1
    (tmp_path / "history.corrupt.json").write_text(SECRET, encoding="utf-8")
    (tmp_path / "dictionary.corrupt.json").write_text("{}", encoding="utf-8")     # other stores' copies stay
    h.clear()
    assert files_with(tmp_path, SECRET) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dictionary.corrupt.json", "history.json"]
    h.clear()                                                     # nothing left to delete: no error


def test_a_backup_copy_that_cannot_be_deleted_is_logged_and_does_not_stop_the_clear(tmp_path, monkeypatch, caplog):
    from pathlib import Path

    from voxflow import store
    h = store.History(tmp_path)
    h.add_entry(store.make_entry(SECRET, SECRET))
    real = Path.unlink

    def locked(self, missing_ok=False):
        if self.name == "history.v1.json":
            raise PermissionError(13, "Access is denied", str(self))
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    with caplog.at_level(logging.WARNING, logger="voxflow"):
        h.clear()
    assert h.count() == 0 and "could not delete history.v1.json: Access is denied" in caplog.text


# =========================================================================== crash reports never quote dictation
def _fail_with_cause():
    try:
        raise KeyError(SECRET)
    except KeyError as inner:
        raise RuntimeError(f"cleaning {SECRET!r} failed") from inner


def test_redacted_traceback_follows_the_chain_without_messages():
    try:
        _fail_with_cause()
    except RuntimeError as exc:
        text = sysmod.redacted_traceback(exc)
    assert "KeyError at " in text and "RuntimeError at " in text and "direct cause" in text
    assert text.index("KeyError") < text.index("RuntimeError"), "oldest first, as Python prints it"
    assert "bank" not in text
    try:
        try:
            raise KeyError(SECRET)
        except KeyError:
            raise ValueError(SECRET)                              # noqa: B904 - implicit context on purpose
    except ValueError as exc:
        text = sysmod.redacted_traceback(exc)
    assert "During handling of the above exception" in text and "bank" not in text


def test_an_uncaught_exception_leaves_no_words_in_the_crash_report_or_the_log(tmp_path, clean_logging,
                                                                               monkeypatch):
    sysmod.setup_logging(tmp_path, logging.DEBUG)
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)     # the console print is not a file
    sysmod.install_crash_handler(tmp_path)
    try:
        _fail_with_cause()
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    th = threading.Thread(target=_fail_with_cause, name="pipeline")
    th.start()
    th.join()
    sysmod.shutdown_logging()
    reports = sorted((tmp_path / "logs").glob("crash-*.log"))
    assert reports, "a crash report is still written"
    for report in reports:
        text = report.read_text(encoding="utf-8")
        assert "RuntimeError at test_history_privacy.py:" in text and "in _fail_with_cause" in text
    log = (tmp_path / "logs" / sysmod.LOG_NAME).read_text(encoding="utf-8")
    assert "uncaught exception in main thread" in log and "uncaught exception in thread pipeline" in log
    assert "RuntimeError at " in log
    assert files_with(tmp_path, SECRET) == [] and "bank" not in log.lower()


def test_a_crash_hook_given_no_exception_object_still_reports(tmp_path):
    path = sysmod.write_crash_report(tmp_path, RuntimeError, None, None)
    assert path is not None and path.read_text(encoding="utf-8").rstrip().endswith("RuntimeError")
    path = sysmod.write_crash_report(tmp_path / "b", None, None, None)
    assert path is not None and "Exception" in path.read_text(encoding="utf-8")
    try:
        _raise(KeyError(SECRET))
    except KeyError as exc:
        tb = exc.__traceback__
    path = sysmod.write_crash_report(tmp_path / "c", KeyError, None, tb)    # a type and a traceback only
    text = path.read_text(encoding="utf-8")
    assert "KeyError at test_history_privacy.py:" in text and "in _raise" in text and "bank" not in text


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
