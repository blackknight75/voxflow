"""Regression tests for the bugs found with real speech and a real first-run download (round 3).

Each input below is what the real recogniser produced (``scripts/speech_matrix.py``) or what a real Hugging Face
download leaves on disk on Windows without symlink rights (``scripts/system_checks.py download``).
"""
from __future__ import annotations

import pytest

from voxflow import cleanup, modelhub


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


# ---------------------------------------------------------------- C-61: amount glued to the previous word
def test_c61_currency_amount_glued_to_the_previous_word_gets_its_space():
    assert cleanup.rule_clean("The total is$23.50.") == "The total is $23.50."


def test_c61_fix_must_not_touch_amounts_that_are_already_fine():
    for text in ("The total is $23.50.", "It costs $5.", "It costs $5 and $7."):
        assert cleanup.rule_clean(text) == text


# ---------------------------------------------------------------- C-62: the recogniser's own e-mail / time forms
def test_c62_email_address_as_the_recogniser_writes_it():
    out = cleanup.smart_format(cleanup.rule_clean("Please email john.smith at example.com"))
    assert "john.smith@example.com" in out


def test_c62_time_as_the_recogniser_writes_it():
    out = cleanup.smart_format(cleanup.rule_clean("The meeting is at 3.30 pm."))
    assert "3:30 PM" in out or "3:30 pm" in out


def test_c62_fix_must_not_turn_ordinary_at_phrases_or_decimals_into_addresses_or_times():
    for text in ("We met at example.com headquarters.", "The value is 3.30 percent.", "Version 3.30 is out."):
        assert cleanup.smart_format(cleanup.rule_clean(text)) == text


# ---------------------------------------------------------------- C-63: Italian / Portuguese taken for English
REAL_FOREIGN = [
    "Nel giro di 2 ore sono esplose altre tre bombe nelle vicinanze degli edifici governativi.",
    "Robin Gutap fez a pontuação mais alta do Inis 70 corridas em apenas 41 bolas, acertando 1142.",
]


@pytest.mark.parametrize("sentence", REAL_FOREIGN)
def test_c63_real_italian_and_portuguese_sentences_are_not_english(sentence):
    assert cleanup.is_english(sentence, "auto") is False


def test_c63_fix_must_keep_english_english():
    for sentence in ("The quick brown fox jumps over the lazy dog.",
                     "Please schedule the meeting for Thursday at 3.30 in the afternoon.",
                     "Send the report to Giovanni Rossi in Milano before noon."):
        assert cleanup.is_english(sentence, "auto") is True


# ---------------------------------------------------------------- C-64: download progress without symlinks
def _repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    root = modelhub.repo_dir("acme/model")
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "abc123").mkdir(parents=True)
    return root


def test_c64_bytes_still_downloading_are_counted(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    (root / "blobs" / "deadbeef.incomplete").write_bytes(b"x" * 1000)
    assert modelhub.downloaded_bytes("acme/model") == 1000


def test_c64_a_finished_file_moved_to_the_snapshot_folder_still_counts(tmp_path, monkeypatch):
    """Windows without symlink rights: the finished file leaves ``blobs/`` and becomes a real file in the snapshot."""
    root = _repo(tmp_path, monkeypatch)
    (root / "snapshots" / "abc123" / "encoder.onnx").write_bytes(b"x" * 4000)      # finished, moved
    (root / "blobs" / "cafe.incomplete").write_bytes(b"x" * 500)                   # the next file, in progress
    assert modelhub.downloaded_bytes("acme/model") == 4500


def test_c64_progress_never_goes_backwards_when_a_file_completes(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    part = root / "blobs" / "deadbeef.incomplete"
    part.write_bytes(b"x" * 3000)
    before = modelhub.downloaded_bytes("acme/model")
    part.rename(root / "snapshots" / "abc123" / "model.gguf")                       # download finished
    assert modelhub.downloaded_bytes("acme/model") >= before


# ---------------------------------------------------------------- C-70: "two million years" written as 2.000.000
GERMAN = ("Die Wissenschaft weist nun darauf hin, dass diese massive Kohlenstoffwirtschaft die Biosphäre aus einem "
          "ihrer stabilen Zustände, der die menschliche Evolution in den letzten zwei Millionen Jahren getragen hat, "
          "herausgebracht hat.")


def test_c70_a_round_number_of_millions_in_running_text_is_not_spelled_out_in_digits():
    """Real human sentence (FLEURS, German). The recogniser wrote it correctly; smart formatting turned
    'zwei Millionen Jahren' into '2.000.000 Jahren', which no German text would print."""
    out = cleanup.Cleaner(use_llm=False).clean(GERMAN, language="de", smart=True)
    assert "2.000.000" not in out
    assert "zwei Millionen Jahren" in out or "2 Millionen Jahren" in out


def test_c70_fix_must_keep_real_figures():
    out = cleanup.Cleaner(use_llm=False).clean("das kostet zwei millionen dreihunderttausend euro", language="de",
                                               smart=True)
    assert "2.300.000" in out
