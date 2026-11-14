"""voxflow.store: remaining corners of the shared save plumbing (exit flush, coalesced writes, nested
batches), of lenient loading and of the bulk ``apply_changes`` used by sync / profile import."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxflow import store as S
from voxflow.store import Dictionary, History, Snippets, Styles


class Topics:
    def __init__(self, store):
        self.seen: list[str] = []
        store.subscribe(self.seen.append)


def on_disk(store) -> dict:
    return json.loads(store.path.read_text(encoding="utf-8"))


# =========================================================================== shared plumbing
def test_flush_all_writes_pending_changes_and_survives_a_store_that_cannot_flush(tmp_path):
    good, broken = Dictionary(tmp_path), Snippets(tmp_path)

    def boom():
        raise RuntimeError("disk vanished")

    good.add("Kubernetes")                         # written at once
    good.add("PyTorch")                            # within the coalescing window: still pending
    assert good._dirty and [e["term"] for e in on_disk(good)["entries"]] == ["Kubernetes"]
    broken.flush = boom
    S.flush_all()                                  # what atexit runs
    assert sorted(e["term"] for e in on_disk(good)["entries"]) == ["Kubernetes", "PyTorch"]
    assert not good._dirty and good._timer is None


def test_second_store_on_the_same_file_loads_even_if_the_first_cannot_flush(tmp_path):
    first = Dictionary(tmp_path)
    first.add("Kubernetes")
    first.flush()

    def boom():
        raise OSError("locked")

    first.flush = boom
    second = Dictionary(tmp_path)
    assert second.terms() == ["Kubernetes"]


def test_unreadable_but_missing_file_starts_empty_without_a_backup_copy(tmp_path, monkeypatch, caplog):
    real_read = Path.read_text

    def read_text(self, *a, **k):
        if self.name == "dictionary.json":
            raise PermissionError("access denied")      # e.g. the folder itself cannot be read
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    with caplog.at_level("WARNING"):
        d = Dictionary(tmp_path)
    assert d.list() == [] and "unreadable" in caplog.text
    assert list(tmp_path.glob("dictionary*")) == [], "there was no file to keep a copy of"


def test_coalescing_timer_does_nothing_inside_a_batch_or_without_pending_changes(tmp_path):
    d = Dictionary(tmp_path)
    d._timer_fired()                               # nothing pending
    assert not d.path.exists()
    with d.batch():
        d.add("Kubernetes")
        d._timer_fired()                           # a timer from before the batch fires in the middle of it
        assert not d.path.exists() and d._dirty, "the batch still owns the write"
    assert on_disk(d)["entries"][0]["term"] == "Kubernetes" and not d._dirty


def test_retry_save_cancels_the_pending_coalesced_write_and_writes_now(tmp_path):
    d = Dictionary(tmp_path)
    d.add("Kubernetes")                            # first change: written at once
    d.add("PyTorch")                               # second within the coalescing window: a timer is pending
    timer = d._timer
    assert timer is not None and [e["term"] for e in on_disk(d)["entries"]] == ["Kubernetes"]
    assert d.retry_save() is True
    assert d._timer is None and not d._dirty
    timer.join(1.0)
    assert not timer.is_alive(), "the cancelled timer is gone"
    assert sorted(e["term"] for e in on_disk(d)["entries"]) == ["Kubernetes", "PyTorch"]


def test_nested_batches_write_once_when_the_outermost_ends(tmp_path, monkeypatch):
    d = Dictionary(tmp_path)
    writes = []
    real_write = S.cfgmod._atomic_write
    monkeypatch.setattr(S.cfgmod, "_atomic_write", lambda path, text: (writes.append(1), real_write(path, text))[1])
    with d.batch():
        d.add("Kubernetes")
        with d.batch():
            d.add("PyTorch")
        assert writes == [] and d._batch == 1, "leaving the inner batch does not write"
        d.add("VoxFlow")
    assert writes == [1] and d._batch == 0
    assert len(on_disk(d)["entries"]) == 3


# =========================================================================== dictionary
def test_dictionary_file_with_the_same_term_twice_keeps_the_first(tmp_path):
    (tmp_path / "dictionary.json").write_text(json.dumps({"entries": [
        {"id": "one", "term": "Kubernetes", "aliases": ["cube"]},
        {"id": "two", "term": "kubernetes", "aliases": ["other"]},
        {"id": "three", "term": "PyTorch"}]}), encoding="utf-8")
    d = Dictionary(tmp_path)
    assert [(e["id"], e["term"], e["aliases"]) for e in d.list()] == [
        ("one", "Kubernetes", ["cube"]), ("three", "PyTorch", [])]


@pytest.mark.parametrize("bad", ["\x01\x02", " \x7f "])
def test_dictionary_term_made_only_of_control_characters_is_refused(tmp_path, bad):
    d = Dictionary(tmp_path)
    with pytest.raises(ValueError, match="Type the word or name"):
        d.add(bad)
    assert d.list() == []


def test_dictionary_apply_changes_skips_bad_rows_and_reports_only_real_changes(tmp_path):
    d = Dictionary(tmp_path)
    d.add("Kubernetes", ["cube"])
    topics = Topics(d)
    before = d.path.read_bytes()

    # nothing here is a change: unknown / non-text removals, unusable rows, and a row identical to what is stored
    assert d.apply_changes(upserts=[None, "text", {"term": ""}, {"term": 7}, {"aliases": ["x"]},
                                    {"term": "Kubernetes", "aliases": ["cube"]}],
                           removals=["Unknown", 42, None]) == 0
    assert topics.seen == [] and d.path.read_bytes() == before

    assert d.apply_changes(upserts=[{"term": "kubernetes", "aliases": ["cube", "kube"]}, {"term": "PyTorch"},
                                    {"term": "x" * 200}]) == 2
    assert {e["term"]: e["aliases"] for e in d.list()} == {"kubernetes": ["cube", "kube"], "PyTorch": []}
    assert topics.seen == ["dictionary", "suggestions"]
    assert d.apply_changes(removals=["  PYTORCH "]) == 1 and d.terms() == ["kubernetes"]


# =========================================================================== snippets
def test_snippets_file_with_duplicates_and_empty_rows_is_loaded_leniently(tmp_path):
    (tmp_path / "snippets.json").write_text(json.dumps({"entries": [
        {"id": "a", "trigger": "my email", "text": "me@example.com"},
        {"id": "b", "trigger": "My Email!", "text": "other@example.com"},     # same trigger once normalised
        {"id": "c", "trigger": "...", "text": "only punctuation"},
        {"id": "d", "trigger": "empty", "text": ""},
        {"id": "e", "trigger": "sig", "text": "Best regards"}]}), encoding="utf-8")
    sn = Snippets(tmp_path)
    assert [(e["id"], e["trigger"]) for e in sn.list()] == [("a", "my email"), ("e", "sig")]


def test_snippet_made_only_of_control_characters_is_refused_with_the_matching_message(tmp_path):
    sn = Snippets(tmp_path)
    with pytest.raises(ValueError, match="Type the phrase you will say"):
        sn.add("\x01\x02", "some text")
    with pytest.raises(ValueError, match="Type the text this snippet should insert"):
        sn.add("my sig", "\x01\x02")
    assert sn.list() == []


def test_snippets_apply_changes_skips_bad_rows_and_reports_only_real_changes(tmp_path):
    sn = Snippets(tmp_path)
    sn.add("my email", "me@example.com")
    topics = Topics(sn)
    before = sn.path.read_bytes()

    assert sn.apply_changes(upserts=[None, {"trigger": "x"}, {"trigger": 1, "text": "t"},
                                     {"trigger": "sig", "text": "   "}, {"trigger": "?!", "text": "t"},
                                     {"trigger": "y" * 200, "text": "t"},
                                     {"trigger": "my email", "text": "me@example.com"}],
                            removals=["unknown", 5, None]) == 0
    assert topics.seen == [] and sn.path.read_bytes() == before

    assert sn.apply_changes(upserts=[{"trigger": "My Email", "text": "new@example.com"},
                                     {"trigger": "sig", "text": "Best regards"}]) == 2
    assert {e["trigger"]: e["text"] for e in sn.list()} == {"My Email": "new@example.com", "sig": "Best regards"}
    assert topics.seen == ["snippets"]
    assert sn.expand("sig") == "Best regards", "the trigger matcher was rebuilt"
    assert sn.apply_changes(removals=["SIG."]) == 1
    assert [e["trigger"] for e in sn.list()] == ["My Email"]


# =========================================================================== history
def test_history_import_skips_malformed_entries(tmp_path):
    h = History(tmp_path)
    added = h.import_entries([None, "text", {"text": 5}, {"text": "   "}, {"raw": "no text"},
                              {"id": "keep", "text": "Hello world.", "ts": 1_700_000_000.0}])
    assert added == 1
    assert [(e["id"], e["text"]) for e in h.list()] == [("keep", "Hello world.")]
    assert h.import_entries([{"id": "keep", "text": "duplicate id"}]) == 0 and h.count() == 1


# =========================================================================== styles
def test_styles_apply_changes_skips_unknown_values_and_reports_only_real_changes(tmp_path):
    st = Styles(tmp_path)
    st.set_app_category("myapp.exe", "work")
    topics = Topics(st)
    before = st.path.read_bytes()
    current = st.tones()

    assert st.apply_changes(tones={"nonsense": "formal", "work": "shouting", "email": current["email"]},
                            apps={"": "work", 7: "work", ".exe": "work", "other.exe": "nonsense",
                                  "MyApp": "work"},
                            removed_apps=["never-mapped.exe"]) == 0
    assert topics.seen == [] and st.path.read_bytes() == before
    assert st.apply_changes() == 0

    assert st.apply_changes(tones={"work": "formal"}, apps={"C:\\Apps\\Editor.EXE": "email"},
                            removed_apps=["MYAPP"]) == 3
    assert st.tones()["work"] == "formal" and st.user_apps() == {"editor.exe": "email"}
    assert topics.seen == ["styles"]
    assert Styles(tmp_path).user_apps() == {"editor.exe": "email"}
