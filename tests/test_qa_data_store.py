"""QA "data" fixes in the stores (``voxflow.store``): each test names the finding of the QA report it covers."""
from __future__ import annotations

import datetime
import json
import time

from voxflow import store


# =========================================================================== D6
def test_history_counts_what_a_retention_would_delete(tmp_path):
    now = 2_000_000_000.0
    h = store.History(tmp_path, clock=lambda: now)
    for age in (0, 0.5, 3, 10, 40):
        h.add(f"said {age} days ago", ts=now - age * 86400.0)
    assert h.count_older_than(7) == 2 and h.count_older_than(1) == 3
    assert h.count_older_than(365) == 0
    assert h.count_older_than(0) == 0 and h.count_older_than(-3) == 0, "forever deletes nothing"
    assert h.count() == 5, "counting deletes nothing"


# =========================================================================== D2 + R8
def _write(path, payload) -> None:
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def test_one_damaged_row_costs_that_row_only_and_is_reported(tmp_path):
    _write(tmp_path / "dictionary.json", {
        "entries": [{"term": "Kubernetes", "aliases": ["kube"]}, {"term": 5}, "junk", {"term": "GitHub"},
                    {"term": "Odd", "aliases": 5}],
        "suggestions": [{"term": "PyTorch", "ts": "yesterday"}, {"nope": 1}], "dismissed": ["x", 3]})
    d = store.Dictionary(tmp_path)
    assert d.terms() == ["GitHub", "Kubernetes", "Odd"], "a bad field never empties the whole dictionary"
    assert [e["aliases"] for e in d.list()] == [[], ["kube"], []]
    assert [(s["term"], s["ts"]) for s in d.suggestions()] == [("PyTorch", 0.0)]
    [copy] = tmp_path.glob("dictionary.corrupt-*.json")
    assert d.load_problem == (f"3 entries in your dictionary could not be read and were left out. The original file "
                              f"was kept as {copy.name} in VoxFlow's data folder.")
    assert json.loads(copy.read_text(encoding="utf-8"))["entries"][1] == {"term": 5}

    _write(tmp_path / "snippets.json", {"entries": [{"trigger": "sig", "text": "Best"}, {"trigger": "x"},
                                                    {"trigger": "...", "text": "y"}]})
    s = store.Snippets(tmp_path)
    assert [e["trigger"] for e in s.list()] == ["sig"] and s.load_problem.startswith("2 entries in your snippets")


def test_a_wrong_layout_is_kept_and_reported_instead_of_silently_starting_empty(tmp_path):
    _write(tmp_path / "history.json", {"items": {"a": {"text": "hello"}}})
    h = store.History(tmp_path)
    assert h.count() == 0 and h.load_problem.startswith("Your history could not be read, so it starts empty.")
    assert len(list(tmp_path.glob("history.corrupt-*.json"))) == 1, "the only copy of the transcripts is kept"
    _write(tmp_path / "styles.json", ["a", "list"])
    assert store.Styles(tmp_path).load_problem.startswith("Your style settings could not be read")
    _write(tmp_path / "history.json", {"items": [{"text": "one"}, {"text": 2}]})
    assert store.History(tmp_path).load_problem.startswith("1 entry in your history could not be read and was left")
    _write(tmp_path / "styles.json", {"tones": ["formal"], "apps": "slack"})
    st = store.Styles(tmp_path)
    assert st.load_problem == "" and st.tones() == store.DEFAULT_TONES, "unknown values are ignored, not an error"
    assert store.Dictionary(tmp_path / "empty").load_problem == "", "no file yet is no problem"


def test_a_second_bad_file_never_overwrites_the_copy_of_the_first(tmp_path, monkeypatch):
    class Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 17, 7, 23, 1)

    monkeypatch.setattr(store._dt, "datetime", Frozen)
    for text in ("{ first", "{ second", "{ third"):
        _write(tmp_path / "dictionary.json", text)
        store.Dictionary(tmp_path)
    names = sorted(p.name for p in tmp_path.glob("dictionary.corrupt*"))
    assert names == ["dictionary.corrupt-20261117-072301-2.json", "dictionary.corrupt-20261117-072301-3.json",
                     "dictionary.corrupt-20261117-072301.json"]
    assert (tmp_path / "dictionary.corrupt-20261117-072301.json").read_text(encoding="utf-8") == "{ first"


def test_a_copy_that_cannot_be_written_is_not_promised(tmp_path, monkeypatch):
    _write(tmp_path / "snippets.json", "{ broken")

    def no_room(self, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store.Path, "write_bytes", no_room)
    s = store.Snippets(tmp_path)
    assert s.load_problem == "Your snippets could not be read, so it starts empty."


def test_the_copy_of_a_damaged_history_survives_pruning_and_goes_with_clear_all(tmp_path):
    now = time.time()
    _write(tmp_path / "history.json", {"items": [{"text": "old secret", "ts": now - 40 * 86400},
                                                 {"text": "recent", "ts": now - 60}, {"text": None}]})
    h = store.History(tmp_path)
    [copy] = tmp_path.glob("history.corrupt-*.json")
    h.configure(retention_days=7)                    # deletes "old secret" from history.json
    h.add("another")
    h.delete(h.list()[0]["id"])
    h.flush()
    assert copy.exists(), "the user was told where the copy is: retention and deletions do not remove it"
    assert "old secret" not in (tmp_path / "history.json").read_text(encoding="utf-8")
    h.clear()
    assert not copy.exists() and not list(tmp_path.glob("history.corrupt*")), "Clear all deletes everything"


def test_a_damaged_store_is_announced_at_start_with_open_data_folder(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "dictionary.json", {"entries": [{"term": "Kubernetes"}, {"term": 7}]})
    rig = make_rig(tmp_path, qtbot, monkeypatch, start=False)
    actions: list = []
    rig.c.notice_action.connect(lambda *a: actions.append(a))
    try:
        rig.start()
        notices = [args for name, args in rig.startup_events if name == "notice"]
        message = rig.c.dictionary.load_problem
        assert ("warning", message) in notices and message.startswith("1 entry in your dictionary")
        assert actions == [(message, "open_data_folder", "Open data folder")]
        assert rig.c.dictionary.terms() == ["Kubernetes"]
    finally:
        rig.close()


# =========================================================================== D3
def test_aliases_written_as_one_string_are_a_comma_separated_list(tmp_path):
    _write(tmp_path / "dictionary.json", {"entries": [{"term": "Kubernetes", "aliases": "kube"},
                                                      {"term": "PostgreSQL", "aliases": "post gress, postgress"},
                                                      {"term": "Plan", "aliases": ["b", "e", "plan bee"]}]})
    d = store.Dictionary(tmp_path)
    assert {e["term"]: e["aliases"] for e in d.list()} == {
        "Kubernetes": ["kube"], "PostgreSQL": ["post gress", "postgress"], "Plan": ["plan bee"]}
    sentence = "I will e-mail k and e about a b c"
    assert d.apply(sentence) == sentence, "single letters are never turned into a term"
    assert d.load_problem == ""

    src = tmp_path / "import.json"
    _write(src, [{"term": "Grafana", "aliases": "graph ana"}, {"term": "Redis", "aliases": 5}])
    fresh = store.Dictionary(tmp_path / "fresh")
    fresh.import_(str(src))
    assert {e["term"]: e["aliases"] for e in fresh.list()} == {"Grafana": ["graph ana"], "Redis": []}
    assert fresh.add("Vim", ["v", "vi m"])["aliases"] == ["vi m"]
