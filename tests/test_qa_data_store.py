"""QA "data" fixes in the stores (``voxflow.store``): each test names the finding of the QA report it covers."""
from __future__ import annotations

import datetime
import json
import time

import pytest

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


# =========================================================================== D4
def test_a_semicolon_csv_from_excel_imports_as_words_not_as_glued_rows(tmp_path):
    src = tmp_path / "excel_de.csv"
    src.write_bytes("term;aliases\r\nKubernetes;cube net ease\r\nMüller;mueller\r\n".encode("cp1252"))
    d = store.Dictionary(tmp_path / "d")
    assert d.import_(str(src)) == {"added": 2, "updated": 0}, "ANSI (cp1252) is read, not a codec error"
    assert {e["term"]: e["aliases"] for e in d.list()} == {"Kubernetes": ["cube net ease"], "Müller": ["mueller"]}
    assert d.apply("we use kubernetes and cube net ease") == "we use Kubernetes and Kubernetes"

    tabbed = tmp_path / "tabs.csv"
    tabbed.write_text("term\taliases\nGrafana\tgraph ana\n", encoding="utf-8")
    d.import_(str(tabbed))
    assert "Grafana" in d.terms()
    utf16 = tmp_path / "unicode.txt"
    utf16.write_text("Köln: koeln\n", encoding="utf-16")         # Notepad's "Unicode"
    d.import_(str(utf16))
    assert {e["term"]: e["aliases"] for e in d.list()}["Köln"] == ["koeln"]


def test_a_term_never_holds_a_semicolon(tmp_path):
    d = store.Dictionary(tmp_path)
    with pytest.raises(ValueError, match="can't contain “;”"):
        d.add("Kubernetes;cube net ease")
    src = tmp_path / "glued.txt"
    src.write_text("Kubernetes;cube net ease\nGitHub\n", encoding="utf-8")
    assert d.import_(str(src)) == {"added": 1, "updated": 0} and d.terms() == ["GitHub"]


# =========================================================================== D14
def test_csv_exports_start_with_a_bom_so_excel_reads_utf8(tmp_path):
    d = store.Dictionary(tmp_path)
    d.add("Köln", ["koeln"])
    d.export(str(tmp_path / "d.csv"))
    h = store.History(tmp_path)
    h.add("Grüße aus Köln")
    h.export(str(tmp_path / "h.csv"))
    for name in ("d.csv", "h.csv"):
        raw = (tmp_path / name).read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf") and "Köln".encode("utf-8") in raw
    fresh = store.Dictionary(tmp_path / "fresh")
    assert fresh.import_(str(tmp_path / "d.csv")) == {"added": 1, "updated": 0}, "the BOM is not part of a term"
    assert fresh.list()[0]["term"] == "Köln"


# =========================================================================== D15
def test_a_txt_export_imports_back_to_the_same_entries(tmp_path):
    d = store.Dictionary(tmp_path)
    entries = {"Ratio: 3": ["ratio three"], "Smith, Jones": ["smith and jones", "smith, jones and co"],
               "#hashtag": ["hash tag"], r"C:\Tools": ["see tools"], "Plain": [], r"a\b": [r"back\slash"]}
    for term, aliases in entries.items():
        d.add(term, aliases)
    d.export(str(tmp_path / "d.txt"))
    text = (tmp_path / "d.txt").read_text(encoding="utf-8")
    assert "Ratio\\: 3: ratio three\n" in text and "\\#hashtag: hash tag\n" in text
    fresh = store.Dictionary(tmp_path / "fresh")
    fresh.import_(str(tmp_path / "d.txt"))
    assert {e["term"]: e["aliases"] for e in fresh.list()} == entries
    hand = tmp_path / "hand.txt"                                  # written by hand: plain lines still work
    hand.write_text("Kubernetes: kube, cube net\n" + r"x\y: a backslash before a letter stays" + "\n", encoding="utf-8")
    other = store.Dictionary(tmp_path / "other")
    other.import_(str(hand))
    assert {e["term"]: e["aliases"] for e in other.list()} == {"Kubernetes": ["kube", "cube net"],
                                                               r"x\y": ["a backslash before a letter stays"]}


# =========================================================================== D16
def test_importing_a_large_dictionary_is_linear(tmp_path):
    src = tmp_path / "big.json"
    src.write_text(json.dumps([{"term": f"Term{i}", "aliases": [f"alias {i}", f"other {i}", "shared"]}
                               for i in range(4000)]), encoding="utf-8")
    d = store.Dictionary(tmp_path / "d")
    d.add("Owner", ["shared"])
    t0 = time.perf_counter()
    assert d.import_(str(src)) == {"added": 4000, "updated": 0}
    took = time.perf_counter() - t0
    assert took < 3.0, f"4,000 entries took {took:.1f} s (it was O(n²): about 12 s)"
    assert all("shared" not in e["aliases"] for e in d.list() if e["term"] != "Owner"), "conflicts still refused"
    assert d.list()[0]["aliases"] == ["shared"]


def test_import_conflict_rules_are_unchanged(tmp_path):
    d = store.Dictionary(tmp_path)
    d.add("GitHub", ["get hub"])
    d.add("GitLab", ["git lab"])
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"term": "get hub", "aliases": ["x y"]},         # someone's alias: not a term
                               {"term": "GitLab", "aliases": ["github", "gitlab two"]},   # a term is taken
                               {"term": "New", "aliases": ["git lab", "new one"]},
                               {"term": "Newer", "aliases": ["new one"]}]), encoding="utf-8")
    assert d.import_(str(src)) == {"added": 2, "updated": 1}
    assert {e["term"]: e["aliases"] for e in d.list()} == {
        "GitHub": ["get hub"], "GitLab": ["git lab", "gitlab two"], "New": ["new one"], "Newer": []}


# =========================================================================== Q3
def test_snippet_import_updates_the_text_and_counts_both(tmp_path):
    s = store.Snippets(tmp_path)
    s.add("zoom link", "https://zoom/OLD")
    s.add("sig", "Best")
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"trigger": "Zoom-link", "text": "https://zoom/NEW"}, {"trigger": "sig", "text": "Best"},
                               {"trigger": "eta", "text": "On my way"}, {"trigger": "eta", "text": "dup"}]),
                   encoding="utf-8")
    assert s.import_(str(src)) == {"added": 1, "updated": 1}
    assert {e["trigger"]: e["text"] for e in s.list()} == {"zoom link": "https://zoom/NEW", "sig": "Best",
                                                           "eta": "On my way"}
    binary = tmp_path / "bin.json"
    binary.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Couldn't read bin.json: it is not a text file"):
        s.import_(str(binary))
    bad16 = tmp_path / "bad16.txt"
    bad16.write_bytes(b"\xff\xfe\x00\xd8")                         # a UTF-16 BOM, then half a character
    with pytest.raises(ValueError, match="not a text file"):
        store.Dictionary(tmp_path / "d").import_(str(bad16))


def test_a_dictionary_import_that_only_adds_spellings_says_so(tmp_path):
    d = store.Dictionary(tmp_path)
    d.add("GitHub", ["get hub"])
    src = tmp_path / "in.csv"
    src.write_text("term,aliases\nGitHub,git hub\n", encoding="utf-8")
    assert d.import_(str(src)) == {"added": 0, "updated": 1}
    assert d.list()[0]["aliases"] == ["get hub", "git hub"]
