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


# =========================================================================== Q4
def test_spoken_numbers_up_to_99_fire_a_snippet(tmp_path):
    s = store.Snippets(tmp_path)
    s.add("room 21", "Room 21, second floor")
    s.add("gate forty two", "Gate 42")
    s.add("zoom link two", "https://zoom/2")
    s.add("agent 007", "Bond")
    for said, typed in [("Room twenty one.", "Room 21, second floor"), ("room twenty-one", "Room 21, second floor"),
                        ("Room 21", "Room 21, second floor"), ("gate 42", "Gate 42"), ("Gate forty-two.", "Gate 42"),
                        ("zoom link 2", "https://zoom/2"), ("Agent 007.", "Bond")]:
        assert s.expand(said) == typed, said
    assert s.expand("Insert room twenty one, please.") == "Room 21, second floor, please."
    assert s.expand("room twenty") == "room twenty" and s.expand("room twenty two") == "room twenty two"
    assert store._trigger_key("Room Twenty-One") == "room 21" and store._trigger_key("ninety") == "90"
    with pytest.raises(ValueError, match="already have a snippet"):
        s.add("room twenty one", "dup")
    assert store._number_words(90) == ["ninety"] and store._number_words(7) == ["seven"]


def test_a_damaged_history_is_copied_before_the_load_rewrites_it(tmp_path):
    now = time.time()
    v1 = [{"text": "kept", "ts": now - 60}, {"text": 5}, {"text": "old", "ts": now - 90 * 86400}]
    _write(tmp_path / "history.json", v1)                      # an old layout that is migrated (and pruned) on load
    h = store.History(tmp_path, retention_days=30)
    [copy] = tmp_path.glob("history.corrupt-*.json")
    assert json.loads(copy.read_text(encoding="utf-8")) == v1, "the copy is the file as it was, damage included"
    assert [e["text"] for e in h.snapshot()] == ["kept"]
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))["version"] == 2


# =========================================================================== D17
def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def _settled(h) -> None:
    """Every background write of ``h`` is on disk (a write holds the lock while it replaces the file)."""
    _wait(lambda: h._timer is None and not h._dirty)
    with h._lock:
        pass


def test_filing_a_transcript_never_writes_the_history_on_the_callers_thread(tmp_path, monkeypatch):
    import threading
    h = store.History(tmp_path)
    writers: list[str] = []
    real = store.cfgmod._atomic_write
    monkeypatch.setattr(store.cfgmod, "_atomic_write", lambda path, text: (
        path == h.path and writers.append(threading.current_thread().name), real(path, text)))
    h.add("first dictation")
    _wait(lambda: writers)
    assert writers == ["voxflow-store-write"], "written on the store's own thread, never the dictation's"
    _settled(h)
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))["items"][0]["text"] == "first dictation"


def test_an_older_background_write_never_lands_after_a_newer_one(tmp_path):
    h = store.History(tmp_path)
    real = h._serialise

    def slow(payload):                                # while the timer thread prepares its text, a flush writes
        if not getattr(slow, "done", False):
            slow.done = True
            h.add("newer")
            h.flush()
        return real(payload)

    h._serialise = slow
    h.add("older")
    _wait(lambda: getattr(slow, "done", False))
    _settled(h)
    on_disk = [e["text"] for e in json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))["items"]]
    assert on_disk == ["newer", "older"], "the stale snapshot (only 'older') was not written over it"


def test_a_change_made_while_writing_is_written_next_and_a_purge_waits_for_it(tmp_path):
    now = time.time()
    _write(tmp_path / "history.json", [{"text": "secret", "ts": now - 60}])     # v1: migrated, a v1 copy kept
    h = store.History(tmp_path)
    assert (tmp_path / "history.v1.json").exists()
    real = h._serialise

    def busy(payload):
        if not getattr(busy, "done", False):
            busy.done = True
            h.add("added meanwhile")                  # not in the payload being written
        return real(payload)

    h._serialise = busy
    h.delete(h.list()[0]["id"])                        # the purge of the v1 copy waits for this write
    _wait(lambda: getattr(busy, "done", False))
    _settled(h)
    assert not (tmp_path / "history.v1.json").exists(), "the file without 'secret' is on disk: the copy went"
    assert [e["text"] for e in json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))["items"]] == [
        "added meanwhile"], "the change made during the write was written right after"


def test_a_background_write_that_fails_is_reported(tmp_path, monkeypatch):
    h = store.History(tmp_path)
    topics: list[str] = []
    h.subscribe(topics.append)
    monkeypatch.setattr(h, "_serialise", lambda payload: ValueError("Out of range float values are not JSON"))
    h.add("x")
    _wait(lambda: "save_failed:history" in topics)
    assert h.last_save_error == "Out of range float values are not JSON"


def test_search_stops_at_the_limit(tmp_path):
    h = store.History(tmp_path)
    with h.batch():
        for i in range(50):
            h.add(f"report number {i}", ts=1_800_000_000.0 + i)
    assert [e["text"] for e in h.search("report", 3)] == ["report number 49", "report number 48", "report number 47"]
    assert len(h.search("", 7)) == 7 and h.search("report", 0) == [] and len(h.search("number")) == 50
    assert h.search("nothing like it", 5) == []


def test_a_timer_that_fires_after_a_flush_writes_nothing(tmp_path, monkeypatch):
    h = store.History(tmp_path)
    h.add("x")
    h.flush()
    calls: list = []
    monkeypatch.setattr(store.cfgmod, "_atomic_write", lambda path, text: path == h.path and calls.append(text))
    h._write_behind()                                 # a timer that was already running when flush() cancelled it
    assert calls == []


def test_a_payload_that_is_not_json_is_a_save_error_not_a_crash(tmp_path):
    h = store.History(tmp_path)
    assert isinstance(h._serialise({"bad": {1, 2}}), TypeError)
    h._items.append({"id": "x", "ts": 1.0, "text": {1}})          # never happens through the API
    h._dirty = True
    h.flush()
    assert "not JSON serializable" in h.last_save_error


def test_the_v1_copy_that_cannot_be_written_is_logged(tmp_path, monkeypatch, caplog):
    _write(tmp_path / "history.json", [{"text": "old", "ts": time.time() - 60}])
    real = store.Path.write_bytes

    def refuse_copy(self, data):
        if self.name == "history.v1.json":
            raise OSError(13, "Access is denied")
        return real(self, data)

    monkeypatch.setattr(store.Path, "write_bytes", refuse_copy)
    with caplog.at_level("WARNING"):
        h = store.History(tmp_path)
    assert h.count() == 1 and "could not keep a copy of history.json" in caplog.text


def test_the_term_index_follows_every_change(tmp_path):
    d = store.Dictionary(tmp_path)
    a = d.add("Wi-Fi", ["why fly"])
    d.update(a["id"], term="WLAN")
    assert d._find_term("wlan")["id"] == a["id"] and d._find_term("wi fi") is None
    with pytest.raises(ValueError, match="already in your dictionary"):
        d.add("Wlan")
    d.remove(a["id"])
    assert d._find_term("WLAN") is None
    d.apply_changes([{"term": "GitHub"}, {"term": "github", "aliases": ["get hub"]}], removals=["nothing"])
    assert d.terms() == ["github"] and d._find_term("GITHUB")["aliases"] == ["get hub"]
    d.apply_changes(removals=["GitHub"])
    assert d.terms() == [] and d._find_term("github") is None
    src = tmp_path / "in.txt"
    src.write_text("Kubernetes: kube\n", encoding="utf-8")
    assert d.import_(str(src)) == {"added": 1, "updated": 0}
    assert d.import_(str(src)) == {"added": 0, "updated": 0}, "nothing new: nothing updated"
    assert d._find_term("kubernetes")["term"] == "Kubernetes"
    d.suggest("PyTorch")
    d.accept_suggestion(d.suggestions()[0]["id"])
    assert d._find_term("pytorch")["term"] == "PyTorch"


def test_search_never_reads_past_the_limit(tmp_path):
    class Untouchable(dict):
        def __getitem__(self, key):
            raise AssertionError("search read an entry after it had enough results")

    h = store.History(tmp_path)
    for i in range(3):
        h.add(f"match {i}")
    h._items.append(Untouchable())                    # the oldest entry: only a scan of everything reaches it
    assert len(h.search("match", 3)) == 3 and len(h.search("", 2)) == 2
