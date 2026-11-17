"""voxflow.store: Dictionary, Snippets, History, Stats, Styles. Every store lives in a temp directory."""
from __future__ import annotations

import csv
import datetime as dt
import json
import threading

import pytest

from voxflow import store as S
from voxflow.store import Dictionary, History, Snippets, Stats, Styles, count_words, make_entry, normalize_exe


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


class Topics:
    """Collects store notifications."""

    def __init__(self, store):
        self.seen: list[str] = []
        store.subscribe(self.seen.append)

    def take(self) -> list[str]:
        out, self.seen[:] = list(self.seen), []
        return out


def reload(store):
    """A second instance reading the same file: proves the change was persisted."""
    return type(store)(store.path)


# =========================================================================== shared plumbing
@pytest.mark.parametrize("cls", [Dictionary, Snippets, History, Stats, Styles])
def test_default_location_is_the_isolated_app_dir(cls, data_dir):
    store = cls()
    assert store.path == data_dir / cls.filename


@pytest.mark.parametrize("cls", [Dictionary, Snippets, History, Stats, Styles])
def test_store_accepts_a_directory_or_a_file_path(cls, tmp_path):
    assert cls(tmp_path).path == tmp_path / cls.filename
    assert cls(tmp_path / "custom.json").path == tmp_path / "custom.json"
    assert cls(str(tmp_path / "not-there-yet")).path == tmp_path / "not-there-yet" / cls.filename


@pytest.mark.parametrize("cls", [Dictionary, Snippets, History, Stats, Styles])
@pytest.mark.parametrize("content", ["{ not json", "", "[1, 2", "\x00\x01\x02", '"just a string"', "42", "null"])
def test_corrupt_or_foreign_file_starts_empty_and_never_raises(cls, content, tmp_path):
    path = tmp_path / cls.filename
    path.write_text(content, encoding="utf-8")
    store = cls(tmp_path)                       # must not raise
    assert store._dump() is not None


@pytest.mark.parametrize("cls", [Dictionary, Snippets, History, Stats, Styles])
def test_unparsable_file_is_backed_up_before_it_can_be_overwritten(cls, tmp_path):
    path = tmp_path / cls.filename
    path.write_text("{ this was my data", encoding="utf-8")
    st = cls(tmp_path)
    [backup] = tmp_path.glob(f"{path.stem}.corrupt-*{path.suffix}")      # named with the time: never overwritten
    assert backup.read_text(encoding="utf-8") == "{ this was my data"
    assert backup.name in st.load_problem and "could not be read, so it starts empty" in st.load_problem


@pytest.mark.parametrize("payload", [{"entries": "nope"}, {"entries": [1, None, "x", {"term": 5}, {"trigger": 1}]},
                                     {"items": [1, {"text": 3}]}, {"days": {"2026-01-01": "x", "bad-date": {}}},
                                     {"tones": {"work": "shouty"}, "apps": {"": "work", "a.exe": "nope"}}])
@pytest.mark.parametrize("cls", [Dictionary, Snippets, History, Stats, Styles])
def test_wrongly_shaped_json_is_tolerated(cls, payload, tmp_path):
    (tmp_path / cls.filename).write_text(json.dumps(payload), encoding="utf-8")
    store = cls(tmp_path)
    json.dumps(store._dump())                   # still serialisable


def test_a_layout_that_breaks_the_loader_is_backed_up_and_reset(tmp_path):
    (tmp_path / "stats.json").write_text(json.dumps({"days": ["not", "a", "dict"]}), encoding="utf-8")
    st = Stats(tmp_path)
    assert st.summary()["total_words"] == 0
    assert len(list(tmp_path.glob("stats.corrupt-*.json"))) == 1 and st.load_problem


def test_save_failure_is_survived_in_memory(tmp_path, monkeypatch, caplog):
    d = Dictionary(tmp_path)
    monkeypatch.setattr(S.cfgmod, "_atomic_write", lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked")))
    with caplog.at_level("WARNING"):
        d.add("Kubernetes")                     # must not raise
    assert d.terms() == ["Kubernetes"]
    assert "could not save" in caplog.text
    assert not (tmp_path / "dictionary.json").exists()


def test_a_failing_listener_does_not_break_the_store_or_other_listeners(tmp_path):
    d = Dictionary(tmp_path)
    d.subscribe(lambda topic: 1 / 0)
    t = Topics(d)
    d.add("Kubernetes")
    assert t.take() == ["dictionary"]


def test_listeners_run_outside_the_lock_so_they_may_call_back_into_the_store(tmp_path):
    d = Dictionary(tmp_path)
    seen = []
    d.subscribe(lambda topic: seen.append(d.terms()))
    done = threading.Event()

    def other_thread_reader(topic):
        th = threading.Thread(target=lambda: (d.list(), done.set()))
        th.start()
        th.join(2)

    d.subscribe(other_thread_reader)
    d.add("Kubernetes")
    assert seen == [["Kubernetes"]]
    assert done.is_set(), "a listener on another thread dead-locked: notify ran with the store lock held"


@pytest.mark.parametrize("cls,payload,probe", [
    (Dictionary, {"entries": [{"term": "Keep"}]}, lambda s: s.terms() == ["Keep"]),
    (Snippets, {"entries": [{"trigger": "sig", "text": "Best"}]}, lambda s: len(s.list()) == 1),
])
def test_c47_files_with_a_utf8_bom_are_read(cls, payload, probe, tmp_path):
    (tmp_path / cls.filename).write_text("\ufeff" + json.dumps(payload), encoding="utf-8")
    assert probe(cls(tmp_path))


# =========================================================================== dictionary
@pytest.fixture
def d(tmp_path):
    return Dictionary(tmp_path)


def test_dictionary_add_list_terms_and_persistence(d):
    t = Topics(d)
    e = d.add("  Kubernetes  ", ["cube are netties", "kube er net ease"])
    assert e["term"] == "Kubernetes" and e["aliases"] == ["cube are netties", "kube er net ease"] and e["id"]
    d.add("anthropic")
    d.add("Zed")
    assert [x["term"] for x in d.list()] == ["anthropic", "Kubernetes", "Zed"]        # case-insensitive sort
    assert d.terms() == ["anthropic", "Kubernetes", "Zed"]
    assert t.take() == ["dictionary"] * 3
    again = reload(d)
    assert again.list() == d.list()


def test_dictionary_returns_copies(d):
    e = d.add("GitHub", ["get hub"])
    e["aliases"].append("hacked")
    d.list()[0]["term"] = "hacked"
    assert d.list()[0] == {"id": e["id"], "term": "GitHub", "aliases": ["get hub"]}


@pytest.mark.parametrize("bad", ["", "   ", None, 5, "x" * 81])
def test_dictionary_rejects_invalid_terms_with_a_human_message(d, bad):
    with pytest.raises(ValueError) as exc:
        d.add(bad)
    assert len(str(exc.value)) > 10 and "Traceback" not in str(exc.value)
    assert d.list() == []


def test_dictionary_rejects_duplicates_and_alias_collisions(d):
    d.add("GitHub", ["get hub"])
    with pytest.raises(ValueError, match="already in your dictionary"):
        d.add("github")
    with pytest.raises(ValueError, match="already belongs to"):
        d.add("GitLab", ["Get Hub"])
    with pytest.raises(ValueError, match="already belongs to"):
        d.add("Hub", ["GITHUB"])
    with pytest.raises(ValueError, match="sounds-like spelling"):
        d.add("get hub")
    assert d.terms() == ["GitHub"]


def test_dictionary_alias_cleaning(d):
    e = d.add("PyTorch", ["pie torch", "Pie  Torch", "pytorch", "", 7, "pie torch,", "y" * 81])
    assert e["aliases"] == ["pie torch"]


def test_dictionary_update(d):
    a = d.add("GitHub", ["get hub"])
    b = d.add("GitLab")
    t = Topics(d)
    assert d.update(a["id"], term="Github Inc")["aliases"] == ["get hub"]            # aliases kept when omitted
    assert d.update(a["id"], aliases=["git hub", "get hub"])["term"] == "Github Inc"
    assert d.update(a["id"], aliases=[])["aliases"] == []
    assert d.update(a["id"], term="GITHUB INC")["term"] == "GITHUB INC"               # re-casing itself is allowed
    assert t.take() == ["dictionary"] * 4
    with pytest.raises(ValueError, match="already in your dictionary"):
        d.update(a["id"], term="gitlab")
    with pytest.raises(ValueError, match="already belongs"):
        d.update(a["id"], aliases=["GitLab"])
    with pytest.raises(ValueError, match="no longer exists"):
        d.update("nope", term="x")
    with pytest.raises(ValueError):
        d.update(b["id"], term="  ")
    assert reload(d).list() == d.list()


def test_dictionary_remove(d):
    a = d.add("GitHub")
    t = Topics(d)
    assert d.remove(a["id"]) is True
    assert d.remove(a["id"]) is False
    assert t.take() == ["dictionary"]
    assert d.list() == [] and reload(d).list() == []


@pytest.mark.parametrize("text,expected", [
    ("push it to get hub today", "push it to GitHub today"),
    ("push it to Get  Hub today", "push it to GitHub today"),
    ("push it to github", "push it to GitHub"),                       # wrong casing of the term itself
    ("push it to git hub", "push it to GitHub"),                      # camel-case split is an implicit alias
    ("push it to git-hub", "push it to GitHub"),
    ("github's api", "GitHub's api"),
    ("usegithub and githubber", "usegithub and githubber"),            # whole words only
    ("i love see plus plus", "i love C++"),
    ("c++ and C++", "C++ and C++"),
    ("use node js here", "use Node.js here"),
    ("nothing to change", "nothing to change"),
    ("", ""),
])
def test_dictionary_apply(d, text, expected):
    d.add("GitHub", ["get hub"])
    d.add("C++", ["see plus plus"])
    d.add("Node.js", ["node js"])
    assert d.apply(text) == expected


def test_dictionary_apply_prefers_the_longest_match_and_follows_changes(d):
    assert d.apply("open ai") == "open ai"                              # empty dictionary
    a = d.add("OpenAI", ["open a i"])
    d.add("OpenAI API", ["open a i a p i"])
    assert d.apply("call the open a i a p i now") == "call the OpenAI API now"
    assert d.apply("ask open ai") == "ask OpenAI"
    d.update(a["id"], term="Open-AI", aliases=["open a i"])
    assert d.apply("ask open a i") == "ask Open-AI"
    d.remove(a["id"])
    assert d.apply("ask open a i") == "ask open a i"


@pytest.mark.parametrize("term,aliases,text,expected", [
    ("Wi-Fi", [], "the wi-fi is down", "the Wi-Fi is down"),
    ("Wi-Fi", [], "the wi fi is down", "the Wi-Fi is down"),
    ("GitHub", ["get-hub"], "push to get-hub now", "push to GitHub now"),
    ("GitHub", ["get-hub"], "push to get hub now", "push to GitHub now"),
])
def test_c44_dictionary_apply_handles_hyphenated_terms_and_aliases(d, term, aliases, text, expected):
    d.add(term, aliases)
    assert d.apply(text) == expected


def test_dictionary_apply_is_safe_with_regex_metacharacters(d):
    d.add("C#", ["see sharp"])
    d.add("a.b(c)", ["weird [thing]"])
    assert d.apply("i write see sharp and weird [thing]") == "i write C# and a.b(c)"


def test_dictionary_export_import_roundtrip_all_formats(d, tmp_path):
    d.add("GitHub", ["get hub", "git hub"])
    d.add("Kubernetes")
    for ext in ("json", "csv", "txt"):
        target = tmp_path / f"dict.{ext}"
        assert d.export(str(target)) == 2
        fresh = Dictionary(tmp_path / f"fresh-{ext}")
        assert fresh.import_(str(target)) == 2
        assert [(e["term"], e["aliases"]) for e in fresh.list()] == [("GitHub", ["get hub", "git hub"]),
                                                                    ("Kubernetes", [])]


def test_dictionary_export_file_contents(d, tmp_path):
    d.add("GitHub", ["get hub"])
    d.export(str(tmp_path / "d.csv"))
    assert list(csv.reader((tmp_path / "d.csv").read_text(encoding="utf-8").splitlines())) == [
        ["term", "aliases"], ["GitHub", "get hub"]]
    d.export(str(tmp_path / "d.txt"))
    assert (tmp_path / "d.txt").read_text(encoding="utf-8") == "GitHub: get hub\n"
    d.export(str(tmp_path / "d.json"))
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8")) == {
        "entries": [{"term": "GitHub", "aliases": ["get hub"]}]}


def test_dictionary_import_merges_skips_invalid_and_conflicting_rows(d, tmp_path):
    d.add("GitHub", ["get hub"])
    d.add("GitLab", ["git lab"])
    src = tmp_path / "in.json"
    src.write_text(json.dumps([
        "Kubernetes",                                         # plain string
        {"term": "github", "aliases": ["gith ub", "git lab"]},   # merges into GitHub; "git lab" belongs to GitLab
        {"term": "  "},                                       # invalid
        {"term": "get hub"},                                  # is an alias of GitHub -> skipped
        {"nope": 1}, 17,
        {"term": "Vercel", "aliases": ["ver sell", 5]},
    ]), encoding="utf-8")
    t = Topics(d)
    assert d.import_(str(src)) == 2
    by_term = {e["term"]: e["aliases"] for e in d.list()}
    assert by_term == {"GitHub": ["get hub", "gith ub"], "GitLab": ["git lab"], "Kubernetes": [],
                       "Vercel": ["ver sell"]}
    assert "dictionary" in t.take()
    assert reload(d).list() == d.list()


def test_dictionary_import_txt_and_csv_details(d, tmp_path):
    txt = tmp_path / "in.txt"
    txt.write_text("\ufeff# my words\n\nKubernetes\nGitHub: get hub, git hub\n   \n", encoding="utf-8")
    assert d.import_(str(txt)) == 2
    csvf = tmp_path / "in.csv"
    csvf.write_text("term,aliases\nVercel,ver sell,\n,,\nSupabase\n", encoding="utf-8")
    assert d.import_(str(csvf)) == 2
    assert {e["term"]: e["aliases"] for e in d.list()} == {
        "GitHub": ["get hub", "git hub"], "Kubernetes": [], "Supabase": [], "Vercel": ["ver sell"]}


@pytest.mark.parametrize("name,content,match", [
    ("missing.json", None, "Couldn't read"),
    ("bad.json", "{ nope", "not a valid JSON"),
    ("obj.json", '{"entries": 5}', "does not contain a list"),
    ("num.json", "5", "does not contain a list"),
    ("binary.txt", b"\xff\xfe\x00\xd8", "Couldn't read"),
])
def test_dictionary_import_errors_are_human_messages(d, tmp_path, name, content, match):
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        d.import_(str(path))
    assert d.list() == []


def test_dictionary_export_to_an_unwritable_place_is_a_human_error(d, tmp_path):
    d.add("GitHub")
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="Couldn't write"):
        d.export(str(blocker / "out.json"))


# ---- auto-learn
def test_suggest_queue_accept_and_dismiss(d):
    t = Topics(d)
    s1 = d.suggest("GitHub", ["get hub"])
    assert s1["term"] == "GitHub" and s1["aliases"] == ["get hub"] and s1["source"] == "cleanup" and s1["ts"] > 0
    assert d.suggest("github", ["get hub"]) is None                      # already queued, nothing new
    merged = d.suggest("GitHub", ["git hub"])
    assert merged["id"] == s1["id"] and merged["aliases"] == ["get hub", "git hub"]
    s2 = d.suggest("PyTorch", ["pie torch"], source="manual")
    assert [s["term"] for s in d.suggestions()] == ["PyTorch", "GitHub"]  # newest first
    assert t.take() == ["suggestions"] * 3

    entry = d.accept_suggestion(s1["id"])
    assert entry["term"] == "GitHub" and entry["aliases"] == ["get hub", "git hub"]
    assert sorted(t.take()) == ["dictionary", "suggestions"]
    assert d.dismiss_suggestion(s2["id"]) is True
    assert d.dismiss_suggestion(s2["id"]) is False
    assert d.suggestions() == []
    assert d.suggest("pytorch", ["pie torch"]) is None, "a dismissed term must never be suggested again"
    with pytest.raises(ValueError, match="no longer available"):
        d.accept_suggestion(s1["id"])
    again = reload(d)
    assert again.suggestions() == [] and again.terms() == ["GitHub"]
    assert again.suggest("PyTorch") is None, "dismissals must survive a restart"


def test_suggest_for_a_known_term_only_offers_new_aliases(d):
    d.add("GitHub", ["get hub"])
    assert d.suggest("GitHub") is None
    assert d.suggest("GitHub", ["get hub", "git hub"]) is None            # "git hub" is the implicit camel split
    s = d.suggest("GitHub", ["gith up"])
    assert s["aliases"] == ["gith up"]
    assert d.accept_suggestion(s["id"])["aliases"] == ["get hub", "gith up"]
    assert len(d.list()) == 1


def test_accepting_a_suggestion_drops_aliases_that_now_belong_to_another_term(d):
    s = d.suggest("GitHub", ["the hub", "get hub"])
    d.add("Hubble", ["the hub"])
    assert d.accept_suggestion(s["id"])["aliases"] == ["get hub"]


def test_adding_or_importing_a_term_removes_its_pending_suggestion(d, tmp_path):
    d.suggest("GitHub", ["get hub"])
    d.suggest("Vercel")
    t = Topics(d)
    d.add("github")
    assert sorted(t.take()) == ["dictionary", "suggestions"]
    src = tmp_path / "in.txt"
    src.write_text("Vercel\n", encoding="utf-8")
    d.import_(str(src))
    assert d.suggestions() == []


def test_suggestion_queue_is_capped(d):
    for i in range(S._MAX_SUGGESTIONS + 10):
        d.suggest(f"Term{i}")
    terms = [s["term"] for s in d.suggestions()]
    assert len(terms) == S._MAX_SUGGESTIONS and terms[0] == f"Term{S._MAX_SUGGESTIONS + 9}"


@pytest.mark.parametrize("bad", ["", None, "x" * 81])
def test_suggest_ignores_invalid_terms(d, bad):
    assert d.suggest(bad) is None and d.suggestions() == []


def test_learn_queues_names_the_cleanup_had_to_repair(d):
    out = d.learn("i pushed the branch to get hub and wrote the read me", "I pushed the branch to GitHub and wrote the ReadMe.")
    assert [(s["term"], s["aliases"]) for s in out] == [("GitHub", ["get hub"]), ("ReadMe", ["read me"])]
    assert d.learn("i pushed to get hub", "I pushed to GitHub.") == []     # nothing new the second time


@pytest.mark.parametrize("raw,text,term,alias", [
    ("we trained it with pie torch", "We trained it with PyTorch.", "PyTorch", "pie torch"),
])
def test_c48_learn_never_suggests_a_fragment_of_the_spoken_phrase_as_alias(d, raw, text, term, alias):
    d.learn(raw, text)
    sugs = {s["term"]: s["aliases"] for s in d.suggestions()}
    for word in raw.split():
        assert [word] != sugs.get(term), f"alias {word!r} alone would rewrite an ordinary word into {term}"
    if term in sugs:
        assert sugs[term] in ([alias], [])


def test_suggestions_and_dismissals_are_loaded_from_disk(tmp_path):
    (tmp_path / "dictionary.json").write_text(json.dumps({"version": 1, "entries": [], "dismissed": ["Vercel", 5],
        "suggestions": [{"id": "s1", "term": " GitHub ", "aliases": ["get hub", 3], "ts": 12.5, "source": "manual"},
                        {"term": "PyTorch"}, {"term": ""}, "junk"]}), encoding="utf-8")
    d = Dictionary(tmp_path)
    sugs = d.suggestions()
    assert sugs[0] == {"id": "s1", "term": "GitHub", "aliases": ["get hub"], "ts": 12.5, "source": "manual"}
    assert sugs[1]["term"] == "PyTorch" and sugs[1]["aliases"] == [] and sugs[1]["source"] == "cleanup" and sugs[1]["id"]
    assert len(sugs) == 2 and d.suggest("vercel") is None


def test_learn_handles_tokens_and_transcripts_without_letters_or_digits(d):
    assert d.learn("...", "GitHub") == []
    assert Dictionary._heard_as("'", ["x"]) is None
    assert Dictionary._heard_as("GitHub", ["'", "--"]) is None


def test_backup_failure_is_ignored(tmp_path, monkeypatch):
    import pathlib as _pl
    (tmp_path / "dictionary.json").write_text("{ broken", encoding="utf-8")
    monkeypatch.setattr(_pl.Path, "write_bytes", lambda self, data: (_ for _ in ()).throw(PermissionError("ro")))
    assert Dictionary(tmp_path).list() == []


def test_base_store_requires_load_and_dump():
    with pytest.raises(NotImplementedError):
        S._JsonStore.__new__(S._JsonStore)._load(None)
    with pytest.raises(NotImplementedError):
        S._JsonStore.__new__(S._JsonStore)._dump()


def test_learn_ignores_ordinary_words_unchanged_names_and_unrelated_insertions(d):
    assert d.learn("hello there my friend", "Hello there, my friend.") == []
    assert d.learn("i use GitHub daily", "I use GitHub daily.") == []       # already written that way
    assert d.learn("we talked about it", "We talked about NASA.") == []     # not derived from the audio
    assert d.learn("", "GitHub") == [] and d.learn("get hub", "") == []


def test_learn_recased_only_terms_get_no_alias_and_limit_is_respected(d):
    out = d.learn("the api uses oauth and graphql and grpc", "The API uses OAuth and GraphQL and gRPC.", limit=2)
    assert len(out) == 2 and all(s["aliases"] == [] for s in out)


def test_learn_skips_terms_already_in_the_dictionary_or_dismissed(d):
    d.add("GitHub", ["get hub"])
    s = d.suggest("PyTorch")
    d.dismiss_suggestion(s["id"])
    assert d.learn("get hub and pie torch", "GitHub and PyTorch") == []


# =========================================================================== snippets
@pytest.fixture
def sn(tmp_path):
    return Snippets(tmp_path)


def test_snippets_crud_and_persistence(sn):
    t = Topics(sn)
    a = sn.add("  My address. ", "221B Baker Street\nLondon")
    assert a["trigger"] == "My address" and a["text"] == "221B Baker Street\nLondon" and a["id"]
    b = sn.add("calendar link", "https://cal.example/dan")
    assert [e["trigger"] for e in sn.list()] == ["calendar link", "My address"]
    assert sn.update(a["id"], text="742 Evergreen Terrace")["trigger"] == "My address"
    assert sn.update(a["id"], trigger="home address")["text"] == "742 Evergreen Terrace"
    assert sn.update(a["id"], trigger="HOME ADDRESS")["trigger"] == "HOME ADDRESS"      # renaming itself is fine
    assert sn.remove(b["id"]) is True and sn.remove(b["id"]) is False
    assert t.take() == ["snippets"] * 6
    assert reload(sn).list() == sn.list() == [{"id": a["id"], "trigger": "HOME ADDRESS",
                                                "text": "742 Evergreen Terrace"}]


@pytest.mark.parametrize("trigger,text,match", [
    ("", "x", "phrase you will say"), ("...", "x", "phrase you will say"), (None, "x", "phrase you will say"),
    ("ok", "", "text this snippet"), ("ok", "   ", "text this snippet"), ("ok", None, "text this snippet"),
    ("x" * 81, "y", "at most"),
])
def test_snippets_reject_invalid_input(sn, trigger, text, match):
    with pytest.raises(ValueError, match=match):
        sn.add(trigger, text)
    assert sn.list() == []


def test_snippets_reject_duplicates_and_missing_ids(sn):
    a = sn.add("my address", "x")
    b = sn.add("sig", "y")
    with pytest.raises(ValueError, match="already have a snippet"):
        sn.add("My Address!", "z")
    with pytest.raises(ValueError, match="already have a snippet"):
        sn.update(b["id"], trigger="my address")
    with pytest.raises(ValueError, match="no longer exists"):
        sn.update("nope", text="z")
    with pytest.raises(ValueError):
        sn.update(a["id"], text=" ")
    assert {e["trigger"]: e["text"] for e in sn.list()} == {"my address": "x", "sig": "y"}


@pytest.mark.parametrize("spoken,expected", [
    ("my address", "221B Baker Street\nLondon"),
    ("My address.", "221B Baker Street\nLondon"),                 # whole utterance: exactly the snippet
    ("  MY ADDRESS!  ", "221B Baker Street\nLondon"),
    ("My, address.", "221B Baker Street\nLondon"),
    ("send it to my address please", "send it to my address please"),   # in a sentence: only after a lead word
    ("send it to, insert my address please", "send it to, 221B Baker Street\nLondon please"),
    ("Send it to, insert my address.", "Send it to, 221B Baker Street\nLondon."),
    ("my addresses are many", "my addresses are many"),          # whole words only
    ("the signature block", "the signature block"),
    ("type sig and paste my address", "Best,\nDan and 221B Baker Street\nLondon"),
    ("my address book", "ADDRESS BOOK"),                          # longest trigger wins
    ("", ""),
])
def test_snippets_expand(sn, spoken, expected):
    sn.add("my address", "221B Baker Street\nLondon")
    sn.add("sig", "Best,\nDan")
    sn.add("my address book", "ADDRESS BOOK")
    assert sn.expand(spoken) == expected


def test_snippets_expand_is_literal_and_tracks_changes(sn):
    assert sn.expand("my address") == "my address"                 # no snippets yet
    a = sn.add("regex test", r"\1 \g<0> $1 (.*)")
    assert sn.expand("a regex test b") == "a regex test b"            # in a sentence: only after a lead word
    assert sn.expand("a insert regex test b") == r"a \1 \g<0> $1 (.*) b"
    sn.update(a["id"], trigger="c++ tip", text="use RAII")
    assert sn.expand("regex test") == "regex test"
    assert sn.expand("c++ tip") == "use RAII"
    sn.remove(a["id"])
    assert sn.expand("c++ tip") == "c++ tip"


def test_snippets_export_import_roundtrip_and_errors(sn, tmp_path):
    sn.add("my address", "221B Baker Street\nLondon")
    sn.add("sig", "Best,\nDan")
    out = tmp_path / "snips.json"
    assert sn.export(str(out)) == 2
    assert json.loads(out.read_text(encoding="utf-8")) == {"entries": [
        {"trigger": "my address", "text": "221B Baker Street\nLondon"}, {"trigger": "sig", "text": "Best,\nDan"}]}
    fresh = Snippets(tmp_path / "fresh")
    t = Topics(fresh)
    assert fresh.import_(str(out)) == 2
    assert fresh.import_(str(out)) == 0                             # duplicates are skipped
    assert t.take() == ["snippets", "snippets"]
    assert [(e["trigger"], e["text"]) for e in fresh.list()] == [("my address", "221B Baker Street\nLondon"),
                                                                 ("sig", "Best,\nDan")]
    plain = tmp_path / "plain.json"
    plain.write_text("\ufeff" + json.dumps([{"trigger": "eta", "text": "On my way"}, "junk", {"trigger": ""}]),
                     encoding="utf-8")
    assert fresh.import_(str(plain)) == 1
    for name, content, match in [("missing.json", None, "Couldn't read"), ("bad.json", "{", "not a valid JSON"),
                                 ("num.json", "5", "does not contain a list")]:
        p = tmp_path / name
        if content is not None:
            p.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            fresh.import_(str(p))
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="Couldn't write"):
        sn.export(str(blocker / "x.json"))


# =========================================================================== history
NOW = 1_800_000_000.0


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def h(tmp_path):
    return History(tmp_path, clock=Clock())


def test_make_entry_and_count_words():
    assert count_words("one  two\nthree") == 3 and count_words("") == 0
    e = make_entry("one two three four", "raw", app_exe="Slack.EXE", app_title="general", duration_s=2.0,
                   latency_ms=612.4, ts=NOW)
    assert e == {"id": e["id"], "ts": NOW, "raw": "raw", "text": "one two three four", "app_exe": "slack.exe",
                 "app_title": "general", "duration_s": 2.0, "words": 4, "wpm": 120.0, "mode": "dictate",
                 "latency_ms": 612}
    assert len(e["id"]) == 12
    assert make_entry("a b", duration_s=0.5)["wpm"] == 0.0            # too short to measure a rate
    assert make_entry("a b", duration_s=10, mode="command")["wpm"] == 0.0
    assert make_entry("a b", mode="weird")["mode"] == "dictate"
    assert make_entry("a b", duration_s=-3)["duration_s"] == 0.0
    assert make_entry("x", raw=None, app_exe=None, app_title=None, duration_s=None, latency_ms=None)["raw"] == ""
    assert abs(make_entry("x")["ts"] - __import__("time").time()) < 5


def test_history_add_list_get_count_and_persistence(h):
    t = Topics(h)
    a = h.add("first entry", "first entry raw", 400, app_exe="slack.exe", ts=NOW - 30)
    b = h.add("second entry", ts=NOW - 20)
    c = h.add("third entry", ts=NOW - 10, mode="command", duration_s=3, app_title="Doc")
    assert t.take() == ["history"] * 3
    assert [e["text"] for e in h.list()] == ["third entry", "second entry", "first entry"]
    assert [e["text"] for e in h.list(limit=2)] == ["third entry", "second entry"]
    assert [e["text"] for e in h.list(limit=2, offset=2)] == ["first entry"]
    assert h.list(limit=0) == [] and h.list(limit=5, offset=99) == [] and len(h.list(limit=None)) == 3
    assert [e["text"] for e in h.list(limit=-4, offset=-9)] == []
    assert h.snapshot() == h.list(limit=None)
    assert h.count() == 3
    assert h.get(a["id"])["latency_ms"] == 400 and h.get("nope") is None
    assert c["mode"] == "command" and b["raw"] == ""
    again = History(h.path, clock=Clock())
    assert again.snapshot() == h.snapshot()
    assert json.loads(h.path.read_text(encoding="utf-8"))["version"] == 2


def test_history_returns_copies(h):
    a = h.add("text", ts=NOW)
    a["text"] = "hacked"
    h.list()[0]["text"] = "hacked"
    h.get(a["id"])["text"] = "hacked"
    h.search("text")[0]["text"] = "hacked"
    assert h.list()[0]["text"] == "text"


def test_history_keeps_newest_first_even_for_out_of_order_inserts(h):
    h.add("new", ts=NOW)
    h.add("old", ts=NOW - 500)
    h.add("middle", ts=NOW - 100)
    assert [e["text"] for e in h.list()] == ["new", "middle", "old"]


def test_history_delete_and_clear(h):
    a = h.add("one", ts=NOW - 2)
    h.add("two", ts=NOW - 1)
    t = Topics(h)
    assert h.delete(a["id"]) is True and h.delete(a["id"]) is False
    assert [e["text"] for e in h.list()] == ["two"]
    h.clear()
    assert h.count() == 0 and t.take() == ["history", "history"]
    assert History(h.path).count() == 0


def test_history_search(h):
    h.add("Ship the quarterly report", "ship the um quarterly report", app_exe="outlook.exe",
          app_title="RE: Budget - Message", ts=NOW - 3)
    h.add("lunch at noon?", app_exe="slack.exe", app_title="#general", ts=NOW - 2)
    h.add("Deploy to staging", ts=NOW - 1)
    assert [e["text"] for e in h.search("REPORT")] == ["Ship the quarterly report"]
    assert [e["text"] for e in h.search("report ship")] == ["Ship the quarterly report"]      # every word, any order
    assert [e["text"] for e in h.search("um")] == ["Ship the quarterly report"]               # raw text
    assert [e["text"] for e in h.search("slack")] == ["lunch at noon?"]                       # app
    assert [e["text"] for e in h.search("budget")] == ["Ship the quarterly report"]           # window title
    assert h.search("report lunch") == [] and h.search("zzz") == []
    assert len(h.search("")) == 3 and len(h.search("   ")) == 3 and len(h.search(None)) == 3
    assert [e["text"] for e in h.search("", limit=1)] == ["Deploy to staging"]
    assert h.search("e", limit=0) == []


def test_history_size_cap(tmp_path):
    h = History(tmp_path, limit=3, clock=Clock())
    for i in range(5):
        h.add(f"entry {i}", ts=NOW + i)
    assert [e["text"] for e in h.list()] == ["entry 4", "entry 3", "entry 2"]
    t = Topics(h)
    h.configure(limit=1)
    assert [e["text"] for e in h.list()] == ["entry 4"] and t.take() == ["history"]
    h.configure(limit=10)
    assert t.take() == []                                              # nothing pruned -> no notification
    h.configure(limit=0)                                               # never drops below one entry
    assert h.count() == 1
    assert History(h.path).count() == 1


def test_history_retention(tmp_path):
    clock = Clock()
    h = History(tmp_path, retention_days=7, clock=clock)
    h.add("ten days old", ts=NOW - 10 * 86400)                          # pruned immediately
    h.add("six days old", ts=NOW - 6 * 86400)
    h.add("today", ts=NOW)
    assert [e["text"] for e in h.list()] == ["today", "six days old"]
    assert h.prune() is False
    clock.t = NOW + 2 * 86400
    t = Topics(h)
    assert h.prune() is True and t.take() == ["history"]
    assert [e["text"] for e in h.list()] == ["today"]
    h.configure(retention_days=0)                                       # forever
    h.add("ancient", ts=NOW - 900 * 86400)
    assert h.count() == 2
    h.configure(retention_days=30)
    assert [e["text"] for e in h.list()] == ["today"]
    reopened = History(h.path, retention_days=1, clock=Clock(NOW + 5 * 86400))
    assert reopened.count() == 0, "retention must be applied when the file is loaded"


def test_history_migrates_the_v1_flat_list(tmp_path):
    path = tmp_path / "history.json"
    v1 = [{"ts": NOW - 5, "text": "newest", "raw": "newest raw", "ms": 640},
          {"ts": NOW - 50, "text": "older", "raw": "", "ms": 0},
          {"text": "no timestamp"}, {"ts": NOW, "text": 5}, "garbage", None]
    path.write_text(json.dumps(v1), encoding="utf-8")
    h = History(tmp_path)
    items = h.snapshot()
    assert sorted(e["text"] for e in items) == ["newest", "no timestamp", "older"]
    newest = next(e for e in items if e["text"] == "newest")
    assert newest["raw"] == "newest raw" and newest["latency_ms"] == 640 and newest["words"] == 1
    assert newest["mode"] == "dictate" and newest["app_exe"] == "" and len(newest["id"]) == 12
    assert all(set(e) == set(S._HISTORY_FIELDS) for e in items)
    assert json.loads((tmp_path / "history.v1.json").read_text(encoding="utf-8")) == v1, "v1 file must be kept"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["version"] == 2 and len(on_disk["items"]) == 3
    ids = [e["id"] for e in items]
    assert sorted(e["id"] for e in History(tmp_path).snapshot()) == sorted(ids), "ids must be stable after migration"


def test_history_v2_file_roundtrips_ids_and_tolerates_partial_entries(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps({"version": 2, "items": [
        {"id": "keep-this-id", "ts": NOW, "text": "full", "raw": 7, "duration_s": "x", "mode": None, "words": 99},
        {"text": "minimal"}]}), encoding="utf-8")
    h = History(tmp_path)
    full = h.get("keep-this-id")
    assert full["raw"] == "" and full["duration_s"] == 0.0 and full["mode"] == "dictate" and full["words"] == 1
    assert h.count() == 2


def test_history_export_formats(h, tmp_path):
    h.add("Hello, \"world\"\nsecond line", "hello world second line", app_exe="slack.exe", app_title="#general",
          duration_s=2, ts=NOW - 60)
    h.add("Plain", ts=NOW)
    stamp = dt.datetime.fromtimestamp(NOW).strftime("%Y-%m-%d %H:%M:%S")

    assert h.export(str(tmp_path / "h.txt")) == 2
    txt = (tmp_path / "h.txt").read_text(encoding="utf-8")
    assert txt.startswith(f"[{stamp}]\nPlain\n\n[") and 'slack.exe\nHello, "world"\nsecond line\n\n' in txt

    assert h.export(str(tmp_path / "h.json")) == 2
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8")) == h.snapshot()

    assert h.export(str(tmp_path / "h.csv")) == 2
    rows = list(csv.reader((tmp_path / "h.csv").read_text(encoding="utf-8").splitlines(keepends=True)))
    assert rows[0] == ["time", "app", "window", "mode", "words", "wpm", "duration_s", "text", "raw"]
    assert rows[1][0] == stamp and rows[1][7] == "Plain"
    assert rows[2][1:4] == ["slack.exe", "#general", "dictate"] and rows[2][7] == 'Hello, "world"\nsecond line'

    assert h.export(str(tmp_path / "noext"), fmt="JSON") == 2           # explicit format wins
    assert json.loads((tmp_path / "noext").read_text(encoding="utf-8"))
    assert h.export(str(tmp_path / "defaults-to-txt")) == 2
    with pytest.raises(ValueError, match=r"\.txt, \.json or \.csv"):
        h.export(str(tmp_path / "h.pdf"))
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="Couldn't write"):
        h.export(str(blocker / "h.txt"))
    empty = History(tmp_path / "empty")
    assert empty.export(str(tmp_path / "e.csv")) == 0


@pytest.mark.parametrize("ts", ["NaN", "1e20", "-5", "Infinity"])
def test_c45_entries_with_absurd_timestamps_cannot_break_export_or_ordering(tmp_path, ts):
    (tmp_path / "history.json").write_text('{"version": 2, "items": [{"text": "odd", "ts": %s}, '
                                           '{"text": "fine", "ts": %r}]}' % (ts, NOW), encoding="utf-8")
    h = History(tmp_path)
    for e in h.snapshot():
        assert 0 < e["ts"] < 4e9, e
    for fmt in ("txt", "csv", "json"):
        assert h.export(str(tmp_path / f"out.{fmt}")) == 2


def test_history_is_thread_safe(tmp_path):
    h = History(tmp_path, limit=10_000)
    errors = []

    def writer(k):
        try:
            for i in range(25):
                h.add(f"writer {k} entry {i}")
                h.search("entry")
                h.list(limit=5)
        except Exception as exc:          # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and h.count() == 150
    assert History(tmp_path, limit=10_000).count() == 150
    assert len({e["id"] for e in h.snapshot()}) == 150


# =========================================================================== stats
def day(offset: int) -> float:
    """Noon, ``offset`` days from the fake 'today'."""
    base = dt.datetime.fromtimestamp(NOW).replace(hour=12, minute=0, second=0, microsecond=0)
    return (base + dt.timedelta(days=offset)).timestamp()


@pytest.fixture
def st(tmp_path):
    return Stats(tmp_path, clock=Clock(day(0)))


def test_stats_empty(st):
    assert st.is_new is True
    assert st.summary() == {"total_words": 0, "sessions": 0, "seconds_spoken": 0.0, "avg_wpm": 0.0,
                            "time_saved_s": 0.0, "streak_days": 0, "today_words": 0}
    assert st.top_apps() == []
    rows = st.daily(3)
    assert [r["words"] for r in rows] == [0, 0, 0]
    assert rows[-1]["date"] == dt.date.fromtimestamp(day(0)).isoformat()


def test_stats_record_summary_and_persistence(st):
    t = Topics(st)
    st.record(120, 60.0, "Slack.exe")                     # 120 wpm
    st.record(60, 60.0, "slack.exe")                      # 60 wpm
    st.record(20, 0.0, "code.exe")                        # untimed (e.g. command mode): not in the rate
    st.record(0, 5.0, "x.exe")
    st.record(-3)
    assert t.take() == ["stats"] * 3
    s = st.summary()
    assert s["total_words"] == 200 and s["sessions"] == 3 and s["seconds_spoken"] == 120.0
    assert s["avg_wpm"] == 90.0                           # 180 timed words in 2 minutes
    assert s["time_saved_s"] == 150.0                     # typing 180 words at 40 wpm = 270 s, minus 120 s spoken
    assert s["today_words"] == 200 and s["streak_days"] == 1
    assert st.top_apps() == [{"exe": "slack.exe", "words": 180, "sessions": 2},
                             {"exe": "code.exe", "words": 20, "sessions": 1}]
    assert st.top_apps(1) == [{"exe": "slack.exe", "words": 180, "sessions": 2}] and st.top_apps(0) == []
    again = Stats(st.path, clock=Clock(day(0)))
    assert again.summary() == s and again.is_new is False


def test_stats_time_saved_is_never_negative(st):
    st.record(10, 600.0)                                  # slower than typing
    assert st.summary()["time_saved_s"] == 0.0 and st.summary()["avg_wpm"] == 1.0


def test_stats_streak_counts_consecutive_days_and_survives_an_unused_today(tmp_path):
    clock = Clock(day(0))
    st = Stats(tmp_path, clock=clock)
    for back in (1, 2, 3, 5):
        st.record(10, 5.0, ts=day(-back))
    assert st.summary()["streak_days"] == 3, "yesterday, -2, -3 are consecutive; -5 is not"
    assert st.summary()["today_words"] == 0
    st.record(7, 3.0)
    assert st.summary()["streak_days"] == 4 and st.summary()["today_words"] == 7
    clock.t = day(2)                                      # two days later, nothing dictated since
    assert st.summary()["streak_days"] == 0 and st.summary()["today_words"] == 0


def test_stats_daily_series(st):
    st.record(10, 5.0, ts=day(-2))
    st.record(5, 2.5, ts=day(-2))
    st.record(30, 9.0, ts=day(0))
    st.record(99, 1.0, ts=day(-40))
    rows = st.daily(4)
    assert [r["date"] for r in rows] == [dt.date.fromtimestamp(day(-k)).isoformat() for k in (3, 2, 1, 0)]
    assert [(r["words"], r["sessions"], r["seconds"]) for r in rows] == [(0, 0, 0.0), (15, 2, 7.5), (0, 0, 0.0),
                                                                        (30, 1, 9.0)]
    assert len(st.daily(0)) == 1 and len(st.daily(14)) == 14 and len(st.daily()) == 14


def test_stats_reset(st):
    st.record(10, 5.0)
    t = Topics(st)
    st.reset()
    assert st.summary()["total_words"] == 0 and t.take() == ["stats"]
    assert Stats(st.path).summary()["total_words"] == 0


def test_stats_seed_from_history_runs_once(st):
    items = [make_entry("one two three four five six", duration_s=3.0, app_exe="slack.exe", ts=day(-1)),
             make_entry("command result words", mode="command", ts=day(-1)),
             make_entry("", ts=day(-1)),
             make_entry("seven eight", ts=day(0))]
    st.seed_from_history(items)
    s = st.summary()
    assert s["total_words"] == 8 and s["sessions"] == 2 and s["streak_days"] == 2 and st.is_new is False
    st.seed_from_history([make_entry("more words here", ts=day(0))])
    assert st.summary()["total_words"] == 8, "seeding must not double count once stats exist"


def test_stats_loader_sanitises_values(tmp_path):
    (tmp_path / "stats.json").write_text(json.dumps({"version": 1, "days": {
        "2026-11-10": {"words": "many", "sessions": 2, "seconds": None, "timed_words": True,
                       "apps": {"a.exe": {"words": 5, "sessions": "x"}, "b.exe": 7, "3": {"words": 1}}},
        "not-a-date": {"words": 5}, "2026-11-11": 5}}), encoding="utf-8")
    st = Stats(tmp_path)
    assert st.summary()["total_words"] == 0 and st.summary()["sessions"] == 2
    assert {"exe": "a.exe", "words": 5, "sessions": 0} in st.top_apps()


# =========================================================================== styles
@pytest.fixture
def sty(tmp_path):
    return Styles(tmp_path)


@pytest.mark.parametrize("raw,expected", [("Slack", "slack.exe"), ("slack.exe", "slack.exe"),
                                          (r"C:\Apps\Slack.EXE", "slack.exe"), ("/opt/x/Code", "code.exe"),
                                          ("  ms-teams.exe ", "ms-teams.exe"), ("", ""), (None, "")])
def test_normalize_exe(raw, expected):
    assert normalize_exe(raw) == expected


def test_styles_defaults(sty):
    assert sty.tones() == {"personal": "casual", "work": "casual", "email": "formal", "other": "formal"}
    assert set(S.CATEGORIES) == set(S.CATEGORY_LABELS) == set(S.DEFAULT_TONES)
    assert set(S.TONES) == set(S.TONE_LABELS) and Styles.CATEGORIES == S.CATEGORIES and Styles.TONES == S.TONES
    for exe, cat in [("slack.exe", "work"), ("TEAMS.EXE", "work"), ("ms-teams.exe", "work"), ("discord.exe", "personal"),
                     ("whatsapp.exe", "personal"), ("outlook.exe", "email"), ("thunderbird", "email"),
                     ("notepad.exe", "other"), ("", "other"), ("chrome.exe", "other")]:
        assert sty.category_for(exe) == cat, exe
    assert all(cat in S.CATEGORIES for cat in S.BUILTIN_APPS.values())
    assert all(exe == exe.lower() and exe.endswith(".exe") for exe in S.BUILTIN_APPS)


@pytest.mark.parametrize("title,expected", [
    ("Inbox (3) - dan@example.com - Gmail - Google Chrome", "email"),
    ("Mail - Dan - Outlook - Microsoft Edge", "email"),
    ("WhatsApp - Google Chrome", "personal"),
    ("(2) Discord | #general - Brave", "personal"),
    ("general (Channel) - Acme - Slack - Google Chrome", "work"),
    ("VOX-123 Fix the thing - Jira", "work"),
    ("Chat | Microsoft Teams", "work"),
    ("How to bake bread - Google Search", "other"),
])
def test_styles_browsers_are_refined_by_window_title(sty, title, expected):
    for browser in ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"):
        assert sty.category_for(browser, title) == expected
    assert sty.category_for("notepad.exe", title) == "other", "titles only matter for browsers"


@pytest.mark.parametrize("title", ["Signal processing - Wikipedia - Google Chrome", "NBA teams ranked - Google Chrome",
                                   "Linear algebra - Wikipedia - Mozilla Firefox", "Zoom lens review - Google Chrome"])
def test_c46_ordinary_page_titles_are_not_mistaken_for_chat_apps(sty, title):
    assert sty.category_for("chrome.exe", title) == "other"


def test_styles_tone_for_follows_category_and_settings(sty):
    assert sty.tone_for("slack.exe") == "casual" and sty.tone_for("outlook.exe") == "formal"
    assert sty.tone_for("chrome.exe", "Inbox (1) - Gmail") == "formal"
    t = Topics(sty)
    sty.set_tone("work", "very_casual")
    sty.set_tone("email", "casual")
    assert sty.tone_for("slack.exe") == "very_casual" and sty.tone_for("outlook.exe") == "casual"
    assert t.take() == ["styles", "styles"]
    assert reload(sty).tones() == {"personal": "casual", "work": "very_casual", "email": "casual", "other": "formal"}
    with pytest.raises(ValueError, match="not a style category"):
        sty.set_tone("boss", "formal")
    with pytest.raises(ValueError, match="not a writing tone"):
        sty.set_tone("work", "shouty")
    tones = sty.tones()
    tones["work"] = "formal"
    assert sty.tones()["work"] == "very_casual", "tones() must return a copy"


def test_styles_app_overrides(sty):
    t = Topics(sty)
    sty.set_app_category(r"C:\Tools\Obsidian.exe", "personal")       # new app
    sty.set_app_category("Slack", "personal")                         # override a built-in
    sty.set_app_category("chrome.exe", "work")                        # override beats the title rules
    assert t.take() == ["styles"] * 3
    assert sty.category_for("obsidian.exe") == "personal" and sty.category_for("SLACK.EXE") == "personal"
    assert sty.category_for("chrome.exe", "Inbox - Gmail") == "work"
    rows = {r["exe"]: r for r in sty.app_categories()}
    assert rows["obsidian.exe"] == {"exe": "obsidian.exe", "category": "personal", "builtin": False,
                                    "overridden": False}
    assert rows["slack.exe"] == {"exe": "slack.exe", "category": "personal", "builtin": True, "overridden": True}
    assert rows["outlook.exe"] == {"exe": "outlook.exe", "category": "email", "builtin": True, "overridden": False}
    assert [r["exe"] for r in sty.app_categories()] == sorted(rows)
    assert reload(sty).category_for("slack.exe") == "personal"

    assert sty.remove_app_category("slack") is True                   # built-in mapping applies again
    assert sty.remove_app_category("slack") is False
    assert sty.category_for("slack.exe") == "work" and t.take() == ["styles"]
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="program name"):
            sty.set_app_category(bad, "work")
    with pytest.raises(ValueError, match="not a style category"):
        sty.set_app_category("x.exe", "boss")
