"""QA "data" fixes in backups, packs and the sync folder (``voxflow.sync``).

D5   entries are matched the way they are said ("zoom link two" == "zoom link 2", "Wi-Fi" == "wi fi")
D7   a backup with a large history can be restored (it has its own, higher size limit; export checks it)
D11  a pack or backup never gives an alias that another word already owns; the user is told
"""
from __future__ import annotations

import json

import pytest

from voxflow import config as cfgmod, store, sync as S


class Profile:
    def __init__(self, root, name="me"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        self.dictionary, self.snippets = store.Dictionary(d), store.Snippets(d)
        self.styles, self.history, self.stats = store.Styles(d), store.History(d), store.Stats(d)
        self.settings = cfgmod.defaults()
        self.ds = S.DataSet(self.dictionary, self.snippets, self.styles, self.history, lambda: dict(self.settings),
                            lambda values: [], self.stats)


def _pack(path, kind, entries):
    path.write_text(json.dumps({"format": "voxflow-pack", "version": 1, "kind": kind, "name": "Team",
                                "entries": entries}), encoding="utf-8")
    return str(path)


# =========================================================================== D5
def test_a_merge_backup_never_overwrites_a_snippet_said_the_same_way(tmp_path):
    other = Profile(tmp_path, "other")
    other.snippets.add("zoom link two", "https://zoom/OLD-from-backup")
    backup = S.export_profile(other.ds, str(tmp_path / "b.voxbackup"), include_history=False)["path"]
    me = Profile(tmp_path)
    me.snippets.add("zoom link 2", "https://zoom/MINE")
    out = S.import_profile(me.ds, backup, "merge")
    assert {e["trigger"]: e["text"] for e in me.snippets.list()} == {"zoom link 2": "https://zoom/MINE"}
    assert out["counts"]["snippets"] == 0, "merge changes nothing that exists, and adds nothing that is there"


def test_a_pack_preview_counts_what_is_already_there_said_another_way(tmp_path):
    me = Profile(tmp_path)
    me.snippets.add("Sign off", "Best, Dan")
    me.dictionary.add("Wi-Fi")
    snips = _pack(tmp_path / "s.voxpack", "snippets", [{"trigger": "sign-off", "text": "Cheers"},
                                                       {"trigger": "eta", "text": "On my way"}])
    info = S.describe_pack(me.ds, snips)
    assert (info["new"], info["existing"]) == (1, 1)
    out = S.import_pack(me.ds, snips)
    assert (out["added"], out["updated"]) == (1, 1), "the existing one is reported as updated, not added"
    assert {e["trigger"]: e["text"] for e in me.snippets.list()} == {"sign-off": "Cheers", "eta": "On my way"}

    words = _pack(tmp_path / "d.voxpack", "dictionary", [{"term": "wi fi", "aliases": ["why fly"]}, "Kubernetes"])
    assert (S.describe_pack(me.ds, words)["new"], S.describe_pack(me.ds, words)["existing"]) == (1, 1)
    S.import_pack(me.ds, words)
    assert {e["term"]: e["aliases"] for e in me.dictionary.list()} == {"Kubernetes": [], "wi fi": ["why fly"]}


def test_the_dictionary_treats_a_hyphen_as_a_space_when_finding_a_term(tmp_path):
    d = store.Dictionary(tmp_path)
    d.add("Wi-Fi")
    with pytest.raises(ValueError, match="already in your dictionary"):
        d.add("wi fi")
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "dictionary.json").write_text(json.dumps({"entries": [{"term": "Wi-Fi"}, {"term": "wi fi"}]}),
                                                    encoding="utf-8")
    assert store.Dictionary(tmp_path / "x").terms() == ["Wi-Fi"], "a duplicate in the file is dropped"


def test_sync_records_and_deletions_use_the_same_keys(tmp_path):
    rec = {"x": 1, "t": 5.0, "d": "dev"}
    assert S.SyncEngine._valid("snippets", "Zoom link two", rec, 10.0)[0] == "zoom link 2"
    assert S.SyncEngine._valid("dictionary", "Wi-Fi", rec, 10.0)[0] == "wi fi"
    got = S.SyncEngine._valid("snippets", "old", {"v": {"trigger": "Sign-off", "text": "x"}, "t": 5.0, "d": "d"}, 10.0)
    assert got[0] == "sign off", "a record's key comes from its value, so an older state file is re-keyed"
    assert S._norm("snippets", None, {"trigger": "!!!", "text": "x"}) is None


# =========================================================================== D11
def test_a_pack_alias_that_another_word_owns_is_left_out_and_reported(tmp_path):
    me = Profile(tmp_path)
    me.dictionary.add("Cuba", ["cube"])
    pack = _pack(tmp_path / "p.voxpack", "dictionary", [{"term": "Kubernetes", "aliases": ["cube", "kube"]},
                                                        {"term": "Kubelet", "aliases": ["kube", "cubelet"]}])
    out = S.import_pack(me.ds, pack)
    by_term = lambda items: sorted(items, key=lambda it: it["term"])      # noqa: E731
    assert by_term(out["aliases_skipped"]) == [{"alias": "kube", "term": "Kubelet", "owner": "Kubernetes"},
                                               {"alias": "cube", "term": "Kubernetes", "owner": "Cuba"}]
    assert {e["term"]: e["aliases"] for e in me.dictionary.list()} == {
        "Cuba": ["cube"], "Kubelet": ["cubelet"], "Kubernetes": ["kube"]}
    assert me.dictionary.apply("the cube and the kube") == "the Cuba and the Kubernetes"

    again = S.import_pack(me.ds, pack)                         # an entry's own aliases never conflict with itself
    assert by_term(again["aliases_skipped"]) == by_term(out["aliases_skipped"])
    assert (again["added"], again["updated"]) == (0, 0)


def test_a_backup_reports_alias_conflicts_too_but_sync_keeps_its_old_rule(tmp_path):
    other = Profile(tmp_path, "other")
    other.dictionary.add("Kubernetes", ["cube"])
    backup = S.export_profile(other.ds, str(tmp_path / "b.voxbackup"), include_history=False)["path"]
    me = Profile(tmp_path)
    me.dictionary.add("Cuba", ["cube"])
    out = S.import_profile(me.ds, backup, "merge")
    assert out["aliases_skipped"] == [{"alias": "cube", "term": "Kubernetes", "owner": "Cuba"}]

    me.dictionary.apply_changes([{"term": "Kubernetes", "aliases": ["cube"]}])     # sync: no conflicts list
    assert me.dictionary.list()[1]["aliases"] == ["cube"]


# =========================================================================== D7
def test_a_backup_larger_than_a_pack_may_be_still_imports(tmp_path, monkeypatch):
    me = Profile(tmp_path)
    for i in range(30):
        me.history.add(f"transcript number {i} with a few words in it")
    path = S.export_profile(me.ds, str(tmp_path / "big.voxbackup"))["path"]
    monkeypatch.setattr(S, "_MAX_FILE_BYTES", 100)               # packs and sync files keep the 50 MB limit
    fresh = Profile(tmp_path, "fresh")
    assert S.import_profile(fresh.ds, path, "merge")["counts"]["history"] == 30
    with pytest.raises(ValueError, match="too large"):
        S.read_pack(path)
    assert S._MAX_BACKUP_BYTES >= 1024 ** 3, "100,000 transcripts (about 80 MB) fit easily"


def test_export_refuses_a_backup_that_could_not_be_imported_again(tmp_path, monkeypatch):
    me = Profile(tmp_path)
    me.history.add("hello")
    monkeypatch.setattr(S, "_MAX_BACKUP_BYTES", 50)
    target = tmp_path / "b.voxbackup"
    with pytest.raises(ValueError, match="too large to import again"):
        S.export_profile(me.ds, str(target))
    assert not target.exists(), "no unusable backup is left behind"
