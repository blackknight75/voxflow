"""Data-layer findings of the 1.0 QA pass, store side (``voxflow.store``): file writes outside the store lock
(R-01), statistics written behind the caller and folded into months (DL-07), exact roll-back of a bulk change
(DL-05), one copy per damaged file (DL-09), duplicate ids (DL-13), hostile JSON (DL-10), and the speed-ups that
must not change a result (D-01, D-06, DL-06, P-03)."""
from __future__ import annotations

import datetime as dt
import difflib
import json
import random
import threading
import time

import pytest

from voxflow import store as S
from voxflow.store import Dictionary, History, Snippets, Stats, Styles

DAY = 86400.0
NOW = dt.datetime(2026, 11, 19, 12).timestamp()


def on_disk(st) -> dict:
    return json.loads(st.path.read_text(encoding="utf-8"))


def wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


class Gate:
    """Holds the first file write to ``path`` until ``release`` is set, as a virus scanner with the file open does."""

    def __init__(self, monkeypatch, path):
        self.entered, self.release, self.texts = threading.Event(), threading.Event(), []
        real = S.cfgmod._atomic_write

        def write(target, text):
            if target == path:
                self.texts.append(text)
                if not self.entered.is_set():
                    self.entered.set()
                    assert self.release.wait(5)
            return real(target, text)

        monkeypatch.setattr(S.cfgmod, "_atomic_write", write)


def in_thread(fn, *args) -> threading.Thread:
    th = threading.Thread(target=fn, args=args, daemon=True)
    th.start()
    return th


# =========================================================================== R-01: the disk is never waited for
#                                                                               with the store lock held
def test_a_file_write_that_has_to_wait_keeps_no_reader_and_no_dictation_waiting(tmp_path, monkeypatch):
    d = Dictionary(tmp_path)
    d.add("Kubernetes", ["cooper netties"])
    d.COALESCE_S = 0                                   # every edit writes at once, on the editing thread
    gate = Gate(monkeypatch, d.path)
    editor = in_thread(d.add, "PyTorch")
    assert gate.entered.wait(3), "the edit is writing its file"
    got: list = []
    reader = in_thread(lambda: got.extend([d.apply("the cooper netties cluster"), d.terms(), len(d.list())]))
    reader.join(2)
    assert got == ["the Kubernetes cluster", ["Kubernetes", "PyTorch"], 2], "nobody waits for the rename"
    assert editor.is_alive(), "the edit itself returns when its file is written"
    gate.release.set()
    editor.join(3)
    assert not editor.is_alive() and [e["term"] for e in on_disk(d)["entries"]] == ["Kubernetes", "PyTorch"]


def test_the_history_page_does_not_wait_for_the_write_behind_thread(tmp_path, monkeypatch):
    h = History(tmp_path)
    gate = Gate(monkeypatch, h.path)
    h.add("first dictation")                           # written by the timer thread
    assert gate.entered.wait(3)
    got: list = []
    reader = in_thread(lambda: got.append([e["text"] for e in h.list()]))
    reader.join(2)
    assert got == [["first dictation"]]
    gate.release.set()
    wait(lambda: h._timer is None and not h._dirty)
    assert [e["text"] for e in on_disk(h)["items"]] == ["first dictation"]


def test_a_second_edit_during_a_slow_write_is_written_after_it_and_is_on_disk_when_it_returns(tmp_path, monkeypatch):
    d = Dictionary(tmp_path)
    d.COALESCE_S = 0
    gate = Gate(monkeypatch, d.path)
    first = in_thread(d.add, "First")
    assert gate.entered.wait(3)
    done = threading.Event()
    second = in_thread(lambda: (d.add("Second"), done.set()))
    wait(lambda: d.terms() == ["First", "Second"])     # the change itself is made at once
    assert not done.wait(0.2), "but the call returns only once its own change is on disk"
    gate.release.set()
    first.join(3)
    second.join(3)
    assert done.is_set() and not d._dirty and d.last_save_error == ""
    assert [[e["term"] for e in json.loads(t)["entries"]] for t in gate.texts] == [["First"], ["First", "Second"]], \
        "two writes, in the order the data was read"
    assert [e["term"] for e in on_disk(d)["entries"]] == ["First", "Second"]


def test_a_write_asked_for_inside_the_lock_runs_when_the_lock_is_let_go(tmp_path):
    h = History(tmp_path)
    with h._lock:
        with h._lock:
            h.add("inside")
            h.flush()
        assert not h.path.exists(), "still inside the outer block"
    assert [e["text"] for e in on_disk(h)["items"]] == ["inside"]


def test_a_write_that_finds_nothing_left_to_write_or_a_batch_in_progress_does_nothing(tmp_path, monkeypatch):
    d = Dictionary(tmp_path)
    writes: list = []
    real = S.cfgmod._atomic_write
    monkeypatch.setattr(S.cfgmod, "_atomic_write", lambda path, text: (writes.append(1), real(path, text))[1])
    d._write_out()                                     # another thread's write took the change along
    assert writes == []
    with d.batch():
        d.add("Kubernetes")
        d._write_out()                                 # a timer that chose to write just before the batch began
        assert writes == [] and d._dirty
        d._write_out(force=True)                       # flush() inside a batch
        assert writes == [1] and not d._dirty
    assert writes == [1]


def test_save_now_reports_only_its_own_write(tmp_path, monkeypatch):
    d = Dictionary(tmp_path)
    assert d.save_now() == ""
    real = S.cfgmod._atomic_write

    def refuse(path, text):
        raise PermissionError("the folder is read-only")

    monkeypatch.setattr(S.cfgmod, "_atomic_write", refuse)
    topics: list[str] = []
    d.subscribe(topics.append)
    with d.batch():
        d.add("Kubernetes")
        assert d.save_now() == "the folder is read-only"
    assert d.last_save_error and d.save_now() == "", "nothing new to write: the old failure is not this call's"
    assert topics.count("save_failed:dictionary") == 1
    monkeypatch.setattr(S.cfgmod, "_atomic_write", real)
    assert d.retry_save() is True and topics[-1] == "save_ok:dictionary" and d.last_save_error == ""
    assert [e["term"] for e in on_disk(d)["entries"]] == ["Kubernetes"]


def test_the_data_handed_to_the_writer_is_a_copy(tmp_path):
    d, sn, st, s = Dictionary(tmp_path), Snippets(tmp_path), Styles(tmp_path), Stats(tmp_path, clock=lambda: NOW)
    d.add("Kubernetes", ["cube"])
    d.suggest("PyTorch", ["pie torch"])
    sn.add("sig", "Best")
    st.set_app_category("editor.exe", "work")
    s.record(5, 2.0, "editor.exe")
    dumps = [x._dump() for x in (d, sn, st, s)]
    before = json.dumps(dumps)
    d.update(d.list()[0]["id"], aliases=["cube", "kube"])
    d.suggest("PyTorch", ["pi torch"])
    sn.update(sn.list()[0]["id"], text="Regards")
    st.set_tone("work", "formal")
    st.set_app_category("editor.exe", "email")
    s.record(7, 3.0, "editor.exe")
    assert json.dumps(dumps) == before, "a later change never reaches a write that is already being prepared"


# =========================================================================== DL-07: statistics
def test_counting_a_dictation_never_writes_the_statistics_on_the_callers_thread(tmp_path, monkeypatch):
    s = Stats(tmp_path, clock=lambda: NOW)
    writers: list[str] = []
    real = S.cfgmod._atomic_write
    monkeypatch.setattr(S.cfgmod, "_atomic_write", lambda path, text: (
        path == s.path and writers.append(threading.current_thread().name), real(path, text))[1])
    s.record(12, 5.0, "notepad.exe")
    wait(lambda: writers)
    assert writers == ["voxflow-store-write"]
    wait(lambda: s._timer is None and not s._dirty)
    text = s.path.read_text(encoding="utf-8")
    assert "\n" not in text, "compact: the file is rewritten after every dictation"
    assert json.loads(text) == {"version": 1, "days": {"2026-11-19": {
        "words": 12, "sessions": 1, "seconds": 5.0, "timed_words": 12,
        "apps": {"notepad.exe": {"words": 12, "sessions": 1}}}}}


def years_of_days(n_days: int, seed: int = 3, gaps: bool = False) -> dict:
    r = random.Random(seed)
    today = dt.date.fromtimestamp(NOW)
    days = {}
    for back in range(n_days):
        if gaps and r.random() < 0.2:
            continue
        apps = {f"app{r.randrange(9)}.exe": {"words": r.randrange(50, 400), "sessions": r.randrange(1, 9)}
                for _ in range(4)}
        words = sum(a["words"] for a in apps.values())
        days[(today - dt.timedelta(days=back)).isoformat()] = {
            "words": words, "sessions": sum(a["sessions"] for a in apps.values()),
            "seconds": round(r.uniform(30, 2000), 2), "timed_words": words - r.randrange(40), "apps": apps}
    return days


def stats_file(folder, days: dict, **more):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stats.json").write_text(json.dumps({"version": 1, "days": days, **more}), encoding="utf-8")
    return folder


@pytest.mark.parametrize("gaps", [False, True])
def test_old_days_are_folded_into_months_without_changing_any_total_streak_or_recent_chart(tmp_path, monkeypatch, gaps):
    days = years_of_days(3 * 365, gaps=gaps)
    with monkeypatch.context() as m:
        m.setattr(S, "_STATS_KEEP_DAYS", 10 ** 9)                  # the statistics as they were: every day kept
        whole = Stats(stats_file(tmp_path / "whole", days), clock=lambda: NOW)
        assert len(whole._days) == len(days) and not whole._months
    s = Stats(stats_file(tmp_path / "folded", days), clock=lambda: NOW)
    assert s.summary() == whole.summary() and s.summary()["streak_days"] == (5 if gaps else 3 * 365)
    assert s.top_apps(50) == whole.top_apps(50)
    assert s.daily(30) == whole.daily(30) and s.daily(365) == whole.daily(365), "a year of days is still there"
    kept = sorted(s._days)
    assert kept[0] >= "2025-10-01" and len(kept) <= 431, "from the month that began 400 days before the newest day"
    assert set(s._months) == {f"{y}-{m:02d}" for y, m in [(2023, 11), (2023, 12)] + [(2024, m) for m in range(1, 13)]
                              + [(2025, m) for m in range(1, 10)]}
    stored = on_disk(s)                                            # the fold is written at once, like a migration
    assert sorted(stored["days"]) == kept and set(stored["months"]) == set(s._months)
    assert stored["months"]["2024-02"]["active"] == sorted(int(day[8:]) for day in days if day.startswith("2024-02"))
    assert len(s.path.read_bytes()) < len((tmp_path / "whole" / "stats.json").read_bytes()) // 2
    again = Stats(tmp_path / "folded", clock=lambda: NOW)
    assert again.summary() == whole.summary() and again.top_apps(50) == whole.top_apps(50) and not again.load_problem
    assert again.export_days() == s.export_days()


def test_a_new_day_folds_the_days_that_left_the_kept_time_and_an_old_day_joins_its_month(tmp_path):
    clock = {"t": NOW}
    s = Stats(stats_file(tmp_path, years_of_days(420)), clock=lambda: clock["t"])
    assert sorted(s._months) == ["2025-09"] and min(s._days) == "2025-10-01"
    total = s.summary()["total_words"]
    s.record(10, 4.0, "notepad.exe")                               # the same day: nothing to fold
    assert sorted(s._months) == ["2025-09"]
    clock["t"] += 45 * DAY                                         # 2027-01-03: the kept time now starts 2025-11-01
    s.record(10, 4.0, "notepad.exe")
    assert sorted(s._months) == ["2025-09", "2025-10"] and min(s._days) == "2025-11-01"
    assert s._months["2025-10"]["active"] == list(range(1, 32))
    before = dict(s._months["2025-10"], apps=None)
    s.record(3, 1.5, "old.exe", ts=dt.datetime(2025, 10, 7, 9).timestamp())    # a late entry for a folded day
    month = s._months["2025-10"]
    assert "2025-10-07" not in s._days and month["active"] == list(range(1, 32))
    assert (month["words"], month["sessions"], month["timed_words"]) == (
        before["words"] + 3, before["sessions"] + 1, before["timed_words"] + 3)
    assert month["seconds"] == round(before["seconds"] + 1.5, 2)
    assert month["apps"]["old.exe"] == {"words": 3, "sessions": 1}
    assert s.summary()["total_words"] == total + 23 and s.top_apps(50)[-1]["exe"] == "old.exe"


def test_the_kept_time_is_counted_from_the_newest_day_with_use_not_from_the_clock(tmp_path):
    days = years_of_days(30)
    s = Stats(stats_file(tmp_path, days), clock=lambda: NOW + 3 * 365 * DAY)     # one start with the clock years ahead
    assert sorted(s._days) == sorted(days) and not s._months, "nothing was used on that day: nothing is folded"
    assert Stats(tmp_path, clock=lambda: NOW).daily(30) == [
        {"date": day, "words": t["words"], "sessions": t["sessions"], "seconds": t["seconds"]}
        for day, t in sorted(days.items())]


def test_seeding_from_an_old_history_folds_once_and_keeps_every_total(tmp_path):
    s = Stats(tmp_path, clock=lambda: NOW)
    items = [{"words": 10, "duration_s": 4.0, "app_exe": "Notepad.exe", "ts": NOW - back * DAY, "mode": "dictate"}
             for back in (0, 1, 2, 500, 500, 501, 900)]
    s.seed_from_history(items)
    assert sorted(s._days) == ["2026-11-17", "2026-11-18", "2026-11-19"] and sorted(s._months) == ["2024-06", "2025-07"]
    assert s._months["2025-07"]["active"] == [6, 7] and s._months["2025-07"]["sessions"] == 3
    assert s.summary()["total_words"] == 70 and s.summary()["streak_days"] == 3
    assert s.top_apps() == [{"exe": "notepad.exe", "words": 70, "sessions": 7}]
    s.seed_from_history([{"words": 99, "ts": NOW}])
    assert s.summary()["total_words"] == 70, "only ever seeded once"
    s._days.clear()
    s.seed_from_history([{"words": 99, "ts": NOW}])
    assert s.summary()["total_words"] == 40, "also when all there is are months"
    s.reset()
    assert s.export_days() == {} and s.summary()["total_words"] == 0


def test_a_streak_runs_through_folded_months(tmp_path):
    days = {(dt.date(2026, 11, 19) - dt.timedelta(days=back)).isoformat(): {"words": 1, "sessions": 1}
            for back in range(1, 700) if back != 650}
    days["2025-02-07"] = {"words": 4, "sessions": 0}               # 650 days ago: totals, but no dictation
    s = Stats(stats_file(tmp_path, days), clock=lambda: NOW)
    assert s._months["2025-02"]["active"] == [n for n in range(1, 29) if n != 7] and s._months["2025-02"]["words"] == 31
    assert s._months and s.summary()["streak_days"] == 649, "today not used yet; the gap 650 days ago ends it"
    assert s.summary()["today_words"] == 0


def test_backups_carry_the_months_and_a_merge_never_counts_a_day_twice(tmp_path):
    src = Stats(stats_file(tmp_path / "src", years_of_days(800)), clock=lambda: NOW)
    exported = src.export_days()
    months = sorted(k for k in exported if len(k) == 7)
    assert months and all("active" in exported[m] for m in months) and "active" not in exported["2026-11-19"]
    exported["2026-11-19"]["apps"].clear()
    exported[months[0]]["active"].clear()
    assert src.export_days()["2026-11-19"]["apps"] and src.export_days()[months[0]]["active"], "a copy"
    exported = src.export_days()

    fresh = Stats(tmp_path / "fresh", clock=lambda: NOW)
    assert fresh.import_days(exported) == len(exported) and fresh.is_new is False
    assert fresh.summary() == src.summary() and fresh.top_apps(50) == src.top_apps(50)
    assert fresh.import_days(exported) == 0, "everything is here already"
    assert fresh.summary() == src.summary()

    replaced = Stats(stats_file(tmp_path / "replaced", years_of_days(50, seed=9)), clock=lambda: NOW)
    assert replaced.import_days(exported, replace=True) == len(exported)
    assert replaced.export_days() == exported and replaced.summary() == src.summary()

    # merge, entry by entry
    here = Stats(tmp_path / "here", clock=lambda: NOW)
    one = {"words": 5, "sessions": 1, "seconds": 2.0, "timed_words": 5, "apps": {}}
    here.import_days({"2026-11-01": one, "2026-10-03": one, "2024-03": {**one, "active": [5]}})
    taken = here.import_days({
        "2024-03": {**one, "words": 500, "active": [5, 6]},        # the month is here
        "2024-03-05": {**one, "words": 500},                       # a day the month here already counts
        "2024-03-30": {**one, "words": 7},                         # a day the month here does not know: added to it
        "2026-10": {**one, "words": 500, "active": [3]},           # a month, but one of its days is here by itself
        "2026-11-01": {**one, "words": 500},                       # the day is here
        "2026-11-02": {**one, "words": 9},                         # new
        "2023-01": {**one, "words": 11, "active": [1, 2]},         # new
    })
    assert taken == 3
    assert sorted(here._days) == ["2026-10-03", "2026-11-01", "2026-11-02"]
    assert sorted(here._months) == ["2023-01", "2024-03"]
    assert here._months["2024-03"]["words"] == 12 and here._months["2024-03"]["active"] == [5, 30]
    assert here.summary()["total_words"] == 5 + 5 + 9 + 12 + 11


def test_stored_statistics_are_validated_months_included(tmp_path):
    parsed = Stats.parse_days({
        "20240305": {"words": 3, "sessions": 1},                   # the same day written another way
        "2024-13": {"words": 1}, "2024-00": {"words": 1}, "2024-3": {"words": 1}, "march": {"words": 1},
        "2024-02": {"words": 10 ** 400, "sessions": 2, "seconds": 1.5, "active": [3, "x", 40, True, 3, 1, 0, 2.0],
                    "apps": {"a.exe": {"words": 10 ** 400, "sessions": 1}}},
        "2024-01": {"words": 4, "active": "all of them"}, 7: {}, "2024-05-05": "not totals"})
    assert parsed == {
        "2024-03-05": {"words": 3, "sessions": 1, "seconds": 0.0, "timed_words": 0, "apps": {}},
        "2024-02": {"words": 0, "sessions": 2, "seconds": 1.5, "timed_words": 0, "active": [1, 3],
                    "apps": {"a.exe": {"words": 0, "sessions": 1}}},
        "2024-01": {"words": 4, "sessions": 0, "seconds": 0.0, "timed_words": 0, "apps": {}, "active": []}}
    assert Stats.parse_days(["not", "a", "dict"]) == {}
    # a file from before months existed, an empty one, one with a date no calendar reaches back from, a broken one
    s = Stats(stats_file(tmp_path / "a", {"2026-11-18": {"words": 2, "sessions": 1}}), clock=lambda: NOW)
    assert s.summary()["total_words"] == 2 and "months" not in s._dump() and not s.load_problem
    assert Stats(stats_file(tmp_path / "b", {}), clock=lambda: NOW).is_new is False
    odd = Stats(stats_file(tmp_path / "c", {"0001-01-05": {"words": 2, "sessions": 1}}), clock=lambda: NOW)
    assert odd.summary()["total_words"] == 2 and not odd.load_problem
    bad = Stats(stats_file(tmp_path / "d", {}, months=["2024-01"]), clock=lambda: NOW)
    assert bad.load_problem and bad.summary()["total_words"] == 0


def test_the_seconds_total_does_not_depend_on_what_was_folded(tmp_path, monkeypatch):
    days = {(dt.date(2026, 11, 19) - dt.timedelta(days=back)).isoformat():
            {"words": 1, "sessions": 1, "seconds": 0.05 + (back % 7) * 0.1, "timed_words": 1} for back in range(900)}
    with monkeypatch.context() as m:
        m.setattr(S, "_STATS_KEEP_DAYS", 10 ** 9)
        whole = Stats(stats_file(tmp_path / "whole", days), clock=lambda: NOW).summary()
    folded = Stats(stats_file(tmp_path / "folded", days), clock=lambda: NOW).summary()
    assert folded == whole
    assert whole["seconds_spoken"] == round(sum(round(d["seconds"] * 100) for d in days.values()) / 100, 1)


# =========================================================================== DL-05: checkpoint / rollback
def test_a_rollback_puts_a_store_back_exactly_ids_suggestions_and_backup_copies_included(tmp_path):
    d, sn, st = Dictionary(tmp_path), Snippets(tmp_path), Styles(tmp_path)
    s = Stats(tmp_path, clock=lambda: NOW)
    (tmp_path / "history.json").write_text(json.dumps([{"text": "old", "ts": NOW - 60}]), encoding="utf-8")   # v1
    (tmp_path / "history.corrupt-20260101-000000.json").write_text("{ damaged", encoding="utf-8")
    h = History(tmp_path, clock=lambda: NOW)
    assert (tmp_path / "history.v1.json").exists()
    d.add("Kubernetes", ["cube"])
    d.add("PyTorch")
    d.suggest("Grafana", ["graphana"])
    d.dismiss_suggestion(d.suggest("Nope")["id"])
    sn.add("sig", "Best")
    st.set_app_category("editor.exe", "work")
    s.record(5, 2.0, "editor.exe")
    stores = (d, sn, st, s, h)
    for x in stores:
        x.flush()
    before = [json.dumps(x._dump(), sort_keys=True) for x in stores]
    saved = [x.checkpoint() for x in stores]
    seen: list[str] = []
    for x in stores:
        x.subscribe(seen.append)

    d.apply_changes([{"term": "Grafana"}, {"term": "kubernetes", "aliases": ["other"]}], removals=["PyTorch"])
    d.suggest("Vercel")
    sn.apply_changes([{"trigger": "sig", "text": "changed"}, {"trigger": "new", "text": "x"}])
    st.apply_changes(tones={"work": "formal"}, apps={"editor.exe": "email"})
    s.import_days({"2020-01-01": {"words": 9}}, replace=True)
    h.import_entries([{"text": "imported"}], replace=True)             # asks for every copy to be deleted
    assert sn.expand("sig") == "changed" and d.suggestions()[0]["term"] == "Vercel"
    del seen[:]
    for x, state in zip(stores, saved):
        x.rollback(state)
    for x in stores:
        x.flush()
    assert [json.dumps(x._dump(), sort_keys=True) for x in stores] == before
    assert [json.dumps(on_disk(x), sort_keys=True) for x in stores] == before
    assert sn.expand("sig") == "Best" and d.apply("a cube") == "a Kubernetes", "what was derived is rebuilt"
    assert (tmp_path / "history.v1.json").exists() and (tmp_path / "history.corrupt-20260101-000000.json").exists()
    assert seen == ["dictionary", "suggestions", "snippets", "styles", "stats", "history"]
    assert d.update(d.list()[1]["id"], aliases=["pie torch"])["term"] == "PyTorch", "the index follows too"
    assert s.is_new is True and h.snapshot()[0]["id"] == json.loads(before[4])["items"][0]["id"]


# =========================================================================== DL-09: one copy per damaged file
def test_a_damaged_file_that_is_never_rewritten_is_copied_once_however_often_voxflow_starts(tmp_path):
    (tmp_path / "styles.json").write_text("{ damaged", encoding="utf-8")
    first = Styles(tmp_path)
    [copy] = tmp_path.glob("styles.corrupt-*.json")
    assert copy.name in first.load_problem
    for _ in range(3):
        again = Styles(tmp_path)
        assert again.load_problem == first.load_problem, "the same copy is named"
    assert list(tmp_path.glob("styles.corrupt-*.json")) == [copy]
    (tmp_path / "styles.json").write_text("{ damaged differently", encoding="utf-8")
    other = Styles(tmp_path)
    copies = sorted(tmp_path.glob("styles.corrupt-*.json"))
    assert len(copies) == 2 and copy in copies and other.load_problem != first.load_problem
    assert {c.read_text(encoding="utf-8") for c in copies} == {"{ damaged", "{ damaged differently"}
    (tmp_path / "styles.corrupt-folder.json").mkdir()               # not a file: never read
    (tmp_path / "styles.json").write_text("{ a third", encoding="utf-8")
    Styles(tmp_path)
    assert len(list(tmp_path.glob("styles.corrupt-2*.json"))) == 3


# =========================================================================== DL-13: two entries with one id
def test_entries_that_share_an_id_get_their_own_at_load_and_are_no_longer_deleted_together(tmp_path):
    (tmp_path / "dictionary.json").write_text(json.dumps({
        "entries": [{"id": "same", "term": "Kubernetes"}, {"id": "same", "term": "PyTorch"}, {"id": 7, "term": "Seven"},
                    {"term": "NoId"}],
        "suggestions": [{"id": "same", "term": "Grafana"}, {"id": "same", "term": "Vercel"}]}), encoding="utf-8")
    (tmp_path / "snippets.json").write_text(json.dumps({"entries": [
        {"id": "same", "trigger": "sig", "text": "A"}, {"id": "same", "trigger": "addr", "text": "B"}]}),
        encoding="utf-8")
    (tmp_path / "history.json").write_text(json.dumps({"version": 2, "items": [
        {"id": "same", "text": "newer", "ts": NOW - 10}, {"id": "same", "text": "older", "ts": NOW - 20},
        {"id": "", "text": "no id", "ts": NOW - 30}, {"id": 5, "text": "odd id", "ts": NOW - 40}]}), encoding="utf-8")
    d, sn, h = Dictionary(tmp_path), Snippets(tmp_path), History(tmp_path, clock=lambda: NOW)
    assert not d.load_problem and not sn.load_problem and not h.load_problem
    ids = {e["term"]: e["id"] for e in d.list()}
    assert ids["Kubernetes"] == "same" and ids["Seven"] == "7" and len(set(ids.values())) == 4
    assert [s["id"] for s in d.suggestions()][0] == "same" and len({s["id"] for s in d.suggestions()}) == 2
    assert [e["id"] for e in on_disk(d)["entries"]] == [ids[t] for t in ("Kubernetes", "PyTorch", "Seven", "NoId")], \
        "written at once: the ids are the same at the next start"
    assert {e["term"]: e["id"] for e in Dictionary(tmp_path).list()} == ids
    assert d.remove("same") is True and d.terms() == ["NoId", "PyTorch", "Seven"]
    sig = {e["trigger"]: e["id"] for e in sn.list()}
    assert sig["sig"] == "same" != sig["addr"] and {e["id"] for e in on_disk(sn)["entries"]} == set(sig.values())
    assert sn.remove("same") is True and [e["trigger"] for e in sn.list()] == ["addr"]
    texts = {e["text"]: e["id"] for e in h.snapshot()}
    assert texts["newer"] == "same" and len(set(texts.values())) == 4 and all(texts.values())
    assert {e["text"]: e["id"] for e in on_disk(h)["items"]} == texts
    assert h.delete("same") is True and [e["text"] for e in h.snapshot()] == ["older", "no id", "odd id"]


def test_a_file_without_duplicate_ids_is_not_rewritten_at_load(tmp_path):
    text = json.dumps({"entries": [{"id": "a", "term": "Kubernetes", "aliases": []}]})
    (tmp_path / "dictionary.json").write_text(text, encoding="utf-8")
    Dictionary(tmp_path)
    assert (tmp_path / "dictionary.json").read_text(encoding="utf-8") == text


# =========================================================================== DL-10: hostile JSON
DEEP = "[" * 100_000 + "]" * 100_000


@pytest.mark.parametrize("cls", [Dictionary, Snippets, History, Stats, Styles])
@pytest.mark.parametrize("content", [DEEP, '{"entries": ' + DEEP + "}"])
def test_a_store_file_nested_too_deeply_is_a_damaged_file_not_a_crash(tmp_path, cls, content):
    (tmp_path / cls.filename).write_text(content, encoding="utf-8")
    st = cls(tmp_path)
    assert "could not be read" in st.load_problem
    [copy] = tmp_path.glob(f"{st.path.stem}.corrupt-*.json")
    assert copy.read_text(encoding="utf-8") == content


def test_a_number_too_large_for_a_float_costs_that_value_only(tmp_path):
    huge = 10 ** 400
    assert S._num(huge) == 0.0 and S._num(-huge) == 0.0 and S._num(3) == 3.0 and S._num(float("inf")) == 0.0
    (tmp_path / "dictionary.json").write_text(json.dumps({
        "entries": [{"id": "a", "term": "Kubernetes"}], "suggestions": [{"term": "Grafana", "ts": huge}]}),
        encoding="utf-8")
    d = Dictionary(tmp_path)
    assert d.terms() == ["Kubernetes"] and d.suggestions()[0]["ts"] == 0.0 and not d.load_problem
    (tmp_path / "history.json").write_text(json.dumps({"version": 2, "items": [
        {"text": "kept", "ts": huge, "duration_s": huge, "latency_ms": huge, "words": huge}]}), encoding="utf-8")
    h = History(tmp_path, clock=lambda: NOW)
    assert not h.load_problem and h.snapshot()[0]["ts"] == NOW and h.snapshot()[0]["words"] == 1
    assert h.import_entries([{"text": "imported", "ts": huge, "duration_s": huge}]) == 1
    (tmp_path / "stats.json").write_text(json.dumps({"days": {"2026-11-19": {"words": huge, "sessions": 2}}}),
                                         encoding="utf-8")
    s = Stats(tmp_path, clock=lambda: NOW)
    assert not s.load_problem and s.summary()["sessions"] == 2 and s.summary()["total_words"] == 0


@pytest.mark.parametrize("name", ["terms.json", "snips.json"])
def test_importing_a_file_nested_too_deeply_is_refused_with_a_sentence(tmp_path, name):
    path = tmp_path / name
    path.write_text(DEEP, encoding="utf-8")
    with pytest.raises(ValueError, match=f"{name} is not a valid JSON file"):
        (Dictionary(tmp_path) if name == "terms.json" else Snippets(tmp_path)).import_(str(path))


# =========================================================================== D-06 / D-01: the history
def old_clean_text(s) -> str:
    s = str(s)
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-8", "replace").decode("utf-8")
    return "".join(ch for ch in s if ch in "\t\n\r" or (ch >= " " and ch != "\x7f"))


def test_the_faster_clean_text_gives_the_same_text_for_everything():
    r = random.Random(11)
    alphabet = ([chr(c) for c in range(0, 0x30)] + list("abc xyz ÄöÜß“”€")
                + ["\x7f", "\x80", "\x9f", "\ud800", "\udfff", "\udc00", "", "\U0001f600", "�"])
    for ch in alphabet:
        assert S.clean_text(ch) == old_clean_text(ch), repr(ch)
    for _ in range(4000):
        text = "".join(r.choice(alphabet) for _ in range(r.randrange(0, 40)))
        assert S.clean_text(text) == old_clean_text(text), repr(text)
    assert S.clean_text(12) == "12" and S.clean_text("plain text\twith\nlines\r") == "plain text\twith\nlines\r"
    assert S.clean_text("a\x00b\x1fc\x7fd") == "abcd"


def test_the_history_file_is_the_same_text_as_before_with_each_entry_encoded_once(tmp_path, monkeypatch):
    h = History(tmp_path, clock=lambda: NOW)
    with h.batch():
        for i in range(40):
            h.add(f"dictation “{i}” with ünïcode \\ and \"quotes\"", raw=f"raw {i}", ms=80.5,
                  app_exe="Notepad.EXE", duration_s=3.21, ts=NOW - i, rules_only=i % 3 == 0)
    assert h.path.read_text(encoding="utf-8") == json.dumps({"version": 2, "items": h.snapshot()}, ensure_ascii=False)
    assert History(tmp_path, clock=lambda: NOW).snapshot() == h.snapshot()
    assert h._serialise({"version": 2, "items": []}) == '{"version": 2, "items": []}'
    encoded: list = []
    real = S._encode_entry
    monkeypatch.setattr(S, "_encode_entry", lambda entry: (encoded.append(entry["text"]), real(entry))[1])
    h.add("one more", ts=NOW + 1)
    h.flush()
    assert encoded == ["one more"], "the 40 entries that were written before are not encoded again"
    assert h.path.read_text(encoding="utf-8") == json.dumps({"version": 2, "items": h.snapshot()}, ensure_ascii=False)
    fresh = History(tmp_path, clock=lambda: NOW)                   # after a start: each entry once, at the first write
    del encoded[:]
    fresh.add("after a restart", ts=NOW + 2)
    fresh.flush()
    fresh.delete(fresh.list()[0]["id"])
    fresh.flush()
    assert len(encoded) == 42 and len(on_disk(fresh)["items"]) == 41


def test_an_entry_whose_time_is_corrected_is_written_with_the_new_time(tmp_path):
    clock = {"t": NOW}
    h = History(tmp_path, clock=lambda: clock["t"])
    for i in range(4):
        h.add(f"entry {i}", ts=NOW - 3 * DAY + i)
    ahead = h.add("clock ran ahead", ts=NOW + 3600)
    h.flush()
    assert ahead["ts"] == NOW + 3600 and on_disk(h)["items"][0]["ts"] == NOW + 3600
    clock["t"] = NOW - 2 * DAY                                     # two days back: that one entry is now "ahead"
    kept = h._items[1]
    assert h.prune() is False and h._items[1] is kept, "the others are the same objects"
    h.add("next", ts=NOW - 2 * DAY + 5)
    h.flush()
    stored = {e["text"]: e["ts"] for e in on_disk(h)["items"]}
    assert stored["clock ran ahead"] == NOW - 2 * DAY == h.search("clock ran ahead")[0]["ts"]
    assert stored["entry 0"] == NOW - 3 * DAY
    late = h.add("stamped by a clock that is ahead", ts=NOW + 5 * DAY)
    assert late["ts"] == NOW - 2 * DAY, "the caller gets the entry as it is stored"
    gone = History(tmp_path / "other", retention_days=1, clock=lambda: NOW).add("too old to keep", ts=NOW - 5 * DAY)
    assert gone["text"] == "too old to keep"


def test_make_entry_keeps_a_given_id_and_makes_one_otherwise():
    assert S.make_entry("x", entry_id="abc")["id"] == "abc"
    assert len(S.make_entry("x")["id"]) == 12 and len(S.make_entry("x", entry_id="")["id"]) == 12


# =========================================================================== DL-06: snippets
def old_matches(sn, text: str) -> bool:
    whole_rx, lead_rx, trail_rx, _texts = sn._compiled()
    return bool(text and whole_rx is not None
                and (whole_rx.fullmatch(text) or lead_rx.search(text) or trail_rx.search(text)))


def old_expand(sn, text: str, transform=None) -> str:
    if not text:
        return text
    whole_rx, lead_rx, trail_rx, texts = sn._compiled()
    if whole_rx is None:
        return transform(text) if transform else text
    whole = whole_rx.fullmatch(text)
    if whole:
        return sn._snippet_of(whole, texts)
    hits = sorted(list(lead_rx.finditer(text)) + list(trail_rx.finditer(text)), key=lambda m: m.start())
    out: list[str] = []
    last = 0
    for m in hits:
        if m.start() < last:
            continue
        out.append(sn._around(text[last:m.start()], transform))
        out.append(sn._snippet_of(m, texts))
        last = m.end()
        before = text[:m.start()].strip(" \t\"“'‘")[-1:]
        if not text[last:].strip(" \t.!?…") and before in ("", ".", "!", "?", "…", "\n"):
            last = len(text)
    if not out:
        return transform(text) if transform else text
    out.append(sn._around(text[last:], transform))
    return "".join(out)


def test_looking_for_a_trailing_verb_first_changes_no_result(tmp_path):
    sn = Snippets(tmp_path)
    triggers = ["my address", "meine Adresse", "sign off", "zoom link two", "mijn adres", "e mail", "c++ header"]
    with sn.batch():
        for i, trigger in enumerate(triggers):
            sn.add(trigger, f"<{i}>")
    r = random.Random(5)
    words = (triggers + list(S._TRAILING_VERBS) + [v.upper() for v in S._TRAILING_VERBS]
             + ["Einfügen", "einfugen", "plak", "insert", "Bitte", "füge", "ein", "paste", "and", "then", "the", "to",
                "budget", "Thanks.", ",", ".", "!", "um", "please", "invoegen.", "einsetzen,"])
    seen = {True: 0, False: 0}
    changed = 0
    for _ in range(3000):
        text = " ".join(r.choice(words) for _ in range(r.randrange(1, 9)))
        assert sn.matches(text) == old_matches(sn, text), text
        assert sn.expand(text) == old_expand(sn, text), text
        assert sn.expand(text, str.upper) == old_expand(sn, text, str.upper), text
        seen[sn.matches(text)] += 1
        changed += sn.expand(text) != text
    assert min(seen.values()) > 300 and changed > 300, "the comparison saw both outcomes often"
    assert sn.expand("meine Adresse einfügen") == "<1>"
    assert sn.expand("Dann mijn adres INVOEGEN bitte") == "Dann <4> bitte"


def test_the_trailing_verb_regex_only_runs_on_a_text_that_has_one_of_its_verbs(tmp_path):
    sn = Snippets(tmp_path)
    sn.add("my address", "1 Main Street")
    whole_rx, lead_rx, trail_rx, texts = sn._compiled()

    class Spy:
        calls = 0

        def search(self, text):
            Spy.calls += 1
            return trail_rx.search(text)

        def finditer(self, text):
            Spy.calls += 1
            return trail_rx.finditer(text)

    sn._regex = (whole_rx, lead_rx, Spy(), texts)
    text = "Send the parcel to my address by Friday, and tell me when it is there."
    assert sn.matches(text) is False and sn.expand(text) == text and Spy.calls == 0
    assert sn.matches("my address einfügen") is True
    assert sn.expand("my address einfügen bitte") == "1 Main Street bitte"
    assert Spy.calls == 2


# =========================================================================== P-03: one matcher per dictionary key
def old_fuzzy_score(cand: str, n_words: int, index: dict):
    from voxflow.cleanup import phonetic_key

    sound = phonetic_key(cand)
    best = None
    for sq, ph, term in index["by_sound"].get(S._first_sound(sound), ()):
        if abs(len(cand) - len(sq)) > max(2, len(sq) // 4):
            continue
        if cand == sq:
            score = 1.0
        else:
            if n_words == 1 and (cand.startswith(sq) or sq.startswith(cand)):
                continue
            m = difflib.SequenceMatcher(None, cand, sq)
            if m.real_quick_ratio() < 0.75 or m.quick_ratio() < 0.75:
                continue
            spelt, heard = m.ratio(), difflib.SequenceMatcher(None, sound, ph).ratio()
            if n_words == 1:
                ok = heard >= 0.84 and spelt >= 0.78 and spelt + heard >= 1.72
            else:
                ok = spelt >= 0.8 and heard >= 0.8
            if not ok:
                continue
            score = (spelt + heard) / 2 - 0.02 * (n_words - 1)
        if best is None or score > best[0]:
            best = (score, term)
    return best


SYLLABLES = "ka zen flo vox tri bex lum qua dro nix pal mer sto vin gra hub net ops dat kor".split()


def made_up_terms(n: int) -> list[str]:
    r = random.Random(7)
    names = {"".join(r.choice(SYLLABLES) for _ in range(r.randint(2, 4))).capitalize() for _ in range(n * 2)}
    return sorted(names)[:n] + ["Kubernetes", "PostgreSQL", "Grafana", "Supercalifragilistic", "Aaaaaa", "Aaaaab"]


def misheard(r: random.Random, word: str) -> str:
    letters = list(word.lower())
    for _ in range(r.randrange(0, 3)):
        i = r.randrange(len(letters))
        kind = r.randrange(3)
        if kind == 0:
            letters[i] = r.choice("aeiouknstz")
        elif kind == 1:
            letters.insert(i, r.choice("aeiouknstz"))
        elif len(letters) > 3:
            del letters[i]
    return "".join(letters)


def test_one_matcher_per_key_scores_every_span_exactly_as_one_matcher_per_pair_did():
    terms = made_up_terms(150)
    index = S._fuzzy_index([{"term": t, "aliases": []} for t in terms])
    r = random.Random(13)
    hits = 0
    for n in range(6000):
        cand = misheard(r, r.choice(terms)) if n % 4 else "".join(r.choice(SYLLABLES) for _ in range(3))
        for n_words in (1, 2):
            new, old = S._fuzzy_score(cand, n_words, index), old_fuzzy_score(cand, n_words, index)
            assert new == old, (cand, n_words)
            hits += new is not None
    assert hits > 1500, "the comparison saw plenty of matches, not only misses"


def test_matchers_are_built_once_per_key_and_never_shared_between_threads(tmp_path, monkeypatch):
    d = Dictionary(tmp_path)
    with d.batch():
        for term in made_up_terms(60):
            d.add(term)
    texts = ["We moved the Kubernetis cluster and the Postgress upgrade to Tuesday.",
             "Graphana shows the Kazenflo and Voxtribex numbers.", "Nothing to correct in this sentence at all."]
    expected = [d.apply(t) for t in texts]
    assert expected[0] == "We moved the Kubernetes cluster and the PostgreSQL upgrade to Tuesday."
    built: list = []
    real = difflib.SequenceMatcher

    class Counting(real):
        def __init__(self, *a, **k):
            built.append(threading.get_ident())
            super().__init__(*a, **k)

    monkeypatch.setattr(S.difflib, "SequenceMatcher", Counting)
    assert [d.apply(t) for t in texts] == expected and built == [], "this thread has its matchers already"
    index = d._fuzzy
    mine = S._matcher(index, "kubernetes")
    assert S._matcher(index, "kubernetes") is mine
    theirs: list = []
    results: list = []

    def work():
        theirs.append(S._matcher(index, "kubernetes"))
        for _ in range(40):
            results.append([d.apply(t) for t in texts])

    threads = [in_thread(work) for _ in range(4)]
    mixed = [[d.apply(t) for t in texts] for _ in range(40)]
    for th in threads:
        th.join(20)
    assert all(got == expected for got in results + mixed) and len(results) == 160
    assert len({id(m) for m in theirs + [mine]}) == 5, "a matcher is state: every thread has its own"
    assert set(built) == {th.ident for th in threads}
    d.add("Zanzibarqa")                                # a change: a new index, and with it new matchers
    del built[:]
    assert d.apply(texts[0]) == expected[0] and built and d._fuzzy is not index
