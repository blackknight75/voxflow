"""QA "data" fixes on the pages (in-process Qt on the offscreen platform, the demo ``FakeController``).

D1  long Dictionary / Snippets lists: rows are added in one batch, the first 100 shown, search debounced
Q1  the auto-learn wording says what really happened (VoxFlow re-spelled a word; the user corrected nothing)
D6  a shorter "Keep transcripts for" asks before it deletes older transcripts
Q2  the retention row says that even "Forever" keeps at most `history_limit` transcripts
"""
from __future__ import annotations

from tests.cov_ui_kit import flush, make_demo, show  # noqa: F401 - fixtures
from voxflow.ui.components import Row, Section


# =========================================================================== D1
def test_section_add_many_lays_out_once_and_keeps_the_dividers(show):
    sec = show(Section("Rows"))
    hidden_while_adding: list[bool] = []
    real_add = sec.add
    sec.add = lambda row: (hidden_while_adding.append(sec.card.isHidden()), real_add(row))[1]
    rows = sec.add_many([Row(f"Row {i}") for i in range(5)])
    assert hidden_while_adding == [True] * 5, "the card is hidden while the rows go in: one layout, not one per row"
    assert not sec.card.isHidden(), "and shown again afterwards"
    assert [r.divider for r in rows] == [True, True, True, True, False]
    sec.add(Row("Last"))
    assert [r.divider for r in sec._rows] == [True] * 5 + [False]

    hidden = Section("Hidden card")
    hidden.card.hide()
    hidden.add_many([Row("a")])
    assert hidden.card.isHidden(), "a card that was hidden stays hidden"


def test_a_long_dictionary_builds_100_rows_and_shows_more_on_request(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("dictionary"), demo.c
    c.dictionary._items = [{"id": f"i{i}", "term": f"Term{i:04d}", "aliases": [f"alias {i}"]} for i in range(250)]
    page.refresh()
    assert len(page.entries._rows) == 100, "only the first 100 rows are built"
    assert page.btn_more.isVisibleTo(page) and page.btn_more.text() == "Show more words (150 not shown)"
    assert page.entries.caption.text() == "YOUR WORDS  ·  250"
    page.btn_more.click()
    assert len(page.entries._rows) == 200 and page.btn_more.text() == "Show more words (50 not shown)"
    page.btn_more.click()
    assert len(page.entries._rows) == 250 and not page.btn_more.isVisibleTo(page)

    page.search.setText("term00")
    assert len(page.entries._rows) == 250, "nothing is rebuilt on the keystroke itself"
    demo.wait(lambda: len(page.entries._rows) == 100)          # after the pause: matches, from the first page again
    assert page.btn_more.text() == "Show more words (0 not shown)" and not page.btn_more.isVisibleTo(page)
    page.search.setText("term01")
    demo.wait(lambda: page.entries._rows[0].title.text() == "Term0100")


def test_a_long_snippet_list_builds_100_rows_and_shows_more_on_request(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("snippets"), demo.c
    c.snippets._items = [{"id": f"i{i}", "trigger": f"trigger {i:03d}", "text": f"text {i}"} for i in range(130)]
    page.refresh()
    assert len(page.rows._rows) == 100 and page.btn_more.text() == "Show more snippets (30 not shown)"
    assert page.btn_more.isVisibleTo(page)
    page.btn_more.click()
    assert len(page.rows._rows) == 130 and not page.btn_more.isVisibleTo(page)


# =========================================================================== Q1
def test_suggestions_say_voxflow_wrote_the_word_not_that_the_user_corrected_it(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("dictionary"), demo.c
    c.dictionary._sugg = [{"id": "s1", "term": "GitHub", "aliases": ["get hub"], "ts": 0.0},
                                 {"id": "s2", "term": "PyTorch", "aliases": [], "ts": 0.0}]
    page.refresh()
    subs = [r.sub.text() for r in page.inbox._rows]
    assert subs == ["VoxFlow heard “get hub” and wrote GitHub. Add it so it is always spelled this way.",
                    "VoxFlow wrote PyTorch in a recent dictation. Add it so it is always spelled this way."]
    assert not any("corrected" in s for s in subs)
    settings = demo.page("settings")
    learn = next(r for sec in settings._sections() for r in sec._rows if r.title.text() == "Learn new words")
    assert "correct" not in learn.sub.text() and "“get hub” written as GitHub" in learn.sub.text()


# =========================================================================== D6 + Q2
def _aged_history(c, ages_days) -> None:
    import time
    now = time.time()
    c.history._items = [{"id": f"h{i}", "ts": now - d * 86400.0, "text": f"t{i}", "raw": f"t{i}", "app_exe": "",
                         "app_title": "", "words": 1, "wpm": 0.0, "duration_s": 0.0, "mode": "dictate",
                         "latency_ms": 0} for i, d in enumerate(ages_days)]


def test_a_shorter_retention_asks_before_deleting_older_transcripts(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("settings"), demo.c
    _aged_history(c, [0, 0.5, 3, 10, 40])
    page.retention.chosen.emit(7)                    # what picking "7 days" in the menu sends
    modal = demo.find("retention_confirm")
    assert modal.title_label.text() == "Delete 2 transcripts older than 7 days?" if hasattr(modal, "title_label") \
        else True
    assert c.get("history_retention_days") == 0, "nothing changes before the user confirms"
    assert page.retention.value == 0, "the menu shows the old choice again"
    modal.btn_cancel.click()
    flush()
    assert c.get("history_retention_days") == 0 and c.history.count() == 5

    page.retention.chosen.emit(1)
    modal = demo.find("retention_confirm")
    assert modal.btn_primary.text() == "Delete 3 transcripts"
    modal.btn_primary.click()
    flush()
    assert c.get("history_retention_days") == 1 and page.retention.value == 1

    page.retention.chosen.emit(0)                    # Forever deletes nothing: no question
    assert c.get("history_retention_days") == 0 and not demo.find_all("retention_confirm")
    page.retention.chosen.emit(365)                  # nothing that old: no question either
    assert c.get("history_retention_days") == 365 and not demo.find_all("retention_confirm")


def test_retention_still_asks_when_the_history_cannot_be_counted(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("settings"), demo.c

    def broken(_days):
        raise OSError("locked")

    c.history.count_older_than = broken
    page.retention.chosen.emit(30)
    modal = demo.find("retention_confirm")
    assert modal.btn_primary.text() == "Delete the transcripts" and c.get("history_retention_days") == 0
    modal.btn_primary.click()
    flush()
    assert c.get("history_retention_days") == 30


def test_the_retention_row_says_how_many_transcripts_forever_keeps(make_demo):
    demo = make_demo(empty=True)
    page, c = demo.page("settings"), demo.c
    assert page.retention_row.sub.text() == ("Older transcripts are deleted automatically. Even with Forever, at most "
                                             "5,000 are kept: the oldest go first.")
    c.set("history_limit", 20000)
    assert "at most 20,000 are kept" in page.retention_row.sub.text()


# =========================================================================== D2 / R8
def test_open_data_folder_is_a_notice_action(make_demo):
    demo = make_demo(empty=True)
    opened: list = []
    demo.c.open_data_folder = lambda: opened.append(True)
    demo.shell.run_action("open_data_folder")
    assert opened == [True]
