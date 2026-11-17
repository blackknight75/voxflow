"""QA "data" fixes in the stores (``voxflow.store``): each test names the finding of the QA report it covers."""
from __future__ import annotations

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
