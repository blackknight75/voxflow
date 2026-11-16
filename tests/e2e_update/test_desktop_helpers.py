"""desktop.py helpers that need no desktop session of their own (run in the normal suite on Windows)."""
import sys

import pytest

from tests.e2e_update import desktop as D


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell API")
def test_the_notification_state_is_read_and_named():
    code, name = D.notification_state()
    assert 1 <= code <= 7 and name == D.QUNS[code]


def test_every_state_has_a_name_that_says_why_balloons_are_dropped():
    assert set(D.QUNS) == set(range(1, 8)) and D.QUNS[5] == "QUNS_ACCEPTS_NOTIFICATIONS"
    assert "locked" in D.QUNS[1] and "full-screen" in D.QUNS[3]
