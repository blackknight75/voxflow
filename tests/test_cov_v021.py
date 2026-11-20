"""Behaviour that v0.2.1's suite left untested (CI's 100 % coverage gate failed at 99.74 %).

One test per behaviour: the controller's dictionary warm-up and snippet fallback, the dictionary's fuzzy matching
and word list, snippet edge cases, the Windows probes (on fake ``ole32`` / ``user32`` / ``kernel32``), and the
updater's guards. Hermetic like ``tests/test_updater.py``, whose fakes it reuses.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import json
import types
from pathlib import Path

import pytest

from tests.controller_rig import make_rig
from tests.cov_ui_kit import demo, flush, make_demo  # noqa: F401 - fixtures
from tests.test_updater import (API, INSTALLER, REPO, Resp, _exe, _raiser, github, make, no_sockets,  # noqa: F401
                                release_json, settle)
from voxflow import store
from voxflow import system as sysmod
from voxflow import updater as U
from voxflow.fakes import FakeCleaner, FakeTranscriber

SETUP = "VoxFlow-Setup-0.3.0.exe"


def say(rig, spoken: str) -> str:
    before = len(rig.injector.pasted)
    rig.dictate(spoken)
    assert rig.c.state != "error", rig.c.message
    assert len(rig.injector.pasted) == before + 1, "nothing was pasted"
    return rig.injector.pasted[-1]


# =========================================================================== controller
def test_a_failing_dictionary_warm_up_neither_stops_the_model_nor_the_dictionary(tmp_path, qtbot, monkeypatch, caplog):
    """The word list is prepared off the dictation path while the speech model loads; when that fails, the model
    still comes up and the dictionary is built on first use instead."""
    calls = []
    monkeypatch.setattr(store.Dictionary, "prepare", lambda self: calls.append(1) or (_ for _ in ()).throw(
        RuntimeError("word list unreadable")))
    rig = make_rig(tmp_path, qtbot, monkeypatch, asr=FakeTranscriber(text="", delay_s=0.0),
                   cleaner=FakeCleaner(delay_s=0.0))
    try:
        assert calls and rig.c.model_status("asr")["state"] == "ready"
        assert "dictionary matchers could not be prepared" in caplog.text
        rig.c.dictionary.add("Kubernetes")
        assert say(rig, "I deploy on Kubernetis every day.") == "I deploy on Kubernetes every day."
    finally:
        rig.close()


def test_a_reworded_trigger_fires_every_time_and_even_when_the_rule_clean_up_fails(tmp_path, qtbot, monkeypatch):
    """The model-free clean-up is built once and reused; if it raises, the raw transcript is used, so saying the
    trigger still inserts the snippet."""
    r = make_rig(tmp_path, qtbot, monkeypatch)                          # TCleaner: scripted "model" output
    try:
        r.c.snippets.add("sig dot", "SIGNATURE")
        r.cleaner.rewrites["Sig dot."] = "Sig."                         # the model "fixed" the spoken word
        assert say(r, "Sig dot.") == "SIGNATURE"
        first = r.c._rule_cleaner
        assert say(r, "Sig dot.") == "SIGNATURE" and r.c._rule_cleaner is first, "built once, then reused"
        first.clean = _raiser(RuntimeError("rules broke"))
        assert say(r, "Sig dot.") == "SIGNATURE", "falls back to what the recogniser heard"
    finally:
        r.close()


def test_a_controller_built_without_updates_never_reaches_the_network(tmp_path, qtbot, monkeypatch, no_sockets):
    rig = make_rig(tmp_path, qtbot, monkeypatch, updates=False)
    try:
        info = rig.c.update_info()
        assert info["kind"] == "source" and not info["can_install"]
        assert not rig.c.updates._check_timer.isActive(), "no automatic check is armed"
        rig.c.check_updates()                                           # even Check now stays offline
        qtbot.waitUntil(lambda: rig.c.update_info()["phase"] == "idle", timeout=3000)
        assert "does not check for updates" in rig.c.update_info()["last_result"] + rig.c.update_info()["error"]
        assert no_sockets == []
        assert rig.c.updates.folder == rig.c.data_dir / "update-offline"
    finally:
        rig.close()


# =========================================================================== store: dictionary
def _dictionary(tmp_path, *terms):
    d = store.Dictionary(tmp_path)
    for t in terms:
        d.add(t)
    return d


def test_a_term_split_into_words_is_joined_back(tmp_path):
    """The squashed words are exactly the term ("kuber netes" -> "Kubernetes"): the best possible score."""
    assert _dictionary(tmp_path, "Kubernetes").apply("I deploy on kuber netes today.") == \
        "I deploy on Kubernetes today."


def test_terms_that_only_start_alike_are_not_confused(tmp_path):
    """"Kafka" sounds like it starts as "Kubernetis" does but is far shorter: it is not even scored."""
    d = _dictionary(tmp_path, "Kafka", "Kubernetes")
    assert d.apply("I deploy on Kubernetis every day.") == "I deploy on Kubernetes every day."
    assert d.apply("Kafka streams.") == "Kafka streams."


def test_the_closest_of_two_similar_terms_wins_whatever_their_order(tmp_path):
    for terms in (("Postgres", "Postgre"), ("Postgre", "Postgres")):
        assert _dictionary(tmp_path / "-".join(terms), *terms).apply("It runs on post gress.") == \
            "It runs on Postgres.", terms


def test_a_term_plus_or_minus_an_ending_is_left_as_said(tmp_path):
    """One word that is the term with an ending added or dropped ("Grafanas", "Kubernete") is another word, not a
    misspelling: it is not replaced, while a real misspelling of the same length is."""
    d = _dictionary(tmp_path, "Grafana", "Kubernetes")
    assert d.apply("Two grafanas run here.") == "Two grafanas run here."
    assert d.apply("One kubernete node.") == "One kubernete node."
    assert d.apply("Open the graphana dashboard.") == "Open the Grafana dashboard."


def test_contractions_and_possessives_count_as_english_words():
    words = frozenset({"isnt", "colour", "rock"})
    assert store._is_word("colour's", words) and store._is_word("Isn’t", words)
    assert store._is_word("rock'n'roll", words), "the part before the apostrophe"
    assert not store._is_word("kuber'netes", words) and not store._is_word("Grafana's", words)


def test_concurrent_first_uses_load_the_word_lists_once(monkeypatch):
    """Double-checked lock: a thread that waited for the lock finds the lists loaded and does not load them again."""
    loaded = (frozenset({"word"}), frozenset({"name"}))

    class Lock:                                        # the other thread finished loading while we waited
        def __enter__(self):
            store._WORDLISTS = loaded

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(store, "_WORDLISTS", None)
    monkeypatch.setattr(store, "_WORDLISTS_LOCK", Lock())
    monkeypatch.setattr(Path, "read_bytes", _raiser(AssertionError("loaded a second time")))
    assert store._wordlists() is loaded


# =========================================================================== store: snippets
def test_a_synced_trigger_without_words_never_fires_and_breaks_nothing(tmp_path):
    """Sync and profile import accept any non-empty trigger; one made of dashes or brackets only ("—", "()") has
    no words to say, so it is left out of the matcher while the other snippets work."""
    s = store.Snippets(tmp_path)
    assert s.apply_changes([{"trigger": "—", "text": "DASH"}]) == 1 and s.list()[0]["trigger"] == "—"
    assert s.expand("—") == "—" and not s.matches("—")
    s.apply_changes([{"trigger": "()", "text": "BRACKETS"}, {"trigger": "sign off", "text": "Best"}])
    assert s.expand("Sign off.") == "Best" and s.expand("— () —") == "— () —"


def test_a_trigger_asked_for_twice_over_is_inserted_once(tmp_path):
    """A lead verb before and a trailing verb after the same trigger ("Paste my address einfügen") overlap: the
    snippet goes in once, not twice."""
    s = store.Snippets(tmp_path)
    s.add("my address", "ADDR")
    _whole, lead, trail, _texts = s._compiled()
    text = "Paste my address einfügen, then go."
    assert lead.search(text) and trail.search(text), "both matchers see it"
    assert s.expand(text) == "ADDR einfügen, then go."


# =========================================================================== system
def test_the_setup_check_says_no_when_windows_cannot_be_asked(monkeypatch):
    monkeypatch.setattr(sysmod.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", _raiser(OSError("kernel32 unavailable")))
    assert sysmod.update_setup_running() is False
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: types.SimpleNamespace())     # no OpenMutexW
    assert sysmod.update_setup_running() is False


# =========================================================================== updater: Windows probes on fakes
RELEASE_FN = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
GETCOST_FN = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p)


class FakeOle32:
    """NetworkListManager on a fake COM object: a vtable whose slot 2 is Release and slot 3 GetCost."""

    def __init__(self, cost=0, hr_cost=0, hr_init=0, hr_create=0, null=False):
        self.cost, self.hr_cost, self.hr_init, self.hr_create, self.null = cost, hr_cost, hr_init, hr_create, null
        self.uninit = self.released = 0
        self._release = RELEASE_FN(self._on_release)
        self._get_cost = GETCOST_FN(self._on_cost)
        self._vtbl = (ctypes.c_void_p * 4)(None, None, ctypes.cast(self._release, ctypes.c_void_p).value,
                                           ctypes.cast(self._get_cost, ctypes.c_void_p).value)
        self._obj = ctypes.c_void_p(ctypes.addressof(self._vtbl))      # an object's first field: its vtable

    def _on_release(self, _this):
        self.released += 1
        return 0

    def _on_cost(self, _this, pcost, _dest):
        pcost[0] = self.cost
        return self.hr_cost

    def CoInitializeEx(self, *_a):  # noqa: N802 - the Win32 names
        return self.hr_init

    def CoUninitialize(self):  # noqa: N802
        self.uninit += 1

    def CLSIDFromString(self, *_a):  # noqa: N802
        return 0

    def CoCreateInstance(self, _clsid, _outer, _ctx, _iid, ppv):  # noqa: N802
        if not self.null:
            ppv._obj.value = ctypes.addressof(self._obj)
        return self.hr_create


def _windll(monkeypatch, **dlls):
    monkeypatch.setattr(U.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(**dlls), raising=False)


@pytest.mark.parametrize("cost, metered", [(0x1, False), (0x2, True), (0x4, True), (0x10000, True),
                                           (0x40000, True), (0x1 | 0x40000, True)])
def test_metered_is_read_from_the_connection_cost(monkeypatch, cost, metered):
    ole = FakeOle32(cost=cost)
    _windll(monkeypatch, ole32=ole)
    assert U.connection_metered() is metered
    assert ole.released == 1 and ole.uninit == 1, "the object is released and COM uninitialised"


def test_metered_is_unknown_when_windows_cannot_tell(monkeypatch):
    for ole in (FakeOle32(hr_create=-2147221164), FakeOle32(null=True), FakeOle32(hr_cost=-2147467259)):
        _windll(monkeypatch, ole32=ole)
        assert U.connection_metered() is None
        assert ole.uninit == 1
    ole = FakeOle32(hr_cost=-2147467259)
    _windll(monkeypatch, ole32=ole)
    U.connection_metered()
    assert ole.released == 1, "released even when GetCost fails"
    _windll(monkeypatch)                                                  # no ole32 at all
    assert U.connection_metered() is None


def test_com_is_only_uninitialised_when_this_call_initialised_it(monkeypatch):
    """CoInitializeEx answers RPC_E_CHANGED_MODE on a thread already in another apartment: no CoUninitialize."""
    ole = FakeOle32(cost=0x1, hr_init=-2147417850)
    _windll(monkeypatch, ole32=ole)
    assert U.connection_metered() is False and ole.uninit == 0
    ole = FakeOle32(cost=0x1, hr_init=1)                                  # S_FALSE: already initialised, still counted
    _windll(monkeypatch, ole32=ole)
    assert U.connection_metered() is False and ole.uninit == 1


class FakeUser32:
    def __init__(self, last=0, ok=True):
        self.last, self.ok = last, ok

    def GetLastInputInfo(self, pinfo):  # noqa: N802
        pinfo._obj.dwTime = self.last
        return self.ok


def test_idle_time_survives_the_49_day_tick_wrap(monkeypatch):
    _windll(monkeypatch, user32=FakeUser32(last=0xFFFFFFFF - 999),
            kernel32=types.SimpleNamespace(GetTickCount=lambda: 5000))
    assert U.idle_seconds() == 6.0
    _windll(monkeypatch, user32=FakeUser32(last=1000), kernel32=types.SimpleNamespace(GetTickCount=lambda: 4000))
    assert U.idle_seconds() == 3.0


def test_idle_time_is_unknown_when_windows_cannot_tell(monkeypatch):
    _windll(monkeypatch, user32=FakeUser32(ok=False), kernel32=types.SimpleNamespace(GetTickCount=lambda: 1))
    assert U.idle_seconds() is None
    _windll(monkeypatch)                                                  # no user32
    assert U.idle_seconds() is None


# =========================================================================== updater: guards
def test_other_assets_of_a_release_are_ignored():
    data = release_json()
    data["assets"].insert(0, {"name": "VoxFlow-0.3.0-portable.zip", "size": 5,
                              "browser_download_url": f"{REPO}/releases/download/v0.3.0/VoxFlow-0.3.0-portable.zip"})
    rel = U.parse_release(data)
    assert rel.installer == SETUP and rel.sums_url.endswith("/SHA256SUMS.txt")
    assert rel.size == len(INSTALLER)


def test_an_unreadable_program_files_root_does_not_hide_the_others(tmp_path, monkeypatch):
    """A Program Files variable that points to a drive Windows cannot resolve is skipped, not fatal."""
    pf86 = tmp_path / "Program Files (x86)"
    exe = _exe(pf86 / "VoxFlow")
    (exe.parent / "unins000.exe").write_bytes(b"")
    broken = str(tmp_path / "unreachable")
    monkeypatch.setenv("ProgramFiles", broken)
    monkeypatch.setenv("ProgramFiles(x86)", str(pf86))
    monkeypatch.delenv("ProgramW6432", raising=False)
    real = Path.resolve

    def resolve(self, *a, **k):
        if str(self) == broken:
            raise OSError("the network drive is gone")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert U.install_kind(exe, frozen=True) == "allusers"
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "elsewhere"))
    assert U.install_kind(exe, frozen=True) == "installer"


def test_a_setting_the_updater_does_not_own_changes_nothing(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto", when="time")
    slot_active = up._slot_timer.isActive()
    changed = []
    up.changed.connect(lambda *a: changed.append(1))
    up.settings_changed("language")
    assert up._slot_timer.isActive() == slot_active and up._http.requests == [] and up.phase == "idle"
    assert changed, "the page still refreshes"
    up.close()


def test_automatic_mode_with_no_known_release_downloads_nothing(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto")
    up.settings["update_install_when"] = "time"
    up.settings_changed("update_install_when")
    assert up._http.requests == [] and up.phase == "idle"
    up.close()


def test_nothing_is_downloaded_after_the_updater_has_closed(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check(user=True)
    settle(up)
    up.close()
    asked = len(up._http.requests)
    up.install("now")
    qtbot.wait(50)
    assert len(up._http.requests) == asked and up.phase == "idle" and not up.info()["downloaded"]


class ApiGate:
    """The API route held open until ``release()``: a check under way."""

    def __init__(self, payload):
        import threading
        self.event, self.payload = threading.Event(), payload

    def __call__(self, headers):
        self.event.wait(5)
        return Resp(200, json.dumps(self.payload).encode())

    def release(self):
        self.event.set()


def test_the_set_time_waits_for_a_check_under_way(tmp_path, qtbot):
    moment = _dt.datetime(2026, 11, 15, 3, 10).astimezone()
    up = make(tmp_path, qtbot, mode="auto", when="time", now=lambda: moment)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"] and up.phase == "idle", timeout=3000)
    up.state.slot_done = ""
    gate = ApiGate(release_json())
    up._http.routes[API] = gate
    up.check(user=False)                                   # the daily check, still the same newest version
    qtbot.waitUntil(lambda: up.phase == "checking", timeout=3000)
    up._slot_tick()
    assert up.events["restart"] == 0 and up.state.slot_done == "" and up._slot_timer.isActive()
    gate.release()
    settle(up)
    up._slot_tick()
    assert up.events["restart"] == 1, "then it installs in the same slot"
    up.close()


def test_an_installer_that_cannot_be_read_is_not_restarted_into(tmp_path, qtbot, monkeypatch):
    """Install now: a file a virus scanner holds cannot be hashed, so VoxFlow does not quit for it."""
    up = make(tmp_path, qtbot)
    up.check(user=True)
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    monkeypatch.setattr(U, "sha256_stream", _raiser(PermissionError("locked by a scanner")))
    up.install("now")
    assert up.events["restart"] == 0 and up.phase == "idle" and up.error
    up.close()


def test_a_read_error_while_hashing_on_quit_launches_nothing(tmp_path, qtbot, monkeypatch):
    up = make(tmp_path, qtbot)
    up.check(user=True)
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    monkeypatch.setattr(U, "sha256_stream", _raiser(OSError("the disk went away")))
    assert up.on_exit(user_quit=True) is False and up.launched == []
    up.close()


# =========================================================================== UI
def test_a_settings_search_keeps_the_install_timing_row_when_updates_change(demo):
    page = demo.page("settings")
    sec = page.updates
    page.find.setText("install")
    flush()
    assert sec.when_row.isVisible()
    sec.refresh(dict(demo.c.update_info(), mode="notify"))          # e.g. a check finished meanwhile
    assert sec.when_row.isVisible(), "the search, not the mode, decides what is shown while it is typed"
    page.find.setText("")
    flush()


def test_tray_quit_asks_the_controller_to_prepare_then_quits(demo, monkeypatch):
    from voxflow.ui import shell as shellmod
    quits, prepared = [], []
    monkeypatch.setattr(shellmod, "QApplication",
                        types.SimpleNamespace(instance=lambda: types.SimpleNamespace(quit=lambda: quits.append(1))))
    monkeypatch.setattr(demo.c, "prepare_quit", lambda: prepared.append(1))
    demo.shell.quit()
    assert prepared == [1] and quits == [1]
    monkeypatch.setattr(demo.c, "prepare_quit", None)                # a controller without update support
    demo.shell.quit()
    assert quits == [1, 1]
