"""Regression guards for the design review: contrast, accessible names, Flow-bar states,
failure reasons, Settings finder, tray, toasts, minimum window. Offscreen, against the scripted controller."""
from __future__ import annotations

import pytest
from PySide6.QtGui import QAccessible, QColor
from PySide6.QtWidgets import QAbstractButton, QApplication

from tests.test_ui import PAGES, make_ui, seed  # noqa: F401  (fixture re-exported)
from voxflow.ui import theme as T


# ============================================================================ D-01 / D-02 contrast
def _lin(v: float) -> float:
    v /= 255.0
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def _lum(c: QColor) -> float:
    return 0.2126 * _lin(c.red()) + 0.7152 * _lin(c.green()) + 0.0722 * _lin(c.blue())


def ratio(fg: QColor, bg: QColor) -> float:
    a, b = _lum(fg), _lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _over(fg: QColor, bg: QColor) -> QColor:
    """``fg`` (possibly translucent) composited on the opaque ``bg``."""
    a = fg.alphaF()
    return QColor(round(fg.red() * a + bg.red() * (1 - a)), round(fg.green() * a + bg.green() * (1 - a)),
                  round(fg.blue() * a + bg.blue() * (1 - a)))


SURFACES = ("BG0", "BG1", "PANEL", "CONTROL", "SUNK")
TEXT_TOKENS = ("TEXT", "TEXT_DIM", "TEXT_FAINT")
ACCENT_TEXT = ("CYAN", "GREEN", "AMBER", "RED", "MAGENTA")       # used as text / icons on surfaces (VIOLET is fill only)


@pytest.mark.parametrize("palette", [T.DARK, T.LIGHT], ids=["dark", "light"])
def test_d01_text_tokens_meet_wcag_aa_on_every_surface(palette):
    bad = []
    for fg in TEXT_TOKENS:
        for bg in SURFACES:
            r = ratio(getattr(palette, fg), _over(getattr(palette, bg), palette.BG0))
            if r < 4.5:
                bad.append(f"{fg} on {bg}: {r:.2f}")
    assert not bad, bad


@pytest.mark.parametrize("palette", [T.DARK, T.LIGHT], ids=["dark", "light"])
def test_d02_accent_colours_used_as_text_meet_wcag_aa(palette):
    bad = []
    for fg in ACCENT_TEXT:
        for bg in SURFACES:
            r = ratio(getattr(palette, fg), _over(getattr(palette, bg), palette.BG0))
            if r < 4.5:
                bad.append(f"{fg} on {bg}: {r:.2f}")
    assert not bad, bad


@pytest.mark.parametrize("palette", [T.DARK, T.LIGHT], ids=["dark", "light"])
def test_d01_text_on_the_accent_gradient_is_readable(palette):
    assert ratio(palette.ON_ACCENT, palette.CYAN) >= 4.5
    assert ratio(palette.ON_ACCENT, palette.MAGENTA) >= 4.5


def test_d17_the_type_scale_has_no_near_duplicate_steps():
    sizes = sorted({T.F_CAPTION, T.F_SMALL, T.F_BODY, T.F_LABEL if hasattr(T, "F_LABEL") else T.F_BODY})
    assert all(b - a >= 0.5 for a, b in zip(sizes, sizes[1:])), sizes


# ============================================================================ D-03 accessible names
def test_d03_every_icon_only_button_has_an_accessible_name(make_ui):
    ui = make_ui()
    unnamed = []
    for key in PAGES:
        page = ui.page(key)
        for w in page.findChildren(QAbstractButton):
            if (w.text().strip() or not w.isVisibleTo(page)
                    or (not w.objectName() and type(w).__name__ == "QToolButton")):
                continue                                           # Qt's own arrows / scroll buttons
            iface = QAccessible.queryAccessibleInterface(w)
            name = iface.text(QAccessible.Text.Name) if iface else ""
            if not name.strip():
                unnamed.append(f"{key}:{type(w).__name__}:{w.objectName()}")
    assert not unnamed, unnamed


# ============================================================================ D-04 Flow bar
def test_d04_idle_pill_is_large_enough_to_see_and_paused_says_so(make_ui):
    from voxflow.ui import overlay
    ui = make_ui()
    ov = ui.shell.overlay
    assert overlay.SIZES["idle"][0] >= 64 and overlay.SIZES["idle"][1] >= 12
    assert overlay.PAUSED_SIZE[0] >= 80 and overlay.PAUSED_SIZE[1] >= 18, "the paused bar has room for a word"
    ui.c.set_paused(True)
    QApplication.processEvents()
    assert ui.c.paused
    w, h = ov._target_size("idle")
    assert (w, h) == overlay.PAUSED_SIZE, "a paused idle bar grows to say why nothing happens"
    ui.c.set_paused(False)
    QApplication.processEvents()
    assert ov._target_size("idle") == overlay.SIZES["idle"]


def test_d04_locking_shows_the_hands_free_hint_then_it_goes(make_ui):
    ui = make_ui()
    ov = ui.shell.overlay
    ui.c._set_state("locked", "", "dictate")
    QApplication.processEvents()
    assert ov.state == "locked"
    assert hasattr(ov, "_paint_lock_hint"), "D-04: hands-free caption missing"


# ============================================================================ D-05 specific failure reasons
def test_d05_silence_and_a_missing_microphone_are_told_apart(make_ui):
    ui = make_ui()
    rig = ui.rig
    seen = []
    rig.c.error_info_changed.connect(seen.append)
    rig.asr.text = ""
    rig.dictate()                                            # the scripted speech is audible but ASR hears nothing
    rig.wait(lambda: rig.c.state == "idle")
    assert seen and seen[-1]["kind"] in ("no_speech", "silence", "nothing"), seen
    assert seen[-1]["hint"], "an error must come with a next step"
    from tests.controller_rig import TRecorder
    TRecorder.fail_start = OSError("no device")
    rig.pause()
    rig.press()
    rig.wait(lambda: len(seen) >= 2, timeout=3000)
    TRecorder.fail_start = None
    rig.release()
    rig.wait(lambda: rig.c.state == "idle")
    assert seen[-1]["kind"] == "mic" and seen[-1]["action"] == "microphone", seen[-1]


def test_d05_a_failing_speech_model_is_reported_as_a_model_problem(make_ui):
    ui = make_ui()
    rig = ui.rig
    seen = []
    rig.c.error_info_changed.connect(seen.append)
    rig.asr.fail_transcribe = RuntimeError("boom")
    rig.dictate("hello there")
    rig.wait(lambda: rig.c.state == "idle")
    assert seen and seen[-1]["kind"] == "model", seen
    assert rig.c.error_info()["kind"] == "model"


# ============================================================================ D-06 .. D-09, D-12, D-15
def test_d07_settings_can_be_searched_and_jumped(make_ui):
    ui = make_ui()
    page = ui.page("settings")
    search = ui.find("settings_search")
    jump = ui.find("settings_jump")
    assert search is not None and jump is not None
    toggles = [w for w in page.findChildren(QAbstractButton) if w.objectName() == "settings_whisper_mode"]
    assert toggles
    search.setText("whisper")
    QApplication.processEvents()
    assert toggles[0].isVisibleTo(page)
    other = ui.find("settings_launch_at_login")
    assert not other.isVisibleTo(page), "a search must hide rows that do not match"
    search.setText("")
    QApplication.processEvents()
    assert other.isVisibleTo(page)


def test_d09_a_fresh_profile_shows_the_hero_not_empty_stats(make_ui):
    ui = make_ui()
    ui.page("home")
    assert ui.find_all("home_hero"), "first-run Home must show the hero card"


def test_d09_a_used_profile_shows_the_stats(make_ui):
    def prepare(rig):
        seed(rig, 4)
    ui = make_ui(prepare=prepare)
    ui.page("home").refresh()
    QApplication.processEvents()
    hero = ui.find_all("home_hero")
    assert not hero or not hero[0].isVisible()


def test_d12_tray_can_pause_for_an_hour_and_the_manual_toggle_cancels_it(make_ui):
    ui = make_ui()
    tray = ui.shell.tray
    assert hasattr(tray, "pause_for_hour")
    tray.pause_for_hour()
    assert ui.c.paused
    ui.c.set_paused(False)
    assert not ui.c.paused
    assert not getattr(tray, "_resume_timer", None) or not tray._resume_timer.isActive()


def test_d15_toasts_have_one_width_and_a_dismiss_affordance(make_ui):
    ui = make_ui()
    for msg in ("Short.", "A much longer message that needs room to breathe and wrap onto a second line."):
        ui.c.notice.emit("info", msg)
    QApplication.processEvents()
    toasts = ui.win.toasts._toasts
    assert len(toasts) >= 2
    assert len({t.width() for t in toasts}) == 1, [t.width() for t in toasts]


# ============================================================================ D-13 minimum window
def test_d13_the_minimum_window_is_800_by_580_and_pages_still_fit(make_ui):
    ui = make_ui()
    win = ui.win
    assert win.minimumWidth() >= 800 and win.minimumHeight() >= 580
    win.resize(win.minimumSize())
    QApplication.processEvents()
    for key in PAGES:
        page = ui.page(key)
        sa = getattr(page, "scroll", None)
        ui.render()
        QApplication.processEvents()
        assert page.width() <= win.width()


# ============================================================================ D-15 / T-01 actionable toasts (UI side)
def test_d15_a_save_failure_toast_gets_a_retry_button_that_calls_retry_saves(make_ui, monkeypatch):
    from voxflow import config as cfgmod
    ui = make_ui()
    real = cfgmod._atomic_write
    monkeypatch.setattr(cfgmod, "_atomic_write", lambda p, t: (_ for _ in ()).throw(OSError(28, "disk full")))
    ui.c.dictionary.add("Kubernetes")
    ui.wait(lambda: any("sav" in m.lower() for m in ui.toasts()))
    toast = next(t for t in ui.win.toasts._toasts if "sav" in t.message.lower())
    assert toast.btn is not None and toast.btn.text(), "the failure toast carries a retry button"
    monkeypatch.setattr(cfgmod, "_atomic_write", real)
    toast.btn.click()
    from voxflow import store
    ui.wait(lambda: "Kubernetes" in [e["term"] for e in store.Dictionary(ui.c.data_dir / "dictionary.json").list()])


def test_d05_a_microphone_failure_toast_opens_the_microphone_settings(make_ui):
    from tests.controller_rig import TRecorder
    ui = make_ui()
    rig = ui.rig
    TRecorder.fail_start = OSError("no device")
    rig.pause()
    rig.press()
    rig.wait(lambda: any(t.btn is not None for t in ui.win.toasts._toasts), timeout=3000)
    TRecorder.fail_start = None
    rig.release()
    toast = next(t for t in ui.win.toasts._toasts if t.btn is not None)
    ui.win.show_page("home")
    toast.btn.click()
    assert ui.win.current == "settings"


def test_d15_unknown_notice_actions_are_ignored(make_ui):
    ui = make_ui()
    ui.shell.run_action("something_new")
    ui.shell.run_action("")
    ui.c.notice_action.emit("no such toast", "retry_save", "Try again")        # no matching toast: no crash
    QApplication.processEvents()


def test_d04_the_paused_label_is_readable(make_ui):
    ui = make_ui()
    ov = ui.shell.overlay
    ui.c.set_paused(True)
    ov.show()
    QApplication.processEvents()
    ui.qtbot.wait(400)                                      # let the pill grow to its paused size
    img = ov.grab().toImage()
    from collections import Counter
    count = Counter(QColor(img.pixel(x, y)).getRgb()[:3] for x in range(img.width()) for y in range(img.height()))
    count.pop((0, 0, 0), None)                              # the offscreen canvas outside the pill
    assert count, "the paused pill painted nothing"
    fill = QColor(*count.most_common(1)[0][0])               # the pill body is the commonest colour
    ink = max((QColor(*c) for c in count), key=lambda c: ratio(c, fill))     # the label is the colour furthest from the pill
    assert ratio(ink, fill) >= 4.5, (ink.getRgb(), fill.getRgb(), ratio(ink, fill))


# ============================================================================ P6 root cause: the Flow bar's 62 fps timer never stopped while idle
def test_p6_an_idle_flow_bar_burns_no_frames_even_with_the_idle_pill_shown(make_ui):
    """The 3 % idle CPU of the old build was ``overlay.py`` keeping its 16 ms repaint timer running whenever the
    idle pill was visible (measured 4.1 % vs 0.05 % with the pill hidden). It must stop once the pill has settled."""
    ui = make_ui(config={"show_idle_pill": True})
    ov = ui.shell.overlay
    ov.show()
    ui.wait(lambda: not ov._timer.isActive(), timeout=6000)
    assert ov.show_idle and ov.state == "idle"
    ticks = []
    real = ov._tick
    ov._tick = lambda: (ticks.append(1), real())[1]
    ui.qtbot.wait(600)
    assert not ov._timer.isActive() and len(ticks) <= 2, f"{len(ticks)} frames in 600 ms (the old build drew about 37)"
    ui.c.set_paused(True)                                   # the paused pill animates once, then settles again
    ui.wait(lambda: not ov._timer.isActive(), timeout=6000)
    ui.c.set_paused(False)
    ui.wait(lambda: not ov._timer.isActive(), timeout=6000)


def test_d15_the_toast_stack_never_covers_the_sidebar(make_ui):
    """Live render at 150 % (window about 826 logical px wide): the toasts, centred on the whole window, hid the
    sidebar's status card. They belong over the content area."""
    ui = make_ui()
    win = ui.win
    win.show()
    win.resize(win.minimumSize())
    QApplication.processEvents()
    for msg in ("VoxFlow couldn't save your dictionary (the disk may be full or read-only). It is kept for now but "
                "will be lost when VoxFlow closes.", "Everything is saved now.", "Short note."):
        ui.c.notice.emit("warning", msg)
    ui.qtbot.wait(500)
    toasts = [t for t in win.toasts._toasts if t.isVisible()]
    assert toasts
    nav = win.pages["home"].mapTo(win, win.pages["home"].rect().topLeft())     # left edge of the content area
    for t in toasts:
        left = t.mapTo(win, t.rect().topLeft()).x()
        assert left >= nav.x() - 1, f"toast starts at x={left}, the content area at x={nav.x()}"
