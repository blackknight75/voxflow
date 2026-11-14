"""Settings > On-device models: the device menus and the speed check row, on the in-memory FakeController.

Also the two pieces under them: ``ChoiceButton`` options with a note / disabled state, and the FakeController's
device API (the twin of ``Controller.device_info`` / ``measure_devices``).
"""
from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QApplication

from tests.cov_ui_kit import Signals, demo, flush, make_demo, show  # noqa: F401 - fixtures
from voxflow.ui import pages as P
from voxflow.ui.fakes import FakeController
from voxflow.ui.widgets import ChoiceButton


def view(auto=None, fastest=None, options=(), **kw):
    return dict(auto=auto, fastest=fastest, options=list(options), **kw)


def opt(id_, ms=None, available=True, reason="", failed=False):
    return {"id": id_, "ms": ms, "available": available, "reason": reason, "failed": failed, "measured": True}


# =========================================================================== the wording
def test_menu_entries_show_the_measured_time_and_why_a_device_is_unavailable():
    v = view("cuda", "cuda", [opt("cuda", 29.0), opt("webgpu", 65.0), opt("cpu", 257.4)])
    assert P._device_options(v) == [("auto", "Automatic (fastest here: NVIDIA CUDA, 29 ms)", "", True),
                                    ("cuda", "NVIDIA CUDA · 29 ms", "", True), ("webgpu", "WebGPU · 65 ms", "", True),
                                    ("cpu", "CPU · 257 ms", "", True)]
    laptop = view("cpu", "vulkan", [opt("cuda", available=False, reason="No NVIDIA driver"),
                                    opt("vulkan", 1200.0), opt("cpu", 2650.0), opt("metal", available=False),
                                    opt("x", failed=True)])
    assert P._device_options(laptop) == [
        ("auto", "Automatic (CPU, 2,650 ms)", "", True), ("cuda", "NVIDIA CUDA", "No NVIDIA driver", False),
        ("vulkan", "Vulkan · 1,200 ms", "", True), ("cpu", "CPU · 2,650 ms", "", True),
        ("metal", "metal", "Not available here", False), ("x", "x", "Failed in the speed check", True)]
    assert P._device_options(view("cpu", None, [opt("cpu")]))[0][1] == "Automatic (CPU)", "not measured yet"
    assert P._device_options({})[0] == ("auto", "Automatic", "", True)


def test_the_row_says_where_the_model_runs_now_and_when_a_change_applies():
    assert P._device_sub("asr", {"current": "cuda"}, "") == (
        "Now running on NVIDIA CUDA. Automatic uses the fastest device measured on this PC.", "TEXT_DIM")
    assert P._device_sub("asr", {"current": "demo"}, "")[0].startswith("Automatic uses")
    assert P._device_sub("cleanup", {"current": "cuda"}, "") == (
        "Now running on NVIDIA CUDA. A change takes effect when VoxFlow restarts.", "TEXT_DIM")
    assert P._device_sub("cleanup", {"current": "cuda", "restart": True, "next": "vulkan"}, "") == (
        "Now running on NVIDIA CUDA. Moves to Vulkan when VoxFlow restarts.", "AMBER")
    assert P._device_sub("cleanup", {"current": "", "restart": True, "next": "?"}, "")[0] == (
        "A change takes effect when VoxFlow restarts.")
    assert P._device_sub("asr", {"current": "cpu"}, "cpu") == ("Set by VOXFLOW_DEVICE=cpu. Now running on CPU.",
                                                               "TEXT_DIM")
    assert P._device_sub("asr", {}, "webgpu") == ("Set by VOXFLOW_DEVICE=webgpu.", "TEXT_DIM")


def test_the_speed_check_row_tells_what_it_is_doing():
    assert P._speed_sub({"available": False}) == "Needs the real models, which are not in use."
    assert P._speed_sub({"state": "running"}) == "Measuring…"
    assert P._speed_sub({"state": "running", "done": 1, "total": 6, "model": "asr", "backend": "webgpu"}) == (
        "Measuring speech recognition on WebGPU (2 of 6)…")
    assert P._speed_sub({"state": "running", "done": 6, "total": 6, "model": "cleanup", "backend": "metal"}) == (
        "Measuring the language model on metal (6 of 6)…")
    assert P._speed_sub({"state": "waiting"}).startswith("Starts when VoxFlow is idle")
    when = time.mktime((2026, 11, 4, 12, 0, 0, 0, 0, -1))
    assert P._speed_sub({"state": "done", "measured_at": when}) == (
        "Measured on this PC on 4 Nov 2026. Automatic picks the fastest device.")
    assert P._speed_sub({}).startswith("Times each device once")


# =========================================================================== the page
def page_of(d):
    page = d.page("settings")
    page.refresh()
    flush()
    return page


def test_the_menus_show_the_measured_devices_and_a_pick_moves_speech_at_once(demo):
    page = page_of(demo)
    pick, row = page.devices["asr"], page.device_rows["asr"]
    assert pick.label() == "Automatic (fastest here: NVIDIA CUDA, 29 ms)" and pick.isEnabled()
    assert row.sub.text().startswith("Now running on NVIDIA CUDA.")
    assert page.btn_measure.text() == "Measure again" and page.btn_measure.isEnabled()
    assert page.speed_row.sub.text().startswith("Measured on this PC on")
    menu = pick.open_menu()
    flush()
    assert [it["text"] for it in menu.items()] == ["Automatic (fastest here: NVIDIA CUDA, 29 ms)",
                                                   "NVIDIA CUDA · 29 ms", "WebGPU · 65 ms", "CPU · 257 ms"]
    menu._activate(2)                                            # WebGPU
    flush()
    assert demo.c.config.asr_device == "webgpu" and demo.c.model_status("asr")["backend"] == "webgpu"
    assert row.sub.text().startswith("Now running on WebGPU.") and pick.label() == "WebGPU · 65 ms"
    pick._pick("auto")
    flush()
    assert demo.c.model_status("asr")["backend"] == "cuda"


def test_a_language_model_pick_says_it_moves_at_the_next_start(demo):
    page = page_of(demo)
    page.devices["cleanup"]._pick("vulkan")
    flush()
    row = page.device_rows["cleanup"]
    assert row.sub.text() == "Now running on NVIDIA CUDA. Moves to Vulkan when VoxFlow restarts."
    assert row.sub._role == "AMBER"
    assert demo.c.model_status("cleanup")["backend"] == "cuda", "nothing reloads in the running app"


def test_a_rejected_pick_snaps_back(demo):
    page = page_of(demo)
    page.devices["asr"]._pick("metal")
    flush()
    assert demo.c.config.asr_device == "auto" and page.devices["asr"].value == "auto"
    assert any("not a valid choice" in t for t in demo.toasts())


def test_measure_again_shows_progress_and_then_the_new_date(demo):
    page = page_of(demo)
    page.btn_measure.click()
    flush()
    assert not page.btn_measure.isEnabled() and page.btn_measure.text() == "Measuring…"
    assert page.speed_bar.isVisible() and page.speed_bar.value == 0.0
    assert page.speed_row.sub.text() == "Measuring speech recognition on NVIDIA CUDA (1 of 6)…"
    demo.c.measure_devices()                                     # a second request while running: ignored
    for _ in range(6):
        demo.c._speed_step()
    flush()
    assert page.btn_measure.isEnabled() and page.btn_measure.text() == "Measure again"
    assert not page.speed_bar.isVisible()
    demo.c.demo_set_devices(state="running", done=0, total=0)    # starting: no step known yet
    flush()
    assert page.speed_bar.value is None and page.speed_row.sub.text() == "Measuring…"


def test_the_override_and_missing_models_disable_the_controls(demo):
    page = page_of(demo)
    demo.c.demo_set_devices(forced="cpu")
    flush()
    assert not page.devices["asr"].isEnabled() and not page.devices["cleanup"].isEnabled()
    assert page.device_rows["asr"].sub.text().startswith("Set by VOXFLOW_DEVICE=cpu.")
    demo.c.demo_set_devices(forced="", available=False, measured_at=None, state="idle")
    flush()
    assert page.devices["asr"].isEnabled() and not page.btn_measure.isEnabled()
    assert page.btn_measure.text() == "Measure now"
    assert page.speed_row.sub.text() == "Needs the real models, which are not in use."


def test_unavailable_devices_cannot_be_picked_from_the_menu(demo, show):
    page = page_of(demo)
    demo.c.demo_set_devices(results={"asr": {"cuda": {"available": False, "reason": "No NVIDIA driver"},
                                             "webgpu": {"ms": 118.0}, "cpu": {"ms": 402.0}},
                                     "cleanup": {"cuda": {"available": False, "reason": "No NVIDIA driver"},
                                                 "vulkan": {"failed": True}, "cpu": {"ms": 2650.0}}},
                            auto={"asr": "webgpu", "cleanup": "cpu"})
    flush()
    menu = page.devices["asr"].open_menu()
    flush()
    items = menu.items()
    assert (items[1]["text"], items[1]["sub"], items[1]["enabled"]) == ("NVIDIA CUDA", "No NVIDIA driver", False)
    menu._activate(1)
    assert demo.c.config.asr_device == "auto", "a disabled device cannot be chosen"
    assert not menu.grab().isNull()                              # paints the note and the dimmed entry
    menu.close()
    cleanup = [it["sub"] for it in page.devices["cleanup"].open_menu().items()]
    assert cleanup == ["", "No NVIDIA driver", "Failed in the speed check", ""]
    page.devices["cleanup"]._menu.close()
    assert page.devices["cleanup"].label() == "Automatic (fastest here: CPU, 2,650 ms)"


def test_a_settings_search_finds_the_device_rows(demo):
    page = page_of(demo)
    page.find.setText("device")
    flush()
    assert page.device_rows["asr"].isVisibleTo(page) and page.device_rows["cleanup"].isVisibleTo(page)
    page.find.setText("speed check")
    flush()
    assert page.speed_row.isVisibleTo(page) and not page.device_rows["asr"].isVisibleTo(page)


# =========================================================================== the pieces under the page
def test_choice_options_may_carry_a_note_and_be_disabled(show):
    b = show(ChoiceButton(name="Device"))
    b.set_options([("a", "Alpha"), ("b", "Beta", "Not here", False), ("c", "Gamma", "", True)], "b")
    assert b.label() == "Beta" and b.sizeHint().width() >= 150
    menu = b.open_menu()
    flush()
    assert [(it["sub"], it["enabled"]) for it in menu.items()] == [("", True), ("Not here", False), ("", True)]
    menu._activate(1)
    assert b.value == "b"
    menu.close()
    b.set_options([], None)
    assert b.label() == b.fallback


def test_the_preview_twin_answers_like_the_controller():
    c = FakeController()
    seen = Signals(devices=c.devices_changed)
    info = c.device_info()
    assert set(info) == {"forced", "check", "asr", "cleanup"}
    assert info["asr"]["options"][0] == {"id": "cuda", "measured": True, "available": True, "reason": "",
                                         "ms": 29.0, "failed": False}
    assert info["asr"]["fastest"] == "cuda" and info["cleanup"]["restart"] is False
    with pytest.raises(ValueError):
        c.set("cleanup_device", "webgpu")
    c.set("cleanup_device", "cpu")
    assert c.device_info()["cleanup"]["restart"] is True and seen.devices
    c.demo_set_devices(results={"asr": {}, "cleanup": {"cpu": {"ms": None}}})
    info = c.device_info()
    assert info["asr"]["fastest"] is None and info["asr"]["options"][0]["measured"] is False
    c.measure_devices()
    assert c._speed_timer.isActive()
    c.shutdown()
    assert not c._speed_timer.isActive()
    QApplication.processEvents()
