"""Render every page, dialog, tray menu, onboarding step and Flow-bar state to build/shots/*.png
(a git-ignored output folder).

Runs offscreen (no window ever appears, nothing is injected) at 100 % and 150 % scale; each scale is rendered in
its own child process because the scale factor has to be set before Qt starts.

    python scripts/demo_ui.py                    # all shots, both scales, preview data
    python scripts/demo_ui.py --scales 100       # one scale
    python scripts/demo_ui.py --controller real  # use voxflow.controller.Controller(demo=True) instead of the twin
    python scripts/demo_ui.py --real-screen      # real desktop, real DPI, every monitor: shots + focus/placement checks

Shots are named <group>_<name>@<scale>.png. Translucent surfaces are composited over a desktop-like backdrop.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "build" / "shots"


def parent(args) -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    code = 0
    for scale in args.scales:
        for old in SHOTS.glob(f"*@{scale}.png"):
            old.unlink()
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=str(scale / 100),
                   QT_ENABLE_HIGHDPI_SCALING="1", VOXFLOW_SHOT_SCALE=str(scale))
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        if fonts.is_dir():
            env.setdefault("QT_QPA_FONTDIR", str(fonts))      # the offscreen plugin ships no fonts of its own
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--controller", args.controller]
        code |= subprocess.call(cmd, env=env, cwd=str(ROOT))
    print(f"{len(list(SHOTS.glob('*.png')))} shots in {SHOTS}")
    return code


# --------------------------------------------------------------------------- child: the actual rendering
def child(args) -> int:
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QPoint, QRect, Qt, qInstallMessageHandler
    from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter
    from PySide6.QtWidgets import QApplication, QWidget

    warnings: list[str] = []
    # inherent to Qt's offscreen platform when a Qt.Popup opens / fonts are loaded; not produced by our code
    benign = ("does not support raise", "does not support grabbing", "Cannot find font directory",
              "does not support propagateSizeHints")
    qInstallMessageHandler(lambda _t, _c, msg: None if any(b in msg for b in benign) else warnings.append(msg))

    app = QApplication([])
    scale = os.environ.get("VOXFLOW_SHOT_SCALE", "100")

    from voxflow.ui import theme as T
    import copy

    from voxflow.ui.fakes import _DEVICE_RESULTS as _FAKE_RESULTS, FakeController
    from voxflow.ui.shell import build_ui
    from voxflow.ui.tray import ICON_STATES, make_icon

    def make_controller(empty: bool = False):
        if args.controller == "real" and not empty:
            from voxflow.controller import Controller
            c = Controller(demo=True)
            c.start()
            return c
        return FakeController(empty=empty)

    def pump(seconds: float = 0.05) -> None:
        end = time.perf_counter() + seconds
        while True:
            app.processEvents()
            if time.perf_counter() >= end:
                return
            time.sleep(0.008)

    def backdrop(w: int, h: int, light: bool) -> QImage:
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        g = QLinearGradient(0, 0, w, h)
        if light:
            g.setColorAt(0, QColor(226, 231, 242))
            g.setColorAt(1, QColor(196, 202, 224))
        else:
            g.setColorAt(0, QColor(30, 38, 70))
            g.setColorAt(1, QColor(70, 46, 96))
        p.fillRect(0, 0, w, h, g)
        p.end()
        return img

    def save(widget, name: str, light_bg: bool = False, crop: QRect | None = None) -> None:
        pm = widget.grab()
        top = pm.toImage()
        top.setDevicePixelRatio(1.0)          # composite in physical pixels, exactly as rendered
        img = backdrop(top.width(), top.height(), light_bg)
        p = QPainter(img)
        p.drawImage(0, 0, top)
        p.end()
        if crop is not None:
            k = pm.devicePixelRatio()
            img = img.copy(int(crop.x() * k), int(crop.y() * k), int(crop.width() * k), int(crop.height() * k))
        img.save(str(SHOTS / f"{name}@{scale}.png"))

    ctl = make_controller()
    ctl.set("onboarding_done", True)
    ctl.set("theme", "dark")
    ui = build_ui(ctl, show_window=True)
    win, pill = ui.window, ui.overlay
    win.resize(1080 + 2 * win.M, 720 + 2 * win.M)       # the default size (the offscreen screen is smaller)
    pump(0.3)

    # ---- Flow bar states
    crop = QRect(40, 30, 440, 112)

    def bar(name: str, state: str, mode: str = "dictate", message: str = "", secs: float = 1.0,
            light_bg: bool = False) -> None:
        ctl.demo_set_state(state, mode, message)
        pump(secs)
        save(pill, f"overlay_{name}", light_bg, crop)

    bar("idle", "idle", secs=0.8)
    save(pill, "overlay_idle_on_light", True, crop)
    ctl.set_paused(True)
    pump(0.6)
    save(pill, "overlay_idle_paused", False, crop)
    ctl.set_paused(False)
    bar("listening", "listening", secs=1.5)
    save(pill, "overlay_listening_on_light", True, crop)
    bar("locked", "locked", secs=1.3)
    bar("command", "listening", "command", secs=1.4)
    bar("processing", "processing", secs=1.0)
    bar("command_processing", "processing", "command", secs=0.9)
    bar("done", "done", message="Pasted", secs=0.7)
    bar("done_rules_only", "done", message="Pasted · basic clean-up", secs=0.7)
    bar("done_rules_only_longest", "done", message="Copied. Press Ctrl+V · basic clean-up", secs=0.7)
    bar("error", "error", message="Didn't catch that", secs=0.9)
    bar("error_long", "error", message="Window changed. Result copied: press Ctrl+V", secs=0.9)
    bar("error_nospeech", "error", message="No speech heard. Check your microphone", secs=0.9)
    bar("error_nomic", "error", message="Microphone disconnected", secs=0.9)
    ctl.demo_set_state("idle")
    pill.set_state("loading", "Downloading speech model", "dictate", 0.43)
    pump(0.9)
    save(pill, "overlay_loading", False, crop)
    pill.set_state("idle")
    pill.set_arranging(True)
    pump(0.9)
    save(pill, "overlay_arrange", False, crop)
    pill.set_arranging(False)
    pump(0.2)

    # ---- main window pages (dark, then light)
    def page(name: str, key: str, scroll: int = 0) -> None:
        win.show_page(key)
        pump(0.15)
        win.pages[key].scroll.verticalScrollBar().setValue(scroll)
        pump(0.15)
        save(win, f"page_{name}")

    def settings_tour(suffix: str = "") -> None:
        sb = win.pages["settings"].scroll.verticalScrollBar()
        win.show_page("settings")
        pump(0.2)
        step = max(1, win.pages["settings"].scroll.viewport().height() - 60)
        pos, i = 0, 1
        while True:
            sb.setValue(pos)
            pump(0.12)
            save(win, f"page_settings_{i}{suffix}")
            if pos >= sb.maximum():
                break
            pos, i = min(sb.maximum(), pos + step), i + 1

    for theme in ("dark", "light"):
        ctl.set("theme", theme)
        pump(0.2)
        sfx = "" if theme == "dark" else "_light"
        page(f"home{sfx}", "home")
        page(f"history{sfx}", "history")
        page(f"dictionary{sfx}", "dictionary")
        page(f"snippets{sfx}", "snippets")
        page(f"style{sfx}", "style")
        ctl.set("language", "de")                # preview in the dictation language; German keeps its capitals
        ctl.styles.set_tone("personal", "very_casual")
        pump(0.2)
        page(f"style_german{sfx}", "style")
        ctl.styles.set_tone("personal", "casual")
        for code in ("es", "pl", "auto"):        # the clean-up hint under the language picker, per level
            ctl.set("language", code)
            win.show_page("settings")
            pump(0.15)
            win.pages["settings"].scroll.ensureWidgetVisible(win.pages["settings"].language_row, 0, 120)
            pump(0.15)
            save(win, f"page_settings_language_{code}{sfx}")
        page(f"style_apps{sfx}", "style", 100000)
        ctl.styles.set_app_category("obsidian.exe", "personal")       # an app the user added
        ctl.styles.set_app_category("slack.exe", "personal")          # a built-in app the user re-categorised
        pump(0.2)
        page(f"style_apps_own{sfx}", "style", 330)
        ctl.styles.remove_app_category("obsidian.exe")
        ctl.styles.remove_app_category("slack.exe")
        settings_tour(sfx)
        # dialogs, toasts and transient states
        win.show_page("dictionary")
        m = win.pages["dictionary"].edit(ctl.dictionary.list()[0])
        pump(0.35)
        save(win, f"modal_dictionary_edit{sfx}")
        m.show_error("“Kubernetes” is already in your dictionary.")
        pump(0.1)
        save(win, f"modal_dictionary_error{sfx}")
        m.close_modal()
        win.show_page("snippets")
        m = win.pages["snippets"].edit(ctl.snippets.list()[0])
        pump(0.35)
        save(win, f"modal_snippet_edit{sfx}")
        m.close_modal()
        win.show_page("history")
        win.pages["history"]._confirm_clear()
        pump(0.35)
        save(win, f"modal_confirm_clear{sfx}")
        win._modal.close_modal()
        win.pages["history"].view.set_value(True, emit=True)
        pump(0.3)
        save(win, f"page_history_raw{sfx}")
        win.pages["history"].view.set_value(False, emit=True)
        win.pages["history"].search.setText("zebra")
        pump(0.4)
        save(win, f"page_history_no_match{sfx}")
        win.pages["history"].search.setText("")
        pump(0.3)
        win.toast("success", "Exported 9 transcripts to voxflow-history.txt.")
        win.toast("error", "Could not save settings: the disk is full.")
        win.toast("info", "New dictionary suggestion: “GitHub”.", "Review", lambda: None)
        pump(0.5)
        save(win, f"toasts{sfx}")
        for t in list(win.toasts._toasts):
            t.dismiss()
        # backup / restore / sync folder / packs (only when the controller has that API)
        if hasattr(ctl, "export_backup"):
            import tempfile
            from voxflow.ui import dialogs
            tmp = Path(tempfile.mkdtemp(prefix="voxflow-shots-"))
            settings = win.pages["settings"]
            win.show_page("settings")
            pump(0.15)

            def show_sync(name: str) -> None:
                settings.scroll.ensureWidgetVisible(settings.sync_row, 0, 150)
                pump(0.2)
                save(win, f"page_settings_sync_{name}{sfx}")

            ctl.set("sync_folder", "")
            show_sync("off")
            ctl.set("sync_folder", str(tmp))
            if hasattr(ctl, "demo_set_sync"):
                ctl.demo_set_sync("idle", "", time.time() - 260, ["surface-laptop", "office-pc"])
                show_sync("idle")
                ctl.demo_set_sync("error", "The sync folder is not reachable (is the network drive connected?)",
                                  time.time() - 7300)
                show_sync("error")
            else:
                show_sync("idle")
            ctl.set("sync_folder", "")
            m = settings.export_backup()
            pump(0.35)
            save(win, f"modal_backup_export{sfx}")
            m.close_modal()
            backup = tmp / ("profile" + getattr(ctl, "BACKUP_SUFFIX", ".voxbackup"))
            ctl.export_backup(str(backup))
            real_open = dialogs.ask_open_path
            dialogs.ask_open_path = lambda *a, **k: str(backup)
            m = settings.import_backup()
            pump(0.35)
            save(win, f"modal_backup_merge{sfx}")
            m.mode.set_value("replace", emit=True)
            m.parts["history"].setChecked(False)
            m.parts["settings"].setChecked(False)
            pump(0.3)
            save(win, f"modal_backup_replace{sfx}")
            m.close_modal()
            pack = tmp / ("team" + getattr(ctl, "PACK_SUFFIX", ".voxpack"))
            ctl.export_pack("dictionary", str(pack), "Platform team glossary", "Service names and acronyms we use every day")
            win.show_page("dictionary")
            dialogs.ask_open_path = lambda *a, **k: str(pack)
            m = win.pages["dictionary"].import_pack(str(pack))
            pump(0.35)
            save(win, f"modal_pack_import{sfx}")
            m.close_modal()
            dialogs.ask_open_path = real_open
            win.show_page("snippets")
            m = win.pages["snippets"].share_pack()
            m.entries[1][1].setChecked(False)
            pump(0.35)
            save(win, f"modal_pack_share{sfx}")
            m.close_modal()
            win.show_page("dictionary")
            m = win.pages["dictionary"].share_pack()
            pump(0.35)
            save(win, f"modal_pack_share_dictionary{sfx}")
            m.close_modal()
        win.show_page("settings")
        win.pages["settings"].scroll.verticalScrollBar().setValue(0)
        pump(0.15)
        rec = win.pages["settings"].recorders["command_hotkey"]
        rec.toggle_capture()
        pump(0.2)
        save(win, f"page_settings_capture{sfx}")
        rec.toggle_capture()
        rec.show_message("Already used for Dictation.")
        pump(0.1)
        save(win, f"page_settings_conflict{sfx}")
        rec.msg.hide()
        # keyboard focus rings: Tab-focus a button, a toggle and the theme switch in turn
        win.activateWindow()
        pump(0.1)
        for i, target in enumerate((win.pages["settings"].recorders["hotkey"].btn,
                                    win.pages["settings"].findChild(QWidget, "settings_double_tap_lock"),
                                    win.pages["settings"].mic, win.nav["history"])):
            target.setFocus(Qt.FocusReason.TabFocusReason)
            pump(0.1)
            save(win, f"focus_ring_{i + 1}{sfx}")
        win.setFocus()
        menu = win.pages["settings"].language.open_menu()
        pump(0.35)
        save(menu, f"menu_language{sfx}")
        menu.close()
    ctl.set("theme", "dark")
    pump(0.2)

    # ---- every page once more at the minimum window size (nothing may clip or overflow)
    win.resize(win.minimumSize())
    pump(0.2)
    for key in ("home", "history", "dictionary", "snippets", "style", "settings"):
        page(f"min_{key}", key)
    page("min_style_apps", "style", 100000)
    # toasts at the minimum size: the stack lives in the content area and must not cover the sidebar status card
    win.show_page("home")
    win.toast("warning", "VoxFlow couldn't save your history (the disk may be full or read-only).", "Try again", lambda: None)
    win.toast("error", "Could not save settings: the disk is full.")
    pump(0.5)
    save(win, "min_toasts")
    for t in list(win.toasts._toasts):
        t.dismiss()
    win.resize(1080 + 2 * win.M, 720 + 2 * win.M)
    pump(0.2)

    # ---- device menus and the speed check (Settings > On-device models)
    settings = win.pages["settings"]

    def show_devices(name: str) -> None:
        win.show_page("settings")
        pump(0.15)
        settings.scroll.ensureWidgetVisible(settings.speed_row, 0, 30)
        pump(0.3)
        save(win, f"page_settings_devices_{name}")

    show_devices("nvidia")
    menu = settings.devices["asr"].open_menu()
    pump(0.35)
    save(menu, "menu_device_asr_nvidia")
    menu.close()
    if hasattr(ctl, "demo_set_devices"):             # the preview twin can pose other PCs
        ctl.measure_devices()
        ctl._speed_timer.stop()
        for _ in range(3):
            ctl._speed_step()
        show_devices("measuring")
        ctl.demo_set_devices(
            results={"asr": {"cuda": {"available": False, "reason": "No NVIDIA driver"}, "webgpu": {"ms": 118.0},
                             "cpu": {"ms": 402.0}},
                     "cleanup": {"cuda": {"available": False, "reason": "No NVIDIA driver"},
                                 "vulkan": {"ms": 6100.0}, "cpu": {"ms": 2650.0}}},
            auto={"asr": "webgpu", "cleanup": "cpu"}, state="done", done=0, total=0)
        ctl.demo_set_model("asr", "ready", "Running on the GPU (WebGPU).", None, "webgpu")
        ctl.demo_set_model("cleanup", "ready", "Running on the CPU.", None, "cpu")
        show_devices("igpu_laptop")
        menu = settings.devices["asr"].open_menu()
        pump(0.35)
        save(menu, "menu_device_asr_igpu_laptop")
        menu.close()
        ctl.demo_set_devices(results=copy.deepcopy(_FAKE_RESULTS), auto={"asr": "cuda", "cleanup": "cuda"})
        ctl.reload_models()
        pump(0.2)

    # ---- model problems on the Settings page
    ctl_models = getattr(ctl, "demo_set_model", None)
    if ctl_models is not None:
        ctl.demo_set_model("asr", "downloading", "Downloading… 312 MB of 640 MB", 0.49)
        ctl.demo_set_model("cleanup", "failed", "The cleanup model could not be loaded. Dictation still works; "
                                                "text is tidied with built-in rules.", None)
        win.show_page("settings")
        sb = win.pages["settings"].scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
        pump(0.6)
        save(win, "page_settings_models_busy")
        pill.set_arranging(False)
        ctl.demo_set_state("idle")
        pump(0.7)
        save(pill, "overlay_loading_auto", False, crop)
        ctl.reload_models()
        pump(0.2)

    # ---- tray menu + icons
    anchor = app.primaryScreen().availableGeometry().bottomRight() - QPoint(220, 0)
    menu = ui.tray.show_menu(anchor)
    pump(0.35)
    save(menu, "tray_menu")
    ui.tray._goto("mics")
    pump(0.2)
    save(menu, "tray_menu_mics")
    menu.close()
    ctl.set("theme", "light")
    menu = ui.tray.show_menu(anchor)
    pump(0.35)
    save(menu, "tray_menu_light", True)
    menu.close()
    ctl.set("theme", "dark")
    k = app.primaryScreen().devicePixelRatio()
    sheet = QImage(int((len(ICON_STATES) * 56 + 16) * k), int(140 * k), QImage.Format.Format_ARGB32_Premultiplied)
    sheet.fill(QColor(24, 28, 40))
    p = QPainter(sheet)
    p.scale(k, k)
    p.fillRect(0, 70, len(ICON_STATES) * 56 + 16, 70, QColor(238, 240, 246))
    for i, look in enumerate(ICON_STATES):
        for row, size in ((0, 32), (1, 16)):
            for band in (0, 70):
                pm = make_icon(look).pixmap(size, size)
                p.drawPixmap(16 + i * 56, band + 8 + row * 38, size, size, pm)
    p.end()
    sheet.save(str(SHOTS / f"tray_icons@{scale}.png"))

    # ---- onboarding
    for theme in ("dark", "light"):
        ctl.set("theme", theme)
        sfx = "" if theme == "dark" else "_light"
        ob = ui.open_onboarding()
        for i, step in enumerate(("welcome", "mic", "hotkey", "models", "done")):
            ob.go(i)
            if step == "mic":
                pump(1.2)
            elif step == "hotkey":
                ctl.demo_set_state("listening")
                pump(0.3)
                ctl.demo_add_transcript("Let's ship the new build on Friday.")
                pump(0.7)
                ctl.demo_set_state("idle")
            elif step == "models" and ctl_models is not None:
                ctl.demo_set_model("asr", "ready", "Running on GPU (CUDA)", None, "cuda")
                ctl.demo_set_model("cleanup", "downloading", "Downloading… 212 MB of 398 MB", 0.53)
            pump(0.4)
            save(ob, f"onboarding_{i + 1}_{step}{sfx}", theme == "light")
        if ctl_models is not None:
            ctl.reload_models()
        ob.skip()
        pump(0.1)
    ctl.set("theme", "dark")

    # ---- empty states (a controller with no data at all)
    ui.close()
    empty = FakeController(empty=True)
    empty.set("onboarding_done", True)
    empty.set("theme", "dark")
    ui2 = build_ui(empty, show_window=True)
    pump(0.3)
    for key in ("home", "history", "dictionary", "snippets"):
        ui2.window.show_page(key)
        pump(0.2)
        save(ui2.window, f"empty_{key}")
    empty.set("save_history", False)
    ui2.window.show_page("history")
    pump(0.2)
    save(ui2.window, "empty_history_off")
    ui2.close()

    print(f"[{scale}%] overlay fps ~{pill.fps:.0f}; qt warnings: {warnings or 'none'}")
    ctl.shutdown()
    return 1 if warnings else 0


# --------------------------------------------------------------------------- real-screen QA
def real_screen(args) -> int:
    """Show the demo app on the REAL desktop for a few seconds, grab what the compositor actually displays and
    check DPI rendering, Flow-bar placement on every monitor and that the Flow bar never takes the foreground.

    Nothing is typed or clicked in other apps. Every shot is taken over our own opaque backdrop window, so no
    pixel of the user's desktop ends up in the PNGs. Results: build/shots/real/*.png + REPORT.txt.

    If the workstation is LOCKED the compositor shows the lock screen instead of our windows and the foreground
    window can never change, so screen photos and the focus check would be meaningless. The run then says so,
    marks those checks NOT CHECKED, and saves each window's own real-DPI render buffer (``buffer_*.png``: real
    DirectWrite text at the monitor's scale, but not what the compositor shows) instead of screen photos.
    """
    import ctypes

    for var in ("QT_QPA_PLATFORM", "QT_SCALE_FACTOR", "QT_QPA_FONTDIR"):
        os.environ.pop(var, None)
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QPoint, QRect, Qt, qInstallMessageHandler
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
    from PySide6.QtWidgets import QApplication, QWidget

    out = SHOTS / "real"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    qt_msgs: list[str] = []
    qInstallMessageHandler(lambda _t, _c, msg: qt_msgs.append(msg))
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    if app.platformName() == "offscreen":
        print("real-screen mode needs the real desktop platform")
        return 2

    from voxflow.controller import Controller
    from voxflow.ui import overlay as ov
    from voxflow.ui.shell import build_ui

    u32 = ctypes.WinDLL("user32", use_last_error=True)
    u32.GetForegroundWindow.restype = ctypes.c_void_p
    u32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    u32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
    u32.GetWindowTextW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)

    def desktop_locked() -> bool:
        """True when the input desktop is not ours (lock screen / secure desktop)."""
        u32.OpenInputDesktop.restype = ctypes.c_void_p
        u32.OpenInputDesktop.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint)
        h = u32.OpenInputDesktop(0, False, 0x0001)            # DESKTOP_READOBJECTS
        if not h:
            return True
        name = ctypes.create_unicode_buffer(64)
        needed = ctypes.c_uint(0)
        u32.GetUserObjectInformationW(ctypes.c_void_p(h), 2, name, ctypes.sizeof(name), ctypes.byref(needed))
        u32.CloseDesktop(ctypes.c_void_p(h))
        return name.value.lower() != "default" or "lock screen" in title(int(u32.GetForegroundWindow() or 0)).lower()

    def foreground() -> int:
        return int(u32.GetForegroundWindow() or 0)

    def title(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(200)
        u32.GetWindowTextW(hwnd, buf, 200)
        return buf.value

    report: list[str] = []
    problems: list[str] = []

    def note(line: str) -> None:
        report.append(line)
        print(line)

    skipped: list[str] = []

    def check(name: str, ok: bool, detail: str = "", needs_desktop: bool = False) -> None:
        """``needs_desktop`` checks say nothing while the workstation is locked: report them as not checked."""
        if needs_desktop and locked:
            note(f"[NOT CHECKED] {name}  (workstation locked)")
            skipped.append(name)
            return
        note(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            problems.append(name)

    def pump(seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.008)

    class Backdrop(QWidget):
        """Opaque stand-in for the desktop behind whatever is being photographed."""

        def __init__(self, topmost: bool, light: bool = False):
            super().__init__(None)
            flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus
            if topmost:
                flags |= Qt.WindowType.WindowStaysOnTopHint
            self.setWindowFlags(flags)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.light = light

        def paintEvent(self, _):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(214, 220, 236) if self.light else QColor(38, 42, 74))

        def cover(self, rect: QRect) -> None:
            self.setGeometry(rect)
            self.show()

    def screen_of(rect: QRect):
        return QGuiApplication.screenAt(rect.center()) or app.primaryScreen()

    def grab(rect: QRect, name: str, widget=None, light: bool = False) -> tuple[int, int, float]:
        """Photograph a global logical rectangle from the real screen. Returns (px width, px height, dpr).
        Locked workstation: save ``widget``'s own render buffer over the backdrop colour instead."""
        s = screen_of(rect)
        g = s.geometry()
        if not locked:
            pm = s.grabWindow(0, rect.x() - g.x(), rect.y() - g.y(), rect.width(), rect.height())
            pm.save(str(out / f"{name}.png"))
            return pm.width(), pm.height(), pm.devicePixelRatio()
        top = widget.grab()
        k = top.devicePixelRatio()
        img = QImage(int(rect.width() * k), int(rect.height() * k), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(214, 220, 236) if light else QColor(38, 42, 74))
        p = QPainter(img)
        off = widget.mapToGlobal(QPoint(0, 0)) - rect.topLeft()
        raw = top.toImage()
        raw.setDevicePixelRatio(1.0)
        p.drawImage(int(off.x() * k), int(off.y() * k), raw)
        p.end()
        img.save(str(out / f"buffer_{name}.png"))
        return img.width(), img.height(), k

    locked = desktop_locked()
    screens = app.screens()
    note(f"VoxFlow real-screen check, {time.strftime('%Y-%m-%d %H:%M')}; Qt platform '{app.platformName()}'")
    for i, s in enumerate(screens):
        g, a = s.geometry(), s.availableGeometry()
        note(f"monitor {i + 1}: '{s.name()}' {int(g.width() * s.devicePixelRatio())}x"
             f"{int(g.height() * s.devicePixelRatio())} px at {int(s.devicePixelRatio() * 100)} % "
             f"(logical {g.width()}x{g.height()} at {g.x()},{g.y()}; work area {a.width()}x{a.height()})"
             + ("  [primary]" if s is app.primaryScreen() else ""))

    if locked:
        note("WORKSTATION IS LOCKED: the screen shows the lock screen, not VoxFlow. Screen photos and the "
             "focus-stealing check are NOT possible in this run; window render buffers are saved instead "
             "(buffer_*.png). Unlock the PC and run this again for the real thing.")
    # pure geometry: a physical-pixel point on each monitor must map back to that monitor
    for i, s in enumerate(screens):
        g, k = s.geometry(), s.devicePixelRatio()
        px, py = int(g.x() + g.width() * k / 2), int(g.y() + g.height() * k / 2)
        check(f"physical point ({px},{py}) maps to monitor {i + 1}", ov.screen_at_physical(px, py) is s)
    ctl = Controller(demo=True)
    ctl.set("onboarding_done", True)
    ctl.set("theme", "dark")
    fg_before = foreground()
    note(f"foreground window before VoxFlow shows anything: {fg_before:#x} '{title(fg_before)[:40]}'")
    ui = build_ui(ctl, show_window=False)
    ctl.start()
    pill, win = ui.overlay, ui.window
    pump(0.8)

    # ---- 1. the Flow bar must never take the foreground (no backdrop, nothing else of ours on screen)
    check("idle Flow bar appeared without taking the foreground", foreground() == fg_before,
          f"foreground now {foreground():#x} '{title(foreground())[:40]}'", needs_desktop=True)
    cycle = (("listening", "dictate", ""), ("locked", "dictate", ""), ("processing", "dictate", ""),
             ("done", "dictate", "Pasted"), ("listening", "command", ""), ("error", "dictate", "Didn't catch that"),
             ("idle", "dictate", ""))
    stolen = []
    for state, mode, msg in cycle:
        ctl.demo_set_state(state, mode, msg)
        pump(0.45)
        if foreground() != fg_before:
            stolen.append(f"{state}/{mode} -> '{title(foreground())[:30]}'")
    check("foreground window unchanged through every Flow bar state", not stolen, "; ".join(stolen),
          needs_desktop=True)
    ex = u32.GetWindowLongPtrW(int(pill.winId()), -20)
    check("Flow bar window is NOACTIVATE + TOOLWINDOW + click-through (WS_EX_TRANSPARENT) + topmost",
          bool(ex & 0x08000000) and bool(ex & 0x80) and bool(ex & 0x20) and bool(ex & 0x8), f"exstyle {ex:#x}")
    check("Flow bar is not the active window and holds no focus", not pill.isActiveWindow() and not pill.hasFocus())

    # ---- 2. Flow bar placement: bottom centre of the screen that holds the foreground window
    def pill_global() -> QRect:
        return QRect(pill.mapToGlobal(pill.pill_rect().topLeft().toPoint()), pill.pill_rect().size().toSize())

    def placement(label: str, s, needs_desktop: bool = False) -> None:
        r, a = pill_global(), s.availableGeometry()
        check(f"Flow bar on {label}: inside the work area, centred, {ov.SCREEN_GAP} px above its bottom edge",
              a.contains(r) and abs(r.center().x() - a.center().x()) <= 2 and abs(a.bottom() - r.bottom() - ov.SCREEN_GAP) <= 2,
              f"pill {r.x()},{r.y()} {r.width()}x{r.height()}; work area {a.x()},{a.y()} {a.width()}x{a.height()}",
              needs_desktop)

    active = ov.active_screen()
    note(f"active screen (holds the foreground window): '{active.name()}'")
    ctl.demo_set_state("listening")
    pump(0.5)
    placement(f"the active monitor '{active.name()}'", active)
    ctl.demo_set_state("idle")
    pump(0.3)

    # ---- 3. Flow bar states, photographed over a top-most backdrop on the active monitor
    back = Backdrop(topmost=True)
    a = active.availableGeometry()
    area = QRect(a.center().x() - 300, a.bottom() - 150, 600, 150)
    back.cover(area)
    pill.raise_()
    pump(0.3)
    shots = (("idle", "idle", "dictate", "", 0.7), ("listening", "listening", "dictate", "", 1.3),
             ("locked", "locked", "dictate", "", 1.2), ("command", "listening", "command", "", 1.2),
             ("processing", "processing", "dictate", "", 0.9), ("done", "done", "dictate", "Pasted", 0.6),
             ("error", "error", "dictate", "Didn't catch that", 0.8))
    for name, state, mode, msg, secs in shots:
        ctl.demo_set_state(state, mode, msg)
        pump(secs)
        w, h, dpr = grab(area, f"flowbar_{name}", pill)
    check("Flow bar shots are native resolution", abs(w - area.width() * active.devicePixelRatio()) <= 2,
          f"{w}x{h} px for {area.width()}x{area.height()} logical at {active.devicePixelRatio()}")
    ctl.demo_set_state("idle")
    pill.set_state("loading", "Downloading speech model", "dictate", 0.43)
    pump(0.8)
    grab(area, "flowbar_loading", pill)
    pill.set_state("idle")
    back.light = True
    back.update()
    ctl.demo_set_state("listening")
    pump(1.2)
    grab(area, "flowbar_listening_on_light", pill, light=True)
    ctl.demo_set_state("idle")
    check("foreground still unchanged after the Flow bar photo session", foreground() == fg_before,
          f"'{title(foreground())[:40]}'", needs_desktop=True)
    back.hide()
    pump(0.2)

    # ---- 4. main window, every page, on every monitor (this legitimately takes the foreground: it is our window)
    back = Backdrop(topmost=False)

    def frame(widget, margin: int = 16) -> QRect:
        return widget.frameGeometry().adjusted(-margin, -margin, margin, margin)

    win.present("home")
    pump(0.6)
    check("opening the main window makes it the foreground window (expected)",
          foreground() == int(win.winId()), f"foreground '{title(foreground())[:40]}'", needs_desktop=True)
    for i, s in enumerate(screens):
        tag = f"m{i + 1}"
        ag = s.availableGeometry()
        w0 = min(1080 + 2 * win.M, ag.width() - 40)
        h0 = min(720 + 2 * win.M, ag.height() - 40)
        win.setGeometry(ag.center().x() - w0 // 2, ag.center().y() - h0 // 2, w0, h0)
        pump(0.5)
        back.cover(frame(win).intersected(ag))
        win.raise_()
        win.activateWindow()
        pump(0.4)
        on = QGuiApplication.screenAt(win.geometry().center())
        check(f"main window moved to monitor {i + 1} and renders at its scale", on is s and
              abs(win.devicePixelRatioF() - s.devicePixelRatio()) < 0.01, f"window dpr {win.devicePixelRatioF()}")
        pages = ("home", "history", "dictionary", "snippets", "style", "settings") if i == 0 else ("home", "settings")
        for key in pages:
            win.show_page(key)
            pump(0.45)
            pw, ph, _d = grab(win.frameGeometry(), f"window_{key}_{tag}", win)
        check(f"window shot on monitor {i + 1} is native resolution (no upscaling blur)",
              abs(pw - win.width() * s.devicePixelRatio()) <= 2, f"{pw}x{ph} px for {win.width()}x{win.height()} logical")
        if i == 0:
            sb = win.pages["settings"].scroll.verticalScrollBar()
            sb.setValue(sb.maximum() // 2)
            pump(0.3)
            grab(win.frameGeometry(), f"window_settings_mid_{tag}", win)
            sb.setValue(sb.maximum())
            pump(0.3)
            grab(win.frameGeometry(), f"window_settings_end_{tag}", win)
            win.show_page("dictionary")
            modal = win.pages["dictionary"].edit(None)
            pump(0.5)
            grab(win.frameGeometry(), f"window_dialog_{tag}", win)
            modal.close_modal()
            win.toast("success", "Exported 9 transcripts to voxflow-history.txt.")
            pump(0.5)
            grab(win.frameGeometry(), f"window_toast_{tag}", win)
            ctl.set("theme", "light")
            for key in ("home", "settings"):
                win.show_page(key)
                pump(0.45)
                grab(win.frameGeometry(), f"window_{key}_light_{tag}", win)
            ctl.set("theme", "dark")
            pump(0.2)
        # the Flow bar follows the foreground window's monitor
        ctl.demo_set_state("listening")
        pump(0.6)
        placement(f"monitor {i + 1} '{s.name()}' (VoxFlow window in front there)", s, needs_desktop=True)
        ctl.demo_set_state("idle")
        pump(0.3)

    # ---- 5. tray menu (popup) next to the window on the primary monitor
    ps = app.primaryScreen().availableGeometry()
    win.setGeometry(ps.center().x() - 554, ps.center().y() - 374, 1108, 748)
    win.show_page("home")
    pump(0.4)
    back.cover(frame(win))
    win.raise_()
    menu = ui.tray.show_menu(win.geometry().center() + QPoint(0, 220))
    pump(0.5)
    grab(menu.frameGeometry(), "tray_menu", menu)
    check("tray menu is fully on screen", ps.contains(menu.frameGeometry()))
    menu.close()
    look = ui.tray.current_look()
    check("tray icon is showing", ui.tray.icon.isVisible(), f"look '{look}'")
    win.hide()
    pump(0.2)

    # ---- 6. onboarding
    ob = ui.open_onboarding()
    pump(0.5)
    back.cover(frame(ob))
    ob.raise_()
    ob.activateWindow()
    for i, step in enumerate(("welcome", "mic", "hotkey", "models", "done")):
        ob.go(i)
        pump(0.9 if step == "mic" else 0.45)
        ow, oh, _d = grab(ob.frameGeometry(), f"onboarding_{i + 1}_{step}", ob)
    check("onboarding is centred on its screen and native resolution",
          abs(ob.frameGeometry().center().x() - screen_of(ob.frameGeometry()).availableGeometry().center().x()) <= 2
          and abs(ow - ob.width() * screen_of(ob.frameGeometry()).devicePixelRatio()) <= 2, f"{ow}x{oh} px")
    ob.skip()
    back.hide()
    pump(0.2)

    # ---- done: take everything down again
    ui.close()
    ctl.shutdown()
    back.close()
    pump(0.3)
    ours = [w for w in app.topLevelWidgets() if w.isVisible()]
    check("every VoxFlow window is closed again", not ours, ", ".join(type(w).__name__ for w in ours))
    noisy = [m for m in qt_msgs if m.strip()]
    check("no Qt warnings on the real platform", not noisy, " | ".join(noisy[:3]))
    kind = "window-buffer renders (NOT screen photos)" if locked else "real-screen shots"
    note(f"{len(list(out.glob('*.png')))} {kind} in {out}")
    verdict = "all checks that ran passed" if not problems else f"{len(problems)} problem(s): " + "; ".join(problems)
    if skipped:
        verdict += f"; {len(skipped)} check(s) NOT CHECKED because the workstation is locked"
    note("RESULT: " + verdict)
    (out / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 1 if problems else (3 if locked else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", type=int, nargs="+", default=[100, 150])
    ap.add_argument("--controller", choices=("preview", "real"), default="preview")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--real-screen", action="store_true",
                    help="show the demo app briefly on the real desktop and photograph it (build/shots/real/)")
    a = ap.parse_args()
    if a.real_screen:
        sys.exit(real_screen(a))
    sys.exit(child(a) if a.child else parent(a))
