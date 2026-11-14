"""Global hold-to-talk hotkey.

* ``HotkeyMachine``  - pure, testable state machine (no OS calls).
* ``HotkeyListener`` - Windows WH_KEYBOARD_LL hook (ctypes) feeding the machine.

Events emitted by a ``hold`` machine: ``start`` (key combo pressed), ``stop`` (released, or tapped
again while locked), ``lock`` (quick double-tap -> hands-free), ``cancel`` (Esc while active).
A ``press`` machine emits a single ``press`` each time its combo goes down.

The listener hosts several named bindings (dictation, command mode, paste-last, ...). Only one binding
can be active at a time: while one is held or locked the others cannot fire.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field

from .context import focus_state
from .inject import EXTRA_MARKER

# --------------------------------------------------------------------------- key names
VK_CTRL, VK_LCTRL, VK_RCTRL = 0x11, 0xA2, 0xA3
VK_SHIFT, VK_LSHIFT, VK_RSHIFT = 0x10, 0xA0, 0xA1
VK_ALT, VK_LALT, VK_RALT = 0x12, 0xA4, 0xA5
VK_LWIN, VK_RWIN = 0x5B, 0x5C
VK_ESC = 0x1B
VK_MASK = 0xE8  # unassigned vk used to "mask" a lone Win/Alt release

_MOD_GROUPS = {
    "ctrl": frozenset({VK_CTRL, VK_LCTRL, VK_RCTRL}),
    "alt": frozenset({VK_ALT, VK_LALT, VK_RALT}),
    "shift": frozenset({VK_SHIFT, VK_LSHIFT, VK_RSHIFT}),
    "win": frozenset({VK_LWIN, VK_RWIN}),
}
_MOD_ORDER = ["ctrl", "alt", "shift", "win"]
_ALIASES = {"control": "ctrl", "ctl": "ctrl", "option": "alt", "windows": "win", "super": "win",
            "meta": "win", "cmd": "win", "return": "enter", "escape": "esc", "spacebar": "space"}
_NAMED = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "backspace": 0x08,
          "capslock": 0x14, "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23,
          "pageup": 0x21, "pagedown": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
          "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE,
          ",": 0xBC, ".": 0xBE, "/": 0xBF, "\\": 0xDC}
for _i in range(1, 25):
    _NAMED[f"f{_i}"] = 0x6F + _i
for _i in range(10):
    _NAMED[f"num{_i}"] = 0x60 + _i
_NAMED.update({"nummul": 0x6A, "numadd": 0x6B, "numsub": 0x6D, "numdot": 0x6E, "numdiv": 0x6F,
               "pause": 0x13, "scrolllock": 0x91, "printscreen": 0x2C, "menu": 0x5D})
_NAME_BY_VK = {v: k for k, v in _NAMED.items()}


def _vk_for_token(tok: str) -> int:
    if tok in _NAMED:
        return _NAMED[tok]
    if len(tok) == 1 and (tok.isalpha() or tok.isdigit()):
        return ord(tok.upper())
    raise ValueError(f"unknown key '{tok}'")


def parse_hotkey(combo: str) -> list[frozenset[int]]:
    """'ctrl+win' -> [{ctrl vks}, {win vks}].  Raises ValueError on garbage/empty."""
    groups: list[frozenset[int]] = []
    for raw in combo.lower().replace(" ", "").split("+"):
        if not raw:
            continue
        tok = _ALIASES.get(raw, raw)
        groups.append(_MOD_GROUPS[tok] if tok in _MOD_GROUPS else frozenset({_vk_for_token(tok)}))
    if not groups:
        raise ValueError("empty hotkey")
    return groups


def vk_group_name(vk: int) -> str | None:
    for name, vks in _MOD_GROUPS.items():
        if vk in vks:
            return name
    if vk in _NAME_BY_VK:
        return _NAME_BY_VK[vk]
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk).lower()
    return None


def combo_from_vks(vks) -> str:
    mods, others = [], []
    for vk in sorted(vks):
        n = vk_group_name(vk)
        if n is None:
            continue
        if n in _MOD_GROUPS:
            if n not in mods:
                mods.append(n)
        elif n not in others:
            others.append(n)
    mods.sort(key=_MOD_ORDER.index)
    return "+".join(mods + others)


def pretty_hotkey(combo: str) -> str:
    names = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win", "space": "Space"}
    return " + ".join(names.get(p, p.upper() if len(p) <= 3 else p.title()) for p in combo.split("+") if p)


def _tokens(combo: str) -> list[str]:
    toks = [_ALIASES.get(t, t) for t in combo.lower().replace(" ", "").split("+") if t]
    return list(dict.fromkeys(toks))


def normalize_hotkey(combo: str) -> str:
    """Canonical spelling: lower-case, aliases resolved, modifiers first in a fixed order."""
    toks = _tokens(combo)
    mods = sorted((t for t in toks if t in _MOD_GROUPS), key=_MOD_ORDER.index)
    return "+".join(mods + [t for t in toks if t not in _MOD_GROUPS])


# Combinations Windows or nearly every app already relies on.
_RESERVED = {frozenset(c.split("+")) for c in (
    "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z", "ctrl+y", "ctrl+a", "ctrl+s", "ctrl+f", "ctrl+p", "ctrl+n",
    "ctrl+t", "ctrl+w", "alt+tab", "alt+f4", "alt+space", "ctrl+alt+delete", "ctrl+shift+esc", "win+l",
    "win+d", "win+e", "win+r", "win+tab", "win+space", "win+v", "win+i", "win+s", "win+a", "win+x",
    "win+m", "win+h", "win+.", "win+;", "alt+enter", "ctrl+alt+tab", "alt+shift", "ctrl+shift",
)}
_FUNCTION_KEYS = {f"f{i}" for i in range(1, 25)}


def hotkey_problem(combo: str, others: dict | None = None) -> str:
    """Why ``combo`` cannot be used as a global shortcut ('' when it is fine). Messages are for the user.

    ``others`` maps a label ("Command mode") to the combination that action already uses."""
    try:
        parse_hotkey(combo)
    except ValueError as exc:
        if "empty" in str(exc):
            return "Press a key combination to set a shortcut."
        return f"VoxFlow can't use that key ({str(exc).replace('unknown key ', '')})."
    toks = _tokens(combo)
    mods = [t for t in toks if t in _MOD_GROUPS]
    keys = [t for t in toks if t not in _MOD_GROUPS]
    pretty = pretty_hotkey(normalize_hotkey(combo))
    if "esc" in keys:
        return "Esc cancels a recording, so it can't be part of a shortcut."
    if len(keys) > 1:
        return "Use modifier keys (Ctrl, Alt, Shift, Win) plus at most one other key."
    if frozenset(toks) in _RESERVED:
        return f"{pretty} is already used by Windows or most apps. Pick another combination."
    if not keys and len(mods) < 2:
        return "A single modifier key would trigger all the time. Combine two, such as Ctrl + Win."
    if keys and keys[0] not in _FUNCTION_KEYS and not set(mods) - {"shift"}:
        return f"{pretty} would get in the way of typing. Add Ctrl, Alt or Win."
    for label, other in (others or {}).items():
        if not other or not combos_clash(combo, other):
            continue
        if normalize_hotkey(other) == normalize_hotkey(combo):
            return f"{pretty} is already used for {label}."
        return f"{pretty} overlaps with the {label} shortcut ({pretty_hotkey(normalize_hotkey(other))})."
    return ""


def combos_clash(a: str, b: str) -> bool:
    """True if holding one combination would also satisfy the other (identical, or one contains the other)."""
    try:
        ga, gb = set(parse_hotkey(a)), set(parse_hotkey(b))
    except ValueError:
        return False
    return ga <= gb or gb <= ga


# --------------------------------------------------------------------------- state machine
IDLE, HOLD, LOCKED, WAIT_RELEASE = "idle", "hold", "locked", "wait_release"
TAP_MAX_S = 0.28       # a press shorter than this counts as a "tap"
DOUBLE_GAP_S = 0.45    # max gap between two taps to lock hands-free
MAX_HOLD_S = 120.0     # safety net: a lost key-up / forgotten lock can never record forever
CAPTURE_TIMEOUT_S = 10.0   # an abandoned "press your shortcut" prompt must not eat a key typed later


@dataclass
class Result:
    swallow: bool = False
    events: list = field(default_factory=list)
    mask: bool = False          # inject a dummy key before letting this event through (kills Start menu)
    captured: str | None = None   # a combo, or "" when the capture was cancelled (Esc)


def _locked(fn):
    """Serialise machine access: the hook thread (feed/tick) and the UI thread (force_idle, set_combo...)."""
    def wrapper(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


class HotkeyMachine:
    def __init__(self, combo: str = "ctrl+win", double_tap: bool = True, clock=time.monotonic,
                 max_hold_s: float = MAX_HOLD_S, mode: str = "hold"):
        if mode not in ("hold", "press"):
            raise ValueError(f"unknown hotkey mode '{mode}'")
        self._lock = threading.RLock()
        self.mode = mode
        self.blocked = False        # set by the listener while another binding is active
        self.double_tap = double_tap
        self.max_hold_s = max_hold_s
        self.clock = clock
        self.down: set[int] = set()
        self.state = IDLE
        self._press_t = 0.0
        self._last_tap_t: float | None = None
        self._swallowed: set[int] = set()
        self._mask_vks: set[int] = set()
        self._capturing = False
        self._cap_max: set[int] = set()
        self._cap_t = 0.0
        self.set_combo(combo)

    # ---- configuration
    @_locked
    def set_combo(self, combo: str) -> None:
        self.groups = parse_hotkey(combo)
        self.combo = combo
        self._trigger_vks = set().union(*self.groups)
        self.state = IDLE
        self._last_tap_t = None

    @_locked
    def force_idle(self) -> None:
        """The recording was ended externally (e.g. max-length cap): stop tracking hold / hands-free lock."""
        if self.state in (HOLD, LOCKED):
            self.state = WAIT_RELEASE
        self._last_tap_t = None

    @_locked
    def start_capture(self) -> None:
        self._capturing, self._cap_max, self._cap_t = True, set(), self.clock()

    @_locked
    def expire_capture(self, timeout_s: float = CAPTURE_TIMEOUT_S) -> bool:
        """End a capture nobody completed in time. True if one was ended."""
        if self._capturing and not self._cap_max and self.clock() - self._cap_t > timeout_s:
            self._capturing = False
            return True
        return False

    @_locked
    def cancel_capture(self) -> None:
        self._capturing = False

    @property
    def capturing(self) -> bool:
        return self._capturing

    @property
    def active(self) -> bool:
        """Held or locked hands-free: the binding currently owns the keyboard."""
        return self.state in (HOLD, LOCKED)

    # ---- helpers
    def _satisfied(self) -> bool:
        return all(g & self.down for g in self.groups)

    @staticmethod
    def _is_modifier(vk: int) -> bool:
        return any(vk in g for g in _MOD_GROUPS.values())

    @staticmethod
    def _maskable(vk: int) -> bool:
        return vk in _MOD_GROUPS["win"] or vk in _MOD_GROUPS["alt"]

    # ---- resync with the real keyboard state
    def _drop_stale(self, probe, keep: int | None = None) -> bool:
        """Forget keys the OS says are up (missed key-up: Win+L, secure desktop, hook timeout).

        Swallowed keys are skipped: blocked events never update GetAsyncKeyState."""
        dropped = False
        for k in list(self.down):
            if k != keep and k not in self._swallowed and not probe(k):
                self.down.discard(k)
                self._mask_vks.discard(k)
                dropped = True
        return dropped

    @_locked
    def tick(self, probe) -> list:
        """Periodic watchdog. ``probe(vk)->bool`` = key physically down. Returns events to emit."""
        res = Result()
        if self._capturing:
            self._drop_stale(probe)
            return []
        self._drop_stale(probe)
        if self.state == HOLD and not self._satisfied():
            self._deactivate(res)
        elif self.state == WAIT_RELEASE and not self.down & self._trigger_vks:
            self.state = IDLE
        if self.state in (HOLD, LOCKED) and self.clock() - self._press_t > self.max_hold_s:
            res.events.append("stop")
            self.state = WAIT_RELEASE
        return res.events

    # ---- main entry
    @_locked
    def feed(self, vk: int, is_down: bool, probe=None) -> Result:
        res = Result()
        if (probe is not None and is_down and self._drop_stale(probe, keep=vk) and self.state == HOLD
                and not all(g & (self.down | {vk}) for g in self.groups)):
            self._deactivate(res)
        was_down = vk in self.down
        if is_down:
            self.down.add(vk)
        else:
            self.down.discard(vk)

        if self._capturing:
            return self._feed_capture(vk, is_down, res)

        in_combo = vk in self._trigger_vks
        mod = self._is_modifier(vk)

        if is_down:
            if was_down:  # auto-repeat
                res.swallow = vk in self._swallowed
                return res
            if vk == VK_ESC and self.state in (HOLD, LOCKED):
                res.events.append("cancel")
                self.state = WAIT_RELEASE
                self._last_tap_t = None
                self._swallowed.add(vk)          # the cancelling Esc must not also reach the focused app
                res.swallow = True
                return res
            if not in_combo and self.state == HOLD:
                # a different shortcut that merely contains ours (Ctrl+Win+Left ...): not a dictation
                res.events.append("cancel")
                self.state = WAIT_RELEASE
                self._last_tap_t = None
                return res
            # exact chord only: with any other key held (Ctrl+Win+Alt, Shift+Ctrl+Win ...) this is some
            # other shortcut, and a binding that was just cancelled by a foreign key must not hand over
            # to another binding within the same key press
            if (in_combo and self._satisfied() and self.state in (IDLE, LOCKED) and not self.blocked
                    and self.down <= self._trigger_vks):
                self._activate(res)
                for g_vk in self._trigger_vks & self.down:
                    if self._maskable(g_vk):
                        self._mask_vks.add(g_vk)
                if not mod:  # non-modifier combo keys are eaten so they can't leak
                    self._swallowed.add(vk)
                    res.swallow = True
            elif in_combo and not mod and self.state != IDLE:
                self._swallowed.add(vk)
                res.swallow = True
            return res

        # key up
        if vk in self._swallowed:
            self._swallowed.discard(vk)
            res.swallow = True
        if vk in self._mask_vks:
            self._mask_vks.discard(vk)
            res.mask = True
        if in_combo and self.state in (HOLD, WAIT_RELEASE) and not self._satisfied():
            self._deactivate(res)
        if self.state == WAIT_RELEASE and not self.down & self._trigger_vks:
            self.state = IDLE
        return res

    def _activate(self, res: Result) -> None:
        if self.mode == "press":
            res.events.append("press")
            self.state = WAIT_RELEASE
            return
        if self.state == LOCKED:
            res.events.append("stop")
            self.state = WAIT_RELEASE
            self._last_tap_t = None
        else:
            self.state = HOLD
            self._press_t = self.clock()
            res.events.append("start")

    def _deactivate(self, res: Result) -> None:
        if self.state == WAIT_RELEASE:
            self.state = IDLE
            return
        now = self.clock()
        is_tap = (now - self._press_t) < TAP_MAX_S
        if (self.double_tap and is_tap and self._last_tap_t is not None
                and now - self._last_tap_t < DOUBLE_GAP_S):
            self.state = LOCKED
            self._last_tap_t = None
            res.events.append("lock")
            return
        self.state = IDLE
        self._last_tap_t = now if is_tap else None
        res.events.append("stop")

    def _feed_capture(self, vk: int, is_down: bool, res: Result) -> Result:
        if is_down:
            if vk == VK_ESC:                     # Esc cancels the capture; it never becomes the shortcut
                self._capturing, self._cap_max = False, set()
                self._swallowed.add(vk)
                res.swallow, res.captured = True, ""
                return res
            if vk_group_name(vk) is None:        # a key we have no name for: ignore it, keep waiting
                return res
            self._cap_max.add(vk)
            res.swallow = not self._is_modifier(vk)
            if res.swallow:
                self._swallowed.add(vk)
        else:
            if vk in self._swallowed:
                self._swallowed.discard(vk)
                res.swallow = True
            if self._maskable(vk) and self._cap_max:
                res.mask = True
            if not self.down and self._cap_max:
                combo = combo_from_vks(self._cap_max)
                self._capturing = False
                self._cap_max = set()
                if combo:  # pragma: no branch - only named keys enter _cap_max, so it is never empty
                    res.captured = combo
        return res


# --------------------------------------------------------------------------- menu guard
MENU_SETTLE_S = 0.12       # time a window gets to react to the Alt key-up before we look at its focus
MENU_VERIFY_S = 0.10       # time it gets to leave its menu after our Alt tap
MENU_RETRIES = 12          # looks while the user keeps a modifier key held, then give up


class MenuGuard:
    """Gives the keyboard focus back when a shortcut that contains Alt moved it into the target's menu.

    A classic window opens its menu on a lone Alt; the mask key prevents that. Windows 11 Notepad (a XAML menu
    bar) moves the focus to its menu after Ctrl+Alt+F13 even though other keys were pressed meanwhile, so the
    paste that follows a dictation goes nowhere. The guard notes which control has the focus when the keys of
    a shortcut start to go down and looks again shortly after Alt came up. If the same window now has the focus
    somewhere else, or is in menu mode, one Alt tap leaves the menu. If that does not bring the old focus back
    (the user moved it, it was no menu), a second tap undoes the first, so the window is left as it was found.

    ``focus()`` -> ``(window, focused control, in menu mode)``; ``tap()`` sends one Alt press and release;
    ``idle()`` -> True when no modifier key is physically held. All calls come from the hook thread except
    ``wait``, which the paste path uses so that it never pastes while a check is still due.
    """

    def __init__(self, focus, tap, idle, clock=time.monotonic):
        self._focus, self._tap, self._idle, self._clock = focus, tap, idle, clock
        self._origin: tuple | None = None
        self._armed = False
        self._phase = ""            # "" | "check" | "verify"
        self._due = 0.0
        self._tries = 0
        self._settled = threading.Event()
        self._settled.set()
        self.taps = 0               # Alt taps sent so far (diagnostics and tests)

    @property
    def pending(self) -> bool:
        return bool(self._phase)

    def chord_began(self) -> None:
        """The first key of a possible shortcut went down: remember where the focus is now."""
        if not self._armed and not self._phase:
            self._origin = self._focus()

    def fired(self) -> None:
        """A shortcut that contains Alt was recognised."""
        if self._origin is not None and not self._phase:
            self._armed = True
            self._settled.clear()

    def alt_released(self) -> None:
        if self._armed and not self._phase:
            self._phase, self._tries = "check", 0
            self._due = self._clock() + MENU_SETTLE_S

    def step(self) -> None:
        """Timer pass: look at the focus once the window had time to react, repair it if it moved."""
        if not self._phase or self._clock() < self._due:
            return
        if not self._idle():                                 # an Alt tap now would combine with the held key
            self._tries += 1
            if self._tries > MENU_RETRIES:
                self._finish()
            else:
                self._due = self._clock() + MENU_SETTLE_S
            return
        origin, now = self._origin, self._focus()
        if self._phase == "check":
            moved = (now[1] and now[1] != origin[1]) or (now[2] and not origin[2])
            if origin[1] and now[0] == origin[0] and moved:
                self._tap()
                self.taps += 1
                self._phase, self._due = "verify", self._clock() + MENU_VERIFY_S
                return
        elif now != origin and now[0] == origin[0]:
            self._tap()                                      # not a menu after all: undo our tap
            self.taps += 1
        self._finish()

    def _finish(self) -> None:
        self._phase, self._armed, self._origin = "", False, None
        self._settled.set()

    def wait(self, timeout: float = 0.6) -> bool:
        """Block until no focus check is outstanding. False if it did not finish in time."""
        return self._settled.wait(timeout)


# --------------------------------------------------------------------------- Windows hook
class HotkeyListener:
    """Installs a low-level keyboard hook on a dedicated thread and feeds the hotkey machines.

    ``on_event(name)`` is called from the hook thread (keep it trivial - emit a Qt signal).
    ``on_captured(combo)`` is called once after ``begin_capture()``.

    The constructor sets up the primary (dictation) binding, reachable as ``machine``. More bindings are
    added with ``add_binding``; ``enabled = False`` lets every key through untouched (dictation paused).
    """

    WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x100, 0x101, 0x104, 0x105

    def __init__(self, combo: str, on_event, on_captured=None, double_tap: bool = True):
        self.machine = HotkeyMachine(combo, double_tap)
        self.on_event = on_event
        self.on_captured = on_captured
        self.enabled = True
        self._extra: dict[str, list] = {}      # name -> [machine | None, on_event, mode]
        self.guard: MenuGuard | None = None    # set by the hook thread; tests may install their own
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._hook = None
        self._ready = threading.Event()
        self._proc = None
        self.error: str | None = None

    # ---- bindings
    def add_binding(self, name: str, combo: str, on_event, mode: str = "hold") -> None:
        """Register another shortcut. ``combo == ""`` registers it switched off (see ``set_binding``)."""
        self._extra[name] = [HotkeyMachine(combo, False, mode=mode) if combo else None, on_event, mode]

    def set_binding(self, name: str, combo: str) -> None:
        """Change (or with ``""`` switch off) an extra binding. Raises ValueError for an unusable combo."""
        slot = self._extra[name]
        if not combo:
            slot[0] = None
        elif slot[0] is None:
            slot[0] = HotkeyMachine(combo, False, mode=slot[2])
        else:
            slot[0].set_combo(combo)

    def binding(self, name: str) -> HotkeyMachine | None:
        return self._extra[name][0]

    def _machines(self) -> list[tuple[HotkeyMachine, object]]:
        """(machine, callback) pairs, primary first. The primary callback is looked up late on purpose."""
        out = [(self.machine, lambda ev: self.on_event(ev))]
        out.extend((slot[0], slot[1]) for slot in list(self._extra.values()) if slot[0] is not None)
        return out

    def force_idle(self) -> None:
        """A recording was ended externally: no binding may keep tracking a hold / hands-free lock."""
        for m, _cb in self._machines():
            m.force_idle()

    def feed(self, vk: int, is_down: bool, probe=None) -> Result:
        """Route one key event through every binding (called by the hook; public for tests)."""
        total = Result()
        if self.machine.capturing:
            res = self.machine.feed(vk, is_down, probe)
            if res.captured is not None and self.on_captured:
                self.on_captured(res.captured)
            return res
        if not self.enabled:
            return total
        pairs = self._machines()
        guard = self.guard
        if (guard is not None and is_down and not self.machine.down
                and any(vk in m._trigger_vks and _MOD_GROUPS["alt"] in m.groups for m, _ in pairs)):
            guard.chord_began()                  # only shortcuts that contain Alt are watched
        for m, cb in pairs:
            m.blocked = any(o.active for o, _ in pairs if o is not m)
            res = m.feed(vk, is_down, probe)
            total.swallow |= res.swallow
            total.mask |= res.mask
            if guard is not None and is_down and res.events and _MOD_GROUPS["alt"] in m.groups:
                guard.fired()
            for ev in res.events:
                cb(ev)
        if guard is not None and not is_down and vk in _MOD_GROUPS["alt"]:
            guard.alt_released()
        return total

    def settle(self, timeout: float = 0.6) -> bool:
        """Wait until the focus check that follows a shortcut with Alt is over (see ``MenuGuard``).
        Called by the paste path; returns at once when nothing is outstanding."""
        guard = self.guard
        return True if guard is None else guard.wait(timeout)

    def tick(self, probe) -> None:
        """Watchdog pass over every binding (called by the hook thread's timer; public for tests)."""
        if self.machine.expire_capture() and self.on_captured:
            self.on_captured("")
        for m, cb in self._machines():
            for ev in m.tick(probe):
                cb(ev)

    # ---- lifecycle
    def start(self) -> bool:
        if sys.platform != "win32":
            self.error = "hook only available on Windows"
            return False
        if self._thread is not None and self._thread.is_alive():
            return self.error is None
        self.error = None
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="voxflow-hook", daemon=True)
        self._thread.start()
        if not self._ready.wait(3.0):
            self.error = "the keyboard hook did not start in time"
        return self.error is None

    def stop(self) -> None:
        """Idempotent: posts WM_QUIT so the hook thread unhooks itself, then waits for it."""
        tid, self._tid = self._tid, 0
        if tid:
            ctypes.windll.user32.PostThreadMessageW(tid, 0x0012, 0, 0)  # WM_QUIT
        th, self._thread = self._thread, None
        if th is not None and th is not threading.current_thread():
            th.join(3.0)

    def set_combo(self, combo: str) -> None:
        self.machine.set_combo(combo)

    def begin_capture(self) -> None:
        self.machine.start_capture()

    def cancel_capture(self) -> None:
        self.machine.cancel_capture()

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        LRESULT = ctypes.c_ssize_t

        class KBD(ctypes.Structure):
            _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        PROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = LRESULT
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, PROC, wintypes.HINSTANCE, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = wintypes.HHOOK

        user32.GetAsyncKeyState.restype = ctypes.c_short

        def _probe(vk: int) -> bool:
            return bool(user32.GetAsyncKeyState(vk) & 0x8000)

        def _tap_alt() -> None:
            from .inject import send_keys
            send_keys([(VK_LALT, True), (VK_LALT, False)])

        def _mods_idle() -> bool:
            return not any(_probe(vk) for vk in (VK_CTRL, VK_ALT, VK_SHIFT, VK_LWIN, VK_RWIN))

        self.guard = guard = MenuGuard(focus_state, _tap_alt, _mods_idle)
        guard_timer = 0

        def proc(n_code, w_param, l_param):
            nonlocal guard_timer
            try:
                if n_code == 0:
                    k = ctypes.cast(l_param, ctypes.POINTER(KBD)).contents
                    if k.dwExtraInfo != EXTRA_MARKER:
                        is_down = w_param in (self.WM_KEYDOWN, self.WM_SYSKEYDOWN)
                        res = self.feed(int(k.vkCode), is_down, _probe)
                        if res.mask:
                            self._send_mask()
                        if guard.pending and not guard_timer:
                            guard_timer = user32.SetTimer(None, 0, 40, None)
                        if res.swallow:
                            return 1
            except Exception:
                pass
            return user32.CallNextHookEx(self._hook, n_code, w_param, l_param)

        self._proc = PROC(proc)
        self._tid = kernel32.GetCurrentThreadId()
        self._hook = user32.SetWindowsHookExW(13, self._proc, None, 0)
        if not self._hook:
            self.error = f"SetWindowsHookEx failed ({ctypes.get_last_error()})"
            self._ready.set()
            return
        self._ready.set()
        user32.SetTimer(None, 0, 250, None)       # WM_TIMER -> watchdog (stuck keys, max hold)
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0113:
                try:
                    if guard_timer and msg.wParam == guard_timer:
                        guard.step()
                        if not guard.pending:
                            user32.KillTimer(None, guard_timer)
                            guard_timer = 0
                    elif self.enabled or self.machine.capturing:
                        self.tick(_probe)
                except Exception:
                    pass
                continue
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None

    @staticmethod
    def _send_mask() -> None:
        from .inject import send_keys
        send_keys([(VK_MASK, True), (VK_MASK, False)])
