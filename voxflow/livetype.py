"""Live typing into the target app: the pure logic, not wired into the app yet (design: section 7).

While the key is held, ``LiveTyper`` types the words the preview engine has *committed* (they are never taken
back) into the focused text field; on release it brings the field to exactly the final, cleaned-up text with one
small edit. Nothing here touches the keyboard, a window or Qt: it drives a ``keys`` object with two methods,

    keys.type_text(text: str)      type ``text`` at the cursor
    keys.backspace(count: int)     press Backspace ``count`` times

so the whole behaviour is testable against a simulated field (``tests/test_livetype.py``).

    plan_edit(typed, target) -> (backspaces, text)    the smallest edit that turns ``typed`` into ``target``
    clusters(text)           -> list[str]             what one Backspace removes at a time
    LiveTyper(keys, ...)     .update(committed) / .window_changed() / .cancel() / .finish(final)

Units. A Backspace removes one *cluster*, not one code point: a letter with its combining marks, an emoji with
its modifiers / ZWJ sequence / variation selector, a flag (two regional indicators), a surrogate pair written as
two characters, and "\\r\\n". ``plan_edit`` counts and cuts only at cluster boundaries, so it never leaves half a
character and never counts one key press as two. Apps that delete by code point instead are a known risk
(see the report that came with this module).
"""
from __future__ import annotations

import unicodedata

ZWJ = "‍"
TYPED, FALLBACK, LEFT = "typed", "fallback", "left"     # what ``LiveTyper.finish`` did (see there)
MAX_BACKSPACES = 40       # more than this many characters to delete: leave the field alone and tell the user


def _extends(ch: str) -> bool:
    """Does ``ch`` attach to the character before it (combining mark, ZWJ, variation selector, skin tone, tag)?"""
    code = ord(ch)
    return (unicodedata.category(ch) in ("Mn", "Me", "Mc") or ch == ZWJ or 0x1F3FB <= code <= 0x1F3FF
            or 0xE0020 <= code <= 0xE007F)


def _regional(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _joins(prev: str, ch: str) -> bool:
    """Is ``ch`` part of the cluster that ends with ``prev``'s last character?"""
    last = prev[-1]
    return (_extends(ch) or last == ZWJ or (last == "\r" and ch == "\n")
            or ("\ud800" <= last <= "\udbff" and "\udc00" <= ch <= "\udfff")
            or (len(prev) == 1 and _regional(last) and _regional(ch)))


def clusters(text: str) -> list[str]:
    """``text`` cut into the pieces one Backspace deletes (``"".join(clusters(t)) == t``)."""
    out: list[str] = []
    for ch in text:
        if out and _joins(out[-1], ch):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def plan_edit(typed: str, target: str) -> tuple[int, str]:
    """``(backspaces, text)``: press Backspace ``backspaces`` times, then type ``text``, and a field that holds
    ``typed`` before the cursor holds ``target`` afterwards. The longest common prefix (by whole clusters) is
    kept."""
    have, want = clusters(typed), clusters(target)
    keep = 0
    for a, b in zip(have, want):
        if a != b:
            break
        keep += 1
    return len(have) - keep, "".join(want[keep:])


class LiveTyper:
    """Types committed words as they come and reconciles the field to the final text at the end.

    ``mode`` is the dictation mode: ``"command"`` never types (the answer replaces a selection, there is
    nothing to preview into). ``max_backspaces`` caps how much may be deleted at the end (or on ``cancel``).

    ``state`` is ``"typing"`` (the normal one), ``"off"`` (command mode), ``"stopped"`` (typing was given up on
    purpose: the window changed, the committed text was rewritten, or it could not start safely; ``reason``
    says which) or ``"failed"`` (the keyboard raised: what stands in the field is unknown).
    ``typed`` is exactly what this object has typed and left in the field.
    """

    def __init__(self, keys, *, mode: str = "dictate", max_backspaces: int = MAX_BACKSPACES):
        self.keys = keys
        self.max_backspaces = max_backspaces
        self.typed = ""
        self.state = "off" if mode == "command" else "typing"
        self.reason = "command mode" if mode == "command" else ""

    # ------------------------------------------------------------------ while the key is held
    def update(self, committed: str) -> bool:
        """Type what ``committed`` (all committed words so far) has beyond what was typed. True when keys were
        sent. Text that is not an extension of what stands in the field stops the typing for good."""
        if self.state != "typing":
            return False
        committed = committed.strip()
        if committed == self.typed:
            return False
        if not committed.startswith(self.typed):
            self._stop("committed text was rewritten")
            return False
        delta = committed[len(self.typed):]
        if not self.typed and _extends(delta[0]):
            self._stop("text starts with a combining mark")      # it would join the character left of the cursor
            return False
        return self._send(0, delta, committed)

    def window_changed(self) -> None:
        """The target window is not the one that got the first keys: type nothing more."""
        if self.state == "typing":
            self._stop("the target window changed")

    def cancel(self) -> bool:
        """The dictation was cancelled: take back what was typed, when it is little and still in the right
        window. True when the field is clean again."""
        if self.state != "typing":
            return not self.typed and self.state != "failed"
        count = len(clusters(self.typed))
        if count > self.max_backspaces:
            self._stop("too much to take back")
            return False
        return self._send(count, "", "")

    # ------------------------------------------------------------------ on release
    def finish(self, final: str) -> str:
        """Bring the field to ``final``. Returns

        ``TYPED``     the field now holds ``final`` (nothing else is to be pasted);
        ``FALLBACK``  nothing usable was typed, or typing stopped: paste / copy ``final`` the usual way;
        ``LEFT``      the field was left as it is (too much to delete, or the keyboard failed): tell the user.
        """
        state, self.state = self.state, "done"
        if state == "failed":
            return LEFT
        if state != "typing" or not self.typed:
            return FALLBACK
        backspaces, text = plan_edit(self.typed, final)
        if backspaces > self.max_backspaces:
            self.reason = "too much to delete"
            return LEFT
        self.state = "typing"
        if not self._send(backspaces, text, final):
            self.state = "done"
            return LEFT
        self.state = "done"
        return TYPED

    # ------------------------------------------------------------------ internals
    def _stop(self, reason: str) -> None:
        self.state, self.reason = "stopped", reason

    def _send(self, backspaces: int, text: str, now: str) -> bool:
        """Press the keys; ``now`` is what the field holds afterwards. A raising keyboard ends in ``failed``."""
        try:
            if backspaces:
                self.keys.backspace(backspaces)
            if text:
                self.keys.type_text(text)
        except Exception as exc:  # noqa: BLE001 - whatever the keyboard layer raises: stop, never retry blind
            self.state, self.reason = "failed", type(exc).__name__
            return False
        self.typed = now
        return True


__all__ = ["LiveTyper", "plan_edit", "clusters", "TYPED", "FALLBACK", "LEFT", "MAX_BACKSPACES"]
