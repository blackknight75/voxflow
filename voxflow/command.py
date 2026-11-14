"""Command mode: speak an instruction, the local LLM rewrites the selected text (or writes new text).

Flow (driven by the controller): the user holds the command hotkey and speaks ("make this more formal",
"translate to Spanish", "turn into bullets"); on release the selection is read through Ctrl+C (clipboard
saved and restored), the instruction is transcribed, the model produces the result and it is pasted over
the selection. With nothing selected the instruction is treated as a request to generate text.

Every failure raises ``CommandError`` carrying a message for the user; the selection is never touched then.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("voxflow.command")

_SYSTEM = (
    "You are the rewriting engine of a dictation app. You get an INSTRUCTION the user spoke and, usually, the "
    "TEXT they selected in another application.\n"
    "- If TEXT is present, apply the instruction to it and output the complete new version of the text.\n"
    "- If there is no TEXT, write what the instruction asks for.\n"
    "- Output ONLY the resulting text, ready to be pasted: no introduction, no explanation, no quotation marks "
    "around it, no markdown code fences.\n"
    "- Keep the language of the TEXT unless the instruction asks for a translation.\n"
    "- Keep line breaks and list structure unless the instruction asks to change them.\n"
    "- Do not add facts that are not in the TEXT unless the instruction asks you to."
)
_TONE_NOTES = {
    "formal": "Unless told otherwise, write in a professional tone with full punctuation.",
    "casual": "Unless told otherwise, write in a relaxed, friendly tone.",
    "very_casual": "Unless told otherwise, write the way people text friends: relaxed and brief.",
}
_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|okay|ok|absolutely|here(?:'s| is| are)|below is)\b[^\n]{0,80}[:!.]\s*\n+", re.I)
_REFUSAL_RE = re.compile(
    r"^\s*(?:i['’]?m sorry,? but|i am sorry,? but|i cannot\b|i can['’]?t (?:help|assist|do|comply)|as an ai\b|"
    r"i['’]?m unable to)", re.I)
_FENCE_RE = re.compile(r"^```[\w-]*\n(.*?)\n?```$", re.S)
_OUTPUT_RESERVE = 96          # tokens kept free for the answer on top of the computed budget
_PROMPT_OVERHEAD = 260        # system prompt + chat template + labels, in tokens (measured, rounded up)


class CommandError(Exception):
    """Command mode could not produce a result. ``str(exc)`` is a complete sentence for the user."""


def clean_instruction(spoken: str, language: str | None = None) -> str:
    """The spoken instruction without filler words (kept verbatim otherwise)."""
    from . import langs
    from .cleanup import resolve_language, rule_clean

    spoken = spoken or ""
    lang = resolve_language(spoken, language)
    if lang == "en":
        text = rule_clean(spoken, formatting=False, terminal=False).strip()
    else:
        text = langs.rule_clean(spoken, langs.get(lang), formatting=False, terminal=False).strip()
    return text or spoken.strip()


def build_messages(instruction: str, selection: str, *, tone: str | None = None, terms=None,
                   language: str | None = None) -> list[dict]:
    system = _SYSTEM
    if tone in _TONE_NOTES:
        system += "\n- " + _TONE_NOTES[tone]
    picked = [" ".join(t.split()) for t in (terms or []) if isinstance(t, str) and t.strip()][:40]
    if picked:
        system += "\n- Spell these names and terms exactly like this when they occur: " + ", ".join(picked) + "."
    if language and language not in ("auto", "en") and not selection.strip():
        system += f"\n- The user dictates in the language with ISO code '{language}'; answer in that language."
    if selection.strip():
        user = f"INSTRUCTION: {instruction}\n\nTEXT:\n{selection}"
    else:
        user = f"INSTRUCTION: {instruction}\n\n(There is no TEXT: write what the instruction asks for.)"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tidy_output(output: str, selection: str = "") -> str:
    """Strip what a chat model adds around the answer: preambles, code fences, wrapping quotes, labels."""
    out = (output or "").strip()
    fenced = _FENCE_RE.match(out)
    if fenced:
        out = fenced.group(1).strip()
    out = _PREAMBLE_RE.sub("", out, count=1).strip()
    out = re.sub(r"^(?:TEXT|RESULT|OUTPUT|REWRITTEN TEXT)\s*:\s*\n?", "", out, count=1, flags=re.I).strip()
    out = re.sub(r"[ \t]+\n", "\n", out)               # markdown hard breaks ("line  \n") are noise once pasted
    for left, right in (('"', '"'), ("“", "”"), ("'", "'")):
        wrapped = len(out) >= 2 and out.startswith(left) and out.endswith(right)
        if wrapped and not selection.strip().startswith(left) and left not in out[1:-1]:
            out = out[1:-1].strip()
            break
    return out


class CommandMode:
    """Selection grab + guarded LLM rewrite.

    ``cleaner``  needs ``has_llm``, ``generate(messages, max_tokens, timeout_s)``, ``count_tokens(text)`` and
                 ``n_ctx`` (``voxflow.cleanup.Cleaner`` or a fake).
    ``injector`` needs ``copy_selection()`` (``voxflow.inject.Injector`` or a fake).
    """

    def __init__(self, cleaner, injector, timeout_s: float = 20.0):
        self.cleaner = cleaner
        self.injector = injector
        self.timeout_s = float(timeout_s)

    @property
    def available(self) -> bool:
        return bool(getattr(self.cleaner, "has_llm", False))

    def grab_selection(self) -> str:
        """Selected text of the focused window ('' when nothing is selected).

        Raises CommandError when the clipboard cannot be used safely."""
        try:
            text = self.injector.copy_selection()
        except Exception as exc:  # noqa: BLE001
            log.warning("selection grab failed: %s", exc)
            text = None
        if text is None:
            raise CommandError("Couldn't read the selection because the clipboard is busy. Nothing was changed.")
        return text

    def _budget(self, instruction: str, selection: str) -> int:
        count = self.cleaner.count_tokens
        n_ctx = int(getattr(self.cleaner, "n_ctx", 4096))
        sel = count(selection) if selection.strip() else 0
        used = _PROMPT_OVERHEAD + count(instruction) + sel
        wanted = max(256, int(sel * 1.6) + 128) if sel else 700
        room = n_ctx - used - _OUTPUT_RESERVE
        if room < min(wanted, max(96, int(sel * 1.1))):
            words = max(50, int((n_ctx - _PROMPT_OVERHEAD - _OUTPUT_RESERVE) / 2.4 / 1.35))
            raise CommandError(f"That selection is too long for command mode (about {words:,} words at most). "
                               "Nothing was changed.")
        return min(wanted, room)

    def run(self, instruction: str, selection: str = "", *, tone: str | None = None, terms=None,
            language: str | None = None) -> str:
        """Return the text to paste. Raises CommandError (with a human message) on any problem."""
        instruction = clean_instruction(instruction, language)
        if not instruction:
            raise CommandError("Didn't catch the instruction. Hold the command shortcut and say what to do.")
        if not self.available:
            raise CommandError("Command mode needs the language model, which isn't loaded. Nothing was changed.")
        max_tokens = self._budget(instruction, selection)
        messages = build_messages(instruction, selection, tone=tone, terms=terms, language=language)
        try:
            raw = self.cleaner.generate(messages, max_tokens=max_tokens, timeout_s=self.timeout_s)
        except TimeoutError:
            raise CommandError("Command mode took too long and was stopped. Nothing was changed.") from None
        except Exception as exc:  # noqa: BLE001
            log.warning("command generation failed: %s", exc)
            raise CommandError("The language model failed on that request. Nothing was changed.") from None
        out = tidy_output(raw, selection)
        if not out:
            raise CommandError("The language model returned nothing. Nothing was changed.")
        if _REFUSAL_RE.match(out) and not _REFUSAL_RE.match(selection or ""):
            raise CommandError("The language model declined that request. Nothing was changed.")
        if selection.strip() and len(out) > max(600, len(selection) * 8):
            raise CommandError("The result looked wrong (far longer than the selection), so nothing was changed.")
        return out
