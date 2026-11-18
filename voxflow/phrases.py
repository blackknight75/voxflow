"""Phrase streaming engine: cut the committed words of a live dictation into phrases, and find where to cut the audio.

    PhraseSplitter     pure: committed words with start times -> Phrase(words, kind, cut window [lo, hi])
    PhraseStream       glue on the preview thread: splitter + the quietest point of the cut window in the audio
    quiet_point        the cut inside a window: the centre of its quietest frame
    finish_chunk       the text to paste for a cleaned phrase, given what was pasted before it in the session
    closing_period     the full stop to end a streamed sentence when nothing is left to say at release

The audio before ``cut_s`` holds exactly the words of the phrases emitted so far and the audio after it the next
words (a cut lies after the estimated end of the last word and before the start of the next one), so the final
pass over ``audio[cut_s:]`` neither loses nor repeats a word. Phrase text is never logged (plan/PHRASE_STREAMING_DESIGN.md
section 9): this module does no logging at all.
"""
from __future__ import annotations

import dataclasses
import re
import threading
from dataclasses import dataclass

import numpy as np

from .stream import norm

SAMPLE_RATE = 16000
PAUSE_S = 0.6           # silence (estimated) after a word that ends a phrase on a pause
LONG_PAUSE_S = 1.5      # a silence this long ends the phrase whatever its length, and confirms a trailing "."
MIN_WORDS = 3           # shortest phrase emitted (except after LONG_PAUSE_S)
MAX_WORDS = 25          # length cap, words
MAX_SPAN_S = 10.0       # length cap, audio seconds (first word start .. last word start)
CUT_MARGIN_S = 0.10     # the cut stays this far before the next word's start
MIN_GAP_S = 0.10        # a boundary needs at least this much room for a cut window
END_BASE_S = 0.25       # estimated word end: start + min(END_MAX_S, END_BASE_S + END_PER_CHAR_S * letters)
END_PER_CHAR_S = 0.06
END_MAX_S = 0.9
_CAP_SCAN = 8           # a length cap looks back at most this many words for a place to cut
_TENTATIVE_SLACK_S = 0.05

_ABBREVIATIONS = frozenset("mr mrs ms dr prof sr jr st vs cf fig approx inc ltd co".split())   # never end a sentence
# these end a sentence only before a capitalised word: "a.m.", "etc.", "bzw.", "13.", "z.B.", "U.S.", "e.g."
_SOFT_ABBREVIATIONS = frozenset("a.m p.m etc bzw usw ca nr evtl".split())
_CLOSERS = "\"')]}"
_NO_SPACE_BEFORE = (" ", "\n", "\r", ",", ".", ";", ":", "!", "?", ")", "]", "}", "%")


@dataclass(frozen=True)
class Phrase:
    seq: int                                  # 1, 2, 3 ... within the session
    words: tuple[tuple[str, float], ...]      # (word as committed, start time s); never empty
    text: str                                 # " ".join(word for word, _ in words): raw ASR, never cleaned
    kind: str                                 # "sentence" | "pause" | "cap"
    ends_sentence: bool                       # the last word carries sentence-final punctuation
    cut_lo: float                             # earliest allowed cut (s, utterance time)
    cut_hi: float                             # latest allowed cut
    cut_s: float                              # (lo + hi) / 2 from the splitter; PhraseStream moves it to the quiet point


def end_est(word: str, start: float) -> float:
    """No word ends exist, only starts: a late estimate (the safe side for a cut)."""
    return start + min(END_MAX_S, END_BASE_S + END_PER_CHAR_S * len(norm(word)))


def _terminal(word: str, later: str | None) -> bool:
    """Does ``word`` end a sentence? ``later``: the next committed word, None when there is none yet. A word followed
    by a lower-case word does not (a "." the model wrote at the end of one hypothesis proves nothing); abbreviations,
    ordinals and dotted tokens end one only before a capitalised word."""
    w = word.rstrip(_CLOSERS)
    if w.endswith("..."):
        return False                                   # a trailing pause, not an end
    if not w.endswith((".", "!", "?", "…")):
        return False
    if later is not None and later[:1].islower():
        return False
    if w.endswith("."):
        stem = w[:-1].lower()
        if re.fullmatch(r"\w", stem) or stem in _ABBREVIATIONS:
            return False                               # an initial, a title
        if stem in _SOFT_ABBREVIATIONS or re.fullmatch(r"\d+", stem) or re.search(r"[^\W\d_]\.\w|\d\.\d", stem + "."):
            return later is not None and later[:1].isupper()
    return True


def _unstop(word: str, nxt: str | None) -> str:
    """A "." that looked like a sentence end but is followed by a lower-case word was only the end of one hypothesis:
    leave it out of the text that goes to clean-up."""
    w = word.rstrip(_CLOSERS)
    if nxt is not None and nxt[:1].islower() and w.endswith(".") and _terminal(word, "X"):
        return w[:-1] + word[len(w):]
    return word


class PhraseSplitter:
    """Committed words (append-only, with start times) -> phrases with a cut window. Deterministic, no clock, no I/O."""

    def __init__(self, *, pause_s=PAUSE_S, long_pause_s=LONG_PAUSE_S,
                 min_words=MIN_WORDS, max_words=MAX_WORDS, max_span_s=MAX_SPAN_S):
        self.pause_s, self.long_pause_s = pause_s, long_pause_s
        self.min_words, self.max_words, self.max_span_s = min_words, max_words, max_span_s
        self.rejected_at: set[float] = set()   # start times of words whose boundary has "no room" right now
        self.reset()

    def reset(self) -> None:
        self._words: list[tuple[str, float]] = []
        self._tent: float | None = None
        self._seq = 0
        self._last_cut = 0.0
        self.rejected_at.clear()
        self.rejected_final = 0     # boundaries skipped for good (the phrase went on past them), for the bench

    @property
    def pending(self) -> tuple[tuple[str, float], ...]:
        """Committed words not yet in a phrase."""
        return tuple(self._words)

    def feed(self, new_committed, tentative_start: float | None, now_s: float) -> list[Phrase]:
        """``new_committed``: the words committed since the last call. ``tentative_start``: start of the first
        tentative word of the newest hypothesis, or None. ``now_s``: audio time covered by the latest pass."""
        self._words.extend((str(w), float(t)) for w, t in new_committed)
        self._tent = tentative_start
        return self._drain(now_s)

    def tick(self, now_s: float) -> list[Phrase]:
        """Time passed without a model pass: the last tentative start still holds."""
        return self._drain(now_s)

    # ------------------------------------------------------------------ boundaries
    def _drain(self, now_s: float) -> list[Phrase]:
        out: list[Phrase] = []
        while self._words:
            found = self._boundary(now_s)
            if found is None:
                break
            i, kind, lo, hi = found
            words = tuple(self._words[:i + 1])
            later = self._words[i + 1][0] if i + 1 < len(self._words) else None
            nxt = [w for w, _ in self._words[1:i + 2]]
            text = " ".join(_unstop(w, n) for (w, _), n in zip(words, nxt + [None]))
            self._seq += 1
            dead = {t for t in self.rejected_at if t < words[-1][1]}      # boundaries that stayed rejected: now inside
            self.rejected_final += len(dead)
            self.rejected_at -= dead
            self.rejected_at.discard(words[-1][1])
            cut = (lo + hi) / 2
            self._last_cut = cut
            out.append(Phrase(self._seq, words, text, kind,
                              _terminal(words[-1][0], later), lo, hi, cut))
            del self._words[:i + 1]
        return out

    def _window(self, i: int, now_s: float):
        """The cut window after word ``i`` (lo, hi) or None when there is no room (noted in ``rejected_at``)."""
        w = self._words
        lo = max(end_est(*w[i]), self._last_cut)
        hi = w[i + 1][1] - CUT_MARGIN_S if i + 1 < len(w) else now_s - _TENTATIVE_SLACK_S
        if hi - lo < MIN_GAP_S:
            self.rejected_at.add(w[i][1])
            return None
        return lo, hi

    def _gap(self, i: int, now_s: float) -> float | None:
        """Silence after word ``i``: to the next committed word, else to now when nothing is tentative before now."""
        w = self._words
        if i + 1 < len(w):
            return w[i + 1][1] - end_est(*w[i])
        if self._tent is None or self._tent >= now_s - _TENTATIVE_SLACK_S:
            return now_s - end_est(*w[i])
        return None

    def _boundary(self, now_s: float):
        w = self._words
        for i in range(len(w)):
            gap = self._gap(i, now_s)
            if gap is None:
                break                                   # a tentative word may start inside the silence: unknown
            later = w[i + 1][0] if i + 1 < len(w) else None
            long_pause = gap >= self.long_pause_s
            if not (i + 1 >= self.min_words or long_pause):
                continue
            term = _terminal(w[i][0], later)
            kind = None
            if later is not None:
                # a sentence end is confirmed by a later word; a pause after any other word may cut too
                kind = "sentence" if term else ("pause" if gap >= self.pause_s else None)
            elif term:
                kind = "sentence" if long_pause else None      # an unconfirmed "." needs a long silence
            elif gap >= self.pause_s:
                kind = "pause"
            if kind is None:
                continue
            win = self._window(i, now_s)
            if win is not None:
                return (i, kind, *win)
        n = len(w)
        if n >= self.max_words or (w[-1][1] - w[0][1] >= self.max_span_s):
            return self._cap(now_s)
        return None

    def _cap(self, now_s: float):
        """The phrase got too long: cut after the last comma-ended word, else after the last word with room."""
        w = self._words
        cands = []
        for i in range(len(w) - 2, max(len(w) - 2 - _CAP_SCAN, -1), -1):
            if i + 1 < self.min_words:
                break
            win = self._window(i, now_s)
            if win is not None:
                cands.append((i, win))
        if not cands:
            return None
        pick = next((c for c in cands if w[c[0]][0].rstrip(_CLOSERS).endswith(",")), cands[0])
        return (pick[0], "cap", *pick[1])


# ---------------------------------------------------------------------- the cut inside a window
def quiet_point(samples, t0: float, lo: float, hi: float, frame_s: float = 0.04) -> float:
    """The cut time inside ``[lo, hi]``: the centre of the quietest ``frame_s`` frame of ``samples`` (``samples[0]``
    is at time ``t0``). Ties go to the frame nearest the middle of the window. An empty or too short slice gives the
    middle of the window."""
    mid = (lo + hi) / 2
    a = max(0, int(round((lo - t0) * SAMPLE_RATE)))
    b = max(a, int(round((hi - t0) * SAMPLE_RATE)))
    seg = np.asarray(samples, dtype=np.float32)[a:b]
    fn = max(1, int(round(frame_s * SAMPLE_RATE)))
    k = seg.size // fn
    if k < 1:
        return mid
    rms = np.sqrt(np.mean(seg[:k * fn].reshape(k, fn) ** 2, axis=1))
    centres = t0 + (a + (np.arange(k) + 0.5) * fn) / SAMPLE_RATE
    best = np.flatnonzero(rms <= rms.min() + 1e-7)
    c = float(centres[best[np.argmin(np.abs(centres[best] - mid))]])
    return min(hi, max(lo, c))


class PhraseStream:
    """The part that lives on the preview thread: ``feed`` is the Previewer's ``on_words``. ``begin``, ``end``, ``feed``
    and ``heard`` are mutually exclusive (one lock), so a new session never receives words of the old one."""

    def __init__(self, recorder, on_phrase, *, splitter=None, frame_s=0.04):
        self._recorder, self._on_phrase, self._frame_s = recorder, on_phrase, frame_s
        self.splitter = splitter if splitter is not None else PhraseSplitter()
        self._lock = threading.Lock()
        self._gen: int | None = None
        self._cut = 0.0
        self._committed: list[tuple[str, float]] = []
        self._tentative: list[tuple[str, float]] = []
        self._covered = 0.0

    def begin(self, gen: int) -> None:
        """New session (UI thread)."""
        with self._lock:
            self.splitter.reset()
            self._cut = self._covered = 0.0
            self._committed, self._tentative = [], []
            self._gen = gen

    def end(self) -> None:
        """Forget the session: later ``feed`` calls are ignored. ``heard()`` stays valid until the next ``begin``."""
        with self._lock:
            self._gen = None

    def heard(self) -> tuple[tuple[tuple[str, float], ...], float]:
        """``(words, covered_s)``: every word the preview heard in this session in time order with its start time
        (the committed ones, then the tentative words of the newest pass), and the audio time up to which a model
        pass has looked (0.0 before the first)."""
        with self._lock:
            return tuple(self._committed) + tuple(self._tentative), self._covered

    def feed(self, gen, new_committed, tentative, now_s) -> None:
        with self._lock:
            if self._gen is None or gen != self._gen:
                return
            self._committed.extend((str(w), float(t)) for w, t in new_committed)
            if tentative is None:
                found = self.splitter.tick(now_s)
            else:
                self._tentative = [(str(w), float(t)) for w, t in tentative]
                self._covered = max(self._covered, float(now_s))
                found = self.splitter.feed(new_committed, tentative[0][1] if tentative else None, now_s)
            for ph in found:
                cut = max(self._cut, self._refine(ph))
                self._cut = cut
                self._on_phrase(gen, dataclasses.replace(ph, cut_s=cut))

    def _refine(self, ph: Phrase) -> float:
        try:
            samples = self._recorder.snapshot(int(ph.cut_lo * SAMPLE_RATE))
        except Exception:  # noqa: BLE001 - the session may have ended under us: the middle will do
            return ph.cut_s
        return quiet_point(samples, int(ph.cut_lo * SAMPLE_RATE) / SAMPLE_RATE, ph.cut_lo, ph.cut_hi, self._frame_s)


# ---------------------------------------------------------------------- text at the edges of a chunk
def tail_hint(words, cut_s: float) -> str | None:
    """The first heard word that starts at or after the cut (the first word of the tail), or None."""
    return next((w for w, t in words if t >= cut_s - 0.05), None)


def finish_chunk(prev: str, cleaned: str, raw: str, kind: str, ends_sentence: bool, hint: str | None = None) -> str:
    """The text to paste for a cleaned phrase (kind "sentence" | "pause" | "cap") or tail ("tail"), given everything
    pasted before it in this session (``prev``, '' for the first chunk)."""
    if not cleaned:
        return ""
    # 1. cleanup adds a full stop or question mark to a phrase that is not a sentence: take it off again
    if kind in ("pause", "cap") and cleaned.endswith((".", "?")) and not raw.rstrip().endswith((".", "!", "?", "…")):
        cleaned = cleaned[:-1]
    # 2. cleanup capitalises the first letter of every phrase (and the model, of every utterance): undo that when the
    #    text before is mid-sentence and the evidence says the word is not capitalised
    tail = prev.rstrip()
    prev_open = bool(tail) and not prev.endswith(("\n", "\r")) and tail[-1] not in ".!?…:"
    if prev_open and cleaned[:1].isupper():
        first = cleaned.split()[0]
        if kind == "tail":
            # the tail is transcribed as a fresh utterance, so its raw casing proves nothing: the word the preview
            # heard at that place does (``hint``); without it only words that are capitalised for another reason stay
            if hint is not None and norm(hint) == norm(first):
                lower = hint[:1].islower()
            else:
                core = re.sub(r"[^\w'’]", "", first)
                caps = len(re.sub(r"[\W\d_]", "", core)) > 1 and core.isupper()
                lower = not (core == "I" or core.startswith(("I'", "I’")) or caps or any(c.isdigit() for c in core))
        else:
            heard = next((w for w in raw.split() if norm(w) == norm(first)), None)
            lower = heard is not None and heard[:1].islower() and not (first.isupper() and len(first) > 1)
        if lower:
            cleaned = cleaned[:1].lower() + cleaned[1:]
    # 3. the space that joins it to the text before (a leading one: the field never ends with a dangling space)
    if prev and not prev.endswith((" ", "\n", "\r")) and not cleaned.startswith(_NO_SPACE_BEFORE):
        cleaned = " " + cleaned
    return cleaned


def closing_period(prev: str, last_kind: str) -> str:
    """"." when the streamed text ends mid-sentence on a pause or cap cut and nothing is left to say; else ""."""
    if last_kind not in ("pause", "cap") or not prev[-1:].isalnum():
        return ""
    last = re.split(r"(?<=[.!?…])\s+|\n+", prev)[-1]
    return "." if len(last.split()) >= 3 else ""
