"""Phrase streaming engine (stub: final public names and signatures, trivial bodies)."""
from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 16000
PAUSE_S = 0.6
SENT_PAUSE_S = 0.35
LONG_PAUSE_S = 1.5
MIN_WORDS = 3
MAX_WORDS = 25
MAX_SPAN_S = 10.0
CUT_MARGIN_S = 0.10
MIN_GAP_S = 0.10


@dataclass(frozen=True)
class Phrase:
    seq: int
    words: tuple
    text: str
    kind: str
    ends_sentence: bool
    cut_lo: float
    cut_hi: float
    cut_s: float


class PhraseSplitter:
    def __init__(self, *, pause_s=PAUSE_S, sent_pause_s=SENT_PAUSE_S, long_pause_s=LONG_PAUSE_S,
                 min_words=MIN_WORDS, max_words=MAX_WORDS, max_span_s=MAX_SPAN_S):
        self.pending = ()

    def reset(self) -> None:
        pass

    def feed(self, new_committed, tentative_start, now_s) -> list:
        return []

    def tick(self, now_s) -> list:
        return []


def quiet_point(samples, t0, lo, hi, frame_s=0.04) -> float:
    return (lo + hi) / 2


class PhraseStream:
    def __init__(self, recorder, on_phrase, *, splitter=None, frame_s=0.04):
        self._on_phrase = on_phrase

    def begin(self, gen: int) -> None:
        pass

    def end(self) -> None:
        pass

    def feed(self, gen, new_committed, tentative, now_s) -> None:
        pass


def finish_chunk(prev: str, cleaned: str, raw: str, kind: str, ends_sentence: bool) -> str:
    if not cleaned:
        return ""
    if prev and not prev.endswith((" ", "\n")) and not cleaned.startswith((" ", "\n", ",", ".", ";", ":", "!", "?", ")", "]", "}", "%")):
        return " " + cleaned
    return cleaned


def closing_period(prev: str, last_kind: str) -> str:
    return ""
