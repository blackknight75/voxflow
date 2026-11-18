"""Live preview engine: the words recognised so far, while the user is still speaking.

There is no streaming model, so a background thread re-transcribes a bounded window of the growing recording every
few hundred milliseconds and stabilises the result with LocalAgreement-2 (as in whisper_streaming by Machacek et al.):
a word is *committed* once two consecutive hypotheses agree on it, the rest of the newest hypothesis is *tentative*.

    Previewer(recorder, transcriber, on_partial)    on_partial(gen, committed, tentative) runs on the preview thread
    Previewer(..., on_words=f)                      f(gen, new_committed, tentative, now_s): the words committed by
                                                    each pass with their times (tentative None = a tick without a
                                                    pass, only time moved); feeds voxflow.phrases
    .start(gen) / .stop()                           one session per utterance; both return at once
    .close()                                        stop and join (shutdown)
    .available                                      False once a governor or an error switched previews off

The preview is display-only: it never touches the target app, the clipboard, the history or the log (no text is
logged, not even at DEBUG). It never queues behind or in front of the final transcription: the transcriber is only
try-locked, and a governor keeps the cost of a preview call small and backs off on a slow PC.
"""
from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np

from .asr import PreviewUnavailable

log = logging.getLogger("voxflow.stream")

SAMPLE_RATE = 16000
MIN_NEW_S = 0.25            # less new audio than this since the last call: wait for more
SILENCE_PEAK = 0.003        # new audio quieter than this is silence
MIN_FIRST_S = 0.7           # no call before this much audio beyond the pre-roll: 0.3-0.4 s of noise alone made the
                            # model write "Yeah." (measured; nothing from 0.5 s on)
HOLD_CAP_S = 2.0            # calls that return no words are spaced further apart (x2, x4, ...) up to this long ...
RISE_FACTOR = 4.0           # ... until the new audio is this many times louder than what the empty call heard
COST_BUDGET_S = 0.25        # a call that costs more shrinks the window
GIVE_UP_S = 0.4             # a call that costs more with the window at its floor ...
GIVE_UP_COUNT = 2           # ... this many times in a row switches the preview off for the session
TRIM_FLOOR_S = 4.0
CLEAR_S = 0.45             # a window that starts this long after the last committed word cannot contain it
REHEARD_S = 0.3             # a word this much after the last committed one can still be that word (timestamp drift)
DUTY = 3.0                  # the next call starts at least DUTY x its cost after the last one (<= 33 % of a core)
_GPU_BACKENDS = ("cuda", "webgpu")
_DEFAULT_TRIM = {True: 15.0, False: 8.0}       # by "runs on a GPU"
_DEFAULT_BUCKET = {True: 1.0, False: 0.0}


def norm(word: str) -> str:
    """Compare words without case or punctuation: Parakeet re-punctuates as more context arrives."""
    return re.sub(r"[^\w']+", "", word.lower())


def words_with_times(tokens, stamps) -> list[tuple[str, float]]:
    """Join sub-word tokens (a leading space starts a word) into (word, start time in s)."""
    out: list[list] = []
    for tok, ts in zip(tokens, stamps):
        if tok.startswith(" ") or not out:
            out.append([tok.strip(), float(ts)])
        else:
            out[-1][0] += tok
    return [(w, t) for w, t in out if w]


class LocalAgreement:
    """Commit the longest prefix on which the last two hypotheses agree (after what is already committed)."""

    def __init__(self) -> None:
        self.committed: list[tuple[str, float]] = []     # (word, absolute start time)
        self.prev: list[tuple[str, float]] = []

    def new_part(self, hyp: list[tuple[str, float]], since: float | None = None) -> list[tuple[str, float]]:
        """The words of a full hypothesis that come after the committed text. Timestamps move by a frame or two
        between passes, so a time cut alone repeats the boundary word: like whisper_streaming, words near the cut
        whose n-gram (n <= 5) equals the end of the committed text are dropped too. ``since``: where the audio of
        this hypothesis starts; when the last committed word lies before it, nothing committed can be heard again
        and a word that repeats it ("very very", "no no") is a new one."""
        if not self.committed:
            return hyp
        last_t = self.committed[-1][1]
        hyp = [(w, t) for w, t in hyp if t > last_t - 1.0]          # keep a margin around the cut
        tail = [norm(w) for w, _ in self.committed[-5:]]
        heard_again = since is None or last_t >= since - 0.05
        for n in range(min(5, len(hyp), len(tail)) if heard_again else 0, 0, -1):
            for skip in range(0, min(3, len(hyp) - n) + 1):           # the n-gram may start a few words in
                # it is the committed text heard again only if it ENDS about where the committed text ends: a word
                # that starts clearly later is new, even when it repeats the last committed one
                if hyp[skip + n - 1][1] <= last_t + REHEARD_S and [
                        norm(w) for w, _ in hyp[skip:skip + n]] == tail[-n:]:
                    return hyp[skip + n:]
        return [(w, t) for w, t in hyp if t > last_t + 0.08]

    def update(self, hyp: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """``hyp``: words of the newest hypothesis that start after the committed ones. Returns the tentative tail."""
        n = 0
        while n < len(hyp) and n < len(self.prev) and norm(hyp[n][0]) == norm(self.prev[n][0]):
            n += 1
        self.committed += hyp[:n]
        self.prev = hyp[n:]
        return self.prev


class Previewer:
    """Live preview sessions over a growing recording.

    ``recorder`` needs ``recorded_samples`` and ``snapshot(start)`` (``voxflow.audio.Recorder``); ``transcriber``
    needs ``transcribe_timed(audio, bucket_s=, blocking=)`` and ``backend`` (``voxflow.asr.Transcriber``);
    ``gain``: optional callable applied to each window (whisper mode). ``trim_s`` / ``bucket_s`` default by back-end:
    a GPU keeps a 15 s window and reuses 1 s input lengths, the CPU a 8 s window and exact lengths."""

    def __init__(self, recorder, transcriber, on_partial, *, interval_s: float = 0.4, trim_s: float | None = None,
                 bucket_s: float | None = None, gain=None, clock=time.perf_counter, on_words=None):
        self._recorder, self._transcriber, self._on_partial = recorder, transcriber, on_partial
        self._on_words = on_words
        self.interval_s = float(interval_s)
        self._trim_arg, self._bucket_arg, self._gain, self._clock = trim_s, bucket_s, gain, clock
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._active = False
        self._available = True
        self._session = 0                  # bumps on every start(): a result of an older one is dropped
        self._gen = 0
        self._trim_s: float | None = None  # current window length limit (the governor shrinks it)
        self._slow = 0                     # consecutive calls over GIVE_UP_S with the window at its floor
        self._begin_state()

    # ------------------------------------------------------------------ state of one utterance
    def _begin_state(self) -> None:
        self._la = LocalAgreement()
        self._win_start = 0                # first sample of the window (offset into the utterance)
        self._last_n = 0                   # samples of the utterance seen by the last call
        self._silent_prev = False
        self._start_n = 0                  # samples already there when the session began (the pre-roll)
        self._empty = 0                    # consecutive calls that returned no words
        self._hold_until = 0.0             # no model call before this time unless the sound rises
        self._ref_peak = 0.0               # peak of the audio the last empty call heard
        self._shown = ("", "")
        self._next_at = 0.0

    @property
    def available(self) -> bool:
        """False once the governor or an error has switched previews off for this app session."""
        return self._available

    def _gpu(self) -> bool:
        return getattr(self._transcriber, "backend", "") in _GPU_BACKENDS

    def _bucket(self) -> float:
        return float(self._bucket_arg) if self._bucket_arg is not None else _DEFAULT_BUCKET[self._gpu()]

    # ------------------------------------------------------------------ public control
    def start(self, gen: int) -> None:
        """Begin the preview of utterance ``gen``. Returns at once; a no-op when previews are off."""
        with self._cv:
            if not self._begin(gen):
                return
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="voxflow-preview", daemon=True)
                self._thread.start()
            self._cv.notify_all()

    def _begin(self, gen: int) -> bool:
        """Reset the per-utterance state and open the session (the caller holds the lock and wakes the thread)."""
        if self._closed or not self._available:
            return False
        self._session += 1
        self._gen = gen
        self._begin_state()
        self._last_n = self._start_n = int(self._recorder.recorded_samples)     # the pre-roll is not "new" audio
        self._trim_s = float(self._trim_arg) if self._trim_arg is not None else _DEFAULT_TRIM[self._gpu()]
        self._next_at = self._clock() + self.interval_s
        self._active = True
        return True

    def stop(self) -> None:
        """End the session. Never blocks: a call that is still running is dropped when it returns."""
        with self._cv:
            self._active = False
            self._begin_state()                     # the words of the utterance are not kept in memory
            self._cv.notify_all()

    def close(self) -> None:
        """Stop and end the thread (shutdown); waits at most half a second for it."""
        with self._cv:
            self._closed = True
            self._active = False
            self._cv.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(0.5)

    # ------------------------------------------------------------------ the thread
    def _run(self) -> None:
        while True:
            with self._cv:
                if self._closed:
                    return
                if not self._active:
                    self._cv.wait()
                    continue
                delay = self._next_at - self._clock()
                if delay > 0:
                    self._cv.wait(delay)
                    continue
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - a preview must never take anything else down
                self._disable(f"preview failed ({type(exc).__name__})")

    def _disable(self, why: str) -> None:
        """Switch the preview off for the app session. A caption that is on screen is cleared (once), so it does
        not freeze there until the key is released."""
        with self._cv:
            was, midway, gen, shown = self._available, self._active, self._gen, self._shown
            self._available = self._active = False
            if midway:
                self._shown = ("", "")
            self._cv.notify_all()
        if was:
            log.info("live preview is off for this session: %s", why)
        if midway and shown != ("", ""):
            self._deliver(gen, "", "")

    def _deliver(self, gen: int, committed: str, tentative: str) -> None:
        try:
            self._on_partial(gen, committed, tentative)
        except Exception as exc:  # noqa: BLE001 - the receiver's problem must not stop the preview
            log.warning("preview receiver failed (%s)", type(exc).__name__)

    def _deliver_words(self, gen: int, new: list, tentative: list | None, now_s: float, gap: bool = False) -> None:
        try:
            self._on_words(gen, new, tentative, now_s, gap=gap)
        except Exception as exc:  # noqa: BLE001 - the receiver's problem must not stop the preview
            log.warning("preview words receiver failed (%s)", type(exc).__name__)

    def _tick(self) -> None:
        """One preview tick. A tick that ended without a model pass (too little audio, silence, back-off, busy model)
        tells ``on_words`` that time has passed, so a pause is seen even when the model is not called."""
        idle = self._pass()
        if idle is not None and self._on_words is not None:
            session, gen, n, gap = idle
            with self._cv:
                live = self._active and self._session == session
            if live:
                self._deliver_words(gen, [], None, n / SAMPLE_RATE, gap)

    def _pass(self):
        """One preview pass: snapshot the window, transcribe it (never waiting), stabilise, emit, adapt. The state
        of the utterance is only read and written under the lock, after a check that it is still the same session;
        the receivers are called after the lock is released. Returns ``(session, gen, samples, gap)`` when the tick ended
        without a model pass (``samples``: the audio actually examined so far, never audio nobody looked at), else
        None. ``gap``: this tick or pass moved the window start past audio that was never transcribed (the two forced
        jumps: calls spaced out, and the window floor in ``_trim``); words in that audio are never heard."""
        with self._cv:
            if not self._active:
                return None
            session, gen, last_n = self._session, self._gen, self._last_n
            win_start, trim_s = self._win_start, self._trim_s
            t_call = self._clock()
            self._next_at = t_call + self.interval_s
        n = int(self._recorder.recorded_samples)
        if n - last_n < MIN_NEW_S * SAMPLE_RATE or n - self._start_n < MIN_FIRST_S * SAMPLE_RATE:
            return session, gen, last_n, False               # that audio was not looked at: time does not move
        stored, jumped = win_start, False
        if n - win_start > (trim_s + self.interval_s) * SAMPLE_RATE:       # calls were spaced out: still bounded
            win_start = n - int(trim_s * SAMPLE_RATE)
        window = self._recorder.snapshot(win_start)
        new = max(0, window.size - max(0, last_n - win_start))
        peak = float(np.max(np.abs(window[-new:]))) if new else 0.0
        silent = peak < SILENCE_PEAK
        with self._cv:
            if not self._active or self._session != session:
                return None
            if silent and self._silent_prev:
                self._last_n = win_start + window.size       # nothing to say: do not look at it again
                return session, gen, n, False
            risen = peak >= RISE_FACTOR * max(self._ref_peak, SILENCE_PEAK)
            if t_call < self._hold_until and not risen:
                return session, gen, last_n, False           # backing off: the audio stays "new" for the next look
            if risen:
                self._empty = 0
            self._silent_prev = silent
            if win_start > stored:
                jumped = True                                # audio before it is never transcribed: a gap
                self._win_start = win_start
                self._la.prev = []                           # a cut through the tentative words: start afresh
        if self._gain is not None:
            window = self._gain(window)
        try:
            words = self._transcriber.transcribe_timed(window, bucket_s=self._bucket(), blocking=False)
        except PreviewUnavailable as exc:
            self._disable(str(exc) or "the speech model cannot do previews")
            return None
        if words is None:                                      # the model was busy (a final pass): try again later
            return session, gen, last_n, jumped                # the window was not examined: time does not move
        cost = self._clock() - t_call
        with self._cv:
            if not self._active or self._session != session:
                return None                                     # released, cancelled or replaced meanwhile
            end = win_start + window.size
            self._last_n = end
            self._empty = 0 if words else self._empty + 1
            self._ref_peak = peak
            self._hold_until = t_call + min(HOLD_CAP_S, self.interval_s * 2 ** self._empty) if self._empty else 0.0
            off = win_start / SAMPLE_RATE
            k0 = len(self._la.committed)
            tentative = self._la.update(self._la.new_part([(w, t + off) for w, t in words], off))
            gained = self._la.committed[k0:]
            shown = (" ".join(w for w, _ in self._la.committed), " ".join(w for w, _ in tentative))
            changed, self._shown = shown != self._shown, shown
            forced = self._trim(end, tentative)
            give_up = self._govern(t_call, cost)
        if changed:
            self._deliver(gen, shown[0], shown[1])
        if self._on_words is not None:
            self._deliver_words(gen, gained, list(tentative), end / SAMPLE_RATE, jumped or forced)
        if give_up:
            self._disable("the speech model is too slow on this PC")
        return None

    def _trim(self, end: int, tentative: list[tuple[str, float]]) -> bool:
        """Keep the window short: start it at the first uncommitted word, or just after the last committed one; and
        never let it be longer than ``trim_s`` whatever was recognised (no words at all, a first word that never
        settles): then the oldest words that are still tentative are given up. True when that floor forced the window
        start past audio that may hold words nobody committed (a gap)."""
        if (end - self._win_start) / SAMPLE_RATE <= self._trim_s:
            return False
        last = self._la.committed[-1][1] if self._la.committed else None
        if tentative:
            start = tentative[0][1] - 0.2
            if last is not None and start < last + CLEAR_S:
                start = last - 0.1       # not right behind the last committed word: it is either clearly in or clearly out
            self._win_start = max(self._win_start, int(start * SAMPLE_RATE))
        elif last is not None:
            self._win_start = max(self._win_start, int(min(end / SAMPLE_RATE, last + CLEAR_S + 0.05) * SAMPLE_RATE))
        floor = end - int(self._trim_s * SAMPLE_RATE)
        if self._win_start < floor:
            self._win_start = floor
            self._la.prev = []                                  # a cut through them: the next pass starts afresh
            return True
        return False

    def _govern(self, t_call: float, cost: float) -> bool:
        """Bound the duty cycle; shrink the window when a call is slow. True: the PC cannot keep up, give up."""
        self._next_at = t_call + max(self.interval_s, DUTY * cost)
        at_floor = self._trim_s <= TRIM_FLOOR_S
        if cost > COST_BUDGET_S:
            self._trim_s = max(TRIM_FLOOR_S, self._trim_s * 0.75)
        self._slow = self._slow + 1 if at_floor and cost > GIVE_UP_S else 0
        return self._slow >= GIVE_UP_COUNT
