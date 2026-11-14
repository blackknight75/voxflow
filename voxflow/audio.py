"""Microphone capture: always-open 16 kHz mono stream with a pre-roll ring buffer.

Also: device discovery by *name* (PortAudio indices change across reboots and hot-plug), stream health
(``Recorder.alive`` / ``reopen``) for unplugged devices, and gain normalisation for whispered speech.
"""
from __future__ import annotations

import logging
import sys
import threading
import time

import numpy as np

log = logging.getLogger("voxflow.audio")

SAMPLE_RATE = 16000
PREROLL_S = 0.3
_RING_S = 2.0
_STALL_S = 1.5              # no audio callback for this long = the device went away

MIC_HELP = ("Check that a microphone is connected and that Windows Settings › Privacy & security › "
            "Microphone lets desktop apps use it.")


# --------------------------------------------------------------------------- devices
def _sd():
    import sounddevice as sd
    return sd


def rescan_devices() -> None:
    """Make PortAudio re-enumerate devices (it caches the list). Only call while no stream is open."""
    sd = _sd()
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:  # noqa: BLE001
        log.warning("audio device rescan failed: %s", exc)


def _input_devices() -> tuple[list[tuple[int, str]], dict[str, str]]:
    """(index, name) of the inputs on the default host API, plus {short name: full name}.

    The default Windows host API (MME) truncates names to 31 characters; the full name is taken from the
    same device as listed by another host API."""
    sd = _sd()
    devices = list(sd.query_devices())
    default_in = sd.default.device[0]
    api = devices[default_in]["hostapi"] if isinstance(default_in, int) and 0 <= default_in < len(devices) else None
    mine: list[tuple[int, str]] = []
    longer: list[str] = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = str(d["name"]).strip()
        if api is None or d["hostapi"] == api:
            mine.append((i, name))
        else:
            longer.append(name)
    full: dict[str, str] = {}
    for _i, name in mine:
        best = name
        if len(name) >= 30:
            cands = sorted((n for n in longer if n.startswith(name) and len(n) > len(name)), key=len)
            if cands:
                best = cands[0]
        full[name] = best
    return mine, full


def list_input_devices() -> list[tuple[int, str]]:
    """``[(portaudio index, display name)]`` of usable microphones (unique names, default host API)."""
    try:
        mine, full = _input_devices()
    except Exception as exc:  # noqa: BLE001 - no audio subsystem: an empty list, not a crash
        log.warning("could not list microphones: %s", exc)
        return []
    out, seen = [], set()
    for idx, name in mine:
        label = full[name]
        if label not in seen and "Microsoft Sound Mapper" not in label:
            seen.add(label)
            out.append((idx, label))
    return out


def default_input_name() -> str:
    """Display name of the system default microphone ('' if there is none)."""
    try:
        sd = _sd()
        idx = sd.default.device[0]
        if not isinstance(idx, int) or idx < 0:
            return ""
        name = str(sd.query_devices(idx)["name"]).strip()
        return _input_devices()[1].get(name, name)
    except Exception:  # noqa: BLE001
        return ""


def match_device(spec, devices: list[tuple[int, str]]):
    """Resolve a stored microphone setting against ``devices``: index, or None when it is not present.

    ``spec`` may be a display name, a (possibly truncated) driver name or a legacy integer index."""
    if spec is None:
        return None
    if isinstance(spec, int) and not isinstance(spec, bool):
        return spec
    want = str(spec).strip().lower()
    if not want:
        return None
    for idx, name in devices:
        if name.lower() == want:
            return idx
    for idx, name in devices:
        low = name.lower()
        if low.startswith(want) or want.startswith(low):
            return idx
    for idx, name in devices:
        if want in name.lower():
            return idx
    return None


def describe_mic_error(exc: BaseException | None) -> str:
    """A message for the user; the technical text goes to the log."""
    text = str(exc or "").lower()
    if "no microphone" in text or "no default input" in text or "-1" in text and "device" in text:
        return "No microphone found. " + MIC_HELP
    if "invalid device" in text or "not found" in text or "no input device matching" in text:
        return "The selected microphone is not available. " + MIC_HELP
    return "The microphone could not be opened. " + MIC_HELP


# --------------------------------------------------------------------------- gain
def normalize_gain(audio, target_rms: float = 0.07, max_gain: float = 40.0) -> np.ndarray:
    """Bring quiet / whispered speech up to a level the recogniser handles well.

    The speech level is the 90th percentile of 20 ms frame RMS, so pauses do not count. The signal is never
    attenuated, digital silence and steady hiss are left alone, and peaks are soft-limited instead of clipped."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = 320
    n = a.size // frame
    if n < 5:
        return a
    rms = np.sqrt((a[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
    level = float(np.percentile(rms, 90))
    floor = float(np.percentile(rms, 10))
    if level < 1e-5 or level < floor * 1.6:      # silence, or noise without any speech-like dynamics
        return a
    gain = min(float(max_gain), float(target_rms) / level)
    if gain <= 1.0:
        return a
    out = a * np.float32(gain)
    knee = 0.8
    over = np.abs(out) > knee
    if over.any():
        mag = np.abs(out[over])
        out[over] = np.sign(out[over]) * (knee + (1.0 - knee) * np.tanh((mag - knee) / (1.0 - knee)))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, device: int | str | None = None, max_seconds: float = 300.0):
        self._device = device
        self._max_samples = int(max_seconds * SAMPLE_RATE)
        self._rec_samples = 0       # samples buffered for the current recording (incl. pre-roll)
        self._preroll_samples = 0   # pre-roll samples included in the current recording
        self._last_speech_s = 0.0
        self._hit_cap = False
        self._lock = threading.Lock()
        self._stream = None
        self._recording = False
        self._chunks: list[np.ndarray] = []
        self._ring = np.zeros(int(_RING_S * SAMPLE_RATE), dtype=np.float32)
        self._level = 0.0
        self._last_cb = 0.0         # monotonic time of the last audio callback
        self._opened = 0.0
        self.level_boost = 1.0      # meter gain (raised in whisper mode so quiet speech still moves it)
        self.error: BaseException | None = None    # why the last attempt to open the stream failed
        self.using_fallback = False                # the chosen device was missing: system default in use
        try:
            self._ensure_stream()
        except Exception as exc:  # noqa: BLE001 - retried on start()
            self._stream = None
            self.error = exc

    def _callback(self, indata, frames, time_info, status):
        x = np.array(indata[:, 0], dtype=np.float32, copy=True)
        n = x.size
        if not n:
            return
        rms = float(np.sqrt(np.mean(x * x)))
        with self._lock:
            self._last_cb = time.monotonic()
            ring = self._ring
            if n >= ring.size:
                ring[:] = x[-ring.size:]
            else:
                ring[:-n] = ring[n:]
                ring[-n:] = x
            if self._recording:
                room = self._max_samples - self._rec_samples
                if room > 0:
                    self._chunks.append(x[:room] if n > room else x)
                    self._rec_samples += min(n, room)
                if self._rec_samples >= self._max_samples:
                    self._hit_cap = True
            lvl = min(1.0, rms * 8.0 * self.level_boost)
            self._level = lvl if lvl > self._level else self._level * 0.85 + lvl * 0.15

    # ---- stream management
    def _open(self, device):
        sd = _sd()
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device,
                                blocksize=512, callback=self._callback)
        stream.start()
        return stream

    def _close_stream(self) -> None:
        with self._lock:
            s, self._stream = self._stream, None
        if s is not None:
            for op in (s.stop, s.close):
                try:
                    op()
                except Exception:  # noqa: BLE001
                    pass

    def _ensure_stream(self) -> None:
        if self._stream is not None and self._stream.active:
            return
        self._close_stream()                 # a dead stream (device unplugged) still holds a handle
        with self._lock:
            self._ring[:] = 0.0              # never hand out pre-roll from before the outage
            self._level = 0.0
        index = self._device
        self.using_fallback = False
        if isinstance(index, str):
            index = match_device(index, list_input_devices())
            if index is None:
                self.using_fallback = True   # the chosen microphone is gone: use the system default
        try:
            stream = self._open(index)
        except Exception as exc:
            if index is None:
                self.error = exc
                raise
            log.warning("microphone %r failed to open (%s); trying the system default", self._device, exc)
            try:
                stream = self._open(None)
            except Exception as exc2:
                self.error = exc2
                raise
            self.using_fallback = True
        self.error = None
        now = time.monotonic()
        with self._lock:
            self._stream = stream
            self._opened = self._last_cb = now

    @property
    def alive(self) -> bool:
        """False once the input stream stopped delivering audio (device unplugged, driver reset)."""
        s = self._stream
        if s is None:
            return False
        try:
            if not s.active:
                return False
        except Exception:  # noqa: BLE001
            return False
        return time.monotonic() - self._last_cb < _STALL_S

    def reopen(self, rescan: bool = True) -> None:
        """Drop the stream and open it again (after a device change). Raises if no microphone works."""
        self._close_stream()
        if rescan:
            rescan_devices()
        self._ensure_stream()

    def close(self) -> None:
        with self._lock:
            self._recording = False
        self._close_stream()

    def set_device(self, device: int | str | None) -> None:
        """Switch microphone. A recording in progress keeps what it has and continues on the new device."""
        self._close_stream()
        self._device = device
        self._ensure_stream()

    @property
    def device(self):
        return self._device

    # ---- recording
    def start(self) -> None:
        self._ensure_stream()
        with self._lock:
            pre = int(PREROLL_S * SAMPLE_RATE)
            self._chunks = [self._ring[-pre:].copy()]
            self._preroll_samples = pre
            self._rec_samples = pre
            self._hit_cap = False
            self._recording = True

    def stop(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            chunks, self._chunks = self._chunks, []
        if not chunks:
            self._last_speech_s = 0.0
            return np.zeros(0, dtype=np.float32)
        out = np.clip(np.concatenate(chunks), -1.0, 1.0).astype(np.float32)
        self._last_speech_s = max(0, out.size - self._preroll_samples) / SAMPLE_RATE
        return out

    @property
    def hit_cap(self) -> bool:
        """True once the current/last recording reached max_seconds; reset by start()."""
        with self._lock:
            return self._hit_cap

    @property
    def last_speech_seconds(self) -> float:
        """Duration of the last stop() result, excluding the PREROLL_S pre-roll."""
        return self._last_speech_s

    def level(self) -> float:
        return float(self._level)

    def recent_samples(self, n: int = 1024) -> np.ndarray:
        with self._lock:
            return self._ring[-min(n, self._ring.size):].copy()

    @property
    def is_recording(self) -> bool:
        return self._recording


def input_device_count() -> int:
    """Cheap device-change probe: the number of wave-in devices Windows reports right now (-1 if unknown)."""
    if sys.platform != "win32":
        return -1
    try:
        import ctypes
        return int(ctypes.windll.winmm.waveInGetNumDevs())
    except Exception:  # noqa: BLE001
        return -1
