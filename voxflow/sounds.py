"""Sound cues synthesised locally, and muting of other apps' audio while dictating.

``SoundPlayer``  - short start / stop / error / command tones generated with numpy (no audio assets).
``MediaMuter``   - mutes every other application's audio session for the length of a dictation and always
                   gives it back: on stop, on cancel, at exit, in the crash handler, and (after a hard kill)
                   at the next start, from a marker file.

Both take an injectable backend so tests never touch the real speakers or the Windows mixer.
"""
from __future__ import annotations

import ctypes
import io
import json
import logging
import os
import queue
import sys
import threading
import uuid
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("voxflow.sounds")

SOUND_NAMES = ("start", "stop", "error", "command")
_RATE = 44100


# =========================================================================== synthesis
def _note(freq: float, dur: float, harmonic: float = 0.18) -> np.ndarray:
    t = np.arange(int(dur * _RATE)) / _RATE
    wave_ = np.sin(2 * np.pi * freq * t) + harmonic * np.sin(2 * np.pi * 2 * freq * t)
    attack = np.minimum(1.0, t / 0.004)
    decay = np.exp(-t / (dur * 0.38))
    release = np.minimum(1.0, (dur - t) / 0.006)
    return (wave_ * attack * decay * release).astype(np.float32)


def _sequence(notes: list[tuple[float, float]], gap: float = 0.012, harmonic: float = 0.18) -> np.ndarray:
    silence = np.zeros(int(gap * _RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    for freq, dur in notes:
        parts += [_note(freq, dur, harmonic), silence]
    return np.concatenate(parts)


def synth(name: str) -> np.ndarray:
    """The cue ``name`` as mono float32 samples in [-1, 1] at 44.1 kHz."""
    if name == "start":
        sig = _sequence([(587.33, 0.055), (880.0, 0.085)])              # rising fifth: "go"
    elif name == "stop":
        sig = _sequence([(880.0, 0.055), (587.33, 0.085)])              # falling fifth: "done"
    elif name == "command":
        sig = _sequence([(659.25, 0.045), (880.0, 0.045), (1174.66, 0.085)])
    elif name == "error":
        sig = _sequence([(207.65, 0.085), (196.0, 0.12)], gap=0.03, harmonic=0.05)
    else:
        raise ValueError(f"unknown sound '{name}'")
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak).astype(np.float32)


def to_wav(samples: np.ndarray, volume: float) -> bytes:
    """16-bit mono WAV bytes of ``samples`` scaled by a perceptual volume in 0..1."""
    amp = 0.55 * max(0.0, min(1.0, float(volume))) ** 1.6
    pcm = np.clip(samples * amp, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_RATE)
        w.writeframes(data)
    return buf.getvalue()


def _winsound_backend(wav: bytes) -> None:
    import winsound
    winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_NODEFAULT)


class SoundPlayer:
    """Plays the cues on a small worker thread so callers never wait for audio.

    ``backend(wav_bytes)`` plays one sound and blocks until it has finished; the default uses ``winsound``.
    """

    def __init__(self, enabled: bool = True, volume: float = 0.5, backend=None):
        self.enabled = enabled
        self._volume = 0.5
        self._backend = backend if backend is not None else (_winsound_backend if sys.platform == "win32" else None)
        self._cache: dict[str, bytes] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.volume = volume

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        with self._lock:
            if value != self._volume:
                self._cache.clear()
            self._volume = value

    def _wav(self, name: str) -> bytes:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = to_wav(synth(name), self._volume)
            return self._cache[name]

    def play(self, name: str, force: bool = False) -> bool:
        """Queue cue ``name``. ``force`` plays it even when cues are switched off (settings preview).

        Returns True if it was queued. Never raises: a missing audio device just means no sound."""
        if name not in SOUND_NAMES or self._backend is None:
            return False
        if not (self.enabled or force) or self._volume <= 0.0:
            return False
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="voxflow-sounds", daemon=True)
                self._thread.start()
        try:
            self._queue.put_nowait(name)
        except queue.Full:
            return False              # cues piling up are worthless a moment later
        return True

    def _run(self) -> None:
        while True:
            name = self._queue.get()
            if name is None:
                return
            try:
                self._backend(self._wav(name))
            except Exception as exc:  # noqa: BLE001
                log.debug("sound '%s' could not be played: %s", name, exc)

    def close(self) -> None:
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            while True:                        # pending cues are worthless now; make room for the sentinel
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(None)
            thread.join(1.5)


# =========================================================================== muting other apps
class _Guid(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16),
                ("d4", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, text: str) -> "_Guid":
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


_CLSID_ENUMERATOR = "BCDE0395-E52F-467C-8E3D-C4579291692E"
_IID_ENUMERATOR = "A95664D2-9614-4F35-A746-DE8DB63617E6"
_IID_SESSION_MANAGER2 = "77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"
_IID_SESSION_CONTROL2 = "BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D"
_IID_SIMPLE_VOLUME = "87CE5498-68D6-44E5-9215-6DA47EF883D8"
_ACTIVE = 1


class _Com:
    """A raw COM interface pointer with vtable calls and deterministic Release."""

    def __init__(self, ptr: ctypes.c_void_p):
        self.ptr = ptr

    def call(self, index: int, *args, argtypes=()) -> int:
        vtable = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[index])
        return int(fn(self.ptr, *args))

    def out(self, index: int, *args, argtypes=()) -> "_Com | None":
        """Call a method whose last parameter is an interface out-pointer."""
        result = ctypes.c_void_p()
        hr = self.call(index, *args, ctypes.byref(result),
                       argtypes=(*argtypes, ctypes.POINTER(ctypes.c_void_p)))
        return _Com(result) if hr >= 0 and result.value else None

    def query(self, iid: str) -> "_Com | None":
        return self.out(0, ctypes.byref(_Guid.of(iid)), argtypes=(ctypes.POINTER(_Guid),))

    def __enter__(self) -> "_Com":
        return self

    def __exit__(self, *exc) -> None:
        if self.ptr and self.ptr.value:
            self.call(2)
            self.ptr = ctypes.c_void_p()


class CoreAudioBackend:
    """Mutes / unmutes other processes' audio sessions through the Windows Core Audio COM API (ctypes)."""

    def _sessions(self):
        """Yield ``(volume, pid, session_id, state)`` for every audio session on every active output."""
        ole32 = ctypes.WinDLL("ole32")
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        raw = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(ctypes.byref(_Guid.of(_CLSID_ENUMERATOR)), None, 23,
                                    ctypes.byref(_Guid.of(_IID_ENUMERATOR)), ctypes.byref(raw))
        if hr < 0 or not raw.value:
            raise OSError(f"audio device enumerator unavailable (0x{hr & 0xFFFFFFFF:08X})")
        u32, i32 = ctypes.c_uint, ctypes.c_int
        with _Com(raw) as enumerator:
            devices = enumerator.out(3, 0, _ACTIVE, argtypes=(i32, ctypes.c_uint32))     # eRender, active
            if devices is None:
                return
            with devices:
                n_dev = u32()
                devices.call(3, ctypes.byref(n_dev), argtypes=(ctypes.POINTER(u32),))
                for i in range(n_dev.value):
                    device = devices.out(4, i, argtypes=(u32,))
                    if device is None:
                        continue
                    with device:
                        manager = device.out(3, ctypes.byref(_Guid.of(_IID_SESSION_MANAGER2)), 23, None,
                                             argtypes=(ctypes.POINTER(_Guid), ctypes.c_uint32, ctypes.c_void_p))
                        if manager is None:
                            continue
                        with manager:
                            sessions = manager.out(5)
                            if sessions is None:
                                continue
                            with sessions:
                                n_ses = i32()
                                sessions.call(3, ctypes.byref(n_ses), argtypes=(ctypes.POINTER(i32),))
                                for j in range(n_ses.value):
                                    control = sessions.out(4, j, argtypes=(i32,))
                                    if control is None:
                                        continue
                                    with control:
                                        yield from self._describe(control, ole32)

    @staticmethod
    def _describe(control: _Com, ole32):
        control2 = control.query(_IID_SESSION_CONTROL2)
        volume = control.query(_IID_SIMPLE_VOLUME)
        if control2 is None or volume is None:
            for c in (control2, volume):
                if c is not None:
                    c.__exit__()
            return
        with control2, volume:
            state = ctypes.c_int()
            control.call(3, ctypes.byref(state), argtypes=(ctypes.POINTER(ctypes.c_int),))
            pid = ctypes.c_uint32()
            control2.call(14, ctypes.byref(pid), argtypes=(ctypes.POINTER(ctypes.c_uint32),))
            text = ctypes.c_void_p()
            session_id = ""
            if control2.call(12, ctypes.byref(text), argtypes=(ctypes.POINTER(ctypes.c_void_p),)) >= 0 and text.value:
                session_id = ctypes.wstring_at(text.value)
                ole32.CoTaskMemFree(text)
            yield volume, int(pid.value), session_id, int(state.value)

    @staticmethod
    def _with_com(work):
        ole32 = ctypes.WinDLL("ole32")
        hr = ole32.CoInitializeEx(None, 0)          # multithreaded; S_FALSE / RPC_E_CHANGED_MODE are fine
        try:
            return work()
        finally:
            if hr in (0, 1):
                ole32.CoUninitialize()

    def mute_others(self) -> list[str]:
        """Mute every active, currently unmuted session of other processes. Returns their session ids."""
        me = os.getpid()

        def work() -> list[str]:
            muted: list[str] = []
            for volume, pid, session_id, state in self._sessions():
                if pid in (0, me) or state != _ACTIVE or not session_id:
                    continue
                already = ctypes.c_int()
                volume.call(6, ctypes.byref(already), argtypes=(ctypes.POINTER(ctypes.c_int),))
                if already.value:
                    continue                         # the user muted it: not ours to unmute later
                if volume.call(5, 1, None, argtypes=(ctypes.c_int, ctypes.c_void_p)) >= 0:
                    muted.append(session_id)
            return muted

        return self._with_com(work)

    def unmute(self, session_ids: list[str]) -> None:
        wanted = set(session_ids)

        def work() -> None:
            for volume, _pid, session_id, _state in self._sessions():
                if session_id in wanted:
                    volume.call(5, 0, None, argtypes=(ctypes.c_int, ctypes.c_void_p))

        if wanted:
            self._with_com(work)


class MediaMuter:
    """Mute other apps while dictating; restore reliably.

    ``backend`` needs ``mute_others() -> list[str]`` and ``unmute(ids)``. Work runs on a private thread (in
    order) so the UI never waits on the audio service; ``threaded=False`` runs inline (tests).
    A marker file in ``state_dir`` records what is muted, so ``recover()`` can undo it after a hard kill.
    """

    def __init__(self, state_dir: str | Path, backend=None, threaded: bool = True):
        self._marker = Path(state_dir) / "muted_sessions.json"
        self._backend = backend if backend is not None else (CoreAudioBackend() if sys.platform == "win32" else None)
        self._threaded = threaded
        self._closing = False
        self._muted: list[str] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._muted)

    # ---- public (non-blocking when threaded)
    def mute(self) -> None:
        if not self._closing:
            self._submit(self._do_mute)

    def unmute(self) -> None:
        self._submit(self._do_unmute)

    def recover(self) -> None:
        """Undo a mute left behind by a previous run that was killed mid-dictation."""
        self._submit(self._do_recover)

    def restore_now(self) -> None:
        """Final, synchronous unmute for exit paths and the crash handler (any thread, repeatedly).

        A mute that is still being applied on the worker undoes itself when it completes."""
        self._closing = True
        self._do_unmute()

    def close(self) -> None:
        """Stop the worker (letting a mute in flight finish first), then give every app its sound back."""
        self._closing = True
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            self._queue.put(None)
            thread.join(5.0)
        self._do_unmute()

    # ---- plumbing
    def _submit(self, job) -> None:
        if self._backend is None:
            return
        if not self._threaded:
            job()
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="voxflow-media-mute", daemon=True)
                self._thread.start()
        self._queue.put(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            job()

    def _write_marker(self, ids: list[str]) -> None:
        try:
            if ids:
                self._marker.parent.mkdir(parents=True, exist_ok=True)
                self._marker.write_text(json.dumps(ids), encoding="utf-8")
            elif self._marker.exists():
                self._marker.unlink()
        except OSError as exc:
            log.debug("mute marker not updated: %s", exc)

    def _do_mute(self) -> None:
        if self._backend is None:
            return
        try:
            ids = self._backend.mute_others()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not mute other apps: %s", exc)
            return
        with self._lock:
            self._muted = list(dict.fromkeys(self._muted + list(ids)))
            current = list(self._muted)
        self._write_marker(current)
        if self._closing:                      # shut down while the mute was being applied: undo it now
            self._do_unmute()

    def _do_unmute(self) -> None:
        with self._lock:
            ids, self._muted = self._muted, []
        if not ids or self._backend is None:
            return
        try:
            self._backend.unmute(ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not unmute other apps: %s", exc)
            with self._lock:
                self._muted = ids + self._muted      # keep them for the next attempt
            return
        self._write_marker([])

    def _do_recover(self) -> None:
        try:
            ids = json.loads(self._marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        ids = [i for i in ids if isinstance(i, str)] if isinstance(ids, list) else []
        if ids and self._backend is not None:
            try:
                self._backend.unmute(ids)
                log.info("restored audio of %d app(s) muted by a previous session", len(ids))
            except Exception as exc:  # noqa: BLE001
                log.warning("could not restore audio muted by a previous session: %s", exc)
                return
        self._write_marker([])
