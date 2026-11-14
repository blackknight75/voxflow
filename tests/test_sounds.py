"""voxflow.sounds: cue synthesis, the SoundPlayer worker, MediaMuter restore guarantees, and the Core Audio
backend logic driven against fake COM objects. No real speaker or mixer is touched."""
from __future__ import annotations

import ctypes
import io
import json
import sys
import threading
import wave

import numpy as np
import pytest

from voxflow import sounds as snd
from voxflow.sounds import MediaMuter, SoundPlayer


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


def read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), frames


# =========================================================================== synthesis
@pytest.mark.parametrize("name", snd.SOUND_NAMES)
def test_cues_are_short_normalised_and_click_free(name):
    sig = snd.synth(name)
    assert sig.dtype == np.float32 and sig.ndim == 1
    assert 0.08 <= sig.size / 44100 <= 0.35, "a cue must be a blip, not a jingle"
    assert abs(float(np.max(np.abs(sig))) - 1.0) < 1e-5
    assert abs(float(sig[0])) < 0.02 and abs(float(sig[-1])) < 0.02, "must start and end at silence (no click)"
    assert np.isfinite(sig).all()


def test_cues_are_distinguishable_and_deterministic():
    sigs = {n: snd.synth(n) for n in snd.SOUND_NAMES}
    assert set(snd.SOUND_NAMES) == {"start", "stop", "error", "command"}
    for a in sigs:
        assert np.array_equal(sigs[a], snd.synth(a))
        for b in sigs:
            if a < b:
                n = min(sigs[a].size, sigs[b].size)
                assert not np.allclose(sigs[a][:n], sigs[b][:n], atol=0.05), (a, b)

    def dominant(sig):
        spectrum = np.abs(np.fft.rfft(sig))
        return float(np.fft.rfftfreq(sig.size, 1 / 44100)[int(np.argmax(spectrum))])

    half = sigs["start"].size // 2
    assert dominant(sigs["start"][:half // 1]) > 0
    assert dominant(sigs["start"][half:]) > dominant(sigs["start"][:half - 600]), "start rises"
    assert dominant(sigs["stop"][half:]) < dominant(sigs["stop"][:half - 600]), "stop falls"
    assert dominant(sigs["error"]) < 300, "the error cue is low"
    with pytest.raises(ValueError, match="unknown sound"):
        snd.synth("fanfare")


def test_to_wav_is_16_bit_mono_and_volume_scales_perceptually():
    sig = snd.synth("start")
    ch, width, rate, full = read_wav(snd.to_wav(sig, 1.0))
    assert (ch, width, rate) == (1, 2, 44100) and full.size == sig.size
    peaks = [float(np.max(np.abs(read_wav(snd.to_wav(sig, v))[3]))) for v in (0.0, 0.25, 0.5, 1.0, 7.0, -3.0)]
    assert peaks[0] == 0.0 and peaks[5] == 0.0
    assert 0 < peaks[1] < peaks[2] < peaks[3] <= 0.56, "never louder than a subtle cue"
    assert peaks[4] == peaks[3], "volume is clamped to 1"


# =========================================================================== SoundPlayer
class Speaker:
    """Blocking fake backend that records what was played."""

    def __init__(self):
        self.played: list[bytes] = []
        self.event = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.fail = False
        self.failed = threading.Event()

    def __call__(self, wav: bytes):
        self.gate.wait(5)
        if self.fail:
            self.failed.set()
            raise OSError("no audio device")
        self.played.append(wav)
        self.event.set()

    def wait(self, n=1, timeout=3.0):
        import time
        end = time.time() + timeout
        while len(self.played) < n and time.time() < end:
            time.sleep(0.005)
        return len(self.played) >= n


@pytest.fixture
def speaker():
    return Speaker()


def test_player_plays_on_a_worker_thread(speaker):
    p = SoundPlayer(backend=speaker)
    try:
        assert p.play("start") is True
        assert speaker.wait(1)
        assert read_wav(speaker.played[0])[3].size == snd.synth("start").size
        assert p._thread.daemon and p._thread.name == "voxflow-sounds"
        assert p._thread is not threading.current_thread()
    finally:
        p.close()
    assert p._thread is None


def test_player_respects_toggle_volume_and_force(speaker):
    p = SoundPlayer(enabled=False, volume=0.5, backend=speaker)
    try:
        assert p.play("start") is False
        assert p.play("start", force=True) is True, "the settings preview must play even when cues are off"
        assert speaker.wait(1)
        p.enabled = True
        p.volume = 0
        assert p.play("stop") is False and p.play("stop", force=True) is False, "volume 0 is silence"
        assert p.play("fanfare") is False and p.play("") is False
        p.volume = 9
        assert p.volume == 1.0
        p.volume = -2
        assert p.volume == 0.0
    finally:
        p.close()
    assert len(speaker.played) == 1


def test_volume_change_rebuilds_the_cached_wave(speaker):
    p = SoundPlayer(volume=0.2, backend=speaker)
    try:
        p.play("start")
        assert speaker.wait(1)
        p.play("start")
        assert speaker.wait(2)
        assert speaker.played[0] is speaker.played[1], "same volume: the WAV is cached"
        p.volume = 1.0
        p.play("start")
        assert speaker.wait(3)
        quiet, loud = (float(np.max(np.abs(read_wav(w)[3]))) for w in (speaker.played[0], speaker.played[2]))
        assert loud > quiet * 3
    finally:
        p.close()


def test_cues_never_pile_up_behind_a_stuck_audio_device(speaker):
    speaker.gate.clear()                                # the device blocks
    p = SoundPlayer(backend=speaker)
    try:
        results = [p.play("start") for _ in range(12)]
        assert results[0] is True and results.count(False) >= 6, "old cues are worthless a moment later"
    finally:
        speaker.gate.set()
        p.close()


def test_backend_failure_is_swallowed_and_the_worker_survives(speaker):
    speaker.fail = True
    p = SoundPlayer(backend=speaker)
    try:
        assert p.play("error") is True
        assert speaker.failed.wait(3)
        speaker.fail = False
        assert p.play("stop") is True
        assert speaker.wait(1), "the worker thread died after a backend error"
    finally:
        p.close()


def test_player_without_a_backend_is_silent(monkeypatch):
    monkeypatch.setattr(snd.sys, "platform", "linux")
    p = SoundPlayer()
    assert p.play("start") is False
    p.close()                                            # nothing to stop


def test_close_is_idempotent_and_the_player_restarts_on_demand(speaker):
    p = SoundPlayer(backend=speaker)
    p.play("start")
    assert speaker.wait(1)
    p.close()
    p.close()
    assert p.play("stop") is True and speaker.wait(2)
    p.close()


def test_close_with_a_full_queue_does_not_hang(speaker):
    speaker.gate.clear()
    p = SoundPlayer(backend=speaker)
    for _ in range(8):
        p.play("start")
    import time
    t0 = time.perf_counter()
    p.close()
    assert time.perf_counter() - t0 < 3.0
    speaker.gate.set()


def test_c53_close_with_a_full_queue_still_stops_the_worker_thread(speaker):
    speaker.gate.clear()
    p = SoundPlayer(backend=speaker)
    for _ in range(8):
        p.play("start")
    worker = p._thread
    p.close()                                            # the device is still stuck while we shut down
    speaker.gate.set()                                   # ... and recovers afterwards
    worker.join(3)
    assert not worker.is_alive(), "the stop sentinel was dropped because the queue was full: thread leak"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_default_backend_uses_winsound_from_memory_synchronously(monkeypatch):
    import winsound
    calls = []
    monkeypatch.setattr(winsound, "PlaySound", lambda data, flags: calls.append((data, flags)))
    snd._winsound_backend(b"RIFF....")
    assert calls == [(b"RIFF....", winsound.SND_MEMORY | winsound.SND_NODEFAULT)]
    assert not calls[0][1] & winsound.SND_ASYNC, "async playback of a temporary buffer is undefined behaviour"
    assert SoundPlayer()._backend is snd._winsound_backend


# =========================================================================== MediaMuter
class Mixer:
    """Fake audio-session backend. ``playing`` = sessions with sound; ``muted`` = sessions we muted."""

    def __init__(self, playing=("spotify", "chrome")):
        self.playing = list(playing)
        self.muted: set[str] = set()
        self.user_muted: set[str] = set()
        self.fail_mute = self.fail_unmute = False
        self.calls: list[str] = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def mute_others(self):
        self.entered.set()
        self.gate.wait(5)
        self.calls.append("mute")
        if self.fail_mute:
            raise OSError("audio service down")
        new = [s for s in self.playing if s not in self.muted and s not in self.user_muted]
        self.muted |= set(new)
        return new

    def unmute(self, ids):
        self.calls.append(f"unmute:{','.join(ids)}")
        if self.fail_unmute:
            raise OSError("audio service down")
        self.muted -= set(ids)


@pytest.fixture
def mixer():
    return Mixer()


def marker(tmp_path):
    return tmp_path / "muted_sessions.json"


def test_mute_then_unmute_restores_exactly_what_was_muted(tmp_path, mixer):
    mixer.user_muted = {"teams"}
    mixer.playing.append("teams")
    m = MediaMuter(tmp_path, backend=mixer, threaded=False)
    assert m.available and not m.active
    m.mute()
    assert mixer.muted == {"spotify", "chrome"} and m.active
    assert json.loads(marker(tmp_path).read_text(encoding="utf-8")) == ["spotify", "chrome"]
    m.unmute()
    assert mixer.muted == set() and not m.active and not marker(tmp_path).exists()
    assert mixer.calls == ["mute", "unmute:spotify,chrome"], "an app the user muted is never touched"
    m.unmute()
    assert mixer.calls == ["mute", "unmute:spotify,chrome"], "unmuting twice does nothing"


def test_repeated_mutes_accumulate_without_duplicates(tmp_path, mixer):
    m = MediaMuter(tmp_path, backend=mixer, threaded=False)
    m.mute()
    mixer.playing.append("vlc")
    m.mute()
    assert json.loads(marker(tmp_path).read_text(encoding="utf-8")) == ["spotify", "chrome", "vlc"]
    m.restore_now()
    assert mixer.muted == set() and mixer.calls[-1] == "unmute:spotify,chrome,vlc"


def test_mute_failure_is_harmless(tmp_path, mixer, caplog):
    mixer.fail_mute = True
    m = MediaMuter(tmp_path, backend=mixer, threaded=False)
    with caplog.at_level("WARNING"):
        m.mute()
    assert not m.active and not marker(tmp_path).exists() and "could not mute" in caplog.text
    m.unmute()
    assert mixer.calls == ["mute"]


def test_failed_unmute_is_retried_and_the_marker_is_kept(tmp_path, mixer, caplog):
    m = MediaMuter(tmp_path, backend=mixer, threaded=False)
    m.mute()
    mixer.fail_unmute = True
    with caplog.at_level("WARNING"):
        m.unmute()
    assert m.active and marker(tmp_path).exists() and "could not unmute" in caplog.text
    mixer.fail_unmute = False
    m.restore_now()
    assert mixer.muted == set() and not m.active and not marker(tmp_path).exists()


def test_recover_unmutes_what_a_killed_session_left_behind(tmp_path, mixer, caplog):
    marker(tmp_path).write_text(json.dumps(["spotify", 7, "chrome"]), encoding="utf-8")
    mixer.muted = {"spotify", "chrome"}
    m = MediaMuter(tmp_path, backend=mixer, threaded=False)
    with caplog.at_level("INFO"):
        m.recover()
    assert mixer.muted == set() and not marker(tmp_path).exists()
    assert "restored audio of 2 app(s)" in caplog.text
    m.recover()                                           # no marker: nothing to do
    assert mixer.calls == ["unmute:spotify,chrome"]


@pytest.mark.parametrize("content", ["{ broken", "{}", '"text"', "[]", "[1, 2]"])
def test_recover_tolerates_a_damaged_marker(tmp_path, mixer, content):
    marker(tmp_path).write_text(content, encoding="utf-8")
    MediaMuter(tmp_path, backend=mixer, threaded=False).recover()
    assert mixer.calls == []
    if content != "{ broken":
        assert not marker(tmp_path).exists()


def test_recover_keeps_the_marker_when_the_audio_service_is_down(tmp_path, mixer):
    marker(tmp_path).write_text('["spotify"]', encoding="utf-8")
    mixer.fail_unmute = True
    MediaMuter(tmp_path, backend=mixer, threaded=False).recover()
    assert marker(tmp_path).exists(), "must be retried at the next start"


def test_muter_without_a_backend_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(snd.sys, "platform", "linux")
    m = MediaMuter(tmp_path)
    assert not m.available
    m.mute()
    m.unmute()
    m.recover()
    m._do_mute()
    m.close()
    assert not m.active and not marker(tmp_path).exists()


def test_marker_write_failure_does_not_break_muting(tmp_path, mixer):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    m = MediaMuter(blocker / "sub", backend=mixer, threaded=False)
    m.mute()
    assert mixer.muted == {"spotify", "chrome"}
    m.unmute()
    assert mixer.muted == set()


def test_threaded_jobs_run_in_order_off_the_calling_thread(tmp_path, mixer):
    m = MediaMuter(tmp_path, backend=mixer)
    caller = threading.current_thread()
    seen = []
    real = mixer.mute_others
    mixer.mute_others = lambda: (seen.append(threading.current_thread()), real())[1]
    for _ in range(5):
        m.mute()
        m.unmute()
    m.close()
    assert mixer.calls == ["mute", "unmute:spotify,chrome"] * 5 and mixer.muted == set()
    assert seen and all(t is not caller and t.name == "voxflow-media-mute" and t.daemon for t in seen)
    assert m._thread is None
    m.close()                                             # idempotent


def test_c52_quitting_while_the_mute_is_still_being_applied_leaves_nothing_muted(tmp_path, mixer):
    mixer.gate.clear()                                    # the audio service is slow
    m = MediaMuter(tmp_path, backend=mixer)
    m.mute()
    assert mixer.entered.wait(3)
    closer = threading.Thread(target=m.close)             # user quits right now
    closer.start()
    mixer.gate.set()
    closer.join(5)
    import time
    end = time.time() + 2
    while mixer.muted and time.time() < end:
        time.sleep(0.01)
    assert mixer.muted == set(), "other apps stay muted after VoxFlow exits (until its next start)"


def test_restore_now_is_safe_from_any_thread(tmp_path, mixer):
    m = MediaMuter(tmp_path, backend=mixer, threaded=False)
    m.mute()
    threads = [threading.Thread(target=m.restore_now) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mixer.muted == set() and mixer.calls.count("unmute:spotify,chrome") == 1


# =========================================================================== Core Audio backend (fake COM)
# Interface layout the backend relies on (vtable slot numbers after IUnknown's 0-2), written down here
# independently of the implementation, from the Windows SDK headers mmdeviceapi.h / audiopolicy.h / audioclient.h.
ENUM_AUDIO_ENDPOINTS, COLLECTION_COUNT, COLLECTION_ITEM, DEVICE_ACTIVATE = 3, 3, 4, 3
GET_SESSION_ENUMERATOR, SESSIONS_COUNT, SESSIONS_GET = 5, 3, 4
CONTROL_GET_STATE, CONTROL2_GET_SESSION_ID, CONTROL2_GET_PID = 3, 12, 14
VOLUME_SET_MUTE, VOLUME_GET_MUTE, RELEASE = 5, 6, 2


class Node:
    """A fake COM object. ``released`` counts Release() calls made through the context manager."""

    live = 0

    def __init__(self, kind, **data):
        self.kind, self.data, self.released = kind, data, 0
        self.ptr = ctypes.c_void_p(1)
        Node.live += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.ptr and self.ptr.value:
            self.released += 1
            Node.live -= 1
            self.ptr = ctypes.c_void_p()

    # --- _Com API
    def out(self, index, *args, argtypes=()):
        k = self.kind
        if k == "enumerator" and index == ENUM_AUDIO_ENDPOINTS:
            assert args[:2] == (0, 1), "eRender, DEVICE_STATE_ACTIVE"
            return Node("devices", items=self.data["devices"]) if self.data["devices"] is not None else None
        if k == "devices" and index == COLLECTION_ITEM:
            return self.data["items"][args[0]]
        if k == "device" and index == DEVICE_ACTIVATE:
            return Node("manager", sessions=self.data["sessions"]) if self.data.get("sessions") is not None else None
        if k == "manager" and index == GET_SESSION_ENUMERATOR:
            return Node("sessions", items=self.data["sessions"])
        if k == "sessions" and index == SESSIONS_GET:
            return self.data["items"][args[0]]
        raise AssertionError(f"unexpected out() slot {index} on {k}")

    def call(self, index, *args, argtypes=()):
        k = self.kind
        if k in ("devices", "sessions") and index == 3:
            args[0]._obj.value = len(self.data["items"])
            return 0
        if k == "control" and index == CONTROL_GET_STATE:
            args[0]._obj.value = self.data["state"]
            return 0
        if k == "control2" and index == CONTROL2_GET_PID:
            args[0]._obj.value = self.data["pid"]
            return 0
        if k == "control2" and index == CONTROL2_GET_SESSION_ID:
            sid = self.data["sid"]
            if sid is None:
                return -1
            buf = ctypes.create_unicode_buffer(sid)
            self.data["keep"] = buf
            args[0]._obj.value = ctypes.addressof(buf)
            return 0
        if k == "volume" and index == VOLUME_GET_MUTE:
            args[0]._obj.value = int(self.data["session"]["muted"])
            return 0
        if k == "volume" and index == VOLUME_SET_MUTE:
            if self.data["session"].get("locked"):
                return -1
            self.data["session"]["muted"] = bool(args[0])
            return 0
        raise AssertionError(f"unexpected call() slot {index} on {k}")

    def query(self, iid):
        s = self.data["session"]
        if iid == snd._IID_SESSION_CONTROL2:
            return Node("control2", pid=s["pid"], sid=s["sid"]) if s.get("has_control2", True) else None
        if iid == snd._IID_SIMPLE_VOLUME:
            return Node("volume", session=s) if s.get("has_volume", True) else None
        raise AssertionError(iid)


class FakeOle32:
    def __init__(self, enumerator, init_hr=0, create_hr=0):
        self.enumerator = enumerator
        self.inits = self.uninits = self.freed = 0

        def co_initialize(reserved, mode):
            self.inits += 1
            return init_hr

        def co_uninitialize():
            self.uninits += 1

        def co_create(clsid, outer, ctx, iid, out):
            if create_hr < 0:
                return create_hr
            out._obj.value = 1
            return 0

        def co_free(ptr):
            self.freed += 1

        # plain functions: the code under test assigns ``argtypes`` on them like on real ctypes functions
        self.CoInitializeEx, self.CoUninitialize = co_initialize, co_uninitialize
        self.CoCreateInstance, self.CoTaskMemFree = co_create, co_free


def session(pid, sid, state=1, muted=False, **extra):
    return dict(pid=pid, sid=sid, state=state, muted=muted, **extra)


@pytest.fixture
def core_audio(monkeypatch):
    """Returns ``build(devices)``; ``devices`` is a list of session-dict lists (None = device without sessions)."""
    Node.live = 0

    def build(devices, init_hr=0, create_hr=0):
        device_nodes = []
        for sessions in devices or []:
            if sessions is None:
                device_nodes.append(Node("device", sessions=None))
            elif sessions == "missing":
                device_nodes.append(None)
            else:
                controls = [Node("control", state=s["state"], session=s) if s is not None else None for s in sessions]
                device_nodes.append(Node("device", sessions=controls))
        enumerator = Node("enumerator", devices=device_nodes if devices is not None else None)
        ole = FakeOle32(enumerator, init_hr, create_hr)
        monkeypatch.setattr(snd.ctypes, "WinDLL", lambda name, **kw: ole)
        monkeypatch.setattr(snd, "_Com", lambda ptr: enumerator)
        return ole

    return build


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_core_audio_mutes_only_active_unmuted_sessions_of_other_apps(core_audio, monkeypatch):
    monkeypatch.setattr(snd.os, "getpid", lambda: 500)
    spotify, chrome = session(100, "spotify-session"), session(200, "chrome-session")
    idle = session(300, "idle-session", state=0)
    user_muted = session(400, "already-muted", muted=True)
    me, system = session(500, "voxflow-own-cues"), session(0, "system-sounds")
    no_id, locked = session(600, None), session(700, "locked", locked=True)
    ole = core_audio([[spotify, idle, user_muted, me, None], None, "missing", [chrome, system, no_id, locked]])
    backend = snd.CoreAudioBackend()
    assert backend.mute_others() == ["spotify-session", "chrome-session"]
    assert spotify["muted"] and chrome["muted"]
    assert not any(s["muted"] for s in (idle, me, system, no_id, locked))
    assert user_muted["muted"], "left exactly as the user had it"
    assert ole.inits == ole.uninits == 1 and ole.freed >= 6, "session-id strings must be freed with CoTaskMemFree"
    assert Node.live == 0, "every COM reference must be released"

    backend.unmute(["spotify-session", "chrome-session"])
    assert not spotify["muted"] and not chrome["muted"] and user_muted["muted"]
    assert Node.live == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_core_audio_unmute_with_nothing_to_do_never_touches_com(core_audio):
    ole = core_audio([[session(100, "a", muted=True)]])
    snd.CoreAudioBackend().unmute([])
    assert ole.inits == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_core_audio_handles_missing_interfaces_and_no_devices(core_audio, monkeypatch):
    monkeypatch.setattr(snd.os, "getpid", lambda: 1)
    no_ctl2, no_vol = session(100, "a", has_control2=False), session(200, "b", has_volume=False)
    core_audio([[no_ctl2, no_vol]])
    assert snd.CoreAudioBackend().mute_others() == []
    assert Node.live == 0, "partially acquired interfaces must still be released"
    core_audio(None)                                      # EnumAudioEndpoints failed
    assert snd.CoreAudioBackend().mute_others() == []
    core_audio([])
    assert snd.CoreAudioBackend().mute_others() == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
@pytest.mark.parametrize("init_hr,uninit_expected", [(0, 1), (1, 1), (-2147417850, 0)])   # S_OK, S_FALSE, RPC_E_CHANGED_MODE
def test_core_audio_balances_com_initialisation(core_audio, init_hr, uninit_expected):
    ole = core_audio([[]], init_hr=init_hr)
    snd.CoreAudioBackend().mute_others()
    assert ole.uninits == uninit_expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_core_audio_failure_surfaces_as_oserror_and_muter_survives(core_audio, tmp_path):
    ole = core_audio([[]], create_hr=-2147221164)         # REGDB_E_CLASSNOTREG
    with pytest.raises(OSError, match="audio device enumerator unavailable"):
        snd.CoreAudioBackend().mute_others()
    assert ole.uninits == 1, "COM must be uninitialised even when the call fails"
    m = MediaMuter(tmp_path, threaded=False)
    assert isinstance(m._backend, snd.CoreAudioBackend)
    m.mute()                                              # logged, not raised
    assert not m.active


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_guid_layout_and_raw_com_wrapper_against_a_real_vtable():
    g = snd._Guid.of(snd._IID_SIMPLE_VOLUME)
    assert (g.d1, g.d2, g.d3) == (0x87CE5498, 0x68D6, 0x44E5) and bytes(g.d4) == bytes.fromhex("92156DA47EF883D8")
    assert ctypes.sizeof(snd._Guid) == 16

    # a hand-made COM object: vtable = [QueryInterface, AddRef, Release, Method3]
    calls = []
    QI = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(snd._Guid), ctypes.POINTER(ctypes.c_void_p))
    PLAIN = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)
    M3 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
    obj_holder = {}

    def query(this, iid, out):
        calls.append(("qi", iid.contents.d1))
        if iid.contents.d1 == 0x87CE5498:
            out[0] = this
            return 0
        return -2147467262                                   # E_NOINTERFACE

    def method3(this, n, out):
        calls.append(("m3", n))
        if n == 1:
            out[0] = this
            return 0
        return -1 if n == 2 else 0                           # n == 0: success but NULL out-pointer

    thunks = [QI(query), PLAIN(lambda this: 2), PLAIN(lambda this: calls.append(("release",)) or 0), M3(method3)]
    vtable = (ctypes.c_void_p * 4)(*[ctypes.cast(t, ctypes.c_void_p) for t in thunks])
    obj = ctypes.c_void_p(ctypes.addressof(vtable))
    obj_holder["o"] = (obj, vtable, thunks)
    com = snd._Com(ctypes.c_void_p(ctypes.addressof(obj)))

    assert com.query(snd._IID_SIMPLE_VOLUME) is not None
    assert com.query(snd._IID_ENUMERATOR) is None
    assert com.out(3, 1, argtypes=(ctypes.c_int,)) is not None
    assert com.out(3, 2, argtypes=(ctypes.c_int,)) is None, "a failed HRESULT is not an interface"
    assert com.out(3, 0, argtypes=(ctypes.c_int,)) is None, "S_OK with a NULL pointer is not an interface"
    with com:
        pass
    with com:                                                # second exit: already released
        pass
    assert calls.count(("release",)) == 1
