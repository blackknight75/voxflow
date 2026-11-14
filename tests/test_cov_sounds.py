"""voxflow.sounds: a device whose session enumerator is missing, and mute requests after shutdown began."""
from __future__ import annotations

import sys

import pytest

from voxflow import sounds as snd
from voxflow.sounds import MediaMuter


class _Node:
    """Minimal fake COM object: ``outs`` maps a vtable slot to what ``out()`` returns; ``count`` answers slot 3."""

    def __init__(self, name, outs=None, count=0):
        self.name, self.outs, self.count = name, outs or {}, count
        self.released = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.released += 1

    def out(self, index, *args, argtypes=()):
        target = self.outs[index]
        return target(*args) if callable(target) else target

    def call(self, index, *args, argtypes=()):
        assert index == 3, f"unexpected call slot {index} on {self.name}"
        args[0]._obj.value = self.count
        return 0


class _Ole32:
    def __init__(self):
        self.inits = self.uninits = 0

        def co_create(clsid, outer, ctx, iid, out):
            out._obj.value = 1
            return 0

        def co_init(reserved, mode):
            self.inits += 1
            return 0

        def co_uninit():
            self.uninits += 1

        self.CoCreateInstance, self.CoTaskMemFree = co_create, (lambda ptr: None)
        self.CoInitializeEx, self.CoUninitialize = co_init, co_uninit


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_device_without_a_session_enumerator_is_skipped_and_released(monkeypatch):
    manager = _Node("manager", {5: None})                       # GetSessionEnumerator failed
    device = _Node("device", {3: lambda *a: manager})
    devices = _Node("devices", {4: lambda i: device}, count=1)
    enumerator = _Node("enumerator", {3: lambda *a: devices})
    ole = _Ole32()
    monkeypatch.setattr(snd.ctypes, "WinDLL", lambda name, **kw: ole)
    monkeypatch.setattr(snd, "_Com", lambda ptr: enumerator)

    assert snd.CoreAudioBackend().mute_others() == []
    assert [n.released for n in (manager, device, devices, enumerator)] == [1, 1, 1, 1]
    assert ole.inits == ole.uninits == 1


class _Backend:
    def __init__(self):
        self.calls = []

    def mute_others(self):
        self.calls.append("mute")
        return ["s1"]

    def unmute(self, ids):
        self.calls.append(("unmute", tuple(ids)))


def test_mute_requested_after_shutdown_started_is_ignored(tmp_path):
    backend = _Backend()
    muter = MediaMuter(tmp_path, backend=backend, threaded=False)
    muter.mute()
    assert muter.active and backend.calls == ["mute"]
    muter.restore_now()                                         # exit path: unmute and refuse new mutes
    assert not muter.active and backend.calls == ["mute", ("unmute", ("s1",))]
    muter.mute()
    assert backend.calls == ["mute", ("unmute", ("s1",))], "nothing is muted once the app is closing"
    assert not muter.active and not (tmp_path / "muted_sessions.json").exists()
