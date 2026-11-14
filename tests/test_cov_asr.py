"""voxflow.asr: the "does it really run there?" check must survive a session object whose provider query raises."""
from __future__ import annotations

from voxflow.asr import _runs_on


class _Broken:
    def get_providers(self):
        raise RuntimeError("session already released")


class _Cuda:
    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _Holder:
    def __init__(self, **parts):
        self.__dict__.update(parts)


def test_runs_on_skips_a_session_that_cannot_report_its_providers():
    assert _runs_on(_Broken(), "CUDAExecutionProvider") is False
    assert _runs_on(_Holder(encoder=_Broken()), "CUDAExecutionProvider") is False


def test_runs_on_keeps_looking_after_a_broken_session():
    # the stack is LIFO: the broken session (added last) is asked first, the working one afterwards
    model = _Holder(decoder=_Cuda(), encoder=_Broken())
    assert _runs_on(model, "CUDAExecutionProvider") is True
