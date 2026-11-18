from voxflow.inject import Injector


class FakeClip:
    def __init__(self, content=None):
        self.content = content          # "snapshot" is just the content
        self.seq = 1
        self.log = []

    def snapshot(self):
        self.log.append("snap")
        return self.content

    def set_text(self, t):
        self.content, self.seq = t, self.seq + 1
        self.log.append("set")
        return self.seq

    def sequence(self):
        return self.seq

    def restore(self, snap):
        self.content, self.seq = snap, self.seq + 1
        self.log.append("restore")


class FakeKeys:
    def __init__(self, clip, held=0):
        self.clip, self.held, self.pasted, self.typed = clip, held, [], []

    def paste(self):
        self.pasted.append(self.clip.content)
        return True

    def type_text(self, t):
        self.typed.append(t)
        return True

    def modifiers_down(self):
        if self.held > 0:
            self.held -= 1
            return True
        return False


def mk(content="old", **kw):
    clip = FakeClip(content)
    keys = FakeKeys(clip, kw.pop("held", 0))
    t = [0.0]
    inj = Injector(clip, keys, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], **kw)
    return inj, clip, keys


def test_paste_then_restore():
    inj, clip, keys = mk("precious")
    assert inj.inject("hello") is True
    assert keys.pasted == ["hello"]               # clipboard held our text when Ctrl+V fired
    assert clip.content == "precious"             # and the old content is back
    assert clip.log == ["snap", "set", "restore"]


def test_no_restore_when_disabled():
    inj, clip, keys = mk("precious", restore=False)
    inj.inject("x")
    assert clip.content == "x" and "snap" not in clip.log


def test_no_clobber_if_user_copied_meanwhile():
    inj, clip, keys = mk("precious")
    orig = keys.paste

    def paste_and_user_copies():
        r = orig()
        clip.content, clip.seq = "user copied this", clip.seq + 1
        return r

    keys.paste = paste_and_user_copies
    inj.inject("x")
    assert clip.content == "user copied this"


def test_waits_for_modifiers_released():
    inj, clip, keys = mk(held=5)
    inj.inject("x")
    assert keys.held == 0 and keys.pasted == ["x"]


def test_modifier_wait_times_out_without_sending_the_paste():
    """Bug C-06: Ctrl+V with Win/Alt/Shift still down would be Win+V / Alt+Ctrl+V. The text is left on the
    clipboard for a manual paste instead."""
    inj, clip, keys = mk(held=10 ** 9, modifier_timeout=0.2)
    assert inj.inject("x") is False and keys.pasted == []
    assert clip.content == "x" and inj.last_outcome == "copied"


def test_empty_text_and_clipboard_failure_falls_back_to_typing():
    inj, clip, keys = mk()
    assert inj.inject("") is False and not keys.pasted
    clip.set_text = lambda t: None
    assert inj.inject("typed") is True and keys.typed == ["typed"]


def test_modifiers_held_reports_the_keyboard_without_touching_the_clipboard():
    inj, clip, keys = mk("keep", held=1)
    assert inj.modifiers_held() is True
    assert inj.modifiers_held() is False
    assert clip.log == [] and keys.pasted == []


def test_modifiers_held_is_false_when_the_keyboard_cannot_be_read():
    inj, clip, keys = mk("keep")
    keys.modifiers_down = lambda: (_ for _ in ()).throw(OSError("no keyboard"))
    assert inj.modifiers_held() is False
