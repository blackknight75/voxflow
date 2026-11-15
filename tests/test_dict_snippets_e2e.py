"""Dictionary and Snippets, end to end, the way a user meets them.

An entry is saved in the real UI (or through the controller's store, which is what the UI calls), then a
dictation runs through the real controller: hotkey -> recorder -> ASR -> clean-up -> snippets / dictionary ->
paste. The ASR is ``voxflow.fakes.FakeTranscriber`` returning what Parakeet really writes for such speech
(capital letters, its own full stop, commas, hyphens), and the clean-up is ``voxflow.fakes.FakeCleaner``: the
real rule pipeline of ``voxflow.cleanup`` with the default settings (smart formatting on, language "auto").
The last test does the same with the real speech model on the CPU and synthetic speech.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import wave

import numpy as np
import pytest

from tests.controller_rig import TCleaner, TRecorder, make_rig
from tests.test_ui import UI, make_ui  # noqa: F401  (make_ui is a fixture)
from voxflow import store
from voxflow.fakes import FakeCleaner, FakeTranscriber

ADDRESS = "221B Baker Street\nLondon NW1 6XE"


def _doubles() -> dict:
    return {"asr": FakeTranscriber(text="", delay_s=0.0), "cleaner": FakeCleaner(delay_s=0.0)}


@pytest.fixture
def rig(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch, **_doubles())
    yield r
    r.close()


@pytest.fixture
def app(make_ui) -> UI:  # noqa: F811
    return make_ui(**_doubles())


def say(rig, spoken: str) -> str:
    """One push-to-talk dictation whose recognised text is ``spoken``; returns what was pasted."""
    before = len(rig.injector.pasted)
    rig.dictate(spoken)
    assert rig.c.state != "error", rig.c.message
    assert len(rig.injector.pasted) == before + 1, "nothing was pasted"
    return rig.injector.pasted[-1]


def add_snippet_in_ui(app: UI, trigger: str, text: str) -> None:
    app.page("snippets")
    app.find("snippets_add").click()
    app.find("snippet_trigger").setText(trigger)
    app.find("snippet_text").setPlainText(text)
    app.find("snippet_editor_ok").click()


def add_word_in_ui(app: UI, term: str, aliases: str = "") -> None:
    app.page("dictionary")
    app.find("dict_add").click()
    app.find("dict_term").setText(term)
    app.find("dict_aliases").setText(aliases)
    app.find("dict_editor_ok").click()


# =========================================================================== snippets
@pytest.mark.parametrize("spoken", [
    "My address.",                 # what Parakeet writes for "my address"
    "my address",
    "Insert my address.",          # people say the verb
    "insert  My Address",
    "Insert, my address!",
    "Paste my address.",
    "My, address.",
    "My. Address.",
    "My-address.",
    "\"My address.\"",
    "Um, my address.",             # a filler the clean-up removes
])
def test_snippet_saved_in_the_ui_fires_however_the_recogniser_writes_the_trigger(app, spoken):
    add_snippet_in_ui(app, "my address", ADDRESS)
    assert say(app.rig, spoken) == ADDRESS


@pytest.mark.parametrize("trigger, spoken", [
    ("sign off", "Sign-off."),
    ("sign off", "Signoff."),
    ("sign-off", "Sign off."),
    ("Sign Off!", "sign off"),
    ("zoom link 2", "Zoom link two."),
    ("zoom link two", "Zoom link 2."),
    ("let's meet", "Lets meet."),
    ("e-mail sig", "Email sig."),
    ("brb", "BRB."),
])
def test_trigger_spelling_and_spoken_form_need_not_agree(rig, trigger, spoken):
    rig.c.snippets.add(trigger, "SNIPPET")
    assert say(rig, spoken) == "SNIPPET"


def test_snippet_inside_a_sentence_and_at_its_end(rig):
    rig.c.snippets.add("my address", ADDRESS)
    rig.c.snippets.add("sign off", "Best regards,\nAlex")
    assert say(rig, "Please send it to my address, thanks.") == f"Please send it to {ADDRESS}, thanks."
    # the recogniser's full stop after a closing snippet is not typed after the sign-off
    assert say(rig, "Thanks for the help. Sign off.") == "Thanks for the help. Best regards,\nAlex"


def test_snippet_edits_take_effect_on_the_next_dictation_without_a_restart(app):
    add_snippet_in_ui(app, "my address", ADDRESS)
    assert say(app.rig, "My address.") == ADDRESS

    app.find("snippets_edit").click()                                   # change the text
    app.find("snippet_text").setPlainText("742 Evergreen Terrace")
    app.find("snippet_editor_ok").click()
    assert say(app.rig, "My address.") == "742 Evergreen Terrace"

    app.find("snippets_edit").click()                                   # change the trigger
    app.find("snippet_trigger").setText("home address")
    app.find("snippet_editor_ok").click()
    assert say(app.rig, "Home address.") == "742 Evergreen Terrace"
    assert say(app.rig, "My address.") == "My address."

    app.find("snippets_remove").click()                                 # delete it
    assert say(app.rig, "Home address.") == "Home address."


def test_snippet_text_is_typed_exactly_as_saved(rig):
    rig.c.dictionary.add("GitHub", ["get hub"])
    rig.c.snippets.add("repo link", "see github.com/acme/get-hub-tools (get hub)")
    assert say(rig, "Repo link.") == "see github.com/acme/get-hub-tools (get hub)"
    # ... while the dictionary still corrects the words around it
    assert say(rig, "Push to get hub, repo link.") == "Push to GitHub, see github.com/acme/get-hub-tools (get hub)"


def test_trigger_reworded_by_the_language_model_still_fires(tmp_path, qtbot, monkeypatch):
    r = make_rig(tmp_path, qtbot, monkeypatch)                          # TCleaner: scripted "model" output
    try:
        r.c.snippets.add("sig dot", "SIGNATURE")
        r.cleaner.rewrites["Sig dot."] = "Sig."                         # the model "fixed" the spoken word
        assert say(r, "Sig dot.") == "SIGNATURE"
        r.cleaner.rewrites["Insert sign off."] = "Insert sign-off."    # harmless rewording: matched as it is
        r.c.snippets.add("sign off", "Best")
        assert say(r, "Insert sign off.") == "Best"
    finally:
        r.close()


def test_duplicate_triggers_are_the_same_spoken_phrase(rig):
    rig.c.snippets.add("sign off", "A")
    for same in ("Sign-Off", "sign off!", "SIGN  OFF"):
        with pytest.raises(ValueError, match="already have a snippet"):
            rig.c.snippets.add(same, "B")
    with pytest.raises(ValueError, match="phrase you will say"):
        rig.c.snippets.add("?!", "B")


# =========================================================================== dictionary
@pytest.mark.parametrize("term, spoken, pasted", [
    ("Kubernetes", "I deploy on Kubernetis every day.", "I deploy on Kubernetes every day."),
    ("Kubernetes", "Kubernetis's scheduler is fine.", "Kubernetes's scheduler is fine."),
    ("PyTorch", "We train it in pie torch.", "We train it in PyTorch."),
    ("GitHub", "Push it to get hub.", "Push it to GitHub."),
    ("Grafana", "Open the graphana dashboard.", "Open the Grafana dashboard."),
    ("Postgres", "It runs on post gress.", "It runs on Postgres."),
    ("Supercalifragilisticexpialidocious", "Supercalifragilistic expialidocio.",
     "Supercalifragilisticexpialidocious."),
    ("VoxFlow", "I dictate with vox flow.", "I dictate with VoxFlow."),
    ("VoxFlow", "I dictate with Vox-Flow.", "I dictate with VoxFlow."),
])
def test_word_added_in_the_ui_without_aliases_corrects_what_the_recogniser_wrote(app, term, spoken, pasted):
    add_word_in_ui(app, term)
    assert say(app.rig, spoken) == pasted


def test_sounds_like_alias_from_the_ui(app):
    add_word_in_ui(app, "Siobhan", "shivon, shiv on")
    assert say(app.rig, "Shivon said hi, and shiv on agreed.") == "Siobhan said hi, and Siobhan agreed."


def test_dictionary_edits_take_effect_on_the_next_dictation_without_a_restart(app):
    add_word_in_ui(app, "Kubernetes")
    assert say(app.rig, "Kubernetis is hard.") == "Kubernetes is hard."
    app.find("dict_edit").click()
    app.find("dict_term").setText("Kubernetes Engine")
    app.find("dict_aliases").setText("cube engine")
    app.find("dict_editor_ok").click()
    assert say(app.rig, "Try cube engine.") == "Try Kubernetes Engine."
    app.find("dict_remove").click()
    assert say(app.rig, "Try cube engine.") == "Try cube engine."


@pytest.mark.parametrize("spoken", [
    "The doctor saw it.",                      # Docker
    "The stripes are linear.",                 # Stripe (an ending), and a term that is a plain word stays one
    "Notions of motion.",                      # Notion
    "Can we talk?",                            # Dan: short terms only match exactly
    "See github.com slash docs.",              # web addresses keep their spelling
])
def test_dictionary_leaves_ordinary_words_alone(rig, spoken):
    for term in ("Docker", "Stripe", "Notion", "Dan", "GitHub"):
        rig.c.dictionary.add(term)
    pasted = say(rig, spoken)
    assert pasted.split() == rig.c._cleaner.clean(spoken, language="auto", smart=True).split()


def test_dictionary_skips_urls_paths_and_identifiers():
    d = store.Dictionary(tempfile.mkdtemp())
    d.add("VoxFlow")
    d.add("GitHub")
    for text in ("https://github.com/acme/voxflow", "set VOXFLOW_DEVICE=cpu", r"C:\voxflow\logs", "me@voxflow.dev"):
        assert d.apply(text) == text
    assert d.apply("I use voxflow. And github!") == "I use VoxFlow. And GitHub!"


def test_dictionary_correction_is_fast_with_a_big_dictionary():
    d = store.Dictionary(tempfile.mkdtemp())
    rng = np.random.default_rng(7)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    with d.batch():
        for i in range(500):
            d.add("".join(rng.choice(letters, size=int(rng.integers(5, 12)))).capitalize() + str(i % 10))
    text = " ".join(["please send the quarterly report to the engineering team before friday"] * 30)  # 330 words
    d.apply("warm up")
    t0 = time.perf_counter()
    d.apply(text)
    assert time.perf_counter() - t0 < 0.5      # measured about 0.03 s; a two-minute dictation is ~300 words


# =========================================================================== the real speech model
def _tts(text: str) -> np.ndarray:
    path = os.path.join(tempfile.gettempdir(), f"voxflow_dict_{abs(hash(text))}.wav")
    ps = ("Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
          f"$s.SetOutputToWaveFile('{path}',$f);$s.Speak('{text}');$s.Dispose()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60, capture_output=True)
    except Exception:
        pytest.skip("System.Speech TTS unavailable")
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class _Loaded:
    """A real Transcriber loaded up front (that takes longer than the rig waits for a start), then shared."""

    def __init__(self, transcriber):
        self._t = transcriber
        self.backend = transcriber.backend

    def load(self):
        pass

    def transcribe(self, audio):
        return self._t.transcribe(audio)


@pytest.fixture(scope="module")
def cpu_asr():
    pytest.importorskip("onnx_asr")
    from voxflow.asr import Transcriber

    t = Transcriber(device="cpu")
    try:
        t.load()
    except Exception as exc:  # noqa: BLE001 - weights not downloaded, no onnxruntime, ...
        pytest.skip(f"the speech model is not available here: {exc}")
    return _Loaded(t)


@pytest.mark.model
def test_real_speech_model_on_cpu_with_snippets_and_dictionary(tmp_path, qtbot, monkeypatch, cpu_asr):
    r = make_rig(tmp_path, qtbot, monkeypatch, asr=cpu_asr, cleaner=TCleaner())
    try:
        assert r.c.model_status("asr")["state"] == "ready"
        r.c.snippets.add("my address", ADDRESS)
        r.c.snippets.add("sign off", "Best regards,\nAlex")
        r.c.dictionary.add("Kubernetes")
        heard = {}
        for spoken, want in (("Insert my address.", ADDRESS), ("Sign off.", "Best regards,\nAlex"),
                             ("We deploy everything on Kubernetes.", "Kubernetes")):
            TRecorder.audio = _tts(spoken)
            TRecorder.speech_seconds = len(TRecorder.audio) / 16000
            r.dictate()
            pasted = r.injector.pasted[-1]
            heard[spoken] = (r.last_entry()["raw"], pasted)
            assert want in pasted, heard
    finally:
        r.close()
