"""The pure parts of the live-preview prototype (scripts/stream_proto.py): word joining and LocalAgreement-2.
The prototype itself needs the real model; ``model``-marked test below runs it on a short slice of the test clip."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "stream_proto.py"
_spec = importlib.util.spec_from_file_location("stream_proto", _PATH)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def test_words_with_times_joins_subword_tokens():
    tokens = [" The", " qu", "ick", " bro", "wn", "."]
    stamps = [0.0, 0.24, 0.4, 0.56, 0.72, 0.8]
    assert sp.words_with_times(tokens, stamps) == [("The", 0.0), ("quick", 0.24), ("brown.", 0.56)]
    assert sp.words_with_times([], []) == []


def test_norm_ignores_case_and_punctuation():
    assert sp.norm("Dog.") == sp.norm("dog") == "dog"
    assert sp.norm("don't") == "don't"


def _h(text, start=0.0, step=0.3):
    return [(w, start + i * step) for i, w in enumerate(text.split())]


def test_local_agreement_commits_only_what_two_hypotheses_share():
    la = sp.LocalAgreement()
    assert la.update(_h("the quick")) == _h("the quick")          # first hypothesis: nothing confirmed yet
    assert la.committed == []
    tent = la.update(_h("The quick brown fox"))                    # case differs: still agreement
    assert [w for w, _ in la.committed] == ["The", "quick"]
    assert [w for w, _ in tent] == ["brown", "fox"]
    la.update(la.new_part(_h("the quick brown fax jumps")))
    assert [w for w, _ in la.committed] == ["The", "quick", "brown"]   # "fox" vs "fax": not committed


def test_new_part_drops_the_committed_boundary_word_even_when_its_time_moved():
    la = sp.LocalAgreement()
    la.committed = _h("send the report")                           # "report" committed at 0.6 s
    hyp = [("send", 0.0), ("the", 0.31), ("report", 0.58), ("to", 0.9), ("the", 1.1)]   # 20 ms earlier this time
    assert [w for w, _ in la.new_part(hyp)] == ["to", "the"]
    assert sp.LocalAgreement().new_part(hyp) == hyp                # nothing committed: everything is new


def test_new_part_falls_back_to_the_time_cut_without_an_ngram_match():
    la = sp.LocalAgreement()
    la.committed = [("alpha", 1.0)]
    hyp = [("beta", 0.9), ("gamma", 1.2), ("delta", 1.5)]
    assert [w for w, _ in la.new_part(hyp)] == ["gamma", "delta"]


@pytest.mark.model
def test_prototype_runs_on_the_real_model():
    from voxflow.app import read_wav_16k
    from voxflow.asr import Transcriber

    clip = read_wav_16k(str(Path(__file__).parent / "tts_sample.wav"))[: 4 * 16000]
    tr = Transcriber(device="cpu")
    tr.load()
    r = sp.run(tr, clip, 0.5, 0.0, quiet=True, bucket_s=1.0)
    assert r["updates"] >= 3
    assert r["final"]
    assert r["similarity"] > 0.6
