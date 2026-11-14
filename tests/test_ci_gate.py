"""The verdict helpers of the real-model workflow (.github/scripts/ci_gate.py): a model that silently did not
load, or model tests that skipped themselves, must turn the run red."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "ci_gate.py"
_spec = importlib.util.spec_from_file_location("ci_gate", _PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

GOOD = """== VoxFlow selftest ==
[ok] UI imports (1.02s)
[ok] keyboard hook (0.01s): installed + removed
[ok] clipboard (0.00s): round-trip, previous content restored
[note] VOXFLOW_DEVICE=cpu
[ok] speech model load (9.31s): backend cpu
[ok] cleanup model load (21.50s): ready=True, backend cpu
run 1: audio 11.60s | asr 900 ms | cleanup 0 ms | total 900 ms
run 2: audio 11.60s | asr 850 ms | cleanup 0 ms | total 850 ms
raw  : 'the quick brown fox'
final: 'The quick brown fox.'
SELFTEST PASSED
"""


def test_selftest_good_output_passes():
    assert gate.evaluate_selftest(GOOD, "cpu") == ([], [])
    steps = gate.selftest_steps(GOOD)
    assert steps["UI imports"] == (True, "") and steps["speech model load"] == (True, "backend cpu")


def test_selftest_language_model_that_fell_back_to_rules_is_a_problem():
    text = GOOD.replace("ready=True, backend cpu", "ready=True, backend rules")
    problems, warnings = gate.evaluate_selftest(text, "cpu")
    assert len(problems) == 1 and "cleanup model load" in problems[0] and "rules" in problems[0] and not warnings
    assert gate.evaluate_selftest(text) == ([], []), "without --backend the backend is not judged"


def test_selftest_desktop_steps_only_warn_unless_strict():
    text = (GOOD.replace("[ok] keyboard hook (0.01s): installed + removed",
                         "[FAIL] keyboard hook: RuntimeError: hook did not start")
            .replace("SELFTEST PASSED", "SELFTEST FAILED"))
    problems, warnings = gate.evaluate_selftest(text, "cpu")
    assert problems == [] and len(warnings) == 1 and "keyboard hook" in warnings[0]
    problems, warnings = gate.evaluate_selftest(text, "cpu", strict_desktop=True)
    assert len(problems) == 1 and warnings == []


@pytest.mark.parametrize("old,new,needle", [
    ("[ok] UI imports (1.02s)\n", "[FAIL] UI imports: ImportError: PySide6\n", "UI imports"),
    ("[ok] UI imports (1.02s)\n", "", "'UI imports' did not run"),
    ("run 2: audio 11.60s | asr 850 ms | cleanup 0 ms | total 850 ms\n", "", "ran 1 time"),
    ("final: 'The quick brown fox.'", "final: ''", "no text"),
    ("SELFTEST PASSED\n", "", "verdict"),
    ("[ok] speech model load (9.31s): backend cpu", "[ok] speech model load (9.31s): backend cuda", "expected backend cpu"),
])
def test_selftest_problems(old, new, needle):
    assert old in GOOD
    problems, _warnings = gate.evaluate_selftest(GOOD.replace(old, new), "cpu")
    assert any(needle in p for p in problems), problems


def test_selftest_that_stopped_at_the_speech_model():
    text = "== VoxFlow selftest ==\n[ok] UI imports (1.0s)\n[FAIL] speech model load: OSError: no network\nSELFTEST FAILED\n"
    problems, _warnings = gate.evaluate_selftest(text, "cpu")
    assert any("'speech model load' failed" in p for p in problems)
    assert any("'cleanup model load' did not run" in p for p in problems)


def _junit(*cases: str) -> str:
    return f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>'


PASSED = '<testcase classname="tests.test_asr" name="test_backend" time="9.1" />'
REAL = '<testcase classname="tests.test_cleanup" name="test_real_self_correction" time="2.0" />'
RULES = '<testcase classname="tests.test_cleanup" name="test_rules" time="0.0" />'
TTS_SKIP = ('<testcase classname="tests.test_asr" name="test_tts_accuracy"><skipped type="pytest.skip" '
            'message="System.Speech TTS unavailable">x</skipped></testcase>')
REAL_SKIP = ('<testcase classname="tests.test_cleanup" name="test_real_latency"><skipped type="pytest.skip" '
             'message="LLM backend unavailable">x</skipped></testcase>')
REAL_FAIL = ('<testcase classname="tests.test_cleanup" name="test_real_does_not_obey"><failure message="assert">x'
             '</failure></testcase>')


def test_junit_all_ran():
    problems, warnings, counts = gate.evaluate_junit(_junit(PASSED, REAL, RULES), ["test_asr", "test_real_"], [])
    assert problems == [] and warnings == [] and counts == {"passed": 3, "failed": 0, "skipped": 0}


def test_junit_a_skipped_model_test_is_a_problem():
    problems, warnings, counts = gate.evaluate_junit(_junit(PASSED, REAL_SKIP, RULES), ["test_asr", "test_real_"], [])
    assert counts["skipped"] == 1 and warnings == []
    assert any("no test matching 'test_real_' passed" in p for p in problems)
    assert any("skipped: tests.test_cleanup::test_real_latency: LLM backend unavailable" in p for p in problems)


def test_junit_allowed_skip_is_a_warning():
    problems, warnings, _counts = gate.evaluate_junit(_junit(PASSED, TTS_SKIP, REAL), ["test_asr", "test_real_"],
                                                      ["test_tts_accuracy"])
    assert problems == [] and len(warnings) == 1 and "test_tts_accuracy" in warnings[0]


def test_junit_nothing_collected_and_failures():
    problems, _warnings, counts = gate.evaluate_junit(_junit(RULES, REAL_FAIL), ["test_asr", "test_real_"], [])
    assert counts == {"passed": 1, "failed": 1, "skipped": 0}
    assert any("no test matches 'test_asr'" in p for p in problems)
    assert any("no test matching 'test_real_' passed" in p for p in problems)


def test_main_junit_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    report = tmp_path / "report.xml"
    report.write_text(_junit(PASSED, REAL), encoding="utf-8")
    assert gate.main(["junit", str(report), "--must-run", "test_asr", "--must-run", "test_real_"]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    report.write_text(_junit(PASSED, REAL_SKIP), encoding="utf-8")
    assert gate.main(["junit", str(report), "--must-run", "test_real_"]) == 1
    out = capsys.readouterr().out
    assert "::error title=model tests::" in out and "RESULT: FAIL" in out


def test_main_selftest_runs_the_command(tmp_path, capsys, monkeypatch):
    import sys
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    script = tmp_path / "fake_selftest.py"
    script.write_text(f"print({GOOD!r})", encoding="utf-8")
    assert gate.main(["selftest", "--backend", "cpu", "--", sys.executable, str(script)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    script.write_text(f"print({GOOD.replace('backend cpu', 'backend rules')!r})", encoding="utf-8")
    assert gate.main(["selftest", "--backend", "cpu", "--", sys.executable, str(script)]) == 1
    assert "FAIL  'cleanup model load' reports" in capsys.readouterr().out
