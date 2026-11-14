"""Verdicts for the real-model workflow (.github/workflows/models.yml). Standard library only.

    python .github/scripts/ci_gate.py selftest --backend cpu -- python -m voxflow --selftest tests/tts_sample.wav
    python .github/scripts/ci_gate.py junit report.xml --must-run test_asr --must-run test_real_ --may-skip test_tts_accuracy

Both exist because a missing model does not fail anything by itself: the cleanup model degrades to the rule pass
("backend rules") and the tests that need it skip themselves. A green run must mean the models really ran.

``selftest`` runs the command, prints its output and reads the ``[ok]`` / ``[FAIL]`` lines of ``voxflow --selftest``:
the UI imports and both model loads must be ok, both models must report the wanted backend, the pipeline must
have run twice and produced text. The keyboard hook and clipboard steps need an interactive desktop, which a
hosted runner may not give: their failure is a warning here (``--strict-desktop`` makes it count).

``junit`` reads a pytest ``--junitxml`` report: every ``--must-run`` pattern (a substring of ``classname::name``)
must match at least one test, and no matched test may be skipped, except those named by ``--may-skip``.

Exit code 0 = fine, 1 = a problem (each one is printed, and as an ``::error::`` line on GitHub Actions).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

DESKTOP_STEPS = ("keyboard hook", "clipboard")
_OK = re.compile(r"^\[ok\] (.+?) \(\d+(?:\.\d+)?s\)(?:: (.*))?$")
_FAIL = re.compile(r"^\[FAIL\] (.+?): (.*)$")
_BACKEND = re.compile(r"\bbackend (\w+)")


def selftest_steps(text: str) -> dict[str, tuple[bool, str]]:
    """``{label: (ok, detail)}`` from the ``[ok] label (1.23s): detail`` / ``[FAIL] label: reason`` lines."""
    steps: dict[str, tuple[bool, str]] = {}
    for line in text.splitlines():
        line = line.rstrip()
        hit = _OK.match(line)
        if hit:
            steps[hit.group(1)] = (True, hit.group(2) or "")
            continue
        hit = _FAIL.match(line)
        if hit:
            steps[hit.group(1)] = (False, hit.group(2))
    return steps


def evaluate_selftest(text: str, backend: str = "", strict_desktop: bool = False) -> tuple[list[str], list[str]]:
    """(problems, warnings) for the output of ``python -m voxflow --selftest WAV``."""
    problems: list[str] = []
    warnings: list[str] = []
    steps = selftest_steps(text)
    for label in ("UI imports", "speech model load", "cleanup model load"):
        if label not in steps:
            problems.append(f"the step '{label}' did not run")
    for label, (ok, detail) in steps.items():
        if ok:
            continue
        if label in DESKTOP_STEPS and not strict_desktop:
            warnings.append(f"'{label}' failed ({detail}); not counted: it needs an interactive desktop")
        else:
            problems.append(f"'{label}' failed: {detail}")
    if backend:
        for label in ("speech model load", "cleanup model load"):
            ok, detail = steps.get(label, (False, ""))
            found = _BACKEND.search(detail)
            if ok and (found is None or found.group(1) != backend):
                problems.append(f"'{label}' reports '{detail}', expected backend {backend} "
                                "(backend rules = the language model did not load)")
    runs = re.findall(r"(?m)^run \d+: ", text)
    if len(runs) < 2:
        problems.append(f"the pipeline ran {len(runs)} time(s), expected 2")
    final = re.search(r"(?m)^final: (.*)$", text)
    if final is None or final.group(1).strip() in ("", "''", '""'):
        problems.append("the pipeline produced no text")
    if not re.search(r"(?m)^SELFTEST (PASSED|FAILED)\s*$", text):
        problems.append("the self-test did not reach its verdict line")
    return problems, warnings


def evaluate_junit(xml_text: str, must_run: list[str], may_skip: list[str]) -> tuple[list[str], list[str], dict]:
    """(problems, warnings, counts) for a pytest junit report."""
    problems: list[str] = []
    warnings: list[str] = []
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    cases = []
    for case in ET.fromstring(xml_text).iter("testcase"):
        ident = f"{case.get('classname', '')}::{case.get('name', '')}"
        skipped = case.find("skipped")
        if skipped is not None:
            state, why = "skipped", (skipped.get("message") or "").strip()
        elif case.find("failure") is not None or case.find("error") is not None:
            state, why = "failed", ""
        else:
            state, why = "passed", ""
        counts[state] += 1
        cases.append((ident, state, why))
    for pattern in must_run:
        matched = [c for c in cases if pattern in c[0]]
        if not matched:
            problems.append(f"no test matches '{pattern}': the real-model tests were not collected")
            continue
        if not any(state == "passed" for _i, state, _w in matched):
            problems.append(f"no test matching '{pattern}' passed")
        for ident, state, why in matched:
            if state != "skipped":
                continue
            if any(name in ident for name in may_skip):
                warnings.append(f"skipped (allowed): {ident}: {why}")
            else:
                problems.append(f"skipped: {ident}: {why}")
    return problems, warnings, counts


def _report(problems: list[str], warnings: list[str], title: str) -> int:
    on_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for text in warnings:
        print(f"WARN  {text}")
        if on_actions:
            print(f"::warning title={title}::{text}")
    for text in problems:
        print(f"FAIL  {text}")
        if on_actions:
            print(f"::error title={title}::{text}")
    print(f"RESULT: {'FAIL' if problems else 'PASS'} ({title}: {len(problems)} problem(s), {len(warnings)} warning(s))")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    st = sub.add_parser("selftest", help="run a voxflow --selftest command and judge its output")
    st.add_argument("--backend", default="", help="backend both models must report, e.g. cpu")
    st.add_argument("--strict-desktop", action="store_true", help="keyboard hook / clipboard failures count")
    st.add_argument("--timeout", type=float, default=1800.0)
    st.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command to run")
    ju = sub.add_parser("junit", help="judge a pytest --junitxml report")
    ju.add_argument("report")
    ju.add_argument("--must-run", action="append", default=[])
    ju.add_argument("--may-skip", action="append", default=[])
    args = ap.parse_args(argv)

    if args.what == "selftest":
        command = [c for c in args.command if c != "--"] if args.command[:1] == ["--"] else args.command
        if not command:
            ap.error("give the command after --")
        try:
            proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", timeout=args.timeout)
            text, note = proc.stdout, f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired as exc:
            text = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            note = f"TIMED OUT after {args.timeout:.0f} s"
        print(text.rstrip())
        print(f"----- {note}")
        problems, warnings = evaluate_selftest(text, args.backend, args.strict_desktop)
        if note.startswith("TIMED OUT"):
            problems.insert(0, note)
        return _report(problems, warnings, "self-test")

    with open(args.report, encoding="utf-8") as fh:
        problems, warnings, counts = evaluate_junit(fh.read(), args.must_run, args.may_skip)
    print(f"junit: {counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped")
    return _report(problems, warnings, "model tests")


if __name__ == "__main__":
    sys.exit(main())
