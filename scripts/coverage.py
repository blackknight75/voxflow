"""Run the test suite under coverage and print a per-module summary.

    python scripts/coverage.py            # whole suite
    python scripts/coverage.py -k store   # extra args go to pytest
    python scripts/coverage.py --fast     # skip the real-model tests

Output: a per-module table, the line coverage of the non-UI modules (target >= 90 %), the line coverage of
``voxflow/ui`` and a machine-readable copy in ``<temp>/voxflow-coverage.json``. Exit code = pytest's exit code,
or 2 when the tests pass but non-UI coverage is below ``--min`` (default 0 = report only).

There is no coverage config file: only coverage.py's default markers apply. Nothing is excluded for needing
real hardware or a desktop session (the tests fake those). The only exclusions are a few defensive spots that
no input can reach, each marked ``# pragma: no cover`` / ``# pragma: no branch`` with its reason; the premises
are guarded by ``tests/test_cov_unreachable.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip tests that load the real ASR / cleanup models")
    ap.add_argument("--min", type=float, default=0.0, help="fail (exit 2) if non-UI line coverage is below this %%")
    ap.add_argument("--json", default=str(Path(tempfile.gettempdir()) / "voxflow-coverage.json"))
    args, pytest_args = ap.parse_known_args(argv)

    env = dict(os.environ)
    if args.fast:
        env["VOXFLOW_SKIP_MODEL_TESTS"] = "1"
    data_file = str(Path(tempfile.gettempdir()) / "voxflow.coverage")
    env["COVERAGE_FILE"] = data_file
    cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "-p", "no:cacheprovider",
           "--cov=voxflow", "--cov-branch", "--cov-report=", *pytest_args]
    print("$", " ".join(cmd))
    test_rc = subprocess.call(cmd, cwd=str(ROOT), env=env)

    rc = subprocess.call([sys.executable, "-m", "coverage", "json", "-q", "-o", args.json,
                          f"--data-file={data_file}"], cwd=str(ROOT), env=env)
    if rc != 0:
        print("coverage data could not be exported")
        return test_rc or 1
    report = json.loads(Path(args.json).read_text(encoding="utf-8"))

    rows, groups = [], {"non-UI": [0, 0], "UI": [0, 0]}
    for name, info in sorted(report["files"].items()):
        rel = name.replace("\\", "/")
        s = info["summary"]
        group = "UI" if "/ui/" in rel else "non-UI"
        groups[group][0] += s["covered_lines"]
        groups[group][1] += s["num_statements"]
        pct = 100.0 * s["covered_lines"] / s["num_statements"] if s["num_statements"] else 100.0
        rows.append((rel, s["num_statements"], s["missing_lines"], pct))

    width = max(len(r[0]) for r in rows) if rows else 10
    print(f"\n{'module'.ljust(width)}  stmts   miss   line%")
    for rel, n, miss, pct in rows:
        print(f"{rel.ljust(width)}  {n:5d}  {miss:5d}  {pct:5.1f}")
    print()
    pcts = {}
    for group, (cov, total) in groups.items():
        pcts[group] = 100.0 * cov / total if total else 100.0
        print(f"{group} line coverage: {pcts[group]:.1f} %  ({cov}/{total} statements)")
    print(f"pytest exit code: {test_rc}")
    if test_rc == 0 and pcts["non-UI"] < args.min:
        print(f"FAIL: non-UI coverage {pcts['non-UI']:.1f} % is below the required {args.min:.1f} %")
        return 2
    return test_rc


if __name__ == "__main__":
    sys.exit(main())
