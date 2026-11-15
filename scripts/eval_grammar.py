"""Grammar / punctuation / capitalisation score of the English cleanup on tests/grammar_eval.json.

    python scripts/eval_grammar.py                     # rules + the default language model, both transcript kinds
    python scripts/eval_grammar.py --rules-only        # the rule pass only (no model; what a "Limited" install gets)
    python scripts/eval_grammar.py --source raw|asr    # only the hand-written or only the Parakeet transcripts
    python scripts/eval_grammar.py --freeze-asr        # (re)make the "asr" transcripts: Windows voice -> Parakeet

Scores per case (lower is better except exact):
  exact   the output is one of the accepted outputs
  pter    punctuation-aware token error rate: edit distance over whitespace tokens that KEEP their punctuation and
          case ("friday," != "Friday"), against the closest accepted output, divided by its length
  wer     plain word error rate (lowercase, punctuation stripped): did the cleanup change the words?
Latency is the wall time of ``Cleaner.clean`` per case (p50 / p95 / max). Cases run as the app runs them: smart
formatting on, no tone.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "grammar_eval.json"


def load_cases(path: Path = DATA) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _edit(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _tokens(s: str) -> list[str]:
    return s.replace("\n", " ⏎ ").split()                   # a line break is a token of its own


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", s.lower()).replace("_", " ").split()


def score(out: str, expect: list[str]) -> dict:
    exact = out in expect
    pter = min(_edit(_tokens(e), _tokens(out)) / max(1, len(_tokens(e))) for e in expect)
    wer = min(_edit(_words(e), _words(out)) / max(1, len(_words(e))) for e in expect)
    return {"exact": exact, "pter": 0.0 if exact else pter, "wer": 0.0 if exact else wer}


def evaluate(cleaner, cases: list[dict], sources=("raw", "asr"), quiet: bool = False) -> dict:
    rows = []
    for case in cases:
        for src in sources:
            text = case.get(src) or ""
            if not text:
                continue
            t = time.perf_counter()
            out = cleaner.clean(text, smart=True)
            ms = (time.perf_counter() - t) * 1000
            s = score(out, case["expect"])
            info = dict(getattr(cleaner, "last_info", {}) or {})
            rows.append({"id": case["id"], "cat": case["cat"], "src": src, "in": text, "out": out, "ms": ms,
                         "llm": bool(info.get("llm")), "rejected": bool(info.get("rejected")), **s})
            if not quiet and not s["exact"]:
                flags = ("LLM" if info.get("llm") else "   ") + (" REJ" if info.get("rejected") else "    ")
                print(f"[BAD] {case['id']:16s} {src:3s} {ms:6.0f}ms {flags} pter={s['pter']:.2f} wer={s['wer']:.2f}")
                print(f"      in : {text!r}\n      out: {out!r}\n      exp: {case['expect'][0]!r}")
    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    lat = sorted(r["ms"] for r in rows)
    llm = sorted(r["ms"] for r in rows if r["llm"])
    by_src = {}
    for src in sorted({r["src"] for r in rows}):
        sub = [r for r in rows if r["src"] == src]
        by_src[src] = {"n": len(sub), "exact": sum(r["exact"] for r in sub),
                       "pter": statistics.mean(r["pter"] for r in sub), "wer": statistics.mean(r["wer"] for r in sub)}
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["cat"], [0, 0])
        c[0] += r["exact"]
        c[1] += 1
    return {
        "n": n, "exact": sum(r["exact"] for r in rows),
        "pter": statistics.mean(r["pter"] for r in rows) if rows else 0.0,
        "wer": statistics.mean(r["wer"] for r in rows) if rows else 0.0,
        "p50_ms": statistics.median(lat) if lat else 0.0, "p95_ms": lat[int(0.95 * (n - 1))] if lat else 0.0,
        "max_ms": lat[-1] if lat else 0.0, "llm_calls": len(llm),
        "llm_p50_ms": statistics.median(llm) if llm else 0.0,
        "rejected": sum(r["rejected"] for r in rows), "by_src": by_src,
        "by_cat": {k: f"{v[0]}/{v[1]}" for k, v in by_cat.items()}, "rows": rows,
    }


def print_summary(label: str, s: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"exact {s['exact']}/{s['n']}  pter {s['pter']:.3f}  wer {s['wer']:.3f}  "
          f"guardrail rejections {s['rejected']}  LLM calls {s['llm_calls']}")
    for src, v in s["by_src"].items():
        print(f"  {src:3s}: exact {v['exact']}/{v['n']}  pter {v['pter']:.3f}  wer {v['wer']:.3f}")
    print("  by category: " + ", ".join(f"{k} {v}" for k, v in s["by_cat"].items()))
    print(f"latency: p50 {s['p50_ms']:.0f} ms  p95 {s['p95_ms']:.0f} ms  max {s['max_ms']:.0f} ms  "
          f"(LLM cases p50 {s['llm_p50_ms']:.0f} ms)")


def freeze_asr(path: Path = DATA, voice: str | None = None) -> None:
    """Speak every case's words with a Windows voice and store what Parakeet wrote as the case's "asr" text.
    Homophone cases are skipped: a voice cannot say which spelling was meant."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from speech_matrix import tts, voices  # noqa: PLC0415 - Windows-only helpers (System.Speech)

    from voxflow.app import read_wav_16k
    from voxflow.asr import Transcriber

    data = json.loads(path.read_text(encoding="utf-8"))
    voice = voice or voices()[0]
    asr = Transcriber()
    asr.load()
    print(f"voice {voice}, asr {asr.backend}")
    with tempfile.TemporaryDirectory(prefix="voxflow-grammar-") as td:
        for case in data["cases"]:
            if case["cat"] in ("homophone", "clean", "heldout"):
                case["asr"] = ""
                continue
            wav = Path(td) / f"{case['id']}.wav"
            tts(case["raw"], voice, wav)
            case["asr"] = asr.transcribe(read_wav_16k(str(wav)))
            print(f"{case['id']:16s} {case['asr']!r}")
    data["asr_voice"] = voice
    lines = ['{', f'  "about": {json.dumps(data["about"])},', f'  "asr_voice": {json.dumps(voice)},', '  "cases": [']
    lines.append(",\n".join("    " + json.dumps(c, ensure_ascii=False) for c in data["cases"]))
    lines += ["  ]", "}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--source", choices=("raw", "asr", "both"), default="both")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--freeze-asr", action="store_true")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.freeze_asr:
        freeze_asr(voice=a.voice)
        return 0
    from voxflow.cleanup import Cleaner

    c = Cleaner(device=a.device, use_llm=not a.rules_only)
    t = time.perf_counter()
    c.load()
    print(f"cleanup backend {c.backend}, load {time.perf_counter() - t:.1f} s")
    sources = ("raw", "asr") if a.source == "both" else (a.source,)
    s = evaluate(c, load_cases(), sources, a.quiet)
    print_summary(f"{'rules only' if a.rules_only else 'rules + LLM'} ({c.backend})", s)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
