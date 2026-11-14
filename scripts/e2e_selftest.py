"""End-to-end check with the REAL models on a recorded clip, without touching the desktop.

    python scripts/e2e_selftest.py                 # GPU (auto) then forced CPU
    python scripts/e2e_selftest.py --device cpu    # one device only
    python scripts/e2e_selftest.py --wav clip.wav --expect "what was said"

What it does, per device:
  1. loads the speech model and the cleanup model (times both, reports the backend actually used);
  2. runs ``app.run_pipeline`` on the clip several times (ASR + cleanup timings, raw and final text);
  3. runs the whole product path through the real ``Controller``: hotkey start -> recorder (plays the clip) ->
     hotkey release -> ASR -> cleanup -> dictionary/snippets -> paste, and reads the release-to-paste latency
     the app itself records. The microphone, clipboard, keyboard, mixer and registry are stand-ins, the data
     folder is a temp dir, Qt runs offscreen: nothing on the desktop is touched;
  4. records every outgoing network connection attempted while all of this runs (expected: none once the
     models are cached).

Exit code 0 = text accurate and (for the GPU run) latency within the design target of 1.5 s for a ~10 s clip.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = ROOT / "tests" / "tts_sample.wav"
DEFAULT_TEXT = ("The quick brown fox jumps over the lazy dog. Please schedule the meeting for Thursday "
                "at three thirty in the afternoon, and send the report to the engineering team before the end of the day.")
TARGET_S = 1.5            # design target: release-to-paste under ~1.5 s for a 10 s utterance
CLEANUP_TARGET_MS = 400   # design target: cleanup under 400 ms for ~40 words on the GPU
# A disfluent utterance (41 words) that needs the language model: fillers, a repeat, a self-correction.
MESSY = ("um so i think we should uh move the the meeting to thursday no wait friday at three thirty and uh send "
         "the report to the engineering team before the end of the day you know so they can review it")
MESSY_MUST_KEEP = ("meeting", "friday", "report", "engineering", "team", "review")
MESSY_MUST_DROP = ("um", "uh", "thursday", "the the")
COMMAND_TEXT = "hey bob can u send me the report by friday thx"


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def word_accuracy(expected: str, got: str) -> float:
    """1 - WER (word-level Levenshtein distance / reference length), floored at 0."""
    ref, hyp = words(expected), words(got)
    if not ref:
        return 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / len(ref))


def run_device(device: str, wav: Path, expect: str, runs: int) -> dict:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    sys.path.insert(0, str(ROOT))

    connections: list[str] = []
    real_connect = socket.socket.connect

    def spy_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            connections.append(str(address))
        return real_connect(self, address)

    socket.socket.connect = spy_connect

    import numpy as np
    from PySide6.QtWidgets import QApplication

    from voxflow import app as appmod
    from voxflow import fakes
    from voxflow.asr import Transcriber
    from voxflow.cleanup import Cleaner
    from voxflow.config import Config
    from voxflow.context import AppContext
    from voxflow.controller import Controller

    cfg = Config()
    audio = appmod.read_wav_16k(str(wav))
    seconds = len(audio) / 16000
    out: dict = {"device": device, "clip_s": round(seconds, 2), "expect": expect}

    asr = Transcriber(cfg.asr_model, device)
    t = time.perf_counter()
    asr.load()
    out["asr_load_s"] = round(time.perf_counter() - t, 2)
    out["asr_backend"] = asr.backend
    cleaner = Cleaner(cfg.cleanup_model, "cpu" if device == "cpu" else "auto", cfg.cleanup_timeout_s)
    t = time.perf_counter()
    cleaner.load()
    out["cleanup_load_s"] = round(time.perf_counter() - t, 2)
    out["cleanup_backend"] = cleaner.backend

    # ---- 2. bare pipeline
    asr_ms, clean_ms = [], []
    res = {}
    for _ in range(runs):
        res = appmod.run_pipeline(audio, asr, cleaner, True, tone="formal", terms=[], language="auto", smart=True)
        asr_ms.append(res["t_asr"] * 1000)
        clean_ms.append(res["t_clean"] * 1000)
    out["pipeline"] = {"asr_ms": [round(v) for v in asr_ms], "cleanup_ms": [round(v) for v in clean_ms],
                       "asr_median_ms": round(statistics.median(asr_ms)),
                       "cleanup_median_ms": round(statistics.median(clean_ms)),
                       "raw": res.get("raw", ""), "final": res.get("text", ""),
                       "cleanup_info": dict(getattr(cleaner, "last_info", {}))}
    out["raw_accuracy"] = round(word_accuracy(expect, res.get("raw", "")), 3)

    # ---- 2b. the language-model path: a disfluent transcript that the rules alone cannot fix
    llm_ms, llm_out, llm_info = [], "", {}
    for _ in range(runs):
        t = time.perf_counter()
        llm_out = cleaner.clean(MESSY, tone="formal", terms=[], language="auto", smart=True)
        llm_ms.append((time.perf_counter() - t) * 1000)
        llm_info = dict(getattr(cleaner, "last_info", {}))
    low = " " + " ".join(words(llm_out)) + " "
    out["llm_cleanup"] = {"ms": [round(v) for v in llm_ms], "median_ms": round(statistics.median(llm_ms)),
                          "words_in": len(MESSY.split()), "out": llm_out, "info": llm_info,
                          "kept": all(f" {w} " in low for w in MESSY_MUST_KEEP),
                          "dropped": all(f" {w} " not in low for w in MESSY_MUST_DROP)}

    # ---- 2c. command mode with the real model
    from voxflow.command import CommandError, CommandMode
    cmd = CommandMode(cleaner, fakes.FakeInjector(selection=COMMAND_TEXT), cfg.command_timeout_s)
    if cmd.available:
        t = time.perf_counter()
        try:
            rewritten = cmd.run("make this more formal", COMMAND_TEXT, tone="formal")
            out["command"] = {"ms": round((time.perf_counter() - t) * 1000), "out": rewritten, "error": ""}
        except CommandError as exc:
            out["command"] = {"ms": round((time.perf_counter() - t) * 1000), "out": "", "error": str(exc)}
    else:
        out["command"] = {"ms": 0, "out": "", "error": "language model not loaded"}

    # ---- 3. the product path through the Controller
    class ClipRecorder(fakes.FakeRecorder):
        """A microphone that 'hears' the clip."""

        def stop(self):
            was = self._rec
            self._rec = False
            self._last_speech_s = seconds if was else 0.0
            return audio.copy() if was else np.zeros(0, dtype=np.float32)

        @property
        def hit_cap(self):
            return False

    qapp = QApplication.instance() or QApplication([])
    data_dir = Path(tempfile.mkdtemp(prefix="voxflow-e2e-"))
    injector = fakes.FakeInjector()
    ctl = Controller(data_dir=data_dir, install_hook=False, modules=(ClipRecorder, asr, cleaner), injector=injector,
                     sound_backend=lambda wav_bytes: None, mute_backend=fakes.FakeMuteBackend(),
                     context_fn=lambda: AppContext("winword.exe", "Report.docx - Word", 1),
                     autostart=fakes.FakeAutostart(False))
    entries: list[dict] = []
    ctl.transcript_ready.connect(entries.append)
    failures: list[str] = []
    ctl.notice.connect(lambda level, message: failures.append(f"{level}: {message}") if level == "error" else None)
    ctl.start()

    def pump(until, timeout_s: float) -> bool:
        end = time.time() + timeout_s
        while time.time() < end:
            qapp.processEvents()
            if until():
                return True
            time.sleep(0.005)
        return False

    ready = pump(lambda: ctl.model_status("asr")["state"] == "ready"
                 and ctl.model_status("cleanup")["state"] in ("ready", "degraded", "failed"), 120)
    out["controller_models"] = {n: ctl.model_status(n) for n in ("asr", "cleanup")}
    latencies = []
    if ready:
        for i in range(runs):
            ctl._on_hotkey("dictate", "start")
            ctl._t_start -= seconds                       # the key was held for the length of the clip
            n = len(entries)
            ctl._on_hotkey("dictate", "stop")
            if not pump(lambda: len(entries) > n or ctl.state == "error", 60):
                failures.append(f"run {i + 1}: no result within 60 s")
                break
            if len(entries) > n:
                latencies.append(entries[-1]["latency_ms"])
            pump(lambda: ctl.state == "idle", 5)
    else:
        failures.append("models did not become ready within 120 s")
    # worst case: the same clip, but the recogniser's text is the disfluent one, so the language model runs too
    class MessyAsr:
        backend = asr.backend

        def load(self):
            pass

        def transcribe(self, clip):
            asr.transcribe(clip)                          # pay the real recognition time
            return MESSY

    messy_latencies = []
    clean_pasted = injector.pasted[-1] if injector.pasted else ""
    if ready:
        ctl._asr = MessyAsr()
        for i in range(runs):
            ctl._on_hotkey("dictate", "start")
            ctl._t_start -= seconds
            n = len(entries)
            ctl._on_hotkey("dictate", "stop")
            if not pump(lambda: len(entries) > n or ctl.state == "error", 60):
                failures.append(f"messy run {i + 1}: no result within 60 s")
                break
            if len(entries) > n:
                messy_latencies.append(entries[-1]["latency_ms"])
            pump(lambda: ctl.state == "idle", 5)
    out["controller_llm"] = {"latency_ms": messy_latencies,
                             "latency_median_ms": round(statistics.median(messy_latencies)) if messy_latencies else None,
                             "pasted": injector.pasted[-1] if messy_latencies else ""}
    ctl.shutdown()
    out["controller"] = {"latency_ms": latencies,
                         "latency_median_ms": round(statistics.median(latencies)) if latencies else None,
                         "latency_worst_ms": max(latencies) if latencies else None,
                         "pasted": clean_pasted,
                         "history_entry": {k: entries[-1][k] for k in ("words", "wpm", "duration_s", "app_exe", "mode")}
                         if entries else {}, "errors": failures}
    out["final_accuracy"] = round(word_accuracy(expect, clean_pasted), 3)
    out["network_connections"] = connections
    for p in sorted(data_dir.rglob("*"), reverse=True):
        try:
            p.unlink() if p.is_file() else p.rmdir()
        except OSError:
            pass
    return out


def verdict(result: dict) -> tuple[bool, list[str]]:
    notes = []
    ok = True
    med = result["controller"]["latency_median_ms"]
    if result["controller"]["errors"] or med is None:
        ok = False
        notes.append("pipeline errors: " + "; ".join(result["controller"]["errors"]))
    if result["raw_accuracy"] < 0.9:
        ok = False
        notes.append(f"speech recognition accuracy {result['raw_accuracy']:.0%} is below 90 %")
    if result["final_accuracy"] < 0.75:
        ok = False
        notes.append(f"final text keeps only {result['final_accuracy']:.0%} of the reference words")
    if med is not None:
        within = med <= TARGET_S * 1000
        notes.append(f"release-to-paste median {med} ms vs the {TARGET_S * 1000:.0f} ms target ({result['clip_s']} s "
                     "clip): " + ("within target" if within else "OVER target"))
        if result["device"] != "cpu" and not within:
            ok = False
    llm, cl = result["llm_cleanup"], result["controller_llm"]
    gpu = result["device"] != "cpu"
    if not llm["info"].get("llm"):
        notes.append("the language model was NOT used for the disfluent sentence (rules only)")
        ok = ok and not gpu
    elif llm["info"].get("timeout") or llm["info"].get("rejected"):
        notes.append(f"language-model output was not used ({llm['info']}); the rule result was pasted instead")
    if not (llm["kept"] and llm["dropped"]):
        notes.append("disfluent sentence: fillers / the corrected word were not all removed, or content was lost")
        ok = ok and not gpu
    notes.append(f"LLM cleanup of {llm['words_in']} words: median {llm['median_ms']} ms "
                 f"(GPU target {CLEANUP_TARGET_MS} ms): "
                 + ("within target" if llm["median_ms"] <= CLEANUP_TARGET_MS else "over target"))
    if gpu and llm["median_ms"] > CLEANUP_TARGET_MS:
        ok = False
    if cl["latency_median_ms"] is not None:
        within = cl["latency_median_ms"] <= TARGET_S * 1000
        notes.append(f"release-to-paste WITH the language model: median {cl['latency_median_ms']} ms vs "
                     f"{TARGET_S * 1000:.0f} ms: " + ("within target" if within else "OVER target"))
        ok = ok and (within or not gpu)
    if result["command"]["error"]:
        notes.append("command mode: " + result["command"]["error"])
        ok = ok and not gpu
    if result["network_connections"]:
        notes.append(f"{len(result['network_connections'])} network connection(s) were attempted: "
                     + ", ".join(sorted(set(result["network_connections"]))[:5]))
    else:
        notes.append("no network connection was attempted")
    return ok, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default=str(DEFAULT_WAV))
    ap.add_argument("--expect", default=DEFAULT_TEXT, help="what is said in the clip (for the accuracy check)")
    ap.add_argument("--device", choices=["auto", "cuda", "webgpu", "cpu", "both"], default="both")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", default="", help="write the raw results to this file")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child:
        print("RESULT " + json.dumps(run_device(args.device, Path(args.wav), args.expect, args.runs)))
        return 0

    devices = ["auto", "cpu"] if args.device == "both" else [args.device]
    results, all_ok = [], True
    for device in devices:                                # one process per device: clean GPU / RAM state
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--device", device, "--wav", args.wav,
               "--expect", args.expect, "--runs", str(args.runs)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"== device {device}: FAILED to run ==\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
            all_ok = False
            continue
        r = json.loads(line[len("RESULT "):])
        results.append(r)
        ok, notes = verdict(r)
        all_ok &= ok
        p, c = r["pipeline"], r["controller"]
        print(f"== device {device}: {'PASS' if ok else 'FAIL'} ==")
        print(f"  clip                 {r['clip_s']} s")
        print(f"  speech model         backend {r['asr_backend']}, load {r['asr_load_s']} s")
        print(f"  cleanup model        backend {r['cleanup_backend']}, load {r['cleanup_load_s']} s")
        print(f"  ASR per run          {p['asr_ms']} ms (median {p['asr_median_ms']})")
        print(f"  cleanup per run      {p['cleanup_ms']} ms (median {p['cleanup_median_ms']}), info {p['cleanup_info']}")
        print(f"  release-to-paste     {c['latency_ms']} ms (median {c['latency_median_ms']}, worst {c['latency_worst_ms']})")
        print(f"  raw transcript       {p['raw']!r}")
        print(f"  pasted text          {c['pasted']!r}")
        print(f"  accuracy             raw {r['raw_accuracy']:.1%}, final {r['final_accuracy']:.1%}")
        print(f"  LLM cleanup per run  {r['llm_cleanup']['ms']} ms (median {r['llm_cleanup']['median_ms']}), "
              f"info {r['llm_cleanup']['info']}")
        print(f"  LLM cleanup output   {r['llm_cleanup']['out']!r}")
        print(f"  with LLM, end to end {r['controller_llm']['latency_ms']} ms "
              f"(median {r['controller_llm']['latency_median_ms']}) -> {r['controller_llm']['pasted']!r}")
        print(f"  command mode         {r['command']['ms']} ms -> {r['command']['out']!r} {r['command']['error']}")
        print(f"  history entry        {c['history_entry']}")
        for note in notes:
            print(f"  - {note}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("E2E SELFTEST", "PASSED" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
