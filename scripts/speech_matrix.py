"""Speech beyond one clip: accuracy and latency of the real models on varied English and on human non-English speech.

    python scripts/speech_matrix.py [--skip-download] [--json out.json]

English (synthesised with the installed Windows voices, System.Speech):
  * every voice on a reference sentence        * a 60 s+ dictation
  * whisper-level gain, whisper mode off / on  * numbers, e-mail, list formatting (smart formatting on)
  * a noisy mix at several signal-to-noise ratios
Non-English (real human read speech): the first few clips of the FLEURS dev split for es, fr, de, it, pt, nl,
streamed from the public dataset on Hugging Face into a temp folder (test data only; only the first megabytes of
each archive are read).

Everything runs through the real ``Transcriber`` and ``Cleaner``; nothing touches the desktop.
"""
from __future__ import annotations

import argparse
import io  # noqa: F401
import json
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REFERENCE = ("Please schedule the meeting for Thursday afternoon, and send the report to the engineering team "
             "before the end of the day.")
LONG = (
    "Good morning everyone, and thank you for joining the weekly planning call. I want to start with a short "
    "summary of where we are. The new release went out on Tuesday, and so far the feedback from customers has "
    "been positive. Support tickets are down compared with last month, which is a good sign. There are still "
    "three open problems that we need to look at. The first one is the slow start on older laptops. The second "
    "one is the missing export option in the reports page. The third one is a crash that happens when the "
    "network connection drops during an upload. I would like the platform team to take the first problem, the "
    "reporting team to take the second, and I will look at the third one myself. Please write down what you "
    "find, and share it in the project channel before Friday. Next week we will also begin planning the work "
    "for the summer. If you have ideas for features, small or large, add them to the shared document. We will "
    "go through the list together, estimate the effort, and choose what fits. One more thing before we finish. "
    "The office will be closed on Monday because of the public holiday, so the next call moves to Wednesday at "
    "the usual time. Thank you all, and have a great rest of the day.")
FORMATTING = [
    ("the total is twenty three dollars and fifty cents", ["$23", "50 cents"]),
    ("the meeting starts at three thirty p m on january fifth", ["3:30 PM", "January 5th"]),
    ("please email john dot smith at example dot com", ["john.smith@example.com"]),
    ("about seventy five percent of the users upgraded", ["75%"]),
    ("number one milk number two eggs number three bread", ["1. Milk", "2. Eggs", "3. Bread"]),
    ("the file is called report dot pdf", ["report.pdf"]),
]
FLEURS = {"es": "es_419", "fr": "fr_fr", "de": "de_de", "it": "it_it", "pt": "pt_br", "nl": "nl_nl"}
HF = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"


# --------------------------------------------------------------------------- helpers
def norm_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE).replace("_", " ")
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = norm_words(reference), norm_words(hypothesis)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def tts(text: str, voice: str, path: Path, rate: int = 0) -> None:
    safe = text.replace("'", "''")
    ps = ("Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          f"$s.SelectVoice('{voice}');$s.Rate={rate};"
          "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
          f"$s.SetOutputToWaveFile('{path}',$f);$s.Speak('{safe}');$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=300)


def voices() -> list[str]:
    ps = ("Add-Type -AssemblyName System.Speech;(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
          ".GetInstalledVoices() | Where-Object {$_.Enabled} | ForEach-Object { $_.VoiceInfo.Name }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    return [v.strip() for v in out.stdout.splitlines() if v.strip()]


def read_any_wav(data: bytes):
    """16 kHz mono float32 from PCM (8/16/24/32 bit) or IEEE-float WAV bytes (the stdlib ``wave`` cannot read float)."""
    import numpy as np
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV file"
    pos, fmt, raw = 12, None, b""
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            fmt = (int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little"),
                   int.from_bytes(body[4:8], "little"), int.from_bytes(body[14:16], "little"))
        elif tag == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    kind, channels, rate, bits = fmt
    if kind == 3:
        a = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    elif bits == 16:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif bits == 24:
        b = np.frombuffer(raw[:len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        a = (np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0).astype(np.float32)
    else:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    if channels > 1:
        a = a[:len(a) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        x = np.arange(int(len(a) * 16000 / rate)) * (rate / 16000)
        a = np.interp(x, np.arange(len(a)), a).astype(np.float32)
    return a.astype(np.float32)


def fetch_fleurs(code: str, folder: Path, n: int) -> list[tuple[Path, str]]:
    """First ``n`` dev clips of one language with their reference transcripts (streamed, a few MB)."""
    config = FLEURS[code]
    folder.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{HF}/{config}/dev.tsv", headers={"User-Agent": "voxflow-test"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = [line.split("\t") for line in r.read().decode("utf-8").splitlines()]
    text_by_file = {row[1]: row[2] for row in rows if len(row) > 3}
    clips: list[tuple[Path, str]] = []
    req = urllib.request.Request(f"{HF}/{config}/audio/dev.tar.gz", headers={"User-Agent": "voxflow-test"})
    with urllib.request.urlopen(req, timeout=120) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or not name.endswith(".wav") or name not in text_by_file:
                continue
            data = tar.extractfile(member).read()
            target = folder / name
            target.write_bytes(data)
            seconds = len(read_any_wav(data)) / 16000
            if 4.0 <= seconds <= 20.0:
                clips.append((target, text_by_file[name]))
            if len(clips) >= n:
                break
    return clips


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="skip the non-English part")
    ap.add_argument("--only-languages", action="store_true", help="skip the English part")
    ap.add_argument("--clips", type=int, default=5, help="clips per language")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    import numpy as np

    from voxflow import app as appmod
    from voxflow import audio
    from voxflow.asr import Transcriber
    from voxflow.cleanup import Cleaner, is_english

    work = Path(tempfile.mkdtemp(prefix="voxflow-speech-"))
    asr, cleaner = Transcriber(), Cleaner()
    asr.load()
    cleaner.load()
    print(f"models: asr {asr.backend}, cleanup {cleaner.backend}; temp folder {work}")
    out: dict = {"backends": [asr.backend, cleaner.backend]}

    def run(clip, **clean_kw) -> dict:
        t = time.perf_counter()
        raw = asr.transcribe(clip)
        t_asr = time.perf_counter() - t
        t = time.perf_counter()
        final = cleaner.clean(raw, **clean_kw) if raw else ""
        t_clean = time.perf_counter() - t
        return {"raw": raw, "final": final, "asr_ms": round(t_asr * 1000), "clean_ms": round(t_clean * 1000),
                "seconds": round(len(clip) / 16000, 1), "info": dict(cleaner.last_info)}

    installed = [] if args.only_languages else voices()
    out["voices"] = installed
    print("voices:", installed)
    english = not args.only_languages

    # ---- every voice on the reference sentence
    out["voices_result"] = []
    base_clip = None
    for v in installed:
        p = work / f"ref_{len(out['voices_result'])}.wav"
        tts(REFERENCE, v, p)
        clip = appmod.read_wav_16k(str(p))
        base_clip = clip if base_clip is None else base_clip
        r = run(clip)
        r.update(voice=v, wer=round(wer(REFERENCE, r["raw"]), 3))
        out["voices_result"].append(r)
        print(f"voice {v}: {r['seconds']} s, WER {r['wer']:.1%}, asr {r['asr_ms']} ms -> {r['final']!r}")

    # ---- long dictation
    out["long"] = []
    for v in installed[:2]:
        p = work / f"long_{len(out['long'])}.wav"
        tts(LONG, v, p)
        clip = appmod.read_wav_16k(str(p))
        r = run(clip, tone="formal", smart=True)
        r.update(voice=v, words=len(LONG.split()), wer_raw=round(wer(LONG, r["raw"]), 3),
                 wer_final=round(wer(LONG, r["final"]), 3), total_ms=r["asr_ms"] + r["clean_ms"])
        out["long"].append(r)
        print(f"long ({v}): {r['seconds']} s, {r['words']} words, WER raw {r['wer_raw']:.1%} final "
              f"{r['wer_final']:.1%}, asr {r['asr_ms']} ms + cleanup {r['clean_ms']} ms = {r['total_ms']} ms, "
              f"cleanup info {r['info']}")

    # ---- whisper-level gain
    out["quiet"] = []
    for gain_db in ((-20, -40, -50, -60) if english else ()):
        quiet = (base_clip * (10 ** (gain_db / 20))).astype(np.float32)
        off = asr.transcribe(quiet)
        on = asr.transcribe(audio.normalize_gain(quiet))
        row = {"gain_db": gain_db, "peak": round(float(np.abs(quiet).max()), 5),
               "wer_off": round(wer(REFERENCE, off), 3), "wer_on": round(wer(REFERENCE, on), 3),
               "app_calls_it_silence": bool(float(np.abs(quiet).max()) < 1e-4)}
        out["quiet"].append(row)
        print(f"quiet {gain_db} dB (peak {row['peak']}): WER whisper mode off {row['wer_off']:.1%}, "
              f"on {row['wer_on']:.1%}" + ("  [below the app's silence gate]" if row["app_calls_it_silence"] else ""))

    # ---- noisy mix
    out["noise"] = []
    rng = np.random.default_rng(7)
    signal_rms = float(np.sqrt(np.mean(base_clip ** 2))) if english else 0.0
    for snr_db in ((30, 20, 10, 5, 0) if english else ()):
        noise = rng.normal(0, signal_rms / (10 ** (snr_db / 20)), base_clip.size).astype(np.float32)
        text = asr.transcribe(np.clip(base_clip + noise, -1, 1))
        row = {"snr_db": snr_db, "wer": round(wer(REFERENCE, text), 3), "text": text}
        out["noise"].append(row)
        print(f"noise SNR {snr_db} dB: WER {row['wer']:.1%}")

    # ---- formatting
    out["formatting"] = []
    for spoken, must in (FORMATTING if english else ()):
        p = work / f"fmt_{len(out['formatting'])}.wav"
        tts(spoken, installed[0], p)
        r = run(appmod.read_wav_16k(str(p)), tone="formal", smart=True)
        hit = [m for m in must if m.lower() in r["final"].lower()]
        r.update(spoken=spoken, expected=must, ok=len(hit) == len(must))
        out["formatting"].append(r)
        print(f"format {'OK ' if r['ok'] else 'MISS'} {spoken!r} -> raw {r['raw']!r} -> {r['final']!r} (want {must})")

    # ---- non-English human speech
    out["languages"] = {}
    if not args.skip_download:
        for code in FLEURS:
            try:
                clips = fetch_fleurs(code, work / code, args.clips)
            except Exception as exc:  # noqa: BLE001
                print(f"{code}: download failed ({type(exc).__name__}: {exc})")
                out["languages"][code] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            rows = []
            for path, reference in clips:
                clip = read_any_wav(path.read_bytes())
                r = run(clip, language="auto", tone="formal", smart=True)
                forced = cleaner.clean(r["raw"], language=code, tone="formal", smart=True) if r["raw"] else ""
                r.update(reference=reference, wer=round(wer(reference, r["raw"]), 3),
                         wer_final=round(wer(reference, r["final"]), 3), detected_english=is_english(r["raw"]),
                         final_forced_language=forced, wer_forced=round(wer(reference, forced), 3))
                rows.append(r)
            if not rows:
                out["languages"][code] = {"error": "no clip"}
                continue
            summary = {"clips": len(rows), "seconds": round(sum(r["seconds"] for r in rows), 1),
                       "wer_mean": round(statistics.mean(r["wer"] for r in rows), 3),
                       "wer_final_mean": round(statistics.mean(r["wer_final"] for r in rows), 3),
                       "wer_forced_mean": round(statistics.mean(r["wer_forced"] for r in rows), 3),
                       "asr_ms_median": round(statistics.median(r["asr_ms"] for r in rows)),
                       "clean_ms_median": round(statistics.median(r["clean_ms"] for r in rows)),
                       "mistaken_for_english": sum(1 for r in rows if r["detected_english"]), "rows": rows}
            out["languages"][code] = summary
            print(f"{code}: {summary['clips']} clips, {summary['seconds']} s, WER raw {summary['wer_mean']:.1%}, after "
                  f"cleanup (auto) {summary['wer_final_mean']:.1%}, (language set) {summary['wer_forced_mean']:.1%}, "
                  f"asr median {summary['asr_ms_median']} ms, cleanup median {summary['clean_ms_median']} ms, "
                  f"taken for English {summary['mistaken_for_english']}/{summary['clips']}")
            for r in rows[:2]:
                print(f"    ref : {r['reference']}\n    raw : {r['raw']}\n    out : {r['final']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SPEECH MATRIX DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
