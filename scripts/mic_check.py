"""Real-microphone check: device, format, level, and (if the room allows) a speaker-to-microphone loopback.

    python scripts/mic_check.py [--wav tests/tts_sample.wav] [--no-loopback]

1. Lists the input devices VoxFlow would offer and the system default.
2. Records 3 s with the real ``voxflow.audio.Recorder`` and reports sample rate, dtype, length, peak and RMS.
3. Loopback: plays the clip through the default speakers while recording with the real Recorder, transcribes the
   recording with the real speech model and compares it with the transcript of the clean clip.

Nothing is written outside the temp folder; no keystroke or clipboard access.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default=str(ROOT / "tests" / "tts_sample.wav"))
    ap.add_argument("--no-loopback", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    import numpy as np
    import sounddevice as sd

    from scripts.e2e_selftest import word_accuracy
    from voxflow import app as appmod
    from voxflow import audio
    from voxflow.asr import Transcriber

    out: dict = {}
    devices = audio.list_input_devices()
    out["devices"] = [name for _i, name in devices]
    out["default"] = audio.default_input_name()
    print("input devices :", out["devices"])
    print("default input :", out["default"] or "(none)")
    if not devices:
        print("MIC CHECK: no input device")
        return 1

    def level(x) -> dict:
        x = np.asarray(x, dtype=np.float32)
        if not x.size:
            return {"seconds": 0.0, "peak": 0.0, "rms": 0.0}
        return {"seconds": round(x.size / audio.SAMPLE_RATE, 2), "peak": round(float(np.max(np.abs(x))), 5),
                "rms": round(float(np.sqrt(np.mean(x * x))), 6)}

    rec = audio.Recorder()
    try:
        if rec.error is not None:
            print("MIC CHECK: could not open the microphone:", audio.describe_mic_error(rec.error))
            return 1
        time.sleep(0.5)
        rec.start()
        time.sleep(3.0)
        quiet = rec.stop()
        out["capture"] = dict(level(quiet), dtype=str(quiet.dtype), ndim=int(quiet.ndim), alive=bool(rec.alive),
                              speech_seconds=round(rec.last_speech_seconds, 2), using_fallback=bool(rec.using_fallback),
                              meter=round(rec.level(), 4))
        print("3 s capture   :", out["capture"])
        ok = (quiet.dtype == np.float32 and quiet.ndim == 1 and abs(quiet.size / audio.SAMPLE_RATE - 3.3) < 0.4
              and rec.alive)
        out["capture_ok"] = bool(ok)
        out["capture_silent"] = bool(out["capture"]["peak"] < 1e-4)

        if not args.no_loopback:
            clip = appmod.read_wav_16k(args.wav)
            asr = Transcriber()
            asr.load()
            reference = asr.transcribe(clip)
            try:
                out_dev = sd.query_devices(kind="output")
                out["speaker"] = str(out_dev["name"])
            except Exception as exc:  # noqa: BLE001
                out["speaker"] = f"none ({exc})"
            print("speaker       :", out["speaker"])
            rec.start()
            t0 = time.time()
            try:
                sd.play(clip, audio.SAMPLE_RATE, blocking=True)
                played = True
            except Exception as exc:  # noqa: BLE001
                played = False
                out["play_error"] = str(exc)
            time.sleep(0.4)
            heard = rec.stop()
            out["loopback"] = dict(level(heard), played=played, wall_s=round(time.time() - t0, 2))
            text = asr.transcribe(audio.normalize_gain(heard)) if heard.size else ""
            out["loopback"]["reference"] = reference
            out["loopback"]["transcript"] = text
            out["loopback"]["accuracy_vs_clean"] = round(word_accuracy(reference, text), 3)
            gain = out["loopback"]["rms"] / max(out["capture"]["rms"], 1e-9)
            out["loopback"]["level_vs_room"] = round(gain, 2)
            print("loopback      :", {k: v for k, v in out["loopback"].items() if k not in ("reference", "transcript")})
            print("  reference   :", repr(reference))
            print("  heard       :", repr(text))
    finally:
        rec.close()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    verdict = "PASSED" if out.get("capture_ok") else "FAILED"
    if out.get("capture_silent"):
        verdict += " (format ok, but the microphone delivers digital silence)"
    print("MIC CHECK", verdict)
    return 0 if out.get("capture_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
