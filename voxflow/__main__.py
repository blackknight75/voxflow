"""python -m voxflow [--demo] [--fake] [--selftest [WAV]] [--settings] [--no-hook] [--autostart]"""
import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="voxflow", description="Local push-to-talk dictation")
    ap.add_argument("--demo", action="store_true",
                    help="show the whole app with simulated audio, models and data; no hotkeys, nothing is pasted")
    ap.add_argument("--fake", action="store_true", help="use stand-in audio/ASR/cleanup but real hotkeys + paste")
    ap.add_argument("--selftest", nargs="?", const="", metavar="WAV",
                    help="load everything, run the pipeline on WAV, print timings + text, exit")
    ap.add_argument("--settings", action="store_true", help="open the main window on start")
    ap.add_argument("--no-hook", action="store_true", help="do not install the global keyboard hook")
    ap.add_argument("--autostart", action="store_true",
                    help="started by Windows at login: stay in the tray without opening the window")
    # the app's own background speed check runs each measurement in a child started with this flag
    ap.add_argument("--speed-check", nargs="+", metavar="ARG", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.speed_check is not None:
        from .speedcheck import child_main
        return child_main(args.speed_check)
    if args.selftest is not None:
        from .app import selftest
        return selftest(args.selftest or None, fake=args.fake)
    from .app import run_app
    return run_app(demo=args.demo, fake=args.fake, open_settings=args.settings, no_hook=args.no_hook,
                   autostart=args.autostart)


if __name__ == "__main__":
    sys.exit(main())
