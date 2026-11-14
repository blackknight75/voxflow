"""Entry point of the frozen app (PyInstaller). Same behaviour as ``python -m voxflow``."""
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # the windowed exe has no console: print() / argparse errors must not crash on a missing stdout
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    from voxflow.__main__ import main
    sys.exit(main())
