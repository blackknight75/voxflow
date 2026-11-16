"""A stand-in for VoxFlow-Setup-<version>.exe (built into a one-file exe by harness.build_stub_installer).

It records how it was started (argv, working folder, a few environment facts) as JSON in the file named by
``VOXFLOW_E2E_RECORD`` and exits 0. It installs nothing.
"""
import json
import os
import sys
import time

record = os.environ.get("VOXFLOW_E2E_RECORD")
if record:
    data = {
        "argv": sys.argv[1:],
        "exe": sys.executable,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        # the stub is itself a PyInstaller exe whose bootloader consumes PYINSTALLER_RESET_ENVIRONMENT, so the
        # test checks the stripping of the parent's PyInstaller state with a canary variable instead
        "meipass2": os.environ.get("_MEIPASS2"),
        "e2e_canary": os.environ.get("VOXFLOW_E2E_CANARY"),
        "time": time.time(),
    }
    tmp = record + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, record)
sys.exit(0)
