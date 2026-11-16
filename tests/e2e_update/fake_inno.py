"""A fake Inno Setup build of VoxFlow, to rehearse .github/scripts/installer_smoke.ps1 on a PC where VoxFlow must
not be installed. One one-file exe plays every part, chosen by its own file name:

* ``VoxFlow-Setup-<v>.exe``: "installs" copies of itself as VoxFlow.exe, VoxFlow-console.exe and unins000.exe
  into ``FAKE_INNO_DIR``, writes the uninstall key under ``HKCU\\<FAKE_INNO_REG>\\Uninstall\\<AppId>_is1``, writes
  the /LOG= file, first ends every VoxFlow.exe running from that folder (what Restart Manager does with
  CloseApplications=force) and, with /VERYSILENT and /RELAUNCH=1, starts VoxFlow.exe again (the [Run] entry).
* ``VoxFlow.exe``: stays running (like the tray app) until it is ended.
* ``VoxFlow-console.exe``: ``--help`` prints a usage line; anything else exits 0.
* ``unins000.exe``: hands over to a copy of itself in %TEMP% (``_iu<pid>.exe``, as Inno does), which removes the
  folder, the key and a Run value that points into the folder.

Nothing outside FAKE_INNO_DIR and HKCU\\<FAKE_INNO_REG> is touched.
"""
import os
import shutil
import subprocess
import sys
import time
import winreg
from pathlib import Path

APP_ID = "{6C0B0B5E-5A9E-4C57-9F1D-4F4F0E2A7B31}"
me = Path(sys.executable)
name = me.name.lower()
target = Path(os.environ.get("FAKE_INNO_DIR", ""))
reg = os.environ.get("FAKE_INNO_REG", "")
args = sys.argv[1:]


def arg(prefix, default=""):
    for a in args:
        if a.upper().startswith(prefix.upper()):
            return a[len(prefix):].strip('"')
    return default


def uninstall_key():
    return rf"{reg}\Uninstall\{APP_ID}_is1"


def end_running_copies():
    exe = str(target / "VoxFlow.exe").lower()
    ps = ("Get-Process -Name VoxFlow -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.ToLower() "
          f"-eq '{exe}' }} | Stop-Process -Force")
    shell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    subprocess.run([str(shell), "-NoProfile", "-Command", ps], capture_output=True, timeout=60, check=True)


if not reg or not str(target) or "voxflowsmokerehearsal" not in reg.lower():
    sys.exit(3)                                           # refuses to run outside a rehearsal

if name.startswith("voxflow-setup-"):
    log = arg("/LOG=")
    lines = [f"Setup started: {args}"]
    end_running_copies()
    lines.append("Shutting down applications using our files.")
    target.mkdir(parents=True, exist_ok=True)
    for n in ("VoxFlow.exe", "VoxFlow-console.exe", "unins000.exe"):
        for _ in range(40):
            try:
                shutil.copy2(me, target / n)
                break
            except PermissionError:
                time.sleep(0.25)
    version = me.stem[len("VoxFlow-Setup-"):]
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, uninstall_key(), 0, winreg.KEY_WRITE) as k:
        winreg.SetValueEx(k, "DisplayVersion", 0, winreg.REG_SZ, version)
        winreg.SetValueEx(k, "InstallLocation", 0, winreg.REG_SZ, str(target) + "\\")
    lines.append("Installation process succeeded.")
    if log:
        Path(log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    silent = any(a.upper() == "/VERYSILENT" for a in args)
    if silent and arg("/RELAUNCH=", "0") == "1" and os.environ.get("FAKE_INNO_NO_RELAUNCH") != "1":
        subprocess.Popen([str(target / "VoxFlow.exe")], close_fds=True,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    sys.exit(0)

if name == "voxflow.exe":
    time.sleep(3600)
    sys.exit(0)

if name == "voxflow-console.exe":
    if "--help" in args:
        print("usage: voxflow [--demo] [--fake] [--selftest [WAV]]")
    sys.exit(0)

if name == "unins000.exe":
    log = arg("/LOG=")
    worker = Path(os.environ.get("TEMP", ".")) / f"_iu{os.getpid()}.exe"
    shutil.copy2(me, worker)
    subprocess.Popen([str(worker), f"/LOG={log}" if log else "/NOLOG"], close_fds=True,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    sys.exit(0)

if name.startswith("_iu"):
    time.sleep(1.0)                                       # the unins000.exe that started this has exited
    end_running_copies()
    for _ in range(40):
        shutil.rmtree(target, ignore_errors=True)
        if not target.exists():
            break
        time.sleep(0.25)
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, uninstall_key())
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{reg}\Run", 0, winreg.KEY_ALL_ACCESS) as k:
            value, _t = winreg.QueryValueEx(k, "VoxFlow")
            if str(target).lower() + "\\" in str(value).lower():
                winreg.DeleteValue(k, "VoxFlow")
    except OSError:
        pass
    log = arg("/LOG=")
    if log:
        Path(log).write_text("Uninstall process succeeded.\n", encoding="utf-8")
    sys.exit(0)

sys.exit(2)
