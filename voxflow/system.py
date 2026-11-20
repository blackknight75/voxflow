"""Operating-system integration: launch at login, rotating log file, crash handler, opening folders."""
from __future__ import annotations

import datetime as _dt
import faulthandler
import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

log = logging.getLogger("voxflow.system")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "VoxFlow"
AUTOSTART_FLAG = "--autostart"
RUN_VALUE_LIMIT = 260        # Windows silently ignores longer Run entries
SETUP_MUTEX = "VoxFlowSetupRunning"     # held by VoxFlow's Setup while it runs (SetupMutex in packaging/voxflow.iss)


# =========================================================================== launch at login
class RunKey:
    """The per-user "Run" registry key (HKCU). Replace ``voxflow.system.registry`` with a fake in tests."""

    def get(self, name: str) -> str | None:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as key:
                value, _kind = winreg.QueryValueEx(key, name)
                return str(value)
        except FileNotFoundError:
            return None

    def set(self, name: str, value: str) -> None:
        import winreg
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete(self, name: str) -> None:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass


registry = RunKey() if sys.platform == "win32" else None


def launch_command(executable: str | None = None, root: str | Path | None = None) -> str:
    """The command line that starts VoxFlow at login (window-less interpreter, marked as an autostart)."""
    exe = Path(executable or sys.executable)
    if getattr(sys, "frozen", False) and executable is None:
        return f'"{exe}" {AUTOSTART_FLAG}'
    if exe.name.lower() == "python.exe" and (exe.with_name("pythonw.exe")).exists():
        exe = exe.with_name("pythonw.exe")                 # no console window at login
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    folder = str(root)
    if '"' in folder or "'" in folder:
        raise ValueError("VoxFlow can't start at login from a folder whose name contains quotes.")
    # a raw string cannot end in a backslash (drive root "C:\"): forward slashes are just as valid there
    literal = f"'{root.as_posix()}'" if folder.endswith("\\") else f"r'{folder}'"
    command = _login_command(exe, literal)
    if len(command) > RUN_VALUE_LIMIT:
        # Deep install path. When the interpreter lives inside the install folder (the usual .venv layout)
        # the folder can be derived from sys.executable instead of being spelled out a second time.
        try:
            depth = len(exe.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth:
            command = _login_command(exe, f"sys.executable.rsplit('\\\\',{depth})[0]")
    return command


def _login_command(exe: Path, folder_expr: str) -> str:
    code = (f"import runpy,sys;sys.path[:0]=[{folder_expr}];sys.argv+=['{AUTOSTART_FLAG}'];"
            "runpy.run_module('voxflow',None,'__main__')")
    return f'"{exe}" -c "{code}"'


def autostart_enabled() -> bool:
    """True if a VoxFlow entry exists in the user's Run key."""
    if registry is None:
        return False
    try:
        return bool(registry.get(RUN_VALUE))
    except OSError as exc:
        log.warning("could not read the launch-at-login setting: %s", exc)
        return False


_SOURCE_FOLDER = re.compile(r"sys\.path\[:0\]=\[r?'([^']+)'\]")


def autostart_target(command: str) -> tuple[str, str]:
    """(program, source folder or "") that a Run entry starts: the quoted (or first) word, and for a copy run from
    source the folder named in its ``-c`` code (see ``launch_command``)."""
    command = command.strip()
    if command.startswith('"'):
        exe = command[1:].split('"', 1)[0]
    else:
        exe = command.split(" ", 1)[0]
    m = _SOURCE_FOLDER.search(command)
    return exe, (m.group(1) if m else "")


def autostart_state() -> tuple[str, str]:
    """What the Run entry does, compared with this copy: ("off" | "this" | "missing" | "other", the program it starts).
    "missing": the program (or source folder) it names is gone, e.g. the install folder moved. "other": it starts
    another copy of VoxFlow that exists."""
    if registry is None:
        return "off", ""
    try:
        value = registry.get(RUN_VALUE)
    except OSError as exc:
        log.warning("could not read the launch-at-login setting: %s", exc)
        return "off", ""
    if not value:
        return "off", ""
    try:
        mine = launch_command()
    except ValueError:
        mine = ""
    if value.strip().lower() == mine.lower():
        return "this", ""
    exe, folder = autostart_target(value)
    if not Path(exe).is_file() or (folder and not Path(folder).is_dir()):
        return "missing", folder or exe
    return "other", folder or exe


def set_autostart(enabled: bool) -> None:
    """Add or remove the Run entry. Raises ValueError with a human message if Windows refuses."""
    if registry is None:
        raise ValueError("Launch at login is only available on Windows.")
    try:
        if enabled:
            command = launch_command()                     # ValueError (human message) if it cannot work
            if len(command) > RUN_VALUE_LIMIT:
                raise ValueError("VoxFlow is installed in a folder whose path is too long for Windows to "
                                 "start it at login. Move it to a shorter path and try again.")
            registry.set(RUN_VALUE, command)
        else:
            registry.delete(RUN_VALUE)
    except OSError as exc:
        log.warning("could not change the launch-at-login setting: %s", exc)
        raise ValueError("Windows did not let VoxFlow change the launch-at-login setting.") from None


# =========================================================================== logging
LOG_NAME = "voxflow.log"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"
_file_handler: logging.Handler | None = None
_fault_file = None


def log_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "logs"


class RedactingFormatter(logging.Formatter):
    """Prints the stack of a logged exception (``log.exception``, ``exc_info=True``) without its message, which can
    quote dictated text (see :func:`redacted_exc`). Used by every VoxFlow handler, so no future call can leak."""

    def formatException(self, ei) -> str:
        exc = ei[1] if ei and isinstance(ei[1], BaseException) else None
        return redacted_traceback(exc) if exc is not None else super().formatException(ei)


def setup_logging(data_dir: str | Path, level: int = logging.INFO) -> Path:
    """Send the ``voxflow`` loggers to ``<data_dir>/logs/voxflow.log`` (1 MB x 3 files). Idempotent.

    Returns the log file path. If the folder cannot be written, logging continues on stderr only."""
    global _file_handler
    root = logging.getLogger("voxflow")
    root.setLevel(level)
    path = log_dir(data_dir) / LOG_NAME
    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8",
                                                       delay=True)
        handler.setFormatter(RedactingFormatter(_FORMAT))
        root.addHandler(handler)
        _file_handler = handler
    except OSError as exc:
        print(f"[voxflow] log file unavailable ({exc}); logging to the console only", file=sys.stderr)
    if sys.stderr is not None and not any(getattr(h, "_voxflow_console", False) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(RedactingFormatter("[voxflow] %(levelname)s %(name)s: %(message)s"))
        console._voxflow_console = True
        root.addHandler(console)
    return path


def shutdown_logging() -> None:
    """Detach and close the log file (so a temporary data folder can be removed)."""
    global _file_handler, _fault_file
    if _file_handler is not None:
        logging.getLogger("voxflow").removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    if _fault_file is not None:
        try:
            faulthandler.disable()
            _fault_file.close()
        except (OSError, ValueError):
            pass
        _fault_file = None


# =========================================================================== dictated text stays out of the log
def redacted_exc(exc: BaseException) -> str:
    """``KeyError at cleanup.py:812 in _fix`` for an exception raised while handling dictated text.

    The message is left out on purpose: it can quote the words being processed (a ``KeyError`` names the word it
    looked up, a ``ValueError`` may repeat its input), and the log file must never hold dictated text. An
    ``OSError`` keeps its reason and file name, which are about the disk, never about the words."""
    tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    where = f" at {Path(tb[-1].filename).name}:{tb[-1].lineno} in {tb[-1].name}" if tb else ""
    detail = ""
    if isinstance(exc, OSError):
        detail = ": " + " ".join(str(p) for p in (exc.strerror, exc.filename) if p)
    return f"{type(exc).__name__}{where}{detail}"


def redacted_traceback(exc: BaseException) -> str:
    """The stack of ``exc`` (file, line, code) without any exception message, chained causes included the way
    ``traceback.format_exception`` prints them; see :func:`redacted_exc`."""
    chain, seen, cur = [], set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ if cur.__cause__ is not None else (None if cur.__suppress_context__ else cur.__context__)
    parts = []
    for i, e in enumerate(reversed(chain)):                       # oldest cause first, as Python prints it
        if i:                                                     # how ``e`` is linked to the one printed above
            parts.append("\nThe above exception was the direct cause of the following exception:\n\n"
                         if e.__cause__ is not None else
                         "\nDuring handling of the above exception, another exception occurred:\n\n")
        frames = traceback.format_list(traceback.extract_tb(e.__traceback__)) if e.__traceback__ else []
        parts.append("Traceback (most recent call last):\n" + "".join(frames) + redacted_exc(e) + "\n")
    return "".join(parts).rstrip("\n")


# =========================================================================== crash handling
def _with_tb(exc_type, exc, tb) -> BaseException:
    """The exception of an ``excepthook`` triple, carrying its traceback (a hook may be handed a bare type)."""
    if not isinstance(exc, BaseException):
        exc = exc_type() if isinstance(exc_type, type) and issubclass(exc_type, BaseException) else Exception()
    if tb is not None and exc.__traceback__ is None:
        exc.__traceback__ = tb
    return exc


CRASH_KEEP = 10                  # crash-*.log files kept; older ones are deleted when a new one is written
FAULT_LOG_MAX = 1_000_000        # above this size fault.log is started afresh (the old one kept as fault.old.log)
_SEEN = "reports-seen.json"      # the crash reports the user was already told about (see new_crash_report)


def write_crash_report(data_dir: str | Path, exc_type, exc, tb, where: str = "main thread") -> Path | None:
    """Write ``logs/crash-<timestamp>.log`` with the traceback. Returns its path (None if it failed).
    Only the ``CRASH_KEEP`` newest reports are kept."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir(data_dir) / f"crash-{stamp}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"VoxFlow crash report {_dt.datetime.now().isoformat(timespec='seconds')}\n")
            fh.write(f"where: {where}\npython: {sys.version}\nplatform: {sys.platform}\n\n")
            # frames and error types only: a message can quote dictated text (see redacted_exc)
            fh.write(redacted_traceback(_with_tb(exc_type, exc, tb)) + "\n")
    except OSError:
        return None
    prune_crash_reports(data_dir)
    return path


def prune_crash_reports(data_dir: str | Path, keep: int = CRASH_KEEP) -> None:
    """Delete all but the ``keep`` newest ``crash-*.log`` files (their names sort by time)."""
    reports = sorted(log_dir(data_dir).glob("crash-*.log"), key=lambda p: p.name, reverse=True)
    for old in reports[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def new_crash_report(data_dir: str | Path) -> Path | None:
    """The newest crash report written since the previous call, i.e. by a run that crashed; None otherwise.

    Reports are ``crash-*.log`` (a Python error that ended VoxFlow) and new text in ``fault.log`` that names a
    fatal error (a native crash). The first call on a profile only takes note of what is already there."""
    folder = log_dir(data_dir)
    seen_path = folder / _SEEN
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
        seen_time, seen_fault = float(seen["time"]), int(seen["fault"])
    except (OSError, ValueError, KeyError, TypeError):
        seen, seen_time, seen_fault = None, 0.0, 0
    found: Path | None = None
    newest = seen_time
    for report in folder.glob("crash-*.log"):
        mtime = report.stat().st_mtime
        if mtime > newest:
            found, newest = report, mtime
    fault = folder / "fault.log"
    try:
        fault_size = fault.stat().st_size
    except OSError:
        fault_size = 0
    if found is None and fault_size != seen_fault:
        try:
            with fault.open("rb") as fh:
                fh.seek(seen_fault if fault_size > seen_fault else 0)     # smaller: started afresh, read it all
                if b"fatal" in fh.read().lower():
                    found = fault
        except OSError:
            pass
    try:
        folder.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps({"time": max(newest, time.time()), "fault": fault_size}), encoding="utf-8")
    except OSError:
        pass
    return found if seen is not None else None


def _cap_fault_log(path: Path) -> None:
    try:
        if path.stat().st_size > FAULT_LOG_MAX:
            os.replace(path, path.with_name("fault.old.log"))
    except OSError:
        pass


def install_crash_handler(data_dir: str | Path, on_crash=None, on_error=None, fatal=None) -> None:
    """Log uncaught exceptions from any thread; for one that ends VoxFlow also write a crash report and run
    ``on_crash()``.

    ``on_crash`` is the place to undo global side effects for good (unmute other apps, release the keyboard hook).
    Most uncaught exceptions do not end VoxFlow: one in a Qt slot is reported while the event loop keeps running,
    and one in a thread ends only that thread. For those ``on_error()`` runs instead (save what is pending, give
    other apps their sound back, switch nothing off) and no crash report is written. ``fatal()`` says whether an
    exception on the main thread ends the app (default: it always does). Also enables ``faulthandler`` into
    ``logs/fault.log`` (started afresh above about 1 MB) so native crashes leave a stack trace."""
    global _fault_file

    def report(exc_type, exc, tb, where: str, is_fatal: bool) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        text = redacted_traceback(_with_tb(exc_type, exc, tb))            # no message: see redacted_exc
        root = logging.getLogger("voxflow")
        if is_fatal:
            root.critical("uncaught exception in %s\n%s", where, text)
            path = write_crash_report(data_dir, exc_type, exc, tb, where)
            if path is not None:
                root.critical("crash report written to %s", path)
        else:
            root.error("uncaught exception in %s (VoxFlow keeps running)\n%s", where, text)
        callback = on_crash if is_fatal else on_error
        if callback is not None:
            try:
                callback()
            except Exception:  # noqa: BLE001
                root.exception("crash clean-up failed")

    def excepthook(exc_type, exc, tb) -> None:
        report(exc_type, exc, tb, "main thread", fatal is None or bool(fatal()))
        sys.__excepthook__(exc_type, exc, tb)

    def thread_hook(args) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "unknown"
        report(args.exc_type, args.exc_value, args.exc_traceback, f"thread {name}", False)

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    try:
        if _fault_file is None:
            path = log_dir(data_dir) / "fault.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            _cap_fault_log(path)
            _fault_file = path.open("a", encoding="utf-8")
            faulthandler.enable(_fault_file)
    except (OSError, RuntimeError, ValueError) as exc:
        log.debug("faulthandler not enabled: %s", exc)


# =========================================================================== an update being installed
def update_setup_running() -> bool:
    """True while VoxFlow's Setup is running (it holds ``SETUP_MUTEX``), e.g. an update the user chose to install
    when quitting. A VoxFlow started then would run from half-replaced files, and Setup would close it anyway."""
    if sys.platform != "win32":
        return False
    import ctypes
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)       # a private copy: argtypes set here stay here
        k32.OpenMutexW.restype = ctypes.c_void_p
        k32.OpenMutexW.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_wchar_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        for name in (SETUP_MUTEX, "Global\\" + SETUP_MUTEX):
            handle = k32.OpenMutexW(0x00100000, False, name)       # SYNCHRONIZE: only asks whether it exists
            if handle:
                k32.CloseHandle(handle)
                return True
    except (OSError, AttributeError):
        pass
    return False


UPDATE_MESSAGE_VAR = "VOXFLOW_UPDATE_MESSAGE"     # the message box's text, handed to the helper process


def tell_update_running(popen=None) -> None:
    """Say why VoxFlow does not start right now, and return at once.

    The message box is shown by a Windows PowerShell started from System32, not by this process: a VoxFlow.exe that
    stayed open in its own message box would keep its exe and DLLs in the install folder locked while Setup is
    replacing them, and a silent Setup then gives up on the locked file and rolls the whole update back (seen with
    Inno Setup 6.7.3: "DeleteFile: The existing file appears to be in use", then "Defaulting to Abort", exit code 5).
    Without PowerShell nothing is shown; VoxFlow still exits."""
    message = ("Finishing the update…\n\nVoxFlow is being updated right now. Start it again in a minute, when the "
               "update has finished.")
    log.info("an update is being installed: not starting")
    if sys.platform != "win32":
        print(message)
        return
    import subprocess
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    shell = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    script = ("Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show("
              f"$env:{UPDATE_MESSAGE_VAR}, 'VoxFlow', 'OK', 'Information')")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("_PYI", "_MEIPASS"))}
    env[UPDATE_MESSAGE_VAR] = message
    try:
        (popen or subprocess.Popen)(
            [str(shell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
             "-Command", script], cwd=root, env=env, close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, ValueError) as exc:
        log.info("could not show the update message: %s", exc)


# =========================================================================== misc
def open_folder(path: str | Path) -> bool:
    """Show ``path`` in the file manager. False (and a log line) if that is not possible."""
    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(target))  # noqa: S606 - opening the user's own data folder
            return True
    except OSError as exc:
        log.warning("could not open %s: %s", path, exc)
    return False


def import_roots() -> list[str]:
    """The folders Python imports from, without the current directory and relative entries. Native libraries are
    looked for under these (``nvidia/...`` of the CUDA wheels): a folder of that name in whatever directory VoxFlow
    was started from must never be put on the DLL search path (security review S-17; source runs only, the
    installed program's path has no such entry)."""
    here = os.path.normcase(os.path.abspath(os.getcwd()))
    return [p for p in sys.path if p and os.path.isabs(p) and os.path.normcase(os.path.abspath(p)) != here]
