"""Operating-system integration: launch at login, rotating log file, crash handler, opening folders."""
from __future__ import annotations

import datetime as _dt
import faulthandler
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path

log = logging.getLogger("voxflow.system")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "VoxFlow"
AUTOSTART_FLAG = "--autostart"
RUN_VALUE_LIMIT = 260        # Windows silently ignores longer Run entries


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
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        _file_handler = handler
    except OSError as exc:
        print(f"[voxflow] log file unavailable ({exc}); logging to the console only", file=sys.stderr)
    if sys.stderr is not None and not any(getattr(h, "_voxflow_console", False) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("[voxflow] %(levelname)s %(name)s: %(message)s"))
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


def write_crash_report(data_dir: str | Path, exc_type, exc, tb, where: str = "main thread") -> Path | None:
    """Write ``logs/crash-<timestamp>.log`` with the traceback. Returns its path (None if it failed)."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir(data_dir) / f"crash-{stamp}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"VoxFlow crash report {_dt.datetime.now().isoformat(timespec='seconds')}\n")
            fh.write(f"where: {where}\npython: {sys.version}\nplatform: {sys.platform}\n\n")
            # frames and error types only: a message can quote dictated text (see redacted_exc)
            fh.write(redacted_traceback(_with_tb(exc_type, exc, tb)) + "\n")
        return path
    except OSError:
        return None


def install_crash_handler(data_dir: str | Path, on_crash=None) -> None:
    """Log uncaught exceptions from any thread, write a crash report, then run ``on_crash()``.

    ``on_crash`` is the place to undo global side effects (unmute other apps, release the keyboard hook).
    Also enables ``faulthandler`` into ``logs/fault.log`` so native crashes leave a stack trace."""
    global _fault_file

    def report(exc_type, exc, tb, where: str) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        logging.getLogger("voxflow").critical("uncaught exception in %s\n%s", where,     # no message: see above
                                              redacted_traceback(_with_tb(exc_type, exc, tb)))
        path = write_crash_report(data_dir, exc_type, exc, tb, where)
        if path is not None:
            logging.getLogger("voxflow").critical("crash report written to %s", path)
        if on_crash is not None:
            try:
                on_crash()
            except Exception:  # noqa: BLE001
                logging.getLogger("voxflow").exception("crash clean-up failed")

    def excepthook(exc_type, exc, tb) -> None:
        report(exc_type, exc, tb, "main thread")
        sys.__excepthook__(exc_type, exc, tb)

    def thread_hook(args) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "unknown"
        report(args.exc_type, args.exc_value, args.exc_traceback, f"thread {name}")

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    try:
        if _fault_file is None:
            path = log_dir(data_dir) / "fault.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            _fault_file = path.open("a", encoding="utf-8")
            faulthandler.enable(_fault_file)
    except (OSError, RuntimeError, ValueError) as exc:
        log.debug("faulthandler not enabled: %s", exc)


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
