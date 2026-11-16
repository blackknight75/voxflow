"""App updates: check GitHub Releases, tell the user, and (only if they allow it) install the new version.

The one network call VoxFlow makes besides the model download. It is an anonymous HTTPS GET with the generic
User-Agent ``VoxFlow/<version>``: no ids, no cookies, no telemetry. With the setting "Off" nothing here touches
the network at all.

* Source: ``https://api.github.com/repos/blackknight75/voxflow/releases/latest`` (GitHub returns the newest
  *published* release only: drafts and pre-releases are never offered).
* Checked at most once a day, first a random few minutes after start, and on "Check now". Failures are quiet:
  they show in Settings > Updates and in the log only.
* Install: ``VoxFlow-Setup-<version>.exe`` is downloaded to a temp folder (resumed after an interruption), its
  SHA-256 is compared with the release's ``SHA256SUMS.txt`` and a mismatch is refused. Every URL, including each
  redirect, must be HTTPS on ``github.com/blackknight75/voxflow`` or GitHub's asset hosts. The installer is run
  silently (``/VERYSILENT /SUPPRESSMSGBOXES /NORESTART``) after VoxFlow has shut down, never while a dictation or
  a command is being recorded or processed. The installer is not code-signed: the checksum proves the file is the
  one published with the release, not who built it.
* A copy installed with Scoop or winget, or run from source, never replaces itself: it can only tell the user.

Pure helpers (version compare, URL allow-list, release parsing, checksums, the download, install-kind detection,
the schedule) are module functions so they can be tested without Qt or a network; :class:`Updater` is the Qt
service the controller owns.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import hashlib
import json
import logging
import os
import random
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QObject, QTimer, Signal

from . import __version__, config as cfgmod

log = logging.getLogger("voxflow.updater")

REPO = "blackknight75/voxflow"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
WINGET_ID = "blackknight75.VoxFlow"

MODES = cfgmod.UPDATE_MODES                  # ("auto", "notify", "off")
DEFAULT_MODE = cfgmod.DEFAULT_UPDATE_MODE    # "notify" (confirmed by the maintainer)

CHECK_EVERY_S = 24 * 3600                    # at most one automatic check a day
STARTUP_DELAY_S = (90, 600)                  # the first automatic check: a random moment in this window after start
REMIND_AFTER_S = 24 * 3600                   # "Remind me later" = tomorrow
TIMEOUT_S = 15.0                             # per connection / read
API_MAX_BYTES = 2 * 1024 * 1024
SUMS_MAX_BYTES = 64 * 1024
INSTALLER_MAX_BYTES = 2 * 1024 ** 3          # GitHub's limit for one release asset
NOTES_MAX_CHARS = 6000
CHUNK = 1024 * 1024
SCHEDULE_WINDOW_S = 3600                     # a 03:00 install may start until 03:59 (the PC may wake late)
IDLE_BEFORE_SCHEDULED_S = 600                # ...and only after 10 minutes without keyboard or mouse input
INSTALL_POLL_MS = 2000                       # "Install now" looks this often whether VoxFlow is idle yet
SCHEDULE_POLL_MS = 60_000
RETRY_BUSY_MS = 15 * 60_000                  # a daily check that found a download under way tries again this soon
ASSET_HOSTS = ("objects.githubusercontent.com", "release-assets.githubusercontent.com")
SUMS_NAME = "SHA256SUMS.txt"
STATE_FILE = "updates.json"
INSTALLER_SWITCHES = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
MANAGED_MARKER = "managed-by.txt"            # written by the installer when winget passes /MANAGEDBY=winget


def user_agent() -> str:
    return f"VoxFlow/{__version__}"


class UpdateError(Exception):
    """A check or a download that did not work; ``str()`` is a sentence for the user."""


class Cancelled(UpdateError):
    pass


# =========================================================================== versions
class Version(NamedTuple):
    major: int
    minor: int
    patch: int
    pre: tuple = ()

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.patch}"
        return core + ("-" + ".".join(str(p) for p in self.pre) if self.pre else "")


_SEMVER = re.compile(r"^[vV]?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
                     r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$")


def parse_version(text) -> Version | None:
    """``"v0.10.0"`` -> ``Version(0, 10, 0)``; ``None`` for anything that is not semantic versioning."""
    m = _SEMVER.match(str(text or "").strip())
    if not m:
        return None
    pre = tuple(int(p) if p.isdigit() else p for p in m.group(4).split(".")) if m.group(4) else ()
    return Version(int(m.group(1)), int(m.group(2)), int(m.group(3)), pre)


def _order(v: Version) -> tuple:
    # semver precedence: a pre-release sorts before its release; numeric identifiers before alphanumeric ones
    pre = tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in v.pre)
    return (v.major, v.minor, v.patch, 0 if v.pre else 1, pre)


def compare_versions(a, b) -> int:
    """-1, 0 or 1 like ``a <=> b``. Raises ValueError for a version that cannot be read."""
    va, vb = parse_version(a), parse_version(b)
    if va is None or vb is None:
        raise ValueError(f"not a version: {a if va is None else b!r}")
    ka, kb = _order(va), _order(vb)
    return (ka > kb) - (ka < kb)


def is_newer(candidate, current) -> bool:
    try:
        return compare_versions(candidate, current) > 0
    except ValueError:
        return False


# =========================================================================== where the bytes may come from
def allowed_url(url: str, kind: str = "asset") -> bool:
    """True for an HTTPS URL VoxFlow may fetch: ``kind="api"`` only this repository's API, ``"asset"`` only this
    repository's release downloads and the hosts GitHub redirects them to. No credentials, no other port."""
    try:
        p = urllib.parse.urlsplit(str(url))
        port = p.port
    except ValueError:
        return False
    if p.scheme != "https" or p.username or p.password or port not in (None, 443):
        return False
    host = (p.hostname or "").lower()
    if kind == "api":
        return host == "api.github.com" and p.path.startswith(f"/repos/{REPO}/")
    if host == "github.com":
        return p.path.startswith(f"/{REPO}/releases/download/")
    return host in ASSET_HOSTS


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another allowed URL of the same kind."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl, self.kind):
            raise UpdateError("The download was redirected to a server VoxFlow does not trust, so it was stopped.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Http:
    """The only door to the network: an anonymous HTTPS GET with certificate checks, no cookies, short timeouts.
    Tests replace it with a fake that has the same ``open(url, kind, headers)`` method."""

    def __init__(self, timeout: float = TIMEOUT_S):
        self.timeout = timeout

    def open(self, url: str, kind: str, headers: dict | None = None):
        """A response (``status``, ``headers``, ``read(n)``, context manager). Raises UpdateError."""
        if not allowed_url(url, kind):
            raise UpdateError("VoxFlow refused to contact a server it does not trust.")
        accept = "application/vnd.github+json" if kind == "api" else "application/octet-stream"
        req = urllib.request.Request(url, headers={"User-Agent": user_agent(), "Accept": accept, **(headers or {})})
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context()), _CheckedRedirect(kind))
        try:
            return opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:                         # a resumed range past the end: the caller starts over
                return exc
            raise UpdateError(_http_problem(exc.code)) from None
        except UpdateError:
            raise
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            raise UpdateError(f"GitHub could not be reached ({_short(reason)}).") from None


class OfflineHttp:
    """The network door of a fake, demo or test copy of the controller: always closed."""

    requests = 0

    def open(self, url: str, kind: str, headers: dict | None = None):
        raise UpdateError("This test copy of VoxFlow does not check for updates.")


def _http_problem(code: int) -> str:
    if code == 404:
        return "No published release was found on GitHub."
    if code in (403, 429):
        return "GitHub is limiting requests right now. VoxFlow will try again later."
    return f"GitHub answered with an error (HTTP {code})."


def _short(exc) -> str:
    text = str(getattr(exc, "strerror", None) or exc).strip() or type(exc).__name__
    return text[:120]


def _status(resp) -> int:
    return int(getattr(resp, "status", None) or getattr(resp, "code", 0) or 0)


def _read_limited(resp, limit: int) -> bytes:
    data = resp.read(limit + 1)
    if len(data) > limit:
        raise UpdateError("GitHub sent an unexpectedly large answer, so it was ignored.")
    return data


# =========================================================================== releases
@dataclass
class Release:
    version: str                # without the "v"
    tag: str
    title: str
    notes: str                  # plain text (see plain_notes)
    page: str                   # the release page on github.com
    published: str = ""
    installer: str = ""         # asset name, "" when the release has no Windows installer
    installer_url: str = ""
    size: int = 0
    sums_url: str = ""

    @property
    def installable(self) -> bool:
        return bool(self.installer_url and self.sums_url)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "Release | None":
        if not isinstance(data, dict):
            return None
        try:
            rel = cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
        except TypeError:
            return None
        if parse_version(rel.version) is None:
            return None
        for name in ("installer_url", "sums_url"):           # a hand-edited state file must not widen the list
            if getattr(rel, name) and not allowed_url(getattr(rel, name)):
                setattr(rel, name, "")
        return rel


def installer_name(version: str) -> str:
    return f"VoxFlow-Setup-{version}.exe"


def parse_release(data) -> Release:
    """A release from the GitHub API answer. Raises UpdateError for drafts, pre-releases and odd answers."""
    if not isinstance(data, dict):
        raise UpdateError("GitHub sent an answer VoxFlow does not understand.")
    if data.get("draft") or data.get("prerelease"):
        raise UpdateError("The newest release on GitHub is not a final release.")
    tag = str(data.get("tag_name") or "")
    version = parse_version(tag)
    if version is None or version.pre:
        raise UpdateError("The newest release on GitHub has no usable version number.")
    ver = str(version)
    page = str(data.get("html_url") or "")
    if not re.fullmatch(rf"https://github\.com/{REPO}/releases/tag/[0-9A-Za-z.+-]+", page):
        page = RELEASES_PAGE                            # the link the dialog offers: this repository only
    rel = Release(version=ver, tag=tag, title=str(data.get("name") or f"VoxFlow {ver}")[:120],
                  notes=plain_notes(data.get("body")), page=page, published=str(data.get("published_at") or ""))
    for asset in data.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name, url = str(asset.get("name") or ""), str(asset.get("browser_download_url") or "")
        if not allowed_url(url) or urllib.parse.urlsplit(url).hostname != "github.com":
            continue
        if name == installer_name(ver):
            size = asset.get("size")
            rel.installer, rel.installer_url = name, url
            rel.size = int(size) if isinstance(size, int) and not isinstance(size, bool) and size > 0 else 0
        elif name == SUMS_NAME:
            rel.sums_url = url
    if rel.size > INSTALLER_MAX_BYTES:
        rel.installer = rel.installer_url = ""
    return rel


_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_MD = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_TAG = re.compile(r"<[^>]{0,500}>")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁦-⁩]")


def plain_notes(body) -> str:
    """Release notes as plain text: no images (nothing remote is ever loaded), no HTML, links reduced to their
    text, list markers as bullets, at most ``NOTES_MAX_CHARS`` characters. The UI shows it as plain text."""
    text = str(body or "")[:NOTES_MAX_CHARS * 4].replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = _IMAGE_MD.sub("", text)
    text = _LINK_MD.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = _CONTROL.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)                 # headings: their text
        line = re.sub(r"^(\s*)[-*+]\s+", lambda m: m.group(1) + "• ", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        if line.strip() in ("---", "***", "___"):
            continue
        lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) > NOTES_MAX_CHARS:
        text = text[:NOTES_MAX_CHARS].rsplit("\n", 1)[0].rstrip() + "\n…"
    return text


def fetch_latest(http) -> Release:
    with http.open(API_LATEST, "api") as resp:
        body = _read_limited(resp, API_MAX_BYTES)
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise UpdateError("GitHub sent an answer VoxFlow does not understand.") from None
    return parse_release(data)


_SUM_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")


def parse_sums(text: str, name: str) -> str | None:
    """The SHA-256 of ``name`` in a ``sha256sum`` listing, lower case; None when it is not listed."""
    for line in str(text).splitlines():
        m = _SUM_LINE.match(line.strip())
        if m and m.group(2) == name:
            return m.group(1).lower()
    return None


def fetch_sum(http, release: Release) -> str:
    if not release.sums_url:
        raise UpdateError("This release has no SHA256SUMS.txt, so its installer cannot be checked and is not used.")
    with http.open(release.sums_url, "asset") as resp:
        text = _read_limited(resp, SUMS_MAX_BYTES).decode("utf-8", "replace")
    digest = parse_sums(text, release.installer)
    if digest is None:
        raise UpdateError("SHA256SUMS.txt does not list the installer, so it cannot be checked and is not used.")
    return digest


def sha256_file(path: Path, cancel=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                return h.hexdigest()
            if cancel is not None and cancel():
                raise Cancelled("The update was cancelled.")
            h.update(block)


def download_folder() -> Path:
    return Path(tempfile.gettempdir()) / "VoxFlow-update"


def download_installer(http, release: Release, folder: Path, progress=None, cancel=None) -> tuple[Path, str]:
    """Download the release's installer into ``folder`` and verify it. Returns ``(path, sha256)``.

    An interrupted download is resumed from its ``.part`` file; older installers and stray files in ``folder``
    are deleted. A file whose SHA-256 differs from ``SHA256SUMS.txt`` is deleted and UpdateError is raised."""
    if not release.installable:
        raise UpdateError("This release has no Windows installer to download.")
    folder.mkdir(parents=True, exist_ok=True)
    expected = fetch_sum(http, release)
    final = folder / release.installer
    part = folder / (release.installer + ".part")
    for stray in folder.iterdir():                     # old versions, a part of another version
        if stray.name not in (final.name, part.name):
            _remove(stray)
    if final.exists():
        if sha256_file(final, cancel) == expected:
            return final, expected
        _remove(final)
    offset = part.stat().st_size if part.exists() else 0
    if release.size and offset >= release.size:
        _remove(part)
        offset = 0
    resp = http.open(release.installer_url, "asset", {"Range": f"bytes={offset}-"} if offset else None)
    with resp:
        status = _status(resp)
        if status == 416:                               # the part was complete or stale: start over
            _remove(part)
            return download_installer(http, release, folder, progress, cancel)
        if status not in (200, 206):
            raise UpdateError(_http_problem(status))
        if status == 200:
            offset = 0
        total = release.size or (offset + int(resp.headers.get("Content-Length") or 0)) or 0
        done = offset
        with open(part, "ab" if offset else "wb") as fh:
            while True:
                if cancel is not None and cancel():
                    raise Cancelled("The update was cancelled.")
                block = resp.read(CHUNK)
                if not block:
                    break
                done += len(block)
                if done > INSTALLER_MAX_BYTES or (release.size and done > release.size):
                    fh.close()
                    _remove(part)
                    raise UpdateError("The download was larger than the release says, so it was deleted.")
                fh.write(block)
                if progress is not None:
                    progress(done, total)
    if release.size and done < release.size:
        raise UpdateError("The download stopped before the end. VoxFlow will continue it next time.")
    got = sha256_file(part, cancel)
    if got != expected:
        _remove(part)
        log.warning("update %s refused: SHA-256 %s, SHA256SUMS.txt says %s", release.version, got, expected)
        raise UpdateError("The downloaded installer does not match its published SHA-256 checksum. It was deleted "
                          "and will not be run.")
    os.replace(part, final)
    return final, expected


def _remove(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not delete %s: %s", path.name, exc.strerror or exc)


def clean_downloads(folder: Path) -> None:
    """Delete everything an earlier update left behind (after a skip, or once the new version runs)."""
    if folder.is_dir():
        for item in folder.iterdir():
            _remove(item)


# =========================================================================== how this copy was installed
INSTALL_KINDS = ("installer", "scoop", "winget", "source", "portable")


def install_kind(exe: str | Path | None = None, frozen: bool | None = None) -> str:
    """``installer`` (our Inno Setup install: it may update itself), ``scoop``, ``winget``, ``source`` (python -m
    voxflow) or ``portable`` (a frozen build that was not installed, e.g. ``dist\\VoxFlow``)."""
    if not (getattr(sys, "frozen", False) if frozen is None else frozen):
        return "source"
    app = Path(exe or sys.executable).parent
    try:
        app = app.resolve()
    except OSError:
        pass
    lowered = [p.lower() for p in app.parts]
    # Scoop: <scoop root>\apps\voxflow\<version or current>\VoxFlow.exe
    if len(lowered) >= 3 and lowered[-2] == "voxflow" and lowered[-3] == "apps":
        return "scoop"
    try:
        if (app / MANAGED_MARKER).read_text(encoding="utf-8", errors="replace").strip().lower() == "winget":
            return "winget"
    except OSError:
        pass
    return "installer" if (app / "unins000.exe").exists() else "portable"


def managed_text(kind: str) -> str:
    """Why this copy does not install updates itself ("" when it does)."""
    return {
        "scoop": "Updates are managed by Scoop: run scoop update voxflow.",
        "winget": f"Updates are managed by winget: run winget upgrade {WINGET_ID}.",
        "source": "VoxFlow is running from source: update it with git pull.",
        "portable": "This copy was not installed with the VoxFlow installer, so it cannot update itself. Get the "
                    "new version from the releases page.",
    }.get(kind, "")


# =========================================================================== Windows: metered connection, idle time
_NLM_FIXED, _NLM_VARIABLE = 0x2, 0x4
_NLM_OVERDATALIMIT, _NLM_ROAMING = 0x10000, 0x40000


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort),
                ("d4", ctypes.c_ubyte * 8)]


def connection_metered() -> bool | None:
    """True when Windows marks the internet connection as metered (or roaming / over its data limit), False when
    it does not, None when that cannot be told. INetworkCostManager through plain COM, no extra package."""
    if sys.platform != "win32":
        return None
    try:
        ole32 = ctypes.windll.ole32
        hr = ole32.CoInitializeEx(None, 0x2)                       # COINIT_APARTMENTTHREADED
        try:
            clsid, iid = _GUID(), _GUID()
            ole32.CLSIDFromString("{DCB00C01-570F-4A9B-8D69-199FDBA5723B}", ctypes.byref(clsid))  # NetworkListManager
            ole32.CLSIDFromString("{DCB00008-570F-4A9B-8D69-199FDBA5723B}", ctypes.byref(iid))    # INetworkCostManager
            obj = ctypes.c_void_p()
            if ole32.CoCreateInstance(ctypes.byref(clsid), None, 0x17, ctypes.byref(iid), ctypes.byref(obj)) != 0 \
                    or not obj.value:
                return None
            vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])
            get_cost = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong),
                                          ctypes.c_void_p)(vtbl[3])
            cost = ctypes.c_ulong(0)
            try:
                if get_cost(obj, ctypes.byref(cost), None) != 0:
                    return None
            finally:
                release(obj)
            return bool(cost.value & (_NLM_FIXED | _NLM_VARIABLE | _NLM_OVERDATALIMIT | _NLM_ROAMING))
        finally:
            if hr in (0, 1):
                ole32.CoUninitialize()
    except (OSError, AttributeError, ValueError):
        return None


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]


def idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input in this session; None when Windows cannot tell."""
    if sys.platform != "win32":
        return None
    try:
        info = _LASTINPUTINFO(ctypes.sizeof(_LASTINPUTINFO), 0)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        return ((ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0
    except (OSError, AttributeError):
        return None


# =========================================================================== schedule
def parse_hhmm(text) -> tuple[int, int]:
    m = re.fullmatch(r"\s*([01]?\d|2[0-3]):([0-5]\d)\s*", str(text or ""))
    if not m:
        raise ValueError("The install time must look like 03:00.")
    return int(m.group(1)), int(m.group(2))


def scheduled_slot(now: _dt.datetime, at: str) -> str | None:
    """The day (``YYYY-MM-DD``) of the install slot ``now`` falls in, or None outside every slot. A slot starts
    at ``at`` and lasts ``SCHEDULE_WINDOW_S`` (it may run past midnight)."""
    h, m = parse_hhmm(at)
    for day in (now.date(), now.date() - _dt.timedelta(days=1)):
        start = _dt.datetime.combine(day, _dt.time(h, m), tzinfo=now.tzinfo)
        if start <= now < start + _dt.timedelta(seconds=SCHEDULE_WINDOW_S):
            return day.isoformat()
    return None


# =========================================================================== the installer
def installer_command(path: Path, relaunch: bool, log_path: Path | None = None) -> list[str]:
    """Inno Setup, silent. ``/RELAUNCH=1`` makes packaging/voxflow.iss start VoxFlow again when it is done."""
    cmd = [str(path), *INSTALLER_SWITCHES, f"/RELAUNCH={1 if relaunch else 0}"]
    if log_path is not None:
        cmd.append(f"/LOG={log_path}")
    return cmd


def _child_env() -> dict:
    """The installer (and the VoxFlow it starts again) must not inherit this frozen process's PyInstaller state."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("_PYI", "_MEIPASS"))}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def launch_installer(path: Path, relaunch: bool, log_path: Path | None = None, popen=subprocess.Popen) -> bool:
    """Start the installer detached from VoxFlow (it outlives it). True when it started."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    cmd = installer_command(path, relaunch, log_path)
    for extra in ((0x01000000,) if sys.platform == "win32" else ()) + (0,):    # CREATE_BREAKAWAY_FROM_JOB first
        try:
            popen(cmd, close_fds=True, creationflags=flags | extra, env=_child_env(), cwd=str(path.parent),
                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("update: started %s (relaunch=%s)", path.name, relaunch)
            return True
        except OSError as exc:
            last = exc
    log.warning("update: the installer could not be started: %s", _short(last))
    return False


# =========================================================================== state between runs
@dataclass
class UpdateState:
    last_check: float = 0.0        # when the last check finished (time.time), 0 = never
    last_ok: bool = True
    last_result: str = ""          # a sentence for Settings > Updates
    latest: dict | None = None     # Release.to_dict() of the newest release seen
    skipped: str = ""              # "Skip this version"
    remind_after: float = 0.0      # "Remind me later": no notification before this
    notified: str = ""             # the version the user was last told about
    install_on_quit: str = ""      # version to install when VoxFlow is quit
    downloaded: str = ""           # version whose verified installer is in the download folder
    sha256: str = ""               # ...and its checksum
    slot_done: str = ""            # the scheduled slot (day) already used

    @classmethod
    def load(cls, path: Path) -> "UpdateState":
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return cls()
        st = cls()
        if not isinstance(data, dict):
            return st
        for f in fields(cls):                          # a hand-edited or damaged file: keep the default per field
            value, default = data.get(f.name), getattr(st, f.name)
            if f.name == "latest":
                ok = isinstance(value, dict)
            elif isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, float):
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                value = float(value) if ok else default
            else:
                ok = isinstance(value, str)
                value = value[:300] if ok else default
            if ok:
                setattr(st, f.name, value)
        return st

    def save(self, path: Path) -> None:
        try:
            cfgmod._atomic_write(path, json.dumps(asdict(self), indent=1))
        except OSError as exc:
            log.warning("could not save %s: %s", path.name, exc)


# =========================================================================== the service
class Updater(QObject):
    """Checks, downloads and installs, on the UI thread with background threads for the network.

    ``get(key)`` reads a setting, ``busy()`` says a dictation or command is under way (or a shortcut is being
    recorded). The controller connects ``changed`` (Settings > Updates), ``available`` (tell the user),
    ``notice`` and ``restart_requested`` (quit now: the installer runs when VoxFlow has shut down)."""

    changed = Signal(dict)
    available = Signal(dict)
    notice = Signal(str, str)
    restart_requested = Signal()
    _sig = Signal(str, object)

    def __init__(self, data_dir: Path, get, busy, *, version: str = __version__, http=None, kind: str | None = None,
                 metered=connection_metered, idle=idle_seconds, clock=time.time, now=None, launcher=launch_installer,
                 folder: Path | None = None, automatic: bool = True, rng: random.Random | None = None, parent=None):
        super().__init__(parent)
        self.version = version
        self._path = Path(data_dir) / STATE_FILE
        self._get, self._busy = get, busy
        self._http = http if http is not None else Http()
        self.kind = kind if kind is not None else install_kind()
        self._metered, self._idle, self._clock = metered, idle, clock
        self._now = now or (lambda: _dt.datetime.now().astimezone())
        self._launcher = launcher
        self.folder = Path(folder) if folder is not None else download_folder()
        self._automatic = automatic
        self._rng = rng or random.Random()
        self.state = UpdateState.load(self._path)
        self.phase = "idle"            # idle | checking | downloading | waiting (install now, for idle) | restarting
        self.progress: float | None = None
        self.error = ""
        self._after_download = ""      # "now" | "quit" | "auto": what the download in progress is for
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_launch: tuple[Path, bool] | None = None
        self._closed = False
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.timeout.connect(lambda: self.check(user=False))
        self._install_timer = QTimer(self)
        self._install_timer.setInterval(INSTALL_POLL_MS)
        self._install_timer.timeout.connect(self._install_tick)
        self._slot_timer = QTimer(self)
        self._slot_timer.setInterval(SCHEDULE_POLL_MS)
        self._slot_timer.timeout.connect(self._slot_tick)
        self._sig.connect(self._on_thread)
        self._forget_installed()

    # ------------------------------------------------------------------ facts
    @property
    def mode(self) -> str:
        return str(self._get("update_mode"))

    @property
    def can_install(self) -> bool:
        """This copy may replace itself (our installer put it here)."""
        return self.kind == "installer"

    def latest(self) -> Release | None:
        rel = Release.from_dict(self.state.latest)
        return rel if rel is not None and is_newer(rel.version, self.version) else None

    def info(self) -> dict:
        rel = self.latest()
        return {
            "version": self.version, "mode": self.mode, "when": str(self._get("update_install_when")),
            "time": str(self._get("update_time")), "kind": self.kind, "can_install": self.can_install,
            "managed": managed_text(self.kind), "phase": self.phase, "progress": self.progress,
            "error": self.error, "last_check": self.state.last_check, "last_ok": self.state.last_ok,
            "last_result": self.state.last_result,
            "available": rel.to_dict() if rel else None,
            "installable": bool(rel and rel.installable and self.can_install),
            "skipped": bool(rel and self.state.skipped == rel.version),
            "install_on_quit": bool(rel and self.state.install_on_quit == rel.version),
            "downloaded": bool(rel and self.state.downloaded == rel.version),
            "releases_page": rel.page if rel else RELEASES_PAGE,
        }

    def _emit(self) -> None:
        self.changed.emit(self.info())

    def _save(self) -> None:
        self.state.save(self._path)

    def _forget_installed(self) -> None:
        """After an update the old installer (and any reminder about it) is no longer needed."""
        st = self.state
        if st.downloaded and not is_newer(st.downloaded, self.version):
            clean_downloads(self.folder)
            st.downloaded = st.sha256 = ""
        if st.install_on_quit and not is_newer(st.install_on_quit, self.version):
            st.install_on_quit = ""

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Arm the daily check (nothing at all when the mode is Off or automatic checks are disabled)."""
        self._arm_check(first=True)
        self._arm_slot()

    def close(self) -> None:
        self._closed = True
        for t in (self._check_timer, self._install_timer, self._slot_timer):
            t.stop()
        self._cancel.set()

    def _arm_check(self, first: bool = False) -> None:
        self._check_timer.stop()
        if self._closed or not self._automatic or self.mode == "off":
            return
        due = self.state.last_check + CHECK_EVERY_S - self._clock()
        delay = max(due, 0.0)
        if first:
            delay = max(delay, self._rng.uniform(*STARTUP_DELAY_S))
        self._check_timer.start(int(min(delay, CHECK_EVERY_S) * 1000))

    def settings_changed(self, key: str) -> None:
        if key == "update_mode":
            if self.mode == "off":                     # Off: no network, nothing armed
                self._cancel.set()
                self._install_timer.stop()
                if self.phase == "waiting":
                    self.phase = "idle"
                self.state.install_on_quit = ""
                self._save()
            self._arm_check(first=True)
        if key in ("update_mode", "update_install_when", "update_time"):
            self._arm_slot()
            if self.mode == "auto":
                self._auto_next()
        self._emit()

    # ------------------------------------------------------------------ check
    def check(self, user: bool = True) -> bool:
        """Ask GitHub for the newest release in the background. False when not started (Off, or already busy)."""
        if self._closed or self.mode == "off":
            return False
        if self.phase != "idle":
            if not user and self._automatic:           # the daily check came during a download: look again later
                self._check_timer.start(RETRY_BUSY_MS)
            return False
        self.phase, self.error = "checking", ""
        self._emit()
        self._run(lambda: ("checked", (user, fetch_latest(self._http))), lambda exc: ("check_failed", (user, exc)))
        return True

    def _run(self, job, on_error) -> None:
        self._cancel.clear()

        def body():
            try:
                kind, payload = job()
            except Exception as exc:  # noqa: BLE001 - every failure is reported, none may escape the thread
                kind, payload = on_error(exc)
            if not self._closed:
                self._sig.emit(kind, payload)

        self._thread = threading.Thread(target=body, name="voxflow-update", daemon=True)
        self._thread.start()

    def _on_thread(self, kind: str, payload) -> None:
        if self._closed:
            return
        handler = getattr(self, "_on_" + kind)
        handler(*payload)

    def _on_checked(self, user: bool, rel: Release) -> None:
        st = self.state
        st.last_check, st.last_ok = self._clock(), True
        st.latest = rel.to_dict()
        newer = is_newer(rel.version, self.version)
        st.last_result = f"VoxFlow {rel.version} is available." if newer else "VoxFlow is up to date."
        log.info("update check: newest release %s, running %s", rel.version, self.version)
        self.phase = "idle"
        self._save()
        self._arm_check()
        self._emit()
        if not newer:
            return
        if self.mode == "auto" and self.can_install and rel.installable and st.skipped != rel.version:
            self._auto_next()                          # it tells the user what it will do (see _downloaded)
            if user:
                self.available.emit(self.info())
        else:
            self._announce(rel, user)

    def _on_check_failed(self, user: bool, exc: Exception) -> None:
        st = self.state
        st.last_check, st.last_ok = self._clock(), False
        st.last_result = str(exc) if isinstance(exc, UpdateError) else "The update check did not work."
        log.warning("update check failed: %s", exc if isinstance(exc, UpdateError) else type(exc).__name__)
        self.phase = "idle"
        self._save()
        self._arm_check()
        self._emit()

    def _announce(self, rel: Release, user: bool) -> None:
        """Tell the user once per version, unless they skipped it or asked to be reminded later. A check the user
        started themselves always shows the dialog."""
        st = self.state
        quiet = st.skipped == rel.version or self._clock() < st.remind_after or st.notified == rel.version
        if user or not quiet:
            st.notified = rel.version
            self._save()
            self.available.emit(self.info())

    # ------------------------------------------------------------------ the user's choices
    def remind_later(self) -> None:
        self.state.remind_after = self._clock() + REMIND_AFTER_S
        self.state.notified = ""                       # tell again after the pause
        self._save()
        self._emit()

    def skip(self) -> None:
        rel = self.latest()
        if rel is None:
            return
        st = self.state
        st.skipped, st.install_on_quit = rel.version, ""
        if self.phase == "downloading":
            self._cancel.set()
        clean_downloads(self.folder)
        st.downloaded = st.sha256 = ""
        self._install_timer.stop()
        if self.phase == "waiting":
            self.phase = "idle"
        self._save()
        self._emit()

    def install(self, when: str) -> None:
        """``when``: ``"now"`` (restart as soon as VoxFlow is idle) or ``"quit"``. Raises ValueError (a sentence)
        when this copy cannot install the update."""
        if when not in ("now", "quit"):
            raise ValueError("Choose to install now or when you quit VoxFlow.")
        rel = self.latest()
        if rel is None:
            raise ValueError("There is no newer version to install.")
        if not self.can_install:
            raise ValueError(managed_text(self.kind))
        if not rel.installable:
            raise ValueError("This release has no Windows installer. Get it from the releases page.")
        if self.phase in ("checking", "restarting"):
            raise ValueError("VoxFlow is busy with the update. Try again in a moment.")
        st = self.state
        if st.skipped == rel.version:
            st.skipped = ""
        if when == "quit":
            st.install_on_quit = rel.version
            if self.phase == "waiting":                # changed their mind: no restart now
                self._install_timer.stop()
                self.phase = "idle"
        self._save()
        self._after_download = when
        if st.downloaded == rel.version and self._file(rel).exists():
            self._downloaded(rel)
        elif self.phase != "downloading":
            self._download(rel)
        self._emit()

    # ------------------------------------------------------------------ download
    def _file(self, rel: Release) -> Path:
        return self.folder / rel.installer

    def _download(self, rel: Release) -> None:
        if self._closed or self.phase in ("checking", "downloading"):
            return
        self.phase, self.progress, self.error = "downloading", 0.0, ""
        self._emit()

        def progress(done, total):
            self._sig.emit("progress", (done / total if total else None,))

        self._run(lambda: ("download_done", (rel, download_installer(self._http, rel, self.folder, progress,
                                                                      self._cancel.is_set))),
                  lambda exc: ("download_failed", (rel, exc)))

    def _on_progress(self, fraction) -> None:
        if self.phase == "downloading":
            self.progress = fraction
            self._emit()

    def _on_download_done(self, rel: Release, result) -> None:
        _path, digest = result
        self.state.downloaded, self.state.sha256 = rel.version, digest
        self.phase, self.progress = "idle", None
        self._save()
        log.info("update %s downloaded and verified", rel.version)
        self._downloaded(rel)
        self._emit()

    def _on_download_failed(self, rel: Release, exc: Exception) -> None:
        self.phase, self.progress = "idle", None
        if isinstance(exc, Cancelled):
            self._after_download = ""
            self._emit()
            return
        self.error = str(exc) if isinstance(exc, UpdateError) else "The update could not be downloaded."
        log.warning("update %s: download failed: %s", rel.version,
                    exc if isinstance(exc, UpdateError) else type(exc).__name__)
        if self._after_download in ("now", "quit"):    # the user asked: say so (an automatic try stays quiet)
            self.notice.emit("warning", self.error)
        self._after_download = ""
        self._emit()

    def _downloaded(self, rel: Release) -> None:
        when, self._after_download = self._after_download, ""
        if when == "now":
            self.phase = "waiting"
            self._install_tick()
            if self.phase == "waiting":
                self._install_timer.start()
        elif when == "auto" and self._get("update_install_when") == "quit":
            self.state.install_on_quit = rel.version
            self._save()
            self.notice.emit("info", f"VoxFlow {rel.version} will be installed when you quit VoxFlow.")
        elif when == "auto":
            self.notice.emit("info", f"VoxFlow {rel.version} will be installed at {self._get('update_time')}, when "
                                     "VoxFlow and the PC are idle.")
        self._arm_slot()

    # ------------------------------------------------------------------ automatic mode
    def _auto_next(self) -> None:
        """Install automatically: fetch the installer in the background (not on a metered connection)."""
        rel = self.latest()
        if (rel is None or not self.can_install or not rel.installable or self.state.skipped == rel.version
                or self.phase != "idle"):
            return
        if self.state.downloaded == rel.version and self._file(rel).exists():
            self._after_download = "auto"
            self._downloaded(rel)
            return
        if self._metered() is True:
            self.error = "Waiting for a connection that is not metered to download the update."
            log.info("update %s: not downloaded automatically on a metered connection", rel.version)
            self._emit()
            return
        self._after_download = "auto"
        self._download(rel)

    def _arm_slot(self) -> None:
        rel = self.latest()
        armed = (not self._closed and self.mode == "auto" and self._get("update_install_when") == "time"
                 and rel is not None and self.state.downloaded == rel.version and self.state.skipped != rel.version)
        if armed and not self._slot_timer.isActive():
            self._slot_timer.start()
        elif not armed:
            self._slot_timer.stop()

    def _slot_tick(self) -> None:
        rel = self.latest()
        if rel is None or self.mode != "auto" or self._get("update_install_when") != "time":
            self._slot_timer.stop()
            return
        try:
            slot = scheduled_slot(self._now(), str(self._get("update_time")))
        except ValueError:
            return
        if slot is None or slot == self.state.slot_done or self._busy():
            return
        idle = self._idle()
        if idle is not None and idle < IDLE_BEFORE_SCHEDULED_S:
            return                                     # someone is at the PC: wait (the slot lasts an hour)
        self.state.slot_done = slot
        self._save()
        self._restart(rel)

    # ------------------------------------------------------------------ installing
    def _install_tick(self) -> None:
        rel = self.latest()
        if rel is None or self.phase != "waiting":
            self._install_timer.stop()
            return
        if self._busy():
            return                                     # never while recording, processing or in command mode
        self._install_timer.stop()
        self._restart(rel)

    def _restart(self, rel: Release) -> None:
        path = self._file(rel)
        if not path.exists():
            self.phase, self.error = "idle", "The downloaded installer is gone. Try again."
            self.state.downloaded = ""
            self._save()
            self._emit()
            return
        self._pending_launch = (path, True)
        self.phase = "restarting"
        self._emit()
        log.info("update %s: restarting VoxFlow to install it", rel.version)
        self.restart_requested.emit()

    def on_exit(self, user_quit: bool) -> bool:
        """VoxFlow is shutting down: start the installer if one is due. ``user_quit`` is False when Windows ends the
        session (no installer then). Verifies the file again right before it runs. True when it was started."""
        rel = self.latest()
        st = self.state
        if self._pending_launch is not None:
            path, relaunch = self._pending_launch
        elif user_quit and rel is not None and st.install_on_quit == rel.version and st.downloaded == rel.version:
            path, relaunch = self._file(rel), False
        else:
            return False
        self._pending_launch = None
        try:
            ok = bool(st.sha256) and path.exists() and sha256_file(path) == st.sha256
        except OSError:
            ok = False
        if not ok:
            log.warning("update: the installer changed on disk since it was verified; it is not run")
            _remove(path)
            st.downloaded = st.sha256 = ""
            self._save()
            return False
        st.install_on_quit = ""
        self._save()
        return bool(self._launcher(path, relaunch, self.folder / "install.log"))
