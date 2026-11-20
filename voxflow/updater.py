"""App updates: check GitHub Releases, tell the user, and (only if they allow it) install the new version.

The one network call VoxFlow makes besides the model download. It is an anonymous HTTPS GET with the generic
User-Agent ``VoxFlow/<version>``: no ids, no cookies, no telemetry. With the setting "Off" nothing here touches
the network by itself (no check at start, none daily): only a click on "Check now" asks GitHub, once.

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
* Nothing is started that cannot finish: free disk space is checked before the download and before Setup, and the
  verified installer is held open from the moment VoxFlow is asked to quit until Setup runs. A Setup run that
  leaves the version unchanged is reported at the next start and repeated automatically only once; a release
  that GitHub no longer publishes is dropped, with its download.
* A copy installed with Scoop or winget, or run from source, never replaces itself: it can only tell the user.

Pure helpers (version compare, URL allow-list, release parsing, checksums, the download, install-kind detection,
the schedule) are module functions so they can be tested without Qt or a network; :class:`Updater` is the Qt
service the controller owns.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QObject, QTimer, Signal

from . import __version__, config as cfgmod

# urllib.request (with http.client, email and ssl behind it) is imported where a request is made, not here: it cost
# about a tenth of the controller's import time, at every start, for a check that runs once a day (QA R-A).

log = logging.getLogger("voxflow.updater")

REPO = "blackknight75/voxflow"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
WINGET_ID = "blackknight75.VoxFlow"

MODES = cfgmod.UPDATE_MODES                  # ("auto", "notify", "off")

CHECK_EVERY_S = 24 * 3600                    # at most one automatic check a day
RETRY_FAILED_S = 3600                        # ... but one that failed (offline at wake-up) is tried again after an hour
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
TIDY_RETRY_MS = 5 * 60_000                   # a cleanup that met a file still in use (Setup) tries again this soon
TIDY_RETRIES = 6
ASSET_HOSTS = ("objects.githubusercontent.com", "release-assets.githubusercontent.com")
SUMS_NAME = "SHA256SUMS.txt"
STATE_FILE = "updates.json"
INSTALLER_SWITCHES = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
MANAGED_MARKER = "managed-by.txt"            # written by the installer when winget passes /MANAGEDBY=winget
SETUP_LOG = "install.log"                    # Setup's /LOG, in the download folder
MAX_AUTO_ATTEMPTS = 2                        # Setup runs that left the version unchanged before automatic mode stops
DOWNLOAD_SPARE_BYTES = 100 * 1024 ** 2       # left free on the temp drive besides the installer itself
# Setup unpacks about 2.2 times the installer's size (3.8 GB from 1.8 GB) and keeps the old program files until the
# new ones are complete (packaging/voxflow.iss), so that much must be free on the install drive.
UNPACKED_FACTOR = 2.5


def user_agent() -> str:
    return f"VoxFlow/{__version__}"


class UpdateError(Exception):
    """A check or a download that did not work; ``str()`` is a sentence for the user. ``status`` is the HTTP status
    when GitHub answered with an error, else 0."""

    status = 0


class NoRelease(UpdateError):
    """GitHub answered, and there is no final release to offer: none is published (``status`` 404), or the newest
    one is a draft, a pre-release or has a tag that is not a version. The check itself worked, so it is not tried
    again an hour later like one that failed."""


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
    if not _plain_path(p.path):
        return False
    host = (p.hostname or "").lower()
    if kind == "api":
        return host == "api.github.com" and p.path.startswith(f"/repos/{REPO}/")
    if host == "github.com":
        return p.path.startswith(f"/{REPO}/releases/download/")
    return host in ASSET_HOSTS


def _plain_path(path: str) -> bool:
    """A URL path the prefix checks can trust: no ``..`` or ``.`` segments, no backslashes, no empty segments and
    no encoded dot, slash or backslash (``%2e``, ``%2f``, ``%5c``), which a server could decode into one."""
    low = path.lower()
    if any(bad in low for bad in ("%2e", "%2f", "%5c", "\\", "//")):
        return False
    return all(seg not in (".", "..") for seg in path.split("/"))


def release_asset_url(url: str) -> bool:
    """A download link as the API lists it: on github.com, in this repository's releases (the CDN hosts are only
    reached through GitHub's own redirect)."""
    return allowed_url(url) and urllib.parse.urlsplit(str(url)).hostname == "github.com"


@functools.cache
def checked_redirect():
    """The urllib handler class that follows a redirect only to another allowed URL of the same kind. Built on
    first use, because its base class is in urllib.request."""
    import urllib.request

    class CheckedRedirect(urllib.request.HTTPRedirectHandler):
        def __init__(self, kind: str):
            super().__init__()
            self.kind = kind

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if not allowed_url(newurl, self.kind):
                raise UpdateError("The download was redirected to a server VoxFlow does not trust, so it was "
                                  "stopped.")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return CheckedRedirect


class Http:
    """The only door to the network: an anonymous HTTPS GET with certificate checks, no cookies, short timeouts.
    Tests replace it with a fake that has the same ``open(url, kind, headers)`` method."""

    def __init__(self, timeout: float = TIMEOUT_S):
        self.timeout = timeout

    def open(self, url: str, kind: str, headers: dict | None = None):
        """A response (``status``, ``headers``, ``read(n)``, context manager). Raises UpdateError."""
        if not allowed_url(url, kind):
            raise UpdateError("VoxFlow refused to contact a server it does not trust.")
        import ssl
        import urllib.error
        import urllib.request
        accept = "application/vnd.github+json" if kind == "api" else "application/octet-stream"
        req = urllib.request.Request(url, headers={"User-Agent": user_agent(), "Accept": accept, **(headers or {})})
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context()), checked_redirect()(kind))
        try:
            return opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:                         # a resumed range past the end: the caller starts over
                return exc
            problem = UpdateError(_http_problem(exc.code))
            problem.status = exc.code
            raise problem from None
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
            if getattr(rel, name) and not release_asset_url(getattr(rel, name)):
                setattr(rel, name, "")
        if rel.installer and rel.installer != installer_name(rel.version):
            rel.installer = rel.installer_url = ""
        if not _own_page(rel.page):                           # the "Releases page" button opens this address
            rel.page = RELEASES_PAGE
        return rel


def _own_page(page) -> bool:
    """True for the page of a release of this repository, the only address the dialog's link may open. Checked for
    the API answer and for the copy read back from updates.json: a state file edited by another program made the
    button open any address or local file (security review S-12)."""
    return isinstance(page, str) and (page == RELEASES_PAGE or bool(
        re.fullmatch(rf"https://github\.com/{REPO}/releases/tag/[0-9A-Za-z.+-]+", page)))


def installer_name(version: str) -> str:
    return f"VoxFlow-Setup-{version}.exe"


def parse_release(data) -> Release:
    """A release from the GitHub API answer. Raises NoRelease for a draft, a pre-release or a tag that is not a
    version, UpdateError for an answer that cannot be read."""
    if not isinstance(data, dict):
        raise UpdateError("GitHub sent an answer VoxFlow does not understand.")
    if data.get("draft") or data.get("prerelease"):
        raise NoRelease("The newest release on GitHub is not a final release.")
    tag = str(data.get("tag_name") or "")
    version = parse_version(tag)
    if version is None or version.pre:
        raise NoRelease("The newest release on GitHub has no usable version number.")
    ver = str(version)
    page = str(data.get("html_url") or "")
    if not _own_page(page):
        page = RELEASES_PAGE                            # the link the dialog offers: this repository only
    rel = Release(version=ver, tag=tag, title=str(data.get("name") or f"VoxFlow {ver}")[:120],
                  notes=plain_notes(data.get("body")), page=page, published=str(data.get("published_at") or ""))
    for asset in data.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name, url = str(asset.get("name") or ""), str(asset.get("browser_download_url") or "")
        if not release_asset_url(url):
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
# real HTML elements only: a placeholder such as "VoxFlow/<version>" is text and stays (the dialog shows plain text,
# so nothing it keeps is ever rendered)
_HTML_ELEMENTS = ("a|abbr|audio|b|base|blockquote|body|br|button|canvas|center|code|dd|del|details|div|dl|dt|em|"
                  "embed|font|form|h[1-6]|head|hr|html|i|iframe|img|input|ins|kbd|li|link|meta|noscript|object|ol|"
                  "p|picture|pre|q|s|samp|script|section|small|source|span|strike|strong|style|sub|summary|sup|svg|"
                  "table|tbody|td|template|textarea|tfoot|th|thead|title|tr|tt|u|ul|var|video|wbr")
_TAG = re.compile(rf"</?(?:{_HTML_ELEMENTS})(?=[\s/>])[^>]{{0,500}}>", re.I)
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
    try:
        resp = http.open(API_LATEST, "api")
    except UpdateError as exc:
        if exc.status != 404:
            raise
        gone = NoRelease(str(exc))                      # no published release (any more): see Updater._withdrawn
        gone.status = 404
        raise gone from None
    with resp:
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
    with open(path, "rb") as fh:
        return sha256_stream(fh, cancel)


def sha256_stream(fh, cancel=None) -> str:
    h = hashlib.sha256()
    while True:
        block = fh.read(CHUNK)
        if not block:
            return h.hexdigest()
        if cancel is not None and cancel():
            raise Cancelled("The update was cancelled.")
        h.update(block)


def open_exclusive(path: Path):
    """Open ``path`` for reading so that, while it is open, nobody can write, rename or delete it (Windows: share
    mode read only; Setup can still be started from it). The installer is hashed through this handle and the
    handle is kept until Setup runs, so the file that runs is the file that was checked."""
    if sys.platform != "win32":
        return open(path, "rb")
    import msvcrt
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)      # a private copy: argtypes set here stay here
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                                ctypes.c_ulong, ctypes.c_void_p]
    handle = k32.CreateFileW(str(path), 0x80000000, 0x1, None, 3, 0x80, None)   # GENERIC_READ, FILE_SHARE_READ
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb")


def download_folder() -> Path:
    return Path(tempfile.gettempdir()) / "VoxFlow-update"


def free_bytes(path: Path) -> int | None:
    """Free space on the drive that holds ``path`` (asked of the nearest folder that exists); None when it cannot
    be told."""
    for folder in (path, *path.parents):
        try:
            return shutil.disk_usage(folder).free
        except OSError:
            continue
    return None


def _size(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1000 * 1024 ** 2 else f"{max(1, round(n / 1024 ** 2))} MB"


def space_problem(what: str, needed: int, folder: Path, free=free_bytes) -> str:
    """A sentence when ``folder``'s drive has less than ``needed`` bytes free ("" when there is room, or when the
    free space cannot be told). ``what``: "download" or "install"."""
    have = free(folder)
    if have is None or have >= needed:
        return ""
    return (f"There is not enough free disk space to {what} the update: {_size(needed)} is needed in {folder}, "
            f"{_size(have)} is free.")


def download_installer(http, release: Release, folder: Path, progress=None, cancel=None,
                       _restarted: bool = False, free=free_bytes) -> tuple[Path, str]:
    """Download the release's installer into ``folder`` and verify it. Returns ``(path, sha256)``.

    An interrupted download is resumed from its ``.part`` file; older installers and stray files in ``folder``
    are deleted. A file whose SHA-256 differs from ``SHA256SUMS.txt`` is deleted and UpdateError is raised. So is
    one for which the drive has no room (``free``: see :func:`free_bytes`), before a byte of it is fetched."""
    if not release.installable:
        raise UpdateError("This release has no Windows installer to download.")
    folder.mkdir(parents=True, exist_ok=True)
    expected = fetch_sum(http, release)
    final = folder / release.installer
    part = folder / (release.installer + ".part")
    for stray in folder.iterdir():                     # old versions, a part of another version
        if stray.name not in (final.name, part.name, SETUP_LOG):   # (Setup's log: Updater._tidy decides)
            _remove(stray)
    if final.exists():
        if sha256_file(final, cancel) == expected:
            return final, expected
        _remove(final)
    offset = part.stat().st_size if part.exists() else 0
    if release.size and offset >= release.size:
        _remove(part)
        offset = 0
    if release.size:                                   # about 1.8 GB: say so now, not when the drive is full
        problem = space_problem("download", release.size - offset + DOWNLOAD_SPARE_BYTES, folder, free)
        if problem:
            raise UpdateError(problem)
    resp = http.open(release.installer_url, "asset", {"Range": f"bytes={offset}-"} if offset else None)
    with resp:
        status = _status(resp)
        if status == 416 and offset and not _restarted:   # the part was complete or stale: start over, once
            _remove(part)
            return download_installer(http, release, folder, progress, cancel, _restarted=True, free=free)
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


def _remove(path: Path) -> bool:
    """Delete a file or folder; False (and a log line, never an exception) when it is in use, e.g. by Setup."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        log.info("could not delete %s yet: %s", path.name, exc.strerror or exc)
        return False


def clean_downloads(folder: Path) -> bool:
    """Delete everything an earlier update left behind (after a skip, or once the new version runs). False when
    something is still in use."""
    ok = True
    if folder.is_dir():
        for item in folder.iterdir():
            ok = _remove(item) and ok
    return ok


# =========================================================================== how this copy was installed

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
    installed = (app / "unins000.exe").exists()        # Inno Setup's uninstaller: our installer put this copy here
    lowered = [p.lower() for p in app.parts]
    # Scoop: <scoop root>\apps\voxflow\<version or current>\VoxFlow.exe. Scoop unpacks the installer without running
    # it, so its copy has no uninstaller; an install by hand into a folder that only looks like Scoop's
    # (D:\apps\voxflow\app) has one and updates itself (QA UP-11).
    if not installed and len(lowered) >= 3 and lowered[-2] == "voxflow" and lowered[-3] == "apps":
        return "scoop"
    try:
        if (app / MANAGED_MARKER).read_text(encoding="utf-8", errors="replace").strip().lower() == "winget":
            return "winget"
    except OSError:
        pass
    if not installed:
        return "portable"
    return "allusers" if _system_wide(app) else "installer"


def _system_wide(app: Path) -> bool:
    """Installed for all users (/ALLUSERS: Program Files) or in a folder this user cannot write: a silent Setup
    there would wait for an administrator prompt with nobody at the PC."""
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(var)
        if root:
            try:
                if app.is_relative_to(Path(root).resolve()):
                    return True
            except (OSError, ValueError):
                continue
    return not _writable(app)


def _writable(folder: Path) -> bool:
    """Whether this user can create a file in ``folder``, found out by doing it: ``os.access(W_OK)`` only looks at
    the read-only attribute on Windows and says True for a folder whose permissions deny writing (QA UP-07)."""
    if not os.access(folder, os.W_OK):
        return False
    # (not tempfile.mkstemp: on Windows it takes "permission denied" for a name clash and tries the next name, for ever)
    probe = Path(folder) / f"voxflow-write-test-{os.getpid()}-{random.getrandbits(32):08x}.tmp"
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except OSError:
        return False
    os.close(fd)
    _remove(probe)
    return True


def managed_text(kind: str) -> str:
    """Why this copy does not install updates itself ("" when it does)."""
    return {
        "scoop": "Updates are managed by Scoop: run scoop update voxflow.",
        "winget": f"Updates are managed by winget: run winget upgrade {WINGET_ID}.",
        "source": "VoxFlow is running from source: update it with git pull.",
        "allusers": "VoxFlow is installed for all users, so it cannot update itself without administrator "
                    "rights. Download the new version from the releases page.",
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


def _spawn(cmd: list[str], cwd: str, flags: int, popen, env: dict | None = None) -> OSError | None:
    """Start ``cmd`` so that it outlives VoxFlow. None when it started, else the error."""
    for extra in ((0x01000000,) if sys.platform == "win32" else ()) + (0,):    # CREATE_BREAKAWAY_FROM_JOB first
        try:
            popen(cmd, close_fds=True, creationflags=flags | extra, env=env or _child_env(), cwd=cwd,
                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return None
        except OSError as exc:
            last = exc
    return last


def launch_installer(path: Path, relaunch: bool, log_path: Path | None = None, popen=subprocess.Popen) -> bool:
    """Start the installer detached from VoxFlow (it outlives it). True when it started."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    failed = _spawn(installer_command(path, relaunch, log_path), str(path.parent), flags, popen)
    if failed is None:
        log.info("update: started %s (relaunch=%s)", path.name, relaunch)
        return True
    log.warning("update: the installer could not be started: %s", _short(failed))
    return False


RESTART_EXE_VAR, RESTART_PID_VAR = "VOXFLOW_RESTART_EXE", "VOXFLOW_RESTART_PID"


def restart_app(tray: bool, popen=subprocess.Popen, exe: str | Path | None = None) -> bool:
    """Start this installed copy of VoxFlow again once this process has ended (``tray``: without its window, like a
    start at login). For an update that made VoxFlow quit and then could not start Setup: VoxFlow must not stay
    closed (QA UP-04). A Windows PowerShell from System32 waits for this process to end first: a second VoxFlow
    started while the first still runs only brings the first one's window up and exits. False when nothing was
    started (not Windows, not a frozen build, no PowerShell)."""
    if sys.platform != "win32" or not (exe or getattr(sys, "frozen", False)):
        return False
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    shell = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    script = (f"Wait-Process -Id $env:{RESTART_PID_VAR} -Timeout 60 -ErrorAction SilentlyContinue; "
              f"Start-Process -FilePath $env:{RESTART_EXE_VAR} -WorkingDirectory (Split-Path $env:{RESTART_EXE_VAR})"
              + (" -ArgumentList '--autostart'" if tray else ""))
    env = {**_child_env(), RESTART_EXE_VAR: str(exe or sys.executable), RESTART_PID_VAR: str(os.getpid())}
    failed = _spawn([str(shell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-WindowStyle",
                     "Hidden", "-Command", script], root,
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP, popen, env)
    if failed is None:
        log.info("update: VoxFlow starts again when this copy has closed")
        return True
    log.warning("update: VoxFlow could not be started again: %s", _short(failed))
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
    quit_by: str = ""              # who chose that: "user" (the dialog) or "auto" (automatic mode, when I quit)
    told: str = ""                 # the "will be installed ..." notice already shown (version and when)
    downloaded: str = ""           # version whose verified installer is in the download folder
    sha256: str = ""               # ...and its checksum
    slot_done: str = ""            # the scheduled slot (day) already used
    attempted: str = ""            # the version Setup was last started for ("" once that version, or a newer, runs)
    attempts: int = 0              # ...how many times in a row
    attempt_time: float = 0.0      # ...when (time.time)
    attempt_tray: bool = False     # ...for a scheduled install: the VoxFlow Setup starts again stays in the tray
    attempt_told: bool = False     # the "was not installed" notice about that run was shown
    exit_error: str = ""           # why an update did not start while VoxFlow was closing: said at the next start

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
            elif isinstance(default, int):
                ok = type(value) is int and value >= 0
            elif isinstance(default, float):
                # finite only: a NaN in a hand-edited file made the first timer of the start raise (QA UP-08)
                ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
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


def relaunch_in_tray(data_dir: Path | None = None) -> bool:
    """For ``--after-update`` (Setup starting VoxFlow again): True when that update was a scheduled one. It ran
    while nobody was at the PC, so VoxFlow comes back in the tray, like a start at login, instead of opening its
    window (QA UP-09)."""
    folder = Path(data_dir) if data_dir is not None else cfgmod.app_dir_path()
    return UpdateState.load(folder / STATE_FILE).attempt_tray


# =========================================================================== the service
class Updater(QObject):
    """Checks, downloads and installs, on the UI thread with background threads for the network.

    ``get(key)`` reads a setting, ``busy()`` says a dictation or command is under way (or a shortcut is being
    recorded). The controller connects ``changed`` (Settings > Updates), ``available`` (tell the user),
    ``notice`` and ``restart_requested`` (quit now: the installer runs when VoxFlow has shut down).

    ``free(path)`` gives the free bytes of a drive, ``app_dir`` is the install folder (where Setup unpacks) and
    ``restarter(tray)`` starts VoxFlow again after an update that could not start; tests replace them."""

    changed = Signal(dict)
    available = Signal(dict)
    notice = Signal(str, str)
    restart_requested = Signal()
    _sig = Signal(str, object)

    def __init__(self, data_dir: Path, get, busy, *, version: str = __version__, http=None, kind: str | None = None,
                 metered=connection_metered, idle=idle_seconds, clock=time.time, now=None, launcher=launch_installer,
                 folder: Path | None = None, automatic: bool = True, rng: random.Random | None = None,
                 free=free_bytes, app_dir: Path | None = None, restarter=restart_app, parent=None):
        super().__init__(parent)
        self.version = version
        self._path = Path(data_dir) / STATE_FILE
        self._get, self._busy = get, busy
        self._http = http if http is not None else Http()
        self.kind = kind if kind is not None else install_kind()
        self._metered, self._idle, self._clock = metered, idle, clock
        self._now = now or (lambda: _dt.datetime.now().astimezone())
        self._launcher, self._restarter, self._free = launcher, restarter, free
        self._app_dir = Path(app_dir) if app_dir is not None else Path(sys.executable).parent
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
        # "Install now" / the set time, between the restart request and on_exit: (path, the open verified file,
        # version, tray)
        self._pending_launch: tuple | None = None
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
        self._tidy_timer = QTimer(self)
        self._tidy_timer.setSingleShot(True)
        self._tidy_timer.setInterval(TIDY_RETRY_MS)
        self._tidy_timer.timeout.connect(self._tidy_again)
        self._tidy_tries = 0
        self._remind_timer = QTimer(self)
        self._remind_timer.setSingleShot(True)
        self._remind_timer.timeout.connect(self._remind)
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
            "auto_stopped": bool(rel and self._gave_up(rel)),
            "releases_page": rel.page if rel else RELEASES_PAGE,
        }

    def _emit(self) -> None:
        self.changed.emit(self.info())

    def _save(self) -> None:
        self.state.save(self._path)

    def _forget_installed(self) -> None:
        """After an update the old installer (and any reminder about it) is no longer needed."""
        st = self.state
        if st.attempted and not is_newer(st.attempted, self.version):
            self._clear_attempt()                      # that version runs now: Setup did its work
        if st.downloaded and not is_newer(st.downloaded, self.version):
            st.downloaded = st.sha256 = ""
        if st.install_on_quit and not is_newer(st.install_on_quit, self.version):
            self._clear_quit()
        self._tidy_again()

    def _tidy_again(self) -> None:
        """Right after an update Setup may still hold its files (the relaunched VoxFlow starts while it finishes):
        what cannot be deleted yet is tried again a few minutes later, a few times. Never an error."""
        if self._closed or self._tidy() or self._tidy_tries >= TIDY_RETRIES:
            return
        self._tidy_tries += 1
        self._tidy_timer.start()

    def _clear_quit(self) -> None:
        self.state.install_on_quit = self.state.quit_by = ""

    def _clear_attempt(self) -> None:
        st = self.state
        st.attempted, st.attempts, st.attempt_time, st.attempt_tray, st.attempt_told = "", 0, 0.0, False, False

    def _gave_up(self, rel: Release) -> bool:
        """Setup was started for this version ``MAX_AUTO_ATTEMPTS`` times and it is still not the one running:
        nothing automatic tries again (VoxFlow would quit and run Setup every night, QA UP-02). The user still
        can, with Install now."""
        return self.state.attempted == rel.version and self.state.attempts >= MAX_AUTO_ATTEMPTS

    def _drop_pending(self) -> None:
        if self._pending_launch is not None:
            self._pending_launch[1].close()
            self._pending_launch = None

    def _tidy(self, keep_latest: bool = True) -> bool:
        """Keep the download folder bounded: at most the installer of the newest release (and its ``.part`` while it
        downloads). Everything else, such as an older installer, Setup's log of a run that worked or a file whose
        release was skipped, is deleted. The log of a Setup run that did not change the version stays: it says why."""
        rel = self.latest()
        keep = set()
        if keep_latest and rel is not None and rel.installer and self.state.skipped != rel.version:
            keep = {rel.installer, rel.installer + ".part"}
        if self.state.downloaded and (rel is None or self.state.downloaded != rel.version or not keep):
            self.state.downloaded = self.state.sha256 = ""
        kept = keep | {SETUP_LOG} if self.state.attempted else keep
        ok = True
        if self.folder.is_dir():
            for item in self.folder.iterdir():
                if item.name not in kept:
                    ok = _remove(item) and ok
        return ok

    def _drop_download(self) -> None:
        """Cancel a download under way and delete what was downloaded (Off, Skip this version)."""
        if self.phase == "downloading":
            self._cancel.set()                         # the thread stops at the next block; see _on_download_failed
        clean_downloads(self.folder)
        self.state.downloaded = self.state.sha256 = ""

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Arm the daily check (nothing at all when the mode is Off or automatic checks are disabled)."""
        self._arm_check(first=True)
        self._arm_slot()
        self._arm_remind(startup=True)
        self._tell_exit_error()
        self._tell_failed_install()
        self._remind_quit_install()

    def _tell_exit_error(self) -> None:
        """An update that could not start while VoxFlow was closing had nobody to tell: say it now, once."""
        st = self.state
        if st.exit_error:
            self.error, st.exit_error = st.exit_error, ""
            self._save()
            self.notice.emit("warning", self.error)

    def _tell_failed_install(self) -> None:
        """Setup was started for a version and this is still the old one (Setup failed, was blocked or was
        cancelled): say so once per run of Setup, with where its log is (QA UP-02)."""
        st = self.state
        if not st.attempted or st.attempt_told:
            return
        st.attempt_told = True
        self._save()
        log.warning("update %s: Setup was started %d time(s) and VoxFlow is still %s", st.attempted, st.attempts,
                    self.version)
        message = f"VoxFlow {st.attempted} was not installed: Setup ran, but this is still VoxFlow {self.version}."
        if (self.folder / SETUP_LOG).exists():
            message += f" Setup's log is {self.folder / SETUP_LOG}."
        if self.mode == "auto" and st.attempts >= MAX_AUTO_ATTEMPTS:
            message += (" VoxFlow will not try again by itself: use Check now in Settings > Updates to install it, "
                        "or get it from the releases page.")
        self.error = message
        self.notice.emit("warning", message)

    def _remind_quit_install(self) -> None:
        """An update that waits for "when I quit" is installed only by Quit in the tray menu, never when Windows ends
        the session. Someone who only ever shuts the PC down was told once and then kept the old version for good
        (QA UP-01): say it again at every start, with what to do."""
        rel, st = self.latest(), self.state
        if (rel is not None and self.mode != "off" and st.install_on_quit == rel.version == st.downloaded
                and is_newer(rel.version, self.version)):
            self.notice.emit("info", f"VoxFlow {rel.version} is downloaded and waiting. It is installed when you "
                                     "quit VoxFlow from its tray menu (not when Windows shuts down).")

    def close(self) -> None:
        """Stop every timer and thread result. An installer held open for a restart stays open: on_exit, which the
        controller calls after this, starts Setup from it."""
        self._closed = True
        for t in (self._check_timer, self._install_timer, self._slot_timer, self._tidy_timer, self._remind_timer):
            t.stop()
        self._cancel.set()

    def _arm_check(self, first: bool = False) -> None:
        self._check_timer.stop()
        if self._closed or not self._automatic or self.mode == "off":
            return
        every = CHECK_EVERY_S if self.state.last_ok else RETRY_FAILED_S
        last = min(self.state.last_check, self._clock())    # a time in the future (the clock was wrong once) = now
        due = last + every - self._clock()
        delay = max(due, 0.0)
        if first:
            delay = max(delay, self._rng.uniform(*STARTUP_DELAY_S))
        self._check_timer.start(int(min(delay, CHECK_EVERY_S) * 1000))

    def settings_changed(self, key: str) -> None:
        if key == "update_mode":
            if self.mode == "off":                     # Off: no automatic traffic, nothing pending, nothing kept
                self._drop_download()
                self._install_timer.stop()
                self._slot_timer.stop()
                if self.phase == "waiting":
                    self.phase = "idle"
                self._clear_quit()
                self._drop_pending()
                self._save()
            elif self.mode != "auto" and self.state.quit_by == "auto":
                self._clear_quit()                     # automatic mode chose it, not the user: no longer wanted
                self._save()
            self._arm_check(first=True)
            self._arm_remind()
        if (key == "update_install_when" and self.mode == "auto" and self._get("update_install_when") == "time"
                and self.state.quit_by == "auto"):
            self._clear_quit()                         # the set time takes over
            self._save()
        if key in ("update_mode", "update_install_when", "update_time"):
            self._arm_slot()
            if self.mode == "auto":
                self._auto_next()
        self._emit()

    # ------------------------------------------------------------------ check
    def check(self, user: bool = True) -> bool:
        """Ask GitHub for the newest release in the background. ``user``: "Check now" was clicked, which works in
        every mode; an automatic check (``user=False``) never runs while the mode is Off. False when not started."""
        if self._closed or (self.mode == "off" and not user):
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
        self._tidy()                                   # a newer release replaces an older download
        newer = is_newer(rel.version, self.version)
        st.last_result = f"VoxFlow {rel.version} is available." if newer else "VoxFlow is up to date."
        log.info("update check: newest release %s, running %s", rel.version, self.version)
        self.phase = "idle"
        self._save()
        self._arm_check()
        self._emit()
        if not newer or (self.mode == "off" and not user):
            return                                     # switched to Off while it ran: no notification
        if self.mode == "auto" and self.can_install and rel.installable and st.skipped != rel.version:
            self._auto_next()                          # it tells the user what it will do (see _downloaded)
            if user:
                self.available.emit(self.info())
        else:
            self._announce(rel, user)

    def _on_check_failed(self, user: bool, exc: Exception) -> None:
        st = self.state
        # GitHub answered and has nothing to offer: the check worked, so the next one is tomorrow's. Only a check
        # that did not get through is tried again in an hour (QA D-05).
        nothing = isinstance(exc, NoRelease)
        st.last_check, st.last_ok = self._clock(), nothing
        st.last_result = str(exc) if isinstance(exc, UpdateError) else "The update check did not work."
        if nothing:
            log.info("update check: %s", exc)
            if exc.status == 404:
                self._withdrawn()
        else:
            log.warning("update check failed: %s", exc if isinstance(exc, UpdateError) else type(exc).__name__)
        self.phase = "idle"
        self._save()
        self._arm_check()
        self._emit()

    def _withdrawn(self) -> None:
        """GitHub has no published release any more, so the one VoxFlow knew was withdrawn (it may have been
        broken): it is no longer offered, its download is deleted and it is not installed on quit or at the set
        time (QA UP-10)."""
        if self.state.latest is not None:
            log.warning("update: the release VoxFlow knew is no longer published; nothing of it is kept")
        self.state.latest = None
        self._drop_download()
        self._clear_quit()
        for t in (self._install_timer, self._slot_timer, self._remind_timer):
            t.stop()

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
        self._arm_remind()
        self._emit()

    def _arm_remind(self, startup: bool = False) -> None:
        """"Remind me later" means tomorrow: a timer for that moment (also armed at start, for a reminder that is
        still pending or fell due while VoxFlow was closed). Waiting for the first daily check after it made it
        about two days (QA UP-05)."""
        self._remind_timer.stop()
        rel, st = self.latest(), self.state
        if (self._closed or rel is None or self.mode == "off" or not st.remind_after
                or rel.version in (st.notified, st.skipped)):
            return
        # never longer than the pause itself (a clock that was wrong when the user clicked)
        wait = min(max(st.remind_after - self._clock(), 0.0), REMIND_AFTER_S)
        if startup:
            wait = max(wait, STARTUP_DELAY_S[0])        # not on top of an app that is still starting
        self._remind_timer.start(int(wait * 1000))

    def _remind(self) -> None:
        rel, st = self.latest(), self.state
        st.remind_after = 0.0
        self._save()
        if (rel is None or self.mode == "off" or self.phase != "idle" or rel.version in (st.notified, st.skipped)
                or st.install_on_quit == rel.version):
            return                                     # told meanwhile, or the user already chose what to do
        self._announce(rel, user=False)

    def skip(self) -> None:
        rel = self.latest()
        if rel is None:
            return
        st = self.state
        st.skipped = rel.version
        self._clear_quit()
        self._drop_download()
        self._install_timer.stop()
        self._remind_timer.stop()
        if self.phase == "waiting":
            self.phase = "idle"
        self._arm_slot()                               # a skipped version is never installed at the set time
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
            st.install_on_quit, st.quit_by = rel.version, "user"
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
                                                                      self._cancel.is_set, free=self._free))),
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
            self._tidy(keep_latest=self.mode != "off")  # Off or Skip: nothing of it is kept (nor the .part)
            self._save()
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
        elif when == "auto" and self.mode == "auto":   # (the user may have left automatic mode meanwhile)
            st = self.state
            if self._get("update_install_when") == "quit":
                if st.install_on_quit != rel.version:
                    st.install_on_quit, st.quit_by = rel.version, "auto"
                self._tell(rel, "quit", f"VoxFlow {rel.version} will be installed when you quit VoxFlow.")
            else:
                at = str(self._get("update_time"))
                self._tell(rel, f"time {at}", f"VoxFlow {rel.version} will be installed at {at}, when VoxFlow and "
                                              "the PC are idle.")
            self._save()
        self._arm_slot()

    def _tell(self, rel: Release, what: str, message: str) -> None:
        """A notice about an automatic install, once per version and plan (not again at every daily check)."""
        key = f"{rel.version} {what}"
        if self.state.told != key:
            self.state.told = key
            self.notice.emit("info", message)

    # ------------------------------------------------------------------ automatic mode
    def _auto_next(self) -> None:
        """Install automatically: fetch the installer in the background (not on a metered connection)."""
        rel = self.latest()
        if (rel is None or not self.can_install or not rel.installable or self.state.skipped == rel.version
                or self.phase != "idle" or self._gave_up(rel)):
            return
        if self.state.downloaded == rel.version and self._file(rel).exists():
            self._after_download = "auto"
            self._downloaded(rel)
            return
        if self._metered() is True:
            self.error = "Waiting for a connection that is not metered to download the update."
            log.info("update %s: not downloaded automatically on a metered connection", rel.version)
            self._emit()
            self._announce(rel, user=False)            # the user can still choose Install now
            return
        self._after_download = "auto"
        self._download(rel)

    def _arm_slot(self) -> None:
        rel = self.latest()
        armed = (not self._closed and self.mode == "auto" and self._get("update_install_when") == "time"
                 and rel is not None and self.state.downloaded == rel.version and self.state.skipped != rel.version
                 and not self._gave_up(rel))
        if armed and not self._slot_timer.isActive():
            self._slot_timer.start()
        elif not armed:
            self._slot_timer.stop()

    def _slot_tick(self) -> None:
        rel = self.latest()
        st = self.state
        if (rel is None or self.mode != "auto" or self._get("update_install_when") != "time"
                or st.downloaded != rel.version or st.skipped == rel.version or self._gave_up(rel)):
            self._slot_timer.stop()
            return
        if self.phase != "idle":
            return                                     # a check or another download is under way
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
        self._restart(rel, tray=True)                  # nobody is at the PC: VoxFlow comes back without its window

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

    def _room_problem(self, path: Path) -> str:
        """A sentence when the install drive has no room for Setup to unpack ``path`` ("" when it has)."""
        try:
            size = path.stat().st_size
        except OSError:
            return ""                                  # gone: _open_checked says so
        return space_problem("install", int(size * UNPACKED_FACTOR), self._app_dir, self._free)

    def _open_checked(self, path: Path):
        """The downloaded installer, opened so that nobody can change it and hashed through that handle. The caller
        keeps it open until Setup has started, so the file that runs is the file that was checked. None when it is
        gone, cannot be opened (another program has it open for writing) or is not the verified download."""
        try:
            fh = open_exclusive(path)
        except OSError:
            return None
        try:
            ok = bool(self.state.sha256) and sha256_stream(fh) == self.state.sha256
        except OSError:
            ok = False
        if ok:
            return fh
        fh.close()
        return None

    def _restart(self, rel: Release, tray: bool = False) -> None:
        """Ask the controller to quit so that Setup can run. Everything that can still stop Setup (no room, a file
        that is gone, changed or locked) is checked first, and the verified file stays open from here until Setup
        starts: VoxFlow never quits for an install that then does not happen (QA UP-04)."""
        path = self._file(rel)
        problem = self._room_problem(path)
        held = None if problem else self._open_checked(path)
        if held is None:
            if not problem:
                log.warning("update %s: the downloaded installer is missing or changed; not restarting", rel.version)
                _remove(path)
                self.state.downloaded = self.state.sha256 = ""
                self._clear_quit()
                problem = "The downloaded installer is gone or changed. Try again."
            self.phase, self.error = "idle", problem
            self._arm_slot()
            self._save()
            self._emit()
            self.notice.emit("warning", problem)
            return
        self._pending_launch = (path, held, rel.version, tray)
        self.phase = "restarting"
        self._emit()
        log.info("update %s: restarting VoxFlow to install it", rel.version)
        self.restart_requested.emit()

    def on_exit(self, user_quit: bool) -> bool:
        """VoxFlow is shutting down: start the installer if one is due. ``user_quit`` is False when Windows ends the
        session (no installer then, unless VoxFlow is closing for it). True when Setup was started.

        What goes wrong here has nobody to tell: it is kept for the next start (``exit_error``), and after a restart
        VoxFlow asked for itself it is started again, so it is not left closed."""
        rel = self.latest()
        st = self.state
        if self._pending_launch is not None:
            (path, held, version, tray), relaunch = self._pending_launch, True
            self._pending_launch = None
        elif user_quit and rel is not None and st.install_on_quit == rel.version and st.downloaded == rel.version:
            path, version, tray, relaunch = self._file(rel), rel.version, False, False
            problem = self._room_problem(path)         # (kept for the next quit: it works once there is room)
            held = None if problem else self._open_checked(path)
            if held is None:
                if not problem:
                    log.warning("update: the installer changed on disk since it was verified; it is not run")
                    _remove(path)
                    st.downloaded = st.sha256 = ""
                    self._clear_quit()
                    problem = (f"VoxFlow {version} was not installed when you quit: the downloaded installer was "
                               "gone or changed.")
                st.exit_error = problem
                self._save()
                return False
        else:
            return False
        with held:                                     # nobody can change the file until Setup has started
            self._clear_quit()
            st.attempts = st.attempts + 1 if st.attempted == version else 1
            st.attempted, st.attempt_time, st.attempt_tray, st.attempt_told = version, self._clock(), tray, False
            self._save()                               # before Setup starts: it may end this process
            started = bool(self._launcher(path, relaunch, self.folder / SETUP_LOG))
        if started:
            return True
        st.attempts -= 1                               # Setup never ran: not a run that left the version unchanged
        if not st.attempts:
            self._clear_attempt()
        st.exit_error = (f"VoxFlow {version} was not installed: Windows did not start its installer. Security "
                         "software may have blocked it.")
        self._save()
        if relaunch:
            self._restarter(tray)
        return False
