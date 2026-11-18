"""App updates (voxflow/updater.py): hermetic. Every request goes to a fake HTTP layer, and a socket spy proves the
setting "Off" (and a refused URL) opens no connection at all."""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import socket
import urllib.request
from pathlib import Path

import pytest

from voxflow import updater as U

REPO = "https://github.com/blackknight75/voxflow"
API = U.API_LATEST


# =========================================================================== fakes
class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status = status
        self._body = io.BytesIO(body)
        self.headers = headers or {"Content-Length": str(len(body))}

    def read(self, n=-1):
        return self._body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHttp:
    """Routes ``url -> callable(headers) -> Resp`` (or an exception to raise). Records every request."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.requests: list[tuple[str, str, dict]] = []

    def open(self, url, kind, headers=None):
        assert U.allowed_url(url, kind), f"the updater asked for a URL outside the allow-list: {url}"
        self.requests.append((url, kind, dict(headers or {})))
        route = self.routes.get(url)
        if route is None:
            raise U.UpdateError("GitHub could not be reached (no route in the test).")
        result = route(headers or {})
        if isinstance(result, Exception):
            raise result
        return result

    def urls(self):
        return [u for u, _k, _h in self.requests]


INSTALLER = b"MZ" + bytes(range(256)) * 40          # 10 KiB of "installer"
DIGEST = hashlib.sha256(INSTALLER).hexdigest()


def release_json(version="0.3.0", *, draft=False, prerelease=False, size=None, assets=None, body="## Added\n- x"):
    name = f"VoxFlow-Setup-{version}.exe"
    if assets is None:
        assets = [{"name": name, "size": len(INSTALLER) if size is None else size,
                   "browser_download_url": f"{REPO}/releases/download/v{version}/{name}"},
                  {"name": "SHA256SUMS.txt", "size": 90,
                   "browser_download_url": f"{REPO}/releases/download/v{version}/SHA256SUMS.txt"}]
    return {"tag_name": f"v{version}", "name": f"VoxFlow {version}", "draft": draft, "prerelease": prerelease,
            "html_url": f"{REPO}/releases/tag/v{version}", "published_at": "2026-12-01T10:00:00Z", "body": body,
            "assets": assets}


def github(version="0.3.0", payload=None, sums=None, data=INSTALLER, **kw):
    """A FakeHttp that serves the API answer, SHA256SUMS.txt and the installer (with Range support)."""
    payload = payload if payload is not None else release_json(version, **kw)
    name = f"VoxFlow-Setup-{version}.exe"
    sums = sums if sums is not None else f"{hashlib.sha256(data).hexdigest()}  {name}\n"

    def installer(headers):
        rng = headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return Resp(206, data[start:])
        return Resp(200, data)

    return FakeHttp({API: lambda h: Resp(200, json.dumps(payload).encode()),
                     f"{REPO}/releases/download/v{version}/SHA256SUMS.txt": lambda h: Resp(200, sums.encode()),
                     f"{REPO}/releases/download/v{version}/{name}": installer})


@pytest.fixture(autouse=True)
def installed_version_is_older(monkeypatch):
    """These tests offer 0.3.0 as the newer release, so an Updater built without ``version`` (the controller's too)
    must take the installed version to be an older one, whatever ``voxflow.__version__`` is today."""
    monkeypatch.setitem(U.Updater.__init__.__kwdefaults__, "version", "0.2.2")


@pytest.fixture
def no_sockets(monkeypatch):
    """Every outgoing connection attempt is recorded and refused."""
    seen = []

    def spy(self, address):
        seen.append(address)
        raise OSError("network disabled in the test")

    monkeypatch.setattr(socket.socket, "connect", spy)
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: spy(None, address))
    return seen


# =========================================================================== versions
@pytest.mark.parametrize("a,b,expected", [
    ("0.2.0", "0.2.1", -1), ("0.2.1", "0.10.0", -1), ("0.10.0", "0.2.0", 1), ("v0.2.0", "0.2.0", 0),
    ("V1.0.0", "v1.0.0", 0), ("1.0.0-rc.1", "1.0.0", -1), ("1.0.0-alpha", "1.0.0-alpha.1", -1),
    ("1.0.0-alpha.1", "1.0.0-alpha.beta", -1), ("1.0.0-beta.2", "1.0.0-beta.11", -1), ("1.0.0+build.5", "1.0.0", 0),
    ("2.0.0", "10.0.0", -1),
])
def test_versions_compare_by_semver(a, b, expected):
    assert U.compare_versions(a, b) == expected
    assert U.compare_versions(b, a) == -expected


@pytest.mark.parametrize("text", ["", "0.2", "1.2.3.4", "01.2.3", "latest", "v", "1.2.x", None, "1.0.0-"])
def test_anything_that_is_not_a_version_is_refused(text):
    assert U.parse_version(text) is None
    assert U.is_newer(text, "0.2.0") is False
    with pytest.raises(ValueError):
        U.compare_versions(text, "0.2.0")


def test_is_newer_and_version_text():
    assert U.is_newer("v0.2.1", "0.2.0") and not U.is_newer("0.2.0", "0.2.0") and not U.is_newer("0.1.9", "0.2.0")
    assert str(U.parse_version("v1.2.3-rc.1+x")) == "1.2.3-rc.1" and str(U.parse_version("1.2.3")) == "1.2.3"


# =========================================================================== allow-list
@pytest.mark.parametrize("url,kind,ok", [
    (API, "api", True),
    (f"{REPO}/releases/download/v1.0.0/VoxFlow-Setup-1.0.0.exe", "asset", True),
    ("https://objects.githubusercontent.com/github-production-release-asset/1/2?x=y", "asset", True),
    ("https://release-assets.githubusercontent.com/github-production-release-asset/1/2", "asset", True),
    ("http://github.com/blackknight75/voxflow/releases/download/v1/x.exe", "asset", False),   # not HTTPS
    ("https://github.com/someone/voxflow/releases/download/v1/x.exe", "asset", False),        # another repository
    ("https://github.com/blackknight75/voxflow/archive/main.zip", "asset", False),            # not a release asset
    ("https://github.com.evil.example/blackknight75/voxflow/releases/download/v1/x", "asset", False),
    ("https://user:pw@github.com/blackknight75/voxflow/releases/download/v1/x", "asset", False),
    ("https://github.com:8443/blackknight75/voxflow/releases/download/v1/x", "asset", False),
    ("https://github.com:99999/blackknight75/voxflow/releases/download/v1/x", "asset", False),
    ("https://evil.example/x.exe", "asset", False),
    ("https://api.github.com/repos/blackknight75/voxflow/releases/latest", "asset", False),   # API is not an asset
    ("https://objects.githubusercontent.com/x", "api", False),
    ("https://api.github.com/repos/other/repo/releases/latest", "api", False),
    ("file:///C:/Windows/notepad.exe", "asset", False),
])
def test_only_this_repository_over_https_is_allowed(url, kind, ok):
    assert U.allowed_url(url, kind) is ok


def test_a_redirect_outside_the_allow_list_is_stopped():
    handler = U._CheckedRedirect("asset")
    req = urllib.request.Request(f"{REPO}/releases/download/v1.0.0/VoxFlow-Setup-1.0.0.exe",
                                 headers={"Range": "bytes=10-"})
    with pytest.raises(U.UpdateError, match="does not trust"):
        handler.redirect_request(req, None, 302, "Found", {}, "https://evil.example/x.exe")
    with pytest.raises(U.UpdateError):
        handler.redirect_request(req, None, 302, "Found", {}, "http://objects.githubusercontent.com/x")
    nxt = handler.redirect_request(req, None, 302, "Found", {}, "https://objects.githubusercontent.com/a?b=c")
    assert nxt.full_url == "https://objects.githubusercontent.com/a?b=c" and nxt.get_header("Range") == "bytes=10-"


def test_the_real_http_layer_refuses_a_foreign_url_before_any_connection(no_sockets):
    with pytest.raises(U.UpdateError, match="does not trust"):
        U.Http().open("https://evil.example/x", "asset")
    with pytest.raises(U.UpdateError):
        U.Http().open("https://evil.example/x", "api")
    assert no_sockets == []


def test_the_real_http_layer_sends_a_generic_user_agent_and_nothing_else(monkeypatch):
    sent = {}

    class Opener:
        def open(self, req, timeout):
            sent.update(url=req.full_url, headers=dict(req.header_items()), timeout=timeout)
            return Resp(200, b"{}")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    U.Http(timeout=7).open(API, "api")
    assert sent["url"] == API and sent["timeout"] == 7
    assert sent["headers"] == {"User-agent": f"VoxFlow/{U.__version__}", "Accept": "application/vnd.github+json"}


@pytest.mark.parametrize("code,text", [(404, "No published release"), (403, "limiting"), (429, "limiting"),
                                       (500, "HTTP 500")])
def test_http_errors_become_sentences(monkeypatch, code, text):
    import urllib.error

    class Opener:
        def open(self, req, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(U.UpdateError, match=text):
        U.Http().open(API, "api")


def test_an_unreachable_github_is_one_quiet_sentence(monkeypatch):
    import urllib.error

    class Opener:
        def open(self, req, timeout):
            raise urllib.error.URLError(OSError("getaddrinfo failed"))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(U.UpdateError, match=r"GitHub could not be reached \(getaddrinfo failed\)"):
        U.Http().open(API, "api")


# =========================================================================== releases and notes
def test_a_release_is_read_from_the_api_answer():
    rel = U.parse_release(release_json("0.3.0"))
    assert (rel.version, rel.tag, rel.installer) == ("0.3.0", "v0.3.0", "VoxFlow-Setup-0.3.0.exe")
    assert rel.installable and rel.size == len(INSTALLER) and rel.page == f"{REPO}/releases/tag/v0.3.0"
    assert U.Release.from_dict(rel.to_dict()) == rel


@pytest.mark.parametrize("change,text", [
    ({"draft": True}, "not a final release"), ({"prerelease": True}, "not a final release"),
    ({"tag_name": "nightly"}, "no usable version"), ({"tag_name": "v1.0.0-beta.1"}, "no usable version"),
])
def test_drafts_prereleases_and_odd_tags_are_never_offered(change, text):
    with pytest.raises(U.UpdateError, match=text):
        U.parse_release({**release_json(), **change})
    with pytest.raises(U.UpdateError):
        U.parse_release(["not", "a", "dict"])


def test_assets_from_elsewhere_are_ignored_and_a_release_without_checksums_cannot_be_installed():
    data = release_json(assets=[
        {"name": "VoxFlow-Setup-0.3.0.exe", "size": 5, "browser_download_url": "https://evil.example/x.exe"},
        {"name": "SHA256SUMS.txt", "browser_download_url": f"{REPO}/releases/download/v0.3.0/SHA256SUMS.txt"},
        "junk"])
    rel = U.parse_release(data)
    assert rel.installer_url == "" and not rel.installable
    data = release_json(assets=[{"name": "VoxFlow-Setup-0.3.0.exe", "size": 5,
                                 "browser_download_url": f"{REPO}/releases/download/v0.3.0/VoxFlow-Setup-0.3.0.exe"}])
    assert not U.parse_release(data).installable, "no SHA256SUMS.txt: never installed"
    huge = U.parse_release(release_json(size=U.INSTALLER_MAX_BYTES + 1))
    assert not huge.installable


def test_a_foreign_release_page_link_is_replaced_by_the_releases_page():
    rel = U.parse_release({**release_json(), "html_url": "https://evil.example/phish"})
    assert rel.page == U.RELEASES_PAGE
    rel = U.parse_release({**release_json(), "html_url": f"{REPO}/releases/tag/v0.3.0?x=<script>"})
    assert rel.page == U.RELEASES_PAGE


def test_a_tampered_state_file_cannot_widen_the_allow_list():
    rel = U.parse_release(release_json()).to_dict()
    rel["installer_url"] = "https://evil.example/x.exe"
    assert U.Release.from_dict(rel).installer_url == ""
    assert U.Release.from_dict({"version": "nope"}) is None and U.Release.from_dict({"bogus": 1}) is None
    assert U.Release.from_dict("x") is None


def test_release_notes_become_plain_text_without_images_html_or_link_targets():
    body = ("## Added\r\n- **Updates** with `choices`\n* [Docs](https://example.com/docs \"t\")\n"
            "![shot](https://example.com/track.png)<img src=\"https://example.com/p.gif\">\n"
            "<script>alert(1)</script><!-- hidden -->\n---\n\n\n\nThe end\u202e\x07")
    text = U.plain_notes(body)
    assert text == "Added\n• Updates with choices\n• Docs\n\nalert(1)\n\nThe end"
    assert "http" not in text and "<" not in text
    long = U.plain_notes("line\n" * 5000)
    assert len(long) <= U.NOTES_MAX_CHARS + 2 and long.endswith("…")
    assert U.plain_notes(None) == ""


def test_release_notes_keep_placeholders_that_are_not_html():
    # v0.2.1's notes said "User-Agent VoxFlow/." because "<version>" was stripped as if it were a tag
    text = U.plain_notes("with the User-Agent `VoxFlow/<version>`, run <path to exe> /S, a <-> b, 1 < 2 > 0\n"
                         "<IMG SRC=x><Script>x</SCRIPT><br/><a href='https://e.com'>e</a><svg/onload=1>")
    assert text == "with the User-Agent VoxFlow/<version>, run <path to exe> /S, a <-> b, 1 < 2 > 0\nxe"


def test_checksums_are_read_like_sha256sum_writes_them():
    sums = f"{'A' * 64}  VoxFlow-Setup-1.0.0.exe\n{'b' * 64} *other.exe\n"
    assert U.parse_sums(sums, "VoxFlow-Setup-1.0.0.exe") == "a" * 64
    assert U.parse_sums(sums, "other.exe") == "b" * 64
    assert U.parse_sums(sums, "missing.exe") is None and U.parse_sums("garbage", "x") is None


def test_fetch_latest_reads_the_api_and_refuses_garbage():
    assert U.fetch_latest(github()).version == "0.3.0"
    bad = FakeHttp({API: lambda h: Resp(200, b"<html>")})
    with pytest.raises(U.UpdateError, match="does not understand"):
        U.fetch_latest(bad)
    big = FakeHttp({API: lambda h: Resp(200, b" " * (U.API_MAX_BYTES + 1))})
    with pytest.raises(U.UpdateError, match="unexpectedly large"):
        U.fetch_latest(big)


# =========================================================================== download and verification
def test_the_installer_is_downloaded_verified_and_progress_is_reported(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    seen = []
    path, digest = U.download_installer(http, rel, tmp_path / "dl", progress=lambda d, t: seen.append((d, t)))
    assert path.read_bytes() == INSTALLER and digest == DIGEST and path.name == "VoxFlow-Setup-0.3.0.exe"
    assert seen[-1] == (len(INSTALLER), len(INSTALLER))
    assert sorted(p.name for p in (tmp_path / "dl").iterdir()) == ["VoxFlow-Setup-0.3.0.exe"]
    before = len(http.requests)
    assert U.download_installer(http, rel, tmp_path / "dl") == (path, DIGEST)       # already there: not fetched
    assert http.urls()[before:] == [rel.sums_url]


def test_an_interrupted_download_is_resumed_with_a_range_request(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / (rel.installer + ".part")).write_bytes(INSTALLER[:4000])
    (folder / "VoxFlow-Setup-0.2.5.exe").write_bytes(b"old")           # an older update: cleaned up
    path, _ = U.download_installer(http, rel, folder)
    assert path.read_bytes() == INSTALLER
    assert http.requests[-1][2] == {"Range": "bytes=4000-"}
    assert sorted(p.name for p in folder.iterdir()) == [rel.installer]


def test_a_server_that_ignores_the_range_starts_over(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    http.routes[rel.installer_url] = lambda h: Resp(200, INSTALLER)
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / (rel.installer + ".part")).write_bytes(b"garbage" * 10)
    path, _ = U.download_installer(http, rel, folder)
    assert path.read_bytes() == INSTALLER


def test_a_part_that_is_complete_or_rejected_is_downloaded_again(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / (rel.installer + ".part")).write_bytes(b"x" * (len(INSTALLER) + 5))   # longer than the release
    path, _ = U.download_installer(http, rel, folder)
    assert path.read_bytes() == INSTALLER and http.requests[-1][2] == {}
    calls = []

    def range_refused(headers):
        calls.append(dict(headers))
        return Resp(416, b"") if headers.get("Range") else Resp(200, INSTALLER)

    http.routes[rel.installer_url] = range_refused
    path.unlink()
    (folder / (rel.installer + ".part")).write_bytes(INSTALLER[:10])
    U.download_installer(http, rel, folder)
    assert calls == [{"Range": "bytes=10-"}, {}]


def test_a_checksum_mismatch_is_refused_and_the_file_deleted(tmp_path, caplog):
    http = github(sums=f"{'0' * 64}  VoxFlow-Setup-0.3.0.exe\n")
    rel = U.fetch_latest(http)
    with pytest.raises(U.UpdateError, match="does not match its published SHA-256"):
        U.download_installer(http, rel, tmp_path / "dl")
    assert list((tmp_path / "dl").iterdir()) == [], "nothing is left to run"
    assert "refused" in caplog.text


def test_a_tampered_finished_file_is_replaced(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    folder = tmp_path / "dl"
    folder.mkdir()
    (folder / rel.installer).write_bytes(b"evil")
    path, _ = U.download_installer(http, rel, folder)
    assert path.read_bytes() == INSTALLER


@pytest.mark.parametrize("sums,text", [("", "does not list the installer"), (None, "has no SHA256SUMS")])
def test_without_a_checksum_nothing_is_downloaded(tmp_path, sums, text):
    http = github(sums=sums if sums is not None else "x")
    rel = U.fetch_latest(http)
    if sums is None:
        rel.sums_url = ""
        with pytest.raises(U.UpdateError, match="no Windows installer"):
            U.download_installer(http, rel, tmp_path / "dl")
        with pytest.raises(U.UpdateError, match=text):
            U.fetch_sum(http, rel)
        return
    with pytest.raises(U.UpdateError, match=text):
        U.download_installer(http, rel, tmp_path / "dl")
    assert rel.installer_url not in http.urls()


def test_a_download_larger_than_announced_or_cut_short_is_not_kept(tmp_path):
    http = github(data=INSTALLER + b"extra")
    rel = U.fetch_latest(http)
    rel.size = len(INSTALLER)
    with pytest.raises(U.UpdateError, match="larger than the release says"):
        U.download_installer(http, rel, tmp_path / "dl")
    assert not (tmp_path / "dl" / (rel.installer + ".part")).exists()
    short = github(data=INSTALLER[:100])
    rel = U.fetch_latest(short)
    rel.size = len(INSTALLER)
    with pytest.raises(U.UpdateError, match="stopped before the end"):
        U.download_installer(short, rel, tmp_path / "dl2")
    assert (tmp_path / "dl2" / (rel.installer + ".part")).stat().st_size == 100, "kept, to be resumed"


def test_a_cancelled_download_keeps_its_part_for_later(tmp_path):
    http = github()
    rel = U.fetch_latest(http)
    with pytest.raises(U.Cancelled):
        U.download_installer(http, rel, tmp_path / "dl", cancel=lambda: True)
    with pytest.raises(U.UpdateError, match="HTTP 503"):
        http.routes[rel.installer_url] = lambda h: Resp(503, b"")
        U.download_installer(http, rel, tmp_path / "dl")


# =========================================================================== how this copy was installed
def _exe(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "VoxFlow.exe"
    exe.write_bytes(b"")
    return exe


def test_install_kind_detects_source_scoop_winget_installer_and_portable(tmp_path):
    assert U.install_kind(frozen=False) == "source"
    assert U.install_kind(_exe(tmp_path / "scoop" / "apps" / "voxflow" / "0.2.0"), frozen=True) == "scoop"
    assert U.install_kind(_exe(tmp_path / "S" / "Apps" / "VoxFlow" / "current"), frozen=True) == "scoop"
    winget = _exe(tmp_path / "Programs" / "VoxFlow")
    (winget.parent / "unins000.exe").write_bytes(b"")
    assert U.install_kind(winget, frozen=True) == "installer"
    (winget.parent / U.MANAGED_MARKER).write_text("winget\n", encoding="utf-8")
    assert U.install_kind(winget, frozen=True) == "winget"
    (winget.parent / U.MANAGED_MARKER).write_text("something else", encoding="utf-8")
    assert U.install_kind(winget, frozen=True) == "installer"
    assert U.install_kind(_exe(tmp_path / "dist" / "VoxFlow"), frozen=True) == "portable"
    for kind in ("scoop", "winget", "source", "portable"):
        assert U.managed_text(kind).endswith(".")
    assert "scoop update voxflow" in U.managed_text("scoop") and U.managed_text("installer") == ""


# =========================================================================== schedule
def test_the_install_slot_starts_at_the_time_and_lasts_an_hour():
    tz = _dt.timezone.utc
    at = lambda h, m=0, d=15: _dt.datetime(2026, 11, d, h, m, tzinfo=tz)  # noqa: E731
    assert U.scheduled_slot(at(3), "03:00") == "2026-11-15"
    assert U.scheduled_slot(at(3, 59), "03:00") == "2026-11-15"
    assert U.scheduled_slot(at(4), "03:00") is None and U.scheduled_slot(at(2, 59), "03:00") is None
    assert U.scheduled_slot(at(0, 10, 16), "23:30") == "2026-11-15", "a slot may run past midnight"
    assert U.parse_hhmm("3:05") == (3, 5)
    for bad in ("24:00", "3", "03:60", "", None, "noon"):
        with pytest.raises(ValueError):
            U.parse_hhmm(bad)


# =========================================================================== the installer process
def test_the_installer_runs_silently_detached_and_without_pyinstaller_state(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", "x")
    monkeypatch.setenv("_MEIPASS2", "y")

    def popen(cmd, **kw):
        calls.append((cmd, kw))
        if len(calls) == 1:
            raise OSError("breakaway not allowed in this job")

    path = tmp_path / "VoxFlow-Setup-1.0.0.exe"
    assert U.launch_installer(path, True, tmp_path / "install.log", popen=popen)
    cmd, kw = calls[-1]
    assert cmd == [str(path), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/RELAUNCH=1",
                   f"/LOG={tmp_path / 'install.log'}"]
    assert kw["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert not any(k.upper().startswith(("_PYI", "_MEIPASS")) for k in kw["env"])
    assert kw["close_fds"] and kw["cwd"] == str(tmp_path)
    assert calls[0][1]["creationflags"] & 0x01000000 and not calls[1][1]["creationflags"] & 0x01000000
    assert U.installer_command(path, False)[-1] == "/RELAUNCH=0"

    def never(cmd, **kw):
        raise OSError("blocked by policy")

    assert U.launch_installer(path, False, popen=never) is False


def test_state_survives_a_damaged_file(tmp_path):
    path = tmp_path / U.STATE_FILE
    path.write_text(json.dumps({"last_check": "yesterday", "last_ok": "no", "skipped": 5, "latest": [1],
                                "remind_after": 12, "notified": "0.3.0", "extra": 1}), encoding="utf-8")
    st = U.UpdateState.load(path)
    assert (st.last_check, st.last_ok, st.skipped, st.latest, st.remind_after, st.notified) == \
        (0.0, True, "", None, 12.0, "0.3.0")
    path.write_text("{broken", encoding="utf-8")
    assert U.UpdateState.load(path) == U.UpdateState()
    path.write_text("[]", encoding="utf-8")
    assert U.UpdateState.load(path) == U.UpdateState()


# =========================================================================== the service
class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, qtbot, http=None, *, mode="notify", when="quit", at="03:00", kind="installer", busy=False,
         metered=False, idle=None, now=None, version="0.2.0", automatic=True):
    settings = {"update_mode": mode, "update_install_when": when, "update_time": at}
    flags = {"busy": busy}
    launched = []
    clock = Clock()
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    up = U.Updater(tmp_path / "data", settings.__getitem__, lambda: flags["busy"], version=version,
                   http=http if http is not None else github(), kind=kind, metered=lambda: metered,
                   idle=(lambda: idle), clock=clock, now=now, folder=tmp_path / "dl",
                   launcher=lambda path, relaunch, log: launched.append((path, relaunch)) or True,
                   automatic=automatic)
    (tmp_path / "data").mkdir(exist_ok=True)
    up.settings, up.flags, up.launched, up.clock = settings, flags, launched, clock
    up.events = {"available": [], "notice": [], "restart": 0}
    up.available.connect(lambda info: up.events["available"].append(info))
    up.notice.connect(lambda level, msg: up.events["notice"].append((level, msg)))
    up.restart_requested.connect(lambda: up.events.__setitem__("restart", up.events["restart"] + 1))
    up.qtbot = qtbot
    return up


def settle(up, timeout=3000):
    up.qtbot.waitUntil(lambda: up.phase in ("idle", "waiting", "restarting"), timeout=timeout)


def test_off_means_no_update_traffic_unless_check_now_is_clicked(tmp_path, qtbot, no_sockets):
    settings = {"update_mode": "off", "update_install_when": "quit", "update_time": "03:00"}
    up = U.Updater(tmp_path, settings.__getitem__, lambda: False, kind="installer",
                   folder=tmp_path / "dl")                                  # the real Http class
    up.start()
    assert not up._check_timer.isActive() and not up._slot_timer.isActive(), "no startup or daily check"
    assert up.check(user=False) is False and up.phase == "idle", "an automatic check never runs while Off"
    up._check_timer.timeout.emit()                     # even a timer that fired anyway does nothing
    qtbot.wait(50)
    assert no_sockets == [] and up.state.last_check == 0
    # the click is the user asking: exactly one request, to the API, and Off stays off afterwards
    up._http = http = github()
    assert up.check(user=True) is True
    qtbot.waitUntil(lambda: up.phase == "idle", timeout=3000)
    assert http.urls() == [API] and up.info()["available"]["version"] == "0.3.0"
    assert not up._check_timer.isActive(), "a manual check arms no automatic one"
    assert [u for u in http.urls() if "Setup" in u or "SHA256" in u] == [], "nothing is downloaded by itself"
    assert no_sockets == []
    up.close()


def test_off_through_the_controller_checks_only_on_click(tmp_path, qtbot, monkeypatch, no_sockets):
    from tests.controller_rig import make_rig
    http = github()
    rig = make_rig(tmp_path, qtbot, monkeypatch, config={"update_mode": "off"},
                   updates=lambda d, get, busy: U.Updater(d, get, busy, http=http, kind="installer",
                                                          folder=tmp_path / "dl"))
    try:
        assert not rig.c.updates._check_timer.isActive() and http.requests == []
        assert rig.c.check_updates() is True               # "Check now"
        rig.wait(lambda: rig.c.update_info()["phase"] == "idle")
        assert http.urls() == [API] and rig.spy.of("update_available")
        assert not rig.c.updates._check_timer.isActive()
        rig.c.set("update_mode", "notify")             # on again: a check is armed (after the startup delay)
        assert rig.c.updates._check_timer.isActive() and rig.c.updates._check_timer.interval() >= 90_000
        rig.c.set("update_mode", "off")
        assert not rig.c.updates._check_timer.isActive()
        assert http.urls() == [API] and no_sockets == []
    finally:
        rig.close()


def test_a_fake_controller_copy_never_reaches_the_network(tmp_path, qtbot, monkeypatch, no_sockets):
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert isinstance(rig.c.updates._http, U.OfflineHttp) and rig.c.updates.kind == "source"
        assert rig.c.check_updates() is True
        rig.wait(lambda: rig.c.update_info()["phase"] == "idle")
        info = rig.c.update_info()
        assert info["last_ok"] is False and "test copy" in info["last_result"] and no_sockets == []
        with pytest.raises(ValueError, match="no newer version"):
            rig.c.install_update("now")
    finally:
        rig.close()


def test_the_first_check_waits_a_random_few_minutes_then_once_a_day(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up._rng.seed(1)
    up.start()
    first = up._check_timer.interval()
    assert U.STARTUP_DELAY_S[0] * 1000 <= first <= U.STARTUP_DELAY_S[1] * 1000
    up.state.last_check = up.clock.t - 3600
    up._arm_check()
    assert abs(up._check_timer.interval() - 23 * 3600 * 1000) < 1000, "a day after the last check"
    up.close()
    up2 = make(tmp_path / "b", qtbot, automatic=False)
    up2.start()
    assert not up2._check_timer.isActive(), "automatic checks disabled: only Check now"
    up2.close()


def test_notify_mode_tells_once_per_version_and_honours_remind_and_skip(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    assert up.check(user=False)
    settle(up)
    info = up.info()
    assert info["available"]["version"] == "0.3.0" and info["last_ok"] and info["installable"]
    assert len(up.events["available"]) == 1 and up.events["available"][0]["available"]["notes"] == "Added\n• x"
    assert [u for u in up._http.urls() if "Setup" in u] == [], "notify mode downloads nothing by itself"
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 1, "the same version is announced once"
    up.remind_later()
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 1, "not before tomorrow"
    up.clock.t += U.REMIND_AFTER_S + 1
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 2, "reminded the next day"
    up.skip()
    assert up.info()["skipped"] and up.state.skipped == "0.3.0"
    up.clock.t += 10 * U.REMIND_AFTER_S
    up.state.notified = ""
    up.check(user=False)
    settle(up)
    assert len(up.events["available"]) == 2, "a skipped version is not offered again"
    up.check(user=True)
    settle(up)
    assert len(up.events["available"]) == 3, "Check now shows it anyway"
    # it all survives a restart
    again = make(tmp_path, qtbot)
    assert again.state.skipped == "0.3.0" and again.latest().version == "0.3.0"
    up.close()
    again.close()


def test_up_to_date_and_failed_checks_are_quiet(tmp_path, qtbot, caplog):
    up = make(tmp_path, qtbot, http=github("0.2.0"), version="0.2.0")
    up.check(user=False)
    settle(up)
    assert up.info()["available"] is None and up.info()["last_result"] == "VoxFlow is up to date."
    assert up.events["available"] == [] and up.events["notice"] == []
    down = make(tmp_path / "b", qtbot, http=FakeHttp())
    down.check(user=False)
    settle(down)
    info = down.info()
    assert info["last_ok"] is False and "could not be reached" in info["last_result"] and info["last_check"]
    assert down.events["notice"] == [] and "update check failed" in caplog.text
    assert down._check_timer.isActive(), "tried again tomorrow"

    def boom(_h):
        raise RuntimeError("bug")

    odd = make(tmp_path / "c", qtbot, http=FakeHttp({API: boom}))
    odd.check(user=True)
    settle(odd)
    assert odd.info()["last_result"] == "The update check did not work."
    for u in (up, down, odd):
        u.close()


def test_install_when_i_quit_downloads_and_runs_setup_only_on_a_real_quit(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    assert up.info()["install_on_quit"] and up.launched == []
    assert up.on_exit(user_quit=False) is False, "Windows ending the session: no installer"
    assert up.on_exit(user_quit=True) is True
    (path, relaunch), = up.launched
    assert path.name == "VoxFlow-Setup-0.3.0.exe" and relaunch is False
    assert up.state.install_on_quit == "" and up.on_exit(user_quit=True) is False, "once"
    up.close()


def test_install_now_waits_until_voxflow_is_idle(tmp_path, qtbot):
    up = make(tmp_path, qtbot, busy=True)
    up.check()
    settle(up)
    up.install("now")
    qtbot.waitUntil(lambda: up.phase == "waiting", timeout=3000)
    assert up._install_timer.isActive() and up.events["restart"] == 0
    up._install_tick()
    assert up.events["restart"] == 0, "never while recording, processing or in command mode"
    up.flags["busy"] = False
    up._install_tick()
    assert up.events["restart"] == 1 and up.phase == "restarting" and not up._install_timer.isActive()
    assert up.on_exit(user_quit=True) is True and up.launched[0][1] is True, "it starts VoxFlow again"
    up.close()


def test_install_now_with_the_file_already_there_restarts_at_once(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    up.install("now")
    assert up.events["restart"] == 1
    up.close()


def test_a_file_changed_after_its_check_is_not_run(tmp_path, qtbot, caplog):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").write_bytes(b"swapped")
    assert up.on_exit(user_quit=True) is False and up.launched == []
    assert not (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").exists() and "changed on disk" in caplog.text
    up.close()


def test_a_failed_download_the_user_asked_for_is_said_and_nothing_runs(tmp_path, qtbot):
    up = make(tmp_path, qtbot, http=github(sums=f"{'0' * 64}  VoxFlow-Setup-0.3.0.exe\n"))
    up.check()
    settle(up)
    up.install("now")
    qtbot.waitUntil(lambda: up.phase == "idle" and bool(up.error), timeout=3000)
    assert "does not match" in up.info()["error"] and up.events["restart"] == 0
    assert up.events["notice"] == [("warning", up.info()["error"])]
    assert up.on_exit(user_quit=True) is False and up.launched == []
    up.close()


def test_automatic_mode_downloads_in_the_background_and_installs_on_quit(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto")
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["install_on_quit"], timeout=3000)
    assert up.events["available"] == [], "nothing to decide: it says what it will do"
    assert up.events["notice"] == [("info", "VoxFlow 0.3.0 will be installed when you quit VoxFlow.")]
    assert up.on_exit(user_quit=True) and up.launched[0][1] is False
    up.close()


def test_automatic_mode_waits_for_an_unmetered_connection(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto", metered=True)
    up.check(user=False)
    settle(up)
    assert "metered" in up.info()["error"] and not up.info()["downloaded"]
    assert [u for u in up._http.urls() if "Setup" in u or "SHA256" in u] == []
    up.install("now")                                   # the user's own choice downloads anyway
    qtbot.waitUntil(lambda: up.events["restart"] == 1, timeout=3000)
    up.close()


def test_automatic_mode_leaves_managed_copies_alone(tmp_path, qtbot):
    for kind in ("scoop", "winget", "source", "portable"):
        up = make(tmp_path / kind, qtbot, mode="auto", kind=kind)
        up.check(user=False)
        settle(up)
        assert len(up.events["available"]) == 1, f"{kind}: notify only"
        assert [u for u in up._http.urls() if "Setup" in u or "SHA256" in u] == []
        info = up.info()
        assert not info["can_install"] and not info["installable"] and info["managed"] == U.managed_text(kind)
        with pytest.raises(ValueError, match=U.managed_text(kind)[:20]):
            up.install("now")
        up.close()


def test_a_scheduled_install_waits_for_its_slot_an_idle_pc_and_an_idle_voxflow(tmp_path, qtbot):
    moment = {"now": _dt.datetime(2026, 11, 15, 2, 0).astimezone()}
    up = make(tmp_path, qtbot, mode="auto", when="time", at="03:00", idle=30.0, now=lambda: moment["now"])
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    assert up._slot_timer.isActive() and "03:00" in up.events["notice"][-1][1]
    up._slot_tick()
    assert up.events["restart"] == 0, "not before 03:00"
    moment["now"] = moment["now"].replace(hour=3, minute=5)
    up._slot_tick()
    assert up.events["restart"] == 0, "someone used the PC 30 s ago"
    up._idle = lambda: 900.0
    up.flags["busy"] = True
    up._slot_tick()
    assert up.events["restart"] == 0, "never while dictating"
    up.flags["busy"] = False
    up._slot_tick()
    assert up.events["restart"] == 1 and up.state.slot_done == "2026-11-15"
    assert up.on_exit(user_quit=True) and up.launched[0][1] is True
    up.close()


def test_a_scheduled_install_runs_once_per_slot_and_stops_when_the_setting_changes(tmp_path, qtbot):
    moment = _dt.datetime(2026, 11, 15, 3, 10).astimezone()
    up = make(tmp_path, qtbot, mode="auto", when="time", idle=None, now=lambda: moment)
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    up.state.slot_done = "2026-11-15"
    up._slot_tick()
    assert up.events["restart"] == 0
    up.settings["update_install_when"] = "quit"
    up.settings_changed("update_install_when")
    assert not up._slot_timer.isActive() and up.info()["install_on_quit"]
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    assert not up.info()["install_on_quit"] and not up._check_timer.isActive()
    up._slot_tick()
    assert up.events["restart"] == 0
    up.close()


def test_skipping_deletes_the_download_and_cancels_install_on_quit(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    up.skip()
    assert not up.info()["install_on_quit"] and list((tmp_path / "dl").iterdir()) == []
    assert up.on_exit(user_quit=True) is False
    up.install("quit")                                  # changing one's mind un-skips it
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    assert not up.info()["skipped"]
    up.close()


def test_install_refuses_what_it_cannot_do(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    with pytest.raises(ValueError, match="no newer version"):
        up.install("now")
    up.skip()                                             # nothing to skip: no error, nothing stored
    assert up.state.skipped == ""
    up.check()
    settle(up)
    with pytest.raises(ValueError, match="install now or when you quit"):
        up.install("tomorrow")
    no_exe = make(tmp_path / "b", qtbot, http=github(assets=[]))
    no_exe.check()
    settle(no_exe)
    with pytest.raises(ValueError, match="no Windows installer"):
        no_exe.install("now")
    up.close()
    no_exe.close()


def test_after_the_update_the_old_installer_is_cleaned_up(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    up.close()
    after = make(tmp_path, qtbot, version="0.3.0")       # the new version starts
    assert after.latest() is None and after.state.downloaded == "" and after.state.install_on_quit == ""
    assert list((tmp_path / "dl").iterdir()) == []
    after.close()


def test_the_controller_never_installs_while_busy_and_runs_setup_last_on_shutdown(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    launched = []
    http = github()
    holder = {}

    def factory(d, get, busy):
        holder["up"] = U.Updater(d, get, busy, http=http, kind="installer", folder=tmp_path / "dl",
                                 launcher=lambda p, r, log: launched.append((p, r)) or True, automatic=False)
        return holder["up"]

    rig = make_rig(tmp_path, qtbot, monkeypatch, updates=factory)
    restarts = []
    rig.c.restart_requested.connect(lambda: restarts.append(1))
    try:
        assert rig.c.check_updates()
        rig.wait(lambda: rig.c.update_info()["available"] is not None)
        assert rig.spy.of("update_available") and rig.spy.of("update_changed")
        rig.c._active = True                            # a dictation is being recorded
        rig.c.install_update("now")
        rig.wait(lambda: rig.c.update_info()["phase"] == "waiting")
        holder["up"]._install_tick()
        assert restarts == []
        rig.c._active = False
        rig.c._capturing = True                         # a shortcut is being recorded
        holder["up"]._install_tick()
        assert restarts == []
        rig.c._capturing = False
        holder["up"]._install_tick()
        assert restarts == [1] and rig.c._quit_by_user
        assert launched == [], "the installer starts only after VoxFlow has shut down"
    finally:
        rig.close()
    assert len(launched) == 1 and launched[0][1] is True


# =========================================================================== edges
def test_http_passes_a_416_back_and_a_refused_redirect_up(monkeypatch):
    import urllib.error

    err = urllib.error.HTTPError(API, 416, "Range Not Satisfiable", {}, io.BytesIO(b""))

    class Opener:
        def __init__(self, exc):
            self.exc = exc

        def open(self, req, timeout):
            raise self.exc

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: Opener(err))
    assert U.Http().open(f"{REPO}/releases/download/v1.0.0/x.exe", "asset") is err
    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: Opener(U.UpdateError("redirected away")))
    with pytest.raises(U.UpdateError, match="redirected away"):
        U.Http().open(API, "api")
    with pytest.raises(U.UpdateError, match="test copy"):
        U.OfflineHttp().open(API, "api")


def _raiser(exc):
    def fn(*_a, **_k):
        raise exc
    return fn


def test_small_helpers_fail_safe(tmp_path, monkeypatch):
    assert U.Release.from_dict({"version": "1.0.0"}) is None, "fields missing"
    assert U.Release.from_dict({**U.parse_release(release_json()).to_dict(), "version": "one"}) is None
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 10)
    with pytest.raises(U.Cancelled):
        U.sha256_file(f, cancel=lambda: True)
    folder = tmp_path / "dl"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.part").write_bytes(b"1")
    U.clean_downloads(folder)
    assert list(folder.iterdir()) == []
    U.clean_downloads(tmp_path / "missing")
    (folder / "locked.exe").write_bytes(b"1")
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", _raiser(PermissionError(13, "Access is denied")))
        U.clean_downloads(folder)                          # logged, not raised
    exe = _exe(tmp_path / "Programs" / "VoxFlow")
    with monkeypatch.context() as m:
        m.setattr(Path, "resolve", _raiser(OSError("gone")))
        assert U.install_kind(exe, frozen=True) == "portable"
    with monkeypatch.context() as m:
        m.setattr(U.cfgmod, "_atomic_write", _raiser(OSError("disk full")))
        U.UpdateState().save(tmp_path / "s.json")          # logged, not raised
    assert not (tmp_path / "s.json").exists()


def test_the_windows_probes_answer_or_say_they_cannot_tell(monkeypatch):
    metered, idle = U.connection_metered(), U.idle_seconds()
    assert metered in (True, False, None) and (idle is None or idle >= 0)
    monkeypatch.setattr(U.sys, "platform", "linux")
    assert U.connection_metered() is None and U.idle_seconds() is None


def test_launch_flags_off_windows(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(U.sys, "platform", "linux")
    assert U.launch_installer(tmp_path / "s.exe", False, popen=lambda cmd, **kw: calls.append(kw))
    assert len(calls) == 1 and calls[0]["creationflags"] == 0


class Gate:
    """An installer route that holds the download open until ``release()``."""

    def __init__(self):
        import threading
        self.event = threading.Event()

    def __call__(self, headers):
        self.event.wait(5)
        return Resp(200, INSTALLER)

    def release(self):
        self.event.set()


def gated(tmp_path, qtbot, **kw):
    http = github()
    gate = Gate()
    http.routes[f"{REPO}/releases/download/v0.3.0/VoxFlow-Setup-0.3.0.exe"] = gate
    up = make(tmp_path, qtbot, http=http, **kw)
    up.check()
    qtbot.waitUntil(lambda: up.phase != "checking", timeout=3000)     # automatic mode goes on downloading
    return up, gate


def test_choices_made_while_a_download_runs(tmp_path, qtbot):
    up, gate = gated(tmp_path, qtbot)
    up.install("quit")
    assert up.phase == "downloading"
    up.install("now")                                    # the running download is reused, not started twice
    assert up._after_download == "now"
    assert up.check() is False, "no check while downloading"
    up._on_progress(0.5)
    assert up.progress == 0.5
    gate.release()
    qtbot.waitUntil(lambda: up.events["restart"] == 1, timeout=3000)
    up._on_progress(0.9)                                 # a late progress report after the end is ignored
    assert up.progress is None
    up.close()


def test_skip_or_off_while_downloading_cancels_quietly(tmp_path, qtbot):
    up, gate = gated(tmp_path, qtbot)
    up.install("quit")
    up.skip()
    gate.release()
    qtbot.waitUntil(lambda: up.phase == "idle", timeout=3000)
    assert up.events["notice"] == [] and not up.info()["downloaded"]
    up2, gate2 = gated(tmp_path / "b", qtbot, mode="auto")
    qtbot.waitUntil(lambda: up2.phase == "downloading", timeout=3000)
    up2.settings["update_mode"] = "off"
    up2.settings_changed("update_mode")
    gate2.release()
    qtbot.waitUntil(lambda: up2.phase == "idle", timeout=3000)
    assert up2.events["notice"] == [] and up2.error == ""
    up.close()
    up2.close()


def test_install_refuses_while_checking_and_an_automatic_failure_is_quiet(tmp_path, qtbot):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.phase = "checking"
    with pytest.raises(ValueError, match="busy with the update"):
        up.install("now")
    up.phase = "idle"
    auto = make(tmp_path / "b", qtbot, mode="auto", http=github(sums=f"{'0' * 64}  VoxFlow-Setup-0.3.0.exe\n"))
    auto.check(user=True)
    qtbot.waitUntil(lambda: auto.phase == "idle" and bool(auto.error), timeout=3000)
    assert auto.events["notice"] == [], "an automatic download that failed is shown in Settings only"
    assert len(auto.events["available"]) == 1, "Check now in automatic mode still shows the dialog"
    up.close()
    auto.close()


def test_automatic_mode_reuses_a_verified_download_after_a_restart(tmp_path, qtbot):
    up = make(tmp_path, qtbot, mode="auto")
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["install_on_quit"], timeout=3000)
    up.close()
    again = make(tmp_path, qtbot, mode="auto")
    again.state.install_on_quit = ""
    requests = len(again._http.requests)
    again._auto_next()
    assert again.info()["install_on_quit"] and len(again._http.requests) == requests
    again.close()


def test_odd_states_of_the_install_timers(tmp_path, qtbot):
    moment = _dt.datetime(2026, 11, 15, 3, 10).astimezone()
    up = make(tmp_path, qtbot, mode="auto", when="time", now=lambda: moment)
    up._slot_tick()                                      # nothing available: the timer stops
    assert not up._slot_timer.isActive()
    up.check(user=False)
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    up.settings["update_time"] = "bogus"
    up._slot_tick()
    assert up.events["restart"] == 0
    up.settings["update_time"] = "03:00"
    (tmp_path / "dl" / "VoxFlow-Setup-0.3.0.exe").unlink()
    up._slot_tick()
    assert up.events["restart"] == 0 and "gone" in up.error and not up.state.downloaded
    up.phase = "waiting"
    up.state.latest = None
    up._install_timer.start()
    up._install_tick()
    assert not up._install_timer.isActive()
    up.close()


def test_on_exit_does_not_run_an_unreadable_file(tmp_path, qtbot, monkeypatch):
    up = make(tmp_path, qtbot)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    monkeypatch.setattr(U, "open_exclusive", _raiser(OSError("locked")))
    assert up.on_exit(user_quit=True) is False and up.launched == []
    up.close()


def test_a_thread_result_after_close_is_dropped(tmp_path, qtbot):
    up, gate = gated(tmp_path, qtbot)
    up.install("quit")
    up.close()
    gate.release()
    qtbot.wait(100)
    assert up.phase == "downloading" and not up.state.downloaded
    up._on_thread("checked", (False, None))               # queued before close: ignored


def test_off_while_install_now_waits_releases_the_wait(tmp_path, qtbot):
    up = make(tmp_path, qtbot, busy=True)
    up.check()
    settle(up)
    up.install("now")
    qtbot.waitUntil(lambda: up.phase == "waiting", timeout=3000)
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    assert up.phase == "idle" and not up._install_timer.isActive()
    up.flags["busy"] = False
    up._install_tick()
    assert up.events["restart"] == 0
    up.close()


def test_skip_or_install_when_i_quit_cancel_a_waiting_install_now(tmp_path, qtbot):
    up = make(tmp_path, qtbot, busy=True)
    up.check()
    settle(up)
    up.install("now")
    qtbot.waitUntil(lambda: up.phase == "waiting", timeout=3000)
    up.install("quit")
    assert up.phase == "idle" and not up._install_timer.isActive() and up.info()["install_on_quit"]
    up.install("now")
    assert up.phase == "waiting"
    up.skip()
    assert up.phase == "idle" and not up._install_timer.isActive()
    up.flags["busy"] = False
    up._install_tick()
    assert up.events["restart"] == 0
    up.close()


def test_a_daily_check_during_a_download_tries_again_later(tmp_path, qtbot):
    up, gate = gated(tmp_path, qtbot)
    up.install("quit")
    assert up.check(user=False) is False
    assert up._check_timer.isActive() and up._check_timer.interval() == U.RETRY_BUSY_MS
    up._check_timer.stop()
    assert up.check(user=True) is False and not up._check_timer.isActive(), "Check now just says no"
    gate.release()
    qtbot.waitUntil(lambda: up.phase == "idle", timeout=3000)
    up.close()


def test_the_controller_passes_choices_through_and_survives_a_failing_installer_start(tmp_path, qtbot, monkeypatch,
                                                                                         caplog):
    from tests.controller_rig import make_rig
    http = github()
    up = U.Updater(tmp_path / "upd", {"update_mode": "notify", "update_install_when": "quit",
                                      "update_time": "03:00"}.__getitem__, lambda: False, http=http, kind="installer",
                   folder=tmp_path / "dl", automatic=False)
    rig = make_rig(tmp_path, qtbot, monkeypatch, updates=up)          # a ready Updater instance
    try:
        assert rig.c.updates is up
        rig.c.check_updates()
        rig.wait(lambda: rig.c.update_info()["available"] is not None)
        rig.c.remind_update_later()
        assert up.state.remind_after > 0
        rig.c.skip_update()
        assert rig.c.update_info()["skipped"]
        rig.c.prepare_quit()
        monkeypatch.setattr(up, "on_exit", _raiser(RuntimeError("boom")))
    finally:
        rig.close()                                    # must not raise
    assert "could not start the update installer" in caplog.text


def test_a_release_date_that_cannot_be_read_is_left_out():
    from voxflow.ui import updates as UP
    assert UP._published({"published": "soon"}) == ""
    assert UP._published({"published": "2026-12-01T10:00:00Z"}) == " Released December 1, 2026."


# =========================================================================== Off and Skip delete; the folder stays bounded
def _downloaded(tmp_path, qtbot, **kw):
    up = make(tmp_path, qtbot, **kw)
    up.check()
    settle(up)
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    return up


def test_switching_to_off_deletes_the_download_and_cancels_every_pending_install(tmp_path, qtbot):
    up = _downloaded(tmp_path, qtbot, busy=True)
    up.install("now")
    assert up.phase == "waiting" and up.info()["install_on_quit"]
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    info = up.info()
    assert list((tmp_path / "dl").iterdir()) == [], "deleted at once"
    assert not info["downloaded"] and not info["install_on_quit"] and info["phase"] == "idle"
    assert not up._install_timer.isActive() and not up._slot_timer.isActive()
    up.flags["busy"] = False
    up._install_tick()
    assert up.events["restart"] == 0 and up.on_exit(user_quit=True) is False and up.launched == []
    again = make(tmp_path, qtbot, mode="off")             # and nothing comes back after a restart
    assert not again.info()["downloaded"] and again.on_exit(user_quit=True) is False
    up.close()
    again.close()


def test_switching_to_off_during_a_download_leaves_no_part_behind(tmp_path, qtbot):
    up, gate = gated(tmp_path, qtbot)
    up.install("quit")
    assert up.phase == "downloading"
    up.settings["update_mode"] = "off"
    up.settings_changed("update_mode")
    gate.release()
    qtbot.waitUntil(lambda: up.phase == "idle", timeout=3000)
    assert not (tmp_path / "dl").exists() or list((tmp_path / "dl").iterdir()) == []
    assert not up.info()["downloaded"] and not up.info()["install_on_quit"]
    up.close()


def test_skipping_deletes_the_file_of_that_version(tmp_path, qtbot):
    up = _downloaded(tmp_path, qtbot)
    up.skip()
    assert list((tmp_path / "dl").iterdir()) == [] and not up.info()["downloaded"]
    up.close()


def test_the_download_folder_keeps_at_most_the_current_installer(tmp_path, qtbot):
    up = _downloaded(tmp_path, qtbot)
    up.close()
    folder = tmp_path / "dl"
    for junk in ("VoxFlow-Setup-0.2.5.exe", "VoxFlow-Setup-0.2.9.exe.part", "install.log", "random.tmp"):
        (folder / junk).write_bytes(b"x" * 100)
    (folder / "leftover").mkdir()
    again = make(tmp_path, qtbot)                         # VoxFlow starts again: the folder is swept
    assert sorted(p.name for p in folder.iterdir()) == ["VoxFlow-Setup-0.3.0.exe"]
    assert again.info()["downloaded"] and again.info()["install_on_quit"]
    again.close()


def test_a_newer_release_replaces_an_older_download(tmp_path, qtbot):
    up = _downloaded(tmp_path, qtbot)
    up._http = github("0.4.0")
    up.check(user=False)
    settle(up)
    assert up.info()["available"]["version"] == "0.4.0"
    assert list((tmp_path / "dl").iterdir()) == [], "the 0.3.0 installer is gone"
    assert not up.info()["downloaded"] and not up.info()["install_on_quit"]
    up.install("quit")
    qtbot.waitUntil(lambda: up.info()["downloaded"], timeout=3000)
    assert sorted(p.name for p in (tmp_path / "dl").iterdir()) == ["VoxFlow-Setup-0.4.0.exe"]
    up.close()


def test_after_the_update_even_setups_log_is_removed(tmp_path, qtbot):
    up = _downloaded(tmp_path, qtbot)
    assert up.on_exit(user_quit=True)
    (tmp_path / "dl" / "install.log").write_text("Setup log", encoding="utf-8")   # what Setup writes
    up.close()
    after = make(tmp_path, qtbot, version="0.3.0")
    assert list((tmp_path / "dl").iterdir()) == []
    after.close()


def test_a_fake_controller_never_touches_the_real_download_folder(tmp_path, qtbot, monkeypatch):
    from tests.controller_rig import make_rig
    rig = make_rig(tmp_path, qtbot, monkeypatch)
    try:
        assert rig.c.updates.folder == rig.data_dir / "update-offline"
        assert rig.c.updates.folder != U.download_folder()
    finally:
        rig.close()


def test_a_still_running_installer_is_deleted_once_windows_releases_it(tmp_path, qtbot, monkeypatch):
    """Seen in tests/e2e_update: Inno Setup's [Run] entry starts the new VoxFlow while Setup (the downloaded exe)
    is still running, so the first delete fails; the retry must remove it once Setup has exited."""
    monkeypatch.setattr(U, "TIDY_RETRY_MS", 50)
    folder = tmp_path / "dl"
    folder.mkdir()
    exe = folder / "VoxFlow-Setup-0.3.0.exe"
    exe.write_bytes(b"MZ")
    (tmp_path / "data").mkdir()
    U.UpdateState(downloaded="0.3.0", sha256="ab" * 32).save(tmp_path / "data" / U.STATE_FILE)
    locked = {"on": True}
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if locked["on"] and self.name == exe.name:
            raise PermissionError(32, "The process cannot access the file because it is being used")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    up = make(tmp_path, qtbot, version="0.3.0")            # the new version starts while Setup still runs
    assert exe.exists() and up._tidy_timer.isActive()
    locked["on"] = False                                   # Setup has exited
    qtbot.waitUntil(lambda: not exe.exists(), timeout=3000)
    assert not up.info()["downloaded"]
    up.close()
    assert not up._tidy_timer.isActive()
