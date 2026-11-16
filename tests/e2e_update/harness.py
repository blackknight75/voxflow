"""A local stand-in for GitHub, for the update end-to-end tests (tests/e2e_update).

What is real: urllib, TLS with certificate and host-name checks, the updater's redirect handler and allow-list,
the files in the download folder, ``launch_installer`` starting a real (tiny) Windows exe, Qt timers.

What is faked, in the test process only:

* DNS: ``socket.create_connection`` sends the GitHub host names to this machine's local server (every other
  host is refused, so a test can never reach the real internet by accident). TLS still checks the original
  host name, so the URLs the updater sees are the real ``https://api.github.com/...`` ones and nothing in the
  shipped allow-list is changed.
* Trust: a throw-away certificate authority made with ``openssl`` for this run, given to OpenSSL through
  ``SSL_CERT_FILE`` (``ssl.create_default_context`` reads it in addition to the Windows store).
* The installer: a PyInstaller one-file exe (``stub_installer.py``) that writes its argv to a JSON file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = "blackknight75/voxflow"
API_HOST = "api.github.com"
WEB_HOST = "github.com"
ASSET_HOST = "objects.githubusercontent.com"       # where github.com redirects release downloads
EVIL_HOST = "downloads.evil.example"                # not on the allow-list
FAKE_HOSTS = (API_HOST, WEB_HOST, ASSET_HOST, "release-assets.githubusercontent.com", EVIL_HOST)
HERE = Path(__file__).resolve().parent


# =========================================================================== certificates
def find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", "")):
        for sub in (r"Git\mingw64\bin\openssl.exe", r"Git\usr\bin\openssl.exe"):
            p = Path(base) / sub
            if base and p.is_file():
                return str(p)
    return None


def make_pki(folder: Path) -> dict:
    """A CA and one server certificate for every fake host. Returns paths (ca, cert, key)."""
    openssl = find_openssl()
    if openssl is None:
        raise RuntimeError("openssl was not found (Git for Windows ships one)")
    folder.mkdir(parents=True, exist_ok=True)
    ca_key, ca, key, csr, cert = (folder / n for n in ("ca.key", "ca.pem", "leaf.key", "leaf.csr", "leaf.pem"))
    ext = folder / "leaf.ext"
    ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                   "extendedKeyUsage=serverAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n"
                   "subjectAltName=" + ",".join(f"DNS:{h}" for h in FAKE_HOSTS) + "\n", encoding="ascii")

    def run(*args):
        subprocess.run([openssl, *args], check=True, capture_output=True, timeout=60)

    run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "2", "-keyout", str(ca_key),
        "-out", str(ca), "-subj", "/CN=VoxFlow update e2e test CA",
        "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        "-addext", "subjectKeyIdentifier=hash")
    run("req", "-newkey", "rsa:2048", "-nodes", "-sha256", "-keyout", str(key), "-out", str(csr),
        "-subj", f"/CN={API_HOST}")
    run("x509", "-req", "-sha256", "-days", "2", "-in", str(csr), "-CA", str(ca), "-CAkey", str(ca_key),
        "-CAcreateserial", "-out", str(cert), "-extfile", str(ext))
    return {"ca": ca, "cert": cert, "key": key}


# =========================================================================== the fake installer
STUB_SOURCE = HERE / "stub_installer.py"


def build_onefile(source: Path, cache: Path) -> Path:
    """A PyInstaller one-file exe of ``source``, cached by the source's hash (a build takes ~20 s)."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    out = cache / digest / f"{source.stem}.exe"
    if out.is_file():
        return out
    work = cache / digest / "build"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--noconfirm", "--console", "--log-level",
                    "WARN", "--name", source.stem, "--distpath", str(out.parent), "--workpath",
                    str(work / "w"), "--specpath", str(work), str(source)],
                   check=True, capture_output=True, timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    return out


def build_stub_installer(cache: Path) -> Path:
    return build_onefile(STUB_SOURCE, cache)


def wait_for_record(path: Path, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.1)
    raise AssertionError(f"the installer did not run within {timeout:.0f} s (no {path.name})")


# =========================================================================== the server
@dataclass
class Asset:
    name: str
    data: bytes
    size: int | None = None              # what the API says (None: the real length)


@dataclass
class Scenario:
    """What the fake GitHub answers. Tests change it between steps; every request is recorded."""
    version: str = "0.3.0"
    body: str = "## Added\n- A new thing\n\n![tracker](https://downloads.evil.example/pixel.png)\n" \
                "<img src=\"https://downloads.evil.example/img.gif\">\nSee [the docs](https://downloads.evil.example/docs)."
    installer: bytes = b""
    draft: bool = False
    prerelease: bool = False
    tag: str | None = None               # default "v" + version
    with_sums: bool = True
    sums_override: str | None = None     # SHA256SUMS.txt text instead of the right one
    api_status: int = 200
    api_raw: bytes | None = None         # a raw API body instead of the JSON
    api_delay: float = 0.0
    asset_redirect: str | None = None    # where github.com sends the installer (default: ASSET_HOST)
    truncate_once: int | None = None     # the next installer download stops after this many bytes
    asset_status: int = 200
    extra_assets: list = field(default_factory=list)
    requests: list = field(default_factory=list)

    def installer_name(self) -> str:
        return f"VoxFlow-Setup-{self.version}.exe"

    def sums_text(self) -> str:
        if self.sums_override is not None:
            return self.sums_override
        return f"{hashlib.sha256(self.installer).hexdigest()}  {self.installer_name()}\n" \
               f"{'0' * 64}  something-else.zip\n"

    def release_json(self) -> dict:
        tag = self.tag or f"v{self.version}"
        dl = f"https://{WEB_HOST}/{REPO}/releases/download/{tag}"
        assets = [{"name": self.installer_name(), "size": len(self.installer),
                   "browser_download_url": f"{dl}/{self.installer_name()}"}]
        if self.with_sums:
            assets.append({"name": "SHA256SUMS.txt", "size": len(self.sums_text()),
                           "browser_download_url": f"{dl}/SHA256SUMS.txt"})
        assets += self.extra_assets
        return {"tag_name": tag, "name": f"VoxFlow {self.version}", "draft": self.draft,
                "prerelease": self.prerelease, "html_url": f"https://{WEB_HOST}/{REPO}/releases/tag/{tag}",
                "published_at": "2026-12-01T10:00:00Z", "body": self.body, "assets": assets}

    def count(self, host: str | None = None, path_part: str = "") -> int:
        return sum(1 for r in self.requests if (host is None or r["host"] == host) and path_part in r["path"])


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeGitHub/1.0"

    def log_message(self, fmt, *args):            # quiet
        pass

    def do_GET(self):  # noqa: N802 - http.server API
        sc: Scenario = self.server.scenario
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        sc.requests.append({"host": host, "path": self.path, "headers": dict(self.headers.items()),
                            "tls": isinstance(self.connection, ssl.SSLSocket), "time": time.time()})
        try:
            self._route(sc, host)
        except (ConnectionError, ssl.SSLError):
            pass

    def _send(self, status: int, body: bytes = b"", headers: dict | None = None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self, sc: Scenario, host: str):
        tag = sc.tag or f"v{sc.version}"
        if host == API_HOST and self.path == f"/repos/{REPO}/releases/latest":
            if sc.api_delay:
                time.sleep(sc.api_delay)
            if sc.api_status != 200:
                return self._send(sc.api_status, b'{"message": "nope"}', {"Content-Type": "application/json"})
            body = sc.api_raw if sc.api_raw is not None else json.dumps(sc.release_json()).encode()
            return self._send(200, body, {"Content-Type": "application/json; charset=utf-8"})
        prefix = f"/{REPO}/releases/download/{tag}/"
        if host == WEB_HOST and self.path.startswith(prefix):
            name = self.path[len(prefix):]
            if name == "SHA256SUMS.txt" and sc.with_sums:
                target = f"https://{ASSET_HOST}/github-production-release-asset/1/sums?name=SHA256SUMS.txt"
            elif name == sc.installer_name():
                target = sc.asset_redirect or \
                    f"https://{ASSET_HOST}/github-production-release-asset/1/exe?name={name}"
            else:
                return self._send(404, b"Not Found")
            return self._send(302, b"", {"Location": target})
        if host in (ASSET_HOST, EVIL_HOST) and self.path.startswith("/github-production-release-asset/1/"):
            if "/sums" in self.path:
                return self._send(200, sc.sums_text().encode(), {"Content-Type": "text/plain"})
            if sc.asset_status != 200:
                return self._send(sc.asset_status, b"error")
            return self._installer(sc)
        return self._send(404, b"Not Found")

    def _installer(self, sc: Scenario):
        data = sc.installer
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes=") and rng.endswith("-"):
            start = int(rng[6:-1])
            if start >= len(data):
                return self._send(416, b"", {"Content-Range": f"bytes */{len(data)}"})
        part = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        cut = sc.truncate_once
        if cut is not None:
            sc.truncate_once = None
            self.wfile.write(part[:cut])
            self.wfile.flush()
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        self.wfile.write(part)


class FakeGitHub:
    """An HTTPS server (all fake hosts, by the Host header) and a plain-HTTP one on 127.0.0.1."""

    def __init__(self, pki: dict, scenario: Scenario):
        self.scenario = scenario
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(pki["cert"]), str(pki["key"]))
        self.https = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.https.socket = ctx.wrap_socket(self.https.socket, server_side=True)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        for srv in (self.https, self.http):
            srv.scenario = scenario
            srv.daemon_threads = True
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.network = True               # False: every connection is refused (no network)
        self.dns_lookups: list[tuple[str, int]] = []

    @property
    def ports(self) -> tuple[int, int]:
        return self.https.server_address[1], self.http.server_address[1]

    def close(self):
        for srv in (self.https, self.http):
            srv.shutdown()
            srv.server_close()

    def connector(self, real):
        """A replacement for socket.create_connection: the fake hosts go to the local servers, nothing else
        goes anywhere."""
        https_port, http_port = self.ports

        def create_connection(address, *args, **kwargs):
            host, port = address[0], address[1]
            self.dns_lookups.append((str(host), int(port)))
            if str(host).lower() not in FAKE_HOSTS:
                raise OSError(f"e2e harness: no connections to {host}")
            if not self.network:
                raise ConnectionRefusedError(10061, "No connection could be made (e2e: network off)")
            local = https_port if int(port) == 443 else http_port
            return real(("127.0.0.1", local), *args, **kwargs)

        return create_connection


class Clock:
    """``time.time`` and ``datetime.now`` the Updater sees; tests move it forward."""

    def __init__(self, start: float | None = None):
        self.t = time.time() if start is None else start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
