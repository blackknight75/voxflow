"""The real GitHub API through the updater's own HTTP code: anonymous GETs only, a few hundred KB in total.

Runs only with ``VOXFLOW_UPDATE_E2E=1`` and ``VOXFLOW_UPDATE_LIVE=1`` (it needs the internet). Only the repository
name is pointed elsewhere, in this test process, because VoxFlow's own releases have no small asset to fetch.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse

import pytest

from voxflow import updater as U

pytestmark = [pytest.mark.update_e2e,
              pytest.mark.skipif(os.environ.get("VOXFLOW_UPDATE_LIVE") != "1",
                                 reason="needs the internet: set VOXFLOW_UPDATE_LIVE=1")]

# fzf publishes "<name>_<version>_checksums.txt" (sha256sum format) next to small archives (~1.5 MB)
OTHER_REPO = "junegunn/fzf"


def note(test: str, evidence: str) -> None:
    out = os.environ.get("VOXFLOW_E2E_EVIDENCE")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"test": test, "evidence": evidence}) + "\n")


@pytest.fixture
def redirects(monkeypatch):
    seen: list[str] = []
    real = U.checked_redirect().redirect_request

    def spy(self, req, fp, code, msg, headers, newurl):
        seen.append(newurl)
        return real(self, req, fp, code, msg, headers, newurl)

    monkeypatch.setattr(U.checked_redirect(), "redirect_request", spy)
    return seen


def test_b_voxflows_own_latest_release_reads_with_the_shipped_code():
    """The very request VoxFlow makes (nothing patched)."""
    try:
        rel = U.fetch_latest(U.Http())
    except U.UpdateError as exc:                           # e.g. no published release yet: a quiet failure
        note("live voxflow", f"UpdateError: {exc}")
        assert str(exc)
        return
    assert U.parse_version(rel.version) is not None
    note("live voxflow", f"{rel.tag}: installable={rel.installable}, installer={rel.installer!r}, size={rel.size}, "
                         f"sums={'yes' if rel.sums_url else 'no'}, notes {len(rel.notes)} chars")


def test_b_real_api_json_redirect_tls_and_user_agent(monkeypatch, tmp_path, redirects):
    monkeypatch.setattr(U, "REPO", OTHER_REPO)
    monkeypatch.setattr(U, "API_LATEST", f"https://api.github.com/repos/{OTHER_REPO}/releases/latest")
    http = U.Http()
    sent = {}
    real_open = urllib.request.OpenerDirector.open

    def capture(self, req, *a, **k):
        sent.setdefault("headers", dict(req.header_items()))
        return real_open(self, req, *a, **k)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", capture)
    with http.open(U.API_LATEST, "api") as resp:
        assert resp.status == 200
        data = json.loads(U._read_limited(resp, U.API_MAX_BYTES).decode("utf-8"))
    assert sent["headers"]["User-agent"] == f"VoxFlow/{U.__version__}", sent["headers"]
    assert set(sent["headers"]) <= {"User-agent", "Accept"}, "nothing but the generic User-Agent and Accept"
    rel = U.parse_release(data)                            # the shape VoxFlow reads: tag, name, body, assets
    assert rel.version == str(U.parse_version(data["tag_name"]))
    assert rel.page.startswith(f"https://github.com/{OTHER_REPO}/releases/tag/")
    assert "<img" not in rel.notes and "![" not in rel.notes
    # a small asset and its checksum list, through GitHub's real redirect
    assets = {a["name"]: a for a in data["assets"]}
    sums_name = next(n for n in assets if n.endswith("checksums.txt"))
    small = min((a for n, a in assets.items() if n.endswith((".zip", ".tar.gz")) and "windows" in n),
                key=lambda a: a["size"])
    rel.installer, rel.installer_url, rel.size = small["name"], small["browser_download_url"], small["size"]
    rel.sums_url = assets[sums_name]["browser_download_url"]
    assert U.allowed_url(rel.installer_url) and U.allowed_url(rel.sums_url)
    path, digest = U.download_installer(http, rel, tmp_path / "dl")
    assert path.stat().st_size == small["size"] and U.sha256_file(path) == digest
    hosts = sorted({urllib.parse.urlsplit(u).hostname for u in redirects})
    assert hosts and all(h in U.ASSET_HOSTS for h in hosts), hosts
    assert all(re.match(r"https://", u) for u in redirects)
    note("live github", f"{OTHER_REPO} {data['tag_name']}: {small['name']} ({small['size']} bytes) verified against "
                        f"{sums_name}; redirect host(s): {hosts}; request headers: {sorted(sent['headers'])}")
