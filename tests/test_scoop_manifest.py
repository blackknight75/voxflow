"""Scoop manifest (bucket/voxflow.json; the repository is its own bucket) and the script that updates it."""
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import voxflow
from voxflow import system

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "bucket" / "voxflow.json"
SCRIPT = ROOT / "packaging" / "scoop" / "update_manifest.py"
SHA = "25e12d0aea4921d93295bf33053dc2ae36f08d93e3eb034e4669ffd456aa128c"
REPO = "https://github.com/blackknight75/voxflow"


def _load():
    spec = importlib.util.spec_from_file_location("scoop_update_manifest", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text("utf-8"))


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.match(r"\d+(?:\.\d+)*", v).group(0).split("."))


def _uninstaller() -> str:
    return "\n".join(_manifest()["uninstaller"]["script"])


def test_the_bucket_folder_holds_only_this_manifest():
    # Scoop lists every *.json under bucket/ (recursively) as an app of the bucket
    assert sorted(p.relative_to(MANIFEST.parent).as_posix() for p in MANIFEST.parent.rglob("*")) == ["voxflow.json"]


def test_manifest_is_written_the_way_the_update_script_writes_it():
    # canonical form, so the workflow's pull request changes only the lines it means to
    assert _load().dumps(_manifest()) == MANIFEST.read_text("utf-8")


def test_version_and_hash_are_consistent_with_the_package():
    m = _manifest()
    mod = _load()
    sha = m["architecture"]["64bit"]["hash"]
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert _vtuple(m["version"]) <= _vtuple(voxflow.__version__)
    if sha == mod.PLACEHOLDER_HASH:
        # before the release exists: only the version being released may carry the placeholder
        assert m["version"] == voxflow.__version__


def test_download_url_is_the_release_asset_and_matches_winget():
    m = _manifest()
    template = m["autoupdate"]["architecture"]["64bit"]["url"]
    assert template == f"{REPO}/releases/download/v$version/VoxFlow-Setup-$version.exe"
    assert m["architecture"]["64bit"]["url"] == template.replace("$version", m["version"])
    # release.yml uploads installer=VoxFlow-Setup-$version.exe under the tag v$version
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
    assert 'echo "installer=VoxFlow-Setup-$version.exe"' in release and 'tag="v$version"' in release
    winget = (ROOT / "packaging" / "winget" / "blackknight75.VoxFlow.installer.yaml").read_text("utf-8")
    assert f"InstallerUrl: {template.replace('$version', '__VERSION__')}\n" in winget
    assert set(m["architecture"]) == {"64bit"} and "url" not in m and "hash" not in m


def test_autoupdate_hash_regex_finds_the_hash_in_sha256sums_txt():
    hash_cfg = _manifest()["autoupdate"]["architecture"]["64bit"]["hash"]
    assert hash_cfg["url"] == f"{REPO}/releases/download/v$version/SHA256SUMS.txt"
    assert 'sha256sum "$INSTALLER" > SHA256SUMS.txt' in (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
    # what Scoop's find_hash_in_textfile does: $sha256 -> a capture group, $basename -> the escaped file name
    regex = hash_cfg["regex"].replace("$sha256", "([a-fA-F0-9]{64})").replace("$basename",
                                                                             re.escape("VoxFlow-Setup-1.2.3.exe"))
    for line in (f"{SHA}  VoxFlow-Setup-1.2.3.exe\n", f"{SHA} *VoxFlow-Setup-1.2.3.exe\n"):
        assert re.search(regex, line).group(1) == SHA
    assert re.search(regex, f"{SHA}  VoxFlow-Setup-1.2.4.exe\n") is None          # another file's line


def test_licence_metadata_and_checkver():
    m = _manifest()
    assert m["license"] == "MIT" and (ROOT / "LICENSE").read_text("utf-8").startswith("MIT License\n")
    assert m["homepage"] == REPO and m["checkver"] == {"github": REPO}
    winget_locale = (ROOT / "packaging" / "winget" / "blackknight75.VoxFlow.locale.en-US.yaml").read_text("utf-8")
    assert f"ShortDescription: {m['description']}\n" in winget_locale
    notes = " ".join(m["notes"])
    assert "Windows 11" in notes and "5 GB" in notes and "%APPDATA%\\VoxFlow" in notes


def test_extracted_not_run_and_the_entry_points_exist():
    m = _manifest()
    assert m["innosetup"] is True and "installer" not in m      # innounp unpacks {app}; Setup never runs
    assert "persist" not in m                                    # the app keeps nothing in its own folder
    spec = (ROOT / "packaging" / "voxflow.spec").read_text("utf-8")
    assert 'name="VoxFlow", console=False' in spec and 'name="VoxFlow-console", console=True' in spec
    assert m["shortcuts"] == [["VoxFlow.exe", "VoxFlow"]]
    assert m["bin"] == [["VoxFlow-console.exe", "voxflow"]]
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    assert 'Name: "{autoprograms}\\VoxFlow"; Filename: "{app}\\VoxFlow.exe"' in iss


def test_uninstaller_removes_only_this_apps_login_entry():
    text = _uninstaller()
    run = f"'HKCU:\\{system.RUN_KEY}'"
    assert f"$run = {run}" in text and text.count(run) == 1
    assert text.count(f"-Name '{system.RUN_VALUE}'") == 3                # read, remove, rewrite
    assert "$app_root = (Split-Path $dir) + '\\'" in text               # ...\scoop\apps\voxflow\ (all versions)
    guard = "if ($command -and $command.ToLower().Contains($app_root.ToLower())) {"
    assert guard in text
    # the registry is only ever changed inside the guard, and only removed when not updating
    before, inside = text.split(guard)
    assert "Remove-ItemProperty" not in before and "Set-ItemProperty" not in before
    assert inside.index("if (!$updating) {") < inside.index("Remove-ItemProperty") < inside.index("} elseif (")
    assert "$updating = [bool](Get-PSCallStack | Where-Object { $_.ScriptName -like '*\\scoop-update.ps1' })" in text
    # an update repoints an entry naming the old version's folder at the 'current' junction, if there is one
    assert '(Test-Path "$app_root\\current") -and $command.ToLower().Contains("$dir\\".ToLower())' in text
    assert "Set-ItemProperty -LiteralPath $run -Name 'VoxFlow' -Value $new" in inside
    # the same rule as the installer's uninstaller
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    assert "if Pos(Lowercase(AddBackslash(ExpandConstant('{app}'))), Lowercase(Command)) > 0 then" in iss


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="needs Windows PowerShell")
def test_uninstaller_script_parses_in_windows_powershell(tmp_path):
    path = tmp_path / "uninstaller.ps1"
    path.write_text(_uninstaller(), encoding="utf-8-sig")
    check = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile($env:SCOOP_HOOK, "
             "[ref]$null, [ref]$e); $e | ForEach-Object { $_.Message }; exit $e.Count")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", check],
                         capture_output=True, text=True, timeout=120, env={**os.environ, "SCOOP_HOOK": str(path)})
    assert out.returncode == 0, out.stdout + out.stderr


def test_render_sets_version_url_and_hash_and_keeps_the_rest():
    mod = _load()
    before = _manifest()
    after = mod.render(before, "v1.2.3", SHA.upper())
    assert after["version"] == "1.2.3"
    assert after["architecture"]["64bit"] == {
        "url": f"{REPO}/releases/download/v1.2.3/VoxFlow-Setup-1.2.3.exe", "hash": SHA}
    for key in set(before) - {"version", "architecture"}:
        assert after[key] == before[key]
    assert before == _manifest()                                      # the input is not modified


@pytest.mark.parametrize("version, sha", [("1.2.3", "abc"), ("1.2.3", "g" * 64), ("latest", SHA), ("", SHA),
                                          ("1.2.3", "0" * 64), ("1", SHA)])
def test_render_rejects_a_bad_version_or_hash(version, sha):
    with pytest.raises(ValueError):
        _load().render(_manifest(), version, sha)


def test_cli_rewrites_the_manifest_in_place_and_refuses_bad_input(tmp_path, capsys):
    mod = _load()
    target = tmp_path / "voxflow.json"
    shutil.copyfile(MANIFEST, target)
    original = target.read_bytes()
    assert mod.main(["0.9.0", "nope", "--manifest", str(target)]) == 2
    assert mod.main(["latest", SHA, "--manifest", str(target)]) == 2
    assert mod.main(["0.9.0", "0" * 64, "--manifest", str(target)]) == 2
    assert target.read_bytes() == original and "error:" in capsys.readouterr().err
    assert mod.main(["0.9.0", SHA, "--manifest", str(target)]) == 0
    data = target.read_bytes()
    assert b"\r\n" not in data
    written = json.loads(data)
    assert written["version"] == "0.9.0" and written["architecture"]["64bit"]["hash"] == SHA
    assert "0.9.0" in capsys.readouterr().out
    assert mod.MANIFEST == MANIFEST


def test_the_workflow_feeds_the_release_hash_to_the_script():
    wf = (ROOT / ".github" / "workflows" / "scoop.yml").read_text("utf-8")
    assert "types: [published]" in wf and "!github.event.release.prerelease" in wf
    assert "gh release download \"$TAG\" --pattern SHA256SUMS.txt" in wf
    assert 'python packaging/scoop/update_manifest.py "$version" "$sha"' in wf
    assert "ref: ${{ github.event.repository.default_branch }}" in wf
    assert "pull-requests: write" in wf and "gh pr create" in wf and "git push --force origin" in wf
