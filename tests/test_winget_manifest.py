"""winget manifest templates (packaging/winget) and the script that fills them for a release."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WINGET = ROOT / "packaging" / "winget"
SHA = "25e12d0aea4921d93295bf33053dc2ae36f08d93e3eb034e4669ffd456aa128c"


def _load():
    spec = importlib.util.spec_from_file_location("update_manifest", WINGET / "update_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_templates_carry_only_the_known_placeholders():
    mod = _load()
    tokens = set()
    for name in mod.TEMPLATES:
        tokens |= set(mod.TOKEN.findall((WINGET / name).read_text("utf-8")))
    assert tokens == {"__VERSION__", "__SHA256__", "__LICENSE__"}


def test_render_fills_version_hash_and_url():
    mod = _load()
    files = mod.render("v1.2.3", SHA, "MIT")
    assert set(files) == set(mod.TEMPLATES) and mod.unfilled(files) == []
    installer = files["blackknight75.VoxFlow.installer.yaml"]
    assert ("InstallerUrl: https://github.com/blackknight75/voxflow/releases/download/v1.2.3/VoxFlow-Setup-1.2.3.exe"
            in installer)
    assert f"InstallerSha256: {SHA.upper()}" in installer
    assert "InstallerType: inno" in installer and "Scope: user" in installer and "Architecture: x64" in installer
    for text in files.values():
        assert "PackageIdentifier: blackknight75.VoxFlow\nPackageVersion: 1.2.3\n" in text
        assert "# Template:" not in text
    assert "License: MIT\n" in files["blackknight75.VoxFlow.locale.en-US.yaml"]


def test_product_code_is_the_uninstall_key_inno_setup_registers():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    app_id = re.search(r"^AppId=\{(\{[0-9A-F-]{36}\})", iss, re.M).group(1)
    installer = (WINGET / "blackknight75.VoxFlow.installer.yaml").read_text("utf-8")
    assert installer.count(f"ProductCode: '{app_id}_is1'") == 2          # top level + AppsAndFeaturesEntries
    name = re.search(r"^UninstallDisplayName=(.+)$", iss, re.M).group(1)
    publisher = re.search(r"^AppPublisher=(.+)$", iss, re.M).group(1)
    assert f"- DisplayName: {name}\n  Publisher: {publisher}\n" in installer


@pytest.mark.parametrize("version, sha", [("1.2.3", "abc"), ("1.2.3", "g" * 64), ("latest", SHA), ("", SHA)])
def test_render_rejects_a_bad_version_or_hash(version, sha):
    with pytest.raises(ValueError):
        _load().render(version, sha)


def test_cli_writes_the_three_files_and_strict_refuses_the_licence_placeholder(tmp_path, capsys):
    mod = _load()
    out = tmp_path / "m"
    assert mod.main(["0.9.0", SHA, "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(mod.TEMPLATES)
    assert "License: __LICENSE__" in (out / "blackknight75.VoxFlow.locale.en-US.yaml").read_text("utf-8")
    assert "__LICENSE__" in capsys.readouterr().err
    assert mod.main(["0.9.0", SHA, "--out", str(tmp_path / "s"), "--strict"]) == 3
    assert not (tmp_path / "s").exists()
    assert mod.main(["0.9.0", SHA, "--out", str(tmp_path / "s"), "--strict", "--license", "MIT"]) == 0
    assert mod.main(["0.9.0", "nope"]) == 2


def test_default_output_follows_the_winget_pkgs_layout():
    out = _load().default_out("1.2.3")
    assert out.parts[-6:] == ("winget", "manifests", "b", "blackknight75", "VoxFlow", "1.2.3")
