"""The release workflow's CHANGELOG.md section extractor (.github/scripts/release_notes.py)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CHANGELOG = """# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [1.2.3] - 2026-11-14

### Added
- Installer.

```
## not a heading, this is code
```

### Fixed
- A bug.

## [1.2.30] - 2026-10-01
- Wrong one.

## v0.9.0 (2026-01-01)
Old notes.

## 0.8.0

[1.2.3]: https://example.invalid/compare/v0.9.0...v1.2.3
[0.9.0]: https://example.invalid/releases/tag/v0.9.0
"""


def _load():
    spec = importlib.util.spec_from_file_location("release_notes", ROOT / ".github" / "scripts" / "release_notes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extracts_exactly_the_section_of_the_version():
    notes = _load().extract(CHANGELOG, "1.2.3")
    assert notes.startswith("### Added\n- Installer.") and notes.endswith("### Fixed\n- A bug.")
    assert "## not a heading, this is code" in notes and "Wrong one" not in notes and "Unreleased" not in notes


@pytest.mark.parametrize("version, expected", [("v1.2.3", "### Added"), ("1.2.30", "- Wrong one."),
                                               ("0.9.0", "Old notes."), ("V0.9.0", "Old notes.")])
def test_heading_variants_and_a_leading_v(version, expected):
    assert _load().extract(CHANGELOG, version).startswith(expected)


@pytest.mark.parametrize("version", ["0.8.0", "2.0.0", "1.2", "Unreleased", "", "v"])
def test_missing_or_empty_sections_give_none(version):
    # 0.8.0 exists but holds nothing except the link definitions at the foot of the file
    assert _load().extract(CHANGELOG, version) is None


def test_cli_exit_codes_and_output_file(tmp_path, capsys):
    mod = _load()
    log = tmp_path / "CHANGELOG.md"
    log.write_text(CHANGELOG, encoding="utf-8")
    out = tmp_path / "notes.md"
    assert mod.main(["1.2.3", "--changelog", str(log), "--out", str(out)]) == 0
    assert out.read_text("utf-8").startswith("### Added")
    assert mod.main(["0.9.0", "--changelog", str(log)]) == 0
    assert capsys.readouterr().out == "Old notes.\n"
    assert mod.main(["2.0.0", "--changelog", str(log)]) == 1
    assert mod.main(["1.2.3", "--changelog", str(tmp_path / "absent.md")]) == 1
