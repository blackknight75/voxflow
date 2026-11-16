# Contributing to VoxFlow

Thanks for your interest. Bug reports, fixes and improvements are welcome.

## Development setup

VoxFlow targets Windows 11 and Python 3.12 (64-bit).

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt --only-binary=:all:
.venv\Scripts\python.exe -m voxflow
```

`requirements-dev.txt` includes the runtime requirements plus pytest, pytest-qt, coverage and PyInstaller.
See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the optional GPU (CUDA) setup.

## Running the tests

```bat
.venv\Scripts\python.exe scripts\coverage.py             # whole suite under coverage, per-module table
.venv\Scripts\python.exe scripts\coverage.py --fast      # skip the tests that load the real models
.venv\Scripts\python.exe scripts\coverage.py -k store    # extra arguments go to pytest
```

The suite is hermetic: it uses a temp profile and offscreen Qt, and the harness (`tests/conftest.py`) fails any
test that reaches the real clipboard, keyboard, microphone, registry or `%APPDATA%\VoxFlow`. The scripts in
`scripts/` that use real hardware (`mic_check.py`, `e2e_selftest.py`, `live_e2e.py`, ...) are optional and are
described in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

The app-update end-to-end tests (`tests/e2e_update`: a local fake GitHub over real TLS, a stub installer exe
started for real, the smoke script rehearsed against a fake Inno Setup; about 4 minutes, needs `openssl` from Git
for Windows and PyInstaller) are skipped unless you set `VOXFLOW_UPDATE_E2E=1`. Add `VOXFLOW_UPDATE_LIVE=1` for
the two tests that make anonymous GETs to the real GitHub API. They never install VoxFlow.

Maintainers: the manual pre-release checks and the release steps are in [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## Building the installer

```bat
.venv\Scripts\python.exe scripts\build_installer.py
```

This runs PyInstaller and then Inno Setup 6 (`winget install --id JRSoftware.InnoSetup -e --scope user`).
Output goes to `dist\`; `build\` and `dist\` are git-ignored.

## Pull requests

* Keep changes focused and add or update tests for behaviour you change.
* Run the test suite before opening a pull request.
* VoxFlow is fully local by design: changes must not add network calls beyond the model download and the
  update check (`voxflow/updater.py`, the only module that may open a connection for updates).
* Do not commit model files, build output, or personal data (paths, recordings, transcripts).

## Reporting security issues

Please do not open a public issue for a vulnerability; see [SECURITY.md](SECURITY.md).
