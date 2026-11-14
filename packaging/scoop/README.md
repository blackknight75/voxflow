# Scoop manifest for VoxFlow

This repository is its own [Scoop](https://scoop.sh) bucket: Scoop reads app manifests from the `bucket/` folder
of a bucket's git repository, so `bucket/voxflow.json` is all it needs. No other bucket is involved. Users run:

```powershell
scoop bucket add voxflow https://github.com/blackknight75/voxflow
scoop install voxflow/voxflow
```

and later `scoop update voxflow` / `scoop uninstall voxflow`. `scoop bucket add` makes a git clone of the whole
repository (a few hundred MB with its history) and `scoop update` pulls it.

## How the install works

* Scoop downloads `VoxFlow-Setup-<version>.exe` from the GitHub release and checks its SHA-256 against the
  manifest's `hash`.
* `"innosetup": true`: Scoop does **not** run the installer. It unpacks the installer's `{app}` files with
  [innounp](https://github.com/jrathlev/InnoUnpacker-Windows-GUI) (installed automatically from Scoop's main bucket)
  into `~\scoop\apps\voxflow\<version>\`. VoxFlow is a self-contained one-folder app, so nothing else is needed:
  the installer itself only adds shortcuts and an uninstall entry, and Scoop makes its own Start-menu shortcut
  ("VoxFlow" in *Scoop Apps*) and a `voxflow` command (`VoxFlow-console.exe`). No Windows uninstall entry, no
  desktop shortcut, no administrator rights.
* Nothing is persisted by Scoop: VoxFlow keeps its settings, history and logs in `%APPDATA%\VoxFlow` and the
  models in the Hugging Face cache, outside the app folder. `scoop uninstall` leaves both, as the installer's
  uninstaller does.
* "Launch at login" (Settings > General) is a value `VoxFlow` under `HKCU\...\CurrentVersion\Run` that the app writes
  itself. The manifest's `uninstaller.script` does what the `[Code]` section of `packaging/voxflow.iss` does for the
  installer: on `scoop uninstall` it removes the value, but only if it starts a program inside
  `~\scoop\apps\voxflow\` (a copy run from source, or the installer's copy, keeps its own entry). On `scoop update`
  Scoop runs the same script for the old version; there it keeps the setting and, if the value names the old
  version's folder, points it at the `current` junction instead. (It tells the two apart by whether
  `scoop-update.ps1` is on the PowerShell call stack. If a future Scoop changed that, an update would switch
  "Launch at login" off, not leave a broken entry.)
* Windows 11 is required. The installer enforces that (`MinVersion`), an extracted copy cannot: the manifest's
  `notes` say so.

## The hash, and releases

Builds are not reproducible, so the installer's SHA-256 is only known once the Release workflow has built it.
Until the first release is published, `bucket/voxflow.json` carries version `0.2.0` with the placeholder hash
`0000...0000` (64 zeros). That is never a real install: before publishing, the download URL does not exist; after
publishing, Scoop downloads the file, compares the hash and refuses it (`Hash check failed!`).

**After a release is published,** `.github/workflows/scoop.yml` reads the hash from the release's
`SHA256SUMS.txt`, rewrites the manifest and opens a pull request `Scoop: VoxFlow <version>` against the default
branch. Merge it; Scoop users get the new version from then on. Two things to know:

* the repository setting *Settings > Actions > General > Allow GitHub Actions to create and approve pull requests*
  must be on, or the workflow cannot open the pull request;
* a pull request opened by the workflow's `GITHUB_TOKEN` does not start CI by itself. Close and reopen it (or push
  to its branch) to run the checks, or merge it on the strength of the diff (two lines: URL/version and hash).

The workflow can also be started by hand (Actions > scoop > Run workflow, with the tag).

**By hand** (the same thing, one command once the hash is known):

```bat
gh release download v0.2.0 --repo blackknight75/voxflow --pattern SHA256SUMS.txt
type SHA256SUMS.txt
python packaging\scoop\update_manifest.py 0.2.0 <sha256 from the file>
```

then commit `bucket/voxflow.json` to the default branch. The script sets `version`, the download URL (from the
`autoupdate` URL) and `hash`, and refuses a malformed version, a malformed hash or the placeholder.

Do not raise the manifest's version ahead of a release: the bucket is live for everyone who added it. Raising
`voxflow.__version__` does not require touching the manifest (`tests/test_scoop_manifest.py` accepts an older
version in the manifest, as long as its hash is real).

`checkver` / `autoupdate` are there for Scoop's own tooling (`checkver.ps1 -u`, which reads the hash from
`SHA256SUMS.txt` with the regex `$sha256\s+\*?$basename`); this repository uses the workflow above instead.

## Check locally

`tests/test_scoop_manifest.py` checks the manifest against the release workflow, the winget template, the licence
and `voxflow.__version__`. To check it against Scoop's JSON schema, in a throwaway virtual environment:

```bat
curl -LO https://raw.githubusercontent.com/ScoopInstaller/Scoop/master/schema.json
python -m pip install jsonschema
python -c "import json, jsonschema; jsonschema.Draft7Validator(json.load(open('schema.json'))).validate(json.load(open('bucket/voxflow.json')))"
```

On a test machine with Scoop, once the release is published and the hash is in:

```powershell
scoop install .\bucket\voxflow.json   # or: scoop bucket add voxflow <repo>; scoop install voxflow/voxflow
voxflow --help
scoop uninstall voxflow
```
