# winget manifest for VoxFlow

Package identifier: `blackknight75.VoxFlow`. Once the package is in the
[winget community repository](https://github.com/microsoft/winget-pkgs), users install and update with:

```bat
winget install blackknight75.VoxFlow
winget upgrade blackknight75.VoxFlow
```

winget downloads `VoxFlow-Setup-<version>.exe` from the GitHub release, checks its SHA-256 and runs it silently
(`/SP- /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`, the defaults for an Inno Setup installer). The install is per user,
with no administrator prompt.

## What is here

| File | Purpose |
| --- | --- |
| `blackknight75.VoxFlow.yaml` | version manifest (template) |
| `blackknight75.VoxFlow.installer.yaml` | installer manifest (template): Inno Setup, user scope, x64, download URL, hash |
| `blackknight75.VoxFlow.locale.en-US.yaml` | default locale manifest (template): name, publisher, description, licence |
| `update_manifest.py` | fills the templates for one release |

The three YAML files are templates. They hold the tokens `__VERSION__`, `__SHA256__` and `__LICENSE__`, so they
are not submitted as they are. `ProductCode` is `{6C0B0B5E-5A9E-4C57-9F1D-4F4F0E2A7B31}_is1`: the `AppId` in
`packaging/voxflow.iss` plus `_is1`, which is the uninstall registry key Inno Setup writes. If the `AppId`,
`AppPublisher` or `UninstallDisplayName` in `voxflow.iss` ever change, change the installer template with them
(`tests/test_winget_manifest.py` checks that they agree).

**Licence:** `License: __LICENSE__` is a placeholder until the project's licence is chosen. Then either replace the
token in `blackknight75.VoxFlow.locale.en-US.yaml` (and add a `LicenseUrl:` line pointing at the licence file), or
pass `--license` each time. winget-pkgs does not accept a manifest without a real licence.

## Order of events

winget validates the download URL, so the release has to be public first:

1. The repository is public and the GitHub release `v<version>` is **published** (not a draft), with
   `VoxFlow-Setup-<version>.exe` and `SHA256SUMS.txt` attached (`.github/workflows/release.yml` builds the draft).
2. Fill the manifest with the hash from that release.
3. Validate, test, submit the pull request to `microsoft/winget-pkgs`.
4. A bot validates the pull request (it installs the package in a clean VM) and a moderator merges it. The first
   submission of a new package usually takes a few days; later versions are faster.

## Fill and validate

```bat
gh release download v0.2.0 --repo blackknight75/voxflow --pattern SHA256SUMS.txt
type SHA256SUMS.txt
python packaging\winget\update_manifest.py 0.2.0 <sha256 from the file> --license <licence> --strict
```

The manifests are written to `dist\winget\manifests\b\blackknight75\VoxFlow\0.2.0\` (the folder layout of
winget-pkgs; `--out` changes it). Without `SHA256SUMS.txt`, the hash is
`(Get-FileHash dist\VoxFlow-Setup-0.2.0.exe -Algorithm SHA256).Hash` in PowerShell.

```bat
winget validate --manifest dist\winget\manifests\b\blackknight75\VoxFlow\0.2.0
```

Optional, on a machine or VM where VoxFlow may be installed: test the real install from the local manifest.

```bat
winget settings --enable LocalManifestFiles
winget install --manifest dist\winget\manifests\b\blackknight75\VoxFlow\0.2.0
```

## Submit

**With wingetcreate** (`winget install Microsoft.WingetCreate`). It forks winget-pkgs under your account, commits
the files and opens the pull request; it asks for a GitHub sign-in the first time.

```bat
wingetcreate submit dist\winget\manifests\b\blackknight75\VoxFlow\0.2.0
```

**By hand:** fork `microsoft/winget-pkgs`, copy the folder `dist\winget\manifests\b\blackknight75\VoxFlow\0.2.0`
to the same path under `manifests\` in the fork, commit on a new branch and open a pull request titled
`New package: blackknight75.VoxFlow version 0.2.0` (later: `New version: ...`). Tick the checklist in the pull
request template.

**Automatically:** `.github/workflows/winget.yml` runs the same steps when a release is published. It does nothing
until the repository variable `WINGET_SUBMIT` is `true`; it also needs the variable `WINGET_LICENSE` and the secret
`WINGET_PAT` (a classic personal access token with the `public_repo` scope). Make the first submission by hand, so
the questions a moderator may ask about a new package reach you directly, and switch the workflow on for later
versions.

## Notes

* `MinimumOSVersion` is `10.0.22000.0` (Windows 11, the supported target). The installer checks the same floor
  (`MinVersion=10.0.22000` in `packaging/voxflow.iss`); a test keeps the two equal.
* The installer is about 1.8 GB. winget has no size limit, but the validation run downloads and installs it, so
  it takes a while.
* The installer is not code-signed. winget does not require a signature.
* The other command-line channel is Scoop: this repository is its own bucket (`bucket/voxflow.json`, see
  [../scoop/README.md](../scoop/README.md)).
