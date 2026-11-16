# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability** (GitHub private security advisories). Do not open a public issue for a vulnerability.

Include what you can: the VoxFlow version, your Windows version, steps to reproduce and the impact you expect.
You will get a reply as soon as possible, and credit in the fix notes if you want it.

## Scope and design notes

VoxFlow runs fully on your PC. Audio, transcripts, the dictionary, snippets and settings stay on the machine
(under `%APPDATA%\VoxFlow`); there is no account, no telemetry and no cloud service. The only network access
is downloading the speech and language models from Hugging Face the first time they are needed (and, if you
opt in, the GPU runtime packages from PyPI), plus the update check: an anonymous daily HTTPS request to the
GitHub Releases API of this repository, which can be switched off (Settings > Updates > Off). An update installer
is only accepted from this repository's releases over HTTPS and only when its SHA-256 matches the release's
`SHA256SUMS.txt`. The installer is not code-signed, so that check proves integrity, not origin.

Things that are especially relevant to report:

* anything that sends audio, text or settings off the machine;
* text being typed or pasted into a window other than the one the user is dictating into;
* problems in the keyboard hook, the clipboard handling, or the installer;
* unsafe handling of imported files (backups, shared packs, the sync folder).

Only the latest release is supported with security fixes.
