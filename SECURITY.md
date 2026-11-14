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
opt in, the GPU runtime packages from PyPI).

Things that are especially relevant to report:

* anything that sends audio, text or settings off the machine;
* text being typed or pasted into a window other than the one the user is dictating into;
* problems in the keyboard hook, the clipboard handling, or the installer;
* unsafe handling of imported files (backups, shared packs, the sync folder).

Only the latest release is supported with security fixes.
