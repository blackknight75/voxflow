# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability** (GitHub private security advisories). Do not open a public issue for a vulnerability.

Include what you can: the VoxFlow version, your Windows version, steps to reproduce and the impact you expect.
You will get a reply as soon as possible, and credit in the fix notes if you want it.

## Scope and design notes

VoxFlow runs fully on your PC. Audio, transcripts, the dictionary, snippets and settings stay on the machine
(under `%APPDATA%\VoxFlow`); there is no account, no telemetry and no cloud service (the usage events that
onnxruntime's Windows builds can emit are switched off). The only network access
is downloading the speech and language models from Hugging Face the first time they are needed (and, if you
opt in, the GPU runtime packages from PyPI), plus the update check: an anonymous daily HTTPS request to the
GitHub Releases API of this repository, which can be switched off (Settings > Updates > Off: then only an
explicit *Check now* contacts GitHub). An update installer is only accepted from this repository's releases over
HTTPS and only when its SHA-256 matches the release's `SHA256SUMS.txt`. The installer is not code-signed, so
that check proves integrity, not origin.

## What the protections cover, and what they do not

* **Updates.** The checksum protects against a damaged or swapped download. It does not protect against a
  release that was itself replaced: `SHA256SUMS.txt` comes from the same GitHub release as the installer, so
  whoever can publish or edit a release of this repository (a stolen maintainer account or token) can replace
  both. There is no signature with a key kept elsewhere. For that reason the default update mode only tells you
  about a new version and installs when you say so; *Install automatically* trusts the repository without a
  second look. Windows SmartScreen shows "Unknown publisher" for the same reason. Scoop and winget check the
  hash that is written in their manifest when the release is published.
* **The sync folder.** Every file in the folder you choose is trusted: whoever can write there (another PC of
  yours, anyone you share the folder with, the cloud account behind it) can add, change and delete dictionary
  entries and snippets on your PC. A snippet is text that VoxFlow types into other programs, so use a folder that
  only you can write to. VoxFlow limits what such a file can do: it cannot change your shortcuts or anything about
  your history (whether it is kept, how much, for how long), control characters and invisible direction marks
  are removed from the text, a change to your snippets that arrived through the folder is announced, and
  history and statistics are never synced.
* **Backups and packs** from other people are read as data: no paths, no code, sizes capped, every entry
  checked. A backup never sets which models are downloaded or how updates are handled, also not one that
  claims to come from this PC.
* **The language model** has no tools, files or network. Text you select for command mode may have been written
  by someone else (a web page, an e-mail) and can try to steer the model; the worst outcome is unwanted text in
  place of your selection. Control characters and direction overrides are removed from the result.
* **Where the text goes.** A dictation is pasted into the window that has the keyboard focus when the text is
  ready, as when you type. If another window takes the focus in between, the text goes there; Esc cancels a
  dictation that has not been pasted yet. Command mode is stricter: if the window changed, the result is copied
  instead of pasted.
* **Other programs on your account** can do what you can: read VoxFlow's files, ask it to quit or show its
  window, send keys. VoxFlow does not defend against software already running as you.
* **Build pipeline.** Release builds install every Python package by the SHA-256 of its file, the extra
  llama.cpp builds and the CUDA runtime are pinned by SHA-256, and the GitHub Actions used are pinned by commit.
  Inno Setup comes from the runner image (or Chocolatey when the image has none) and is not pinned.

Things that are especially relevant to report:

* anything that sends audio, text or settings off the machine;
* text being typed or pasted into a window other than the one the user is dictating into;
* problems in the keyboard hook, the clipboard handling, or the installer;
* unsafe handling of imported files (backups, shared packs, the sync folder).

Only the latest release is supported with security fixes.
