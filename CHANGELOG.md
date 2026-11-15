# Changelog

All notable changes to VoxFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The setup guide downloads and loads the speech model before the hotkey practice, which now opens only once the
  model is ready. A model that is already downloaded skips the wait; a failed download can be retried from the
  guide, or the practice skipped.
- Turning off *Keep history* offers to delete the transcripts already saved, and the History page says when
  history is off while older transcripts are still listed.
- While *Keep history* is off, *Learn new words* is paused: nothing from a dictation is added to the dictionary
  suggestions. Editing the dictionary by hand still works.

### Fixed

- Errors while cleaning up or pasting a dictation could quote the dictated words in `voxflow.log`. They are now
  logged by error type and code location only. Crash reports and uncaught errors are written the same way.
- Deleting history (one entry, retention, the size limit, *Clear all*, an import that replaces it) now also
  deletes its backup copies (`history.v1.json`, `history.corrupt.json`) once the updated file is written, so
  deleted text never survives in a backup.
- In the setup guide, a retried download that is slow again now shows the "taking a while" note.
- The setup guide said the models are about 1 GB; they are about 5 GB.
- Snippets fire when you ask for them, the way people say it: dictate just the phrase ("My address.", "Um, sign
  off", "E-mail") to get exactly the snippet, or say "insert", "paste" or "type" right before it inside a sentence
  ("Insert my address, please" types the address, without "Insert"). The same works with the lead words of
  Spanish, French, German ("füge meine Adresse ein", "meine Adresse einfügen"), Italian, Portuguese and Dutch.
  Case, punctuation, hyphens and "zoom link two" / "zoom link 2" do not matter, and a trigger the clean-up
  reworded still fires. A phrase that merely occurs in a sentence ("I will sign off on the budget", "send me an
  email") is left alone.
- Dictionary words work without "sounds like" aliases: misspellings the recogniser makes that are not English
  words ("Kubernetis", "Graphana", "Postgress" spoken as "post gress") are corrected after recognition. Real
  words are never turned into a term ("copper" stays, not "Cooper"; "Red is" stays, not "Redis"), and a term that
  is itself an ordinary word ("Linear", "Hunter") no longer capitalises that word in prose; add a "sounds like"
  alias for those. Web addresses, paths and identifiers are left alone, and a snippet's text is never changed by
  the dictionary. The check uses a bundled English word list (SCOWL, about 200 KB).

## [0.2.0] - 2026-11-14

First public release. VoxFlow is a push-to-talk dictation app for Windows 11 that runs entirely on your PC:
hold **Ctrl + Win**, speak, release, and clean text is pasted into the app you are using.

### Highlights

- **100% local.** Speech recognition (NVIDIA Parakeet TDT 0.6b v3) and text cleanup (Qwen3-4B through llama.cpp)
  both run on your machine. No cloud, no account, no telemetry. The only network use is the one-time model
  download on first run.
- **Works offline** after that first run.
- **Fast.** On an RTX 5080, 32 ms from releasing the hotkey to the paste for a clean sentence and 192 ms when the
  language model is needed (11.6 s clip, medians of 5 runs). On CPU: 245 ms and about 1.9 s.
- **25 European languages** with auto-detection; full cleanup in English, Spanish, French, German, Italian,
  Portuguese and Dutch.
- **A one-file installer.** Per-user, no administrator rights, no Python, GPU runtime included.

### Install

1. Download `VoxFlow-Setup-0.2.0.exe` from the assets of this release and run it.
2. Windows SmartScreen will say "Windows protected your PC / Unknown publisher" because the installer is not
   code-signed. Choose *More info*, then *Run anyway*.
3. Start VoxFlow. The setup guide checks your microphone, lets you practise the hotkey and downloads the two
   models (about 5 GB in total, once).

`winget install blackknight75.VoxFlow` will work once the package has been accepted into the winget community
repository. With [Scoop](https://scoop.sh), from this repository (it is its own bucket):
`scoop bucket add voxflow https://github.com/blackknight75/voxflow`, then `scoop install voxflow/voxflow`
(`scoop update voxflow` / `scoop uninstall voxflow` later).

To update later, run the newer installer over the old one. To uninstall, use Windows Settings > Apps; your
settings and the downloaded models are left in place (see Known limitations).

### System requirements

- Windows 11, 64-bit.
- An NVIDIA GPU is recommended (developed on an RTX 5080 with 16 GB): both models run on it. On an AMD or Intel
  graphics card (a separate card with 4 GB or more) speech recognition runs on the GPU through WebGPU and the
  language model through Vulkan. Without a suitable card everything runs on the CPU: plain dictation stays quick,
  sentences that need the language model take about 2 seconds.
- Disk space for the installer (about 1.8 GB), the installed program and the models (about 5 GB).
- Internet once, for the model download on first run.
- A microphone.

### Added

Dictation

- Push-to-talk: hold Ctrl + Win to talk, release to paste into the focused window. The Start menu does not open
  and no keys leak into the app.
- Hands-free lock by double-tapping the hotkey; Esc cancels; pressing another key while holding the hotkey
  cancels as well.
- Paste last transcript (Alt + Shift + Z) and an optional undo-last-paste shortcut. All shortcuts are
  configurable, and unsafe combinations are refused.
- Your previous clipboard content is restored after each paste. If a paste cannot be sent (for example into a
  window running as administrator), the text stays on the clipboard and in History.
- Whisper mode boosts quiet speech. Recordings are capped at 2 minutes by default (configurable); taps shorter
  than 0.3 s are ignored.
- Microphone selection by name, a microphone test, and recovery when a microphone is unplugged mid-sentence.

Cleanup and formatting

- Smart cleanup: fillers removed, punctuation and capitals fixed, self-corrections resolved ("at five, no wait,
  six"), spoken commands such as "new line", "new paragraph" and "bullet point". The language model is guarded
  so it never answers or obeys what you dictate.
- Smart formatting: spoken numbers, money, percentages, dates, times, e-mail addresses, links, file names and
  numbered lists are written the way you would type them.
- A "Basic clean-up" signal on the Flow bar and in History when the language model did not answer in time and
  the text was cleaned by rules only.
- Command mode: hold Alt + Win and say what to do with the selected text ("make this more formal", "translate to
  Spanish", "turn into bullets"). With nothing selected it writes new text.
- Tone per app: formal, casual or very casual per category (personal, work, e-mail, other), with apps mapped to
  categories and browsers recognised by window title.

Languages

- Auto-detect or a fixed choice among the 25 European languages Parakeet v3 supports.
- Full cleanup (fillers, self-corrections, spoken punctuation, smart formatting, tone and a language-model pass
  in that language) for English, Spanish, French, German, Italian, Portuguese and Dutch.
- Basic cleanup (hesitation sounds removed, capitals, a final stop) for the other languages.

Your words and data

- Dictionary: names and jargon always spelled your way, "sounds like" aliases, import and export, and
  suggestions when the cleanup had to correct a name.
- Snippets: say a short phrase and a longer saved text is inserted.
- History: searchable, cleaned and raw text, app and time, copy, paste again, delete, export to txt, csv or
  json, and a retention setting. History can be switched off.
- Statistics: words dictated, speed, time saved, streak, activity chart and top apps.
- Backup: one `.voxbackup` file for the whole profile, with a choice of parts and merge or replace on import.
- Shareable packs (`.voxpack`) of dictionary terms or snippets.
- Sync folder: point VoxFlow at a folder your own sync tool shares between computers to keep dictionary,
  snippets, styles and settings in step. Files only; history and statistics never leave the computer.

Interface and system

- Flow bar: a small always-on-top, click-through pill with a live waveform that shows listening, hands-free,
  processing, done, errors, command mode and model loading. It appears on the screen you are working on while a
  hotkey is in use, and its position can be moved and is remembered.
- Main window with Home, History, Dictionary, Snippets, Style and Settings; light, dark or system theme; a
  reduced-motion option; a first-run setup guide with model download progress.
- Tray icon with quick toggles, pause, microphone choice and paste last transcript.
- Sound cues for start, stop and error (generated locally), and optional muting of other apps while you dictate,
  restored afterwards and after a crash.
- Launch at login, start in the tray, single instance, a rotating log and crash logs, all kept on your PC.
- Speed check: once per PC, in the background while you are not dictating, each device that works here (NVIDIA
  CUDA, any graphics card through WebGPU or Vulkan, integrated ones included, and the CPU) runs both models
  briefly in a separate low-priority process. "Automatic" then uses the fastest one when it is at least 15% and
  10 ms faster than the fixed order the app starts with, so it never makes dictation slower. Settings >
  On-device models shows the measured times, lets you pick a device per model and has **Measure again**. It is
  measured again after a driver, hardware or VoxFlow update. A speech recognition change applies at once, a
  language model change at the next start (new setting `cleanup_device`). `VOXFLOW_DEVICE` still overrides it.
- Per-user Windows installer (PyInstaller and Inno Setup) that includes the GPU runtime, plus a console build
  (`VoxFlow-console.exe`) for debugging.

### Privacy

- Audio, transcripts, dictionary and settings stay on your PC, under `%APPDATA%\VoxFlow`.
- The models are stored in the standard Hugging Face cache in your user folder.
- The only network access is the model download from Hugging Face. Hugging Face usage telemetry is switched off.
  There is no update check and no crash upload.

### Known limitations

- **Windows 11 only.** macOS and Linux are not supported.
- **The installer is not code-signed**, so SmartScreen warns about an unknown publisher.
- **Large downloads:** the installer is about 1.8 GB and the models are about 5 GB, downloaded on first run.
- **CPU-only machines:** sentences that need the language model take about 2 seconds, above the 1.5 s design
  target. Switching Smart cleanup off removes the wait, but then the raw transcript is pasted.
- **AMD and Intel graphics cards are untested.** The paths for them (WebGPU for speech recognition, Vulkan for the
  language model) were verified on an NVIDIA card and, forced, on an AMD integrated GPU only. If one is too slow
  or fails on your card VoxFlow falls back to the CPU by itself; starting it with the environment variable
  `VOXFLOW_DEVICE=cpu` skips the GPU altogether. Integrated graphics are not picked automatically (the one
  measured was slower than the CPU); `VOXFLOW_DEVICE=webgpu`, or `asr_device` set to `webgpu`, tries one anyway.
- **Elevated windows:** Windows blocks keyboard hooks and synthetic input for windows running as administrator
  and for UAC prompts, so the hotkey and the paste do not work there unless VoxFlow also runs as administrator.
- **Language coverage of the cleanup:** only seven languages get the full cleanup; the other 18 get the basic one.
- **Uninstalling leaves your data:** settings, history and dictionary in `%APPDATA%\VoxFlow` and the downloaded
  models in the Hugging Face cache are not removed. Delete those folders by hand for a complete removal.
- **No automatic updates:** install a newer version by running its installer.
- Features that would need a cloud service, an account or a phone app are out of scope by design.

[Unreleased]: https://github.com/blackknight75/voxflow/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/blackknight75/voxflow/releases/tag/v0.2.0
