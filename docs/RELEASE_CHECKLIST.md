# Release checklist

What has to be done by hand before a release, and then the release steps. The automated part is in
`.github/workflows/`; this page is the rest. Commands are run from the repository root in the project venv
(setup: [DEVELOPMENT.md](DEVELOPMENT.md)).

## What the workflows already check

| Workflow | Checks | Does not check |
| --- | --- | --- |
| `ci.yml` (every push and pull request) | the test suite in fast mode, with the coverage gate | the real models |
| `models.yml` (twice a week, by hand, and before every release build) | the real speech and language models **on the CPU**: the real-model tests, `voxflow --selftest`, `scripts/e2e_selftest.py --device cpu` | the GPU: CUDA and WebGPU for speech recognition, the CUDA and Vulkan builds of llama.cpp (a runner has no graphics driver, so the app picks its CPU build, as on a PC without a GPU) |
| `release.yml` (tag `v*`) | builds the installer, then installs it on the runner, runs the installed program, uninstalls it and checks what is left (`.github/scripts/installer_smoke.ps1`) | the setup wizard (the install is silent), SmartScreen, an upgrade over an older version |

No hosted runner has a microphone, a GPU or a desktop someone is typing on. Those are the manual checks below.

## Manual checks (Windows 11 PC with an NVIDIA GPU, a microphone and speakers)

Do these on the commit that will be tagged. The first run downloads the models (about 5 GB) if they are not in
the Hugging Face cache yet. Each script prints its verdict on the last line.

- [ ] **Whole suite, real models included, on the GPU.** Takes several minutes; needs both models.

  ```bat
  .venv\Scripts\python.exe scripts\coverage.py
  ```

  Expect `pytest exit code: 0`, no `HERMETICITY VIOLATION` section, and no skipped real-model test
  (add `-rs` to list skips: `scripts\coverage.py -rs`).

- [ ] **Real models end to end, GPU then forced CPU**, with the latency targets. Touches nothing on the
  desktop. Close games and other GPU-heavy programs first: the GPU run fails when it is over the 1.5 s target.

  ```bat
  .venv\Scripts\python.exe scripts\e2e_selftest.py
  ```

  Expect `== device auto: PASS ==` with `backend cuda` for both models, `== device cpu: PASS ==`,
  `no network connection was attempted` for both, and `E2E SELFTEST PASSED`. If the numbers in
  [DEVELOPMENT.md](DEVELOPMENT.md#measured-speed) or the README changed noticeably, update them.

- [ ] **Real microphone.** Needs a connected microphone, and speakers at a normal volume for the loopback part
  (it plays `tests\tts_sample.wav` aloud and listens to it; `--no-loopback` skips that part, for headphones).

  ```bat
  .venv\Scripts\python.exe scripts\mic_check.py
  ```

  Expect `MIC CHECK PASSED` without the "digital silence" note, your microphone in `input devices`, and a
  `heard` line that resembles the `reference` line.

- [ ] **Built-in self-test** (real keyboard hook, real clipboard, both models). Your clipboard is put back.

  ```bat
  .venv\Scripts\python.exe -m voxflow --selftest tests\tts_sample.wav
  ```

  Expect `[ok]` on every line, `backend cuda` for both models, and `SELFTEST PASSED`.

- [ ] **Both models on the any-vendor GPU path (WebGPU, Vulkan), and on the CPU.** The same self-test with the back-end named:

  ```bat
  set VOXFLOW_DEVICE=vulkan && .venv\Scripts\python.exe -m voxflow --selftest tests\tts_sample.wav
  set VOXFLOW_DEVICE=cpu && .venv\Scripts\python.exe -m voxflow --selftest tests\tts_sample.wav
  ```

  Expect `speech model load: ... backend webgpu` and `cleanup model load: ... backend vulkan` in the first run,
  `backend cpu` for both in the second (never `backend rules`), and `SELFTEST PASSED` twice. On an NVIDIA card
  this exercises the WebGPU plugin and the Vulkan build, not AMD or Intel hardware.

- [ ] **A separate AMD graphics card (not tested so far: no such card was available).** On a PC with a
  discrete AMD card with 4 GB or more and no NVIDIA card, install VoxFlow and dictate a sentence with a
  self-correction ("move it to Thursday, no wait, Friday"). Settings > On-device models must show the speech
  model as "Running on the GPU (WebGPU)." and the cleanup model as "Running on the GPU (Vulkan)."; the paste must
  not carry the "basic clean-up" hint. In `%APPDATA%\VoxFlow\logs\voxflow.log` the line
  `speech recognition runs on webgpu` follows a line `WebGPU adapter: ...` that must name the AMD card, and
  `llama.cpp libraries: vulkan build` names the card's driver path; `speech recognition runs on cpu (... webgpu:
  failed ...)` or `too slow for dictation` mean the card was tried and the CPU was used instead. Compare the
  speech latency in History with a run under `VOXFLOW_DEVICE=cpu`: the GPU must be the faster one. Record the
  card and the result in the release notes; until this is done the README must keep saying that these cards are
  untested.

- [ ] **A separate Intel graphics card (Arc; not tested so far).** The same check as for AMD.

- [ ] **A laptop with integrated graphics and a separate card** (not tested so far): the line `WebGPU adapter:`
  must name the separate card, and the speech latency must match a card, not the 1 to 5 s of an integrated GPU.
  This is the case where the card is selected by `adapterIndex`.

- [ ] **The speed check's choice on a PC with integrated graphics and a separate card** (not tested so far).
  Start VoxFlow with an empty profile, wait about a minute without dictating, and read the two
  `speed check: speech: ...` / `speed check: language model: ...` lines in `voxflow.log`: every back-end has a
  time or a reason, and Settings > On-device models shows the same times in both device menus. The choice must be
  the separate card (or CUDA) unless the log shows the other device at least 15% and 10 ms faster; dictating
  while it runs must log `speed check postponed` and the dictation must not be slower than usual. Restart
  VoxFlow: no new `speed check started` line. Record the table in the release notes.

- [ ] **A PC with no separate graphics card** (integrated graphics only, or a VM without any graphics driver):
  Settings > On-device models shows both models as "Running on the CPU.", the cleanup model not as "Limited"; a
  dictated self-correction is resolved; the log says `speech recognition runs on cpu` with the reason
  (`no discrete GPU ...` or `the WebGPU plugin did not load ...`). No crash at start on the VM: the WebGPU
  plugin is loaded there for the first time without a Direct3D 12 driver.

- [ ] **Live paste into real apps.** Presses real keys and pastes for real into windows the script opens (a Qt
  box, a Win32 edit box, Notepad, an Electron editor, a browser field). Needs an unlocked desktop and about two
  minutes with your hands off the keyboard and mouse. It works in a temp profile and puts your clipboard back;
  close your own Notepad windows first, or the Notepad step is skipped.

  ```bat
  .venv\Scripts\python.exe scripts\live_e2e.py
  .venv\Scripts\python.exe scripts\live_e2e.py --alt-shortcuts
  set VOXFLOW_DESKTOP_TESTS=1 && .venv\Scripts\python.exe -m pytest tests -m desktop -rs
  ```

  Expect `LIVE E2E PASSED` (exit code 0) twice. `LIVE E2E BLOCKED` (exit code 3) means nothing was tested
  (locked workstation, or another window took the focus): run it again. The third command runs the desktop
  tests of the suite: none may fail, and a skipped one was not tested (the reason is printed).

- [ ] **Five minutes of real use.** Start the app (`run.bat`), dictate into two or three programs you use
  every day with your own microphone, try command mode, paste-last and undo, then quit from the tray.
  No test can stand in for these, so try each once:
  - end a sentence and let go at the same moment, several times: the last word must be whole (the recording
    goes on for 80 ms after the release);
  - on the CPU (`set VOXFLOW_DEVICE=cpu`), dictate a self-correction, let go and press Esc before the text
    appears: nothing is pasted, nothing is in History, and the Esc does not reach the app;
  - dictate, let go, and at once hold the hotkey for the next sentence: the first text is pasted when you
    release, before the second;
  - pause dictation from the tray and press the hotkey: the Flow bar says "Dictation is paused";
  - quit from the tray while a dictation is being worked on: window, tray icon and Flow bar disappear at once.

### The installer (after the release workflow has built it)

- [ ] On a spare PC or a VM, **not** your own machine: download `VoxFlow-Setup-<version>.exe` from the draft
  release and compare its SHA-256 with `SHA256SUMS.txt`
  (`(Get-FileHash VoxFlow-Setup-<version>.exe -Algorithm SHA256).Hash` in PowerShell).
- [ ] Run it by double-click as a user would: SmartScreen shows "Unknown publisher" (the installer is not
  signed), the wizard installs without an administrator prompt, "Launch VoxFlow" starts the app, the first-run
  guide downloads the models, a dictation works. If that PC has an NVIDIA GPU, Settings > On-device models
  must show both models on the GPU. Without one, the speech model must show "Running on the GPU (WebGPU)." or
  "Running on the CPU.", and the cleanup model "Running on the GPU (Vulkan)." or "Running on the CPU.", never
  "Limited". In `dist\VoxFlow\_internal\onnxruntime_ep_webgpu\` both `onnxruntime_providers_webgpu.dll` and
  `dxcompiler.dll` must be present, and `set VOXFLOW_DEVICE=webgpu` before `VoxFlow-console.exe --selftest` must
  report `backend webgpu` for the speech model on a PC with a graphics card: the plugin has not been run from a
  frozen build yet. About a minute after the models are ready, `voxflow.log` must show `speed check finished`
  with a time for every back-end: the frozen exe starts its own measuring children (`--speed-check`), which has
  not been run from a frozen build yet either.
- [ ] Switch on Settings > General > Launch at login, sign out and in again: VoxFlow starts in the tray.
- [ ] If an older VoxFlow is installed there: run the new installer over it. Settings, history and dictionary
  are kept, and no `previous` folder is left in the install folder. The upgrade needs about 3.8 GB free: Setup
  keeps the old program files in `previous` until the new ones are complete and puts them back if it does not
  finish. That step was run with a dummy program only (see DEVELOPMENT.md, "Building the installer"), not yet
  with the real 3.8 GB runtime: to try it, open `_internal\python312.dll` of the installed copy in a program
  that locks it (or fill the disk), run Setup with `/VERYSILENT /SUPPRESSMSGBOXES /NOCLOSEAPPLICATIONS`, and
  check that the old version still starts afterwards.
- [ ] In-app update, once the release is published (it needs a published release newer than the installed one):
  with the previous version installed by its installer, Settings > Updates > *Check now* shows the new version and
  its notes. *Install now* downloads it, VoxFlow quits, Setup runs without any window, and VoxFlow starts again
  with the new version in Settings > About. `%TEMP%\VoxFlow-update\install.log` is Inno Setup's log of that run.
  Also try *Install when I quit*: quitting from the tray installs it, and VoxFlow does not start again.
- [ ] Uninstall from Windows Settings > Apps: the program folder and the login entry are gone,
  `%APPDATA%\VoxFlow` is still there.
- [ ] Only if the release workflow's installer test was skipped (the run summary says so), run it by hand on
  that spare PC. The script installs and uninstalls VoxFlow for real and refuses to run where VoxFlow is
  already installed:

  ```bat
  powershell -ExecutionPolicy Bypass -File .github\scripts\installer_smoke.ps1 -Installer VoxFlow-Setup-<version>.exe -RunValue Foreign -HubCheck -StrictDesktop -IUnderstandThisInstalls
  powershell -ExecutionPolicy Bypass -File .github\scripts\installer_smoke.ps1 -Installer VoxFlow-Setup-<version>.exe -RunValue Owned -IUnderstandThisInstalls
  ```

  Expect `RESULT: PASS` from both.

## Release steps

1. `voxflow/__init__.py` has the new `__version__`, `CHANGELOG.md` has a section for it, and both are merged
   to the default branch. CI is green on that commit.
2. Do the manual checks above on that commit.
3. Tag it and push the tag. The tag must be `v` + `__version__`, or the release workflow stops.

   ```bat
   git tag v<version>
   git push origin v<version>
   ```

4. Wait for the **Release** workflow (the real-model tests, then the build and the installer test: allow
   two hours). On a failure in the installer test, the install and uninstall logs are in the run's
   `installer-test-logs` artifact. A gate that fails for a reason outside the code (Hugging Face unreachable,
   a runner image change) can be switched off for one run: see the comment at the top of
   `.github/workflows/release.yml`. The run summary then says which gate was skipped; do its checks by hand.
5. Open the **draft** release the workflow created. Check the notes (taken from `CHANGELOG.md`), the two
   assets (`VoxFlow-Setup-<version>.exe`, `SHA256SUMS.txt`) and the run summary, then do "The installer"
   checks above with the file from the draft.
6. Publish the draft. Nothing is public before this step. Releases are not rebuilt in place: to fix a
   published release, raise the version and tag again.
7. winget, after the release is published and the repository is public (winget validates the download URL):
   follow [packaging/winget/README.md](../packaging/winget/README.md). Make the first submission by hand
   (`update_manifest.py`, `winget validate`, `wingetcreate submit`); for later versions
   `.github/workflows/winget.yml` does it when the repository variable `WINGET_SUBMIT` is `true`.
8. Scoop (this repository is its own bucket, `bucket/voxflow.json`): publishing the release starts
   `.github/workflows/scoop.yml`, which takes the installer's SHA-256 from the release's `SHA256SUMS.txt` and opens
   the pull request "Scoop: VoxFlow `<version>`". It needs *Settings > Actions > General > Allow GitHub Actions to
   create and approve pull requests*. CI does not start on a pull request opened by the workflow: close and reopen
   it to run the checks, then merge it. Scoop users get the release only after the merge. If the workflow did not
   run, do it by hand (see [packaging/scoop/README.md](../packaging/scoop/README.md)):

   ```bat
   python packaging\scoop\update_manifest.py <version> <sha256 from SHA256SUMS.txt>
   ```

   and commit `bucket/voxflow.json` to the default branch. For the first release the manifest carries a
   placeholder hash (64 zeros) until then, which Scoop refuses.
