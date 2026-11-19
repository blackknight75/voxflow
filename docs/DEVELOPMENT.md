# Developing VoxFlow

Everything a developer needs that does not belong on the front page: installing from source, launch flags, the
self-test, building the installer, the test suite, how the speed numbers are measured, and the architecture.
For the contribution workflow see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Requirements

* Windows 11, **Python 3.12** (64-bit)
* An NVIDIA GPU is recommended (developed on an RTX 5080, 16 GB). Everything also runs on CPU: plain dictation
  stays fast, sentences that need the language model take about 2 s (see [Measured speed](#measured-speed)).
  On separate AMD and Intel cards the language model is meant to run through Vulkan and speech recognition
  through WebGPU (see [Where the language model runs](#where-the-language-model-runs) and
  [Where speech recognition runs](#where-speech-recognition-runs)); neither was tested on such a card.
* About 6 GB free disk for the packages and models, internet for the first install and first run

## Install from source

```bat
git clone https://github.com/blackknight75/voxflow.git
cd voxflow
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt --only-binary=:all:
.venv\Scripts\python.exe -m voxflow.cleanup --install-cuda
.venv\Scripts\python.exe scripts\fetch_llama_backends.py
```

* `--only-binary=:all:` is required. PyPI only has a source package of `llama-cpp-python`; the prebuilt CUDA
  wheel comes from the extra index declared at the top of `requirements.txt`. Without the flag pip tries to
  compile it.
* **Never install plain `onnxruntime`** next to `onnxruntime-gpu`; they clash. Always call
  `.venv\Scripts\python.exe` instead of a global Python.
* `--install-cuda` (needs internet) downloads the CUDA 12 runtime DLLs the cleanup wheel needs
  (`cudart64_12`, `cublas64_12`, `cublasLt64_12`) and copies just those next to `llama.dll`. Skip it and
  the CUDA build of llama.cpp cannot be loaded.
* `fetch_llama_backends.py` (needs internet once, about 200 MB) adds the CUDA 13, Vulkan and CPU builds of
  llama.cpp next to the CUDA 12 one. Skip it and the language model only loads on a PC with an NVIDIA driver;
  anywhere else VoxFlow falls back to the rule-based cleanup, and an RTX 50xx card compiles its GPU code on the
  first start (see below). See [Where the language model runs](#where-the-language-model-runs).
* **Windows 260-character path limit:** installing PySide6 into a deep folder can fail with
  `OSError: [Errno 2] ... PySide6\qml\...`. Use a short install path such as `C:\voxflow`, or enable Windows long
  paths (`reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f`
  from an admin prompt, then restart).
* Developers: `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` adds pytest, pytest-qt, coverage
  and PyInstaller.

## First run and self-test

The first launch opens a short setup guide (microphone check, hotkey practice, model status) and downloads the
Parakeet v3 ONNX model (about 2.5 GB) and the Qwen3-4B GGUF cleanup model (about 2.5 GB) from Hugging Face into
the normal Hugging Face cache. This needs internet once; afterwards VoxFlow works fully offline. Progress is
shown in the Flow bar, in the setup guide and under Settings > On-device models. You can check the install
without the UI:

```bat
.venv\Scripts\python.exe -m voxflow --selftest tests\tts_sample.wav
```

It checks the UI imports, the keyboard hook and the clipboard (your clipboard content is put back), loads both
models, runs the pipeline twice on the clip and prints timings plus the final text. Every step prints `[ok]` or
`[FAIL]`; the exit code is 0 only if all passed.

## Launch

```bat
run.bat                  # tray app, no console window
run.bat --console        # with a console for logs
.venv\Scripts\python.exe -m voxflow --demo       # the whole app with simulated audio, models and data; nothing is pasted
.venv\Scripts\python.exe -m voxflow --fake       # real hotkeys + paste, stand-in audio and models
.venv\Scripts\python.exe -m voxflow --settings   # open the main window on start
.venv\Scripts\python.exe -m voxflow --no-hook    # do not install the global keyboard hook
.venv\Scripts\python.exe -m voxflow --autostart  # what "Launch at login" runs: start in the tray
```

Only one instance runs per user; starting VoxFlow again brings the running instance's window to the front.

## Building the installer

```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt                  :: includes PyInstaller
winget install --id JRSoftware.InnoSetup -e --scope user                          :: Inno Setup 6, once
.venv\Scripts\python.exe scripts\build_installer.py --clean                      :: dist\VoxFlow-Setup-<version>.exe
```

PyInstaller (one-folder, `packaging\voxflow.spec`) freezes the app from the venv, then Inno Setup
(`packaging\voxflow.iss`) packs it. The version comes from `voxflow/__init__.py`. `build\` and `dist\` are not
committed.

What the installer does:

* `VoxFlow-Setup-<version>.exe` (about 1.8 GB) installs for the current user only: no administrator rights,
  default folder `%LOCALAPPDATA%\Programs\VoxFlow`, a Start menu entry, an optional desktop shortcut and a
  "Launch VoxFlow" box on the last page. Running a newer installer updates the old copy in place.
* The GPU runtime (CUDA 13 for speech recognition, CUDA 12 and, for RTX 50xx, CUDA 13 for the language model; the
  CUDA 13 llama.cpp build adds about 150 MB unpacked) is inside the installer, so no
  CUDA toolkit is needed. So are the Vulkan and CPU builds of llama.cpp: without an NVIDIA card the language model
  uses another GPU through Vulkan or the CPU. So is the WebGPU plugin for onnxruntime (about 30 MB): without an
  NVIDIA card speech recognition uses another separate GPU through it, or the CPU.
* The speech and language models (about 5 GB) are not in the installer; the app downloads them on first run.
* The installer is not code-signed, so Windows SmartScreen shows "Unknown publisher".
* `VoxFlow-console.exe` next to `VoxFlow.exe` is the same program with a console window, for debugging
  (`VoxFlow-console.exe --selftest [clip.wav]`; `set VOXFLOW_DEVICE=cpu` forces the CPU path; `cuda`, and `webgpu` or
  `vulkan`, force a GPU path: see the table under "Where speech recognition runs").
* The winget manifests live in `packaging\winget\`.

## Where the language model runs

llama.cpp is built once per kind of hardware, and a build can only be loaded where its driver is installed. VoxFlow
therefore carries four builds of the same llama-cpp-python release, each in its own folder next to the
`llama_cpp` package, and picks one before `llama_cpp` is imported (`LLAMA_BACKENDS` in `voxflow/cleanup.py`, in
this order):

| back-end | folder | chosen when |
| --- | --- | --- |
| `cuda` | `llama_cpp\lib_cuda13` (CUDA 13: native code for sm_86, sm_89, sm_120, sm_121) | every NVIDIA GPU is one the CUDA 12 build has no native code for and this one has (today: RTX 50xx), the driver supports CUDA 13.0 or newer, and the build loads |
| `cuda` | `llama_cpp\lib` (the installed wheel, plus `--install-cuda`: native code for sm_60 to sm_90) | the NVIDIA driver is installed and the build loads |
| `vulkan` | `llama_cpp\lib_vulkan` | Vulkan reports a separate (discrete) graphics card with at least 3500 MiB, any vendor, and the build loads |
| `cpu` | `llama_cpp\lib_cpu` | always loads: the last resort |

* The machine is asked in `voxflow/gpuprobe.py` (NVIDIA driver present, Vulkan devices with vendor, kind and
  memory). A build is then really loaded (`llama.dll` with its dependencies from its own folder); one that does not
  load is passed over for the next. The log line `llama.cpp libraries: ...` names the build and why the ones
  before it were skipped.
* After the model is loaded on a GPU, one sentence is timed. A GPU that fails to load the model, or needs more
  than half the cleanup timeout for that sentence, gives way to the CPU in the same run. The rule-based cleanup
  ("basic clean-up") remains only when no build can load the model at all.
* Integrated graphics are not in the fixed order: on an AMD Radeon integrated GPU the default model took 5.4 s
  for a sentence through Vulkan against 1.0 s on the CPU of the same PC. The speed check (below) measures them
  anyway, and `VOXFLOW_DEVICE=vulkan` or the `cleanup_device` setting still tries one.
* The `cleanup_device` setting (`auto`, `cuda`, `vulkan`, `cpu`) and `VOXFLOW_DEVICE=cpu`, `cuda` or `vulkan`
  (environment variable, wins over the setting) name the back-end instead of the automatic choice; a named GPU
  back-end that cannot be used ends on the CPU. Both are read at start only: a process can load one llama.cpp
  build, so a change applies when VoxFlow restarts.
* The Vulkan and CPU builds come from the wheels the llama-cpp-python project publishes for the same version,
  pinned by SHA-256 in `scripts/fetch_llama_backends.py`. When `requirements.txt` moves to another
  llama-cpp-python version, add that version's builds there; a folder built for another version is never loaded.
* Why a CUDA 13 build: a GPU needs machine code (SASS) for its own generation, or the driver compiles the build's
  PTX on first use of each kernel module and caches it (`%APPDATA%\VoxFlow\cuda-cache`). The CUDA 12 wheel has SASS
  for sm_60 to sm_90 only, so on an RTX 5080 a first start took 25-30 s and the first dictation of a new length
  stalled 4-8 s more (QA C-01). No llama-cpp-python wheel has sm_120 code (the cu130 and cu132 wheels stop at
  sm_90 too), but ggml-org's own CUDA 13.4 build of the llama.cpp release that llama-cpp-python vendors (b11312
  for 0.3.36) has. `lib_cuda13` is that build's `ggml-cuda.dll` next to the wheel's own `llama.dll` and ggml
  DLLs; it uses the CUDA 13 cuBLAS that onnxruntime-gpu already brings. Measured on the RTX 5080 against the CUDA
  12 build: empty cache 1.0 s load instead of 24.7 s and no stall, warm 2 % less time over 30 dictations,
  identical text for all of them, the same GPU memory.
* Where the loaded CUDA build has no native code for a GPU (the CUDA 12 build on a PC whose driver is too old for
  CUDA 13, say), the load also prefills one short user turn, so the compiling happens while the UI says
  "Preparing the GPU" instead of inside a dictation: 8 s on an empty cache, 30-40 ms once compiled. GPUs with
  native code skip it.
* Measured on the development PC (RTX 5080, one 19-word sentence that needs the language model, warm): CUDA 82 ms,
  Vulkan on the same card 82 ms, CPU build 0.9 s. **Vulkan was only ever run on that NVIDIA card (and, forced, on
  the integrated AMD GPU above); no AMD or Intel graphics card has been tested.** Speech recognition is separate:
  see the next section.

## Where speech recognition runs

The speech model runs on onnxruntime. `onnxruntime-gpu` carries the CUDA execution provider (NVIDIA only); every
other card is reached through the WebGPU execution provider, a plugin (`onnxruntime-ep-webgpu`: Dawn on Direct3D
12, with `dxcompiler.dll`) that `voxflow/asr.py` registers into the same onnxruntime at run time, once per
process and only when it gets that far. `Transcriber.load()` walks `ASR_BACKENDS` in this order:

| back-end | runs through | chosen when |
| --- | --- | --- |
| `cuda` | the CUDA execution provider (CUDA 13 wheels) | the NVIDIA driver is installed and onnxruntime offers the provider |
| `webgpu` | the WebGPU plugin | the plugin registers and lists a separate (discrete) graphics card with at least 3500 MiB, any vendor |
| `cpu` | the CPU execution provider | always: the last resort |

* Each GPU back-end is probed (a few milliseconds: 1.5 ms for CUDA, 2.5 ms for WebGPU on the development PC),
  then built, warmed up with one second of audio and checked: the encoder session must list the intended
  provider. Anything that fails gives way to the next back-end. The log line `speech recognition runs on ...`
  names the back-end and why the ones before it were skipped. A GPU back-end that fails during a dictation is
  replaced by the CPU for the rest of the session. `Transcriber.backend` names where the encoder runs.
* On both GPU back-ends only the encoder (nearly all of the work) runs on the GPU. The small decoder is called
  once per output token, and on a GPU every call pays a round trip: moving it to the CPU took the 11.6 s clip
  from 52 ms to 29 ms on CUDA and from about 190 ms to 65 ms on WebGPU (RTX 5080). onnx-asr has no option for
  this, so `_decoder_on_cpu` replaces its private `_decoder_joint` session; if a later onnx-asr no longer has it,
  the model is used as built (slower, same text).
* Two things the plugin needs, both found by measurement: it attaches only through
  `SessionOptions.add_provider_for_devices` (naming `WebGpuExecutionProvider` in `providers` gives a session that
  silently runs on the CPU), and the adapter is chosen by Dawn, not by the device passed in: the high-performance
  adapter by default, another one through the `adapterIndex` option. One process holds one WebGPU adapter.
* The fixed order is only the instant first choice. An integrated GPU is not in it because the one measured (AMD
  Radeon) took 1.7 s for the clip the CPU of the same PC does in 0.25 s; that is one measurement, not a rule.
  `asr.time_backends(model, names, runs, audio)` builds each named back-end in turn (an integrated GPU counts for
  `webgpu`), times warm runs on a built-in 5 s clip and returns the timings, so the choice can be made by
  measurement on each PC. It takes about 10 s for all three here and is never called at start-up: the speed check
  (below) calls it in a child process, one back-end per process.
* The `asr_device` setting (`auto`, `cuda`, `webgpu`, `cpu`) names a back-end; `webgpu` then takes any adapter,
  an integrated one included. A named GPU back-end that cannot be used ends on the CPU. Changing the setting
  reloads the models without a restart.
* `VOXFLOW_DEVICE` (environment variable, read at start) overrides the setting and names the back-end of both
  models. The mapping lives in one place, `DEVICE_NAMES` in `voxflow/app.py`:

  | `VOXFLOW_DEVICE` | speech recognition | language model |
  | --- | --- | --- |
  | `cpu` | `cpu` | `cpu` |
  | `cuda` | `cuda` | `cuda` |
  | `webgpu` or `vulkan` | `webgpu` | `vulkan` |

* GPU memory on CUDA: onnxruntime's memory arena grows with the longest clip and keeps that memory (Parakeet on
  the RTX 5080: 3.3 GB after the load, +1.1 GB after an 80 s clip, +5.2 GB after 240 s). A clip longer than 30 s
  that starts while the GPU has less than 2 GB free gives its working memory back when it ends
  (`SHRINK_AFTER_S` / `SHRINK_BELOW_FREE_MB` in `voxflow/asr.py`; free memory from NVML, as nvidia-smi shows it,
  0.02 ms). With room to spare nothing changes, because the next long clip would pay for allocating it again
  (measured 5-25 ms per long clip).
* Measured on the development PC (RTX 5080 / Ryzen 9800X3D, the 11.6 s clip, warm, medians of 5): CUDA 29 ms,
  WebGPU on the same card 65 ms, CPU 257 ms; load and warm-up 2.4 s, 3.5 s and 2.9 s. The text was identical on
  all three. **WebGPU was only ever run on that NVIDIA card and, forced, on the integrated AMD GPU; no separate
  AMD or Intel card has been tested, and neither has a PC whose discrete card is not the adapter Windows ranks
  first for performance (the `adapterIndex` case).** The memory check does not know about the language model: on
  a 4 GB card both models may not fit together, and each then falls back on its own.

## The speed check

`voxflow/speedcheck.py` replaces the fixed order by a measurement, once per PC, without making anyone wait: the
app always starts on the fixed order, and 30 s after both models are ready, with no dictation recorded or
processed, the controller measures every back-end of both models, each in a child process of its own
(`python -m voxflow --speed-check MODEL BACKEND OUT ...`, or the frozen exe with the same hidden flag) at
below-normal priority with a time limit (3 min for speech, 5 min for the language model). A child keeps the app's
own WebGPU adapter and llama.cpp build unpinned, frees its video memory when it exits, and cannot crash the app;
it writes JSON to a file it is given (and prints it), and the parent checks every field. Each child probes its
back-end as if the user had named it, so integrated GPUs are candidates; speech is timed with
`asr.time_backends` on the built-in 5 s clip, the language model on three short real clean-ups (its LLM pass),
medians of 3. A hotkey press kills the running child and the check starts over when VoxFlow is idle again. The
fastest back-end replaces the fixed choice only when it is at least 15% **and** 10 ms faster than the fixed choice
measured in the same run (`MIN_GAIN`, `MIN_SAVING_MS`); if the fixed choice could not be measured (for example
out of memory beside the app's own models), nothing changes. The result goes to `speedcheck.json` in the profile
folder, keyed by a fingerprint of the GPU names and driver versions and the CPU name (read from the registry,
`gpuprobe.display_adapters` / `cpu_name`), the VoxFlow version, the installed onnxruntime-gpu,
onnxruntime-ep-webgpu, onnx-asr and llama-cpp-python versions and the two model names; it is measured again only
when that key changes or the user presses **Measure again**. It is not a setting: never in a backup, a pack or the
sync folder. Precedence: `VOXFLOW_DEVICE` (which also switches the automatic check off), then `asr_device` /
`cleanup_device`, then the measured choice, then the fixed order (`app.model_devices`). A new speech device is
applied at once by reloading only the speech model (when no dictation is running); a new language-model device
at the next start. The log has the table (`speed check: speech: cuda 28 ms, webgpu 63 ms, cpu 143 ms -> keep the
automatic choice (cuda)`). `VOXFLOW_SPEED_CHECK=0` switches the automatic check off (the test suite does).
Limits: "webgpu" and "vulkan" each measure the adapter they would use when named (the discrete card first), so on
a PC with both an integrated and a discrete GPU the integrated one is not timed separately.

## Tests

```bat
.venv\Scripts\python.exe -m pytest tests -q              # hermetic: temp profile, offscreen Qt, no real desktop access
.venv\Scripts\python.exe scripts\coverage.py             # the suite under coverage, per-module table
.venv\Scripts\python.exe scripts\e2e_selftest.py         # real models on tests\tts_sample.wav, GPU then forced CPU
.venv\Scripts\python.exe scripts\demo_ui.py              # screenshots of every page and Flow bar state in build\shots\
.venv\Scripts\python.exe scripts\caption_gif.py          # docs\assets\live-preview.gif (offscreen; needs ffmpeg)
.venv\Scripts\python.exe scripts\record_demo.py          # docs\assets\demo.gif: a real dictation on the REAL desktop (needs ffmpeg)
```

* `set VOXFLOW_SKIP_MODEL_TESTS=1` skips the tests that load the real models.
* The suite points `CUDA_CACHE_PATH` at your real NVIDIA compute cache, so the real-model test does not recompile
  the GPU kernels on every run (about 25 s saved).
* The test harness (`tests/conftest.py`) fails any test that reaches the real clipboard, keyboard, hook,
  microphone, autostart registry key or `%APPDATA%\VoxFlow`.

### Checks on the real machine

These use the real hardware and the real Windows APIs. Each one works in a temp profile, puts back what it changes
and prints `PASSED`, `FAILED` or `BLOCKED`.

```bat
.venv\Scripts\python.exe scripts\mic_check.py                   # real microphone: device, format, level, speaker-to-mic loopback
.venv\Scripts\python.exe scripts\speech_matrix.py               # real models on varied speech: voices, 79 s dictation, quiet, noise,
                                                                #   formatting, and human clips in es fr de it pt nl (public FLEURS data)
.venv\Scripts\python.exe scripts\system_checks.py mute          # real Core Audio: mute another app's sound, restore, recover after a kill
.venv\Scripts\python.exe scripts\system_checks.py autostart     # real HKCU Run key under a throwaway value name
.venv\Scripts\python.exe scripts\system_checks.py download      # first run with an empty temp model cache (downloads about 5 GB)
.venv\Scripts\python.exe scripts\system_checks.py soak --minutes 30   # repeated dictations; memory, handles, threads must stay flat
.venv\Scripts\python.exe scripts\system_checks.py screen        # real screen(s): scaling, native window geometry, Flow bar, tray
```

The next ones press real keys and paste for real, so they need an unlocked desktop and about two minutes with your
hands off the keyboard:

```bat
set VOXFLOW_DESKTOP_TESTS=1 && .venv\Scripts\python.exe -m pytest tests -m desktop -rs
.venv\Scripts\python.exe -m voxflow --selftest tests\tts_sample.wav
.venv\Scripts\python.exe scripts\hook_probe.py                  # hold the shortcut through the real keyboard hook
.venv\Scripts\python.exe scripts\live_e2e.py                    # (add --alt-shortcuts for shortcuts with Alt) real hotkeys, real models, real paste into windows the script opens:
                                                                #   Qt box, Win32 edit, Notepad, an Electron editor, a browser field;
                                                                #   command mode, paste-last, undo, Esc, hands-free lock
```

They are safe to run on a desktop in use (`tests/desktop_kit.py`): a key press or paste is sent only if a window
the script itself opened is the foreground window at that instant, otherwise the step stops; your clipboard is
saved first and put back; if another VoxFlow is running, the scripts use their own shortcuts (Ctrl+Shift+F13..F16)
so your copy does not start recording. On a locked workstation they report `BLOCKED` and send nothing.

## Measured speed

`scripts\e2e_selftest.py`, real models, 11.6 s clip (`tests\tts_sample.wav`), RTX 5080 / Ryzen 9800X3D, medians
of 5 runs (2026-11-14; the GPU column is CUDA):

| | GPU | forced CPU |
| --- | --- | --- |
| speech recognition | 30 ms | 263 ms |
| release to paste, clean sentence (no language model needed) | 32 ms | 245 ms |
| language-model cleanup of a 41-word disfluent sentence | 155 ms | 1630 ms |
| release to paste when the language model is needed | 192 ms | 1901 ms |

Method: the script loads the real Parakeet and Qwen models, runs the pipeline on the clip, then drives the real
controller (hotkey start, a recorder that plays the clip, hotkey release, recognition, cleanup, dictionary and
snippets, paste into a recording stand-in) and reads the latency the app itself stores in the history entry.
It runs once with the device on `auto` (GPU) and once forced to CPU (`--device webgpu` runs speech through WebGPU), and it also records any outgoing network
connection attempted while the models load and run; that count was 0 on both devices.

The design target is 1.5 s from release to paste for a 10 s utterance (and 400 ms for the language-model pass on
about 40 words on the GPU). On CPU, sentences that need the language model exceed it. Switching Smart cleanup off
avoids that wait, but then the raw transcript is pasted with no cleanup at all.

`scripts\bench_asr.py` and `scripts\bench_cleanup.py` benchmark the two models on their own.

## How it is built

| file | role |
| --- | --- |
| `voxflow/controller.py` | the one object the UI talks to: settings, stores, models, hotkeys, pipeline worker, status |
| `voxflow/app.py`, `__main__.py` | process start-up, single instance, logging, `--selftest` |
| `voxflow/hotkey.py` | pure state machines + `WH_KEYBOARD_LL` hook; several named shortcuts, capture mode, watchdog |
| `voxflow/audio.py` | always-open 16 kHz microphone stream with pre-roll, device discovery by name, hot-unplug recovery, whisper gain |
| `voxflow/asr.py` | Parakeet TDT via onnx-asr on onnxruntime (CUDA with CPU fallback) |
| `voxflow/cleanup.py` | rule pass, guarded local LLM (llama.cpp) and the choice of its build, smart formatting, tone, language handling |
| `voxflow/gpuprobe.py` | which graphics hardware the drivers report (NVIDIA driver, Vulkan devices); no model code |
| `voxflow/langs.py` | language packs (es, fr, de, it, pt, nl) and the neutral fallback: rules, prompts, guardrails, number formatting |
| `voxflow/command.py` | command mode: prompt, token budget, guardrails |
| `voxflow/sync.py` | backups, shareable packs, sync folder (one file per device, last writer wins per entry, tombstones); no network code |
| `voxflow/updater.py` | app updates: the daily GitHub Releases check, version compare, URL allow-list, resumable and SHA-256-verified installer download, Scoop / winget / source detection, the silent Inno Setup run after VoxFlow quits (`Updater`, owned by the controller); `VOXFLOW_UPDATES=0` turns the automatic check off |
| `voxflow/inject.py` | clipboard snapshot -> set text -> `SendInput` Ctrl+V -> restore; typed fallback; selection grab |
| `voxflow/store.py` | dictionary, snippets, history, stats, styles (JSON, atomic writes) |
| `voxflow/sounds.py`, `system.py`, `context.py`, `modelhub.py` | cues + media mute, autostart / logging / crash handler, foreground app, model cache |
| `voxflow/ui/` | design system, Flow bar, main window and pages, setup guide, tray |

The default models are `istupakov/parakeet-tdt-0.6b-v3-onnx` (speech) and
`bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`, file `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (cleanup).

## README and release assets

Everything under `docs/assets/` is referenced by the README or by the GitHub repository settings.

* `screenshots/` are renders of the real UI with sample data, produced by `scripts\demo_ui.py`.
* `hero.png` (README banner, 1280x420) and `social-preview.png` (GitHub social card, 1280x640): the waveform
  artwork was generated locally with ComfyUI (Z-Image-Turbo) from a text-free prompt; the logo mark (the same
  shape `packaging\make_icon.py` renders), the wordmark and the tagline were then composited locally with
  Pillow in Segoe UI, using the app's own cyan and violet from `voxflow/ui/theme.py`. No online image service
  was used.
* `demo.gif` is a screen recording of one real dictation, made with `scripts\record_demo.py` (0.3.0, RTX 5080):
  the real keyboard hook, speech recognition, live preview, cleanup model, Flow bar and paste, into a note page
  open in a Microsoft Edge app window. The one stand-in is the microphone: a clip from Windows' offline speech
  engine (voice *Microsoft Zira Desktop*) is fed in real time into the recorder's audio callback, so the level
  meter, the live-preview caption above the Flow bar and the final transcript all hear it as microphone audio. The
  line at the top shows the words in the clip; the caption is what the live preview recognised; the text in the
  note is what VoxFlow pasted. The script takes over the middle of the primary screen for about 30 seconds and
  sends keys only into its own windows (as `scripts\live_e2e.py` does); it needs ffmpeg on PATH.
* `live-preview.gif` is rendered by `scripts\caption_gif.py`: the real Flow bar widget, offscreen, at 150 %, with
  scripted words fed to its live-preview caption and the preview controller twin driving its states (no model,
  no microphone). It needs ffmpeg on PATH.
