"""Packaging round: single version source, installer script wiring, frozen-mode helpers (no build is run here)."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

import voxflow
from voxflow import system

ROOT = Path(__file__).resolve().parent.parent


def _load_build_script():
    spec = importlib.util.spec_from_file_location("build_installer", ROOT / "scripts" / "build_installer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_installer_version_comes_from_the_package():
    assert _load_build_script().version() == voxflow.__version__
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    assert "{#AppVersion}" in iss and not re.search(r'AppVersion=\d', iss)      # never hard-coded in the script


def test_the_installer_is_per_user_with_a_stable_id_and_keeps_the_profile():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    assert "PrivilegesRequired=lowest" in iss
    assert r"{localappdata}\Programs\VoxFlow" in iss
    assert re.search(r"AppId=\{\{[0-9A-F-]{36}\}", iss)
    assert "{userappdata}" not in iss                                           # profile and models are never removed
    deletes = iss.split("\n[UninstallDelete]\n", 1)[1].split("\n[", 1)[0]
    entries = [ln for ln in deletes.splitlines() if ln.strip() and not ln.lstrip().startswith(";")]
    assert entries == [r'Type: files; Name: "{app}\managed-by.txt"',              # the winget marker, and the
                       r'Type: filesandordirs; Name: "{localappdata}\VoxFlow\cuda-cache"',   # GPU kernel cache
                       r'Type: dirifempty; Name: "{localappdata}\VoxFlow"'], "never the profile or the models"
    assert 'Description: "Launch VoxFlow"' in iss and "postinstall" in iss


def _iss_define(iss: str, name: str) -> str:
    return re.search(rf'^#define {name} "(.*)"\s*$', iss, re.M).group(1)


def test_the_installer_adds_no_autostart_entry_and_the_uninstaller_removes_the_apps_own():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    code = iss.split("\n[Code]\n", 1)[1]
    # the names are the app's: a rename in voxflow/system.py without the installer would leave the entry behind
    assert _iss_define(iss, "RunKey") == system.RUN_KEY
    assert _iss_define(iss, "RunValue") == system.RUN_VALUE
    # Setup never creates an autostart entry: no [Registry] section, no registry write from code, no Startup shortcut,
    # and the Run key is spelled out nowhere but in that one #define
    assert not re.search(r"^\s*\[Registry\]", iss, re.M | re.I) and not re.search(r"RegWrite\w*", iss, re.I)
    assert not re.search(r"\{(user|common|auto)startup\}", iss, re.I)
    assert iss.count("CurrentVersion\\Run") == 1
    # removal happens in the uninstaller only (an upgrade keeps the user's setting), on HKCU, for that one value
    events = re.findall(r"^(?:procedure|function)\s+(\w+)", code, re.M)
    assert events == ["StopRunningCopy", "PrepareToInstall", "RelaunchAfterUpdate", "CurStepChanged",
                      "CurUninstallStepChanged"]
    assert "CurUninstallStep <> usUninstall" in code
    assert code.count("RegDeleteValue(HKEY_CURRENT_USER, '{#RunKey}', '{#RunValue}')") == 1
    assert len(re.findall(r"RegDelete\w+", code)) == 1                          # nothing else is deleted
    assert "ExpandConstant('{app}')" in code            # ... and only when the entry starts this install's exe


def test_the_installer_relaunches_only_for_the_in_app_updater_and_marks_winget_installs():
    """voxflow/updater.py runs Setup silently with /RELAUNCH=1 (Install now) or 0 (when I quit); winget passes
    /MANAGEDBY=winget, which leaves managed-by.txt for updater.install_kind."""
    from voxflow import updater
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    run = iss.split("\n[Run]\n", 1)[1].split("\n[", 1)[0]
    lines = [ln for ln in run.splitlines() if ln.startswith("Filename:")]
    assert len(lines) == 2 and "skipifsilent" in lines[0] and "Check: RelaunchAfterUpdate" in lines[1]
    code = iss.split("\n[Code]\n", 1)[1]
    assert "WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1')" in code
    assert "ExpandConstant('{param:MANAGEDBY|}')) = 'winget'" in code and updater.MANAGED_MARKER in code
    assert "CurStep <> ssPostInstall" in code
    cmd = updater.installer_command(Path("C:/t/VoxFlow-Setup-1.0.0.exe"), True)
    assert cmd[1:] == ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/RELAUNCH=1"]
    manifest = (ROOT / "packaging" / "winget" / "blackknight75.VoxFlow.installer.yaml").read_text("utf-8")
    assert re.search(r"^InstallerSwitches:\n(?:  #.*\n)*  Custom: /MANAGEDBY=winget$", manifest, re.M)
    assert "CloseApplications=force" in iss            # Setup also closes a VoxFlow that is still exiting


def test_the_app_writes_its_login_entry_with_a_path_inside_the_install_folder(monkeypatch):
    """What the uninstaller matches on: the frozen app's Run command contains ``{app}\\``."""
    app_dir = r"C:\Users\x\AppData\Local\Programs\VoxFlow"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", app_dir + r"\VoxFlow.exe")
    assert (app_dir + "\\").lower() in system.launch_command().lower()


def test_the_installer_requires_windows_11_like_the_winget_manifest():
    iss = (ROOT / "packaging" / "voxflow.iss").read_text("utf-8")
    setup = iss.split("\n[Setup]\n", 1)[1].split("\n[", 1)[0]
    minimum = re.search(r"^MinVersion=([\d.]+)\s*$", setup, re.M).group(1)
    assert minimum == "10.0.22000"                                              # the first Windows 11 build
    manifest = (ROOT / "packaging" / "winget" / "blackknight75.VoxFlow.installer.yaml").read_text("utf-8")
    winget = re.search(r"^MinimumOSVersion: ([\d.]+)\s*$", manifest, re.M).group(1)
    assert winget.split(".")[:3] == minimum.split(".") and winget.split(".")[3:] in ([], ["0"])


def test_version_resource_is_generated_from_the_version(tmp_path):
    pytest.importorskip("PyInstaller")
    out = tmp_path / "v.txt"
    _load_build_script().write_version_file(out, "1.2.3")
    text = out.read_text("utf-8")
    assert "filevers=(1, 2, 3, 0)" in text and "ProductVersion', '1.2.3'" in text


def test_icon_file_is_a_valid_multi_size_ico(tmp_path, qapp):
    spec = importlib.util.spec_from_file_location("make_icon", ROOT / "packaging" / "make_icon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "x.ico"
    mod.build(out)
    data = out.read_bytes()
    assert data[:4] == b"\x00\x00\x01\x00" and int.from_bytes(data[4:6], "little") >= 6
    assert b"\x89PNG" in data


def test_a_frozen_app_starts_at_login_through_its_own_exe(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Users\x\AppData\Local\Programs\VoxFlow\VoxFlow.exe")
    cmd = system.launch_command()
    assert cmd == r'"C:\Users\x\AppData\Local\Programs\VoxFlow\VoxFlow.exe" --autostart'


def test_entry_point_survives_a_missing_stdout(monkeypatch):
    """The windowed exe has no console: sys.stdout is None there; the entry stub must give it a sink."""
    src = (ROOT / "packaging" / "voxflow_entry.py").read_text("utf-8")
    assert "freeze_support" in src and "is None" in src and "voxflow.__main__" in src


def test_spec_drops_what_the_app_does_not_use():
    spec = (ROOT / "packaging" / "voxflow.spec").read_text("utf-8")
    for unused in ("PySide6.QtWebEngineCore", "PySide6.QtPdf", "PySide6.QtQml", "pytest", "onnxruntime_providers_tensorrt"):
        assert unused in spec
    assert "llama_cpp/lib" in spec and "nvidia" in spec and "copy_metadata" in spec


def test_spec_bundles_the_webgpu_plugin_next_to_its_package():
    spec = (ROOT / "packaging" / "voxflow.spec").read_text("utf-8")
    for needed in ('"onnxruntime_providers_webgpu.dll", "dxcompiler.dll"', '"onnxruntime_ep_webgpu")',
                   '"onnxruntime_ep_webgpu", "huggingface_hub"', '"onnxruntime-ep-webgpu"'):
        assert needed in spec
    # the size filter drops files by a part of their name: it must not catch the plugin's two DLLs
    drop = spec[spec.index("DROP_SUBSTR = ("):spec.index("KEEP_IMAGEFORMATS")]
    dropped = [s.lower() for s in __import__("re").findall(r'"([^"]+)"', drop)]
    for dll in ("onnxruntime_providers_webgpu.dll", "dxcompiler.dll"):
        assert not [s for s in dropped if s in dll], dll
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
    for needed in ("_internal/onnxruntime_ep_webgpu/onnxruntime_providers_webgpu.dll",
                   "_internal/onnxruntime_ep_webgpu/dxcompiler.dll"):
        assert needed in release
    requirements = (ROOT / "requirements.txt").read_text("utf-8")
    assert "onnxruntime-ep-webgpu==0.5.0" in requirements.split()


# ---- the llama.cpp builds: CUDA (the wheel's own), Vulkan and CPU, each in its own folder ------------------------
def _load_fetch_script():
    spec = importlib.util.spec_from_file_location("fetch_llama_backends", ROOT / "scripts" / "fetch_llama_backends.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wheel(files: dict) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def _sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_spec_bundles_every_llama_build_folder_the_app_chooses_from():
    from voxflow.cleanup import LLAMA_BACKENDS
    spec = (ROOT / "packaging" / "voxflow.spec").read_text("utf-8")
    assert [b.folder for b in LLAMA_BACKENDS] == ["lib_cuda13", "lib", "lib_vulkan", "lib_cpu"]
    assert 'for folder in ("lib_cuda13", "lib_vulkan", "lib_cpu"):' in spec and 'f"llama_cpp/{folder}"' in spec
    assert 'for sub in ("cu13", "cudnn"):' in spec, "the CUDA 13 cuBLAS that lib_cuda13's ggml-cuda.dll loads"
    assert "version.txt" in spec, "the folder's release marker is bundled: the app refuses a build without a match"
    assert "raise SystemExit" in spec, "a freeze without the Vulkan / CPU builds must fail, not ship a CUDA-only app"


def test_the_build_fetches_the_extra_llama_builds_before_the_freeze():
    src = (ROOT / "scripts" / "build_installer.py").read_text("utf-8")
    assert src.index("fetch_llama_backends.fetch()") < src.index('"-m", "PyInstaller"')
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
    for needed in ("scripts/fetch_llama_backends.py", "_internal/llama_cpp/lib_cpu/llama.dll",
                   "_internal/llama_cpp/lib_vulkan/ggml-vulkan.dll", "_internal/llama_cpp/lib_cpu/version.txt",
                   "_internal/llama_cpp/lib_cuda13/ggml-cuda.dll", "_internal/llama_cpp/lib_cuda13/version.txt"):
        assert needed in release


def test_the_extra_llama_builds_are_pinned_for_the_required_version():
    fetch = _load_fetch_script()
    required = re.search(r"^llama-cpp-python==(\S+)\s*$", (ROOT / "requirements.txt").read_text("utf-8"), re.M).group(1)
    pins = fetch.PINS[required]
    assert list(pins) == fetch.backend_folders() == ["lib_cuda13", "lib_vulkan", "lib_cpu"]
    for folder, (kind, url, sha256) in pins.items():
        assert re.fullmatch(r"[0-9a-f]{64}", sha256)
        if kind == "wheel":
            assert url.startswith("https://github.com/abetlen/llama-cpp-python/releases/download/v" + required)
            assert url.endswith(f"llama_cpp_python-{required}-py3-none-win_amd64.whl")
    assert "-vulkan/" in pins["lib_vulkan"][1] and pins["lib_vulkan"][2] != pins["lib_cpu"][2]
    # llama-cpp-python 0.3.36 vendors llama.cpp 0c1e570, which ggml-org released as b11312
    assert pins["lib_cuda13"][:2] == ("cuda13", "https://github.com/ggml-org/llama.cpp/releases/download/b11312/"
                                                "llama-b11312-bin-win-cuda-13.4-x64.zip")
    assert fetch.WHEEL_SHARED == ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll")


def test_fetch_installs_only_the_dlls_of_a_wheel_that_matches_its_pin(tmp_path, monkeypatch, capsys):
    fetch = _load_fetch_script()
    vulkan = _wheel({"llama_cpp/lib/llama.dll": b"VK-LLAMA", "llama_cpp/lib/ggml-vulkan.dll": b"VK",
                     "llama_cpp/lib/llama.lib": b"import library", "llama_cpp/llama.py": b"python",
                     "bin/llama.dll": b"another copy"})
    cpu = _wheel({"llama_cpp/lib/llama.dll": b"CPU-LLAMA", "llama_cpp/lib/ggml-cpu.dll": b"CPU"})
    wheels = {"https://example.invalid/vulkan.whl": vulkan, "https://example.invalid/cpu.whl": cpu}
    monkeypatch.setattr(fetch.metadata, "version", lambda dist: "9.9.9")
    monkeypatch.setattr(fetch, "backend_folders", lambda: ["lib_vulkan", "lib_cpu"])
    monkeypatch.setattr(fetch, "PINS", {"9.9.9": {
        "lib_vulkan": ("wheel", "https://example.invalid/vulkan.whl", _sha(vulkan)),
        "lib_cpu": ("wheel", "https://example.invalid/cpu.whl", _sha(cpu))}})
    dest = tmp_path / "llama_cpp"
    stale = dest / "lib_cpu"
    stale.mkdir(parents=True)
    (stale / "old-release.dll").write_bytes(b"stale")
    (stale / "llama.dll").write_bytes(b"stale")
    (stale / "version.txt").write_text("9.9.8\n", "utf-8")
    asked = []

    assert fetch.fetch(dest, check_only=True) == 1 and "lib_cpu: MISSING" in capsys.readouterr().out
    assert fetch.fetch(dest, fetcher=lambda url: asked.append(url) or wheels[url]) == 0
    assert asked == ["https://example.invalid/vulkan.whl", "https://example.invalid/cpu.whl"]
    assert sorted(p.name for p in (dest / "lib_vulkan").iterdir()) == ["ggml-vulkan.dll", "llama.dll", "version.txt"]
    assert (dest / "lib_vulkan" / "llama.dll").read_bytes() == b"VK-LLAMA"
    assert sorted(p.name for p in stale.iterdir()) == ["ggml-cpu.dll", "llama.dll", "version.txt"], "no mix of releases"
    assert (stale / "version.txt").read_text("utf-8").strip() == "9.9.9"
    assert "lib_vulkan: installed ggml-vulkan.dll, llama.dll" in capsys.readouterr().out

    # a second run needs no network, and --check is satisfied
    assert fetch.fetch(dest, fetcher=lambda url: pytest.fail("nothing may be downloaded")) == 0
    assert fetch.fetch(dest, check_only=True) == 0
    assert capsys.readouterr().out.count("in place") == 4


def test_fetch_refuses_a_download_that_does_not_match_its_pin(tmp_path, monkeypatch):
    fetch = _load_fetch_script()
    good = _wheel({"llama_cpp/lib/llama.dll": b"REAL"})
    monkeypatch.setattr(fetch.metadata, "version", lambda dist: "9.9.9")
    monkeypatch.setattr(fetch, "backend_folders", lambda: ["lib_vulkan", "lib_cpu"])
    monkeypatch.setattr(fetch, "PINS", {"9.9.9": {"lib_vulkan": ("wheel", "u", _sha(good)),
                                                  "lib_cpu": ("wheel", "u", _sha(good))}})
    with pytest.raises(SystemExit) as exc:
        fetch.fetch(tmp_path, fetcher=lambda url: _wheel({"llama_cpp/lib/llama.dll": b"TAMPERED"}))
    assert "refusing the download for lib_vulkan: SHA-256 is" in str(exc.value)
    assert not (tmp_path / "lib_vulkan").exists(), "nothing is unpacked from a wheel that fails the check"

    empty = _wheel({"llama_cpp/llama.py": b"no libraries"})
    monkeypatch.setattr(fetch, "PINS", {"9.9.9": {"lib_vulkan": ("wheel", "u", _sha(empty)),
                                                  "lib_cpu": ("wheel", "u", _sha(empty))}})
    with pytest.raises(SystemExit) as exc:
        fetch.fetch(tmp_path, fetcher=lambda url: empty)
    assert "has no llama_cpp/lib/llama.dll" in str(exc.value)


def test_fetch_stops_when_the_installed_version_has_no_pins_or_llama_cpp_is_missing(monkeypatch):
    fetch = _load_fetch_script()
    monkeypatch.setattr(fetch.metadata, "version", lambda dist: "0.0.1")
    with pytest.raises(SystemExit) as exc:
        fetch.fetch(fetcher=lambda url: pytest.fail("nothing may be downloaded"))
    assert "llama-cpp-python 0.0.1 is installed" in str(exc.value) and "PINS" in str(exc.value)
    from voxflow import cleanup
    monkeypatch.setattr(cleanup, "_llama_pkg_dir", lambda: None)
    with pytest.raises(SystemExit) as exc:
        fetch.package_dir()
    assert "llama-cpp-python is not installed" in str(exc.value)
    monkeypatch.setattr(cleanup, "_llama_pkg_dir", lambda: r"C:\site\llama_cpp")
    assert fetch.package_dir() == Path(r"C:\site\llama_cpp")


def test_fetch_command_line(tmp_path, monkeypatch):
    fetch = _load_fetch_script()
    calls = []
    monkeypatch.setattr(fetch, "fetch", lambda dest, check: calls.append((dest, check)) or 0)
    assert fetch.main(["--check", "--dest", str(tmp_path)]) == 0 and fetch.main([]) == 0
    assert calls == [(tmp_path, True), (None, False)]


def test_fetch_downloads_over_https_with_a_generic_client_name(monkeypatch):
    fetch = _load_fetch_script()
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"wheel bytes"

    def urlopen(request, timeout):
        seen.update(url=request.full_url, agent=request.get_header("User-agent"), timeout=timeout)
        return Response()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    assert fetch.download("https://github.com/abetlen/llama-cpp-python/releases/download/x.whl") == b"wheel bytes"
    assert seen["url"].startswith("https://github.com/abetlen/") and seen["agent"] == "python-urllib"


def _cuda_wheel_lib(lib: Path, ggml=b"MZ imports ggml-cuda.dll and ggml-cpu.dll") -> Path:
    lib.mkdir(parents=True, exist_ok=True)
    for name in ("llama.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll"):
        (lib / name).write_bytes(b"WHEEL-" + name.encode())
    (lib / "ggml.dll").write_bytes(ggml)
    return lib


def test_qa_c01_the_cuda13_folder_is_the_archives_ggml_cuda_next_to_the_wheels_own_dlls(tmp_path, monkeypatch,
                                                                                        capsys):
    fetch = _load_fetch_script()
    archive = _wheel({"ggml-cuda.dll": b"SM120", "llama.dll": b"UPSTREAM-LLAMA", "ggml-cpu-zen4.dll": b"X",
                      "llama-server.exe": b"exe"})
    dest = tmp_path / "llama_cpp"
    _cuda_wheel_lib(dest / "lib")
    monkeypatch.setattr(fetch.metadata, "version", lambda dist: "9.9.9")
    monkeypatch.setattr(fetch, "backend_folders", lambda: ["lib_cuda13"])
    monkeypatch.setattr(fetch, "PINS", {"9.9.9": {"lib_cuda13": ("cuda13", "https://example.invalid/b1.zip",
                                                                 _sha(archive))}})
    assert fetch.fetch(dest, fetcher=lambda url: archive) == 0
    folder = dest / "lib_cuda13"
    assert sorted(p.name for p in folder.iterdir()) == ["ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll",
                                                        "llama.dll", "mtmd.dll", "version.txt"]
    assert (folder / "ggml-cuda.dll").read_bytes() == b"SM120", "the only file taken from the archive"
    assert (folder / "llama.dll").read_bytes() == b"WHEEL-llama.dll", "the bindings keep talking to the wheel's llama"
    assert (folder / "version.txt").read_text("utf-8").strip() == "9.9.9"
    assert "lib_cuda13: installed ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll" in capsys.readouterr().out
    assert fetch.fetch(dest, fetcher=lambda url: pytest.fail("nothing may be downloaded")) == 0


def test_qa_c01_the_cuda13_folder_refuses_a_bad_archive_or_a_wheel_that_is_not_the_cuda_one(tmp_path):
    fetch = _load_fetch_script()
    good = _wheel({"bin/ggml-cuda.dll": b"SM120"})
    lib = _cuda_wheel_lib(tmp_path / "lib")
    with pytest.raises(SystemExit, match="refusing the download for lib_cuda13: SHA-256 is"):
        fetch.install_cuda13(tmp_path / "lib_cuda13", _wheel({"ggml-cuda.dll": b"TAMPERED"}), _sha(good), "1", lib)
    assert not (tmp_path / "lib_cuda13").exists()
    for wrong in (_wheel({"llama.dll": b"no cuda"}), _wheel({"a/ggml-cuda.dll": b"1", "b/ggml-cuda.dll": b"2"})):
        with pytest.raises(SystemExit, match="has no single ggml-cuda.dll"):
            fetch.install_cuda13(tmp_path / "lib_cuda13", wrong, _sha(wrong), "1", lib)
    vulkan_lib = _cuda_wheel_lib(tmp_path / "vk", ggml=b"MZ imports ggml-vulkan.dll")
    with pytest.raises(SystemExit, match="is not from the CUDA wheel"):
        fetch.install_cuda13(tmp_path / "lib_cuda13", good, _sha(good), "1", vulkan_lib)
    (lib / "mtmd.dll").unlink()
    with pytest.raises(SystemExit, match=r"mtmd.dll is missing: install requirements.txt"):
        fetch.install_cuda13(tmp_path / "lib_cuda13", good, _sha(good), "1", lib)
    assert not (tmp_path / "lib_cuda13").exists(), "nothing is written before every check passed"
    (lib / "mtmd.dll").write_bytes(b"WHEEL-mtmd.dll")
    assert "ggml-cuda.dll" in fetch.install_cuda13(tmp_path / "lib_cuda13", good, _sha(good), "1", lib)
