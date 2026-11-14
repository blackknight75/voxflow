# PyInstaller spec for VoxFlow (onedir; GPU and CPU builds of the models' runtimes). Run through
# scripts/build_installer.py.
# Inputs by environment: VOXFLOW_ICON (.ico), VOXFLOW_VERSION_FILE (PyInstaller version resource).
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
SITE = Path(next(p for p in sys.path if p.endswith("site-packages")))
ICON = os.environ.get("VOXFLOW_ICON") or None
VERSION_FILE = os.environ.get("VOXFLOW_VERSION_FILE") or None

# What the app does not use. Qt: only Core/Gui/Widgets/Network are imported by voxflow.
EXCLUDES = [
    "tkinter", "pytest", "pytest_qt", "_pytest", "coverage", "pip", "setuptools", "pkg_resources", "unittest",
    "IPython", "matplotlib", "scipy", "pandas", "PIL", "onnx", "torch", "tensorflow",
    "tests", "scripts", "llama_cpp.server", "uvicorn", "fastapi", "starlette", "pydantic_settings", "sse_starlette",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel",
    "PySide6.QtWebSockets", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtQuickWidgets", "PySide6.QtQuick3D", "PySide6.QtQuickControls2", "PySide6.Qt3DCore", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtSvg", "PySide6.QtSvgWidgets", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtPrintSupport", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtStateMachine", "PySide6.QtTextToSpeech", "PySide6.QtSpatialAudio", "PySide6.QtGraphs",
    "PySide6.QtHttpServer", "PySide6.QtNetworkAuth", "PySide6.QtConcurrent", "PySide6.QtDBus", "PySide6.QtXml",
    "PySide6.QtUiTools", "PySide6.QtAxContainer", "PySide6.QtSerialBus", "PySide6.QtShaderTools",
]

# DLLs / files that are in the wheels but never used here (saves ~1.5 GB)
DROP_SUBSTR = (
    "onnxruntime_providers_tensorrt", "nvblas", "cufftw",          # TensorRT EP, NVBLAS, FFTW shim
    "Qt6WebEngine", "Qt6Pdf", "Qt6Quick", "Qt6Qml", "Qt63D", "Qt6Multimedia", "Qt6Designer", "opengl32sw",
    "Qt6OpenGL", "Qt6Sql", "Qt6Test", "Qt6Svg", "Qt6Charts", "Qt6DataVisualization", "Qt6Graphs", "Qt6Bluetooth",
    "Qt6Positioning", "Qt6Sensors", "Qt6Nfc", "Qt6Scxml", "Qt6RemoteObjects", "Qt6VirtualKeyboard", "Qt6ShaderTools",
    "Qt6Concurrent", "Qt6DBus", "Qt6Xml", "Qt6PrintSupport", "Qt6Help", "Qt6Location", "Qt6HttpServer",
    "Qt6SerialPort", "Qt6SerialBus", "Qt6Test", "Qt6TextToSpeech", "Qt6SpatialAudio", "Qt6StateMachine",
    "QtWebEngineProcess", "d3dcompiler", "Qt6Labs", "Qt6Protobuf", "Qt6Grpc", "Qt6WebChannel", "Qt6WebSockets",
    "Qt6NetworkAuth", "Qt6Quick3D", "Qt6Core5Compat", "Qt6StateMachine", "Qt6UiTools", "Qt6Lottie",
)
KEEP_IMAGEFORMATS = ("qico", "qjpeg", "qgif")


def wanted(dest: str, src: str) -> bool:
    norm = dest.replace("\\", "/")
    base = norm.rsplit("/", 1)[-1]
    if any(s.lower() in base.lower() for s in DROP_SUBSTR):
        return False
    if "/translations/" in norm or norm.startswith("PySide6/translations") or "/qml/" in norm.lower():
        return False
    if norm.startswith("PySide6/plugins/"):                       # keep only what a widgets app on Windows needs
        sub = norm.split("/")[2]
        if sub == "platforms":
            return base == "qwindows.dll"
        if sub == "styles":
            return base in ("qmodernwindowsstyle.dll", "qwindowsvistastyle.dll")
        if sub == "imageformats":
            return base.split(".")[0] in KEEP_IMAGEFORMATS
        return False
    # the CUDA 12 duplicates inside the nvidia wheels: llama_cpp/lib carries its own copies
    if norm.startswith("nvidia/cublas/") or norm.startswith("nvidia/cuda_runtime/"):
        return False
    return True


# ---- CUDA 13 runtime for onnxruntime-gpu: keep the wheel layout, voxflow.asr walks sys.path for "nvidia/**/*.dll"
extra_binaries = []
for sub in ("cu13", "cudnn"):
    base = SITE / "nvidia" / sub
    for dll in base.rglob("*.dll"):
        rel = dll.relative_to(SITE).parent.as_posix()
        extra_binaries.append((str(dll), rel))

# ---- llama.cpp (CUDA 12 build): the package loads its DLLs from llama_cpp/lib with ctypes, so PyInstaller cannot see
# them. The wheel's lib folder already holds cudart64_12 / cublas64_12 / cublasLt64_12 (what --install-cuda adds).
for dll in (SITE / "llama_cpp" / "lib").glob("*.dll"):
    extra_binaries.append((str(dll), "llama_cpp/lib"))

# ---- the other llama.cpp builds of the same release, each in its own folder: Vulkan (AMD / Intel / NVIDIA cards)
# and plain CPU (a PC without any GPU driver, where the CUDA and Vulkan builds cannot even be loaded).
# voxflow.cleanup picks one folder at start; scripts/fetch_llama_backends.py puts them into site-packages.
llama_extra_data = []
for folder in ("lib_vulkan", "lib_cpu"):
    src = SITE / "llama_cpp" / folder
    if not (src / "llama.dll").is_file() or not (src / "version.txt").is_file():
        raise SystemExit(f"{src} is missing or incomplete: run scripts/fetch_llama_backends.py "
                         "(scripts/build_installer.py does that before the freeze)")
    for dll in src.glob("*.dll"):
        extra_binaries.append((str(dll), f"llama_cpp/{folder}"))
    llama_extra_data.append((str(src / "version.txt"), f"llama_cpp/{folder}"))

# ---- the WebGPU execution provider for onnxruntime (speech recognition on AMD / Intel / NVIDIA cards): a plugin that
# voxflow.asr registers at run time. The package finds its DLL next to its own __init__ (get_library_path), and
# dxcompiler.dll (the shader compiler Dawn needs for Direct3D 12) is loaded from the same folder: both stay in
# onnxruntime_ep_webgpu/. PyInstaller cannot see them: nothing imports them.
webgpu_dir = SITE / "onnxruntime_ep_webgpu"
for name in ("onnxruntime_providers_webgpu.dll", "dxcompiler.dll"):
    if not (webgpu_dir / name).is_file():
        raise SystemExit(f"{webgpu_dir / name} is missing: install requirements.txt (onnxruntime-ep-webgpu)")
    extra_binaries.append((str(webgpu_dir / name), "onnxruntime_ep_webgpu"))

# importlib.metadata lookups (onnx-asr asks for its own and onnxruntime's version at import time)
metadata = collect_data_files("onnx_asr")   # preprocessor .onnx / .npz data
metadata += llama_extra_data

for dist in ("onnx-asr", "onnxruntime-gpu", "onnxruntime-ep-webgpu", "huggingface-hub", "numpy", "llama_cpp_python", "tqdm", "packaging",
             "filelock", "requests", "httpx", "pyyaml", "hf-xet", "sounddevice", "PySide6", "shiboken6"):
    try:
        metadata += copy_metadata(dist)
    except Exception as exc:  # noqa: BLE001 - optional ones may not be installed
        print("no metadata for", dist, exc)

a = Analysis(
    [str(ROOT / "packaging" / "voxflow_entry.py")],
    pathex=[str(ROOT)],
    binaries=extra_binaries,
    datas=metadata,
    hiddenimports=collect_submodules("voxflow") + collect_submodules("llama_cpp", filter=lambda n: not n.startswith("llama_cpp.server")) + ["sounddevice", "onnx_asr", "onnxruntime_ep_webgpu", "huggingface_hub", "winsound", "winreg",
                                                    "certifi"],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
a.binaries = [b for b in a.binaries if wanted(b[0], b[1])]
a.datas = [d for d in a.datas if wanted(d[0], d[1])]
pyz = PYZ(a.pure)

common = dict(icon=ICON, version=VERSION_FILE, strip=False, upx=False, bootloader_ignore_signals=False)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VoxFlow", console=False, **common)
con = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VoxFlow-console", console=True, **common)
coll = COLLECT(gui, con, a.binaries, a.datas, strip=False, upx=False, name="VoxFlow")
