"""Which graphics hardware this PC can compute on, asked from the installed drivers.

One small place for every "is there a usable GPU, and whose?" question, so the model back-ends do not each grow
their own: ``voxflow.cleanup`` picks its llama.cpp build with it, speech recognition can use it the same way.

    nvidia_driver()      -> bool         the NVIDIA driver's CUDA library loads (an NVIDIA card with its driver)
    cuda_info()          -> CudaInfo     the CUDA version that driver supports and each device's compute capability
    cuda_free_mb()       -> int | None   free memory of CUDA device 0 in MiB
    vulkan_gpus()        -> list[Gpu]    the devices the Vulkan loader reports: any vendor (AMD, Intel, NVIDIA, ...)
    usable_vulkan_gpu(n) -> Gpu | None   the best real GPU among them with at least ``n`` MiB of memory
    library_loads(name)  -> bool         a DLL can be loaded by name
    display_adapters()   -> list         (name, driver version) of every display adapter Windows has a driver for
    cpu_name()           -> str          the processor's marketing name

The last two read the registry only (no driver is loaded): they are what the speed check's cache is keyed by.

Standard library only (ctypes): nothing is installed, downloaded or written, and no model is touched. Every
probe answers "no" instead of raising, because a broken driver must never stop the app from starting.
A new platform (Apple Metal later) adds its own probe function here; nothing below is specific to one vendor
except ``nvidia_driver``.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass

log = logging.getLogger("voxflow.gpuprobe")

VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x1022: "AMD", 0x8086: "Intel", 0x5143: "Qualcomm", 0x1414: "Microsoft"}
# VkPhysicalDeviceType
_KINDS = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}
# a software rasteriser ("cpu") or an unknown device is not a GPU worth offloading a model to
GPU_KINDS = ("discrete", "integrated")


@dataclass(frozen=True)
class Gpu:
    """One graphics device as its driver describes it."""
    name: str
    vendor_id: int
    kind: str            # "discrete" | "integrated" | "virtual" | "cpu" | "other"
    vram_mb: int         # device-local memory; for an integrated GPU the share of system RAM it may use

    @property
    def vendor(self) -> str:
        return VENDORS.get(self.vendor_id, f"vendor 0x{self.vendor_id:04X}")

    def __str__(self) -> str:
        return f"{self.name} ({self.vendor}, {self.kind}, {self.vram_mb} MiB)"


def _load_library(name: str):
    """The one place a DLL is loaded by name (tests replace it)."""
    if sys.platform != "win32":
        raise OSError(f"{name}: Windows only")
    return ctypes.WinDLL(name)


def library_loads(name: str) -> bool:
    try:
        _load_library(name)
        return True
    except OSError:
        return False


def nvidia_driver() -> bool:
    """True when the NVIDIA display driver is installed: ``nvcuda.dll`` is part of the driver, not of any app."""
    return library_loads("nvcuda.dll")


@dataclass(frozen=True)
class CudaInfo:
    """What the NVIDIA driver reports: the CUDA version it supports (13000 = 13.0) and each device's compute
    capability, e.g. ``((12, 0),)`` for one RTX 50xx."""
    driver_version: int
    capabilities: tuple

    def __str__(self) -> str:
        caps = ", ".join(f"sm_{a}{b}" for a, b in self.capabilities) or "no devices"
        return f"CUDA {self.driver_version // 1000}.{self.driver_version % 1000 // 10} driver, {caps}"


_CU_COMPUTE_MAJOR, _CU_COMPUTE_MINOR = 75, 76       # CUdevice_attribute
_cuda_cache: list = []


def _query_cuda(cu) -> CudaInfo:
    def check(status: int, what: str) -> None:
        if status != 0:
            raise OSError(f"{what} returned CUDA error {status}")

    version, count = ctypes.c_int(0), ctypes.c_int(0)
    check(cu.cuInit(0), "cuInit")
    check(cu.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
    check(cu.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    caps = []
    for i in range(count.value):
        device, major, minor = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        check(cu.cuDeviceGet(ctypes.byref(device), i), "cuDeviceGet")
        check(cu.cuDeviceGetAttribute(ctypes.byref(major), _CU_COMPUTE_MAJOR, device), "cuDeviceGetAttribute")
        check(cu.cuDeviceGetAttribute(ctypes.byref(minor), _CU_COMPUTE_MINOR, device), "cuDeviceGetAttribute")
        caps.append((major.value, minor.value))
    return CudaInfo(version.value, tuple(caps))


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


_nvml_cache: list = []             # [(nvml library, device handle)] or [None]


def _nvml_device0():
    """NVML's handle for CUDA device 0 (matched by PCI bus id: the two number devices differently)."""
    cu = _load_library("nvcuda.dll")
    dev, bus = ctypes.c_int(0), ctypes.create_string_buffer(64)
    if cu.cuDeviceGet(ctypes.byref(dev), 0) != 0 or cu.cuDeviceGetPCIBusId(bus, 64, dev) != 0:
        raise OSError("CUDA device 0 has no PCI bus id")
    nvml = _load_library("nvml.dll")
    handle = ctypes.c_void_p()
    if nvml.nvmlInit_v2() != 0 or nvml.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(handle)) != 0:
        raise OSError(f"NVML does not know the device at {bus.value!r}")
    return nvml, handle


def cuda_free_mb() -> int | None:
    """Free memory of CUDA device 0 (the one onnxruntime and llama.cpp use) in MiB, counting every process (a game
    too), as nvidia-smi shows it; None when it cannot be told. Asks NVML, part of the driver: the first call sets it
    up (tens of milliseconds, most of it the driver start every CUDA program pays), later ones take microseconds.
    The driver's own cuMemGetInfo is no use here: on Windows it leaves the memory of other processes out."""
    if cuda_info() is None:
        return None
    if not _nvml_cache:
        try:
            _nvml_cache.append(_nvml_device0())
        except Exception as exc:  # noqa: BLE001 - a question about memory must never break a dictation
            log.info("GPU free memory unknown (%s: %s)", type(exc).__name__, exc)
            _nvml_cache.append(None)
    if _nvml_cache[0] is None:
        return None
    nvml, handle = _nvml_cache[0]
    memory = _NvmlMemory()
    if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
        return None
    return int(memory.free >> 20)


def cuda_info() -> CudaInfo | None:
    """The driver's CUDA version and the compute capability of every NVIDIA device; None without a working
    driver. Asked once per process (``cuInit`` is the start every CUDA program pays anyway, so later users of the
    driver in this process do not pay it again)."""
    if not _cuda_cache:
        try:
            _cuda_cache.append(_query_cuda(_load_library("nvcuda.dll")))
        except Exception as exc:  # noqa: BLE001 - no driver, or one that misbehaves: no CUDA device to speak of
            log.info("no CUDA devices (%s: %s)", type(exc).__name__, exc)
            _cuda_cache.append(None)
    return _cuda_cache[0]


# --------------------------------------------------------------------------- Vulkan (vendor neutral)
class _AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class _InstanceInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_AppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class _DeviceHead(ctypes.Structure):
    """The start of VkPhysicalDeviceProperties; the rest (limits, sparse properties) is padding here."""
    _fields_ = [("apiVersion", ctypes.c_uint32), ("driverVersion", ctypes.c_uint32), ("vendorID", ctypes.c_uint32),
                ("deviceID", ctypes.c_uint32), ("deviceType", ctypes.c_uint32), ("deviceName", ctypes.c_char * 256),
                ("rest", ctypes.c_uint8 * 1024)]           # the real struct is 824 bytes: room to spare


class _Heap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _Memory(ctypes.Structure):
    """VkPhysicalDeviceMemoryProperties."""
    _fields_ = [("memoryTypeCount", ctypes.c_uint32), ("memoryTypes", ctypes.c_uint32 * 64),
                ("memoryHeapCount", ctypes.c_uint32), ("memoryHeaps", _Heap * 16)]


_STYPE_APPLICATION_INFO, _STYPE_INSTANCE_CREATE_INFO = 0, 1
_API_VERSION_1_0 = 1 << 22
_HEAP_DEVICE_LOCAL = 1
_vulkan_cache: list[Gpu] | None = None


def _enumerate_vulkan(vk) -> list[Gpu]:
    vk.vkCreateInstance.argtypes = [ctypes.POINTER(_InstanceInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    vk.vkCreateInstance.restype = ctypes.c_int32
    vk.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                              ctypes.POINTER(ctypes.c_void_p)]
    vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    vk.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DeviceHead)]
    vk.vkGetPhysicalDeviceProperties.restype = None
    vk.vkGetPhysicalDeviceMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Memory)]
    vk.vkGetPhysicalDeviceMemoryProperties.restype = None
    vk.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    vk.vkDestroyInstance.restype = None

    app = _AppInfo(_STYPE_APPLICATION_INFO, None, b"VoxFlow", 0, b"VoxFlow", 0, _API_VERSION_1_0)
    info = _InstanceInfo(_STYPE_INSTANCE_CREATE_INFO, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    if vk.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance)) != 0:
        return []                                      # a loader without any driver behind it
    try:
        count = ctypes.c_uint32(0)
        if vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != 0 or not count.value:
            return []
        devices = (ctypes.c_void_p * count.value)()
        if vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices) != 0:
            return []
        found = []
        for device in devices[:count.value]:
            head, memory = _DeviceHead(), _Memory()
            vk.vkGetPhysicalDeviceProperties(device, ctypes.byref(head))
            vk.vkGetPhysicalDeviceMemoryProperties(device, ctypes.byref(memory))
            local = sum(h.size for h in memory.memoryHeaps[:min(memory.memoryHeapCount, 16)]
                        if h.flags & _HEAP_DEVICE_LOCAL)
            found.append(Gpu(head.deviceName.decode("utf-8", "replace"), head.vendorID,
                             _KINDS.get(head.deviceType, "other"), int(local // (1 << 20))))
        return found
    finally:
        vk.vkDestroyInstance(instance, None)


def vulkan_gpus() -> list[Gpu]:
    """Every device the Vulkan loader reports ([] without a loader, a driver or a device). Asked once per process."""
    global _vulkan_cache
    if _vulkan_cache is None:
        try:
            _vulkan_cache = _enumerate_vulkan(_load_library("vulkan-1.dll"))
        except Exception as exc:  # noqa: BLE001 - no loader, or a driver that misbehaves: there is no Vulkan GPU
            log.info("no Vulkan devices (%s: %s)", type(exc).__name__, exc)
            _vulkan_cache = []
    return list(_vulkan_cache)


def usable_vulkan_gpu(min_vram_mb: int = 0, kinds: tuple = GPU_KINDS) -> Gpu | None:
    """The real GPU (discrete before integrated, then the most memory) with at least ``min_vram_mb`` MiB.
    ``kinds`` narrows what counts, e.g. ``("discrete",)``."""
    fit = [g for g in vulkan_gpus() if g.kind in kinds and g.vram_mb >= min_vram_mb]
    return max(fit, key=lambda g: (g.kind == "discrete", g.vram_mb), default=None)


# --------------------------------------------------------------------------- registry (no driver is loaded)
_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


def _registry_values(path: str, names: tuple) -> dict:
    """``{name: value}`` of the HKLM key ``path`` ({} when the key cannot be read; tests replace this)."""
    import winreg

    found = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            for name in names:
                try:
                    found[name] = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    pass
    except OSError:
        pass
    return found


def _registry_subkeys(path: str) -> list[str]:
    """The names of the subkeys of the HKLM key ``path`` ([] when it cannot be read; tests replace this)."""
    import winreg

    names = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                names.append(winreg.EnumKey(key, i))
    except OSError:
        pass
    return names


def display_adapters() -> list[tuple[str, str]]:
    """``(adapter name, driver version)`` for every display adapter with a driver, sorted. [] off Windows."""
    if sys.platform != "win32":
        return []
    found = set()
    for sub in _registry_subkeys(_DISPLAY_CLASS):
        if not sub.isdigit():                    # "Properties", "Configuration": not an adapter
            continue
        values = _registry_values(f"{_DISPLAY_CLASS}\\{sub}", ("DriverDesc", "DriverVersion"))
        if values.get("DriverDesc"):
            found.add((str(values["DriverDesc"]).strip(), str(values.get("DriverVersion", "")).strip()))
    return sorted(found)


def cpu_name() -> str:
    """The processor's name as Windows reports it ("" when unknown)."""
    if sys.platform != "win32":
        return ""
    return str(_registry_values(_CPU_KEY, ("ProcessorNameString",)).get("ProcessorNameString", "")).strip()
