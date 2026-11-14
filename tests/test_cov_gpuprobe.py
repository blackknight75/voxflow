"""``voxflow.gpuprobe``: what the machine probe reports, with the drivers faked. No GPU, no driver, no DLL is used."""
from __future__ import annotations

import ctypes

import pytest

from voxflow import gpuprobe as G

NVIDIA, AMD, INTEL = 0x10DE, 0x1002, 0x8086
MIB = 1 << 20


class Fn:
    """A stand-in for a ctypes function: accepts argtypes / restype and calls ``impl``."""

    def __init__(self, impl):
        self.impl = impl

    def __call__(self, *args):
        return self.impl(*args)


class FakeVulkan:
    """vulkan-1.dll with scripted devices: (name, vendor id, VkPhysicalDeviceType, [(heap MiB, device local)])."""

    def __init__(self, devices=(), create=0, count_result=0, list_result=0):
        self.devices, self.destroyed = list(devices), 0
        self.app_name = None

        def create_instance(info, allocator, instance):
            self.app_name = info._obj.pApplicationInfo.contents.pApplicationName
            instance._obj.value = 0xABC
            return create

        def enumerate_devices(instance, count, out):
            assert instance.value == 0xABC
            if out is None:
                count._obj.value = len(self.devices)
                return count_result
            for i in range(len(self.devices)):
                out[i] = i + 1
            return list_result

        def properties(device, head):
            name, vendor, kind, _heaps = self.devices[device - 1]
            head._obj.vendorID, head._obj.deviceType, head._obj.deviceName = vendor, kind, name.encode()

        def memory(device, mem):
            heaps = self.devices[device - 1][3]
            mem._obj.memoryHeapCount = len(heaps)
            for i, (mib, local) in enumerate(heaps):
                mem._obj.memoryHeaps[i].size, mem._obj.memoryHeaps[i].flags = mib * MIB, 1 if local else 0

        def destroy(instance, allocator):
            self.destroyed += 1

        self.vkCreateInstance, self.vkEnumeratePhysicalDevices = Fn(create_instance), Fn(enumerate_devices)
        self.vkGetPhysicalDeviceProperties, self.vkGetPhysicalDeviceMemoryProperties = Fn(properties), Fn(memory)
        self.vkDestroyInstance = Fn(destroy)


RTX = ("NVIDIA GeForce RTX 5080", NVIDIA, 2, [(15977, True), (32000, False)])
RADEON_IGPU = ("AMD Radeon(TM) Graphics", AMD, 1, [(10791, True)])
ARC = ("Intel(R) Arc(TM) A380", INTEL, 2, [(6000, True)])
SOFTWARE = ("llvmpipe", 0x10005, 4, [(64000, True)])
ODD = ("Mystery", 0x1234, 9, [])


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(G, "_vulkan_cache", None)


def use(monkeypatch, vk):
    loads = []

    def load(name):
        loads.append(name)
        if isinstance(vk, Exception):
            raise vk
        return vk

    monkeypatch.setattr(G, "_load_library", load)
    return loads


# =========================================================================== loading a driver library
def test_a_library_is_loaded_by_name_on_windows_only(monkeypatch):
    asked = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: asked.append(name) or "handle", raising=False)
    monkeypatch.setattr(G.sys, "platform", "win32")
    assert G._load_library("nvcuda.dll") == "handle" and asked == ["nvcuda.dll"]
    monkeypatch.setattr(G.sys, "platform", "darwin")
    with pytest.raises(OSError, match="Windows only"):
        G._load_library("nvcuda.dll")
    assert asked == ["nvcuda.dll"]


def test_the_nvidia_driver_is_present_when_its_cuda_library_loads(monkeypatch):
    loads = use(monkeypatch, object())
    assert G.nvidia_driver() is True and loads == ["nvcuda.dll"]
    use(monkeypatch, OSError("Could not find module 'nvcuda.dll'"))
    assert G.nvidia_driver() is False and G.library_loads("anything.dll") is False


# =========================================================================== Vulkan devices
def test_vulkan_reports_every_device_with_vendor_kind_and_its_own_memory(monkeypatch):
    vk = FakeVulkan([RTX, RADEON_IGPU, SOFTWARE, ODD])
    loads = use(monkeypatch, vk)
    gpus = G.vulkan_gpus()
    assert loads == ["vulkan-1.dll"] and vk.app_name == b"VoxFlow" and vk.destroyed == 1
    assert [(g.name, g.vendor, g.kind, g.vram_mb) for g in gpus] == [
        ("NVIDIA GeForce RTX 5080", "NVIDIA", "discrete", 15977),      # the heap that is not device-local is left out
        ("AMD Radeon(TM) Graphics", "AMD", "integrated", 10791),
        ("llvmpipe", "vendor 0x10005", "cpu", 64000),
        ("Mystery", "vendor 0x1234", "other", 0),
    ]
    assert str(gpus[1]) == "AMD Radeon(TM) Graphics (AMD, integrated, 10791 MiB)"
    # asked once per process, and the caller gets its own list
    gpus.clear()
    assert len(G.vulkan_gpus()) == 4 and loads == ["vulkan-1.dll"]


@pytest.mark.parametrize("vk,destroyed", [
    (FakeVulkan([RTX], create=-9), 0),       # VK_ERROR_INCOMPATIBLE_DRIVER: a loader with no driver behind it
    (FakeVulkan([RTX], count_result=-3), 1),
    (FakeVulkan([]), 1),                     # a driver, but no device
    (FakeVulkan([RTX], list_result=-3), 1),
])
def test_vulkan_without_a_usable_driver_or_device_reports_nothing(monkeypatch, vk, destroyed):
    use(monkeypatch, vk)
    assert G.vulkan_gpus() == []
    assert vk.destroyed == destroyed, "an instance that was created is always destroyed"


def test_no_vulkan_loader_or_a_misbehaving_driver_means_no_vulkan_gpu(monkeypatch, caplog):
    use(monkeypatch, OSError("Could not find module 'vulkan-1.dll'"))
    with caplog.at_level("INFO"):
        assert G.vulkan_gpus() == [] and G.usable_vulkan_gpu() is None
    assert "no Vulkan devices (OSError" in caplog.text

    class Broken:
        def __getattr__(self, name):
            raise AttributeError(f"function '{name}' not found")

    monkeypatch.setattr(G, "_vulkan_cache", None)
    use(monkeypatch, Broken())
    assert G.vulkan_gpus() == []


def test_the_usable_gpu_is_a_real_one_with_enough_memory_discrete_first(monkeypatch):
    use(monkeypatch, FakeVulkan([SOFTWARE, RADEON_IGPU, ARC, RTX]))
    assert G.usable_vulkan_gpu().name == "NVIDIA GeForce RTX 5080"
    assert G.usable_vulkan_gpu(20000) is None, "nothing has that much memory; the software device never counts"
    assert G.usable_vulkan_gpu(8000, ("integrated",)).name == "AMD Radeon(TM) Graphics"
    monkeypatch.setattr(G, "_vulkan_cache", [g for g in G.vulkan_gpus() if g.vendor != "NVIDIA"])
    assert G.usable_vulkan_gpu(3500).name == "Intel(R) Arc(TM) A380", "a discrete card beats a larger integrated one"
    assert G.usable_vulkan_gpu(3500, ("discrete",)).vendor == "Intel"
    assert G.usable_vulkan_gpu(8000).kind == "integrated"
    assert G.usable_vulkan_gpu(8000, ("discrete",)) is None


def test_the_structures_have_the_layout_vulkan_documents():
    assert ctypes.sizeof(G._Memory) == 520 and G._Memory.memoryHeapCount.offset == 260
    assert G._DeviceHead.deviceName.offset == 20 and ctypes.sizeof(G._DeviceHead) >= 824
    assert ctypes.sizeof(G._InstanceInfo) == 64 and ctypes.sizeof(G._AppInfo) == 48
