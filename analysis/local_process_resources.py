"""Read-only Windows process/system memory and NVIDIA resource samples."""
import ctypes
import os
import subprocess


def host_memory():
    if os.name != "nt":
        return {}
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"total_physical_bytes": value.total, "available_physical_bytes": value.available}


def process_memory(pid):
    if os.name != "nt":
        return {"pid": pid}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        return {"pid": pid, "unavailable": True}
    try:
        value = Counters()
        value.cb = ctypes.sizeof(value)
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not get_memory(handle, ctypes.byref(value), value.cb):
            return {"pid": pid, "unavailable": True}
        return {"pid": pid, "working_set_bytes": value.rss, "peak_working_set_bytes": value.peak_rss}
    finally:
        kernel.CloseHandle(handle)


def sample(pids=()):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    used, total, utilization = [int(v.strip()) for v in output.splitlines()[0].split(",")]
    processes = [process_memory(pid) for pid in pids]
    return {**host_memory(), "gpu_used_mib": used, "gpu_total_mib": total, "gpu_utilization_percent": utilization,
            "process_count": len(processes), "processes": processes,
            "process_working_set_total_bytes": sum(p.get("working_set_bytes", 0) for p in processes)}
