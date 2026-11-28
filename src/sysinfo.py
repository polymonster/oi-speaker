"""A small snapshot of how the box is doing, for the web ui: cpu, memory, temperature, and on a
Raspberry Pi whether it is throttling or short of power; on a box with an nvidia card, the gpu.

cpu and memory come from psutil on every platform. Temperature is what the platform will tell an
unprivileged process: the kernel's thermal zones on linux (the Pi's cpu_thermal among them).
Windows has no reliable source for cpu temperature without admin rights or a driver, and macOS
none at all, so there it is left out rather than guessed."""

import shutil
import subprocess
import time
from pathlib import Path

import psutil

# a Pi's sensor first, then the usual names on desktop linux
_TEMP_SENSORS = ("cpu_thermal", "coretemp", "k10temp", "zenpower", "acpitz")

# vcgencmd get_throttled bits: the low ones are happening now, the 16+ ones have happened since boot
_THROTTLE_FLAGS = {
    0: "under-voltage",
    1: "freq capped",
    2: "throttled",
    3: "soft temp limit",
}

_REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _version() -> dict:
    """What this process is running, read once at start: a pull without a restart isn't running yet.
    The update tool checks out tags, which leaves no branch, so the tag stands in for one."""
    branch = _git("symbolic-ref", "--quiet", "--short", "HEAD")
    if branch is None:
        branch = _git("describe", "--tags", "--exact-match") or "detached"
    return {
        "branch": branch,
        "sha": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


VERSION = _version()

_cache: dict[str, tuple[float, object]] = {}
psutil.cpu_percent(None)  # the first call has nothing to compare against and returns 0


def _cached(key: str, ttl: float, fn):
    """Keep slow lookups (a subprocess each) from running on every poll of every open browser tab."""
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        value = fn()
    except Exception:
        value = None
    _cache[key] = (now, value)
    return value


def _cpu_temp() -> float | None:
    if not hasattr(psutil, "sensors_temperatures"):  # windows, macos
        return None
    sensors = psutil.sensors_temperatures()
    for name in _TEMP_SENSORS:
        if sensors.get(name):
            return round(max(s.current for s in sensors[name]), 1)
    for readings in sensors.values():
        if readings:
            return round(readings[0].current, 1)
    return None


def _pi_throttled() -> dict | None:
    if not shutil.which("vcgencmd"):
        return None
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
    value = int(out.strip().split("=")[1], 16)  # "throttled=0x50000"
    return {
        "now": [name for bit, name in _THROTTLE_FLAGS.items() if value & (1 << bit)],
        "since_boot": [name for bit, name in _THROTTLE_FLAGS.items() if value & (1 << (bit + 16))],
    }


def _nvidia_gpu() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3).stdout
    util, temp, used, total = (float(v) for v in out.splitlines()[0].split(","))
    return {"percent": util, "temp_c": temp, "mem_percent": round(used / total * 100, 1) if total else None}


def snapshot() -> dict:
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(None),  # since the last call, ie. over the poll interval
        "cpu_count": psutil.cpu_count(),
        "mem_percent": mem.percent,
        "temp_c": _cached("temp", 2, _cpu_temp),
        "throttled": _cached("throttled", 5, _pi_throttled),
        "gpu": _cached("gpu", 5, _nvidia_gpu),
        "uptime_s": int(time.time() - psutil.boot_time()),
        "version": VERSION,
    }
