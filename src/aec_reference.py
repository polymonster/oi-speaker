"""Feed the speaker's own playback to the XVF3800 so its echo canceller can remove it from the mic.

The XVF3800's AEC uses whatever the host plays to it over USB as the far-end reference. Our audio
goes out through another device (the HiFiBerry amp), so the array never sees it and hears music,
radio and TTS as if someone in the room were talking. This builds a PulseAudio combine sink (works
on pipewire-pulse too) that plays to both the real output and the XVF3800 at once, keeping the two
clocks in step, and makes it the default sink so mpv and a "pulse"/"default" sounddevice output
follow it. The array's own 3.5mm jack also gets the audio; leave it unplugged.
"""

import shutil
import subprocess
import sys

SINK_NAME = "oi_aec"
MODULE = "module-combine-sink"
# how the array's playback sink is named: pulse uses the usb product string, pipewire the vendor
XVF_MATCH = ("xvf3800", "respeaker", "seeed", "2886")


def _pactl(*args: str) -> str:
    return subprocess.run(["pactl", *args], check=True, capture_output=True, text=True, timeout=10).stdout.strip()


def _sinks() -> list[str]:
    # "id\tname\tdriver\tformat\tstate"
    return [line.split("\t")[1] for line in _pactl("list", "short", "sinks").splitlines() if "\t" in line]


def _is_xvf(sink: str) -> bool:
    return any(m in sink.lower() for m in XVF_MATCH)


def _our_modules() -> list[tuple[str, str]]:
    """(id, args) of every combine sink we loaded, from this run or an earlier one."""
    found = []
    for line in _pactl("list", "short", "modules").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == MODULE and f"sink_name={SINK_NAME}" in parts[2]:
            found.append((parts[0], parts[2]))
    return found


def _slaves(args: str) -> list[str]:
    for arg in args.split():
        if arg.startswith("slaves="):
            return arg[len("slaves="):].split(",")
    return []


def _pick_output(sinks: list[str], output_sink: str, previous: list[str]) -> str | None:
    """The sink people actually hear: the configured one, else the default (or what was default
    before we took over), else the first that is neither ours nor the array."""
    candidates = [s for s in sinks if s != SINK_NAME and not _is_xvf(s)]
    if output_sink:
        return next((s for s in candidates if output_sink.lower() in s.lower()), None)
    default = _pactl("get-default-sink")
    if default in candidates:
        return default
    for s in previous:
        if s in candidates:
            return s
    return candidates[0] if candidates else None


def _remove(log):
    modules = _our_modules()
    if not modules:
        return
    restore = next((s for _, args in modules for s in _slaves(args) if not _is_xvf(s)), None)
    if _pactl("get-default-sink") == SINK_NAME and restore:
        _pactl("set-default-sink", restore)
    for module_id, _ in modules:
        _pactl("unload-module", module_id)
    log(f"aec reference: off, default sink back to {restore}")


def apply(enabled: bool, output_sink: str = "", log=print):
    """Set up (or tear down) the combine sink. Safe to call repeatedly: an existing sink with the
    same outputs is left alone, so a settings save doesn't interrupt what is playing."""
    if sys.platform != "linux" or not shutil.which("pactl"):
        if enabled:
            log("aec reference: needs linux with pulseaudio or pipewire-pulse (pactl), skipping")
        return
    try:
        if not enabled:
            _remove(log)
            return

        sinks = _sinks()
        xvf = next((s for s in sinks if _is_xvf(s)), None)
        if xvf is None:
            log("aec reference: no XVF3800 playback sink found, is the array plugged in? skipping")
            return
        modules = _our_modules()
        output = _pick_output(sinks, output_sink, [s for _, args in modules for s in _slaves(args)])
        if output is None:
            log(f"aec reference: no output sink found{f' matching {output_sink!r}' if output_sink else ''}, skipping")
            return

        wanted = [output, xvf]
        if len(modules) == 1 and _slaves(modules[0][1]) == wanted:
            if _pactl("get-default-sink") != SINK_NAME:
                _pactl("set-default-sink", SINK_NAME)
            return
        for module_id, _ in modules:
            _pactl("unload-module", module_id)
        _pactl("load-module", MODULE, f"sink_name={SINK_NAME}", f"slaves={','.join(wanted)}")
        _pactl("set-default-sink", SINK_NAME)
        log(f"aec reference: on, playing to {output} and {xvf} (default sink {SINK_NAME})")
    except (subprocess.SubprocessError, OSError) as e:
        detail = getattr(e, "stderr", "") or e
        log(f"aec reference: pactl failed: {str(detail).strip()}")
