"""What transcribe.py and voice.py share: each runs one of several backends picked in config, only
offers the ones installed, and describes each backend's options for the settings ui."""
import importlib.util
import io
import wave

import numpy as np


def option(key: str, choices, default, free_text: bool = False, depends_on: str | None = None) -> dict:
    """One setting of a backend, as the settings ui renders it."""
    opt = {"key": key, "choices": choices, "default": default, "free_text": free_text}
    if depends_on:
        # choices and default are dicts keyed by the value of the option this depends on
        opt["depends_on"] = depends_on
    return opt


def installed(backends: dict) -> list[str]:
    """Backends whose libraries are installed, in preference order, found without importing them."""
    return [name for name, b in backends.items() if importlib.util.find_spec(b["module"]) is not None]


def openai_key(config: dict) -> str | None:
    """The openai backends share the llm's openai key rather than holding their own. None falls
    through to OPENAI_API_KEY, which the sdk reads itself."""
    return config.get("llm", {}).get("keys", {}).get("openai") or None


def signature(backend: str, options: dict, api_key: str | None, worker_url: str | None) -> tuple:
    """What a settings change is compared on, to decide whether a model needs reloading."""
    return (backend, tuple(sorted(options.items())), api_key, worker_url)


def fill_defaults(section: dict, backends: dict, defaults: dict):
    """Pick the first installed backend if none is set, and give every backend its default options."""
    if "backend" not in section:
        found = installed(backends)
        section["backend"] = found[0] if found else next(iter(backends))
    for name, options in defaults.items():
        sub = section.setdefault(name, {})
        for key, value in options.items():
            sub.setdefault(key, value)


def resolve(section: dict, backends: dict, defaults: dict, kind: str, log=print) -> tuple[str | None, dict]:
    """The backend to run and its options, falling back to another installed backend when the
    configured one isn't. Backend is None when nothing at all is installed."""
    backend = section["backend"]
    found = installed(backends)
    if backend not in found:
        fallback = found[0] if found else None
        extra = backends.get(backend, {}).get("extra")
        hint = f' (pip install -e ".[{extra}]")' if extra else ""
        log(f"{kind}: backend '{backend}' not installed{hint}, using {fallback}")
        backend = fallback
    if backend is None:
        return None, {}
    return backend, {key: section[backend][key] for key in defaults[backend]}


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """int16 mono audio as a wav file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """A 16 bit mono wav file as int16 audio and its sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16), wf.getframerate()
