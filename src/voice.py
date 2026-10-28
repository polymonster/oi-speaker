"""Text to speech behind one Voice class, so the speaker can talk with piper, Kokoro, Chatterbox
Turbo or OpenAI's cloud api — or have a worker synthesise for it over rpc. Each backend's libraries are
imported only on its own path, so a box only needs the ones it uses installed."""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import requests

import backends

PIPER_DIR = Path("models/piper")
KOKORO_DIR = Path("models/kokoro")
KOKORO_VOICES = KOKORO_DIR / "voices-v1.0.bin"  # every voice's style vectors, shared by the models
CHATTERBOX_DIR = Path("models/chatterbox")  # reference clips to clone a voice from
OPENAI_SAMPLE_RATE = 24000  # the "pcm" response format is 24kHz 16 bit mono
OPENAI_CHUNK_BYTES = 4096
WORKER_TIMEOUT = 30.0
OPENAI_TIMEOUT = 30.0

# in preference order, the first installed is the default. module is what has to be importable for
# the backend to be offered; extra is the pip extra that installs it (pip install -e ".[piper]")
BACKENDS = {
    "piper": {"label": "piper (local)", "module": "piper", "extra": "piper"},
    "kokoro": {"label": "Kokoro (local)", "module": "kokoro_onnx", "extra": "kokoro"},
    "chatterbox": {"label": "Chatterbox Turbo (local)", "module": "chatterbox", "extra": "chatterbox"},
    "openai": {"label": "OpenAI (cloud)", "module": "openai", "extra": None},
}

DEFAULT_OPTIONS = {
    "piper": {"model": "models/piper/en_GB-northern_english_male-medium.onnx", "device": "cpu"},
    # the full model, not the int8 one: on cpu its quantised ops ran ~3.5x slower
    "kokoro": {"model": "models/kokoro/kokoro-v1.0.onnx", "voice": "bm_george", "device": "cpu"},
    # an empty reference speaks with chatterbox's built-in voice
    "chatterbox": {"device": "cpu", "reference": ""},
    "openai": {"model": "gpt-4o-mini-tts", "voice": "ash", "instructions": ""},
}
OPENAI_FALLBACK_MODELS = ("gpt-4o-mini-tts", "tts-1", "tts-1-hd")
OPENAI_MODEL_EXCLUDE = ("realtime", "audio")
# a kokoro voice's first letter is its language, which espeak needs to phonemise the text
KOKORO_LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}
OPENAI_VOICES = ("alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse")


class Voice:
    """A loaded text to speech model. Built whole by create() and replaced rather than mutated, so a
    settings change can load the next one while this one keeps speaking."""

    def __init__(self, backend: str, options: dict, api_key: str | None = None, worker_url: str | None = None, log=print):
        self.backend = "worker" if worker_url else backend
        self.options = dict(options)
        self.worker_url = worker_url
        self.log = log
        self.signature = backends.signature(self.backend, self.options, api_key, worker_url)
        self.sample_rate: int | None = None  # None for a worker, whose rate comes back with each clip
        self._model = None

        if self.backend == "worker":
            pass  # a satellite holds no model, the worker it points at does
        elif self.backend == "piper":
            from piper.voice import PiperVoice
            self._model = PiperVoice.load(self.options["model"], use_cuda=self.options["device"] == "cuda")
            self.sample_rate = self._model.config.sample_rate
        elif self.backend == "kokoro":
            import onnxruntime
            from kokoro_onnx import Kokoro
            # kokoro picks its own providers from what is installed, so the device is pinned here
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.options["device"] == "cuda" else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(self.options["model"], providers=providers)
            self._model = Kokoro.from_session(session, str(KOKORO_VOICES))
            self.sample_rate = 24000  # fixed by the model, create() hands it back with every clip
        elif self.backend == "chatterbox":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.options["device"])
            if self.options["reference"]:
                # chatterbox's loudness normalise hands back float64, which its tokenizer then rejects
                normalise = self._model.norm_loudness
                self._model.norm_loudness = lambda wav, sr, **kw: normalise(wav, sr, **kw).astype(np.float32)
                # conditioning is kept on the model, so it is done once here rather than per sentence
                self._model.prepare_conditionals(self.options["reference"])
            self.sample_rate = self._model.sr
        elif self.backend == "openai":
            import openai
            self._model = openai.OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)
            self.sample_rate = OPENAI_SAMPLE_RATE
        else:
            raise ValueError(f"unknown voice backend '{backend}'")

    def describe(self) -> str:
        if self.backend == "worker":
            return f"worker {self.worker_url}"
        return " ".join([self.backend, *(str(v) for v in self.options.values() if v)])

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Synthesise `text` as int16 chunks at self.sample_rate, yielded as they are ready."""
        if self.backend == "piper":
            for chunk in self._model.synthesize(text):
                yield np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        elif self.backend == "kokoro":
            name = self.options["voice"]
            wav, _ = self._model.create(text, voice=name, lang=KOKORO_LANGS.get(name[:1], "en-us"))
            yield (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        elif self.backend == "chatterbox":
            wav = self._model.generate(text).squeeze(0).numpy()
            yield (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        elif self.backend == "openai":
            request = {"model": self.options["model"], "voice": self.options["voice"], "input": text, "response_format": "pcm"}
            if self.options["instructions"]:
                request["instructions"] = self.options["instructions"]  # tone and accent, gpt-4o-mini-tts only
            with self._model.audio.speech.with_streaming_response.create(**request) as resp:
                left = b""  # a chunk can split a sample across its end
                for data in resp.iter_bytes(OPENAI_CHUNK_BYTES):
                    data = left + data
                    whole = len(data) - len(data) % 2
                    left = data[whole:]
                    if whole:
                        yield np.frombuffer(data[:whole], dtype=np.int16)
        else:
            raise ValueError(f"{self.backend} voice cannot stream")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesise `text` whole, as int16 audio and its sample rate. Empty audio on a failure,
        which is logged — a missed sentence must not take the turn down."""
        if self.backend == "worker":
            self.log(f"delegating speak to worker: {self.worker_url} — {text[:60]}")
            try:
                resp = requests.post(f"{self.worker_url}/rpc/speak", json={"text": text}, timeout=WORKER_TIMEOUT)
                resp.raise_for_status()
                return backends.read_wav(resp.content)
            except Exception as e:
                self.log(f"worker speak error: {e}")
                return np.array([], dtype=np.int16), 16000
        try:
            chunks = list(self.stream(text))
        except Exception as e:
            self.log(f"{self.backend} speak error: {e}")
            chunks = []
        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
        return audio, self.sample_rate


def _openai_models(api_key: str | None) -> list[str]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=5.0)
        models = sorted(
            m.id for m in client.models.list()
            if "tts" in m.id and not any(w in m.id for w in OPENAI_MODEL_EXCLUDE)
        )
        return models or list(OPENAI_FALLBACK_MODELS)
    except Exception:
        return list(OPENAI_FALLBACK_MODELS)  # no key yet, or offline


def backend_options(backend: str, api_key: str | None = None) -> list[dict]:
    """The options a backend takes, with the choices this machine actually supports, for the
    settings ui to offer. Imports the backend's libraries, so only call it for installed ones."""
    if backend == "piper":
        import onnxruntime
        models = sorted(p.as_posix() for p in PIPER_DIR.glob("*.onnx"))
        devices = ["cpu"] + (["cuda"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else [])
        return [
            backends.option("model", models, DEFAULT_OPTIONS["piper"]["model"], free_text=True),
            backends.option("device", devices, devices[-1]),
        ]
    if backend == "kokoro":
        import onnxruntime
        models = sorted(p.as_posix() for p in KOKORO_DIR.glob("*.onnx"))
        try:
            with np.load(KOKORO_VOICES) as voices:
                names = sorted(voices.files)
        except OSError:
            names = [DEFAULT_OPTIONS["kokoro"]["voice"]]  # not downloaded yet
        devices = ["cpu"] + (["cuda"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else [])
        return [
            backends.option("model", models, DEFAULT_OPTIONS["kokoro"]["model"], free_text=True),
            backends.option("voice", names, DEFAULT_OPTIONS["kokoro"]["voice"]),
            backends.option("device", devices, devices[-1]),
        ]
    if backend == "chatterbox":
        import torch
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
        references = [""] + sorted(p.as_posix() for p in CHATTERBOX_DIR.glob("*.wav"))
        return [
            backends.option("device", devices, devices[-1]),
            backends.option("reference", references, "", free_text=True),
        ]
    if backend == "openai":
        return [
            backends.option("model", _openai_models(api_key), DEFAULT_OPTIONS["openai"]["model"], free_text=True),
            backends.option("voice", list(OPENAI_VOICES), DEFAULT_OPTIONS["openai"]["voice"]),
            backends.option("instructions", [], "", free_text=True),
        ]
    return []


def migrate_config(config: dict) -> dict:
    """Move the piper model from before backend choice ([voice] model) under [voice.piper], and fill
    in defaults. Idempotent, so it runs on every load rather than rewriting anyone's config file.
    Returns `config`."""
    section = config.setdefault("voice", {})
    if isinstance(section.get("model"), str):
        section.setdefault("piper", {}).setdefault("model", section.pop("model"))
    backends.fill_defaults(section, BACKENDS, DEFAULT_OPTIONS)
    return config


def settings_from_config(config: dict, log=print) -> tuple[str | None, dict]:
    return backends.resolve(migrate_config(config)["voice"], BACKENDS, DEFAULT_OPTIONS, "voice", log)


def create(config: dict, worker_url: str | None = None, log=print) -> Voice | None:
    """Load the voice config.toml asks for, or a worker proxy under --worker-ip. None when no
    backend is installed — the speaker still works, it just cannot talk back."""
    if worker_url:
        model = Voice("worker", {}, worker_url=worker_url, log=log)
    else:
        backend, options = settings_from_config(config, log)
        if backend is None:
            log('voice: no backend installed — pip install -e ".[piper]", ".[kokoro]" or ".[chatterbox]"')
            return None
        model = Voice(backend, options, api_key=backends.openai_key(config) if backend == "openai" else None, log=log)
    log(f"voice: {model.describe()}")
    return model
