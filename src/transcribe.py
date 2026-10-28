"""Speech to text behind one Transcribe class, so the speaker can run faster-whisper, parakeet or
OpenAI's cloud api — or hand the audio to a worker over rpc. Each backend's libraries are imported
only on its own path, so a box only needs the ones it uses installed."""
import numpy as np
import requests

import backends

SAMPLE_RATE = 16000
NO_SPEECH_PROB_LIMIT = 0.6  # whisper segments above this are hallucinations on near-silence
WORKER_TIMEOUT = 30.0
OPENAI_TIMEOUT = 30.0

# in preference order, the first installed is the default. module is what has to be importable for
# the backend to be offered; extra is the pip extra that installs it (pip install -e ".[whisper]")
BACKENDS = {
    "whisper": {"label": "faster-whisper (local)", "module": "faster_whisper", "extra": "whisper"},
    "parakeet": {"label": "parakeet (local, onnx)", "module": "onnx_asr", "extra": "parakeet"},
    "openai": {"label": "OpenAI (cloud)", "module": "openai", "extra": None},
}

DEFAULT_OPTIONS = {
    "whisper": {"model": "small", "device": "cpu", "compute_type": "int8"},
    "parakeet": {"model": "nemo-parakeet-tdt-0.6b-v2", "device": "cpu", "quantization": "int8"},
    "openai": {"model": "gpt-4o-mini-transcribe"},
}
WHISPER_DEFAULT_COMPUTE = {"cpu": "int8", "cuda": "float16"}
PARAKEET_QUANTIZATIONS = ("int8", "none")
OPENAI_FALLBACK_MODELS = ("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1")
OPENAI_MODEL_WORDS = ("transcribe", "whisper")
OPENAI_MODEL_EXCLUDE = ("diarize", "realtime", "live")  # streaming or different response shapes, not one-shot text


class Transcribe:
    """A loaded speech to text model. Built whole by create() and replaced rather than mutated, so a
    settings change can load the next one while this one keeps serving."""

    def __init__(self, backend: str, options: dict, api_key: str | None = None, worker_url: str | None = None, log=print):
        self.backend = "worker" if worker_url else backend
        self.options = dict(options)
        self.worker_url = worker_url
        self.log = log
        self.signature = backends.signature(self.backend, self.options, api_key, worker_url)
        self._model = None

        if self.backend == "worker":
            pass  # a satellite holds no model, the worker it points at does
        elif self.backend == "whisper":
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.options["model"],
                device=self.options["device"],
                compute_type=self.options["compute_type"],
            )
        elif self.backend == "parakeet":
            import onnx_asr
            providers = ["CPUExecutionProvider"]
            if self.options["device"] == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            quantization = self.options["quantization"]
            self._model = onnx_asr.load_model(
                self.options["model"],
                quantization=None if quantization == "none" else quantization,
                providers=providers,
            )
        elif self.backend == "openai":
            import openai
            self._model = openai.OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)
        else:
            raise ValueError(f"unknown transcribe backend '{backend}'")

    def describe(self) -> str:
        if self.backend == "worker":
            return f"worker {self.worker_url}"
        return " ".join([self.backend, *(str(v) for v in self.options.values())])

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 16kHz mono audio to English text."""
        if self.backend == "worker":
            self.log(f"delegating transcribe to worker: {self.worker_url}")
            try:
                resp = requests.post(
                    f"{self.worker_url}/rpc/transcribe",
                    files={"audio": ("audio.wav", backends.wav_bytes(audio, SAMPLE_RATE), "audio/wav")},
                    timeout=WORKER_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json().get("text", "")
            except Exception as e:
                self.log(f"worker transcribe error: {e}")
                return ""

        if self.backend == "whisper":
            segments, _ = self._model.transcribe(
                audio.astype(np.float32) / 32768.0,
                language="en",
                beam_size=1,
                # whisper invents filler ("yeah", "bye", "thanks for watching") on near-silence, and by default
                # feeds each turn's text forward as a prompt so one invention seeds the next
                condition_on_previous_text=False,
            )
            kept = []
            for segment in segments:
                if segment.no_speech_prob > NO_SPEECH_PROB_LIMIT:
                    self.log(f"dropping hallucinated segment (no_speech={segment.no_speech_prob:.2f}): {segment.text.strip()!r}")
                    continue
                kept.append(segment.text)
            return " ".join(kept)

        if self.backend == "parakeet":
            return self._model.recognize(audio.astype(np.float32) / 32768.0, sample_rate=SAMPLE_RATE)

        if self.backend == "openai":
            try:
                result = self._model.audio.transcriptions.create(
                    model=self.options["model"],
                    file=("audio.wav", backends.wav_bytes(audio, SAMPLE_RATE), "audio/wav"),
                    language="en",
                )
                return result.text
            except Exception as e:
                self.log(f"openai transcribe error: {e}")
                return ""

        return ""


def _openai_models(api_key: str | None) -> list[str]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=5.0)
        models = sorted(
            m.id for m in client.models.list()
            if any(w in m.id for w in OPENAI_MODEL_WORDS) and not any(w in m.id for w in OPENAI_MODEL_EXCLUDE)
        )
        return models or list(OPENAI_FALLBACK_MODELS)
    except Exception:
        return list(OPENAI_FALLBACK_MODELS)  # no key yet, or offline


def backend_options(backend: str, api_key: str | None = None) -> list[dict]:
    """The options a backend takes, with the choices this machine actually supports, for the
    settings ui to offer. Imports the backend's libraries, so only call it for installed ones."""
    if backend == "whisper":
        import ctranslate2
        import faster_whisper
        devices = ["cpu"] + (["cuda"] if ctranslate2.get_cuda_device_count() > 0 else [])
        computes = {d: sorted(ctranslate2.get_supported_compute_types(d)) for d in devices}
        return [
            backends.option("model", list(faster_whisper.available_models()), DEFAULT_OPTIONS["whisper"]["model"], free_text=True),
            backends.option("device", devices, devices[-1]),
            backends.option("compute_type", computes, {d: WHISPER_DEFAULT_COMPUTE[d] for d in devices}, depends_on="device"),
        ]
    if backend == "parakeet":
        import typing
        import onnx_asr.loader
        import onnxruntime
        models = [m for m in typing.get_args(onnx_asr.loader.AsrNames) if "parakeet" in m]
        devices = ["cpu"] + (["cuda"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else [])
        return [
            backends.option("model", models, DEFAULT_OPTIONS["parakeet"]["model"]),
            backends.option("device", devices, devices[-1]),
            backends.option("quantization", list(PARAKEET_QUANTIZATIONS), DEFAULT_OPTIONS["parakeet"]["quantization"]),
        ]
    if backend == "openai":
        return [backends.option("model", _openai_models(api_key), DEFAULT_OPTIONS["openai"]["model"], free_text=True)]
    return []


def migrate_config(config: dict) -> dict:
    """Move the whisper settings from before backend choice ([inference] whisper_model,
    whisper_compute, device) under [transcribe.whisper], and fill in defaults. Idempotent, so it
    runs on every load rather than rewriting anyone's config file. Returns `config`."""
    inference = config.get("inference", {})
    section = config.setdefault("transcribe", {})
    whisper = section.setdefault("whisper", {})
    for old, new in (("whisper_model", "model"), ("whisper_compute", "compute_type"), ("device", "device")):
        if old in inference:
            whisper.setdefault(new, inference.pop(old))
    if "device" in whisper and "compute_type" not in whisper:
        whisper["compute_type"] = WHISPER_DEFAULT_COMPUTE.get(whisper["device"], "int8")
    backends.fill_defaults(section, BACKENDS, DEFAULT_OPTIONS)
    return config


def settings_from_config(config: dict, log=print) -> tuple[str | None, dict]:
    return backends.resolve(migrate_config(config)["transcribe"], BACKENDS, DEFAULT_OPTIONS, "transcribe", log)


def create(config: dict, worker_url: str | None = None, log=print) -> Transcribe | None:
    """Load the transcribe model config.toml asks for, or a worker proxy under --worker-ip. None
    when no backend is installed — text chat still works, voice requests are dropped."""
    if worker_url:
        model = Transcribe("worker", {}, worker_url=worker_url, log=log)
    else:
        backend, options = settings_from_config(config, log)
        if backend is None:
            log('transcribe: no backend installed — pip install -e ".[whisper]" or ".[parakeet]"')
            return None
        model = Transcribe(backend, options, api_key=backends.openai_key(config) if backend == "openai" else None, log=log)
    log(f"transcribe: {model.describe()}")
    return model
