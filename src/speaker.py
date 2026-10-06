import subprocess
import sys
import threading
import queue
import re
from collections import deque
import logging as _logging
import mpv
import openwakeword
import sounddevice as dev
import numpy as np
import time
import math
import random
import faster_whisper
import tomllib
import json
import requests
import wave
import uvicorn
import os
import urllib.request
import urllib.error

# litellm fetches its model registry from github on import unless told otherwise — a network round
# trip on every start, and a hang on a box with no route out. the bundled copy is what we want
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import litellm
import openai  # litellm raises openai's exception types, whichever vendor is behind the call


from pathlib import Path
from piper.voice import PiperVoice
from openwakeword.model import Model
from faster_whisper import WhisperModel
from dataclasses import dataclass, field
from enum import Enum, auto

TARGET_SAMPLE_RATE = 16000 # 16khz
WAKE_CHUNK = 1280 # 80ms at 16khz
VAD_CHUNK = 512 # 32ms at 16kHz (Silero VAD frame size)
VAD_ONSET_FRAMES = 3 # consecutive speech frames required before onset is accepted (~96ms)
VAD_PREROLL_FRAMES = 10 # frames kept from before onset so the first phoneme survives (~320ms)
VAD_MIN_SPEECH = 0.35 # a recording holding less speech than this is noise, not a request
NO_SPEECH_PROB_LIMIT = 0.6 # whisper segments above this are hallucinations on near-silence
ONSET_TIMEOUT = 8.0 # seconds after a wake to start speaking before the turn is dropped

ACK_SOUND_DIR = "sounds/aknowledge" # one played at random as a request goes to the llm
NUDGE_SOUND_DIR = "sounds/nudge" # one played at random when half the onset budget passes in silence
SOUND_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

LEVEL_BUFFER_FRAMES = 2000 # ~64s at 31Hz (recording), ~160s at 12.5Hz (wake listen)
MARKER_BUFFER_LEN = 200 # pipeline events kept alongside the level frames

HISTORY_DIR = Path(".history")
CHAT_HISTORY_PATH = HISTORY_DIR / "chat.jsonl"
PLAY_HISTORY_PATH = HISTORY_DIR / "plays.jsonl"
HISTORY_LOAD_LIMIT = 30  # messages loaded into context on startup
API_HISTORY_LIMIT = 20  # api-format messages kept in the rolling context window
MAX_TOOL_ROUNDS = 8  # backstop on the agent loop so no stop_reason can spin it forever
TERMINAL_TOOLS = ("play_url", "stop", "follow_on")  # tools that end the turn once they succeed
LLM_MAX_TOKENS = 4096  # thinking shares this budget, so it needs more room than the reply
WEB_SEARCH_RESULTS = 5  # results returned by the local web_search fallback
VOLUME_STEPS = 16  # notches from silent to full for change_volume, as on iOS

# Vendors offered in the web ui. Models are litellm "provider/model" strings. native_search means the
# vendor's own search can run alongside our function tools — only Anthropic's can: Gemini drops
# googleSearch when function declarations are present, and OpenAI chat models reject
# web_search_options outright. Everyone else gets the local web_search tool instead.
PROVIDERS = {
    "anthropic": {
        "label": "Anthropic",
        "default_model": "anthropic/claude-opus-5",
        "default_followup": "anthropic/claude-sonnet-5",
        "needs_key": True,
        "needs_api_base": False,
        "native_search": True,
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "openai/gpt-5.5",
        "default_followup": "openai/gpt-5.4-mini",
        "needs_key": True,
        "needs_api_base": False,
        "native_search": False,
    },
    "gemini": {
        "label": "Gemini",
        "default_model": "gemini/gemini-3.1-pro-preview",
        "default_followup": "gemini/gemini-3.5-flash",
        "needs_key": True,
        "needs_api_base": False,
        "native_search": False,
    },
    "openrouter": {
        "label": "OpenRouter",
        "default_model": "openrouter/anthropic/claude-opus-5",
        "default_followup": "openrouter/anthropic/claude-sonnet-5",
        "needs_key": True,
        "needs_api_base": False,
        "native_search": False,
    },
    "ollama": {
        "label": "Ollama (local)",
        "default_model": "ollama_chat/qwen3",
        "default_followup": "ollama_chat/qwen3",
        "default_api_base": "http://localhost:11434",
        "needs_key": False,
        "needs_api_base": True,
        "native_search": False,
    },
}
DEFAULT_PROVIDER = "anthropic"
LLM_EFFORTS = ("low", "medium", "high")
# chat-mode registry entries that cannot hold a spoken conversation with tools
LLM_MODEL_EXCLUDE = ("audio", "realtime", "tts", "image", "live", "transcribe", "search", "robotics", "computer-use")

# YouTube rejects player clients unpredictably — one that worked an hour ago starts returning 403 —
# so resolution tries each in turn. "" means yt-dlp's own default client chain.
YTDLP_CLIENTS = ("tv_embedded", "", "android_vr")
YTDLP_TIMEOUT = 60.0
PROBE_BYTES = 2048  # a resolved URL can still 403 on first read; find out before mpv does
# the probe is not enough — youtube serves the first small range and 403s the reads after it. so
# play_url holds the turn until this much audio has actually played, or mpv gives up
PLAYBACK_CONFIRM_SECONDS = 2.0
PLAYBACK_CONFIRM_TIMEOUT = 20.0  # slow buffering is not a failure — past this, assume it is playing
MPV_END_FILE_ERROR = 4  # mpv_end_file_reason MPV_END_FILE_REASON_ERROR
_HTTP_ERROR = re.compile(r"HTTP error (\d{3})|(\d{3}) Forbidden")

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# LED ring colours per speaker state, 0xRRGGBB
RING_COLOR_WAKE = 0xFF8800  # orange
RING_COLOR_RECORDING = 0x40C0C0  # soft cyan
RING_COLOR_TRANSCRIBING = 0xB57EDC  # mauve
RING_COLOR_LLM = 0xFF8800  # orange, breathing
RING_FLASH_INTERVAL = 0.4  # startup flash half-period

TOOLS = [
    {
        "name": "search_youtube",
        "description": "Find music or videos on YouTube. Call this when the user asks to play something you don't already have a URL for. Returns candidate matches ranked by relevance — pick the best one yourself and play it with play_url. Do not read the candidates out to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_soundcloud",
        "description": "Find music on SoundCloud. Use this when search_youtube or play_url reports that YouTube rejected the request, and as the first choice for remixes, DJ sets and underground electronic music, which SoundCloud carries more of. Returns candidate matches — pick the best one yourself and play it with play_url. Do not read the candidates out to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "play_url",
        "description": "Stream audio from a URL via mpv. Use start_time to resume from a saved position (check play history for saved positions). Returns 'playing' once audio is actually flowing, or 'playback failed: ...' when the source rejected the request — YouTube does this intermittently. On failure, call this again with the next candidate from the search results rather than reporting defeat; only tell the user if several have failed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to play"},
                "title": {"type": "string", "description": "Human-readable title for this track (e.g. the title from search results). Used in play history."},
                "start_time": {"type": "number", "description": "Seconds from start to begin playback. Use the saved position from play history to resume where you left off."},
                "headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional HTTP headers to send with the request, e.g. [\"Origin: https://example.com\"]"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "stop",
        "description": "Stop any in-progress audio playback.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "set_volume",
        "description": "Set the system output volume to an exact percentage. Only use this when the user names a level ('volume 40', 'half volume'); for 'turn it up/down' use change_volume, for 'mute' use set_mute.",
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "Volume level 0 (mute) to 100"}
            },
            "required": ["level"]
        }
    },
    {
        "name": "change_volume",
        "description": "Turn the volume up or down in steps, like pressing a volume button. There are 16 steps from silent to full. Use 1 or -1 for 'turn it up/down', 2-3 for 'a bit more', 4 or more for 'a lot'. Turning it up also unmutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer", "description": "Positive to turn up, negative to turn down"}
            },
            "required": ["steps"]
        }
    },
    {
        "name": "get_volume",
        "description": "Read the current output volume and whether it is muted. Use when the user asks what the volume is, or before a relative change phrased as a target ('turn it up to about half').",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "set_mute",
        "description": "Mute or unmute the output. Unmuting restores the volume it had before.",
        "input_schema": {
            "type": "object",
            "properties": {
                "muted": {"type": "boolean", "description": "true to mute, false to unmute"}
            },
            "required": ["muted"]
        }
    },
    {
        "name": "set_timer",
        "description": "Set a named countdown timer. When it fires it plays an alarm and announces the name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A short name for the timer, e.g. 'pasta'"},
                "seconds": {"type": "integer", "description": "Duration in seconds"}
            },
            "required": ["name", "seconds"]
        }
    },
    {
        "name": "cancel_timer",
        "description": "Cancel a named timer before it fires.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer to cancel"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "list_timers",
        "description": "List all active timers and their remaining time.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_history",
        "description": "Read recent chat or play history. Use type='chat' to recall past conversations, type='plays' to see recently played tracks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["chat", "plays"], "description": "Which history to read"},
                "limit": {"type": "integer", "description": "Number of recent entries to return (default 20)"}
            },
            "required": ["type"]
        }
    },
    {
        "name": "update",
        "description": "Update the speaker software from git and restart. Target can be 'latest' for the newest release tag, a tag name like 'v1.2', or a branch name like 'dev'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "'latest', a version tag, or a branch name"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "follow_on",
        "description": "Keep the microphone open for the user's reply. Call this only when your spoken response ended in a genuine question that you need answered to continue. Do not call it after a statement, a confirmation, or once you have finished the task.",
        "input_schema": {"type": "object", "properties": {}}
    }
]

# Anthropic's server-side search. Passed through litellm untouched — its own web_search_options
# would map to an older tool version
ANTHROPIC_WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
SERVER_TOOL_ID_PREFIX = "srvtoolu_"  # calls Anthropic ran itself, which litellm lists among ours

# run by the speaker itself, for vendors with no native search that coexists with our tools
LOCAL_WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information — news, weather, scores, facts you are unsure of. Returns titles, urls and snippets.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}


class _PerfTimer:
    """Lap timer that prints interval and accumulated time at each named step."""
    def __init__(self):
        self._start: float | None = None
        self._last: float | None = None

    def start(self):
        """Reset and start the timer (call on wake detection)."""
        self._start = self._last = time.monotonic()

    def lap(self, label: str, indent: str = ""):
        """Print elapsed time since last lap and since start, then advance the lap mark."""
        now = time.monotonic()
        interval = now - self._last
        total = now - self._start
        log(f"{indent}[timing] {label}: +{interval:.2f}s  total={total:.2f}s")
        self._last = now


class SpeakerState(Enum):
    TEXT_CHAT = auto()
    LISTEN_FOR_WAKE = auto()
    RECORDING = auto()
    LLM_AGENT = auto()
    RESET = auto()
    CACHE_WAKE = auto()
    VAD_RECORD = auto()


@dataclass
class SpeakerContext:
    system: str
    voice_model: any
    whisper_model: any
    vad: any
    output_dev_index: int
    output_sample_rate: int
    wake_model: any
    input_dev_index: int
    input_sample_rate: int
    worker_mode: bool = False
    worker_url: str | None = None
    ring: any = None  # XVF3800 LED ring, None on platforms/boxes without one
    wake_threshold: float = 0.4
    wake_triggers: int = 1
    speaker_state: SpeakerState = SpeakerState.LISTEN_FOR_WAKE
    interrupt: threading.Event = field(default_factory=threading.Event)
    shutdown: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _LLM:
    """The vendor, models and key in use. Replaced whole by apply_llm_settings rather than mutated,
    so a turn already in flight finishes on the settings it started with."""
    provider: str
    model: str
    followup_model: str  # set equal to `model` to keep one prompt cache
    effort: str
    api_key: str | None
    api_base: str | None


class _PlaybackWatch:
    """One stream handed to mpv, followed from mpv's event thread so a failure has somewhere to go.

    `settled` is set once the stream has either played PLAYBACK_CONFIRM_SECONDS of audio or failed
    — play_url waits on it so a rejection reaches the agent inside the turn. A failure after that is
    late, and is retried or reported by _on_late_playback_failure."""

    def __init__(self, url: str, start_time: float, title: str | None, retry: bool = False):
        self.url = url  # what the user asked for, not the resolved stream
        self.start_time = start_time
        self.title = title
        self.retry = retry  # already a second attempt — a further failure is reported, not retried
        self.first_position: float | None = None
        self.position: float | None = None
        self.http_error: str | None = None
        self.failure: str | None = None
        self.settled = threading.Event()
        self.cancelled = False  # stopped or replaced on purpose — what mpv reports next is no failure


class _Player:
    def __init__(self):
        self.cmd_queue: queue.Queue = queue.Queue()
        self.active: bool = False
        self.current_url: str | None = None


class _Timers:
    def __init__(self):
        self._store: dict[str, tuple[threading.Timer, float]] = {}
        self._lock = threading.Lock()

    def set(self, name: str, timer: threading.Timer, end_time: float):
        with self._lock:
            existing = self._store.pop(name, None)
            if existing:
                existing[0].cancel()
            self._store[name] = (timer, end_time)

    def pop(self, name: str):
        with self._lock:
            return self._store.pop(name, None)

    def remove(self, name: str):
        with self._lock:
            self._store.pop(name, None)

    def keys(self):
        with self._lock:
            return list(self._store.keys())

    def items(self):
        with self._lock:
            return list(self._store.items())

    def __bool__(self):
        with self._lock:
            return bool(self._store)


class _SileroVAD:
    """Silero VAD via the silero-vad package — stateful, same is_speech() interface as webrtcvad."""

    def __init__(self, threshold: float = 0.5, mic_gain: float = 1.0, verbose: bool = False):
        from silero_vad import load_silero_vad
        self._model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self._mic_gain = mic_gain
        self._verbose = verbose
        log(f"[VAD] Silero loaded — threshold={threshold} mic_gain={mic_gain}")

    def reset(self):
        self._model.reset_states()

    def is_speech(self, audio_bytes: bytes, sample_rate: int) -> bool:
        import torch
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.clip(audio * self._mic_gain, -1.0, 1.0)
        tensor = torch.from_numpy(audio)
        prob = float(self._model(tensor, sample_rate))
        _levels.set_vad(prob)
        if self._verbose:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            log(f"[VAD] rms={rms:.4f} prob={prob:.3f} speech={prob > self.threshold}")
        return prob > self.threshold


class _VoiceProbe:
    """Silero VAD run purely for the timeline, so speech is visible while the speaker is listening
    for the wake word. The recording VAD cannot serve this — it only starts once a turn already has,
    which is after the moment you want to look at.

    Holds its own model instance rather than sharing ctx.vad: silero is stateful, and a display tap
    driving the LSTM state that recording decisions depend on would be a real bug. Loaded on first
    use, so a speaker that never turns capture on never pays for it."""

    def __init__(self):
        self._model = None
        self._carry: np.ndarray = np.array([], dtype=np.int16)
        self.mic_gain: float = 1.0

    def reset(self):
        self._carry = np.array([], dtype=np.int16)
        if self._model is not None:
            self._model.reset_states()

    def probe(self, audio: np.ndarray) -> float | None:
        """Feed one chunk of 16 kHz int16 audio and return the highest speech probability across the
        512-sample frames it completed, or None if it did not complete one. Wake chunks are 1280
        samples against silero's fixed 512, so the remainder carries into the next call — 2 frames,
        then 3, then 2, rather than dropping a fifth of the audio every chunk."""
        import torch
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            log("[timeline] voice probe loaded")
        self._carry = np.concatenate([self._carry, audio])
        best = None
        while len(self._carry) >= VAD_CHUNK:
            frame, self._carry = self._carry[:VAD_CHUNK], self._carry[VAD_CHUNK:]
            scaled = np.clip(frame.astype(np.float32) / 32768.0 * self.mic_gain, -1.0, 1.0)
            prob = float(self._model(torch.from_numpy(scaled), TARGET_SAMPLE_RATE))
            best = prob if best is None else max(best, prob)
        return best


class _Log:
    def __init__(self):
        self._buffer: deque = deque(maxlen=2000)
        self._counter: int = 0
        self._lock = threading.Lock()

    def append(self, text: str):
        with self._lock:
            self._buffer.append({"index": self._counter, "text": text, "ts": time.time()})
            self._counter += 1

    def get_since(self, since: int) -> dict:
        with self._lock:
            lines = [l for l in self._buffer if l["index"] > since]
            total = self._counter
        return {"lines": lines, "total": total}


class _Levels:
    """Ring buffer of mic amplitude frames plus pipeline event markers, for the web timeline.

    Frames arrive at a variable rate — 12.5/s while listening for the wake word (80ms chunks) and
    31.25/s while recording (32ms chunks) — so each carries its own timestamp and the UI plots
    against time rather than index. `score` and `vad` are whichever detector was running at the
    time and stay None otherwise."""

    def __init__(self):
        self._buffer: deque = deque(maxlen=LEVEL_BUFFER_FRAMES)
        self._markers: deque = deque(maxlen=MARKER_BUFFER_LEN)
        self._counter: int = 0
        self._marker_counter: int = 0
        self._lock = threading.Lock()
        # opt-in — this is a debug view, and the capture runs on every audio frame. off on every
        # start, the web ui turns it on for the length of a debugging session
        self.enabled: bool = False

    def set_enabled(self, enabled: bool):
        with self._lock:
            self.enabled = enabled
            if not enabled:
                # drop what was captured rather than leave a stale window to be shown on re-enable.
                # the counters keep climbing so a client's cursor stays valid across the gap
                self._buffer.clear()
                self._markers.clear()

    def append(self, rms: float, peak: float):
        if not self.enabled:
            return
        with self._lock:
            self._buffer.append({"i": self._counter, "ts": time.time(), "rms": rms, "peak": peak,
                                 "score": None, "vad": None})
            self._counter += 1

    def _stamp(self, key: str, value: float):
        # the detector runs immediately after the read that produced the frame, so the most recent
        # frame is always the one this score belongs to
        if not self.enabled:
            return
        with self._lock:
            if self._buffer:
                # cast here rather than trusting call sites — the models hand back np.float32, which
                # only fails at json encoding time, as a 500 from /levels rather than anything local
                self._buffer[-1][key] = float(value)

    def set_score(self, score: float):
        self._stamp("score", score)

    def set_vad(self, prob: float):
        self._stamp("vad", prob)

    def mark(self, kind: str, text: str = ""):
        if not self.enabled:
            return
        with self._lock:
            self._markers.append({"i": self._marker_counter, "ts": time.time(), "kind": kind, "text": text})
            self._marker_counter += 1

    def get_since(self, since: int, marker_since: int = -1) -> dict:
        # markers carry their own cursor — resending the whole window four times a second is real
        # bandwidth on a pi, and a client watching the count cannot tell that the ring has rotated
        with self._lock:
            frames = [f for f in self._buffer if f["i"] > since]
            markers = [m for m in self._markers if m["i"] > marker_since]
            total = self._counter
            enabled = self.enabled
        return {"frames": frames, "markers": markers, "total": total, "enabled": enabled}


ctx: SpeakerContext | None = None
chat_history: list[dict] = []
# The conversation as the API sees it — content blocks, tool_use and tool_result included. Distinct
# from chat_history, which is flattened text for the web UI and .history/chat.jsonl.
api_messages: list[dict] = []
_llm: _LLM | None = None
_player = _Player()
_context_notes: list[str] = []  # things that happened between turns, handed to the next llm turn
_duck_volume: int = 0
_mono_output: bool = False # downmix mpv playback to mono, for devices with a single speaker
_silence_timeout: float = 1.6 # continuous silence that ends a recording, overridable from config
_perf_timer = _PerfTimer()
_timers = _Timers()
_log = _Log()
_levels = _Levels()
_voice_probe = _VoiceProbe()
_ring_startup_thread: threading.Thread | None = None
_ring_startup_stop = threading.Event()


def log(text: str):
    print(text)
    _log.append(text)


def get_log_lines(since: int = 0) -> dict:
    return _log.get_since(since)


def mark(kind: str, text: str = ""):
    """Record a pipeline event on the timeline (wake, record, transcribe, ...)."""
    _levels.mark(kind, text)


def get_level_frames(since: int = -1, marker_since: int = -1) -> dict:
    return _levels.get_since(since, marker_since)


def set_levels_enabled(enabled: bool):
    """Turn timeline capture on or off at runtime. Off costs nothing on the audio thread."""
    _levels.set_enabled(enabled)
    _voice_probe.reset()  # drop carried samples and LSTM state from the last session
    log(f"timeline capture {'enabled' if enabled else 'disabled'}")


def start_perf_timer():
    global _perf_timer
    _perf_timer.start()


def enumerate_audio_devices() -> dict:
    """Return lists of available input and output device names."""
    devices = dev.query_devices()
    inputs, outputs, all = [], [], []
    for d in devices:
        if d['max_input_channels'] > 0:
            inputs.append(d['name'])
        if d['max_output_channels'] > 0:
            outputs.append(d['name'])
        all.append(d['name'])
    return {"inputs": inputs, "outputs": outputs, "all": all}


def _get_audio_device_index(name: str):
    """Find and return the sounddevice info dict for the first device whose name contains `name`."""
    devices = dev.query_devices()
    index = 0
    for device in devices:
        if device['name'].find(name) != -1:
            return dev.query_devices(index)
        index += 1
    return None


def _flush_stream(stream, sample_rate: int):
    """Restart the input stream and discard buffered audio to prevent stale data re-triggering the wake word."""
    stream.stop()
    stream.start()
    # drain buffered audio to avoid re-triggering on stale data (Windows WASAPI retains buffer on restart)
    sample_ratio = int(math.ceil(sample_rate / TARGET_SAMPLE_RATE))
    chunk = WAKE_CHUNK * sample_ratio
    discard_count = int(0.5 * sample_rate / chunk) + 1
    for _ in range(discard_count):
        try:
            stream.read(chunk)
        except Exception:
            break



def _test_tone(dev, dev_index: int, sample_rate: int):
    """Play a 440 Hz test tone on the given output device."""
    # generate a simple 440hz test tone
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    dev.play(tone, samplerate=sample_rate, device=dev_index)
    dev.wait()


def _resample_audio(audio, orig_rate: int, target_rate: int, target_length: int | None = None):
    """Linearly interpolate `audio` from `orig_rate` to `target_rate`, returning int16.
    `target_length` forces the output size so callers needing an exact frame count do not have to
    over-read and truncate, which silently discards audio at fractional rate ratios."""
    new_length = target_length if target_length is not None else int(len(audio) * target_rate / orig_rate)
    return np.interp(
        np.linspace(0, len(audio), new_length),
        np.arange(len(audio)),
        audio
    ).astype(np.int16)


def _read_audio(stream, chunk_size16: int, sample_rate: int) -> np.ndarray:
    """Read one chunk from `stream` at native `sample_rate` and resample to exactly `chunk_size16`
    samples of 16 kHz int16."""
    if sample_rate == TARGET_SAMPLE_RATE:
        audio, _ = stream.read(chunk_size16)
        flat = np.squeeze(audio)
    else:
        # read the exact native span that maps onto chunk_size16 samples. reading ceil(rate/16k) * chunk
        # and truncating loses 8% of every chunk at 44.1 kHz, which both mangles the audio handed to
        # whisper and breaks the frame continuity silero's LSTM state depends on
        native_frames = int(round(chunk_size16 * sample_rate / TARGET_SAMPLE_RATE))
        audio, _ = stream.read(native_frames)
        flat = _resample_audio(np.squeeze(audio), sample_rate, TARGET_SAMPLE_RATE, target_length=chunk_size16)
    # every live frame — wake listen, recording and barge-in — passes through here, so this is the
    # one tap the timeline needs. float32 before squaring, int16 ** 2 overflows. checked before the
    # conversion, not inside append(), so a disabled timeline costs a bool read per frame
    if _levels.enabled:
        scaled = flat.astype(np.float32) / 32768.0
        _levels.append(float(np.sqrt(np.mean(scaled * scaled))), float(np.abs(scaled).max()))
    return flat



def _vad_record(vad, stream, sample_rate: int, silence_timeout: float = 1.5) -> np.ndarray:
    """Record raw audio at native sample rate, using resampled audio only for VAD checks."""
    vad.reset()
    raw_chunk = int(round(VAD_CHUNK * sample_rate / TARGET_SAMPLE_RATE))
    raw_chunks = []
    # wait for speech onset
    while True:
        raw, _ = stream.read(raw_chunk)
        raw_flat = np.squeeze(raw)
        audio_16k = _resample_audio(raw_flat, sample_rate, TARGET_SAMPLE_RATE, target_length=VAD_CHUNK)
        if vad.is_speech(audio_16k.tobytes(), TARGET_SAMPLE_RATE):
            raw_chunks.append(raw_flat)
            break
    # record until silence
    silent_duration = 0.0
    while silent_duration < silence_timeout:
        raw, _ = stream.read(raw_chunk)
        raw_flat = np.squeeze(raw)
        raw_chunks.append(raw_flat)
        audio_16k = _resample_audio(raw_flat, sample_rate, TARGET_SAMPLE_RATE, target_length=VAD_CHUNK)
        silent_duration = 0.0 if vad.is_speech(audio_16k.tobytes(), TARGET_SAMPLE_RATE) else silent_duration + VAD_CHUNK / TARGET_SAMPLE_RATE
    return np.concatenate(raw_chunks)


def _play_nudge(stream, sample_rate: int) -> float:
    """Prompt a user who woke the speaker but has not spoken, and return the seconds it took.

    Waits for the clip to finish and then flushes the mic: a nudge is speech, and left in the buffer
    the VAD would take it for the user's request and whisper would transcribe the speaker to itself."""
    start = time.monotonic()
    clip = _play_random_sound(NUDGE_SOUND_DIR)
    if clip is None:
        return 0.0
    log("no speech yet, nudging")
    mark("nudge", "")
    clip.join()
    _flush_stream(stream, sample_rate)
    return time.monotonic() - start


def _record_until_silence(vad, stream, sample_rate: int, silence_timeout: float | None = None, onset_timeout: float=8.0,
                          nudge_after: float | None = None) -> tuple[np.ndarray | None, float]:
    """Wait for sustained speech onset then record until `silence_timeout` seconds of continuous silence.
    Returns (audio, onset_elapsed). audio is None if no speech begins within `onset_timeout` seconds, or if
    what was captured holds too little speech to be a request. If `nudge_after` seconds pass with no
    onset, a nudge clip plays once."""
    silence_timeout = _silence_timeout if silence_timeout is None else silence_timeout
    vad.reset()  # silero is stateful — carrying the previous turn's LSTM state in skews early frames
    preroll = deque(maxlen=VAD_PREROLL_FRAMES)
    chunks = []
    silent_duration = 0.0
    speech_duration = 0.0
    onset_run = 0
    speech_started = False
    onset_elapsed = 0.0
    chunk_duration = VAD_CHUNK / TARGET_SAMPLE_RATE
    log_limiter = 0
    while True:
        audio_flat = _read_audio(stream, VAD_CHUNK, sample_rate)
        is_speech = vad.is_speech(audio_flat.tobytes(), TARGET_SAMPLE_RATE)
        if not speech_started:
            # one frame over threshold is the wake chime bleeding back through the mic, a breath or a
            # click. committing on it starts the silence countdown before the user has spoken, so the
            # turn ends on a ~1s clip of room noise. require a sustained run, and keep a pre-roll so
            # the leading phoneme is not lost to the frames it took to confirm
            preroll.append(audio_flat)
            onset_run = onset_run + 1 if is_speech else 0
            if onset_run >= VAD_ONSET_FRAMES:
                speech_started = True
                speech_duration = onset_run * chunk_duration
                chunks.extend(preroll)
                mark("onset", "speech started")
                continue
            if int(onset_elapsed) > log_limiter:
                log_limiter = int(onset_elapsed)
                log(f"waiting for onset: {log_limiter}s")
            onset_elapsed += chunk_duration
            if nudge_after is not None and onset_elapsed >= nudge_after:
                nudge_after = None
                onset_elapsed += _play_nudge(stream, sample_rate)
                # start the onset search clean — nothing heard during the nudge belongs to the request
                vad.reset()
                preroll.clear()
                onset_run = 0
            if onset_elapsed >= onset_timeout:
                log("no speech detected, returning to wake listen")
                mark("warn", f"no speech within {onset_timeout:.1f}s onset budget")
                return None, onset_elapsed
            continue
        chunks.append(audio_flat)
        if is_speech:
            speech_duration += chunk_duration
            silent_duration = 0.0
        else:
            silent_duration += chunk_duration
            if silent_duration >= silence_timeout:
                break

    if speech_duration < VAD_MIN_SPEECH:
        log(f"discarding {speech_duration:.2f}s of speech, below the {VAD_MIN_SPEECH}s minimum")
        mark("warn", f"discarded {speech_duration:.2f}s, below {VAD_MIN_SPEECH}s minimum")
        return None, onset_elapsed
    log(f"recorded {len(chunks) * chunk_duration:.1f}s ({speech_duration:.1f}s speech)")
    mark("recorded", f"{len(chunks) * chunk_duration:.1f}s ({speech_duration:.1f}s speech)")
    return np.concatenate(chunks), onset_elapsed


def _listen_for_wake(wake_model, stream, sample_rate: int,
                    threshold: float = 0.79, num_triggers: int = 2,
                    window_size: int = 5, buffer_duration: float = 0.0) -> np.ndarray:
    """Block until wake word is detected. Returns buffered pre-trigger audio (empty if buffer_duration=0)."""
    score_window = deque(maxlen=window_size)
    rolling_buffer = None
    if buffer_duration > 0.0:
        buffer_chunks = int(buffer_duration / (WAKE_CHUNK / TARGET_SAMPLE_RATE))
        rolling_buffer = deque(maxlen=buffer_chunks)

    while True:
        audio_flat = _read_audio(stream, WAKE_CHUNK, sample_rate)
        if rolling_buffer is not None:
            rolling_buffer.append(audio_flat)
        # the only place speech can be told from room noise before a turn starts. debug-only, so it
        # runs behind the capture flag rather than on every wake frame forever
        if _levels.enabled:
            voice_prob = _voice_probe.probe(audio_flat)
            if voice_prob is not None:
                _levels.set_vad(voice_prob)
        prediction = wake_model.predict(audio_flat)
        for _, score in prediction.items():
            _levels.set_score(float(score))
            score_window.append(score > threshold)
            hits = sum(score_window)
            if score > 0.0:
                if "--verbose" in sys.argv:
                    log(f"[wakeword] score: {score:.3f} ({hits}/{num_triggers})")
            if hits >= num_triggers:
                log("ello mate!")
                mark("wake", f"score {score:.2f}")
                if rolling_buffer is not None:
                    return np.concatenate(rolling_buffer)
                return np.array([], dtype=np.int16)


def _write_wav(audio: np.ndarray, sample_rate: int, filepath: str, gain: float = 4.0):
    """Write int16 audio to a mono WAV file, applying an optional amplitude gain."""
    boosted = np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(boosted.tobytes())



def _transcribe_audio(whisper_model, audio: np.ndarray, worker_url: str | None = None) -> str:
    """Transcribe int16 audio to English text using Whisper, or delegate to worker."""
    if worker_url:
        log(f"delegating transcribe to worker: {worker_url}")
        import io as _io
        buf = _io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        try:
            resp = requests.post(
                f"{worker_url}/rpc/transcribe",
                files={"audio": ("audio.wav", buf.getvalue(), "audio/wav")},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
        except Exception as e:
            log(f"worker transcribe error: {e}")
            return ""
    audio_float = audio.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(
        audio_float,
        language="en",
        beam_size=1,
        # whisper invents filler ("yeah", "bye", "thanks for watching") on near-silence, and by default
        # feeds each turn's text forward as a prompt so one invention seeds the next
        condition_on_previous_text=False,
    )
    kept = []
    for segment in segments:
        if segment.no_speech_prob > NO_SPEECH_PROB_LIMIT:
            log(f"dropping hallucinated segment (no_speech={segment.no_speech_prob:.2f}): {segment.text.strip()!r}")
            continue
        kept.append(segment.text)
    return " ".join(kept)


def _execute_tool(tool_name: str, tool_input: dict) -> tuple[str, SpeakerState | None]:
    """Dispatch an LLM tool call and return its result string and optional next state."""
    if tool_name == "search_youtube":
        return search_youtube(tool_input["query"]), None
    elif tool_name == "search_soundcloud":
        return search_soundcloud(tool_input["query"]), None
    elif tool_name == "web_search":
        return web_search(tool_input["query"]), None
    elif tool_name == "play_url":
        result = play_url(tool_input["url"], tool_input.get("headers"), tool_input.get("start_time", 0.0), tool_input.get("title"))
        if result != "playing":
            # Returning no state keeps the turn alive so the agent can try another candidate.
            return f"playback failed: {result}", None
        return "playing", SpeakerState.RESET
    elif tool_name == "stop":
        active_timers = _timers.keys()
        if active_timers:
            for name in active_timers:
                entry = _timers.pop(name)
                if entry:
                    entry[0].cancel()
            return f"cancelled timers: {', '.join(active_timers)}", SpeakerState.RESET
        _player.cmd_queue.put(('stop',))
        return "stopped", SpeakerState.RESET
    elif tool_name == "follow_on":
        return "listening", SpeakerState.RECORDING
    elif tool_name == "set_volume":
        return set_volume(tool_input["level"]), None
    elif tool_name == "change_volume":
        return change_volume(tool_input["steps"]), None
    elif tool_name == "set_mute":
        return set_mute(tool_input["muted"]), None
    elif tool_name == "get_volume":
        return get_volume(), None
    elif tool_name == "set_timer":
        return set_timer(tool_input["name"], tool_input["seconds"]), None
    elif tool_name == "cancel_timer":
        return cancel_timer(tool_input["name"]), None
    elif tool_name == "list_timers":
        return list_timers(), None
    elif tool_name == "get_history":
        return get_history(tool_input["type"], tool_input.get("limit", 20)), None
    elif tool_name == "update":
        return update(tool_input["target"]), SpeakerState.RESET
    return "unknown tool", None


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split `text` on sentence boundaries, returning completed sentences and the trailing fragment."""
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        return [], text
    return parts[:-1], parts[-1]


def _is_plain_user_turn(message: dict) -> bool:
    """True if `message` is a spoken user turn rather than a carrier for tool_result blocks."""
    return message["role"] == "user" and isinstance(message["content"], str)


def _trim_api_messages():
    """Trim `api_messages` to the rolling window, cutting only where the conversation can legally start.

    A tool_use block must always be followed by its matching tool_result, so a naive slice can orphan
    a pair and get the whole request rejected. Cut back to a plain spoken user turn instead."""
    if len(api_messages) <= API_HISTORY_LIMIT:
        return
    for index in range(len(api_messages) - API_HISTORY_LIMIT, len(api_messages)):
        if _is_plain_user_turn(api_messages[index]):
            if index:
                log(f"trimming {index} message(s) from context")
                del api_messages[:index]
            return
    # No safe cut point in the window — the tail is one long tool exchange, so keep it whole.


def _to_openai_tool(tool: dict) -> dict:
    """Wrap one of our tool definitions in the OpenAI function shape litellm takes."""
    return {
        "type": "function",
        "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]},
    }


def _model_supports(check, model: str) -> bool:
    """Ask litellm's registry about a capability. A model it has never heard of — any local ollama
    model — can raise rather than answer, and that is a no."""
    try:
        return bool(check(model=model))
    except Exception:
        return False


def _llm_request(llm: _LLM, model: str, system: str, messages: list[dict]) -> dict:
    """Arguments for one streamed completion, shaped for the vendor in `llm`."""
    tools = [_to_openai_tool(t) for t in TOOLS]
    if PROVIDERS.get(llm.provider, {}).get("native_search"):
        tools.append(ANTHROPIC_WEB_SEARCH_TOOL)
    else:
        tools.append(_to_openai_tool(LOCAL_WEB_SEARCH_TOOL))
    if llm.provider == "anthropic":
        # Tools render before system, so one breakpoint here caches the whole stable prefix. Everything
        # that varies per turn lives in messages, after it. Other vendors cache prefixes unprompted
        system_content = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_content = system
    request = {
        "model": model,
        "messages": [{"role": "system", "content": system_content}, *messages],
        "tools": tools,
        "max_tokens": LLM_MAX_TOKENS,
        "api_key": llm.api_key,
        "api_base": llm.api_base,
        "stream": True,
        # a param one vendor lacks is dropped rather than failing the turn
        "drop_params": True,
    }
    if _model_supports(litellm.supports_reasoning, model):
        # Thinking stays ON deliberately. Disabling it on Opus 5 can make the model write a tool call
        # as plain text instead of a tool_use block — the turn looks fine and the call silently never
        # runs. effort=low buys the latency back safely. On Anthropic litellm maps this to adaptive
        # thinking plus output_config.effort
        request["reasoning_effort"] = llm.effort
    return request


def _assistant_message(message) -> dict:
    """The assistant turn as it goes back into history. thinking_blocks and provider_specific_fields
    carry what a vendor needs replayed alongside a tool call — Anthropic's thinking signatures,
    Gemini's thought signatures — so they stay. reasoning_content is display text, and strict
    openai-compatible endpoints reject it on the way back in."""
    entry = {"role": "assistant", "content": message.content or None}
    dumped = message.model_dump(exclude_none=True)
    for key in ("tool_calls", "thinking_blocks", "provider_specific_fields"):
        if dumped.get(key):
            entry[key] = dumped[key]
    return entry


def _parse_tool_input(arguments: str | None) -> dict | None:
    """Tool call arguments arrive as a JSON string. Smaller models get it wrong now and then."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def query_llm(text: str, mic_stream=None, followup: bool = False, speak_after: threading.Thread | None = None) -> SpeakerState:
    """Send `text` to the LLM, stream TTS as sentences arrive, handle tool calls, and return the next state.
    Speech waits for `speak_after` (the acknowledge clip) to finish, so a quick reply does not talk over it."""
    ctx.interrupt.clear()

    _history_start = len(chat_history)
    stop_flag = threading.Event()
    wake_thread = None
    if mic_stream is not None:
        ctx.wake_model.reset()  # clear residual activation from the wake that triggered this session
        wake_thread = threading.Thread(
            target=_interrupt_wake_listen,
            args=(ctx.wake_model, mic_stream, ctx.input_sample_rate, stop_flag, ctx.interrupt, ctx.wake_threshold, ctx.wake_triggers),
            daemon=True
        )
        wake_thread.start()

    # TTS runs in a separate thread so the LLM stream loop is never stalled waiting for playback.
    tts_queue: queue.Queue = queue.Queue()

    def _tts_worker():
        while True:
            sentence = tts_queue.get()
            if sentence is None:
                tts_queue.task_done()
                break
            if speak_after is not None:
                speak_after.join()
            if not ctx.interrupt.is_set():
                _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model, sentence)
            tts_queue.task_done()

    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    def _record_partial(spoken: str):
        """Keep what the assistant had said before an interrupt cut it off.

        Without this the model has no idea it was halfway through listing options, so a follow-up
        like "the second one" has nothing to refer back to."""
        if spoken.strip():
            api_messages.append({"role": "assistant", "content": spoken})

    def _speak_now(line: str):
        """Speak a canned line and wait for it, so the user hears why the turn ended."""
        tts_queue.put(line)
        tts_queue.join()
        entry = {"role": "assistant", "text": line}
        chat_history.append(entry)

    def _discard_tts():
        """Drop queued-but-unspoken sentences (interrupt already set, so worker drains fast)."""
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
                tts_queue.task_done()
            except queue.Empty:
                break

    try:
        # The rolling conversation is the whole point: without it the model cannot resolve "that one"
        # or remember the search results it just described.
        _trim_api_messages()
        if _context_notes:
            # events from between turns ride on the user's message — no turn of their own to sit in
            notes = "\n".join(f"[{note}]" for note in _context_notes)
            _context_notes.clear()
            text = f"{notes}\n{text}"
        api_messages.append({"role": "user", "content": text})
        messages = api_messages
        llm = _llm  # held for the whole turn, so a settings save mid-turn cannot split it across vendors
        model = llm.followup_model if followup else llm.model
        next_state = SpeakerState.RESET

        for _round in range(MAX_TOOL_ROUNDS):
            accumulated_text = ""
            sentence_buffer = ""
            _live_entry = None  # mutable chat_history entry updated live during streaming
            chunks = []

            try:
                for chunk in litellm.completion(**_llm_request(llm, model, ctx.system, messages)):
                    if ctx.interrupt.is_set():
                        break
                    chunks.append(chunk)
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if not delta:
                        continue
                    sentence_buffer += delta
                    accumulated_text += delta
                    # Update chat_history live so web UI polls see the response during TTS
                    if _live_entry is None:
                        _live_entry = {"role": "assistant", "text": accumulated_text}
                        chat_history.append(_live_entry)
                    else:
                        _live_entry["text"] = accumulated_text
                    sentences, sentence_buffer = _split_sentences(sentence_buffer)
                    for s in sentences:
                        if s.strip():
                            tts_queue.put(s)
            except openai.APIError as e:
                # litellm raises openai's exception types whatever the vendor. Without this a bad key
                # or a retired model kills the speak loop, and the speaker goes deaf until a restart
                log(f"llm error: {type(e).__name__}: {e}")
                _discard_tts()
                if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
                    _speak_now("I can't reach the language model. Check the API key in settings.")
                elif isinstance(e, openai.NotFoundError):
                    _speak_now("The language model in settings wasn't found.")
                else:
                    _speak_now("Sorry, I couldn't reach the language model.")
                return next_state

            if ctx.interrupt.is_set():
                _discard_tts()
                _record_partial(accumulated_text)
                return SpeakerState.RECORDING

            if sentence_buffer.strip():
                tts_queue.put(sentence_buffer)

            # Wait for all queued sentences to finish playing before acting on finish_reason
            tts_queue.join()

            if ctx.interrupt.is_set():
                _record_partial(accumulated_text)
                return SpeakerState.RECORDING

            response = litellm.stream_chunk_builder(chunks) if chunks else None
            if response is None:
                log("llm returned an empty stream")
                return next_state
            choice = response.choices[0]
            message = choice.message
            if _live_entry is not None and not accumulated_text.strip():
                chat_history.remove(_live_entry)

            if choice.finish_reason == "length":
                # Checked before tool calls: a truncated call has truncated arguments, and replaying a
                # call with no result alongside it gets the next request rejected.
                log("llm hit max_tokens")
                _record_partial(accumulated_text)
                _speak_now("Sorry, I lost my thread there.")
                return next_state

            # Anthropic's web search (and the code execution it filters results with) already ran
            # server-side — litellm reports those as tool calls too, but answering one is rejected.
            # They stay in the assistant message, which litellm replays with their results.
            client_calls = [c for c in message.tool_calls or [] if not c.id.startswith(SERVER_TOOL_ID_PREFIX)]

            # Keyed on the calls rather than finish_reason — ollama and some gemini models report
            # "stop" on a turn that carries tool calls.
            if client_calls:
                messages.append(_assistant_message(message))
                terminal = False
                for call in client_calls:
                    tool_input = _parse_tool_input(call.function.arguments)
                    log(f"\ttool: {call.function.name} {tool_input}")
                    _perf_timer.lap(call.function.name, indent="\t")
                    if tool_input is None:
                        result, state = "invalid arguments: expected a JSON object", None
                    else:
                        try:
                            result, state = _execute_tool(call.function.name, tool_input)
                        except (KeyError, TypeError) as e:
                            result, state = f"invalid arguments: {e}", None
                    if state is not None:
                        next_state = state
                        # A terminal tool only ends the turn if it actually worked — a failed
                        # play_url returns no state, so the agent gets a round to try another.
                        terminal = terminal or call.function.name in TERMINAL_TOOLS
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

                # Terminal actions don't need another LLM round-trip
                if terminal:
                    return next_state
                continue

            if choice.finish_reason == "content_filter":
                log("llm refused the request")
                # A refusal carries no usable content, so stand the spoken line in as the turn.
                _speak_now("Sorry, I can't help with that one.")
                api_messages.append({"role": "assistant", "content": "Sorry, I can't help with that one."})
                return next_state

            if choice.finish_reason != "stop":
                log(f"unexpected finish_reason: {choice.finish_reason}")
            # A turn with nothing in it (thinking alone) stays out of history — an empty assistant
            # message is rejected on the next request.
            if message.content or message.tool_calls or getattr(message, "thinking_blocks", None):
                messages.append(_assistant_message(message))
            return next_state

        log(f"tool loop hit {MAX_TOOL_ROUNDS} rounds, giving up")
        _speak_now("Sorry, I got stuck on that.")
        return next_state

    finally:
        _discard_tts()
        tts_queue.put(None)
        tts_thread.join(timeout=2.0)
        stop_flag.set()
        if wake_thread is not None:
            wake_thread.join(timeout=0.5)
        ctx.interrupt.clear()
        for entry in chat_history[_history_start:]:
            _save_history(CHAT_HISTORY_PATH, entry)


def _strip_markdown(text: str) -> str:
    """Remove common markdown so TTS doesn't read out symbols."""
    # headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # bold / italic
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    # inline code and code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # links and images
    text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # blockquotes and list markers
    text = re.sub(r'^[>\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _speak(dev, dev_index, sample_rate, voice_model, text: str):
    """Synthesise `text` to audio and play it, stopping early if interrupted."""
    text = _strip_markdown(text)
    if not text:
        return
    if ctx is not None and ctx.interrupt.is_set():
        return
    worker_url = ctx.worker_url if ctx else None
    if worker_url:
        log(f"delegating speak to worker: {worker_url} — {text[:60]}")
        import io as _io
        try:
            resp = requests.post(
                f"{worker_url}/rpc/speak",
                json={"text": text},
                timeout=30,
            )
            resp.raise_for_status()
            buf = _io.BytesIO(resp.content)
            with wave.open(buf, 'rb') as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            audio_int16 = np.frombuffer(raw, dtype=np.int16)
        except Exception as e:
            log(f"worker speak error: {e}")
            return
        resampled = _resample_audio(audio_int16, sr, sample_rate)
        dev.play(resampled, samplerate=sample_rate, device=dev_index)
        done = threading.Event()
        threading.Thread(target=lambda: (dev.wait(), done.set()), daemon=True).start()
        while not done.wait(timeout=0.05):
            if ctx is not None and ctx.interrupt.is_set():
                dev.stop()
                return
        return
    chunks = []
    for audio_chunk in voice_model.synthesize(text):
        int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
        chunks.append(int_data)
    if chunks:
        full_audio = np.concatenate(chunks)
        resampled = _resample_audio(full_audio, voice_model.config.sample_rate, sample_rate)
        dev.play(resampled, samplerate=sample_rate, device=dev_index)
        done = threading.Event()
        threading.Thread(target=lambda: (dev.wait(), done.set()), daemon=True).start()
        while not done.wait(timeout=0.05):
            if ctx is not None and ctx.interrupt.is_set():
                dev.stop()
                return


def _interrupt_wake_listen(wake_model, stream, input_sample_rate: int, stop_flag: threading.Event, interrupt: threading.Event,
                           threshold: float = 0.4, num_triggers: int = 1):
    """Poll the wake model in a background thread and set `interrupt` if the wake word is detected during TTS."""
    score_window = deque(maxlen=5)
    while not stop_flag.is_set():
        audio_flat = _read_audio(stream, WAKE_CHUNK, input_sample_rate)
        prediction = wake_model.predict(audio_flat)
        for _, score in prediction.items():
            _levels.set_score(float(score))
            score_window.append(score > threshold)
            if sum(score_window) >= num_triggers:
                log("interrupt: wake word during TTS")
                mark("interrupt", f"wake during TTS, score {score:.2f}")
                interrupt.set()
                return


def _watch_mpv_log(watch: _PlaybackWatch, level: str, component: str, message: str):
    print(f"[mpv/{component}] {level}: {message}")
    match = _HTTP_ERROR.search(message)
    if match:
        watch.http_error = f"HTTP {match.group(1) or match.group(2)}"


def _watch_position(watch: _PlaybackWatch, position: float | None):
    if position is None or watch.cancelled:
        return
    if watch.first_position is None:
        watch.first_position = position
    watch.position = position
    if position - watch.first_position >= PLAYBACK_CONFIRM_SECONDS:
        watch.settled.set()


def _watch_end_file(watch: _PlaybackWatch, event):
    """mpv finished the file. A failure if mpv says so, or if an HTTP error was logged on the way —
    a stream cut off by a 403 often ends as a plain EOF, so the reason alone misses it."""
    if watch.cancelled:
        return
    data = getattr(event, "data", None)
    reason = getattr(data, "reason", None)
    if reason is None and isinstance(event, dict):  # python-mpv before 1.0 hands over a dict
        reason = event.get("event", {}).get("reason")
    if reason not in (MPV_END_FILE_ERROR, "error") and not watch.http_error:
        return  # the track simply ended
    watch.failure = watch.http_error or "the stream stopped with an error"
    if watch.settled.is_set():
        threading.Thread(target=_on_late_playback_failure, args=(watch,), daemon=True).start()
    else:
        watch.settled.set()  # play_url is still waiting, and will report it to the agent


def _describe_failure(reason: str) -> str:
    if "403" in reason:
        return "the source refused the stream"
    return reason


def _on_late_playback_failure(watch: _PlaybackWatch):
    """A stream that started fine and died later — after the turn that started it had ended, so no
    agent is waiting to hear about it. Try once more from where it stopped (a fresh resolve usually
    gets a URL that works), and if that fails too, say so rather than leave the room in silence."""
    position = watch.position or watch.start_time
    log(f"playback of {watch.url} failed at {position:.0f}s: {watch.failure}")
    if not watch.retry:
        log("retrying playback from the same position")
        result = _play(watch.url, None, position, watch.title, retry=True)
        if result == "playing":
            return
        watch.failure = result
    _report_playback_failure(watch.title or "that track", _describe_failure(watch.failure))


def _report_playback_failure(title: str, reason: str):
    line = f"Sorry, {title} stopped playing. {reason[0].upper()}{reason[1:]}."
    # into the next turn's context, so "try another one" or "what happened" makes sense to the llm
    _context_notes.append(f"Playback of '{title}' failed after it had started: {reason}. The user was told.")
    entry = {"role": "assistant", "text": line}
    chat_history.append(entry)
    _save_history(CHAT_HISTORY_PATH, entry)
    # only speak into an idle room — mid-turn, the note above carries it instead
    if ctx is not None and ctx.speaker_state == SpeakerState.LISTEN_FOR_WAKE:
        _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model, line)
    else:
        log(f"not speaking playback failure mid-turn: {line}")


def _player_loop():
    """Background thread: consume commands from `_player.cmd_queue` and drive the mpv player."""
    log("oi!! player! loop!!")

    player: mpv.MPV | None = None
    cleanup_dir: str | None = None
    watch: _PlaybackWatch | None = None

    def _stop():
        nonlocal player, cleanup_dir, watch
        if watch is not None:
            watch.cancelled = True  # terminating mpv ends the file — that is not a failure
            watch.settled.set()
            watch = None
        if player is not None:
            try:
                player.terminate()
            except Exception:
                pass
            player = None
            _player.current_url = None
        if cleanup_dir is not None:
            try:
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            except Exception:
                pass
            cleanup_dir = None
        _player.active = False

    while True:
        try:
            cmd, *args = _player.cmd_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if cmd == 'play':
            _stop()
            url, headers, start_time, original_url, watch = args[0], args[1], args[2], args[3], args[4]
            log(f"play_url: {url[:80]}...")
            current = watch  # bound now — the callbacks must not see a later play's watch
            player = mpv.MPV(vid=False, terminal=False,
                             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                             cache=True,
                             cache_secs=30,
                             audio_buffer=2,
                             demuxer_max_bytes="50MiB",
                             audio_channels="mono" if _mono_output else "auto-safe",
                             log_handler=lambda level, component, message: _watch_mpv_log(current, level, component, message),
                             loglevel="warn")
            player.observe_property("time-pos", lambda _name, value: _watch_position(current, value))
            player.event_callback("end-file")(lambda event: _watch_end_file(current, event))
            if headers:
                for header in headers:
                    player.command("change-list", "http-header-fields", "append", header)
            if start_time:
                player['start'] = str(int(start_time))
            player.play(url)
            _player.current_url = original_url
            _player.active = True

        elif cmd == 'play_file':
            _stop()
            path, cleanup_dir = args[0], args[1]
            log(f"playing: {path}")
            player = mpv.MPV(vid=False, terminal=False,
                             audio_channels="mono" if _mono_output else "auto-safe",
                             log_handler=lambda level, component, message: print(f"[mpv/{component}] {level}: {message}"),
                             loglevel="warn")
            player.play(path)
            _player.active = True

        elif cmd == 'stop':
            _stop()

        elif cmd == 'pause':
            if player is not None:
                try:
                    player.pause = True
                except Exception:
                    pass

        elif cmd == 'resume':
            if player is not None:
                try:
                    player.pause = False
                except Exception:
                    pass

        elif cmd == 'duck':
            if player is not None:
                try:
                    player.volume = args[0]
                except Exception:
                    pass

        elif cmd == 'unduck':
            if player is not None:
                try:
                    player.volume = 100
                except Exception:
                    pass

        elif cmd == 'quit':
            _stop()
            break


def _ytdlp(args: list[str], client: str = "") -> subprocess.CompletedProcess:
    """Run yt-dlp. `client` pins a YouTube player client; empty uses yt-dlp's own default chain."""
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    if client:
        cmd += ["--extractor-args", f"youtube:player_client={client}"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=YTDLP_TIMEOUT)


def _ytdlp_error(stderr: str) -> str:
    """Condense yt-dlp's stderr to the one line worth reporting."""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    errors = [l for l in lines if l.startswith("ERROR")] or lines
    return (errors[-1] if errors else "no output")[:160]


def _stream_is_playable(url: str) -> str:
    """Range-probe a resolved stream, returning "" if it serves bytes or a short reason if not.

    A successful yt-dlp resolve does not mean the URL works: YouTube hands back throttled or
    IP-bound URLs that 403 on first read. Finding out here means we can still try another client,
    rather than mpv failing silently after the turn has already ended."""
    try:
        request = urllib.request.Request(url, headers={
            "Range": f"bytes=0-{PROBE_BYTES - 1}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        })
        with urllib.request.urlopen(request, timeout=15) as response:
            return "" if response.read(1) else "empty response"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:
        return type(e).__name__


def _resolve_youtube_stream(url: str) -> tuple[str, str, str]:
    """Resolve a YouTube URL to a stream that actually serves bytes.

    Returns (stream_url, title, error). Tries each player client in turn — a 403 from one client
    is routine and says nothing about the others."""
    problems = []
    for client in YTDLP_CLIENTS:
        label = client or "default"
        try:
            result = _ytdlp(["-f", "bestaudio[ext=m4a]/bestaudio", "--print", "url",
                             "--print", "%(title)s", "--no-playlist", url], client)
        except subprocess.TimeoutExpired:
            problems.append(f"{label}: timed out")
            continue
        if result.returncode != 0 or not result.stdout.strip():
            problems.append(f"{label}: {_ytdlp_error(result.stderr)}")
            continue
        lines = result.stdout.strip().splitlines()
        stream_url = lines[0]
        rejected = _stream_is_playable(stream_url)
        if rejected:
            problems.append(f"{label}: stream rejected ({rejected})")
            continue
        if client != YTDLP_CLIENTS[0]:
            log(f"yt-dlp: '{YTDLP_CLIENTS[0]}' failed, fell back to '{label}'")
        return stream_url, (lines[1] if len(lines) > 1 else ""), ""
    return "", "", "; ".join(problems)


def _resolve_soundcloud_and_play(url: str, start_time: float, watch: _PlaybackWatch) -> str:
    """Resolve a SoundCloud URL and enqueue it for playback. Returns "" on success, else the reason.

    No player-client fallback here — that is a YouTube concept, and SoundCloud needs no API key:
    yt-dlp lifts a client_id from the public web player."""
    log("resolving soundcloud stream url...")
    try:
        result = _ytdlp(["-f", "bestaudio/best", "--print", "url",
                         "--print", "%(title)s", "--no-playlist", url])
    except subprocess.TimeoutExpired:
        return "timed out"
    if result.returncode != 0 or not result.stdout.strip():
        error = _ytdlp_error(result.stderr)
        log(f"yt-dlp could not resolve {url}: {error}")
        return error
    lines = result.stdout.strip().splitlines()
    stream_url = lines[0]
    if watch.title is None and len(lines) > 1:
        watch.title = lines[1]
    _save_play_history(url, start_time, watch.title)
    log(f"streaming: {stream_url[:80]}...")
    _player.cmd_queue.put(('play', stream_url, None, start_time, url, watch))
    return ""


def _resolve_youtube_and_play(url: str, start_time: float, watch: _PlaybackWatch) -> str:
    """Resolve a YouTube URL and enqueue it for playback. Returns "" on success, else the reason."""
    log("resolving youtube stream url...")
    stream_url, resolved_title, error = _resolve_youtube_stream(url)
    if error:
        log(f"yt-dlp could not resolve {url}: {error}")
        return error
    if watch.title is None and resolved_title:
        watch.title = resolved_title
    _save_play_history(url, start_time, watch.title)
    log(f"streaming: {stream_url[:80]}...")
    _player.cmd_queue.put(('play', stream_url, None, start_time, url, watch))
    return ""


def _download_youtube_and_play(url: str):
    """Download a YouTube video to a temp file via yt-dlp and enqueue it for playback."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    output_template = os.path.join(tmpdir, "audio.%(ext)s")
    log("downloading youtube audio...")
    result = _ytdlp(["-f", "bestaudio[ext=m4a]/bestaudio", "-o", output_template,
                     "--no-playlist", url], YTDLP_CLIENTS[0])
    if result.returncode != 0:
        log(f"yt-dlp error: {_ytdlp_error(result.stderr)}")
        return
    files = os.listdir(tmpdir)
    if not files:
        log("yt-dlp: no output file found")
        return
    audio_file = os.path.join(tmpdir, files[0])
    _player.cmd_queue.put(('play_file', audio_file, tmpdir))


def _play_oneshot_audio_file(path: str) -> threading.Thread:
    """Play a local audio file once in a background thread without affecting the main player.
    Returns the thread, so a caller that must not talk over it can join it."""
    def _run():
        player = mpv.MPV(vid=False, terminal=False,
                         audio_channels="mono" if _mono_output else "auto-safe")
        player.play(path)
        player.wait_for_playback()
        try:
            player.terminate()
        except Exception:
            pass
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _play_random_sound(directory: str) -> threading.Thread | None:
    """Play one audio file picked at random from `directory`. Read on every call, so clips can be
    added or removed without a restart. None if the directory is missing or holds no audio."""
    try:
        clips = [f for f in os.listdir(directory) if f.lower().endswith(SOUND_EXTENSIONS)]
    except FileNotFoundError:
        return None
    if not clips:
        return None
    return _play_oneshot_audio_file(os.path.join(directory, clips[random.randrange(len(clips))]))


def stop_playback():
    """Stop active playback."""
    _player.cmd_queue.put(('stop',))


def shutdown():
    """Signal all background threads to exit cleanly."""
    if ctx is not None:
        ctx.shutdown.set()
        ctx.interrupt.set()
        ring_off(ctx)
    _player.cmd_queue.put(('quit',))


def pause_playback():
    """Pause the current mpv player."""
    _player.cmd_queue.put(('pause',))


def resume_playback():
    """Resume a paused mpv player."""
    _player.cmd_queue.put(('resume',))


def apply_audio_settings(config: dict):
    """Apply the [audio] settings that can change at runtime, so a config save takes effect
    without a restart. mono_output applies to the next track, not the one already playing."""
    global _duck_volume, _mono_output
    audio_cfg = config.get("audio", {})
    _duck_volume = max(0, min(100, int(audio_cfg.get("duck_volume", _duck_volume))))
    _mono_output = bool(audio_cfg.get("mono_output", False))
    log(f"audio settings applied — duck_volume={_duck_volume} mono_output={_mono_output}")


def llm_key_paths() -> list[str]:
    """Config paths of every vendor key, for the confidential list the web ui masks."""
    return [f"llm.keys.{name}" for name, spec in PROVIDERS.items() if spec["needs_key"]]


def migrate_llm_config(config: dict) -> dict:
    """Bring an [llm] block from before vendor choice up to date, in place: the lone anthropic key
    moves under [llm.keys] and bare claude model names gain their vendor prefix. Idempotent, so it
    runs on every load rather than rewriting anyone's config file. Returns `config`."""
    llm_cfg = config.setdefault("llm", {})
    keys = llm_cfg.setdefault("keys", {})
    legacy_key = llm_cfg.pop("anthropic_api_key", None)
    if legacy_key and not keys.get("anthropic"):
        keys["anthropic"] = legacy_key
    for name, spec in PROVIDERS.items():
        if spec["needs_key"]:
            keys.setdefault(name, "")
    llm_cfg.setdefault("provider", DEFAULT_PROVIDER)
    for field_name in ("model", "followup_model"):
        model = llm_cfg.get(field_name, "")
        if model.startswith("claude-"):
            llm_cfg[field_name] = f"anthropic/{model}"
    # an existing confidential list only names the old key — without this the new ones show in clear
    confidential = config.setdefault("meta", {}).setdefault("confidential", [])
    if "llm.anthropic_api_key" in confidential:
        confidential.remove("llm.anthropic_api_key")
    for path in llm_key_paths():
        if path not in confidential:
            confidential.append(path)
    return config


def apply_llm_settings(config: dict):
    """Apply the [llm] settings, so a vendor, model or key change takes effect without a restart."""
    global _llm
    llm_cfg = migrate_llm_config(config)["llm"]
    provider = llm_cfg["provider"]
    spec = PROVIDERS.get(provider, {})  # any other litellm provider works from config alone
    model = llm_cfg.get("model") or spec.get("default_model")
    if not model:
        log(f"llm settings: no model set for provider '{provider}'")
        return
    previous = _llm
    _llm = _LLM(
        provider=provider,
        model=model,
        followup_model=llm_cfg.get("followup_model") or model,
        effort=llm_cfg.get("effort", "low"),
        # empty falls through to litellm's own env lookup (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...)
        api_key=llm_cfg["keys"].get(provider) or None,
        api_base=llm_cfg.get("api_base") or spec.get("default_api_base"),
    )
    if previous is not None and previous.provider != provider and api_messages:
        # another vendor cannot read this one's thinking blocks or tool call ids
        api_messages.clear()
        log("llm vendor changed — conversation context cleared")
    log(f"llm settings applied — provider={provider} model={_llm.model} followup_model={_llm.followup_model} effort={_llm.effort}")


def list_llm_models(provider: str) -> list[str]:
    """Tool-calling chat models litellm knows for `provider`, as "provider/model" strings, for the
    settings ui to offer. The registry is the one bundled with litellm, so it is only as fresh as
    the installed version — the ui still takes any name typed in."""
    models = []
    for name in litellm.models_by_provider.get(provider, []):
        info = litellm.model_cost.get(name, {})
        if info.get("mode") != "chat" or not info.get("supports_function_calling"):
            continue
        if any(word in name for word in LLM_MODEL_EXCLUDE):
            continue
        models.append(name if name.startswith(f"{provider}/") else f"{provider}/{name}")
    return sorted(models)


def test_llm(provider: str, model: str, api_key: str, api_base: str) -> str | None:
    """One tiny completion to prove a key and model work, behind the settings test button. Returns
    the error text, or None when the model answered."""
    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say ok."}],
            max_tokens=16,
            api_key=api_key or None,
            api_base=api_base or PROVIDERS.get(provider, {}).get("default_api_base"),
            drop_params=True,
        )
    except Exception as e:  # anything at all is worth showing the user here
        return f"{type(e).__name__}: {e}"
    return None


def duck_playback():
    """Reduce mpv player volume to _duck_volume (0 = silent, 100 = full)."""
    _player.cmd_queue.put(('duck', _duck_volume))


def unduck_playback():
    """Restore mpv player volume to full after ducking."""
    _player.cmd_queue.put(('unduck',))


def resolve_url(url: str) -> str:
    """Resolve a URL to a direct streamable URL. For YouTube, uses yt-dlp. Other URLs pass through."""
    if "youtube.com" not in url and "youtu.be" not in url:
        return url
    stream_url, _, error = _resolve_youtube_stream(url)
    if error:
        raise RuntimeError(error)
    return stream_url


def _save_play_history(url: str, start_time: float, title: str | None):
    entry: dict = {"url": url, "start_time": start_time}
    if title:
        entry["title"] = title
    _save_history(PLAY_HISTORY_PATH, entry)


def _play(url: str, headers: list[str] | None, start_time: float, title: str | None, retry: bool = False) -> str:
    """Resolve and start `url`, then wait until audio is actually flowing. Returns "playing" or the
    reason it failed.

    YouTube resolution runs inline rather than in a background thread so a rejection reaches the
    caller — resolving off-thread meant a 403 produced silence that nothing in the system noticed.
    The wait after it is for the same reason: a stream can pass the probe and still be refused
    a moment later, once mpv asks for more."""
    watch = _PlaybackWatch(url, start_time, title, retry=retry)
    if "youtube.com" in url or "youtu.be" in url:
        _player.cmd_queue.put(('stop',))
        error = _resolve_youtube_and_play(url, start_time, watch)
    elif "soundcloud.com" in url:
        _player.cmd_queue.put(('stop',))
        error = _resolve_soundcloud_and_play(url, start_time, watch)
    else:
        _save_play_history(url, start_time, title)
        _player.cmd_queue.put(('play', url, headers, start_time, url, watch))
        error = ""
    if error:
        return error
    if not watch.settled.wait(PLAYBACK_CONFIRM_TIMEOUT):
        log(f"playback not confirmed within {PLAYBACK_CONFIRM_TIMEOUT:.0f}s, assuming it is buffering")
    if watch.failure:
        log(f"playback failed on start: {watch.failure}")
        return watch.failure
    return "playing"


def play_url(url: str, headers: list[str] | None = None, start_time: float = 0.0, title: str | None = None) -> str:
    """Stream audio from `url`. Returns "playing" once audio is flowing, or a failure reason."""
    return _play(url, headers, start_time, title)


def _search(search_spec: str, url_template: str) -> str:
    """Run a yt-dlp search and return a JSON list of title/url/duration/channel results."""
    try:
        result = _ytdlp([search_spec, "--dump-json", "--flat-playlist", "--no-download"])
    except subprocess.TimeoutExpired:
        return "search timed out"
    results = []
    for line in result.stdout.strip().splitlines():
        if line.startswith("{"):
            item = json.loads(line)
            results.append({
                "title": item.get("title"),
                "url": item.get("url") or url_template.format(id=item.get("id")),
                "duration": item.get("duration"),
                "channel": item.get("channel") or item.get("uploader"),
            })
    # An empty result set and a rejected request look identical to the caller otherwise, and the
    # agent needs to tell them apart to decide between rephrasing and trying another source.
    if not results:
        if result.returncode != 0:
            return f"search failed: {_ytdlp_error(result.stderr)}"
        return "no results found"
    return json.dumps(results)


def search_youtube(query: str, max_results: int = 5) -> str:
    """Search YouTube for `query` and return a JSON list of title/url/duration/channel results."""
    return _search(f"ytsearch{max_results}:{query}", "https://www.youtube.com/watch?v={id}")


def search_soundcloud(query: str, max_results: int = 5) -> str:
    """Search SoundCloud for `query` and return a JSON list of title/url/duration/channel results."""
    return _search(f"scsearch{max_results}:{query}", "https://soundcloud.com/{id}")


def web_search(query: str, max_results: int = WEB_SEARCH_RESULTS) -> str:
    """Search the web via DuckDuckGo and return a JSON list of title/url/snippet results. The
    fallback for vendors whose own search cannot run alongside our tools — needs no key."""
    from ddgs import DDGS  # only the fallback path pays for the import
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"search failed: {e}"
    if not hits:
        return "no results found"
    return json.dumps([{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits])


def _run_output(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _windows_volume_endpoint():
    from pycaw.pycaw import AudioUtilities
    speakers = AudioUtilities.GetSpeakers()
    # pycaw 2024+ wraps the device and hands the interface over directly; older releases return the
    # raw COM device, which has to be activated by hand
    if hasattr(speakers, "EndpointVolume"):
        return speakers.EndpointVolume
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _get_system_volume() -> tuple[int, bool]:
    """The default output's level (0-100) and whether it is muted."""
    if sys.platform == "linux":
        # "Volume: front-left: 29491 /  45% / -20.81 dB, ..." — channels move together, first will do
        level = int(re.search(r"(\d+)%", _run_output(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])).group(1))
        muted = _run_output(["pactl", "get-sink-mute", "@DEFAULT_SINK@"]).endswith("yes")
    elif sys.platform == "darwin":
        level = int(_run_output(["osascript", "-e", "output volume of (get volume settings)"]))
        muted = _run_output(["osascript", "-e", "output muted of (get volume settings)"]) == "true"
    elif sys.platform == "win32":
        endpoint = _windows_volume_endpoint()
        level = round(endpoint.GetMasterVolumeLevelScalar() * 100)
        muted = bool(endpoint.GetMute())
    else:
        raise RuntimeError(f"volume control is not supported on {sys.platform}")
    return max(0, min(100, level)), muted


def _set_system_volume(level: int | None = None, muted: bool | None = None):
    """Set the default output's level, mute state, or both. Mute is kept apart from the level so
    unmuting brings back the volume from before, rather than whatever 'mute' had set it to."""
    if sys.platform == "linux":
        if level is not None:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"], check=True)
        if muted is not None:
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if muted else "0"], check=True)
    elif sys.platform == "darwin":
        if level is not None:
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=True)
        if muted is not None:
            subprocess.run(["osascript", "-e", f"set volume output muted {'true' if muted else 'false'}"], check=True)
    elif sys.platform == "win32":
        endpoint = _windows_volume_endpoint()
        if level is not None:
            endpoint.SetMasterVolumeLevelScalar(level / 100, None)
        if muted is not None:
            endpoint.SetMute(1 if muted else 0, None)
    else:
        raise RuntimeError(f"volume control is not supported on {sys.platform}")


def _describe_volume(level: int, muted: bool) -> str:
    step = round(level / 100 * VOLUME_STEPS)
    return f"volume {step} of {VOLUME_STEPS} ({level}%)" + (", muted" if muted else "")


def get_volume() -> str:
    try:
        return _describe_volume(*_get_system_volume())
    except Exception as e:
        log(f"get_volume error: {e}")
        return f"error reading volume: {e}"


def set_volume(level: int) -> str:
    """Set an exact percentage. 0 mutes rather than zeroing the level, so unmute has one to restore."""
    level = max(0, min(100, level))
    try:
        if level == 0:
            _set_system_volume(muted=True)
        else:
            _set_system_volume(level=level, muted=False)
        return _describe_volume(*_get_system_volume())
    except Exception as e:
        log(f"set_volume error: {e}")
        return f"error setting volume: {e}"


def change_volume(steps: int) -> str:
    """Move the level by `steps` notches of the VOLUME_STEPS grid, like a volume button. A level set
    by percentage snaps to the nearest notch first, so up then down lands back on a notch."""
    try:
        level, muted = _get_system_volume()
        notch = max(0, min(VOLUME_STEPS, round(level / 100 * VOLUME_STEPS) + steps))
        new_level = round(notch * 100 / VOLUME_STEPS)
        # turning it up unmutes, as on a phone — turning it down leaves a mute alone
        _set_system_volume(level=new_level, muted=False if steps > 0 else None)
        return _describe_volume(*_get_system_volume())
    except Exception as e:
        log(f"change_volume error: {e}")
        return f"error changing volume: {e}"


def set_mute(muted: bool) -> str:
    try:
        level, _ = _get_system_volume()
        if not muted and level == 0:
            # unmuting into silence would sound like it failed
            _set_system_volume(level=round(100 / VOLUME_STEPS), muted=False)
        else:
            _set_system_volume(muted=muted)
        return _describe_volume(*_get_system_volume())
    except Exception as e:
        log(f"set_mute error: {e}")
        return f"error setting mute: {e}"


def set_timer(name: str, seconds: int) -> str:
    def _fire(timer_name: str):
        log(f"timer fired: {timer_name}")
        _play_oneshot_audio_file("sounds/midnight.wav")
        if ctx is not None:
            _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model,
                   f"{timer_name} timer done")
        _timers.remove(timer_name)

    end_time = time.time() + seconds
    t = threading.Timer(seconds, _fire, args=(name,))
    t.daemon = True
    _timers.set(name, t, end_time)
    t.start()

    mins, secs = divmod(seconds, 60)
    duration = f"{mins}m {secs}s" if mins else f"{secs}s"
    return f"{name} timer set for {duration}"


def cancel_timer(name: str) -> str:
    entry = _timers.pop(name)
    if entry:
        entry[0].cancel()
        return f"{name} timer cancelled"
    active = _timers.keys()
    if active:
        return f"no timer named '{name}'. Active timers: {', '.join(active)}"
    return f"no timer named '{name}'"


def list_timers() -> str:
    if not _timers:
        return "no active timers"
    now = time.time()
    parts = []
    for name, (_, end_time) in _timers.items():
        remaining = max(0, int(end_time - now))
        mins, secs = divmod(remaining, 60)
        parts.append(f"{name}: {f'{mins}m {secs}s' if mins else f'{secs}s'} remaining")
    return ", ".join(parts)


def _save_history(path: Path, entry: dict):
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({**entry, "ts": time.time()}) + "\n")



def _load_history(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def get_history(type: str, limit: int = 20) -> str:
    path = CHAT_HISTORY_PATH if type == "chat" else PLAY_HISTORY_PATH
    entries = _load_history(path, limit)
    if not entries:
        return f"no {type} history found"
    if type == "plays":
        now = time.time()
        for entry in entries:
            elapsed = now - entry.get("ts", now)
            entry["inferred_position"] = int(elapsed + entry.get("start_time", 0))
    return json.dumps(entries)


def update(target: str) -> str:
    try:
        log(f"update: fetching tags...")
        subprocess.run(["git", "fetch", "--tags"], check=True, capture_output=True)

        if target == "latest":
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, check=True
            )
            target = result.stdout.strip()
            log(f"update: latest tag is {target}")

        log(f"update: checking out {target}")
        subprocess.run(["git", "checkout", target], check=True, capture_output=True)

        # pull if on a branch
        is_branch = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "HEAD"],
            capture_output=True
        ).returncode == 0
        if is_branch:
            subprocess.run(["git", "pull"], check=True, capture_output=True)

        log("update: installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                       check=True, capture_output=True)

        log(f"update: done, restarting...")
        if ctx is not None:
            _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model,
                   f"Updated to {target}. Restarting now.")

        def _restart():
            if "--service" in sys.argv:
                import getpass
                subprocess.run(["systemctl", "restart", f"oi-speaker@{getpass.getuser()}"])
            else:
                os.execv(sys.executable, [sys.executable] + sys.argv)

        threading.Timer(1.5, _restart).start()
        return f"updated to {target}"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log(f"update error: {err}")
        return f"update failed: {err}"


def _speak_loop(ctx):
    """Background thread: main state machine driving wake detection, recording, and LLM interaction."""
    log("oi!! speak! loop!!")
    _play_oneshot_audio_file("sounds/startup.m4a")

    # training data capture
    wake_audio = None
    buffer_duration = 0.0
    record_dir = None
    if "--record-negatives" in sys.argv:
        log("recording negatives")
        buffer_duration = 4.0
        record_dir = "training/training_data/recordings/false_positives"
        os.makedirs(record_dir, exist_ok=True)
    elif "--record-positives" in sys.argv:
        log("recording positives")
        record_dir = "training/training_data/recordings/positives"
        ctx.speaker_state = SpeakerState.VAD_RECORD
        os.makedirs(record_dir, exist_ok=True)

    # the loop
    transcribed_request = ""
    onset_remaining = ONSET_TIMEOUT
    followup = False  # a turn re-entered via follow_on or an interrupt already has full context
    with dev.InputStream(samplerate=ctx.input_sample_rate, channels=1, dtype='int16', device=ctx.input_dev_index) as stream:
        while not ctx.shutdown.is_set():
            if ctx.speaker_state == SpeakerState.LISTEN_FOR_WAKE:
                ring_startup_stop()  # no-op once the first wake loop has stopped it
                ring_off(ctx)
                wake_audio = _listen_for_wake(
                    ctx.wake_model,
                    stream,
                    ctx.input_sample_rate,
                    threshold=ctx.wake_threshold,
                    num_triggers=ctx.wake_triggers,
                    buffer_duration=buffer_duration
                )
                _perf_timer.start()
                onset_remaining = ONSET_TIMEOUT
                followup = False
                if record_dir:
                    filepath = f"{record_dir}/{int(time.time() * 1000)}.wav"
                    log(f"caching {filepath}")
                    _write_wav(wake_audio, TARGET_SAMPLE_RATE, filepath)
                    ctx.speaker_state = SpeakerState.LISTEN_FOR_WAKE
                else:
                    ring_wake(ctx)
                    _play_oneshot_audio_file("sounds/wake.wav")
                    ctx.speaker_state = SpeakerState.RECORDING
            elif ctx.speaker_state == SpeakerState.RECORDING:
                log(f"recording (onset budget: {onset_remaining:.1f}s)")
                mark("record", f"onset budget {onset_remaining:.1f}s")
                ring_recording(ctx)
                duck_playback()
                # nudge once, halfway through a wake's onset budget. the budget only shrinks — across
                # false onsets too — so once past halfway this stays None. not on a follow-up: the
                # assistant has just asked its own question
                to_halfway = onset_remaining - ONSET_TIMEOUT / 2
                nudge_after = to_halfway if to_halfway > 0 and not followup else None
                audio_request, onset_elapsed = _record_until_silence(
                    ctx.vad, stream, ctx.input_sample_rate, onset_timeout=onset_remaining, nudge_after=nudge_after)
                transcribed_request = ""
                if audio_request is not None:
                    _perf_timer.lap("recorded")
                    mark("transcribe", "")
                    ring_transcribing(ctx)
                    transcribed_request = _transcribe_audio(ctx.whisper_model, audio_request, ctx.worker_url)
                    _perf_timer.lap("transcribed")
                    mark("transcribed", transcribed_request.strip())
                # nothing usable captured — keep listening on the remaining budget rather than dropping
                # the turn, so a false onset or a discarded noise clip does not send us back to the wake word
                if not transcribed_request.strip():
                    onset_remaining -= onset_elapsed
                    log(f"no request captured, onset budget remaining: {onset_remaining:.1f}s")
                    if onset_remaining > 0.5:
                        ctx.speaker_state = SpeakerState.RECORDING
                    else:
                        log("onset budget exhausted, returning to wake listen")
                        unduck_playback()
                        onset_remaining = ONSET_TIMEOUT
                        ctx.speaker_state = SpeakerState.RESET
                    continue
                onset_remaining = ONSET_TIMEOUT
                log(transcribed_request)
                chat_history.append({"role": "user", "text": transcribed_request})
                _save_history(CHAT_HISTORY_PATH, {"role": "user", "text": transcribed_request})
                ctx.speaker_state = SpeakerState.LLM_AGENT
            elif ctx.speaker_state == SpeakerState.LLM_AGENT:
                mark("llm", transcribed_request.strip())
                ring_llm_agent(ctx)
                # let the user know they were heard — the first spoken sentence can be seconds away
                ack = _play_random_sound(ACK_SOUND_DIR)
                ctx.speaker_state = query_llm(transcribed_request, stream, followup=followup, speak_after=ack)
                _perf_timer.lap("llm")
                mark("llm-done", "")
                if ctx.speaker_state == SpeakerState.RECORDING:
                    onset_remaining = ONSET_TIMEOUT
                    followup = True
                    _flush_stream(stream, ctx.input_sample_rate)
                    ctx.wake_model.reset()
                else:
                    unduck_playback()
            elif ctx.speaker_state == SpeakerState.RESET:
                _flush_stream(stream, ctx.input_sample_rate)
                ctx.wake_model.reset()
                ctx.speaker_state = SpeakerState.LISTEN_FOR_WAKE
                log("return to listen for wake")
            elif ctx.speaker_state == SpeakerState.VAD_RECORD:
                ring_recording(ctx)
                audio = _vad_record(ctx.vad, stream, ctx.input_sample_rate)
                filepath = f"{record_dir}/{int(time.time() * 1000)}.wav"
                log(f"caching {filepath}")
                _write_wav(audio, ctx.input_sample_rate, filepath, gain=1.0)


def _start_worker():
    """Initialise only the inference models and serve RPC — no audio I/O, no background threads."""
    global ctx
    if ctx is not None:
        return
    log("oi! worker mode!!")
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    inf = config.get("inference", {})
    device = inf.get("device", "cpu")
    compute = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel(
        inf.get("whisper_model", "small"),
        device=device,
        compute_type=inf.get("whisper_compute", compute),
    )
    piper_model = config.get("voice", {}).get("model", "models/piper/en_GB-northern_english_male-medium.onnx")
    voice_model = PiperVoice.load(piper_model)
    ctx = SpeakerContext(
        system="",
        voice_model=voice_model,
        whisper_model=whisper_model,
        vad=None,
        output_dev_index=0,
        output_sample_rate=voice_model.config.sample_rate,
        wake_model=None,
        input_dev_index=0,
        input_sample_rate=16000,
        worker_mode=True,
    )
    log("worker ready")


def open_ring():
    """Open the mic array's LED ring, or return None when this box hasn't got one.

    The import is deferred so a platform without pyusb or a libusb backend still starts.
    """
    try:
        from xvf3800 import XVF3800
        ring = XVF3800.open()
        if ring is None:
            log("ring: no XVF3800 found")
        else:
            log(f"ring: XVF3800 firmware {'.'.join(str(v) for v in ring.version())}")
        return ring
    except Exception as e:
        log(f"ring: unavailable ({e})")
        return None


def _ring_call(fn, *args):
    """The ring is cosmetic — a usb hiccup must never take the state machine down."""
    try:
        fn(*args)
    except Exception as e:
        log(f"ring: {e}")


def ring_startup(ring):
    """Flash mauve while the models load. Runs on its own thread until ring_startup_stop().

    Takes the ring rather than ctx — startup runs before there is a SpeakerContext.
    """
    global _ring_startup_thread
    if ring is None:
        return

    def _flash():
        lit = True
        while True:
            _ring_call(ring.solid, RING_COLOR_TRANSCRIBING if lit else 0x000000)
            lit = not lit
            if _ring_startup_stop.wait(RING_FLASH_INTERVAL):
                break
        _ring_call(ring.off)

    _ring_startup_stop.clear()
    _ring_startup_thread = threading.Thread(target=_flash, daemon=True)
    _ring_startup_thread.start()


def ring_startup_stop():
    """Stop the startup flash. Joins the thread so it can't repaint over the next state."""
    global _ring_startup_thread
    if _ring_startup_thread is None:
        return
    _ring_startup_stop.set()
    _ring_startup_thread.join(timeout=1.0)
    _ring_startup_thread = None


def ring_off(ctx):
    """Idle — nothing to show."""
    if ctx.ring is None:
        return
    _ring_call(ctx.ring.off)


def ring_wake(ctx):
    """Wake word heard."""
    if ctx.ring is None:
        return
    _ring_call(ctx.ring.solid, RING_COLOR_WAKE)


def ring_recording(ctx):
    """Capturing the request."""
    if ctx.ring is None:
        return
    _ring_call(ctx.ring.solid, RING_COLOR_RECORDING)


def ring_transcribing(ctx):
    """Whisper is running on the captured audio."""
    if ctx.ring is None:
        return
    _ring_call(ctx.ring.solid, RING_COLOR_TRANSCRIBING)


def ring_llm_agent(ctx):
    """Agent loop: thinking, calling tools, speaking."""
    if ctx.ring is None:
        return
    _ring_call(ctx.ring.breathe, RING_COLOR_LLM, 1)


def start():
    """Initialise all models and devices from config.toml and launch the background threads."""
    global ctx

    # Derive from sys.argv directly — immune to the double-import problem where
    # python src/speaker.py runs as __main__ but web.py imports a second speaker instance.
    worker_mode = "--worker" in sys.argv
    worker_url: str | None = None
    if "--worker-ip" in sys.argv:
        log("oi! starting with help from worker")
        idx = sys.argv.index("--worker-ip")
        ip_arg = sys.argv[idx + 1]
        worker_url = f"http://{ip_arg}" if ":" in ip_arg else f"http://{ip_arg}:8000"

    if worker_mode:
        _start_worker()
        return

    log("oi! start!!")

    if ctx is not None:
        return

    # opened up front, before the models load, so the startup flash covers the whole wait
    ring = open_ring()
    ring_startup(ring)

    chat_history.extend(_load_history(CHAT_HISTORY_PATH, HISTORY_LOAD_LIMIT))

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        if "--verbose" in sys.argv:
            log(json.dumps(config, indent=4))

    with open("system.json", "rb") as f:
        system = json.load(f)["system"]
        if "--verbose" in sys.argv:
            log(json.dumps(system, indent=4))

    hints = config.get("hints", [])
    if hints:
        system += "\n\n# Hints\n"
        for hint in hints:
            value = hint.get("url") or hint.get("text", "")
            extra = hint.get("extra_info", "")
            suffix = f" ({extra})" if extra else ""
            system += f"- [{hint['category']}] {hint['name']}: {value}{suffix}\n"

    apply_llm_settings(config)

    # whisper / inference config
    inf = config.get("inference", {})
    device = inf.get("device", "cpu")
    compute = "int8"
    if device == "cuda":
        compute = "float16"
    vad_threshold = float(inf.get("vad_threshold", 0.5))
    vad_mic_gain = float(inf.get("vad_mic_gain", 1.0))
    vad_verbose = "--verbose" in sys.argv
    _voice_probe.mic_gain = vad_mic_gain  # match what the recording VAD hears, or the two disagree
    global _silence_timeout
    _silence_timeout = float(inf.get("silence_timeout", _silence_timeout))

    # audio config
    if "--verbose" in sys.argv:
        log(json.dumps(enumerate_audio_devices(), indent=4))
    audio_cfg = config["audio"]
    apply_audio_settings(config)
    input_dev_info = _get_audio_device_index(audio_cfg["input_device"])
    output_dev_info = _get_audio_device_index(audio_cfg["output_device"])

    ctx = SpeakerContext(
        system=system,
        wake_model=Model(
            inference_framework="onnx",
            wakeword_models=["models/openwakeword/oi_speaker.onnx"],
            vad_threshold=0.5,
            enable_speex_noise_suppression=False
        ),
        voice_model=PiperVoice.load("models/piper/en_GB-northern_english_male-medium.onnx"),
        whisper_model=WhisperModel(
            inf.get("whisper_model", "small"),
            device=device,
            compute_type=inf.get("whisper_compute", compute)
        ),
        vad=_SileroVAD(threshold=vad_threshold, mic_gain=vad_mic_gain, verbose=vad_verbose),
        input_dev_index=int(input_dev_info['index']),
        input_sample_rate=int(input_dev_info['default_samplerate']),
        output_dev_index=int(output_dev_info['index']),
        output_sample_rate=int(output_dev_info['default_samplerate']),
        worker_url=worker_url,
        ring=ring,
    )

    threading.Thread(
        target=_player_loop,
        daemon=True
    ).start()

    threading.Thread(
        target=_speak_loop,
        args=(ctx,),
        daemon=True
    ).start()



def main():
    """Entry point: start the speaker and serve the web UI via uvicorn."""
    log("oi! oi!!")

    if "--enum-audio" in sys.argv:
        log(json.dumps(enumerate_audio_devices(), indent=4))
        return

    sys.path.insert(0, str(__file__).replace("speaker.py", ""))
    from web import app
    with open("config.toml", "rb") as _f:
        _cfg = tomllib.load(_f)
    _port = int(_cfg.get("network", {}).get("port", 8000))
    uvicorn.run(app, host="0.0.0.0", port=_port)
    os._exit(0)


if __name__ == "__main__":
    main()