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
import webrtcvad
import math
import faster_whisper
import anthropic
import tomllib
import json
import requests
import wave
import uvicorn
import os

from pathlib import Path
from piper.voice import PiperVoice
from openwakeword.model import Model
from faster_whisper import WhisperModel
from dataclasses import dataclass
from enum import Enum, auto

# TODO:
# update function, check git and auto update

TARGET_SAMPLE_RATE = 16000 # 16khz
WAKE_CHUNK = 1280 # 80ms at 16khz
VAD_CHUNK = 480 # 30ms at 16kHz

HISTORY_DIR = Path(".history") 
CHAT_HISTORY_PATH = HISTORY_DIR / "chat.jsonl"
PLAY_HISTORY_PATH = HISTORY_DIR / "plays.jsonl"
PLAY_POSITIONS_PATH = HISTORY_DIR / "positions.json"
HISTORY_LOAD_LIMIT = 30  # messages loaded into context on startup

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

TOOLS = [
    {
        "name": "search_youtube",
        "description": "Search for music or videos on YouTube. Returns a list of results to choose from.",
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
        "description": "Stream audio from a URL via mpv. Use start_time to resume from a saved position (check play history for saved positions).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to play"},
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
        "type": "web_search_20250305",
        "name": "web_search"
    },
    {
        "name": "set_volume",
        "description": "Set the system output volume. Use 0 to mute, 1-100 for a percentage level.",
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "description": "Volume level 0 (mute) to 100"}
            },
            "required": ["level"]
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
        "name": "chat",
        "description": "Speak a plain-text response aloud via text-to-speech.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Plain text only, no markdown"},
                "follow_on": {"type": "boolean", "description": "Keep listening for user response after speaking"}
            },
            "required": ["text"]
        }
    }
]


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
    llm_client: any
    system: str
    voice_model: any
    whisper_model: any
    output_dev_index: int
    output_sample_rate: int
    wake_model: any
    input_dev_index: int
    input_sample_rate: int


ctx: SpeakerContext | None = None
chat_history: list[dict] = []  # {"role": "user"|"assistant", "text": str}
speaker_state: SpeakerState = SpeakerState.LISTEN_FOR_WAKE
_player_cmd_queue: queue.Queue = queue.Queue()
_player_active: bool = False
_current_play_url: str | None = None
_play_positions: dict[str, float] = {}  # original url -> last position (seconds)
_worker_mode: bool = False   # this instance serves RPC only, no local audio
_worker_url: str | None = None  # delegate transcription/TTS to this worker base URL
_interrupt: threading.Event = threading.Event()
_shutdown: threading.Event = threading.Event()
_perf_timer = _PerfTimer()
_timers: dict[str, tuple[threading.Timer, float]] = {}  # name -> (timer, end_time)
_timers_lock = threading.Lock()
_log_buffer: deque = deque(maxlen=2000)
_log_counter: int = 0
_log_lock = threading.Lock()


def log(text: str):
    """Print to stdout and append to the web-accessible log buffer."""
    global _log_counter
    print(text)
    with _log_lock:
        _log_buffer.append({"index": _log_counter, "text": text, "ts": time.time()})
        _log_counter += 1


def get_log_lines(since: int = 0) -> dict:
    with _log_lock:
        lines = [l for l in _log_buffer if l["index"] > since]
        total = _log_counter
    return {"lines": lines, "total": total}


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


def _wait_for_silence(stream, sample_rate: int, threshold: float = 800.0, settle: float = 0.6, timeout: float = 3.0):
    """Read input audio until RMS stays below `threshold` for `settle` seconds (or `timeout` expires).
    Called after pausing playback so speaker bleed dies down before VAD starts."""
    sample_ratio = int(math.ceil(sample_rate / TARGET_SAMPLE_RATE))
    chunk = VAD_CHUNK * sample_ratio
    chunk_duration = VAD_CHUNK / TARGET_SAMPLE_RATE
    silent_duration = 0.0
    elapsed = 0.0
    while elapsed < timeout:
        raw, _ = stream.read(chunk)
        rms = float(np.sqrt(np.mean(np.square(np.squeeze(raw).astype(np.float32)))))
        elapsed += chunk_duration
        if rms < threshold:
            silent_duration += chunk_duration
            if silent_duration >= settle:
                return
        else:
            silent_duration = 0.0


def _test_tone(dev, dev_index: int, sample_rate: int):
    """Play a 440 Hz test tone on the given output device."""
    # generate a simple 440hz test tone
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    dev.play(tone, samplerate=sample_rate, device=dev_index)
    dev.wait()


def _resample_audio(audio, orig_rate: int, target_rate: int):
    """Linearly interpolate `audio` from `orig_rate` to `target_rate`, returning int16."""
    ratio = target_rate / orig_rate
    new_length = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio), new_length),
        np.arange(len(audio)),
        audio
    ).astype(np.int16)


def _read_audio(stream, chunk_size16: int, sample_rate: int) -> np.ndarray:
    """Read one chunk from `stream` at native `sample_rate` and resample to 16 kHz int16."""
    # read audio
    sample_ratio = int(math.ceil(sample_rate / TARGET_SAMPLE_RATE))
    chunk = chunk_size16 * sample_ratio
    audio, _ = stream.read(chunk)
    # resample to TARGET_SAMPLE_RATE and flatten
    if sample_ratio != 1:
        audio_flat = np.squeeze(audio)
        audio_flat = _resample_audio(audio_flat, sample_rate, TARGET_SAMPLE_RATE)
        audio_flat = audio_flat[:chunk_size16]
    else:
        audio_flat = np.squeeze(audio)
    return audio_flat


def _vad_record(vad, stream, sample_rate: int, silence_timeout: float = 1.5) -> np.ndarray:
    """Record raw audio at native sample rate, using resampled audio only for VAD checks."""
    sample_ratio = int(math.ceil(sample_rate / TARGET_SAMPLE_RATE))
    raw_chunk = VAD_CHUNK * sample_ratio
    raw_chunks = []
    # wait for speech onset
    while True:
        raw, _ = stream.read(raw_chunk)
        raw_flat = np.squeeze(raw)
        audio_16k = _resample_audio(raw_flat, sample_rate, TARGET_SAMPLE_RATE)[:VAD_CHUNK]
        if vad.is_speech(audio_16k.tobytes(), TARGET_SAMPLE_RATE):
            raw_chunks.append(raw_flat)
            break
    # record until silence
    silent_duration = 0.0
    while silent_duration < silence_timeout:
        raw, _ = stream.read(raw_chunk)
        raw_flat = np.squeeze(raw)
        raw_chunks.append(raw_flat)
        audio_16k = _resample_audio(raw_flat, sample_rate, TARGET_SAMPLE_RATE)[:VAD_CHUNK]
        silent_duration = 0.0 if vad.is_speech(audio_16k.tobytes(), TARGET_SAMPLE_RATE) else silent_duration + VAD_CHUNK / TARGET_SAMPLE_RATE
    return np.concatenate(raw_chunks)


def _record_until_silence(vad, stream, sample_rate: int, silence_timeout: float=1.0, onset_timeout: float=8.0) -> tuple[np.ndarray | None, float]:
    """Wait for speech onset then record until `silence_timeout` seconds of continuous silence.
    Returns (audio, onset_elapsed). audio is None if no speech begins within `onset_timeout` seconds."""
    chunks = []
    silent_duration = 0.0
    speech_started = False
    onset_elapsed = 0.0
    chunk_duration = VAD_CHUNK / TARGET_SAMPLE_RATE
    log_limiter = 0
    while True:
        audio_flat = _read_audio(stream, VAD_CHUNK, sample_rate)
        chunks.append(audio_flat)
        is_speech = vad.is_speech(audio_flat.tobytes(), TARGET_SAMPLE_RATE)
        if is_speech:
            speech_started = True
            silent_duration = 0.0
        elif speech_started:
            silent_duration += chunk_duration
            if silent_duration >= silence_timeout:
                break
        else:
            if int(onset_elapsed) > log_limiter:
                log_limiter = int(onset_elapsed)
                log(f"waiting for onset: {log_limiter}s")
            onset_elapsed += chunk_duration
            if onset_elapsed >= onset_timeout:
                log("no speech detected, returning to wake listen")
                return None, onset_elapsed

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
        prediction = wake_model.predict(audio_flat)
        for _, score in prediction.items():
            score_window.append(score > threshold)
            hits = sum(score_window)
            if score > 0.0:
                log(f"score: {score:.3f} ({hits}/{num_triggers})")
            if hits >= num_triggers:
                log("ello mate!")
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



def _transcribe_audio(whisper_model, audio: np.ndarray) -> str:
    """Transcribe int16 audio to English text using Whisper, or delegate to worker."""
    global _worker_url
    if _worker_url:
        import io as _io
        buf = _io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        try:
            resp = requests.post(
                f"{_worker_url}/rpc/transcribe",
                files={"audio": ("audio.wav", buf.getvalue(), "audio/wav")},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
        except Exception as e:
            log(f"worker transcribe error: {e}")
            return ""
    audio_float = audio.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio_float, language="en", beam_size=1)
    return " ".join(segment.text for segment in segments)


def _execute_tool(tool_name: str, tool_input: dict) -> tuple[str, SpeakerState | None]:
    """Dispatch an LLM tool call and return its result string and optional next state."""
    if tool_name == "search_youtube":
        return search_youtube(tool_input["query"]), None
    elif tool_name == "play_url":
        play_url(tool_input["url"], tool_input.get("headers"), tool_input.get("start_time", 0.0))
        return "playing", SpeakerState.RESET
    elif tool_name == "stop":
        stop_playback()
        return "stopped", SpeakerState.RESET
    elif tool_name == "chat":
        chat_history.append({"role": "assistant", "text": tool_input["text"]})
        _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model, tool_input["text"])
        next_state = SpeakerState.RECORDING if tool_input.get("follow_on") else SpeakerState.RESET
        return "spoken", next_state
    elif tool_name == "set_volume":
        return set_volume(tool_input["level"]), None
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


def query_llm(llm_client, system: str, text: str, mic_stream=None) -> SpeakerState:
    """Send `text` to the LLM, stream TTS as sentences arrive, handle tool calls, and return the next state."""
    global _interrupt
    _interrupt.clear()

    _history_start = len(chat_history)
    stop_flag = threading.Event()
    wake_thread = None
    if mic_stream is not None:
        ctx.wake_model.reset()  # clear residual activation from the wake that triggered this session
        wake_thread = threading.Thread(
            target=_interrupt_wake_listen,
            args=(ctx.wake_model, mic_stream, ctx.input_sample_rate, stop_flag),
            daemon=True
        )
        wake_thread.start()
    try:
        messages = [{"role": "user", "content": text}]
        next_state = SpeakerState.RESET

        while True:
            accumulated_text = ""
            sentence_buffer = ""
            _live_entry = None  # mutable chat_history entry updated live during streaming

            with llm_client.messages.stream(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages
            ) as stream:
                for delta in stream.text_stream:
                    if _interrupt.is_set():
                        break
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
                            _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model, s)
                    if _interrupt.is_set():
                        return SpeakerState.RESET

                if _interrupt.is_set():
                    return SpeakerState.RESET

                if sentence_buffer.strip():
                    _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model, sentence_buffer)

                if _interrupt.is_set():
                    return SpeakerState.RESET

                final_message = stream.get_final_message()

            if final_message.stop_reason == "end_turn":
                # Remove live entry if nothing was streamed
                if _live_entry is not None and not accumulated_text.strip():
                    chat_history.remove(_live_entry)
                return next_state

            if final_message.stop_reason == "pause_turn":
                # server-side tool (web search) hit iteration limit — re-send to continue
                messages.append({"role": "assistant", "content": final_message.content})
                continue

            if final_message.stop_reason == "tool_use":
                # Remove live entry if nothing was streamed (chat tool will add its own entry)
                if _live_entry is not None and not accumulated_text.strip():
                    chat_history.remove(_live_entry)
                messages.append({"role": "assistant", "content": final_message.content})
                tool_results = []
                for block in final_message.content:
                    if block.type == "tool_use":
                        log(f"\ttool: {block.name} {block.input}")
                        _perf_timer.lap(block.name, indent="\t")
                        result, state = _execute_tool(block.name, block.input)
                        if state is not None:
                            next_state = state
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                messages.append({"role": "user", "content": tool_results})

                # Terminal actions don't need another LLM round-trip
                if next_state == SpeakerState.RESET and any(
                    block.type == "tool_use" and block.name in ("play_url", "stop", "chat")
                    for block in final_message.content
                ):
                    return next_state

    finally:
        stop_flag.set()
        if wake_thread is not None:
            wake_thread.join(timeout=0.5)
        _interrupt.clear()
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
    """Synthesise `text` to audio and play it, stopping early if `_interrupt` is set."""
    text = _strip_markdown(text)
    if not text:
        return
    if _interrupt.is_set():
        return
    if _worker_url:
        import io as _io
        try:
            resp = requests.post(
                f"{_worker_url}/rpc/speak",
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
            if _interrupt.is_set():
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
            if _interrupt.is_set():
                dev.stop()
                return


def _interrupt_wake_listen(wake_model, stream, input_sample_rate: int, stop_flag: threading.Event):
    """Poll the wake model in a background thread and set `_interrupt` if the wake word is detected during TTS."""
    score_window = deque(maxlen=5)
    while not stop_flag.is_set():
        audio_flat = _read_audio(stream, WAKE_CHUNK, input_sample_rate)
        prediction = wake_model.predict(audio_flat)
        for _, score in prediction.items():
            score_window.append(score > 0.9)
            if sum(score_window) >= 3:
                log("interrupt: wake word during TTS")
                _interrupt.set()
                return


def _player_loop():
    """Background thread: consume commands from `_player_cmd_queue` and drive the mpv player."""
    log("oi!! player! loop!!")

    player: mpv.MPV | None = None
    cleanup_dir: str | None = None

    def _stop():
        global _player_active, _current_play_url
        nonlocal player, cleanup_dir
        if player is not None:
            try:
                pos = player.time_pos
                if pos is not None and _current_play_url:
                    _play_positions[_current_play_url] = pos
                    _save_positions()
            except Exception:
                pass
            try:
                player.terminate()
            except Exception:
                pass
            player = None
            _current_play_url = None
        if cleanup_dir is not None:
            try:
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            except Exception:
                pass
            cleanup_dir = None
        _player_active = False

    while True:
        try:
            cmd, *args = _player_cmd_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if cmd == 'play':
            global _player_active, _current_play_url
            _stop()
            url, headers, start_time, original_url = args[0], args[1], args[2], args[3]
            log(f"play_url: {url[:80]}...")
            player = mpv.MPV(vid=False, terminal=False,
                             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                             log_handler=lambda level, component, message: print(f"[mpv/{component}] {level}: {message}"),
                             loglevel="warn")
            if start_time:
                player['start'] = f'+{int(start_time)}'
            if headers:
                for header in headers:
                    player.command("change-list", "http-header-fields", "append", header)
            player.play(url)
            _current_play_url = original_url
            _player_active = True

        elif cmd == 'play_file':
            _stop()
            path, cleanup_dir = args[0], args[1]
            log(f"playing: {path}")
            player = mpv.MPV(vid=False, terminal=False,
                             log_handler=lambda level, component, message: print(f"[mpv/{component}] {level}: {message}"),
                             loglevel="warn")
            player.play(path)
            _player_active = True

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

        elif cmd == 'quit':
            _stop()
            break


def _resolve_youtube_and_play(url: str, start_time: float = 0.0):
    """Resolve a YouTube URL to a direct stream URL via yt-dlp and enqueue it for playback."""
    log("resolving youtube stream url...")
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio",
         "--get-url", "--no-playlist", "--extractor-args", "youtube:player_client=tv_embedded", url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"yt-dlp error: {result.stderr}")
        return
    stream_url = result.stdout.strip().splitlines()[0]
    if not stream_url:
        log("yt-dlp: no stream url found")
        return
    log(f"streaming: {stream_url[:80]}...")
    _player_cmd_queue.put(('play', stream_url, None, start_time, url))


def _download_youtube_and_play(url: str):
    """Download a YouTube video to a temp file via yt-dlp and enqueue it for playback."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    output_template = os.path.join(tmpdir, "audio.%(ext)s")
    log("downloading youtube audio...")
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", output_template,
         "--no-playlist", "--extractor-args", "youtube:player_client=tv_embedded", url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"yt-dlp error: {result.stderr}")
        return
    files = os.listdir(tmpdir)
    if not files:
        log("yt-dlp: no output file found")
        return
    audio_file = os.path.join(tmpdir, files[0])
    _player_cmd_queue.put(('play_file', audio_file, tmpdir))


def _play_oneshot_audio_file(path: str):
    """Play a local audio file once in a background thread without affecting the main player."""
    def _run():
        player = mpv.MPV(vid=False, terminal=False)
        player.play(path)
        player.wait_for_playback()
        try:
            player.terminate()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def stop_playback():
    """Stop active playback and signal any in-progress TTS to interrupt."""
    _interrupt.set()
    _player_cmd_queue.put(('stop',))


def shutdown():
    """Signal all background threads to exit cleanly."""
    _shutdown.set()
    _interrupt.set()
    _player_cmd_queue.put(('quit',))


def pause_playback():
    """Pause the current mpv player."""
    _player_cmd_queue.put(('pause',))


def resume_playback():
    """Resume a paused mpv player."""
    _player_cmd_queue.put(('resume',))


def resolve_url(url: str) -> str:
    """Resolve a URL to a direct streamable URL. For YouTube, uses yt-dlp. Other URLs pass through."""
    if "youtube.com" not in url and "youtu.be" not in url:
        return url
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio",
         "--get-url", "--no-playlist", "--extractor-args", "youtube:player_client=tv_embedded", url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    stream_url = result.stdout.strip().splitlines()[0]
    if not stream_url:
        raise RuntimeError("yt-dlp returned no URL")
    return stream_url


def play_url(url: str, headers: list[str] | None = None, start_time: float = 0.0):
    """Stream audio from `url`, using yt-dlp resolution for YouTube URLs on Windows."""
    _save_history(PLAY_HISTORY_PATH, {"url": url, "start_time": start_time})
    is_youtube = "youtube.com" in url or "youtu.be" in url
    if is_youtube:
        threading.Thread(target=_resolve_youtube_and_play, args=(url, start_time), daemon=True).start()
    else:
        _player_cmd_queue.put(('play', url, headers, start_time, url))


def search_youtube(query: str, max_results: int = 5) -> str:
    """Search YouTube for `query` and return a JSON list of title/url/duration/channel results."""
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", f"ytsearch{max_results}:{query}", "--dump-json", "--flat-playlist", "--no-download"],
        capture_output=True, text=True
    )
    results = []
    for line in result.stdout.strip().splitlines():
        if line:
            item = json.loads(line)
            results.append({
                "title": item.get("title"),
                "url": item.get("url") or f"https://www.youtube.com/watch?v={item.get('id')}",
                "duration": item.get("duration"),
                "channel": item.get("channel") or item.get("uploader"),
            })
    return json.dumps(results)


def set_volume(level: int) -> str:
    level = max(0, min(100, level))
    try:
        if sys.platform == "linux":
            if level == 0:
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], check=True)
            else:
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], check=True)
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=True)
        elif sys.platform == "win32":
            # Use nircmd if available, otherwise ctypes
            result = subprocess.run(["nircmd", "setsysvolume", str(int(level / 100 * 65535))],
                                    capture_output=True)
            if result.returncode != 0:
                import ctypes
                from ctypes import POINTER, cast
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                if level == 0:
                    volume.SetMute(1, None)
                else:
                    volume.SetMute(0, None)
                    volume.SetMasterVolumeLevelScalar(level / 100, None)
        return f"volume set to {level}"
    except Exception as e:
        log(f"set_volume error: {e}")
        return f"error setting volume: {e}"


def set_timer(name: str, seconds: int) -> str:
    def _fire(timer_name: str):
        log(f"timer fired: {timer_name}")
        _play_oneshot_audio_file("sounds/midnight.wav")
        if ctx is not None:
            _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model,
                   f"{timer_name} timer done")
        with _timers_lock:
            _timers.pop(timer_name, None)

    with _timers_lock:
        existing = _timers.pop(name, None)
        if existing:
            existing[0].cancel()
        end_time = time.time() + seconds
        t = threading.Timer(seconds, _fire, args=(name,))
        t.daemon = True
        _timers[name] = (t, end_time)
        t.start()

    mins, secs = divmod(seconds, 60)
    duration = f"{mins}m {secs}s" if mins else f"{secs}s"
    return f"{name} timer set for {duration}"


def cancel_timer(name: str) -> str:
    with _timers_lock:
        entry = _timers.pop(name, None)
    if entry:
        entry[0].cancel()
        return f"{name} timer cancelled"
    active = list(_timers.keys())
    if active:
        return f"no timer named '{name}'. Active timers: {', '.join(active)}"
    return f"no timer named '{name}'"


def list_timers() -> str:
    with _timers_lock:
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


def _save_positions():
    PLAY_POSITIONS_PATH.parent.mkdir(exist_ok=True)
    PLAY_POSITIONS_PATH.write_text(json.dumps(_play_positions), encoding="utf-8")


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
        for entry in entries:
            pos = _play_positions.get(entry.get("url", ""))
            if pos is not None:
                entry["saved_position"] = round(pos)
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

        if sys.platform == "linux":
            import getpass
            subprocess.Popen(["systemctl", "restart", f"oi-speaker@{getpass.getuser()}"])
        else:
            # on non-Linux, exit and let the user restart
            threading.Timer(1.0, lambda: os._exit(0)).start()

        return f"updated to {target}"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log(f"update error: {err}")
        return f"update failed: {err}"


def _speak_loop(wake_model, vad, whisper_model, input_dev_index, input_sample_rate):
    """Background thread: main state machine driving wake detection, recording, and LLM interaction."""
    log("oi!! speak! loop!!")
    _play_oneshot_audio_file("sounds/startup.m4a")
    global speaker_state

    # training data capture
    wake_audio = None
    buffer_duration = 0.0
    threshold = 0.4
    triggers = 1
    record_dir = None
    if "--record-negatives" in sys.argv:
        log("recording negatives")
        buffer_duration = 4.0
        record_dir = "training/training_data/recordings/false_positives"
        os.makedirs(record_dir, exist_ok=True)
    elif "--record-positives" in sys.argv:
        log("recording positives")
        record_dir = "training/training_data/recordings/positives"
        speaker_state = SpeakerState.VAD_RECORD
        os.makedirs(record_dir, exist_ok=True)

    ONSET_TIMEOUT = 8.0

    # the loop
    transcribed_request = ""
    onset_remaining = ONSET_TIMEOUT
    with dev.InputStream(samplerate=input_sample_rate, channels=1, dtype='int16', device=input_dev_index) as stream:
        while not _shutdown.is_set():
            if speaker_state == SpeakerState.LISTEN_FOR_WAKE:
                wake_audio = _listen_for_wake(
                    wake_model,
                    stream,
                    input_sample_rate,
                    threshold=threshold,
                    num_triggers=triggers,
                    buffer_duration=buffer_duration
                )
                _perf_timer.start()
                onset_remaining = ONSET_TIMEOUT
                if record_dir:
                    filepath = f"{record_dir}/{int(time.time() * 1000)}.wav"
                    log(f"caching {filepath}")
                    _write_wav(wake_audio, TARGET_SAMPLE_RATE, filepath)
                    speaker_state = SpeakerState.LISTEN_FOR_WAKE
                else:
                    _play_oneshot_audio_file("sounds/wake.wav")
                    speaker_state = SpeakerState.RECORDING
            elif speaker_state == SpeakerState.RECORDING:
                log(f"recording (onset budget: {onset_remaining:.1f}s)")
                pause_playback()
                _flush_stream(stream, input_sample_rate)
                _wait_for_silence(stream, input_sample_rate)
                audio_request, onset_elapsed = _record_until_silence(vad, stream, input_sample_rate, onset_timeout=onset_remaining)
                if audio_request is None:
                    log("onset timeout, returning to wake listen")
                    resume_playback()
                    onset_remaining = ONSET_TIMEOUT
                    speaker_state = SpeakerState.RESET
                    continue
                _perf_timer.lap("recorded")
                transcribed_request = _transcribe_audio(whisper_model, audio_request)
                _perf_timer.lap("transcribed")
                if not transcribed_request.strip():
                    onset_remaining -= onset_elapsed
                    log(f"empty transcription, onset budget remaining: {onset_remaining:.1f}s")
                    if onset_remaining > 0.5:
                        speaker_state = SpeakerState.RECORDING
                    else:
                        log("onset budget exhausted, returning to wake listen")
                        resume_playback()
                        onset_remaining = ONSET_TIMEOUT
                        speaker_state = SpeakerState.RESET
                    continue
                onset_remaining = ONSET_TIMEOUT
                log(transcribed_request)
                chat_history.append({"role": "user", "text": transcribed_request})
                _save_history(CHAT_HISTORY_PATH, {"role": "user", "text": transcribed_request})
                speaker_state = SpeakerState.LLM_AGENT
            elif speaker_state == SpeakerState.LLM_AGENT:
                speaker_state = query_llm(ctx.llm_client, ctx.system, transcribed_request, stream)
                _perf_timer.lap("llm")
                resume_playback()
            elif speaker_state == SpeakerState.RESET:
                _flush_stream(stream, input_sample_rate)
                wake_model.reset()
                speaker_state = SpeakerState.LISTEN_FOR_WAKE
                log("return to listen for wake")
            elif speaker_state == SpeakerState.VAD_RECORD:
                audio = _vad_record(vad, stream, input_sample_rate)
                filepath = f"{record_dir}/{int(time.time() * 1000)}.wav"
                log(f"caching {filepath}")
                _write_wav(audio, input_sample_rate, filepath, gain=1.0)


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
        llm_client=None,
        system="",
        voice_model=voice_model,
        whisper_model=whisper_model,
        output_dev_index=0,
        output_sample_rate=voice_model.config.sample_rate,
        wake_model=None,
        input_dev_index=0,
        input_sample_rate=16000,
    )
    log("worker ready")


def start():
    """Initialise all models and devices from config.toml and launch the background threads."""
    global ctx, _worker_mode, _worker_url

    # Derive from sys.argv directly — immune to the double-import problem where
    # python src/speaker.py runs as __main__ but web.py imports a second speaker instance.
    _worker_mode = "--worker" in sys.argv
    if "--worker-ip" in sys.argv:
        idx = sys.argv.index("--worker-ip")
        ip_arg = sys.argv[idx + 1]
        _worker_url = f"http://{ip_arg}" if ":" in ip_arg else f"http://{ip_arg}:8000"

    if _worker_mode:
        _start_worker()
        return

    log("oi! start!!")

    if ctx is not None:
        return

    chat_history.extend(_load_history(CHAT_HISTORY_PATH, HISTORY_LOAD_LIMIT))
    if PLAY_POSITIONS_PATH.exists():
        _play_positions.update(json.loads(PLAY_POSITIONS_PATH.read_text(encoding="utf-8")))

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

    # wake
    wake_model = Model(
        inference_framework="onnx",
        wakeword_models=["models/openwakeword/oi_speaker.onnx"],
        vad_threshold=0.5,
        enable_speex_noise_suppression=False
    )

    # vad
    vad = webrtcvad.Vad(3)

    # whisper
    inf = config.get("inference", {})
    device = inf.get("device", "cpu")
    compute = "int8"
    if device == "cuda":
        compute = "float16"
    whisper_model = WhisperModel(
        inf.get("whisper_model", "small"),
        device=device,
        compute_type=inf.get("whisper_compute", compute)
    )

    llm_client = anthropic.Anthropic(api_key=config["llm"]["anthropic_api_key"])
    voice_model = PiperVoice.load("models/piper/en_GB-northern_english_male-medium.onnx")

    if "--verbose" in sys.argv:
        log(json.dumps(enumerate_audio_devices(), indent=4))
    input_dev_info = _get_audio_device_index(config["audio"]["input_device"])
    output_dev_info = _get_audio_device_index(config["audio"]["output_device"])
    input_dev_index = int(input_dev_info['index'])
    input_sample_rate = int(input_dev_info['default_samplerate'])
    output_dev_index = int(output_dev_info['index'])
    output_sample_rate = int(output_dev_info['default_samplerate'])

    ctx = SpeakerContext(
        llm_client=llm_client,
        system=system,
        voice_model=voice_model,
        whisper_model=whisper_model,
        output_dev_index=output_dev_index,
        output_sample_rate=output_sample_rate,
        wake_model=wake_model,
        input_dev_index=input_dev_index,
        input_sample_rate=input_sample_rate,
    )

    threading.Thread(
        target=_player_loop,
        daemon=True
    ).start()

    threading.Thread(
        target=_speak_loop,
        args=(wake_model, vad, whisper_model, input_dev_index, input_sample_rate),
        daemon=True
    ).start()



def main():
    """Entry point: start the speaker and serve the web UI via uvicorn."""
    global _worker_mode, _worker_url
    log("oi! oi!!")

    if "--enum-audio" in sys.argv:
        log(json.dumps(enumerate_audio_devices(), indent=4))
        return

    if "--worker" in sys.argv:
        print("oi! worker!!")
        _worker_mode = True

    if "--worker-ip" in sys.argv:
        idx = sys.argv.index("--worker-ip")
        _worker_url = f"http://{sys.argv[idx + 1]}"
        # append default port if none given
        if ":" not in sys.argv[idx + 1]:
            with open("config.toml", "rb") as _f:
                _port_cfg = tomllib.load(_f)
            _worker_url += f":{int(_port_cfg.get('network', {}).get('port', 8000))}"
        log(f"using worker: {_worker_url}")

    sys.path.insert(0, str(__file__).replace("speaker.py", ""))
    from web import app
    with open("config.toml", "rb") as _f:
        _cfg = tomllib.load(_f)
    _port = int(_cfg.get("network", {}).get("port", 8000))
    uvicorn.run(app, host="0.0.0.0", port=_port)
    os._exit(0)


if __name__ == "__main__":
    main()