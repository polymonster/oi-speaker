import subprocess
import sys
import threading
from collections import deque
import mpv
import openwakeword
import sounddevice as dev
import numpy as np
import time
import webrtcvad
import math
import faster_whisper
import anthropic
import re
import tomllib
import json
import requests
import wave
import uvicorn
import os
    
from piper.voice import PiperVoice
from openwakeword.model import Model
from faster_whisper import WhisperModel
from dataclasses import dataclass
from enum import Enum

TARGET_SAMPLE_RATE = 16000 # 16khz
WAKE_CHUNK = 1280 # 80ms at 16khz
VAD_CHUNK = 480 # 30ms at 16kHz


class SpeakerState(Enum):
    TEXT_CHAT = "text_chat"
    LISTEN_FOR_WAKE = "listen_for_wake"
    CHATTING = "chatting"
    RESET = "reset"


@dataclass
class SpeakerContext:
    llm_client: any
    system: str
    config: dict
    voice_model: any
    output_dev_index: int
    output_sample_rate: int


def get_audio_device_index(name: str):
    print(dev.query_devices())
    devices = str(dev.query_devices()).splitlines()
    index = 0
    for device in devices:
        if device.find(name) != -1:
            print(f"audio device index: {device}")
            return dev.query_devices(index)
        index += 1
    return None


def flush_stream(stream, sample_rate: int):
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


def test_tone(dev, dev_index: int, sample_rate: int):
    # generate a simple 440hz test tone
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    dev.play(tone, samplerate=sample_rate, device=dev_index)
    dev.wait()


def resample_audio(audio, orig_rate: int, target_rate: int):
    ratio = target_rate / orig_rate
    new_length = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio), new_length),
        np.arange(len(audio)),
        audio
    ).astype(np.int16)


def read_audio(stream, chunk_size16: int, sample_rate: int) -> np.ndarray:
    # read audio
    sample_ratio = int(math.ceil(sample_rate / TARGET_SAMPLE_RATE))
    chunk = chunk_size16 * sample_ratio
    audio, _ = stream.read(chunk)
    # resample to TARGET_SAMPLE_RATE and flatten
    if sample_ratio != 1:
        audio_flat = np.squeeze(audio)
        audio_flat = resample_audio(audio_flat, sample_rate, TARGET_SAMPLE_RATE)
        audio_flat = audio_flat[:chunk_size16]
    else:
        audio_flat = np.squeeze(audio)
    return audio_flat


def record_until_silence(vad, stream, sample_rate: int, silence_timeout: float=1.5) -> np.ndarray:
    chunks = []
    silent_duration = 0.0
    while silent_duration < silence_timeout:
        audio_flat = read_audio(stream, VAD_CHUNK, sample_rate)
        chunks.append(audio_flat)
        is_speech = vad.is_speech(audio_flat.tobytes(), TARGET_SAMPLE_RATE)
        if is_speech:
            silent_duration = 0.0
        else:
            silent_duration += VAD_CHUNK / TARGET_SAMPLE_RATE

    return np.concatenate(chunks)


def listen_for_wake(wake_model, stream, sample_rate: int) -> bool:
    # read audio
    audio_flat = read_audio(stream, WAKE_CHUNK, sample_rate)
    # predict and wake
    prediction = wake_model.predict(audio_flat)
    for _, score in prediction.items():
        if score > 0.5:
            print("ello mate!")
            return True
    return False


def write_wav(audio: np.ndarray, sample_rate: int, filepath: str, gain: float = 4.0):
    boosted = np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(boosted.tobytes())


def transcribe_audio(whisper_model, audio: np.ndarray) -> str:
    audio_float = audio.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio_float, language="en", beam_size=1)
    return " ".join(segment.text for segment in segments)


def query_llm(llm_client, system, text: str) -> str:
    message = llm_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": text}
        ]
    )
    return message.content[0].text


def _speak(dev, dev_index, sample_rate, voice_model, text: str):
    chunks = []
    for audio_chunk in voice_model.synthesize(text):
        int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
        chunks.append(int_data)
    if chunks:
        full_audio = np.concatenate(chunks)
        resampled = resample_audio(full_audio, voice_model.config.sample_rate, sample_rate)
        dev.play(resampled, samplerate=sample_rate, device=dev_index)
        dev.wait()


ctx: SpeakerContext | None = None
speaker_state: SpeakerState = SpeakerState.LISTEN_FOR_WAKE

_player: mpv.MPV | None = None

def stop_playback():
    global _player
    if _player is not None:
        _player.stop()
        _player = None


def pause_playback():
    if _player is not None:
        _player.pause = True


def resume_playback():
    if _player is not None:
        _player.pause = False


def _play_youtube_windows(url: str):
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    output_template = os.path.join(tmpdir, "audio.%(ext)s")
    print(f"downloading youtube audio...")
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", output_template,
         "--no-playlist", "--extractor-args", "youtube:player_client=tv_embedded", url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr}")
        return
    files = os.listdir(tmpdir)
    if not files:
        print("yt-dlp: no output file found")
        return
    audio_file = os.path.join(tmpdir, files[0])
    print(f"playing: {audio_file}")
    global _player
    _player = mpv.MPV(vid=False, terminal=False)
    _player.play(audio_file)
    _player.wait_for_playback()
    try:
        os.unlink(audio_file)
        os.rmdir(tmpdir)
    except Exception:
        pass


def play_url(url: str):
    global _player
    stop_playback()
    print(f"play_url: {url}")
    is_youtube = "youtube.com" in url or "youtu.be" in url
    if is_youtube and sys.platform == "win32":
        threading.Thread(target=_play_youtube_windows, args=(url,), daemon=True).start()
    else:
        _player = mpv.MPV(vid=False, terminal=False)
        def _run():
            _player.play(url)
            _player.wait_for_playback()
        threading.Thread(target=_run, daemon=True).start()


def search_youtube(query: str, max_results: int = 5) -> str:
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


def search_web(query: str):
    return requests.get("https://api.duckduckgo.com/", params={"q": query, "format": "json"})


def search_radio(query: str, config: dict):
    stations = json.dumps(config["radio"])
    request = "make me a play_url request for " + query
    request += "selected from these list of stations " + stations
    return request


def _parse_llm_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    return json.loads(text)


def _handle_llm_function(ctx: SpeakerContext, response: str) -> SpeakerState | None:
    response = _parse_llm_json(response)
    print(json.dumps(response))
    if "function" in response:
        if response["function"] == "search_youtube":
            results = search_youtube(response["payload"])
            print(results)
            response = query_llm(ctx.llm_client, ctx.system, results)
            return _handle_llm_function(ctx, response)
        elif response["function"] == "search_radio":
            results = search_radio(response["payload"], ctx.config)
            response = query_llm(ctx.llm_client, ctx.system, results)
            return _handle_llm_function(ctx, response)
        elif response["function"] == "play_url":
            play_url(response["payload"])
            return SpeakerState.RESET
        elif response["function"] == "stop":
            stop_playback()
            return SpeakerState.RESET
        elif response["function"] == "search_web":
            results = search_web(response["payload"])
            print(results.text)
            response = query_llm(ctx.llm_client, ctx.system, results.text)
            return _handle_llm_function(ctx, response)
        elif response["function"] == "chat":
            _speak(dev, ctx.output_dev_index, ctx.output_sample_rate, ctx.voice_model, response["payload"])
            if response.get("follow_on"):
                return SpeakerState.CHATTING
            return SpeakerState.RESET
    return None


def _audio_loop(wake_model, vad, whisper_model, input_dev_index, input_sample_rate):
    print("audio! loop!!")
    global speaker_state
    with dev.InputStream(samplerate=input_sample_rate, channels=1, dtype='int16', device=input_dev_index) as stream:
        while True:
            if speaker_state == SpeakerState.LISTEN_FOR_WAKE:
                if listen_for_wake(wake_model, stream, input_sample_rate):
                    speaker_state = SpeakerState.CHATTING
            elif speaker_state == SpeakerState.CHATTING:
                print("chatting")
                pause_playback()
                audio_request = record_until_silence(vad, stream, input_sample_rate)
                transcribed_request = transcribe_audio(whisper_model, audio_request)
                print(transcribed_request)
                llm_response = query_llm(ctx.llm_client, ctx.system, transcribed_request)
                next_state = _handle_llm_function(ctx, llm_response)
                if next_state is not None:
                    speaker_state = next_state
                resume_playback()
            elif speaker_state == SpeakerState.RESET:
                flush_stream(stream, input_sample_rate)
                wake_model.reset()
                speaker_state = SpeakerState.LISTEN_FOR_WAKE
                print("return to listen for wake")


def start():
    global ctx
    if ctx is not None:
        return

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    with open("system.json", "rb") as f:
        system = json.load(f)
        system = json.dumps(system)

    wake_model = Model(
        # wakeword_models=["hey_jarvis"],
        # inference_framework="tflite",
        wakeword_models=["models/oi_speaker.onnx"],
        vad_threshold=0.5,
        enable_speex_noise_suppression=False # TODO: Pi
    )

    vad = webrtcvad.Vad(2)
    inf = config.get("inference", {})
    device = inf.get("device", "cpu")
    try:
        whisper_model = WhisperModel(
            inf.get("whisper_model", "small"),
            device=device,
            compute_type=inf.get("whisper_compute", "int8")
        )
    except Exception as e:
        print(f"whisper init failed on {device} ({e}), falling back to cpu")
        whisper_model = WhisperModel(inf.get("whisper_model", "small"), device="cpu", compute_type="int8")
    llm_client = anthropic.Anthropic(api_key=config["llm"]["anthropic_api_key"])
    voice_model = PiperVoice.load("models/en_GB-northern_english_male-medium.onnx")

    input_dev_info = get_audio_device_index(config["audio"]["input_device"])
    output_dev_info = get_audio_device_index(config["audio"]["output_device"])
    input_dev_index = int(input_dev_info['index'])
    input_sample_rate = int(input_dev_info['default_samplerate'])
    output_dev_index = int(output_dev_info['index'])
    output_sample_rate = int(output_dev_info['default_samplerate'])

    ctx = SpeakerContext(
        llm_client=llm_client,
        system=system,
        config=config,
        voice_model=voice_model,
        output_dev_index=output_dev_index,
        output_sample_rate=output_sample_rate,
    )

    threading.Thread(
        target=_audio_loop,
        args=(wake_model, vad, whisper_model, input_dev_index, input_sample_rate),
        daemon=True
    ).start()


def wake_word_capture(dir: str):
    print("wake_word_capture")

    # make dir to write to
    os.makedirs(dir, exist_ok=True)

    # load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    wake_model = Model(
        wakeword_models=["models/oi_speaker.onnx"],
        vad_threshold=0.5,
        enable_speex_noise_suppression=False
    )

    # setup audio
    input_dev_info = get_audio_device_index(config["audio"]["input_device"])
    input_dev_index = int(input_dev_info['index'])
    input_sample_rate = int(input_dev_info['default_samplerate'])

    # 2 seconds of rolling context at 80ms per chunk
    buffer_chunks = int(2.0 / (WAKE_CHUNK / TARGET_SAMPLE_RATE))
    raw_chunk = WAKE_CHUNK * sample_ratio
    rolling_buffer = deque(maxlen=buffer_chunks)
    sample_ratio = int(math.ceil(input_sample_rate / TARGET_SAMPLE_RATE))
    score_window = deque(maxlen=5)

    # listening loop
    print("listening")
    with dev.InputStream(samplerate=input_sample_rate, channels=1, dtype='int16', device=input_dev_index) as stream:
        while True:
            raw, _ = stream.read(raw_chunk)
            raw_flat = np.squeeze(raw)
            rolling_buffer.append(raw_flat)
            # resample only for the wake model
            audio_flat = resample_audio(raw_flat, input_sample_rate, TARGET_SAMPLE_RATE)[:WAKE_CHUNK]
            prediction = wake_model.predict(audio_flat)
            for _, score in prediction.items():
                score_window.append(score > 0.9)
                hits = sum(score_window)
                if score > 0.0:
                    print(f"score: {score:.3f} ({hits}/5)")
                if hits >= 3:
                    print("triggered!")
                    filepath = f"{dir}/{int(time.time() * 1000)}.wav"
                    write_wav(np.concatenate(rolling_buffer), input_sample_rate, filepath)
                    print(f"wrote: {filepath}")
                    rolling_buffer.clear()
                    score_window.clear()
                    wake_model.reset()


def main():
    print("oi! oi!!")

    if "-wake_word_capture" in sys.argv:
        wake_word_capture(sys.argv[sys.argv.index("-wake_word_capture") + 1])
    else:
        # main speaker loop / app
        sys.path.insert(0, str(__file__).replace("speaker.py", ""))
        from web import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
        os._exit(0)


if __name__ == "__main__":
    main()