import openwakeword
import sounddevice as dev
import numpy as np
import time
import webrtcvad
import math
import faster_whisper
import anthropic
import re

from piper.voice import PiperVoice
from openwakeword.model import Model
from faster_whisper import WhisperModel
from enum import Enum

TARGET_SAMPLE_RATE = 16000 # 16khz
WAKE_CHUNK = 1280 # 80ms at 16khz
VAD_CHUNK = 480 # 30ms at 16kHz


class SpeakerState(Enum):
    LISTEN_FOR_WAKE = "listen_for_wake"
    CHATTING = "chatting"
    RESET = "reset"


def strip_markdown(text: str) -> str:
    text = re.sub(r'#{1,6}\s*', '', text) # headers
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text) # bold/italic
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text) # code
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # links
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE) # list bullets
    text = re.sub(r'\n+', ' ', text) # newlines to spaces
    return text.strip()


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


def record_until_silence_snappy(vad, stream, sample_rate, silence_timeout=1.5) -> np.ndarray:
    chunks = []
    silent_duration = 0.0
    was_speaking = False
    
    while silent_duration < silence_timeout:
        audio_flat = read_audio(stream, VAD_CHUNK, sample_rate)
        chunks.append(audio_flat)
        is_speech = vad.is_speech(audio_flat.tobytes(), TARGET_SAMPLE_RATE)
                
        if is_speech:
            was_speaking = True
            silent_duration = 0.0
        else:
            silent_duration += VAD_CHUNK / TARGET_SAMPLE_RATE
            
            # speech just ended - don't wait full timeout
            if was_speaking and silent_duration > 0.3:
                break
                
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


def transcribe_audio(whisper_model, audio: np.ndarray) -> str:
    audio_float = audio.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio_float, language="en", beam_size=1)
    return " ".join(segment.text for segment in segments)


def flush_stream(stream):
    stream.stop()
    stream.start()


def query_llm(llm_client, text: str) -> str:
    SYSTEM_PROMPT = """You are Oi-Speaker!!, a smart speaker assistant, designed and built by Alex Dixon.
    Respond in plain conversational text only.
    No markdown, no bullet points, no headers.
    Speak as if talking out loud, keep responses concise."""

    message = llm_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": text}
        ]
    )
    return message.content[0].text


def speak(dev, dev_index, sample_rate, voice_model, text: str):
    stream = dev.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        device=dev_index
    )
    stream.start()
    for audio_chunk in voice_model.synthesize(text):
        int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
        resampled_audio = resample_audio(int_data, voice_model.config.sample_rate, sample_rate)
        stream.write(resampled_audio)
    stream.stop()
    stream.close()


def speak_debug(dev, dev_index, sample_rate, voice_model, text: str):
    chunks = []
    for audio_chunk in voice_model.synthesize(text):
        int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
        chunks.append(int_data)
    
    print(f"chunks: {len(chunks)}")
    
    if chunks:
        full_audio = np.concatenate(chunks)
        resampled = resample_audio(full_audio, voice_model.config.sample_rate, sample_rate)
        dev.play(resampled, samplerate=sample_rate, device=dev_index)
        dev.wait()


def test_tone(dev, dev_index: int, sample_rate: int):
    # generate a simple 440hz test tone
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    dev.play(tone, samplerate=sample_rate, device=dev_index)
    dev.wait()


def main():
    print("oi! oi!!")

    # init openwakeword model
    wake_model = Model(
        wakeword_models = ["hey_jarvis"],
        inference_framework="tflite",
        vad_threshold=0.5,
        enable_speex_noise_suppression=False
    )

    # init voice active detection
    vad = webrtcvad.Vad(2) # aggressiveness 0-3

    # init whisper model
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    # llm client
    llm_client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    # text to voice model
    voice_model = PiperVoice.load("models/en_GB-northern_english_male-medium.onnx")

    # grab device, this should come from config
    input_dev_info = get_audio_device_index("DualSense")
    output_dev_info = get_audio_device_index("Scarlett 2i2 USB") 
    input_dev_index = int(input_dev_info['index'])
    input_sample_rate = int(input_dev_info['default_samplerate'])
    output_dev_index = int(output_dev_info['index'])
    output_sample_rate = int(output_dev_info['default_samplerate'])

    # test_tone(dev, output_dev_index, output_sample_rate)

    # main loop
    state = SpeakerState.LISTEN_FOR_WAKE
    with dev.InputStream(samplerate=input_sample_rate, channels=1, dtype='int16', device=input_dev_index) as stream:
        while True:
            if state == SpeakerState.LISTEN_FOR_WAKE:
                if listen_for_wake(wake_model, stream, input_sample_rate):
                    state = SpeakerState.CHATTING
            elif state == SpeakerState.CHATTING:
                audio_request = record_until_silence(vad, stream, input_sample_rate)
                transcribed_request = transcribe_audio(whisper_model, audio_request)
                llm_response = query_llm(llm_client, transcribed_request)
                print(llm_response)
                voice_response = strip_markdown(llm_response)
                speak_debug(dev, output_dev_index, output_sample_rate, voice_model, voice_response)
                if not llm_response.endswith("?"):
                    state = SpeakerState.RESET
            elif state == SpeakerState.RESET:
                flush_stream(stream)
                wake_model.reset()
                state = SpeakerState.LISTEN_FOR_WAKE
                print("return to listen for wake")


if __name__ == "__main__":
    main()