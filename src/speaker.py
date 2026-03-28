import openwakeword
import sounddevice as dev
import numpy as np
import time
import webrtcvad
import math
import faster_whisper

from openwakeword.model import Model
from faster_whisper import WhisperModel
from enum import Enum

TARGET_SAMPLE_RATE = 16000 # 16khz
WAKE_CHUNK = 1280 # 80ms at 16khz
VAD_CHUNK = 480 # 30ms at 16kHz

class SpeakerState(Enum):
    LISTEN_FOR_WAKE = "listen_for_wake"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


def get_audio_device_index(name: str):
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

    # grab device, this should come from config
    dev_info = get_audio_device_index("Scarlett 2i2 USB")
    dev_index = int(dev_info['index'])
    sample_rate = int(dev_info['default_samplerate'])

    state = SpeakerState.LISTEN_FOR_WAKE

    with dev.InputStream(samplerate=sample_rate, channels=1, dtype='int16', device=dev_index) as stream:
        while True:
            if state == SpeakerState.LISTEN_FOR_WAKE:
                if listen_for_wake(wake_model, stream, sample_rate):
                    state = SpeakerState.RECORDING
            elif state == SpeakerState.RECORDING:
                print("recording")
                audio_request = record_until_silence(vad, stream, sample_rate)
                transcribe_request = transcribe_audio(whisper_model, audio_request)
                print(f"{transcribe_request}")
                flush_stream(stream)
                wake_model.reset()
                state = SpeakerState.LISTEN_FOR_WAKE
                print("return to listen for wake")
                



if __name__ == "__main__":
    main()