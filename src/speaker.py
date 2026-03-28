import openwakeword
import sounddevice as dev
import numpy as np

from openwakeword.model import Model

model = Model(
    wakeword_models = ["hey_jarvis"],
    inference_framework="tflite",
    vad_threshold=0.5,
    enable_speex_noise_suppression=False
)

TARGET_SAMPLE_RATE = 16000
TARGET_CHUNK = 1280

def resample_audio(audio, orig_rate, target_rate):
    ratio = target_rate / orig_rate
    new_length = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio), new_length),
        np.arange(len(audio)),
        audio
    ).astype(np.int16)


def get_audio_device_index(name: str):
    devices = str(dev.query_devices()).splitlines()
    index = 0
    for device in devices:
        if device.find(name) != -1:
            print(f"audio device index: {device}")
            return dev.query_devices(index)
        index += 1
    return None


def main():
    print("oi! oi!!")

    # grab device, this should come from config
    dev_info = get_audio_device_index("Scarlett 2i2 USB")
    sample_rate = int(dev_info['default_samplerate'])

    with dev.InputStream(samplerate=sample_rate, channels=1, dtype='int16', device=dev_info['index']) as stream:
        while True:
            # read audio
            sample_ratio = int(sample_rate / TARGET_SAMPLE_RATE)
            chunk = TARGET_CHUNK * sample_ratio
            audio, _ = stream.read(chunk)

            # resample to TARGET_SAMPLE_RATE and flatten
            if sample_ratio != 1:
                audio_flat = np.squeeze(audio)
                audio_flat = resample_audio(audio_flat, sample_rate, TARGET_SAMPLE_RATE)
            else:
                audio_flat = np.squeeze(audio)

            # predict and wake
            prediction = model.predict(audio_flat)
            for wake_word, score in prediction.items():
                if score > 0.5:
                    print("ello mate!")


if __name__ == "__main__":
    main()