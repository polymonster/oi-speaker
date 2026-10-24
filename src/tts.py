"""Render text to a wav file with the speaker's piper voice, for recording out canned lines.

    tts out.wav "Oi! What do you want?"
    tts out.wav "Hello" --voice models/piper/other-voice.onnx
"""
import argparse
import tomllib
import wave
from pathlib import Path

from piper.voice import PiperVoice

DEFAULT_VOICE = "models/piper/en_GB-northern_english_male-medium.onnx"
CONFIG_PATH = Path("config.toml")


def _configured_voice() -> str:
    """The voice the speaker itself uses — config.toml's [voice] model if set, else the default."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f).get("voice", {}).get("model", DEFAULT_VOICE)
    return DEFAULT_VOICE


def synthesize_to_file(voice: PiperVoice, text: str, out_path: Path):
    """Synthesise `text` and write it as a 16-bit mono wav at the voice's native sample rate."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        for chunk in voice.synthesize(text):
            wf.writeframes(chunk.audio_int16_bytes)


def main():
    parser = argparse.ArgumentParser(description="Render text to a wav file with the speaker's piper voice.")
    parser.add_argument("output", type=Path, help="wav file to write")
    parser.add_argument("text", help="text to speak")
    parser.add_argument("--voice", help=f"piper .onnx voice model (default: config.toml [voice] model, else {DEFAULT_VOICE})")
    args = parser.parse_args()

    voice_path = args.voice or _configured_voice()
    voice = PiperVoice.load(voice_path)
    synthesize_to_file(voice, args.text, args.output)
    print(f"wrote {args.output} ({voice_path})")


if __name__ == "__main__":
    main()
