"""Render text to a wav file with the speaker's voice, for recording out canned lines.

    tts-record out.wav "Oi! What do you want?"
    tts-record out.wav "Hello" --backend openai
    tts-record out.wav "Hello" --voice models/piper/other-voice.onnx
    tts-record out.wav "Hello" --backend kokoro --voice bm_daniel
    tts-record out.wav "Hello" --backend chatterbox --voice reference.wav
"""
import argparse
import tomllib
import wave
from pathlib import Path

import voice

CONFIG_PATH = Path("config.toml")
VOICE_OPTION = {"piper": "model", "kokoro": "voice", "chatterbox": "reference", "openai": "voice"}  # what --voice sets


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Render text to a wav file with the speaker's voice.")
    parser.add_argument("output", type=Path, help="wav file to write")
    parser.add_argument("text", help="text to speak")
    parser.add_argument("--backend", choices=list(voice.BACKENDS), help="voice backend (default: config.toml [voice] backend)")
    parser.add_argument("--voice", help="the backend's voice: a piper .onnx model (implies --backend piper), a kokoro "
                                        "or openai voice name, or a chatterbox reference wav")
    args = parser.parse_args()

    # the speaker's own [voice] settings, so canned lines sound like the live ones
    config = voice.migrate_config(_load_config())
    if args.voice and args.voice.endswith(".onnx"):
        args.backend = args.backend or "piper"
    if args.backend:
        config["voice"]["backend"] = args.backend
    if args.voice:
        backend = config["voice"]["backend"]
        config["voice"][backend][VOICE_OPTION[backend]] = args.voice

    model = voice.create(config)
    if model is None:
        raise SystemExit("no voice backend installed")
    audio, sample_rate = model.synthesize(args.text)
    if not len(audio):
        raise SystemExit("synthesis failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"wrote {args.output} ({model.describe()})")


if __name__ == "__main__":
    main()
