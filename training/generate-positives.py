#!/usr/bin/env python3
"""
Generate synthetic positive wake-word samples using the project's Piper TTS model.

Each clip is generated with fully randomised synthesis parameters so no two are
identical. length_scale controls speaking rate (and therefore clip duration),
noise_scale controls overall prosody variation, noise_w_scale controls per-phoneme
timing variation.

Usage:
    python training/generate-positives.py
    python training/generate-positives.py --output_dir training/training_data/positive_samples --count 500
    python training/generate-positives.py --model models/piper/other-voice.onnx --count 200
"""

import argparse
import os
import numpy as np
import soxr
import scipy.io.wavfile as wavfile
from piper.voice import PiperVoice, SynthesisConfig

# Slight text variants — all phonetically equivalent to how a user would say it.
# Punctuation affects Piper's prosody and pause length, adding natural variety.
TEXT_VARIANTS = [
    "oi speaker",
    "oi, speaker",
    "oi! speaker",
    "oi speaker!",
]

TARGET_RATE = 16000


def synthesize(voice, text, config):
    chunks = []
    for chunk in voice.synthesize(text, config):
        data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        chunks.append(data)
    if not chunks:
        return None
    audio = np.concatenate(chunks)
    if voice.config.sample_rate != TARGET_RATE:
        audio = soxr.resample(audio.astype(np.float32) / 32767.0,
                               voice.config.sample_rate, TARGET_RATE)
        audio = (audio * 32767).astype(np.int16)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/piper/en_GB-northern_english_male-medium.onnx")
    parser.add_argument("--output_dir", default="training/training_data/positive_samples")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model}")
    voice = PiperVoice.load(args.model)

    rng = np.random.default_rng(args.seed)

    generated = 0
    for i in range(args.count):
        # Fully randomised params — no fixed grid, no repeats
        length_scale  = float(rng.uniform(0.85, 1.25))   # speaking rate; drives clip duration
        noise_scale   = float(rng.uniform(0.55, 0.90))   # overall prosody variation (avoid flat <0.5)
        noise_w_scale = float(rng.uniform(0.60, 1.10))   # per-phoneme timing variation
        text = TEXT_VARIANTS[i % len(TEXT_VARIANTS)]

        config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        audio = synthesize(voice, text, config)
        if audio is None:
            continue

        out_path = os.path.join(args.output_dir, f"tts_{i:04d}.wav")
        wavfile.write(out_path, TARGET_RATE, audio)
        generated += 1
        if generated % 100 == 0:
            print(f"  {generated}/{args.count}")

    print(f"Done. {generated} clips saved to {args.output_dir}")


if __name__ == "__main__":
    main()
