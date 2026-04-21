#!/usr/bin/env python3
"""
Generate synthetic positive wake-word samples using the project's Piper TTS model.
Varies length_scale, noise_scale, and noise_w_scale to produce diverse clips.

Usage:
    python training/generate-positives.py
    python training/generate-positives.py --output_dir training/positives --count 200
    python training/generate-positives.py --model models/piper/other-voice.onnx --count 100
"""

import argparse
import os
import wave
import itertools
import numpy as np
import soxr
import scipy.io.wavfile as wavfile
from piper.voice import PiperVoice, SynthesisConfig

TEXT = "oi speaker"
TARGET_RATE = 16000


def synthesize(voice, config):
    chunks = []
    for chunk in voice.synthesize(TEXT, config):
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
    parser.add_argument("--output_dir", default="training/positives")
    parser.add_argument("--count", type=int, default=200,
                        help="Number of samples to generate (default: 200)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model}")
    voice = PiperVoice.load(args.model)

    length_scales  = [0.8, 0.9, 1.0, 1.1, 1.2]
    noise_scales   = [0.4, 0.667, 0.9]
    noise_w_scales = [0.6, 0.8, 1.0]

    combos = list(itertools.product(length_scales, noise_scales, noise_w_scales))

    generated = 0
    for i in range(args.count):
        ls, ns, nw = combos[i % len(combos)]
        config = SynthesisConfig(length_scale=ls, noise_scale=ns, noise_w_scale=nw)
        audio = synthesize(voice, config)
        if audio is None:
            continue
        out_path = os.path.join(args.output_dir, f"tts_{i:04d}.wav")
        wavfile.write(out_path, TARGET_RATE, audio)
        generated += 1
        if generated % 50 == 0:
            print(f"  {generated}/{args.count}")

    print(f"Done. {generated} clips saved to {args.output_dir}")


if __name__ == "__main__":
    main()
