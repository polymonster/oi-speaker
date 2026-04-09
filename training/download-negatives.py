#!/usr/bin/env python3
"""
Download and prepare negative training data for openwakeword.

Grabs speech data from HuggingFace (LibriSpeech clean) and chops it
into clips matching your positive sample duration. Also generates
silence/noise clips for variety.

Usage:
    pip install datasets soundfile
    python download_negatives.py --output_dir ../oww-training/negative_samples --n_clips 3000

This will give you a mix of:
  - General English speech (people saying things that aren't your wake word)
  - Silence / low-level noise
"""

import os
import sys
import argparse
import logging
import numpy as np
import scipy.io.wavfile as wavfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def download_librispeech_clips(output_dir, n_clips=2500, clip_duration_sec=1.0, sr=16000):
    """Download LibriSpeech data from HuggingFace and chop into fixed-length clips."""
    from datasets import load_dataset, Audio

    target_samples = int(clip_duration_sec * sr)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading LibriSpeech clean-100 from HuggingFace (streaming)...")
    logger.info(f"Target: {n_clips} clips of {clip_duration_sec}s at {sr}Hz")

    # Stream the dataset, force soundfile backend to avoid torchcodec
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
        trust_remote_code=True
    ).cast_column("audio", Audio(sampling_rate=sr, decode=True))

    clip_count = 0

    for example in dataset:
        if clip_count >= n_clips:
            break

        audio = example["audio"]
        audio_array = audio["array"]
        audio_sr = audio["sampling_rate"]

        # Convert to int16 (HF returns float32 normalized to [-1, 1])
        if audio_array.dtype == np.float64 or audio_array.dtype == np.float32:
            audio_array = (audio_array * 32767).astype(np.int16)

        # Chop into fixed-length clips
        for start in range(0, len(audio_array) - target_samples, target_samples):
            if clip_count >= n_clips:
                break

            clip = audio_array[start:start + target_samples]
            out_path = os.path.join(output_dir, f"speech_{clip_count:05d}.wav")
            wavfile.write(out_path, sr, clip)
            clip_count += 1

            if clip_count % 500 == 0:
                logger.info(f"  Generated {clip_count}/{n_clips} speech clips...")

    logger.info(f"Generated {clip_count} speech clips from LibriSpeech")
    return clip_count


def generate_noise_clips(output_dir, n_clips=300, clip_duration_sec=1.0, sr=16000):
    """Generate various noise/silence clips as additional negatives."""
    target_samples = int(clip_duration_sec * sr)
    os.makedirs(output_dir, exist_ok=True)

    clip_count = 0
    rng = np.random.RandomState(42)

    for i in range(n_clips):
        noise_type = i % 4

        if noise_type == 0:
            # Near-silence with very low noise
            clip = (rng.randn(target_samples) * 50).astype(np.int16)
        elif noise_type == 1:
            # White noise at moderate level
            clip = (rng.randn(target_samples) * 500).astype(np.int16)
        elif noise_type == 2:
            # Pink-ish noise (low freq bias)
            white = rng.randn(target_samples)
            # Simple low-pass via cumulative sum + decay
            pink = np.zeros(target_samples)
            pink[0] = white[0]
            for j in range(1, target_samples):
                pink[j] = 0.98 * pink[j-1] + white[j]
            pink = pink / (np.max(np.abs(pink)) + 1e-10) * 1000
            clip = pink.astype(np.int16)
        else:
            # Silence
            clip = np.zeros(target_samples, dtype=np.int16)

        out_path = os.path.join(output_dir, f"noise_{clip_count:05d}.wav")
        wavfile.write(out_path, sr, clip)
        clip_count += 1

    logger.info(f"Generated {clip_count} noise/silence clips")
    return clip_count


def main():
    parser = argparse.ArgumentParser(description='Download negative training data')
    parser.add_argument('--output_dir', default='../oww-training/negative_samples',
                        help='Output directory for negative clips')
    parser.add_argument('--n_clips', type=int, default=3000,
                        help='Total number of negative clips to generate')
    parser.add_argument('--clip_duration', type=float, default=1.0,
                        help='Clip duration in seconds (match your positive clips)')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate (16000 for openwakeword)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Allocate: 85% speech, 15% noise/silence
    n_speech = int(args.n_clips * 0.85)
    n_noise = args.n_clips - n_speech

    logger.info("=" * 60)
    logger.info(f"Downloading {n_speech} speech clips + {n_noise} noise clips")
    logger.info(f"Clip duration: {args.clip_duration}s, Sample rate: {args.sr}Hz")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    speech_count = download_librispeech_clips(
        args.output_dir, n_clips=n_speech,
        clip_duration_sec=args.clip_duration, sr=args.sr
    )

    noise_count = generate_noise_clips(
        args.output_dir, n_clips=n_noise,
        clip_duration_sec=args.clip_duration, sr=args.sr
    )

    total = speech_count + noise_count
    logger.info("=" * 60)
    logger.info(f"Done! {total} negative clips saved to {args.output_dir}")
    logger.info(f"  Speech: {speech_count}")
    logger.info(f"  Noise/silence: {noise_count}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()