#!/usr/bin/env python3
"""
Download and prepare negative training data for openwakeword.

Downloads LibriSpeech test-clean directly (no HuggingFace datasets audio
decoding nonsense) and chops into clips. Also generates noise clips.

Usage:
    pip install soundfile soxr requests
    python download_negatives.py --output_dir ../oww-training/negative_samples --n_clips 3000
"""

import os
import sys
import tarfile
import argparse
import logging
import tempfile
import glob
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
# ~350MB download, ~5.4 hours of clean English speech


def download_and_extract(url, extract_dir):
    """Download a tar.gz and extract it."""
    tar_path = os.path.join(extract_dir, "download.tar.gz")

    if not os.path.exists(tar_path):
        logger.info(f"Downloading {url}...")
        logger.info("(~350MB, this may take a few minutes)")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        with open(tar_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (50 * 1024 * 1024) < len(chunk):
                    pct = downloaded / total * 100
                    logger.info(f"  {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.0f}%)")
        logger.info("Download complete.")
    else:
        logger.info(f"Using cached download at {tar_path}")

    logger.info("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    logger.info("Extraction complete.")


def find_flac_files(search_dir):
    """Find all FLAC files recursively."""
    return sorted(glob.glob(os.path.join(search_dir, '**', '*.flac'), recursive=True))


def chop_audio_to_clips(flac_files, output_dir, n_clips, clip_duration_sec, target_sr=16000):
    """Read FLAC files with soundfile and chop into fixed-length WAV clips."""
    target_samples = int(clip_duration_sec * target_sr)
    os.makedirs(output_dir, exist_ok=True)
    clip_count = 0

    for flac_path in flac_files:
        if clip_count >= n_clips:
            break

        try:
            audio, sr = sf.read(flac_path, dtype='float32')
        except Exception as e:
            logger.warning(f"Failed to read {flac_path}: {e}")
            continue

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import soxr
                audio = soxr.resample(audio, sr, target_sr)
            except ImportError:
                from scipy.signal import resample as sp_resample
                n_samples = int(len(audio) * target_sr / sr)
                audio = sp_resample(audio, n_samples)

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Chop into clips
        for start in range(0, len(audio_int16) - target_samples, target_samples):
            if clip_count >= n_clips:
                break

            clip = audio_int16[start:start + target_samples]
            out_path = os.path.join(output_dir, f"speech_{clip_count:05d}.wav")
            wavfile.write(out_path, target_sr, clip)
            clip_count += 1

            if clip_count % 500 == 0:
                logger.info(f"  Generated {clip_count}/{n_clips} speech clips...")

    logger.info(f"Generated {clip_count} speech clips")
    return clip_count


def generate_noise_clips(output_dir, n_clips=300, clip_duration_sec=1.0, sr=16000):
    """Generate various noise/silence clips as additional negatives."""
    target_samples = int(clip_duration_sec * sr)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    for i in range(n_clips):
        noise_type = i % 4

        if noise_type == 0:
            clip = (rng.randn(target_samples) * 50).astype(np.int16)
        elif noise_type == 1:
            clip = (rng.randn(target_samples) * 500).astype(np.int16)
        elif noise_type == 2:
            white = rng.randn(target_samples)
            pink = np.zeros(target_samples)
            pink[0] = white[0]
            for j in range(1, target_samples):
                pink[j] = 0.98 * pink[j-1] + white[j]
            pink = pink / (np.max(np.abs(pink)) + 1e-10) * 1000
            clip = pink.astype(np.int16)
        else:
            clip = np.zeros(target_samples, dtype=np.int16)

        out_path = os.path.join(output_dir, f"noise_{i:05d}.wav")
        wavfile.write(out_path, sr, clip)

    logger.info(f"Generated {n_clips} noise/silence clips")
    return n_clips


def main():
    parser = argparse.ArgumentParser(description='Download negative training data')
    parser.add_argument('--output_dir', default='../oww-training/negative_samples')
    parser.add_argument('--n_clips', type=int, default=3000)
    parser.add_argument('--clip_duration', type=float, default=1.0)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--cache_dir', default=None,
                        help='Directory to cache the LibriSpeech download (default: temp dir)')
    args = parser.parse_args()

    n_speech = int(args.n_clips * 0.85)
    n_noise = args.n_clips - n_speech

    logger.info("=" * 60)
    logger.info(f"Target: {n_speech} speech clips + {n_noise} noise clips")
    logger.info(f"Clip duration: {args.clip_duration}s @ {args.sr}Hz")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # Download LibriSpeech
    cache_dir = args.cache_dir or os.path.join(tempfile.gettempdir(), "librispeech_cache")
    os.makedirs(cache_dir, exist_ok=True)

    download_and_extract(LIBRISPEECH_URL, cache_dir)

    flac_files = find_flac_files(cache_dir)
    logger.info(f"Found {len(flac_files)} FLAC files")

    if not flac_files:
        logger.error("No FLAC files found after extraction!")
        sys.exit(1)

    # Chop into clips
    speech_count = chop_audio_to_clips(
        flac_files, args.output_dir, n_speech, args.clip_duration, args.sr
    )

    noise_count = generate_noise_clips(
        args.output_dir, n_noise, args.clip_duration, args.sr
    )

    logger.info("=" * 60)
    logger.info(f"Done! {speech_count + noise_count} clips in {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()