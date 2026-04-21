#!/usr/bin/env python3
"""
Preprocess positive wake-word recordings:
  - Splits a file containing multiple utterances into individual clips
    (splits on silence gaps >= --split_silence_ms)
  - Trims leading/trailing silence from each clip
  - Writes one numbered wav per clip

Usage:
    python preprocess-positives.py --input_dir raw/ --output_dir positives/
    python preprocess-positives.py --input raw/long_session.wav --output_dir positives/
"""

import os
import sys
import glob
import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import soxr


def load_wav_16k_mono(filepath):
    sr, data = wavfile.read(filepath)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
    if sr != 16000:
        data = soxr.resample(data.astype(np.float32) / 32767.0, sr, 16000)
        data = (data * 32767).astype(np.int16)
    return data


def rms_frames(audio, frame_samples):
    """Return RMS energy per frame."""
    n_frames = len(audio) // frame_samples
    frames = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples).astype(np.float32)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def find_speech_regions(audio, sr=16000, frame_ms=10, energy_threshold=200,
                        split_silence_ms=400, min_duration_ms=200, pad_ms=80):
    """
    Returns a list of (start_sample, end_sample) tuples, one per utterance.
    Frames below energy_threshold are silence.
    Silence gaps >= split_silence_ms cause a new utterance boundary.
    Regions shorter than min_duration_ms are discarded.
    Each region is padded by pad_ms on both sides.
    """
    frame_samples = int(sr * frame_ms / 1000)
    rms = rms_frames(audio, frame_samples)
    is_speech = rms > energy_threshold

    split_frames = int(split_silence_ms / frame_ms)
    min_frames = int(min_duration_ms / frame_ms)
    pad_frames = int(pad_ms / frame_ms)

    regions = []
    in_speech = False
    start = 0
    silent_run = 0

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                start = i
                in_speech = True
            silent_run = 0
        else:
            if in_speech:
                silent_run += 1
                if silent_run >= split_frames:
                    end = i - silent_run + 1
                    if end - start >= min_frames:
                        regions.append((start, end))
                    in_speech = False
                    silent_run = 0

    if in_speech:
        end = len(is_speech)
        if end - start >= min_frames:
            regions.append((start, end))

    # Convert frame indices to samples, add padding
    result = []
    for (fs, fe) in regions:
        s = max(0, (fs - pad_frames) * frame_samples)
        e = min(len(audio), (fe + pad_frames) * frame_samples)
        result.append((s, e))

    return result


def trim_silence(audio, sr=16000, frame_ms=10, energy_threshold=200):
    """Trim leading and trailing silence."""
    frame_samples = int(sr * frame_ms / 1000)
    rms = rms_frames(audio, frame_samples)
    speech_frames = np.where(rms > energy_threshold)[0]
    if len(speech_frames) == 0:
        return audio
    start = speech_frames[0] * frame_samples
    end = (speech_frames[-1] + 1) * frame_samples
    return audio[start:end]


def process_file(filepath, output_dir, args, file_index):
    audio = load_wav_16k_mono(filepath)

    regions = find_speech_regions(
        audio,
        split_silence_ms=args.split_silence_ms,
        energy_threshold=args.energy_threshold,
        min_duration_ms=args.min_duration_ms,
        pad_ms=args.pad_ms,
    )

    if not regions:
        print(f"  {os.path.basename(filepath)}: no speech detected, skipping")
        return 0

    saved = 0
    for i, (s, e) in enumerate(regions):
        clip = audio[s:e]
        clip = trim_silence(clip, energy_threshold=args.energy_threshold)
        if len(clip) < 16000 * args.min_duration_ms / 1000:
            continue
        stem = os.path.splitext(os.path.basename(filepath))[0]
        suffix = f"_{i}" if len(regions) > 1 else ""
        out_path = os.path.join(output_dir, f"{stem}{suffix}.wav")
        wavfile.write(out_path, 16000, clip)
        duration_ms = len(clip) / 16000 * 1000
        print(f"  -> {os.path.basename(out_path)}  ({duration_ms:.0f}ms)")
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Preprocess positive wake-word recordings")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Directory of wav files to process")
    group.add_argument("--input", help="Single wav file to process")
    parser.add_argument("--output_dir", required=True, help="Where to write output clips")
    parser.add_argument("--split_silence_ms", type=float, default=400,
                        help="Silence gap (ms) that splits utterances (default: 400)")
    parser.add_argument("--energy_threshold", type=float, default=200,
                        help="RMS energy threshold for speech vs silence (default: 200)")
    parser.add_argument("--min_duration_ms", type=float, default=200,
                        help="Discard clips shorter than this (default: 200ms)")
    parser.add_argument("--pad_ms", type=float, default=80,
                        help="Padding around each detected utterance (default: 80ms)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        files = [args.input]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        files += sorted(glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True))
        files = sorted(set(files))

    if not files:
        print("No wav files found.")
        sys.exit(1)

    total = 0
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
        total += process_file(f, args.output_dir, args, i)

    print(f"\nDone. {total} clips saved to {args.output_dir}")


if __name__ == "__main__":
    main()
