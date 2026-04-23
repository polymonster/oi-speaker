"""Score distribution diagnostic for the wake word model.

Scores are cached by model name so re-runs are instant.
Compare two models by passing --compare to overlay them on one plot.

Usage:
    python training/diagnose.py
    python training/diagnose.py --model models/openwakeword/oi_speaker.onnx
    python training/diagnose.py --model models/openwakeword/oi_speaker_v2.onnx --compare models/openwakeword/oi_speaker.onnx
"""

import argparse
import os
import glob
import numpy as np
import scipy.io.wavfile as wavfile
from openwakeword.model import Model

WAKE_CHUNK = 1280  # 80ms at 16kHz
TARGET_SR = 16000
CACHE_DIR = "training/scores/score_cache"


def _cache_path(model_path: str, label: str) -> str:
    name = os.path.splitext(os.path.basename(model_path))[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_{label}.npy")


def _scores_csv_path(model_path: str, label: str) -> str:
    name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(CACHE_DIR, f"{name}_{label}_scores.csv")


def score_clips(model, model_path: str, wav_dir: str, label: str) -> np.ndarray:
    """Return peak score per clip, loading from cache if available."""
    cache = _cache_path(model_path, label)
    csv_path = _scores_csv_path(model_path, label)
    if os.path.exists(cache):
        scores = np.load(cache)
        print(f"  loaded {len(scores)} cached scores from {cache}")
        return scores

    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not files:
        print(f"  no wav files found in {wav_dir}")
        return np.array([])

    scores = []
    for i, path in enumerate(files):
        try:
            sr, audio = wavfile.read(path)
        except Exception as e:
            print(f"  skipping {path}: {e}")
            continue
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        if sr != TARGET_SR:
            ratio = TARGET_SR / sr
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * ratio)),
                np.arange(len(audio)), audio
            ).astype(np.int16)
        model.reset()
        peak = 0.0
        for j in range(0, len(audio) - WAKE_CHUNK, WAKE_CHUNK):
            prediction = model.predict(audio[j:j + WAKE_CHUNK])
            for _, s in prediction.items():
                if s > peak:
                    peak = s
        scores.append(peak)
        if (i + 1) % 500 == 0:
            print(f"  scored {i + 1}/{len(files)}...")

    arr = np.array(scores)
    np.save(cache, arr)
    with open(csv_path, "w") as f:
        f.write("filename,score\n")
        for path, score in zip(files, arr):
            f.write(f"{os.path.basename(path)},{score:.6f}\n")
    print(f"  cached to {cache}")
    print(f"  scores written to {csv_path}")
    return arr


def print_stats(label: str, scores: np.ndarray):
    if not len(scores):
        return
    print(f"\n{label} ({len(scores)} clips):")
    print(f"  mean={scores.mean():.3f}  std={scores.std():.3f}")
    print(f"  p10={np.percentile(scores,10):.3f}  p50={np.percentile(scores,50):.3f}  p90={np.percentile(scores,90):.3f}  max={scores.max():.3f}")


def suggest_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    """Find threshold maximising TPR - FPR."""
    best_t, best_gap = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 200):
        gap = (pos >= t).mean() - (neg >= t).mean()
        if gap > best_gap:
            best_gap, best_t = gap, t
    tpr = (pos >= best_t).mean()
    fpr = (neg >= best_t).mean()
    print(f"\nsuggested threshold: {best_t:.2f}  →  TPR={tpr:.1%}  FPR={fpr:.1%}")
    return best_t


def plot(results: list[tuple[str, np.ndarray, np.ndarray, float]]):
    """results: list of (model_label, pos_scores, neg_scores, threshold)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot (pip install matplotlib)")
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 4), squeeze=False)
    bins = np.linspace(0, 1, 50)

    for ax, (name, pos, neg, t) in zip(axes[0], results):
        ax.hist(neg, bins=bins, alpha=0.6, color="tomato",   label=f"negatives (n={len(neg)})")
        ax.hist(pos, bins=bins, alpha=0.6, color="steelblue", label=f"positives (n={len(pos)})")
        ax.axvline(t, color="black", linestyle="--", label=f"threshold {t:.2f}")
        ax.set_title(name)
        ax.set_xlabel("peak score per clip")
        ax.set_ylabel("count")
        ax.legend()

    plt.tight_layout()
    name = results[0][0] if len(results) == 1 else "comparison"
    out = os.path.join(CACHE_DIR, f"{name}_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"plot saved to {out}")
    plt.show()


def load_model_and_score(model_path, pos_dir, neg_dir, skip_pos=False, skip_neg=False):
    print(f"\nloading model: {model_path}")
    model = Model(wakeword_models=[model_path], vad_threshold=0.5, enable_speex_noise_suppression=False)
    pos = np.array([])
    neg = np.array([])
    if not skip_pos:
        print(f"scoring positives: {pos_dir}")
        pos = score_clips(model, model_path, pos_dir, "pos")
    if not skip_neg:
        print(f"scoring negatives: {neg_dir}")
        neg = score_clips(model, model_path, neg_dir, "neg")
    return pos, neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="models/openwakeword/oi_speaker.onnx")
    parser.add_argument("--compare", default=None, help="second model to overlay")
    parser.add_argument("--pos", default="training/training_data/positive_samples")
    parser.add_argument("--neg", default="training/training_data/negative_samples")
    parser.add_argument("--pos-only", action="store_true", help="score positives only")
    parser.add_argument("--neg-only", action="store_true", help="score negatives only")
    args = parser.parse_args()

    models = [args.model]
    if args.compare:
        models.append(args.compare)

    results = []
    for m in models:
        pos, neg = load_model_and_score(m, args.pos, args.neg, skip_pos=args.neg_only, skip_neg=args.pos_only)
        name = os.path.splitext(os.path.basename(m))[0]
        print_stats(f"{name} positives", pos)
        print_stats(f"{name} negatives", neg)
        if len(pos) and len(neg):
            t = suggest_threshold(pos, neg)
            results.append((name, pos, neg, t))

    if results:
        plot(results)
    else:
        print("need both positive and negative samples to compare")


if __name__ == "__main__":
    main()
    
# Scores are cached per model name in `training/score_cache/` so the negatives only need to run once per model.
