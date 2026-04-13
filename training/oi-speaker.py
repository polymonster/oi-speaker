#!/usr/bin/env python3
"""
Train a custom openwakeword model from your own audio clips.
Skips all Piper/TTS synthetic generation — just uses your wav files directly.

Usage:
    python train_oi_speaker.py \
        --positive_dir ../oww-training/positive_samples \
        --negative_dir ../oww-training/negative_samples \
        --model_name oi_speaker \
        --output_dir oi_speaker_model \
        --epochs 100 \
        --batch_size 32

Requirements (clean venv):
    pip install numpy scipy scikit-learn onnxruntime torch onnx soundfile soxr
    pip install -e /path/to/openWakeWord --no-deps
    # Plus melspectrogram.onnx and embedding_model.onnx in openwakeword/resources/models/
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─── Audio helpers ───

def load_wav_16k_mono(filepath):
    """Load a wav file, convert to 16kHz mono int16."""
    sr, data = wavfile.read(filepath)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (data * 32767).astype(np.int16)

    if sr != 16000:
        import soxr
        data = data.astype(np.float32) / 32767.0
        data = soxr.resample(data, sr, 16000)
        data = (data * 32767).astype(np.int16)

    return data.astype(np.int16)


def pad_or_trim(audio, target_length):
    """Pad with zeros or trim audio to exact target length."""
    if len(audio) >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - len(audio)), mode='constant')


def load_clips_from_dir(directory, clip_duration_sec=3.0, sr=16000):
    """Load all wav files from a directory, pad/trim to fixed length."""
    target_length = int(clip_duration_sec * sr)
    clips = []
    files = sorted(glob.glob(os.path.join(directory, '*.wav')))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True))

    logger.info(f"Found {len(files)} wav files in {directory}")

    for f in files:
        try:
            audio = load_wav_16k_mono(f)
            audio = pad_or_trim(audio, target_length)
            clips.append(audio)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    return clips, files


# ─── Feature extraction ───

def compute_features(audio_clips, batch_size=32):
    """
    Compute openwakeword embedding features for audio clips.
    Uses openwakeword's AudioFeatures class directly.
    Returns numpy array of shape (n_clips, n_frames, 96).
    """
    from openwakeword.utils import AudioFeatures
    F = AudioFeatures(device='cpu', ncpu=4)

    # Check what methods are available
    if hasattr(F, 'embed_clips'):
        logger.info(f"Using embed_clips (batch_size={batch_size})")
        audio_array = np.array(audio_clips)
        return F.embed_clips(audio_array, batch_size=batch_size)

    # Fallback: process one at a time
    logger.info("Using single-clip embedding (embed_clips not available)")
    all_features = []
    for i, clip in enumerate(audio_clips):
        features = F.embed(clip)
        all_features.append(features)
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(audio_clips)} clips...")

    return np.array(all_features)


# ─── Augmentation ───

def augment_with_noise(positive_clips, negative_clips, n_augmented=None, snr_range=(5, 20)):
    """Mix positive clips with random negative clips at various SNR levels."""
    if n_augmented is None:
        n_augmented = len(positive_clips) * 3

    augmented = []
    rng = np.random.RandomState(42)

    for i in range(n_augmented):
        pos_idx = i % len(positive_clips)
        neg_idx = rng.randint(0, len(negative_clips))

        pos = positive_clips[pos_idx].astype(np.float32)
        neg = negative_clips[neg_idx].astype(np.float32)

        snr_db = rng.uniform(snr_range[0], snr_range[1])
        pos_power = np.mean(pos ** 2) + 1e-10
        neg_power = np.mean(neg ** 2) + 1e-10
        scale = np.sqrt(pos_power / (neg_power * 10 ** (snr_db / 10)))

        mixed = pos + neg * scale
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        augmented.append(mixed)

    return augmented


# ─── Model ───

class WakeWordModel(nn.Module):
    """Small FC model on top of frozen openwakeword features."""
    def __init__(self, n_features=16, feature_dim=96):
        super().__init__()
        input_dim = n_features * feature_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.model(x)


# ─── Export ───

def export_to_onnx(model, n_features, feature_dim, output_path):
    """Export trained PyTorch model to ONNX."""
    model.eval()
    dummy_input = torch.randn(1, n_features, feature_dim)
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['x'], output_names=['output'],
        dynamic_axes={'x': {0: 'batch_size'}},
        opset_version=12,
    )
    logger.info(f"Exported ONNX model to {output_path}")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description='Train custom openwakeword model')
    parser.add_argument('--positive_dir', required=True)
    parser.add_argument('--negative_dir', required=True)
    parser.add_argument('--model_name', default='oi_speaker')
    parser.add_argument('--output_dir', default='oi_speaker_model')
    parser.add_argument('--clip_duration', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--augment_ratio', type=int, default=3)
    parser.add_argument('--val_split', type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load clips ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading audio clips")
    logger.info("=" * 60)

    positive_clips, _ = load_clips_from_dir(args.positive_dir, args.clip_duration)
    negative_clips, _ = load_clips_from_dir(args.negative_dir, args.clip_duration)

    if not positive_clips:
        logger.error(f"No positive clips found in {args.positive_dir}")
        sys.exit(1)
    if not negative_clips:
        logger.error(f"No negative clips found in {args.negative_dir}")
        sys.exit(1)

    logger.info(f"Loaded {len(positive_clips)} positive, {len(negative_clips)} negative")

    # ── Step 2: Augment ──
    logger.info("=" * 60)
    logger.info("Step 2: Augmenting positive clips with background noise")
    logger.info("=" * 60)

    n_aug = len(positive_clips) * args.augment_ratio
    augmented = augment_with_noise(positive_clips, negative_clips, n_augmented=n_aug)
    all_positive = positive_clips + augmented
    logger.info(f"Total positive after augmentation: {len(all_positive)}")

    # ── Step 3: Compute features ──
    logger.info("=" * 60)
    logger.info("Step 3: Computing openwakeword features")
    logger.info("=" * 60)

    logger.info("Computing positive features...")
    pos_features = compute_features(all_positive, batch_size=args.batch_size)
    logger.info(f"Positive features shape: {pos_features.shape}")

    logger.info("Computing negative features...")
    neg_features = compute_features(negative_clips, batch_size=args.batch_size)
    logger.info(f"Negative features shape: {neg_features.shape}")

    # ── Step 4: Prepare training data ──
    logger.info("=" * 60)
    logger.info("Step 4: Preparing training data")
    logger.info("=" * 60)

    n_feature_frames = 16
    feature_dim = 96

    def extract_windows(features, n_frames=16):
        """Extract the last n_frames from each clip's features."""
        windows = []
        for feat in features:
            if feat.shape[0] >= n_frames:
                windows.append(feat[-n_frames:])
            else:
                padded = np.zeros((n_frames, feat.shape[1]))
                padded[-feat.shape[0]:] = feat
                windows.append(padded)
        return np.array(windows)

    pos_windows = extract_windows(pos_features, n_feature_frames)
    neg_windows = extract_windows(neg_features, n_feature_frames)
    logger.info(f"Positive windows: {pos_windows.shape}, Negative windows: {neg_windows.shape}")

    X = np.concatenate([pos_windows, neg_windows], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(pos_windows)), np.zeros(len(neg_windows))]).astype(np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    val_size = int(len(X) * args.val_split)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # ── Step 5: Train ──
    logger.info("=" * 60)
    logger.info("Step 5: Training")
    logger.info("=" * 60)

    model = WakeWordModel(n_features=n_feature_frames, feature_dim=feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        for i in range(0, len(X_train_t), args.batch_size):
            bx = X_train_t[i:i+args.batch_size]
            by = y_train_t[i:i+args.batch_size]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_acc = ((val_out > 0.5).float() == y_val_t).float().mean().item()
            train_out = model(X_train_t)
            train_acc = ((train_out > 0.5).float() == y_train_t).float().mean().item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {epoch_loss/n_batches:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        logger.info(f"Loaded best model (val_loss={best_val_loss:.4f})")

    # ── Step 6: Export ──
    logger.info("=" * 60)
    logger.info("Step 6: Exporting model")
    logger.info("=" * 60)

    onnx_path = os.path.join(args.output_dir, f'{args.model_name}.onnx')
    export_to_onnx(model, n_feature_frames, feature_dim, onnx_path)

    torch_path = os.path.join(args.output_dir, f'{args.model_name}.pt')
    torch.save(model.state_dict(), torch_path)
    logger.info(f"Saved PyTorch model to {torch_path}")

    # ── Step 7: Sanity check ──
    logger.info("=" * 60)
    logger.info("Step 7: Sanity check")
    logger.info("=" * 60)

    model.eval()
    with torch.no_grad():
        pos_scores = model(torch.from_numpy(pos_windows[:5]).float()).numpy().flatten()
        neg_scores = model(torch.from_numpy(neg_windows[:5]).float()).numpy().flatten()
        logger.info(f"Positive scores: {pos_scores}")
        logger.info(f"Negative scores: {neg_scores}")

    logger.info("=" * 60)
    logger.info(f"Done! Model at: {onnx_path}")
    logger.info(f"Usage:")
    logger.info(f"  from openwakeword.model import Model")
    logger.info(f"  model = Model(wakeword_models=['{onnx_path}'])")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()