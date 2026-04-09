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
    pip install openwakeword numpy scipy scikit-learn onnxruntime torch torchaudio onnx
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
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─── Audio feature extraction using openwakeword's frozen models ───

def get_feature_extractor():
    """Load the openwakeword melspec + embedding ONNX models for feature extraction."""
    try:
        from openwakeword.utils import AudioFeatures
        return AudioFeatures(device='cpu', ncpu=4)
    except ImportError:
        pass

    # Fallback: load the ONNX models directly
    import onnxruntime as ort

    # Find the model files
    oww_dir = None
    for candidate in [
        os.path.join(os.path.dirname(__file__), 'openwakeword', 'resources', 'models'),
        os.path.expanduser('~/.local/lib/python3.10/site-packages/openwakeword/resources/models'),
    ]:
        if os.path.exists(candidate):
            oww_dir = candidate
            break

    if oww_dir is None:
        raise RuntimeError("Cannot find openwakeword model files. Make sure openwakeword is installed.")

    melspec_path = os.path.join(oww_dir, 'melspectrogram.onnx')
    embed_path = os.path.join(oww_dir, 'embedding_model.onnx')

    if not os.path.exists(melspec_path) or not os.path.exists(embed_path):
        # Try tflite versions
        melspec_path = os.path.join(oww_dir, 'melspectrogram.tflite')
        embed_path = os.path.join(oww_dir, 'embedding_model.tflite')

    logger.info(f"Loading melspec model: {melspec_path}")
    logger.info(f"Loading embedding model: {embed_path}")

    return {
        'melspec': ort.InferenceSession(melspec_path, providers=['CPUExecutionProvider']),
        'embedding': ort.InferenceSession(embed_path, providers=['CPUExecutionProvider']),
    }


def load_wav_16k_mono(filepath):
    """Load a wav file, convert to 16kHz mono int16."""
    sr, data = wavfile.read(filepath)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Convert to int16 if float
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (data * 32767).astype(np.int16)

    # Resample to 16kHz if needed
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
    else:
        return np.pad(audio, (0, target_length - len(audio)), mode='constant')


def compute_features_batch(audio_clips, feature_extractor):
    """
    Compute openwakeword embedding features for a batch of audio clips.
    Each clip should be int16 16kHz mono.
    Returns numpy array of shape (n_clips, n_frames, 96).
    """
    try:
        # Try using openwakeword's built-in AudioFeatures
        from openwakeword.utils import AudioFeatures
        if isinstance(feature_extractor, AudioFeatures):
            features = feature_extractor.embed_clips(audio_clips, batch_size=32)
            return features
    except (ImportError, AttributeError):
        pass

    # Manual ONNX approach
    melspec_session = feature_extractor['melspec']
    embed_session = feature_extractor['embedding']

    all_features = []
    for clip in audio_clips:
        clip_f32 = clip.astype(np.float32) / 32767.0

        # Compute melspectrogram
        mel_input = clip_f32.reshape(1, -1)
        mel_out = melspec_session.run(None, {melspec_session.get_inputs()[0].name: mel_input})[0]

        # Compute embeddings (process in windows of 76 mel frames)
        n_frames = mel_out.shape[1]
        window_size = 76
        embeddings = []
        for i in range(0, n_frames - window_size + 1, 8):  # step of 8
            window = mel_out[:, i:i+window_size, :]
            emb = embed_session.run(None, {embed_session.get_inputs()[0].name: window})[0]
            embeddings.append(emb)

        if embeddings:
            all_features.append(np.concatenate(embeddings, axis=0))

    return np.array(all_features) if all_features else np.array([])


# ─── The wake word detection model (small DNN) ───

class WakeWordModel(nn.Module):
    """
    Small fully-connected model that takes openwakeword features
    and predicts wake word presence. This matches the architecture
    used in the official openwakeword training notebook.
    """
    def __init__(self, n_features=16, feature_dim=96, n_classes=1):
        super().__init__()
        input_dim = n_features * feature_dim  # flatten the feature window

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, n_frames, 96) -> flatten
        x = x.reshape(x.shape[0], -1)
        return self.model(x)


# ─── Data loading ───

def load_clips_from_dir(directory, clip_duration_sec=3.0, sr=16000):
    """Load all wav files from a directory, pad/trim to fixed length."""
    target_length = int(clip_duration_sec * sr)
    clips = []
    files = sorted(glob.glob(os.path.join(directory, '*.wav')))

    if not files:
        # Also try recursive
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


def augment_with_noise(positive_clips, negative_clips, n_augmented=None, snr_range=(5, 20)):
    """
    Mix positive clips with random negative clips at various SNR levels.
    This simulates the wake word being spoken over background noise/speech.
    """
    if n_augmented is None:
        n_augmented = len(positive_clips) * 3  # 3x augmentation

    augmented = []
    rng = np.random.RandomState(42)

    for i in range(n_augmented):
        pos_idx = i % len(positive_clips)
        neg_idx = rng.randint(0, len(negative_clips))

        pos = positive_clips[pos_idx].astype(np.float32)
        neg = negative_clips[neg_idx].astype(np.float32)

        # Random SNR
        snr_db = rng.uniform(snr_range[0], snr_range[1])
        pos_power = np.mean(pos ** 2) + 1e-10
        neg_power = np.mean(neg ** 2) + 1e-10
        scale = np.sqrt(pos_power / (neg_power * 10 ** (snr_db / 10)))

        mixed = pos + neg * scale
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        augmented.append(mixed)

    return augmented


# ─── Export to ONNX ───

def export_to_onnx(model, n_features, feature_dim, output_path):
    """Export the trained PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, n_features, feature_dim)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['x'],
        output_names=['output'],
        dynamic_axes={'x': {0: 'batch_size'}},
        opset_version=12,
    )
    logger.info(f"Exported ONNX model to {output_path}")


# ─── Main training loop ───

def main():
    parser = argparse.ArgumentParser(description='Train custom openwakeword model')
    parser.add_argument('--positive_dir', required=True, help='Directory with positive wake word wav files')
    parser.add_argument('--negative_dir', required=True, help='Directory with negative wav files')
    parser.add_argument('--model_name', default='oi_speaker', help='Name for the output model')
    parser.add_argument('--output_dir', default='oi_speaker_model', help='Output directory')
    parser.add_argument('--clip_duration', type=float, default=3.0, help='Clip duration in seconds')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--augment_ratio', type=int, default=3, help='Augmentation multiplier for positive clips')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Step 1: Load audio clips ───
    logger.info("=" * 60)
    logger.info("Step 1: Loading audio clips")
    logger.info("=" * 60)

    positive_clips, pos_files = load_clips_from_dir(args.positive_dir, args.clip_duration)
    negative_clips, neg_files = load_clips_from_dir(args.negative_dir, args.clip_duration)

    if len(positive_clips) == 0:
        logger.error(f"No positive clips found in {args.positive_dir}")
        sys.exit(1)
    if len(negative_clips) == 0:
        logger.error(f"No negative clips found in {args.negative_dir}")
        sys.exit(1)

    logger.info(f"Loaded {len(positive_clips)} positive clips, {len(negative_clips)} negative clips")

    # ─── Step 2: Augment positive clips with noise ───
    logger.info("=" * 60)
    logger.info("Step 2: Augmenting positive clips with background noise")
    logger.info("=" * 60)

    n_augmented = len(positive_clips) * args.augment_ratio
    augmented_positive = augment_with_noise(positive_clips, negative_clips, n_augmented=n_augmented)
    all_positive = positive_clips + augmented_positive
    logger.info(f"Total positive clips after augmentation: {len(all_positive)}")

    # ─── Step 3: Compute features ───
    logger.info("=" * 60)
    logger.info("Step 3: Computing openwakeword features (this may take a while)")
    logger.info("=" * 60)

    feature_extractor = get_feature_extractor()

    logger.info("Computing positive features...")
    pos_features = compute_features_batch(all_positive, feature_extractor)
    logger.info(f"Positive features shape: {pos_features.shape}")

    logger.info("Computing negative features...")
    neg_features = compute_features_batch(negative_clips, feature_extractor)
    logger.info(f"Negative features shape: {neg_features.shape}")

    # ─── Step 4: Prepare training data ───
    logger.info("=" * 60)
    logger.info("Step 4: Preparing training data")
    logger.info("=" * 60)

    # Use the last N frames as features (the model looks at a window)
    n_feature_frames = 16  # standard openwakeword window
    feature_dim = 96       # embedding dimension

    def extract_windows(features, n_frames=16):
        """Extract the last n_frames from each clip's features."""
        windows = []
        for feat in features:
            if feat.shape[0] >= n_frames:
                windows.append(feat[-n_frames:])
            else:
                # Pad with zeros if too short
                padded = np.zeros((n_frames, feat.shape[1]))
                padded[-feat.shape[0]:] = feat
                windows.append(padded)
        return np.array(windows)

    pos_windows = extract_windows(pos_features, n_feature_frames)
    neg_windows = extract_windows(neg_features, n_feature_frames)

    logger.info(f"Positive windows: {pos_windows.shape}")
    logger.info(f"Negative windows: {neg_windows.shape}")

    # Create labels
    X = np.concatenate([pos_windows, neg_windows], axis=0).astype(np.float32)
    y = np.concatenate([
        np.ones(len(pos_windows)),
        np.zeros(len(neg_windows))
    ]).astype(np.float32)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Train/val split
    val_size = int(len(X) * args.val_split)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # ─── Step 5: Train the model ───
    logger.info("=" * 60)
    logger.info("Step 5: Training")
    logger.info("=" * 60)

    model = WakeWordModel(n_features=n_feature_frames, feature_dim=feature_dim, n_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Shuffle training data each epoch
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        for i in range(0, len(X_train_t), args.batch_size):
            batch_X = X_train_t[i:i+args.batch_size]
            batch_y = y_train_t[i:i+args.batch_size]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()
            val_preds = (val_output > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

            # Training accuracy
            train_output = model(X_train_t)
            train_preds = (train_output > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model (val_loss={best_val_loss:.4f})")

    # ─── Step 6: Export ───
    logger.info("=" * 60)
    logger.info("Step 6: Exporting model")
    logger.info("=" * 60)

    onnx_path = os.path.join(args.output_dir, f'{args.model_name}.onnx')
    export_to_onnx(model, n_feature_frames, feature_dim, onnx_path)

    # Save PyTorch model too
    torch_path = os.path.join(args.output_dir, f'{args.model_name}.pt')
    torch.save(model.state_dict(), torch_path)
    logger.info(f"Saved PyTorch model to {torch_path}")

    # ─── Step 7: Quick sanity test ───
    logger.info("=" * 60)
    logger.info("Step 7: Sanity check")
    logger.info("=" * 60)

    model.eval()
    with torch.no_grad():
        # Test on a few positive clips
        test_pos = torch.from_numpy(pos_windows[:5])
        test_neg = torch.from_numpy(neg_windows[:5])
        pos_scores = model(test_pos).numpy().flatten()
        neg_scores = model(test_neg).numpy().flatten()
        logger.info(f"Sample positive scores: {pos_scores}")
        logger.info(f"Sample negative scores: {neg_scores}")

    logger.info("=" * 60)
    logger.info("Done! To use the model:")
    logger.info(f"  from openwakeword.model import Model")
    logger.info(f"  model = Model(wakeword_models=['{onnx_path}'])")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()