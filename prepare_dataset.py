#!/usr/bin/env python3
"""
Dataset preparation script for ESKMeans.

Extracts MFCC or HuBERT features from WAV files and computes energy-based
landmarks, saving outputs in the directory structure expected by data_utils.py.

Set ESKMEANS_DATA to the output base directory (same value used when running):
    export ESKMEANS_DATA=/path/to/data

Usage (MFCC):
    python prepare_dataset.py \
        --wav_dir data/wavs/s01 \
        --language buckeye \
        --speaker s01 \
        --feature_type mfcc

Usage (HuBERT):
    python prepare_dataset.py \
        --wav_dir data/wavs/s01 \
        --language buckeye \
        --speaker s01 \
        --feature_type hubert_base_ls960 \
        --layer 10 \
        --device cuda

Required packages:
    pip install librosa scipy numpy
    pip install transformers torch   # only for HuBERT
"""

import argparse
import os
import pickle
import numpy as np
from pathlib import Path

SR = 16000
MFCC_N_MFCC = 13
MFCC_HOP_LENGTH = 160    # 10 ms at 16 kHz  →  1 frame = 1 centisecond
HUBERT_HOP_LENGTH = 320  # 20 ms at 16 kHz  →  1 frame = 2 centiseconds

# Minimum segment duration = 200 ms (matches min_duration=20 in run.py)
MFCC_MIN_DIST_FRAMES = 20   # 20 frames × 10 ms = 200 ms
HUBERT_MIN_DIST_FRAMES = 10  # 10 frames × 20 ms = 200 ms


def load_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def extract_mfcc(y):
    import librosa
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=MFCC_N_MFCC,
        hop_length=MFCC_HOP_LENGTH,
        win_length=400,  # 25 ms window
    )
    return mfcc.T  # (frames, 13)


def extract_hubert(y, model, processor, layer, device):
    import torch
    inputs = processor(y, return_tensors='pt', sampling_rate=SR)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states[0] = CNN features, [1..12] = transformer layers
    return out.hidden_states[layer].squeeze(0).cpu().numpy()  # (frames, 768)


def load_hubert_model(device):
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    print("Loading facebook/hubert-base-ls960 ...")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval().to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    return model, processor


def compute_landmarks(feats, min_distance_frames):
    """
    Compute energy-based landmarks from feature frames.

    Uses peaks in the L2-norm derivative as segment boundaries.
    Always includes the last frame as the final landmark.

    Returns a list of frame indices.
    """
    from scipy.signal import find_peaks
    energy = np.linalg.norm(feats, axis=1)
    diff = np.abs(np.diff(energy))
    peaks, _ = find_peaks(diff, distance=min_distance_frames)
    return sorted(set(peaks.tolist() + [len(feats) - 1]))


def to_centiseconds(frame_idx, hop_length):
    """Convert a frame index to centiseconds (10 ms units)."""
    ms = frame_idx * hop_length / SR * 1000
    return int(ms / 10)


def resolve_output_base():
    base = os.environ.get('ESKMEANS_DATA')
    if base is None:
        base = os.path.join(os.path.dirname(__file__), 'data', 'prepared')
        print(f"ESKMEANS_DATA not set, writing to {base}")
    return base


def main():
    parser = argparse.ArgumentParser(description='Prepare ESKMeans dataset')
    parser.add_argument('--wav_dir', required=True,
                        help='Directory containing WAV files for this speaker')
    parser.add_argument('--language', required=True, choices=['buckeye', 'mandarin'],
                        help='Dataset / language name')
    parser.add_argument('--speaker', required=True,
                        help='Speaker ID (e.g. s01)')
    parser.add_argument('--feature_type', required=True,
                        choices=['mfcc', 'hubert_base_ls960'],
                        help='Feature type to extract')
    parser.add_argument('--layer', type=int, default=10,
                        help='HuBERT transformer layer (1-12, default: 10)')
    parser.add_argument('--device', default='cpu',
                        help='Device for HuBERT inference: cpu or cuda (default: cpu)')
    args = parser.parse_args()

    wav_files = sorted(Path(args.wav_dir).glob('*.wav'))
    if not wav_files:
        print(f"No WAV files found in {args.wav_dir}")
        return

    print(f"Found {len(wav_files)} WAV file(s) in {args.wav_dir}")

    if args.feature_type == 'hubert_base_ls960':
        model, processor = load_hubert_model(args.device)
        hop_length = HUBERT_HOP_LENGTH
        min_dist = HUBERT_MIN_DIST_FRAMES
    else:
        hop_length = MFCC_HOP_LENGTH
        min_dist = MFCC_MIN_DIST_FRAMES

    features = {}
    landmarks = {}

    for wav_path in wav_files:
        print(f"  {wav_path.name} ...", end=' ', flush=True)
        y = load_audio(wav_path)

        if args.feature_type == 'mfcc':
            feats = extract_mfcc(y)
        else:
            feats = extract_hubert(y, model, processor, args.layer, args.device)

        raw_lm = compute_landmarks(feats, min_dist)
        lm_cs = [to_centiseconds(i, hop_length) for i in raw_lm]
        n_cs = to_centiseconds(len(feats) - 1, hop_length)

        # Utterance ID format: <stem>_<start_cs>-<end_cs>
        utt_id = f"{wav_path.stem}_0-{n_cs}"
        features[utt_id] = feats
        landmarks[utt_id] = lm_cs
        print(f"{len(feats)} frames, {len(lm_cs)} landmarks")

    base = resolve_output_base()

    # Landmarks are always saved under the mfcc_herman path (shared across feature types)
    lm_dir = Path(base) / 'zerospeech_seg' / 'mfcc_herman' / args.language / args.speaker
    lm_dir.mkdir(parents=True, exist_ok=True)
    lm_path = lm_dir / 'landmarks.pkl'
    with open(lm_path, 'wb') as f:
        pickle.dump(landmarks, f)
    print(f"\nSaved landmarks ({len(landmarks)} utterances) → {lm_path}")

    # Features
    if args.feature_type == 'mfcc':
        feat_path = lm_dir / 'raw_mfcc.npz'
    else:
        feat_dir = (Path(base) / 'hubert_data' / 'seg' / 'zsc' /
                    args.feature_type / str(args.layer) / args.language / 'prevad')
        feat_dir.mkdir(parents=True, exist_ok=True)
        feat_path = feat_dir / f"{args.speaker}_features_frame.npz"

    np.savez(feat_path, **features)
    print(f"Saved features → {feat_path}")
    print("Done.")


if __name__ == '__main__':
    main()
