#!/usr/bin/env python3
"""
Dataset preparation for ESKMeans — ZeroSpeech / Buckeye.

Steps:
  1. (External, MATLAB) Run thetaOscillator on each recording to get syllable
     boundaries. Save one text file per recording under --landmark_dir:
         s0101a.txt   (one boundary in seconds per line, absolute time in recording)
  2. (This script) Run feature extraction + package everything as one pickle
     per recording per feature type.

Output:
    <ESKMEANS_DATA>/<language>/<recording_id>/mfcc.pkl
    <ESKMEANS_DATA>/<language>/<recording_id>/hubert_base_ls960_l10.pkl
    ...

Each pickle:  utt_id → {'features': np.ndarray, 'landmarks': [int, ...]}
Utterance IDs: <recording_id>_<start_cs>-<end_cs>  (centiseconds)

Landmark file format (produced by thetaOscillator):
    One boundary per line in seconds (absolute time in the recording).
    Example s0101a.txt:
        0.321
        0.487
        1.203
        ...

ZeroSpeech VAD file:
    mkdir -p data/zerospeech
    curl -o data/zerospeech/english_vad.txt \
        https://raw.githubusercontent.com/zerospeech/Zerospeech2015/master/english_vad.txt

Usage:
    export ESKMEANS_DATA=/shared/ramon/experiments_old

    python prepare_dataset.py \
        --buckeye_dir /path/to/buckeye \
        --vad_file data/zerospeech/english_vad.txt \
        --landmark_dir /path/to/theta_oscillator_output \
        --recording s0101a --language buckeye --feature_type mfcc

    # all recordings in parallel
    cat data/file_list/buckeye_spk | parallel \
        python prepare_dataset.py \
            --buckeye_dir /path/to/buckeye \
            --vad_file data/zerospeech/english_vad.txt \
            --landmark_dir /path/to/theta_oscillator_output \
            --recording {} --language buckeye \
            --feature_type hubert_base_ls960 --layer 10 --device cuda

Feature types:  mfcc | hubert_base_ls960 | mhubert | wavlm_large
"""

import argparse
import os
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

SR = 16000
MFCC_HOP = 160    # 10 ms  →  1 frame = 1 centisecond
NEURAL_HOP = 320  # 20 ms  →  1 frame = 2 centiseconds
DEFAULT_DATA_DIR = "/shared/ramon/experiments_old"

NEURAL_MODELS = {
    'hubert_base_ls960': 'facebook/hubert-base-ls960',
    'mhubert':           'utter-project/mHuBERT-147',
    'wavlm_large':       'microsoft/wavlm-large',
}


def data_path(language, recording_id, feature_type, layer):
    base = os.environ.get('ESKMEANS_DATA', DEFAULT_DATA_DIR)
    fname = 'mfcc.pkl' if feature_type == 'mfcc' else f'{feature_type}_l{layer}.pkl'
    return Path(base) / language / recording_id / fname


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def parse_vad(vad_file):
    """ZeroSpeech VAD format: file_id,start_sec,end_sec"""
    vad = defaultdict(list)
    with open(vad_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                vad[parts[0].strip()].append((float(parts[1]), float(parts[2])))
    return {k: sorted(v) for k, v in vad.items()}


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

def load_landmarks(landmark_dir, recording_id):
    """
    Load thetaOscillator boundaries for a recording.
    Expected file: <landmark_dir>/<recording_id>.txt
    One boundary per line in seconds (absolute time in the recording).
    Returns sorted list of floats, or None if file not found.
    """
    path = Path(landmark_dir) / f"{recording_id}.txt"
    if not path.exists():
        return None
    boundaries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    boundaries.append(float(line))
                except ValueError:
                    pass
    return sorted(boundaries)


def landmarks_for_segment(all_boundaries, start_sec, end_sec, hop):
    """
    Select boundaries that fall within [start_sec, end_sec] and convert
    to centisecond offsets relative to the segment start.
    Always appends the last frame of the segment as a boundary.
    """
    dur_cs = int((end_sec - start_sec) * 100)
    lm = []
    for t in all_boundaries:
        if start_sec <= t < end_sec:
            cs = int((t - start_sec) * 100)
            if cs > 0:
                lm.append(cs)
    if not lm or lm[-1] != dur_cs:
        lm.append(dur_cs)
    return sorted(set(lm))


# ---------------------------------------------------------------------------
# Audio & features
# ---------------------------------------------------------------------------

def find_wav(buckeye_dir, recording_id):
    for path in [
        Path(buckeye_dir) / recording_id[:3] / f"{recording_id}.wav",
        Path(buckeye_dir) / f"{recording_id}.wav",
    ]:
        if path.exists():
            return path
    return None


def load_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def extract_mfcc(y):
    import librosa
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13,
                                 hop_length=MFCC_HOP, win_length=400).T


def extract_neural(y, model, processor, layer, device):
    import torch
    inputs = {k: v.to(device)
              for k, v in processor(y, return_tensors='pt', sampling_rate=SR).items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer].squeeze(0).cpu().numpy()


def load_model(feature_type, device):
    from transformers import AutoModel, AutoFeatureExtractor
    hf_id = NEURAL_MODELS[feature_type]
    print(f"Loading {hf_id} ...")
    return (AutoModel.from_pretrained(hf_id).eval().to(device),
            AutoFeatureExtractor.from_pretrained(hf_id))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--buckeye_dir',  required=True)
    parser.add_argument('--vad_file',     required=True)
    parser.add_argument('--landmark_dir', required=True,
                        help='Directory with thetaOscillator output files '
                             '(<recording_id>.txt, one boundary in seconds per line)')
    parser.add_argument('--recording', default=None,
                        help='Recording ID (e.g. s0101a). Omit for all in VAD.')
    parser.add_argument('--language',     required=True, choices=['buckeye', 'mandarin'])
    parser.add_argument('--feature_type', required=True,
                        choices=['mfcc'] + list(NEURAL_MODELS))
    parser.add_argument('--layer',  type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    vad = parse_vad(args.vad_file)
    recordings = ([args.recording] if args.recording
                  else [r for r in sorted(vad) if r in vad])

    is_neural = args.feature_type in NEURAL_MODELS
    model, processor = load_model(args.feature_type, args.device) if is_neural else (None, None)
    hop = NEURAL_HOP if is_neural else MFCC_HOP

    print(f"{len(recordings)} recording(s)  feature={args.feature_type}")

    for rec_id in recordings:
        if rec_id not in vad:
            print(f"[{rec_id}] not in VAD, skipping")
            continue

        wav = find_wav(args.buckeye_dir, rec_id)
        if wav is None:
            print(f"[{rec_id}] WAV not found, skipping")
            continue

        boundaries = load_landmarks(args.landmark_dir, rec_id)
        if boundaries is None:
            print(f"[{rec_id}] landmark file not found in {args.landmark_dir}, skipping")
            continue

        print(f"\n[{rec_id}]  {len(vad[rec_id])} segments  {len(boundaries)} boundaries")
        y_full = load_audio(str(wav))

        data = {}
        for start_sec, end_sec in vad[rec_id]:
            y_seg = y_full[int(start_sec * SR): min(int(end_sec * SR), len(y_full))]
            if len(y_seg) < SR * 0.05:
                continue

            feats = (extract_mfcc(y_seg) if not is_neural
                     else extract_neural(y_seg, model, processor, args.layer, args.device))
            if len(feats) < 2:
                continue

            utt_id = f"{rec_id}_{int(start_sec * 100)}-{int(end_sec * 100)}"
            data[utt_id] = {
                'features':  feats,
                'landmarks': landmarks_for_segment(boundaries, start_sec, end_sec, hop),
            }

        print(f"  {len(data)} utterances")
        out = data_path(args.language, rec_id, args.feature_type, args.layer)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(data, f)
        print(f"  → {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
