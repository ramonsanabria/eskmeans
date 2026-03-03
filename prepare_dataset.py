#!/usr/bin/env python3
"""
Dataset preparation for ESKMeans — ZeroSpeech / Buckeye / Xitsonga.

Computes syllable-like landmarks using the thetaOscillator (Räsänen et al. 2018)
and extracts acoustic features, packaging everything as one pickle per speaker.

Output:
    <ESKMEANS_DATA>/<language>/<speaker_id>/mfcc.pkl
    <ESKMEANS_DATA>/<language>/<speaker_id>/hubert_base_ls960_l10.pkl
    ...

Each pickle:  utt_id → {'features': np.ndarray, 'landmarks': [int, ...]}
Utterance IDs: <recording_id>_<start_cs>-<end_cs>  (centiseconds)

VAD files (download once):
    curl -o data/zerospeech/english_vad.txt \
        https://raw.githubusercontent.com/zerospeech/Zerospeech2015/master/english_vad.txt
    curl -o data/zerospeech/xitsonga_vad.txt \
        https://raw.githubusercontent.com/zerospeech/Zerospeech2015/master/xitsonga_vad.txt

Usage:
    export ESKMEANS_DATA=/shared/ramon/experiments_old

    # ZeroSpeech 2015 languages (english/french/german/mandarin/wolof)
    # --audio_dir points to the language root, --subset selects 1s / 10s / 120s
    # Use 120s for TDE evaluation (recommended).
    python prepare_dataset.py \
        --audio_dir /shared/ramon/experiments_old/english \
        --subset 120s --language english --feature_type mfcc

    # process all languages in parallel
    for lang in english french german mandarin wolof; do
        python prepare_dataset.py \
            --audio_dir /shared/ramon/experiments_old/$lang \
            --subset 120s --language $lang --feature_type mfcc &
    done; wait

    # Buckeye
    python prepare_dataset.py \
        --audio_dir /path/to/buckeye \
        --vad_file data/zerospeech/english_vad.txt \
        --recording s0101a --language buckeye --feature_type mfcc

    # Xitsonga
    python prepare_dataset.py \
        --audio_dir /path/to/nchlt_tso/audio \
        --vad_file data/zerospeech/xitsonga_vad.txt \
        --recording 001 --language xitsonga --feature_type mfcc

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


def data_path(language, speaker_id, feature_type, layer):
    base = os.environ.get('ESKMEANS_DATA', DEFAULT_DATA_DIR)
    fname = 'mfcc.pkl' if feature_type == 'mfcc' else f'{feature_type}_l{layer}.pkl'
    return Path(base) / language / speaker_id / fname


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def parse_vad_buckeye(vad_file):
    """ZeroSpeech English VAD: comma-separated  file_id,start_sec,end_sec
    Returns {recording_id: [(start, end), ...]}"""
    vad = defaultdict(list)
    with open(vad_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    vad[parts[0].strip()].append((float(parts[1]), float(parts[2])))
                except ValueError:
                    pass  # skip header
    return {k: sorted(v) for k, v in vad.items()}


def parse_vad_from_words(words_path, merge_gap=0.3, min_dur=0.1):
    """Derive VAD from a Buckeye .words file (Kamper-style).

    Each line after the header:  end_time  channel  word; phon; canon; POS
    Labels wrapped in <> or {} are non-speech (SIL, NOISE, IVER, etc.).
    Consecutive speech intervals separated by <= merge_gap are merged.
    Returns [(start_sec, end_sec), ...]
    """
    spans = []
    prev_end = 0.0
    in_header = True
    with open(words_path) as f:
        for line in f:
            line = line.strip()
            if in_header:
                if line == '#':
                    in_header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                end_time = float(parts[0])
            except ValueError:
                continue
            label = parts[2].split(';')[0]   # word field before first ';'
            start_time = prev_end
            prev_end = end_time
            if label.startswith('<') or label.startswith('{'):
                continue   # non-speech
            spans.append((start_time, end_time))

    if not spans:
        return []

    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged if e - s >= min_dur]


def parse_vad_xitsonga(vad_file):
    """ZeroSpeech Xitsonga VAD: space-separated  file_id start_sec end_sec
    Returns {speaker_id: [(file_id, start, end), ...]} grouped by speaker."""
    by_speaker = defaultdict(list)
    with open(vad_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    file_id, start, end = parts[0], float(parts[1]), float(parts[2])
                    # nchlt_tso_001m_0007 → speaker 001
                    spk = file_id.split('_')[2][:3]
                    by_speaker[spk].append((file_id, start, end))
                except (ValueError, IndexError):
                    pass
    return dict(by_speaker)


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

def compute_landmarks(y_seg, dur_cs):
    """
    Compute thetaOscillator syllable boundaries for an audio segment.
    Returns a list of centisecond offsets (relative to segment start),
    always ending with dur_cs.
    """
    from theta_oscillator import get_boundaries
    boundary_times = get_boundaries(y_seg, fs=SR)
    lm = [int(t * 100) for t in boundary_times if 0 < t * 100 < dur_cs]
    lm = sorted(set(lm))
    if not lm or lm[-1] != dur_cs:
        lm.append(dur_cs)
    return lm


def load_landmarks(landmark_dir, recording_id):
    """Load pre-computed thetaOscillator boundary file (one sec per line)."""
    path = Path(landmark_dir) / f"{recording_id}.txt"
    if not path.exists():
        return None
    boundaries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    boundaries.append(float(line))
                except ValueError:
                    pass
    return sorted(boundaries)


def landmarks_from_file(all_boundaries, start_sec, end_sec, dur_cs):
    """Select pre-computed boundaries within [start_sec, end_sec] → centisecond offsets."""
    lm = []
    for t in all_boundaries:
        if start_sec <= t < end_sec:
            cs = int((t - start_sec) * 100)
            if 0 < cs < dur_cs:
                lm.append(cs)
    lm = sorted(set(lm))
    if not lm or lm[-1] != dur_cs:
        lm.append(dur_cs)
    return lm


# ---------------------------------------------------------------------------
# Audio & features
# ---------------------------------------------------------------------------

def find_wav_buckeye(audio_dir, recording_id):
    """Buckeye: <audio_dir>/<spk3>/<recording_id>.wav"""
    for path in [
        Path(audio_dir) / recording_id[:3] / f"{recording_id}.wav",
        Path(audio_dir) / f"{recording_id}.wav",
    ]:
        if path.exists():
            return path
    return None


def find_wav_xitsonga(audio_dir, file_id):
    """NCHLT: <audio_dir>/<spk3>/<file_id>.wav"""
    spk = file_id.split('_')[2][:3]
    path = Path(audio_dir) / spk / f"{file_id}.wav"
    return path if path.exists() else None


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
    return (AutoModel.from_pretrained(hf_id, attn_implementation='eager').eval().to(device),
            AutoFeatureExtractor.from_pretrained(hf_id))


# ---------------------------------------------------------------------------
# Per-segment processing (shared by both languages)
# ---------------------------------------------------------------------------

def process_segment(y_seg, start_sec, end_sec, rec_id,
                    is_neural, model, processor, layer, device,
                    file_boundaries):
    import librosa
    if len(y_seg) < SR * 0.05:
        return None, None

    feats = (extract_mfcc(y_seg) if not is_neural
             else extract_neural(y_seg, model, processor, layer, device))
    if len(feats) < 2:
        return None, None

    start_cs = round(start_sec * 100)
    end_cs = round(end_sec * 100)
    dur_cs = end_cs - start_cs
    if file_boundaries is not None:
        lm = landmarks_from_file(file_boundaries, start_sec, end_sec, dur_cs)
    else:
        lm = compute_landmarks(y_seg, dur_cs)

    utt_id = f"{rec_id}_{start_cs}-{end_cs}"
    return utt_id, {'features': feats, 'landmarks': lm}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio_dir',    required=True,
                        help='Language root dir (ZeroSpeech) or WAV dir (Buckeye/Xitsonga)')
    parser.add_argument('--vad_file',     default=None,
                        help='VAD CSV file (Buckeye/Xitsonga). Not needed for ZeroSpeech subsets.')
    parser.add_argument('--subset',       default=None, choices=['1s', '10s', '120s'],
                        help='ZeroSpeech subset to process (1s/10s/120s). '
                             'Omit for Buckeye/Xitsonga VAD-based mode.')
    parser.add_argument('--landmark_dir', default=None,
                        help='Optional: pre-computed thetaOscillator boundary files')
    parser.add_argument('--recording',    default=None,
                        help='Speaker/recording ID to process. Omit for all.')
    parser.add_argument('--language',     required=True,
                        choices=['buckeye', 'xitsonga',
                                 'english', 'french', 'german', 'mandarin', 'wolof'])
    parser.add_argument('--feature_type', required=True,
                        choices=['mfcc'] + list(NEURAL_MODELS))
    parser.add_argument('--layer',  type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.subset is None and args.vad_file is None and args.language not in ('buckeye',):
        parser.error('Either --subset (ZeroSpeech mode) or --vad_file (VAD mode) is required '
                     '(buckeye can derive VAD from .phones files without --vad_file).')

    is_neural = args.feature_type in NEURAL_MODELS
    model, processor = load_model(args.feature_type, args.device) if is_neural else (None, None)

    # ---- ZeroSpeech 2015 (english / french / german / mandarin / wolof) ----
    if args.subset is not None:
        subset_dir = Path(args.audio_dir) / args.subset
        if not subset_dir.exists():
            print(f"Subset directory not found: {subset_dir}")
            return

        wav_files = sorted(subset_dir.glob('*.wav'),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

        if args.recording is not None:
            wav_files = [f for f in wav_files if f.stem == args.recording]

        print(f"[{args.language}/{args.subset}]  {len(wav_files)} WAV files  "
              f"feature={args.feature_type}")

        data = {}
        for wav_path in wav_files:
            file_id = wav_path.stem
            y = load_audio(str(wav_path))
            dur_sec = len(y) / SR
            dur_cs  = int(dur_sec * 100)

            if len(y) < SR * 0.05:
                continue

            feats = (extract_mfcc(y) if not is_neural
                     else extract_neural(y, model, processor, args.layer, args.device))
            if len(feats) < 2:
                continue

            file_boundaries = (load_landmarks(args.landmark_dir, file_id)
                                if args.landmark_dir else None)
            if file_boundaries is not None:
                lm = landmarks_from_file(file_boundaries, 0.0, dur_sec, dur_cs)
            else:
                lm = compute_landmarks(y, dur_cs)

            utt_id = f"{file_id}_0-{dur_cs}"
            data[utt_id] = {'features': feats, 'landmarks': lm}

        print(f"  {len(data)} utterances extracted")
        out = data_path(args.language, args.subset, args.feature_type, args.layer)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(data, f)
        print(f"  → {out}")

    # ---- Xitsonga ----
    elif args.language == 'xitsonga':
        vad = parse_vad_xitsonga(args.vad_file)
        speakers = ([args.recording] if args.recording else sorted(vad))
        print(f"{len(speakers)} speaker(s)  feature={args.feature_type}")

        for spk in speakers:
            if spk not in vad:
                print(f"[{spk}] not in VAD, skipping")
                continue

            entries = vad[spk]
            print(f"\n[{spk}]  {len(entries)} utterances")
            data = {}

            for file_id, start_sec, end_sec in entries:
                wav = find_wav_xitsonga(args.audio_dir, file_id)
                if wav is None:
                    print(f"  WAV not found: {file_id}, skipping")
                    continue

                y_full = load_audio(str(wav))
                y_seg = y_full[int(start_sec * SR): min(int(end_sec * SR), len(y_full))]

                file_boundaries = (load_landmarks(args.landmark_dir, file_id)
                                   if args.landmark_dir else None)

                utt_id, entry = process_segment(
                    y_seg, start_sec, end_sec, file_id,
                    is_neural, model, processor, args.layer, args.device,
                    file_boundaries)
                if utt_id:
                    data[utt_id] = entry

            print(f"  {len(data)} segments extracted")
            out = data_path(args.language, spk, args.feature_type, args.layer)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'wb') as f:
                pickle.dump(data, f)
            print(f"  → {out}")

    # ---- Buckeye (phones-based VAD) or CSV VAD ----
    else:
        if args.vad_file:
            vad_csv = parse_vad_buckeye(args.vad_file)
            recordings = ([args.recording] if args.recording
                          else sorted(vad_csv))
        else:
            # discover all recordings from WAV files in the audio_dir tree
            vad_csv = None
            wav_paths = sorted(Path(args.audio_dir).rglob('*.wav'))
            recordings = ([args.recording] if args.recording
                          else [p.stem for p in wav_paths])

        print(f"{len(recordings)} recording(s)  feature={args.feature_type}")

        for rec_id in recordings:
            wav = find_wav_buckeye(args.audio_dir, rec_id)
            if wav is None:
                print(f"[{rec_id}] WAV not found, skipping")
                continue

            if vad_csv is not None:
                if rec_id not in vad_csv:
                    print(f"[{rec_id}] not in VAD, skipping")
                    continue
                segments = vad_csv[rec_id]
            else:
                # derive VAD from the .words file next to the WAV (Kamper-style)
                words_path = wav.with_suffix('.words')
                if not words_path.exists():
                    print(f"[{rec_id}] no .words file, skipping")
                    continue
                segments = parse_vad_from_words(words_path)
                if not segments:
                    print(f"[{rec_id}] no speech found in words file, skipping")
                    continue

            file_boundaries = (load_landmarks(args.landmark_dir, rec_id)
                                if args.landmark_dir else None)
            if args.landmark_dir and file_boundaries is None:
                print(f"[{rec_id}] landmark file not found, skipping")
                continue

            print(f"\n[{rec_id}]  {len(segments)} segments")
            y_full = load_audio(str(wav))
            data = {}

            for start_sec, end_sec in segments:
                y_seg = y_full[int(start_sec * SR): min(int(end_sec * SR), len(y_full))]
                utt_id, entry = process_segment(
                    y_seg, start_sec, end_sec, rec_id,
                    is_neural, model, processor, args.layer, args.device,
                    file_boundaries)
                if utt_id:
                    data[utt_id] = entry

            print(f"  {len(data)} utterances")
            out = data_path(args.language, rec_id, args.feature_type, args.layer)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'wb') as f:
                pickle.dump(data, f)
            print(f"  → {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
