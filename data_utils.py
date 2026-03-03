import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = "/shared/ramon/experiments_old"


def data_dir(language, speaker):
    base = os.environ.get('ESKMEANS_DATA', DEFAULT_DATA_DIR)
    return Path(base) / language / speaker


def feature_filename(feature_type, feature_layer):
    if feature_type == 'mfcc':
        return 'mfcc.pkl'
    return f'{feature_type}_l{feature_layer}.pkl'


def write_ramons(unsup_landmarks, unsup_transcript, speaker_id, output_folder):
    class_dict = defaultdict(list)
    print("WRITING RESULTS AT: " + str(os.path.join(output_folder, speaker_id + ".tdev")))

    for key in unsup_transcript.keys():
        for idx, class_id in enumerate(unsup_transcript[key]):
            start = float(key.split("_")[-1].split("-")[0]) / 100
            new_key = "_".join(key.split("_")[:-1])
            final_start = start + float(unsup_landmarks[key][idx][0] / 100)
            final_end = start + float(unsup_landmarks[key][idx][1] / 100)
            class_dict[class_id].append((final_start, final_end, new_key))

    with open(os.path.join(output_folder, speaker_id + ".tdev"), "w") as result_file:
        for class_id in class_dict.keys():
            result_file.write("Class " + str(class_id) + "\n")
            for segment in class_dict[class_id]:
                result_file.write(str(segment[2]) + " " + str(segment[0]) + " " + str(segment[1]) + "\n")
            result_file.write("\n")


def filter_short_segments(landmarks_dict, feat_dict, minimum_duration):
    deleted = 0
    new_feats = {}
    new_landmarks = {}
    for key in list(landmarks_dict.keys()):
        start, end = key.split("_")[-1].split("-")
        if (int(end) - int(start)) < minimum_duration:
            deleted += 1
        else:
            new_feats[key] = feat_dict[key]
            new_landmarks[key] = landmarks_dict[key]
    print(f"{deleted} segments removed (duration < {minimum_duration} cs)")
    return new_landmarks, new_feats


def load_dataset(language, speaker, feature_type, minimum_duration,
                 unit_test_flag, feature_layer="10"):
    import unit_test

    pkl_path = data_dir(language, speaker) / feature_filename(feature_type, feature_layer)

    if not pkl_path.exists():
        print(f"Data file not found: {pkl_path}")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    # raw is a dict: utt_id → {'features': ndarray, 'landmarks': [int, ...]}
    feat_dict = {k: v['features'] for k, v in sorted(raw.items())}
    landmarks_dict = {k: v['landmarks'] for k, v in sorted(raw.items())}

    if unit_test_flag:
        unit_test.utterance_ids(feat_dict, language, speaker)

    return filter_short_segments(landmarks_dict, feat_dict, minimum_duration)


def load_all_speakers(language, feature_type, minimum_duration,
                      unit_test_flag, feature_layer="10"):
    """Load and merge data from all speaker directories for a language."""
    lang_dir = Path(os.environ.get('ESKMEANS_DATA', DEFAULT_DATA_DIR)) / language
    feat_dict = {}
    landmarks_dict = {}

    speakers = sorted(d.name for d in lang_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('nchlt'))
    print(f"Loading {len(speakers)} speakers: {speakers}")

    for spk in speakers:
        pkl_path = lang_dir / spk / feature_filename(feature_type, feature_layer)
        if not pkl_path.exists():
            print(f"  [{spk}] missing {pkl_path.name}, skipping")
            continue
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)
        for k, v in raw.items():
            feat_dict[k] = v['features']
            landmarks_dict[k] = v['landmarks']
        print(f"  [{spk}] {len(raw)} utterances")

    print(f"Total: {len(feat_dict)} utterances")
    return filter_short_segments(landmarks_dict, feat_dict, minimum_duration)
