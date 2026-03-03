#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np
import data_utils
from eskmeans_init import initialize_clusters
from pooling import PoolingEngine
import pathlib
from eskmeans import eskmeans, eskmeans_em
from centroids import Centroids, CentroidsEm

FEAT_DIMS = {
    "mfcc":               13,
    "hubert_base_ls960":  768,
    "mhubert":            768,
    "wavlm_large":        1024,
}

# Per-language hyperparameters tuned to landmark spacing and word duration distributions.
# min_duration: centiseconds. max_edges: landmark intervals.
# xitsonga: median landmark spacing ~23cs, word p10=15cs, p99=133cs (→ ~5.8 edges)
# english:  faster speech, shorter landmark spacing — tighter defaults kept
LANG_PARAMS = {
    #              min_dur  max_edges
    "xitsonga": (   10,       8  ),
    "buckeye":  (   10,       8  ),
    "english":  (   20,       6  ),
    "french":   (   20,       6  ),
    "german":   (   20,       6  ),
    "mandarin": (   20,       6  ),
    "wolof":    (   20,       6  ),
}

parser = argparse.ArgumentParser(description='ESKMeans segmentation')
parser.add_argument('--language',     dest='lan', type=str,
                    choices=["buckeye", "xitsonga",
                             "english", "french", "german", "mandarin", "wolof"])
parser.add_argument('--speaker',      dest='spk', type=str, default=None,
                    help='Speaker ID. Omit to run across all speakers (speaker-independent).')
parser.add_argument('--feature_type', type=str, choices=list(FEAT_DIMS))
parser.add_argument('--pooling_type', type=str, choices=["herman", "average"])
parser.add_argument('--kmeans_type',  type=str, choices=["herman", "em"])
parser.add_argument('--layer',        type=str, default="10",
                    help='Feature layer (for neural models, default: 10)')
parser.add_argument('--seed',         type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

language         = args.lan
speaker          = args.spk
feature_type     = args.feature_type
feature_layer    = args.layer
pooling_type     = args.pooling_type
kmeans_type      = args.kmeans_type
centroid_init_type = "herman"
unit_test_flag   = False

min_edges    = 0
min_duration, max_edges = LANG_PARAMS[language]
nepochs      = 10
print(f"[{language}] min_duration={min_duration}cs  max_edges={max_edges}")

feat_dim = FEAT_DIMS[feature_type]
if speaker is None:
    landmarks_dict, feat_npy = data_utils.load_all_speakers(
        language, feature_type, min_duration, unit_test_flag, feature_layer)
    speaker = "all"
else:
    landmarks_dict, feat_npy = data_utils.load_dataset(
        language, speaker, feature_type, min_duration, unit_test_flag, feature_layer)

pooling_engine = PoolingEngine(pooling_type, feat_dim, feature_type)

num_centroids, den_centroids, initial_segments, centroid_rands = initialize_clusters(
    landmarks_dict, feat_npy, centroid_init_type, pooling_engine,
    language, speaker, max_edges, unit_test_flag)

if kmeans_type == "herman":
    centroids = Centroids(num_centroids, den_centroids, centroid_rands)
    landmarks, transcriptions = eskmeans(
        landmarks_dict, dict(feat_npy), centroids, nepochs, pooling_engine,
        initial_segments, language, speaker,
        min_edges, max_edges, min_duration, unit_test_flag)

elif kmeans_type == "em":
    data_base = os.environ.get('ESKMEANS_DATA', '/shared/ramon/experiments_old')
    centroids_path = os.path.join(data_base, language, speaker,
                                  f"centroids_l{feature_layer}.npy")
    centroids_gt = np.load(centroids_path).transpose()
    centroids = CentroidsEm(centroids_gt)
    landmarks, transcriptions = eskmeans_em(
        landmarks_dict, dict(feat_npy), centroids, nepochs, pooling_engine,
        min_edges, max_edges, min_duration)

result_folder = os.path.join(
    "results",
    f"{feature_type}_{pooling_type}_l{feature_layer}" if feature_type != "mfcc"
    else f"{feature_type}_{pooling_type}",
    language)

pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
data_utils.write_ramons(landmarks, transcriptions, speaker, result_folder)
