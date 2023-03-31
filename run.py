#!/usr/bin/env python3

import argparse
import data_utils
from eskmeans_init import initialize_clusters
from pooling import PoolingEngine
from eskmeans import eskmeans

parser = argparse.ArgumentParser(description='ESKMeans segmentattion')
parser.add_argument('--language', dest='lan', type=str,  choices=["buckeye", "mandarin"], help='datraset to use only available [buckeye, dummy]')
parser.add_argument('--speaker', dest='spk', type=str, help='speaker to train for')
parser.add_argument('--centroids', dest='n_c', type=int, help='number of centroids to use')
parser.add_argument('--min_duration', dest='m_d', type=int, help='min duration of segments')

args = parser.parse_args()

language=args.lan
speaker=args.spk
ncentroids=args.n_c
min_duration=args.m_d

#arguments eskmeans
pooling_method = "herman"
centroid_init_method = "spread_herman"
min_segments = 0
max_segments = 6
nepochs = 5
max_number_centroids = 895


#landmarks_dict, feat_scp_path = data_utils.load_dataset(language, speaker, "npy")
landmarks_dict, feat_npy = data_utils.load_dataset(language, speaker, "npz")

feat_dim = 13

pooling_engine = PoolingEngine(pooling_method, feature_dim=feat_dim)

num_centroids, den_centroids, initial_segments = initialize_clusters(landmarks_dict,
                                                  feat_npy,
                                                  ncentroids,
                                                  centroid_init_method,
                                                  pooling_engine,
                                                  "npz",
                                                  language,
                                                  speaker)

#segment
landmarks, transcriptions = eskmeans(landmarks_dict,
                                     dict(feat_npy),
                                     num_centroids,
                                     den_centroids,
                                     max_number_centroids,
                                     nepochs,
                                     pooling_engine,
                                     initial_segments)


