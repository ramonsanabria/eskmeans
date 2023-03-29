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
min_segments = 0
max_segments = 6
pooling_methos = "herman"
centroid_init = "spread_herman"
nepochs = 5


#landmarks_dict, feat_scp_path = data_utils.load_dataset(language, speaker, "npy")
landmarks_dict, feat_npy = data_utils.load_dataset(language, speaker, "npz")

feat_dim = 13

pooling_engine = PoolingEngine(pooling_methos, feature_dim=feat_dim)

centroids, initial_segments = initialize_clusters(landmarks_dict,
                                                  max_segments,
                                                  feat_npy,
                                                  ncentroids,
                                                  centroid_init,
                                                  pooling_engine,
                                                  "npz",
                                                  language,
                                                  speaker)

landmarks, transcriptions = eskmeans(landmarks_dict,
                                     dict(feat_npy),
                                     centroids,
                                     nepochs,
                                     min_duration,
                                     pooling_engine,
                                     initial_segments)


