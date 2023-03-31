#!/usr/bin/env python3

import argparse
import data_utils
from eskmeans_init import initialize_clusters
from pooling import PoolingEngine
from eskmeans import eskmeans
import os

parser = argparse.ArgumentParser(description='ESKMeans segmentattion')
parser.add_argument('--language', dest='lan', type=str,  choices=["buckeye", "mandarin"], help='datraset to use only available [buckeye, dummy]')
parser.add_argument('--speaker', dest='spk', type=str, help='speaker to train for')
parser.add_argument('--centroids', dest='n_c', type=int, help='number of centroids to use')
parser.add_argument('--min_duration', dest='m_d', type=int, help='min duration of segments')
parser.add_argument('--feature_type',
                    choices = ["red", "green", "blue"],
                    dest='f_t', type=str,
                    help='feature type to use (mfcc, hubert, hubert_cp, hubert_lp)')

parser.add_argument('--pooling_type', dest='p_t',
                    choices = ["herman", "subsample", "average"],
                    type=str,
                    help='feature type to use (mfcc, hubert, hubert cp)')

args = parser.parse_args()

language = args.lan
speaker = args.spk

#arguments eskmeans
max_number_centroids = args.n_c
min_duration = args.m_d
feature_type = args.f_t

pooling_method = "herman"
centroid_init_method = "spread_herman"
min_segments = 0
max_segments = 6
nepochs = 5
feat_dim = 13

landmarks_dict, feat_npy = data_utils.load_dataset(language, speaker, "npz")

pooling_engine = PoolingEngine(pooling_method, feature_dim=feat_dim)

#initialize clustering
num_centroids, den_centroids, initial_segments = initialize_clusters(landmarks_dict,
                                                  feat_npy,
                                                  max_number_centroids,
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

#save results
result_folder = os.path.join("results", feature_type+"_"+pooling_method)
data_utils.write_ramons(landmarks, transcriptions, language, speaker, result_folder)



