#!/usr/bin/env python3

import argparse
import os
import data_utils
from eskmeans_init import initialize_clusters
from pooling import PoolingEngine
import pathlib
from eskmeans import eskmeans
from centroids import Centroids


parser = argparse.ArgumentParser(description='ESKMeans segmentattion')
parser.add_argument('--language', dest='lan', type=str,  choices=["buckeye", "mandarin"], help='datraset to use only available [buckeye, dummy]')
parser.add_argument('--speaker', dest='spk', type=str, help='speaker to train for')
parser.add_argument('--feature_type',
                    choices = ["mfcc", "hubert_base_ls960"],
                    type=str,
                    help='feature type to use (mfcc, hubert_base_ls960)')

parser.add_argument('--pooling_type',
                    choices = ["herman", "average"],
                    type=str,
                    help='feature type to use (mfcc, hubert, hubert cp)')

args = parser.parse_args()

#arguments eskmeans
language = args.lan
speaker = args.spk
feature_type = args.feature_type
pooling_type = args.pooling_type


centroid_init_method = "spread_herman"

#MANUAL SETTING
#pooling_method = "herman"
#pooling_method = "average"

#feature_type = "mfcc"
#feature_type = "hubert_base_ls960"

unit_test_flag = False

min_edges = 0
max_edges = 6
min_duration = 20

nepochs = 7

if(feature_type == "mfcc"):
    feat_dim = 13
else:
    feat_dim = 768

landmarks_dict, feat_npy = data_utils.load_dataset(language, speaker, feature_type, unit_test_flag)

pooling_engine = PoolingEngine(pooling_type, feat_dim, feature_type)

#initialize clustering
num_centroids, den_centroids, initial_segments, centroid_rands = initialize_clusters(landmarks_dict,
                                                                                     feat_npy,
                                                                                     centroid_init_method,
                                                                                     pooling_engine,
                                                                                     "npz",
                                                                                     language,
                                                                                     speaker,
                                                                                     max_edges,
                                                                                     unit_test_flag)

#create centroid object
centroids = Centroids(num_centroids, den_centroids, language, speaker, centroid_rands)

#segment
landmarks, transcriptions = eskmeans(landmarks_dict,
                                     dict(feat_npy),
                                     centroids,
                                     nepochs,
                                     pooling_engine,
                                     initial_segments,
                                     language,
                                     speaker,
                                     min_edges,
                                     max_edges,
                                     min_duration,
                                     unit_test_flag)

#save results
result_folder = os.path.join("results", feature_type+"_"+pooling_type,language)
pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
data_utils.write_ramons(landmarks, transcriptions, speaker, result_folder)
