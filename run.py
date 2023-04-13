#!/usr/bin/env python3

import argparse
import pickle
import data_utils
from eskmeans_init import initialize_clusters
from pooling import PoolingEngine
from eskmeans import eskmeans
import sys
from centroids import Centroids


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
#pooling_method = "average"

centroid_init_method = "spread_herman"

feature_type = "mfcc"
#feature_type = "hubert_base_ls960"

unit_test = True

min_edges = 0
max_edges = 6
min_duration = 20

nepochs = 10

if(feature_type == "mfcc"):
    feat_dim = 13
else:
    feat_dim = 768

landmarks_dict, feat_npy = data_utils.load_dataset(language, speaker, feature_type)

pooling_engine = PoolingEngine(pooling_method, feat_dim, feature_type)

#initialize clustering
num_centroids, den_centroids, initial_segments, centroid_rands = initialize_clusters(landmarks_dict,
                                                                                     feat_npy,
                                                                                     centroid_init_method,
                                                                                     pooling_engine,
                                                                                     "npz",
                                                                                     language,
                                                                                     speaker,
                                                                                     max_edges,
                                                                                     unit_test)

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
                                     min_duration)

#save results
#result_folder = os.path.join("results", feature_type+"_"+pooling_method)
#pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
#data_utils.write_ramons(landmarks, transcriptions, language, speaker, result_folder)



