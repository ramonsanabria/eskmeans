#!/usr/bin/env python3

import socket
import argparse
import os
import data_utils
from eskmeans_init import initialize_clusters
from pooling import PoolingEngine
import pathlib
import numpy as np
from eskmeans import eskmeans, eskmeans_em
from centroids import Centroids, CentroidsEm


parser = argparse.ArgumentParser(description='ESKMeans segmentattion')
parser.add_argument('--language', dest='lan', type=str,  choices=["buckeye", "mandarin"], help='datraset to use only available [buckeye, dummy]')
parser.add_argument('--speaker', dest='spk', type=str, help='speaker to train for')
parser.add_argument('--feature_type',
                    choices = ["mfcc", "hubert_base_ls960", "mhubert", "wavlm_large"],
                    type=str,
                    help='feature type to use')

parser.add_argument('--pooling_type',
                    choices = ["herman", "average"],
                    type=str,
                    help='feature type to use (mfcc, hubert, hubert cp)')

parser.add_argument('--kmeans_type',
                    choices = ["herman", "em"],
                    type=str,
                    help='how type of kmeans to train the model with (herman, em)')

args = parser.parse_args()

#arguments eskmeans
language = args.lan
speaker = args.spk

#feature definition
feature_type = args.feature_type
feature_layer = "10"
vad_position = "prevad"

pooling_type = args.pooling_type
kmeans_type = args.kmeans_type
centroid_init_type = "herman"

unit_test_flag = False

min_edges = 0
max_edges = 6
min_duration = 20

nepochs = 10

FEAT_DIMS = {
    "mfcc": 13,
    "hubert_base_ls960": 768,
    "mhubert": 768,
    "wavlm_large": 1024,
}

feat_dim = FEAT_DIMS[feature_type]

if feature_type == "mfcc":
    landmarks_dict, feat_npy = data_utils.load_dataset(language,
                                                       speaker,
                                                       feature_type,
                                                       min_duration,
                                                       unit_test_flag)
else:
    landmarks_dict, feat_npy = data_utils.load_dataset(language,
                                                       speaker,
                                                       feature_type,
                                                       min_duration,
                                                       unit_test_flag,
                                                       feature_layer,
                                                       vad_position)

pooling_engine = PoolingEngine(pooling_type, feat_dim, feature_type)

#initialize clustering
num_centroids, den_centroids, initial_segments, centroid_rands = initialize_clusters(landmarks_dict,
                                                                                     feat_npy,
                                                                                     centroid_init_type,
                                                                                     pooling_engine,
                                                                                     "npz",
                                                                                     language,
                                                                                     speaker,
                                                                                     max_edges,
                                                                                     unit_test_flag)


if(kmeans_type == "herman"):

    #create centroid object
    centroids = Centroids(num_centroids, den_centroids, centroid_rands)

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
elif(kmeans_type == "em"):

    #create centroid object
    if socket.gethostname() == "banff.inf.ed.ac.uk":
        default_data_base = "/disk/scratch_fast/ramons/data"
    else:
        default_data_base = "/disk/scratch1/ramons/data"
    data_base = os.environ.get('ESKMEANS_DATA', default_data_base)

    centroids_gt = np.load(os.path.join(data_base,
                                        "hubert_data/word_centroids/zsc/hubert_base_ls960",
                                        feature_layer, "prevad/norm",
                                        language, speaker + ".npy")).transpose()

    centroids = CentroidsEm(centroids_gt)

    #segment
    landmarks, transcriptions = eskmeans_em(landmarks_dict,
                                            dict(feat_npy),
                                            centroids,
                                            nepochs,
                                            pooling_engine,
                                            min_edges,
                                            max_edges,
                                            min_duration)

if feature_type == "mfcc":
    result_folder = os.path.join("results", feature_type + "_" + pooling_type, language)
else:
    result_folder = os.path.join("results", feature_type + "_" + pooling_type + "_l" + feature_layer + "_" + vad_position,
                                 language)

pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
data_utils.write_ramons(landmarks, transcriptions, speaker, result_folder)
