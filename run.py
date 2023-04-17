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
                    choices = ["mfcc", "hubert_base_ls960"],
                    type=str,
                    help='feature type to use (mfcc, hubert_base_ls960)')

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

#MANUAL SETTING
#pooling_method = "herman"
#pooling_method = "average"

#feature_type = "mfcc"
#feature_type = "hubert_base_ls960"

unit_test_flag = False

min_edges = 0
max_edges = 6
min_duration = 20

nepochs = 10

if("hubert" in feature_type):
    feat_dim = 768
    landmarks_dict, feat_npy = data_utils.load_dataset(language,
                                                       speaker,
                                                       feature_type,
                                                       unit_test_flag,
                                                       feature_layer,
                                                       vad_position)
else:
    feat_dim = 13
    landmarks_dict, feat_npy = data_utils.load_dataset(language,
                                                       speaker,
                                                       feature_type,
                                                       unit_test_flag)

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
    current_hostname = socket.gethostname()

    if ("banff.inf.ed.ac.uk" == current_hostname):

        centroids_gt = np.load("/disk/scratch_fast/ramons/data/hubert_data/word_centroids/zsc/hubert_base_ls960/10"
                               "/prevad"/norm/"+language+"/"+speaker+".npy").transpose()
    else:
        centroids_gt = np.load("/disk/scratch1/ramons/data/hubert_data/word_centroids/zsc/hubert_base_ls960/10/prevad"
                               "/norm/"+language+"/"+speaker+".npy").transpose()

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

if("hubert" in feature_type):
    #save results
    result_folder = os.path.join("results", feature_type+"_"+pooling_type+"_l"+feature_layer+"_"+vad_position,
                                 language)
else:
    result_folder = os.path.join("results", feature_type+"_"+pooling_type,language)

pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
data_utils.write_ramons(landmarks, transcriptions, speaker, result_folder)
