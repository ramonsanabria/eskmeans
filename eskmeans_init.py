import numpy as np

import kaldi_io

from collections import defaultdict

import unit_test

import random
import sys


def spread_herman(landmarks, feats, pooling_function, feats_format, language, speaker_id, unit_test_flag):

    #number of centroids is the 20% of the number of landmarks
    n_ladmarks = sum([len(value) for value in landmarks.values()])
    ncentroids = int(0.2*n_ladmarks)
    centroids = np.zeros((pooling_function.get_out_feat_dim(), ncentroids))
    num_centroids = np.zeros((pooling_function.get_out_feat_dim(), ncentroids))

    #we truncate the left over in the end due the mod of the division
    n_initial_segments = sum([len(value) for value in landmarks.values()])
    assignments = (list(range(ncentroids))*int(np.ceil(float(n_initial_segments)/ncentroids)))[:n_initial_segments]

    den_centroids = np.zeros(ncentroids, dtype=int)

    for assignment in assignments:
        den_centroids[assignment] += 1

    random.seed(2)
    random.shuffle(assignments)


    if(feats_format == 'npz'):
        feats_dict = feats
    elif(feats_format == 'scp'):
        feats_dict = kaldi_io.read_mat_scp(feats)
    else:
        print("feats_format not supported: "+str(feats_format))
        sys.exit()

    #maximum amount of landmarks in the dataset
    lengths = [ len(landmarks[key]) for key in landmarks.keys() ]
    max_lengths = max(lengths)
    initial_segments_count = 0
    initial_segments = defaultdict(list)

    for utt_id in sorted(feats_dict.keys()):


        #note that we are not considering 0 as a landmark. num_segments = num_landmarks - 1
        num_segments = len(landmarks[utt_id])
        p_boundary_init = 1.0

        #replicate boundaries format from herman
        boundaries = np.zeros((max_lengths), dtype=bool)
        boundaries[0:num_segments] = (np.random.rand(num_segments) < p_boundary_init)
        boundaries[num_segments - 1] = True

        mat = feats_dict[utt_id]

        #needed for inside utterance loop
        j_prev = 0
        landmarks_aux = [0,] + landmarks[utt_id]

        for j in np.where(boundaries)[0]:
            start_idx, end_idx = j_prev, j + 1

            start_frame = landmarks_aux[start_idx]
            end_frame =  landmarks_aux[end_idx]


            embedding = pooling_function.pool(mat, start_frame, end_frame)
            k = assignments[initial_segments_count]

            centroids[:,k] += embedding/den_centroids[k]
            num_centroids[:,k] += embedding

            initial_segments[utt_id].append((k, (start_frame, end_frame)))

            initial_segments_count += 1
            j_prev = j + 1

    centroids = centroids.transpose()

    if(unit_test_flag):
        unit_test.initial_centroids_and_segments(centroids, initial_segments, language, speaker_id)

    return num_centroids, den_centroids, initial_segments

def create_centroid_rands(num_centroids,
                          feats,
                          landmarks,
                          pooling,
                          max_edges,
                          unit_test_flag):


    feats = dict(sorted(feats.items()))
    centroids_rand_init = np.zeros((pooling.get_out_feat_dim(), num_centroids))
    list_land_feats = []

    for utt_id in feats.keys():
        prev_landmark = 0
        for i in range(len(landmarks[utt_id])):
            for j in landmarks[utt_id][i:i + max_edges]:
                list_land_feats.append((prev_landmark, j, utt_id))
            prev_landmark = landmarks[utt_id][i]

    #if(unit_test_flag):
    np.random.seed(2)
    idx_segment = np.random.choice(len(list_land_feats), num_centroids, replace=True).tolist()

    for idx_centroids, idx_segment in enumerate(idx_segment):
        i, j, utt_id = list_land_feats[idx_segment]
        feats_utt = feats[utt_id]

        centroids_rand_init[:, idx_centroids] = pooling.pool(feats_utt, i, j)

    return centroids_rand_init

def initialize_clusters(landmarks,
                        feats,
                        cluster_init_type,
                        pooling_function,
                        format,
                        language,
                        speaker_id,
                        max_edges,
                        unit_test_flag):

    if(cluster_init_type == "herman"):
        num_centroids, den_centroids, initial_segments = spread_herman(landmarks,
                                                                       feats,
                                                                       pooling_function,
                                                                       format,
                                                                       language,
                                                                       speaker_id,
                                                                       unit_test_flag)


        ncentroids = num_centroids.shape[1]
        centroid_rand_init = create_centroid_rands( ncentroids,
                                                feats,
                                                landmarks,
                                                pooling_function,
                                                max_edges,
                                                unit_test_flag)

        if(unit_test_flag):
            unit_test.centroid_rands(centroid_rand_init, language, speaker_id)

    else:
        print("init_technique (cluster initialization) not supported: " + str(cluster_init_type))
        sys.exit()

    return num_centroids, den_centroids, initial_segments, centroid_rand_init