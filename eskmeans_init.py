import numpy as np
import kaldi_io

import random
from tqdm import tqdm
import sys


def init_random_hao(landmark_sets, feat_scp, ncentroid):
    f = open(feat_scp[0][1], 'rb')
    f.seek(feat_scp[0][2])
    feats = kaldiark.parse_feat_matrix(f)
    feat_dim = feats.shape[1]
    f.close()

    centroids = numpy.zeros((ncentroid, feat_dim * 10))
    ncentroid_per_scp = int(ncentroid / len(feat_scp)) + 1

    k = 0

    for i, scp in enumerate(feat_scp):
        landmarks = landmark_sets[i]

        f = open(scp[1], 'rb')
        f.seek(scp[2])
        feats = kaldiark.parse_feat_matrix(f)
        f.close()

        g = build_graph(landmarks)
        g.feats = feats

        edges = list(range(g.edges))
        random.shuffle(edges)


        for e in edges[:ncentroid_per_scp]:
            if k == ncentroid:
                break

            centroids[k] = g.feat(e)
            k += 1

        if k == ncentroid:
            break
    return centroids

def get_durations(landmarks_aux):
    """
    Return a list of tuple, where every tuple is the duration and the
    :param landmarks_aux: list of landmarks (without the 0 landmarks)
    :return: list of tuple, where every tuple is the duration and the (start, end) landmarks
    """
    
    landmarks = [0,] + landmarks_aux

    N = len(landmarks)
    #durations = -1*np.ones(int(((N - 1)**2 + (N - 1))/2), dtype=int)
    durations = []
    j = 0
    for t in range(1, N):
        for i in range(t):
            if t - i > N - 1:
                continue
            durations.append((landmarks[t] - landmarks[i], (landmarks[t], landmarks[i])))
    return durations

def spread_herman(landmarks, feats, n_landmarks_max, pooling_function, ncentroids, feats_format):

    centroids = np.zeros((pooling_function.get_out_feat_dim(), ncentroids))

    #we truncate the left over in the end due the mod of the division
    n_initial_segments = sum([len(value) for value in landmarks.values()])
    assignments = (list(range(ncentroids))*int(np.ceil(float(n_initial_segments)/ncentroids)))[:n_initial_segments]
    num_initial_segments_k = np.zeros(ncentroids, dtype=int)

    for assignment in assignments:
        num_initial_segments_k[assignment] += 1

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

    for utt_id in sorted(feats_dict.keys()):


        #landmark_utt = get_durations_and_plandmarks(landmarks[utt_id], n_landmarks_max)
        #note that we are not considering 0 as a landmark. num_segments = num_landmarks - 1
        num_segments = len(landmarks[utt_id])
        durations = get_durations(landmarks[utt_id])
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

            embedding = pooling_function.subsample(mat[start_frame:end_frame+1,:])
            k = assignments[initial_segments_count]
            centroids[:,k] += embedding/num_initial_segments_k[k]

            initial_segments_count += 1
            j_prev = j + 1
    print(centroids.transpose())
    print(centroids.shape)
    return centroids

def random_herman(landmarks, feats_scps, max_segments, pooling_function):
     print("here we will use initial_segments vecotr as above and assign a cluster ID to each one randomly")


def initialize_clusters(landmarks, max_segments, feats_scps, ncentroids, init_technique, pooling_function, format):
    
    if(init_technique == "init_hao"):
        return init_random_hao(landmarks, feats_scps, ncentroids)

    elif(init_technique == "spread_herman"):
        return spread_herman(landmarks, feats_scps, max_segments, pooling_function, ncentroids, format)

    elif(init_technique == "random_herman"):
        return random_herman(landmarks, feats_scps, max_segments, pooling_function)

