import numpy as np
import kaldiark
import random


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


#TODO
def get_durations_and_plandmarks(landmarks_aux, n_landmarks_max):
    
    landmarks = [0,] + landmarks_aux

    N = len(landmarks)
    durations = -1*np.ones(int(((N - 1)**2 + (N - 1))/2), dtype=int)
    j = 0
    for t in range(1, N):
        for i in range(t):
            if t - i > N - 1:
                j += 1
                continue
            durations[j] = landmarks[t] - landmarks[i]
            j += 1
    return durations

#def initialize_segmentation(landmarks_aux, n_landmarks_max):

def get_segmented_landmark_indices(self, i):
    """
    Return a list of tuple, where every tuple is the start (inclusive) and
    end (exclusive) landmark index for the segmented embeddings.
    """
    indices = []
    j_prev = 0
    print(self.boundaries)
    for j in np.where(self.boundaries[i][:self.lengths[i]])[0]:
        indices.append((j_prev, j + 1))
        j_prev = j + 1
    return indices

def spread_herman(landmarks, feats_scps, max_segments, pooling_function, ncentroids):

    feats_scps

    centroids = np.zeors((130, ncentroids))

    #done for Herman Recovery
    p_boundary_init = 0.1
    idx_counter = 0

    np.random.seed(5) 

    landmarks_aux = {}

    initial_segments=[]

    for idx, landmark in enumerate(landmarks):
        #print(landmark[-1])

        durations = get_durations_and_plandmarks(landmark[0], max_segments)
        N = len(landmark[0])

        boundaries = (np.random.rand(N) < p_boundary_init)

        boundaries[N-1] = True
        idx_boundaries = list(range(N+1))

        j_prev = 0
        for j in np.where(idx_boundaries)[0]:
            initial_segments.append(((j_prev, j + 1), feats_scps[idx]))
            j_prev = j + 1
        
    n_initial_segments = len(initial_segments)
    
    #each utterance has different clusters assigned. 
    #we truncate the left over in the end due the mod of the division
    random.seed(2) 
    assignment_list = (list(range(ncentroids))*int(np.ceil(float(n_initial_segments)/ncentroids)))[:n_initial_segments]
    random.shuffle(assignment_list)

    # compute centroids
    for k in range(ncentroids + 1):
        seg_idx_to_k = np.where(assignments == k)[0]
        for i in seg_idx_to_k:
            centroids[:,k] += feature_vector/len(seg_idx_to_k)

    return centroids

def random_herman(landmarks, feats_scps, max_segments, pooling_function):
     print("here we will use initial_segments vecotr as above and assign a cluster ID to each one randomly")


def initialize_clusters(landmarks, max_segments, feats_scps, ncentroids, init_technique, pooling_function):
    
    if(init_technique == "init_hao"):
        return init_random_hao(landmarks, feats_scps, ncentroids)

    elif(init_technique == "spread_herman"):
        return spread_herman(landmarks, feats_scps, max_segments, pooling_function, ncentroids)

    elif(init_technique == "random_herman"):
        return random_herman(landmarks, feats_scps, max_segments, pooling_function)

