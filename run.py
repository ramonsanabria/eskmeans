#!/usr/bin/env python3

import collections
import argparse
import sys
import pickle
import kaldiark
import data_utils
import random
import numpy
import numpy.linalg
from tqdm import tqdm
from eskmeans_init import initialize_clusters
from pooling import factory_function


random.seed(0)

def up_centroids_and_comp_weights(prev_segments, path, g, sums, counts, scp, k_herman):

    path_weight=0

    if(scp in prev_segments):
        for e in prev_segments[scp]:
            v = g.feat(e)
            arg, m = assign_cluster(v, centroids)

            sums[arg] -= v
            counts[arg] -= 1


    for e in path:
        v = g.feat(e)
        d = g.duration(e)

        arg, m = assign_cluster(v, centroids)

        if arg > k_herman:
            arg = k_herman
        if arg == k_herman:
            k_herman += 1


        sums[arg] += v
        counts[arg] += 1

        path_weight += m * d

    
    return sums, counts, path_weight, k_herman




def test_model(g, feat_scp, gt_phone, gt_words):
    
    segments = []
    for i, scp in tqdm(enumerate(feat_scp), total=len(feat_scp)):
        path = shortest_path(g)
        path_weight=0
        segment_i = []
        for e in path:
            path_weight += m * d
            segment_i.append(e[0])
    eval_utils.get_word_token_scores(gt_phone, gt_words, segments, 0.2)

    return None

#TODO matrice this function
def assign_cluster(v, centroids):
    m = float('-inf')
    arg = -1
    for i, u in enumerate(centroids):
        cand = numpy.linalg.norm(u - v)
        cand = -cand * cand 
            
        if cand > m:
            m = cand
            arg = i

    return arg, m


class Graph:
    def __init__(self):
        self.vertices = 0
        self.edges = 0
        self.min_duration = 20
        self.tail = {}
        self.head = {}
        self.time = {}
        self.in_edges = collections.defaultdict(list)

        self.feats = None
        self.centroids = None

    def add_vertex(self):
        v = self.vertices
        self.vertices += 1
        return v

    def add_edge(self, u, v):
        e = self.edges
        self.edges += 1
        self.tail[e] = u
        self.head[e] = v
        self.in_edges[v].append(e)
        return e

    def feat(self, e):
        s = self.time[self.tail[e]]
        t = self.time[self.head[e]]
        v = subsample_herman(self.feats[s:t+1], 10)
        return v

    def weight(self, e):
        v = self.feat(e)
        d = self.duration(e)
        if(d < self.min_duration):
            d = float('inf')
        _, m = assign_cluster(v, centroids)

        return m * d

    def duration(self, e):

        s = self.time[self.tail[e]]
        t = self.time[self.head[e]]

        return t-s

def build_graph(landmarks):
    g = Graph()
    r = g.add_vertex()
    g.time[r] = 0

    for e in landmarks:
        v = g.add_vertex()
        g.time[v] = e

    for u in range(g.vertices):
        for v in range(u + 1, g.vertices):
            g.add_edge(u, v)

    return g


def shortest_path(g):
    d = {}
    back = {}

    d[0] = 0
    back[0] = -1

    #
    # Assuming that vertices are topologically sorted
    #
    for v in range(1, g.vertices):

        arg = -1
        m = float('-inf')
        for e in g.in_edges[v]:
            #print(g.weight(e))
            cand = d[g.tail[e]] + g.weight(e)
            if cand > m:
                m = cand
                arg = e

        d[v] = m
        back[v] = arg

    #
    # Assuming that the last vertex is the final vertex
    # and that 0 is the initial vertex
    #
    path = []
    v = g.vertices - 1
    while v != 0:
        e = back[v]
        path.append(e)
        v = g.tail[e]

    path.reverse()
    return path


def eskmeans(landmarks, feat_scps, phn_gt, wrd_gt, centroids, nepoch, min_duration):

    
    
    feat_scp_dev = feat_scps["dev"]
    feat_scp_test = feat_scps["test"]

    landmark_dev = landmarks["dev"]
    landmark_test = landmarks["test"]

    prev_paths = {}

    k_herman = 0

    for epoch in range(nepoch):
        sums =  numpy.zeros(centroids.shape)
        counts = numpy.zeros(centroids.shape[0])
        #print(centroids[:,:5])

        for i, scp in tqdm(enumerate(feat_scp_dev), total=len(feat_scp_dev)):
            landmarks = landmark_dev[i]

            f = open(scp[1], 'rb')
            f.seek(scp[2])
            feats = kaldiark.parse_feat_matrix(f)
            f.close()

            g = build_graph(landmarks)
            g.feats = feats
            g.centroids = centroids

            path = shortest_path(g)

            sums, counts, path_weight, k_herman  = up_centroids_and_comp_weights(prev_paths, path, g, sums, counts, scp, k_herman)
            

            for idx in range(centroids.shape[0]):
                if counts[idx] > 0:
                    centroids[idx,:] = sums[idx,:]/counts[idx]

            prev_paths[scp[0]] = path

            print('epoch: {}'.format(epoch))
            print('sample: {}'.format(scp[0]))
            print('path: {}'.format([(g.time[g.tail[e]], g.time[g.head[e]]) for e in path]))
            print('path weight: {}'.format(path_weight))
            print('')


    return centroids



parser = argparse.ArgumentParser(description='ESKMeans segmentattion')
parser.add_argument('--language', dest='lan', type=str,  choices=["buckeye", "mandarin"], help='datraset to use only available [buckeye, dummy]')
parser.add_argument('--speaker', dest='spk', type=str, help='speaker to train for')
parser.add_argument('--centroids', dest='n_c', type=int, help='number of centroids to use')
parser.add_argument('--min_duration', dest='m_d', type=int, help='min duration of segments')

args = parser.parse_args()

language=args.lan
speaker=args.spk
n_centroids=args.n_c
min_duration=args.m_d

#arguments eskmeans
ncentroids = 815
min_segments = 0
max_segments = 6
pooling_methos = "herman"
centroid_init = "spread_herman"

pooling_function = factory_function(pooling_methos)
landmarks, feats_scps  = data_utils.load_dataset(language, speaker)

clusters = initialize_clusters(landmarks, max_segments, feats_scps, ncentroids, centroid_init, pooling_function) 
#numpy.save("/disk/scratch1/ramons/segmentation/code/eskmeans/centroids_100.npy",centroids)
sys.exit()


landmarks, transcriptions = eskmeans(landmarks, feats_scps, phn_gt, wrd_gt, centroids, 5, min_duration)

