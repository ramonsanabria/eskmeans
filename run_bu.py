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
import scipy.signal as signal
from tqdm import tqdm
import eval_utils


random.seed(0)

def update_centroids(prev_segments, path, g, centroids, counts):
    for e in path:
        v = g.feat(e)
        arg, m = assign_cluster(v, centroids)

        sums[arg] = sums[arg] * (counts[arg] / (counts[arg] + 1)) + v / (counts[arg] + 1)
        counts[arg] += 1
        path_weight += m * d






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



def subsample(feats, n):
    k = len(feats) / n

    result = []
    for i in range(n):
        result.extend(feats[int(k * i)])

    return numpy.array(result)

def subsample_herman(feats, n):
    feats_t = feats.T

    y_new = signal.resample(feats_t, n, axis=1).flatten("C")
    #print(y_new)
    #k = len(feats) / n

    #result = []
    #for i in range(n):
    #    result.extend(feats[int(k * i)])

    return numpy.array(y_new)

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


def eskmeans_init(landmark_sets, feat_scp, ncentroid):
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
    centroids = numpy.load("./centroids.npy")
    return centroids


def eskmeans(landmarks, feat_scps, phn_gt, wrd_gt, centroids, nepoch, min_duration):

    
    feat_scp_dev = feat_scps["dev"]
    feat_scp_test = feat_scps["test"]

    landmark_dev = landmarks["dev"]
    landmark_test = landmarks["test"]

    prev_path = {}

    for epoch in range(nepoch):
        sums = numpy.zeros(centroids.shape)
        counts = numpy.zeros(centroids.shape[0])

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



            path_weight = 0

            for e in path:
                v = g.feat(e)
                d = g.duration(e)

                arg, m = assign_cluster(v, centroids)

                sums[arg] = sums[arg] * (counts[arg] / (counts[arg] + 1)) + v / (counts[arg] + 1)
                counts[arg] += 1
                path_weight += m * d

            
            print('epoch: {}'.format(epoch))
            print('epoch: {}'.format(scp[0]))
            print('path: {}'.format([(g.time[g.tail[e]], g.time[g.head[e]]) for e in path]))
            print('path weight: {}'.format(path_weight))
            print('')

            prev_path[scp[0]] = path

            #update centroids
            centroids, counts = update_centroids(prev_segments, path, g, centroids, counts)
            

        #test_model(g, feat_scp_dev)
        #test_model(g, feat_scp_test)
            


        #centroids = sums

    return centroids



parser = argparse.ArgumentParser(description='ESKMeans segmentattion')
parser.add_argument('--dataset', dest='d', type=str,  choices=["buckeye", "dummy"], help='datraset to use only available [buckeye, dummy]')
parser.add_argument('--centroids', dest='n_c', type=int, help='number of centroids to use')
parser.add_argument('--min_duration', dest='m_d', type=int, help='min duration of segments')


args = parser.parse_args()

dataset=args.d
n_centroids=args.n_c
min_duration=args.m_d


#TODO sanity check dataset
landmarks, feats_scps, phn_gt, wrd_gt = data_utils.load_dataset(dataset)


#TODO remove
centroids = eskmeans_init(landmarks["dev"], feats_scps["dev"], 5)
centroids = eskmeans(landmarks, feats_scps, phn_gt, wrd_gt, centroids, 1, min_duration)

