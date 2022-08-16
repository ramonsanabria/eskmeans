#!/usr/bin/env python3

import collections
import pickle
import kaldiark
import random
import numpy
import numpy.linalg


def subsample(feats, n):
    k = len(feats) / n

    result = []
    for i in range(n):
        result.extend(feats[int(k * i)])

    return numpy.array(result)


def assign_cluster(v, centroids):
    m = float('inf')
    arg = -1
    for i, u in enumerate(centroids):
        cand = numpy.linalg.norm(u - v)
        cand = cand * cand
        if cand < m:
            m = cand
            arg = i

    return arg, m


class Graph:
    def __init__(self):
        self.vertices = 0
        self.edges = 0
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
        v = subsample(self.feats[s:t], 10)
        return v

    def weight(self, e):
        v = self.feat(e)
        arg, m = assign_cluster(v, centroids)
        return m


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
        m = float('inf')
        for e in g.in_edges[v]:
            cand = d[g.tail[e]] + g.weight(e)
            if cand < m:
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

    return centroids


def eskmeans(landmark_sets, feat_scp, centroids, nepoch):
    for epoch in range(nepoch):
        sums = numpy.zeros(centroids.shape)
        counts = numpy.zeros(centroids.shape[0])

        for i, scp in enumerate(feat_scp):
            landmarks = landmark_sets[i]

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
                arg, m = assign_cluster(v, centroids)
                sums[arg] = sums[arg] * (counts[arg] / (counts[arg] + 1)) + v / (counts[arg] + 1)
                counts[arg] += 1
                path_weight += m

            print('epoch: {}'.format(epoch))
            print('key: {}'.format(scp[0]))
            print('path: {}'.format([(g.time[g.tail[e]], g.time[g.head[e]]) for e in path]))
            print('path weight: {}'.format(path_weight))
            print('')

        centroids = sums

    return centroids


landmark_file = open('data/landmarks.pkl', 'rb')
landmarks = pickle.load(landmark_file)
landmark_file.close()

feat_scp_file = open('data/feats/feats_local.scp')
feat_scp = []
for line in feat_scp_file:
    parts = line.strip().split()
    file, shift = parts[1].split(':')
    feat_scp.append((parts[0], file, int(shift)))
feat_scp_file.close()

centroids = eskmeans_init(landmarks, feat_scp, 5)
centroids = eskmeans(landmarks, feat_scp, centroids, 10)

