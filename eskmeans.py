import random
import numpy
import numpy.linalg
from tqdm import tqdm
import sys
import pickle
import kaldiark
import collections

def up_centroids_and_comp_weights(prev_segments,
                                  edges,
                                  g,
                                  num_centroids,
                                  den_centroids):

    centroids = num_centroids/den_centroids
    centroids = centroids.transpose()

    #removing previous segments from the centroid
    all_args = []
    for arg, segment in prev_segments:

        v = g.feat_s(segment[0],segment[1])
        all_args.append(arg)
        num_centroids[:,arg] -= v
        den_centroids[arg] -= 1

    for e in edges:
        v = g.feat(e)
        arg, _ = assign_cluster(v, centroids)

        num_centroids[:,arg] += v
        den_centroids[arg] += 1

    for idx, den in enumerate(den_centroids):
        if(den != 0):
            centroids[idx,:] = num_centroids[:,idx]/den_centroids[idx]

    return num_centroids, den_centroids

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
    def __init__(self, pooling_engine, centroids, feats):
        self.vertices = 0
        self.edges = 0

        self.min_duration = 20

        self.min_edges = 0
        self.max_edges = 6

        self.tail = {}
        self.head = {}
        self.time = {}
        self.in_edges = collections.defaultdict(list)

        self.feats = feats
        self.centroids = centroids
        self.pooling_engine = pooling_engine

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

    def feat_s(self, s, t):

        return self.pooling_engine.subsample(self.feats[s:t+1])

    def feat(self, e):
        s = self.time[self.tail[e]]
        t = self.time[self.head[e]]
        return self.pooling_engine.subsample(self.feats[s:t+1])

    def weight(self, e):
        v = self.feat(e)

        d = self.duration(e)

        num_edges = self.head[e] - self.tail[e]

        if num_edges < self.min_edges or num_edges > self.max_edges:
            d = float('inf')

        if self.duration(e) < self.min_duration:
            d = float('inf')

        c_id, m = assign_cluster(v, self.centroids)

        return m * d, c_id

    def duration(self, e):

        t = self.time[self.tail[e]]
        h = self.time[self.head[e]]

        return h-t

def build_graph(landmarks, pooling_engine, centroids, feats):
    g = Graph(pooling_engine, centroids, feats)
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

            w, _ = g.weight(e)
            cand = d[g.tail[e]] + w

            if cand > m:
                m = cand
                arg = e

        d[v] = m
        back[v] = arg
    #
    # Assuming that the last vertex is the final vertex
    # and that 0 is the initial vertex
    #
    path_e = []
    state_sequence = []
    nll = 0

    v = g.vertices - 1
    while v != 0:
        e = back[v]
        w, c = g.weight(e)
        path_e.append(e)
        v = g.tail[e]
        state_sequence.append((c,(v, g.head[e])))

    path_e.reverse()
    state_sequence.reverse()
    segments = [ (g.time[g.tail[e]], g.time[g.head[e]]) for e in path_e ]

    return state_sequence, path_e, segments, nll

def eskmeans(landmarks,
             feats,
             num_centroids,
             den_centroids,
             max_number_centroids,
             nepoch,
             pooling_engine,
             initial_segments):


    centroids = num_centroids/den_centroids
    centroids = centroids.transpose()

    prev_segments = dict(sorted(initial_segments.items()))
    feats = dict(sorted(feats.items()))

    utt_ids = list(feats.keys())
    utt_idxs = list(range(len(feats)))

    nll_epoch = 0

    for epoch in range(nepoch):

        #TODO uncomment this after replicating one epoch
        #utt_order = random.shuffle(feat_idxs)
        utt_order = utt_idxs

        for idx_sample in tqdm(utt_order):

            #get utterance id so we can use it to retrive
            utt_id = utt_ids[idx_sample]

            #expectation: find the best path
            g = build_graph(landmarks[utt_id], pooling_engine, centroids, feats[utt_id])
            transcription, edges, segments, nll = shortest_path(g)
            nll_epoch += nll

            #maximitzation: modify centroids
            num_centroids, den_centroids, = up_centroids_and_comp_weights(prev_segments[utt_id],
                                                                         edges,
                                                                         g,
                                                                         num_centroids,
                                                                         den_centroids)

            centroids = num_centroids / den_centroids


        print('epoch: {}'.format(epoch))
        print('nll: {}'.format(nll_epoch))
        print('')
        sys.exit()

    return None

