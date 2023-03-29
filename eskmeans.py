import random
import numpy
import numpy.linalg
from tqdm import tqdm
import sys
import pickle
import kaldiark
import collections

def up_centroids_and_comp_weights(prev_segments, path, g, sums, counts, scp, k_herman, centroids):

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

    def feat(self, e):
        s = self.time[self.tail[e]]
        t = self.time[self.head[e]]
        v = self.pooling_engine.subsample(self.feats[s:t+1])
        return v

    def weight(self, e):
        v = self.feat(e)

        d = self.duration(e)

        num_edges = self.head[e] - self.tail[e]

        if num_edges < self.min_edges or num_edges > self.max_edges:
            d = float('inf')

        if self.duration(e) < self.min_duration:
            d = float('inf')

        _, m = assign_cluster(v, self.centroids)

        return m * d

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
        can_list = []
        for e in g.in_edges[v]:
            if(v == 8):
                print(str(d[g.tail[e]])+" + "+str(g.weight(e))+" = "+str(d[g.tail[e]] + g.weight(e))+" duration = "
                                                                                                     +str(g.duration(
                    e)))

            cand = d[g.tail[e]] + g.weight(e)
            can_list.append(cand)
            if cand > m:
                m = cand
                arg = e

        d[v] = m
        back[v] = arg

    sys.exit()
    #
    # Assuming that the last vertex is the final vertex
    # and that 0 is the initial vertex
    #
    path = []
    nll = 0

    v = g.vertices - 1
    while v != 0:
        e = back[v]
        nll += g.weight(e)
        path.append(e)
        v = g.tail[e]

    path.reverse()
    path_time = [ (g.time[g.tail[e]], g.time[g.head[e]]) for e in path ]

    return path_time, nll


def eskmeans(landmarks, feats, centroids, nepoch, min_duration, pooling_engine, initial_segments):

    inital_segments = dict(sorted(initial_segments.items()))
    feats = dict(sorted(feats.items()))

    utt_ids = list(feats.keys())
    utt_idxs = list(range(len(feats)))

    for epoch in range(nepoch):

        sums =  centroids
        counts = numpy.zeros(centroids.shape[0])

        #TODO uncomment this after replicating one epoch
        #utt_order = random.shuffle(feat_idxs)
        utt_order = utt_idxs

        for idx_sample in tqdm(utt_order):

            #get utterance id so we can use it to retrive
            utt_id = utt_ids[idx_sample]
            g = build_graph(landmarks[utt_id], pooling_engine, centroids, feats[utt_id])
            path, nll = shortest_path(g)

            print(nll)
            print(path)
            sys.exit()
            sums, counts, path_weight, k_herman  = up_centroids_and_comp_weights(prev_paths,
                                                                                 path,
                                                                                 g,
                                                                                 sums,
                                                                                 counts,
                                                                                 idx_utterance,
                                                                                 k_herman,
                                                                                 centroids)


            for idx in range(centroids.shape[0]):
                if counts[idx] > 0:
                    centroids[idx,:] = sums[idx,:]/counts[idx]

            prev_paths[scp[0]] = path

            print('epoch: {}'.format(epoch))
            print('sample: {}'.format(idx_utterance))
            print('path: {}'.format([(g.time[g.tail[e]], g.time[g.head[e]]) for e in path]))
            print('path weight: {}'.format(path_weight))
            print('')


    return centroids

