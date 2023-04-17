import random
import numpy as np
import numpy.linalg
import multiprocessing as mp
from tqdm import tqdm
import unit_test
import collections
from collections import defaultdict

class Graph:
    def __init__(self,
                 pooling_engine,
                 centroids,
                 feats,
                 min_edges,
                 max_edges,
                 min_duration):


        self.vertices = 0
        self.edges = 0

        self.min_edges = min_edges
        self.max_edges = max_edges

        self.min_duration = min_duration

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


    def _assign_cluster(self, v, centroids):
        dists = -np.linalg.norm(centroids - v, axis=1) ** 2
        arg = np.argmax(dists)
        m = dists[arg]

        return arg, m

    def feat_s(self, s, t):
        return self.pooling_engine.pool(self.feats, s ,t)

    def feat(self, e):
        s = self.time[self.tail[e]]
        t = self.time[self.head[e]]

        return self.pooling_engine.pool(self.feats, s ,t)

    def weight(self, e):
        v = self.feat(e)

        d = self.duration(e)

        num_edges = self.head[e] - self.tail[e]

        if num_edges < self.min_edges or num_edges > self.max_edges:
            d = float('inf')

        if d < self.min_duration:
            d = float('inf')

        c_id, m = self._assign_cluster(v, self.centroids)

        if(m == 0.0 and d == float('inf')):
             return -float('inf'), c_id

        return m * d, c_id

    def duration(self, e):

        t = self.time[self.tail[e]]
        h = self.time[self.head[e]]

        return h-t

def build_graph(landmarks,
                pooling_engine,
                centroids,
                feats,
                min_edges,
                max_edges,
                min_duration):

    g = Graph(pooling_engine,
              centroids,
              feats,
              min_edges,
              max_edges,
              min_duration)

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

        nll += w

    path_e.reverse()
    state_sequence.reverse()


    segments = [ (c_id, (g.time[edge[0]], g.time[edge[1]])) for c_id, edge in state_sequence ]

    return path_e, segments, nll


def update_previous_segments(prev_segments, rules):

    #substitute the cluster id for the corresponding one
    for utt_id in prev_segments.keys():
        aux_list = []
        for c_id, segment in prev_segments[utt_id]:
            flag_rule = False
            for rule in rules:
                if rule[0] == c_id:
                    aux_list.append((rule[1], segment))
                    flag_rule = True

            if not flag_rule:
                aux_list.append((c_id, segment))

        prev_segments[utt_id] = aux_list

    return prev_segments



def convert_to_segments_and_transcriptions(segments_and_transcriptioss):
    transcriptions = defaultdict(list)
    segments = defaultdict(list)

    for utt_id in segments_and_transcriptioss.keys():
        transcriptions[utt_id] = [el[0] for el in segments_and_transcriptioss[utt_id]]
        segments[utt_id] = [el[1] for el in segments_and_transcriptioss[utt_id]]

    return segments, transcriptions

def eskmeans(landmarks,
             feats,
             centroids,
             nepochs,
             pooling_engine,
             initial_segments,
             language,
             speaker,
             min_edges,
             max_edges,
             min_duration,
             unit_test_flag):

    prev_segments = dict(sorted(initial_segments.items()))
    feats = dict(sorted(feats.items()))

    utt_ids = list(feats.keys())
    utt_idxs = list(range(len(feats)))

    nll_prev = -float('inf')

    for epoch in range(nepochs):

        print("ITERATION: ", epoch)

        if not unit_test:
            random.shuffle(utt_idxs)

        nll_epoch = 0

        #for idx_sample in tqdm(utt_idxs):
        for idx_sample in utt_idxs:
        #for idx_sample in utt_order:

            #get utterance id so we can use it to retrive
            utt_id = utt_ids[idx_sample]

            #expectation: compute shortest path
            g = build_graph(landmarks[utt_id],
                            pooling_engine,
                            centroids.get_centroids(),
                            feats[utt_id],
                            min_edges,
                            max_edges,
                            min_duration)

            edges, _, nll = shortest_path(g)

            nll_epoch += nll

            #maximitzation: modify centroids
            #we return segments just in case some reordering is needed
            rules, seg_and_cids = centroids.up_centroids_and_comp_weights( prev_segments[utt_id],
                            edges,
                            g,
                            epoch)

            prev_segments[utt_id] = seg_and_cids


            #if some reorderings happened in centroids, we update cluster id from previous segments
            if len(rules) > 0:
                prev_segments = update_previous_segments(prev_segments, rules)

        if(unit_test_flag):
            unit_test.segments_and_transcriptions(prev_segments, language, speaker, epoch)
            unit_test.centroids(centroids.get_final_centroids(), language, speaker, epoch)

        print("EPOCH "+str(epoch)+" NLL: ", nll_epoch)

        if(nll_epoch == nll_prev):
            print("NLL did not improve, stopping training")
            break

        nll_prev=nll_epoch

    return convert_to_segments_and_transcriptions(prev_segments)


def process_sample(args):

    utt_id, landmarks, pooling_engine, centroids, feats, min_edges, max_edges, min_duration = args

    # expectation: compute shortest path
    g = build_graph(landmarks,
                    pooling_engine,
                    centroids.get_centroids(),
                    feats,
                    min_edges,
                    max_edges,
                    min_duration)

    edges, seg_and_cids, nll = shortest_path(g)

    return utt_id, seg_and_cids, nll


def eskmeans_em(landmarks,
                feats,
                centroids,
                nepochs,
                pooling_engine,
                min_edges,
                max_edges,
                min_duration):

    feats = dict(sorted(feats.items()))
    prev_segments = {}

    utt_ids = list(feats.keys())

    nll_prev = -float('inf')

    for epoch in range(nepochs):

        print("ITERATION: ", epoch)

        nll_epoch = 0
        prev_segments = {}

        print("\texpectation")
        input_data = [(utt_id, landmarks[utt_id], pooling_engine, centroids, feats[utt_id], min_edges, max_edges,
                       min_duration)
                      for utt_id in utt_ids]

        #with mp.Pool(mp.cpu_count()) as pool:
        with mp.Pool(3) as pool:

            for utt_id, seg_and_cids, nll in tqdm(pool.imap_unordered(process_sample,
                                                                      input_data),
                                                                    total=len(utt_ids)):
                nll_epoch += nll
                prev_segments[utt_id] = seg_and_cids

        print("\tmaximization")
        print("\t\tacomulating")

        centroids.reset()
        for utt_id in utt_ids:

            #acomulate centroids
            centroids.add_to_centroids(prev_segments[utt_id], feats[utt_id], pooling_engine)


        print("\t\tmaximizing")
        centroids.compute_centroids()

        print("EPOCH "+str(epoch)+" NLL: ", nll_epoch)

        if(nll_epoch == nll_prev):
            print("NLL did not improve, stopping training")
            break

        nll_prev=nll_epoch

    return convert_to_segments_and_transcriptions(prev_segments)
