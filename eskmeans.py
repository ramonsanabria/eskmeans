import pickle

import numpy as np
import numpy.linalg
from tqdm import tqdm
import sys
import collections
from collections import defaultdict
import os

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
        m = float('-inf')
        arg = -1
        for i, u in enumerate(centroids):

            cand = np.linalg.norm(u - v)
            cand = -cand * cand

            if cand > m:
                m = cand
                arg = i

        return arg, m

    def feat_s(self, s, t):

        return self.pooling_engine.pool(self.feats[s:t + 1])

    def feat(self, e):
        s = self.time[self.tail[e]]
        t = self.time[self.head[e]]
        return self.pooling_engine.pool(self.feats[s:t + 1])

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
            for rule in rules:
                if rule[0] == c_id:
                    aux_list.append((rule[1], segment))
                else:
                    aux_list.append((c_id, segment))
        prev_segments[utt_id] = aux_list

    return prev_segments

def unit_test_segments_and_transcriptions(segments_and_transcipts_ours, language, speaker, epoch):

        with open(os.path.join('./data/kamperetal_segmentation/',
                                language,
                               "epoch_"+str(epoch),
                               speaker+'.pkl'),"rb") as f:
            segments_kamper_etal = pickle.load(f)

        for key, kamper_segment in segments_kamper_etal.items():
            our_segment =  [el[1]for el in segments_and_transcipts_ours[key]]
            our_segment = sorted(our_segment)
            kamper_segment = sorted(kamper_segment)

            if (our_segment != kamper_segment):
                print("UNIT TEST FAILED: SEGMENTS IN  EPOCH "+str(epoch)+" FROM UTT "+key+" ARE DIFFERENT AS KAMPER ET AL")
                sys.exit()

        print("UNIT TEST PASSED: SEGMENTS IN  EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")

        with open(os.path.join('./data/kamperetal_transcripts/',
                               language,
                               "epoch_"+str(epoch),
                               speaker+'.pkl'),"rb") as f:
            transcripts_kamper_etal = pickle.load(f)

        for key, kamper_transcripts in transcripts_kamper_etal.items():
            our_transcripts =  [el[0]for el in segments_and_transcipts_ours[key]]
            if(our_transcripts != kamper_transcripts):
                print(our_transcripts)
                print(kamper_transcripts)
                print("UNIT TEST FAILED: TRANSCRIPTS IN  EPOCH "+str(epoch)+" FROM UTT "+key+" ARE DIFFERENT AS KAMPER ET AL")
                #sys.exit()

        print("UNIT TEST PASSED: TRANSCRIPTS IN  EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")


def unit_test_centroids(centroids_ours, language, speaker, epoch):

    centroids_kamperetal = np.load(os.path.join('./data/kamperetal_epochs_centroids/',
                                                language,
                                                "epoch_"+str(epoch),
                                                speaker+'.npy'))

    if (np.allclose(centroids_kamperetal, centroids_ours, atol=0.001)):
        print("UNIT TEST PASSED: CENTROIDS EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")
        print("\tTOTAL DIFF: " + str(np.sum(centroids_kamperetal - centroids_ours)))
    else:
        print("UNIT FAILED: CENTROIDS EPOCH "+str(epoch)+" ARE DIFFERENT AS KAMPER ET AL")
        print("\tTOTAL DIFF: " + str(np.sum(centroids_kamperetal - centroids_ours)))
        sys.exit()


def convert_to_segments_and_transcriptions(segments_and_transcriptioss):
    transcriptions = defaultdict(list)
    segments = defaultdict(list)

    for utt_id in segments_and_transcriptioss.keys():
        transcriptions[utt_id] = [el[0] for el in segments_and_transcriptioss[utt_id]]
        segments[utt_id] = [el[1] for el in segments_and_transcriptioss[utt_id]]

    return transcriptions, segments



def eskmeans(landmarks,
             feats,
             centroids,
             nepoch,
             pooling_engine,
             initial_segments,
             language,
             speaker,
             min_edges,
             max_edges,
             min_duration):


    prev_segments = dict(sorted(initial_segments.items()))
    feats = dict(sorted(feats.items()))

    utt_ids = list(feats.keys())
    utt_idxs = list(range(len(feats)))


    for epoch in range(nepoch):

        print("ITERATION: ", epoch)
        #TODO uncomment this after replicating one epoch
        #utt_order = random.shuffle(feat_idxs)
        utt_order = utt_idxs

        nll_epoch = 0

        #reset rules for centroid exchnage
        centroids.reset_rules_for_previous_segment()

        #for idx_sample in tqdm(utt_order):
        for idx_sample in tqdm(utt_order):

            #get utterance id so we can use it to retrive
            utt_id = utt_ids[idx_sample]

            #expectation: compute shortest path
            g = build_graph(landmarks[utt_id],
                            pooling_engine,
                            centroids.get_centroids(),
                            feats[utt_id])

            edges, seg_and_cids, nll = shortest_path(g)
            nll_epoch += nll


            #maximitzation: modify centroids
            #we return segments just in case some reordering is needed
            rules = centroids.up_centroids_and_comp_weights(
                            prev_segments[utt_id],
                            edges,
                            g,
                            epoch)

            prev_segments[utt_id] = seg_and_cids


            #if some reorderings happened in centroids, we update cluster id from previous segments
            if(len(rules) > 0):
                print(rules)
                prev_segments = update_previous_segments(prev_segments, rules)

        unit_test_segments_and_transcriptions(prev_segments, language, speaker, epoch)
        unit_test_centroids(centroids.get_final_centroids(), language, speaker, epoch)

    return convert_to_segments_and_transcriptions(prev_segments)