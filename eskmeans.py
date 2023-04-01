import numpy as np
import numpy.linalg
from tqdm import tqdm
import sys
import collections


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

        if d < self.min_duration:
            d = float('inf')

        c_id, m = self._assign_cluster(v, self.centroids)

        # if(c_id == 38):
        #     #print()
        #     print("centroid: "+str(self.centroids[38,:][:10]))
        #     print("segment: "+str(v_aux[:10]))
        #
        if(m == 0.0 and d == float('inf')):
             return -float('inf'), c_id

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
    segments = [ (state_sequence[idx], (g.time[g.tail[e]], g.time[g.head[e]])) for idx, e in enumerate(path_e) ]

    return path_e, segments, nll

def eskmeans(landmarks,
             feats,
             centroids,
             nepoch,
             pooling_engine,
             initial_segments,
             language,
             speaker):


    prev_segments = dict(sorted(initial_segments.items()))
    feats = dict(sorted(feats.items()))

    utt_ids = list(feats.keys())
    utt_idxs = list(range(len(feats)))


    for epoch in range(nepoch):

        #TODO uncomment this after replicating one epoch
        #utt_order = random.shuffle(feat_idxs)
        utt_order = utt_idxs
        nll_epoch = 0

        for idx_sample in tqdm(utt_order):

            #get utterance id so we can use it to retrive
            utt_id = utt_ids[idx_sample]

            #expectation: find the best path
            #if(utt_id == "C19_031956-032684"):
            # print("pre: "+str(den_centroids[38]))

            g = build_graph(landmarks[utt_id],
                            pooling_engine,
                            centroids.get_centroids(),
                            feats[utt_id])

            edges, segments, nll = shortest_path(g)
            nll_epoch += nll

            #maximitzation: modify centroids
            centroids.up_centroids_and_comp_weights(prev_segments[utt_id],
                                                    edges,
                                                    g)


            prev_segments[utt_id] = segments

        centroids_kampereral = np.load('./data/kamperetal_epochs_centroid/' + language + '/' + speaker + '_'+str(
            epoch)+'.npy')
        print(centroids_kampereral.shape)
        print(centroids.get_centroids().shape)
        sys.exit()

        centroids = centroids.transpose()

        if (np.allclose(centroid_kampereral, centroids, atol=0.001)):
            print("TEST PASSED: CENTROIDS EPOCH "+str(epoch)+" ARE THE SAME AS KAMPER ET AL")
            print("\tTOTAL DIFF: " + str(np.sum(centroid_kampereral - centroids)))



