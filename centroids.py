import numpy

class Centroids:

    def __init__(self, num_centroids, den_centroids):
        """
        Constructor for the Centroids class.

        Parameters:
            num_centroids (numpy.ndarray): The numerator of the centroids.
            den_centroids (numpy.ndarray): The denominator of the centroids.
        """
        self.__num_centroids = num_centroids
        self.__den_centroids = den_centroids

        #rules for reassigment of centroids
        self.rules = []

    #TODO matrice this function
    def __assign_cluster(self, v, centroids):
        m = float('-inf')
        arg = -1
        for i, u in enumerate(centroids):

            cand = numpy.linalg.norm(u - v)
            cand = -cand * cand

            if cand > m:
                m = cand
                arg = i

        return arg, m

    def up_centroids_and_comp_weights(self,
                                      prev_segments,
                                      edges,
                                      g):
        """
        Updates the centroids and the components weights.
        :param prev_segments: The segments assigned to that utterance in the previous iteration
        :param edges: The edges of the graph assigned in the current iteraion
        :param g: the graph of the utterance in the current iteration.
        :return:
        """

        centroids = self.__num_centroids / self.__den_centroids
        centroids = centroids.transpose()
        idx_last_component = self.__den_centroids.shape[0] - 1

        #removing previous segments from the centroid
        all_args = []
        for arg, segment in prev_segments:

            #reassigning the centroid (if some centroid has been removed)
            for rule in self.rules:
                if(rule[0] == arg):
                    arg = rule[1]

            v = g.feat_s(segment[0],segment[1])
            all_args.append(arg)
            self.__num_centroids[:, arg] -= v
            self.__den_centroids[arg] -= 1

        for e in edges:
            v = g.feat(e)
            arg, _ = self.__assign_cluster(v, centroids)

            self.__num_centroids[:, arg] += v
            self.__den_centroids[arg] += 1

        #removing centroids without components
        if numpy.any(numpy.equal(self.__den_centroids, 0)):
            for idx_den_zero in numpy.where(self.__den_centroids == 0)[0][::-1]:

                #we substitue the empty centroid with the last centroid
                if(idx_last_component != idx_den_zero):
                    self.__num_centroids[:, idx_den_zero] = self.__num_centroids[:, idx_last_component]
                    self.__den_centroids[idx_den_zero] = self.__den_centroids[idx_last_component]
                    self.rules.append((idx_last_component,idx_den_zero))

                #we remove the last centroid
                self.__num_centroids = numpy.delete(self.__num_centroids, idx_last_component, axis=1)
                self.__den_centroids = numpy.delete(self.__den_centroids, idx_last_component, axis=0)

    def get_centroids(self):
        """
        Returns the centroids.
        :return (numpy.ndarray): The centroids.
        """
        centroids = self.__num_centroids / self.__den_centroids
        return centroids.transpose()
