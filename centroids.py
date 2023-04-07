import numpy as np
import sys

class Centroids:

    def __init__(self, num_centroids, den_centroids, language, speaker_id, centroids_rand):
        """
        Constructor for the Centroids class.

        Parameters:
            num_centroids (numpy.ndarray): The numerator of the centroids.
            den_centroids (numpy.ndarray): The denominator of the centroids.
        """

        #private object
        self.__num_centroids = num_centroids
        self.den_centroids = den_centroids
        self.__centroids = self.__num_centroids / self.den_centroids

        self.__centroids_rand = centroids_rand

        #rules to reorder previous segments
        self.__rules_previous_segments = []

        #keep tracking of randomly sampled centroids
        self.__non_randomly_initialized_centroids = den_centroids.shape[0]

    #TODO matrice this function
    def __assign_cluster(self, v, centroids):
        m = float('-inf')
        arg = -1
        for i, u in enumerate(centroids):

            cand = np.linalg.norm(u - v)
            cand = -cand * cand

            if cand > m:
                m = cand
                arg = i

        return arg, m

    def up_centroids_and_comp_weights(self,
                                      prev_segments,
                                      edges,
                                      g,
                                      epoch):
        """
        Updates the centroids and the components weights.
        :param prev_segments: The segments assigned to that utterance in the previous iteration
        :param edges: The edges of the graph assigned in the current iteraion
        :param g: the graph of the utterance in the current iteration.
        :return:
        """

        #removing previous segments from the centroid
        all_args = []
        new_rules = []

        #print("deleting segments: "+str([el[0] for el in prev_segments]))
        for arg, segment in prev_segments:
            #C19_041232-041902

            #chage segment ids according to the rules
            #for rule in self.__rules_previous_segments:
            #    if(rule[0] == arg):
            #        print("applying rule:"+str(rule))
            #        arg = rule[1]

            #if(arg == 231):
            #    print("PRE del k:"+str(arg)+" count: "+str(self.den_centroids[arg]))

            v = g.feat_s(segment[0],segment[1])
            all_args.append(arg)
            self.__num_centroids[:, arg] -= v
            self.den_centroids[arg] -= 1


        #we incorporate the new segments into the centroids
        #we need to recompute component so we can incorporate them into centroids
        #we do not apply rules here (rearranging segments do not effect current components)
        current=[]
        for e in edges:

            v = g.feat(e)
            arg, _ = self.__assign_cluster(v, self.__centroids.transpose())

            current.append(arg)

            if(arg > self.__non_randomly_initialized_centroids):
                arg = self.__non_randomly_initialized_centroids

            if(arg == self.__non_randomly_initialized_centroids):
                self.__non_randomly_initialized_centroids += 1

            self.__num_centroids[:, arg] += v
            self.den_centroids[arg] += 1


        for idx_den_zero in np.where(self.den_centroids[:self.__non_randomly_initialized_centroids] == 0)[0][::-1]:

            #we track which element is substitued
            self.__non_randomly_initialized_centroids -= 1

            #we substitue the empty centroid with the last centroid
            if(self.__non_randomly_initialized_centroids != idx_den_zero):

                self.__num_centroids[:, idx_den_zero] = self.__num_centroids[:, self.__non_randomly_initialized_centroids]
                self.den_centroids[idx_den_zero] = self.den_centroids[self.__non_randomly_initialized_centroids]

                self.__centroids[:,self.__non_randomly_initialized_centroids] = \
                    self.__centroids_rand[:,self.__non_randomly_initialized_centroids]

                self.__rules_previous_segments.append((self.__non_randomly_initialized_centroids, idx_den_zero))
                new_rules.append((self.__non_randomly_initialized_centroids, idx_den_zero))

        #we remove the last centroid(s)
        self.__centroids[:,:self.__non_randomly_initialized_centroids] = \
            self.__num_centroids[:,:self.__non_randomly_initialized_centroids]/ \
            self.den_centroids[:self.__non_randomly_initialized_centroids]

        #set them to zero and prepare for recieving a new component
        self.__num_centroids[:, self.__non_randomly_initialized_centroids:] = 0
        self.den_centroids[self.__non_randomly_initialized_centroids:] = 0

        #return the only the new rules (otherwise new rules will be empty)
        return new_rules

    def get_final_centroids(self):

        return self.__centroids.transpose()[:self.__non_randomly_initialized_centroids,:]

    def get_centroids(self):
        """
        Returns the centroids.
        :return (numpy.ndarray): The centroids.
        """
        return self.__centroids.transpose()

    def reset_rules_for_previous_segment(self):
        """
        Resets the rules -- this is excuted at end of every epoch
        """

        self.__rules_previous_segments = []
