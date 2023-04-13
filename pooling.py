import scipy.signal as signal
import numpy as np

class PoolingEngine:
    
    def __init__(self, method, feature_dim, feature_type):
        """
        Constructor for the Subsampler class.
        
        Parameters:
            method (str): The subsampling method to use. Can be either 'random' or 'first'.
        """
        self.method = method
        self.feature_dim = feature_dim
        if("hubert" in feature_type):
            self.freq_red = 2
        else:
            self.freq_red = 1

    def __subsample_herman(self, feats):
        n=10
        feats_t = feats.T
        y_new = signal.resample(feats_t, n, axis=1).flatten("C")

        if (np.linalg.norm(y_new) != 0):
            return np.array(y_new / np.linalg.norm(y_new))
        else:
            return np.array(y_new)

    def __subsample(self, feats):
        n=10
        k = len(feats) / n

        result = []
        for i in range(n):
            result.extend(feats[int(k * i)])

        return np.array(result)

    def __average(self, feats):

        return np.average(feats, axis=0)

    def get_out_feat_dim(self):

        if self.method == 'subsample':
            return self.feature_dim*10
        elif self.method == 'herman':
            return self.feature_dim*10
        elif self.method == 'average':
            return self.feature_dim
        else:
            raise ValueError('Invalid subsampling method: {}'.format(self.method))


    def pool(self, feats, start, end):
        """
        Pool the features into one unique vector using start and end time in millisecond

        Parameters:
            feats (np.array): Matrix of size (feats,time)
        Returns:
            np.array: one unique vector. Its size depends on the subsampling method.
        """

        end = end+1
        start = int(start/self.freq_red)
        end = int(end/self.freq_red)
        feats =  feats[start:end,:]

        if self.method == 'subsample':
            return self.__subsample(feats)
        elif self.method == 'herman':
            return self.__subsample_herman(feats)
        elif self.method == 'average':
            return self.__average(feats)
        else:
            raise ValueError('Invalid subsampling method: {}'.format(self.method))

