import scipy.signal as signal
import numpy as np

class PoolingEngine:
    
    def __init__(self, method, feature_dim):
        """
        Constructor for the Subsampler class.
        
        Parameters:
            method (str): The subsampling method to use. Can be either 'random' or 'first'.
        """
        self.method = method
        self.feature_dim = feature_dim

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
    
    def get_out_feat_dim(self):

        if self.method == 'subsample':
            return self.feature_dim*10
        elif self.method == 'herman':
            return self.feature_dim*10
        else:
            raise ValueError('Invalid subsampling method: {}'.format(self.method))


    def subsample(self, feats):
        """
        Subsamples the input data according to the chosen subsampling method.
        
        Parameters:
            data (list): The input data to subsample.
            
        Returns:
            A list containing the subsampled data.
        """
        if self.method == 'subsample':
            return self.__subsample(feats)
        elif self.method == 'herman':
            return self.__subsample_herman(feats)
        else:
            raise ValueError('Invalid subsampling method: {}'.format(self.method))

