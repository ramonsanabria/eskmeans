import scipy.signal as signal
import numpy

def subsample(feats):

    n=10
    k = len(feats) / n

    result = []
    for i in range(n):
        result.extend(feats[int(k * i)])

    return numpy.array(result)


def subsample_herman(feats):

    n=10
    feats_t = feats.T

    y_new = signal.resample(feats_t, n, axis=1).flatten("C")

    return numpy.array(y_new)

def factory_function(pooling_method):

    if("sub" in pooling_method):
        return subsample

    elif("herman" in pooling_method):
        return subsample_herman
