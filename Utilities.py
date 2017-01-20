import numpy as np
import sys, time, datetime
from sklearn.model_selection import KFold

from kNN import kNN

def compress_by_col(data, selectors):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    for idx, s in enumerate(selectors):
        if s:
            yield data[:, idx]

def normalize(array, norm_feats=None, minm=None, maxm=None):
    # Function normalizes a given array in the range [0, 1]
    # norm_feats is a boolean(or int) vector, which specifies which features should be normalized.
    # Works with arrays, which have number represented as strings
    # If minm and maxm are provided, the normaliztion is performed within this range

    assert isinstance(array, np.ndarray)
    N, D = array.shape

    # Ensure the provided vector is correct shape and type
    if norm_feats is None:
        norm_feats = np.ones( D, dtype=np.bool)
    else:
        assert isinstance(norm_feats, np.ndarray)
        assert norm_feats.shape[0] == D
        assert norm_feats.dtype == np.int or norm_feats.dtype == np.dtype(bool)
        norm_feats = norm_feats.astype(bool)

    # Select only required features
    array_ = np.array(list(compress_by_col(array, norm_feats)))
    array_ = array_.T


    # Convert to numerical form
    data_type = array.dtype
    if data_type != np.float:
        array_ = array_.astype(np.float)

    # Find minimum and maximum
    if minm is None and maxm is None:
        feat_min = np.min(array_, 0)
        feat_max = np.max(array_, 0)
    else:
        assert isinstance(minm, np.ndarray) and isinstance(maxm, np.ndarray)
        assert minm.shape[0] == array_.shape[1] and maxm.shape[0] == array_.shape[1]
        assert minm.dtype == np.float and maxm.dtype == np.float
        feat_min = minm
        feat_max = maxm

    # Tile the minimum and maximum values to have the as arrays of the same shape as input array
    feat_min_tiled = np.tile(feat_min, (N, 1))
    feat_max_tiled = np.tile(feat_max, (N, 1))

    # Normalize
    array_ = (array_ - feat_min_tiled )/ (feat_max_tiled - feat_min_tiled)

    # Convert back to the original data type
    if data_type != np.float:
        array_ = array_.astype(dtype=data_type)

    if not norm_feats.all():
        ref = 0
        for idx, bl in enumerate(norm_feats):
            if bl:
                array[:, idx] = array_[:, ref]
                ref += 1
        return array, feat_min, feat_max
    else:
        return array_, feat_min, feat_max


def denormalize(value, minm, maxm):
    return value * (maxm - minm) + minm

class ProgressBar:
    # Progress Bar
    def __init__(self, max_val):
        self.max_val = max_val
        pass

    def update(self, current):

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-20s] %d%% (%d of %d)" %
                         ('#'*int(round(float(current)/self.max_val*20,2)),
                          round(float(current)/self.max_val*100,2),
                          current,
                          self.max_val))
        sys.stdout.flush()
class Timer:
    # A tool for measuring run time of the function
    def __init__(self):
        self.clockings = []
        pass
    def tic(self):
        self.clockings.append(time.time())
        pass
    def toc(self, message=None):
        t_end = time.time()
        if len(self.clockings)>0:
            t_start = self.clockings.pop()
            elapsed = round((t_end - t_start) * 1000, 3)
            if message is not None:
                print "**TIMER: ", message, elapsed, "ms"
            return elapsed