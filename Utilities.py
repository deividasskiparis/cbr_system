##@package Utilities
# Package containing helper function for CBR cycle
#
# This package contains:
# 1. Case_Base object used to represent the database.
# 2. Retrieved_CaseBase which is a container for running the CBR cycle. Contains all the necessary information to
# complete all the sage of CBR cycle

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona

import numpy as np
import sys, time, datetime
from sklearn.model_selection import KFold
from kNN import kNN


## Returns only columns of the data, for which the selector is True
#  @param data Array of data [N x D]
#  @param selectors Boolean values indicating which column to return [1 x D]
#  @retval  Iterator object of the reduced columns
def compress_by_col(data, selectors):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    for idx, s in enumerate(selectors):
        if s:
            yield data[:, idx]

## Normalizes values within minimum and maximum values. Works with arrays, which have number represented as strings
#  @param array Array of data [N x D]
#  @param norm_feats Boolean vector, indicating which features should be normalized
#  @param minm Vector of minimum values of each column for normalization.
#  If not provided, it is extracted from the array
#  @param maxm Vector of maximum values of each column for normalization.
#  If not provided, it is extracted from the array
#  @retval Norm_Array Normalized array
def normalize(array, norm_feats=None, minm=None, maxm=None):

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
    diff = array_ - feat_min_tiled
    rng = feat_max_tiled - feat_min_tiled
    rng[rng==0] = 1
    array_ = diff/rng

    # Convert back to the original data type
    if data_type != np.float:
        array_ = array_.astype(dtype=data_type)

    # If not all of the features were normalized, map back the normalized features to the original array
    if not norm_feats.all():
        ref = 0
        for idx, bl in enumerate(norm_feats):
            if bl:
                array[:, idx] = array_[:, ref]
                ref += 1
        return array, feat_min, feat_max
    else:
        return array_, feat_min, feat_max


## Denormalizes value to the original scale, based on minimum and maximum values
#  @param value Value to be denormalized
#  @param minm Minimum of the original range
#  @param maxm Maximum of the original range
#  @retval denorm_val Denormalized value
def denormalize(value, minm, maxm):
    return value * (maxm - minm) + minm

## Documentation for the ProgressBar class. This was used for testing to show the progress
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


## Matlab style tool for measuring function run-time
class Timer:
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