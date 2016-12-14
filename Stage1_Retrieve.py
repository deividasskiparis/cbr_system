import numpy as np
from kNN import kNN


def CBR_retrieve(case_base, new_case, k):

    _, idxs = kNN(case_base, new_case, k, dist_meas="DIST_EUCL")

    return idxs