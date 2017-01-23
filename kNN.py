##@package kNN
# Contains custom kNN function used for Retrieval in CBR process
#

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona


import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from itertools import compress


## k-nearest neighbour (kNN) function.
#  Compares a new case in the feature space and returns k most closely matching instances, based on the selected
#  similarity measure
#  @param case_base numpy array of data of shape (row c column) N x D, where N is instances and D - dimensionality
#  @param new_case A new instance of data to be compared to all instances in case_base [1 x D]
#  @param k Number of nearest neighbours
#  @param dist_meas Similarity measure. Avalable values are "EUCL+W", "EUCL", "MANH+W", "MANH", "LEIX" for Euclidean
#  Manhattan and Eixample similarity measure. +W = Weighted.
#  @param attr_weights attribute weights for weighted retrieval. If the dist_meas has "+W" suffix or is "LEIX", this is
#  required.
#  @param EIX_map An array, containing information about data. Created using make_Eixample_map.py
#  @param eix_alpha Eixample threshold. Numerical features with weights above this threshold are treated as lineal
#  @retval dists Distances from k most similar cases
#  @retval idxs Row indices of the most ismilar cases in the case_base
#  @retval new_case_allNum All numerical representation of new_case. Used for Eixample distance measure.
#  @retval case_base_allNum All numerical representation of retrieved cases. Used for Eixample distance measure.
def kNN(case_base, new_case, k, dist_meas = "EUCL", attr_weights = None, EIX_map = None, EIX_alpha = 0.5):
    # Returns k-nearest-neighbors to new_case from the case_base
    # dist_measure: DIST_EUCL - Euclidean distance
    #               DIST_MANH - Manhattan distance
    #               DIST_LEIX - Eixample distance

    ## Return Kronecker_delta of two variables
    #  @param val1 Value 1. Can be numerical or categorical
    #  @param val2 Value 2. Can be numerical or categorical
    #  @retval equal Returns 1 if the values are identical. Otherwise, 0
    def Kronecker_delta(val1, val2):
        if val1 == val2:
            return 1
        return 0

    ## Calculates Eixample distance between two vectors
    #  @param vector1 Numpy array of values [1 x D]
    #  @param vector2 Numpy array of values [1 x D]
    #  @param EIX_map An array, containing information about data. Created using make_Eixample_map.py
    #  @param attr_weights attribute weights for weighted retrieval. If the dist_meas has "+W" suffix or is "LEIX", this is
    #  required.
    #  @param Eix_alpha Eixample threshold. Numerical features with weights above this threshold are treated as lineal

    #  @param val2 Value 2. Can be numerical or categorical
    #  @retval equal Returns 1 if the values are identical. Otherwise, 0
    def LEIX_dist(vector1, vector2, EIX_map, attr_weights, EIX_alpha):
        np.seterr(all='warn')
        # Calculates L'Eixample distance between two vectors
        assert vector1.shape == vector2.shape
        assert vector1.shape[0] == EIX_map.shape[1]

        numr = 0
        denom = 0
        dist = 0

        # Numerical representation of input vectors
        vec1_allNum = np.zeros(vector1.shape, dtype = np.float)
        vec2_allNum = np.zeros(vector1.shape, dtype = np.float)

        # Iterate through every feature
        for col in range(EIX_map.shape[1]):
            L_col = EIX_map[:, col]
            att_type = L_col[0]

            att_val1 = vector1[col]
            att_val2 = vector2[col]

            # Check for missing values
            if att_val1 == '?' or att_val2 == '?':
                continue

            # Calculate weight according to the formulas provided in the paper
            if att_type == '0' and attr_weights[0, col] < EIX_alpha:
                att_val1 = float(vector1[col])
                att_val2 = float(vector2[col])

                up_val = float(L_col[3])
                low_val = float(L_col[2])
                dist = float(abs(att_val1 - att_val2))/(up_val - low_val)
                vec1_allNum[col] = att_val1 /(up_val - low_val)
                vec2_allNum[col] = att_val2 / (up_val - low_val)

            elif (att_type == '0' and attr_weights[0, col] > EIX_alpha):
                att_val1 = float(vector1[col])
                att_val2 = float(vector2[col])

                cat_val1 = 0
                cat_val2 = 0
                n_mods = int(L_col[1])
                for m in range(n_mods-1):
                    # Get the categorical values for both vector attribute values
                    cat_val1 = m
                    if att_val1 > float(L_col[m+4]) :
                        continue
                    else:
                        break

                for m in range(n_mods-1):
                    # Get the categorical values for both vector attribute values
                    cat_val2 = m
                    if att_val2 > float(L_col[m+4]) :
                        continue
                    else:
                        break

                dist = float(abs(cat_val1 - cat_val2))/n_mods
                vec1_allNum[col] = float(cat_val1) / n_mods
                vec2_allNum[col] = float(cat_val2) / n_mods

            elif att_type == '1':

                cat_val1 = 0
                cat_val2 = 0
                n_mods = int(L_col[1])
                for m in range(n_mods - 1):
                    # Get the categorical values for both vector attribute values
                    cat_val1 = m
                    if att_val1 == L_col[m + 4]:
                        break
                    else:
                        continue

                for m in range(n_mods - 1):
                    # Get the categorical values for both vector attribute values
                    cat_val2 = m
                    if att_val2 == L_col[m + 4]:
                        break
                    else:
                        continue


                dist = float(abs(cat_val1 - cat_val2)) / n_mods
                vec1_allNum[col] = float(cat_val1) / n_mods
                vec2_allNum[col] = float(cat_val2) / n_mods


            elif att_type == '2':
                dist = 1 - Kronecker_delta(att_val1, att_val2)
                vec1_allNum[col] = 0
                vec2_allNum[col] = dist

            with np.errstate(over='ignore'):
                numr += np.exp(attr_weights[0, col] * dist)
                denom += np.exp(attr_weights[0, col])


        return numr/denom, np.array([vec1_allNum]), np.array([vec2_allNum])

    new_case_n = new_case.shape[0]
    new_case_dims = new_case.shape[1]

    case_base_n = case_base.shape[0]
    case_base_dims = case_base.shape[1]

    d_meas = dist_meas[:4]
    weighted = (dist_meas[-2:] == "+W" or d_meas == "LEIX")

    # Check for data correctness
    assert (d_meas == "EUCL" or d_meas == "MANH" or d_meas == "LEIX")
    assert (case_base_dims == new_case_dims and new_case_n == 1)
    if weighted:
        assert attr_weights is not None
        assert  attr_weights.shape[1] == case_base_dims and attr_weights.shape[0] == 1

    if d_meas == "LEIX":
        assert attr_weights is not None and EIX_map is not None
        assert EIX_map.shape[1] == case_base_dims
        assert EIX_alpha >= 0 and EIX_alpha <= 1


    if d_meas == "EUCL":
        # Preprocess data
        new_tiled = np.tile(new_case,(case_base_n,1))

        log1 = log2 = False
        if not (np.issubdtype(new_tiled.dtype, np.float) or np.issubdtype(new_tiled.dtype, np.int)):
            # String data was loaded, meaning there are missing values
            log1 = new_tiled == '?'

        if not (np.issubdtype(case_base.dtype, np.float) or np.issubdtype(case_base.dtype, np.int)):
            # String data was loaded, meaning there are missing values
            log2 = case_base == '?'

        log = log1 + log2
        if isinstance(log, np.ndarray):
            #Change missing values in both array to 0
            new_tiled[log] = 0
            case_base[log] = 0

        case_base = case_base.astype(np.float, copy=False)
        new_tiled = new_tiled.astype(np.float, copy=False)

        # Calculate second norm (Euclidean distnace) between the new case and case_base

        diff = np.subtract(case_base, new_tiled)
        if weighted:
            attr_tiled = np.tile(attr_weights[0], (case_base_n, 1))
            diff *= attr_tiled

        dist_eucl = np.linalg.norm(diff,ord=2, axis=1)
        # Get k minimum (unsorted!!!). Using argpartition for speed
        uns_min_idxs = np.argpartition(dist_eucl,k)[:k]

        # Get distances of k nearest neighbours (unsorted!!!)
        uns_red_dists = dist_eucl[uns_min_idxs]

        # Get indices of sorted distances from lowest to highest
        s_min_idxs = [i[0] for i in sorted(enumerate(uns_red_dists), key=lambda x: x[1])]

        # Get sorted indices relative to case_base
        sorted_idxs = uns_min_idxs[s_min_idxs]

        # Get all numerical features
        new_case_allNum = np.array([new_tiled[0, :]])
        case_base_allNum = case_base[sorted_idxs, :]

        return uns_red_dists[s_min_idxs], sorted_idxs, new_case_allNum, case_base_allNum

    elif d_meas == "MANH":

        # Preprocess data
        new_tiled = np.tile(new_case, (case_base_n, 1))

        log1 = False
        log2 = False
        if not (np.issubdtype(new_tiled.dtype, np.float) or np.issubdtype(new_tiled.dtype, np.int)):
            # String data was loaded, meaning there are missing values
            log1 = new_tiled == '?'

        if not (np.issubdtype(case_base.dtype, np.float) or np.issubdtype(case_base.dtype, np.int)):
            # String data was loaded, meaning there are missing values
            log2 = case_base == '?'

        log = log1 + log2
        if isinstance(log, np.ndarray):
            # Change missing values in both array to 0
            new_tiled[log] = 0
            case_base[log] = 0

            case_base = case_base.astype(np.float, copy=False)
            new_tiled = new_tiled.astype(np.float, copy=False)

        # Calculate first norm (Manhattan distnace) between the new case and case_base
        diff = np.subtract(case_base, new_tiled)
        if weighted:
            attr_tiled = np.tile(attr_weights[0], (case_base_n, 1))
            diff *= attr_tiled

        dist_manh = np.linalg.norm(diff,ord=1, axis=1)

        # Get k minimum (unsorted!!!). Using argpartition for speed
        uns_min_idxs = np.argpartition(dist_manh,k)[:k]

        # Get distances of k nearest neighbours (unsorted!!!)
        uns_red_dists = dist_manh[uns_min_idxs]

        # Get indices of sorted distances from lowest to highest
        s_min_idxs = [i[0] for i in sorted(enumerate(uns_red_dists), key=lambda x: x[1])]

        # Get sorted indices relative to case_base
        sorted_idxs = uns_min_idxs[s_min_idxs]

        # Get all numerical features
        new_case_allNum = np.array([new_tiled[0, :]])
        case_base_allNum = case_base[sorted_idxs, :]

        return uns_red_dists[s_min_idxs], sorted_idxs, new_case_allNum, case_base_allNum

    elif d_meas == "LEIX":

        dist_leix = np.zeros(case_base_n,dtype=float)

        # Loop through all cases in the case_base and compare to new case individually
        dist_leix[0], new_case_allNum, case_base_allNum = LEIX_dist(new_case[0], case_base[0, :], EIX_map, attr_weights, EIX_alpha)
        for row in range(1, case_base_n):
            dist_leix[row], _, all_num_ = LEIX_dist(new_case[0], case_base[row, :], EIX_map, attr_weights, EIX_alpha)
            case_base_allNum = np.append(case_base_allNum, all_num_,axis=0)

        # Get k minimum (unsorted!!!). Using argpartition for speed
        uns_min_idxs = np.argpartition(dist_leix,k)[:k]

        # Get distances of k nearest neighbours (unsorted!!!)
        uns_red_dists = dist_leix[uns_min_idxs]

        # Get indices of sorted distances from lowest to highest
        s_min_idxs = [i[0] for i in sorted(enumerate(uns_red_dists), key=lambda x: x[1])]

        # Get sorted indices relative to case_base
        sorted_idxs = uns_min_idxs[s_min_idxs]

        return uns_red_dists[s_min_idxs], sorted_idxs, new_case_allNum, case_base_allNum[sorted_idxs, :]