import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from itertools import compress


def kNN(case_base, new_case, k, dist_meas = "EUCL", attr_weights = None, LEIX_map = None, LEIX_alpha = 0.5):
    # Returns k-nearest-neighbors to new_case from the case_base
    # dist_measure: DIST_EUCL - Euclidean distance
    #               DIST_MANH - Manhattan distance
    #               DIST_LEIX - l'Eixample distance


    def Kronecker_delta(val1, val2):
        if val1 == val2:
            return 1
        return 0
    def LEIX_dist(vector1, vector2, LEIX_map, attr_weights, LEIX_alpha):
        np.seterr(all='warn')
        # Calculates L'Eixample distance between two vectors
        assert vector1.shape == vector2.shape
        assert vector1.shape[0] == LEIX_map.shape[1]

        numr = 0
        denom = 0
        dist = 0

        vec1_allNum = np.zeros(vector1.shape, dtype = np.float)
        vec2_allNum = np.zeros(vector1.shape, dtype = np.float)
        for col in range(LEIX_map.shape[1]):
            L_col = LEIX_map[:, col]
            att_type = L_col[0]

            att_val1 = vector1[col]
            att_val2 = vector2[col]

            if att_val1 == '?' or att_val2 == '?':
                continue



            if att_type == '0' and attr_weights[0, col] < LEIX_alpha:
                att_val1 = float(vector1[col])
                att_val2 = float(vector2[col])

                up_val = float(L_col[3])
                low_val = float(L_col[2])
                dist = float(abs(att_val1 - att_val2))/(up_val - low_val)
                vec1_allNum[col] = att_val1 /(up_val - low_val)
                vec2_allNum[col] = att_val2 / (up_val - low_val)

            elif (att_type == '0' and attr_weights[0, col] > LEIX_alpha):
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

    assert (d_meas == "EUCL" or d_meas == "MANH" or d_meas == "LEIX")
    assert (case_base_dims == new_case_dims and new_case_n == 1)
    if weighted:
        assert attr_weights is not None
        assert  attr_weights.shape[1] == case_base_dims and attr_weights.shape[0] == 1

    if d_meas == "LEIX":
        assert attr_weights is not None and LEIX_map is not None
        assert LEIX_map.shape[1] == case_base_dims
        assert LEIX_alpha >= 0 and LEIX_alpha <= 1


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
        # Get k minimum (unsorted!!!)
        uns_min_idxs = np.argpartition(dist_eucl,k)[:k]

        uns_red_dists = dist_eucl[uns_min_idxs]
        s_min_idxs = [i[0] for i in sorted(enumerate(uns_red_dists), key=lambda x: x[1])]

        sorted_idxs = uns_min_idxs[s_min_idxs]

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
        # Get k minimum (unsorted!!!)
        uns_min_idxs = np.argpartition(dist_manh,k)[:k]

        uns_red_dists = dist_manh[uns_min_idxs]
        s_min_idxs = [i[0] for i in sorted(enumerate(uns_red_dists), key=lambda x: x[1])]

        sorted_idxs = uns_min_idxs[s_min_idxs]

        new_case_allNum = np.array([new_tiled[0, :]])
        case_base_allNum = case_base[sorted_idxs, :]

        return uns_red_dists[s_min_idxs], sorted_idxs, new_case_allNum, case_base_allNum

    elif d_meas == "LEIX":

        dist_leix = np.zeros(case_base_n,dtype=float)
        dist_leix[0], new_case_allNum, case_base_allNum = LEIX_dist(new_case[0], case_base[0, :], LEIX_map, attr_weights, LEIX_alpha)
        for row in range(1, case_base_n):
            dist_leix[row], _, all_num_ = LEIX_dist(new_case[0], case_base[row, :], LEIX_map, attr_weights, LEIX_alpha)
            case_base_allNum = np.append(case_base_allNum, all_num_,axis=0)

        # Get k minimum (unsorted!!!)
        uns_min_idxs = np.argpartition(dist_leix,k)[:k]

        uns_red_dists = dist_leix[uns_min_idxs]
        s_min_idxs = [i[0] for i in sorted(enumerate(uns_red_dists), key=lambda x: x[1])]

        sorted_idxs = uns_min_idxs[s_min_idxs]

        return uns_red_dists[s_min_idxs], sorted_idxs, new_case_allNum, case_base_allNum[sorted_idxs, :]