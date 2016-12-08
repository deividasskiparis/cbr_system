import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt


def kNN(case_base, new_case, k, dist_meas = "DIST_EUCL", attr_weights = None, LEIX_map = None, LEIX_alpha = 0.5):
    # Returns k-nearest-neighbors to new_case from the case_base
    # dist_measure: DIST_EUCL - Euclidean distance
    #               DIST_MANH - Manhattan distance
    #               DIST_LEIX - l'Eixample distance

    def Kronecker_delta(val1, val2):
        if val1 == val2:
            return 1
        return 0
    def LEIX_dist(vector1, vector2, LEIX_map, attr_weights, LEIX_alpha):
        # Calculates L'Eixample distance between two vectors
        assert vector1.shape == vector2.shape
        assert vector1.shape == LEIX_map.shape[1]

        numr = 0
        denom = 0
        dist = 0

        for col in range(LEIX_map.shape[1]):
            L_col = LEIX_map[col]
            att_type = L_col[0]

            att_val1 = vector1[col]
            att_val2 = vector2[col]

            if att_type == '0' and attr_weights[col] < LEIX_alpha:
                up_val = L_col[3]
                low_val = L_col[2]
                dist = abs(att_val1 - att_val2)/(up_val - low_val)

            elif (att_type == '0' and attr_weights[col] > LEIX_alpha):

                cat_val1 = 0
                cat_val2 = 0
                n_mods = L_col[1]
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

                dist = abs(cat_val1 - cat_val2)/n_mods

            elif att_type == '1':

                cat_val1 = 0
                cat_val2 = 0
                n_mods = L_col[1]
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

                dist = abs(cat_val1 - cat_val2) / n_mods


            elif att_type == '2':
                dist = 1 - Kronecker_delta(att_val1, att_val2)


            numr += np.exp(attr_weights[col] * dist)
            denom += np.exp(attr_weights[col])

        return numr/denom

    new_case_n = new_case.shape[0]
    new_case_dims = new_case.shape[1]

    case_base_n = case_base.shape[0]
    case_base_dims = case_base.shape[1]

    assert (dist_meas == "DIST_EUCL" or dist_meas == "DIST_MANH" or dist_meas == "DIST_LEIX")
    assert (case_base_dims == new_case_dims and new_case_n == 1)
    if dist_meas == "DIST_LEIX":
        assert attr_weights is not None and LEIX_map is not None
        assert  attr_weights.shape[0] == case_base_dims
        assert LEIX_map.shape[1] == case_base_dims
        assert LEIX_alpha >= 0 and LEIX_alpha <= 1


    if dist_meas == "DIST_EUCL":
        # Calculate second norm (Euclidean distnace) between the new case and case_base
        new_tiled = np.tile(new_case,(case_base_n,1))
        diff = np.subtract(case_base, new_tiled)
        dist_eucl = np.linalg.norm(diff,ord=2, axis=1)
        # Get k minimum (unsorted!!!)
        min_idxs = np.argpartition(dist_eucl,k)[:k]

        return dist_eucl[min_idxs], min_idxs

    elif dist_meas == "DIST_MANH":
        # Calculate first norm (Manhattan distnace) between the new case and case_base
        new_tiled = np.tile(new_case,(case_base_n,1))
        diff = np.subtract(case_base, new_tiled)
        dist_manh = np.linalg.norm(diff,ord=1, axis=1)
        # Get k minimum (unsorted!!!)
        min_idxs = np.argpartition(dist_manh,k)[:k]

        return dist_manh[min_idxs], min_idxs

    elif dist_meas == "DIST_LEIX":

        dist_leix = np.zeros((case_base_n,1),dtype=int)
        for row in range(case_base_n):
            dist_leix[row] = LEIX_dist(new_case, case_base[row], LEIX_map, attr_weights, LEIX_alpha)

        # Get k minimum (unsorted!!!)
        min_idxs = np.argpartition(dist_leix,k)[:k]

        return dist_leix[min_idxs], min_idxs


if __name__ == "__main__":
    new_case = np.random.rand(1,2)

    # new_case = np.array([[0.25,0.75]])
 #    case_base = np.array([[ 0.40461157,  0.55683227],
 # [ 0.06919205,  0.25065274],
 # [ 0.49946357,  0.3386246 ],
 # [ 0.83712319,  0.5933376 ],
 # [ 0.99707458,  0.12873529],
 # [ 0.24541566,  0.64939248],
 # [ 0.25441935,  0.78398782],
 # [ 0.69697643,  0.45492324],
 # [ 0.06706135, 0.90091   ],
 # [ 0.09709043,  0.47938234],
 # [ 0.14319426,  0.87855187],
 # [ 0.54265014,  0.17606687],
 # [ 0.43625492,  0.40710672],
 # [ 0.83566863,  0.26068208],
 # [ 0.98470511,  0.89974885],
 # [ 0.80506879,  0.38633246],
 # [ 0.22509562,  0.82582272],
 # [ 0.60282755,  0.07223626],
 # [ 0.68949023,  0.91043417],
 # [ 0.65924711, 0.11932163],
 # [ 0.70315676,  0.16224709],
 # [ 0.47023523,  0.20062633],
 # [ 0.42361451,  0.17478604],
 # [ 0.46979067,  0.43666728],
 # [ 0.20277721,  0.55286279],
 # [ 0.28054345,  0.36781067],
 # [ 0.8858598 ,  0.83365171],
 # [ 0.08422862,  0.66685995],
 # [ 0.24148748,  0.17760375],
 # [ 0.50074202,  0.92862401],
 # [ 0.34632474,  0.88975747],
 # [ 0.73150337,  0.0391518 ],
 # [ 0.48519422,  0.63790444],
 # [ 0.49225382,  0.74642141],
 # [ 0.88343699, 0.91802455],
 # [ 0.67481844,  0.53916975],
 # [ 0.21909941,  0.75180143],
 # [ 0.21652872,  0.50401636],
 # [ 0.77721192,  0.93575716],
 # [ 0.7915698 ,  0.53548471],
 # [ 0.07750677,  0.59857752],
 # [ 0.98288997,  0.31496038],
 # [ 0.06162153,  0.88657062],
 # [ 0.01734003,  0.04875837],
 # [ 0.14516041,  0.79879213],
 # [ 0.19609885,  0.64874039],
 # [ 0.995986  ,  0.88778617],
 # [ 0.1821503 ,  0.9571085 ],
 # [ 0.797197  ,  0.59645775],
 # [ 0.3945454 ,  0.89013917],
 # [ 0.34877914,  0.82034393],
 # [ 0.60277962,  0.55905761],
 # [ 0.44306522,  0.89516162],
 # [ 0.2984943 ,  0.76707728],
 # [ 0.32163129,  0.01441598],
 # [ 0.78752523,  0.78902307],
 # [ 0.23407774,  0.89419401],
 # [ 0.34642316,  0.904942  ],
 # [ 0.37468917,  0.42059626],
 # [ 0.84160886,  0.05838244],
 # [ 0.4477369 ,  0.57966743],
 # [ 0.36527768,  0.415644  ],
 # [ 0.02802242,  0.67820692],
 # [ 0.6077316 ,  0.33394001],
 # [ 0.463749  ,  0.26074356],
 # [ 0.18846697,  0.82079949],
 # [ 0.05561293,  0.19105988],
 # [ 0.06638597,  0.07731496],
 # [ 0.36186306,  0.58146746],
 # [ 0.99937748,  0.1438968 ],
 # [ 0.9918541 ,  0.18782745],
 # [ 0.48900022,  0.32856536],
 # [ 0.30693204,  0.83985446],
 # [ 0.22066551,  0.0604647 ],
 # [ 0.76746069,  0.75023093],
 # [ 0.47696016,  0.53512471],
 # [ 0.6801349 ,  0.1420225 ],
 # [ 0.36009533,  0.60566293],
 # [ 0.79537591,  0.70275951],
 # [ 0.07076218,  0.48546268],
 # [ 0.89182799,  0.84594476],
 # [ 0.35572974,  0.75357359],
 # [ 0.94285518,  0.56944846],
 # [ 0.92629538,  0.84875368],
 # [ 0.53182334,  0.36330019],
 # [ 0.15440881,  0.984087  ],
 # [ 0.88757943,  0.7744527 ],
 # [ 0.2132219 ,  0.05050276],
 # [ 0.68267327,  0.77963597],
 # [ 0.24185286, 0.16911466],
 # [ 0.66840388,  0.02705929],
 # [ 0.24713764,  0.87568706],
 # [ 0.97011753,  0.27838145],
 # [ 0.76138327,  0.15748868],
 # [ 0.85283881,  0.95842301],
 # [ 0.95972444,  0.4407556 ],
 # [ 0.48010497,  0.11834339],
 # [ 0.00514256,  0.00616358],
 # [ 0.28065987,  0.67775544],
 # [ 0.17296524,  0.63245103]])

    weights = np.array([0.05,0.95])

    case_base = np.random.rand(100,2)
    LEIX_map = None #genfromtxt('LEIX_map.csv', dtype='|S30', skip_header=0, names=None, delimiter=',')

    _, idxs = kNN(case_base,new_case,5,"DIST_MANH", attr_weights=weights, LEIX_map=LEIX_map)
    _NNs = case_base[idxs, :]
    _case_base = case_base
    _case_base = np.delete(_case_base, idxs, 0)
    print (_NNs)
    print (_case_base)
    plt.plot(_case_base[:,0], _case_base[:,1],'bo',new_case[:,0], new_case[:,1],'g*', _NNs[:,0], _NNs[:,1],'rs' )
    plt.show()
    print("")