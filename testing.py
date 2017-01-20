# This is a main executable used for testing the CBR pipeline

from Utilities import *
from Stage0_CaseBase import Case_Base
from Stage1_Retrieve import CBR_retrieve
from Stage2_Reuse import CBR_Reuse
from Stage3_Revise import CBR_Revise
from Stage4_Retain import CBR_Retain

def test_1():
    # ==================== Test for Euclidean and Manhattan distances =================================
    data_all = np.genfromtxt(fname='dataFromKaggle/train301.csv', dtype=np.float, skip_header=1, delimiter=',')

    X, y = np.hsplit(data_all, np.array([data_all.shape[1] - 1]))

    y = y.astype(np.float, copy=False)

    CB = Case_Base()
    CB.load_weights(file_name='kNNweights/weights301.csv')


    # Testing parameters
    n_splits = 5
    distance_metrics = ["EUCL+W", "EUCL", "MANH+W", "MANH"]
    ret_strategies = ["Regular", "Always", "Never"]
    ks = [2,3,5,7,10,15]


    total_iters = len(distance_metrics) * len(ret_strategies) * len(ks) * n_splits


    kf = KFold(n_splits=n_splits, shuffle=True)

    stats1 = np.zeros((total_iters, 13),dtype='|S20')
    # 0 - iter_id
    # 1 - distance metric
    # 2 - retention strategy
    # 3 - k
    # 4 - leix_alpha (if applicable)
    # 5 - no of test cases
    # 6 - Case_size_start
    # 7 - Case_size_end
    # 8 - sum of errors
    # 9 - minimum of errors
    # 10 - maximum of errors
    # 11 - stddev of errors
    # 12 - percentage errors

    iter_id = 0

    for distance_metric in distance_metrics:
        for ret_strategy in ret_strategies:
            for k in ks:

                split = 0

                for train_index, test_index in kf.split(X):
                    stats1[iter_id, 0] = iter_id
                    stats1[iter_id, 1] = distance_metric
                    stats1[iter_id, 2] = ret_strategy
                    stats1[iter_id, 3] = str(k)
                    stats1[iter_id, 4] = str(0)
                    stats1[iter_id, 5] = str(test_index.shape[0])


                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    n_test_cases = X_test.shape[0]
                    CB.load_data(X_train, y_train, norm_feats=True, norm_labels=True)

                    stats1[iter_id, 6] = str(CB.data.shape[0])

                    error = np.zeros((X_test.shape[0]))
                    prcts = np.zeros((X_test.shape[0]))

                    print "Iteration: ", iter_id + 1, "of", total_iters
                    print "Distance metric: ", distance_metric
                    print "Retention Strategy: ", ret_strategy
                    print "k : ", k
                    print "Fold : ", split + 1, " of ", n_splits
                    prg = ProgressBar(n_test_cases)

                    for idx, nc in enumerate(X_test):

                        CB_ret = CBR_retrieve(case_base=CB,
                                              new_case=np.array([nc]),
                                              new_label=np.array([y_test[idx]]),
                                              k=k,
                                              dist_meas=distance_metric)

                        CBR_Reuse(CB_ret)
                        er, prct = CBR_Revise(CB_ret)
                        # print "case", test_index[idx], "\terror:", er, "\tpercentage:", prct

                        error[idx] = er
                        prcts[idx] = prct
                        CBR_Retain(CB_ret, strategy=ret_strategy, std_thresh=0.015)

                        prg.update(idx+1)

                    print "\n\n"
                    error = np.abs(error)
                    prcts = np.abs(prcts)

                    stats1[iter_id, 7] = str(CB.data.shape[0])
                    stats1[iter_id, 8] = str(np.average(error))
                    stats1[iter_id, 9] = str(np.min(error))
                    stats1[iter_id, 10] = str(np.max(error))
                    stats1[iter_id, 11] = str(np.std(error))
                    stats1[iter_id, 12] = str(np.average(prcts))
                    split += 1
                    iter_id += 1


                    out_fn1 = "testStats_EUCL_MANH.csv"
                    np.savetxt(out_fn1, stats1, fmt='%s', delimiter=',')

def test_2():
    # ==================== Test for L'Eixample distance =================================
    data_all = np.genfromtxt(fname='dataFromKaggle/train79.csv', dtype=None, skip_header=0, delimiter=',')
    X, y = np.hsplit(data_all[1:, :], np.array([data_all.shape[1] - 1]))
    y = y.astype(np.float, copy=False)

    CB = Case_Base()
    CB.load_LEIX_map(file_name='dataFromKaggle/LEIX_map_20170119_1715.csv')
    CB.load_weights(file_name='kNNweights/weights79.csv')

    # Testing parameters
    distance_metrics = ["LEIX"]
    leix_alphas = [0.95,0.75,0.6,0.5]
    n_splits = 5
    ret_strategies = ["Always", "Never", "Regular"]
    ks = [2,3,5,7,10,15]

    total_iters = len(distance_metrics) * len(ret_strategies) * len(ks) * len(leix_alphas) * n_splits

    kf = KFold(n_splits=n_splits, shuffle=True)
    stats2 = np.zeros((total_iters,13),dtype='|S20')

    iter_id = 0

    for distance_metric in distance_metrics:
        for ret_strategy in ret_strategies:
            for k in ks:
                for leix_alpha in leix_alphas:

                    split = 0

                    for train_index, test_index in kf.split(X):

                        stats2[iter_id, 0] = iter_id
                        stats2[iter_id, 1] = distance_metric
                        stats2[iter_id, 2] = ret_strategy
                        stats2[iter_id, 3] = str(k)
                        stats2[iter_id, 4] = str(leix_alpha)
                        stats2[iter_id, 5] = str(test_index.shape[0])


                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        n_test_cases = X_test.shape[0]

                        N, D = X.shape
                        feat_select = np.zeros( D, dtype=bool)
                        feat_select[43:59] = True
                        feat_select[61:69] = True
                        feat_select[72] = True
                        feat_select[74:78] = True

                        CB.load_data(X_train, y_train, norm_feats=True,feats_select=feat_select, norm_labels=True)

                        CB.leix_alpha = leix_alpha

                        stats2[iter_id, 6] = CB.data.shape[0]

                        error = np.zeros((X_test.shape[0]))
                        prcts = np.zeros((X_test.shape[0]))

                        print "Iteration: ", iter_id + 1, "of", total_iters
                        print "Distance metric: ", distance_metric
                        print "Retention Strategy: ", ret_strategy
                        print "k : ", k
                        print "Fold : ", split + 1, " of ", n_splits
                        prg = ProgressBar(n_test_cases)

                        for idx, nc in enumerate(X_test):

                            CB_ret = CBR_retrieve(case_base=CB,
                                                  new_case=np.array([nc]),
                                                  new_label=np.array([y_test[idx]]),
                                                  k=k,
                                                  dist_meas=distance_metric)

                            CBR_Reuse(CB_ret)
                            er, prct = CBR_Revise(CB_ret)
                            # print "case", test_index[idx], "\terror:", er, "\tpercentage:", prct
                            error[idx] = er
                            prcts[idx] = prct
                            CBR_Retain(CB_ret, strategy=ret_strategy, std_thresh=10000)
                            prg.update(idx + 1)

                        print "\n\n"
                        error = np.abs(error)
                        prcts = np.abs(prcts)

                        stats2[iter_id, 7] = str(CB.data.shape[0])
                        stats2[iter_id, 8] = str(np.average(error))
                        stats2[iter_id, 9] = str(np.min(error))
                        stats2[iter_id, 10] = str(np.max(error))
                        stats2[iter_id, 11] = str(np.std(error))
                        stats2[iter_id, 12] = str(np.average(prcts))
                        split += 1
                        iter_id += 1


                        out_fn2 = "testStats_LEIX.csv"
                        np.savetxt(out_fn2, stats2, fmt='%s', delimiter=',')



if __name__ == "__main__":
    S1_start = datetime.datetime.now()
    print "Stage 1 started: ", S1_start
    test_1()

    S2_start = datetime.datetime.now()
    print "Stage 2 started: ", S2_start
    # test_2()

    test_end_time = datetime.datetime.now()
    print "Stage 1 started: ", S1_start
    print "Stage 2 started: ", S2_start
    print "Testing Finished: ", test_end_time