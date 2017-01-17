# This is a main executable to demonstrate how the CBR system should be used

import numpy as np
from sklearn.model_selection import KFold
from Stage0_CaseBase import Case_Base
from Stage1_Retrieve import CBR_retrieve
from Stage2_Reuse import CBR_Reuse
from Stage3_Revise import CBR_Revise
from Stage4_Retain import CBR_Retain

data_all = np.genfromtxt(fname='dataFromKaggle/train301.csv', dtype=None, names=None, delimiter=',')
X, y = np.hsplit(data_all[1:, :], np.array([data_all.shape[1] - 1]))
y = y.astype(np.float, copy=False)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
stats = np.zeros((6,n_splits))
# 0 - k
# 1 - no of test cases
# 2 - Case_size_start
# 3 - Case_size_end
# 4 - Error
# 5 - stddev of error

CB = Case_Base()
CB.load_LEIX_map(file_name='dataFromKaggle/LEIX_map_20170108_1540.csv', set_alpha=0.3)
CB.load_weights(file_name='dataFromKaggle/weights301.csv')


for k in range(5,15,2):
    split = 0
    for train_index, test_index in kf.split(X):
        stats[0, split] = k
        stats[1, split] = test_index.shape[0]


        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        n_test_cases = X_test.shape[0]
        CB.load_data(X_train, y_train)
        stats[2, split] = CB.data.shape[0]

        error = np.zeros((1,1))
        for id, nc in enumerate(X_test):
            print "Fold", split + 1, " of ", n_splits, ", ",
            print "\tk = ", k, ", ",
            print"\tCase ", id + 1, " of ", n_test_cases
            CB_ret = CBR_retrieve(case_base=CB,
                                  new_case=np.array([nc]),
                                  new_label=np.array([y_test[id]]),
                                  k=k,
                                  dist_meas="DIST_EUCL")
            CBR_Reuse(CB_ret)
            er = CBR_Revise(CB_ret)
            np.append(error,er)
            CBR_Retain(CB_ret, std_thresh=10000)

        stats[3, split] = CB.data.shape[0]
        stats[4, split] = np.sum(error)
        stats[5, split] = np.std(error)
        split += 1
        print stats