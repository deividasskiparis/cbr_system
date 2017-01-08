# This is module containing definition for the main Case Base object and the temporary Retrieved Case Base. The latter
# is mostly used as an information carrier withing the CBR cycle

import numpy as np
from kNN import kNN

class Case_Base(object): # SIMONS PART
    def __init__(self):
        self.data = np.array([])
        self.GTlabels = np.array([])
        self.weights = None
        self.leix_map = None
        self.leix_alpha = None

    def load_data(self, file_name):
        data_all = np.genfromtxt(file_name, dtype=None, names=None, delimiter=',')
        self.data, self.GTlabels = np.hsplit(data_all[1:,:], np.array([data_all.shape[1] - 1]))

    def load_LEIX_map(self, file_name, set_alpha):
        assert set_alpha > 0 and set_alpha < 1
        self.leix_map = np.genfromtxt(file_name, dtype='|S30', names=None, delimiter=',')
        self.leix_alpha = set_alpha

    def load_weights(self, file_name):
        self.weights = np.array([np.genfromtxt(file_name, dtype=np.float, names=None, delimiter=',')])
    pass

class Retrieved_CaseBase(object):
    def __init__(self, case_base, new_case, new_label, k, dist_meas="DIST_EUCL"):
        self.case_base = case_base

        self.new_case = new_case
        self.new_case_y  = new_label #Label
        self.K = k
        self.dist_meas = dist_meas

        _, idxs, new_case_all_num, ret_all_num = kNN(self.case_base.data, self.new_case, self.K,
                                                     dist_meas=self.dist_meas,
                                                     LEIX_map=case_base.leix_map,
                                                     LEIX_alpha=case_base.leix_alpha,
                                                     attr_weights=case_base.weights)

        # Retrieved indices
        self.idxs = idxs
        # Retrieved data rows
        self.ret_data = self.case_base.data[self.idxs, :]
        self.ret_labels = self.case_base.GTlabels[self.idxs, :]

        # Retrieved cases - all numerical
        self.RC_all_num = ret_all_num
        # New case all numerical in case it is not
        self.NC_all_num = new_case_all_num

        # Reused label will be here after stage 2
        self.reused_label = np.array([0])

    def add_new_case(self):
        # Will be used in Retrieve stage
        self.case_base.data = np.append(self.case_base.data, self.new_case, 0)
        self.case_base.GTlabels = np.append(self.case_base.GTlabels, self.reused_label, 0)
        print ("New case added to the case base")
