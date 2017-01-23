##@package Stage0_CaseBase
# This is module containing definition for the main Case Base object and the temporary Retrieved Case Base. The latter
# is mostly used as an information carrier withing the CBR cycle
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




from Utilities import *

## Documentation for the Case_Base class.
class Case_Base(object):
    ## Standard object constructor.
    #  @param self The object pointer.
    #  @param data The input data, where columns represent features and rows data instances
    #  @param GTlabels Ground truth labels.
    #  @param weights Weights representing the importance of each feature for the retrieval stage
    #  @param leix_map An array created by make_Eixample_map.py function
    #  @param leix_alpha Eixample threshold. Numerical features with weights above this threshold are treated as lineal
    def __init__(self):
        self.data = np.array([])
        self.GTlabels = np.array([])
        self.weights = None
        self.leix_map = None
        self.leix_alpha = None
        self.data_normalized = False
        self.labels_normalized = False

    ## Loads data into the CaseBase obejct
    #  @param X Predictors
    #  @param y labels
    #  @param norm_feats Defines whether the data should be normalized [True, False]
    #  @param feats_select If normalization is needed, this gives an option to normalize only particular features in
    # dataset. Especially useful for Retrieval using Eixample distance, since nominal variables cannot be normalized
    #  @param norm_labels Defines whether the labels should be normalized too
    def load_data(self, X, y, norm_feats=False, feats_select=None, norm_labels=True):
        if norm_feats:
            self.data, self.feat_min, self.feat_max = normalize(X, feats_select)
            self.norm_feats = feats_select
            self.data_normalized = True
        else:
            self.data, self.feat_min, self.feat_max = X, None, None

        if norm_labels:
            self.GTlabels, self.lbl_min, self.lbl_max = normalize(y)
            self.labels_normalized = True
        else:
            self.GTlabels, self.lbl_min, self.lbl_max = y, 0, 1

    ## Load Eixample map into the Case_Base
    #  @param file_name Filename of the saved eixample map csv, produced to make_Eixample_map.py
    #  @param set_alpha Eixample threshold. Numerical features with weights above this threshold are treated as lineal
    def load_LEIX_map(self, file_name, set_alpha=0.3):
        assert set_alpha > 0 and set_alpha < 1
        self.leix_map = np.genfromtxt(file_name, dtype='|S30', names=None, delimiter=',')
        self.leix_alpha = set_alpha

    ## Load feature weights into the Case_Base
    #  @param file_name Filename of the feature weights csv
    def load_weights(self, file_name):
        self.weights = np.array([np.genfromtxt(file_name, dtype=np.float, names=None, delimiter=',')])
    pass

## Documentation for the Retrieved_CaseBase class.
class Retrieved_CaseBase(object):
    ## Standard object constructor.
    #  @param case_base Case_Base obejct
    #  @param k Number of nearest neighbours for retrieval
    #  @param new_case A new instance to be tested on CBR engine
    #  @param new_label Ground truth label of the new case
    #  @param dist_measure Distance measure to be used for Retrieval
    def __init__(self, case_base, k, new_case, new_label=None, dist_meas="DIST_EUCL"):

        self.case_base = case_base

        # Check if data normalziation is required
        if self.case_base.data_normalized:
            self.new_case,_,_ = normalize(new_case, case_base.norm_feats,case_base.feat_min, case_base.feat_max)
        else:
            self.new_case = new_case

        # Check if label normalziation is required
        if self.case_base.labels_normalized and new_label is not None:
            self.new_case_y,_,_ = normalize(new_label,minm=case_base.lbl_min, maxm=case_base.lbl_max)
        else:
            self.new_case_y  = new_label #Label

        self.K = k
        self.dist_meas = dist_meas

        # Perform KNN retrieval
        _, idxs, new_case_all_num, ret_all_num = kNN(self.case_base.data, self.new_case, self.K,
                                                     dist_meas=self.dist_meas,
                                                     EIX_map=case_base.leix_map,
                                                     EIX_alpha=case_base.leix_alpha,
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

    ## Adds new case to the corresponding Case_Base
    def add_new_case(self):
        # Will be used in Retrieve stage
        self.case_base.data = np.append(self.case_base.data, self.new_case, 0)
        self.case_base.GTlabels = np.append(self.case_base.GTlabels, self.reused_label, 0)
        # print ("New case added to the case base")