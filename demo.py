##@package demo
# This is a demo executable to demonstrate how this CBR system might be used

from Utilities import *
from Stage0_CaseBase import Case_Base
from Stage1_Retrieve import CBR_retrieve
from Stage2_Reuse import CBR_Reuse
from Stage3_Revise import CBR_Revise
from Stage4_Retain import CBR_Retain

# Load data (For Eixample distance)
# data_all = np.genfromtxt(fname='dataFromKaggle/train79.csv', dtype=None, skip_header=1, delimiter=',')

# Load data (For Euclidean and Manhattan distances)
data_all = np.genfromtxt(fname='dataFromKaggle/train301.csv', dtype=np.float, skip_header=1, delimiter=',')

X, y = np.hsplit(data_all, np.array([data_all.shape[1] - 1]))
y = y.astype(np.float, copy=False)

# Determine the shape of the training data
N,d = X.shape

# Get a random row for demo
idx = np.random.randint(0,N-1)
new_case_demo_X = np.array([X[idx, :]])
new_case_demo_y = np.array([y[idx, :]])
X = np.delete(X,idx,0)
y = np.delete(y,idx,0)
N, D = X.shape

# Create the case base object
CB = Case_Base()

# Load the L'Eixample map in case the L'Eixample distance will be used
CB.load_LEIX_map(file_name='dataFromKaggle/LEIX_map_20170119_1715.csv', set_alpha=0.9)

# Load feature weights if the weight-sensitive distance is to be used
CB.load_weights(file_name='kNNweights/weights301.csv')

# FOR EIXAMPLE DISNTACE ONLY
    # Load data into the case base
    # Select features to normalize. Known from the data, only the following features are numerical
    # feat_select = np.zeros( D, dtype=bool)
    # feat_select[43:59] = True
    # feat_select[61:69] = True
    # feat_select[72] = True
    # feat_select[74:78] = True
# CB.load_data(X, y, norm_feats=True, feats_select=feat_select, norm_labels=True)


CB.load_data(X, y, norm_feats=True, feats_select=None, norm_labels=True)

# Stage 1 - Retrieve
CB_ret = CBR_retrieve(case_base=CB,
                      new_case=new_case_demo_X,
                      new_label=new_case_demo_y,
                      k=2,
                      dist_meas="EUCL")


# Stage 2 - Reuse
CBR_Reuse(CB_ret)

# Stage 3 - Revise
error_reuse, prct_error = CBR_Revise(CB_ret)

# Stage 4 - Retain
retained = CBR_Retain(CB_ret, strategy="Regular", std_thresh=0.015)

# Print some statistics
print "#####    Demo run for CBR system    #####\n"
print "Randomly selected test case id:", idx
print "Test case label - ", round(new_case_demo_y[0][0],2)
print "Reused case label - ", round(denormalize(CB_ret.reused_label[0][0], CB_ret.case_base.lbl_min, CB_ret.case_base.lbl_max),2)
print "Error - ", round(denormalize(CB_ret.reused_label[0][0], CB_ret.case_base.lbl_min, CB_ret.case_base.lbl_max),2) ,"(", round(prct_error[0][0],2), "%)"
print "Test case retained? - ", retained
print "Case-base size before/after - ", N, "/", CB.data.shape[0]