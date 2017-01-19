# This is a demo executable to demonstrate how this CBR system might be used

import numpy as np
from Stage0_CaseBase import Case_Base
from Stage1_Retrieve import CBR_retrieve
from Stage2_Reuse import CBR_Reuse
from Stage3_Revise import CBR_Revise
from Stage4_Retain import CBR_Retain


# Load data (For L'Eixample distance)
# data_all = np.genfromtxt(fname='dataFromKaggle/train79.csv', dtype=None, names=None, delimiter=',')


# Load data (For Euclidean and Manhattan distances)
data_all = np.genfromtxt(fname='dataFromKaggle/train301.csv', dtype=None, names=None, delimiter=',')


X, y = np.hsplit(data_all[1:, :], np.array([data_all.shape[1] - 1]))
y = y.astype(np.float, copy=False)

# Determine the shape of the training data
N,d = X.shape


# Get a random row for demo
idx = np.random.randint(0,N-1)
new_case_demo_X = np.array([X[idx, :]])
new_case_demo_y = np.array([y[idx, :]])
X = np.delete(X,idx,0)
y = np.delete(y,idx,0)
N,_ = X.shape


# Create the case base object
CB = Case_Base()

# Load the L'Eixample map in case the L'Eixample distance will be used
CB.load_LEIX_map(file_name='dataFromKaggle/LEIX_map_20170108_1540.csv', set_alpha=0.9)

# Load feature weights if the weight-sensitive distance is to be used
CB.load_weights(file_name='kNNweights/weights301.csv')

# Load data into the case base
CB.load_data(X, y)

# Stage 1 - Retrieve
CB_ret = CBR_retrieve(case_base=CB,
                      new_case=new_case_demo_X,
                      new_label=new_case_demo_y,
                      k=3,
                      dist_meas="EUCL+W")


# Stage 2 - Reuse
CBR_Reuse(CB_ret)

# Stage 3 - Revise
error_reuse, prct_error = CBR_Revise(CB_ret)

# Stage 4 - Retain
retained = CBR_Retain(CB_ret, std_thresh=10000)

# Print some statistics
print "New case label - ", new_case_demo_y
print "Retrieved labels - ", CB_ret.ret_labels
print "Reused case label - ", CB_ret.reused_label
print "Error - ", error_reuse[0][0] ,"(", prct_error[0][0], ")"
print "New case retained - ", retained
print "Case-base size before/after - ", N, "/", CB.data.shape[0]