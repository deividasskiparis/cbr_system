import numpy as np
from Stage0_CaseBase import Retrieved_CaseBase


def CBR_Revise(ret_CB, eval_within = 1000):
    assert isinstance(ret_CB, Retrieved_CaseBase)
    # Function compares the labels to ground truth labels. If the labels are numerical, eval_within threshold is
    # used to have equality intervals. If eval_within = 5, 45 == 45+-5
    # Returns number of mismatches
    # labels - np array of values
    # gt_labels - np array of the same shape and same type of values

    assert ret_CB.reused_label.shape == ret_CB.new_case_y.shape # Same dimensionality
    assert ret_CB.reused_label.dtype == ret_CB.new_case_y.dtype # Same data type

    if lbls.dtype == np.dtype(np.float) or lbls.dtype == np.dtype(np.int):
        diff = np.abs(ret_CB.reused_label - ret_CB.new_case_y)
        error_ = np.sum(diff > eval_within)
    else:
        error_ = np.sum(np.array(ret_CB.reused_label != ret_CB.new_case_y, dtype=np.int))

    return error_

if __name__ == "__main__":
    print("Demo run\nComparing arrays:")
    lbls = np.array([5000,1000,20000,150000,15200.5])
    lbls_gt = np.array([5100,1000,25000,150001,15200.4])

    print ("lbls = " + str(lbls))
    print("lbls_gt = " + str(lbls_gt))

    no_errors = CBR_Revise(lbls, lbls_gt, 99)

    print("Number of mismatches = " + str(no_errors))