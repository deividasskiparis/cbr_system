import numpy as np
from Stage0_CaseBase import Retrieved_CaseBase


def CBR_Revise(ret_CB):
    # Function compares the labels to ground truth labels
    # Returns the error between the values
    # labels - np array of values
    # gt_labels - np array of the same shape and same type of values

    assert isinstance(ret_CB, Retrieved_CaseBase)

    if ret_CB.new_case_y is None:
        return 0, 0

    prct = 0
    assert ret_CB.reused_label.shape == ret_CB.new_case_y.shape # Same dimensionality
    assert ret_CB.reused_label.dtype == ret_CB.new_case_y.dtype # Same data type


    if ret_CB.reused_label.dtype == np.dtype(np.float) or ret_CB.reused_label.dtype == np.dtype(np.int):
        error_ = ret_CB.reused_label - ret_CB.new_case_y
        prct = error_/ret_CB.new_case_y
    else:
        error_ = np.sum(np.array(ret_CB.reused_label != ret_CB.new_case_y, dtype=np.int))

    return error_, prct