##@package Stage3_Revise
# This module contains Revise stage for the CBR cycle

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona

import numpy as np
from Stage0_CaseBase import Retrieved_CaseBase

## The function to perform case adaptation for the CBR cycle.
#  @param ret_CB Retrieved_CaseBase
#  @retval error Difference between the ground truth value and adapted value
#  @retval prct Ratio of error/'true value'
def CBR_Revise(ret_CB):
    # Function compares the labels to ground truth labels
    # Returns the error between the values
    # labels - np array of values
    # gt_labels - np array of the same shape and same type of values

    assert isinstance(ret_CB, Retrieved_CaseBase)
    # Check if ground truth is available
    if ret_CB.new_case_y is None:
        return 0, 0

    prct = 0
    assert ret_CB.reused_label.shape == ret_CB.new_case_y.shape # Same dimensionality
    assert ret_CB.reused_label.dtype == ret_CB.new_case_y.dtype # Same data type

    # Check data type
    if ret_CB.reused_label.dtype == np.dtype(np.float) or ret_CB.reused_label.dtype == np.dtype(np.int):
        # If numerical:
        error_ = ret_CB.reused_label - ret_CB.new_case_y
        prct = error_/ret_CB.new_case_y
    else:
        # If categorical:
        error_ = np.sum(np.array(ret_CB.reused_label != ret_CB.new_case_y, dtype=np.int))

    return error_, prct