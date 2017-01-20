##@package Stage4_Retain
# Retention stage of the CBR cycle

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona

import numpy as np
from Stage0_CaseBase import Retrieved_CaseBase

## The function checks if the multi-dimensional point is within the bounds of the array. It is considered to be within
# bounds if for each dimension, the point's values are between the minimum and maximum of the array.
#  @param array Multidimensional array, where rows are instances and columns are dimensions
#  @param point Point to be tested, having the same dimensionality as array
#  @retval Logical Boolean of whether the point is contained by the array
def contains_point(array, point):
    # Find minimum nad maximum
    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)

    # Check if point is within bounds
    l1 = point <= maxs
    l2 = point >= mins
    l = l1 * l2  # Bitwise AND
    return l.all()

## The main function for performing retention for CBR cycle
#  @param ret_CB Retrieved_CaseBase
#  @param strategy Retention strategy ["Always", "Never", "Regular"]
# strategy="Always" - always retains new cases
# strategy="Never" - Never retains new cases
# strategy="Regular" - keeps new cases base on if it is treated as exceptional or non-redundant
#
# Exceptional case has standard deviation of retrieved labels > std_thresh * mean_value
#
# Redundant cases are included within the multidimensional ellipse, bounded by points of retrieved cases
# in the feature space
#    @retval Logical Boolean whether the case was retained or not
#  @retval prct Ratio of error/'true value'
def CBR_Retain(ret_CB, strategy="Regular", std_thresh=0.1):

    assert isinstance(ret_CB, Retrieved_CaseBase)

    if strategy == "Always":
        ret_CB.add_new_case()
        return True

    elif strategy == "Never":
        return False

    elif strategy == "Regular":

        # Check if exceptional
        stddev = np.std(ret_CB.ret_labels)
        ret_mean = np.average(ret_CB.ret_labels)
        # print("S4_Retain\tstdDev = ", stddev)

        if stddev > std_thresh * ret_mean:
            # print "Exceptional"
            ret_CB.add_new_case()
            return True

        # If not exceptional, check if not redundant
        redundant = ~contains_point(ret_CB.RC_all_num, ret_CB.NC_all_num)

        if not redundant: # KEEP
            # print "Not redundant"
            ret_CB.add_new_case()
            return True
        # print "Not exceptional, redundant"
        return False

    else:
        raise("No such retention strategy")