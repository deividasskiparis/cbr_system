import numpy as np
from Stage0_CaseBase import Retrieved_CaseBase
import numpy.linalg as la


def contains_point(array, point):
    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)

    l1 = point <= maxs
    l2 = point >= mins
    l = l1 * l2
    return l.all()


def CBR_Retain(ret_CB, std_thresh):
    # Function retains new case or discards it based on whether it is treated as exceptional or redundant
    #
    # Exceptional case has standard deviation of labels above the std_thresh
    #
    # Redundant cases are included within the multidimensional ellipse, bounded by points of retrieved cases
    # in the feature space

    assert isinstance(ret_CB, Retrieved_CaseBase)

    # Check if exceptional
    stddev = np.std(ret_CB.ret_labels)
    # print("S4_Retain\tstdDev = ", stddev)

    if stddev > std_thresh:
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

