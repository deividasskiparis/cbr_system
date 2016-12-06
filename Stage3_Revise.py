import numpy as np

def CBR_Revise(labels, gt_labels, eval_within = 1000):
    # Function compares the labels to ground truth labels. If the labels are numerical, eval_within threshold is
    # used to have equality intervals. If eval_within = 5, 45 == 45+-5
    # Returns number of mismatches
    # labels - np array of values
    # gt_labels - np array of the same shape and same type of values

    assert labels.shape == gt_labels.shape
    assert labels.dtype == gt_labels.dtype

    if lbls.dtype ==np.dtype(np.float) or lbls.dtype ==np.dtype(np.int):
        diff = np.abs(labels-gt_labels)
        error_count = np.sum(diff > eval_within)
    else:
        error_count = np.sum(np.array(labels != gt_labels, dtype=np.bool))

    return error_count

if __name__ == "__main__":
    print("Demo run\nComparing arrays:")
    lbls = np.array([5000,1000,20000,150000,15200.5])
    lbls_gt = np.array([5100,1000,25000,150001,15200.4])

    print ("lbls = " + str(lbls))
    print("lbls_gt = " + str(lbls_gt))

    no_errors = CBR_Revise(lbls, lbls_gt, 99)

    print("Number of mismatches = " + str(no_errors))