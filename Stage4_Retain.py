import numpy as np

def CBR_Retain(case_base, retrieved_labels, reused_case, std_thresh):
    # Function test if standard deviation of the retrieved labels are within the threshold std_thresh

    stddev = np.std(retrieved_labels)
    print(stddev)
    if stddev < std_thresh:
        case_base = np.append(case_base,reused_case,0)

    return case_base

if __name__ == "__main__":
    print("Demo run")
    retrieved_labels = np.array([12000,11000,12500,10000,10100])
    case_base = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
    reused_case = np.array([[7,8,9]])
    print("old_case_base = " + str(case_base))

    new_case_base = CBR_Retain(case_base, retrieved_labels,reused_case,2000)
    print("new_case_base = " + str(new_case_base))
