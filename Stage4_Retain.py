import numpy as np
from Stage0_CaseBase import Retrieved_CaseBase
import numpy.linalg as la


def contains_point(array, point):

    def MinVolEllipse(points, el_tol=0.001):
        """
        AUTHOR: unutbu, http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python

        Finds the minimum volume ellipse.
        Return A, c where the equation for the ellipse given in "center form" is
        (x-c).T * A * (x-c) = 1
        """

        points = np.asmatrix(points)
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        err = el_tol + 1.0
        u = np.ones(N) / N
        while err > el_tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
        c = u * points
        A = la.inv(points.T * np.diag(u) * points - c.T * c) / d
        return np.asarray(A), np.squeeze(np.asarray(c))

    # Provide tolerance since the ellipse is not 100% accurate
    tolerance = 0.1

    # Calculate the elipse
    A, c = MinVolEllipse(array, el_tol=0.001)
    x_c = point - c
    d = np.dot(x_c, A * x_c)

    # Check if if all dimensions are within the ellipse
    lgg = d[0] < (1 + tolerance)
    return np.all(lgg)

def CBR_Retain(ret_CB, std_thresh):
    # Function retains new case or discards it based on whether it is treated as exceptional or redundant

    assert isinstance(ret_CB, Retrieved_CaseBase)

    # Exceptional case has standard deviation of labels above the std_thresh
    #
    # Redundant cases are included within the multidimensional ellipse, bounded by points of retrieved cases
    # in the feature space


    # Check if exceptional
    stddev = np.std(ret_CB.ret_labels)
    print("S4_Retain\tstdDev = ", stddev)

    if stddev > std_thresh:
        ret_CB.add_new_case()
        return

    # If not exceptional, check if not redundant
    redundant = contains_point(ret_CB.RC_all_num, ret_CB.NC_all_num)

    if not redundant: # KEEP
        ret_CB.add_new_case()

    return

