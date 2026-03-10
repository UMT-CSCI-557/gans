import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def empirical_wasserstein_1(X, Y, p=2):
    """
    Empirical Wasserstein-1 distance between two equally weighted point clouds.

    Parameters
    ----------
    X : (n, d) array
        First sample set.
    Y : (n, d) array
        Second sample set.
    p : int or float
        Ground metric exponent. p=2 means Euclidean cost is used.

    Returns
    -------
    w1 : float
        Mean optimal transport cost.
    """
    X = np.asarray(X.cpu().detach(), dtype=float)
    Y = np.asarray(Y.cpu().detach(), dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays of shape (n, d)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same ambient dimension")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("For this simple implementation, X and Y must have the same number of samples")

    # Pairwise ground cost matrix
    C = cdist(X, Y, metric="minkowski", p=p)

    # Solve optimal bipartite matching
    row_ind, col_ind = linear_sum_assignment(C)

    # Mean transport cost
    return C[row_ind, col_ind].mean()
