"""
Procrustes alignment distance for comparing representation geometries.

Uses scipy's orthogonal Procrustes to optimally align two matrices
(allowing rotation, reflection, and scaling) and returns the residual
disparity as a dissimilarity measure.
"""

import numpy as np
from scipy.spatial import procrustes


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Procrustes distance between representation matrices X and Y.

    Both matrices are standardised (zero-mean, unit Frobenius norm) by
    scipy.spatial.procrustes before optimal orthogonal alignment.  The
    returned disparity value lies in [0, 1], where 0 means perfectly
    alignable representations and larger values indicate greater structural
    differences.

    If the two matrices have different numbers of features (columns), the
    smaller one is zero-padded to match the larger.

    Parameters
    ----------
    X : np.ndarray, shape (n, d1)
        First representation matrix with n samples.
    Y : np.ndarray, shape (n, d2)
        Second representation matrix with n samples.

    Returns
    -------
    float
        Procrustes disparity in [0, 1].
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(
            f"X and Y must have the same number of samples; "
            f"got {n} and {Y.shape[0]}."
        )

    d1, d2 = X.shape[1], Y.shape[1]
    if d1 < d2:
        X = np.pad(X, ((0, 0), (0, d2 - d1)))
    elif d2 < d1:
        Y = np.pad(Y, ((0, 0), (0, d1 - d2)))

    _, _, disparity = procrustes(X, Y)
    return float(disparity)
