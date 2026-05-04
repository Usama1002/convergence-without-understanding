"""
Mutual Nearest Neighbors (MNN) similarity metric.

Measures representational similarity by comparing k-nearest-neighbor
graphs in two embedding spaces and computing the mean fraction of
neighbors that are shared.
"""

import numpy as np


def mutual_nearest_neighbors(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Compute the mean fraction of mutual nearest neighbors between X and Y.

    For each sample i, the k nearest neighbors (by Euclidean distance,
    excluding self) are identified in X-space and Y-space independently.
    The overlap fraction for sample i is |NN_X(i) ∩ NN_Y(i)| / k.
    The return value is the mean overlap fraction across all n samples,
    lying in [0, 1].  A value of 1 means the neighborhood structure is
    identical in both spaces; 0 means no neighbors are shared.

    Parameters
    ----------
    X : np.ndarray, shape (n, d1)
        First representation matrix with n samples and d1 features.
    Y : np.ndarray, shape (n, d2)
        Second representation matrix with n samples and d2 features.
    k : int
        Number of nearest neighbors to consider (default 5).

    Returns
    -------
    float
        Mean fraction of shared nearest neighbors in [0, 1].
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(
            f"X and Y must have the same number of samples; "
            f"got {n} and {Y.shape[0]}."
        )

    k = min(k, n - 1)

    def _knn_indices(Z: np.ndarray, num_neighbors: int) -> np.ndarray:
        """Return (n, num_neighbors) array of nearest-neighbor indices."""
        sq_dists = (
            np.sum(Z ** 2, axis=1, keepdims=True)
            + np.sum(Z ** 2, axis=1)
            - 2 * (Z @ Z.T)
        )
        np.fill_diagonal(sq_dists, np.inf)  # exclude self
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.argsort(sq_dists, axis=1)[:, :num_neighbors]

    nn_x = _knn_indices(X, k)  # shape (n, k)
    nn_y = _knn_indices(Y, k)  # shape (n, k)

    overlaps = np.array([
        len(np.intersect1d(nn_x[i], nn_y[i])) / k
        for i in range(n)
    ])

    return float(overlaps.mean())
