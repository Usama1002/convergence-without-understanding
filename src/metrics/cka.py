"""
Centered Kernel Alignment (CKA) similarity metrics.

Provides linear CKA, uncentered CKA, and RBF kernel CKA for comparing
neural network representations across models or layers.
"""

import numpy as np


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix using the centering matrix H = I - 1/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) for centered Gram matrices."""
    n = K.shape[0]
    return np.sum(K * L) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA between representation matrices X and Y.

    Parameters
    ----------
    X : np.ndarray, shape (n, d1)
        First representation matrix with n samples and d1 features.
    Y : np.ndarray, shape (n, d2)
        Second representation matrix with n samples and d2 features.

    Returns
    -------
    float
        Linear CKA value in [0, 1]. 1.0 means identical representations
        (up to linear transformation); 0.0 means unrelated.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    K = X @ X.T
    L = Y @ Y.T

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    hsic_kl = _hsic(Kc, Lc)
    norm_k = np.sqrt(_hsic(Kc, Kc))
    norm_l = np.sqrt(_hsic(Lc, Lc))

    if norm_k < 1e-10 or norm_l < 1e-10:
        return 0.0

    return float(hsic_kl / (norm_k * norm_l))


def uncentered_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute uncentered Linear CKA between representation matrices X and Y.

    Same as linear_cka but without mean-centering the feature matrices.
    Useful as an ablation to assess the effect of centering.

    Parameters
    ----------
    X : np.ndarray, shape (n, d1)
        First representation matrix.
    Y : np.ndarray, shape (n, d2)
        Second representation matrix.

    Returns
    -------
    float
        Uncentered CKA value.
    """
    K = X @ X.T
    L = Y @ Y.T

    n = K.shape[0]
    hsic_kl = np.sum(K * L) / (n ** 2)
    norm_k = np.sqrt(np.sum(K * K) / (n ** 2))
    norm_l = np.sqrt(np.sum(L * L) / (n ** 2))

    if norm_k < 1e-10 or norm_l < 1e-10:
        return 0.0

    return float(hsic_kl / (norm_k * norm_l))


def kernel_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    Compute RBF kernel CKA between representation matrices X and Y.

    Uses the median heuristic to select sigma if not provided.
    Kernel matrices are centered with H = I - 1/n before computing HSIC.

    Parameters
    ----------
    X : np.ndarray, shape (n, d1)
        First representation matrix.
    Y : np.ndarray, shape (n, d2)
        Second representation matrix.
    sigma : float or None
        RBF bandwidth. If None, the median of pairwise distances is used.

    Returns
    -------
    float
        Kernel CKA value.
    """
    def _rbf_kernel(Z: np.ndarray, bw: float) -> np.ndarray:
        sq_dists = np.sum(Z ** 2, axis=1, keepdims=True) \
                   + np.sum(Z ** 2, axis=1) \
                   - 2 * (Z @ Z.T)
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-sq_dists / (2 * bw ** 2))

    def _median_bandwidth(Z: np.ndarray) -> float:
        sq_dists = np.sum(Z ** 2, axis=1, keepdims=True) \
                   + np.sum(Z ** 2, axis=1) \
                   - 2 * (Z @ Z.T)
        sq_dists = np.maximum(sq_dists, 0.0)
        n = Z.shape[0]
        # upper triangle (excluding diagonal)
        upper = sq_dists[np.triu_indices(n, k=1)]
        median_sq = np.median(upper)
        return float(np.sqrt(median_sq)) if median_sq > 0 else 1.0

    sigma_x = sigma if sigma is not None else _median_bandwidth(X)
    sigma_y = sigma if sigma is not None else _median_bandwidth(Y)

    K = _rbf_kernel(X, sigma_x)
    L = _rbf_kernel(Y, sigma_y)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    hsic_kl = _hsic(Kc, Lc)
    norm_k = np.sqrt(_hsic(Kc, Kc))
    norm_l = np.sqrt(_hsic(Lc, Lc))

    if norm_k < 1e-10 or norm_l < 1e-10:
        return 0.0

    return float(hsic_kl / (norm_k * norm_l))
