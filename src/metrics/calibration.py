"""
Null calibration for representational similarity metrics.

A raw similarity value is only interpretable relative to what the metric
returns when there is NO real alignment. This module estimates that null
distribution and reports a calibrated ("gated") score:

    gated = max(0, (raw - tau) / (s_max - tau))

where tau is the (1 - alpha) quantile of the null. A gated score of 0 means
the observed similarity is indistinguishable from the null; values near the
raw score mean the null is negligible.

Two nulls are provided:

  * permutation_null: permutes the rows (samples) of Y. Destroys the
    per-sample correspondence between X and Y while keeping each
    representation's geometry intact. This is the standard test for
    "is there sample-level alignment at all".

  * spectrum_matched_null: replaces Y with a random matrix that has exactly
    Y's singular values but random singular vectors. Destroys all structure
    EXCEPT the spectrum (anisotropy / effective dimensionality). If the
    observed similarity survives this null, it cannot be explained by the
    two representations merely having similar spectra (the width/anisotropy
    confound of biased CKA).

Method follows Groeger et al. 2026 (arXiv:2602.14486). Pure numpy; no new
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class CalibratedScore:
    """Result of calibrating one similarity value against a null."""

    raw: float
    tau: float
    gated: float
    p_value: float
    null_mean: float
    null_std: float
    n_null: int


def permutation_null(
    X: np.ndarray,
    Y: np.ndarray,
    metric_fn: MetricFn,
    n_permutations: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of metric_fn under random row permutations of Y.

    Returns an array of n_permutations null samples.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = Y.shape[0]
    samples = np.empty(n_permutations)
    for i in range(n_permutations):
        samples[i] = metric_fn(X, Y[rng.permutation(n)])
    return samples


def spectrum_matched_view(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random matrix with exactly Y's singular values (random singular vectors).

    Preserves the spectrum (anisotropy, effective rank) while destroying the
    actual per-sample structure.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n, d = Y.shape
    r = min(n, d)
    s = np.linalg.svd(Y, compute_uv=False)
    Qa, _ = np.linalg.qr(rng.standard_normal((n, r)))
    Qb, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return (Qa * s) @ Qb.T


def spectrum_matched_null(
    X: np.ndarray,
    Y: np.ndarray,
    metric_fn: MetricFn,
    n_draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of metric_fn against spectrum-matched surrogates of Y.

    Note: each draw requires a QR decomposition of an (n x r) and a (d x r)
    matrix, so this is noticeably slower than permutation_null for large d.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    samples = np.empty(n_draws)
    for i in range(n_draws):
        samples[i] = metric_fn(X, spectrum_matched_view(Y, rng))
    return samples


def _gate(raw: float, null_samples: np.ndarray, alpha: float, s_max: float) -> CalibratedScore:
    """Turn (raw, null distribution) into a CalibratedScore."""
    # Include the observed value when estimating tau (conservative convention).
    tau = float(np.quantile(np.append(null_samples, raw), 1.0 - alpha))

    if raw <= tau:
        gated = 0.0
    elif s_max <= tau:
        gated = 1.0
    else:
        gated = (raw - tau) / (s_max - tau)

    p_value = (1.0 + float(np.sum(null_samples >= raw))) / (len(null_samples) + 1.0)

    return CalibratedScore(
        raw=float(raw),
        tau=tau,
        gated=float(gated),
        p_value=float(p_value),
        null_mean=float(null_samples.mean()),
        null_std=float(null_samples.std()),
        n_null=len(null_samples),
    )


def calibrate(
    X: np.ndarray,
    Y: np.ndarray,
    metric_fn: MetricFn,
    *,
    null: str = "permutation",
    n_permutations: int = 200,
    alpha: float = 0.05,
    s_max: float = 1.0,
    seed: int = 0,
) -> CalibratedScore:
    """Calibrate metric_fn(X, Y) against a null distribution.

    Generic version: re-evaluates metric_fn from the features for every null
    draw, which costs O(n^2 d) per draw. Fine for occasional checks or
    non-Gram metrics (e.g. mutual_nearest_neighbors); for CKA on full-size
    hidden states use calibrated_cka, which precomputes the Gram matrices
    once and is much faster.

    Parameters
    ----------
    X, Y : np.ndarray, shape (n, d1) and (n, d2)
        Paired representation matrices (same samples, same order).
    metric_fn : callable
        Similarity function metric_fn(X, Y) -> float, e.g. cka.unbiased_cka
        or a lambda wrapping mutual_nearest_neighbors.
    null : str
        'permutation' or 'spectrum' (see module docstring).
    n_permutations : int
        Number of null samples.
    alpha : float
        Significance level; tau is the (1 - alpha) null quantile.
    s_max : float
        The metric's maximum attainable value (1.0 for CKA / MNN overlap).
    seed : int
        Random seed for the null draws.

    Returns
    -------
    CalibratedScore
        raw, tau, gated, p_value (add-one permutation p), null mean/std.
    """
    raw = float(metric_fn(X, Y))
    if null == "permutation":
        null_samples = permutation_null(X, Y, metric_fn, n_permutations, seed)
    elif null == "spectrum":
        null_samples = spectrum_matched_null(X, Y, metric_fn, n_permutations, seed)
    else:
        raise ValueError(f"null must be 'permutation' or 'spectrum', got {null!r}")

    return _gate(raw, null_samples, alpha, s_max)


# ---------------------------------------------------------------------------
# Fast Gram-based calibrated CKA
# ---------------------------------------------------------------------------


def _center_gram(K: np.ndarray) -> np.ndarray:
    row = K.mean(1, keepdims=True)
    col = K.mean(0, keepdims=True)
    return K - row - col + K.mean()


def _hsic_unbiased_gram(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC (Song et al. 2012) from raw (uncentered) Gram matrices."""
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"unbiased HSIC requires n >= 4 samples, got {n}")
    Kt = K.copy()
    Lt = L.copy()
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)
    return float(
        (np.sum(Kt * Lt)
         + Kt.sum() * Lt.sum() / ((n - 1) * (n - 2))
         - 2.0 * (Kt @ Lt).sum() / (n - 2)) / (n * (n - 3))
    )


def _cka_unbiased_gram(Gx: np.ndarray, Gy: np.ndarray) -> float:
    denom = np.sqrt(max(
        _hsic_unbiased_gram(Gx, Gx) * _hsic_unbiased_gram(Gy, Gy), 0.0))
    if denom < 1e-12:
        return 0.0
    return _hsic_unbiased_gram(Gx, Gy) / denom


def _cka_linear_gram(Kx: np.ndarray, Ky: np.ndarray) -> float:
    """Biased linear CKA from CENTERED Gram matrices."""
    denom = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(Kx * Ky) / denom)


def calibrated_cka(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    estimator: str = "unbiased",
    null: str = "permutation",
    n_permutations: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> CalibratedScore:
    """Calibrated CKA with the null computed on precomputed Gram matrices.

    Equivalent to calibrate(X, Y, cka_fn) but far faster: the n x n Gram
    matrices are built once and every null draw is an O(n^2)-O(n^3)
    operation instead of rebuilding O(n^2 d) feature products.

    Null equivalences used (both exact, not approximations):
      * permutation: relabeling samples of Y permutes its Gram symmetrically,
        G' = G[p][:, p].
      * spectrum: the ARH spectrum-matched surrogate Y' = Qa diag(s) Qb^T has
        Gram Y'Y'^T = Qa diag(s^2) Qa^T, and s^2 are the eigenvalues of the
        uncentered Gram - so drawing a random orthogonal Q and forming
        Q diag(lambda) Q^T reproduces the surrogate's Gram exactly.

    estimator: 'unbiased' (Song et al. 2012; recommended) or 'linear'
    (biased; reproduces the paper's metric, calibrated).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"X and Y must have the same number of samples; got {n} and {Y.shape[0]}")
    if estimator not in ("unbiased", "linear"):
        raise ValueError(f"estimator must be 'unbiased' or 'linear', got {estimator!r}")
    if null not in ("permutation", "spectrum"):
        raise ValueError(f"null must be 'permutation' or 'spectrum', got {null!r}")

    Gx = X @ X.T
    Gy = Y @ Y.T
    biased = estimator == "linear"
    # linear_cka centers the features; on the Gram side that is double
    # centering, and centering commutes with symmetric permutation.
    Kx = _center_gram(Gx) if biased else Gx
    Ky = _center_gram(Gy) if biased else Gy
    raw = _cka_linear_gram(Kx, Ky) if biased else _cka_unbiased_gram(Gx, Gy)

    rng = np.random.default_rng(seed)
    null_samples = np.empty(n_permutations)
    if null == "permutation":
        base = Ky if biased else Gy
        for i in range(n_permutations):
            p = rng.permutation(n)
            Gp = base[p][:, p]
            null_samples[i] = (
                _cka_linear_gram(Kx, Gp) if biased else _cka_unbiased_gram(Gx, Gp)
            )
    else:
        # The spectrum comes from the UNCENTERED Gram for both estimators,
        # matching spectrum_matched_view; for the linear estimator the
        # surrogate then also carries the mean component that centering
        # removes from the real data, so its null is slightly harsher and
        # the gated score errs conservative.
        lam = np.clip(np.linalg.eigvalsh(Gy), 0.0, None)
        for i in range(n_permutations):
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            S = (Q * lam) @ Q.T
            null_samples[i] = (
                _cka_linear_gram(Kx, _center_gram(S)) if biased
                else _cka_unbiased_gram(Gx, S)
            )

    return _gate(raw, null_samples, alpha, s_max=1.0)
