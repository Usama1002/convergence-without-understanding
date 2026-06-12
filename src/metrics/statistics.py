"""Statistical utility functions for bootstrap CIs, permutation tests, FDR, and effect sizes."""

import numpy as np
from scipy import stats


def bootstrap_ci(data, stat_fn=np.mean, n_bootstrap=1000, ci=0.95, seed=42):
    """BCa bootstrap confidence interval.

    Parameters
    ----------
    data : array-like
        1-D data array.
    stat_fn : callable
        Statistic function (default: np.mean).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (0 < ci < 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)

    observed = stat_fn(data)

    # Bootstrap distribution
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # BCa: bias-correction and acceleration. z0 is infinite when all
    # bootstrap stats fall on one side of the observed value (min/max-like
    # or tie-heavy statistics); fall back to the plain percentile CI.
    z0 = stats.norm.ppf(np.mean(boot_stats < observed))
    if not np.isfinite(z0):
        alpha = (1.0 - ci) / 2.0
        return (
            np.percentile(boot_stats, alpha * 100),
            np.percentile(boot_stats, (1.0 - alpha) * 100),
        )

    # Acceleration factor via jackknife
    jack_stats = np.array([stat_fn(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    alpha = (1.0 - ci) / 2.0
    z_alpha_lo = stats.norm.ppf(alpha)
    z_alpha_hi = stats.norm.ppf(1.0 - alpha)

    # Adjusted percentiles
    def adjusted_alpha(z_a):
        return stats.norm.cdf(z0 + (z0 + z_a) / (1.0 - a * (z0 + z_a)))

    p_lo = adjusted_alpha(z_alpha_lo)
    p_hi = adjusted_alpha(z_alpha_hi)

    lower = np.percentile(boot_stats, p_lo * 100)
    upper = np.percentile(boot_stats, p_hi * 100)

    return (lower, upper)


def bootstrap_ci_two_sample(
    X, Y, metric_fn, n_bootstrap=1000, ci=0.95, seed=42, subsample=0.8
):
    """Subsampling CI for metric(X, Y) with paired resampling.

    Resamples WITHOUT replacement (m-out-of-n subsampling, m = subsample * n).
    Classic with-replacement bootstrap is invalid for Gram-based metrics such
    as CKA: duplicated rows make the Gram matrix artificially low-rank, which
    inflates every resampled value, biasing the interval upward and making it
    too narrow.

    The interval is recentered on the observed full-sample statistic: for
    metrics whose bias depends on n (biased CKA), the size-m subsample
    distribution is shifted relative to the size-n value. The residual
    sqrt(n/m) width inflation (~1.12 at the default 0.8) is conservative.

    Parameters
    ----------
    X : array-like, shape (n, ...)
        First sample (rows are observations).
    Y : array-like, shape (n, ...)
        Second sample (same size as X).
    metric_fn : callable
        Function metric_fn(X, Y) -> float.
    n_bootstrap : int
        Number of resamples.
    ci : float
        Confidence level.
    seed : int
        Random seed.
    subsample : float
        Fraction of rows drawn (without replacement) per resample.

    Returns
    -------
    (observed, lower, upper) : tuple of float
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = X.shape[0]

    observed = metric_fn(X, Y)

    m = max(4, int(np.ceil(subsample * n)))
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=min(m, n), replace=False)
        boot_stats.append(metric_fn(X[idx], Y[idx]))

    boot_stats = np.array(boot_stats)
    alpha = (1.0 - ci) / 2.0
    center = np.median(boot_stats)
    lower = observed + (np.percentile(boot_stats, alpha * 100) - center)
    upper = observed + (np.percentile(boot_stats, (1.0 - alpha) * 100) - center)

    return (observed, lower, upper)


def permutation_test(X, Y, metric_fn, n_permutations=1000, seed=42):
    """Permutation test for metric_fn(X, Y).

    Shuffles rows of X, recomputes metric. Returns
    p-value = (count of permuted >= observed + 1) / (n_permutations + 1).

    Parameters
    ----------
    X : array-like, shape (n, ...)
        First sample.
    Y : array-like, shape (n, ...)
        Second sample.
    metric_fn : callable
        Function metric_fn(X, Y) -> float.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    p_value : float in [0, 1]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)

    observed = metric_fn(X, Y)

    count = 0
    for _ in range(n_permutations):
        X_perm = rng.permutation(X)
        perm_stat = metric_fn(X_perm, Y)
        if perm_stat >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return float(p_value)


def fdr_correction(pvalues, q=0.05):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : array-like
        Array of p-values.
    q : float
        False discovery rate threshold.

    Returns
    -------
    corrected_pvalues : np.ndarray
        BH-adjusted p-values.
    rejected : np.ndarray of bool
        Boolean array indicating rejected null hypotheses.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    # BH adjusted p-values
    corrected = np.minimum(1.0, pvalues * n / ranks)

    # Enforce monotonicity from right to left (cumulative minimum in reverse)
    corrected_ordered = corrected[order]
    for i in range(n - 2, -1, -1):
        corrected_ordered[i] = min(corrected_ordered[i], corrected_ordered[i + 1])
    corrected[order] = corrected_ordered

    rejected = corrected <= q
    return corrected, rejected


def cohens_d(group_a, group_b):
    """Cohen's d effect size with pooled standard deviation.

    Positive if group_b mean > group_a mean.

    Parameters
    ----------
    group_a : array-like
    group_b : array-like

    Returns
    -------
    d : float
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def cliffs_delta(group_a, group_b):
    """Cliff's delta non-parametric effect size.

    Returns a value in [-1, 1] where positive means group_b tends to
    have larger values than group_a.

    Parameters
    ----------
    group_a : array-like
    group_b : array-like

    Returns
    -------
    delta : float in [-1, 1]
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    # Vectorised comparison
    dominance = np.sign(b[:, None] - a[None, :])  # shape (n_b, n_a)
    delta = float(np.mean(dominance))
    return delta
