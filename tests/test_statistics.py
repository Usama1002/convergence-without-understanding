"""Tests for src/metrics/statistics.py"""

import numpy as np
import pytest

from src.metrics.statistics import (
    bootstrap_ci,
    bootstrap_ci_two_sample,
    cliffs_delta,
    cohens_d,
    fdr_correction,
    permutation_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def normal_data(rng):
    """100 samples from N(5, 1)."""
    return rng.normal(loc=5.0, scale=1.0, size=100)


@pytest.fixture
def two_groups(rng):
    """Two groups: a ~ N(0,1), b ~ N(1,1). b > a on average."""
    a = rng.normal(0.0, 1.0, 80)
    b = rng.normal(1.0, 1.0, 80)
    return a, b


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_tuple_of_two(self, normal_data):
        result = bootstrap_ci(normal_data)
        assert len(result) == 2

    def test_lo_less_than_hi(self, normal_data):
        lo, hi = bootstrap_ci(normal_data)
        assert lo < hi

    def test_mean_inside_ci(self, normal_data):
        lo, hi = bootstrap_ci(normal_data)
        mean = np.mean(normal_data)
        assert lo < mean < hi

    def test_custom_stat_fn(self, normal_data):
        lo, hi = bootstrap_ci(normal_data, stat_fn=np.median)
        median = np.median(normal_data)
        assert lo < median < hi

    def test_narrower_ci_at_lower_level(self, normal_data):
        lo_90, hi_90 = bootstrap_ci(normal_data, ci=0.90)
        lo_99, hi_99 = bootstrap_ci(normal_data, ci=0.99)
        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99
        assert width_90 < width_99

    def test_reproducible_with_seed(self, normal_data):
        r1 = bootstrap_ci(normal_data, seed=7)
        r2 = bootstrap_ci(normal_data, seed=7)
        assert r1 == r2

    def test_different_seeds_may_differ(self, normal_data):
        r1 = bootstrap_ci(normal_data, seed=1)
        r2 = bootstrap_ci(normal_data, seed=2)
        # Very unlikely to be identical with different seeds
        assert r1 != r2


# ---------------------------------------------------------------------------
# bootstrap_ci_two_sample
# ---------------------------------------------------------------------------

class TestBootstrapCITwoSample:
    def test_returns_three_values(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 3))
        Y = rng.normal(0, 1, (50, 3))
        result = bootstrap_ci_two_sample(X, Y, metric_fn=lambda x, y: np.mean(x - y))
        assert len(result) == 3

    def test_lo_less_than_hi(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 3))
        Y = rng.normal(1, 1, (50, 3))
        observed, lo, hi = bootstrap_ci_two_sample(
            X, Y, metric_fn=lambda x, y: float(np.mean(y - x))
        )
        assert lo < hi

    def test_observed_value_correct(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (40, 2))
        Y = rng.normal(2, 1, (40, 2))
        metric_fn = lambda x, y: float(np.mean(y - x))
        expected_obs = metric_fn(X, Y)
        observed, lo, hi = bootstrap_ci_two_sample(X, Y, metric_fn=metric_fn)
        assert np.isclose(observed, expected_obs)


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def _simple_metric(self, X, Y):
        """Mean of X minus mean of Y (scalar)."""
        return float(np.mean(X) - np.mean(Y))

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 1))
        Y = rng.normal(0, 1, (30, 1))
        p = permutation_test(X, Y, self._simple_metric)
        assert isinstance(p, float)

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 1))
        Y = rng.normal(0, 1, (30, 1))
        p = permutation_test(X, Y, self._simple_metric)
        assert 0.0 <= p <= 1.0

    def test_null_hypothesis_not_rejected(self):
        """Identical distributions -> large p-value on average."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 1))
        Y = rng.normal(0, 1, (50, 1))
        p = permutation_test(X, Y, self._simple_metric, n_permutations=500)
        # Should not be extremely small
        assert p > 0.01

    def test_pvalue_formula(self):
        """p = (count + 1) / (n_permutations + 1) -> minimum is 1/(n+1)."""
        rng = np.random.default_rng(0)
        X = rng.normal(100, 1, (20, 1))  # clearly different from Y
        Y = rng.normal(0, 1, (20, 1))
        n_perm = 99
        p = permutation_test(X, Y, self._simple_metric, n_permutations=n_perm)
        min_possible = 1.0 / (n_perm + 1)
        assert p >= min_possible


# ---------------------------------------------------------------------------
# fdr_correction
# ---------------------------------------------------------------------------

class TestFDRCorrection:
    def test_returns_two_arrays(self):
        pvals = [0.01, 0.04, 0.03, 0.20, 0.50]
        corrected, rejected = fdr_correction(pvals)
        assert corrected is not None and rejected is not None

    def test_correct_length(self):
        pvals = [0.01, 0.04, 0.03, 0.20, 0.50]
        corrected, rejected = fdr_correction(pvals)
        assert len(corrected) == len(pvals)
        assert len(rejected) == len(pvals)

    def test_rejected_is_bool_array(self):
        pvals = [0.001, 0.01, 0.5, 0.9]
        _, rejected = fdr_correction(pvals)
        assert rejected.dtype == bool

    def test_very_small_pvalues_rejected(self):
        pvals = [0.0001, 0.0002, 0.9, 0.95]
        _, rejected = fdr_correction(pvals, q=0.05)
        assert rejected[0] and rejected[1]

    def test_large_pvalues_not_rejected(self):
        pvals = [0.9, 0.8, 0.7]
        _, rejected = fdr_correction(pvals, q=0.05)
        assert not any(rejected)

    def test_corrected_pvalues_gte_original(self):
        """BH-corrected p-values should be >= original p-values."""
        pvals = np.array([0.01, 0.03, 0.05, 0.10, 0.20])
        corrected, _ = fdr_correction(pvals)
        assert np.all(corrected >= pvals - 1e-10)

    def test_corrected_pvalues_bounded(self):
        pvals = [0.01, 0.02, 0.03]
        corrected, _ = fdr_correction(pvals)
        assert np.all(corrected <= 1.0)
        assert np.all(corrected >= 0.0)


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_positive_when_b_greater(self, two_groups):
        a, b = two_groups
        d = cohens_d(a, b)
        assert d > 0.0

    def test_negative_when_b_less(self, two_groups):
        a, b = two_groups
        d = cohens_d(b, a)
        assert d < 0.0

    def test_zero_identical_groups(self):
        x = np.ones(50)
        assert cohens_d(x, x) == 0.0

    def test_returns_float(self, two_groups):
        a, b = two_groups
        assert isinstance(cohens_d(a, b), float)

    def test_magnitude_reasonable(self):
        """Cohen's d ~ 1.0 for groups separated by 1 std."""
        rng = np.random.default_rng(99)
        a = rng.normal(0.0, 1.0, 500)
        b = rng.normal(1.0, 1.0, 500)
        d = cohens_d(a, b)
        assert 0.7 < d < 1.3


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------

class TestCliffsDelta:
    def test_returns_float(self, two_groups):
        a, b = two_groups
        delta = cliffs_delta(a, b)
        assert isinstance(delta, float)

    def test_in_range(self, two_groups):
        a, b = two_groups
        delta = cliffs_delta(a, b)
        assert -1.0 <= delta <= 1.0

    def test_positive_when_b_greater(self, two_groups):
        a, b = two_groups
        delta = cliffs_delta(a, b)
        assert delta > 0.0

    def test_negative_when_b_less(self, two_groups):
        a, b = two_groups
        delta = cliffs_delta(b, a)
        assert delta < 0.0

    def test_zero_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(x, x) == 0.0

    def test_antisymmetric(self, two_groups):
        """cliffs_delta(a, b) == -cliffs_delta(b, a)."""
        a, b = two_groups
        assert np.isclose(cliffs_delta(a, b), -cliffs_delta(b, a))

    def test_extreme_values(self):
        """All of b > all of a -> delta = 1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert cliffs_delta(a, b) == 1.0

    def test_random_magnitude(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 200)
        b = rng.normal(1, 1, 200)
        delta = cliffs_delta(a, b)
        assert 0.1 < delta < 0.9
