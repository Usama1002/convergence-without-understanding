"""
Tests for src/metrics/calibration.py and the unbiased CKA estimator.

The motivating property: the biased CKA estimator returns large values for
INDEPENDENT representations (growing with feature dimension d), while the
unbiased estimator stays near 0, and null calibration gates chance-level
similarity to 0 without touching genuine alignment.
"""

import numpy as np
import pytest

from src.metrics.calibration import (
    calibrate,
    calibrated_cka,
    permutation_null,
    spectrum_matched_view,
)
from src.metrics.cka import linear_cka, unbiased_cka


def correlated_pair(n=100, d=64, noise=0.5, seed=0):
    """Two views of a shared latent signal (genuinely aligned)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 16))
    X = z @ rng.standard_normal((16, d)) + noise * rng.standard_normal((n, d))
    Y = z @ rng.standard_normal((16, d)) + noise * rng.standard_normal((n, d))
    return X, Y


def independent_pair(n=100, d=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.standard_normal((n, d))


# ---------------------------------------------------------------------------
# TestUnbiasedCKA
# ---------------------------------------------------------------------------


class TestUnbiasedCKA:

    def test_identical_matrices_return_one(self):
        X, _ = independent_pair()
        assert abs(unbiased_cka(X, X) - 1.0) < 1e-6

    def test_near_zero_on_independent_data(self):
        X, Y = independent_pair(n=100, d=512)
        assert abs(unbiased_cka(X, Y)) < 0.1

    def test_dimension_invariance_of_null(self):
        # The reason this estimator exists: biased CKA on independent data
        # GROWS with d, the unbiased one stays ~0 at every width.
        for d in (128, 512, 2048):
            X, Y = independent_pair(n=100, d=d, seed=d)
            assert abs(unbiased_cka(X, Y)) < 0.1, f"d={d}"
            assert linear_cka(X, Y) > 0.5, f"d={d} (biased should be inflated)"

    def test_detects_genuine_alignment(self):
        X, Y = correlated_pair()
        assert unbiased_cka(X, Y) > 0.5

    def test_rotation_invariance(self):
        rng = np.random.default_rng(1)
        X, _ = independent_pair(n=80, d=32)
        Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
        assert abs(unbiased_cka(X, X @ Q) - 1.0) < 1e-6

    def test_requires_four_samples(self):
        X = np.eye(3)
        with pytest.raises(ValueError):
            unbiased_cka(X, X)


# ---------------------------------------------------------------------------
# TestCalibrate (permutation null)
# ---------------------------------------------------------------------------


class TestCalibratePermutation:

    def test_independent_data_gates_to_zero(self):
        X, Y = independent_pair()
        result = calibrate(X, Y, unbiased_cka, n_permutations=100, seed=0)
        assert result.gated == 0.0
        assert result.p_value > 0.05

    def test_biased_cka_on_noise_also_gates_to_zero(self):
        # Even the biased estimator's inflated noise value is killed by its
        # own permutation null: raw is high, but so is tau.
        X, Y = independent_pair(n=100, d=512)
        result = calibrate(X, Y, linear_cka, n_permutations=100, seed=0)
        assert result.raw > 0.5
        assert result.gated < 0.05

    def test_genuine_alignment_survives(self):
        X, Y = correlated_pair()
        result = calibrate(X, Y, unbiased_cka, n_permutations=100, seed=0)
        assert result.gated > 0.3
        assert result.p_value < 0.05

    def test_gated_within_unit_interval(self):
        X, Y = correlated_pair()
        result = calibrate(X, Y, unbiased_cka, n_permutations=50, seed=0)
        assert 0.0 <= result.gated <= 1.0

    def test_null_samples_count(self):
        X, Y = independent_pair(n=50, d=16)
        null = permutation_null(X, Y, unbiased_cka, n_permutations=37, seed=0)
        assert len(null) == 37


# ---------------------------------------------------------------------------
# TestCalibrateSpectrum (spectrum-matched null)
# ---------------------------------------------------------------------------


class TestCalibrateSpectrum:

    def test_view_preserves_singular_values(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((60, 40)) @ np.diag(np.linspace(2, 0.1, 40))
        view = spectrum_matched_view(Y, rng)
        s_orig = np.linalg.svd(Y, compute_uv=False)
        s_view = np.linalg.svd(view, compute_uv=False)
        assert np.allclose(s_orig, s_view, rtol=1e-8)

    def test_view_destroys_alignment(self):
        rng = np.random.default_rng(0)
        X, Y = correlated_pair()
        view = spectrum_matched_view(Y, rng)
        assert unbiased_cka(X, Y) > 0.5
        assert abs(unbiased_cka(X, view)) < 0.1

    def test_genuine_alignment_survives_spectrum_null(self):
        # The headline check: alignment that is NOT just shared anisotropy
        # beats surrogates that carry Y's exact spectrum.
        X, Y = correlated_pair(n=80, d=48)
        result = calibrate(
            X, Y, unbiased_cka, null="spectrum", n_permutations=60, seed=0
        )
        assert result.gated > 0.3
        assert result.p_value < 0.05

    def test_unknown_null_raises(self):
        X, Y = independent_pair(n=50, d=16)
        with pytest.raises(ValueError):
            calibrate(X, Y, unbiased_cka, null="bogus")


# ---------------------------------------------------------------------------
# TestCalibratedCKAFastPath (Gram-precomputed == generic feature-level)
# ---------------------------------------------------------------------------


class TestCalibratedCKAFastPath:

    def test_raw_matches_feature_level(self):
        X, Y = correlated_pair(n=70, d=48)
        fast = calibrated_cka(X, Y, estimator="unbiased", n_permutations=10, seed=0)
        assert fast.raw == pytest.approx(unbiased_cka(X, Y), abs=1e-10)
        fast_lin = calibrated_cka(X, Y, estimator="linear", n_permutations=10, seed=0)
        assert fast_lin.raw == pytest.approx(linear_cka(X, Y), abs=1e-8)

    def test_permutation_null_identical_to_generic(self):
        # Same seed -> same permutation sequence -> the Gram-side null must
        # equal the feature-side null draw for draw, hence identical gating.
        X, Y = correlated_pair(n=60, d=32, seed=3)
        fast = calibrated_cka(X, Y, estimator="unbiased", n_permutations=50, seed=5)
        slow = calibrate(X, Y, unbiased_cka, n_permutations=50, seed=5)
        assert fast.tau == pytest.approx(slow.tau, abs=1e-8)
        assert fast.gated == pytest.approx(slow.gated, abs=1e-8)
        assert fast.p_value == pytest.approx(slow.p_value, abs=1e-12)
        assert fast.null_mean == pytest.approx(slow.null_mean, abs=1e-8)

    def test_spectrum_null_gram_identity(self):
        # The surrogate's Gram eigenvalues equal the uncentered Gram's
        # eigenvalues (Y'Y'^T = Qa diag(s^2) Qa^T), so the Gram-side draw
        # must produce the same null LEVEL as the feature-side draw.
        X, Y = correlated_pair(n=60, d=40, seed=4)
        fast = calibrated_cka(
            X, Y, estimator="unbiased", null="spectrum", n_permutations=40, seed=0
        )
        slow = calibrate(
            X, Y, unbiased_cka, null="spectrum", n_permutations=40, seed=0
        )
        assert abs(fast.null_mean - slow.null_mean) < 0.05
        assert fast.gated == pytest.approx(slow.gated, abs=0.05)

    def test_genuine_alignment_survives_both_nulls(self):
        X, Y = correlated_pair(n=80, d=48)
        for null in ("permutation", "spectrum"):
            r = calibrated_cka(X, Y, null=null, n_permutations=50, seed=0)
            assert r.gated > 0.3, null

    def test_independent_data_gates_to_zero_both_estimators(self):
        X, Y = independent_pair(n=80, d=256)
        for estimator in ("unbiased", "linear"):
            r = calibrated_cka(X, Y, estimator=estimator, n_permutations=50, seed=0)
            assert r.gated < 0.05, estimator


# ---------------------------------------------------------------------------
# Strict fast-vs-naive equivalence (linear estimator + exact spectrum identity)
# ---------------------------------------------------------------------------


class TestFastPathStrictEquivalence:

    def test_linear_permutation_null_identical_to_generic(self):
        # Strict check of the "centering commutes with symmetric permutation"
        # claim: same seed -> same permutations -> the centered-Gram null must
        # equal the feature-level linear_cka null draw for draw.
        X, Y = correlated_pair(n=60, d=32, seed=8)
        fast = calibrated_cka(X, Y, estimator="linear", n_permutations=50, seed=9)
        slow = calibrate(X, Y, linear_cka, n_permutations=50, seed=9)
        assert fast.raw == pytest.approx(slow.raw, abs=1e-8)
        assert fast.tau == pytest.approx(slow.tau, abs=1e-8)
        assert fast.gated == pytest.approx(slow.gated, abs=1e-8)
        assert fast.p_value == pytest.approx(slow.p_value, abs=1e-12)
        assert fast.null_mean == pytest.approx(slow.null_mean, abs=1e-8)

    def test_spectrum_surrogate_gram_identity_exact(self):
        # The algebra behind the fast spectrum null, checked exactly:
        # Y' = Qa diag(s) Qb^T  =>  Y'Y'^T = Qa diag(s^2) Qa^T, and
        # s^2 = eigvals(Y Y^T). So the fast path's Q diag(lam) Q^T draw is
        # the surrogate's Gram with Q := Qa, not an approximation.
        rng = np.random.default_rng(0)
        n, d = 40, 24
        Y = rng.standard_normal((n, d)) @ np.diag(np.linspace(2.0, 0.1, d))
        s = np.linalg.svd(Y, compute_uv=False)
        r = min(n, d)
        Qa, _ = np.linalg.qr(rng.standard_normal((n, r)))
        Qb, _ = np.linalg.qr(rng.standard_normal((d, r)))
        view = (Qa * s) @ Qb.T
        gram_view = view @ view.T
        gram_fast = (Qa * s**2) @ Qa.T
        assert np.allclose(gram_view, gram_fast, atol=1e-10)
        lam = np.sort(np.linalg.eigvalsh(Y @ Y.T))[::-1][:r]
        assert np.allclose(lam, s**2, rtol=1e-9)

    def test_linear_spectrum_null_level_matches_generic(self):
        X, Y = correlated_pair(n=60, d=40, seed=10)
        fast = calibrated_cka(
            X, Y, estimator="linear", null="spectrum", n_permutations=40, seed=0
        )
        slow = calibrate(
            X, Y, linear_cka, null="spectrum", n_permutations=40, seed=0
        )
        assert abs(fast.null_mean - slow.null_mean) < 0.05
        assert fast.gated == pytest.approx(slow.gated, abs=0.05)
