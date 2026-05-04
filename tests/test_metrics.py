"""
Tests for similarity metrics: CKA, Procrustes, and MNN.
"""

import numpy as np
import pytest

from src.metrics.cka import linear_cka, uncentered_cka, kernel_cka
from src.metrics.procrustes import procrustes_distance
from src.metrics.mnn import mutual_nearest_neighbors


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_matrix(n: int, d: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = RNG
    return rng.standard_normal((n, d))


def random_rotation(d: int, rng=None) -> np.ndarray:
    """Return a random d x d orthogonal matrix via QR decomposition."""
    if rng is None:
        rng = RNG
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


# ---------------------------------------------------------------------------
# TestLinearCKA
# ---------------------------------------------------------------------------

class TestLinearCKA:

    def test_identical_matrices_return_one(self):
        X = random_matrix(50, 20)
        result = linear_cka(X, X)
        assert abs(result - 1.0) < 1e-6, f"Expected ~1.0, got {result}"

    def test_identical_matrices_return_one_uncentered(self):
        X = random_matrix(50, 20)
        result = uncentered_cka(X, X)
        assert abs(result - 1.0) < 1e-6, f"Expected ~1.0, got {result}"

    def test_random_unrelated_matrices_low_cka(self):
        rng = np.random.default_rng(0)
        X = random_matrix(100, 32, rng)
        Y = random_matrix(100, 32, rng)
        result = linear_cka(X, Y)
        assert result < 0.5, f"Expected low CKA for unrelated matrices, got {result}"

    def test_different_feature_dims(self):
        X = random_matrix(60, 16)
        Y = random_matrix(60, 32)
        result = linear_cka(X, Y)
        assert 0.0 <= result <= 1.0, f"CKA out of range: {result}"

    def test_returns_float(self):
        X = random_matrix(30, 10)
        Y = random_matrix(30, 10)
        assert isinstance(linear_cka(X, Y), float)
        assert isinstance(uncentered_cka(X, Y), float)

    def test_cka_is_symmetric(self):
        X = random_matrix(40, 12)
        Y = random_matrix(40, 8)
        assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-6

    def test_cka_range(self):
        for _ in range(5):
            X = random_matrix(30, 10)
            Y = random_matrix(30, 10)
            val = linear_cka(X, Y)
            assert 0.0 <= val <= 1.0 + 1e-8, f"CKA out of [0,1]: {val}"

    def test_rotated_matrix_high_cka(self):
        """Linear CKA is invariant to orthogonal transformations of X."""
        X = random_matrix(50, 10)
        Q = random_rotation(10)
        X_rot = X @ Q
        result = linear_cka(X, X_rot)
        assert result > 0.99, f"Expected CKA ~1 after rotation, got {result}"

    def test_uncentered_cka_range(self):
        X = random_matrix(40, 15)
        Y = random_matrix(40, 15)
        val = uncentered_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-8, f"Uncentered CKA out of [0,1]: {val}"

    def test_zero_matrix_returns_zero(self):
        X = np.zeros((20, 5))
        Y = random_matrix(20, 5)
        result = linear_cka(X, Y)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestKernelCKA
# ---------------------------------------------------------------------------

class TestKernelCKA:

    def test_identical_matrices_return_one(self):
        X = random_matrix(40, 10)
        result = kernel_cka(X, X)
        assert abs(result - 1.0) < 1e-5, f"Expected ~1.0, got {result}"

    def test_random_matrices_low_cka(self):
        rng = np.random.default_rng(7)
        X = random_matrix(80, 20, rng)
        Y = random_matrix(80, 20, rng)
        result = kernel_cka(X, Y)
        assert result < 0.5, f"Expected low kernel CKA for random matrices, got {result}"

    def test_returns_float(self):
        X = random_matrix(25, 8)
        Y = random_matrix(25, 8)
        assert isinstance(kernel_cka(X, Y), float)

    def test_custom_sigma(self):
        X = random_matrix(30, 5)
        result = kernel_cka(X, X, sigma=1.0)
        assert abs(result - 1.0) < 1e-5

    def test_range(self):
        X = random_matrix(30, 8)
        Y = random_matrix(30, 8)
        val = kernel_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-8


# ---------------------------------------------------------------------------
# TestProcrustes
# ---------------------------------------------------------------------------

class TestProcrustes:

    def test_identical_matrices_return_zero(self):
        X = random_matrix(50, 10)
        result = procrustes_distance(X, X)
        assert result < 1e-6, f"Expected ~0.0, got {result}"

    def test_rotated_matrix_near_zero(self):
        X = random_matrix(50, 8)
        Q = random_rotation(8)
        X_rot = X @ Q
        result = procrustes_distance(X, X_rot)
        assert result < 1e-6, f"Expected ~0.0 after rotation, got {result}"

    def test_random_unrelated_matrices_nonzero(self):
        rng = np.random.default_rng(1)
        X = random_matrix(50, 10, rng)
        Y = random_matrix(50, 10, rng)
        result = procrustes_distance(X, Y)
        assert result > 0.01, f"Expected nonzero disparity, got {result}"

    def test_different_feature_dims(self):
        X = random_matrix(40, 8)
        Y = random_matrix(40, 16)
        result = procrustes_distance(X, Y)
        assert 0.0 <= result <= 1.0 + 1e-8, f"Disparity out of range: {result}"

    def test_returns_float(self):
        X = random_matrix(30, 6)
        Y = random_matrix(30, 6)
        assert isinstance(procrustes_distance(X, Y), float)

    def test_range(self):
        for _ in range(5):
            X = random_matrix(30, 6)
            Y = random_matrix(30, 6)
            val = procrustes_distance(X, Y)
            assert 0.0 <= val <= 1.0 + 1e-8, f"Disparity out of [0,1]: {val}"

    def test_mismatched_samples_raises(self):
        X = random_matrix(30, 5)
        Y = random_matrix(40, 5)
        with pytest.raises(ValueError, match="same number of samples"):
            procrustes_distance(X, Y)


# ---------------------------------------------------------------------------
# TestMNN
# ---------------------------------------------------------------------------

class TestMNN:

    def test_identical_matrices_return_one(self):
        X = random_matrix(50, 10)
        result = mutual_nearest_neighbors(X, X, k=5)
        assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"

    def test_random_matrices_low_mnn(self):
        rng = np.random.default_rng(2)
        X = random_matrix(200, 32, rng)
        Y = random_matrix(200, 32, rng)
        result = mutual_nearest_neighbors(X, Y, k=5)
        assert result < 0.5, f"Expected low MNN for random matrices, got {result}"

    def test_returns_float(self):
        X = random_matrix(30, 8)
        Y = random_matrix(30, 8)
        assert isinstance(mutual_nearest_neighbors(X, Y), float)

    def test_range(self):
        for _ in range(5):
            X = random_matrix(30, 8)
            Y = random_matrix(30, 8)
            val = mutual_nearest_neighbors(X, Y, k=3)
            assert 0.0 <= val <= 1.0 + 1e-8, f"MNN out of [0,1]: {val}"

    def test_different_feature_dims(self):
        X = random_matrix(40, 8)
        Y = random_matrix(40, 32)
        result = mutual_nearest_neighbors(X, Y, k=5)
        assert 0.0 <= result <= 1.0 + 1e-8

    def test_large_k_clipped_to_n_minus_one(self):
        X = random_matrix(10, 4)
        # k=100 should be silently clipped to n-1 = 9 without error
        result = mutual_nearest_neighbors(X, X, k=100)
        assert abs(result - 1.0) < 1e-6

    def test_mismatched_samples_raises(self):
        X = random_matrix(30, 5)
        Y = random_matrix(40, 5)
        with pytest.raises(ValueError, match="same number of samples"):
            mutual_nearest_neighbors(X, Y)

    def test_rotated_identical_structure_high_mnn(self):
        """Rotating X should preserve neighborhood structure."""
        X = random_matrix(50, 8)
        Q = random_rotation(8)
        X_rot = X @ Q
        result = mutual_nearest_neighbors(X, X_rot, k=5)
        assert result > 0.9, f"Expected high MNN after rotation, got {result}"
