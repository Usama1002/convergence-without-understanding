"""
Tests for experiment-level fixes: equal-size difficulty strata (exp08) and
the missing-post-state guard (exp10).
"""

import numpy as np

from src.experiments.exp08_difficulty_stratified import equalize_strata
from src.experiments.exp10_pre_post_decision import _compute_pair_cka


# ---------------------------------------------------------------------------
# TestEqualizeStrata
# ---------------------------------------------------------------------------


class TestEqualizeStrata:

    def test_all_strata_get_common_size(self):
        strata = {0: list(range(14)), 7: list(range(100, 150)), 14: list(range(200, 290))}
        equalized, equal_n = equalize_strata(strata, min_size=5, seed=42)
        assert equal_n == 14
        assert all(len(v) == 14 for v in equalized.values())

    def test_small_strata_dropped(self):
        strata = {0: [1, 2, 3], 7: list(range(20)), 14: list(range(50, 80))}
        equalized, equal_n = equalize_strata(strata, min_size=5, seed=42)
        assert 0 not in equalized
        assert equal_n == 20

    def test_subsample_is_within_stratum(self):
        strata = {3: list(range(30)), 9: list(range(100, 160))}
        equalized, _ = equalize_strata(strata, seed=0)
        assert set(equalized[3]) <= set(range(30))
        assert set(equalized[9]) <= set(range(100, 160))

    def test_deterministic_for_seed(self):
        strata = {1: list(range(40)), 2: list(range(50, 120))}
        a, _ = equalize_strata(strata, seed=7)
        b, _ = equalize_strata(strata, seed=7)
        assert a == b

    def test_empty_when_nothing_eligible(self):
        equalized, equal_n = equalize_strata({0: [1, 2]}, min_size=5)
        assert equalized == {}
        assert equal_n == 0


# ---------------------------------------------------------------------------
# TestExp10MissingPostStates
# ---------------------------------------------------------------------------


class TestExp10MissingPostStates:

    @staticmethod
    def _write_npz(path, post_value=None):
        rng = np.random.default_rng(0)
        pre = rng.standard_normal((30, 21, 8)).astype(np.float32)
        post = rng.standard_normal((30, 21, 8)).astype(np.float32)
        if post_value is not None:
            post[:] = post_value
        np.savez(path, pre_decision=pre, post_decision=post)
        return str(path)

    def test_valid_states_compute_cka(self, tmp_path):
        a = self._write_npz(tmp_path / "a.npz")
        b = self._write_npz(tmp_path / "b.npz")
        rec = _compute_pair_cka(("ma", "mb", "post_decision", a, b))
        assert "error" not in rec
        assert len(rec["cka_per_layer"]) == 21

    def test_nan_post_states_are_skipped(self, tmp_path):
        # Models >10B store NaN post-decision states (extraction skips the
        # post pass); the pair must be excluded, not compared as pre vs pre.
        a = self._write_npz(tmp_path / "a.npz", post_value=np.nan)
        b = self._write_npz(tmp_path / "b.npz")
        rec = _compute_pair_cka(("ma", "mb", "post_decision", a, b))
        assert "error" in rec
        assert "missing" in rec["error"]
