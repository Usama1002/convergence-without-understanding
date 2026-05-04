"""
Tests for src/extraction.py

Covers get_nearest_layer_index and get_layer_mapping without requiring GPU.
"""

import numpy as np
import pytest

from src.extraction import get_layer_mapping, get_nearest_layer_index


# ---------------------------------------------------------------------------
# Tests for get_nearest_layer_index
# ---------------------------------------------------------------------------


class TestGetNearestLayerIndex:
    def test_position_zero_maps_to_first_layer(self):
        assert get_nearest_layer_index(0.0, 10) == 0

    def test_position_one_maps_to_last_layer(self):
        assert get_nearest_layer_index(1.0, 10) == 9

    def test_exact_midpoint_single_layer(self):
        # With 1 layer, only index 0 exists
        assert get_nearest_layer_index(0.5, 1) == 0

    def test_exact_midpoint_two_layers(self):
        # linspace(0,1,2) = [0.0, 1.0]; 0.5 is equidistant, argmin picks 0
        result = get_nearest_layer_index(0.5, 2)
        assert result in (0, 1)

    def test_midpoint_with_odd_layers(self):
        # linspace(0,1,5) = [0.0, 0.25, 0.5, 0.75, 1.0]; 0.5 → index 2
        assert get_nearest_layer_index(0.5, 5) == 2

    def test_position_near_zero(self):
        # 0.05 is closest to 0.0 (index 0) in a 5-layer model
        assert get_nearest_layer_index(0.05, 5) == 0

    def test_position_near_end(self):
        # 0.95 is closest to 1.0 (index 4) in a 5-layer model
        assert get_nearest_layer_index(0.95, 5) == 4

    def test_returns_int(self):
        result = get_nearest_layer_index(0.3, 10)
        assert isinstance(result, int)

    def test_result_in_valid_range(self):
        for total in [1, 5, 10, 28, 33]:
            for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
                idx = get_nearest_layer_index(pos, total)
                assert 0 <= idx < total, (
                    f"Index {idx} out of range [0, {total}) for pos={pos}"
                )

    def test_position_0_25_with_5_layers(self):
        # linspace(0,1,5) = [0.0, 0.25, 0.5, 0.75, 1.0]; 0.25 → index 1
        assert get_nearest_layer_index(0.25, 5) == 1

    def test_position_0_75_with_5_layers(self):
        # linspace(0,1,5) = [0.0, 0.25, 0.5, 0.75, 1.0]; 0.75 → index 3
        assert get_nearest_layer_index(0.75, 5) == 3

    def test_single_layer_always_returns_zero(self):
        for pos in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert get_nearest_layer_index(pos, 1) == 0

    def test_large_model_boundary(self):
        # 33-layer model: position 0.0 → 0, position 1.0 → 32
        assert get_nearest_layer_index(0.0, 33) == 0
        assert get_nearest_layer_index(1.0, 33) == 32

    def test_exact_grid_positions_map_correctly(self):
        # linspace(0,1,10) = [0, 1/9, 2/9, ..., 1]
        total = 10
        positions = np.linspace(0.0, 1.0, total)
        for expected_idx, pos in enumerate(positions):
            result = get_nearest_layer_index(float(pos), total)
            assert result == expected_idx, (
                f"pos={pos}, expected {expected_idx}, got {result}"
            )


# ---------------------------------------------------------------------------
# Tests for get_layer_mapping
# ---------------------------------------------------------------------------


class TestGetLayerMapping:
    def test_default_returns_21_indices(self):
        mapping = get_layer_mapping(10)
        assert len(mapping) == 21

    def test_custom_n_positions(self):
        mapping = get_layer_mapping(10, n_positions=11)
        assert len(mapping) == 11

    def test_single_position(self):
        mapping = get_layer_mapping(10, n_positions=1)
        assert len(mapping) == 1
        # Single position at 0.0 → index 0
        assert mapping[0] == 0

    def test_all_indices_in_valid_range(self):
        for total in [5, 10, 28, 33]:
            mapping = get_layer_mapping(total)
            for idx in mapping:
                assert 0 <= idx < total, (
                    f"Index {idx} out of range [0, {total}) for total={total}"
                )

    def test_first_index_is_zero(self):
        # Position 0.0 always maps to index 0
        mapping = get_layer_mapping(20)
        assert mapping[0] == 0

    def test_last_index_is_total_minus_one(self):
        # Position 1.0 always maps to last index
        for total in [5, 10, 28]:
            mapping = get_layer_mapping(total)
            assert mapping[-1] == total - 1, (
                f"Last index {mapping[-1]} != {total - 1} for total={total}"
            )

    def test_returns_list_of_ints(self):
        mapping = get_layer_mapping(10)
        assert isinstance(mapping, list)
        for idx in mapping:
            assert isinstance(idx, int)

    def test_non_decreasing(self):
        # Normalized positions are increasing, so layer indices should be
        # non-decreasing
        for total in [5, 10, 21, 33]:
            mapping = get_layer_mapping(total)
            for j in range(len(mapping) - 1):
                assert mapping[j] <= mapping[j + 1], (
                    f"Mapping not non-decreasing at {j}: {mapping[j]} > {mapping[j+1]}"
                )

    def test_single_layer_all_zeros(self):
        mapping = get_layer_mapping(1)
        assert all(idx == 0 for idx in mapping)

    def test_two_layers_21_positions(self):
        # total=2: linspace(0,1,2)=[0,1]
        # All 21 positions map to either 0 or 1
        mapping = get_layer_mapping(2)
        assert len(mapping) == 21
        assert all(idx in (0, 1) for idx in mapping)

    def test_21_layers_maps_identity(self):
        # With total=21 and n_positions=21, linspace matches exactly
        # Each position maps to exactly its own index
        mapping = get_layer_mapping(21, n_positions=21)
        assert mapping == list(range(21))

    def test_large_model_28_layers(self):
        # Typical 28-layer model (e.g. 7B)
        mapping = get_layer_mapping(29)  # 28 transformer + 1 embedding = 29
        assert len(mapping) == 21
        assert mapping[0] == 0
        assert mapping[-1] == 28

    def test_n_positions_2(self):
        # With n_positions=2: linspace(0,1,2)=[0.0, 1.0]
        # First → 0, last → total-1
        for total in [5, 10, 28]:
            mapping = get_layer_mapping(total, n_positions=2)
            assert len(mapping) == 2
            assert mapping[0] == 0
            assert mapping[1] == total - 1

    def test_consistency_with_get_nearest_layer_index(self):
        # get_layer_mapping should match calling get_nearest_layer_index manually
        total = 15
        n_pos = 21
        mapping = get_layer_mapping(total, n_positions=n_pos)
        positions = np.linspace(0.0, 1.0, n_pos)
        expected = [get_nearest_layer_index(float(p), total) for p in positions]
        assert mapping == expected
