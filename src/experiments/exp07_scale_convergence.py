"""
Experiment 7: Within-Family Scale Convergence.

For each model family in FAMILY_SCALE_TRAJECTORIES, computes pairwise
CKA / Procrustes / MNN with bootstrap CIs across all 21 layers.
Also computes cross-family CKA at matched scale tiers:
    small  : params_b <= 2.5
    medium : 2.5 < params_b <= 4.0
    large  : params_b > 4.0
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np

from src.config import (
    FAMILY_SCALE_TRAJECTORIES,
    MODEL_REGISTRY,
    NORMALIZED_LAYER_POSITIONS,
    N_BOOTSTRAP,
    PATHS,
    ensure_all_dirs,
)
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.metrics.cka import linear_cka
from src.metrics.mnn import mutual_nearest_neighbors
from src.metrics.procrustes import procrustes_distance
from src.metrics.statistics import bootstrap_ci_two_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tier(params_b: float) -> str:
    if params_b <= 2.5:
        return "small"
    elif params_b <= 4.0:
        return "medium"
    else:
        return "large"


def _load_norm_states(model_name: str) -> np.ndarray | None:
    """Load and return (n_problems, 21, hidden_dim) normalized pre-decision states."""
    hs_dir = PATHS["hidden_states"]
    npz_path = hs_dir / f"{model_name}.npz"
    info_path = hs_dir / f"{model_name}_info.json"
    if not npz_path.exists() or not info_path.exists():
        return None
    hs_data = load_hidden_states(model_name)
    norm = get_states_at_normalized_positions(hs_data["pre_decision"], hs_data["model_info"])
    return norm  # (n_problems, 21, hidden_dim)


def _compute_pair_metrics(
    states_a: np.ndarray,
    states_b: np.ndarray,
) -> list[dict]:
    """Compute CKA and MNN for all 21 layers (no bootstrap — fast)."""
    n_layers = states_a.shape[1]
    layer_results: list[dict] = []
    for layer_idx in range(n_layers):
        X = states_a[:, layer_idx, :]
        Y = states_b[:, layer_idx, :]

        if len(X) < 3:
            layer_results.append({"layer_idx": layer_idx, "skipped": True})
            continue

        layer_results.append({
            "layer_idx": layer_idx,
            "cka": float(linear_cka(X, Y)),
            "mnn": float(mutual_nearest_neighbors(X, Y)),
        })
    return layer_results


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_7() -> dict:
    """Run within-family scale convergence experiment.

    For each family in FAMILY_SCALE_TRAJECTORIES, computes pairwise
    CKA/Procrustes/MNN with bootstrap CIs. Also computes cross-family
    metrics at matched scale tiers.

    Returns
    -------
    dict
        Results saved to PATHS['metrics_scale']/exp07_results.json.
    """
    ensure_all_dirs()
    out_dir = PATHS["metrics_scale"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build param lookup
    params_lookup = {m["short_name"]: m["params_b"] for m in MODEL_REGISTRY}
    family_lookup = {m["short_name"]: m["family"] for m in MODEL_REGISTRY}

    # ------------------------------------------------------------------
    # Within-family analysis
    # ------------------------------------------------------------------
    within_family_results: list[dict] = []

    for family, members in FAMILY_SCALE_TRAJECTORIES.items():
        # Load states for all available members
        states_cache: dict[str, np.ndarray] = {}
        for model_name in members:
            s = _load_norm_states(model_name)
            if s is not None:
                states_cache[model_name] = s

        available = list(states_cache.keys())
        if len(available) < 2:
            continue

        for model_a, model_b in combinations(available, 2):
            sa = states_cache[model_a]
            sb = states_cache[model_b]

            # Align sample counts
            n = min(sa.shape[0], sb.shape[0])
            sa = sa[:n]
            sb = sb[:n]

            layer_metrics = _compute_pair_metrics(sa, sb)

            within_family_results.append({
                "family": family,
                "model_a": model_a,
                "model_b": model_b,
                "params_a": params_lookup.get(model_a),
                "params_b_val": params_lookup.get(model_b),
                "layer_metrics": layer_metrics,
            })

    # ------------------------------------------------------------------
    # Cross-family analysis at matched scale tiers
    # ------------------------------------------------------------------
    tier_small = [m["short_name"] for m in MODEL_REGISTRY if _get_tier(m["params_b"]) == "small"]
    tier_medium = [m["short_name"] for m in MODEL_REGISTRY if _get_tier(m["params_b"]) == "medium"]
    tier_large = [m["short_name"] for m in MODEL_REGISTRY if _get_tier(m["params_b"]) == "large"]

    cross_family_results: list[dict] = []

    for tier_name, tier_models in [("small", tier_small), ("medium", tier_medium), ("large", tier_large)]:
        # Only keep models with different families and available hidden states
        states_cache_tier: dict[str, np.ndarray] = {}
        for model_name in tier_models:
            s = _load_norm_states(model_name)
            if s is not None:
                states_cache_tier[model_name] = s

        available_tier = list(states_cache_tier.keys())
        for model_a, model_b in combinations(available_tier, 2):
            if family_lookup.get(model_a) == family_lookup.get(model_b):
                continue  # skip same family

            sa = states_cache_tier[model_a]
            sb = states_cache_tier[model_b]
            n = min(sa.shape[0], sb.shape[0])
            sa = sa[:n]
            sb = sb[:n]

            layer_metrics = _compute_pair_metrics(sa, sb)

            cross_family_results.append({
                "tier": tier_name,
                "model_a": model_a,
                "family_a": family_lookup.get(model_a),
                "model_b": model_b,
                "family_b": family_lookup.get(model_b),
                "layer_metrics": layer_metrics,
            })

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = {
        "within_family": within_family_results,
        "cross_family": cross_family_results,
        "families": list(FAMILY_SCALE_TRAJECTORIES.keys()),
        "tiers": {"small": tier_small, "medium": tier_medium, "large": tier_large},
    }
    out_path = out_dir / "exp07_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
