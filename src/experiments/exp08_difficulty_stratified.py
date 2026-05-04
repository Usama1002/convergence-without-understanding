"""
Experiment 8: Difficulty-Stratified CKA Analysis.

Groups problems by difficulty stratum (fraction of models that answered
correctly) and computes mean CKA across all model pairs at all 21 layers
for each stratum with at least 5 problems.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from src.config import (
    MODEL_SHORT_NAMES,
    PATHS,
    ensure_all_dirs,
)
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.metrics.cka import linear_cka


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_correctness_matrix() -> tuple[np.ndarray, list[str], list[str]] | None:
    """Load all evaluation results and build (n_problems, n_models) correctness matrix.

    Returns
    -------
    (correctness, problem_ids, model_names) or None if no data available.
    """
    eval_dir = PATHS["evaluations"]
    all_results: list[list[dict]] = []
    model_names: list[str] = []

    for model_name in MODEL_SHORT_NAMES:
        eval_path = eval_dir / f"{model_name}.json"
        if not eval_path.exists():
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        all_results.append(results)
        model_names.append(model_name)

    if not all_results:
        return None

    problem_ids = [r["problem_id"] for r in all_results[0]]
    n_problems = len(problem_ids)
    n_models = len(all_results)

    correctness = np.zeros((n_problems, n_models), dtype=bool)
    for j, results in enumerate(all_results):
        for i, result in enumerate(results):
            correctness[i, j] = bool(result["correct"])

    return correctness, problem_ids, model_names


def _load_norm_states(model_name: str) -> np.ndarray | None:
    """Load (n_problems, 21, hidden_dim) normalized pre-decision states."""
    hs_dir = PATHS["hidden_states"]
    npz_path = hs_dir / f"{model_name}.npz"
    info_path = hs_dir / f"{model_name}_info.json"
    if not npz_path.exists() or not info_path.exists():
        return None
    hs_data = load_hidden_states(model_name)
    return get_states_at_normalized_positions(hs_data["pre_decision"], hs_data["model_info"])


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_8() -> dict:
    """Run difficulty-stratified CKA analysis.

    Computes n_correct per problem, groups by difficulty stratum
    (1/14 through 14/14), and for each stratum with >=5 problems
    computes mean CKA across all model pairs at all 21 layers.

    Returns
    -------
    dict
        Results saved to PATHS['metrics_difficulty']/exp08_results.json.
    """
    ensure_all_dirs()
    out_dir = PATHS["metrics_difficulty"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build correctness matrix
    # ------------------------------------------------------------------
    result = _load_correctness_matrix()
    if result is None:
        return {"error": "No evaluation results found."}

    correctness, problem_ids, model_names = result
    n_problems, n_models = correctness.shape

    # n_correct per problem
    n_correct_per_problem = correctness.sum(axis=1)  # shape (n_problems,)

    # ------------------------------------------------------------------
    # Step 2: Load hidden states for all available models
    # ------------------------------------------------------------------
    states_cache: dict[str, np.ndarray] = {}
    for model_name in model_names:
        s = _load_norm_states(model_name)
        if s is not None:
            states_cache[model_name] = s

    available_models = list(states_cache.keys())
    if len(available_models) < 2:
        return {"error": "Insufficient hidden states for analysis."}

    model_pairs = list(combinations(available_models, 2))

    # ------------------------------------------------------------------
    # Step 3: Group problems by difficulty stratum
    # ------------------------------------------------------------------
    # Strata: 0/n_models through n_models/n_models, but we group by n_correct value.
    strata: dict[int, list[int]] = {}
    for prob_idx in range(n_problems):
        nc = int(n_correct_per_problem[prob_idx])
        strata.setdefault(nc, []).append(prob_idx)

    # ------------------------------------------------------------------
    # Step 4: For each stratum with >=5 problems, compute mean CKA
    # ------------------------------------------------------------------
    stratum_results: list[dict] = []

    for n_correct_val in sorted(strata.keys()):
        indices = strata[n_correct_val]
        if len(indices) < 5:
            continue

        idx_arr = np.array(indices)
        fraction_correct = n_correct_val / n_models

        # Per-layer mean CKA across all model pairs
        layer_cka_sums = np.zeros(21)
        pair_count = 0

        for model_a, model_b in model_pairs:
            sa = states_cache[model_a]
            sb = states_cache[model_b]

            # Align problem count
            n_avail = min(sa.shape[0], sb.shape[0], max(idx_arr) + 1)
            valid_idx = idx_arr[idx_arr < n_avail]
            if len(valid_idx) < 5:
                continue

            for layer_idx in range(21):
                X = sa[valid_idx, layer_idx, :]
                Y = sb[valid_idx, layer_idx, :]
                layer_cka_sums[layer_idx] += linear_cka(X, Y)

            pair_count += 1

        if pair_count == 0:
            continue

        mean_cka_per_layer = (layer_cka_sums / pair_count).tolist()

        stratum_results.append({
            "n_correct": n_correct_val,
            "fraction_correct": fraction_correct,
            "n_problems": len(indices),
            "n_pairs_used": pair_count,
            "mean_cka_per_layer": mean_cka_per_layer,
        })

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    output = {
        "n_problems": n_problems,
        "n_models": n_models,
        "available_models": available_models,
        "strata": stratum_results,
    }
    out_path = out_dir / "exp08_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
