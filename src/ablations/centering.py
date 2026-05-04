"""
Ablation Study: Effect of Centering on CKA.

Compares centered (standard linear CKA) vs uncentered CKA across all
model pairs and 21 normalized layer positions using the all-correct
problem subset.  Reports per-layer means and the correlation between
the two variants.

Results saved to: cka/ablations/centering.json
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.config import MODEL_SHORT_NAMES, PATHS, ensure_all_dirs
from src.extraction import get_states_at_normalized_positions, load_hidden_states
from src.metrics.cka import linear_cka, uncentered_cka


def run_centering_ablation() -> dict:
    """Compare centered vs uncentered CKA across all pairs and 21 layers.

    Returns
    -------
    dict
        Results dictionary that was saved to disk.
    """
    ensure_all_dirs()

    # ------------------------------------------------------------------
    # Load problem categories
    # ------------------------------------------------------------------
    cats_path = os.path.join(str(PATHS["evaluations"]), "problem_categories.json")
    with open(cats_path, "r", encoding="utf-8") as f:
        categories = json.load(f)
    all_correct_indices = categories["all_correct"]

    if len(all_correct_indices) < 5:
        all_correct_indices = list(range(50))

    idx = np.array(all_correct_indices)

    # ------------------------------------------------------------------
    # Load hidden states for all models
    # ------------------------------------------------------------------
    all_states: dict[str, np.ndarray] = {}
    for name in MODEL_SHORT_NAMES:
        hs_data = load_hidden_states(name)
        normalized = get_states_at_normalized_positions(
            hs_data["pre_decision"], hs_data["model_info"]
        )
        all_states[name] = normalized

    model_pairs = list(itertools.combinations(MODEL_SHORT_NAMES, 2))
    n_layers = 21

    # Collect per-layer values across all pairs
    centered_by_layer: list[list[float]] = [[] for _ in range(n_layers)]
    uncentered_by_layer: list[list[float]] = [[] for _ in range(n_layers)]

    for name_i, name_j in model_pairs:
        states_i = all_states[name_i]
        states_j = all_states[name_j]
        for layer_idx in range(n_layers):
            X = states_i[idx, layer_idx, :]
            Y = states_j[idx, layer_idx, :]
            c_val = linear_cka(X, Y)
            u_val = uncentered_cka(X, Y)
            centered_by_layer[layer_idx].append(c_val)
            uncentered_by_layer[layer_idx].append(u_val)

    # Per-layer summaries
    layer_results = []
    all_centered_flat: list[float] = []
    all_uncentered_flat: list[float] = []

    for layer_idx in range(n_layers):
        c_vals = centered_by_layer[layer_idx]
        u_vals = uncentered_by_layer[layer_idx]
        all_centered_flat.extend(c_vals)
        all_uncentered_flat.extend(u_vals)
        layer_results.append({
            "layer_pos_idx": layer_idx,
            "mean_centered_cka": float(np.mean(c_vals)),
            "std_centered_cka": float(np.std(c_vals)),
            "mean_uncentered_cka": float(np.mean(u_vals)),
            "std_uncentered_cka": float(np.std(u_vals)),
        })

    # Overall correlation between centered and uncentered
    pearson_r, pearson_p = pearsonr(all_centered_flat, all_uncentered_flat)
    spearman_r, spearman_p = spearmanr(all_centered_flat, all_uncentered_flat)

    results = {
        "description": (
            "Ablation comparing centered (linear_cka) vs uncentered CKA "
            "across all model pairs at 21 normalized layer positions, "
            "using the all-correct problem subset."
        ),
        "n_pairs": len(model_pairs),
        "n_layers": n_layers,
        "n_problems_used": int(len(idx)),
        "correlation_centered_vs_uncentered": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        },
        "layer_results": layer_results,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(str(PATHS["metrics_cka"]), "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "centering.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Centering ablation results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_centering_ablation()
