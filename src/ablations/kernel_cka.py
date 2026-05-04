"""
Ablation Study: Linear CKA vs Kernel (RBF) CKA.

Compares linear_cka and kernel_cka across all model pairs and 21
normalized layer positions using the all-correct problem subset.
Reports per-layer means and overall correlation between the two metrics.

Results saved to: cka/kernel_cka/kernel_cka_ablation.json
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.config import MODEL_SHORT_NAMES, PATHS, ensure_all_dirs
from src.extraction import get_states_at_normalized_positions, load_hidden_states
from src.metrics.cka import kernel_cka, linear_cka


def run_kernel_cka_ablation() -> dict:
    """Compare linear CKA vs kernel CKA across all pairs and 21 layers.

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

    linear_by_layer: list[list[float]] = [[] for _ in range(n_layers)]
    kernel_by_layer: list[list[float]] = [[] for _ in range(n_layers)]

    for name_i, name_j in model_pairs:
        states_i = all_states[name_i]
        states_j = all_states[name_j]
        for layer_idx in range(n_layers):
            X = states_i[idx, layer_idx, :]
            Y = states_j[idx, layer_idx, :]
            lin_val = linear_cka(X, Y)
            kern_val = kernel_cka(X, Y)
            linear_by_layer[layer_idx].append(lin_val)
            kernel_by_layer[layer_idx].append(kern_val)

    # Per-layer summaries
    layer_results = []
    all_linear_flat: list[float] = []
    all_kernel_flat: list[float] = []

    for layer_idx in range(n_layers):
        lin_vals = linear_by_layer[layer_idx]
        kern_vals = kernel_by_layer[layer_idx]
        all_linear_flat.extend(lin_vals)
        all_kernel_flat.extend(kern_vals)
        layer_results.append({
            "layer_pos_idx": layer_idx,
            "mean_linear_cka": float(np.mean(lin_vals)),
            "std_linear_cka": float(np.std(lin_vals)),
            "mean_kernel_cka": float(np.mean(kern_vals)),
            "std_kernel_cka": float(np.std(kern_vals)),
        })

    pearson_r, pearson_p = pearsonr(all_linear_flat, all_kernel_flat)
    spearman_r, spearman_p = spearmanr(all_linear_flat, all_kernel_flat)

    results = {
        "description": (
            "Ablation comparing linear CKA vs RBF kernel CKA "
            "across all model pairs at 21 normalized layer positions, "
            "using the all-correct problem subset."
        ),
        "n_pairs": len(model_pairs),
        "n_layers": n_layers,
        "n_problems_used": int(len(idx)),
        "correlation_linear_vs_kernel": {
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
    out_dir = os.path.join(str(PATHS["metrics_cka"]), "kernel_cka")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kernel_cka_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Kernel CKA ablation results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_kernel_cka_ablation()
