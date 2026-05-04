"""
Ablation Study: Effect of Sample Size on CKA Estimates.

Uses a subset of 4 models for efficiency.  At the mid-range layer
(normalized position index 10), evaluates CKA stability across
sample sizes [50, 100, 150, 200] with 10 random subsamples each.

Results saved to: cka/ablations/sample_size.json
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

from src.config import MODEL_SHORT_NAMES, PATHS, SAMPLE_SIZE_ABLATION, ensure_all_dirs
from src.extraction import get_states_at_normalized_positions, load_hidden_states
from src.metrics.cka import linear_cka

# Use first 4 models for efficiency
_SUBSET_MODELS = MODEL_SHORT_NAMES[:4]
_MID_LAYER_IDX = 10
_N_SUBSAMPLES = 10


def run_sample_size_ablation() -> dict:
    """Evaluate CKA stability across different sample sizes.

    Returns
    -------
    dict
        Results dictionary that was saved to disk.
    """
    ensure_all_dirs()

    # ------------------------------------------------------------------
    # Load hidden states for the model subset
    # ------------------------------------------------------------------
    all_states: dict[str, np.ndarray] = {}
    for name in _SUBSET_MODELS:
        hs_data = load_hidden_states(name)
        normalized = get_states_at_normalized_positions(
            hs_data["pre_decision"], hs_data["model_info"]
        )
        all_states[name] = normalized

    model_pairs = list(itertools.combinations(_SUBSET_MODELS, 2))
    n_problems_total = next(iter(all_states.values())).shape[0]

    sample_size_results = []

    for sample_size in SAMPLE_SIZE_ABLATION:
        if sample_size > n_problems_total:
            actual_size = n_problems_total
        else:
            actual_size = sample_size

        pair_results = []
        for name_i, name_j in model_pairs:
            states_i = all_states[name_i]
            states_j = all_states[name_j]

            subsample_cka_values = []
            for subsample_seed in range(_N_SUBSAMPLES):
                rng = np.random.default_rng(subsample_seed)
                chosen = rng.choice(n_problems_total, size=actual_size, replace=False)
                X = states_i[chosen, _MID_LAYER_IDX, :]
                Y = states_j[chosen, _MID_LAYER_IDX, :]
                subsample_cka_values.append(linear_cka(X, Y))

            pair_results.append({
                "model_i": name_i,
                "model_j": name_j,
                "cka_values": subsample_cka_values,
                "mean_cka": float(np.mean(subsample_cka_values)),
                "std_cka": float(np.std(subsample_cka_values)),
            })

        # Aggregate across pairs
        all_means = [pr["mean_cka"] for pr in pair_results]
        all_stds = [pr["std_cka"] for pr in pair_results]

        sample_size_results.append({
            "sample_size": actual_size,
            "layer_pos_idx": _MID_LAYER_IDX,
            "n_subsamples": _N_SUBSAMPLES,
            "mean_cka_across_pairs": float(np.mean(all_means)),
            "mean_std_across_pairs": float(np.mean(all_stds)),
            "pair_results": pair_results,
        })

    results = {
        "description": (
            "Ablation evaluating CKA stability across sample sizes "
            f"{SAMPLE_SIZE_ABLATION} with {_N_SUBSAMPLES} random subsamples each. "
            f"Computed at layer position index {_MID_LAYER_IDX} "
            f"using models: {_SUBSET_MODELS}."
        ),
        "models_used": _SUBSET_MODELS,
        "layer_pos_idx": _MID_LAYER_IDX,
        "n_subsamples": _N_SUBSAMPLES,
        "sample_sizes": SAMPLE_SIZE_ABLATION,
        "sample_size_results": sample_size_results,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(str(PATHS["metrics_cka"]), "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_size.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Sample size ablation results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_sample_size_ablation()
