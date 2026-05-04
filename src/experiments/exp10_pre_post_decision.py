"""
Experiment 10: Pre vs Post Decision Representation Convergence.

Computes mean CKA across all model pairs at each of 21 layers separately
for pre_decision and post_decision hidden states, then compares which
phase shows higher CKA.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np

from src.config import (
    MODEL_SHORT_NAMES,
    NORMALIZED_LAYER_POSITIONS,
    PATHS,
    ensure_all_dirs,
)
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.metrics.cka import linear_cka


# ---------------------------------------------------------------------------
# Worker (module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _compute_pair_cka(args: tuple) -> dict:
    """Compute CKA per layer for one model pair and one phase."""
    model_a, model_b, phase, npz_a_path, npz_b_path = args

    data_a = np.load(npz_a_path)
    data_b = np.load(npz_b_path)

    raw_a = data_a[phase]  # (n_problems, n_layers_p1, hidden_dim)
    raw_b = data_b[phase]

    n_layers_a = raw_a.shape[1]
    n_layers_b = raw_b.shape[1]

    from src.extraction import get_layer_mapping
    layers_a = get_layer_mapping(n_layers_a, n_positions=21)
    layers_b = get_layer_mapping(n_layers_b, n_positions=21)

    n = min(raw_a.shape[0], raw_b.shape[0])
    cka_per_layer: list[float] = []

    from src.metrics.cka import linear_cka
    for pos_idx in range(21):
        X = raw_a[:n, layers_a[pos_idx], :]
        Y = raw_b[:n, layers_b[pos_idx], :]
        cka_per_layer.append(linear_cka(X, Y))

    return {
        "model_a": model_a,
        "model_b": model_b,
        "phase": phase,
        "cka_per_layer": cka_per_layer,
    }


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_10(n_workers: int = 16) -> dict:
    """Run pre vs post decision CKA comparison.

    Loads both pre_decision and post_decision hidden states for all models,
    computes mean CKA across all pairs at each of 21 layers for both
    phases, then compares which phase shows higher CKA.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    dict
        Results saved to PATHS['metrics_pre_vs_post']/exp10_results.json.
    """
    ensure_all_dirs()
    out_dir = PATHS["metrics_pre_vs_post"]
    out_dir.mkdir(parents=True, exist_ok=True)

    hs_dir = PATHS["hidden_states"]

    # ------------------------------------------------------------------
    # Step 1: Find models with available hidden states
    # ------------------------------------------------------------------
    available_models = [
        m for m in MODEL_SHORT_NAMES
        if (hs_dir / f"{m}.npz").exists()
    ]

    if len(available_models) < 2:
        return {"error": "Insufficient hidden states for analysis."}

    model_pairs = list(combinations(available_models, 2))
    phases = ["pre_decision", "post_decision"]

    npz_paths = {m: str(hs_dir / f"{m}.npz") for m in available_models}

    # ------------------------------------------------------------------
    # Step 2: Build job list and parallelize
    # ------------------------------------------------------------------
    jobs = []
    for phase in phases:
        for model_a, model_b in model_pairs:
            jobs.append((model_a, model_b, phase, npz_paths[model_a], npz_paths[model_b]))

    pair_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_pair_cka, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                pair_results.append(future.result())
            except Exception as exc:
                job = futures[future]
                pair_results.append({
                    "model_a": job[0],
                    "model_b": job[1],
                    "phase": job[2],
                    "error": str(exc),
                })

    # ------------------------------------------------------------------
    # Step 3: Compute mean CKA per layer per phase
    # ------------------------------------------------------------------
    phase_cka_sums: dict[str, np.ndarray] = {
        p: np.zeros(21) for p in phases
    }
    phase_pair_counts: dict[str, int] = {p: 0 for p in phases}

    for rec in pair_results:
        phase = rec.get("phase", "")
        if "error" in rec or phase not in phases:
            continue
        cka_vals = rec.get("cka_per_layer", [])
        if len(cka_vals) == 21:
            phase_cka_sums[phase] += np.array(cka_vals)
            phase_pair_counts[phase] += 1

    mean_cka_pre = (
        (phase_cka_sums["pre_decision"] / phase_pair_counts["pre_decision"]).tolist()
        if phase_pair_counts["pre_decision"] > 0
        else [None] * 21
    )
    mean_cka_post = (
        (phase_cka_sums["post_decision"] / phase_pair_counts["post_decision"]).tolist()
        if phase_pair_counts["post_decision"] > 0
        else [None] * 21
    )

    # ------------------------------------------------------------------
    # Step 4: Compare phases
    # ------------------------------------------------------------------
    comparison: dict = {}
    if phase_pair_counts["pre_decision"] > 0 and phase_pair_counts["post_decision"] > 0:
        pre_arr = np.array(mean_cka_pre)
        post_arr = np.array(mean_cka_post)
        overall_pre = float(np.mean(pre_arr))
        overall_post = float(np.mean(post_arr))
        higher_phase = "pre_decision" if overall_pre > overall_post else "post_decision"
        comparison = {
            "overall_mean_pre": overall_pre,
            "overall_mean_post": overall_post,
            "higher_phase": higher_phase,
            "difference_post_minus_pre": float(overall_post - overall_pre),
            "layer_differences_post_minus_pre": (post_arr - pre_arr).tolist(),
        }

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    output = {
        "n_models": len(available_models),
        "n_pairs": len(model_pairs),
        "mean_cka_pre_decision": mean_cka_pre,
        "mean_cka_post_decision": mean_cka_post,
        "comparison": comparison,
        "pair_results": pair_results,
    }
    out_path = out_dir / "exp10_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
