"""
Experiment 1: Cross-Model Representational Similarity.

For every pair of the 14 models, at each of 21 normalized layer positions,
and for three problem conditions (all_correct, all_incorrect, mixed), computes:
  - Linear CKA (primary metric)
  - Mutual nearest neighbors (topological metric)
  - Bootstrap 95% CI for CKA (200 resamples, only for sets with < 100 samples)

Procrustes is skipped in the main loop (too slow on high-dim matrices)
and computed separately for key comparisons only.

Results are saved incrementally per model pair for resume support.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from src.config import (
    MODEL_SHORT_NAMES,
    N_MODELS,
    PATHS,
    ensure_all_dirs,
)
from src.evaluation import (
    categorize_problems,
    compute_correctness_matrix,
    load_evaluation_results,
)
from src.extraction import get_states_at_normalized_positions, load_hidden_states
from src.metrics.cka import linear_cka
from src.metrics.mnn import mutual_nearest_neighbors
from src.metrics.statistics import bootstrap_ci_two_sample


BOOT_N = 200  # Bootstrap resamples (only for small sets)


def _compute_pair_all_layers(
    states_i: np.ndarray,
    states_j: np.ndarray,
    condition_indices: dict[str, list[int]],
    model_i_name: str,
    model_j_name: str,
) -> list[dict]:
    """Compute CKA and MNN for one model pair across all layers and conditions."""
    results = []
    n_layers = states_i.shape[1]

    for layer_idx in range(n_layers):
        for cond_name, indices in condition_indices.items():
            if len(indices) < 5:
                results.append({
                    "model_a": model_i_name,
                    "model_b": model_j_name,
                    "layer_pos_idx": layer_idx,
                    "condition": cond_name,
                    "n_samples": len(indices),
                    "skipped": True,
                })
                continue

            idx = np.array(indices)
            X = states_i[idx, layer_idx, :]
            Y = states_j[idx, layer_idx, :]

            cka_val = linear_cka(X, Y)
            mnn_val = mutual_nearest_neighbors(X, Y)

            result = {
                "model_a": model_i_name,
                "model_b": model_j_name,
                "layer_pos_idx": layer_idx,
                "condition": cond_name,
                "n_samples": len(indices),
                "skipped": False,
                "cka": float(cka_val),
                "mnn": float(mnn_val),
            }

            # Bootstrap CI only for small sets (< 100 samples)
            # Large sets (mixed=771) are already very stable
            if len(indices) < 100:
                _, ci_lo, ci_hi = bootstrap_ci_two_sample(
                    X, Y, linear_cka, n_bootstrap=BOOT_N, seed=42
                )
                result["cka_ci_lo"] = float(ci_lo)
                result["cka_ci_hi"] = float(ci_hi)

            results.append(result)

    return results


def run_experiment_1(n_workers: int = 8) -> None:
    """Run Experiment 1: cross-model representational similarity.

    Iterates over model pairs sequentially. No ProcessPoolExecutor —
    avoids memory duplication that caused swap thrashing.
    Results saved incrementally per pair for resume support.
    """
    ensure_all_dirs()
    out_dir = Path(str(PATHS["cka_pairwise"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load evaluation results and compute categories
    # ------------------------------------------------------------------
    print("Loading evaluation results...")
    all_eval_list = []
    for name in MODEL_SHORT_NAMES:
        all_eval_list.append(load_evaluation_results(name))

    correctness, problem_ids, model_names = compute_correctness_matrix(all_eval_list)
    categories = categorize_problems(correctness, problem_ids)

    cats_path = Path(str(PATHS["evaluations"])) / "problem_categories.json"
    categories_json = {
        "all_correct": [int(i) for i in categories["all_correct"]],
        "all_incorrect": [int(i) for i in categories["all_incorrect"]],
        "mixed": [int(i) for i in categories["mixed"]],
        "difficulty": categories["difficulty"].tolist() if hasattr(categories["difficulty"], "tolist") else categories["difficulty"],
        "problem_ids": problem_ids,
    }
    with open(cats_path, "w") as f:
        json.dump(categories_json, f, indent=2)
    print(f"Problem categories saved to {cats_path}")
    print(f"  All-correct: {len(categories['all_correct'])} problems")
    print(f"  All-incorrect: {len(categories['all_incorrect'])} problems")
    print(f"  Mixed: {len(categories['mixed'])} problems")

    condition_indices = {
        "all_correct": categories["all_correct"],
        "all_incorrect": categories["all_incorrect"],
        "mixed": categories["mixed"],
    }

    # ------------------------------------------------------------------
    # 2. Load hidden states for all models
    # ------------------------------------------------------------------
    print("\nLoading hidden states...")
    all_states = {}
    for name in MODEL_SHORT_NAMES:
        hs_data = load_hidden_states(name)
        normalized = get_states_at_normalized_positions(
            hs_data["pre_decision"], hs_data["model_info"]
        )
        all_states[name] = normalized
        print(f"  {name}: {normalized.shape}")

    # ------------------------------------------------------------------
    # 3. Compute per pair, save incrementally
    # ------------------------------------------------------------------
    pairs = list(itertools.combinations(MODEL_SHORT_NAMES, 2))
    n_pairs = len(pairs)
    print(f"\nComputing {n_pairs} pairs × 21 layers × 3 conditions")
    print(f"  Bootstrap: {BOOT_N} (only for sets < 100 samples)")
    print(f"  Metrics: CKA + MNN (Procrustes skipped — too slow on high-dim)")

    all_results = []
    t_start = time.time()

    for pair_idx, (name_i, name_j) in enumerate(pairs):
        pair_file = out_dir / f"pair_{name_i}__{name_j}.json"

        if pair_file.exists():
            with open(pair_file) as f:
                pair_results = json.load(f)
            all_results.extend(pair_results)
            elapsed_total = time.time() - t_start
            print(f"  [{pair_idx+1}/{n_pairs}] {name_i} × {name_j} — cached")
            continue

        t0 = time.time()
        pair_results = _compute_pair_all_layers(
            all_states[name_i],
            all_states[name_j],
            condition_indices,
            name_i,
            name_j,
        )
        elapsed = time.time() - t0

        with open(pair_file, "w") as f:
            json.dump(pair_results, f)
        all_results.extend(pair_results)

        elapsed_total = time.time() - t_start
        done = pair_idx + 1
        eta = (elapsed_total / done) * (n_pairs - done)
        print(f"  [{done}/{n_pairs}] {name_i} × {name_j} — "
              f"{elapsed:.1f}s, ETA {eta/60:.0f}min")

    # ------------------------------------------------------------------
    # 4. Save combined results
    # ------------------------------------------------------------------
    combined_path = out_dir / "exp01_all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = (time.time() - t_start) / 60
    print(f"\nDone! {len(all_results)} results saved to {combined_path}")
    print(f"Total time: {total_time:.1f} minutes")


if __name__ == "__main__":
    run_experiment_1()
