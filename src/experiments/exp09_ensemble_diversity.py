"""
Experiment 9: Ensemble Diversity Validation.

Generates all C(14,3) = 364 model triples, computes majority-vote ensemble
accuracy and mean pairwise divergence (1 - CKA), then correlates accuracy
vs divergence using Pearson and Spearman correlations.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.config import (
    MODEL_SHORT_NAMES,
    PATHS,
    ensure_all_dirs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_correctness_vectors() -> dict[str, np.ndarray] | None:
    """Load per-model binary correctness vectors.

    Returns dict mapping model_name -> np.ndarray shape (n_problems,).
    """
    eval_dir = PATHS["evaluations"]
    correctness: dict[str, np.ndarray] = {}
    for model_name in MODEL_SHORT_NAMES:
        eval_path = eval_dir / f"{model_name}.json"
        if not eval_path.exists():
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        correctness[model_name] = np.array([int(r["correct"]) for r in results])
    return correctness if correctness else None


def _load_pairwise_cka() -> dict[tuple[str, str], list[float]] | None:
    """Load pairwise CKA per layer from Exp 1 results.

    Returns dict mapping (model_a, model_b) -> list of 21 CKA values
    (averaged across layer positions).
    """
    cka_path = PATHS["cka_pairwise"] / "exp01_all_results.json"
    if not cka_path.exists():
        return None

    with open(cka_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_list = data if isinstance(data, list) else data.get("results", [])

    # Accumulate CKA per (model_a, model_b, layer_idx)
    from collections import defaultdict
    cka_accum: dict[tuple[str, str, int], list[float]] = defaultdict(list)

    for rec in results_list:
        model_a = rec.get("model_a", "")
        model_b = rec.get("model_b", "")
        layer_idx = int(rec.get("layer_idx", rec.get("layer", 0)))
        cka_val = rec.get("cka")
        if cka_val is not None and not (isinstance(cka_val, float) and np.isnan(cka_val)):
            cka_accum[(model_a, model_b, layer_idx)].append(float(cka_val))

    # Average per (pair, layer_idx) and then mean across all layers
    pair_layer_cka: dict[tuple[str, str], dict[int, float]] = {}
    for (ma, mb, li), vals in cka_accum.items():
        key = (ma, mb)
        if key not in pair_layer_cka:
            pair_layer_cka[key] = {}
        pair_layer_cka[key][li] = float(np.mean(vals))

    # For each pair, build ordered list of 21 CKA values
    pair_cka_list: dict[tuple[str, str], list[float]] = {}
    for key, layer_dict in pair_layer_cka.items():
        cka_list = [layer_dict.get(li, float("nan")) for li in range(21)]
        pair_cka_list[key] = cka_list

    return pair_cka_list if pair_cka_list else None


def _majority_vote_accuracy(
    correctness_a: np.ndarray,
    correctness_b: np.ndarray,
    correctness_c: np.ndarray,
) -> float:
    """Compute majority-vote ensemble accuracy for a 3-model triple."""
    votes = correctness_a + correctness_b + correctness_c
    ensemble_correct = (votes >= 2).astype(int)
    return float(ensemble_correct.mean())


def _mean_pairwise_divergence(
    pair_cka: dict[tuple[str, str], list[float]],
    model_a: str,
    model_b: str,
    model_c: str,
) -> float:
    """Compute mean (1 - mean_CKA) for the 3 pairs in a triple."""
    pairs = [(model_a, model_b), (model_a, model_c), (model_b, model_c)]
    divergences: list[float] = []
    for ma, mb in pairs:
        cka_list = pair_cka.get((ma, mb)) or pair_cka.get((mb, ma))
        if cka_list is None:
            continue
        valid = [v for v in cka_list if not np.isnan(v)]
        if valid:
            mean_cka = float(np.mean(valid))
            divergences.append(1.0 - mean_cka)
    return float(np.mean(divergences)) if divergences else float("nan")


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_9() -> dict:
    """Run ensemble diversity validation.

    Generates all C(14,3)=364 triples, computes majority-vote accuracy and
    mean pairwise divergence, then correlates accuracy vs divergence.

    Returns
    -------
    dict
        Results saved to PATHS['metrics_ensemble']/exp09_results.json.
    """
    ensure_all_dirs()
    out_dir = PATHS["metrics_ensemble"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load correctness vectors
    # ------------------------------------------------------------------
    correctness = _load_correctness_vectors()
    if correctness is None:
        return {"error": "No evaluation results found."}

    available_models = list(correctness.keys())
    n_models = len(available_models)

    # ------------------------------------------------------------------
    # Step 2: Load pairwise CKA
    # ------------------------------------------------------------------
    pair_cka = _load_pairwise_cka()

    # ------------------------------------------------------------------
    # Step 3: Generate all C(n,3) triples and compute metrics
    # ------------------------------------------------------------------
    ensemble_results: list[dict] = []

    for model_a, model_b, model_c in combinations(available_models, 3):
        acc = _majority_vote_accuracy(
            correctness[model_a],
            correctness[model_b],
            correctness[model_c],
        )

        if pair_cka is not None:
            div = _mean_pairwise_divergence(pair_cka, model_a, model_b, model_c)
        else:
            div = float("nan")

        ensemble_results.append({
            "models": [model_a, model_b, model_c],
            "ensemble_accuracy": acc,
            "mean_divergence": div,
        })

    # ------------------------------------------------------------------
    # Step 4: Correlate accuracy vs divergence
    # ------------------------------------------------------------------
    accuracies = np.array([r["ensemble_accuracy"] for r in ensemble_results])
    divergences = np.array([r["mean_divergence"] for r in ensemble_results])

    valid_mask = ~np.isnan(divergences)
    correlations: dict = {}

    if valid_mask.sum() >= 3:
        pearson_r, pearson_p = pearsonr(accuracies[valid_mask], divergences[valid_mask])
        spearman_r, spearman_p = spearmanr(accuracies[valid_mask], divergences[valid_mask])
        correlations = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n_valid": int(valid_mask.sum()),
        }
    else:
        correlations = {"note": "Insufficient valid pairs for correlation."}

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    output = {
        "n_models": n_models,
        "n_ensembles": len(ensemble_results),
        "correlations": correlations,
        "ensemble_results": ensemble_results,
    }
    out_path = out_dir / "exp09_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
