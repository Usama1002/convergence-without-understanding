"""
Experiment 4: Domain-Specific Convergence.

For each of 4 benchmark domains, identifies the all-correct problem subset
and computes CKA / Procrustes / MNN across all model pairs × 21 layers.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Any

import numpy as np

from src.config import (
    DATASET_CONFIGS,
    MODEL_SHORT_NAMES,
    NORMALIZED_LAYER_POSITIONS,
    PATHS,
    ensure_all_dirs,
)
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.metrics.cka import linear_cka
from src.metrics.procrustes import procrustes_distance
from src.metrics.mnn import mutual_nearest_neighbors


# ---------------------------------------------------------------------------
# Worker function (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _worker(args: tuple) -> dict:
    """Compute CKA/Procrustes/MNN for one (model_a, model_b, domain, layer_idx) job."""
    (
        model_a,
        model_b,
        domain,
        layer_idx,
        indices,
        pre_a_path,
        pre_b_path,
    ) = args

    # Load arrays
    data_a = np.load(pre_a_path)
    data_b = np.load(pre_b_path)
    states_a = data_a["pre_decision"]  # (n_problems, n_layers_p1, hidden_dim)
    states_b = data_b["pre_decision"]

    # Select domain indices and layer
    X = states_a[indices, layer_idx, :]
    Y = states_b[indices, layer_idx, :]

    if X.shape[0] < 5:
        cka = float("nan")
        proc = float("nan")
        mnn = float("nan")
    else:
        cka = linear_cka(X, Y)
        proc = procrustes_distance(X, Y)
        mnn = mutual_nearest_neighbors(X, Y)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "domain": domain,
        "layer_idx": layer_idx,
        "cka": cka,
        "procrustes": proc,
        "mnn": mnn,
        "n_problems": len(indices),
    }


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_4(n_workers: int = 16) -> dict:
    """Run domain-specific convergence experiment.

    For each domain, finds all-correct problem indices (problems where every
    model is correct within that domain) then computes CKA/Procrustes/MNN
    for all model pairs × 21 layers.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    dict
        Results structure saved to exp04_domain_results.json.
    """
    ensure_all_dirs()

    domains = [cfg["name"] for cfg in DATASET_CONFIGS]
    hidden_states_dir = PATHS["hidden_states"]
    out_dir = PATHS["cka_pairwise"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load evaluation results to build per-domain correctness
    # ------------------------------------------------------------------
    eval_dir = PATHS["evaluations"]
    # correctness_by_model[model] = {problem_id: bool}
    correctness_by_model: dict[str, dict[str, bool]] = {}
    problem_domains: dict[str, str] = {}

    for model_name in MODEL_SHORT_NAMES:
        eval_path = eval_dir / f"{model_name}.json"
        if not eval_path.exists():
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        correctness_by_model[model_name] = {
            r["problem_id"]: bool(r["correct"]) for r in results
        }
        for r in results:
            problem_domains[r["problem_id"]] = r["domain"]

    available_models = list(correctness_by_model.keys())
    if not available_models:
        return {"error": "No evaluation results found."}

    # ------------------------------------------------------------------
    # Step 2: For each domain, find all-correct problem indices
    # ------------------------------------------------------------------
    # We need an ordered list of problems for index alignment with hidden states.
    # Use the first available model's result list to get ordered problem IDs.
    first_model = available_models[0]
    first_eval_path = eval_dir / f"{first_model}.json"
    with open(first_eval_path, "r", encoding="utf-8") as f:
        first_results = json.load(f)
    all_problem_ids = [r["problem_id"] for r in first_results]

    domain_all_correct_indices: dict[str, list[int]] = {}
    for domain in domains:
        # Problems in this domain
        domain_problem_ids = [
            pid for pid in all_problem_ids if problem_domains.get(pid) == domain
        ]
        all_correct = []
        for pid in domain_problem_ids:
            global_idx = all_problem_ids.index(pid)
            if all(
                correctness_by_model.get(m, {}).get(pid, False)
                for m in available_models
            ):
                all_correct.append(global_idx)
        domain_all_correct_indices[domain] = all_correct

    # ------------------------------------------------------------------
    # Step 3: Determine layer count from first available model
    # ------------------------------------------------------------------
    n_layers_position = len(NORMALIZED_LAYER_POSITIONS)  # 21

    # Check which models have hidden states
    models_with_hs = [
        m for m in available_models
        if (hidden_states_dir / f"{m}.npz").exists()
    ]
    if not models_with_hs:
        return {"error": "No hidden states found."}

    # Build path lookup
    npz_paths = {m: str(hidden_states_dir / f"{m}.npz") for m in models_with_hs}

    # For layer indices we use 0..20 (normalized positions mapped during analysis)
    # Since get_states_at_normalized_positions is per-model, we pre-compute
    # the states. But for workers we need consistent layer indexing.
    # We'll pass raw layer_idx 0-20 and expect states already normalized.
    # Pre-compute normalized states for all models and save temporarily.
    # Actually: workers load raw npz; we pass pre-computed layer indices.
    # Simpler: workers load the npz and use all_problem_ids.

    # ------------------------------------------------------------------
    # Step 4: Build job list and parallelize
    # ------------------------------------------------------------------
    jobs = []
    model_pairs = list(combinations(models_with_hs, 2))

    # We need layer indices into the normalized 21-position space.
    # Since extraction saves (n_problems, n_layers_plus_one, hidden_dim),
    # we need to map normalized positions to actual layer indices per model.
    # Load model_info for each model.
    model_info_cache: dict[str, dict] = {}
    for m in models_with_hs:
        info_path = hidden_states_dir / f"{m}_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                model_info_cache[m] = json.load(f)

    for domain in domains:
        indices = domain_all_correct_indices.get(domain, [])
        if len(indices) < 5:
            continue
        for model_a, model_b in model_pairs:
            for layer_pos_idx in range(n_layers_position):
                jobs.append((
                    model_a,
                    model_b,
                    domain,
                    layer_pos_idx,  # index into 21-position list
                    indices,
                    npz_paths[model_a],
                    npz_paths[model_b],
                ))

    results_list: list[dict] = []
    if jobs:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_normalized, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    results_list.append(future.result())
                except Exception as exc:
                    job = futures[future]
                    results_list.append({
                        "model_a": job[0],
                        "model_b": job[1],
                        "domain": job[2],
                        "layer_idx": job[3],
                        "error": str(exc),
                    })

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    output = {
        "domains": domains,
        "domain_all_correct_counts": {
            d: len(v) for d, v in domain_all_correct_indices.items()
        },
        "n_model_pairs": len(model_pairs),
        "results": results_list,
    }
    out_path = out_dir / "exp04_domain_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def _worker_normalized(args: tuple) -> dict:
    """Compute metrics for one (model_a, model_b, domain, layer_pos_idx) job."""
    (
        model_a,
        model_b,
        domain,
        layer_pos_idx,
        indices,
        pre_a_path,
        pre_b_path,
    ) = args

    data_a = np.load(pre_a_path)
    data_b = np.load(pre_b_path)
    raw_a = data_a["pre_decision"]  # (n_problems, n_layers_p1, hidden_dim)
    raw_b = data_b["pre_decision"]

    n_layers_a = raw_a.shape[1]
    n_layers_b = raw_b.shape[1]

    # Map normalized position to layer index
    from src.extraction import get_layer_mapping
    layers_a = get_layer_mapping(n_layers_a, n_positions=21)
    layers_b = get_layer_mapping(n_layers_b, n_positions=21)

    layer_idx_a = layers_a[layer_pos_idx]
    layer_idx_b = layers_b[layer_pos_idx]

    X = raw_a[np.array(indices), layer_idx_a, :]
    Y = raw_b[np.array(indices), layer_idx_b, :]

    if X.shape[0] < 5:
        return {
            "model_a": model_a, "model_b": model_b,
            "domain": domain, "layer_idx": layer_pos_idx,
            "cka": None, "procrustes": None, "mnn": None,
            "n_problems": len(indices),
        }

    from src.metrics.cka import linear_cka
    from src.metrics.procrustes import procrustes_distance
    from src.metrics.mnn import mutual_nearest_neighbors

    cka = linear_cka(X, Y)
    proc = procrustes_distance(X, Y)
    mnn = mutual_nearest_neighbors(X, Y)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "domain": domain,
        "layer_idx": layer_pos_idx,
        "cka": cka,
        "procrustes": proc,
        "mnn": mnn,
        "n_problems": len(indices),
    }
