"""
Experiment 6: Transfer Probing.

Learns linear projections between model representation spaces and evaluates
whether probes trained in one model's space transfer to another model via
the learned projection. Produces a 14×14 transfer matrix and hierarchical
clustering of models based on transfer accuracy.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODEL_SHORT_NAMES,
    NORMALIZED_LAYER_POSITIONS,
    PATHS,
    SEEDS,
    N_CV_FOLDS,
    ensure_all_dirs,
)
from src.extraction import load_hidden_states, get_states_at_normalized_positions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def learn_projection(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Learn a least-squares linear mapping from source to target space.

    Finds W such that X_source @ W ≈ X_target (minimises Frobenius norm).

    Parameters
    ----------
    X_source : np.ndarray, shape (n, d_s)
    X_target : np.ndarray, shape (n, d_t)

    Returns
    -------
    W : np.ndarray, shape (d_s, d_t)
        Linear projection matrix.
    """
    W, _, _, _ = np.linalg.lstsq(X_source, X_target, rcond=None)
    return W


def _peak_layer_idx(model_name: str) -> int:
    """Return normalized layer index to use as 'peak' layer (default: layer 10 = midpoint)."""
    # exp02_all_results.json is a top-level list with key "peak_layer_idx"
    probe_results_path = PATHS["probes_linear"] / "exp02_all_results.json"
    if probe_results_path.exists():
        try:
            with open(probe_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else data.get("results", [])
            for rec in records:
                if rec.get("model") == model_name:
                    return int(rec.get("peak_layer_idx", 10))
        except Exception:
            pass
    return 10  # midpoint of 21 layers


def _load_peak_states_and_labels(
    model_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load peak-layer hidden states and correctness labels for a model.

    Returns (X, y) where X shape is (n_problems, hidden_dim) and y is binary.
    Returns None if data not available.
    """
    hs_dir = PATHS["hidden_states"]
    npz_path = hs_dir / f"{model_name}.npz"
    info_path = hs_dir / f"{model_name}_info.json"
    if not npz_path.exists() or not info_path.exists():
        return None

    hs_data = load_hidden_states(model_name)
    pre_states = hs_data["pre_decision"]
    model_info = hs_data["model_info"]

    # Normalized states: (n_problems, 21, hidden_dim)
    norm_states = get_states_at_normalized_positions(pre_states, model_info)

    layer_idx = _peak_layer_idx(model_name)
    X = norm_states[:, layer_idx, :]  # (n_problems, hidden_dim)

    # Load labels
    eval_path = PATHS["evaluations"] / f"{model_name}.json"
    if not eval_path.exists():
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    y = np.array([int(r["correct"]) for r in results])

    return X, y


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_6() -> dict:
    """Run transfer probing experiment.

    For each (source, target) model pair: uses 5 seeds × 5-fold CV to
    learn a projection from source to target space, trains a probe in
    target space on training data, and evaluates transfer by projecting
    source validation data through the learned mapping.

    Builds a 14×14 accuracy transfer matrix and applies hierarchical
    clustering.

    Returns
    -------
    dict
        Results saved to relevant PATHS locations.
    """
    ensure_all_dirs()

    # Create output directories
    transfer_dir = PATHS["metrics_transfer"]
    transfer_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load peak states and labels for all available models
    # ------------------------------------------------------------------
    model_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODEL_SHORT_NAMES:
        result = _load_peak_states_and_labels(model_name)
        if result is not None:
            model_data[model_name] = result

    available_models = list(model_data.keys())
    n_models = len(available_models)

    if n_models < 2:
        return {"error": "Insufficient model data for transfer probing."}

    # ------------------------------------------------------------------
    # Step 2: 5 seeds × 5-fold CV transfer probing for each pair
    # ------------------------------------------------------------------
    transfer_acc_matrix = np.full((n_models, n_models), float("nan"))
    transfer_auc_matrix = np.full((n_models, n_models), float("nan"))

    pair_results: list[dict] = []

    for i, source_name in enumerate(available_models):
        X_source, y_source = model_data[source_name]
        n_problems = X_source.shape[0]

        for j, target_name in enumerate(available_models):
            if i == j:
                transfer_acc_matrix[i, j] = 1.0
                transfer_auc_matrix[i, j] = 1.0
                continue

            X_target, y_target = model_data[target_name]
            if X_target.shape[0] != n_problems:
                continue

            seed_accs: list[float] = []
            seed_aucs: list[float] = []

            for seed in SEEDS:
                skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
                fold_accs: list[float] = []
                fold_aucs: list[float] = []

                for train_idx, val_idx in skf.split(X_source, y_source):
                    X_src_train = X_source[train_idx]
                    X_tgt_train = X_target[train_idx]
                    y_tgt_train = y_target[train_idx]
                    X_src_val = X_source[val_idx]
                    y_tgt_val = y_target[val_idx]

                    # Learn projection: source → target (on training set)
                    W = learn_projection(X_src_train, X_tgt_train)

                    # Scale target training features
                    scaler = StandardScaler()
                    X_tgt_train_scaled = scaler.fit_transform(X_tgt_train)

                    # Train probe in target space
                    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                    if len(np.unique(y_tgt_train)) < 2:
                        continue
                    probe.fit(X_tgt_train_scaled, y_tgt_train)

                    # Project source validation data to target space
                    X_src_val_projected = X_src_val @ W
                    X_src_val_projected_scaled = scaler.transform(X_src_val_projected)

                    # Predict
                    y_pred = probe.predict(X_src_val_projected_scaled)
                    acc = float(np.mean(y_pred == y_tgt_val))

                    try:
                        y_prob = probe.predict_proba(X_src_val_projected_scaled)[:, 1]
                        auc = roc_auc_score(y_tgt_val, y_prob)
                    except Exception:
                        auc = float("nan")

                    fold_accs.append(acc)
                    fold_aucs.append(auc)

                if fold_accs:
                    seed_accs.append(float(np.mean(fold_accs)))
                if fold_aucs:
                    seed_aucs.append(float(np.nanmean(fold_aucs)))

            if seed_accs:
                mean_acc = float(np.mean(seed_accs))
                mean_auc = float(np.nanmean(seed_aucs)) if seed_aucs else float("nan")
                transfer_acc_matrix[i, j] = mean_acc
                transfer_auc_matrix[i, j] = mean_auc
                pair_results.append({
                    "source": source_name,
                    "target": target_name,
                    "mean_accuracy": mean_acc,
                    "mean_auc": mean_auc,
                    "seed_accuracies": seed_accs,
                })

    # ------------------------------------------------------------------
    # Step 3: Hierarchical clustering on transfer accuracy matrix
    # ------------------------------------------------------------------
    # Use (1 - mean accuracy) as distance; fill diagonal with 0
    dist_matrix = 1.0 - np.nan_to_num(transfer_acc_matrix, nan=0.5)
    np.fill_diagonal(dist_matrix, 0.0)
    # Transfer accuracy is directional; symmetrize before squareform, which
    # reads only the upper triangle with checks=False.
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

    # Convert to condensed form for linkage
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    cluster_labels = fcluster(Z, t=3, criterion="maxclust")

    clustering_result = {
        "model_names": available_models,
        "linkage_matrix": Z.tolist(),
        "cluster_labels": cluster_labels.tolist(),
        "n_clusters": 3,
    }

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    # Save transfer matrices
    np.save(str(transfer_dir / "exp06_transfer_accuracy_matrix.npy"), transfer_acc_matrix)
    np.save(str(transfer_dir / "exp06_transfer_auc_matrix.npy"), transfer_auc_matrix)

    # Save clustering
    clustering_path = transfer_dir / "exp06_clustering.json"
    with open(clustering_path, "w", encoding="utf-8") as f:
        json.dump(clustering_result, f, indent=2)

    # Save summary
    summary = {
        "n_models": n_models,
        "model_names": available_models,
        "transfer_accuracy_matrix": transfer_acc_matrix.tolist(),
        "transfer_auc_matrix": transfer_auc_matrix.tolist(),
        "pair_results": pair_results,
        "clustering": clustering_result,
    }
    summary_path = transfer_dir / "exp06_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
