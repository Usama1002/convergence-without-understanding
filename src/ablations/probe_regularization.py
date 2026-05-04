"""
Ablation Study: Effect of L2 Regularization on Linear Probes.

For each model, sweeps L2 regularization values from PROBE_L2_SWEEP
([1e-4 to 1e1]) at the model's peak layer (loaded from Exp 2 results).
Uses 5 seeds × 5-fold stratified cross-validation per L2 value.

Results saved to: probes/ablations/regularization_sweep.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import (
    MODEL_SHORT_NAMES,
    N_CV_FOLDS,
    PATHS,
    PROBE_L2_SWEEP,
    SEEDS,
    ensure_all_dirs,
)
from src.evaluation import load_evaluation_results
from src.extraction import get_states_at_normalized_positions, load_hidden_states


def run_probe_regularization_ablation() -> dict:
    """Sweep L2 regularization for linear probes at peak layers.

    Returns
    -------
    dict
        Results dictionary that was saved to disk.
    """
    ensure_all_dirs()

    # ------------------------------------------------------------------
    # Load peak layers from Exp 2 results
    # ------------------------------------------------------------------
    exp2_path = os.path.join(str(PATHS["probes_linear"]), "exp02_all_results.json")
    peak_layers: dict[str, int] = {}
    if os.path.exists(exp2_path):
        with open(exp2_path, "r", encoding="utf-8") as f:
            exp2_results = json.load(f)
        for entry in exp2_results:
            peak_layers[entry["model"]] = entry["peak_layer_idx"]
    else:
        # Fallback: use layer index 10 (mid-range) for all models
        for name in MODEL_SHORT_NAMES:
            peak_layers[name] = 10

    all_model_results = []

    for model_name in MODEL_SHORT_NAMES:
        print(f"\n--- Model: {model_name} ---")
        peak_layer_idx = peak_layers.get(model_name, 10)

        hs_data = load_hidden_states(model_name)
        normalized = get_states_at_normalized_positions(
            hs_data["pre_decision"], hs_data["model_info"]
        )
        X = normalized[:, peak_layer_idx, :]  # (n_problems, hidden_dim)

        eval_results = load_evaluation_results(model_name)
        labels = np.array([int(r["correct"]) for r in eval_results])

        l2_sweep_results = []

        for l2_val in PROBE_L2_SWEEP:
            C = 1.0 / l2_val
            fold_accs: list[float] = []
            fold_aucs: list[float] = []

            for seed in SEEDS:
                skf = StratifiedKFold(
                    n_splits=N_CV_FOLDS, shuffle=True, random_state=seed
                )
                for train_idx, val_idx in skf.split(X, labels):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    probe = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
                    probe.fit(X_train, y_train)

                    preds = probe.predict(X_val)
                    probs = probe.predict_proba(X_val)[:, 1]

                    acc = float(np.mean(preds == y_val))
                    try:
                        auc = float(roc_auc_score(y_val, probs))
                    except ValueError:
                        auc = float("nan")

                    fold_accs.append(acc)
                    fold_aucs.append(auc)

            l2_sweep_results.append({
                "l2": l2_val,
                "C": C,
                "mean_accuracy": float(np.mean(fold_accs)),
                "std_accuracy": float(np.std(fold_accs)),
                "mean_auc": float(np.nanmean(fold_aucs)),
                "std_auc": float(np.nanstd(fold_aucs)),
                "fold_accuracies": fold_accs,
                "fold_aucs": fold_aucs,
            })

            print(
                f"  L2={l2_val:.4f}: acc={np.mean(fold_accs):.4f} "
                f"± {np.std(fold_accs):.4f}"
            )

        all_model_results.append({
            "model": model_name,
            "peak_layer_idx": peak_layer_idx,
            "l2_sweep": l2_sweep_results,
        })

    results = {
        "description": (
            "Ablation sweeping L2 regularization values for linear probes "
            "at each model's peak layer (from Exp 2). "
            f"5 seeds × {N_CV_FOLDS}-fold CV per L2 value."
        ),
        "l2_values_swept": PROBE_L2_SWEEP,
        "n_seeds": len(SEEDS),
        "n_cv_folds": N_CV_FOLDS,
        "model_results": all_model_results,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(str(PATHS["metrics_probes"]), "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regularization_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRegularization sweep results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_probe_regularization_ablation()
