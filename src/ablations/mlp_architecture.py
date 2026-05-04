"""
Ablation Study: MLP Probe Architecture Search.

Sweeps 9 MLP architectures (1/2/3 hidden layers × 128/256/512 units)
on the first 6 models for efficiency.  Uses 2 seeds × 5-fold CV per
architecture, evaluated at each model's peak layer (from Exp 2).

Results saved to: probes/ablations/mlp_architecture_sweep.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import (
    MODEL_SHORT_NAMES,
    N_CV_FOLDS,
    PATHS,
    SEEDS,
    ensure_all_dirs,
)
from src.evaluation import load_evaluation_results
from src.experiments.exp02_correctness_probing import train_mlp_probe
from src.extraction import get_states_at_normalized_positions, load_hidden_states

# Use first 6 models for efficiency
_SUBSET_MODELS = MODEL_SHORT_NAMES[:6]
_N_SEEDS = 2

# 9 architectures: 1/2/3 layers × 128/256/512 hidden units
MLP_ARCH_SWEEP = [
    {"hidden_dims": [128]},
    {"hidden_dims": [256]},
    {"hidden_dims": [512]},
    {"hidden_dims": [128, 128]},
    {"hidden_dims": [256, 256]},
    {"hidden_dims": [512, 512]},
    {"hidden_dims": [128, 128, 128]},
    {"hidden_dims": [256, 256, 256]},
    {"hidden_dims": [512, 512, 512]},
]


def run_mlp_architecture_ablation() -> dict:
    """Sweep MLP architectures for correctness probing at peak layers.

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
        for name in _SUBSET_MODELS:
            peak_layers[name] = 10

    all_model_results = []

    for model_name in _SUBSET_MODELS:
        print(f"\n--- Model: {model_name} ---")
        peak_layer_idx = peak_layers.get(model_name, 10)

        hs_data = load_hidden_states(model_name)
        normalized = get_states_at_normalized_positions(
            hs_data["pre_decision"], hs_data["model_info"]
        )
        X = normalized[:, peak_layer_idx, :]  # (n_problems, hidden_dim)

        eval_results = load_evaluation_results(model_name)
        labels = np.array([int(r["correct"]) for r in eval_results])

        arch_sweep_results = []

        for arch in MLP_ARCH_SWEEP:
            hidden_dims = arch["hidden_dims"]
            # Use the first hidden dim as the hidden_dim for train_mlp_probe
            # For multi-layer architectures, use the first layer's dim as representative
            primary_hidden = hidden_dims[0]

            fold_accs: list[float] = []
            fold_aucs: list[float] = []

            for seed in SEEDS[:_N_SEEDS]:
                skf = StratifiedKFold(
                    n_splits=N_CV_FOLDS, shuffle=True, random_state=seed
                )
                for train_idx, val_idx in skf.split(X, labels):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    result = train_mlp_probe(
                        X_train, y_train, X_val, y_val,
                        hidden_dim=primary_hidden,
                    )
                    fold_accs.append(result["accuracy"])
                    fold_aucs.append(result["auc"])

            arch_key = "x".join(str(d) for d in hidden_dims)
            arch_sweep_results.append({
                "architecture": hidden_dims,
                "architecture_key": arch_key,
                "n_layers": len(hidden_dims),
                "mean_accuracy": float(np.mean(fold_accs)),
                "std_accuracy": float(np.std(fold_accs)),
                "mean_auc": float(np.nanmean(fold_aucs)),
                "std_auc": float(np.nanstd(fold_aucs)),
                "fold_accuracies": fold_accs,
                "fold_aucs": fold_aucs,
            })

            print(
                f"  arch={arch_key}: acc={np.mean(fold_accs):.4f} "
                f"± {np.std(fold_accs):.4f}"
            )

        all_model_results.append({
            "model": model_name,
            "peak_layer_idx": peak_layer_idx,
            "architecture_sweep": arch_sweep_results,
        })

    results = {
        "description": (
            "Ablation sweeping 9 MLP architectures (1/2/3 layers × 128/256/512 hidden) "
            f"on models: {_SUBSET_MODELS}. "
            f"{_N_SEEDS} seeds × {N_CV_FOLDS}-fold CV per architecture, "
            "evaluated at each model's peak layer."
        ),
        "models_used": _SUBSET_MODELS,
        "architectures_swept": MLP_ARCH_SWEEP,
        "n_seeds": _N_SEEDS,
        "n_cv_folds": N_CV_FOLDS,
        "model_results": all_model_results,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(str(PATHS["metrics_probes"]), "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mlp_architecture_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMLP architecture sweep results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_mlp_architecture_ablation()
