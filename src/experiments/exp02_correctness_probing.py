"""
Experiment 2: Correctness Probing.

For each of 14 models and each of 21 normalized layer positions:
  - 5 seeds × 5-fold stratified cross-validation
  - Trains a linear probe (LogisticRegression) and an MLP probe
  - Records accuracy and AUC for both probe types

At the peak layer (highest mean linear probe accuracy) trains a final linear
probe on all data and saves:
  - Probe weights to PATHS["probes_weights"]/{name}_peak_probe.pkl
  - Correctness direction (coef_[0]) to PATHS["probes_weights"]/{name}_correctness_direction.npy

All results saved to PATHS["probes_linear"]/exp02_all_results.json.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    MLP_HIDDEN_DIM,
    MODEL_SHORT_NAMES,
    N_CV_FOLDS,
    PATHS,
    PROBE_L2_DEFAULT,
    SEEDS,
    ensure_all_dirs,
)
from src.evaluation import load_evaluation_results
from src.extraction import get_states_at_normalized_positions, load_hidden_states

# ---------------------------------------------------------------------------
# MLP Probe
# ---------------------------------------------------------------------------


class MLPProbe(nn.Module):
    """2-layer MLP probe for binary correctness classification.

    Architecture: Linear(d, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = MLP_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MLP training helper
# ---------------------------------------------------------------------------


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = MLP_HIDDEN_DIM,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train an MLP probe and evaluate on a validation set.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, d)
    y_train : np.ndarray, shape (n_train,), binary labels
    X_val   : np.ndarray, shape (n_val, d)
    y_val   : np.ndarray, shape (n_val,), binary labels
    hidden_dim : int
    epochs  : int
    lr      : float
    seed    : int (torch init/training seed; without it MLP numbers are
              irreproducible run-to-run and architecture-sweep comparisons
              mix architecture effects with initialization noise)

    Returns
    -------
    dict with keys: accuracy, auc, model
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MLPProbe(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_v).cpu().numpy()

    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_preds = (val_probs >= 0.5).astype(int)
    y_val_np = y_val.astype(int)

    accuracy = float(np.mean(val_preds == y_val_np))
    try:
        auc = float(roc_auc_score(y_val_np, val_probs))
    except ValueError:
        auc = float("nan")

    return {"accuracy": accuracy, "auc": auc, "model": model}


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_2() -> None:
    """Run Experiment 2: correctness probing across all models and layers."""
    ensure_all_dirs()

    all_model_results: list[dict] = []

    for model_name in MODEL_SHORT_NAMES:
        print(f"\n--- Model: {model_name} ---")

        # Load hidden states and evaluation results
        hs_data = load_hidden_states(model_name)
        pre_states = hs_data["pre_decision"]   # (n_problems, n_layers+1, hidden_dim)
        model_info = hs_data["model_info"]
        normalized_states = get_states_at_normalized_positions(pre_states, model_info)
        # shape: (n_problems, 21, hidden_dim)

        eval_results = load_evaluation_results(model_name)
        labels = np.array([int(r["correct"]) for r in eval_results])  # (n_problems,)

        n_problems, n_layer_positions, hidden_dim = normalized_states.shape
        n_layer_positions = 21  # always 21

        layer_mean_lin_acc: list[float] = []  # track peak layer
        layer_results: list[dict] = []

        for layer_pos_idx in range(n_layer_positions):
            X = normalized_states[:, layer_pos_idx, :]  # (n_problems, hidden_dim)

            fold_lin_accs: list[float] = []
            fold_lin_aucs: list[float] = []
            fold_mlp_accs: list[float] = []
            fold_mlp_aucs: list[float] = []

            for seed in SEEDS:
                skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
                for train_idx, val_idx in skf.split(X, labels):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    # Standardize per fold (fit on train only); raw dimension
                    # scales differ by orders of magnitude and would dominate
                    # the L2 penalty.
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                    # --- Linear probe ---
                    C = 1.0 / PROBE_L2_DEFAULT
                    lin_probe = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
                    lin_probe.fit(X_train, y_train)
                    lin_preds = lin_probe.predict(X_val)
                    lin_probs = lin_probe.predict_proba(X_val)[:, 1]

                    lin_acc = float(np.mean(lin_preds == y_val))
                    try:
                        lin_auc = float(roc_auc_score(y_val, lin_probs))
                    except ValueError:
                        lin_auc = float("nan")

                    fold_lin_accs.append(lin_acc)
                    fold_lin_aucs.append(lin_auc)

                    # --- MLP probe ---
                    mlp_result = train_mlp_probe(X_train, y_train, X_val, y_val)
                    fold_mlp_accs.append(mlp_result["accuracy"])
                    fold_mlp_aucs.append(mlp_result["auc"])

            mean_lin_acc = float(np.mean(fold_lin_accs))
            mean_lin_auc = float(np.nanmean(fold_lin_aucs))
            mean_mlp_acc = float(np.mean(fold_mlp_accs))
            mean_mlp_auc = float(np.nanmean(fold_mlp_aucs))

            layer_mean_lin_acc.append(mean_lin_acc)
            layer_results.append({
                "layer_pos_idx": layer_pos_idx,
                "linear_probe": {
                    "mean_accuracy": mean_lin_acc,
                    "mean_auc": mean_lin_auc,
                    "fold_accuracies": fold_lin_accs,
                    "fold_aucs": fold_lin_aucs,
                },
                "mlp_probe": {
                    "mean_accuracy": mean_mlp_acc,
                    "mean_auc": mean_mlp_auc,
                    "fold_accuracies": fold_mlp_accs,
                    "fold_aucs": fold_mlp_aucs,
                },
            })
            print(
                f"  Layer {layer_pos_idx:2d}: lin_acc={mean_lin_acc:.4f}  "
                f"mlp_acc={mean_mlp_acc:.4f}"
            )

        # ------------------------------------------------------------------
        # Peak layer: train final linear probe on all data
        # ------------------------------------------------------------------
        peak_layer_idx = int(np.argmax(layer_mean_lin_acc))
        peak_acc = layer_mean_lin_acc[peak_layer_idx]
        print(
            f"  Peak layer: {peak_layer_idx} "
            f"(mean linear acc = {peak_acc:.4f})"
        )

        X_peak = normalized_states[:, peak_layer_idx, :]
        final_scaler = StandardScaler()
        X_peak_scaled = final_scaler.fit_transform(X_peak)
        C = 1.0 / PROBE_L2_DEFAULT
        final_probe = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        final_probe.fit(X_peak_scaled, labels)

        # Save probe weights (with its scaler, so the probe can be reapplied)
        probe_path = PATHS["probes_weights"] / f"{model_name}_peak_probe.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump({"scaler": final_scaler, "probe": final_probe}, f)

        # Save correctness direction mapped back to RAW activation space
        # (w . (x - mu) / sigma = (w / sigma) . x + const), since the causal
        # interventions (exp03) apply the direction to raw hidden states.
        direction = final_probe.coef_[0] / final_scaler.scale_  # (hidden_dim,)
        direction_path = PATHS["probes_weights"] / f"{model_name}_correctness_direction.npy"
        np.save(direction_path, direction)

        all_model_results.append({
            "model": model_name,
            "peak_layer_idx": peak_layer_idx,
            "peak_layer_mean_linear_acc": peak_acc,
            "n_problems": n_problems,
            "hidden_dim": int(hidden_dim),
            "layer_results": layer_results,
        })

    # Save all results
    out_path = PATHS["probes_linear"] / "exp02_all_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    run_experiment_2()
