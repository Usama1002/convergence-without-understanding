"""
Experiment 5: Convergence Trajectory Fitting.

Loads Exp 1 results, fits quadratic / sigmoid / piecewise-linear curves
to each (model_a, model_b, condition) → layer → CKA trajectory via
scipy.optimize.curve_fit, selects the best fit using AIC, and identifies
convergence_point and reasoning_zone per trajectory.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from src.config import PATHS, ensure_all_dirs


# ---------------------------------------------------------------------------
# Curve model definitions
# ---------------------------------------------------------------------------


def _quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x ** 2 + b * x + c


def _sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def _piecewise_linear(x: np.ndarray, x_break: float, a1: float, b1: float, a2: float) -> np.ndarray:
    x_break = np.clip(x_break, x.min() + 1e-6, x.max() - 1e-6)
    y = np.where(
        x < x_break,
        a1 * x + b1,
        a1 * x_break + b1 + a2 * (x - x_break),
    )
    return y


# ---------------------------------------------------------------------------
# AIC helper
# ---------------------------------------------------------------------------


def _aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Compute AIC for a fitted model."""
    n = len(y_true)
    residuals = y_true - y_pred
    sse = float(np.sum(residuals ** 2))
    if sse <= 0:
        sse = 1e-15
    sigma2 = sse / n
    log_likelihood = -n / 2.0 * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2)
    return -2.0 * log_likelihood + 2.0 * n_params


# ---------------------------------------------------------------------------
# Fit single trajectory
# ---------------------------------------------------------------------------


def _fit_trajectory(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit three curve families and select best by AIC.

    Returns dict with best_model, parameters, aic, and per-model results.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fits: dict[str, dict] = {}

    # --- Quadratic ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_quadratic, x, y, maxfev=5000)
        y_pred = _quadratic(x, *popt)
        aic = _aic(y, y_pred, n_params=3)
        fits["quadratic"] = {"params": popt.tolist(), "aic": aic, "y_pred": y_pred.tolist()}
    except Exception as exc:
        fits["quadratic"] = {"error": str(exc), "aic": float("inf")}

    # --- Sigmoid ---
    try:
        y_range = float(np.max(y) - np.min(y))
        p0 = [y_range, 1.0, float(np.median(x)), float(np.min(y))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _sigmoid, x, y, p0=p0, maxfev=10000,
                bounds=([-np.inf, -np.inf, x.min(), -np.inf], [np.inf, np.inf, x.max(), np.inf]),
            )
        y_pred = _sigmoid(x, *popt)
        aic = _aic(y, y_pred, n_params=4)
        fits["sigmoid"] = {"params": popt.tolist(), "aic": aic, "y_pred": y_pred.tolist()}
    except Exception as exc:
        fits["sigmoid"] = {"error": str(exc), "aic": float("inf")}

    # --- Piecewise linear ---
    try:
        x_break0 = float(np.median(x))
        slope0 = float((y[-1] - y[0]) / (x[-1] - x[0] + 1e-9))
        p0 = [x_break0, slope0, float(y[0]), 0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_piecewise_linear, x, y, p0=p0, maxfev=10000)
        y_pred = _piecewise_linear(x, *popt)
        aic = _aic(y, y_pred, n_params=4)
        fits["piecewise_linear"] = {"params": popt.tolist(), "aic": aic, "y_pred": y_pred.tolist()}
    except Exception as exc:
        fits["piecewise_linear"] = {"error": str(exc), "aic": float("inf")}

    # Select best by AIC
    best_model = min(fits, key=lambda k: fits[k].get("aic", float("inf")))

    return {"best_model": best_model, "fits": fits}


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment_5() -> dict:
    """Run trajectory fitting on Exp 1 results.

    Loads exp01_all_results.json, organizes by (model_a, model_b, condition),
    fits curve families, selects best by AIC, and identifies per-trajectory
    convergence_point and reasoning_zone.

    Returns
    -------
    dict
        Results saved to PATHS['cka_pairwise']/exp05_trajectories.json.
    """
    ensure_all_dirs()

    in_path = PATHS["cka_pairwise"] / "exp01_all_results.json"
    if not in_path.exists():
        return {"error": f"Exp 1 results not found at {in_path}"}

    with open(in_path, "r", encoding="utf-8") as f:
        exp1_data = json.load(f)

    # ------------------------------------------------------------------
    # Organise records by (model_a, model_b, condition) → list of
    # {layer_idx, cka, procrustes, mnn}
    # ------------------------------------------------------------------
    from collections import defaultdict

    trajectories: dict[str, list[dict]] = defaultdict(list)

    results_list = exp1_data if isinstance(exp1_data, list) else exp1_data.get("results", [])

    for rec in results_list:
        model_a = rec.get("model_a", "")
        model_b = rec.get("model_b", "")
        condition = rec.get("condition", "all_correct")
        layer_idx = rec.get("layer_idx", rec.get("layer", 0))
        key = f"{model_a}||{model_b}||{condition}"
        trajectories[key].append({
            "layer_idx": int(layer_idx),
            "cka": rec.get("cka"),
            "procrustes": rec.get("procrustes"),
            "mnn": rec.get("mnn"),
        })

    # Sort each trajectory by layer index
    for key in trajectories:
        trajectories[key].sort(key=lambda r: r["layer_idx"])

    # ------------------------------------------------------------------
    # Fit each trajectory
    # ------------------------------------------------------------------
    x_layers = np.arange(21, dtype=float)

    fitted_trajectories: list[dict] = []

    for key, records in trajectories.items():
        model_a, model_b, condition = key.split("||")

        # Build y from CKA values (use NaN-safe extraction)
        cka_values = [r["cka"] for r in records]
        layer_indices = [r["layer_idx"] for r in records]

        valid_mask = [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in cka_values]
        if sum(valid_mask) < 4:
            continue

        x = np.array([layer_indices[i] for i, ok in enumerate(valid_mask) if ok], dtype=float)
        y = np.array([cka_values[i] for i, ok in enumerate(valid_mask) if ok], dtype=float)

        fit_result = _fit_trajectory(x, y)

        # Convergence point: first layer where CKA > 0.9
        convergence_point = None
        for i, (xi, yi) in enumerate(zip(x.tolist(), y.tolist())):
            if yi > 0.9:
                convergence_point = int(xi)
                break

        # Reasoning zone: layer with minimum CKA
        reasoning_zone = int(x[int(np.argmin(y))]) if len(y) > 0 else None

        fitted_trajectories.append({
            "model_a": model_a,
            "model_b": model_b,
            "condition": condition,
            "n_points": len(x),
            "convergence_point": convergence_point,
            "reasoning_zone": reasoning_zone,
            "fit": fit_result,
            "cka_values": y.tolist(),
            "layer_indices": x.tolist(),
        })

    output = {
        "n_trajectories": len(fitted_trajectories),
        "trajectories": fitted_trajectories,
    }

    out_path = PATHS["cka_pairwise"] / "exp05_trajectories.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
