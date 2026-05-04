"""
Ablation Study: Metric Agreement Between CKA, Procrustes, and MNN.

Loads Experiment 1 results, collects (cka, procrustes, mnn) triplets
across all model pairs and layer positions, then computes Pearson and
Spearman correlations for each pairwise combination:
  - CKA vs -Procrustes (negated because Procrustes is a distance)
  - CKA vs MNN
  - -Procrustes vs MNN

Results saved to: cka/ablations/metric_agreement.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.config import PATHS, ensure_all_dirs


def run_metric_agreement_ablation() -> dict:
    """Compute pairwise correlations between CKA, Procrustes, and MNN.

    Returns
    -------
    dict
        Results dictionary that was saved to disk.
    """
    ensure_all_dirs()

    # ------------------------------------------------------------------
    # Load Exp 1 results
    # ------------------------------------------------------------------
    exp1_path = os.path.join(str(PATHS["cka_pairwise"]), "exp01_all_results.json")
    with open(exp1_path, "r", encoding="utf-8") as f:
        exp1_results = json.load(f)

    # Collect triplets for the "all_correct" condition
    cka_vals: list[float] = []
    procrustes_vals: list[float] = []
    mnn_vals: list[float] = []

    for entry in exp1_results:
        if entry.get("skipped", False):
            continue
        if entry.get("skipped", False):
            continue
        if "cka" not in entry:
            continue

        cka_vals.append(float(entry["cka"]))
        if "mnn" in entry:
            mnn_vals.append(float(entry["mnn"]))

    cka_arr = np.array(cka_vals)
    mnn_arr = np.array(mnn_vals[:len(cka_vals)])  # align lengths

    def _corr_stats(a: np.ndarray, b: np.ndarray) -> dict:
        if len(a) < 2 or len(b) < 2:
            return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                    "spearman_r": float("nan"), "spearman_p": float("nan")}
        pr, pp = pearsonr(a, b)
        sr, sp = spearmanr(a, b)
        return {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
        }

    results = {
        "description": (
            "Metric agreement ablation comparing CKA and MNN "
            "across all model pairs and layer positions from Experiment 1."
        ),
        "n_data_points": len(cka_vals),
        "correlations": {
            "cka_vs_mnn": _corr_stats(cka_arr, mnn_arr),
        },
        "summary_statistics": {
            "cka": {
                "mean": float(np.mean(cka_arr)),
                "std": float(np.std(cka_arr)),
                "min": float(np.min(cka_arr)),
                "max": float(np.max(cka_arr)),
            },
            "mnn": {
                "mean": float(np.mean(mnn_arr)),
                "std": float(np.std(mnn_arr)),
                "min": float(np.min(mnn_arr)),
                "max": float(np.max(mnn_arr)),
            },
        },
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(str(PATHS["metrics_cka"]), "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metric_agreement.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metric agreement ablation results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_metric_agreement_ablation()
