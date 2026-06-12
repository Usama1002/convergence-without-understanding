"""
Figure 06: Scale Convergence Scatter Plot.

Load exp07 results. Scatter plot of scale_gap vs mean_cka per family pair,
colored by family.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PATHS
from src.figures.fig_utils import COLORS, save_figure, setup_style


def _exp07_results_path() -> str:
    scale_dir = os.fspath(PATHS["metrics_scale"])
    candidates = [
        os.path.join(scale_dir, "exp07_all_results.json"),
        os.path.join(scale_dir, "exp07_results.json"),
        os.path.join(scale_dir, "scale_results.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Return first candidate even if missing (will raise on open)
    return candidates[0]


def generate_fig06() -> str:
    """Generate and save Figure 06.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    setup_style()

    with open(_exp07_results_path(), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expected structure: list of dicts with keys:
    # family, model_i, model_j, scale_gap, mean_cka  (or similar)
    # Flatten if nested
    records: list[dict] = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item.setdefault("family", key)
                        records.append(item)
            elif isinstance(val, dict):
                val.setdefault("family", key)
                records.append(val)

    fig, ax = plt.subplots(figsize=(7, 5))

    seen_families: set[str] = set()
    for r in records:
        family = r.get("family", "Other")
        color = COLORS.get(family, "#888888")
        scale_gap = r.get("scale_gap") or r.get("param_gap") or r.get("params_gap")
        mean_cka = r.get("mean_cka") or r.get("avg_cka") or r.get("cka")
        # exp07 records carry params_a/params_b_val and per-layer metrics
        # instead of the aliases above; derive the values from those.
        if scale_gap is None and r.get("params_a") is not None and r.get("params_b_val") is not None:
            scale_gap = abs(float(r["params_b_val"]) - float(r["params_a"]))
        if mean_cka is None and r.get("layer_metrics"):
            vals = [
                lm["cka"] for lm in r["layer_metrics"]
                if isinstance(lm, dict) and lm.get("cka") is not None
            ]
            mean_cka = float(np.mean(vals)) if vals else None
        if scale_gap is None or mean_cka is None:
            continue
        label = family if family not in seen_families else None
        ax.scatter(
            float(scale_gap),
            float(mean_cka),
            color=color,
            label=label,
            alpha=0.75,
            s=50,
            edgecolors="white",
            linewidths=0.4,
        )
        if label:
            seen_families.add(family)

    ax.set_xlabel("Scale Gap (|params_i − params_j|, B)")
    ax.set_ylabel("Mean CKA")
    ax.set_title("Scale Gap vs Representational Similarity")
    ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    return save_figure(fig, "fig06_scale_convergence")
