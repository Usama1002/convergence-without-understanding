"""
Figure 02: CKA Heatmaps at Peak Layer.

Load exp01 results, find the peak layer (highest average CKA for all_correct),
then build 14×14 CKA matrices for correct and incorrect conditions at that layer
and display side-by-side heatmaps.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import MODEL_SHORT_NAMES, N_MODELS, PATHS
from src.figures.fig_utils import save_figure, setup_style


def _load_exp01_results() -> list[dict]:
    """Load exp01 pairwise CKA results from JSON."""
    results_path = os.path.join(
        os.fspath(PATHS["cka_pairwise"]), "exp01_all_results.json"
    )
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_cka_matrix(
    results: list[dict],
    condition: str,
    layer_pos_idx: int,
    model_names: list[str],
) -> np.ndarray:
    """Build an N×N symmetric CKA matrix for a given condition and layer."""
    n = len(model_names)
    idx_map = {name: i for i, name in enumerate(model_names)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    for r in results:
        if r.get("skipped"):
            continue
        if r["condition"] != condition:
            continue
        if r["layer_pos_idx"] != layer_pos_idx:
            continue
        i = idx_map.get(r["model_a"])
        j = idx_map.get(r["model_b"])
        if i is None or j is None:
            continue
        val = r.get("cka", np.nan)
        mat[i, j] = val
        mat[j, i] = val

    # Replace remaining NaN off-diagonal with 0
    mat = np.nan_to_num(mat, nan=0.0)
    np.fill_diagonal(mat, 1.0)
    return mat


def generate_fig02() -> str:
    """Generate and save Figure 02.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    setup_style()
    results = _load_exp01_results()

    # -----------------------------------------------------------------------
    # Find peak layer: highest average CKA for all_correct across all pairs
    # -----------------------------------------------------------------------
    layer_cka: dict[int, list[float]] = {}
    for r in results:
        if r.get("skipped"):
            continue
        if r["condition"] != "all_correct":
            continue
        lp = r["layer_pos_idx"]
        layer_cka.setdefault(lp, []).append(r.get("cka", 0.0))

    peak_layer = max(layer_cka, key=lambda k: float(np.mean(layer_cka[k])))

    # -----------------------------------------------------------------------
    # Build matrices
    # -----------------------------------------------------------------------
    mat_correct = _build_cka_matrix(results, "all_correct", peak_layer, MODEL_SHORT_NAMES)
    mat_incorrect = _build_cka_matrix(results, "all_incorrect", peak_layer, MODEL_SHORT_NAMES)

    short_labels = [n.replace("-Instruct", "") for n in MODEL_SHORT_NAMES]

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mat, title in zip(
        axes,
        [mat_correct, mat_incorrect],
        ["All Correct", "All Incorrect"],
    ):
        sns.heatmap(
            mat,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap="YlOrRd",
            xticklabels=short_labels,
            yticklabels=short_labels,
            square=True,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Linear CKA"},
        )
        ax.set_title(f"{title} — Layer {peak_layer}")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.suptitle("Cross-Model CKA Similarity at Peak Layer", fontsize=12, y=1.01)
    fig.tight_layout()

    return save_figure(fig, "fig02_cka_heatmap")
