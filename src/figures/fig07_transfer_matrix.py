"""
Figure 07: Transfer Accuracy Matrix.

Load transfer accuracy matrix + clustering from exp06 results.
Reorder by cluster, display as heatmap with annotations.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import MODEL_SHORT_NAMES, PATHS
from src.figures.fig_utils import save_figure, setup_style


def _transfer_paths() -> tuple[str, str]:
    """Return (matrix_path, clustering_path) for exp06 transfer results."""
    transfer_dir = os.fspath(PATHS["metrics_transfer"])
    matrix_candidates = [
        os.path.join(transfer_dir, "exp06_summary.json"),
        os.path.join(transfer_dir, "exp06_transfer_matrix.json"),
        os.path.join(transfer_dir, "transfer_matrix.json"),
    ]
    cluster_candidates = [
        os.path.join(transfer_dir, "exp06_clustering.json"),
        os.path.join(transfer_dir, "clustering.json"),
    ]
    matrix_path = next((c for c in matrix_candidates if os.path.exists(c)), matrix_candidates[0])
    cluster_path = next((c for c in cluster_candidates if os.path.exists(c)), cluster_candidates[0])
    return matrix_path, cluster_path


def generate_fig07() -> str:
    """Generate and save Figure 07.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    setup_style()

    transfer_dir = os.fspath(PATHS["metrics_transfer"])
    n = len(MODEL_SHORT_NAMES)
    model_names = MODEL_SHORT_NAMES[:]

    # Load transfer accuracy matrix from .npy
    npy_path = os.path.join(transfer_dir, "exp06_transfer_accuracy_matrix.npy")
    if os.path.exists(npy_path):
        mat = np.load(npy_path)
    else:
        mat = np.zeros((n, n))

    # Load clustering
    cluster_order = list(range(n))
    cluster_path = os.path.join(transfer_dir, "exp06_clustering.json")
    if os.path.exists(cluster_path):
        with open(cluster_path) as f:
            cluster_data = json.load(f)
        if isinstance(cluster_data, dict) and "cluster_order" in cluster_data:
            cluster_order = cluster_data["cluster_order"]
        elif isinstance(cluster_data, dict) and "labels" in cluster_data:
            # Sort by cluster label
            labels = cluster_data["labels"]
            cluster_order = sorted(range(len(labels)), key=lambda i: labels[i])

    # Reorder matrix
    reordered_mat = mat[np.ix_(cluster_order, cluster_order)]
    reordered_labels = [model_names[i] for i in cluster_order]
    short_labels = [lb.split("-")[0] + "-" + lb.split("-")[-1] if "-" in lb else lb
                    for lb in reordered_labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        reordered_mat,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
        xticklabels=short_labels,
        yticklabels=short_labels,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Transfer Accuracy"},
    )
    ax.set_title("Cross-Model Transfer Accuracy (Reordered by Cluster)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()

    return save_figure(fig, "fig07_transfer_matrix")
