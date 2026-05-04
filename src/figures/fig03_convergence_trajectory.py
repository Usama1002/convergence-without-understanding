"""
Figure 03: Convergence Trajectory.

Average CKA across all pairs at each layer, split by condition (all_correct,
all_incorrect, mixed). Line plot with shaded ±1 std band.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PATHS
from src.figures.fig_utils import save_figure, setup_style

CONDITION_STYLE = {
    "all_correct": {"color": "green", "label": "All Correct"},
    "all_incorrect": {"color": "red", "label": "All Incorrect"},
    "mixed": {"color": "#E9C46A", "label": "Mixed"},
}


def generate_fig03() -> str:
    """Generate and save Figure 03.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    setup_style()

    results_path = os.path.join(
        os.fspath(PATHS["cka_pairwise"]), "exp01_all_results.json"
    )
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Collect CKA values per (condition, layer)
    data: dict[str, dict[int, list[float]]] = {}
    for r in results:
        if r.get("skipped"):
            continue
        cond = r["condition"]
        lp = r["layer_pos_idx"]
        val = r.get("cka")
        if val is None:
            continue
        data.setdefault(cond, {}).setdefault(lp, []).append(val)

    n_layers = 21
    layer_indices = list(range(n_layers))
    normalized_positions = np.linspace(0.0, 1.0, n_layers)

    fig, ax = plt.subplots(figsize=(7, 4))

    for cond, style in CONDITION_STYLE.items():
        if cond not in data:
            continue
        means = []
        stds = []
        xs = []
        for lp in layer_indices:
            vals = data[cond].get(lp, [])
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                xs.append(normalized_positions[lp])

        xs_arr = np.array(xs)
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(xs_arr, means_arr, color=style["color"], label=style["label"], linewidth=1.8)
        ax.fill_between(
            xs_arr,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=style["color"],
            alpha=0.2,
        )

    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Mean Linear CKA")
    ax.set_title("CKA Convergence Trajectory by Condition")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    return save_figure(fig, "fig03_convergence_trajectory")
