"""
Figure 04: Probe Accuracy vs Normalized Layer Depth.

Load exp02 results. Plot linear and MLP probe accuracy vs normalized layer depth,
one curve per model, colored by family. Side-by-side panels.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_REGISTRY, PATHS
from src.figures.fig_utils import COLORS, save_figure, setup_style


def generate_fig04() -> str:
    """Generate and save Figure 04.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    setup_style()

    results_path = os.path.join(
        os.fspath(PATHS["probes_linear"]), "exp02_all_results.json"
    )
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Build a lookup: model -> layer_pos_idx -> {linear_acc, mlp_acc}
    # results is expected to be a list of dicts with keys:
    #   model_name, layer_pos_idx, linear_accuracy, mlp_accuracy (mean across seeds/folds)
    model_data: dict[str, dict[int, dict]] = {}
    for r in results:
        model = r.get("model_name") or r.get("model")
        if model is None:
            continue
        lp = r.get("layer_pos_idx")
        if lp is None:
            continue
        model_data.setdefault(model, {})[lp] = r

    n_layers = 21
    normalized_positions = np.linspace(0.0, 1.0, n_layers)

    family_map = {m["short_name"]: m["family"] for m in MODEL_REGISTRY}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    probe_types = [
        ("linear_accuracy", "Linear Probe"),
        ("mlp_accuracy", "MLP Probe"),
    ]

    seen_families: set[str] = set()
    for ax, (acc_key, panel_title) in zip(axes, probe_types):
        for model_name, layer_dict in model_data.items():
            family = family_map.get(model_name, "Other")
            color = COLORS.get(family, "#888888")
            xs = []
            ys = []
            for lp in sorted(layer_dict.keys()):
                val = layer_dict[lp].get(acc_key)
                if val is None:
                    # Try nested key structures
                    val = layer_dict[lp].get("mean_" + acc_key)
                if val is not None:
                    xs.append(normalized_positions[lp])
                    ys.append(float(val))
            if not xs:
                continue
            label = family if family not in seen_families else None
            ax.plot(xs, ys, color=color, label=label, linewidth=1.2, alpha=0.85)
            if label:
                seen_families.add(family)

        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Accuracy")
        ax.set_title(panel_title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Probe Accuracy vs Layer Depth", fontsize=12)
    fig.tight_layout()

    return save_figure(fig, "fig04_probe_accuracy")
