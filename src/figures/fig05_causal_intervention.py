"""
Figure 05: Causal Intervention — Ablation and Amplification Flip Rates.

Load exp03 summary, produce bar charts for ablation and amplification flip rates,
grouped by causal magnitude, colored by model family.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CAUSAL_MAGNITUDES, MODEL_REGISTRY, PATHS
from src.figures.fig_utils import COLORS, save_figure, setup_style


def _causal_summary_path() -> str:
    causal_dir = os.path.join(os.fspath(PATHS["metrics_causal"]), "ablation")
    summary = os.path.join(causal_dir, "exp03_summary.json")
    if os.path.exists(summary):
        return summary
    # Fallback to metrics_causal directly
    summary2 = os.path.join(os.fspath(PATHS["metrics_causal"]), "exp03_summary.json")
    return summary2


def generate_fig05() -> str:
    """Generate and save Figure 05.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    setup_style()

    summary_path = _causal_summary_path()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # summary expected structure:
    # list of dicts: {model_name, magnitude, ablation_flip_rate, amplification_flip_rate}
    # OR a dict keyed by model_name

    if isinstance(summary, dict):
        records = []
        for model_name, model_data in summary.items():
            if isinstance(model_data, list):
                for entry in model_data:
                    entry["model_name"] = model_name
                    records.append(entry)
            elif isinstance(model_data, dict):
                model_data["model_name"] = model_name
                records.append(model_data)
        summary = records

    family_map = {m["short_name"]: m["family"] for m in MODEL_REGISTRY}
    magnitudes = sorted({float(r.get("magnitude", 1.0)) for r in summary})
    if not magnitudes:
        magnitudes = CAUSAL_MAGNITUDES

    model_names = sorted({r.get("model_name", "") for r in summary if r.get("model_name")})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    intervention_types = [
        ("ablation_flip_rate", "Ablation Flip Rate"),
        ("amplification_flip_rate", "Amplification Flip Rate"),
    ]

    bar_width = 0.6 / max(len(model_names), 1)
    x_pos = np.arange(len(magnitudes))

    for ax, (flip_key, panel_title) in zip(axes, intervention_types):
        for k, model_name in enumerate(model_names):
            family = family_map.get(model_name, "Other")
            color = COLORS.get(family, "#888888")
            rates = []
            for mag in magnitudes:
                val = None
                for r in summary:
                    if r.get("model_name") == model_name and abs(
                        float(r.get("magnitude", -999)) - mag
                    ) < 1e-6:
                        val = r.get(flip_key)
                        break
                rates.append(float(val) if val is not None else 0.0)

            offset = (k - len(model_names) / 2) * bar_width
            ax.bar(
                x_pos + offset,
                rates,
                width=bar_width * 0.9,
                color=color,
                label=model_name,
                alpha=0.85,
            )

        ax.set_xlabel("Intervention Magnitude")
        ax.set_ylabel("Flip Rate")
        ax.set_title(panel_title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(m) for m in magnitudes])
        ax.set_ylim(0.0, 1.05)

    # Shared legend — use family colors instead of per-model
    seen: set[str] = set()
    legend_handles = []
    for model_name in model_names:
        family = family_map.get(model_name, "Other")
        if family not in seen:
            import matplotlib.patches as mpatches
            legend_handles.append(
                mpatches.Patch(color=COLORS.get(family, "#888888"), label=family)
            )
            seen.add(family)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Causal Intervention Flip Rates", fontsize=12)
    fig.tight_layout()

    return save_figure(fig, "fig05_causal_intervention")
