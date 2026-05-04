"""Generate all Paper B figures.

Paper B: "Convergence Without Understanding"
8 figures covering the three reveals + baselines.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

OUT_DIR = Path("data/figures")

# NeurIPS style
NEURIPS_RC = {
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
}


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}", flush=True)


def fig1_difficulty_inversion():
    """Figure 1: CKA vs fraction of models correct (main finding)."""
    plt.rcParams.update(NEURIPS_RC)

    with open("data/metrics/extended/b6_agreement_threshold_cka.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: CKA per exact n_correct
    thresholds = data["thresholds"]
    x = [t["n_correct"] for t in thresholds]
    cka = [t["mean_cka"] for t in thresholds]
    mnn = [t["mean_mnn"] for t in thresholds]
    sizes = [t["n_problems"] * 2 for t in thresholds]

    ax1.scatter(x, cka, s=sizes, c=x, cmap="RdYlGn", edgecolor="black", linewidth=0.5, zorder=5)
    ax1.plot(x, cka, color="gray", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Number of Models Correct (out of 14)")
    ax1.set_ylabel("Mean CKA")
    ax1.set_title("(a) CKA vs Problem Difficulty", fontweight="bold")
    ax1.set_xlim(-0.5, 14.5)
    ax1.grid(True, alpha=0.3)

    # Add annotation for the inversion
    ax1.annotate("Models converge\nMORE on failures",
                xy=(2, 0.96), fontsize=8, ha="center", color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax1.annotate("Models DIVERGE\non successes",
                xy=(12, 0.87), fontsize=8, ha="center", color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Panel B: MNN per exact n_correct
    ax2.scatter(x, mnn, s=sizes, c=x, cmap="RdYlGn", edgecolor="black", linewidth=0.5, zorder=5)
    ax2.plot(x, mnn, color="gray", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Number of Models Correct (out of 14)")
    ax2.set_ylabel("Mean MNN")
    ax2.set_title("(b) MNN vs Problem Difficulty", fontweight="bold")
    ax2.set_xlim(-0.5, 14.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Difficulty Inversion: Models Converge More on Problems They Get Wrong",
                fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig1_difficulty_inversion")


def fig2_per_domain_difficulty():
    """Figure 2: Per-domain difficulty inversion (4 panels)."""
    plt.rcParams.update(NEURIPS_RC)

    with open("data/metrics/extended/b1_per_domain_difficulty.json") as f:
        data = json.load(f)

    domains = sorted(data.keys())
    n_domains = len(domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(4 * n_domains, 4), sharey=True)
    if n_domains == 1:
        axes = [axes]

    domain_labels = {"math": "Math (GSM8K)", "science": "Science (ARC-C)",
                     "truthfulness": "Truthfulness (TQA)", "commonsense": "Commonsense (HS)"}

    for ax, domain in zip(axes, domains):
        strata = data[domain]
        x = [s["n_correct"] for s in strata]
        y = [s["mean_cka"] for s in strata]
        sizes = [s["n_problems"] * 5 for s in strata]

        ax.scatter(x, y, s=sizes, c=x, cmap="RdYlGn", edgecolor="black", linewidth=0.5)
        ax.plot(x, y, color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Models Correct")
        ax.set_title(domain_labels.get(domain, domain), fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 14.5)

    axes[0].set_ylabel("Mean CKA")
    fig.suptitle("Difficulty Inversion Across Reasoning Domains", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig2_per_domain_difficulty")


def fig3_per_layer_difficulty():
    """Figure 3: Heatmap of CKA per layer per difficulty bin."""
    plt.rcParams.update(NEURIPS_RC)

    with open("data/metrics/extended/b7_per_layer_difficulty.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Panel A: CKA trajectories by difficulty bin
    colors = {"hard (0-4)": "#E63946", "medium-hard (5-7)": "#F4A261",
              "medium-easy (8-10)": "#2A9D8F", "easy (11-14)": "#264653"}

    for bin_name in ["hard (0-4)", "medium-hard (5-7)", "medium-easy (8-10)", "easy (11-14)"]:
        if bin_name in data:
            ckas = data[bin_name]["cka_per_layer"]
            x = np.linspace(0, 1, len(ckas))
            ax1.plot(x, ckas, label=f"{bin_name} (n={data[bin_name]['n_problems']})",
                    color=colors[bin_name], linewidth=2)

    ax1.set_xlabel("Normalized Layer Depth")
    ax1.set_ylabel("Mean CKA")
    ax1.set_title("(a) CKA Trajectory by Difficulty", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel B: Inversion strength per layer
    if "inversion_per_layer" in data:
        inv = data["inversion_per_layer"]
        x = np.linspace(0, 1, len(inv))
        ax2.bar(x, inv, width=0.04, color=["#E63946" if v > 0 else "#2A9D8F" for v in inv])
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_xlabel("Normalized Layer Depth")
        ax2.set_ylabel("CKA(hard) - CKA(easy)")
        ax2.set_title("(b) Inversion Strength per Layer", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        peak = data.get("peak_inversion_layer", 0)
        ax2.annotate(f"Peak at layer {peak}",
                    xy=(peak / 20, max(inv)), fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.tight_layout()
    _save(fig, "fig3_per_layer_difficulty")


def fig4_pre_post_decision():
    """Figure 4: Pre vs post decision CKA."""
    plt.rcParams.update(NEURIPS_RC)

    with open("data/metrics/pre_vs_post/exp10_results.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    pre_cka = data.get("mean_cka_pre_decision", [])
    post_cka = data.get("mean_cka_post_decision", [])

    if isinstance(pre_cka, list) and isinstance(post_cka, list):
        x = np.linspace(0, 1, len(pre_cka))
        ax.plot(x, pre_cka, label="Pre-decision", color="#2A9D8F", linewidth=2.5)
        ax.plot(x, post_cka, label="Post-decision", color="#E63946", linewidth=2.5)
        ax.fill_between(x, pre_cka, post_cka, alpha=0.15, color="gray")

        mean_pre = np.mean(pre_cka)
        mean_post = np.mean(post_cka)
    else:
        mean_pre = float(pre_cka) if not isinstance(pre_cka, list) else np.mean(pre_cka)
        mean_post = float(post_cka) if not isinstance(post_cka, list) else np.mean(post_cka)
        ax.bar(["Pre-decision", "Post-decision"], [mean_pre, mean_post],
               color=["#2A9D8F", "#E63946"])

    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Mean Cross-Model CKA")
    ax.set_title("The Generation Gap: Convergence Vanishes After Answer Production",
                fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(f"Pre: {mean_pre:.3f}\nPost: {mean_post:.3f}\nDrop: {mean_pre-mean_post:.3f}",
               xy=(0.7, 0.5), fontsize=9,
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    _save(fig, "fig4_pre_post_decision")


def fig5_transfer_vs_causal():
    """Figure 5: Transfer probe accuracy vs causal flip rate (the paradox)."""
    plt.rcParams.update(NEURIPS_RC)

    # Load probe data
    with open("data/metrics/probes/linear/exp02_all_results.json") as f:
        probe_data = json.load(f)

    # Load causal data
    with open("data/metrics/causal/exp03_summary.json") as f:
        causal_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Probe accuracy per model
    models = []
    probe_accs = []
    for entry in probe_data:
        models.append(entry["model"])
        probe_accs.append(entry["peak_layer_mean_linear_acc"])

    y_pos = range(len(models))
    ax1.barh(y_pos, probe_accs, color="#2A9D8F", alpha=0.8)
    ax1.axvline(x=0.5, color="red", linestyle="--", label="Chance (50%)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m[:15] for m in models], fontsize=7)
    ax1.set_xlabel("Peak Probe Accuracy")
    ax1.set_title("(a) Correctness Is Detectable", fontweight="bold")
    ax1.legend()

    # Panel B: Causal flip rates
    causal_models = []
    flip_rates = []
    for model_name, model_data in causal_data.get("models", {}).items():
        causal_models.append(model_name)
        # Get flip rate at magnitude 1.0
        abl = model_data.get("ablation_flip_rates", {}).get("1.0", 0)
        flip_rates.append(abl)

    y_pos2 = range(len(causal_models))
    ax2.barh(y_pos2, flip_rates, color="#E63946", alpha=0.8)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([m[:15] for m in causal_models], fontsize=7)
    ax2.set_xlabel("Ablation Flip Rate")
    ax2.set_title("(b) But Not Causally Used", fontweight="bold")

    fig.suptitle("The Paradox: Correctness Is Encoded but Not Causal",
                fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig5_transfer_vs_causal")


def fig6_random_baseline():
    """Figure 6: Random model CKA baseline."""
    plt.rcParams.update(NEURIPS_RC)

    with open("data/metrics/extended/b2_random_model_baseline.json") as f:
        data = json.load(f)

    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Random vs\nRandom", "Trained vs\nTrained", "Trained vs\nRandom"]
    values = [summary["mean_random_vs_random"],
              summary["mean_trained_vs_trained"],
              summary["mean_trained_vs_random"]]
    colors = ["#E63946", "#2A9D8F", "#F4A261"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean CKA")
    ax.set_title("Random Untrained Models Show Higher CKA Than Trained Models",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotation
    ax.annotate("Convergence is\narchitectural,\nnot learned!",
               xy=(0, summary["mean_random_vs_random"]),
               xytext=(0.5, 0.5), fontsize=9, color="red",
               arrowprops=dict(arrowstyle="->", color="red"),
               bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.tight_layout()
    _save(fig, "fig6_random_baseline")


def fig7_embedding_vs_deep():
    """Figure 7: Embedding-only vs deep CKA."""
    plt.rcParams.update(NEURIPS_RC)

    with open("data/metrics/extended/b3_embedding_cka.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    layers = ["Embedding\n(Layer 0)", "Mid Layer\n(~50%)", "Deep Layer\n(Last)"]
    values = [data["embedding_layer"]["mean_cka"],
              data["mid_layer"]["mean_cka"],
              data["deep_layer"]["mean_cka"]]
    errors = [data["embedding_layer"]["std_cka"],
              data["mid_layer"]["std_cka"],
              data["deep_layer"]["std_cka"]]
    colors = ["#264653", "#2A9D8F", "#E9C46A"]

    bars = ax.bar(layers, values, yerr=errors, color=colors, edgecolor="black",
                  linewidth=0.5, capsize=5, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean CKA")
    ax.set_title("Convergence Emerges in Transformer Layers, Not From Tokenizers",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, "fig7_embedding_vs_deep")


def generate_all_figures():
    """Generate all Paper B figures."""
    print("Generating Paper B figures...", flush=True)
    fig1_difficulty_inversion()
    fig2_per_domain_difficulty()
    fig3_per_layer_difficulty()
    fig4_pre_post_decision()
    fig5_transfer_vs_causal()
    fig6_random_baseline()
    fig7_embedding_vs_deep()
    print(f"\nAll figures saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    generate_all_figures()
