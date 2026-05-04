"""Generate all Paper B figures — v2 (publication quality).

Rules:
- No figure-wide suptitle (captions in LaTeX handle this)
- Clean, professional aesthetic
- High data-to-ink ratio
- Legible fonts at publication column width
- Consistent color palette
- No amateur annotations
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

OUT_DIR = Path("figures/output")

# Consistent professional palette
PAL = {
    "hard": "#c0392b",
    "medium": "#e67e22",
    "easy": "#27ae60",
    "pre": "#2980b9",
    "post": "#c0392b",
    "bar1": "#2c3e50",
    "bar2": "#2980b9",
    "bar3": "#e67e22",
    "gray": "#7f8c8d",
}

RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", pad_inches=0.05)
    fig.savefig(OUT_DIR / f"{name}.png", pad_inches=0.05)
    plt.close(fig)
    print(f"  {name}", flush=True)


def fig1_difficulty_inversion():
    """CKA and MNN vs number of models correct."""
    plt.rcParams.update(RC)

    with open("data/metrics/extended/b6_agreement_threshold_cka.json") as f:
        data = json.load(f)

    thresholds = data["thresholds"]
    x = [t["n_correct"] for t in thresholds]
    cka = [t["mean_cka"] for t in thresholds]
    mnn = [t["mean_mnn"] for t in thresholds]
    n_probs = [t["n_problems"] for t in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # CKA panel
    colors_cka = [plt.cm.RdYlGn(nc / 14.0) for nc in x]
    ax1.scatter(x, cka, s=[n * 3 for n in n_probs], c=colors_cka,
                edgecolor="0.3", linewidth=0.4, zorder=5)
    # Trend line
    z = np.polyfit([xi for xi, c in zip(x, cka) if 1 <= xi <= 13],
                   [c for xi, c in zip(x, cka) if 1 <= xi <= 13], 1)
    x_trend = np.linspace(1, 13, 50)
    ax1.plot(x_trend, np.polyval(z, x_trend), color=PAL["gray"], linewidth=1,
             linestyle="--", alpha=0.7)
    ax1.set_xlabel("Number of models correct (out of 14)")
    ax1.set_ylabel("Mean CKA")
    ax1.set_title("(a) CKA vs. problem difficulty")
    ax1.set_xlim(-0.5, 14.5)
    ax1.set_ylim(0.78, 1.0)

    # MNN panel — exclude n=14/14 outlier (only 5 problems, unreliable)
    x_mnn = [xi for xi, n in zip(x, n_probs) if xi < 14]
    mnn_filt = [m for xi, m in zip(x, mnn) if xi < 14]
    n_filt = [n for xi, n in zip(x, n_probs) if xi < 14]
    colors_mnn = [plt.cm.RdYlGn(nc / 14.0) for nc in x_mnn]
    ax2.scatter(x_mnn, mnn_filt, s=[n * 3 for n in n_filt], c=colors_mnn,
                edgecolor="0.3", linewidth=0.4, zorder=5)
    z2 = np.polyfit([xi for xi, m in zip(x_mnn, mnn_filt) if 1 <= xi <= 13],
                    [m for xi, m in zip(x_mnn, mnn_filt) if 1 <= xi <= 13], 1)
    ax2.plot(x_trend, np.polyval(z2, x_trend), color=PAL["gray"], linewidth=1,
             linestyle="--", alpha=0.7)
    ax2.set_xlabel("Number of models correct (out of 14)")
    ax2.set_ylabel("Mean MNN")
    ax2.set_title("(b) MNN vs. problem difficulty")
    ax2.set_xlim(-0.5, 14.5)

    plt.tight_layout(w_pad=2.5)
    _save(fig, "fig1_difficulty_inversion")


def fig2_per_domain_difficulty():
    """Per-domain difficulty inversion (4 panels)."""
    plt.rcParams.update(RC)

    with open("data/metrics/extended/b1_per_domain_difficulty.json") as f:
        data = json.load(f)

    domain_labels = {
        "math": "Math (GSM8K)",
        "science": "Science (ARC-C)",
        "truthfulness": "Truthfulness (TQA)",
        "commonsense": "Commonsense (HS)",
    }
    domains = sorted(data.keys())

    fig, axes = plt.subplots(1, len(domains), figsize=(7, 2.5), sharey=True)

    for ax, domain in zip(axes, domains):
        strata = data[domain]
        x = [s["n_correct"] for s in strata]
        y = [s["mean_cka"] for s in strata]
        sizes = [max(s["n_problems"] * 4, 15) for s in strata]
        colors = [plt.cm.RdYlGn(nc / 14.0) for nc in x]

        ax.scatter(x, y, s=sizes, c=colors, edgecolor="0.3", linewidth=0.4, zorder=5)

        # Trend line for points with enough data
        valid = [(xi, yi) for xi, yi, s in zip(x, y, strata) if s["n_problems"] >= 3 and 1 <= xi <= 13]
        if len(valid) >= 3:
            vx, vy = zip(*valid)
            z = np.polyfit(vx, vy, 1)
            xt = np.linspace(min(vx), max(vx), 50)
            ax.plot(xt, np.polyval(z, xt), color=PAL["gray"], linewidth=1,
                    linestyle="--", alpha=0.6)
            slope_dir = "inverted" if z[0] < -0.001 else "normal" if z[0] > 0.001 else "flat"
            ax.text(0.95, 0.05, slope_dir, transform=ax.transAxes, fontsize=7,
                    ha="right", va="bottom", color=PAL["hard"] if slope_dir == "inverted" else PAL["easy"],
                    fontstyle="italic")

        ax.set_xlabel("Models correct")
        ax.set_title(domain_labels.get(domain, domain), fontsize=9)
        ax.set_xlim(-0.5, 14.5)

    axes[0].set_ylabel("Mean CKA")
    plt.tight_layout(w_pad=1.0)
    _save(fig, "fig2_per_domain_difficulty")


def fig3_per_layer_difficulty():
    """CKA trajectory by difficulty bin + inversion strength per layer."""
    plt.rcParams.update(RC)

    with open("data/metrics/extended/b7_per_layer_difficulty.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    bin_colors = {
        "hard (0-4)": PAL["hard"],
        "medium-hard (5-7)": PAL["medium"],
        "medium-easy (8-10)": PAL["pre"],
        "easy (11-14)": PAL["easy"],
    }
    bin_labels = {
        "hard (0-4)": "Hard (0\u20134 correct)",
        "medium-hard (5-7)": "Med-hard (5\u20137)",
        "medium-easy (8-10)": "Med-easy (8\u201310)",
        "easy (11-14)": "Easy (11\u201314 correct)",
    }

    for bin_name in ["hard (0-4)", "medium-hard (5-7)", "medium-easy (8-10)", "easy (11-14)"]:
        if bin_name not in data:
            continue
        ckas = data[bin_name]["cka_per_layer"]
        x = np.linspace(0, 1, len(ckas))
        n = data[bin_name]["n_problems"]
        ax1.plot(x, ckas, label=f"{bin_labels[bin_name]} (n={n})",
                 color=bin_colors[bin_name], linewidth=1.5)

    ax1.set_xlabel("Normalized layer depth")
    ax1.set_ylabel("Mean CKA")
    ax1.set_title("(a) CKA trajectory by difficulty")
    ax1.legend(fontsize=6, loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.7, 1.0)

    # Inversion strength
    if "inversion_per_layer" in data:
        inv = data["inversion_per_layer"]
        x = np.linspace(0, 1, len(inv))
        colors = [PAL["hard"] if v > 0 else PAL["easy"] for v in inv]
        ax2.bar(x, inv, width=0.04, color=colors, edgecolor="none")
        ax2.axhline(y=0, color="0.5", linewidth=0.5)
        ax2.set_xlabel("Normalized layer depth")
        ax2.set_ylabel("CKA(hard) \u2212 CKA(easy)")
        ax2.set_title("(b) Inversion strength per layer")
        ax2.set_xlim(0, 1)

    plt.tight_layout(w_pad=2.5)
    _save(fig, "fig3_per_layer_difficulty")


def fig4_pre_post_decision():
    """Pre vs post decision CKA trajectory."""
    plt.rcParams.update(RC)

    with open("data/metrics/pre_vs_post/exp10_results.json") as f:
        data = json.load(f)

    pre_cka = data.get("mean_cka_pre_decision", [])
    post_cka = data.get("mean_cka_post_decision", [])

    fig, ax = plt.subplots(figsize=(4, 3))

    if isinstance(pre_cka, list) and isinstance(post_cka, list):
        x = np.linspace(0, 1, len(pre_cka))
        ax.plot(x, pre_cka, label="Pre-decision", color=PAL["pre"], linewidth=2)
        ax.plot(x, post_cka, label="Post-decision", color=PAL["post"], linewidth=2)
        ax.fill_between(x, pre_cka, post_cka, alpha=0.08, color="0.5")
        mean_pre = np.mean(pre_cka)
        mean_post = np.mean(post_cka)
    else:
        mean_pre = float(pre_cka) if not isinstance(pre_cka, list) else np.mean(pre_cka)
        mean_post = float(post_cka) if not isinstance(post_cka, list) else np.mean(post_cka)
        ax.bar([0, 1], [mean_pre, mean_post], color=[PAL["pre"], PAL["post"]], width=0.5)

    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Mean cross-model CKA")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Clean annotation — positioned away from legend
    gap = mean_pre - mean_post
    ax.text(0.75, 0.45, f"\u0394 = {gap:.3f}", fontsize=9, transform=ax.transAxes,
            color="0.3", ha="center")

    plt.tight_layout()
    _save(fig, "fig4_pre_post_decision")


def fig5_transfer_vs_causal():
    """Transfer probe accuracy vs causal flip rate — the paradox.

    Panel (a): probe accuracy per model (sorted, with chance line).
    Panel (b): scatter of probe accuracy vs causal flip rate per model,
    showing the disconnect. This is more informative than a bar chart
    with mostly-zero bars.
    """
    plt.rcParams.update(RC)

    with open("data/metrics/probes/linear/exp02_all_results.json") as f:
        probe_data = json.load(f)

    with open("data/metrics/causal/exp03_summary.json") as f:
        causal_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Panel A: Probe accuracy sorted
    models = []
    accs = []
    for entry in probe_data:
        models.append(entry["model"])
        accs.append(entry["peak_layer_mean_linear_acc"])

    order = np.argsort(accs)
    models_sorted = [models[i] for i in order]
    accs_sorted = [accs[i] for i in order]

    # Distinct color per model family
    family_colors = {
        "Qwen": "#e74c3c", "Gemma": "#2ecc71", "LLaMA": "#3498db",
        "Phi": "#f39c12", "Mistral": "#9b59b6", "OLMo": "#1abc9c",
        "InternLM": "#e67e22", "SmolLM": "#34495e", "Nemotron": "#3498db",
    }
    bar_colors = []
    for m in models_sorted:
        fam = m.split("-")[0].replace("2.5", "").replace("2", "").replace("3.5", "")
        matched = next((c for f, c in family_colors.items() if f.lower() in m.lower()), PAL["gray"])
        bar_colors.append(matched)

    y_pos = np.arange(len(models_sorted))
    bars = ax1.barh(y_pos, accs_sorted, color=bar_colors, alpha=0.88, height=0.7,
                    edgecolor="0.4", linewidth=0.3)

    # Chance line — prominent with label
    ax1.axvline(x=0.5, color=PAL["hard"], linestyle="--", linewidth=1.5, alpha=0.9, zorder=4)
    ax1.text(0.51, 1.0, "chance", fontsize=6.5, color=PAL["hard"],
             va="center", ha="left", fontstyle="italic", transform=ax1.get_xaxis_transform(),
             clip_on=False)

    # Value at tip of each bar
    for i, (bar, val) in enumerate(zip(bars, accs_sorted)):
        ax1.text(val + 0.01, i, f"{val:.2f}", fontsize=5.5, va="center")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m[:14] for m in models_sorted], fontsize=6.5)
    ax1.set_xlabel("Peak probe accuracy")
    ax1.set_title("(a) Correctness is detectable")
    ax1.set_xlim(0.4, 1.05)

    # Panel B: Scatter — probe accuracy vs causal flip rate
    probe_accs_for_causal = []
    flip_rates = []
    model_labels = []
    for model_name, model_data in sorted(causal_data.get("models", {}).items()):
        probe_entry = next((e for e in probe_data if e["model"] == model_name), None)
        if probe_entry is None:
            continue
        pa = probe_entry["peak_layer_mean_linear_acc"]
        abl = model_data.get("ablation_flip_rates", {})
        max_fr = max(
            (abl.get(str(m), 0) for m in [0.5, 1.0, 1.5, 2.0]),
            default=0,
        )
        if not isinstance(max_fr, (int, float)):
            max_fr = 0
        probe_accs_for_causal.append(pa)
        flip_rates.append(max_fr)
        model_labels.append(model_name)

    ax2.scatter(probe_accs_for_causal, flip_rates, s=60,
                color=PAL["post"], edgecolor="0.3", linewidth=0.4, zorder=5)

    # Label outlier (Gemma-1-2B) — offset to avoid overlap with annotation
    for pa, fr, name in zip(probe_accs_for_causal, flip_rates, model_labels):
        if fr > 0.05:
            ax2.annotate(name[:10], (pa, fr), fontsize=5.5,
                         xytext=(-50, -12), textcoords="offset points",
                         arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.5))

    ax2.axhline(y=0, color="0.5", linewidth=0.5)
    ax2.set_xlabel("Peak probe accuracy")
    ax2.set_ylabel("Max ablation flip rate")
    ax2.set_title("(b) Probe accuracy vs. causal effect")
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(-0.02, 0.25)

    # Annotation — bottom right, away from data
    ax2.text(0.95, 0.55, "n = 5 ablation\nproblems per model",
             transform=ax2.transAxes, fontsize=6.5, ha="right", va="top",
             color="0.4", fontstyle="italic")

    plt.tight_layout(w_pad=2)
    _save(fig, "fig5_transfer_vs_causal")


def fig6_random_baseline():
    """Random model CKA baseline — bar chart."""
    plt.rcParams.update(RC)

    with open("data/metrics/extended/b2_random_model_baseline.json") as f:
        data = json.load(f)

    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    categories = ["Random\nvs. Random", "Trained\nvs. Trained", "Trained\nvs. Random"]
    values = [summary["mean_random_vs_random"],
              summary["mean_trained_vs_trained"],
              summary["mean_trained_vs_random"]]
    colors = [PAL["hard"], PAL["pre"], PAL["medium"]]

    bars = ax.bar(categories, values, color=colors, edgecolor="0.3",
                  linewidth=0.4, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", fontsize=8)

    ax.set_ylabel("Mean CKA")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    plt.tight_layout()
    _save(fig, "fig6_random_baseline")


def fig7_embedding_vs_deep():
    """Embedding-only vs deep CKA."""
    plt.rcParams.update(RC)

    with open("data/metrics/extended/b3_embedding_cka.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))

    layers = ["Embedding\n(layer 0)", "Mid-layer\n(\u224850%)", "Deep layer\n(last)"]
    values = [data["embedding_layer"]["mean_cka"],
              data["mid_layer"]["mean_cka"],
              data["deep_layer"]["mean_cka"]]
    errors = [data["embedding_layer"]["std_cka"],
              data["mid_layer"]["std_cka"],
              data["deep_layer"]["std_cka"]]
    colors = [PAL["bar1"], PAL["bar2"], PAL["bar3"]]

    bars = ax.bar(layers, values, yerr=errors, color=colors, edgecolor="0.3",
                  linewidth=0.4, capsize=3, width=0.5, error_kw={"linewidth": 0.8})

    for bar, val in zip(bars, values):
        y_pos = max(bar.get_height() + 0.03, 0.04)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.3f}", ha="center", fontsize=8)

    ax.set_ylabel("Mean CKA")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    _save(fig, "fig7_embedding_vs_deep")


def fig8_attention_entropy():
    """Attention entropy vs difficulty."""
    plt.rcParams.update(RC)

    with open("data/metrics/extended/b4_attention_entropy.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    models = list(data["models"].keys())
    n_models = len(models)
    model_colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Panel A: Entropy by difficulty
    x = np.arange(3)
    width = 0.12
    for i, (model, color) in enumerate(zip(models, model_colors)):
        ent = data["models"][model]["mean_entropy_by_difficulty"]
        vals = [ent.get("hard", 0), ent.get("medium", 0), ent.get("easy", 0)]
        ax1.bar(x + i * width - (n_models / 2) * width + width / 2, vals, width,
                label=model[:12], color=color, edgecolor="0.4", linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Hard\n(0\u20134 correct)", "Medium\n(5\u20139)", "Easy\n(10\u201314)"])
    ax1.set_ylabel("Mean attention entropy")
    ax1.set_title("(a) Entropy by problem difficulty")
    ax1.legend(fontsize=5.5, ncol=3, loc="lower left")

    # Panel B: Correlation per model
    model_names_short = [m[:12] for m in models]
    correlations = [data["models"][m]["pearson_r"] for m in models]

    y_pos = range(len(models))
    ax2.barh(y_pos, correlations, color=model_colors, edgecolor="0.4",
             linewidth=0.3, height=0.6)
    ax2.axvline(x=0, color="0.5", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names_short, fontsize=7)
    ax2.set_xlabel("Pearson r (entropy vs. difficulty)")
    ax2.set_title("(b) Entropy-difficulty correlation")

    # Add r values inside bars
    for i, r in enumerate(correlations):
        ax2.text(r + 0.02, i, f"r = {r:.2f}", fontsize=6, va="center", ha="left")

    plt.tight_layout(w_pad=2)
    _save(fig, "fig8_attention_entropy")


def generate_all():
    print("Generating Paper B figures (v2)...", flush=True)
    fig1_difficulty_inversion()
    fig2_per_domain_difficulty()
    fig3_per_layer_difficulty()
    fig4_pre_post_decision()
    fig5_transfer_vs_causal()
    fig6_random_baseline()
    fig7_embedding_vs_deep()
    fig8_attention_entropy()
    print(f"All 8 figures saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    generate_all()
