"""
Shared utilities for figure generation.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import PATHS, ensure_all_dirs

# ---------------------------------------------------------------------------
# NeurIPS-style rcParams
# ---------------------------------------------------------------------------

NEURIPS_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# ---------------------------------------------------------------------------
# Model family colors
# ---------------------------------------------------------------------------

COLORS = {
    "Qwen": "#E63946",
    "SmolLM": "#457B9D",
    "Gemma": "#2A9D8F",
    "LLaMA": "#E9C46A",
    "Phi": "#F4A261",
    "Mistral": "#264653",
    "OLMo": "#6A4C93",
    "InternLM": "#1982C4",
    "Nemotron": "#8B8BAE",
}


def setup_style() -> None:
    """Apply NeurIPS rcParams globally."""
    plt.rcParams.update(NEURIPS_RC)


def save_figure(fig: plt.Figure, name: str) -> str:
    """Save *fig* as both PDF and PNG inside PATHS['figures'].

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Base filename without extension.

    Returns
    -------
    str
        Absolute path to the saved PDF file.
    """
    ensure_all_dirs()
    figures_dir = PATHS.get("figures")
    if figures_dir is None:
        import pathlib
        figures_dir = pathlib.Path(PATHS["metrics"]).parent / "figures"
    figures_dir = os.fspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    pdf_path = os.path.join(figures_dir, f"{name}.pdf")
    png_path = os.path.join(figures_dir, f"{name}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=NEURIPS_RC["figure.dpi"])
    plt.close(fig)
    return pdf_path
