"""
Table generation for the agree-disagree paper.

Tables:
  - Table 1: Model details + per-domain accuracy
  - Table 2: Shared problem set sizes from problem_categories.json
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DATASET_CONFIGS, MODEL_REGISTRY, PATHS, ensure_all_dirs


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _tables_dir() -> Path:
    """Return the directory for saving CSV and LaTeX table files."""
    p = PATHS.get("metrics_latex_tables")
    if p is not None:
        return Path(p)
    return Path(PATHS["metrics"]) / "latex_tables"


def _ensure_tables_dir() -> Path:
    td = _tables_dir()
    td.mkdir(parents=True, exist_ok=True)
    return td


def _evaluations_dir() -> Path:
    return Path(PATHS["evaluations"])


# ---------------------------------------------------------------------------
# Table 1: Model details + accuracy per domain
# ---------------------------------------------------------------------------


def generate_table_1() -> tuple[str, str]:
    """Generate Table 1: model details and per-domain accuracy.

    Loads per-model evaluation summaries and compiles:
      short_name | family | params_b | tier | gsm8k_acc | arc_acc | tqa_acc | hs_acc

    Returns
    -------
    tuple[str, str]
        (csv_path, latex_path)
    """
    ensure_all_dirs()
    td = _ensure_tables_dir()

    domain_names = [cfg["name"] for cfg in DATASET_CONFIGS]
    headers = ["model", "family", "params_b", "tier"] + [f"{d}_acc" for d in domain_names]

    rows: list[list[Any]] = []
    for model in MODEL_REGISTRY:
        short_name = model["short_name"]
        row: list[Any] = [
            short_name,
            model["family"],
            model["params_b"],
            model["tier"],
        ]

        # Try to load accuracy from evaluation results
        for cfg in DATASET_CONFIGS:
            dataset_name = cfg["name"]
            acc = _load_model_accuracy(short_name, dataset_name)
            row.append(f"{acc:.3f}" if acc is not None else "N/A")

        rows.append(row)

    # --- CSV ---
    csv_path = str(td / "table1_model_details.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # --- LaTeX ---
    latex_path = str(td / "table1_model_details.tex")
    col_spec = "l" * len(headers)
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Model Details and Per-Domain Accuracy}",
        r"\label{tab:model_details}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(" & ".join(str(v) for v in row) + r" \\")
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")

    print(f"Table 1 saved: {csv_path}, {latex_path}")
    return csv_path, latex_path


def _load_model_accuracy(model_name: str, dataset_name: str) -> float | None:
    """Try to load scalar accuracy for *model_name* on *dataset_name*."""
    evals_dir = _evaluations_dir()

    # Pattern 1: {model_name}/{dataset_name}_results.json
    candidates = [
        evals_dir / model_name / f"{dataset_name}_results.json",
        evals_dir / f"{model_name}_{dataset_name}_results.json",
        evals_dir / model_name / f"{dataset_name}.json",
        evals_dir / f"{model_name}_{dataset_name}.json",
        evals_dir / model_name / "summary.json",
    ]
    for cand in candidates:
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Navigate to accuracy field
            acc = _extract_accuracy(data, dataset_name)
            if acc is not None:
                return acc
    return None


def _extract_accuracy(data: Any, dataset_name: str) -> float | None:
    """Recursively search *data* for an accuracy value related to *dataset_name*."""
    if isinstance(data, dict):
        for key in ("accuracy", "acc", "correct_rate", "mean_accuracy"):
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key])
        # Try dataset-specific sub-key
        if dataset_name in data:
            return _extract_accuracy(data[dataset_name], dataset_name)
        for v in data.values():
            result = _extract_accuracy(v, dataset_name)
            if result is not None:
                return result
    elif isinstance(data, list) and data:
        if isinstance(data[0], dict):
            correct = sum(1 for r in data if r.get("correct") or r.get("is_correct"))
            if correct > 0:
                return correct / len(data)
    return None


# ---------------------------------------------------------------------------
# Table 2: Shared problem set sizes
# ---------------------------------------------------------------------------


def generate_table_2() -> tuple[str, str]:
    """Generate Table 2: shared problem set sizes from problem_categories.json.

    Returns
    -------
    tuple[str, str]
        (csv_path, latex_path)
    """
    ensure_all_dirs()
    td = _ensure_tables_dir()

    cats_path = _evaluations_dir() / "problem_categories.json"
    with open(cats_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    n_all_correct = len(categories.get("all_correct", []))
    n_all_incorrect = len(categories.get("all_incorrect", []))
    n_mixed = len(categories.get("mixed", []))
    n_total = n_all_correct + n_all_incorrect + n_mixed

    headers = ["condition", "n_problems", "fraction"]
    rows = [
        ["all_correct", n_all_correct, f"{n_all_correct / max(n_total, 1):.3f}"],
        ["all_incorrect", n_all_incorrect, f"{n_all_incorrect / max(n_total, 1):.3f}"],
        ["mixed", n_mixed, f"{n_mixed / max(n_total, 1):.3f}"],
        ["total", n_total, "1.000"],
    ]

    # --- CSV ---
    csv_path = str(td / "table2_shared_set_sizes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # --- LaTeX ---
    latex_path = str(td / "table2_shared_set_sizes.tex")
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Problem Set Sizes by Agreement Condition}",
        r"\label{tab:problem_set_sizes}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        "Condition & $N$ & Fraction" + r" \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(f"{row[0]} & {row[1]} & {row[2]}" + r" \\")
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")

    print(f"Table 2 saved: {csv_path}, {latex_path}")
    return csv_path, latex_path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_all_tables() -> None:
    """Generate all paper tables."""
    generate_table_1()
    generate_table_2()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")
