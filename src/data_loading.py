"""
Data loading module for the agree-disagree research pipeline.

Loads and standardizes problems from GSM8K, ARC-Challenge, TruthfulQA,
and HellaSwag datasets into a uniform dict format for evaluation.
"""

from __future__ import annotations

import re
from typing import Any

import datasets as hf_datasets

from src.config import ALTERNATIVE_PROMPTS, DATASET_CONFIGS, SYSTEM_PROMPTS

# ---------------------------------------------------------------------------
# Letter helpers
# ---------------------------------------------------------------------------

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _idx_to_letter(idx: int) -> str:
    if idx < len(_LETTERS):
        return _LETTERS[idx]
    return str(idx)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_gsm8k_answer(answer_str: str) -> str:
    """Extract the numerical answer from a GSM8K answer string.

    GSM8K answers end with '#### <number>'.  Returns the number as a string
    with commas stripped, e.g. '1,234' → '1234'.
    """
    match = re.search(r"####\s*([^\n]+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback: return the whole string stripped
    return answer_str.strip().replace(",", "")


def _format_choices(choices_list: list[str]) -> str:
    """Format a list of choice strings as 'A) ... B) ...' etc."""
    parts = [f"{_idx_to_letter(i)}) {choice}" for i, choice in enumerate(choices_list)]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Per-dataset row parsers
# ---------------------------------------------------------------------------


def _parse_gsm8k(row: dict, idx: int) -> dict:
    problem_id = f"gsm8k_{idx:04d}"
    return {
        "problem_id": problem_id,
        "domain": "math",
        "task_type": "math",
        "question": row["question"],
        "gold_answer": _extract_gsm8k_answer(row["answer"]),
    }


def _parse_arc_challenge(row: dict, idx: int) -> dict:
    problem_id = f"arc_challenge_{idx:04d}"
    choices_text = row["choices"]["text"]
    choice_labels = row["choices"]["label"]
    return {
        "problem_id": problem_id,
        "domain": "science",
        "task_type": "science",
        "question": row["question"],
        "choices": choices_text,
        "choice_labels": choice_labels,
        "gold_answer": row["answerKey"],
    }


def _parse_truthfulqa(row: dict, idx: int) -> dict:
    problem_id = f"truthfulqa_{idx:04d}"
    choices = row["mc1_targets"]["choices"]
    labels = row["mc1_targets"]["labels"]
    # gold = letter corresponding to the index where label == 1
    gold_idx = labels.index(1)
    gold_answer = _idx_to_letter(gold_idx)
    return {
        "problem_id": problem_id,
        "domain": "truthfulness",
        "task_type": "truthfulness",
        "question": row["question"],
        "choices": choices,
        "gold_answer": gold_answer,
    }


def _parse_hellaswag(row: dict, idx: int) -> dict:
    problem_id = f"hellaswag_{idx:04d}"
    choices = row["endings"]
    gold_answer = _idx_to_letter(int(row["label"]))
    return {
        "problem_id": problem_id,
        "domain": "commonsense",
        "task_type": "commonsense",
        "question": row["ctx"],
        "choices": choices,
        "gold_answer": gold_answer,
    }


_PARSERS = {
    "gsm8k": _parse_gsm8k,
    "arc_challenge": _parse_arc_challenge,
    "truthfulqa": _parse_truthfulqa,
    "hellaswag": _parse_hellaswag,
}

# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------


def load_dataset_for_eval(dataset_name: str) -> list[dict]:
    """Load a single dataset and return a list of standardized problem dicts.

    Parameters
    ----------
    dataset_name:
        One of 'gsm8k', 'arc_challenge', 'truthfulqa', 'hellaswag'.

    Returns
    -------
    list[dict]
        Each dict contains at minimum:
        problem_id, domain, task_type, question, gold_answer.
        MC tasks also include 'choices'.
        arc_challenge also includes 'choice_labels'.
    """
    cfg = next((c for c in DATASET_CONFIGS if c["name"] == dataset_name), None)
    if cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid names: {[c['name'] for c in DATASET_CONFIGS]}"
        )

    hf_id: str = cfg["hf_id"]
    subset: str | None = cfg["hf_subset"]
    split: str = cfg["split"]
    n_problems: int = cfg["n_problems"]

    # Load from HuggingFace
    if subset:
        ds = hf_datasets.load_dataset(hf_id, subset, split=split)
    else:
        ds = hf_datasets.load_dataset(hf_id, split=split)

    # Shuffle with fixed seed and take first n_problems
    ds = ds.shuffle(seed=42).select(range(n_problems))

    parser = _PARSERS[dataset_name]
    problems = [parser(row, i) for i, row in enumerate(ds)]
    return problems


def load_all_problems() -> list[dict]:
    """Load all 800 problems (200 per domain) across all four datasets."""
    all_problems: list[dict] = []
    for cfg in DATASET_CONFIGS:
        all_problems.extend(load_dataset_for_eval(cfg["name"]))
    return all_problems


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(problem: dict, prompt_variant: int = 0) -> tuple[str, str]:
    """Format a problem dict into (user_prompt, system_prompt).

    Parameters
    ----------
    problem:
        Standardized problem dict as returned by load_dataset_for_eval.
    prompt_variant:
        0 = default system prompt from SYSTEM_PROMPTS.
        1-3 = alternative prompts from ALTERNATIVE_PROMPTS.

    Returns
    -------
    tuple[str, str]
        (user_prompt, system_prompt)
    """
    task_type: str = problem["task_type"]

    # System prompt
    if prompt_variant == 0:
        system_prompt = SYSTEM_PROMPTS[task_type]
    else:
        alts = ALTERNATIVE_PROMPTS[task_type]
        # variants 1-3 map to indices 0-2
        alt_idx = (prompt_variant - 1) % len(alts)
        system_prompt = alts[alt_idx]

    # User prompt
    question = problem["question"]

    if "choices" in problem:
        choices_str = _format_choices(problem["choices"])
        user_prompt = f"{question}\n\nChoices: {choices_str}"
    else:
        user_prompt = question

    return user_prompt, system_prompt
