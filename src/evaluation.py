"""
Model evaluation pipeline for the agree-disagree research project.

Handles answer extraction, correctness checking, model inference,
and result aggregation across all problems and models.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np

from src.config import MAX_NEW_TOKENS, MAX_SEQ_LEN, PATHS, ensure_all_dirs
from src.data_loading import format_prompt


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_number_answer(text: str) -> str | None:
    """Extract last number from response.

    Checks for the GSM8K-style '#### <number>' pattern first, then falls
    back to the last number found in the text.  Commas are stripped from
    numbers.  Returns None if no number is found.
    """
    if text is None:
        return None

    # Try #### pattern first
    hash_match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    # Fallback: find all numbers and return the last one
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_letter_answer(text: str) -> str | None:
    """Extract first A-D letter answer from response.

    Tries standalone word-boundary match \\b[A-D]\\b first, then looks for
    [A-D] followed by punctuation (e.g. 'A.' or 'A)').
    Returns None if not found.
    """
    if text is None:
        return None

    # Try standalone letter (word boundary), case-insensitive
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    # Fallback: letter followed by punctuation
    match = re.search(r"([A-Da-d])[.),:]", text)
    if match:
        return match.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------


def check_correctness(extracted: str | None, gold: str, task_type: str) -> bool:
    """Check whether the extracted answer matches the gold answer.

    For 'math' task type: float comparison with 1e-3 tolerance.
    For all other task types (MC): case-insensitive letter match.
    Returns False if extracted is None.
    """
    if extracted is None:
        return False

    if task_type == "math":
        try:
            return abs(float(extracted) - float(gold)) < 1e-3
        except (ValueError, TypeError):
            return False
    else:
        # Multiple-choice: case-insensitive comparison
        return extracted.strip().upper() == gold.strip().upper()


# ---------------------------------------------------------------------------
# Single response evaluation
# ---------------------------------------------------------------------------


def evaluate_response(response: str, gold_answer: str, task_type: str) -> dict:
    """Evaluate a single model response against the gold answer.

    Returns a dict with:
        extracted_answer: the parsed answer (str or None)
        correct: bool
        raw_response: the original response string
    """
    if task_type == "math":
        extracted = extract_number_answer(response)
    else:
        extracted = extract_letter_answer(response)

    correct = check_correctness(extracted, gold_answer, task_type)

    return {
        "extracted_answer": extracted,
        "correct": correct,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Full model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_cfg: dict,
    problems: list[dict],
    device: str = "cuda",
) -> list[dict]:
    """Load a model and evaluate it on all problems with greedy decoding.

    Parameters
    ----------
    model_cfg:
        Entry from MODEL_REGISTRY (must have 'hf_id' and 'short_name').
    problems:
        List of standardised problem dicts from load_all_problems().
    device:
        Torch device string, defaults to 'cuda'.

    Returns
    -------
    list[dict]
        One result dict per problem with keys:
        problem_id, domain, model, correct, extracted_answer,
        gold_answer, raw_response, input_tokens, output_tokens,
        gen_time_s.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Patch DynamicCache for Phi-3.5 compatibility (get_max_length removed in transformers 4.41+)
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    hf_id: str = model_cfg["hf_id"]
    short_name: str = model_cfg["short_name"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    # Use device_map="auto" for large models (>10B) to split across GPU/CPU
    params_b = model_cfg.get("params_b", 0)
    dm = "auto" if params_b > 10 else device

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map=dm,
        trust_remote_code=True,
    )
    model.eval()

    results: list[dict] = []

    for problem in problems:
        user_prompt, system_prompt = format_prompt(problem)
        task_type: str = problem["task_type"]
        gold_answer: str = problem["gold_answer"]

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template; fall back if model doesn't support system role
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            ).to(device)
        except Exception:
            # Merge system prompt into user message
            merged_user = f"{system_prompt}\n\n{user_prompt}"
            messages_fallback = [{"role": "user", "content": merged_user}]
            input_ids = tokenizer.apply_chat_template(
                messages_fallback,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            ).to(device)

        n_input_tokens: int = input_ids.shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        gen_time_s: float = time.perf_counter() - t0

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, n_input_tokens:]
        n_output_tokens: int = len(new_tokens)
        raw_response: str = tokenizer.decode(new_tokens, skip_special_tokens=True)

        eval_result = evaluate_response(raw_response, gold_answer, task_type)

        results.append(
            {
                "problem_id": problem["problem_id"],
                "domain": problem["domain"],
                "model": short_name,
                "correct": eval_result["correct"],
                "extracted_answer": eval_result["extracted_answer"],
                "gold_answer": gold_answer,
                "raw_response": raw_response,
                "input_tokens": n_input_tokens,
                "output_tokens": n_output_tokens,
                "gen_time_s": gen_time_s,
            }
        )

    # Cleanup
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_evaluation_results(results: list[dict], model_short_name: str) -> None:
    """Save evaluation results to PATHS['evaluations']/<model_short_name>.json."""
    ensure_all_dirs()
    out_path = PATHS["evaluations"] / f"{model_short_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_evaluation_results(model_short_name: str) -> list[dict]:
    """Load evaluation results from PATHS['evaluations']/<model_short_name>.json."""
    in_path = PATHS["evaluations"] / f"{model_short_name}.json"
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Matrix construction and problem categorisation
# ---------------------------------------------------------------------------


def compute_correctness_matrix(
    all_results: list[list[dict]],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a (n_problems, n_models) boolean correctness array.

    Parameters
    ----------
    all_results:
        List of per-model result lists, each as returned by evaluate_model().
        All lists must cover the same set of problems in the same order.

    Returns
    -------
    correctness : np.ndarray, shape (n_problems, n_models), dtype bool
    problem_ids : list[str]
    model_names : list[str]
    """
    if not all_results:
        return np.empty((0, 0), dtype=bool), [], []

    problem_ids: list[str] = [r["problem_id"] for r in all_results[0]]
    model_names: list[str] = [results[0]["model"] for results in all_results]

    n_problems = len(problem_ids)
    n_models = len(all_results)

    correctness = np.zeros((n_problems, n_models), dtype=bool)
    for j, results in enumerate(all_results):
        for i, result in enumerate(results):
            correctness[i, j] = bool(result["correct"])

    return correctness, problem_ids, model_names


def categorize_problems(
    correctness: np.ndarray,
    problem_ids: list[str],
) -> dict:
    """Categorise problems by how many models answered them correctly.

    Assumes the study uses 14 models.  Returns a dict with:
        all_correct   : list of indices where all 14 models are correct
        all_incorrect : list of indices where no model is correct
        mixed         : list of indices where some (but not all) models are correct
        difficulty    : np.ndarray of shape (n_problems,) with fraction correct per problem
    """
    n_models = correctness.shape[1]
    n_correct_per_problem: np.ndarray = correctness.sum(axis=1)
    difficulty: np.ndarray = n_correct_per_problem / n_models

    all_correct = list(np.where(n_correct_per_problem == n_models)[0])
    all_incorrect = list(np.where(n_correct_per_problem == 0)[0])
    mixed = list(
        np.where((n_correct_per_problem > 0) & (n_correct_per_problem < n_models))[0]
    )

    return {
        "all_correct": all_correct,
        "all_incorrect": all_incorrect,
        "mixed": mixed,
        "difficulty": difficulty,
    }
