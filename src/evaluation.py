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

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def valid_letters_for(problem: dict) -> str:
    """Option letters actually presented for a problem ("ABCD" if none).

    Pass this to evaluate_response / extract_letter_answer so problems with
    more than four choices (TruthfulQA, some ARC items) stay answerable.
    """
    n_choices = len(problem.get("choices", []) or [])
    return _LETTERS[:n_choices] if n_choices else "ABCD"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_number_answer(text: str) -> str | None:
    """Extract the answer number from a response.

    Checks for the GSM8K-style '#### <number>' pattern first, then for an
    explicit answer phrase ('the answer is 42', 'answer: 42', '= 42'), and
    only then falls back to the last number found in the text (the fallback
    misfires on multi-step traces whose final sentence is not the answer).
    Commas are stripped from numbers.  Returns None if no number is found.
    """
    if text is None:
        return None

    # Try #### pattern first
    hash_match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    # Explicit answer phrase
    phrase_match = re.search(
        r"(?:answer|result)\s*(?:is|:|=)\s*\$?([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if phrase_match:
        return phrase_match.group(1).replace(",", "")

    # Fallback: find all numbers and return the last one
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_letter_answer(text: str, valid_letters: str = "ABCD") -> str | None:
    """Extract the first multiple-choice letter answer from a response.

    ``valid_letters`` are the option letters actually presented for this
    problem (e.g. "ABCDE" for a 5-choice ARC item, up to "ABCDEFGHIJKLM"
    for TruthfulQA); with a fixed A-D range, a problem whose gold answer
    sits beyond "D" is unanswerable for every model.

    Tries an explicit answer phrase ('the answer is C', 'option B') first,
    then a standalone UPPERCASE letter, then a letter followed by
    punctuation (any case, e.g. 'b)'). Standalone matching is case-sensitive
    so the article 'a' and the 'd in "I'd" cannot be read as answers; a
    bare standalone 'I' is skipped for the same reason (it still matches in
    an answer phrase or as 'I)').
    Returns None if not found.
    """
    if text is None:
        return None

    letters = valid_letters.upper()
    rng = f"[{letters}{letters.lower()}]"
    # In the phrase tier a lowercase 'a' is excluded: "the answer is a bit
    # tricky" must not parse as answer "A" (uppercase 'A' and the other
    # lowercase letters stay accepted: 'the answer is c').
    phrase = f"[{letters}{letters.lower().replace('a', '')}]"
    up_bare = f"[{letters.replace('I', '')}]"  # standalone 'I' = pronoun

    # Explicit answer phrase; the cue words are case-insensitive but the
    # letter class is NOT (re.IGNORECASE would let 'a' back in via 'A').
    match = re.search(
        rf"(?i:answer|option|choice)\s*(?i:is|:)?\s*\(?({phrase})\)?\b", text,
    )
    if match:
        return match.group(1).upper()

    # Standalone letter (word boundary), UPPERCASE only (a lowercase 'a'/'d'
    # alone is far more likely to be the article or a contraction)
    match = re.search(rf"\b({up_bare})\b", text)
    if match:
        return match.group(1)

    # Fallback: letter directly followed by punctuation ('b)' is an answer,
    # not an article), any case
    match = re.search(rf"\b({rng})[.),:]", text)
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


def evaluate_response(
    response: str,
    gold_answer: str,
    task_type: str,
    valid_letters: str = "ABCD",
) -> dict:
    """Evaluate a single model response against the gold answer.

    ``valid_letters`` are the option letters presented for this problem
    (see extract_letter_answer); ignored for math.

    Returns a dict with:
        extracted_answer: the parsed answer (str or None)
        correct: bool
        raw_response: the original response string
    """
    if task_type == "math":
        extracted = extract_number_answer(response)
    else:
        extracted = extract_letter_answer(response, valid_letters=valid_letters)

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

        eval_result = evaluate_response(
            raw_response, gold_answer, task_type,
            valid_letters=valid_letters_for(problem),
        )

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
        Results are joined on problem_id (the row order of the FIRST model
        defines the output order), so per-model files may be reordered or
        have extra problems. A model missing any problem_id raises.

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
        # Join on problem_id; a positional join misaligns correctness with
        # hidden states whenever a model's results are reordered.
        by_id = {r["problem_id"]: r for r in results}
        missing = [pid for pid in problem_ids if pid not in by_id]
        if missing:
            raise ValueError(
                f"model '{model_names[j]}' is missing {len(missing)} problems "
                f"(first few: {missing[:5]})"
            )
        for i, pid in enumerate(problem_ids):
            correctness[i, j] = bool(by_id[pid]["correct"])

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
