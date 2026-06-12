"""
Re-score existing evaluation results after answer-parsing / gold-label fixes.

The label fixes (ARC answer-key mapping, letter/number parsing, see
data_loading.py and evaluation.py) change how saved generations are SCORED,
not what the models generated. This script re-derives correctness from the
stored raw responses, so the corrected numbers are available without
re-running any model:

    python -m src.rescore

The one exception is TruthfulQA: its fix shuffles the answer choices, which
changes the PROMPT the model sees. Existing TruthfulQA generations were
produced with the old (unshuffled) prompts and cannot be re-scored against
the new gold letters, so they are left untouched and reported; re-run
phase 1 for that domain to regenerate them.
"""

from __future__ import annotations

import json

from src.config import PATHS
from src.data_loading import load_all_problems
from src.evaluation import evaluate_response, valid_letters_for

# Generations for these task types are prompt-compatible with the fixed
# golds and can be re-scored offline. TruthfulQA prompts changed (shuffled
# choices), so its old generations cannot be re-scored.
RESCORABLE_TASK_TYPES = {"math", "science", "commonsense"}


def rescore_results(results: list[dict], problems_by_id: dict[str, dict]) -> dict:
    """Re-score one model's result list in place against corrected problems.

    Returns a summary dict with counts of rescored / skipped / flipped
    entries and the accuracy before and after.
    """
    n_rescored = 0
    n_skipped = 0
    n_flipped = 0
    correct_before = 0
    correct_after = 0

    for result in results:
        correct_before += bool(result.get("correct"))
        problem = problems_by_id.get(result.get("problem_id", ""))
        if problem is None or problem["task_type"] not in RESCORABLE_TASK_TYPES:
            n_skipped += 1
            correct_after += bool(result.get("correct"))
            continue

        rescored = evaluate_response(
            result.get("raw_response", ""),
            problem["gold_answer"],
            problem["task_type"],
            valid_letters=valid_letters_for(problem),
        )
        if bool(rescored["correct"]) != bool(result.get("correct")):
            n_flipped += 1
        result["extracted_answer"] = rescored["extracted_answer"]
        result["gold_answer"] = problem["gold_answer"]
        result["correct"] = rescored["correct"]
        n_rescored += 1
        correct_after += bool(result["correct"])

    n = max(len(results), 1)
    return {
        "n_results": len(results),
        "n_rescored": n_rescored,
        "n_skipped": n_skipped,
        "n_flipped": n_flipped,
        "accuracy_before": correct_before / n,
        "accuracy_after": correct_after / n,
    }


def main() -> None:
    eval_dir = PATHS["evaluations"]
    eval_paths = sorted(eval_dir.glob("*.json"))
    if not eval_paths:
        print(f"No evaluation results found in {eval_dir}")
        return

    print("Loading problems with corrected gold answers ...")
    problems_by_id = {p["problem_id"]: p for p in load_all_problems()}

    # Sanity check against dataset drift: gold answers for math and
    # commonsense are NOT changed by the label fixes, so the stored golds
    # must match the reloaded ones. A mismatch means the datasets resolved
    # differently than in the original run and re-scoring would be invalid.
    sample = json.loads(eval_paths[0].read_text())
    stable = [
        r for r in sample
        if problems_by_id.get(r.get("problem_id", ""), {}).get("task_type")
        in ("math", "commonsense")
    ]
    mismatched = sum(
        1 for r in stable
        if str(r.get("gold_answer")) != str(problems_by_id[r["problem_id"]]["gold_answer"])
    )
    if stable and mismatched / len(stable) > 0.01:
        raise SystemExit(
            f"Gold-answer mismatch on {mismatched}/{len(stable)} unchanged problems: "
            "the datasets did not reload identically to the original run; "
            "re-scoring aborted (re-run the evaluation instead)."
        )

    for path in eval_paths:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)

        summary = rescore_results(results, problems_by_id)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(
            f"{path.stem}: rescored {summary['n_rescored']}/{summary['n_results']} "
            f"({summary['n_flipped']} flipped), "
            f"accuracy {summary['accuracy_before']:.3f} -> {summary['accuracy_after']:.3f}"
        )

    print(
        "\nNote: TruthfulQA entries were NOT re-scored (the fix shuffles the "
        "answer choices, so its prompts changed); re-run phase 1 for that "
        "domain to regenerate them."
    )


if __name__ == "__main__":
    main()
