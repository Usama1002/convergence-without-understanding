"""
Ablation Study: Prompt Sensitivity.

For a subset of 4 models, evaluates accuracy under 4 prompt variants
(variant 0 = default, variants 1-3 = alternatives from ALTERNATIVE_PROMPTS)
on a subset of n_problems problems.  Records accuracy per variant and
computes the standard deviation across prompts as a sensitivity measure.

Results saved to: prompt_sensitivity/prompt_sensitivity.json
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from src.config import (
    ALTERNATIVE_PROMPTS,
    MAX_NEW_TOKENS,
    MAX_SEQ_LEN,
    MODEL_REGISTRY,
    MODEL_SHORT_NAMES,
    PATHS,
    SYSTEM_PROMPTS,
    ensure_all_dirs,
)
from src.data_loading import format_prompt, load_all_problems
from src.evaluation import evaluate_response

# Use first 4 models for efficiency
_SUBSET_MODELS = MODEL_SHORT_NAMES[:4]
_N_PROMPT_VARIANTS = 4  # variants 0, 1, 2, 3


def run_prompt_sensitivity_ablation(
    n_problems: int = 100,
    device: str = "cuda",
) -> dict:
    """Evaluate model accuracy under different prompt variants.

    Parameters
    ----------
    n_problems : int
        Number of problems to evaluate per model (default 100).
    device : str
        Torch device string (default "cuda").

    Returns
    -------
    dict
        Results dictionary that was saved to disk.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_all_dirs()

    problems = load_all_problems()[:n_problems]

    all_model_results = []

    for model_cfg in MODEL_REGISTRY:
        model_name = model_cfg["short_name"]
        if model_name not in _SUBSET_MODELS:
            continue

        print(f"\n--- Model: {model_name} ---")
        hf_id = model_cfg["hf_id"]

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
        )
        model.eval()

        variant_accuracies: list[float] = []

        for variant_idx in range(_N_PROMPT_VARIANTS):
            n_correct = 0

            for problem in problems:
                user_prompt, system_prompt = format_prompt(problem, prompt_variant=variant_idx)
                task_type = problem["task_type"]
                gold_answer = problem["gold_answer"]

                messages_with_system = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                messages_no_system = [
                    {"role": "user", "content": system_prompt + "\n\n" + user_prompt},
                ]

                input_ids = None
                for messages in [messages_with_system, messages_no_system]:
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        enc = tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=MAX_SEQ_LEN,
                        )
                        input_ids = enc["input_ids"].to(device)
                        break
                    except Exception:
                        continue

                if input_ids is None:
                    plain = system_prompt + "\n\n" + user_prompt
                    enc = tokenizer(
                        plain,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_SEQ_LEN,
                    )
                    input_ids = enc["input_ids"].to(device)

                n_input = input_ids.shape[-1]

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )

                new_tokens = output_ids[0, n_input:]
                raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                eval_result = evaluate_response(raw_response, gold_answer, task_type)

                if eval_result["correct"]:
                    n_correct += 1

            accuracy = float(n_correct) / len(problems)
            variant_accuracies.append(accuracy)
            print(f"  Variant {variant_idx}: accuracy={accuracy:.4f}")

        sensitivity_std = float(np.std(variant_accuracies))
        all_model_results.append({
            "model": model_name,
            "n_problems": n_problems,
            "variant_accuracies": variant_accuracies,
            "mean_accuracy": float(np.mean(variant_accuracies)),
            "std_accuracy_across_prompts": sensitivity_std,
        })

        # Cleanup
        del model
        torch.cuda.empty_cache()

    results = {
        "description": (
            "Prompt sensitivity ablation evaluating accuracy under 4 prompt "
            f"variants (0=default, 1-3=alternatives) for models: {_SUBSET_MODELS}. "
            f"Evaluated on {n_problems} problems."
        ),
        "models_used": _SUBSET_MODELS,
        "n_problems": n_problems,
        "n_prompt_variants": _N_PROMPT_VARIANTS,
        "model_results": all_model_results,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = str(PATHS["metrics_prompt_sensitivity"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prompt_sensitivity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nPrompt sensitivity results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_prompt_sensitivity_ablation()
