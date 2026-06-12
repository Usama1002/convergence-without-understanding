"""Base model comparison experiment.

Run the core difficulty inversion analysis on base (non-instruction-tuned)
models to control for alignment confounds. If the inversion holds on base
models, it is not an artifact of instruction tuning.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.config import PATHS, MAX_SEQ_LEN, MAX_NEW_TOKENS, ensure_all_dirs
from src.data_loading import load_all_problems, format_prompt
from src.evaluation import evaluate_response, save_evaluation_results, valid_letters_for
from src.extraction import get_states_at_normalized_positions
from src.metrics.cka import linear_cka


BASE_MODELS = [
    {"short_name": "Qwen-2.5-3B-base", "hf_id": "Qwen/Qwen2.5-3B", "params_b": 3.0, "family": "Qwen"},
    {"short_name": "LLaMA-3.2-3B-base", "hf_id": "meta-llama/Llama-3.2-3B", "params_b": 3.0, "family": "LLaMA"},
    {"short_name": "Gemma-2-2B-base", "hf_id": "google/gemma-2-2b", "params_b": 2.0, "family": "Gemma"},
    {"short_name": "Mistral-7B-base", "hf_id": "mistralai/Mistral-7B-v0.3", "params_b": 7.0, "family": "Mistral"},
]


def evaluate_base_model(model_cfg, problems, device="cuda"):
    """Evaluate a base model on all problems.

    Base models don't follow instructions well, so we use a simple
    completion format and check if the answer appears in the output.
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    hf_id = model_cfg["hf_id"]
    name = model_cfg["short_name"]

    print(f"  Loading {name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    results = []
    for i, problem in enumerate(problems):
        user_prompt, sys_prompt = format_prompt(problem)
        # For base models, just use the prompt directly (no chat template)
        text = f"{sys_prompt}\n\n{user_prompt}\n\nAnswer:" if sys_prompt else f"{user_prompt}\n\nAnswer:"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )

        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        eval_result = evaluate_response(
            response, problem["gold_answer"], problem["task_type"],
            valid_letters=valid_letters_for(problem),
        )

        results.append({
            "problem_id": problem["problem_id"],
            "domain": problem["domain"],
            "model": name,
            "correct": eval_result["correct"],
            "extracted_answer": eval_result["extracted_answer"],
            "gold_answer": problem["gold_answer"],
            "raw_response": eval_result["raw_response"][:200],
            "task_type": problem["task_type"],
        })

    del model
    torch.cuda.empty_cache()
    return results


def extract_base_hidden_states(model_cfg, problems, device="cuda"):
    """Extract hidden states for a base model."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    hf_id = model_cfg["hf_id"]
    name = model_cfg["short_name"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Detect dims
    dummy = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True)
    n_layers_p1 = len(out.hidden_states)
    hidden_dim = out.hidden_states[0].shape[-1]
    del out

    n_problems = len(problems)
    states = np.zeros((n_problems, n_layers_p1, hidden_dim), dtype=np.float32)

    for i, problem in enumerate(problems):
        user_prompt, sys_prompt = format_prompt(problem)
        text = f"{sys_prompt}\n\n{user_prompt}\n\nAnswer:" if sys_prompt else f"{user_prompt}\n\nAnswer:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(out.hidden_states):
            states[i, layer_idx] = hs[0, -1].float().cpu().numpy()
        del out
        if i % 100 == 0:
            torch.cuda.empty_cache()

    model_info = {
        "n_layers": n_layers_p1 - 1,
        "n_layers_plus_one": n_layers_p1,
        "hidden_dim": hidden_dim,
        "short_name": name,
    }

    del model
    torch.cuda.empty_cache()
    return states, model_info


def run_base_comparison(device="cuda"):
    """Run the full base model comparison."""
    print("[Base Model Comparison]", flush=True)

    problems = load_all_problems()
    base_names = [m["short_name"] for m in BASE_MODELS]

    # Eval + extract for each base model
    all_eval = {}
    all_states = {}
    all_infos = {}

    for model_cfg in BASE_MODELS:
        name = model_cfg["short_name"]

        eval_path = Path(f"data/metrics/extended/base_eval_{name}.json")
        states_path = Path(f"data/metrics/extended/base_states_{name}.npz")

        if eval_path.exists() and states_path.exists():
            print(f"  {name}: cached, loading...", flush=True)
            with open(eval_path) as f:
                all_eval[name] = json.load(f)
            data = np.load(str(states_path))
            all_states[name] = data["states"]
            with open(str(states_path).replace(".npz", "_info.json")) as f:
                all_infos[name] = json.load(f)
            continue

        print(f"\n  Evaluating {name}...", flush=True)
        t0 = time.time()
        eval_results = evaluate_base_model(model_cfg, problems, device)
        correct = sum(1 for r in eval_results if r["correct"])
        print(f"    Accuracy: {correct}/{len(eval_results)} ({correct/len(eval_results)*100:.1f}%)", flush=True)
        print(f"    Time: {time.time()-t0:.0f}s", flush=True)

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        all_eval[name] = eval_results

        print(f"  Extracting {name} hidden states...", flush=True)
        t0 = time.time()
        states, model_info = extract_base_hidden_states(model_cfg, problems, device)
        np.savez_compressed(str(states_path), states=states)
        with open(str(states_path).replace(".npz", "_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        print(f"    Time: {time.time()-t0:.0f}s", flush=True)

        all_states[name] = states
        all_infos[name] = model_info

    # Normalize to 21 positions
    norm_states = {}
    for name in base_names:
        if name in all_states:
            norm_states[name] = get_states_at_normalized_positions(
                all_states[name], all_infos[name]
            )

    # Compute difficulty-stratified CKA for base models
    n_problems = len(problems)
    available_base = [n for n in base_names if n in all_eval]
    n_base = len(available_base)

    correctness_count = np.zeros(n_problems, dtype=int)
    for name in available_base:
        for i, r in enumerate(all_eval[name]):
            if r["correct"]:
                correctness_count[i] += 1

    pairs = list(itertools.combinations(available_base, 2))

    # Bin by difficulty
    strata = {}
    for threshold in range(n_base + 1):
        indices = np.where(correctness_count == threshold)[0]
        if len(indices) >= 5:
            strata[threshold] = indices

    results = {
        "models": available_base,
        "n_base_models": n_base,
        "model_accuracies": {},
        "difficulty_strata": [],
    }

    for name in available_base:
        correct = sum(1 for r in all_eval[name] if r["correct"])
        results["model_accuracies"][name] = correct / n_problems

    for n_correct, indices in sorted(strata.items()):
        mid_ckas = []
        for m_a, m_b in pairs:
            for layer_idx in range(5, 16):
                X = norm_states[m_a][indices, layer_idx, :]
                Y = norm_states[m_b][indices, layer_idx, :]
                mid_ckas.append(linear_cka(X, Y))

        results["difficulty_strata"].append({
            "n_correct": n_correct,
            "n_problems": int(len(indices)),
            "mean_cka": float(np.mean(mid_ckas)),
        })
        print(f"  Base {n_correct}/{n_base} correct: {len(indices)} problems, CKA={np.mean(mid_ckas):.4f}", flush=True)

    # Check if inversion holds
    if results["difficulty_strata"]:
        hard = [s for s in results["difficulty_strata"] if s["n_correct"] <= 1]
        easy = [s for s in results["difficulty_strata"] if s["n_correct"] >= n_base - 1]
        if hard and easy:
            hard_cka = np.mean([s["mean_cka"] for s in hard])
            easy_cka = np.mean([s["mean_cka"] for s in easy])
            results["inversion_gap"] = float(hard_cka - easy_cka)
            results["inversion_holds"] = hard_cka > easy_cka
            print(f"\n  Base model inversion: hard={hard_cka:.4f}, easy={easy_cka:.4f}, gap={hard_cka-easy_cka:+.4f}", flush=True)
            print(f"  Inversion holds: {results['inversion_holds']}", flush=True)

    out_path = Path("data/metrics/extended/base_model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_base_comparison()
