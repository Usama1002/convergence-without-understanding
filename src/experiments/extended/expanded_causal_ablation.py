"""Expanded causal ablation using agreement thresholds.

Instead of ablating only on the 5 all-correct problems (underpowered),
use problems where 10+/14 models answer correctly (n=224).
For each model, ablate only on problems THAT MODEL got correct within
the high-agreement set.
"""
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.config import MODEL_REGISTRY, MODEL_SHORT_NAMES, PATHS, MAX_SEQ_LEN, MAX_NEW_TOKENS
from src.data_loading import load_all_problems, format_prompt
from src.evaluation import evaluate_response, load_evaluation_results, valid_letters_for
from src.extraction import get_layer_mapping


def run_expanded_causal(device="cuda", min_agreement=10, max_problems_per_model=100):
    """Run causal ablation on high-agreement problems.

    For each model:
    1. Select problems where >= min_agreement models are correct AND this model is correct
    2. Load correctness direction from Exp 2
    3. Ablate at peak layer, measure flip rate
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    print(f"[Expanded Causal] Agreement threshold: {min_agreement}/14", flush=True)

    problems = load_all_problems()
    all_eval = {m: load_evaluation_results(m) for m in MODEL_SHORT_NAMES}
    n_problems = len(all_eval[MODEL_SHORT_NAMES[0]])

    # Per-problem correctness count
    correctness_count = np.zeros(n_problems, dtype=int)
    for m in MODEL_SHORT_NAMES:
        for i, r in enumerate(all_eval[m]):
            if r["correct"]:
                correctness_count[i] += 1

    high_agreement = set(np.where(correctness_count >= min_agreement)[0])
    print(f"  High-agreement problems (>={min_agreement}/14): {len(high_agreement)}", flush=True)

    # Load peak layers from Exp 2
    exp2_path = Path(str(PATHS["metrics"])) / "probes" / "linear" / "exp02_all_results.json"
    with open(exp2_path) as f:
        exp2_data = json.load(f)
    peak_layers = {e["model"]: e["peak_layer_idx"] for e in exp2_data}

    # Use models that fit on GPU
    model_subset = [m for m in MODEL_REGISTRY if m["params_b"] <= 7.0][:8]

    results = {}

    for model_cfg in model_subset:
        name = model_cfg["short_name"]
        hf_id = model_cfg["hf_id"]
        # peak_layer_idx indexes the 21 normalized probe positions, not the
        # model's transformer blocks; mapped to a block once the model loads.
        peak_pos = peak_layers.get(name, 10)

        # Problems this model got correct within high-agreement set
        model_eval = all_eval[name]
        ablation_indices = [i for i in range(n_problems)
                          if i in high_agreement and model_eval[i]["correct"]]
        ablation_indices = ablation_indices[:max_problems_per_model]

        if len(ablation_indices) < 10:
            print(f"  {name}: only {len(ablation_indices)} problems, skipping", flush=True)
            continue

        print(f"  {name}: {len(ablation_indices)} problems, peak position {peak_pos}", flush=True)

        # Load correctness direction
        direction_path = Path(str(PATHS["metrics"])) / "probes" / "weights" / f"{name}_correctness_direction.npy"
        if not direction_path.exists():
            print(f"    No correctness direction, skipping", flush=True)
            continue
        direction = np.load(str(direction_path))
        direction_norm = direction / (np.linalg.norm(direction) + 1e-12)
        direction_tensor = torch.tensor(direction_norm, dtype=torch.float32)

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        )
        model.eval()

        # Get layers
        if hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            print(f"    Can't find layers, skipping", flush=True)
            del model; torch.cuda.empty_cache()
            continue

        # Hidden-state row L is the output of block L-1 (row 0 = embeddings)
        hs_row = get_layer_mapping(len(layers) + 1, n_positions=21)[peak_pos]
        peak_layer = max(hs_row - 1, 0)

        # Run ablation at magnitude 1.0
        n_flips = 0
        n_tested = 0

        for prob_idx in ablation_indices:
            problem = problems[prob_idx]
            user_prompt, sys_prompt = format_prompt(problem)
            messages = [{"role": "user", "content": (sys_prompt or "") + "\n" + user_prompt}]

            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = user_prompt

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

            def ablation_hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                d = direction_tensor.to(hs.device).to(hs.dtype)
                proj = torch.einsum("...d,d->...", hs, d).unsqueeze(-1) * d
                hs_modified = hs - proj
                if isinstance(output, tuple):
                    return (hs_modified,) + output[1:]
                return hs_modified

            hook = layers[min(peak_layer, len(layers) - 1)].register_forward_hook(ablation_hook)

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    )
                input_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
                eval_result = evaluate_response(
                    response, problem["gold_answer"], problem["task_type"],
                    valid_letters=valid_letters_for(problem),
                )
                if not eval_result["correct"]:
                    n_flips += 1
                n_tested += 1
            except Exception as e:
                print(f"    Error on problem {prob_idx}: {e}", flush=True)
            finally:
                hook.remove()

        del model
        torch.cuda.empty_cache()

        flip_rate = n_flips / max(n_tested, 1)
        results[name] = {
            "n_tested": n_tested,
            "n_flips": n_flips,
            "flip_rate": flip_rate,
            "peak_layer": peak_layer,
            "min_agreement": min_agreement,
        }
        print(f"    Flip rate: {n_flips}/{n_tested} = {flip_rate:.1%}", flush=True)

    # Summary
    all_flip_rates = [r["flip_rate"] for r in results.values()]
    results["_summary"] = {
        "mean_flip_rate": float(np.mean(all_flip_rates)) if all_flip_rates else 0,
        "n_models": len(results),
        "min_agreement": min_agreement,
        "n_high_agreement_problems": len(high_agreement),
    }

    out_path = Path("data/metrics/extended/expanded_causal_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"Mean flip rate: {results['_summary']['mean_flip_rate']:.1%}", flush=True)
    return results


if __name__ == "__main__":
    run_expanded_causal()
