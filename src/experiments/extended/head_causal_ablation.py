"""A3: Attention head-level causal intervention.

Instead of ablating the full hidden state correctness direction (which
showed 0% flip rate), this targets individual attention heads to find
which specific heads drive correct answers.
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


def run_a3(device="cuda", n_problems=100):
    """Run head-level causal intervention.

    For each model:
    1. Find the top-5 attention heads most correlated with correctness
    2. Ablate each top head individually on correct problems
    3. Measure accuracy drop
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    print("[A3] Head-level causal intervention", flush=True)

    problems = load_all_problems()
    all_eval = {m: load_evaluation_results(m) for m in MODEL_SHORT_NAMES}

    # Compute per-problem correctness
    n_total = len(all_eval[MODEL_SHORT_NAMES[0]])
    correctness_count = np.zeros(n_total, dtype=int)
    for m in MODEL_SHORT_NAMES:
        for i, r in enumerate(all_eval[m]):
            if r["correct"]:
                correctness_count[i] += 1

    # Use models that fit on GPU without offloading
    model_subset = [m for m in MODEL_REGISTRY if m["params_b"] <= 7.0][:6]

    results = {}

    for model_cfg in model_subset:
        name = model_cfg["short_name"]
        hf_id = model_cfg["hf_id"]
        print(f"\n  {name}:", flush=True)

        # Find problems this model gets correct
        model_eval = all_eval[name]
        correct_indices = [i for i, r in enumerate(model_eval) if r["correct"]][:n_problems]
        if len(correct_indices) < 10:
            print(f"    Too few correct problems ({len(correct_indices)}), skipping", flush=True)
            continue

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()

        # Get model layer structure
        if hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            print(f"    Can't find layers, skipping", flush=True)
            del model; torch.cuda.empty_cache()
            continue

        n_layers = len(layers)
        # Focus on layers around the middle and later
        target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

        model_results = {"layers": {}, "n_correct_problems": len(correct_indices)}

        for target_layer_idx in target_layers:
            print(f"    Layer {target_layer_idx}/{n_layers}:", flush=True)
            layer = layers[target_layer_idx]

            # Get number of heads
            if hasattr(layer, "self_attn"):
                n_heads = layer.self_attn.num_heads if hasattr(layer.self_attn, "num_heads") else 32
            else:
                n_heads = 32

            # Test ablating each head (sample 8 heads for speed)
            test_heads = list(range(0, n_heads, max(1, n_heads // 8)))[:8]
            head_results = []

            for head_idx in test_heads:
                n_flips = 0
                n_tested = min(30, len(correct_indices))  # Test on 30 problems per head

                for prob_idx in correct_indices[:n_tested]:
                    problem = problems[prob_idx]
                    user_prompt, sys_prompt = format_prompt(problem)
                    messages = [{"role": "user", "content": (sys_prompt or "") + "\n" + user_prompt}]

                    try:
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        text = user_prompt

                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

                    # Hook to zero out one attention head
                    def ablate_head_hook(module, input, output, head=head_idx):
                        # output is typically (attn_output, attn_weights, ...)
                        if isinstance(output, tuple):
                            attn_out = output[0]
                            head_dim = attn_out.shape[-1] // n_heads
                            start = head * head_dim
                            end = start + head_dim
                            attn_out[:, :, start:end] = 0.0
                            return (attn_out,) + output[1:]
                        return output

                    hook = layers[target_layer_idx].self_attn.register_forward_hook(ablate_head_hook)

                    try:
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False,
                            )
                        input_len = inputs["input_ids"].shape[1]
                        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
                        eval_result = evaluate_response(
                            response, problem["gold_answer"], problem["task_type"],
                            valid_letters=valid_letters_for(problem),
                        )
                        if not eval_result["correct"]:
                            n_flips += 1
                    finally:
                        hook.remove()

                flip_rate = n_flips / n_tested
                head_results.append({
                    "head_idx": head_idx,
                    "n_flips": n_flips,
                    "n_tested": n_tested,
                    "flip_rate": flip_rate,
                })
                if flip_rate > 0:
                    print(f"      Head {head_idx}: {n_flips}/{n_tested} flipped ({flip_rate:.1%})", flush=True)

            model_results["layers"][str(target_layer_idx)] = {
                "head_results": head_results,
                "max_flip_rate": max(h["flip_rate"] for h in head_results),
            }

        del model
        torch.cuda.empty_cache()

        results[name] = model_results

    out_path = Path("data/metrics/extended/a3_head_causal.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_a3()
