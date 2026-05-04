"""B4: Attention entropy vs difficulty.

Tests whether models show more uniform (less focused) attention on harder
problems, supporting the "shared confusion" interpretation.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.config import MODEL_REGISTRY, MODEL_SHORT_NAMES, PATHS, MAX_SEQ_LEN
from src.data_loading import load_all_problems, format_prompt
from src.evaluation import load_evaluation_results


def compute_attention_entropy(attn_weights):
    """Compute entropy of attention distribution.

    Args:
        attn_weights: (n_heads, seq_len, seq_len) attention matrix

    Returns:
        float: mean entropy across heads and query positions
    """
    # Clip for numerical stability
    attn = np.clip(attn_weights, 1e-10, 1.0)
    # Entropy per head per query position: -sum(p * log(p))
    entropy = -np.sum(attn * np.log(attn), axis=-1)  # (n_heads, seq_len)
    return float(np.mean(entropy))


def run_b4(device="cuda", n_problems=200):
    """Extract attention weights and compute entropy per difficulty level.

    Uses a subset of models and problems for speed.
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    print("[B4] Attention entropy vs difficulty", flush=True)

    # Use 6 diverse models
    model_subset = ["Qwen-2.5-3B", "LLaMA-3.2-3B", "Gemma-2-2B",
                    "Mistral-7B", "OLMo-2-7B", "InternLM-2.5-7B"]

    # Load eval results for correctness
    all_eval = {m: load_evaluation_results(m) for m in MODEL_SHORT_NAMES}
    n_total = len(all_eval[MODEL_SHORT_NAMES[0]])
    correctness_count = np.zeros(n_total, dtype=int)
    for m in MODEL_SHORT_NAMES:
        for i, r in enumerate(all_eval[m]):
            if r["correct"]:
                correctness_count[i] += 1

    problems = load_all_problems()[:n_problems]

    results = {"models": {}}

    for model_name in model_subset:
        model_cfg = next(m for m in MODEL_REGISTRY if m["short_name"] == model_name)
        hf_id = model_cfg["hf_id"]
        print(f"  {model_name}: loading...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
            attn_implementation="eager",  # Required for output_attentions=True
        )
        model.eval()

        entropies = []
        difficulties = []

        for i, problem in enumerate(problems):
            user_prompt, sys_prompt = format_prompt(problem)
            messages = [{"role": "user", "content": (sys_prompt or "") + "\n" + user_prompt}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = user_prompt

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Get attention from middle layer
            n_layers = len(outputs.attentions)
            mid_layer = n_layers // 2
            attn = outputs.attentions[mid_layer][0].float().cpu().numpy()  # (n_heads, seq, seq)
            entropy = compute_attention_entropy(attn)

            entropies.append(entropy)
            difficulties.append(int(correctness_count[i]))

        del model
        torch.cuda.empty_cache()

        # Correlate entropy with difficulty
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(entropies, difficulties)
        r_s, p_s = spearmanr(entropies, difficulties)

        # Bin by difficulty
        bins = {}
        for ent, diff in zip(entropies, difficulties):
            bin_name = "hard" if diff <= 4 else ("medium" if diff <= 9 else "easy")
            bins.setdefault(bin_name, []).append(ent)

        results["models"][model_name] = {
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_r": float(r_s),
            "spearman_p": float(p_s),
            "mean_entropy_by_difficulty": {k: float(np.mean(v)) for k, v in bins.items()},
            "n_problems": len(entropies),
        }
        print(f"    Pearson r(entropy, difficulty) = {r_p:.3f} (p={p_p:.4f})", flush=True)
        for bin_name, vals in sorted(bins.items()):
            print(f"    {bin_name}: mean entropy = {np.mean(vals):.4f}", flush=True)

    out_path = Path("data/metrics/extended/b4_attention_entropy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_b4()
