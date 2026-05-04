"""B2: Random/untrained model CKA baseline.

Computes CKA between trained models and their randomly-initialized versions,
and between random models of different architectures. If random models show
high CKA, then the observed convergence is architectural, not learned.
"""
import json
import itertools
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_REGISTRY, PATHS, MAX_SEQ_LEN
from src.data_loading import load_all_problems, format_prompt
from src.metrics.cka import linear_cka
from src.metrics.mnn import mutual_nearest_neighbors
from src.extraction import get_states_at_normalized_positions, load_hidden_states


# Use a subset of models for speed (different architectures)
RANDOM_MODELS = [
    {"short_name": "Qwen-2.5-1.5B", "hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "params_b": 1.5},
    {"short_name": "Gemma-2-2B", "hf_id": "google/gemma-2-2b-it", "params_b": 2.0},
    {"short_name": "LLaMA-3.2-3B", "hf_id": "meta-llama/Llama-3.2-3B-Instruct", "params_b": 3.0},
    {"short_name": "Mistral-7B", "hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "params_b": 7.0},
]


def extract_random_model_states(model_cfg, problems, n_problems=100, device="cuda"):
    """Extract hidden states from a randomly initialized model."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    from transformers import AutoConfig, DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    hf_id = model_cfg["hf_id"]
    name = model_cfg["short_name"]

    print(f"  Loading random {name}...", flush=True)
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize with random weights
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    # Detect dimensions
    dummy = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True)
    n_layers_p1 = len(out.hidden_states)
    hidden_dim = out.hidden_states[0].shape[-1]
    del out

    # Extract on subset of problems
    states = np.zeros((n_problems, n_layers_p1, hidden_dim), dtype=np.float32)
    for i, problem in enumerate(problems[:n_problems]):
        user_prompt, sys_prompt = format_prompt(problem)
        messages = [{"role": "user", "content": (sys_prompt or "") + "\n" + user_prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = user_prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(out.hidden_states):
            states[i, layer_idx] = hs[0, -1].float().cpu().numpy()

    del model
    torch.cuda.empty_cache()

    return states, {"n_layers": n_layers_p1 - 1, "n_layers_plus_one": n_layers_p1,
                    "hidden_dim": hidden_dim, "short_name": f"random_{name}"}


def run_b2(device="cuda"):
    """Compute CKA between trained and random models."""
    print("[B2] Random model CKA baseline", flush=True)

    problems = load_all_problems()
    n_problems = 100  # Use subset for speed

    # Load trained model states (first 100 problems)
    trained_states = {}
    for cfg in RANDOM_MODELS:
        name = cfg["short_name"]
        hs = load_hidden_states(name)
        info = hs["model_info"]
        norm = get_states_at_normalized_positions(hs["pre_decision"][:n_problems], info)
        trained_states[name] = norm

    # Extract random model states
    random_states = {}
    for cfg in RANDOM_MODELS:
        name = cfg["short_name"]
        raw_states, info = extract_random_model_states(cfg, problems, n_problems, device)
        norm = get_states_at_normalized_positions(raw_states, info)
        random_states[name] = norm
        print(f"  Random {name}: {norm.shape}", flush=True)

    # 1. Trained vs Random (same architecture)
    trained_vs_random = {}
    for cfg in RANDOM_MODELS:
        name = cfg["short_name"]
        ckas = []
        for layer_idx in range(21):
            X = trained_states[name][:, layer_idx, :]
            Y = random_states[name][:, layer_idx, :]
            ckas.append(linear_cka(X, Y))
        trained_vs_random[name] = {
            "mean_cka": float(np.mean(ckas)),
            "cka_per_layer": [float(c) for c in ckas],
        }
        print(f"  Trained vs Random {name}: CKA={np.mean(ckas):.4f}", flush=True)

    # 2. Random vs Random (different architectures)
    random_pairs = list(itertools.combinations(RANDOM_MODELS, 2))
    random_vs_random = {}
    for cfg_a, cfg_b in random_pairs:
        key = f"{cfg_a['short_name']}_vs_{cfg_b['short_name']}"
        ckas = []
        for layer_idx in range(21):
            X = random_states[cfg_a["short_name"]][:, layer_idx, :]
            Y = random_states[cfg_b["short_name"]][:, layer_idx, :]
            ckas.append(linear_cka(X, Y))
        random_vs_random[key] = {
            "mean_cka": float(np.mean(ckas)),
            "cka_per_layer": [float(c) for c in ckas],
        }
        print(f"  Random {cfg_a['short_name']} vs Random {cfg_b['short_name']}: "
              f"CKA={np.mean(ckas):.4f}", flush=True)

    # 3. Trained vs Trained (for reference, already computed — just summarize)
    trained_vs_trained = {}
    for cfg_a, cfg_b in random_pairs:
        key = f"{cfg_a['short_name']}_vs_{cfg_b['short_name']}"
        ckas = []
        for layer_idx in range(21):
            X = trained_states[cfg_a["short_name"]][:, layer_idx, :]
            Y = trained_states[cfg_b["short_name"]][:, layer_idx, :]
            ckas.append(linear_cka(X, Y))
        trained_vs_trained[key] = {
            "mean_cka": float(np.mean(ckas)),
        }

    results = {
        "description": "CKA baselines: trained vs random (same arch), random vs random (diff arch), trained vs trained",
        "n_problems": n_problems,
        "models_used": [c["short_name"] for c in RANDOM_MODELS],
        "trained_vs_random": trained_vs_random,
        "random_vs_random": random_vs_random,
        "trained_vs_trained": trained_vs_trained,
        "summary": {
            "mean_trained_vs_random": float(np.mean([v["mean_cka"] for v in trained_vs_random.values()])),
            "mean_random_vs_random": float(np.mean([v["mean_cka"] for v in random_vs_random.values()])),
            "mean_trained_vs_trained": float(np.mean([v["mean_cka"] for v in trained_vs_trained.values()])),
        },
    }

    print(f"\n  SUMMARY:", flush=True)
    print(f"    Trained vs Trained: {results['summary']['mean_trained_vs_trained']:.4f}", flush=True)
    print(f"    Trained vs Random:  {results['summary']['mean_trained_vs_random']:.4f}", flush=True)
    print(f"    Random vs Random:   {results['summary']['mean_random_vs_random']:.4f}", flush=True)

    out_path = Path("data/metrics/extended/b2_random_model_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_b2()
