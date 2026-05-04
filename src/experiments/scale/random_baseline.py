# Scale validation: random initialization baseline across full cohort
import os, json, sys, itertools, time
import numpy as np
import torch
sys.path.insert(0, '.')

os.environ['TORCHDYNAMO_DISABLE'] = '1'
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

from src.config import MODEL_REGISTRY, MAX_SEQ_LEN
from src.data_loading import load_all_problems, format_prompt
from src.extraction import get_states_at_normalized_positions, load_hidden_states
from src.metrics.cka import linear_cka

def extract_random_states(model_cfg, problems, n_problems=800, device="cuda:0"):
    """Extract hidden states from a randomly initialized model."""
    hf_id = model_cfg["hf_id"]
    name = model_cfg["short_name"]
    print(f"  Random {name}...", flush=True)

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    dummy = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True)
    n_layers_p1 = len(out.hidden_states)
    hidden_dim = out.hidden_states[0].shape[-1]
    del out

    states = np.zeros((n_problems, n_layers_p1, hidden_dim), dtype=np.float32)
    for i, problem in enumerate(problems[:n_problems]):
        user_prompt, sys_prompt = format_prompt(problem)
        text = (sys_prompt or "") + "\n" + user_prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(out.hidden_states):
            states[i, layer_idx] = hs[0, -1].float().cpu().numpy()
        if (i+1) % 100 == 0:
            torch.cuda.empty_cache()
            print(f"    [{i+1}/{n_problems}]", flush=True)

    info = {"n_layers": n_layers_p1-1, "n_layers_plus_one": n_layers_p1,
            "hidden_dim": hidden_dim, "short_name": f"random_{name}"}
    del model
    torch.cuda.empty_cache()
    return states, info

def run_full_random_baseline():
    print("Full Random Baseline (all models, 800 problems)", flush=True)
    problems = load_all_problems()

    # Use all models that fit on one GPU (<= 14B in bf16 = ~28GB)
    models_to_test = [m for m in MODEL_REGISTRY if m["params_b"] <= 14.0]
    print(f"Testing {len(models_to_test)} models", flush=True)

    # Alternate GPUs to parallelize
    trained_states = {}
    random_states = {}

    for i, model_cfg in enumerate(models_to_test):
        name = model_cfg["short_name"]
        device = f"cuda:{i % 2}"  # Alternate GPUs

        # Load trained states
        try:
            hs = load_hidden_states(name)
            trained_states[name] = get_states_at_normalized_positions(hs["pre_decision"], hs["model_info"])
        except:
            print(f"  Skipping {name} (no trained states)", flush=True)
            continue

        # Extract random states
        raw, info = extract_random_states(model_cfg, problems, n_problems=800, device=device)
        random_states[name] = get_states_at_normalized_positions(raw, info)

    available = [n for n in trained_states if n in random_states]
    pairs = list(itertools.combinations(available, 2))

    # Compute: trained vs trained, random vs random, trained vs random
    results = {"trained_vs_trained": {}, "random_vs_random": {}, "trained_vs_random": {}}

    for m_a, m_b in pairs:
        key = f"{m_a}_vs_{m_b}"
        t_ckas = [float(linear_cka(trained_states[m_a][:, l, :], trained_states[m_b][:, l, :])) for l in range(21)]
        r_ckas = [float(linear_cka(random_states[m_a][:, l, :], random_states[m_b][:, l, :])) for l in range(21)]
        results["trained_vs_trained"][key] = {"mean_cka": float(np.mean(t_ckas))}
        results["random_vs_random"][key] = {"mean_cka": float(np.mean(r_ckas))}

    for name in available:
        t_ckas = [float(linear_cka(trained_states[name][:, l, :], random_states[name][:, l, :])) for l in range(21)]
        results["trained_vs_random"][name] = {"mean_cka": float(np.mean(t_ckas))}

    results["summary"] = {
        "n_models": len(available),
        "n_problems": 800,
        "mean_trained_vs_trained": float(np.mean([v["mean_cka"] for v in results["trained_vs_trained"].values()])),
        "mean_random_vs_random": float(np.mean([v["mean_cka"] for v in results["random_vs_random"].values()])),
        "mean_trained_vs_random": float(np.mean([v["mean_cka"] for v in results["trained_vs_random"].values()])),
    }

    print(f"\nSUMMARY (n={len(available)} models, 800 problems):", flush=True)
    print(f"  Trained vs Trained: {results['summary']['mean_trained_vs_trained']:.4f}", flush=True)
    print(f"  Random vs Random:  {results['summary']['mean_random_vs_random']:.4f}", flush=True)
    print(f"  Trained vs Random: {results['summary']['mean_trained_vs_random']:.4f}", flush=True)

    out_path = "data/metrics/extended/full_random_baseline.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}", flush=True)

if __name__ == "__main__":
    run_full_random_baseline()
