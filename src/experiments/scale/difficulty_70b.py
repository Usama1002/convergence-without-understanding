# Scale validation: difficulty-stratified analysis at 70B scale
import os, json, sys, itertools
import numpy as np
sys.path.insert(0, '.')

from src.config import MODEL_SHORT_NAMES
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.evaluation import load_evaluation_results
from src.metrics.cka import linear_cka
from src.metrics.mnn import mutual_nearest_neighbors

def run_difficulty_with_70b():
    """Compute difficulty-stratified CKA including the 70B model."""
    print("Loading all evaluation results...", flush=True)

    # All models including 70B
    all_models = list(MODEL_SHORT_NAMES) + ["LLaMA-3.1-70B"]
    all_eval = {}
    for m in all_models:
        try:
            all_eval[m] = load_evaluation_results(m)
        except FileNotFoundError:
            print(f"  Skipping {m} (no eval file)", flush=True)

    available = [m for m in all_models if m in all_eval]
    n_models = len(available)
    n_problems = len(all_eval[available[0]])
    print(f"Models: {n_models}, Problems: {n_problems}", flush=True)

    # Correctness count using ALL available models
    correctness_count = np.zeros(n_problems, dtype=int)
    for m in available:
        for i, r in enumerate(all_eval[m]):
            if r["correct"]:
                correctness_count[i] += 1

    # Load hidden states
    print("Loading hidden states...", flush=True)
    all_states = {}
    for m in available:
        try:
            hs = load_hidden_states(m)
            all_states[m] = get_states_at_normalized_positions(hs["pre_decision"], hs["model_info"])
            print(f"  {m}: {all_states[m].shape}", flush=True)
        except:
            print(f"  Skipping {m} (no hidden states)", flush=True)

    available_with_states = [m for m in available if m in all_states]
    pairs = list(itertools.combinations(available_with_states, 2))

    # Pairs involving the 70B model
    pairs_with_70b = [(a, b) for a, b in pairs if "70B" in a or "70B" in b]
    print(f"Total pairs: {len(pairs)}, Pairs with 70B: {len(pairs_with_70b)}", flush=True)

    # Difficulty strata
    results = {"strata": [], "pairs_with_70b": []}

    # Overall difficulty analysis (all pairs)
    for threshold in range(n_models + 1):
        indices = np.where(correctness_count == threshold)[0]
        if len(indices) < 5:
            continue
        mid_ckas = []
        for m_a, m_b in pairs:
            for layer_idx in range(5, 16):
                X = all_states[m_a][indices, layer_idx, :]
                Y = all_states[m_b][indices, layer_idx, :]
                mid_ckas.append(linear_cka(X, Y))

        results["strata"].append({
            "n_correct": threshold, "n_models": n_models,
            "n_problems": int(len(indices)),
            "mean_cka": float(np.mean(mid_ckas)),
        })
        print(f"  {threshold}/{n_models} correct: {len(indices)} problems, CKA={np.mean(mid_ckas):.4f}", flush=True)

    # 70B-specific pairs: CKA between 70B and each other model
    if "LLaMA-3.1-70B" in all_states:
        for m_other in available_with_states:
            if m_other == "LLaMA-3.1-70B":
                continue
            ckas = []
            for layer_idx in range(21):
                X = all_states["LLaMA-3.1-70B"][:, layer_idx, :]
                Y = all_states[m_other][:, layer_idx, :]
                ckas.append(float(linear_cka(X, Y)))
            results["pairs_with_70b"].append({
                "model": m_other,
                "cka_per_layer": ckas,
                "mean_cka": float(np.mean(ckas)),
            })
            print(f"  70B vs {m_other}: CKA={np.mean(ckas):.4f}", flush=True)

    # Check if difficulty inversion holds with 70B included
    if results["strata"]:
        hard = [s for s in results["strata"] if s["n_correct"] <= n_models // 3]
        easy = [s for s in results["strata"] if s["n_correct"] >= 2 * n_models // 3]
        if hard and easy:
            hard_cka = np.mean([s["mean_cka"] for s in hard])
            easy_cka = np.mean([s["mean_cka"] for s in easy])
            results["inversion_with_70b"] = {
                "hard_cka": float(hard_cka),
                "easy_cka": float(easy_cka),
                "gap": float(hard_cka - easy_cka),
                "holds": hard_cka > easy_cka,
            }
            print(f"\nDifficulty inversion with 70B: hard={hard_cka:.4f}, easy={easy_cka:.4f}, gap={hard_cka-easy_cka:+.4f}", flush=True)
            print(f"Inversion holds: {hard_cka > easy_cka}", flush=True)

    out_path = "data/metrics/extended/scale_70b_difficulty.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

if __name__ == "__main__":
    run_difficulty_with_70b()
