"""B7: Per-layer difficulty analysis.

Shows at which network depths the difficulty inversion is strongest.
If strongest in early layers → supports "surface processing" interpretation.
If strongest in late layers → different interpretation needed.
"""
import json
import itertools
from pathlib import Path

import numpy as np

from src.config import MODEL_SHORT_NAMES
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.evaluation import load_evaluation_results
from src.metrics.cka import linear_cka


def run_b7():
    """Compute CKA per layer per difficulty stratum."""
    print("[B7] Per-layer difficulty analysis", flush=True)

    # Load eval
    all_eval = {m: load_evaluation_results(m) for m in MODEL_SHORT_NAMES}
    n_problems = len(all_eval[MODEL_SHORT_NAMES[0]])

    correctness_count = np.zeros(n_problems, dtype=int)
    for m in MODEL_SHORT_NAMES:
        for i, r in enumerate(all_eval[m]):
            if r["correct"]:
                correctness_count[i] += 1

    # Group into difficulty bins
    bins = {
        "hard (0-4)": np.where(correctness_count <= 4)[0],
        "medium-hard (5-7)": np.where((correctness_count >= 5) & (correctness_count <= 7))[0],
        "medium-easy (8-10)": np.where((correctness_count >= 8) & (correctness_count <= 10))[0],
        "easy (11-14)": np.where(correctness_count >= 11)[0],
    }

    # Load hidden states
    print("  Loading hidden states...", flush=True)
    all_states = {}
    for m in MODEL_SHORT_NAMES:
        hs = load_hidden_states(m)
        all_states[m] = get_states_at_normalized_positions(hs["pre_decision"], hs["model_info"])

    pairs = list(itertools.combinations(MODEL_SHORT_NAMES, 2))
    results = {}

    for bin_name, indices in bins.items():
        print(f"  {bin_name}: {len(indices)} problems", flush=True)
        if len(indices) < 3:
            continue

        layer_ckas = []
        for layer_idx in range(21):
            pair_ckas = []
            for m_a, m_b in pairs:
                X = all_states[m_a][indices, layer_idx, :]
                Y = all_states[m_b][indices, layer_idx, :]
                pair_ckas.append(linear_cka(X, Y))
            layer_ckas.append(float(np.mean(pair_ckas)))

        results[bin_name] = {
            "n_problems": int(len(indices)),
            "cka_per_layer": layer_ckas,
        }

    # Compute inversion strength per layer:
    # inversion = CKA(hard) - CKA(easy) at each layer
    if "hard (0-4)" in results and "easy (11-14)" in results:
        hard_cka = results["hard (0-4)"]["cka_per_layer"]
        easy_cka = results["easy (11-14)"]["cka_per_layer"]
        inversion = [h - e for h, e in zip(hard_cka, easy_cka)]
        results["inversion_per_layer"] = inversion
        peak_layer = int(np.argmax(inversion))
        results["peak_inversion_layer"] = peak_layer
        results["peak_inversion_strength"] = float(inversion[peak_layer])
        print(f"  Peak inversion at layer {peak_layer}/20: "
              f"{inversion[peak_layer]:.4f}", flush=True)

    out_path = Path("data/metrics/extended/b7_per_layer_difficulty.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_b7()
