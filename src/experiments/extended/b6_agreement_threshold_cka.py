"""B6: Agreement-threshold CKA.

Recompute CKA using agreement thresholds (8+, 10+, 12+, 13+, 14/14)
instead of strict all-correct. Shows difficulty inversion with larger sample sizes.
"""
import json
import itertools
from pathlib import Path

import numpy as np

from src.config import MODEL_SHORT_NAMES, PATHS
from src.extraction import load_hidden_states, get_states_at_normalized_positions
from src.evaluation import load_evaluation_results
from src.metrics.cka import linear_cka
from src.metrics.mnn import mutual_nearest_neighbors


def run_b6():
    """Compute CKA at various agreement thresholds."""
    print("[B6] Agreement-threshold CKA", flush=True)

    # Load eval
    all_eval = {m: load_evaluation_results(m) for m in MODEL_SHORT_NAMES}
    n_problems = len(all_eval[MODEL_SHORT_NAMES[0]])

    # Per-problem correctness count
    correctness_count = np.zeros(n_problems, dtype=int)
    for m in MODEL_SHORT_NAMES:
        for i, r in enumerate(all_eval[m]):
            if r["correct"]:
                correctness_count[i] += 1

    # Load hidden states
    print("  Loading hidden states...", flush=True)
    all_states = {}
    for m in MODEL_SHORT_NAMES:
        hs = load_hidden_states(m)
        all_states[m] = get_states_at_normalized_positions(hs["pre_decision"], hs["model_info"])

    pairs = list(itertools.combinations(MODEL_SHORT_NAMES, 2))
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    results = {"thresholds": []}

    for thresh in thresholds:
        # Problems where EXACTLY thresh models are correct
        indices = np.where(correctness_count == thresh)[0]
        if len(indices) < 3:
            continue

        # CKA at each of 21 layers
        layer_ckas = []
        layer_mnns = []
        for layer_idx in range(21):
            pair_ckas = []
            pair_mnns = []
            for m_a, m_b in pairs:
                X = all_states[m_a][indices, layer_idx, :]
                Y = all_states[m_b][indices, layer_idx, :]
                pair_ckas.append(linear_cka(X, Y))
                pair_mnns.append(mutual_nearest_neighbors(X, Y))
            layer_ckas.append(float(np.mean(pair_ckas)))
            layer_mnns.append(float(np.mean(pair_mnns)))

        result = {
            "n_correct": thresh,
            "n_problems": int(len(indices)),
            "fraction_correct": thresh / 14,
            "cka_per_layer": layer_ckas,
            "mnn_per_layer": layer_mnns,
            "mean_cka": float(np.mean(layer_ckas)),
            "mean_mnn": float(np.mean(layer_mnns)),
        }
        results["thresholds"].append(result)
        print(f"  {thresh:2d}/14 correct: {len(indices):3d} problems, "
              f"CKA={result['mean_cka']:.4f}, MNN={result['mean_mnn']:.4f}", flush=True)

    # Also compute for "at least N" thresholds
    results["at_least_thresholds"] = []
    for min_thresh in [8, 10, 12, 13, 14]:
        indices = np.where(correctness_count >= min_thresh)[0]
        if len(indices) < 3:
            continue

        mid_ckas = []
        for m_a, m_b in pairs:
            for layer_idx in range(5, 16):
                X = all_states[m_a][indices, layer_idx, :]
                Y = all_states[m_b][indices, layer_idx, :]
                mid_ckas.append(linear_cka(X, Y))

        results["at_least_thresholds"].append({
            "min_correct": min_thresh,
            "n_problems": int(len(indices)),
            "mean_cka_mid_layers": float(np.mean(mid_ckas)),
        })
        print(f"  >= {min_thresh}/14: {len(indices):3d} problems, "
              f"CKA={np.mean(mid_ckas):.4f}", flush=True)

    out_path = Path("data/metrics/extended/b6_agreement_threshold_cka.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_b6()
