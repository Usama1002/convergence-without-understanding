"""B1: Per-domain difficulty breakdown.

Tests whether the difficulty inversion (harder problems = higher CKA)
holds across all four reasoning domains independently.
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


def run_b1():
    """Compute difficulty-stratified CKA per domain."""
    print("[B1] Per-domain difficulty breakdown", flush=True)

    # Load eval results
    all_eval = {m: load_evaluation_results(m) for m in MODEL_SHORT_NAMES}
    n_problems = len(all_eval[MODEL_SHORT_NAMES[0]])

    # Identify domains per problem
    domains_per_problem = [all_eval[MODEL_SHORT_NAMES[0]][i].get("domain", "unknown")
                           for i in range(n_problems)]
    unique_domains = sorted(set(domains_per_problem))

    # Compute per-problem correctness count
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
    results = {}

    for domain in unique_domains:
        print(f"  Domain: {domain}", flush=True)
        domain_mask = [i for i, d in enumerate(domains_per_problem) if d == domain]

        # Group by difficulty within this domain
        domain_strata = {}
        for i in domain_mask:
            nc = int(correctness_count[i])
            domain_strata.setdefault(nc, []).append(i)

        domain_results = []
        for nc in sorted(domain_strata.keys()):
            indices = np.array(domain_strata[nc])
            if len(indices) < 3:
                continue

            # Compute mean CKA across all pairs at mid layers (5-15)
            cka_vals = []
            for m_a, m_b in pairs:
                for layer_idx in range(5, 16):
                    X = all_states[m_a][indices, layer_idx, :]
                    Y = all_states[m_b][indices, layer_idx, :]
                    cka_vals.append(linear_cka(X, Y))

            domain_results.append({
                "n_correct": nc,
                "n_problems": len(indices),
                "mean_cka": float(np.mean(cka_vals)),
                "std_cka": float(np.std(cka_vals)),
            })

        results[domain] = domain_results
        # Print summary
        for s in domain_results:
            print(f"    {s['n_correct']:2d}/14 correct: {s['n_problems']:3d} problems, "
                  f"CKA={s['mean_cka']:.4f}", flush=True)

    # Save
    out_path = Path("data/metrics/extended/b1_per_domain_difficulty.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_b1()
