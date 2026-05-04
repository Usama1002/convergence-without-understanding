"""B3: Tokenizer/embedding-only CKA baseline.

Computes CKA using ONLY layer 0 (embedding layer) hidden states.
If embedding CKA is already high, convergence is partly a tokenizer artifact.
"""
import json
import itertools
from pathlib import Path

import numpy as np

from src.config import MODEL_SHORT_NAMES, PATHS
from src.extraction import load_hidden_states
from src.metrics.cka import linear_cka
from src.metrics.mnn import mutual_nearest_neighbors


def run_b3():
    """Compute CKA using only embedding layer (layer 0) vs deep layers."""
    print("[B3] Embedding-only CKA baseline", flush=True)

    # Load hidden states — use raw (not normalized) to get actual layer 0
    all_states_raw = {}
    for m in MODEL_SHORT_NAMES:
        hs = load_hidden_states(m)
        all_states_raw[m] = hs["pre_decision"]  # (800, n_layers+1, hidden_dim)
        print(f"  {m}: {hs['pre_decision'].shape}", flush=True)

    pairs = list(itertools.combinations(MODEL_SHORT_NAMES, 2))

    # Compute CKA at embedding layer (index 0) and at deep layers (last 3)
    embedding_ckas = []
    deep_ckas = []
    mid_ckas = []

    for m_a, m_b in pairs:
        states_a = all_states_raw[m_a]
        states_b = all_states_raw[m_b]
        n_layers_a = states_a.shape[1]
        n_layers_b = states_b.shape[1]

        # Embedding layer (0)
        X_emb = states_a[:, 0, :]
        Y_emb = states_b[:, 0, :]
        embedding_ckas.append(linear_cka(X_emb, Y_emb))

        # Mid layer (~50% depth)
        mid_a = n_layers_a // 2
        mid_b = n_layers_b // 2
        X_mid = states_a[:, mid_a, :]
        Y_mid = states_b[:, mid_b, :]
        mid_ckas.append(linear_cka(X_mid, Y_mid))

        # Deep layer (last layer before output)
        X_deep = states_a[:, -1, :]
        Y_deep = states_b[:, -1, :]
        deep_ckas.append(linear_cka(X_deep, Y_deep))

    results = {
        "description": "CKA at different depth levels: embedding (layer 0), mid (~50%), deep (last layer)",
        "n_pairs": len(pairs),
        "n_problems": 800,
        "embedding_layer": {
            "mean_cka": float(np.mean(embedding_ckas)),
            "std_cka": float(np.std(embedding_ckas)),
            "min_cka": float(np.min(embedding_ckas)),
            "max_cka": float(np.max(embedding_ckas)),
        },
        "mid_layer": {
            "mean_cka": float(np.mean(mid_ckas)),
            "std_cka": float(np.std(mid_ckas)),
            "min_cka": float(np.min(mid_ckas)),
            "max_cka": float(np.max(mid_ckas)),
        },
        "deep_layer": {
            "mean_cka": float(np.mean(deep_ckas)),
            "std_cka": float(np.std(deep_ckas)),
            "min_cka": float(np.min(deep_ckas)),
            "max_cka": float(np.max(deep_ckas)),
        },
    }

    print(f"  Embedding CKA: {results['embedding_layer']['mean_cka']:.4f} "
          f"± {results['embedding_layer']['std_cka']:.4f}", flush=True)
    print(f"  Mid-layer CKA: {results['mid_layer']['mean_cka']:.4f} "
          f"± {results['mid_layer']['std_cka']:.4f}", flush=True)
    print(f"  Deep-layer CKA: {results['deep_layer']['mean_cka']:.4f} "
          f"± {results['deep_layer']['std_cka']:.4f}", flush=True)

    out_path = Path("data/metrics/extended/b3_embedding_cka.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_b3()
