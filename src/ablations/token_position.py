"""
Ablation Study: Token Position / Pooling Strategy.

NOTE: Full mean-pooling over token positions requires re-extracting
hidden states from all models (currently only the last input token,
i.e. pre-decision, is stored on disk).  This ablation is therefore
deferred: a note JSON is saved explaining the limitation and the
steps needed to complete it.

Results saved to: cka/ablations/token_position.json
"""

from __future__ import annotations

import json
import os

from src.config import PATHS, ensure_all_dirs


def run_token_position_ablation() -> dict:
    """Save a note explaining the token position ablation limitation.

    Full mean-pooling requires re-extraction of hidden states with
    all token positions retained.  The current extraction pipeline only
    stores the last-input-token (pre-decision) state per layer.

    Returns
    -------
    dict
        Note dictionary that was saved to disk.
    """
    ensure_all_dirs()

    results = {
        "description": (
            "Token position / pooling strategy ablation."
        ),
        "status": "deferred",
        "reason": (
            "Full mean-pooling over all token positions requires re-extracting "
            "hidden states for every model with the full token sequence retained. "
            "The current extraction pipeline (src/extraction.py) stores only the "
            "last input token (pre-decision state) at each layer, reducing memory "
            "footprint but preventing post-hoc mean-pooling."
        ),
        "current_strategy": (
            "Last input token (pre-decision): the hidden state at the final "
            "token of the input prompt is used as the representation for each "
            "problem-layer pair."
        ),
        "alternative_strategies_considered": [
            {
                "strategy": "mean_pooling",
                "description": (
                    "Average hidden states across all input token positions. "
                    "Requires storing the full (n_tokens, hidden_dim) tensor per layer."
                ),
                "blocker": "Re-extraction required; not available from stored .npz files.",
            },
            {
                "strategy": "first_token_pooling",
                "description": (
                    "Use the first token (BOS / system prompt start). "
                    "Also requires re-extraction."
                ),
                "blocker": "Re-extraction required.",
            },
            {
                "strategy": "last_token_pooling",
                "description": (
                    "Use the last input token — this is the current strategy "
                    "already implemented."
                ),
                "blocker": "None — already implemented.",
            },
        ],
        "steps_to_complete": [
            "Modify src/extraction.py to store all token positions per layer "
            "(increase memory requirement ~seq_len× per model).",
            "Re-run extraction for all 14 models.",
            "Re-run CKA computation with mean-pooled representations.",
            "Compare mean-pooled CKA to last-token CKA across layers.",
        ],
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(str(PATHS["metrics_cka"]), "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "token_position.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Token position ablation note saved to {out_path}")

    return results


if __name__ == "__main__":
    run_token_position_ablation()
