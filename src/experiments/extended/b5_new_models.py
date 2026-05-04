"""B5: Add Gemma-3/4 + Phi-4 models.

Evaluate and extract hidden states for 3 new models to show findings
generalize to newest architectures. Also enables cross-generational analysis.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.config import PATHS, MAX_SEQ_LEN, MAX_NEW_TOKENS, ensure_all_dirs
from src.data_loading import load_all_problems, format_prompt
from src.evaluation import evaluate_model, save_evaluation_results, evaluate_response
from src.extraction import extract_hidden_states_for_model, save_hidden_states


NEW_MODELS = [
    {"short_name": "Gemma-3-4B", "hf_id": "google/gemma-3-4b-it", "params_b": 4.0, "family": "Gemma", "tier": "Scale"},
    {"short_name": "Gemma-4-E2B", "hf_id": "google/gemma-4-E2B-it", "params_b": 2.0, "family": "Gemma", "tier": "Scale"},
    {"short_name": "Phi-4-Mini", "hf_id": "microsoft/Phi-4-mini-instruct", "params_b": 3.8, "family": "Phi", "tier": "Scale"},
]


def run_b5(device="cuda"):
    """Evaluate and extract hidden states for new models."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    ensure_all_dirs()
    print("[B5] Adding new models: Gemma-3, Gemma-4, Phi-4", flush=True)

    problems = load_all_problems()
    print(f"  Loaded {len(problems)} problems", flush=True)

    for model_cfg in NEW_MODELS:
        name = model_cfg["short_name"]
        eval_path = Path(str(PATHS["evaluations"])) / f"{name}.json"
        hs_path = Path(str(PATHS["hidden_states"])) / f"{name}.npz"

        try:
            # Evaluation
            if eval_path.exists():
                print(f"  {name}: eval exists, skipping", flush=True)
            else:
                print(f"  {name}: evaluating...", flush=True)
                t0 = time.time()
                results = evaluate_model(model_cfg, problems, device=device)
                save_evaluation_results(results, name)
                correct = sum(1 for r in results if r["correct"])
                print(f"    Done in {time.time()-t0:.0f}s, accuracy={correct}/{len(results)}", flush=True)

            # Extraction
            if hs_path.exists():
                print(f"  {name}: hidden states exist, skipping", flush=True)
            else:
                print(f"  {name}: extracting hidden states...", flush=True)
                t0 = time.time()
                extraction = extract_hidden_states_for_model(model_cfg, problems, device=device)
                save_hidden_states(extraction, name)
                print(f"    Done in {time.time()-t0:.0f}s", flush=True)

        except Exception as e:
            print(f"  ERROR on {name}: {e}", flush=True)
            continue

        torch.cuda.empty_cache()

    print("[B5] All new models done", flush=True)


if __name__ == "__main__":
    run_b5()
