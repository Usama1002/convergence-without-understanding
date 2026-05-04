"""
Experiment 3: Causal Intervention — Ablation and Amplification of
the Correctness Direction.

For each model:
- Load the correctness direction (from probe training) and the peak layer.
- Run ablation on all-correct problems (project out the correctness direction)
  and track correct→incorrect flips.
- Run amplification on all-incorrect problems (add scaled correctness direction)
  and track incorrect→correct flips.
- Save per-model JSON results and an aggregate summary.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    CAUSAL_MAGNITUDES,
    MAX_NEW_TOKENS,
    MAX_SEQ_LEN,
    MODEL_REGISTRY,
    MODEL_SHORT_NAMES,
    PATHS,
    ensure_all_dirs,
)
from src.data_loading import format_prompt, load_all_problems
from src.evaluation import evaluate_response


# ---------------------------------------------------------------------------
# Path helpers (keys may not be in PATHS; fall back gracefully)
# ---------------------------------------------------------------------------

def _probes_weights_dir() -> Path:
    """Return the directory that holds probe weight files."""
    if "probes_weights" in PATHS:
        return Path(PATHS["probes_weights"])
    # Fallback: same parent as metrics_probes
    if "metrics_probes" in PATHS:
        return Path(PATHS["metrics_probes"])
    return Path(PATHS["metrics"]) / "probes"


def _causal_ablation_dir() -> Path:
    """Return the directory for per-model causal ablation JSON files."""
    if "causal_ablation" in PATHS:
        return Path(PATHS["causal_ablation"])
    return Path(PATHS["metrics_causal"]) / "ablation"


def _causal_dir() -> Path:
    """Return the directory for the causal experiment summary."""
    if "causal" in PATHS:
        return Path(PATHS["causal"])
    return Path(PATHS["metrics_causal"])


# ---------------------------------------------------------------------------
# Model layer access
# ---------------------------------------------------------------------------

def _get_layers(model: Any) -> Any:
    """Return the list of transformer layers for a model.

    Tries model.model.layers first (LLaMA-style), then model.transformer.h
    (GPT-2-style).  Raises AttributeError if neither is found.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Cannot locate transformer layers: tried model.model.layers and "
        "model.transformer.h"
    )


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def _make_ablation_hook(direction_norm: "np.ndarray", magnitude: float) -> Any:
    """Return a forward hook that projects OUT the correctness direction.

    Hook signature: hook(module, input, output) → modified output tuple.
    direction_norm must be a 1-D unit vector of shape (hidden_size,).
    """
    import torch

    d = torch.tensor(direction_norm, dtype=torch.float32)

    def hook(module: Any, input: Any, output: Any) -> Any:
        # output is a tuple; hidden states are the first element
        hs = output[0].float()            # (batch, seq, hidden)
        d_dev = d.to(hs.device)           # move to same device
        # projection of hs onto direction: (hs · d) * d
        proj_scalar = (hs @ d_dev).unsqueeze(-1)   # (batch, seq, 1)
        proj = proj_scalar * d_dev                  # (batch, seq, hidden)
        hs_mod = hs - magnitude * proj
        hs_mod = hs_mod.to(output[0].dtype)
        return (hs_mod,) + output[1:]

    return hook


def _make_amplification_hook(direction_norm: "np.ndarray", magnitude: float) -> Any:
    """Return a forward hook that ADDS the correctness direction.

    The addition is scaled by 0.1 * magnitude * ||hs|| so that the
    perturbation is proportional to the activation scale.
    """
    import torch

    d = torch.tensor(direction_norm, dtype=torch.float32)

    def hook(module: Any, input: Any, output: Any) -> Any:
        hs = output[0].float()                       # (batch, seq, hidden)
        d_dev = d.to(hs.device)
        # scale: 0.1 * magnitude * per-token norm
        scale = 0.1 * magnitude * hs.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
        hs_mod = hs + scale * d_dev
        hs_mod = hs_mod.to(output[0].dtype)
        return (hs_mod,) + output[1:]

    return hook


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _generate_answer(
    model: Any,
    tokenizer: Any,
    problem: dict,
    device: str,
) -> str:
    """Run greedy decoding for a single problem and return the raw response."""
    import torch

    user_prompt, system_prompt = format_prompt(problem)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
    except Exception:
        merged = f"{system_prompt}\n\n{user_prompt}"
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)

    n_input = input_ids.shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_tokens = output_ids[0, n_input:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-model experiment
# ---------------------------------------------------------------------------

def _run_model(
    model_cfg: dict,
    problems: list[dict],
    problem_categories: dict,
    device: str,
) -> dict:
    """Run ablation + amplification experiments for a single model.

    Returns a dict with the full per-magnitude results.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    # Patch DynamicCache for Phi-3.5 compatibility
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    name: str = model_cfg["short_name"]
    hf_id: str = model_cfg["hf_id"]

    # ------------------------------------------------------------------
    # Load probe direction and peak layer
    # ------------------------------------------------------------------
    probes_dir = _probes_weights_dir()
    direction_path = probes_dir / f"{name}_correctness_direction.npy"
    peak_probe_path = probes_dir / f"{name}_peak_probe.pkl"

    if not direction_path.exists():
        raise FileNotFoundError(f"Correctness direction not found: {direction_path}")
    if not peak_probe_path.exists():
        raise FileNotFoundError(f"Peak probe file not found: {peak_probe_path}")

    direction: np.ndarray = np.load(str(direction_path))
    direction_norm: np.ndarray = direction / (np.linalg.norm(direction) + 1e-12)

    # Get peak layer from Exp 2 JSON results
    exp2_json_path = probes_dir / "linear" / "exp02_all_results.json"
    if exp2_json_path.exists():
        with open(exp2_json_path) as fh:
            exp2_data = json.load(fh)
        # exp2_data is a list of dicts, find our model
        model_entry = next((m for m in exp2_data if m["model"] == name), None)
        if model_entry:
            peak_layer: int = int(model_entry["peak_layer_idx"])
        else:
            peak_layer = 0
    else:
        peak_layer = 0

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    # Use device_map="auto" for large models (>10B) to split across GPU/CPU
    params_b = model_cfg.get("params_b", 0)
    dm = "auto" if params_b > 10 else device

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map=dm,
        trust_remote_code=True,
    )
    model.eval()

    layers = _get_layers(model)
    target_layer = layers[peak_layer]

    # ------------------------------------------------------------------
    # Problem subsets
    # ------------------------------------------------------------------
    all_correct_indices: list[int] = problem_categories.get("all_correct", [])
    all_incorrect_indices: list[int] = problem_categories.get("all_incorrect", [])

    # Cap at 100 each
    ablation_indices = all_correct_indices[:100]
    amplification_indices = all_incorrect_indices[:100]

    model_results: dict = {
        "model": name,
        "peak_layer": peak_layer,
        "n_ablation_problems": len(ablation_indices),
        "n_amplification_problems": len(amplification_indices),
        "magnitudes": {},
    }

    for magnitude in CAUSAL_MAGNITUDES:
        mag_key = str(magnitude)
        ablation_results = []
        amplification_results = []

        # --------------------------------------------------------------
        # Ablation: project OUT correctness direction on correct problems
        # --------------------------------------------------------------
        ablation_hook = _make_ablation_hook(direction_norm, magnitude)
        n_ablation_flips = 0

        for idx in ablation_indices:
            problem = problems[idx]
            handle = target_layer.register_forward_hook(ablation_hook)
            try:
                response = _generate_answer(model, tokenizer, problem, device)
            finally:
                handle.remove()

            eval_result = evaluate_response(
                response, problem["gold_answer"], problem["task_type"]
            )
            correct_after = bool(eval_result["correct"])
            flipped = not correct_after  # was correct, now checking if incorrect
            if flipped:
                n_ablation_flips += 1

            ablation_results.append(
                {
                    "problem_id": problem["problem_id"],
                    "correct_after": correct_after,
                    "flipped": flipped,
                }
            )

        # --------------------------------------------------------------
        # Amplification: ADD correctness direction on incorrect problems
        # --------------------------------------------------------------
        amp_hook = _make_amplification_hook(direction_norm, magnitude)
        n_amp_flips = 0

        for idx in amplification_indices:
            problem = problems[idx]
            handle = target_layer.register_forward_hook(amp_hook)
            try:
                response = _generate_answer(model, tokenizer, problem, device)
            finally:
                handle.remove()

            eval_result = evaluate_response(
                response, problem["gold_answer"], problem["task_type"]
            )
            correct_after = bool(eval_result["correct"])
            flipped = correct_after  # was incorrect, now checking if correct
            if flipped:
                n_amp_flips += 1

            amplification_results.append(
                {
                    "problem_id": problem["problem_id"],
                    "correct_after": correct_after,
                    "flipped": flipped,
                }
            )

        n_abl = len(ablation_indices)
        n_amp = len(amplification_indices)
        model_results["magnitudes"][mag_key] = {
            "magnitude": magnitude,
            "ablation": {
                "n_problems": n_abl,
                "n_flips": n_ablation_flips,
                "flip_rate": n_ablation_flips / n_abl if n_abl > 0 else 0.0,
                "details": ablation_results,
            },
            "amplification": {
                "n_problems": n_amp,
                "n_flips": n_amp_flips,
                "flip_rate": n_amp_flips / n_amp if n_amp > 0 else 0.0,
                "details": amplification_results,
            },
        }

    # Cleanup
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return model_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment_3(device: str = "cuda") -> dict:
    """Run Experiment 3: Causal Intervention for all 14 models.

    For each model:
    1. Load the correctness direction and peak probe layer.
    2. Run ablation on all-correct problems (cap 100): project out direction.
    3. Run amplification on all-incorrect problems (cap 100): add direction.
    4. Save per-model JSON to PATHS["causal_ablation"]/{name}.json.

    Finally saves a summary to PATHS["causal"]/exp03_summary.json and returns
    the summary dict.

    Parameters
    ----------
    device:
        Torch device string (default "cuda").

    Returns
    -------
    dict
        Summary of flip rates across all models and magnitudes.
    """
    ensure_all_dirs()

    causal_ablation_dir = _causal_ablation_dir()
    causal_dir = _causal_dir()
    causal_ablation_dir.mkdir(parents=True, exist_ok=True)
    causal_dir.mkdir(parents=True, exist_ok=True)

    # Load problems and categories
    problems = load_all_problems()

    categories_path = Path(PATHS["evaluations"]) / "problem_categories.json"
    with open(categories_path, "r", encoding="utf-8") as fh:
        problem_categories = json.load(fh)

    summary: dict = {"models": {}, "magnitudes": CAUSAL_MAGNITUDES}

    for model_cfg in MODEL_REGISTRY:
        name: str = model_cfg["short_name"]

        # Resume support: skip if already done
        result_path = causal_ablation_dir / f"{name}.json"
        if result_path.exists():
            print(f"[exp03] Skipping {name} (already done)")
            with open(result_path, "r") as fh:
                model_results = json.load(fh)
            summary["models"][name] = model_results.get("summary", {})
            continue

        print(f"[exp03] Processing model: {name}")

        try:
            model_results = _run_model(
                model_cfg=model_cfg,
                problems=problems,
                problem_categories=problem_categories,
                device=device,
            )
        except FileNotFoundError as exc:
            print(f"[exp03] Skipping {name}: {exc}")
            continue
        except Exception as exc:
            print(f"[exp03] Error on {name}: {exc}")
            raise

        # Save per-model results
        out_path = causal_ablation_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(model_results, fh, indent=2)
        print(f"[exp03]   Saved {out_path}")

        # Aggregate into summary
        summary["models"][name] = {}
        for mag_key, mag_data in model_results["magnitudes"].items():
            summary["models"][name][mag_key] = {
                "ablation_flip_rate": mag_data["ablation"]["flip_rate"],
                "amplification_flip_rate": mag_data["amplification"]["flip_rate"],
            }

    # Save summary
    summary_path = causal_dir / "exp03_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp03] Summary saved to {summary_path}")

    return summary
