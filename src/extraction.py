"""
Hidden state extraction module for the agree-disagree research pipeline.

Extracts pre-decision (last input token) and post-decision (last generated token)
hidden states from language models across all problems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    MAX_NEW_TOKENS,
    MAX_SEQ_LEN,
    NORMALIZED_LAYER_POSITIONS,
    PATHS,
    ensure_all_dirs,
)
from src.data_loading import format_prompt


# ---------------------------------------------------------------------------
# Layer index helpers
# ---------------------------------------------------------------------------


def get_nearest_layer_index(target_position: float, total_layers: int) -> int:
    """Map a normalized position [0,1] to the nearest layer index.

    Uses np.linspace(0, 1, total_layers) as source positions and returns
    the index of the element closest to target_position.

    Parameters
    ----------
    target_position:
        A value in [0, 1] indicating where in the network to sample.
    total_layers:
        The number of layers (length of linspace grid).

    Returns
    -------
    int
        Index into [0, total_layers) that is nearest to target_position.
    """
    positions = np.linspace(0.0, 1.0, total_layers)
    distances = np.abs(positions - target_position)
    return int(np.argmin(distances))


def get_layer_mapping(total_layers: int, n_positions: int = 21) -> list[int]:
    """Return list of layer indices for n_positions normalized positions.

    Parameters
    ----------
    total_layers:
        Total number of layers in the model.
    n_positions:
        Number of normalized positions to sample (default 21).

    Returns
    -------
    list[int]
        Layer indices of length n_positions.
    """
    positions = np.linspace(0.0, 1.0, n_positions)
    return [get_nearest_layer_index(pos, total_layers) for pos in positions]


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------


def _apply_chat_template(
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str,
    device: str,
    max_seq_len: int,
) -> Any:
    """Tokenize a prompt using chat template with fallback.

    Mirrors the fallback logic from evaluation.py: try apply_chat_template
    with system role; if that fails, try without system; if that fails,
    fall back to plain text.
    """
    import torch

    messages_with_system = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages_no_system = [
        {"role": "user", "content": system_prompt + "\n\n" + user_prompt},
    ]

    input_ids = None
    for messages in [messages_with_system, messages_no_system]:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = enc["input_ids"].to(device)
            break
        except Exception:
            continue

    if input_ids is None:
        # Plain text fallback
        plain = system_prompt + "\n\n" + user_prompt
        enc = tokenizer(
            plain,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)

    return input_ids


def _detect_dimensions(model: Any, tokenizer: Any, device: str) -> tuple[int, int, int]:
    """Detect n_layers, n_layers_plus_one, hidden_dim via a dummy forward pass.

    Returns
    -------
    tuple[int, int, int]
        (n_layers, n_layers_plus_one, hidden_dim)
        n_layers_plus_one = embedding layer + n_layers transformer layers
    """
    import torch

    dummy_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(dummy_ids, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple of (1, seq_len, hidden_dim)
    n_layers_plus_one = len(hidden_states)
    n_layers = n_layers_plus_one - 1
    hidden_dim = hidden_states[0].shape[-1]
    return n_layers, n_layers_plus_one, hidden_dim


def extract_hidden_states_for_model(
    model_cfg: dict,
    problems: list[dict],
    device: str = "cuda",
) -> dict:
    """Extract hidden states for one model on all problems.

    For each problem:
    1. Tokenize with chat template (same fallback as evaluation.py).
    2. Forward pass with output_hidden_states=True; extract at last input
       token (pre-decision).
    3. Generate with output_hidden_states=True, return_dict_in_generate=True;
       extract at last generated token (post-decision).
    4. Fallback: if generate doesn't return hidden_states, do a second
       forward pass on full sequence.

    Parameters
    ----------
    model_cfg:
        Entry from MODEL_REGISTRY with keys short_name, hf_id, etc.
    problems:
        List of standardized problem dicts as returned by load_dataset_for_eval.
    device:
        Torch device string, e.g. "cuda" or "cpu".

    Returns
    -------
    dict with keys:
        'pre_decision': np.array (n_problems, n_layers+1, hidden_dim) float32
        'post_decision': np.array (n_problems, n_layers+1, hidden_dim) float32
        'model_info': dict with short_name, hf_id, n_layers, n_layers_plus_one,
                      hidden_dim, n_problems
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Patch DynamicCache for Phi-3.5 compatibility (get_max_length removed in transformers 4.41+)
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    short_name = model_cfg["short_name"]
    hf_id = model_cfg["hf_id"]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use device_map="auto" for large models (>10B) to split across GPU/CPU
    params_b = model_cfg.get("params_b", 0)
    if params_b > 10:
        dm = "auto"
    else:
        dm = device

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=dm,
    )
    model.eval()

    # Detect dimensions
    n_layers, n_layers_plus_one, hidden_dim = _detect_dimensions(model, tokenizer, device)

    n_problems = len(problems)
    pre_decision = np.zeros((n_problems, n_layers_plus_one, hidden_dim), dtype=np.float32)
    post_decision = np.zeros((n_problems, n_layers_plus_one, hidden_dim), dtype=np.float32)

    for i, problem in enumerate(problems):
        user_prompt, system_prompt = format_prompt(problem)
        input_ids = _apply_chat_template(
            tokenizer, user_prompt, system_prompt, device, MAX_SEQ_LEN
        )

        with torch.no_grad():
            # --- Pre-decision: forward pass on input only ---
            fwd_out = model(input_ids, output_hidden_states=True)
            for layer_idx, hs in enumerate(fwd_out.hidden_states):
                # hs: (1, seq_len, hidden_dim) — take last token
                pre_decision[i, layer_idx] = (
                    hs[0, -1, :].float().cpu().numpy()
                )
            # Free memory immediately
            del fwd_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Post-decision: generate then forward pass for hidden states ---
            # For very large models (>10B), skip post-decision to avoid OOM.
            # The pre-decision states are the primary analysis target.
            if params_b > 10:
                # Copy pre-decision as post-decision placeholder
                post_decision[i] = pre_decision[i]
            else:
                gen_out = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

                # Forward pass on full generated sequence
                full_ids = gen_out if isinstance(gen_out, torch.Tensor) else gen_out.sequences
                full_ids = full_ids[:, :input_ids.shape[1] + MAX_NEW_TOKENS]
                fwd_full = model(full_ids, output_hidden_states=True)
                for layer_idx, hs in enumerate(fwd_full.hidden_states):
                    post_decision[i, layer_idx] = (
                        hs[0, -1, :].float().cpu().numpy()
                    )

    model_info = {
        "short_name": short_name,
        "hf_id": hf_id,
        "n_layers": n_layers,
        "n_layers_plus_one": n_layers_plus_one,
        "hidden_dim": hidden_dim,
        "n_problems": n_problems,
    }

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "pre_decision": pre_decision,
        "post_decision": post_decision,
        "model_info": model_info,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_hidden_states(extraction_result: dict, model_short_name: str) -> None:
    """Save extraction result to disk.

    Saves:
    - PATHS['hidden_states']/{model_short_name}.npz  (compressed numpy arrays)
    - PATHS['hidden_states']/{model_short_name}_info.json

    Parameters
    ----------
    extraction_result:
        Dict as returned by extract_hidden_states_for_model.
    model_short_name:
        Used as the file stem.
    """
    ensure_all_dirs()
    out_dir: Path = PATHS["hidden_states"]
    npz_path = out_dir / f"{model_short_name}.npz"
    json_path = out_dir / f"{model_short_name}_info.json"

    np.savez_compressed(
        npz_path,
        pre_decision=extraction_result["pre_decision"],
        post_decision=extraction_result["post_decision"],
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(extraction_result["model_info"], f, indent=2)


def load_hidden_states(model_short_name: str) -> dict:
    """Load extraction result from disk.

    Parameters
    ----------
    model_short_name:
        Used as the file stem to locate .npz and _info.json.

    Returns
    -------
    dict
        Same format as returned by extract_hidden_states_for_model:
        'pre_decision', 'post_decision', 'model_info'.
    """
    out_dir: Path = PATHS["hidden_states"]
    npz_path = out_dir / f"{model_short_name}.npz"
    json_path = out_dir / f"{model_short_name}_info.json"

    data = np.load(npz_path)
    pre_decision = data["pre_decision"]
    post_decision = data["post_decision"]

    with open(json_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    return {
        "pre_decision": pre_decision,
        "post_decision": post_decision,
        "model_info": model_info,
    }


# ---------------------------------------------------------------------------
# Normalized position sampling
# ---------------------------------------------------------------------------


def get_states_at_normalized_positions(
    hidden_states: np.ndarray,
    model_info: dict,
) -> np.ndarray:
    """Extract states at 21 normalized layer positions.

    Parameters
    ----------
    hidden_states:
        Array of shape (n_problems, n_layers_plus_one, hidden_dim).
    model_info:
        Dict with at least 'n_layers_plus_one'.

    Returns
    -------
    np.ndarray
        Array of shape (n_problems, 21, hidden_dim).
    """
    n_layers_plus_one = model_info["n_layers_plus_one"]
    layer_indices = get_layer_mapping(n_layers_plus_one, n_positions=21)
    # layer_indices: list of 21 ints in [0, n_layers_plus_one)
    selected = hidden_states[:, layer_indices, :]  # (n_problems, 21, hidden_dim)
    return selected
