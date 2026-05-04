"""
Central configuration module for the agree-disagree cross-model reasoning
representation convergence research pipeline.
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------

PATHS = {
    "evaluations": DATA_ROOT / "evaluations",
    "hidden_states": DATA_ROOT / "hidden_states",
    "metrics": DATA_ROOT / "metrics",
    "metrics_cka": DATA_ROOT / "metrics" / "cka",
    "metrics_procrustes": DATA_ROOT / "metrics" / "procrustes",
    "metrics_mnn": DATA_ROOT / "metrics" / "mnn",
    "metrics_probes": DATA_ROOT / "metrics" / "probes",
    "metrics_causal": DATA_ROOT / "metrics" / "causal",
    "metrics_transfer": DATA_ROOT / "metrics" / "transfer",
    "metrics_scale": DATA_ROOT / "metrics" / "scale",
    "metrics_difficulty": DATA_ROOT / "metrics" / "difficulty",
    "metrics_ensemble": DATA_ROOT / "metrics" / "ensemble",
    "metrics_pre_vs_post": DATA_ROOT / "metrics" / "pre_vs_post",
    "metrics_prompt_sensitivity": DATA_ROOT / "metrics" / "prompt_sensitivity",
    "metrics_statistical_tests": DATA_ROOT / "metrics" / "statistical_tests",
    "metrics_summary": DATA_ROOT / "metrics" / "summary",
    "metrics_latex_tables": DATA_ROOT / "metrics" / "latex_tables",
    "cka_pairwise": DATA_ROOT / "metrics" / "cka" / "pairwise",
    "probes_linear": DATA_ROOT / "metrics" / "probes" / "linear",
    "probes_weights": DATA_ROOT / "metrics" / "probes" / "weights",
}


def ensure_all_dirs() -> None:
    """Create all directories defined in PATHS if they do not exist."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    {
        "short_name": "Qwen-2.5-1.5B",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "family": "Qwen",
        "tier": "small",
    },
    {
        "short_name": "SmolLM2-1.7B",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params_b": 1.7,
        "family": "SmolLM",
        "tier": "small",
    },
    {
        "short_name": "Gemma-1-2B",
        "hf_id": "google/gemma-2b-it",
        "params_b": 2.0,
        "family": "Gemma",
        "tier": "small",
    },
    {
        "short_name": "Gemma-2-2B",
        "hf_id": "google/gemma-2-2b-it",
        "params_b": 2.0,
        "family": "Gemma",
        "tier": "small",
    },
    {
        "short_name": "Qwen-2.5-3B",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
        "family": "Qwen",
        "tier": "small",
    },
    {
        "short_name": "LLaMA-3.2-3B",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "params_b": 3.0,
        "family": "LLaMA",
        "tier": "small",
    },
    {
        "short_name": "Phi-3.5-Mini",
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "params_b": 3.8,
        "family": "Phi",
        "tier": "small",
    },
    {
        "short_name": "Qwen-2.5-7B",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7.0,
        "family": "Qwen",
        "tier": "medium",
    },
    {
        "short_name": "Mistral-7B",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.0,
        "family": "Mistral",
        "tier": "medium",
    },
    {
        "short_name": "OLMo-2-7B",
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "params_b": 7.0,
        "family": "OLMo",
        "tier": "medium",
    },
    {
        "short_name": "InternLM-2.5-7B",
        "hf_id": "internlm/internlm2_5-7b-chat",
        "params_b": 7.0,
        "family": "InternLM",
        "tier": "medium",
    },
    {
        "short_name": "Nemotron-8B",
        "hf_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "params_b": 8.0,
        "family": "Nemotron",
        "tier": "medium",
    },
    {
        "short_name": "Gemma-2-9B",
        "hf_id": "google/gemma-2-9b-it",
        "params_b": 9.0,
        "family": "Gemma",
        "tier": "medium",
    },
    {
        "short_name": "Qwen-2.5-14B",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14.0,
        "family": "Qwen",
        "tier": "large",
    },
]

MODEL_SHORT_NAMES = [m["short_name"] for m in MODEL_REGISTRY]
MODEL_HF_IDS = [m["hf_id"] for m in MODEL_REGISTRY]
N_MODELS = len(MODEL_REGISTRY)

# ---------------------------------------------------------------------------
# Family scale trajectories
# ---------------------------------------------------------------------------

FAMILY_SCALE_TRAJECTORIES = {
    "Qwen": ["Qwen-2.5-1.5B", "Qwen-2.5-3B", "Qwen-2.5-7B", "Qwen-2.5-14B"],
    "Gemma": ["Gemma-1-2B", "Gemma-2-2B", "Gemma-2-9B"],
    "LLaMA": ["LLaMA-3.2-3B", "Nemotron-8B"],
}

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIGS = [
    {
        "name": "gsm8k",
        "hf_id": "gsm8k",
        "hf_subset": "main",
        "split": "test",
        "n_problems": 200,
        "task_type": "math",
        "question_field": "question",
        "answer_field": "answer",
    },
    {
        "name": "arc_challenge",
        "hf_id": "allenai/ai2_arc",
        "hf_subset": "ARC-Challenge",
        "split": "test",
        "n_problems": 200,
        "task_type": "science",
        "question_field": "question",
        "answer_field": "answerKey",
    },
    {
        "name": "truthfulqa",
        "hf_id": "truthful_qa",
        "hf_subset": "multiple_choice",
        "split": "validation",
        "n_problems": 200,
        "task_type": "truthfulness",
        "question_field": "question",
        "answer_field": "mc1_targets",
    },
    {
        "name": "hellaswag",
        "hf_id": "Rowan/hellaswag",
        "hf_subset": None,
        "split": "validation",
        "n_problems": 200,
        "task_type": "commonsense",
        "question_field": "ctx",
        "answer_field": "label",
    },
]

# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1024]
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
FDR_Q = 0.05
N_CV_FOLDS = 5

NORMALIZED_LAYER_POSITIONS = list(np.linspace(0.0, 1.0, 21))

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256

PROBE_L2_DEFAULT = 1.0
MLP_HIDDEN_DIM = 256

CAUSAL_MAGNITUDES = [0.5, 1.0, 1.5, 2.0]

PROBE_L2_SWEEP = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

MLP_ARCH_SWEEP = [
    {"hidden_dims": [128]},
    {"hidden_dims": [256]},
    {"hidden_dims": [512]},
    {"hidden_dims": [256, 128]},
    {"hidden_dims": [512, 256]},
]

SAMPLE_SIZE_ABLATION = [50, 100, 150, 200]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "math": (
        "You are a precise mathematical reasoner. "
        "Solve the problem step by step and provide the final numerical answer."
    ),
    "science": (
        "You are a knowledgeable science assistant. "
        "Answer the multiple-choice question by selecting the best option."
    ),
    "truthfulness": (
        "You are a truthful assistant. "
        "Answer the question accurately based on verified facts."
    ),
    "commonsense": (
        "You are a commonsense reasoning assistant. "
        "Choose the most plausible continuation for the given context."
    ),
}

ALTERNATIVE_PROMPTS = {
    "math": [
        "Solve this math problem carefully, showing each step.",
        "Work through the following math problem and give the final answer.",
        "As a math tutor, solve this problem step by step.",
    ],
    "science": [
        "Select the correct answer for this science question.",
        "Using your scientific knowledge, answer this multiple-choice question.",
        "Identify the best answer to the following science question.",
    ],
    "truthfulness": [
        "Answer this question truthfully and accurately.",
        "Provide a factually correct response to the following question.",
        "Based on verified information, answer this question.",
    ],
    "commonsense": [
        "Pick the most logical continuation for this scenario.",
        "Using everyday reasoning, choose the best ending for this context.",
        "Select the most sensible completion for the following situation.",
    ],
}
