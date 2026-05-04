"""
Master pipeline orchestrator for the agree-disagree research project.

Runs all five phases in sequence (or a single phase via --phase):
  Phase 1  (GPU) — Evaluate all models and extract hidden states
  Phase 2  (CPU) — Analysis experiments (exp1, exp2, exp4-exp10)
  Phase 3  (GPU) — Causal interventions (exp3)
  Phase 4        — Ablation studies
  Phase 5  (CPU) — Figure and table generation
"""

import argparse
import json
import os
import sys
import time

# Disable torch.compile / CUDA graphs to avoid OOM on some models (e.g. Gemma-2)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

from src.config import MODEL_REGISTRY, MODEL_SHORT_NAMES, PATHS, ensure_all_dirs


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def log_metadata() -> None:
    """Log environment info to data/metadata/environment.json."""
    metadata_dir = PATHS["evaluations"].parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = None
    gpu_memory_gb = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )

    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = metadata_dir / "environment.json"
    with open(out_path, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"  Environment metadata written to {out_path}")


# ---------------------------------------------------------------------------
# Phase 1 — GPU: evaluate models + extract hidden states
# ---------------------------------------------------------------------------


def phase_1_evaluate_and_extract() -> None:
    """Phase 1 (GPU): Evaluate all 14 models and extract hidden states."""
    from src.data_loading import load_all_problems
    from src.evaluation import evaluate_model, save_evaluation_results, load_evaluation_results
    from src.extraction import extract_hidden_states_for_model, save_hidden_states, load_hidden_states

    problems = load_all_problems()
    print(f"  Loaded {len(problems)} problems.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    for model_cfg in MODEL_REGISTRY:
        short_name = model_cfg["short_name"]
        hf_id = model_cfg["hf_id"]
        print(f"\n  --- Model: {short_name} ---")

        # ------------------------------------------------------------------
        # Evaluation (resume support)
        # ------------------------------------------------------------------
        try:
            eval_path = PATHS["evaluations"] / f"{short_name}.json"
            if eval_path.exists():
                print(f"    Evaluation already exists, skipping: {eval_path}")
            else:
                print(f"    Evaluating {short_name} ...")
                results = evaluate_model(model_cfg, problems, device=device)
                save_evaluation_results(results, short_name)
                print(f"    Evaluation saved.")

            # --------------------------------------------------------------
            # Hidden state extraction (resume support)
            # --------------------------------------------------------------
            hs_path = PATHS["hidden_states"] / f"{short_name}.npz"
            if hs_path.exists():
                print(f"    Hidden states already exist, skipping: {hs_path}")
            else:
                print(f"    Extracting hidden states for {short_name} ...")
                extraction_result = extract_hidden_states_for_model(
                    model_cfg, problems, device=device
                )
                save_hidden_states(extraction_result, short_name)
                print(f"    Hidden states saved.")
        except Exception as e:
            print(f"    ERROR processing {short_name}: {e}")
            print(f"    Skipping {short_name} and continuing with next model...")

        torch.cuda.empty_cache()

    print("\n  Phase 1 complete.")


# ---------------------------------------------------------------------------
# Phase 2 — CPU: analysis experiments
# ---------------------------------------------------------------------------


def phase_2_analysis() -> None:
    """Phase 2 (CPU): All analysis experiments (exp1, exp2, exp4-exp10)."""
    from src.experiments.exp01_cross_model_similarity import run_experiment_1
    from src.experiments.exp02_correctness_probing import run_experiment_2
    from src.experiments.exp04_domain_specific import run_experiment_4
    from src.experiments.exp05_convergence_trajectory import run_experiment_5
    from src.experiments.exp06_transfer_probing import run_experiment_6
    from src.experiments.exp07_scale_convergence import run_experiment_7
    from src.experiments.exp08_difficulty_stratified import run_experiment_8
    from src.experiments.exp09_ensemble_diversity import run_experiment_9
    from src.experiments.exp10_pre_post_decision import run_experiment_10

    experiments = [
        ("Experiment 1 — Cross-Model Similarity", run_experiment_1),
        ("Experiment 2 — Correctness Probing", run_experiment_2),
        ("Experiment 4 — Domain-Specific Analysis", run_experiment_4),
        ("Experiment 5 — Convergence Trajectory", run_experiment_5),
        ("Experiment 6 — Transfer Probing", run_experiment_6),
        ("Experiment 7 — Scale Convergence", run_experiment_7),
        ("Experiment 8 — Difficulty-Stratified", run_experiment_8),
        ("Experiment 9 — Ensemble Diversity", run_experiment_9),
        ("Experiment 10 — Pre vs Post Decision", run_experiment_10),
    ]

    for label, fn in experiments:
        print(f"\n  Running {label} ...")
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"  {label} done in {elapsed:.1f}s.")

    print("\n  Phase 2 complete.")


# ---------------------------------------------------------------------------
# Phase 3 — GPU: causal interventions
# ---------------------------------------------------------------------------


def phase_3_causal() -> None:
    """Phase 3 (GPU): Causal interventions (exp3)."""
    from src.experiments.exp03_causal_intervention import run_experiment_3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    print("  Running Experiment 3 — Causal Intervention ...")
    t0 = time.time()
    run_experiment_3(device=device)
    elapsed = time.time() - t0
    print(f"  Experiment 3 done in {elapsed:.1f}s.")
    torch.cuda.empty_cache()

    print("\n  Phase 3 complete.")


# ---------------------------------------------------------------------------
# Phase 4 — Ablation studies
# ---------------------------------------------------------------------------


def phase_4_ablations() -> None:
    """Phase 4: All ablation studies."""
    ablation_specs = [
        ("centering", "src.ablations.centering", "run_centering_ablation"),
        ("kernel_cka", "src.ablations.kernel_cka", "run_kernel_cka_ablation"),
        ("sample_size", "src.ablations.sample_size", "run_sample_size_ablation"),
        ("metric_agreement", "src.ablations.metric_agreement", "run_metric_agreement_ablation"),
        ("probe_regularization", "src.ablations.probe_regularization", "run_probe_regularization_ablation"),
        ("mlp_architecture", "src.ablations.mlp_architecture", "run_mlp_architecture_ablation"),
        ("token_position", "src.ablations.token_position", "run_token_position_ablation"),
        ("prompt_sensitivity", "src.ablations.prompt_sensitivity", "run_prompt_sensitivity_ablation"),
    ]

    for label, module_path, func_name in ablation_specs:
        print(f"\n  Running ablation: {label} ...")
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            fn()
            elapsed = time.time() - t0
            print(f"  Ablation '{label}' done in {elapsed:.1f}s.")
        except (ImportError, AttributeError) as exc:
            print(f"  WARNING: Skipping ablation '{label}' — {exc}")

    print("\n  Phase 4 complete.")


# ---------------------------------------------------------------------------
# Phase 5 — CPU: figures and tables
# ---------------------------------------------------------------------------


def phase_5_figures_and_tables() -> None:
    """Phase 5 (CPU): Generate all figures and tables."""
    figure_specs = [
        ("fig02", "src.figures.fig02", "run_fig02"),
        ("fig03", "src.figures.fig03", "run_fig03"),
        ("fig04", "src.figures.fig04", "run_fig04"),
        ("fig05", "src.figures.fig05", "run_fig05"),
        ("fig06", "src.figures.fig06", "run_fig06"),
        ("fig07", "src.figures.fig07", "run_fig07"),
    ]

    for label, module_path, func_name in figure_specs:
        print(f"\n  Generating figure: {label} ...")
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            fn()
            elapsed = time.time() - t0
            print(f"  Figure '{label}' done in {elapsed:.1f}s.")
        except (ImportError, AttributeError) as exc:
            print(f"  WARNING: Skipping figure '{label}' — {exc}")

    # Tables
    print("\n  Generating all tables ...")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module("src.tables.generate_all_tables")
        fn = getattr(mod, "generate_all_tables")
        fn()
        elapsed = time.time() - t0
        print(f"  Tables done in {elapsed:.1f}s.")
    except (ImportError, AttributeError) as exc:
        print(f"  WARNING: Skipping tables — {exc}")

    print("\n  Phase 5 complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agree-disagree experimental pipeline."
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help=(
            "Which phase to run (1-5). "
            "0 = run all phases in sequence (default)."
        ),
    )
    args = parser.parse_args()

    ensure_all_dirs()
    log_metadata()

    phases = {
        1: ("Phase 1 — Evaluate Models & Extract Hidden States", phase_1_evaluate_and_extract),
        2: ("Phase 2 — Analysis Experiments", phase_2_analysis),
        3: ("Phase 3 — Causal Interventions", phase_3_causal),
        4: ("Phase 4 — Ablation Studies", phase_4_ablations),
        5: ("Phase 5 — Figures & Tables", phase_5_figures_and_tables),
    }

    run_phases = list(phases.keys()) if args.phase == 0 else [args.phase]

    if args.phase != 0 and args.phase not in phases:
        parser.error(f"--phase must be 0-5, got {args.phase}")

    pipeline_start = time.time()

    for phase_num in run_phases:
        label, fn = phases[phase_num]
        print("\n" + "=" * 60)
        print(f"  {label}")
        print("=" * 60)
        phase_start = time.time()
        fn()
        phase_elapsed = time.time() - phase_start
        print(f"\n  [{label}] finished in {phase_elapsed:.1f}s.")

    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
