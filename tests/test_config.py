"""
Tests for src/config.py
"""

import pytest
from src.config import (
    MODEL_REGISTRY,
    MODEL_SHORT_NAMES,
    MODEL_HF_IDS,
    N_MODELS,
    SEEDS,
    DATASET_CONFIGS,
    PATHS,
    NORMALIZED_LAYER_POSITIONS,
    FAMILY_SCALE_TRAJECTORIES,
    PROJECT_ROOT,
    DATA_ROOT,
    CAUSAL_MAGNITUDES,
    PROBE_L2_SWEEP,
    MLP_ARCH_SWEEP,
    SAMPLE_SIZE_ABLATION,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    FDR_Q,
    N_CV_FOLDS,
    MAX_SEQ_LEN,
    MAX_NEW_TOKENS,
    PROBE_L2_DEFAULT,
    MLP_HIDDEN_DIM,
    SYSTEM_PROMPTS,
    ALTERNATIVE_PROMPTS,
    ensure_all_dirs,
)


class TestModelRegistry:
    def test_exactly_14_models(self):
        assert len(MODEL_REGISTRY) == 14

    def test_n_models_matches_registry(self):
        assert N_MODELS == 14

    def test_required_fields_present(self):
        required_fields = {"short_name", "hf_id", "params_b", "family", "tier"}
        for model in MODEL_REGISTRY:
            assert required_fields.issubset(
                model.keys()
            ), f"Model {model.get('short_name', '?')} missing fields"

    def test_short_names_list_length(self):
        assert len(MODEL_SHORT_NAMES) == 14

    def test_hf_ids_list_length(self):
        assert len(MODEL_HF_IDS) == 14

    def test_short_names_match_registry(self):
        assert MODEL_SHORT_NAMES == [m["short_name"] for m in MODEL_REGISTRY]

    def test_hf_ids_match_registry(self):
        assert MODEL_HF_IDS == [m["hf_id"] for m in MODEL_REGISTRY]

    def test_specific_hf_ids(self):
        expected = {
            "Qwen-2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
            "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "Gemma-1-2B": "google/gemma-2b-it",
            "Gemma-2-2B": "google/gemma-2-2b-it",
            "Qwen-2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
            "LLaMA-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
            "Phi-3.5-Mini": "microsoft/Phi-3.5-mini-instruct",
            "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
            "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
            "OLMo-2-7B": "allenai/OLMo-2-1124-7B-Instruct",
            "InternLM-2.5-7B": "internlm/internlm2_5-7b-chat",
            "Nemotron-8B": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
            "Gemma-2-9B": "google/gemma-2-9b-it",
            "Qwen-2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
        }
        hf_map = {m["short_name"]: m["hf_id"] for m in MODEL_REGISTRY}
        for short_name, hf_id in expected.items():
            assert hf_map[short_name] == hf_id, (
                f"{short_name}: expected {hf_id}, got {hf_map.get(short_name)}"
            )

    def test_params_b_are_positive_floats(self):
        for model in MODEL_REGISTRY:
            assert isinstance(model["params_b"], (int, float))
            assert model["params_b"] > 0

    def test_tier_values_valid(self):
        valid_tiers = {"small", "medium", "large"}
        for model in MODEL_REGISTRY:
            assert model["tier"] in valid_tiers, (
                f"{model['short_name']} has invalid tier: {model['tier']}"
            )


class TestSeeds:
    def test_seeds_list(self):
        assert SEEDS == [42, 123, 456, 789, 1024]

    def test_seeds_length(self):
        assert len(SEEDS) == 5

    def test_seeds_are_integers(self):
        for s in SEEDS:
            assert isinstance(s, int)


class TestDatasetConfigs:
    def test_exactly_4_datasets(self):
        assert len(DATASET_CONFIGS) == 4

    def test_required_fields_present(self):
        required_fields = {
            "name",
            "hf_id",
            "hf_subset",
            "split",
            "n_problems",
            "task_type",
            "question_field",
            "answer_field",
        }
        for ds in DATASET_CONFIGS:
            assert required_fields.issubset(
                ds.keys()
            ), f"Dataset {ds.get('name', '?')} missing fields"

    def test_dataset_names(self):
        names = [ds["name"] for ds in DATASET_CONFIGS]
        assert "gsm8k" in names
        assert "arc_challenge" in names
        assert "truthfulqa" in names
        assert "hellaswag" in names

    def test_n_problems_200_for_all(self):
        for ds in DATASET_CONFIGS:
            assert ds["n_problems"] == 200, (
                f"{ds['name']} has n_problems={ds['n_problems']}, expected 200"
            )

    def test_task_types(self):
        task_types = {ds["name"]: ds["task_type"] for ds in DATASET_CONFIGS}
        assert task_types["gsm8k"] == "math"
        assert task_types["arc_challenge"] == "science"
        assert task_types["truthfulqa"] == "truthfulness"
        assert task_types["hellaswag"] == "commonsense"


class TestPaths:
    def test_paths_dict_has_evaluations(self):
        assert "evaluations" in PATHS

    def test_paths_dict_has_hidden_states(self):
        assert "hidden_states" in PATHS

    def test_paths_dict_has_metrics(self):
        assert "metrics" in PATHS

    def test_paths_dict_has_all_metric_subdirs(self):
        expected_subdirs = [
            "metrics_cka",
            "metrics_procrustes",
            "metrics_mnn",
            "metrics_probes",
            "metrics_causal",
            "metrics_transfer",
            "metrics_scale",
            "metrics_difficulty",
            "metrics_ensemble",
            "metrics_pre_vs_post",
            "metrics_prompt_sensitivity",
            "metrics_statistical_tests",
            "metrics_summary",
            "metrics_latex_tables",
        ]
        for key in expected_subdirs:
            assert key in PATHS, f"PATHS missing key: {key}"

    def test_paths_are_path_objects(self):
        from pathlib import Path
        for key, path in PATHS.items():
            assert isinstance(path, Path), f"PATHS['{key}'] is not a Path object"

    def test_project_root_is_path(self):
        from pathlib import Path
        assert isinstance(PROJECT_ROOT, Path)

    def test_data_root_under_project_root(self):
        assert str(DATA_ROOT).startswith(str(PROJECT_ROOT))

    def test_ensure_all_dirs_creates_directories(self):
        import tempfile
        from pathlib import Path
        import src.config as cfg

        original_paths = cfg.PATHS.copy()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg.PATHS = {
                "test_dir_a": tmp_path / "a",
                "test_dir_b": tmp_path / "b" / "c",
            }
            cfg.ensure_all_dirs()
            assert (tmp_path / "a").exists()
            assert (tmp_path / "b" / "c").exists()
        cfg.PATHS = original_paths


class TestNormalizedLayerPositions:
    def test_has_21_points(self):
        assert len(NORMALIZED_LAYER_POSITIONS) == 21

    def test_starts_at_zero(self):
        assert abs(NORMALIZED_LAYER_POSITIONS[0] - 0.0) < 1e-9

    def test_ends_at_one(self):
        assert abs(NORMALIZED_LAYER_POSITIONS[-1] - 1.0) < 1e-9

    def test_evenly_spaced(self):
        diffs = [
            NORMALIZED_LAYER_POSITIONS[i + 1] - NORMALIZED_LAYER_POSITIONS[i]
            for i in range(len(NORMALIZED_LAYER_POSITIONS) - 1)
        ]
        expected_step = 1.0 / 20
        for d in diffs:
            assert abs(d - expected_step) < 1e-9

    def test_all_in_range(self):
        for pos in NORMALIZED_LAYER_POSITIONS:
            assert 0.0 <= pos <= 1.0


class TestFamilyScaleTrajectories:
    def test_has_qwen_family(self):
        assert "Qwen" in FAMILY_SCALE_TRAJECTORIES

    def test_has_gemma_family(self):
        assert "Gemma" in FAMILY_SCALE_TRAJECTORIES

    def test_has_llama_family(self):
        assert "LLaMA" in FAMILY_SCALE_TRAJECTORIES

    def test_qwen_has_4_models(self):
        assert len(FAMILY_SCALE_TRAJECTORIES["Qwen"]) == 4

    def test_gemma_has_3_models(self):
        assert len(FAMILY_SCALE_TRAJECTORIES["Gemma"]) == 3

    def test_llama_has_2_models(self):
        assert len(FAMILY_SCALE_TRAJECTORIES["LLaMA"]) == 2

    def test_trajectory_models_are_in_registry(self):
        for family, models in FAMILY_SCALE_TRAJECTORIES.items():
            for short_name in models:
                assert short_name in MODEL_SHORT_NAMES, (
                    f"{short_name} in trajectory for {family} not found in registry"
                )


class TestExperimentalParams:
    def test_n_bootstrap(self):
        assert N_BOOTSTRAP == 1000

    def test_n_permutations(self):
        assert N_PERMUTATIONS == 1000

    def test_fdr_q(self):
        assert FDR_Q == 0.05

    def test_n_cv_folds(self):
        assert N_CV_FOLDS == 5

    def test_max_seq_len(self):
        assert MAX_SEQ_LEN == 1024

    def test_max_new_tokens(self):
        assert MAX_NEW_TOKENS == 256

    def test_probe_l2_default(self):
        assert PROBE_L2_DEFAULT == 1.0

    def test_mlp_hidden_dim(self):
        assert MLP_HIDDEN_DIM == 256

    def test_causal_magnitudes(self):
        assert CAUSAL_MAGNITUDES == [0.5, 1.0, 1.5, 2.0]

    def test_sample_size_ablation(self):
        assert SAMPLE_SIZE_ABLATION == [50, 100, 150, 200]

    def test_probe_l2_sweep_has_entries(self):
        assert len(PROBE_L2_SWEEP) > 0

    def test_mlp_arch_sweep_has_entries(self):
        assert len(MLP_ARCH_SWEEP) > 0


class TestSystemPrompts:
    def test_has_all_task_types(self):
        task_types = {"math", "science", "truthfulness", "commonsense"}
        assert task_types.issubset(SYSTEM_PROMPTS.keys())

    def test_prompts_are_strings(self):
        for task_type, prompt in SYSTEM_PROMPTS.items():
            assert isinstance(prompt, str), f"SYSTEM_PROMPTS['{task_type}'] is not a string"
            assert len(prompt) > 0


class TestAlternativePrompts:
    def test_has_all_task_types(self):
        task_types = {"math", "science", "truthfulness", "commonsense"}
        assert task_types.issubset(ALTERNATIVE_PROMPTS.keys())

    def test_each_task_has_multiple_prompts(self):
        for task_type, prompts in ALTERNATIVE_PROMPTS.items():
            assert isinstance(prompts, list), f"ALTERNATIVE_PROMPTS['{task_type}'] is not a list"
            assert len(prompts) >= 2, (
                f"ALTERNATIVE_PROMPTS['{task_type}'] has fewer than 2 prompts"
            )
