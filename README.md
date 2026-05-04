# Convergence Without Understanding

Code for "Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning." We evaluate whether representational convergence across language models implies reasoning convergence by studying 16 models from 8 families (1.5B--72B parameters) on 800 reasoning problems across four domains. Our analysis reveals three dissociations: a difficulty inversion (models converge more on problems they fail), a generation gap (convergence collapses in late layers), and epiphenomenal correctness (shared information is not causally deployed). These findings indicate that representational convergence reflects shared input processing constraints rather than shared reasoning strategies.

## Requirements

- Python 3.10+
- PyTorch 2.7+ with CUDA support
- 32 GB+ GPU VRAM for models up to 14B (single GPU)
- 2x 80 GB GPUs for 70B-scale models (`device_map="auto"`)

```bash
pip install -r requirements.txt
```

For gated models (LLaMA, Gemma), authenticate with Hugging Face:

```bash
huggingface-cli login
```

## Repository Structure

```
src/
  config.py                 # Model registry, paths, and constants
  data_loading.py           # Dataset loading (GSM8K, ARC, TruthfulQA, HellaSwag)
  evaluation.py             # Model evaluation on reasoning problems
  extraction.py             # Hidden state extraction at all layers
  run_pipeline.py           # Master pipeline orchestrator (all phases)
  metrics/
    cka.py                  # Centered Kernel Alignment (linear + kernel)
    mnn.py                  # Mutual Nearest Neighbors overlap
    procrustes.py           # Procrustes distance
    statistics.py           # Bootstrap CIs, permutation tests, effect sizes
  experiments/
    exp01_cross_model_similarity.py   # Pairwise CKA across all model pairs
    exp02_correctness_probing.py      # Transfer probes for correctness
    exp03_causal_intervention.py      # Causal ablation of correctness subspace
    exp04_domain_specific.py          # Per-domain difficulty analysis
    exp05_convergence_trajectory.py   # CKA as function of layer depth
    exp06_transfer_probing.py         # Cross-model probe transfer matrix
    exp07_scale_convergence.py        # Scale-dependent convergence
    exp08_difficulty_stratified.py    # Difficulty-bin stratified CKA
    exp09_ensemble_diversity.py       # Ensemble diversity metrics
    exp10_pre_post_decision.py        # Pre-decision vs post-decision CKA
    extended/                         # Additional analyses
      b1_per_domain_difficulty.py     # Per-domain difficulty inversion
      b2_random_model_baseline.py     # Random initialization baseline
      b3_embedding_cka.py            # Embedding vs deep layer CKA
      b4_attention_entropy.py         # Attention entropy mechanism
      b6_agreement_threshold_cka.py   # Answer-agreement CKA control
      b7_per_layer_difficulty.py      # Layer-wise difficulty inversion
      base_model_comparison.py        # Base vs instruction-tuned control
      expanded_causal_ablation.py     # Relaxed causal ablation protocol
      head_causal_ablation.py         # Per-head causal ablation (MHA vs GQA)
    scale/                            # 70B-scale validation
      eval_70b.py                     # Evaluate 70B models
      extract_70b.py                  # Extract hidden states from 70B models
      difficulty_70b.py               # Difficulty inversion at 70B scale
      random_baseline.py              # Full-cohort random baseline
  ablations/                          # Robustness checks
    sample_size.py                    # Sample size stability
    kernel_cka.py                     # RBF kernel CKA comparison
    probe_regularization.py           # Probe regularization sweep
    prompt_sensitivity.py             # Prompt format sensitivity
    centering.py                      # Centered vs uncentered CKA
    metric_agreement.py               # MNN and SVCCA validation
  figures/                            # Figure generation scripts
  tables/                             # Table generation scripts
tests/                                # Unit tests
```

## Running the Full Pipeline

The master pipeline runs all experiments in sequence:

```bash
# Run everything (GPU required for phases 1 and 3)
python -m src.run_pipeline

# Run a specific phase
python -m src.run_pipeline --phase 1   # Evaluate models + extract hidden states
python -m src.run_pipeline --phase 2   # Similarity analysis (CKA, MNN, probes)
python -m src.run_pipeline --phase 3   # Causal interventions (GPU)
python -m src.run_pipeline --phase 4   # Ablation studies
python -m src.run_pipeline --phase 5   # Generate figures and tables
```

### Phase 1: Model Evaluation and Hidden State Extraction (GPU)

Evaluates all 14 models on 800 reasoning problems and extracts hidden-state activations at every layer.

```bash
python -m src.run_pipeline --phase 1
```

Output: `data/evaluations/*.json`, `data/hidden_states/*.npz`

Approximate time: 8--12 hours on a single 32 GB GPU.

### Phase 2: Analysis Experiments (CPU)

Runs all similarity, probing, and stratification analyses.

```bash
python -m src.run_pipeline --phase 2
```

Runs experiments 1, 2, 4--10 from `src/experiments/`.

### Phase 3: Causal Interventions (GPU)

Runs the causal ablation experiment (exp03), which requires loading models to measure prediction changes.

```bash
python -m src.run_pipeline --phase 3
```

### Phase 4: Ablation Studies (CPU)

Runs all robustness checks: sample size stability, kernel CKA, probe regularization sweep, prompt sensitivity, and metric agreement validation.

```bash
python -m src.run_pipeline --phase 4
```

### Phase 5: Figures and Tables

Generates all figures and LaTeX tables from computed metrics.

```bash
python -m src.run_pipeline --phase 5
```

## Running Individual Experiments

Each experiment can also be run standalone:

```bash
# Core experiments
python -m src.experiments.exp01_cross_model_similarity
python -m src.experiments.exp02_correctness_probing
python -m src.experiments.exp03_causal_intervention

# Extended analyses
python -m src.experiments.extended.b1_per_domain_difficulty
python -m src.experiments.extended.b4_attention_entropy
python -m src.experiments.extended.head_causal_ablation

# 70B scale validation (requires 2x 80GB GPUs)
python -m src.experiments.scale.eval_70b
python -m src.experiments.scale.extract_70b
python -m src.experiments.scale.difficulty_70b
python -m src.experiments.scale.random_baseline
```

## Models

The 14-model core cohort spans 8 architectural families:

| Model | Family | Params | HuggingFace ID |
|-------|--------|--------|----------------|
| Qwen-2.5-1.5B | Qwen | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| SmolLM2-1.7B | SmolLM | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Gemma-1-2B | Gemma | 2.0B | `google/gemma-2b-it` |
| Gemma-2-2B | Gemma | 2.6B | `google/gemma-2-2b-it` |
| Qwen-2.5-3B | Qwen | 3.0B | `Qwen/Qwen2.5-3B-Instruct` |
| LLaMA-3.2-3B | LLaMA | 3.2B | `meta-llama/Llama-3.2-3B-Instruct` |
| Phi-3.5-Mini | Phi | 3.8B | `microsoft/Phi-3.5-mini-instruct` |
| Qwen-2.5-7B | Qwen | 7.0B | `Qwen/Qwen2.5-7B-Instruct` |
| Mistral-7B | Mistral | 7.0B | `mistralai/Mistral-7B-Instruct-v0.3` |
| OLMo-2-7B | OLMo | 7.0B | `allenai/OLMo-2-1124-7B-Instruct` |
| InternLM-2.5-7B | InternLM | 7.0B | `internlm/internlm2_5-7b-chat` |
| Nemotron-8B | LLaMA | 8.0B | `nvidia/Nemotron-4-340B-Instruct` |
| Gemma-2-9B | Gemma | 9.0B | `google/gemma-2-9b-it` |
| Qwen-2.5-14B | Qwen | 14.0B | `Qwen/Qwen2.5-14B-Instruct` |

Scale validation (requires 2x 80 GB GPUs):

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| LLaMA-3.1-70B | 70B | `meta-llama/Llama-3.1-70B-Instruct` |
| Qwen-2.5-72B | 72B | `Qwen/Qwen2.5-72B-Instruct` |

## Datasets

All datasets are loaded automatically via the `datasets` library:

- **GSM8K** (200 problems): Multi-step math reasoning
- **ARC-Challenge** (200 problems): Science question answering
- **TruthfulQA** (200 problems): Common misconceptions / truthfulness
- **HellaSwag** (200 problems): Commonsense completion

Problems are selected by shuffling each dataset with seed 42 and taking the first 200.

## Key Results

| Finding | Metric | Value |
|---------|--------|-------|
| Hard problems (0--4/14 correct) | CKA | 0.897 |
| Easy problems (10--14/14 correct) | CKA | 0.830 |
| Inversion gap (1.5B--14B) | Delta CKA | +0.067 |
| Inversion gap (70B scale) | Delta CKA | +0.062 |
| Pre-decision layers | CKA | 0.875 |
| Post-decision layers | CKA | 0.274 |
| Transfer probe accuracy | Accuracy | 66% |
| Causal ablation flip rate | Flip rate | 1.5% |
| Random vs. random models | CKA | 0.864 |
| Trained vs. trained models | CKA | 0.612 |
| Attention entropy correlation | Pearson r | -0.43 |

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{convergence2026,
  title={Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
