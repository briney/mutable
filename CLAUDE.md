# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mutable is a PyTorch framework for generative modeling of antibody somatic hypermutation. It implements a two-phase training pipeline:
- **Phase 1 (Denoising)**: BART-style encoder-decoder pre-training with information-weighted masking
- **Phase 2 (Flow matching)**: Continuous generative model (OT-CFM) that learns to transform germline latents into mutated latents

## Common Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[adaptive]"   # includes torchdiffeq for adaptive ODE solvers

# Run all tests
pytest tests/ -v

# Skip slow tests (e2e training smoke tests)
pytest tests/ -m "not slow"

# Run a single test file
pytest tests/unit/test_tokenizer.py -v

# Run a single test
pytest tests/unit/eval/test_classification.py::test_masked_accuracy_metric -v

# CLI
mutable --help
mutable train-denoising data.train=/path/to/train.csv
mutable train-flow -p outputs/denoising/final data.train=/path/to/pairs.csv
mutable model-size model=large
```

## Architecture

### Two-Phase Model Pipeline

**Phase 1 — `MutableForDenoising`** (`models/mutable_denoising.py`):
Corrupted sequence → `MutableEncoder` (bidirectional, rotary embeddings) → `PerceiverBottleneck` (variable-length → fixed latents via cross-attention) → `MutableDecoder` (autoregressive, cross-attends to latents) → LM head → reconstructed sequence.

**Phase 2 — `MutableFlowMatching`** (`models/flow_matching.py`):
Freezes the Phase 1 encoder/decoder. A `FlowNetwork` predicts velocity fields conditioned on timestep and mutation intensity via AdaLN. At inference, ODE integration (Euler/RK4/adaptive) generates mutated latents from germline latents.

The **Perceiver bottleneck** (`models/bottleneck.py`) is the bridge: it compresses variable-length encoder outputs into fixed-size latents that the flow model operates on.

### Configuration System

Hydra/OmegaConf with hierarchical YAML composition. Main config: `src/mutable/configs/config.yaml` composes defaults from subdirectories (`model/`, `train/`, `data/`, `masking/`, `flow/`, `eval/`, `noise/`, `log/`). CLI overrides use dotpath syntax: `model=small train.learning_rate=1e-3`.

`config/from_hydra.py` converts Hydra `DictConfig` objects into HuggingFace-compatible `PretrainedConfig` instances via introspection-based field filtering.

### Masking System

`InformationWeightedMasker` (`masking/masking.py`) prioritizes biologically informative positions (CDRs, nongermline residues) during masking. Uses Gumbel-top-K ("sampled") or deterministic top-K ("ranked") selection. Falls back to `UniformMasker` when annotation columns are absent. Masking is applied at batch level inside `DenoisingTrainer`.

### Evaluation Harness

Registry-based metric system (`eval/registry.py`) using `@register_metric` decorator. Metrics implement a `Metric` protocol (`eval/base.py`). The `Evaluator` (`eval/evaluator.py`) coordinates multi-dataset, multi-metric, region-aware evaluation (CDR/FWR-level metrics via `eval/regions.py`).

### Tokenizer

Custom 32-token amino acid vocabulary (`tokenizer.py`). Special tokens: `<cls>`(0), `<pad>`(1), `<eos>`(2), `<sep>`(29), `<mask>`(31). Heavy and light chains are concatenated with `<sep>`.

## Key Patterns

- **Model base classes** (`models/base.py`): `MutablePreTrainedModel` extends HuggingFace `PreTrainedModel`. Mixins: `FreezeBaseModelMixin`, `ParameterCountMixin`.
- **Trainer extension**: `DenoisingTrainer` and `FlowTrainer` extend `transformers.Trainer` with custom batch processing (masking, flow matching loss).
- **Outputs**: Frozen dataclasses in `outputs/` following HuggingFace `ModelOutput` conventions.
- **Neural modules** (`modules/`): Composable attention, FFN, embedding, and layer components. SwiGLU activations, rotary embeddings.

## Data Format

- **Denoising**: CSV with `sequence_heavy`, `sequence_light` columns. Optional annotation columns: `cdr_mask_heavy/light`, `nongermline_mask_heavy/light`, `gene_segment_mask_heavy/light` (for weighted masking).
- **Flow matching**: CSV with `germline_heavy/light` and `mutated_heavy/light` column pairs.
