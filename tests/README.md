# Eval Harness Test Suite

Tests for the multi-dataset, multi-metric evaluation system in `src/mutable/eval/`.

Run the full suite:
```bash
python -m pytest tests/ -v
```

Exclude slow tests:
```bash
python -m pytest tests/ -m "not slow"
```

---

## Metric Protocol and Registry

**`unit/eval/test_base.py`** — Verifies that all metrics conform to the `Metric` protocol and that `MetricBase` provides correct defaults.

| Test | Purpose |
|------|---------|
| `test_classification_metrics_satisfy_protocol` | `isinstance(metric, Metric)` for Loss, MaskedAccuracy, Perplexity |
| `test_contact_metric_satisfies_protocol` | `isinstance(PrecisionAtLMetric(), Metric)` |
| `test_metric_base_defaults` | `state_tensors()` returns `[]`, `state_objects()` returns `None` |
| `test_class_attributes_present` | Every metric has `name`, `requires_coords`, `needs_attentions` |
| `test_classification_no_coords_no_attentions` | Classification metrics don't require coords or attentions |
| `test_contact_requires_coords_and_attentions` | P@L metric requires both |

**`unit/eval/test_registry.py`** — Tests the `@register_metric` decorator, `METRIC_REGISTRY`, and `build_metrics` factory including config merging.

| Test | Purpose |
|------|---------|
| `test_all_metrics_registered` | Registry contains `loss`, `masked_accuracy`, `perplexity`, `p_at_l` |
| `test_register_metric_decorator` | Applying decorator adds a class to the registry |
| `test_build_metrics_default_config` | Default config builds 3 classification metrics (no coords) |
| `test_build_metrics_with_coords` | With coords, also builds P@L (4 total) |
| `test_build_metrics_only_whitelist` | `metrics.only: [loss]` returns only LossMetric |
| `test_build_metrics_disabled` | `metrics.loss.enabled: false` excludes LossMetric |
| `test_num_layers_auto_resolution` | `num_layers=null` resolves to `ceil(num_encoder_layers * 0.1)` |
| `test_config_merging_per_dataset` | Per-dataset overrides take precedence over global config |

---

## Eval Masking

**`unit/eval/test_masking.py`** — Tests `EvalMasker` reproducibility, mask rate accuracy, and special token handling.

| Test | Purpose |
|------|---------|
| `test_reproducibility_same_seed` | Two maskers with seed=42 produce identical masks |
| `test_different_seeds_differ` | Different seeds produce different masks |
| `test_mask_rate_approximate` | Actual mask rate is within tolerance of 0.15 |
| `test_special_tokens_not_masked` | BOS, EOS, SEP positions are never masked |
| `test_returns_masked_ids_and_labels` | Returns correct dtypes (`long`, `bool`) and shapes |
| `test_mask_token_id` | Masked positions contain `mask_token_id=31` |

---

## Classification Metrics

**`unit/eval/test_classification.py`** — Tests `LossMetric`, `MaskedAccuracyMetric`, and `PerplexityMetric` computation, edge cases, and state management.

| Test | Purpose |
|------|---------|
| `test_loss_perfect_predictions` | Perfect logits yield loss near 0 |
| `test_loss_random_predictions` | Random logits yield loss near `-log(1/V)` |
| `test_reset_clears_state` | `reset()` zeros accumulators, `compute()` returns 0 |
| `test_multi_batch_accumulation` | Two identical `update()` calls give same average loss |
| `test_accuracy_perfect` | All-correct logits yield accuracy = 1.0 |
| `test_accuracy_zero` | All-wrong logits yield accuracy = 0.0 |
| `test_perplexity_value` | Verifies `ppl = exp(loss)` relationship |
| `test_perplexity_clamp` | Very high loss is clamped at `exp(100)`, no overflow |
| `test_metrics_ignore_minus_100` | Positions with `labels=-100` are excluded |
| `test_metrics_ignore_unmasked` | Only `mask_labels=True` positions contribute |
| `test_state_tensors_round_trip` | `state_tensors()` -> `load_state_tensors()` -> `compute()` is consistent |

---

## Contact Prediction (P@L)

**`unit/eval/test_contact.py`** — Tests `PrecisionAtLMetric` and its helper functions: contact map computation, APC, pair masking, and chain-aware filtering.

| Test | Purpose |
|------|---------|
| `test_compute_contact_map_simple` | Known CA coordinates produce expected binary contact map |
| `test_compute_contact_map_nan_handling` | NaN coordinates produce no contacts |
| `test_apply_apc` | Verified APC correction on a known 2x2 matrix |
| `test_extract_attention_contacts_last_layer` | `num_layers=1` uses only the last attention layer |
| `test_extract_attention_contacts_mean` | Head aggregation averages correctly |
| `test_none_attentions` | Returns `None` when no attentions provided |
| `test_build_pair_mask_min_seq_sep` | Pairs closer than `min_seq_sep` are excluded |
| `test_build_pair_mask_special_tokens` | Special token positions excluded from all pairs |
| `test_filter_by_mode_all` | `"all"` mode keeps all valid pairs |
| `test_filter_by_mode_cross_chain` | Only heavy-light pairs survive |
| `test_filter_by_mode_intra_heavy` | Only heavy-heavy pairs survive |
| `test_filter_by_mode_intra_light` | Only light-light pairs survive |
| `test_filter_by_mode_cdr_contact` | At least one CDR position required per pair |
| `test_p_at_l_perfect_attention` | Attention matching contacts yields precision > 0.5 |
| `test_p_at_l_multi_mode` | Multiple modes return separate result keys |
| `test_state_tensors_per_mode` | Each mode has its own `[correct, total]` accumulator |

---

## Region Evaluation

**`unit/eval/test_regions.py`** — Tests `derive_chain_ids`, `extract_region_masks`, and `aggregate_region_masks` for antibody region breakdowns.

| Test | Purpose |
|------|---------|
| `test_derive_chain_ids_basic` | `<sep>` at position 10 gives 0s before, 1s after |
| `test_derive_chain_ids_no_sep` | No `<sep>` token results in all zeros |
| `test_derive_chain_ids_batch` | Batch of 4 with varying `<sep>` positions |
| `test_extract_region_masks_cdr` | Known CDR positions produce correct HCDR/LCDR masks |
| `test_extract_region_masks_framework` | Framework regions inferred correctly between CDRs |
| `test_extract_region_masks_returns_all_14` | All 14 `AntibodyRegion` values present in output |
| `test_special_tokens_excluded` | BOS/SEP/EOS are `False` in every region mask |
| `test_aggregate_cdr_group` | `aggregate_by="cdr"` unions all CDR masks |
| `test_aggregate_chain` | `aggregate_by="chain"` gives heavy/light splits |

**`unit/eval/test_region_config.py`** — Tests `RegionEvalConfig` dataclass defaults and `build_region_eval_config` factory.

| Test | Purpose |
|------|---------|
| `test_defaults` | `enabled=False`, all region/aggregate bools `False` |
| `test_get_enabled_regions` | Returns only individually enabled region names |
| `test_get_enabled_aggregates` | Returns only enabled aggregate group names |
| `test_has_any_enabled` | `True` when any region or aggregate is enabled |
| `test_build_from_dict` | Builds config from a plain dict |
| `test_empty_dict` | Empty dict returns default config |
| `test_none_dict` | `None` returns default config |
| `test_unknown_keys_ignored` | Unrecognized keys don't raise errors |

---

## Eval Datasets

**`unit/eval/test_eval_dataset.py`** — Tests `EvalDenoisingDataset` output format, clean labels, and annotation mask handling.

| Test | Purpose |
|------|---------|
| `test_output_keys` | Returns `input_ids`, `labels`, `special_tokens_mask`, `chain_ids` |
| `test_labels_equal_input_ids` | Clean eval: no masking applied, labels match input |
| `test_special_tokens_mask` | BOS, SEP, EOS positions are `True` |
| `test_chain_ids_from_sep` | 0 before `<sep>`, 1 after |
| `test_annotation_masks_present` | CDR/nongermline masks included when columns provided |
| `test_annotation_masks_absent` | Annotation keys absent when columns not provided |

**`unit/eval/test_structure_dataset.py`** — Tests `AntibodyStructureDataset` and `StructureCollator` with synthetic PDB files. Requires BioPython (`pytest.importorskip`).

| Test | Purpose |
|------|---------|
| `test_dataset_output_keys` | Returns `input_ids`, `labels`, `special_tokens_mask`, `coords`, `chain_boundary` |
| `test_coords_shape` | Coords are `(seq_len, 3, 3)` |
| `test_coords_nan_at_special_tokens` | BOS, SEP, EOS positions have NaN coords |
| `test_chain_boundary_at_sep` | `chain_boundary` matches `<sep>` token position |
| `test_collator_padding` | Variable-length sequences padded to max length |
| `test_collator_coords_nan_padding` | Padded coordinate positions are NaN |

**`unit/eval/test_structure_parser.py`** — Tests `parse_paired_structure` with synthetic PDB fixtures. Requires BioPython.

| Test | Purpose |
|------|---------|
| `test_parse_synthetic_pdb` | Parses sequences and coord shapes from minimal PDB |
| `test_coord_atom_order` | `coords[:, 0]` is N, `[:, 1]` is CA, `[:, 2]` is C |
| `test_fallback_chain_ids` | Chains A/B fall back to first two protein chains |
| `test_missing_atoms_nan` | Missing backbone atoms produce NaN coordinates |

---

## Integration: Evaluator

**`integration/eval/test_evaluator.py`** — End-to-end tests for the `Evaluator` class using a tiny `MutableForDenoising` model and synthetic data.

| Test | Purpose |
|------|---------|
| `test_evaluate_returns_classification_metrics` | Returns `loss`, `mask_acc`, `ppl` as finite floats |
| `test_evaluate_all_multiple_datasets` | `evaluate_all()` returns nested dict with both dataset names |
| `test_evaluate_with_regions` | Region-enabled config produces `"region/"` prefixed keys |
| `test_evaluate_no_structure_skips_p_at_l` | Sequence-only data excludes P@L metric |
| `test_model_eval_mode` | Model is set to `eval()` mode during evaluation |
| `test_metric_reset_between_evals` | Consecutive evaluations return independent results |
| `test_evaluate_max_samples` | `max_samples` limits how many samples a metric processes |

---

## End-to-End Smoke Test

**`e2e/test_training_smoke.py`** — Full training loop with evaluation. Marked `@pytest.mark.slow`.

| Test | Purpose |
|------|---------|
| `test_denoising_training_with_eval` | Runs `DenoisingTrainer.train()` for 2 steps with `eval_steps=1`, verifies `trainer.evaluate()` returns finite metrics |
