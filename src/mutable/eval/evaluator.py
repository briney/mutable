# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Evaluator orchestrator for multi-dataset, multi-metric evaluation."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from .base import Metric
from .masking import EvalMasker
from .region_config import RegionEvalConfig, build_region_eval_config
from .regions import (
    CDR_REGIONS,
    FWR_REGIONS,
    HEAVY_REGIONS,
    LIGHT_REGIONS,
    AntibodyRegion,
    extract_region_masks,
)
from .registry import build_metrics

__all__ = ["Evaluator"]


def _get_model_device(model: Any, accelerator: Any | None = None) -> torch.device:
    """Get the device the model is on."""
    if accelerator is not None:
        return accelerator.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _derive_chain_ids(input_ids: Tensor, sep_token_id: int = 29) -> Tensor:
    """Derive chain_ids from input_ids by finding the <sep> token.

    Parameters
    ----------
    input_ids
        Token IDs of shape (batch, seq_len).
    sep_token_id
        Token ID of the ``<sep>`` token.

    Returns
    -------
    Tensor
        Integer tensor of shape (batch, seq_len).
        0 for positions before <sep> (heavy chain),
        1 for positions after <sep> (light chain).
        The <sep> position itself gets 0 (treated as boundary).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        sep_positions = (input_ids[b] == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_pos = sep_positions[0].item()
            chain_ids[b, sep_pos + 1 :] = 1
    return chain_ids


class Evaluator:
    """Orchestrator for multi-dataset, multi-metric evaluation.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.
    model
        The model to evaluate.
    tokenizer
        Tokenizer (used for mask_token_id, etc.).
    accelerator
        Optional Accelerate accelerator for distributed evaluation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: Any,
        tokenizer: Any,
        accelerator: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator

        # Cache built metrics per eval dataset name
        self._metrics_cache: dict[str, list[Metric]] = {}
        self._needs_attentions_cache: dict[str, bool] = {}

        # Build eval masker from config
        eval_cfg = OmegaConf.select(cfg, "eval", default=None)
        mask_rate = 0.15
        mask_seed = 42
        if eval_cfg is not None:
            masking_cfg = OmegaConf.select(eval_cfg, "masking", default=None)
            if masking_cfg is not None:
                mask_rate = getattr(masking_cfg, "mask_rate", 0.15)
                mask_seed = getattr(masking_cfg, "seed", 42)

        mask_token_id = getattr(tokenizer, "mask_token_id", 31)
        self.eval_masker = EvalMasker(
            mask_rate=mask_rate,
            mask_token_id=mask_token_id,
            seed=mask_seed,
        )

        # Build region config
        region_dict = OmegaConf.to_container(
            OmegaConf.select(cfg, "eval.regions", default=OmegaConf.create({})),
            resolve=True,
        )
        self.region_cfg = build_region_eval_config(region_dict)

    def _get_metrics(self, eval_name: str, has_coords: bool = False) -> list[Metric]:
        """Get or build metrics for an eval dataset."""
        if eval_name not in self._metrics_cache:
            self._metrics_cache[eval_name] = build_metrics(
                self.cfg, has_coords=has_coords, eval_name=eval_name
            )
        return self._metrics_cache[eval_name]

    def _needs_attentions(self, eval_name: str, has_coords: bool = False) -> bool:
        """Check if any metric for this dataset needs attention weights."""
        if eval_name not in self._needs_attentions_cache:
            metrics = self._get_metrics(eval_name, has_coords)
            self._needs_attentions_cache[eval_name] = any(
                m.needs_attentions for m in metrics
            )
        return self._needs_attentions_cache[eval_name]

    def _get_metric_limits(self, eval_name: str) -> dict[str, int | None]:
        """Get per-metric max_samples from config."""
        limits: dict[str, int | None] = {}
        metrics = self._get_metrics(eval_name)
        for m in metrics:
            metric_cfg = OmegaConf.select(
                self.cfg, f"eval.metrics.{m.name}", default=None
            )
            if metric_cfg is not None:
                limits[m.name] = getattr(metric_cfg, "max_samples", None)
            else:
                limits[m.name] = None
        return limits

    def _gather_metric_states(self, metrics: list[Metric]) -> None:
        """Gather metric states across distributed processes."""
        if self.accelerator is None or not self.accelerator.use_distributed:
            return

        for metric in metrics:
            # Try object-based gathering first
            objects = metric.state_objects()
            if objects is not None:
                gathered = self.accelerator.gather_object(objects)
                metric.load_state_objects(gathered)
                continue

            # Tensor-based gathering
            tensors = metric.state_tensors()
            if not tensors:
                continue

            gathered_tensors = []
            for t in tensors:
                t = t.to(self.accelerator.device)
                gathered = self.accelerator.reduce(t, reduction="sum")
                gathered_tensors.append(gathered)
            metric.load_state_tensors(gathered_tensors)

    def evaluate(
        self,
        eval_loader: DataLoader,
        eval_name: str,
    ) -> dict[str, float]:
        """Evaluate on a single dataset.

        Parameters
        ----------
        eval_loader
            DataLoader for the eval dataset.
        eval_name
            Name of the eval dataset (for config lookup and metric prefixing).

        Returns
        -------
        dict[str, float]
            Flat dictionary of metric names to values.
        """
        # Determine if dataset has coordinates
        has_coords = False
        if hasattr(eval_loader.dataset, "has_coords"):
            has_coords = eval_loader.dataset.has_coords

        metrics = self._get_metrics(eval_name, has_coords)
        if not metrics:
            return {}

        # Reset all metrics
        for m in metrics:
            m.reset()

        # Check if we need attention weights
        need_attentions = self._needs_attentions(eval_name, has_coords)

        # Per-metric sample limits
        metric_limits = self._get_metric_limits(eval_name)
        metric_sample_counts: dict[str, int] = {m.name: 0 for m in metrics}

        # Create eval masker generator
        device = _get_model_device(self.model, self.accelerator)
        generator = self.eval_masker.get_generator(device)

        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }

                batch_size = batch["input_ids"].shape[0]

                # Apply eval masking
                masked_ids, mask_labels = self.eval_masker.apply_mask(
                    batch, generator=generator
                )

                # Forward pass
                outputs = self.model(
                    input_ids=masked_ids,
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_attentions=need_attentions,
                    return_dict=True,
                )

                # Update each metric
                for m in metrics:
                    limit = metric_limits.get(m.name)
                    if limit is not None and metric_sample_counts[m.name] >= limit:
                        continue
                    m.update(outputs, batch, mask_labels)
                    metric_sample_counts[m.name] += batch_size

        # Gather states across processes
        self._gather_metric_states(metrics)

        # Compute final values
        results: dict[str, float] = {}
        for m in metrics:
            results.update(m.compute())

        # Region evaluation
        if self.region_cfg.enabled and self.region_cfg.has_any_enabled():
            region_results = self._evaluate_regions(eval_loader, device, generator)
            results.update(region_results)

        return results

    def _evaluate_regions(
        self,
        eval_loader: DataLoader,
        device: torch.device,
        generator: torch.Generator,
    ) -> dict[str, float]:
        """Run region-based metric breakdowns.

        Computes accuracy/loss/perplexity per antibody region by intersecting
        eval mask_labels with region-specific masks.

        Parameters
        ----------
        eval_loader
            DataLoader for the eval dataset.
        device
            Device to run on.
        generator
            Re-seeded generator for reproducible masking.

        Returns
        -------
        dict[str, float]
            Results prefixed with ``"region/"``.
        """
        # Re-seed generator for reproducibility (same masks as main eval)
        generator = self.eval_masker.get_generator(device)

        enabled_regions = self.region_cfg.get_enabled_regions()
        enabled_aggregates = self.region_cfg.get_enabled_aggregates()

        # Determine which AntibodyRegion enums we need
        needed_regions: set[AntibodyRegion] = set()
        for name in enabled_regions:
            try:
                needed_regions.add(AntibodyRegion(name))
            except ValueError:
                pass

        # For aggregates, add the component regions
        if "all_cdr" in enabled_aggregates:
            needed_regions.update(CDR_REGIONS)
        if "all_fwr" in enabled_aggregates:
            needed_regions.update(FWR_REGIONS)
        if "heavy" in enabled_aggregates:
            needed_regions.update(HEAVY_REGIONS)
        if "light" in enabled_aggregates:
            needed_regions.update(LIGHT_REGIONS)
        if "overall" in enabled_aggregates:
            needed_regions.update(AntibodyRegion)

        if not needed_regions and "germline" not in enabled_aggregates and "nongermline" not in enabled_aggregates:
            return {}

        # Accumulators: {region_name: {correct, total_loss, count}}
        region_accumulators: dict[str, dict[str, float]] = {}
        sample_count = 0
        max_samples = getattr(self.region_cfg, "max_samples", None)

        needs_germline = "germline" in enabled_aggregates
        needs_nongermline = "nongermline" in enabled_aggregates

        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                if max_samples is not None and sample_count >= max_samples:
                    break

                batch = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                batch_size = batch["input_ids"].shape[0]

                # Apply same eval masking
                masked_ids, mask_labels = self.eval_masker.apply_mask(
                    batch, generator=generator
                )

                # Forward pass (no attentions needed for region eval)
                outputs = self.model(
                    input_ids=masked_ids,
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )

                logits = outputs["logits"]
                labels = batch["labels"]
                valid = mask_labels & (labels != -100)

                # Derive chain_ids if not in batch
                if "chain_ids" not in batch:
                    batch["chain_ids"] = _derive_chain_ids(batch["input_ids"])

                # Ensure special_tokens_mask exists
                if "special_tokens_mask" not in batch:
                    batch["special_tokens_mask"] = torch.zeros_like(
                        batch["input_ids"], dtype=torch.bool
                    )

                # Extract region masks (requires cdr_mask)
                if "cdr_mask" in batch and needed_regions:
                    try:
                        region_masks = extract_region_masks(batch, needed_regions)
                    except (ValueError, KeyError) as e:
                        warnings.warn(f"Region mask extraction failed: {e}")
                        region_masks = {}

                    # Compute per-region metrics
                    for region, rmask in region_masks.items():
                        region_valid = valid & rmask
                        count = region_valid.sum().item()
                        if count == 0:
                            continue

                        region_name = region.value
                        if region_name not in region_accumulators:
                            region_accumulators[region_name] = {
                                "correct": 0.0,
                                "total_loss": 0.0,
                                "count": 0,
                            }
                        acc = region_accumulators[region_name]

                        # Accuracy
                        preds = logits.argmax(dim=-1)
                        correct = ((preds == labels) & region_valid).sum().item()
                        acc["correct"] += correct

                        # Loss
                        logits_flat = logits.view(-1, logits.size(-1))
                        labels_flat = labels.view(-1)
                        region_valid_flat = region_valid.view(-1)
                        loss = torch.nn.functional.cross_entropy(
                            logits_flat[region_valid_flat],
                            labels_flat[region_valid_flat],
                            reduction="sum",
                        ).item()
                        acc["total_loss"] += loss
                        acc["count"] += count

                # Germline/nongermline metrics
                if (needs_germline or needs_nongermline) and "nongermline_mask" in batch:
                    nongermline_mask = batch["nongermline_mask"]

                    if needs_germline:
                        germline_valid = valid & (nongermline_mask == 0)
                        count = germline_valid.sum().item()
                        if count > 0:
                            if "germline" not in region_accumulators:
                                region_accumulators["germline"] = {
                                    "correct": 0.0,
                                    "total_loss": 0.0,
                                    "count": 0,
                                }
                            acc = region_accumulators["germline"]
                            preds = logits.argmax(dim=-1)
                            acc["correct"] += ((preds == labels) & germline_valid).sum().item()
                            logits_flat = logits.view(-1, logits.size(-1))
                            labels_flat = labels.view(-1)
                            gv_flat = germline_valid.view(-1)
                            acc["total_loss"] += torch.nn.functional.cross_entropy(
                                logits_flat[gv_flat], labels_flat[gv_flat], reduction="sum"
                            ).item()
                            acc["count"] += count

                    if needs_nongermline:
                        ng_valid = valid & (nongermline_mask == 1)
                        count = ng_valid.sum().item()
                        if count > 0:
                            if "nongermline" not in region_accumulators:
                                region_accumulators["nongermline"] = {
                                    "correct": 0.0,
                                    "total_loss": 0.0,
                                    "count": 0,
                                }
                            acc = region_accumulators["nongermline"]
                            preds = logits.argmax(dim=-1)
                            acc["correct"] += ((preds == labels) & ng_valid).sum().item()
                            logits_flat = logits.view(-1, logits.size(-1))
                            labels_flat = labels.view(-1)
                            ng_flat = ng_valid.view(-1)
                            acc["total_loss"] += torch.nn.functional.cross_entropy(
                                logits_flat[ng_flat], labels_flat[ng_flat], reduction="sum"
                            ).item()
                            acc["count"] += count

                sample_count += batch_size

        # Compute final region results
        results: dict[str, float] = {}

        # Individual regions
        for region_name, acc in region_accumulators.items():
            if acc["count"] > 0 and (
                region_name in enabled_regions
                or region_name in enabled_aggregates
            ):
                results[f"region/{region_name}/accuracy"] = acc["correct"] / acc["count"]
                avg_loss = acc["total_loss"] / acc["count"]
                results[f"region/{region_name}/loss"] = avg_loss
                results[f"region/{region_name}/ppl"] = math.exp(min(avg_loss, 100.0))

        # Aggregates
        results.update(
            self._compute_region_aggregates(region_accumulators, enabled_aggregates)
        )

        return results

    def _compute_region_aggregates(
        self,
        accumulators: dict[str, dict[str, float]],
        enabled_aggregates: set[str],
    ) -> dict[str, float]:
        """Compute aggregate region metrics from per-region accumulators."""
        results: dict[str, float] = {}

        aggregate_groups = {
            "all_cdr": {r.value for r in CDR_REGIONS},
            "all_fwr": {r.value for r in FWR_REGIONS},
            "heavy": {r.value for r in HEAVY_REGIONS},
            "light": {r.value for r in LIGHT_REGIONS},
            "overall": {r.value for r in AntibodyRegion},
        }

        for agg_name, member_names in aggregate_groups.items():
            if agg_name not in enabled_aggregates:
                continue

            total_correct = 0.0
            total_loss = 0.0
            total_count = 0
            for rn in member_names:
                if rn in accumulators:
                    total_correct += accumulators[rn]["correct"]
                    total_loss += accumulators[rn]["total_loss"]
                    total_count += accumulators[rn]["count"]

            if total_count > 0:
                results[f"region/{agg_name}/accuracy"] = total_correct / total_count
                avg_loss = total_loss / total_count
                results[f"region/{agg_name}/loss"] = avg_loss
                results[f"region/{agg_name}/ppl"] = math.exp(min(avg_loss, 100.0))

        return results

    def evaluate_all(
        self,
        eval_loaders: dict[str, DataLoader],
    ) -> dict[str, dict[str, float]]:
        """Evaluate on all configured eval datasets.

        Parameters
        ----------
        eval_loaders
            Dictionary mapping eval dataset names to DataLoaders.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping eval names to metric result dicts.
        """
        all_results: dict[str, dict[str, float]] = {}
        for eval_name, loader in eval_loaders.items():
            all_results[eval_name] = self.evaluate(loader, eval_name)
        return all_results

    def clear_cache(self) -> None:
        """Clear cached metrics and attention flags."""
        self._metrics_cache.clear()
        self._needs_attentions_cache.clear()
