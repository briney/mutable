# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Metric registration and factory utilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from .base import Metric

if TYPE_CHECKING:
    pass

__all__ = ["METRIC_REGISTRY", "register_metric", "build_metrics"]

METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(name: str):
    """Class decorator to register a metric.

    Parameters
    ----------
    name
        Registry name (used in config to enable/disable).
    """

    def decorator(cls):
        METRIC_REGISTRY[name] = cls
        if not hasattr(cls, "name") or cls.name == "":
            cls.name = name
        return cls

    return decorator


def _get_metric_config(
    cfg: DictConfig,
    metric_name: str,
    eval_name: str | None,
) -> dict:
    """Merge global and per-dataset metric config.

    Priority (highest to lowest):
    1. Per-dataset override: ``data.eval.{eval_name}.metrics.{metric_name}``
    2. Global default: ``eval.metrics.{metric_name}``
    3. Empty dict (all defaults)
    """
    result = {}

    # Global defaults
    global_cfg = OmegaConf.select(cfg, f"eval.metrics.{metric_name}", default=None)
    if global_cfg is not None:
        result.update(OmegaConf.to_container(global_cfg, resolve=True))

    # Per-dataset overrides
    if eval_name is not None:
        per_ds = OmegaConf.select(
            cfg, f"data.eval.{eval_name}.metrics.{metric_name}", default=None
        )
        if per_ds is not None:
            result.update(OmegaConf.to_container(per_ds, resolve=True))

    return result


def build_metrics(
    cfg: DictConfig,
    has_coords: bool = False,
    eval_name: str | None = None,
) -> list[Metric]:
    """Build metric instances from config.

    Parameters
    ----------
    cfg
        Full Hydra config.
    has_coords
        Whether the eval dataset provides coordinate data.
    eval_name
        Name of the eval dataset (for per-dataset config overrides).

    Returns
    -------
    list[Metric]
        Instantiated metric objects.
    """
    # Trigger registration by importing metrics subpackage
    import mutable.eval.metrics  # noqa: F401

    # Check for per-dataset whitelist
    whitelist = None
    if eval_name is not None:
        only = OmegaConf.select(
            cfg, f"data.eval.{eval_name}.metrics.only", default=None
        )
        if only is not None:
            whitelist = set(OmegaConf.to_container(only, resolve=True))

    metrics = []
    for name, cls in METRIC_REGISTRY.items():
        # Check whitelist
        if whitelist is not None and name not in whitelist:
            continue

        # Get merged config for this metric
        metric_cfg = _get_metric_config(cfg, name, eval_name)

        # Check if explicitly disabled
        if not metric_cfg.pop("enabled", True):
            continue

        # Filter by dataset capability
        if cls.requires_coords and not has_coords:
            continue

        # Remove meta-keys not passed to constructor
        metric_cfg.pop("max_samples", None)
        metric_cfg.pop("seed", None)

        # Resolve dynamic parameters
        if name == "p_at_l" and metric_cfg.get("num_layers") is None:
            num_encoder_layers = OmegaConf.select(
                cfg, "model.num_encoder_layers", default=12
            )
            metric_cfg["num_layers"] = max(1, math.ceil(num_encoder_layers * 0.1))

        metrics.append(cls(**metric_cfg))

    return metrics
