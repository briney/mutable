# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Evaluation harness for Mutable denoising pre-training."""

from .base import Metric, MetricBase
from .evaluator import Evaluator
from .masking import EvalMasker
from .region_config import RegionEvalConfig, build_region_eval_config
from .regions import (
    AGGREGATE_GROUP_NAMES,
    CDR_REGIONS,
    FWR_REGIONS,
    HEAVY_REGIONS,
    INDIVIDUAL_REGION_NAMES,
    LIGHT_REGIONS,
    AntibodyRegion,
    aggregate_region_masks,
    extract_region_masks,
)
from .registry import METRIC_REGISTRY, build_metrics, register_metric

# Trigger metric registration
import mutable.eval.metrics  # noqa: F401

__all__ = [
    "Metric",
    "MetricBase",
    "Evaluator",
    "EvalMasker",
    "register_metric",
    "build_metrics",
    "METRIC_REGISTRY",
    "AntibodyRegion",
    "CDR_REGIONS",
    "FWR_REGIONS",
    "HEAVY_REGIONS",
    "LIGHT_REGIONS",
    "INDIVIDUAL_REGION_NAMES",
    "AGGREGATE_GROUP_NAMES",
    "extract_region_masks",
    "aggregate_region_masks",
    "RegionEvalConfig",
    "build_region_eval_config",
]
