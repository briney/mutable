# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Configuration for region-specific evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .regions import AGGREGATE_GROUP_NAMES, INDIVIDUAL_REGION_NAMES

__all__ = ["RegionEvalConfig", "build_region_eval_config"]


@dataclass
class RegionEvalConfig:
    """Configuration for region-specific evaluation.

    Parameters
    ----------
    enabled
        Master switch to enable/disable region evaluation.
    mode
        Evaluation mode: "standard" or "per_position".
    max_samples
        Maximum number of samples for region evaluation (None = all).
    seed
        Random seed for subsetting.
    hcdr1 ... lfwr4
        Enable evaluation for individual regions.
    all_cdr ... nongermline
        Enable aggregate statistics.
    """

    enabled: bool = False
    mode: str = "standard"
    max_samples: int | None = None
    seed: int = 42

    # Individual CDR regions (6 total)
    hcdr1: bool = False
    hcdr2: bool = False
    hcdr3: bool = False
    lcdr1: bool = False
    lcdr2: bool = False
    lcdr3: bool = False

    # Individual framework regions (8 total)
    hfwr1: bool = False
    hfwr2: bool = False
    hfwr3: bool = False
    hfwr4: bool = False
    lfwr1: bool = False
    lfwr2: bool = False
    lfwr3: bool = False
    lfwr4: bool = False

    # Aggregate groups (7 total)
    all_cdr: bool = False
    all_fwr: bool = False
    heavy: bool = False
    light: bool = False
    overall: bool = False
    germline: bool = False
    nongermline: bool = False

    def get_enabled_regions(self) -> set[str]:
        """Return set of individually enabled region names."""
        return {r for r in INDIVIDUAL_REGION_NAMES if getattr(self, r, False)}

    def get_enabled_aggregates(self) -> set[str]:
        """Return set of enabled aggregate group names."""
        return {a for a in AGGREGATE_GROUP_NAMES if getattr(self, a, False)}

    def has_any_enabled(self) -> bool:
        """Check if any region or aggregate is enabled."""
        return bool(self.get_enabled_regions() or self.get_enabled_aggregates())


def build_region_eval_config(cfg_dict: dict) -> RegionEvalConfig:
    """Build RegionEvalConfig from a dictionary.

    Parameters
    ----------
    cfg_dict
        Dictionary with config values. Unknown keys are ignored.

    Returns
    -------
    RegionEvalConfig
    """
    if not cfg_dict:
        return RegionEvalConfig()

    config_kwargs = {}
    for field_name in RegionEvalConfig.__dataclass_fields__:
        if field_name in cfg_dict:
            config_kwargs[field_name] = cfg_dict[field_name]

    return RegionEvalConfig(**config_kwargs)
