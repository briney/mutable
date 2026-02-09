# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for RegionEvalConfig and build_region_eval_config."""

import pytest

from mutable.eval.region_config import RegionEvalConfig, build_region_eval_config


class TestRegionEvalConfig:
    def test_defaults(self):
        cfg = RegionEvalConfig()
        assert cfg.enabled is False
        # All individual region bools default to False
        assert cfg.hcdr1 is False
        assert cfg.hcdr3 is False
        assert cfg.lfwr4 is False
        # All aggregate bools default to False
        assert cfg.all_cdr is False
        assert cfg.overall is False

    def test_get_enabled_regions(self):
        cfg = RegionEvalConfig(hcdr1=True, hcdr3=True, lcdr1=True)
        enabled = cfg.get_enabled_regions()
        assert "hcdr1" in enabled
        assert "hcdr3" in enabled
        assert "lcdr1" in enabled
        assert "hcdr2" not in enabled

    def test_get_enabled_aggregates(self):
        cfg = RegionEvalConfig(all_cdr=True, heavy=True)
        enabled = cfg.get_enabled_aggregates()
        assert "all_cdr" in enabled
        assert "heavy" in enabled
        assert "all_fwr" not in enabled

    def test_has_any_enabled(self):
        cfg = RegionEvalConfig()
        assert cfg.has_any_enabled() is False

        cfg2 = RegionEvalConfig(hcdr3=True)
        assert cfg2.has_any_enabled() is True

        cfg3 = RegionEvalConfig(all_cdr=True)
        assert cfg3.has_any_enabled() is True


class TestBuildRegionEvalConfig:
    def test_build_from_dict(self):
        cfg = build_region_eval_config({"enabled": True, "hcdr3": True})
        assert cfg.enabled is True
        assert cfg.hcdr3 is True
        assert cfg.hcdr1 is False  # default

    def test_empty_dict(self):
        cfg = build_region_eval_config({})
        assert cfg.enabled is False

    def test_none_dict(self):
        cfg = build_region_eval_config(None)
        assert cfg.enabled is False

    def test_unknown_keys_ignored(self):
        cfg = build_region_eval_config({"enabled": True, "nonexistent_key": True})
        assert cfg.enabled is True
