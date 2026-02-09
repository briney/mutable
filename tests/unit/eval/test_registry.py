# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for metric registry and build_metrics."""

import pytest
from omegaconf import OmegaConf

from mutable.eval.base import MetricBase
from mutable.eval.registry import METRIC_REGISTRY, build_metrics, register_metric
from mutable.eval.metrics.classification import LossMetric


class TestRegistry:
    """Test METRIC_REGISTRY contents and registration."""

    def test_all_metrics_registered(self):
        assert "loss" in METRIC_REGISTRY
        assert "masked_accuracy" in METRIC_REGISTRY
        assert "perplexity" in METRIC_REGISTRY
        assert "p_at_l" in METRIC_REGISTRY

    def test_register_metric_decorator(self):
        @register_metric("test_metric_unique")
        class _TestMetric(MetricBase):
            def update(self, outputs, batch, mask_labels):
                pass
            def compute(self):
                return {}
            def reset(self):
                pass

        assert "test_metric_unique" in METRIC_REGISTRY
        assert METRIC_REGISTRY["test_metric_unique"] is _TestMetric
        # Cleanup
        del METRIC_REGISTRY["test_metric_unique"]


class TestBuildMetrics:
    """Test build_metrics factory."""

    def test_build_metrics_default_config(self, eval_cfg):
        metrics = build_metrics(eval_cfg, has_coords=False)
        names = {type(m).__name__ for m in metrics}
        assert "LossMetric" in names
        assert "MaskedAccuracyMetric" in names
        assert "PerplexityMetric" in names
        # p_at_l excluded because no coords
        assert "PrecisionAtLMetric" not in names

    def test_build_metrics_with_coords(self, eval_cfg):
        metrics = build_metrics(eval_cfg, has_coords=True)
        names = {type(m).__name__ for m in metrics}
        assert "PrecisionAtLMetric" in names
        assert len(metrics) == 4

    def test_build_metrics_only_whitelist(self, eval_cfg):
        cfg = OmegaConf.merge(eval_cfg, {
            "data": {"eval": {"test_ds": {"metrics": {"only": ["loss"]}}}}
        })
        metrics = build_metrics(cfg, has_coords=False, eval_name="test_ds")
        assert len(metrics) == 1
        assert isinstance(metrics[0], LossMetric)

    def test_build_metrics_disabled(self, eval_cfg):
        cfg = OmegaConf.merge(eval_cfg, {
            "eval": {"metrics": {"loss": {"enabled": False}}}
        })
        metrics = build_metrics(cfg, has_coords=False)
        names = {type(m).__name__ for m in metrics}
        assert "LossMetric" not in names

    def test_num_layers_auto_resolution(self):
        cfg = OmegaConf.create({
            "model": {"num_encoder_layers": 20},
            "eval": {
                "metrics": {
                    "p_at_l": {
                        "enabled": True,
                        "num_layers": None,
                        "contact_mode": "all",
                    },
                },
            },
        })
        metrics = build_metrics(cfg, has_coords=True)
        p_at_l = [m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"]
        assert len(p_at_l) == 1
        assert p_at_l[0].num_layers == 2  # ceil(20 * 0.1)

    def test_config_merging_per_dataset(self, eval_cfg):
        cfg = OmegaConf.merge(eval_cfg, {
            "data": {"eval": {"test_ds": {"metrics": {"loss": {"enabled": False}}}}}
        })
        metrics = build_metrics(cfg, has_coords=False, eval_name="test_ds")
        names = {type(m).__name__ for m in metrics}
        assert "LossMetric" not in names
        # Other metrics still present
        assert "MaskedAccuracyMetric" in names
