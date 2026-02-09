# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for Metric protocol and MetricBase."""

import pytest

from mutable.eval.base import Metric, MetricBase
from mutable.eval.metrics.classification import (
    LossMetric,
    MaskedAccuracyMetric,
    PerplexityMetric,
)
from mutable.eval.metrics.contact import PrecisionAtLMetric


class TestMetricProtocol:
    """Verify all metrics satisfy the Metric protocol."""

    @pytest.mark.parametrize("cls", [LossMetric, MaskedAccuracyMetric, PerplexityMetric])
    def test_classification_metrics_satisfy_protocol(self, cls):
        assert isinstance(cls(), Metric)

    def test_contact_metric_satisfies_protocol(self):
        assert isinstance(PrecisionAtLMetric(), Metric)


class TestMetricBase:
    """Verify MetricBase default implementations."""

    def test_metric_base_defaults(self):
        """MetricBase subclass has empty state_tensors() and None state_objects()."""
        # Create a minimal concrete subclass
        class DummyMetric(MetricBase):
            name = "dummy"
            def update(self, outputs, batch, mask_labels):
                pass
            def compute(self):
                return {}
            def reset(self):
                pass

        m = DummyMetric()
        assert m.state_tensors() == []
        assert m.state_objects() is None

    @pytest.mark.parametrize("cls", [LossMetric, MaskedAccuracyMetric, PerplexityMetric, PrecisionAtLMetric])
    def test_class_attributes_present(self, cls):
        """Each metric class has name, requires_coords, needs_attentions."""
        assert hasattr(cls, "name")
        assert hasattr(cls, "requires_coords")
        assert hasattr(cls, "needs_attentions")

    @pytest.mark.parametrize("cls", [LossMetric, MaskedAccuracyMetric, PerplexityMetric])
    def test_classification_no_coords_no_attentions(self, cls):
        assert cls.requires_coords is False
        assert cls.needs_attentions is False

    def test_contact_requires_coords_and_attentions(self):
        assert PrecisionAtLMetric.requires_coords is True
        assert PrecisionAtLMetric.needs_attentions is True
