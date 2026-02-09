# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for classification metrics: LossMetric, MaskedAccuracyMetric, PerplexityMetric."""

import math

import torch
import pytest

from mutable.eval.metrics.classification import (
    LossMetric,
    MaskedAccuracyMetric,
    PerplexityMetric,
)
from mutable.outputs import DenoisingOutput


def _make_output_and_batch(B=2, L=10, V=32, perfect=False, all_wrong=False):
    """Build a DenoisingOutput and batch for testing classification metrics.

    Parameters
    ----------
    perfect : bool
        If True, logits have very high value at the correct label.
    all_wrong : bool
        If True, logits have very high value at wrong labels.
    """
    torch.manual_seed(42)
    labels = torch.randint(0, V, (B, L))

    if perfect:
        logits = torch.full((B, L, V), -100.0)
        for b in range(B):
            for pos in range(L):
                logits[b, pos, labels[b, pos]] = 100.0
    elif all_wrong:
        logits = torch.full((B, L, V), -100.0)
        for b in range(B):
            for pos in range(L):
                wrong = (labels[b, pos].item() + 1) % V
                logits[b, pos, wrong] = 100.0
    else:
        logits = torch.randn(B, L, V)

    mask_labels = torch.ones(B, L, dtype=torch.bool)

    output = DenoisingOutput(logits=logits)
    batch = {"labels": labels}
    return output, batch, mask_labels


class TestLossMetric:
    def test_loss_perfect_predictions(self):
        output, batch, mask = _make_output_and_batch(perfect=True)
        m = LossMetric()
        m.update(output, batch, mask)
        result = m.compute()
        assert result["loss"] < 0.01

    def test_loss_random_predictions(self):
        output, batch, mask = _make_output_and_batch()
        m = LossMetric()
        m.update(output, batch, mask)
        result = m.compute()
        expected_approx = -math.log(1 / 32)  # ~3.47
        assert abs(result["loss"] - expected_approx) < 1.5

    def test_reset_clears_state(self):
        output, batch, mask = _make_output_and_batch()
        m = LossMetric()
        m.update(output, batch, mask)
        m.reset()
        result = m.compute()
        assert result["loss"] == 0.0

    def test_multi_batch_accumulation(self):
        output, batch, mask = _make_output_and_batch()
        m = LossMetric()
        m.update(output, batch, mask)
        val1 = m.compute()["loss"]
        m.update(output, batch, mask)
        val2 = m.compute()["loss"]
        # Two identical batches should give same avg loss
        assert abs(val1 - val2) < 1e-5


class TestMaskedAccuracyMetric:
    def test_accuracy_perfect(self):
        output, batch, mask = _make_output_and_batch(perfect=True)
        m = MaskedAccuracyMetric()
        m.update(output, batch, mask)
        result = m.compute()
        assert result["mask_acc"] == pytest.approx(1.0)

    def test_accuracy_zero(self):
        output, batch, mask = _make_output_and_batch(all_wrong=True)
        m = MaskedAccuracyMetric()
        m.update(output, batch, mask)
        result = m.compute()
        assert result["mask_acc"] == pytest.approx(0.0)


class TestPerplexityMetric:
    def test_perplexity_value(self):
        output, batch, mask = _make_output_and_batch()
        m = PerplexityMetric()
        m.update(output, batch, mask)
        result = m.compute()

        # Verify ppl = exp(loss)
        loss_m = LossMetric()
        loss_m.update(output, batch, mask)
        loss_val = loss_m.compute()["loss"]
        assert result["ppl"] == pytest.approx(math.exp(loss_val), rel=1e-4)

    def test_perplexity_clamp(self):
        """Very high loss is clamped, no inf/overflow."""
        m = PerplexityMetric()
        m._total_loss = 200.0
        m._total_count = 1
        result = m.compute()
        assert math.isfinite(result["ppl"])
        assert result["ppl"] == pytest.approx(math.exp(100.0))


class TestClassificationMetricEdgeCases:
    def test_metrics_ignore_minus_100(self):
        """Positions with labels=-100 are excluded from computation."""
        output, batch, mask = _make_output_and_batch(perfect=True)
        batch["labels"][:, 0] = -100  # mark first position
        m = MaskedAccuracyMetric()
        m.update(output, batch, mask)
        result = m.compute()
        assert result["mask_acc"] == pytest.approx(1.0)

    def test_metrics_ignore_unmasked(self):
        """Only masked positions contribute to metrics."""
        output, batch, mask = _make_output_and_batch(perfect=True)
        # Only mask position 0
        mask = torch.zeros_like(mask)
        mask[:, 0] = True
        m = MaskedAccuracyMetric()
        m.update(output, batch, mask)
        result = m.compute()
        # Position 0 has perfect prediction so accuracy is 1.0
        assert result["mask_acc"] == pytest.approx(1.0)

    def test_state_tensors_round_trip(self):
        output, batch, mask = _make_output_and_batch()
        m = LossMetric()
        m.update(output, batch, mask)

        tensors = m.state_tensors()
        assert len(tensors) == 1

        # Modify state via load
        doubled = [t * 2 for t in tensors]
        m.load_state_tensors(doubled)
        result = m.compute()

        # Original result
        m2 = LossMetric()
        m2.update(output, batch, mask)
        orig = m2.compute()

        # After doubling both numerator and denominator, loss should be same
        # because both _total_loss and _total_count are stored together
        # Actually _total_loss is doubled, _total_count is doubled => same ratio
        assert result["loss"] == pytest.approx(orig["loss"], rel=1e-4)
