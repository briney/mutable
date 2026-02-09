# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Sequence-level classification metrics: loss, masked accuracy, perplexity."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import MetricBase
from ..registry import register_metric
from ...outputs import DenoisingOutput

__all__ = ["LossMetric", "MaskedAccuracyMetric", "PerplexityMetric"]


def _valid_mask(batch: dict[str, Tensor], mask_labels: Tensor) -> Tensor:
    """Return boolean mask for positions that are both masked and have valid labels.

    Parameters
    ----------
    batch
        Batch dict containing ``labels``.
    mask_labels
        Boolean mask of eval-masked positions.

    Returns
    -------
    Tensor
        Boolean mask of shape (batch, seq_len).
    """
    labels = batch["labels"]
    return mask_labels & (labels != -100)


@register_metric("loss")
class LossMetric(MetricBase):
    """Cross-entropy loss on masked positions."""

    name = "loss"
    requires_coords = False
    needs_attentions = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total_loss = 0.0
        self._total_count = 0

    def update(
        self,
        outputs: DenoisingOutput,
        batch: dict[str, Tensor],
        mask_labels: Tensor,
    ) -> None:
        valid = _valid_mask(batch, mask_labels)
        if not valid.any():
            return

        logits = outputs["logits"]  # (B, L, V)
        labels = batch["labels"]

        # Flatten and select valid positions
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        valid_flat = valid.view(-1)

        loss = F.cross_entropy(
            logits_flat[valid_flat],
            labels_flat[valid_flat],
            reduction="sum",
        )
        self._total_loss += loss.item()
        self._total_count += valid_flat.sum().item()

    def compute(self) -> dict[str, float]:
        if self._total_count == 0:
            return {"loss": 0.0}
        return {"loss": self._total_loss / self._total_count}

    def reset(self) -> None:
        self._total_loss = 0.0
        self._total_count = 0

    def state_tensors(self) -> list[Tensor]:
        return [torch.tensor([self._total_loss, self._total_count], dtype=torch.float64)]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._total_loss = t[0].item()
            self._total_count = int(t[1].item())


@register_metric("masked_accuracy")
class MaskedAccuracyMetric(MetricBase):
    """Top-1 accuracy on masked positions."""

    name = "mask_acc"
    requires_coords = False
    needs_attentions = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._correct = 0
        self._total = 0

    def update(
        self,
        outputs: DenoisingOutput,
        batch: dict[str, Tensor],
        mask_labels: Tensor,
    ) -> None:
        valid = _valid_mask(batch, mask_labels)
        if not valid.any():
            return

        logits = outputs["logits"]
        labels = batch["labels"]

        preds = logits.argmax(dim=-1)  # (B, L)
        correct = (preds == labels) & valid

        self._correct += correct.sum().item()
        self._total += valid.sum().item()

    def compute(self) -> dict[str, float]:
        if self._total == 0:
            return {"mask_acc": 0.0}
        return {"mask_acc": self._correct / self._total}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def state_tensors(self) -> list[Tensor]:
        return [torch.tensor([self._correct, self._total], dtype=torch.float64)]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._correct = int(t[0].item())
            self._total = int(t[1].item())


@register_metric("perplexity")
class PerplexityMetric(MetricBase):
    """Perplexity (exp of average loss) on masked positions."""

    name = "ppl"
    requires_coords = False
    needs_attentions = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total_loss = 0.0
        self._total_count = 0

    def update(
        self,
        outputs: DenoisingOutput,
        batch: dict[str, Tensor],
        mask_labels: Tensor,
    ) -> None:
        valid = _valid_mask(batch, mask_labels)
        if not valid.any():
            return

        logits = outputs["logits"]
        labels = batch["labels"]

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        valid_flat = valid.view(-1)

        loss = F.cross_entropy(
            logits_flat[valid_flat],
            labels_flat[valid_flat],
            reduction="sum",
        )
        self._total_loss += loss.item()
        self._total_count += valid_flat.sum().item()

    def compute(self) -> dict[str, float]:
        if self._total_count == 0:
            return {"ppl": 0.0}
        avg_loss = self._total_loss / self._total_count
        return {"ppl": math.exp(min(avg_loss, 100.0))}  # clamp to avoid overflow

    def reset(self) -> None:
        self._total_loss = 0.0
        self._total_count = 0

    def state_tensors(self) -> list[Tensor]:
        return [torch.tensor([self._total_loss, self._total_count], dtype=torch.float64)]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._total_loss = t[0].item()
            self._total_count = int(t[1].item())
