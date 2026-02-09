# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Base protocol and abstract class for evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..outputs import DenoisingOutput

__all__ = ["Metric", "MetricBase"]


@runtime_checkable
class Metric(Protocol):
    """Protocol defining the interface for evaluation metrics.

    All metrics must implement this protocol to be used with the evaluation system.

    Attributes
    ----------
    name : str
        Unique identifier for the metric (used in logging).
    requires_coords : bool
        Whether this metric requires coordinate data (structure datasets).
    needs_attentions : bool
        Whether this metric requires encoder attention weights.
    """

    name: ClassVar[str]
    requires_coords: ClassVar[bool]
    needs_attentions: ClassVar[bool]

    def update(
        self,
        outputs: DenoisingOutput,
        batch: dict[str, Tensor],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate metric values from a single batch.

        Parameters
        ----------
        outputs
            Model outputs from ``MutableForDenoising.forward()``.
        batch
            Collated batch dict from the DataLoader.
        mask_labels
            Boolean mask of shape (batch, seq_len) indicating which positions
            were masked during eval.
        """
        ...

    def compute(self) -> dict[str, float]:
        """Compute final metric value(s) from accumulated state.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to float values.
        """
        ...

    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        ...

    def state_tensors(self) -> list[Tensor]:
        """Return internal state as tensors for distributed aggregation.

        Returns
        -------
        list[Tensor]
            List of tensors representing the metric's accumulated state.
        """
        ...

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Restore state from gathered tensors (for distributed training).

        Parameters
        ----------
        tensors
            List of tensors as returned by ``state_tensors()``,
            potentially aggregated across processes.
        """
        ...


class MetricBase(ABC):
    """Abstract base class for metrics with default implementations.

    Provides default implementations for ``state_tensors()`` and
    ``load_state_tensors()`` that work for simple scalar accumulators.
    Subclasses should override these if they have more complex state.

    Class Attributes
    ----------------
    name : str
        Unique identifier for the metric.
    requires_coords : bool
        Whether this metric requires coordinate data.
    needs_attentions : bool
        Whether this metric requires encoder attention weights.
    """

    name: ClassVar[str] = ""
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(
        self,
        outputs: DenoisingOutput,
        batch: dict[str, Tensor],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate metric values from a single batch."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute final metric value(s) from accumulated state."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        ...

    def state_tensors(self) -> list[Tensor]:
        """Return internal state as tensors for distributed aggregation.

        Default implementation returns an empty list.
        """
        return []

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Restore state from gathered tensors.

        Default implementation does nothing.
        """
        pass

    def state_objects(self) -> list[Any] | None:
        """Return state as Python objects for distributed gathering.

        Used for metrics with variable-length state that cannot use
        tensor-based gathering. Returns None by default (use tensor gathering).
        """
        return None

    def load_state_objects(self, gathered: list[Any]) -> None:
        """Load state from gathered Python objects.

        Default implementation does nothing.
        """
        pass
