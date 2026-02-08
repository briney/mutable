# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["ConditionalFlowMatcher", "optimal_transport_plan"]


class ConditionalFlowMatcher:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).

    Implements straight-line interpolation paths between source (x_0) and
    target (x_1) distributions with optimal transport coupling.

    Parameters
    ----------
    sigma_min : float, default=1e-4
        Minimum noise level.
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def sample_location_and_conditional_flow(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the interpolated location x_t and the conditional velocity u_t.

        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        u_t = x_1 - (1 - sigma_min) * x_0

        Parameters
        ----------
        x_0 : torch.Tensor
            Source samples, shape (batch, ...).
        x_1 : torch.Tensor
            Target samples, shape (batch, ...).
        t : torch.Tensor
            Timesteps, shape (batch,) or (batch, 1, ...).

        Returns
        -------
        x_t : torch.Tensor
            Interpolated samples at time t.
        u_t : torch.Tensor
            Target conditional velocity.
        """
        # expand t to match spatial dimensions
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)

        sigma_min = self.sigma_min
        x_t = (1.0 - (1.0 - sigma_min) * t) * x_0 + t * x_1
        u_t = x_1 - (1.0 - sigma_min) * x_0

        return x_t, u_t

    def compute_loss(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the OT-CFM MSE loss.

        Parameters
        ----------
        predicted_velocity : torch.Tensor
            Model's predicted velocity field.
        target_velocity : torch.Tensor
            Target conditional velocity.

        Returns
        -------
        torch.Tensor
            Scalar MSE loss.
        """
        return nn.functional.mse_loss(predicted_velocity, target_velocity)


def optimal_transport_plan(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minibatch optimal transport coupling using the Hungarian algorithm.

    Pairs source and target samples to minimize total squared distance,
    returning reordered tensors.

    Parameters
    ----------
    x_0 : torch.Tensor
        Source samples, shape (batch, ...).
    x_1 : torch.Tensor
        Target samples, shape (batch, ...).

    Returns
    -------
    x_0_paired : torch.Tensor
        Reordered source samples.
    x_1_paired : torch.Tensor
        Reordered target samples.
    """
    batch_size = x_0.shape[0]
    # flatten spatial dims for cost computation
    x_0_flat = x_0.detach().reshape(batch_size, -1)
    x_1_flat = x_1.detach().reshape(batch_size, -1)

    # compute pairwise squared distances
    cost_matrix = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2).cpu().numpy()

    # solve assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    return x_0[row_indices], x_1[col_indices]
