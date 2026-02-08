# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import torch

__all__ = ["TimestepSampler"]


class TimestepSampler:
    """
    Timestep sampler for flow matching training.

    Parameters
    ----------
    method : str, default="uniform"
        Sampling method ("uniform" or "logit_normal").
    logit_normal_mean : float, default=0.0
        Mean for logit-normal sampling.
    logit_normal_std : float, default=1.0
        Standard deviation for logit-normal sampling.
    """

    def __init__(
        self,
        method: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ):
        self.method = method
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps in (0, 1).

        Parameters
        ----------
        batch_size : int
            Number of timesteps to sample.
        device : torch.device
            Device for the output tensor.

        Returns
        -------
        torch.Tensor
            Timesteps of shape (batch_size,).
        """
        if self.method == "logit_normal":
            return self._logit_normal(batch_size, device)
        return self._uniform(batch_size, device)

    def _uniform(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def _logit_normal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from logit-normal distribution, biasing towards middle timesteps."""
        normal_samples = torch.randn(batch_size, device=device)
        normal_samples = normal_samples * self.logit_normal_std + self.logit_normal_mean
        return torch.sigmoid(normal_samples)
