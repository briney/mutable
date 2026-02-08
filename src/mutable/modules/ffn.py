# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from .activation import get_activation_fn

__all__ = ["DenseFFN", "GluFFN"]


class DenseFFN(nn.Module):
    """
    Standard (dense) feed-forward network, used for RELU and GELU activations.

    Parameters
    ----------
    model_dim : int
        Token embedding dimension.
    ffn_dim : int
        Feed-forward network dimension.
    bias : bool
        Whether to use bias.
    activation : str
        Activation function to use.
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        bias: bool,
        activation: str,
    ):
        super().__init__()
        self.wi = nn.Linear(model_dim, ffn_dim, bias=bias)
        self.activation = get_activation_fn(activation)
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wi(x)
        x = self.activation(x)
        x = self.wo(x)
        return x


class GluFFN(nn.Module):
    """
    GLU feed-forward network, used for GLU, SwiGLU, GeGLU, and ReGLU activations.

    Parameters
    ----------
    model_dim : int
        Token embedding dimension.
    ffn_dim : int
        Feed-forward network dimension.
    bias : bool
        Whether to use bias.
    activation : str
        Activation function to use.
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        bias: bool,
        activation: str,
    ):
        super().__init__()
        self.activation = get_activation_fn(
            activation, input_dim=model_dim, output_dim=ffn_dim, bias=bias
        )
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.wo(x)
        return x
