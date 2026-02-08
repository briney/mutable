# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["get_activation_fn"]


def get_activation_fn(
    activation: str,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    bias: Optional[bool] = None,
) -> nn.Module:
    """
    Get the activation function from a string.

    Parameters
    ----------
    activation : str
        The activation function to get.
    input_dim : int, optional
        The input dimension, used for GLU activations.
    output_dim : int, optional
        The output dimension, used for GLU activations.
    bias : bool, optional
        Whether to use bias in the linear layers, used for GLU activations.
    """
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "gelu":
        return GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation in GLU_VARIANTS:
        return GLU(
            in_dim=input_dim, out_dim=output_dim, activation=activation, bias=bias
        )
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class GELU(nn.Module):
    """
    GELU activation function from original ESM repo.
    Using F.gelu yields subtly wrong results.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


GLU_VARIANTS = {
    "glu": F.sigmoid,
    "swiglu": F.silu,
    "geglu": F.gelu,
    "reglu": F.relu,
}


class GLU(nn.Module):
    """
    Activation function for GLU variants.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    activation : str
        Type of GLU variant.
    bias : bool
        Whether to use bias in linear layers.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str, bias: bool):
        super().__init__()
        self.value_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.gate_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation_fn = GLU_VARIANTS[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_linear(x)
        gate = self.gate_linear(x)
        return value * self.activation_fn(gate)
