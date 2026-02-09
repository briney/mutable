# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RotaryPositionalEmbedding",
    "SinusoidalTimestepEmbedding",
    "LearnedLatentEmbedding",
]


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (from BALM)
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE).

    Parameters
    ----------
    dim : int
        The embedding dimension (per head).
    """

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension):
        seq_len = x.shape[seq_dimension]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached).to(
                dtype=q.dtype
            ),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached).to(
                dtype=k.dtype
            ),
        )


# ---------------------------------------------------------------------------
# Sinusoidal Timestep Embedding (new for flow matching)
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding for continuous scalar inputs (timestep t, mutation intensity mu).

    Maps a scalar (batch,) -> (batch, embedding_dim) using sinusoidal frequencies,
    then projects through a linear layer.

    Parameters
    ----------
    embedding_dim : int
        Output dimension of the embedding.
    max_period : int, default=10000
        Controls the range of frequencies.
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Scalar input of shape (batch,).

        Returns
        -------
        torch.Tensor
            Embedding of shape (batch, embedding_dim).
        """
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=x.device, dtype=torch.float32)
            / half_dim
        )
        args = x[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # handle odd embedding_dim
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Learned Latent Embedding (new for Perceiver bottleneck)
# ---------------------------------------------------------------------------


class LearnedLatentEmbedding(nn.Module):
    """
    Learned latent query vectors for the Perceiver bottleneck.

    Parameters
    ----------
    num_latents : int
        Number of latent vectors.
    latent_dim : int
        Dimension of each latent vector.
    """

    def __init__(self, num_latents: int, latent_dim: int):
        super().__init__()
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, latent_dim)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_latents, latent_dim).
        """
        return self.latent_queries.expand(batch_size, -1, -1)
