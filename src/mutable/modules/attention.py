# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import RotaryPositionalEmbedding

__all__ = ["SelfAttention", "CrossAttention"]


class SelfAttention(nn.Module):
    """
    Self-attention block with optional rotary positional embeddings.

    Parameters
    ----------
    model_dim : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    dropout : float, default=0.1
        Dropout rate.
    position_embedding_type : str, default="rotary"
        Position embedding type.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"Model dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.in_proj = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.rotary_embed = (
            RotaryPositionalEmbedding(self.head_dim)
            if position_embedding_type == "rotary"
            else None
        )
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        q, k, v = self._in_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask[:, None, None, :]

        if self.rotary_embed is not None:
            q, k = self.rotary_embed(q, k)

        # use SDPA's is_causal only when no explicit mask is provided
        use_causal = is_causal and attention_mask is None
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=use_causal,
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.model_dim)
        )
        attn_out = self.out_proj(attn_out)

        attn_weights = None
        if need_weights:
            with torch.no_grad():
                scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                if attention_mask is not None:
                    scores = scores.masked_fill(~attention_mask, float("-inf"))
                if is_causal:
                    causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
                    scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)

        return attn_out, attn_weights

    def _in_proj(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.in_proj(x)
        return qkv.chunk(3, dim=-1)


class CrossAttention(nn.Module):
    """
    Cross-attention block. No positional embeddings (operates on latent sets).

    Parameters
    ----------
    model_dim : int
        Query dimension.
    kv_dim : int, default=None
        Key/value dimension. Defaults to model_dim.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.1
        Dropout rate.
    """

    def __init__(
        self,
        model_dim: int,
        kv_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        kv_dim = kv_dim or model_dim
        if model_dim % num_heads != 0:
            raise ValueError(
                f"Model dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.k_proj = nn.Linear(kv_dim, model_dim, bias=True)
        self.v_proj = nn.Linear(kv_dim, model_dim, bias=True)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.attn_dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_value_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        query : torch.Tensor
            Shape (batch, q_len, model_dim).
        key_value : torch.Tensor
            Shape (batch, kv_len, kv_dim).
        key_value_mask : Optional[torch.Tensor]
            Shape (batch, kv_len). True = attend, False = ignore.
        need_weights : bool
            Whether to return attention weights.
        """
        batch_size, q_len, _ = query.shape
        kv_len = key_value.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_value_mask is not None:
            attn_mask = key_value_mask.bool()[:, None, None, :]

        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.model_dim)
        )
        attn_out = self.out_proj(attn_out)

        attn_weights = None
        if need_weights:
            with torch.no_grad():
                scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)

        return attn_out, attn_weights
