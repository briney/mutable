# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .attention import SelfAttention, CrossAttention
from .ffn import DenseFFN, GluFFN

__all__ = [
    "EncoderLayer",
    "DecoderLayer",
    "PerceiverBottleneckLayer",
    "AdaptiveLayerNorm",
    "FlowTransformerLayer",
]


class EncoderLayer(nn.Module):
    """
    Standard pre-norm encoder transformer layer (identical to BALM's DenseTransformerLayer).

    Pre-norm -> SelfAttention -> residual -> pre-norm -> FFN -> residual.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # attention
        self.attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # FFN
        self.ffn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        ffn_class = GluFFN if "glu" in config.activation else DenseFFN
        self.ffn = ffn_class(
            model_dim=config.hidden_size,
            ffn_dim=config.intermediate_size,
            activation=config.activation,
            bias=config.ffn_bias,
        )
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        # attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out, attn_vals = self.attention(
            x, attention_mask=attention_mask, need_weights=need_weights
        )
        x = residual + self.attn_dropout(attn_out)

        # FFN
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + self.ffn_dropout(x)

        return (x, attn_vals) if need_weights else x


class DecoderLayer(nn.Module):
    """
    Decoder transformer layer with causal self-attention and cross-attention to bottleneck latents.

    Pre-norm -> causal SelfAttention -> residual ->
    pre-norm -> CrossAttention(to latents) -> residual ->
    pre-norm -> FFN -> residual.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # causal self-attention
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.self_attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.self_attn_dropout = nn.Dropout(config.attention_dropout)

        # cross-attention to bottleneck latents
        self.cross_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.cross_attention = CrossAttention(
            model_dim=config.hidden_size,
            kv_dim=getattr(config, "latent_dim", config.hidden_size),
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.cross_attn_dropout = nn.Dropout(config.attention_dropout)

        # FFN
        self.ffn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        ffn_class = GluFFN if "glu" in config.activation else DenseFFN
        self.ffn = ffn_class(
            model_dim=config.hidden_size,
            ffn_dim=config.intermediate_size,
            activation=config.activation,
            bias=config.ffn_bias,
        )
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Decoder input, shape (batch, seq_len, hidden_size).
        encoder_hidden_states : torch.Tensor
            Bottleneck latents, shape (batch, num_latents, latent_dim).
        attention_mask : Optional[torch.Tensor]
            Padding mask for self-attention, shape (batch, seq_len).
        need_weights : bool
            Whether to return attention weights.
        """
        # causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        self_attn_out, self_attn_weights = self.self_attention(
            x, attention_mask=attention_mask, is_causal=True, need_weights=need_weights
        )
        x = residual + self.self_attn_dropout(self_attn_out)

        # cross-attention to bottleneck latents
        residual = x
        x = self.cross_attn_layer_norm(x)
        cross_attn_out, cross_attn_weights = self.cross_attention(
            query=x, key_value=encoder_hidden_states, need_weights=need_weights
        )
        x = residual + self.cross_attn_dropout(cross_attn_out)

        # FFN
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + self.ffn_dropout(x)

        if need_weights:
            return x, self_attn_weights, cross_attn_weights
        return x


class PerceiverBottleneckLayer(nn.Module):
    """
    Perceiver bottleneck layer: latent queries cross-attend to encoder KV.

    Pre-norm -> CrossAttention(latent queries -> encoder output) -> residual ->
    pre-norm -> FFN -> residual.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        latent_dim = getattr(config, "latent_dim", config.hidden_size)

        # cross-attention: latent queries attend to encoder output
        self.cross_attn_layer_norm = nn.LayerNorm(
            latent_dim, eps=config.layer_norm_eps
        )
        self.cross_attention = CrossAttention(
            model_dim=latent_dim,
            kv_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.cross_attn_dropout = nn.Dropout(config.attention_dropout)

        # FFN
        self.ffn_layer_norm = nn.LayerNorm(latent_dim, eps=config.layer_norm_eps)
        ffn_class = GluFFN if "glu" in config.activation else DenseFFN
        self.ffn = ffn_class(
            model_dim=latent_dim,
            ffn_dim=config.intermediate_size,
            activation=config.activation,
            bias=config.ffn_bias,
        )
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        latent_queries: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        latent_queries : torch.Tensor
            Shape (batch, num_latents, latent_dim).
        encoder_output : torch.Tensor
            Shape (batch, seq_len, hidden_size).
        encoder_mask : Optional[torch.Tensor]
            Shape (batch, seq_len). True = attend.
        """
        # cross-attention
        residual = latent_queries
        x = self.cross_attn_layer_norm(latent_queries)
        cross_out, cross_weights = self.cross_attention(
            query=x,
            key_value=encoder_output,
            key_value_mask=encoder_mask,
            need_weights=need_weights,
        )
        x = residual + self.cross_attn_dropout(cross_out)

        # FFN
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + self.ffn_dropout(x)

        if need_weights:
            return x, cross_weights
        return x


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (DiT-style).

    LN(x) * (1 + scale) + shift, where scale and shift are derived from conditioning.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension.
    conditioning_dim : int
        Dimension of the conditioning vector.
    eps : float, default=1e-5
        Layer norm epsilon.
    """

    def __init__(self, hidden_size: int, conditioning_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(conditioning_dim, 2 * hidden_size)
        # initialize to identity transform
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, hidden_size).
        conditioning : torch.Tensor
            Shape (batch, conditioning_dim).
        """
        scale_shift = self.proj(conditioning)
        scale, shift = scale_shift.chunk(2, dim=-1)
        # add sequence dimension: (batch, hidden_size) -> (batch, 1, hidden_size)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1.0 + scale) + shift


class FlowTransformerLayer(nn.Module):
    """
    Flow transformer layer with AdaLN conditioning.

    AdaLN -> SelfAttention (no rotary) -> residual ->
    AdaLN -> CrossAttention (to germline latents) -> residual ->
    AdaLN -> FFN -> residual.

    Parameters
    ----------
    config : FlowMatchingConfig
        Flow matching configuration.
    conditioning_dim : int
        Dimension of the conditioning vector (projected t + mu).
    """

    def __init__(self, config: PretrainedConfig, conditioning_dim: int):
        super().__init__()
        hidden_size = config.hidden_size

        # self-attention (no rotary — latents are a set, not a sequence)
        self.self_attn_norm = AdaptiveLayerNorm(
            hidden_size, conditioning_dim, eps=config.layer_norm_eps
        )
        self.self_attention = SelfAttention(
            model_dim=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type="none",
        )
        self.self_attn_dropout = nn.Dropout(config.attention_dropout)

        # cross-attention to germline latents
        self.cross_attn_norm = AdaptiveLayerNorm(
            hidden_size, conditioning_dim, eps=config.layer_norm_eps
        )
        self.cross_attention = CrossAttention(
            model_dim=hidden_size,
            kv_dim=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.cross_attn_dropout = nn.Dropout(config.attention_dropout)

        # FFN
        self.ffn_norm = AdaptiveLayerNorm(
            hidden_size, conditioning_dim, eps=config.layer_norm_eps
        )
        ffn_class = GluFFN if "glu" in config.activation else DenseFFN
        self.ffn = ffn_class(
            model_dim=hidden_size,
            ffn_dim=config.intermediate_size,
            activation=config.activation,
            bias=config.ffn_bias,
        )
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        x: torch.Tensor,
        germline_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Noisy latent vectors, shape (batch, num_latents, hidden_size).
        germline_latents : torch.Tensor
            Germline latent vectors, shape (batch, num_latents, hidden_size).
        conditioning : torch.Tensor
            Conditioning vector (t + mu), shape (batch, conditioning_dim).
        """
        # self-attention
        residual = x
        x = self.self_attn_norm(x, conditioning)
        attn_out, _ = self.self_attention(x)
        x = residual + self.self_attn_dropout(attn_out)

        # cross-attention to germline latents
        residual = x
        x = self.cross_attn_norm(x, conditioning)
        cross_out, _ = self.cross_attention(query=x, key_value=germline_latents)
        x = residual + self.cross_attn_dropout(cross_out)

        # FFN
        residual = x
        x = self.ffn_norm(x, conditioning)
        x = self.ffn(x)
        x = residual + self.ffn_dropout(x)

        return x
